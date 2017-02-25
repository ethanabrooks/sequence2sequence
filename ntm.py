"""Module for constructing RNN Cells.

@@RNNCell
@@NTMCell
@@NTMStateTuple

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools
import math

import tensorflow as tf
# from tensorflow.python.ops.rnn_cell import RNNCell

_NTMStateTuple = collections.namedtuple("LSTMStateTuple", ("M", "h", "w"))


# noinspection PyClassHasNoInit
class NTMStateTuple(_NTMStateTuple):
    """Tuple used by LSTM Cells for `state_size`, `zero_state`, and output state.

  Stores two elements: `(c, h)`, in that order.
  """
    __slots__ = ()

    @property
    def dtype(self):
        (M, h, w) = self
        if not M.dtype == h.dtype == w.dtype:
            raise TypeError("Inconsistent internal state: {} vs {} vs {}"
                            .format(M.dtype, h.dtype, w.dtype))
        return M.dtype


class Cell(tf.contrib.rnn.RNNCell):
    """Basic LSTM recurrent network cell.

  The implementation is based on: http://arxiv.org/abs/1409.2329.

  We add forget_bias (default: 1) to the biases of the forget gate in order to
  reduce the scale of forgetting in the beginning of the training.

  It does not allow cell clipping, a projection layer, and does not
  use peep-hole connections: it is the basic baseline.

  For advanced models, please use the full LSTMCell that follows.
  """

    def __init__(self,
                 num_units,
                 forget_bias=1.0,
                 activation=tf.tanh,
                 size_memory=8):
        """Initialize the basic LSTM cell.

    Args:
      num_units: int, The number of units in the LSTM cell.
      forget_bias: float, The bias added to forget gates (see above).
      input_size: Deprecated and unused.
      activation: Activation function of the inner states.
    """
        assert isinstance(num_units, (float, int)), \
            '`num_units` ({}) must be a float or an int'.format(num_units)
        self._dim = num_units
        self._forget_bias = forget_bias
        self._activation = activation
        self._size_memory = size_memory

    @property
    def state_size(self):
        return NTMStateTuple(self._dim * self._size_memory,
                             self._dim,
                             self._size_memory)

    @property
    def output_size(self):
        return self._dim

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):  # "NTMCell"
            if len(inputs.get_shape()) == 1:
                inputs = tf.expand_dims(inputs, axis=0)

            M, h, w = state
            M = tf.reshape(M, (-1, self._dim, self._size_memory))  # ?, dim, size_mem

            c = tf.squeeze(
                tf.matmul(M, tf.expand_dims(w, axis=2)),
                axis=2
            )  # ?, dim

            concat = tf.concat([inputs, c], 1)  # ?, dim + dim

            dims = {
                inputs: inputs.get_shape()[1],
                M: self._dim,
                w: self._size_memory,
                c: self._dim,
                h: self._dim
            }

            var_dims = [
                1,  # g weight_interpolation
                dims[M],  # k = memory_key
                1,  # b = read_sharpness
                self._size_memory,  # e = erase_vector
                dims[M],  # v = new memory vector
            ]

            total_variable_dim = sum(var_dims)
            dt = state.dtype

            weights = [tf.get_variable('weight0', [dims[inputs] + dims[c], dims[h]], dt),
                       tf.get_variable('weight1', [dims[h], total_variable_dim], dt)]

            biases = [tf.get_variable('bias0', [dims[h]], dt),
                      tf.get_variable('bias1', [total_variable_dim], dt)]

            new_h = tf.sigmoid(tf.matmul(concat, weights[0]) + biases[0])
            new_h = tf.verify_tensor_all_finite(new_h, 'new_h')

            # Parameters of gates are concatenated into one multiply for efficiency.
            concat = tf.matmul(new_h, weights[1]) + biases[1]
            concat = tf.verify_tensor_all_finite(concat, 'concat')
            g, k, b, e, v = tf.split(concat, var_dims, axis=1)

            # cosine distances
            k = tf.expand_dims(
                tf.nn.l2_normalize(k, dim=1), axis=2
            )  # ?, dim, 1
            k = tf.verify_tensor_all_finite(k, 'k')
            M_hat = tf.nn.l2_normalize(M, dim=1)  # ?, dim, size_mem
            M_hat = tf.verify_tensor_all_finite(M_hat, 'M_hat')
            cosine_distance = tf.squeeze(
                tf.matmul(k, M_hat, transpose_a=True), axis=1
            )  # ?, size_mem
            cosine_distance = tf.verify_tensor_all_finite(cosine_distance, 'cosine_distance')

            # w
            g = tf.sigmoid(g)  # ?, 1
            v = tf.tanh(v)  # ?, dim
            b = tf.nn.softplus(b)  # ?, 1
            b = tf.verify_tensor_all_finite(b, 'b')
            w_hat = tf.nn.softmax(b * cosine_distance)  # ?, size_mem
            w_hat = tf.verify_tensor_all_finite(w_hat, 'w_hat')
            new_w = (1 - g) * w + g * w_hat  # ?, size_mem
            new_w = tf.verify_tensor_all_finite(new_w, 'new_w')

            # M
            e = tf.sigmoid(e)
            f = tf.expand_dims(new_w * e, axis=1)  # ?, 1, size_mem
            f = tf.verify_tensor_all_finite(f, 'f')
            v = tf.expand_dims(v, axis=2)  # ?, dim, 1
            new_content = tf.matmul(v, f)  # ?, dim, size_mem
            new_content = tf.verify_tensor_all_finite(new_content, 'new_content')
            new_M = M * (1 - f) + new_content  # ?, dim, size_mem
            new_M = tf.verify_tensor_all_finite(new_M, 'new_M')
            new_M = tf.reshape(new_M, [-1, self._size_memory * self._dim])

            return new_h, NTMStateTuple(new_M, new_h, new_w)
            # return h, NTMStateTuple(M, h, w)
