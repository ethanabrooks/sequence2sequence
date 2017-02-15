"""Module for constructing RNN Cells.

@@RNNCell
@@NTMCell
@@NTMStateTuple

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math

import tensorflow as tf
# from tensorflow.python.ops.rnn_cell import RNNCell
from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell

_NTMStateTuple = collections.namedtuple("LSTMStateTuple", ("M", "h", "w"))


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


class NTMCell(RNNCell):
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
            # Parameters of gates are concatenated into one multiply for efficiency.
            M, h, w = state
            M = tf.reshape(M, (-1, self._dim, self._size_memory))  # ?, dim, size_mem

            c = tf.squeeze(
                tf.batch_matmul(M, tf.expand_dims(w, axis=2)),
                axis=2
            )  # ?, dim

            concat = tf.concat(1, [inputs, c])  # ?, dim + dim

            input_dims = {
                inputs: inputs.get_shape()[1],
                M: self._dim,
                w: self._size_memory,
                c: self._dim,
                h: self._dim
            }

            output_dims = [
                1,  # g weight_interpolation
                input_dims[M],  # k = memory_key
                1,  # b = read_sharpness
                self._size_memory,  # e = erase_vector
                input_dims[M],  # v = new memory vector
            ]

            total_variable_dim = sum(output_dims)

            weight_dims = [
                # (in_size, out_size)
                (input_dims[inputs] + input_dims[c], input_dims[h]),
                (input_dims[h], total_variable_dim),
            ]

            weights = []
            biases = []
            for i, (in_size, out_size) in enumerate(weight_dims):
                dtype = state.dtype
                weights.append(
                    tf.get_variable("weight" + str(i), [in_size, out_size], dtype)
                )
                biases.append(
                    tf.get_variable("bias" + str(i), [out_size], dtype)
                )

            new_h = tf.sigmoid(tf.matmul(concat, weights[0]) + biases[0])
            new_h = tf.verify_tensor_all_finite(new_h, 'new_h')

            concat = tf.matmul(new_h, weights[1]) + biases[1]
            concat = tf.verify_tensor_all_finite(concat, 'concat')

            g, k, b, e, v = tf.split_v(concat, output_dims, split_dim=1)

            # cosine distances
            k = tf.expand_dims(
                tf.nn.l2_normalize(k, dim=1), axis=2
            )  # ?, dim, 1
            k = tf.verify_tensor_all_finite(k, 'k')
            M_hat = tf.nn.l2_normalize(M, dim=1)  # ?, dim, size_mem
            M_hat = tf.verify_tensor_all_finite(M_hat, 'M_hat')
            cosine_distance = tf.squeeze(
                tf.batch_matmul(k, M_hat, adj_x=True), axis=1
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
            new_content = tf.batch_matmul(v, f)  # ?, dim, size_mem
            new_content = tf.verify_tensor_all_finite(new_content, 'new_content')
            new_M = M * (1 - f) + new_content  # ?, dim, size_mem
            new_M = tf.verify_tensor_all_finite(new_M, 'new_M')
            new_M = tf.reshape(new_M, [-1, self._size_memory * self._dim])
            # M = tf.reshape(M, [-1, self._size_memory * self._dim])

            return new_h, NTMStateTuple(new_M, new_h, new_w)
            # return h, NTMStateTuple(M, h, w)


def _get_concat_variable(name, shape, dtype, num_shards):
    """Get a sharded variable concatenated into one tensor."""
    sharded_variable = _get_sharded_variable(name, shape, dtype, num_shards)
    if len(sharded_variable) == 1:
        return sharded_variable[0]

    concat_name = name + "/concat"
    concat_full_name = tf.get_variable_scope().name + "/" + concat_name + ":0"
    for value in tf.get_collection(tf.GraphKeys.CONCATENATED_VARIABLES):
        if value.name == concat_full_name:
            return value

    concat_variable = tf.concat(0, sharded_variable, name=concat_name)
    tf.add_to_collection(tf.GraphKeys.CONCATENATED_VARIABLES,
                         concat_variable)
    return concat_variable


def _get_sharded_variable(name, shape, dtype, num_shards):
    """Get a list of sharded variables with the given dtype."""
    if num_shards > shape[0]:
        raise ValueError("Too many shards: shape=%s, num_shards=%d" %
                         (shape, num_shards))
    unit_shard_size = int(math.floor(shape[0] / num_shards))
    remaining_rows = shape[0] - unit_shard_size * num_shards

    shards = []
    for i in range(num_shards):
        current_size = unit_shard_size
        if i < remaining_rows:
            current_size += 1
        shards.append(tf.get_variable(name + "_%d" % i, [current_size] + shape[1:],
                                      dtype=dtype))
    return shards


def _linear(args, output_size, bias, bias_start=0.0, scope=None):
    """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_start: starting value to initialize the bias; 0 by default.
    scope: VariableScope for the created subgraph; defaults to "Linear".

  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """
    if args is None or (tf.nest.is_sequence(args) and not args):
        raise ValueError("`args` must be specified")
    if not tf.nest.is_sequence(args):
        args = [args]

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
        if len(shape) != 2:
            raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
        if not shape[1]:
            raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
        else:
            total_arg_size += shape[1]

    dtype = [a.dtype for a in args][0]

    # Now the computation.
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable(
            "Matrix", [total_arg_size, output_size], dtype=dtype)
        if len(args) == 1:
            res = tf.matmul(args[0], matrix)
        else:
            res = tf.matmul(tf.concat(1, args), matrix)
        if not bias:
            return res
        bias_term = tf.get_variable(
            "Bias", [output_size],
            dtype=dtype,
            initializer=tf.constant_initializer(
                bias_start, dtype=dtype))
    return res + bias_term
