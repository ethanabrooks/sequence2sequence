import tensorflow as tf
import numpy as np

from model import Model


class MLP(Model):
    def __init__(self, layer_sizes, dtype=tf.float32):
        self.dtype = dtype
        self._in_layer_created = False
        self._out_layer_created = False
        assert layer_sizes
        layer_sizes = list(layer_sizes)
        self._layer_sizes = layer_sizes
        self.weights, self.biases = [], []

        # interior layers
        for i, (in_size, out_size) in enumerate(zip(layer_sizes, layer_sizes[1:])):
            self.weights.append(tf.get_variable('weight' + str(i + 1), [in_size, out_size], dtype))
            self.biases.append(tf.get_variable('bias' + str(i + 1), [out_size], dtype))

    @property
    def params(self):
        assert self._in_layer_created and self._out_layer_created
        return self.weights + self.biases

    def forward(self, x):
        assert self._in_layer_created and self._out_layer_created
        _, in_size = x.get_shape()
        for weight, bias in zip(self.weights, self.biases)[:-1]:
            x = tf.sigmoid(tf.matmul(x, weight) + bias)

        return tf.matmul(x, self.weights[-1]) + self.biases[-1]

    def create_in_layer(self, in_size):
        self._in_layer_created = True
        out_size = self._layer_sizes[0]
        self.weights.insert(0, tf.get_variable('weight0', [in_size, out_size], self.dtype))
        self.biases.insert(0, tf.get_variable('bias0', [out_size], self.dtype))

    def create_out_layer(self, out_size):
        self._out_layer_created = True
        in_size = self._layer_sizes[-1]
        i = str(len(self._layer_sizes))
        self.weights.append(tf.get_variable('weight' + i, [in_size, out_size], self.dtype))
        self.biases.append(tf.get_variable('bias' + i, [out_size], self.dtype))
