from model import Model
import tensorflow as tf


class Conv(Model):
    def __init__(self, filters, kernel_size, strides, padding):
        self._filters = filters
        self._kernel_size = kernel_size
        self._strides = strides
        self._padding = padding

    def forward(self, x):
        tf.layers.conv3d(x,
                         self._filters,
                         self._kernel_size,
                         self._strides,
                         self._padding)

    def create_in_layer(self, in_size):
        pass

    def params(self):
        pass

    def create_out_layer(self, in_size):
        pass