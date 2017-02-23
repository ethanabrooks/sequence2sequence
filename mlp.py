import tensorflow as tf


class MLP:
    def __init__(self, layer_sizes, dtype=tf.float32):
        layer_sizes = list(layer_sizes)
        self.weights, self.biases = [], []
        for i, (in_size, out_size) in enumerate(zip(layer_sizes, layer_sizes[1:])):
            self.weights.append(tf.get_variable('weight' + str(i), [in_size, out_size], dtype))
            self.biases.append(tf.get_variable('bias' + str(i), [out_size], dtype))

    @property
    def params(self):
        return self.weights + self.biases

    def forward(self, x):
        for weight, bias in zip(self.weights, self.biases)[:-1]:
            x = tf.sigmoid(tf.matmul(x, weight) + bias)

        return tf.matmul(x, self.weights[-1]) + self.biases[-1]
