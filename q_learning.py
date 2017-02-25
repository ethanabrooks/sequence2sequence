import gym.spaces
import numpy as np
import tensorflow as tf
from tensorflow.python.training.adam import AdamOptimizer
from tensorflow.python.training.ftrl import FtrlOptimizer
from tensorflow.python.training.rmsprop import RMSPropOptimizer

learning_rate = 0.1
optimizers = {
    1: AdamOptimizer(learning_rate),
    2: FtrlOptimizer(learning_rate),
    3: RMSPropOptimizer(learning_rate),
}
optimizer = optimizers[2]


def get_base_name(var):
    return var.name.split(':')[0]


env = gym.make('CartPole-v1')

dtype = env.observation_space.low.dtype

assert (isinstance(env.action_space, gym.spaces.Discrete))
obs_size, = env.observation_space.shape
act_size = env.action_space.n

hidden_size = 3
sample_size = 10

memory_buffer = []


class Model:
    def __init__(self, hidden_size):
        self.weights = [tf.get_variable('weight0', [obs_size, hidden_size], dtype),
                        tf.get_variable('weight1', [hidden_size, act_size], dtype)]

        self.biases = [tf.get_variable('bias0', [hidden_size], dtype),
                       tf.get_variable('bias1', [act_size], dtype)]

    def perceptron(self, i, x):
        return tf.matmul(x, self.weights[i]) + self.biases[i]

    def forward(self, x):
        hidden = tf.sigmoid(self.perceptron(0, x), name='hidden')
        return self.perceptron(1, hidden)


model = Model(hidden_size)

shape = [sample_size, 2 * obs_size]
observations_pair_ph = tf.placeholder(dtype, shape, name='state_action')
actions_ph = tf.placeholder(tf.int32, [sample_size, 2])
rewards_ph = tf.placeholder(dtype, [sample_size])

q = tf.gather_nd(model.forward(observations_pair_ph), actions_ph)
loss = tf.nn.l2_loss((q[0] + rewards_ph) - q[1])
train_op = optimizer.minimize(loss)

observation_ph = tf.placeholder(dtype, obs_size, name='state_action')
q = model.forward(observation_ph)
tf_action = tf.argmax(q)

show_off = False
with tf.Session() as sess:
    tf.global_variables_initializer().run()

    # update every epoch
    while True:
        observation = env.reset()
        done = False
        t = 0
        cumulative_reward = 0
        cumulative_scores = zeros_like_params()

        # steps
        while not done:
            if show_off:
                env.render()
            action, new_scores = sess.run([tf_action, tf_scores],
                                          {memory_buffer_ph: observation.squeeze()})
            observation, reward, done, info = env.step(action)

            cumulative_reward += reward
            for old_score, new_score in zip(cumulative_scores, new_scores):
                old_score += new_score
            t += 1

        mean_reward += cumulative_reward / batches
        for gradient, cumulative_score in zip(gradients, cumulative_scores):
            gradient -= cumulative_score * cumulative_reward / batches

        print("Epoch: {}. Reward: {}".format(e, mean_reward))
        feed_dict = dict(zip(gradient_phs, gradients))
        sess.run(train_op, feed_dict)
        if mean_reward >= 200:
            show_off = True
