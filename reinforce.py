import tensorflow as tf
import numpy as np
import gym.spaces
import copy
from tensorflow.python.training.adam import AdamOptimizer

import ntm


def get_base_name(var):
    return var.name.split(':')[0]


env = gym.make('CartPole-v1')

dtype = env.observation_space.low.dtype

assert (isinstance(env.action_space, gym.spaces.Discrete))
obs_size, = env.observation_space.shape
act_size = env.action_space.n

hidden_size = 50


def get_params():
    weights = [tf.get_variable('weight0', [obs_size, hidden_size], dtype),
               tf.get_variable('weight1', [hidden_size, act_size], dtype)]

    biases = [tf.get_variable('bias0', [hidden_size], dtype),
              tf.get_variable('bias1', [act_size], dtype)]
    return weights, biases


weights, biases = get_params()


def perceptron(i, x):
    return tf.matmul(x, weights[i]) + biases[i]


observation_ph = tf.placeholder(dtype, obs_size, name='observation')
hidden = perceptron(0, tf.expand_dims(observation_ph, 0))
action_dist = perceptron(1, hidden)
tf_action = tf.squeeze(tf.multinomial(action_dist, 1), name='action')

prob = tf.gather(tf.squeeze(action_dist), tf_action, name='prob')
log_prob = tf.log(prob, name='log_prob')
tf_scores = tf.gradients(log_prob, weights + biases, name='gradient')

params = weights + biases
gradient_phs = [tf.placeholder(dtype, param.get_shape(),
                               name=get_base_name(param) + '_placeholder')
                for param in params]
train_op = AdamOptimizer().apply_gradients(zip(gradient_phs, params))

epochs = 200
batches = 500

for _ in range(epochs):
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        gradients = [np.zeros(param.get_shape()) for param in params]
        mean_reward = 0
        for _ in range(batches):
            observation = env.reset()
            done = False
            while not done:
                # env.render()
                action, new_scores = sess.run([tf_action, tf_scores],
                                              {observation_ph: observation.squeeze()})
                observation, reward, done, info = env.step(env.action_space.sample())

                mean_reward += reward / batches
                for gradient, score in zip(gradients, new_scores):
                    gradient += reward * score / batches

        print("Reward: {}".format(mean_reward))
        feed_dict = dict(zip(gradient_phs, gradients))
        sess.run(train_op, feed_dict)
