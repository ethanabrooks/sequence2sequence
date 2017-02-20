import tensorflow as tf
import numpy as np
import gym.spaces
from tensorflow.python.training.adam import AdamOptimizer

import ntm

env = gym.make('CartPole-v1')

dtype = env.observation_space.low.dtype

assert (isinstance(env.action_space, gym.spaces.Discrete))
obs_size, = env.observation_space.shape
act_size = env.action_space.n

hidden_size = 50


# model = ntm.Cell(hidden_size)
# state_tuple = model.zero_state(1, dtype)

def get_params():
    weights = [tf.get_variable('weight0', [obs_size, hidden_size], dtype),
               tf.get_variable('weight1', [hidden_size, act_size], dtype)]

    biases = [tf.get_variable('bias0', [hidden_size], dtype),
              tf.get_variable('bias1', [act_size], dtype)]
    return weights, biases


weights, biases = get_params()


def perceptron(i, x):
    return tf.matmul(x, weights[i]) + biases[i]


observation_tf = tf.placeholder(dtype, obs_size, name='observation')
hidden = perceptron(0, tf.expand_dims(observation_tf, 0))
action_dist = perceptron(1, hidden)
tf_action = tf.squeeze(tf.multinomial(action_dist, 1), name='action')

prob = tf.gather(tf.squeeze(action_dist), tf_action, name='prob')
log_prob = tf.log(prob, name='log_prob')
tf_gradient = tf.gradients(log_prob, weights + biases, name='gradient')

params = weights + biases
tf_gradients = [tf.placeholder(dtype, [None] + list(param.get_shape()))
                for param in params]
# tf_rewards = tf.placeholder(dtype, [None])
mean_gradients = [tf.reduce_mean(gradient, 0) for gradient in tf_gradients]
train_op = AdamOptimizer().apply_gradients(zip(mean_gradients, params))

for _ in range(200):
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        observation_np = env.reset()
        gradients, rewards = [], []
        for t in range(100):
            env.render()
            obs = observation_np.squeeze()
            action, np_gradient = sess.run([tf_action, tf_gradient], {observation_tf: obs})
            observation_np, reward, done, info = env.step(action)
            gradients.append(np_gradient)
            rewards.append(reward)

            if done:
                print("Timesteps: {}. Reward: {}".format(t + 1, sum(rewards)))
                break

        np_gradients = [np.stack(same_var_gradients) for same_var_gradients in zip(*gradients)]
        sess.run(train_op, dict(zip(tf_gradients, np_gradients)))
