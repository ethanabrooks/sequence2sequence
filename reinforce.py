import gym.spaces
import numpy as np
import tensorflow as tf
from tensorflow.python.training.adam import AdamOptimizer
from tensorflow.python.training.ftrl import FtrlOptimizer
from tensorflow.python.training.rmsprop import RMSPropOptimizer

learning_rate = 0.01
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

hidden_size = 50


def get_params():
    weights = [tf.get_variable('weight0', [obs_size, hidden_size], dtype),
               tf.get_variable('weight1', [hidden_size, act_size], dtype)]

    biases = [tf.get_variable('bias0', [hidden_size], dtype),
              tf.get_variable('bias1', [act_size], dtype)]
    return weights, biases


weights, biases = get_params()
params = weights + biases


def perceptron(i, x):
    return tf.matmul(x, weights[i]) + biases[i]


observation_ph = tf.placeholder(dtype, obs_size, name='observation')
hidden = perceptron(0, tf.expand_dims(observation_ph, 0))
output = perceptron(1, tf.sigmoid(hidden))
action_dist = tf.nn.softmax(output, name='action_dist')
tf_action = tf.squeeze(tf.multinomial(action_dist, 1), name='action')

prob = tf.gather(tf.squeeze(action_dist), tf_action, name='prob')
tf_scores = tf.gradients(tf.log(prob), params, name='gradient')

gradient_phs = [tf.placeholder(dtype, param.get_shape(),
                               name=get_base_name(param) + '_placeholder')
                for param in params]
train_op = optimizer.apply_gradients(zip(gradient_phs, params))

epochs = 200
batches = 500


def zeros_like_params():
    return [np.zeros(param.get_shape()) for param in params]


show_off = False
with tf.Session() as sess:
    tf.global_variables_initializer().run()

    # update every epoch
    for e in range(epochs):
        gradients = zeros_like_params()
        mean_reward = 0

        # average over batches
        for b in range(batches):
            observation = env.reset()
            done = False
            cumulative_scores = zeros_like_params()
            cumulative_reward = 0
            t = 0

            # steps
            while not done:
                if show_off:
                    env.render()
                action, new_scores = sess.run([tf_action, tf_scores],
                                              {observation_ph: observation.squeeze()})
                observation, reward, done, info = env.step(action)

                cumulative_reward += reward
                for old_score, new_score in zip(cumulative_scores, new_scores):
                    old_score += new_score
                t += 1

            mean_reward += cumulative_reward / batches
            for gradient, cumulative_score in zip(gradients, cumulative_scores):
                gradient -= cumulative_score * cumulative_reward / batches

        print("Epoch: {}. Reward: {}".format(e, mean_reward))
        if mean_reward > 199:
            show_off = True
        feed_dict = dict(zip(gradient_phs, gradients))
        sess.run(train_op, feed_dict)
