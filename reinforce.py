import gym.spaces
import numpy as np
import tensorflow as tf
from tensorflow.python.training.adam import AdamOptimizer
from tensorflow.python.training.ftrl import FtrlOptimizer
from tensorflow.python.training.rmsprop import RMSPropOptimizer


# learning_rate = 0.1
# optimizers = {
#     1: AdamOptimizer(learning_rate),
#     2: FtrlOptimizer(learning_rate),
#     3: RMSPropOptimizer(learning_rate),
# }
# optimizer = optimizers[2]

#
def get_base_name(var):
    return var.name.split(':')[0]


class Reinforce:
    def __init__(self, env, model,
                 optimizer=AdamOptimizer(learning_rate=.1),
                 epochs=200,
                 batches=500):
        self._env = env
        self._model = model
        dtype = env.observation_space.low.dtype

        assert (isinstance(env.action_space, gym.spaces.Discrete))
        obs_size, = env.observation_space.shape
        act_size = env.action_space.n

        model.create_in_layer(obs_size)
        model.create_out_layer(act_size)

        # get action
        self._observation = tf.placeholder(dtype, obs_size, name='observation')
        logits = model.forward(tf.expand_dims(self._observation, 0))
        action_dist = tf.nn.softmax(logits, name='action_dist')
        self._action = tf.squeeze(tf.multinomial(logits, 1), name='action')

        # get score
        prob = tf.gather(tf.squeeze(action_dist), self._action, name='prob')
        self._scores = tf.gradients(tf.log(prob), model.params, name='gradient')

        # apply gradient
        self._gradients = [tf.placeholder(dtype, param.get_shape(),
                                          name=get_base_name(param) + '_placeholder')
                           for param in model.params]
        self._train_op = optimizer.apply_gradients(zip(self._gradients, model.params))

        self._epochs = epochs
        self._batches = batches

    def train(self):
        def zeros_like_params():
            return [np.zeros(param.get_shape()) for param in self._model.params]

        show_off = False
        with tf.Session() as sess:
            tf.global_variables_initializer().run()

            # update every epoch
            for e in range(self._epochs):
                gradients = zeros_like_params()
                mean_reward = 0

                # average over batches
                for b in range(self._batches):
                    observation = self._env.reset()
                    done = False
                    t = 0
                    cumulative_reward = 0
                    cumulative_scores = zeros_like_params()

                    # steps
                    while not done:
                        if show_off:
                            self._env.render()
                        action, new_scores = sess.run([self._action, self._scores],
                                                      {self._observation: observation.squeeze()})
                        observation, reward, done, info = self._env.step(action)

                        cumulative_reward += reward
                        for old_score, new_score in zip(cumulative_scores, new_scores):
                            old_score += new_score
                        t += 1

                    mean_reward += cumulative_reward / self._batches
                    for gradient, cumulative_score in zip(gradients, cumulative_scores):
                        gradient -= cumulative_score * cumulative_reward / self._batches

                print("Epoch: {}. Reward: {}".format(e, mean_reward))
                feed_dict = dict(zip(self._gradients, gradients))
                sess.run(self._train_op, feed_dict)
                if mean_reward >= 200:
                    show_off = True
