import tensorflow as tf
import gym.spaces

import ntm

env = gym.make('Pendulum-v0')

dtype = env.observation_space.low.dtype

assert (isinstance(env.action_space, gym.spaces.Box))
obs_shape, = env.observation_space.shape
act_shape, = env.action_space.shape

hidden_size = obs_shape
observation_tf = tf.placeholder(dtype, obs_shape)
observation_np = env.reset()

model = ntm.Cell(hidden_size)
state_tuple = model.zero_state(1, dtype)
h, state_tuple = model(observation_tf, state_tuple)
out_weight = tf.get_variable("out", [hidden_size, act_shape], dtype)
action_tf = tf.matmul(h, out_weight)

for _ in range(20):
    with tf.Session() as sess:
        for t in range(100):
            env.render()
            sess.run(tf.global_variables_initializer())
            action_np = sess.run(action_tf,
                                 feed_dict={observation_tf: observation_np.squeeze()})
            observation_np, reward, done, info = env.step(action_np)

            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break
