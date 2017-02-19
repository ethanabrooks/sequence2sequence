import tensorflow as tf
import gym

import ntm

env = gym.make('CartPole-v0')
hidden_size = 100

# def in_layer(input):
#     return input * tf.get_variable("in_weight", [env.observation_space.shape])


shape = env.observation_space.shape
model = ntm.Cell(shape)


def out_layer(input):
    shape = env.action_space.shape
    print(shape)
    return input * tf.get_variable("out_weight", [hidden_size, shape])


state = tf.Variable(expected_shape=model.state_size)
for i_episode in range(20):
    observation = env.reset()
    for t in range(1):
        env.render()
        # action = env.action_space.sample()
        action = model(observation, state)
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break
