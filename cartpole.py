import gym.spaces
import numpy as np
import tensorflow as tf
from tensorflow.python.training.adam import AdamOptimizer
from tensorflow.python.training.ftrl import FtrlOptimizer
from tensorflow.python.training.rmsprop import RMSPropOptimizer

import reinforce
from mlp import MLP

learning_rate = 0.1
optimizers = {
    1: AdamOptimizer(learning_rate),
    2: FtrlOptimizer(learning_rate),
    3: RMSPropOptimizer(learning_rate),
}
optimizer = optimizers[1]

env = gym.make('CartPole-v1')
dtype = env.observation_space.low.dtype

model = MLP([5], dtype)
reinforce.train(env, model, dtype)
