import gym
import numpy as np

from rl_portfolio_management.environments import PortfolioEnv
from rl_portfolio_management.wrappers import ConcatStates, SoftmaxActions, TransposeHistory


def test_concat():
    env = gym.make("CryptoPortfolioEIIE-v0")
    env = ConcatStates(env)
    obs = env.reset()
    assert len(obs.shape) == 3
    action = env.action_space.sample()
    action /= action.sum()
    obs, rew, done, info = env.step(action)
    assert len(obs.shape) == 3


def test_softmax():
    env = gym.make("CryptoPortfolioEIIE-v0")
    env = SoftmaxActions(env)
    obs = env.reset()
    # should be no problem with actions that don't sum to one
    action = env.action_space.sample() * 100
    obs, rew, done, info = env.step(action)


def test_transpose():
    env0 = gym.make("CryptoPortfolioEIIE-v0")
    obs0 = env0.reset()
    transposed_shape = np.transpose(obs0["history"], (2, 1, 0)).shape

    env = gym.make("CryptoPortfolioEIIE-v0")
    env = TransposeHistory(env)
    obs = env.reset()
    env.observation_space.contains(obs)
    assert obs["history"].shape == transposed_shape
    # should be no problem with actions that don't sum to one
    action = env.action_space.sample()
    action /= action.sum()
    obs, rew, done, info = env.step(action)
    assert obs["history"].shape == transposed_shape
