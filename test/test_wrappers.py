import gym

from rl_portfolio_management.environments import PortfolioEnv, env_specs
from rl_portfolio_management.wrappers import ConcatStates, SoftmaxActions


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
