# silence the many message from the gym logger, yuck
import logging
logger = logging.getLogger('gym.envs.tests.spec_list')
logger.setLevel(logging.ERROR)

import pytest
import pandas as pd
import numpy as np
import gym
from gym.envs.tests import test_envs

from rl_portfolio_management.environments import PortfolioEnv, env_specs


@pytest.mark.parametrize("spec_id", env_specs)
def test_gym_env(spec_id):
    """Run openai test to check for repeatable, observation shape, etc."""
    spec = gym.envs.spec(spec_id)
    test_envs.test_env(spec)


@pytest.mark.parametrize("spec_id", env_specs)
def test_env_outputs(spec_id):
    """Check outputs."""
    env = gym.envs.spec(spec_id).make()
    env.seed(0)

    action = env.action_space.sample()
    action /= action.sum()

    obs1, reward, done, info = env.step(action)
    obs2 = env.reset()

    assert obs1["history"].shape == obs2["history"].shape, 'rest and step should output same shaped observations'
    assert obs1["weights"].shape == obs2["weights"].shape, 'rest and step should output same shaped observations'
    assert env.observation_space.contains(
        obs1), 'state should be within observation space'
    assert np.isfinite(reward), 'reward should be finite'
    assert not done
    for k, v in info.items():
        assert np.isfinite(v), 'env info item %s=%s should be finite' % (k, v)
        assert isinstance(
            v, (int, float)), 'env info item %s=%s should be int or float' % (k, v)


@pytest.mark.parametrize("spec_id", env_specs)
def test_portfolio_env_random_agent(spec_id):
    """Test random actions for 20 steps."""
    env = gym.envs.spec(spec_id).make()
    env.seed(0)

    obs = env.reset()
    for i in range(20):

        action = env.action_space.sample()
        action /= action.sum()

        obs, reward, done, info = env.step(action)
        assert env.observation_space.contains(
            obs), 'state should be within observation space'
        assert np.isfinite(reward), 'reward should be finite'
        assert not done, "shouldn't be done after %s steps" % i

    df_info = pd.DataFrame(env.infos)
    final_value = df_info.portfolio_value.iloc[-1]
    assert final_value > 0.75, 'should retain most value with 20 random steps'
    assert final_value < 1.10, 'should retain most value with 20 random steps'


@pytest.mark.parametrize("spec_id", env_specs)
def test_portfolio_env_hold(spec_id):
    """Test that holding cash gives stable value."""
    env = gym.envs.spec(spec_id).make()
    env.seed(0)
    env.reset()
    for _ in range(5):
        action = env.action_space.sample() * 0
        action[0] = 1
        obs, reward, done, info = env.step(action)

    df = pd.DataFrame(env.infos)
    assert df.portfolio_value.iloc[-1] > 0.9999, 'portfolio should retain value if holding bitcoin'
    assert df.portfolio_value.iloc[-1] < 1.01, 'portfolio should retain value if holding bitcoin'

def test_scaled_non_price_cols():
    """Test env with scaled option."""
    df = pd.read_hdf('./data/poloniex_30m_vol.hf', key='train')
    env1 = PortfolioEnv(df=df, scale=True, window_length=40000)
    env1.seed(0)
    obs1 = env1.reset()

    nb_cols = len(env1.src._data.columns.levels[1]) - 1 # minus cash...
    means = obs1["history"][:, :, :].reshape((-1, nb_cols)).mean(0)
    stds = obs1["history"][:, :, :].reshape((-1, nb_cols)).std(0)

    non_price_means = means[3:]

    # if normalized: for a large window, mean non_prices should be near mean=0, std=1
    non_price_std = stds[3:]
    np.testing.assert_almost_equal(non_price_means, [0, 0], decimal=1, err_msg='non price columns should be normalized to be close to one')
    np.testing.assert_almost_equal(non_price_std, [1, 1], decimal=1, err_msg='non price columns should be normalized to be close to one')


def test_scaled():
    """Test env with scaled option."""
    df = pd.read_hdf('./data/poloniex_30m.hf', key='train')

    env0 = PortfolioEnv(df=df, scale=False, window_length=40)
    env0.seed(0)
    obs0 = env0.reset()

    env1 = PortfolioEnv(df=df, scale=True, window_length=40)
    env1.seed(0)
    obs1 = env1.reset()

    assert (obs0["history"] != obs1["history"]).all(), 'scaled and non-scaled data should differ'

    # if scaled by last opening price: for a small window, mean prices should be near 1
    nb_cols = len(env1.src._data.columns.levels[1]) - 1
    means = obs1["history"][:, :, :].reshape((-1, nb_cols)).mean(0)
    price_means = means[:3]
    np.testing.assert_almost_equal(price_means, [1, 1, 1], decimal=1, err_msg='prices should be normalized to be close to one')


@pytest.mark.parametrize("spec_id", env_specs)
def test_invalid_actions(spec_id):
    """Test that holding cash gives stable value."""
    env = gym.envs.spec(spec_id).make()
    env.seed(0)
    env.reset()

    valid_action = env.action_space.sample()
    valid_action /= valid_action.sum()

    invalid_actions = [
        # invalid dont sum to one
        valid_action * 0,
        valid_action * -1,
        valid_action * 10,
        # invalid shapes
        valid_action[:1],
        [valid_action],
        # invalid values
        valid_action * np.nan,
        valid_action * np.inf,
    ]
    for action in invalid_actions:
        try:
            env.step(action)
        except Exception as e:
            pass
        else:
            raise Exception('Expected error for invalid action %s' % action)

@pytest.mark.parametrize("spec_id", env_specs)
def test_costs(spec_id):
    """Test that simple transaction have the cost we expect."""
    env = gym.envs.spec(spec_id).make()
    env.seed(0)

    env.reset()
    obs, reward, done, info = env.step(np.array([1, 0, 0, 0]))
    obs, reward, done, info = env.step(np.array([0, 1, 0, 0]))
    np.testing.assert_almost_equal(info['cost'], env.sim.cost, err_msg='trading 100% cash for asset1 should cost 1*trading_cost')

    obs, reward, done, info = env.step(np.array([0, 0, 1, 0]))
    np.testing.assert_almost_equal(info['cost'], env.sim.cost * 2, err_msg='trading 100% asset1 for asset2 should cost 2*trading_cost')
