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
def test_src_outputs(spec_id):
    """check we arn't giving future prices to model"""
    env = gym.envs.spec(spec_id).make()
    X0, y0, _ = env.src._step()
    _, y1, _ = env.src._step()
    # so relative price vector for calulating this steps returns is
    # y(t) = y = v(t)/v(t-1)
    # while y(t-1) = y_last =  v(t-1)/v(t-2)
    # the last steps price data is
    # X(t-1) = ..., v(t-3)/v(t-1), v(t-2)/v(t-1), v(t-1)/v(t-1)]
    # so 1/X[-2]==y[-2] as a unit test
    np.testing.assert_almost_equal(1 / X0[:, -2, 0], y0[1:])
    # on the other hand these should not
    try:
        np.testing.assert_almost_equal(1 / X0[:, -2, 0], y1[1:])
    except AssertionError:
        pass
    else:
        raise AssertionError(
            "These should not be equal, or we are giving future prices to the model")

    # also make sure that the last price X, is 1 because we scaled it by itself as in eq 18
    assert (X0[:, -1, 0] == 1).all()


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
    market_value = df_info.market_value.iloc[-1]
    np.testing.assert_allclose(final_value, market_value, rtol=0.1,
                               err_msg='should be similar to market values after 20 random steps')


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
    env1 = PortfolioEnv(df=df, scale=True, window_length=len(df) - 300)
    env1.seed(0)
    obs1 = env1.reset()

    nb_cols = len(env1.src.features)
    nb_price_cols = len(env1.src.price_columns)
    means = obs1["history"].reshape((-1, nb_cols)).mean(0)
    stds = obs1["history"].reshape((-1, nb_cols)).std(0)

    non_price_means = means[nb_price_cols:]

    # if normalized: for a large window, mean non_prices should be near mean=0, std=1
    non_price_std = stds[nb_price_cols:]
    np.testing.assert_almost_equal(non_price_means, [
                                   0, 0], decimal=1, err_msg='non price columns should be normalized to be close to one')
    np.testing.assert_allclose(non_price_std, [
                               1, 1], rtol=0.1, err_msg='non price columns should be normalized to be close to one')


def test_scaled():
    """Test env with scaled option."""
    df = pd.read_hdf('./data/poloniex_30m_vol.hf', key='train')

    env0 = PortfolioEnv(df=df, scale=False, window_length=40)
    env0.seed(0)
    obs0 = env0.reset()

    env1 = PortfolioEnv(df=df, scale=True, window_length=40)
    env1.seed(0)
    obs1 = env1.reset()

    nb_price_cols = len(env1.src.price_columns)
    assert (obs0["history"][:, :, :nb_price_cols] != obs1["history"][:, :,
                                                                     :nb_price_cols]).all(), 'scaled and non-scaled data should differ'

    # if scaled by last opening price: for a small window, mean prices should be near 1
    np.testing.assert_allclose(obs1["history"][:, -1, :nb_price_cols], 1,
                               rtol=0.1, err_msg='last prices should be normalized to be close to one')


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
    np.testing.assert_almost_equal(
        info['cost'], env.sim.cost, err_msg='trading 100% cash for asset1 should cost 1*trading_cost')

    obs, reward, done, info = env.step(np.array([0, 0, 1, 0]))
    np.testing.assert_almost_equal(
        info['cost'], env.sim.cost * 2, err_msg='trading 100% asset1 for asset2 should cost 2*trading_cost')
