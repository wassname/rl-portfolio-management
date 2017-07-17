import pandas as pd
import numpy as np
from src.environments.portfolio import PortfolioEnv

def test_env_outputs():
    df = pd.read_hdf('./data/poliniex_30m.hf', key='train')
    env = PortfolioEnv(df=df)

    action = np.random.random(env.action_space.shape)
    action /= action.sum()

    obs1, reward, done, info = env.step(action)
    obs2 = env.reset()
    assert obs1.shape == obs2.shape

    assert np.isfinite(reward)
    assert not done
    for v in info.values():
        assert np.isfinite(v)

def test_portfolio_env():
    df = pd.read_hdf('./data/poliniex_30m.hf', key='train')
    asset_names = df.columns.levels[0]

    env = PortfolioEnv(df=df)
    np.random.seed(0)

    obs = env.reset()
    for _ in range(20):

        w = np.random.random((len(asset_names)))
        w /= w.sum()

        obs, reward, done, info = env.step(w)
        assert not done

    df_info = pd.DataFrame(env.infos)
    final_value = df_info.portfolio_value.iloc[-1]
    assert final_value > 0.75, 'should retain most value with 20 random steps'


def test_portfolio_env_hold():
    df = pd.read_hdf('./data/poliniex_30m.hf', key='train')
    asset_names = df.columns.levels[0]

    np.random.seed(0)
    env = PortfolioEnv(df=df)
    env.reset()
    for _ in range(5):
        w = np.array([1.0] + [0] * (len(asset_names) - 1))
        obs, reward, done, info = env.step(w)

    df = pd.DataFrame(env.infos)
    assert df.portfolio_value.iloc[-1] > 0.9999, 'portfolio should retain value if holding bitcoin'


def test_return_not_scaled():
    df = pd.read_hdf('./data/poliniex_30m.hf', key='train')
    np.random.seed(0)
    env1 = PortfolioEnv(df=df, scale=True)
    np.random.seed(0)
    env0 = PortfolioEnv(df=df, scale=False)
    a = env0.src._data.xs('return', axis=1, level='Price').tail(5)
    b = env1.src._data.xs('return', axis=1, level='Price').tail(5)
    assert (a == b).all().all(), 'returns should not be scaled'

    assert (env.step(env.sim.w0)[0][:,-1,3]==1).all(), 'close prices should be scaled to one at end'
