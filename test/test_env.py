import pandas as pd
import numpy as np
from src.environments.portfolio import PortfolioEnv


def test_portfolio_env():
    df = pd.read_pickle('./data/poliniex_30m_train.pickle')
    asset_names = df.columns.levels[0]
    # action
    w = np.random.random((len(asset_names)))
    w /= w.sum()

    env = PortfolioEnv(df=df)
    env.reset()
    obs, reward, done, info = env.step(w)


def test_portfolio_env_hold():
    df = pd.read_pickle('./data/poliniex_30m_train.pickle')
    asset_names = df.columns.levels[0]

    env = PortfolioEnv(df=pd.read_pickle('./data/poliniex_30m_train.pickle'))
    env.reset()
    for _ in range(5):
        w = np.array([1.0] + [0] * (len(asset_names) - 1))
        obs, reward, done, info = env.step(w)

    df = pd.DataFrame(info)
    assert df.portfolio_value.iloc[-1] > 0.9999
