import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import gym
import gym.spaces

from ..config import eps
from ..data.utils import normalize, random_shift, scale_to_start


class DataSrc(object):
    """Acts as data provider for each new episode."""

    def __init__(self, df, steps=252, scale=True, augument=0.00, window_length=50):
        """
        DataSrc.

        df - csv for data frame index of timestamps
             and multi-index columns levels=[['LTCBTC'],...],['close',...]]
        steps - total steps in episode
        scale - scale the data for each episode
        augument - fraction to augument the data by
        """
        self.steps = steps + 1
        self.augument = augument
        self.scale = scale
        self.window_length = window_length

        df = df.copy()

        # get rid of NaN's
        df.replace(np.nan, 0, inplace=True)
        df = df.fillna(method="pad")

        self._data = df.copy()
        self.asset_names = self._data.columns.levels[0].tolist()

        self.reset()

    def _step(self):
        # get observation matrix from dataframe
        data_window = self.data.iloc[self.step:self.step + self.window_length].copy()

        # (eq 18) prices are divided by open price
        # While the paper says open/close, it only makes sense with close/open
        if self.scale:
            open = data_window.xs('open', axis=1, level='Price')
            data_window = data_window.divide(open.iloc[-1], level='Pair')
            data_window = data_window.drop('open', axis=1, level='Price')

        # convert to matrix (window, assets, prices)
        obs = np.array([data_window[asset].as_matrix() for asset in self.asset_names])

        self.step += 1
        done = self.step >= self.steps
        return obs, done

    def reset(self):
        self.step = 0

        # get data for this episode
        self.idx = np.random.randint(
            low=self.window_length, high=len(self._data.index) - self.steps)
        data = self._data[self.idx - self.window_length:self.idx + self.steps + 1].copy()

        # augument data to prevent overfitting
        data = data.apply(lambda x: random_shift(x, self.augument))

        self.data = data


class PortfolioSim(object):
    """
    Portfolio management sim.

    Params:
    - cost e.g. 0.0025 is max in Poliniex

    Based of [Jiang 2017](https://arxiv.org/abs/1706.10059)
    """

    def __init__(self, asset_names=[], steps=128, trading_cost=0.0025, time_cost=0.0):
        self.cost = trading_cost
        self.time_cost = time_cost
        self.steps = steps
        self.asset_names = asset_names
        self.reset()

    def _step(self, w1, y1):
        """
        Step.

        w1 - new action of portfolio weights - e.g. [0.1,0.9, 0.0]
        y1 - price relative vector also called return
            e.g. [1.0, 0.9, 1.1]
        Numbered equations are from https://arxiv.org/abs/1706.10059
        """
        w0 = self.w0
        p0 = self.p0

        dw1 = (y1 * w0) / (np.dot(y1, w0) + eps)  # (eq7) weights evolve into

        mu1 = self.cost * (
            np.abs(dw1 - w1)).sum()  # (eq16) cost to change portfolio

        p1 = p0 * (1 - mu1) * np.dot(y1, w0)  # (eq11) final portfolio value

        p1 = p1 * (1 - self.time_cost)  # we can add a cost to holding

        p1 = np.clip(p1, 0, np.inf)
        # if p1 > 1e3:
            # raise Exception("really? check this")
        # print(dict(mu1=mu1,p1=p1,dw1=dw1,y1=y1))

        rho1 = p1 / p0 - 1  # rate of returns
        r1 = np.log((p1 + eps) / (p0 + eps))  # log rate of return
        reward = r1 / self.steps  # (22) average logarithmic cumulated return

        # rememeber for next step
        self.w0 = w1
        self.p0 = p1

        # if we run out of money, we're done
        done = p1 == 0

        info = {
            "reward": reward,
            "log_return": r1,
            "portfolio_value": p1,
            "returns": y1,
            "rate_of_return": rho1,
            "weights": w1,
            "cost": mu1,
        }
        self.infos.append(info)
        return reward, info, done

    def reset(self):
        self.infos = []
        self.w0 = np.array([1.0] + [0.0] * (len(self.asset_names) - 1))
        self.p0 = 1.0


class PortfolioEnv(gym.Env):
    """
    An environment for financial portfolio management.

    Financial portfolio management is the process of constant redistribution of a fund into different
    financial products.

    Based on [Jiang 2017](https://arxiv.org/abs/1706.10059)
    """

    metadata = {'render.modes': ['human']}

    def __init__(self,
                 df,
                 steps=256,
                 scale=True,
                 augument=0.00,
                 trading_cost=0.0025,
                 time_cost=0.00,
                 window_length=50,
                 ):
        """
        An environment for financial portfolio management.

        Params:
            df - data frame index of timestamps
                 and multi-index columns levels=[['LTCBTC'],...],['close',...]]
            steps - steps in episode
            scale - scale data and each episode (except return)
            augument - fraction to randomly shift data by
            trading_cost - cost of trade as a fraction
            time_cost - cost of holding as a fraction
            window_length - how many past observations to return
        """
        self.src = DataSrc(df=df, steps=steps, scale=scale, augument=augument, window_length=window_length)

        self.sim = PortfolioSim(
            asset_names=self.src.asset_names,
            trading_cost=trading_cost,
            time_cost=time_cost,
            steps=steps)

        # openai gym attributes
        # action will be the portfolio weights from 0 to 1 for each asset
        self.action_space = gym.spaces.Box(
            0, 1, shape=len(self.src.asset_names))

        # get the observation space from the data min and max
        self.observation_space = gym.spaces.Box(
            0,
            1,
            (
                len(self.src.asset_names) - 1,
                window_length,
                len(self.src._data.columns.levels[1]) - 1
            )
        )
        self._reset()

    def _step(self, action):
        """
        Step the env.

        Actions should be portfolio [cash_bias, w0...]
        - Where wn is a portfolio weight from 0 to 1. The first is cash_bias
        - cn is the portfolio conversion weights see PortioSim._step for description
        """
        np.testing.assert_almost_equal(
            action.shape,
            (len(self.sim.asset_names),)
        )

        # normalise just in case
        action = np.clip(action, 0, 1)

        weights = action  # [cash_bias, w0, w1...]
        weights /= (weights.sum() + eps)
        weights[0] += np.clip(1  -weights.sum(), 0, 1)  # so if weights are all zeros we normalise to [1,0...]

        assert ((action >= 0) * (action <= 1)).all(), 'all action values should be between 0 and 1. Not %s' % action
        np.testing.assert_almost_equal(
            np.sum(weights), 1.0, 3, err_msg='weights should sum to 1. action="%s"' % weights)

        observation, done1 = self.src._step()

        y1 = observation[:, -1, 0]  # relative price vector (open/close)
        reward, info, done2 = self.sim._step(weights, y1)
        observation = observation[1:, :, :]  # remove cash columns

        # add dates
        info['index'] = self.src.data.index[self.src.step]
        info['steps'] = self.src.step
        self.infos.append(info)

        # for keras-rl it only wants a single dict of numberic values FIXME
        if 'weights' in info:
            del info['weights']
        info['returns'] = info['returns'].mean()
        del info['index']

        return observation, reward, done1 + done2, info

    def _reset(self):
        self.sim.reset()
        self.src.reset()
        self.infos = []
        action = self.sim.w0
        observation, reward, done, info = self.step(action)
        return observation

    def _render(self, mode='human', close=False):
        if close:
            return
        pass
        # if mode == 'ansi':
        #     info = self.sim.infos
        #     df_info = pd.DataFrame(info)
        #     df_returns = pd.DataFrame(
        #         np.stack(df_info.returns.values),
        #         columns=self.sim.asset_names,
        #         index=df_info.index)
        #     df_weights = pd.DataFrame(
        #         np.stack(df_info.weights.values),
        #         columns=self.sim.asset_names,
        #         index=df_info.index)
        #     print('Gain:\n', df_weights.iloc[-1] * (df_returns.iloc[-1] - 1))
        # elif mode=='human':
        #     self.plot()

    def plot(self):

        info = self.sim.infos
        # add dates
        for i in range(len(info)):
            info[i]['index'] = self.src.data.index[:self.src.step][i]

        # make dataframes
        df_info = pd.DataFrame(info)
        df_info.index = df_info['index']
        del df_info['index']

        df_returns = pd.DataFrame(
            np.stack(df_info.returns.values),
            columns=self.sim.asset_names,
            index=df_info.index)
        df_weights = pd.DataFrame(
            np.stack(df_info.weights.values),
            columns=self.sim.asset_names,
            index=df_info.index)
        df_quantity = (df_returns * df_weights)
        df_info = df_info.drop(['returns', 'weights'], axis=1)

        # plots
        # (df_returns-1).plot(title='returns')
        # plt.show()

        # df_info.portfolio_value.plot(title='portfolio_value', ylim=[0, 2])
        # plt.show()
        df_quantity.plot.area(title='portfolio value')
        plt.show()

        df_weights.plot.area(title='portfolio weights', ylim=[0, 1])
        plt.show()

        # (df_returns.iloc[-1]-1).plot('bar', title='change')
        # plt.show()

        # (df_quantity.iloc[-1]).plot('pie', title='step distribution')
        # plt.show()
