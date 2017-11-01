import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pprint import pprint
import logging
import os
import tempfile
import time
import gym
import gym.spaces

from ..config import eps
from ..data.utils import normalize, random_shift, scale_to_start
from ..util import MDD as max_drawdown, sharpe, softmax
from ..callbacks.notebook_plot import LivePlotNotebook

logger = logging.getLogger(__name__)


class DataSrc(object):
    """Acts as data provider for each new episode."""

    def __init__(self, df, steps=252, scale=True, augment=0.00, window_length=50):
        """
        DataSrc.

        df - csv for data frame index of timestamps
             and multi-index columns levels=[['LTCBTC'],...],['open','low','high','close']]
        steps - total steps in episode
        scale - scale the data for each episode
        augment - fraction to augment the data by
        """
        self.steps = steps + 1
        self.augment = augment
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
        data_window = self.data.iloc[self.step:self.step +
                                     self.window_length].copy()

        # (eq 18) prices are divided by open price
        # While the paper says open/close, it only makes sense with close/open
        if self.scale:
            open = data_window.xs('open', axis=1, level='Price')
            data_window = data_window.divide(open.iloc[-1], level='Pair')
            data_window = data_window.drop('open', axis=1, level='Price')

        # convert to matrix (window, assets, prices)
        obs = np.array([data_window[asset].as_matrix()
                        for asset in self.asset_names])

        self.step += 1
        done = bool(self.step >= self.steps)
        return obs, done

    def reset(self):
        self.step = 0

        # get data for this episode
        self.idx = np.random.randint(
            low=self.window_length, high=len(self._data.index) - self.steps)
        data = self._data[self.idx -
                          self.window_length:self.idx + self.steps + 1].copy()

        # augment data to prevent overfitting
        data = data.apply(lambda x: random_shift(x, self.augment))

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

        p1 = np.clip(p1, 0, np.inf)  # can't have negative holdings in this model (no shorts)

        rho1 = p1 / p0 - 1  # rate of returns
        r1 = np.log((p1 + eps) / (p0 + eps))  # (eq10) log rate of return
        reward = r1 / self.steps  # (eq22) immediate reward is log rate of return scaled by episode length

        # remember for next step
        self.w0 = w1
        self.p0 = p1

        # if we run out of money, we're done
        done = bool(p1 == 0)

        # should only return single values, not list
        info = {
            "reward": reward,
            "log_return": r1,
            "portfolio_value": p1,
            "market_return": y1.mean(),
            "rate_of_return": rho1,
            "weights_mean": w1.mean(),
            "weights_std": w1.std(),
            "cost": mu1,
        }
        # record weights and prices
        for i in range(len(self.asset_names)):
            info['weight_' + self.asset_names[i]] = w1[i]
            info['price_' + self.asset_names[i]] = y1[i]

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

    metadata = {'render.modes': ['notebook', 'ansi']}

    def __init__(self,
                 df,
                 steps=256,
                 scale=True,
                 augment=0.00,
                 trading_cost=0.0025,
                 time_cost=0.00,
                 window_length=50,
                 output_mode='EIIE'
                 ):
        """
        An environment for financial portfolio management.

        Params:
            df - csv for data frame index of timestamps
                 and multi-index columns levels=[['LTCBTC'],...],['open','low','high','close']]
            steps - steps in episode
            scale - scale data and each episode (except return)
            augment - fraction to randomly shift data by
            trading_cost - cost of trade as a fraction,  e.g. 0.0025 corresponding to max rate of 0.25% at Poloniex (2017)
            time_cost - cost of holding as a fraction
            window_length - how many past observations to return
            output_mode: decides observation shape
                - 'EIIE' for (assets, window, 3)
                - 'atari' for (window, window, 3) (assets is padded)
                - 'mlp' for (assets*window*3)
        """
        self.src = DataSrc(df=df, steps=steps, scale=scale,
                           augment=augment, window_length=window_length)
        self._plot = self._plot2 = self._plot3 = None
        self.output_mode = output_mode
        self.sim = PortfolioSim(
            asset_names=self.src.asset_names,
            trading_cost=trading_cost,
            time_cost=time_cost,
            steps=steps)

        # openai gym attributes
        # action will be the portfolio weights from 0 to 1 for each asset
        self.action_space = gym.spaces.Box(
            0.0, 1.0, shape=len(self.src.asset_names))

        # get the observation space from the data min and max
        if output_mode == 'EIIE':
            obs_shape = (
                len(self.src.asset_names) - 1,  # don't observe cash column
                window_length,
                len(self.src._data.columns.levels[1]) - 1
            )
        elif output_mode == 'atari':
            obs_shape = (
                window_length,  # don't observe cash column
                window_length,
                len(self.src._data.columns.levels[1]) - 1
            )
        elif output_mode == 'mlp':
            obs_shape = (len(self.src.asset_names) - 1) * window_length * \
                (len(self.src._data.columns.levels[1]) - 1)
        else:
            raise Exception('Invalid value for output_mode: %s' %
                            self.output_mode)

        self.observation_space = gym.spaces.Box(
            0,
            2 if scale else 1,  # if scale=True observed price changes return could be large fractions
            obs_shape
        )
        self._reset()

    def _step(self, action):
        """
        Step the env.

        Actions should be portfolio [w0...]
        - Where wn is a portfolio weight from 0 to 1. The first is cash_bias
        - cn is the portfolio conversion weights see PortioSim._step for description
        """
        logger.debug('action: %s', action)

        weights = np.clip(action, 0.0, 1.0)
        weights /= weights.sum() + eps

        # Sanity checks
        np.testing.assert_almost_equal(
            action.shape,
            (len(self.sim.asset_names),),
            err_msg='Action should contain %s floats, not %s' % (len(self.sim.asset_names), action.shape)
        )
        assert ((action >= 0) * (action <= 1)
                ).all(), 'all action values should be between 0 and 1. Not %s' % action
        np.testing.assert_almost_equal(
            np.sum(weights), 1.0, 3, err_msg='weights should sum to 1. action="%s"' % weights)

        observation, done1 = self.src._step()

        y1 = observation[:, -1, 0]  # relative price vector (open/close)
        reward, info, done2 = self.sim._step(weights, y1)

        # Bit of a HACK. We want it to know last steps portfolio weights
        # but don't want to make dual inputs so I'll replace the oldest data
        # with them
        weight_insert_shape = (observation.shape[0], observation.shape[2])
        observation[:, 0, :] = np.ones(
            weight_insert_shape) * weights[:, np.newaxis]

        # remove cash columns, they are just meaningless values
        observation = observation[1:, :, :]

        # calculate return for buy and hold a bit of each asset
        info['market_value'] = np.cumprod(
            [inf["market_return"] for inf in self.infos + [info]])[-1]
        # add dates
        info['date'] = self.src.data.index[self.src.step].timestamp()
        info['steps'] = self.src.step

        self.infos.append(info)

        # reshape output
        if self.output_mode == 'EIIE':
            pass
        elif self.output_mode == 'atari':
            padding = observation.shape[1] - observation.shape[0]
            observation = np.pad(observation, [[0, padding], [
                                 0, 0], [0, 0]], mode='constant')
        elif self.output_mode == 'mlp':
            observation = observation.flatten()

        return observation, reward, done1 or done2, info

    def _reset(self):
        self.sim.reset()
        self.src.reset()
        self.infos = []
        action = self.sim.w0
        observation, reward, done, info = self.step(action)
        return observation

    def _render(self, mode='notebook', close=False):
        # if close:
            # return
        if mode == 'ansi':
            pprint(self.infos[-1])
        elif mode == 'notebook':
            self.plot_notebook(close)

    def plot_notebook(self, close=False):
        """Live plot using the jupyter notebook rendering of matplotlib."""

        if close:
            self._plot = self._plot2 = self._plot3 = None

        df_info = pd.DataFrame(self.infos)
        df_info.index = pd.to_datetime(df_info["date"], unit='s')

        # plot prices and performance
        if not self._plot:
            self._plot_dir = os.path.join(tempfile.gettempdir(), 'notebook_plot_prices_' + str(time.time()))
            self._plot = LivePlotNotebook(
                self._plot_dir, title='prices & performance', labels=self.sim.asset_names + ["Portfolio"], ylabel='value')
        x = df_info.index
        y_portfolio = df_info["portfolio_value"]
        y_assets = [df_info['price_' + name].cumprod()
                    for name in self.sim.asset_names]
        self._plot.update(x, y_assets + [y_portfolio])

        # plot portfolio weights
        if not self._plot2:
            self._plot_dir2 = os.path.join(tempfile.gettempdir(), 'notebook_plot_weights_' + str(time.time()))
            os.makedirs(self._plot_dir2)
            self._plot2 = LivePlotNotebook(
                self._plot_dir2, labels=self.sim.asset_names, title='weights', ylabel='weight')
        ys = [df_info['weight_' + name] for name in self.sim.asset_names]
        self._plot2.update(x, ys)

        if close:
            self._plot = self._plot2 = self._plot3 = None
