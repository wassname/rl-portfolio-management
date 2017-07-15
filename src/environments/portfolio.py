import numpy as np

import gym
import gym.spaces

from ..config import eps
from ..data.utils import normalize, random_shift, scale_to_start


class DataSrc(object):
    """Acts as data provider for each new episode."""

    def __init__(self, df, steps=252, scale=True, augument=0.00):
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


        # add return/y1 as last col
        pairs = df.columns.levels[0]
        for pair in pairs:
            x = df[pair].close
            df[pair, "return"] = (x + eps*2) / (x.shift() + eps)
        df = df[1:]

        # data processing
        if scale:
            # df = (df - df.mean(0) + eps) / (df.max(0) - df.min(0) + eps)
            df = df.apply(lambda x: normalize(x))

        # get rid of NaN's
        df = df.fillna(method="pad")
        df.replace(np.nan, 0, inplace=True)

        self._data = df.copy()
        self.asset_names = self._data.columns.levels[0].tolist()

        self.reset()

    def _step(self):
        tdx = self.data.index[self.step]
        obs = self.data.xs(
            tdx, axis=0).unstack().as_matrix()  # shape = (prices, assets)

        self.step += 1
        done = self.step >= self.steps
        return obs, done

    def reset(self):
        self.step = 0

        # get data for this episode
        self.idx = np.random.randint(
            low=0, high=len(self._data.index) - self.steps)
        data = self._data[self.idx:self.idx + self.steps].copy()

        # scale each run to the begining of the episode so they look the same
        if self.scale:
            data = data.apply(lambda x: scale_to_start(x))

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

    def __init__(self, asset_names=[], trading_cost_bps=0.0025):
        self.cost = trading_cost_bps
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
        # y1 = v1 / v0 # (equation 1) price relative vector / return
        w0 = self.w0
        p0 = self.p0

        dw1 = (y1 * w0) / (np.dot(y1, w0) + eps)  # (eq7) weights evolve into

        mu1 = self.cost * (
            np.abs(dw1 - w1)).sum()  # (eq16) cost to change portfolio

        p1 = p0 * (1 - mu1) * np.dot(y1, w0)  # (eq11) final portfolio value
        p1 = np.clip(p1, 0, np.inf)
        # print(dict(mu1=mu1,p1=p1,dw1=dw1,y1=y1))

        rho1 = p1 / p0 - 1  # rate of returns
        r1 = np.log((p1 + eps) / (p0 + eps))  # log rate of return

        # rememeber for next step
        self.w0 = w1
        self.p0 = p1

        # if we run out of money, we're done
        done = p1 == 0

        info = {
            "log_reward": r1,
            "portfolio_value": p1,
            "returns": y1,
            "rate_of_return": rho1,
            "weights": w1,
            "cost": mu1,
        }
        self.infos.append(info)
        return r1, self.infos, done  # reward

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

    def __init__(self,
                 df,
                 steps=256,
                 scale=True,
                 augument=0.00,
                 trading_cost_bps=0.0025):
        self.src = DataSrc(df=df, steps=steps, scale=scale, augument=augument)

        self.sim = PortfolioSim(
            asset_names=self.src.asset_names,
            trading_cost_bps=trading_cost_bps)

        # openai gym attributes
        self.action_space = gym.spaces.Box(
            0, 1, shape=len(self.src.asset_names))

        gym.spaces.Box(
            np.array([
                self.src._data[pair].min().values
                for pair in self.src.asset_names
            ]),
            np.array([
                self.src._data[pair].max().values
                for pair in self.src.asset_names
            ]), )

        self._reset()

    def _step(self, action):
        np.testing.assert_almost_equal(
            np.sum(action), 1.0, 4, err_msg='action should be sum to 1')
        observation, done1 = self.src._step()

        y1 = observation[:, -1]  # relative price vector (return)
        reward, info, done2 = self.sim._step(action, y1)

        return observation, reward, done1 + done2, info

    def _reset(self):
        self.sim.reset()
        self.src.reset()
        return self.step(self.sim.w0)

    def _render(self, mode='human', close=False):
        pass
