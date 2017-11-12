import gym.wrappers
import numpy as np


class TransposeHistory(gym.Wrapper):
    """Transpose history."""

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        state["history"] = np.transpose(state["history"], (2, 1, 0))
        return state, reward, done, info

    def reset(self):
        state = self.env.reset()
        state["history"] = np.transpose(state["history"], (2, 1, 0))
        return state
