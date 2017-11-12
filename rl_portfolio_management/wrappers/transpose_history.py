import gym.spaces
import numpy as np


class TransposeHistory(gym.Wrapper):
    """Transpose history."""

    def __init__(self, env, axes=(2, 1, 0)):
        super().__init__(env)
        self.axes = axes

        hist_space = self.observation_space.spaces["history"]
        hist_shape = hist_space.shape
        self.observation_space = gym.spaces.Dict({
            'history': gym.spaces.Box(
                hist_space.low.min(),
                hist_space.high.max(),
                (hist_shape[axes[0]], hist_shape[axes[1]], hist_shape[axes[2]])
            ),
            'weights': self.observation_space.spaces["weights"]
        })

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        state["history"] = np.transpose(state["history"], self.axes)
        return state, reward, done, info

    def reset(self):
        state = self.env.reset()
        state["history"] = np.transpose(state["history"], self.axes)
        return state
