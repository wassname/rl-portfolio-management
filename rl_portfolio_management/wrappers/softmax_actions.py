import gym.wrappers

from ..util import softmax


class SoftmaxActions(gym.Wrapper):
    """
    Environment wrapper to softmax actions.

    Usage:
        env = gym.make('Pong-v0')
        env = SoftmaxActions(env)

    Ref: https://github.com/openai/gym/blob/master/gym/wrappers/README.md

    """

    def step(self, action):
        # also it puts it in a list
        # print('action in softmax', type(action))
        if isinstance(action, list):
            # action = action[0]
            # Junchen: change to fit new Horizon framework
            action = action

        if isinstance(action, dict):
            action = list(action[k] for k in sorted(action.keys()))

        action = softmax(action, t=1)
        # print('action in softmax after', action.shape)  

        return self.env.step(action)
