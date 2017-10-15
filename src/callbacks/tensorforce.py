from tqdm import tqdm_notebook
import tensorflow as tf
import numpy as np
import pandas as pd
import os

from ..util import sharpe, MDD


class TensorBoardLogger(object):
    """
    Log scalar and histograms/distributions to tensorboard.
    Usage:
    ```
    logger = TensorBoardLogger(log_dir = '/tmp/test')
    for i in range(10):
        logger.log(
            logs=dict(
                float_test=np.random.random(),
                int_test=np.random.randint(0,4),
            ),
            histograms=dict(
                actions=np.random.randint(0,3,size=np.random.randint(5,20))
            )
        )
    ```
    Ref: https://github.com/fchollet/keras/blob/master/keras/callbacks.py
    Url: https://gist.github.com/wassname/b692f8e8686655011618dfbe8d8a9e3f
    """

    def __init__(self, log_dir, session=None, episode=0):
        self.log_dir = log_dir
        self.writer = tf.summary.FileWriter(self.log_dir)
        self.episode = episode
        print(
            'TensorBoardLogger started. Run `tensorboard --logdir={}` to visualize'.
            format(os.path.abspath(self.log_dir)))

        self.histograms = {}
        self.histogram_inputs = {}
        self.session = session or tf.get_default_session() or tf.Session()

    def log(self, logs={}, histograms={}, episode=None):
        episode = episode or self.episode
        # scalar logging
        for name, value in logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = name
            self.writer.add_summary(summary, episode)

        # histograms
        for name, value in histograms.items():
            if name not in self.histograms:
                # make a tensor with no fixed shape
                self.histogram_inputs[name] = tf.Variable(
                    value, validate_shape=False)
                self.histograms[name] = tf.summary.histogram(
                    name, self.histogram_inputs[name])

            input_tensor = self.histogram_inputs[name]
            summary = self.histograms[name]
            summary_str = summary.eval(
                session=self.session, feed_dict={input_tensor.name: value})
            self.writer.add_summary(summary_str, episode)

        self.writer.flush()
        self.episode += 1


# Callback function printing episode statistics
class EpisodeFinished(object):
    """Logger callback for tensorforce runner"""

    def __init__(self, log_intv):
        self.log_intv = log_intv
        self.portfolio_values = []
        self.mdds = []
        self.sharpes = []

    def __call__(self, r):
        if len(r.environment.gym.sim.infos):
            self.portfolio_values.append(
                r.environment.gym.sim.infos[-1]['portfolio_value'])

            df = pd.DataFrame(r.environment.gym.sim.infos)
            self.mdds.append(MDD(df.rate_of_return + 1))
            self.sharpes.append(sharpe(df.rate_of_return))
        if r.episode % self.log_intv == 0:
            print(
                "Finished episode {ep} after {ts} timesteps (reward: {reward: 2.4f} [{rewards_min: 2.4f}, {rewards_max: 2.4f}]) portfolio_value: {portfolio_value: 2.4f} [{portfolio_value_min: 2.4f}, {portfolio_value_max: 2.4f}] mdd={mdd: 2.2%} sharpe={sharpe: 2.2f}".
                format(
                    ep=r.episode,
                    ts=r.timestep,
                    reward=np.mean(r.episode_rewards[-self.log_intv:]),
                    rewards_min=np.min(r.episode_rewards[-self.log_intv:]),
                    rewards_max=np.max(r.episode_rewards[-self.log_intv:]),
                    portfolio_value=np.mean(self.portfolio_values[
                        -self.log_intv:]),
                    portfolio_value_min=np.min(self.portfolio_values[
                        -self.log_intv:]),
                    portfolio_value_max=np.max(self.portfolio_values[
                        -self.log_intv:]),
                    mdd=np.mean(self.mdds[-self.log_intv:]),
                    sharpe=np.mean(self.sharpes[-self.log_intv:]), ))
        return True


class EpisodeFinishedTQDM(EpisodeFinished):
    """Logger for tensorforce using tqdm_notebook for jupyter-notebook."""

    def __init__(self,
                 episodes,
                 log_intv,
                 session=None,
                 log_dir=None,
                 episode=0):
        """
        log_intv - print the mean metrics every log_intv episodes
        """
        super().__init__(log_intv=log_intv)
        self.episodes = episodes
        self.progbar = tqdm_notebook(
            desc='', total=episodes, leave=True, mininterval=5)

        # tensorboard
        if log_dir:
            self.log_dir = log_dir
        else:
            self.log_dir = '/tmp/StepsProgressBar'
        self.tensor_board_logger = TensorBoardLogger(
            self.log_dir, session=session, episode=episode)

    def __call__(self, r):
        super().__call__(r)
        oai_env = r.environment.gym.unwrapped
        exploration = r.agent.exploration.get('action0', lambda x, y: 0)(
            r.episode, np.sum(r.episode_lengths))
        desc = "reward: {reward: 2.4f} [{rewards_min: 2.4f}, {rewards_max: 2.4f}], portfolio_value: {portfolio_value: 2.4f} [{portfolio_value_min: 2.4f}, {portfolio_value_max: 2.4f}] expl={exploration: 2.2%}".format(
            reward=np.mean(r.episode_rewards[-1:]),
            rewards_min=np.min(r.episode_rewards[-1:]),
            rewards_max=np.max(r.episode_rewards[-1:]),
            portfolio_value=np.mean(self.portfolio_values[-1:]),
            portfolio_value_min=np.min(self.portfolio_values[-1:]),
            portfolio_value_max=np.max(self.portfolio_values[-1:]),
            exploration=exploration)
        self.progbar.desc = desc
        self.progbar.update(1)  # update

        # log to tensorboard
        logs = dict(
            episode_rewards=r.episode_rewards[-1],
            episode_lengths=r.episode_lengths[-1],
            episode_time=r.episode_times[-1],
            portfolio_value=np.mean(self.portfolio_values[-1:]),
            portfolio_value_min=np.min(self.portfolio_values[-1:]),
            portfolio_value_max=np.max(self.portfolio_values[-1:]),
            exploration=exploration)
        df_info = pd.DataFrame(oai_env.infos)
        ep_infos = df_info.mean().to_dict()
        logs.update(ep_infos)
        self.tensor_board_logger.log(
            logs=logs,
            histograms=dict(
                returns=df_info['return'].values,
                portfolio_value=df_info.portfolio_value.values,
                market_value=df_info.market_value.values, ),
            episode=r.episode)
        return True
