from tqdm import tqdm_notebook
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import datetime

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
        print('TensorBoardLogger started. Run `tensorboard --logdir={}` to visualize'.format(
            os.path.dirname(os.path.abspath(log_dir))))

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
                "Finished episode {ep} after {ts} timesteps (reward: {reward: 2.6f} [{rewards_min: 2.6f}, {rewards_max: 2.6f}]) portfolio_value: {portfolio_value: 2.4f} [{portfolio_value_min: 2.4f}, {portfolio_value_max: 2.4f}] mdd={mdd: 2.2%} sharpe={sharpe: 2.2f}".
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


class EpisodeFinishedTQDM(object):
    """Logger for tensorforce using tqdm_notebook for jupyter-notebook."""

    def __init__(self,
                 steps,
                 log_intv,
                 session=None,
                 log_dir=None,
                 episode=0, mean_of=10,
                 save_dir=None, save_every=10
                 ):
        """
        log_intv - print the mean metrics every log_intv episodes
        """
        # super().__init__(log_intv=log_intv)
        self.steps = steps
        self.mean_of = mean_of
        self.log_intv = log_intv
        self.save_dir = save_dir
        self.save_every = save_every
        self.progbar = tqdm_notebook(
            desc='', total=steps, unit='steps', leave=True, mininterval=3)

        # tensorboard
        if log_dir:
            self.log_dir = log_dir
        else:
            self.log_dir = '/tmp/StepsProgressBar'
        self.tensor_board_logger = TensorBoardLogger(
            self.log_dir, session=session, episode=episode)

    def __call__(self, r):
        # super().__call__(r)
        oai_env = r.environment.gym.unwrapped
        df_info = pd.DataFrame(oai_env.infos)
        rate_of_return = df_info["rate_of_return"].values
        weights = dict(zip(oai_env.src.asset_names, np.round(oai_env.sim.w0, 4).tolist()))
        exploration = r.agent.exploration.get('action', lambda x, y: 0)(
            r.episode, np.sum(r.episode_timesteps))

        desc = "ep reward: {reward: 2.8f} [{rewards_min: 2.8f}, {rewards_max: 2.8f}], portfolio_value: {portfolio_value: 2.4f} mdd={mdd:2.2%} sharpe={sharpe:2.4f}, expl={exploration: 2.2%} eps={episode:} weights={weights:}".format(
            reward=np.mean(df_info["reward"].mean()),
            rewards_min=np.min(df_info["reward"].min()),
            rewards_max=np.max(df_info["reward"].max()),
            episode=r.episode,
            portfolio_value=oai_env.sim.p0,
            mdd=MDD(rate_of_return + 1),
            sharpe=sharpe(rate_of_return),
            exploration=exploration,
            weights=weights)
        self.progbar.desc = desc
        self.progbar.update(r.episode_timesteps[-1])

        # print every now and again
        if r.episode % self.log_intv == 0:
            print(desc)
            oai_env.render(mode='notebook')

        # log to tensorboard
        logs = dict(
            # episode_rewards=r.episode_rewards[-1],
            episode_timesteps=r.episode_timesteps[-1],
            episode_time=r.episode_times[-1],
            portfolio_value=oai_env.sim.p0,
            exploration=exploration,
            sharpe=sharpe(rate_of_return),
            mdd=MDD(rate_of_return + 1),
        )

        # logs info means
        ep_infos = df_info.mean().to_dict()
        logs.update(ep_infos)

        # and some arrays
        histograms = {}
        # histograms = dict(
        #     return_hist=df_info['market_return'].values,
        #     rate_of_return_hist=rate_of_return,
        #     portfolio_value_hist=df_info.portfolio_value.values,
        #     market_value_hist=df_info.market_value.values,
        #     last_weight_hist=oai_env.sim.w0,
        # )
        # # and each weight dist
        # for i, w in enumerate(oai_env.sim.w0):
        #     histograms[oai_env.src.asset_names[i] + '_hist'] = df_info['weight_' + oai_env.src.asset_names[i]]

        self.tensor_board_logger.log(
            logs=logs,
            histograms=histograms,
            episode=r.episode)

        # save sometimes
        if self.save_dir and (r.episode % self.save_every == 0):
            r.agent.save_model(self.save_dir)
            print('saved to', self.save_dir)

        # and halt if we are beyond out max steps
        return r.timestep < self.steps
