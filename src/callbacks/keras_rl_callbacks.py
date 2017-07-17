from rl.callbacks import Callback
from rl.callbacks import TrainIntervalLogger
from keras import backend as K
import warnings
from tqdm import tqdm_notebook
import timeit
import numpy as np

class TrainIntervalLoggerTQDMNotebook(TrainIntervalLogger):
    """TrainIntervalLogger using tqdm_notebook for jupyter-notebook."""

    def reset(self):
        self.interval_start = timeit.default_timer()
        self.metrics = []
        self.infos = []
        self.info_names = None
        self.episode_rewards = []

    def on_train_begin(self, logs):
        self.progbar = tqdm_notebook(desc='', total=self.params['nb_steps'], leave=True, mininterval=0.5)
        self.train_start = timeit.default_timer()
        self.metrics_names = self.model.metrics_names
        print('Training for {} steps ...'.format(self.params['nb_steps']))

    def on_step_end(self, step, logs):
        if self.info_names is None:
            self.info_names = logs['info'].keys()
        values = ''
        for k, v in logs['info'].items():
            if isinstance(v, float) or isinstance(v, int):
                values += '{}: {:2.4f}, '.format(k, v)
            elif isinstance(v, str):
                pass
            elif hasattr(v, 'tolist'):
                values += '{}: {:2.4f}, '.format(k, v.tolist())
        # values = ', '.join(['{}: {:2.2f}'.format(k, v) for k, v in logs['info'].items() if not isinstance(v, str)])
        self.progbar.desc = 'reward={reward: 2.4f} info=({values:})'.format(
            reward=logs['reward'],
            values=values
        )
        self.progbar.update(1)  # update
        self.step += 1
        self.metrics.append(logs['metrics'])
        if len(self.info_names) > 0:
            self.infos.append([logs['info'][k] for k in self.info_names])
