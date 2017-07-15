from rl.callbacks import Callback
from rl.callbacks import TrainIntervalLogger
from keras import backend as K
import warnings
from tqdm import tqdm_notebook
import timeit
import numpy as np

#
# class ReduceLROnPlateau(Callback):
#     """Reduce learning rate when a metric has stopped improving.
#     Models often benefit from reducing the learning rate by a factor
#     of 2-10 once learning stagnates. This callback monitors a
#     quantity and if no improvement is seen for a 'patience' number
#     of epochs, the learning rate is reduced.
#     # Example
#         ```python
#         reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
#                                       patience=5, min_lr=0.001)
#         model.fit(X_train, Y_train, callbacks=[reduce_lr])
#         ```
#     # Arguments
#         monitor: quantity to be monitored.
#
#         factor: factor by which the learning rate will
#             be reduced. new_lr = lr * factor
#         patience: number of epochs with no improvement
#             after which learning rate will be reduced.
#         verbose: int. 0: quiet, 1: update messages.
#         mode: one of {auto, min, max}. In `min` mode,
#             lr will be reduced when the quantity
#             monitored has stopped decreasing; in `max`
#             mode it will be reduced when the quantity
#             monitored has stopped increasing; in `auto`
#             mode, the direction is automatically inferred
#             from the name of the monitored quantity.
#         epsilon: threshold for measuring the new optimum,
#             to only focus on significant changes.
#         cooldown: number of epochs to wait before resuming
#             normal operation after lr has been reduced.
#         min_lr: lower bound on the learning rate
#
#     Modified to work with keras-rl from https://github.com/fchollet/keras/blob/master/keras/callbacks.py#L792
#     Untested
#     """
#
#     def __init__(self, monitor='val_loss', factor=0.1, patience=10,
#                  verbose=0, mode='auto', epsilon=1e-4, cooldown=0, min_lr=0):
#         super(Callback, self).__init__()
#
#         self.monitor = monitor
#         if factor >= 1.0:
#             raise ValueError('ReduceLROnPlateau '
#                              'does not support a factor >= 1.0.')
#         self.factor = factor
#         self.min_lr = min_lr
#         self.epsilon = epsilon
#         self.patience = patience
#         self.verbose = verbose
#         self.cooldown = cooldown
#         self.cooldown_counter = 0  # Cooldown counter.
#         self.wait = 0
#         self.best = 0
#         self.mode = mode
#         self.monitor_op = None
#         self._reset()
#
#     def _reset(self):
#         """Resets wait counter and cooldown counter.
#         """
#         if self.mode not in ['auto', 'min', 'max']:
#             warnings.warn('Learning Rate Plateau Reducing mode %s is unknown, '
#                           'fallback to auto mode.' % (self.mode),
#                           RuntimeWarning)
#             self.mode = 'auto'
#         if (self.mode == 'min' or
#                 (self.mode == 'auto' and 'acc' not in self.monitor)):
#             self.monitor_op = lambda a, b: np.less(a, b - self.epsilon)
#             self.best = np.Inf
#         else:
#             self.monitor_op = lambda a, b: np.greater(a, b + self.epsilon)
#             self.best = -np.Inf
#         self.cooldown_counter = 0
#         self.wait = 0
#         self.lr_epsilon = self.min_lr * 1e-4
#
#     def on_train_begin(self, logs=None):
#         self._reset()
#
#     def on_episode_end(self, epoch, logs=None):
#         logs = logs or {}
#         logs['lr'] = K.get_value(self.model.model.optimizer.lr)
#         current = logs.get(self.monitor)
#         if current is None:
#             warnings.warn(
#                 'Reduce LR on plateau conditioned on metric `%s` '
#                 'which is not available. Available metrics are: %s' %
#                 (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
#             )
#
#         else:
#             if self.in_cooldown():
#                 self.cooldown_counter -= 1
#                 self.wait = 0
#
#             if self.monitor_op(current, self.best):
#                 self.best = current
#                 self.wait = 0
#             elif not self.in_cooldown():
#                 if self.wait >= self.patience:
#                     old_lr = float(K.get_value(self.model.model.optimizer.lr))
#                     if old_lr > self.min_lr + self.lr_epsilon:
#                         new_lr = old_lr * self.factor
#                         new_lr = max(new_lr, self.min_lr)
#                         K.set_value(self.model.model.optimizer.lr, new_lr)
#                         if self.verbose > 0:
#                             print('\nEpoch %05d: reducing learning rate to %s.' % (epoch, new_lr))
#                         self.cooldown_counter = self.cooldown
#                         self.wait = 0
#                 self.wait += 1
#
#     def in_cooldown(self):
#         return self.cooldown_counter > 0


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
