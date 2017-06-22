import keras as k
from keras.callbacks import EarlyStopping, ModelCheckpoint
from collections import deque

import numpy as np

class EarlyStoppingAvg(EarlyStopping):

    def __init__(self, monitor='val_loss',
                 min_delta=0, patience=0, verbose=0, avgsteps=10, mode='auto'):

        EarlyStopping.__init__(self, monitor='val_loss',
                     min_delta=min_delta, patience=patience, verbose=verbose, mode='auto')

        self.avgsteps = avgsteps
        self.monitors = deque([])

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0

        self.best = np.Inf if self.monitor_op == np.less else -np.Inf

        self.first = 0

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn(
                'Early stopping conditioned on metric `%s` '
                'which is not available. Available metrics are: %s' %
                (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
            )

        steps = len(self.monitors)
        #first step normal not avg early stopping
        if steps < self.avgsteps:
            if steps == 0:
                self.best = current
                self.monitors.append(current)
            else:
                # if there are less values than the avg step required just add it and do the average

                # updating the average with the new value
                self.best = (self.best * steps + current)/(steps + 1)
                self.monitors.append(current)
        else:
            if steps == self.avgsteps:

                # check current with the average
                if self.monitor_op(current - self.min_delta, self.best):
                    self.wait = 0
                else:
                    if self.wait >= self.patience:
                        self.stopped_epoch = epoch
                        self.model.stop_training = True
                    self.wait += 1

                # print(self.wait)
                # print(self.patience)
                # updating the average with the new value
                self.best = (self.best * steps - self.monitors.popleft() + current)/steps
                self.monitors.append(current)

        print('\nEarly stopping avg %05f' % (self.best))
        # print(self.monitors)

    # def on_train_end(self, logs=None):
    #     if self.stopped_epoch > 0 and self.verbose > 0:
    #         print('Epoch %05d: early stopping' % (self.stopped_epoch))
