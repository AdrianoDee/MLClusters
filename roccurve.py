import os
import numpy as np
from sklearn import metrics
import pickle

falsPath = "half_batch_5_epoch_10_falses.out";
truePath = "half_batch_5_epoch_10_thresh.out";

with open(falsPath, 'rb') as f:
    falses = np.loadtxt(f,delimiter='\n',dtype = np.float32)

with open(truePath, 'rb') as f:
    trues = np.loadtxt(f,delimiter='\n',dtype = np.float32)

print(metrics.auc(falses,trues))
