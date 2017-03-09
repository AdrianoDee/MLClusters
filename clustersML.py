#import keras
from __future__ import division

import tensorflow as tf
tf.python.control_flow_ops = tf

import numpy as np
import collections
import clusterutils as cu
import keras as k

from keras.models import Sequential,Model
from keras.layers import Dense, Activation,Input, Dense, Dropout, merge
from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.layers import Dense, Dropout, Activation, Convolution2D, Convolution3D, Flatten, MaxPooling2D, MaxPooling3D, Merge
from keras.callbacks import History
from sklearn.metrics import roc_curve, auc
#import ROOT

import cPickle as pkl
import matplotlib.pyplot as plt
import gzip
#from tensorflow.python.framework import dtypes

import numpy
import os
import sys
from time import gmtime, strftime

from io import StringIO

sanitize = False
filters = {}
batchsize = 50
epoch = 1

size = 8

numlabels=17

lDel = False
batchsize=30

fileOutput = strftime("./outputs/%Y-%m-%d_%H_%M_%S_outputs.pkl.gz", gmtime())

data    = cu.datasetload(fileslimit=1)
filters = {"detCounterIn":[0.],"detCounterOut":[1.]}
datafil = cu.datafiltering(filters,data,sanitize=True,sanratio=0.5)
(data_train, label_train) = cu.clustersInput(datafil)

clustercnn = Sequential()
clustercnn.add(Convolution3D(32,4,4,1,input_shape = (8,8,2,1), activation = 'sigmoid',border_mode='valid'))
clustercnn.add(MaxPooling3D(pool_size=(2,2,2),border_mode='valid'))
clustercnn.add(Flatten())
clustercnn.add(Dense(512, activation='sigmoid'))
clustercnn.add(Dropout(0.5))
clustercnn.add(Dense(numlabels, activation='softmax'))
clustercnn.compile(loss='binary_crossentropy',optimizer='sgd', metrics=['accuracy'])
early = EarlyStopping(monitor='val_acc', patience=4, verbose=1, mode='auto')
history = clustercnn.fit(data_train, label_train,validation_split=0.33, nb_epoch=epoch, batch_size=batchsize,shuffle=True,callbacks=[early])
predicted_target = clustercnn.predict(data_train)

label_score = label_train[:,0]
preds_score = predicted_target[:,0]

fpr, tpr, _ = roc_curve(label_score, preds_score)
print("\n ROC Area : %g"%(auc(fpr, tpr)))

with gzip.open(fileOutput, 'wb') as f:
    result = (batchsize,filters,predicted_target,label_train)
    print("\n Saving to file " + fileOutput)
    pkl.dump(result, f, -1)
