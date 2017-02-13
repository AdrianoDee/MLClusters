#import keras
from __future__ import division

import tensorflow as tf
#tf.python.control_flow_ops = tf

import numpy as np
import collections
import kerutils as ku
import nn_graphics
import keras as ker
from keras.models import Sequential,Model
from keras.layers import Dense, Activation,Input, Dense, Dropout, merge
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import model_from_json, Sequential
from keras.layers import Dense, Dropout, Activation, Convolution2D, Convolution3D, Flatten, MaxPooling2D, MaxPooling3D, Merge
from keras.callbacks import History
from sklearn.metrics import roc_curve, auc
#import ROOT
import pickle

import matplotlib.pyplot as plt
#from tensorflow.python.framework import dtypes

import numpy
import os
import sys

from io import StringIO


dIn = 0
dOu = 1
mIn = 144
mOu = 80
size = 8
pile = 2
#epoch = 1
lDel = False

#trainS = ['clusterstrain0_1.txt','clusterstrainlabels0_1.txt']
#testS  = ['clusterstest0_1.txt','clusterstestlabels0_1.txt']

# trainS = ['clusters_0_1.txt','clusterslabels0_1.txt']
# testS  = ['clusters_0_1.txt','clusterslabels0_1.txt']

# trainS = ['dets_0_1_mods_176_144train.txt','dets_0_1_mods_176_144labelstrain.txt']
# testS  = ['dets_0_1_mods_176_144test.txt','dets_0_1_mods_176_144labelstest.txt']

#trainS = ['dets_0_1_mods_192_176train.txt','dets_0_1_mods_192_176labelstrain.txt']
#testS  = ['dets_0_1_mods_192_176test.txt','dets_0_1_mods_192_176labelstest.txt']

batchsizes = [30]
#batchsizes = [250,500,1000,10000]#,500,1000,10000]
epochs = [1,10,20,50]

#(X_train, y_train) = ku.doublets_read_data_sets_PU(trainsets=trainS,testsets=testS,train_dir='./Hits/datasets/',cols=size,rows=size,stack=pile)

datas = ['0001','0002'];

(X_train, y_train), (X_test, y_test) = ku.doubletsReadPostMod(detIn=dIn,detIn=dOu,modIn=mIn,modOu=mOu,datasets=datas,filedir='./Hits/',cols=size,rows=size,stack=pile)

clustercnn = Sequential()
#clustercnn.add(Convolution2D(64,3,1,input_shape = (8,8,2), activation = 'sigmoid',border_mode='valid'))
clustercnn.add(Convolution3D(32,5,2,1,input_shape = (8,8,2,1), activation = 'sigmoid',border_mode='valid'))
#clustercnn.add(MaxPooling2D(pool_size=(2,2),border_mode='valid'))
clustercnn.add(MaxPooling3D(pool_size=(2,2,2),border_mode='valid'))
#clustercnn.add(Convolution2D(32,2,2,activation = 'sigmoid'))
#clustercnn.add(MaxPooling2D(pool_size=(2,2),border_mode='valid'))
#clustercnn.add(Convolution2D(64,3,1,input_shape = (8,8,2), activation = 'sigmoid',border_mode='valid'))
#clustercnn.add(Convolution3D(16,2,2,1,input_shape = (8,8,2,1), activation = 'sigmoid',border_mode='valid'))
#clustercnn.add(MaxPooling2D(pool_size=(2,2),border_mode='valid'))
#clustercnn.add(MaxPooling3D(pool_size=(2,2,2),border_mode='valid'))
clustercnn.add(Flatten())
clustercnn.add(Dense(512, activation='relu'))
clustercnn.add(Dropout(0.25))
clustercnn.add(Dense(2, activation='softmax'))
#clustercnn.add(Activation('softmax'))
#clustercnn.add(Dense(2, activation='sigmoid'))
#clustercnn.add(Dense(1, activation='softmax'))

#(X_train, y_train), (X_test, y_test) = ku.base_read_data_sets(trainsets=trainS,testsets=testS,train_dir='./datasets/doublets/01/',cols=size,rows=size,stack=pile,l_delimiter=lDel)
clustercnn.compile(loss='binary_crossentropy',optimizer='sgd', metrics=['accuracy'])
#early = EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto')

results = []

for batchsize in batchsizes:
    for epoch in epochs:
        print('-- Training with %g (%g epochs)'%(batchsize,epoch))
        history = History()
        #early = ker.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.01, patience=0, verbose=0, mode='auto')
        history = clustercnn.fit(X_train, y_train,validation_split=0.33, nb_epoch=epoch, batch_size=batchsize,shuffle=True,verbose=0)#,callbacks=[early])

        predicted_target = clustercnn.predict(X_train)
        loss_test = clustercnn.evaluate(X_train,predicted_target,batch_size=batchsize)
        #print(predicted_target.shape())
        #print(predicted_target)
        #print(y_test)

        print(y_train)
        print(predicted_target)

        y_test_score = y_train[:,1]

        falses = (y_test_score == 0.0).sum();
        trues = (y_test_score == 1.0).sum();

        print(' \n -- Testing with dataset with %g false and %g true'%(falses,trues))

        predicted_target_score = predicted_target[:,1]

        truePosivites = []
        falsePositive = []
        values = []

        #thresholds = np.linspace(np.amin(predicted_target_score), 1.0, num=10000)

        #print (predicted_target_score)
        #print (np.amin(predicted_target_score)*0.99999)
        #print (thresholds)

        # TruePositives = []
        # FalsePositives = []

        fpr, tpr, _ = roc_curve(y_test_score, predicted_target_score)
        print('  - ROC Area : %g'%(auc(fpr, tpr)))


        # fpr, tpr, Thresholds = metrics.roc_curve(y_test_score, predicted_target_score, pos_label=2)
        #
        # print(fpr)
        #
        # print(tpr)
        #
        # print(Thresholds)
        #
        # areaSK = metrics.auc(fpr,tpr)

        #print(falses)


        # for i in range(len(thresholds)):
        #     low_values_indices_true = y_test_score >0.5
        #     low_values_indices = predicted_target_score > thresholds[i]
        #     #print(low_values_indices)
        #     TruePositive = 0
        #     FalsePositive = 0
        #     TrueNegative = 0
        #     FalseNegative = 0
        #     #All = len(low_values_indices)
        #
        #     for j in range(len(low_values_indices)):
        #         if(low_values_indices_true[j]==False and low_values_indices[j]==False):
        #             TrueNegative+=1
        #         #if(low_values_indices_true[j]==True and low_values_indices[j]==True):
        #             #TruePositive+=1
        # #                  trues +=1
        #         #if(low_values_indices_true[j]==True and low_values_indices[j]==False):
        #         #    FalseNegative+=1
        # #                  trues += 1
        #         if(low_values_indices_true[j]==False and low_values_indices[j]==True):
        #             FalsePositive+=1
        # #                  falses +=1
        #
        #
        #     TruePositives.append(TruePositive/trues)
        #     FalsePositives.append(FalsePositive/falses)
        #
        # #result = (epoch,batchsize,TruePositives,FalsePositives)
        # #results.append(result);
        #
        # #falsePositive = np.asarray(falsePositive,dtype=np.float32)
        #
        #
        # # plt.plot(FalsePositives,TruePositives)
        # # plt.savefig('mod_batch_%g_epoch_%g_trues.png'%(batchsize,epoch))

        TruePositives = np.asarray(tpr,dtype=np.float32)
        FalsePositives = np.asarray(fpr,dtype=np.float32)

        truesPath = './outputs/dets_' + str(dIn) + '_' + str(dOu) + '_mods_' + str(mIn) + '_' + str(mOu) + '_batch_' + str(batchsize) + 'epoch' + str(epoch) + '_trues.out'
        falsesPath = './outputs/dets_' + str(dIn) + '_' + str(dOu) + '_mods_' + str(mIn) + '_' + str(mOu) + '_batch_' + str(batchsize) + 'epoch' + str(epoch) + '_falses.out'

        np.savetxt(truesPath,(TruePositives))
        np.savetxt(falsesPath,(FalsePositives))

        result = (epochs,batchsize,history.history['acc'],history.history['loss'],history.history['val_loss'],history.history['val_acc'],TruePositives,FalsePositives)
        results.append(result)

filePKName = './outputs/dets' + str(dIn) + '_' + str(dOu) + '_mods_' + str(mIn) + '_' + str(mOu) + '_batch_' + str(batchsize) + 'epoch' + str(epoch) + '.pik'

with open(filePKName, 'wb') as f:
  pickle.dump(results, f, -1)
        #np.savetxt('mod_batch_%g_epoch_%g_thresh.out'%(batchsize,epoch),(thresholds))

        #print('  - ROC Area  = %g '%(auc(FalsePositives,TruePositives)))



              #plt.savefig('trueFalse.png')

              #plt.figure(2)
              #plt.plot(truePosivites,thresholds)
              #plt.savefig('threshs.png')

              #for i in range(len(histories)):
              #      nn_graphics.show_losses(histories[i])
              #      plt.plot(histories[i].history['acc'])
              #      plt.plot(histories[i].history['val_acc'])
              #      plt.title('model accuracy batchsize %d'%(batchsize))
              #      plt.ylabel('accuracy')
              #      plt.xlabel('epoch')
              #      plt.legend(['train', 'test'], loc='upper left')
              #      plt.savefig('myfig')

              #summarize history for loss
              #print(clustercnn.summary())

              #print('Threshold %g \n TruePositive %g \n FalsePositive %g \n TrueNegative %g \n FalseNegative %g'%(thresholds[i],TruePositive,FalsePositive,TrueNegative,FalseNegative))

              #predicted_target_score_thr[low_values_indices] = 0.0
              #predicted_target_score_thr[high_values_indices] = 1.0
              #print('Threshold %g Ones %g zeros %g '%(thresholds[i],(predicted_target_score <= thresholds[i]).sum(),(predicted_target_score >= thresholds[i]).sum()))
              #print(low_values_indices)
              #print(high_values_indices)
              #TruePositive, FalsePositive, TrueNegative, FalseNegative = ku.accuracy_measure(y_test_score, predicted_target_score_thr)
              #value = [thresholds[i],TruePositive, FalsePositive, TrueNegative, FalseNegative]
              #values.append(value)
              #truePosivites.append(TruePositive)
              #falsePositive.append(FalsePositive)
