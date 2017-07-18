#import keras
from __future__ import division

import imp

import tensorflow as tf
tf.control_flow_ops = tf

import numpy as np
import collections
import clusterutils as cu
import keras as k

from keras.models import Sequential,Model
from keras.layers import Dense, Activation,Input, Dense, Dropout, merge
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers import Dense, Dropout, Activation, Convolution2D, Convolution3D, Flatten, MaxPooling2D, MaxPooling3D, Merge
from keras.callbacks import History

import time

from keras import backend as K
from keras.engine.topology import Layer
import numpy as np

sklearnflag = True

try:
    imp.find_module('sklearn')
    from sklearn.metrics import roc_curve, auc
except ImportError:
    print("No SKLEARN")
    sklearnflag = False

#import ROOT

import cPickle as pkl
#import matplotlib.pyplot as plt
import gzip
#from tensorflow.python.framework import dtypes

import os
import sys
from time import gmtime, strftime

from io import StringIO

size = 8
numlabels=2


if __name__ == "__main__":

    args = sys.argv[1:]

    sanitize = False
    angC = False
    BandW = False
    writedata = False

    epochs = 1000
    batchsize=100
    fileL = 100

    print("Starting Clusters CNN Training with: ")

    for i in range(1,len(sys.argv)):
        if(sys.argv[i] == "-bw"):
            BandW = True
            print("- B&W configuration")
        else:
            if (sys.argv[i] == "-a"):
                angC = True
                print("- angular correction")
            else:
                if(sys.argv[i] == "-e"):
                    if i+1 < sys.argv[0]:
                        i = i + 1
                        epochs = int(sys.argv[i])
                else:
                    if(sys.argv[i] == "-b"):
                        if i+1 < sys.argv[0]:
                            i = i + 1
                            batchsize = int(sys.argv[i])
                    else:
                        if(sys.argv[i] == "-f"):
                            if i+1 < sys.argv[0]:
                                i = i + 1
                                fileL = int(sys.argv[i])
                        else:
                            if(sys.argv[i] == "-w"):
                                writedata = True
			    else:
				if(sys.argv[i] == "-read"):
                     			if i+1 < sys.argv[0]:
                                		i = i + 1
                                		filespath = sys.argv[i]
    print("- batchsize    : " + str(batchsize))
    print("- epochs limit : " + str(epochs))
    data     = cu.datasetload(fileslimit=fileL,path=filespath)
    filetest = filespath + "/test/"
    datatest = cu.datasetload(fileslimit=35,path=filetest)
    print(data.shape[0])
    # results = []

    filters = [{"detCounterIn":[4.],"detCounterOut":[5.]}]#,{"isBarrelIn":[1.],"isBarrelOut":[0.]},cu.datastats(data,mode="ladderdisk")]#,cu.datastats(data,mode="ladderdisk"),
    #cu.datastats(data,mode="disk"),{"isBarrelIn":[1.],"isBarrelOut":[1.]},
    #{"detCounterIn":[4.],"detCounterOut":[5.]},
    #{"isBarrelIn":[0.],"isBarrelOut":[0.]},
    #cu.datamodule(data),{}]

    # {"isBarrelIn":[0.],"isBarrelOut":[0.]},
    # {"isBarrelIn":[1.],"isBarrelOut":[1.]},
    # {},
    # {"ladderIn":[0.],"ladderOut":[1.]},
    # {"detCounterIn":[4.],"detCounterOut":[5.]},
    # {"inZ":[-5.0,5.0],"outZ":[-5.0,5.0]}]

#   1 - Whole dataset
#   2 - Only barrel
#   3 - Only forward
#   4 - detIn 0 detOut 1
#   5 - Most populated module
#   6 - Barrel Ladders couple
#   7 - detIn 4 detOut 5
#   8 - inZ OutZ : [-5.0,5.0]


    bathsizes = [1000,int(data.shape[0]/10),100]
    BandW = False

    print(data.shape)
    for batchsize in bathsizes:
        for angC in [True]:
            for sanrt in [0.5,0.25]:
                for filt in filters:

                   # opt = raw_input('Insert filter(s)? ')

                    timings = []
                    fileOutput = strftime("./outputs/%Y-%m-%d_%H_%M_%S_outputs.pkl.gz", gmtime())
                    fileOuttxt = strftime("./outputs/%Y-%m-%d_%H_%M_%S_", gmtime())

                    datafil = cu.datafiltering(filt,data)
                    (data_train, label_train) = cu.clustersInput(datafil,sanitize=True,sanratio=sanrt,angularcorrection=angC,bAndW=BandW,writedata=False)
                    datafiltest = cu.datafiltering(filt,datatest)
                    (data_test, label_test) = cu.clustersInput(datafiltest,sanitize=True,sanratio=sanrt,angularcorrection=angC,bAndW=BandW)

                    print(data_train.shape)

                    # Create the clustercnn
                    clustercnn = Sequential()

                    inSh = ()

                    if angC:
                        inSh = (8, 16,3)
                    else:
                        inSh = (8, 16,1)

                    clustercnn.add(Convolution2D(128, (2, 2), input_shape=inSh, activation='relu', padding='same'))
                    clustercnn.add(Dropout(0.2))
                    clustercnn.add(Convolution2D(64, (2, 2), activation='relu', padding='same'))
                    s = 1
                    if angC:
                        s = 2
                    clustercnn.add(MaxPooling2D(pool_size=(s, 2)))
                    clustercnn.add(Convolution2D(128, (2, 2), activation='relu', padding='same'))
                    clustercnn.add(Dropout(0.2))
                    clustercnn.add(Convolution2D(128, (2, 2), activation='relu', padding='same'))
                    clustercnn.add(MaxPooling2D(pool_size=(1, 2)))
                    clustercnn.add(Convolution2D(256, (2, 2), activation='relu', padding='same'))
                    clustercnn.add(Dropout(0.2))
                    # clustercnn.add(Convolution2D(128, (2, 2), activation='relu', padding='same'))
                    # clustercnn.add(MaxPooling2D(pool_size=(2, 2)))
                    clustercnn.add(Flatten())
                    clustercnn.add(Dropout(0.2))
                    clustercnn.add(Dense(256, activation='relu', kernel_constraint=maxnorm(3)))
                    clustercnn.add(Dropout(0.2))
                    clustercnn.add(Dense(128, activation='relu', kernel_constraint=maxnorm(3)))
                    clustercnn.add(Dropout(0.2))
                    clustercnn.add(Dense(2, activation='softmax'))

                    clustercnn.summary()

                    lrate = 0.01
                    decay = 10.0 * lrate/epochs
                    sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
                    clustercnn.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

                    # clustercnn = Sequential()
                    # clustercnn.add(Convolution3D(2048,3,4,1,input_shape = data_train.shape[1:],border_mode='valid'))
                    # clustercnn.add(Activation('sigmoid'))
                    # clustercnn.add(MaxPooling3D(pool_size=(1,2,2),border_mode='valid'))
                    #
                    # #clustercnn.add(Convolution2D(1024,3,3,input_shape = data_train.shape[1:],border_mode='valid'))
                    # # clustercnn.add(Activation('sigmoid'))
                    # # clustercnn.add(MaxPooling2D(pool_size=(2,2),border_mode='valid'))
                    #
                    # # clustercnn.add(Convolution3D(32,2,2,1))
                    # # clustercnn.add(MaxPooling3D(pool_size=(2,2,1),border_mode='valid'))
                    # # clustercnn.add(Activation('sigmoid'))
                    #
                    # clustercnn.add(Flatten())
                    # clustercnn.add(Dense(512, activation='sigmoid'))
                    # clustercnn.add(Dropout(0.5))
                    # clustercnn.add(Dense(numlabels, activation='softmax'))

                    # clustercnn.compile(loss='binary_crossentropy',optimizer='sgd', metrics=['accuracy'])

                    early = EarlyStopping(monitor='val_loss', patience=1000, verbose=1, mode='auto')
                    start = time.time()
                    history = clustercnn.fit(data_train, label_train,validation_split=0.33, nb_epoch=epochs, batch_size=batchsize,shuffle=True,callbacks=[early])
                    end = time.time()
                    timings.append(end-start)

                    start = time.time()
                    predicted_target = clustercnn.predict(data_train)
                    end = time.time()
                    timings.append(end-start)

                    clustercnn.summary()
         	    #clustercnn.save(fileOuttxt + "_model.h5")
		    #clustercnn.save_weights(fileOuttxt + "_weights.h5")
                    # print(label_train)
                    # print(predicted_target)
                    #
                    # print(np.sum(label_train[0,1:]))
                    # print(np.sum(predicted_target[0,1:]))

                    label_score = label_train[:,0]
                    preds_score = predicted_target[:,0]

                    # print(label_score)
                    # print(preds_score)

                    #print(label_score)
                    print(label_score.shape)
                    if sklearnflag:
                        fpr, tpr, _ = roc_curve(label_score, preds_score)
                        print("\nROC Area Train : %g"%(auc(fpr, tpr)))


                    # Final evaluation of the model
                    scores = clustercnn.evaluate(data_test, label_test, verbose=0)

                    print("Filters : ")
                    print(filt)
                    print("Accuracy test [%.2f] : %.2f%%" % (data_test.shape[0],scores[1]*100))

                    start = time.time()
                    predicted_target_test = clustercnn.predict(data_test)
                    end = time.time()
                    timings.append(end-start)

                    label_score_test = label_test[:,0]
                    preds_score_test = predicted_target_test[:,0]

                    if sklearnflag:
                        fpr, tpr, _ = roc_curve(label_score_test, preds_score_test)
                        print("\nROC Area Test : %g"%(auc(fpr, tpr)))

                    traaccHist = np.array(history.history['acc'])
                    valaccHist = np.array(history.history['val_acc'])
                    tralosHist = np.array(history.history['loss'])
                    vallosHist = np.array(history.history['val_loss'])

                    print("Saving txt files in " + fileOuttxt)
                    clustercnn.save(fileOuttxt + "_model.h5")
                    clustercnn.save_weights(fileOuttxt + "_weights.h5")
		    np.savetxt(fileOuttxt + 'acc.txt', traaccHist, delimiter=' ')
                    np.savetxt(fileOuttxt + 'valacc.txt', valaccHist, delimiter=' ')
                    np.savetxt(fileOuttxt + 'tralos.txt', tralosHist, delimiter=' ')
                    np.savetxt(fileOuttxt + 'vallos.txt', vallosHist, delimiter=' ')

                    np.savetxt(fileOuttxt + 'labtest.txt', label_test, delimiter=' ')
                    np.savetxt(fileOuttxt + 'predtest.txt', predicted_target_test, delimiter=' ')

                    np.savetxt(fileOuttxt + 'labtrain.txt', label_train, delimiter=' ')
                    np.savetxt(fileOuttxt + 'predtrain.txt', predicted_target, delimiter=' ')

                    with open(fileOuttxt + 'filters.txt', 'wb') as filtfile:
                        u = "Filters with data "+ str(data_train.shape[0]) + " shape " + str(data_train.shape[1]) + " batchsize : " + str(batchsize) + "\n"
                        u += " - BW " + str(BandW) + "\n"
                        u += " - Ang " + str(angC) + "\n"
                        u += "Timings : \n"

                        for t in timings:
                            u += str(t) + " "

                        u += "\n"

                        for k, v in filt.iteritems():
                            u += str(k) + " : " + str(v) + " - "
                        u += "\n" + filespath + "\n"
                        u += "\n" + str(fileL) + "\n"
                        u += "\n" + str(sanrt) + "\n"

                        filtfile.write(u)

                #     result = (batchsize,filters,predicted_target,label_train,predicted_target_test,label_test,history,data_train.shape)
                #     results.append(result)
                #
                # with gzip.open(fileOutput, 'wb') as f:
                #     print("\nSaving to file " + fileOutput)
                #     pkl.dump(results, f, -1)

def runCNN(data,datatest,filters,batchsize,epochs=2000,san=True,one=False,ang=True):

    fileOutput = strftime("./outputs/%Y-%m-%d_%H_%M_%S_outputs.pkl.gz", gmtime())
    timings = []

    datafil = cu.datafiltering(filters,data,sanitize=san,sanratio=0.5)
    (data_train, label_train) = cu.clustersInput(datafil,sanitize=True,sanratio=0.50,angularcorrection=ang,bAndW=one)

    datafil_test = cu.datafiltering(filters,datatest,sanitize=san,sanratio=0.5)
    (data_test, label_test) = cu.clustersInput(datafil_test,sanitize=True,sanratio=0.50,angularcorrection=ang,bAndW=one)

    clustercnn = Sequential()

    inSh = ()

    if angC:
        inSh = (3, 8, 16)
    else:
        inSh = (1, 8, 16)

    clustercnn.add()
    clustercnn.add(Convolution2D(64, (2, 2), input_shape=inSh, activation='relu', padding='same'))
    clustercnn.add(Dropout(0.2))
    clustercnn.add(Convolution2D(64, (2, 2), activation='relu', padding='same'))
    clustercnn.add(MaxPooling2D(pool_size=(1, 2)))
    clustercnn.add(Convolution2D(128, (2, 2), activation='relu', padding='same'))
    clustercnn.add(Dropout(0.2))
    clustercnn.add(Convolution2D(128, (2, 2), activation='relu', padding='same'))
    clustercnn.add(MaxPooling2D(pool_size=(1, 2)))
    clustercnn.add(Convolution2D(256, (2, 2), activation='relu', padding='same'))
    clustercnn.add(Dropout(0.2))
    # clustercnn.add(Convolution2D(128, (2, 2), activation='relu', padding='same'))
    # clustercnn.add(MaxPooling2D(pool_size=(2, 2)))
    clustercnn.add(Flatten())
    clustercnn.add(Dropout(0.2))
    clustercnn.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
    clustercnn.add(Dropout(0.2))
    clustercnn.add(Dense(256, activation='relu', kernel_constraint=maxnorm(3)))
    clustercnn.add(Dropout(0.2))
    clustercnn.add(Dense(2, activation='softmax'))

    lrate = 0.01
    decay = lrate/epochs
    sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
    clustercnn.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

    early = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')

    start = time.time()
    history = clustercnn.fit(data_train, label_train,validation_split=0.33, nb_epoch=epochs, batch_size=batchsize,shuffle=True,callbacks=[early])
    end = time.time()
    timings.append(end-start)

    start = time.time()
    predicted_target = clustercnn.predict(data_train)
    end = time.time()
    timings.append(end-start)

    clustercnn.summary()

    label_score = label_train[:,0]
    preds_score = predicted_target[:,0]

    if sklearnflag:
        fpr, tpr, _ = roc_curve(label_score, preds_score)
        print("\n ROC Area Train : %g"%(auc(fpr, tpr)))

    # Final evaluation of the model
    start = time.time()
    scores = clustercnn.evaluate(data_test, label_test, verbose=0)
    end = time.time()
    timings.append(end-start)

    print("Accuracy test: %.2f%%" % (scores[1]*100))
    predicted_target_test = clustercnn.predict(data_test)
    label_score_test = label_test[:,0]
    preds_score_test = predicted_target_test[:,0]

    with gzip.open(fileOutput, 'wb') as f:
        result = (batchsize,filters,predicted_target,label_train,predicted_target_test,label_test,history,data_train.shape)
        print("\n Saving to file " + fileOutput)
        pkl.dump(result, f, -1)

    print(" ============= Training end")
