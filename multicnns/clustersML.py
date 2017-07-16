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
from keras.utils import plot_model

from avgcallbacks import EarlyStoppingAvg

import time

from os import listdir
from os.path import isfile, join

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

    san = True
    edg = True
    bad = True

    phi = False
    theta = False

    BandW = False

    flipping = "-1"

    writedata = False

    epochs = 1000
    batchsize=100
    fileL = 100

    filesPath = "./datasets/"

    splitting = 0.3

    fnum = 0

    print("Starting Clusters CNN Training with: ")

    for i in range(1,len(sys.argv)):
        if(sys.argv[i] == "-bw"):
            BandW = True
            print("- B&W configuration")
        else:
            if (sys.argv[i] == "-a"):
                theta = True
                phi = True
                print("- theta correction")
                print("- phi correction")
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
                                        filesPath = str(sys.argv[i])
                                else:
                                    if (sys.argv[i] == "-phi"):
                                        phi = True
                                        print("- phi correction")
                                    else:
                                        if (sys.argv[i] == "-theta"):
                                            theta = True
                                            print("- theta correction")
                                        else:
                                            if (sys.argv[i] == "-flip"):
                                                print("- flipped drop")
                                                if i+1 < sys.argv[0]:
                                                    i = i + 1
                                                    flipping = str(sys.argv[i])
                                            else:
                                                if (sys.argv[i] == "-filt"):
                                                    print("- flipped drop")
                                                    if i+1 < sys.argv[0]:
                                                        i = i + 1
                                                        fnum = str(sys.argv[i])


    print("- batchsize    : " + str(batchsize))
    print("- epochs limit : " + str(epochs))

    allFiles = np.array([f for f in listdir(filesPath) if (isfile(join(filesPath, f)) and  f.lower().endswith(('.txt',".gz")))])
    trainSize = allFiles.shape[0] - int(float(allFiles.shape[0])*splitting)
    idx = np.random.permutation(allFiles.shape[0])
    training_idx, test_idx = idx[:trainSize], idx[trainSize:]
    trainFiles,testFiles = allFiles[training_idx], allFiles[test_idx]

    if(len(testFiles)<1):
        testFiles=trainFiles

    data     = cu.datasetload(filesList=trainFiles,path=filesPath,fileslimit=fileL-int(float(fileL)*splitting))
    datatest = cu.datasetload(filesList=testFiles,path=filesPath,fileslimit=int(float(fileL)*splitting))

    # results = []

    filters = [cu.datastats(data,mode="laddder"),cu.datastats(data,mode="ladderdisk"),
    cu.datastats(data,mode="disk"),
    {"detCounterIn":[0.],"detCounterOut":[1.]},
    {"isBarrelIn":[0.],"isBarrelOut":[0.]},
    cu.datamodule(data)]

    filt = filters[min(fnum,len(filters))]

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

    timings = []
    fileOutput = strftime("./outputs/%Y-%m-%d_%H_%M_%S_outputs.pkl.gz", gmtime())
    fileOuttxt = strftime("./outputs/%Y-%m-%d_%H_%M_%S_", gmtime())

    datafil = cu.datafiltering(filters[3],data)
    (data_train, label_train) = cu.clustersInput(datafil,sanitize=san,sanratio=0.5,phicorrection=phi,thetacorrection=theta,bAndW=BandW,writedata=False,flip=flipping)
    datafiltest = cu.datafiltering(filters[3],datatest)
    (data_test, label_test) = cu.clustersInput(datafiltest,sanitize=san,sanratio=0.5,phicorrection=phi,thetacorrection=theta,bAndW=BandW,flip=flipping)

    # print(data_train.shape)

    # Create the clustercnn
    clustercnn = Sequential()

    inSh = ()

    if theta or phi:
        if theta and phi:
            inSh = (5, 8, 16)
        else:
            inSh = (3, 8, 16)
    else:
        inSh = (1, 8, 16)

    clustercnn.add(Convolution2D(64, (3, 3), input_shape=inSh, activation='relu', padding='same'))
    clustercnn.add(Dropout(0.2))
    clustercnn.add(Convolution2D(64, (2, 2), activation='relu', padding='same'))
    wh = (1,2)
    if theta or phi:
        if theta and phi:
            wh = (2,2)
        wh  = (2,2)
    clustercnn.add(MaxPooling2D(pool_size=wh))
    wh = (2,2)
    if theta and phi:
         wh = (2,2)
    clustercnn.add(Convolution2D(128, wh, activation='relu', padding='same'))
    clustercnn.add(Dropout(0.2))
    # clustercnn.add(Convolution2D(32, (2, 2), activation='relu', padding='same'))

    s = 1
    if theta and phi:
        s = 2

    clustercnn.add(MaxPooling2D(pool_size=(1, 3)))
    # clustercnn.add(Convolution2D(512, (2, 2), activation='relu', padding='same'))
    clustercnn.add(Dropout(0.2))
    # clustercnn.add(Convolution2D(128, (2, 2), activation='relu', padding='same'))
    # clustercnn.add(MaxPooling2D(pool_size=(2, 2)))
    clustercnn.add(Flatten())
    clustercnn.add(Dropout(0.2))
    clustercnn.add(Dense(64, activation='relu', kernel_constraint=maxnorm(3)))
    clustercnn.add(Dropout(0.2))
    clustercnn.add(Dense(32, activation='relu', kernel_constraint=maxnorm(3)))
    clustercnn.add(Dropout(0.2))
    clustercnn.add(Dense(2, activation='softmax'))

    lrate = 0.01
    decay = 10.0 * lrate/epochs
    sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
    clustercnn.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

    clustercnn.summary()

    # clustercnnold = Sequential()
    # clustercnnold.add(Convolution3D(2048,3,2,1,input_shape = (8,8,2,1),border_mode='valid'))
    # clustercnnold.add(Activation('sigmoid'))
    # clustercnnold.add(MaxPooling3D(pool_size=(2,2,1),border_mode='valid'))
    #
    # #clustercnn.add(Convolution2D(1024,3,3,input_shape = data_train.shape[1:],border_mode='valid'))
    # # clustercnn.add(Activation('sigmoid'))
    # # clustercnn.add(MaxPooling2D(pool_size=(2,2),border_mode='valid'))
    #
    # # clustercnn.add(Convolution3D(32,2,2,1))
    # # clustercnn.add(MaxPooling3D(pool_size=(2,2,1),border_mode='valid'))
    # # clustercnn.add(Activation('sigmoid'))
    #
    # clustercnnold.add(Flatten())
    # clustercnnold.add(Dense(512, activation='sigmoid'))
    # clustercnnold.add(Dropout(0.5))
    # clustercnnold.add(Dense(numlabels, activation='softmax'))
    #
    # clustercnnold.compile(loss='binary_crossentropy',optimizer='sgd', metrics=['accuracy'])
    #
    # clustercnnold.summary()

    # plot_model(clustercnn, to_file='clustercnn.png')

    early = EarlyStopping(monitor='loss', patience=20, verbose=1, mode='auto')
    early = EarlyStoppingAvg(monitor='loss', patience=5, verbose=1, mode='auto',avgsteps=5)

    start = time.time()
    history = clustercnn.fit(data_train, label_train,validation_split=0.33, nb_epoch=epochs, batch_size=batchsize,shuffle=True,callbacks=[early])
    end = time.time()
    timings.append(end-start)

    # plot_model(clustercnn, to_file='clustercnnfit.png')

    start = time.time()
    predicted_target = clustercnn.predict(data_train)
    end = time.time()
    timings.append(end-start)

    clustercnn.summary()
    # print(label_train)
    # print(predicted_target)
    #
    # print(np.sum(label_train[0,1:]))
    # print(np.sum(predicted_target[0,1:]))

    label_score = label_train[:,0]
    preds_score = predicted_target[:,0]

    # print(label_score)
    # print(preds_score)

    print(label_score)
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
        u += " - BW \t" + str(BandW) + "\n"
        u += " - Phi \t" + str(phi) + "\n"
        u += " - Theta \t" + str(theta) + "\n"
        u += " - Flip \t" + str(flipping) + "\n"
        u += "Timings : \n"

        for t in timings:
            u += str(t) + " "

        u += "\n"

        for k, v in filt.iteritems():
            u += str(k) + " : " + str(v) + " - "
        filtfile.write(u)



                #     result = (batchsize,filters,predicted_target,label_train,predicted_target_test,label_test,history,data_train.shape)
                #     results.append(result)
                #
                # with gzip.open(fileOutput, 'wb') as f:
                #     print("\nSaving to file " + fileOutput)
                #     pkl.dump(results, f, -1)

def runComplete(fileL=100,filesPath="./datasets/"):

    san = False
    writedata = False

    epochs = 1000
    batchsize=100

    splitting = 0.3

    print("Starting Clusters CNN Training with: ")
    print("- batchsize    : " + str(batchsize))
    print("- epochs limit : " + str(epochs))

    allFiles = np.array([f for f in listdir(filesPath) if (isfile(join(filesPath, f)) and  f.lower().endswith(('.txt',".gz")))])
    trainSize = allFiles.shape[0] - int(float(allFiles.shape[0])*splitting)
    idx = np.random.permutation(allFiles.shape[0])
    training_idx, test_idx = idx[:trainSize], idx[trainSize:]
    trainFiles,testFiles = allFiles[training_idx], allFiles[test_idx]

    data     = cu.datasetload(filesList=trainFiles,path=filesPath,fileslimit=fileL-int(float(fileL)*splitting))
    datatest = cu.datasetload(filesList=testFiles,path=filesPath,fileslimit=int(float(fileL)*splitting))

    # results = []

    filters = [cu.datastats(data,mode="laddder"),cu.datastats(data,mode="ladderdisk"),
    cu.datastats(data,mode="disk"),
    {"detCounterIn":[0.],"detCounterOut":[1.]},
    {"isBarrelIn":[0.],"isBarrelOut":[0.]},
    cu.datamodule(data)]

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

    bathsizes = [100,20,200,50]

    print(data.shape)
    for batchsize in bathsizes:
        for filt in filters:
            for phi in [True,False]:
                for theta in [True,False]:
                    for BandW in [True,False]:
                        for flipping in ["-1","1","0"]:
                            # opt = raw_input('Insert filter(s)? ')

                            timings = []
                            fileOutput = strftime("./outputs/%Y-%m-%d_%H_%M_%S_outputs.pkl.gz", gmtime())
                            fileOuttxt = strftime("./outputs/%Y-%m-%d_%H_%M_%S_", gmtime())

                            datafil = cu.datafiltering(filt,data)
                            (data_train, label_train) = cu.clustersInput(datafil,sanitize=san,sanratio=0.5,phicorrection=phi,thetacorrection=theta,bAndW=BandW,writedata=False,flip=flipping)
                            datafiltest = cu.datafiltering(filt,datatest)
                            (data_test, label_test) = cu.clustersInput(datafiltest,sanitize=san,sanratio=0.5,phicorrection=phi,thetacorrection=theta,bAndW=BandW,flip=flipping)

                            print(data_train.shape)

                            # Create the clustercnn
                            clustercnn = Sequential()

                            inSh = ()

                            if theta or phi:
                                if theta and phi:
                                    inSh = (5, 8, 16)
                                else:
                                    inSh = (3, 8, 16)
                            else:
                                inSh = (1, 8, 16)

                            clustercnn.add(Convolution2D(64, (2, 2), input_shape=inSh, activation='relu', padding='same'))
                            clustercnn.add(Dropout(0.2))
                            clustercnn.add(Convolution2D(64, (2, 2), activation='relu', padding='same'))
                            s = 1
                            if theta or phi:
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

                            lrate = 0.01
                            decay = 10.0 * lrate/epochs
                            sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
                            clustercnn.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

                            # clustercnnold = Sequential()
                            # clustercnnold.add(Convolution3D(2048,3,2,1,input_shape = (8,8,2,1),border_mode='valid'))
                            # clustercnnold.add(Activation('sigmoid'))
                            # clustercnnold.add(MaxPooling3D(pool_size=(2,2,1),border_mode='valid'))
                            #
                            # #clustercnn.add(Convolution2D(1024,3,3,input_shape = data_train.shape[1:],border_mode='valid'))
                            # # clustercnn.add(Activation('sigmoid'))
                            # # clustercnn.add(MaxPooling2D(pool_size=(2,2),border_mode='valid'))
                            #
                            # # clustercnn.add(Convolution3D(32,2,2,1))
                            # # clustercnn.add(MaxPooling3D(pool_size=(2,2,1),border_mode='valid'))
                            # # clustercnn.add(Activation('sigmoid'))
                            #
                            # clustercnnold.add(Flatten())
                            # clustercnnold.add(Dense(512, activation='sigmoid'))
                            # clustercnnold.add(Dropout(0.5))
                            # clustercnnold.add(Dense(numlabels, activation='softmax'))
                            #
                            # clustercnnold.compile(loss='binary_crossentropy',optimizer='sgd', metrics=['accuracy'])
                            #
                            # clustercnnold.summary()

                            # plot_model(clustercnn, to_file='clustercnn.png')

                            early = EarlyStopping(monitor='val_loss', patience=500, verbose=1, mode='auto')
                            start = time.time()
                            history = clustercnn.fit(data_train, label_train,validation_split=0.33, nb_epoch=epochs, batch_size=batchsize,shuffle=True,callbacks=[early])
                            end = time.time()
                            timings.append(end-start)

                            # plot_model(clustercnn, to_file='clustercnnfit.png')

                            start = time.time()
                            predicted_target = clustercnn.predict(data_train)
                            end = time.time()
                            timings.append(end-start)

                            clustercnn.summary()
                            # print(label_train)
                            # print(predicted_target)
                            #
                            # print(np.sum(label_train[0,1:]))
                            # print(np.sum(predicted_target[0,1:]))

                            label_score = label_train[:,0]
                            preds_score = predicted_target[:,0]

                            # print(label_score)
                            # print(preds_score)

                            print(label_score)
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
                                u += " - BW \t" + str(BandW) + "\n"
                                u += " - Phi \t" + str(phi) + "\n"
                                u += " - Theta \t" + str(theta) + "\n"
                                u += " - Flip \t" + str(flipping) + "\n"
                                u += "Timings : \n"

                                for t in timings:
                                    u += str(t) + " "

                                u += "\n"

                                for k, v in filt.iteritems():
                                    u += str(k) + " : " + str(v) + " - "
                                filtfile.write(u)

def runCSV(filesPath="./datasets/",filename="Hits.csv"):

    san = False
    writedata = Falsemain


    epochs = 1000
    batchsize=100

    splitting = 0.3

    print("Starting Clusters CNN Training with: ")
    print("- batchsize    : " + str(batchsize))
    print("- epochs limit : " + str(epochs))

    dataall = cu.csvLoad(filen=filename,path=filesPath)

    numpy.random.shuffle(dataall)

    data, datatest = dataall[:int(dataall.shape[0]*0.7),:], dataall[int(dataall.shape[0]*0.3):,:]

    # results = []

    filters = [{}]

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

    bathsizes = [100,20,200,50]

    print(data.shape)
    for batchsize in bathsizes:
        for filt in filters:
            for phi in [True,False]:
                for theta in [True,False]:
                    for BandW in [True,False]:
                        for flipping in ["-1","1","0"]:
                            # opt = raw_input('Insert filter(s)? ')

                            timings = []
                            fileOutput = strftime("./outputs/%Y-%m-%d_%H_%M_%S_outputs.pkl.gz", gmtime())
                            fileOuttxt = strftime("./outputs/%Y-%m-%d_%H_%M_%S_", gmtime())

                            datafil = cu.datafiltering(filt,data)
                            (data_train, label_train) = cu.clustersInput(datafil,sanitize=san,sanratio=0.5,phicorrection=phi,thetacorrection=theta,bAndW=BandW,writedata=False,flip=flipping)
                            datafiltest = cu.datafiltering(filt,datatest)
                            (data_test, label_test) = cu.clustersInput(datafiltest,sanitize=san,sanratio=0.5,phicorrection=phi,thetacorrection=theta,bAndW=BandW,flip=flipping)

                            print(data_train.shape)

                            # Create the clustercnn
                            clustercnn = Sequential()

                            inSh = ()

                            if theta or phi:
                                if theta and phi:
                                    inSh = (5, 8, 16)
                                else:
                                    inSh = (3, 8, 16)
                            else:
                                inSh = (1, 8, 16)

                            clustercnn.add(Convolution2D(64, (2, 2), input_shape=inSh, activation='relu', padding='same'))
                            clustercnn.add(Dropout(0.2))
                            clustercnn.add(Convolution2D(64, (2, 2), activation='relu', padding='same'))
                            s = 1
                            if theta or phi:
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

                            lrate = 0.01
                            decay = 10.0 * lrate/epochs
                            sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
                            clustercnn.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

                            # clustercnnold = Sequential()
                            # clustercnnold.add(Convolution3D(2048,3,2,1,input_shape = (8,8,2,1),border_mode='valid'))
                            # clustercnnold.add(Activation('sigmoid'))
                            # clustercnnold.add(MaxPooling3D(pool_size=(2,2,1),border_mode='valid'))
                            #
                            # #clustercnn.add(Convolution2D(1024,3,3,input_shape = data_train.shape[1:],border_mode='valid'))
                            # # clustercnn.add(Activation('sigmoid'))
                            # # clustercnn.add(MaxPooling2D(pool_size=(2,2),border_mode='valid'))
                            #
                            # # clustercnn.add(Convolution3D(32,2,2,1))
                            # # clustercnn.add(MaxPooling3D(pool_size=(2,2,1),border_mode='valid'))
                            # # clustercnn.add(Activation('sigmoid'))
                            #
                            # clustercnnold.add(Flatten())
                            # clustercnnold.add(Dense(512, activation='sigmoid'))
                            # clustercnnold.add(Dropout(0.5))
                            # clustercnnold.add(Dense(numlabels, activation='softmax'))
                            #
                            # clustercnnold.compile(loss='binary_crossentropy',optimizer='sgd', metrics=['accuracy'])
                            #
                            # clustercnnold.summary()

                            # plot_model(clustercnn, to_file='clustercnn.png')

                            early = EarlyStopping(monitor='val_loss', patience=500, verbose=1, mode='auto')
                            start = time.time()
                            history = clustercnn.fit(data_train, label_train,validation_split=0.33, nb_epoch=epochs, batch_size=batchsize,shuffle=True,callbacks=[early])
                            end = time.time()
                            timings.append(end-start)

                            # plot_model(clustercnn, to_file='clustercnnfit.png')

                            start = time.time()
                            predicted_target = clustercnn.predict(data_train)
                            end = time.time()
                            timings.append(end-start)

                            clustercnn.summary()
                            # print(label_train)
                            # print(predicted_target)
                            #
                            # print(np.sum(label_train[0,1:]))
                            # print(np.sum(predicted_target[0,1:]))

                            label_score = label_train[:,0]
                            preds_score = predicted_target[:,0]

                            # print(label_score)
                            # print(preds_score)

                            print(label_score)
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
                                u += " - BW \t" + str(BandW) + "\n"
                                u += " - Phi \t" + str(phi) + "\n"
                                u += " - Theta \t" + str(theta) + "\n"
                                u += " - Flip \t" + str(flipping) + "\n"
                                u += "Timings : \n"

                                for t in timings:
                                    u += str(t) + " "

                                u += "\n"

                                for k, v in filt.iteritems():
                                    u += str(k) + " : " + str(v) + " - "
                                filtfile.write(u)
                    #     result = (batchsize,filters,predicted_target,label_train,predicted_target_test,label_test,history,data_train.shape)
                    #     results.append(result)
                    #
                    # with gzip.open(fileOutput, 'wb') as f:
                    #     print("\nSaving to file " + fileOutput)
                    #     pkl.dump(results, f, -1)
