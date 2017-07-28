from clusterLoad import dataToPandas, sanitizData, clusterData, filtersData, ladderTop
from dataLabels import dataLab,tailLab,outPixLab,inPixLab,headLab,XYZ,inXYZ,outXYZ

import numpy as np
import tensorflow as tf
import keras as k

from keras.models import Sequential,Model
from keras.layers import Dense, Activation,Input, Dense, Dropout, merge,InputLayer
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.constraints import maxnorm
from keras.optimizers import SGD,Adam,Adagrad
from keras.layers import Dense, Dropout, Activation
from keras.layers import Convolution2D, Flatten, MaxPooling2D, Merge,BatchNormalization,Reshape
from keras.callbacks import History,ReduceLROnPlateau,CSVLogger,TensorBoard,ModelCheckpoint
from keras import backend
from keras.models import load_model
from keras.engine.topology import Layer
from keras.preprocessing.image import ImageDataGenerator



from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

import collections
import argparse
import time
import os
import sys
from time import gmtime, strftime
from io import StringIO


batches = [1000,200,5000,100,50]
lratios = [0.01,0.025,0.05,0.1,0.5]
moments = np.arange(0.0,0.9,0.1)
moments = moments[::-1]
recipies = ["TTBar","ZEE"]

parsercnn = argparse.ArgumentParser(prog="clustercnn")
parsercnn.add_argument('--epochs', type=int, default=1000,help='no. of epochs')
parsercnn.add_argument('--read', type=str, default="/eos/cms/store/cmst3/group/dehep/convPixels/mixed/shuff/",help='files path')
parsercnn.add_argument('--no-theta', dest='theta', action='store_false')
parsercnn.set_defaults(theta=True)
parsercnn.add_argument('--no-phi', dest='phi', action='store_false')
parsercnn.set_defaults(phi=True)
parsercnn.add_argument('--flimit', type=int, default=-1,help='max no. of files')
parsercnn.add_argument('--checks', type=str, default=None,help='load checkpoint')
parsercnn.add_argument('--ext', type=str, default="h5",help='load checkpoint')
parsercnn.add_argument('--dense', dest='dense', action='store_true')
parsercnn.set_defaults(dense=False)



argscnn = parsercnn.parse_args()



if __name__ == "__main__":

    path = argscnn.read
    epochs = argscnn.epochs
    fileL = argscnn.flimit

    checkpoint = False
    checkpath = "checkpoints"

    if argscnn.checks is not None:
        checkpath = argscnn.checks
        checkpoint = True

    filenames = np.array([f for f in os.listdir(path) if (os.path.isfile(os.path.join(path, f)) and  f.lower().endswith(('.csv')))])

    filenames = np.array([f for f in os.listdir(path) if (os.path.isfile(os.path.join(path, f)) and  f.lower().endswith(('.h5')))])

    dataname = "uknRec"

    for l in path.split("/"):
        if any(r in l for ext in recipies):
            dataname = l

    print(dataname)

    print("Starting Clusters CNN Training with: ")

    #imgNorm = ImageDataGenerator(featurewise_center=True,samplewise_center=True,featurewise_std_normalization=True,samplewise_std_normalization=True)
    dataall = dataToPandas(path,fileL,argscnn.ext)

    dataset = sanitizData(dataall,0.5)
    dataset = ladderTop(dataset)
    #dataset = filtersData(dataset,"isBarrelIn",1.)
    #dataset = filtersData(dataset,"isBarrelOut",1.)
    (clusters,labels) = clusterData(dataset)
    #sys.exit()
    X_train, X_test, l_train, l_test = train_test_split(clusters,labels, test_size=0.33, random_state=int(time.time()))

    #imgNorm.fit(X_train)
    #create outputs dirs
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")

    # filters = [{"detCounterIn":[4.],"detCounterOut":[5.]}]#,{"isBarrelIn":[1.],"isBarrelOut":[0.]},cu.datastats(data,mode="ladderdisk")]#,cu.datastats(data,mode="ladderdisk"),
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

    for b in batches:
        for m in moments:
            for l in lratios:

                dayTime = strftime("%Y%m%d%H%M%S_",gmtime())
                checkps = "checkpoints/" + dataname + "_" + dayTime + "_check.h5"
                csvlogs = "checkpoints/" + dataname + "_" + dayTime + "_logs.csv"
                datasav = "checkpoints/" + dataname + "_" + dayTime + "_dataset.h5"
                dataset.to_hdf(datasav,'data',append=True)

                calls = []

                #redulratio = ReduceLROnPlateau(patience=20,factor=0.1)
                calls.append(ModelCheckpoint(checkps,save_best_only=True))
                calls.append(CSVLogger(csvlogs, separator=',', append=True))
                calls.append(EarlyStopping(monitor='val_loss', patience=200, verbose=1, mode='auto'))
                #tensorboar = TensorBoard(log_dir='./logs', histogram_freq=10, batch_size=b, write_graph=True)

                clusterShape = X_train.shape[1:]

                print(clusterShape)
                print(X_train.shape)

                clustercnn = Sequential()
                clustercnn.add(Convolution2D(64, (3,3), input_shape=clusterShape, activation='relu', padding='same'))#,data_format="channels_last"))
                clustercnn.add(Dropout(0.2))
                clustercnn.add(Convolution2D(128, (3,3), activation='relu', padding='same'))
                clustercnn.add(MaxPooling2D(pool_size=(2,2)))
                clustercnn.add(BatchNormalization())

                clustercnn.add(Convolution2D(128, (3,3), activation='relu', padding='same'))
                clustercnn.add(Dropout(0.2))
                clustercnn.add(Convolution2D(64, (2,2), activation='relu', padding='same'))
                clustercnn.add(MaxPooling2D(pool_size=(2, 2)))
                clustercnn.add(Dropout(0.2))
                clustercnn.add(BatchNormalization())

                #clustercnn = Sequential()
                #clustercnn.add(Convolution2D(64, (2,2), input_shape=clusterShape, activation='relu', padding='same'))
                #clustercnn.add(BatchNormalization())

                clustercnn.add(Flatten())
                clustercnn.add(Dense(128, activation='relu', kernel_constraint=maxnorm(3)))
                clustercnn.add(Dropout(0.2))
                clustercnn.add(BatchNormalization())

                clustercnn.add(Dense(64, activation='relu', kernel_constraint=maxnorm(3)))
                clustercnn.add(Dropout(0.2))
                clustercnn.add(Dense(2, activation='softmax'))

                if argscnn.dense:

                    clustercnn = Sequential()

                    clustercnn.add(ll.InputLayer(X_train.shape))
                    clustercnn.add(ll.Flatten())

                    # clustercnn.add(Convolution2D(64, (4,4), input_shape=clusterShape, activation='relu', padding='same'))
                    #clustercnn.add(MaxPooling2D(pool_size=(2,2)))
                    # clustercnn.add(Dropout(0.2))
                    # X_train = X_train.reshape(X_train.shape[0],-1)
                    # X_test = X_test.reshape(X_test.shape[0],-1)
                    ### Dense layer can only accept 2D tensor,
                    ### so reshaping is done to flatten image tensor.
                    #clustercnn.add(Flatten())
                    clustercnn.add(Dense(512, activation='relu',kernel_constraint=maxnorm(3)))
                    clustercnn.add(BatchNormalization())
                    clustercnn.add(Dropout(0.2))

                    clustercnn.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
                    clustercnn.add(BatchNormalization())
                    clustercnn.add(Dropout(0.2))

                    clustercnn.add(Dense(256, activation='relu', kernel_constraint=maxnorm(3)))
                    clustercnn.add(BatchNormalization())
                    clustercnn.add(Dropout(0.2))

                    clustercnn.add(Dense(256, activation='relu', kernel_constraint=maxnorm(3)))
                    clustercnn.add(BatchNormalization())
                    clustercnn.add(Dropout(0.2))

                    clustercnn.add(Dense(2, activation=tf.nn.log_softmax))

                if checkpoint:
                    clustercnn = load_model(checkpath)



                clustercnn.summary()

                #sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
                opt =Adagrad(lr=0.001)
                #sgd = SGD(lr=l, momentum=m, decay=0.0, nesterov=True)

                clustercnn.compile(loss='binary_crossentropy', optimizer = opt, metrics=['accuracy'])
                #clustercnn.fit_generator(imgNorm.flow(X_train, y_train, batch_size=500,callbacks=calls,shuffle=True,validation_split=0.2),steps_per_epoch=len(x_train) / 32, epochs=epochs)

                history = clustercnn.fit(X_train,l_train,validation_split=0.2, nb_epoch=epochs, batch_size=b,shuffle=True,callbacks=calls)

                l_pred = clustercnn.predict(X_test)[:,0]
                l_score = l_test[:,0]
                fpr, tpr, _ = roc_curve(l_score, l_pred)
                print("\nROC Area Test  : %g"%(auc(fpr, tpr)))

                l_pred = clustercnn.predict(X_train)[:,0]
                l_score = l_train[:,0]
                fpr, tpr, _ = roc_curve(l_score, l_pred)
                print("\nROC Area Train : %g"%(auc(fpr, tpr)))
