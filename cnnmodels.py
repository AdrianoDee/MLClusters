import keras

from keras.utils import plot_model

from keras.models import Sequential,Model
from keras.layers import Dense, Activation,Input, Dense, Dropout, merge
from keras.callbacks import EarlyStopping, ModelCheckpoint
# from keras.callbacks import model_from_json, model_to_json
from keras.constraints import maxnorm

from keras.layers import Dense, Dropout, Activation, Convolution2D, Convolution3D, Flatten, MaxPooling2D, MaxPooling3D, Merge
from keras.callbacks import History
from keras.utils import plot_model

clustercnns = []

# modello singolo
# modello singolo + angolo
# modello singolo + angolo + angolo
# modello singolo 3D
# 3 modelli -> dense layer -> output
#

def cnnModels(actv='relu',no=0):

    clustercnns = []

    clustercnns.append(Sequential())

    clustercnns[0].add(Convolution2D(128, (2, 2), input_shape=(1, 8, 16), activation=actv, padding='same'))
    clustercnns[0].add(Dropout(0.2))
    clustercnns[0].add(Convolution2D(128, (2, 2), activation=actv, padding='same'))
    clustercnns[0].add(MaxPooling2D(pool_size=(1,2)))

    clustercnns[0].add(Convolution2D(256, (2,2), activation=actv, padding='same'))
    clustercnns[0].add(Dropout(0.2))
    clustercnns[0].add(Convolution2D(256, (2, 2), activation=actv, padding='same'))

    clustercnns[0].add(MaxPooling2D(pool_size=(1, 2)))
    clustercnns[0].add(Convolution2D(512, (2, 2), activation=actv, padding='same'))
    clustercnns[0].add(Dropout(0.2))

    clustercnns[0].add(Flatten())
    clustercnns[0].add(Dropout(0.2))
    clustercnns[0].add(Dense(512, activation=actv, kernel_constraint=maxnorm(3)))
    clustercnns[0].add(Dropout(0.2))
    clustercnns[0].add(Dense(256, activation=actv, kernel_constraint=maxnorm(3)))
    clustercnns[0].add(Dropout(0.2))
    clustercnns[0].add(Dense(2, activation='softmax'))

    #CLUSTERCNN 1 - cluster + cluster_theta + cluster_phi
    #
    clustercnns.append(Sequential())

    # clustercnn = cm.clustercnns[1

    clustercnns[1].add(Convolution2D(64, (2, 3), input_shape=(3, 8, 16), activation=actv, padding='same'))
    clustercnns[1].add(Dropout(0.2))
    clustercnns[1].add(Convolution2D(64, (1, 4), activation=actv, padding='same'))
    clustercnns[1].add(MaxPooling2D(pool_size=(1,2)))

    clustercnns[1].add(Convolution2D(128, (2,2), activation=actv, padding='same'))
    clustercnns[1].add(Dropout(0.2))
    clustercnns[1].add(Convolution2D(128, (2, 2), activation=actv, padding='same'))

    clustercnns[1].add(MaxPooling2D(pool_size=(1, 2)))
    clustercnns[1].add(Convolution2D(256, (2, 2), activation=actv, padding='same'))
    clustercnns[1].add(Dropout(0.2))

    clustercnns[1].add(Flatten())
    clustercnns[1].add(Dropout(0.2))
    clustercnns[1].add(Dense(256, activation=actv, kernel_constraint=maxnorm(3)))
    clustercnns[1].add(Dropout(0.2))
    clustercnns[1].add(Dense(128, activation=actv, kernel_constraint=maxnorm(3)))
    clustercnns[1].add(Dropout(0.2))
    clustercnns[1].add(Dense(2, activation='softmax'))

    #CLUSTERCNN 2 - single input

    clustercnns.append(Sequential())

    # clustercnn = cm.clustercnns[2

    clustercnns[2].add(Convolution2D(64, (2, 2), input_shape=(5, 8, 16), activation=actv, padding='same'))
    clustercnns[2].add(Dropout(0.2))
    clustercnns[2].add(Convolution2D(64, (2, 2), activation=actv, padding='same'))
    clustercnns[2].add(MaxPooling2D(pool_size=(2,2)))

    clustercnns[2].add(Convolution2D(128, (2,2), activation=actv, padding='same'))
    clustercnns[2].add(Dropout(0.2))
    clustercnns[2].add(Convolution2D(128, (2, 2), activation=actv, padding='same'))

    clustercnns[2].add(MaxPooling2D(pool_size=(1, 2)))
    clustercnns[2].add(Convolution2D(256, (2, 2), activation=actv, padding='same'))
    clustercnns[2].add(Dropout(0.2))

    clustercnns[2].add(Flatten())
    clustercnns[2].add(Dropout(0.2))
    clustercnns[2].add(Dense(256, activation=actv, kernel_constraint=maxnorm(3)))
    clustercnns[2].add(Dropout(0.2))
    clustercnns[2].add(Dense(128, activation=actv, kernel_constraint=maxnorm(3)))
    clustercnns[2].add(Dropout(0.2))
    clustercnns[2].add(Dense(2, activation='softmax'))

    #cnn_3

    clustercnns.append(Sequential())

    clustercnns[3].add(Convolution2D(64, (2, 2), input_shape=(2, 8, 16), activation=actv, padding='same'))
    clustercnns[3].add(Dropout(0.2))
    clustercnns[3].add(Convolution2D(64, (2, 2), activation=actv, padding='same'))
    clustercnns[3].add(MaxPooling2D(pool_size=(2,2)))

    clustercnns[3].add(Convolution2D(128, (2,2), activation=actv, padding='same'))
    clustercnns[3].add(Dropout(0.2))
    clustercnns[3].add(Convolution2D(128, (2, 2), activation=actv, padding='same'))

    clustercnns[3].add(MaxPooling2D(pool_size=(1, 2)))
    clustercnns[3].add(Convolution2D(256, (2, 2), activation=actv, padding='same'))
    clustercnns[3].add(Dropout(0.2))

    clustercnns[3].add(Flatten())
    clustercnns[3].add(Dropout(0.2))
    clustercnns[3].add(Dense(256, activation=actv, kernel_constraint=maxnorm(3)))
    clustercnns[3].add(Dropout(0.2))
    clustercnns[3].add(Dense(128, activation=actv, kernel_constraint=maxnorm(3)))
    clustercnns[3].add(Dropout(0.2))
    clustercnns[3].add(Dense(2, activation='softmax'))


    #clustercnn 4
    clustercnns.append(Sequential())

    clustercnns[4].add(Convolution2D(64, (2, 2), input_shape=(2, 8, 16), activation=actv, padding='same'))
    clustercnns[4].add(Dropout(0.2))
    clustercnns[4].add(Convolution2D(64, (2, 2), activation=actv, padding='same'))
    clustercnns[4].add(MaxPooling2D(pool_size=(2,2)))

    clustercnns[4].add(Convolution2D(128, (2,2), activation=actv, padding='same'))
    clustercnns[4].add(Dropout(0.2))
    clustercnns[4].add(Convolution2D(128, (2, 2), activation=actv, padding='same'))

    clustercnns[4].add(MaxPooling2D(pool_size=(1, 2)))
    clustercnns[4].add(Convolution2D(256, (2, 2), activation=actv, padding='same'))
    clustercnns[4].add(Dropout(0.2))

    clustercnns[4].add(Flatten())
    clustercnns[4].add(Dropout(0.2))
    clustercnns[4].add(Dense(256, activation=actv, kernel_constraint=maxnorm(3)))
    clustercnns[4].add(Dropout(0.2))
    clustercnns[4].add(Dense(128, activation=actv, kernel_constraint=maxnorm(3)))
    clustercnns[4].add(Dropout(0.2))
    clustercnns[4].add(Dense(2, activation='softmax'))

    if no > len(clustercnns):
        no=0

    return clustercnns[no]

def cnnLayersOutput(cnn,X,actv):

    dummycnn= Sequential()

    dummycnn.add(Convolution2D(64, (2, 2), input_shape=(2, 8, 16), activation=actv, padding='same',weights=cnn.layers[0].get_weights()))
    dummycnn.add(Dropout(0.2))
    dummycnn.add(Convolution2D(64, (2, 2), activation=actv, padding='same',weights=cnn.layers[2].get_weights()))
    dummycnn.add(MaxPooling2D(pool_size=(2,2)))

    dummycnn.add(Convolution2D(128, (2,2), activation=actv, padding='same',weights=cnn.layers[4].get_weights()))
    dummycnn.add(Dropout(0.2))
    dummycnn.add(Convolution2D(128, (2, 2), activation=actv, padding='same',weights=cnn.layers[6].get_weights()))

    dummycnn.add(MaxPooling2D(pool_size=(1, 2)))
    dummycnn.add(Convolution2D(256, (2, 2), activation=actv, padding='same',weights=cnn.layers[8].get_weights()))
    dummycnn.add(Dropout(0.2))

    return dummycnn.predict(X)
