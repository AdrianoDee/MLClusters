# coding: utf-8

# Nikolaus Howe, May 2016
import numpy as np
import sys
import ast
import h5py
from sklearn.cross_validation import train_test_split
from keras.models import model_from_json, Sequential
from keras.layers import Dense, Dropout, Activation, Convolution2D, Convolution3D, Flatten, MaxPooling2D, MaxPooling3D, Merge
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle

# Set directories here #
model_dir = "/home/nhowe/Notebooks/models/"

def convertFile(filename):
    with open(filename) as myfile:
        my_events_string = myfile.read().replace('\n', '')
    my_events = ast.literal_eval(my_events_string)
    event_list = []
    for event in my_events:
        event_list.append( np.array(event) )
    energy_array_list = []
    for event in event_list:
        energy_array_list.append(getEventArray(event))
    store = h5py.File(filename+'.h5')
    store['images'] = e
    store.close()

def get_dataset(directory):
    import os
    big=None
    for fn in filter(None,os.popen('ls %s*.h5'%directory).read().split('\n')):
        if 'fulldataset' in fn: continue
        store=h5py.File(fn,'r')
        if big==None:
            big = np.asarray(store['images'])
            print 'init'
        else:
            a = np.asarray(store['images'])
            big = np.concatenate((big, a), axis=0)
        store.close()
    if big!=None:
        print big.shape
    return big

def save_dataset(directory, dataset):
    store = h5py.File('%s_fulldataset.h5'%directory, 'w')
    store['images'] = dataset
    store.close()

def load_dataset(directory):
    store = h5py.File('%s_fulldataset.h5'%directory, 'r')
    return store['images']

def train_test( shape=None , split=0.33):
    signal = load_dataset("/data/vlimant/LCD/Gamma100GeV")
    bkg = load_dataset("/data/vlimant/LCD/Pi0100GeV")
    X = np.concatenate( (signal, bkg), axis=0 )
    Y = np.zeros( (len(X)) )
    Y[:len(signal)] = 1
    if shape:
        X = X.reshape((X.shape[0],)+shape)
    train_data, test_data, train_labels, test_labels = train_test_split(X, Y, test_size=split, random_state=42)
    return train_data, test_data, train_labels, test_labels

def store_model(model, history, label, predictions):
    # Save model as JSON
    open(model_dir+label+".json", 'w').write(model.to_json())
    # Save the weights
    model.save_weights(model_dir+label+"_w.h5", overwrite=True)
    # Save the history
    pickle.dump(history, open(model_dir+label+"_h.pkl",'w'))
    # Predictions is an nx2 matrix of the predicted and truth values
    pickle.dump(predictions, open(model_dir+label+"_p.pkl",'w'))

def load_model(label):
    model = model_from_json(open(model_dir+label+".json").read())
    model.load_weights(model_dir+label+"_w.h5")
    history = pickle.load(open(model_dir+label+"_h.pkl",'r'))
    try:
        predictions = pickle.load(open(model_dir+label+"_p.pkl",'r'))
        return model, history, predictions
    except:
        return model, history, None
