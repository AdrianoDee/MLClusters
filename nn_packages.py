#Kaustuv Datta, June 2016 
#one of several small classes for streamlining code

import theano
device="gpu0"

import numpy as np
import sys
import ast
import h5py
from sklearn.cross_validation import train_test_split
from keras.models import model_from_json, Sequential
from keras.layers import Dense, Dropout, Activation, Convolution2D, Convolution3D, Flatten, MaxPooling2D, MaxPooling3D, Merge
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle as p
import os
import ROOT as rt
import root_numpy as rnp
from numpy import random
#matplotlib inline
import matplotlib.pyplot as plt

def savemodel(model,name="neural network"):

    model_name = name
    model.summary()
    model.save_weights('%s.h5'%model_name, overwrite=True)
    model_json = model.to_json()
    with open("%s.json"%model_name, "w") as json_file:
        json_file.write(model_json)


def show_losses( histories,fname ):
    plt.figure(figsize=(10,10))
    #plt.ylim(bottom=0)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Error by Epoch')
    
    colors=[]
    do_acc=False
    for label,loss in histories:
        color = tuple(np.random.random(3))
        colors.append(color)
        l = label
        vl= label+" validation"
        if 'acc' in loss.history:
            l+=' (acc %2.4f)'% (loss.history['acc'][-1])
            do_acc = True
        if 'val_acc' in loss.history:
            vl+=' (acc %2.4f)'% (loss.history['val_acc'][-1])
            do_acc = True
        plt.plot(loss.history['loss'], label=l, color=color)
        if 'val_loss' in loss.history:
            plt.plot(loss.history['val_loss'], lw=2, ls='dashed', label=vl, color=color)


    plt.legend()
    plt.yscale('log')
    plt.savefig('%s.pdf'%fname)
    plt.show()
    
    if not do_acc: 
	return

    #plt.figure(figsize=(10,10))
    #plt.xlabel('Epoch')
    #plt.ylabel('Accuracy')
    #for i,(label,loss) in enumerate(histories):
    #    color = colors[i]
    #    if 'acc' in loss.history:
    #        plt.plot(loss.history['acc'], lw=2, label=label+" accuracy", color=color)
    #    if 'val_acc' in loss.history:
    #        plt.plot(loss.history['val_acc'], lw=2, ls='dashed', label=label+" validation accuracy", color=color)
    #plt.legend(loc='lower right')
    #plt.savefig('%s.png'%fname)
   
    #plt.show()

    #Class to make generator returning both energy and label
class My_Gen_EL:
    def __init__( self, batch_size, filesize, filepattern='/data/shared/LCD/EnergyScan_Gamma_Shuffled/GammaEscan_*GeV_fulldataset.h5'):
        self.batch_size = batch_size
        self.filelist = filter(None, os.popen('ls %s'%filepattern).read().split('\n'))
        #You can change the fraction of train, validation and test set here
        self.train_split = 0.6 
        self.test_split = 0.2 
        self.validation_split = 0.2
        self.fileindex = 0
        self.filesize = filesize
        self.position = 0
    #function to call when generating data for training  
    def train(self,cnn=False):
        return self.batches(cnn)
    #function to call when generating data for validating
    def validation(self, cnn=False):
        return self.batches(cnn)
    #function to call when generating data for testing
    def test(self, cnn=False):
        return self.batches(cnn)
    #The function which reads files and returns data of batch size of N
    def batch_helper(self, fileindex, position, batch_size, train=True):
        '''
        Yields batches of data of size N
        '''
        f = h5py.File(self.filelist[fileindex],'r')
        #If the data to be read can be read from the current file
        if (position + batch_size < self.filesize*train_split):
            data = np.array(f['images'][position : position + batch_size])
            target = np.array(f['target'][position : position + batch_size])
            #incrementing the position to start from while reading the next batch
            position += batch_size
            f.close()
            
            return data, target, fileindex, position
        
        else:
        #if the data to be read exceeds the current file
        #Read the data as much as we can from the current file
            data = np.array(f['images'][position:])
            target = np.array(f['target'][position:])
            f.close()
            #Read the remaining data from the next files by calling the same function recursively
            #Also a safety check to see if the file opened is the last file
            if (fileindex+1 < len(self.filelist)):
                data_, target_, fileindex, position = self.batch_helper(fileindex + 1, 0, batch_size - self.filesize + position)
                data = np.concatenate((data, data_), axis=0)
                target = np.concatenate((target, target_), axis=0)
            #if the file read is the last file, loop back to the beginning of the filname list
            else:
                fileindex = 0
                position = 0
            
            return data, target, fileindex, position
    #The function which loops indefinitely and continues to return data of the specified batch size
    def batches(self, cnn):
        #loop indefinetly
        while (self.fileindex < len(self.filelist)):
            data, target, self.fileindex, self.position = self.batch_helper(self.fileindex, self.position, self.batch_size)
            if data.shape[0]!=self.batch_size:
                continue
            if cnn==True:
                data = np.swapaxes(data,1,3)
            else:
                data= np.reshape(data,(self.batch_size,-1))
            target0=target[:,0]
            target1=target[:,1]
            yield (data, [target[:,0], target[:,1]/110.])
        self.fileindex = 0
    
class My_Gen_E:
    #Data generator for regression over energy 
    def __init__( self, batch_size, filesize, filepattern='/data/shared/LCD/EnergyScan_Gamma_Shuffled/GammaEscan_*GeV_fulldataset.h5'):
        self.batch_size = batch_size
        self.filelist = filter(None, os.popen('ls %s'%filepattern).read().split('\n'))
        
        self.train_split = 0.6 
        self.test_split = 0.2 
        self.validation_split = 0.2
        self.fileindex = 0
        self.filesize = filesize
        self.position = 0
    #function to call when generating data for training  
    def train(self, cnn=False):
        return self.batches(cnn)
    #function to call when generating data for validation 
    def validation(self, cnn=False):
        return self.batches(cnn)
    #function to call when generating data for testing  
    def test(self, cnn=False):
        return self.batches(cnn)
        
    #The function which reads files to gather data until batch size is satisfied
    def batch_helper(self, fileindex, position, batch_size):
        '''
        Yields batches of data of size N
        '''
        f = h5py.File(self.filelist[fileindex],'r')
        if (position + batch_size < self.filesize):
            data = np.array(f['images'][position : position + batch_size])
            target = np.array(f['target'][position : position + batch_size])
            target = np.delete(target,0,1)

            position += batch_size
            f.close()
            
            return data, target, fileindex, position
        
        else:
            data = np.array(f['images'][position:])
            target = np.array(f['target'][position:])
            target = np.delete(target,0,1)
            f.close()
            
            if (fileindex+1 < len(self.filelist)):
                data_, target_, fileindex, position = self.batch_helper(fileindex + 1, 0, batch_size - self.filesize + position)
                data = np.concatenate((data, data_), axis=0)
                target = np.concatenate((target, target_), axis=0)
            
            else:
                fileindex = 0
                position = 0
            
            return data, target, fileindex, position
    #The function which loops indefinitely and continues to return data of the specified batch size
    def batches(self, cnn):
        while (self.fileindex < len(self.filelist)):
            data, target, self.fileindex, self.position = self.batch_helper(self.fileindex, self.position, self.batch_size)
            if data.shape[0]!=self.batch_size:
                continue
            if cnn==True:
                data = np.swapaxes(data, 1, 3)
                #data = np.swapaxes(data, 1, 2)
                #data = np.swapaxes(data, 0, 1)
                #data=data.reshape((data.shape[0],1,20,20,25))
                
            else:
                data= np.reshape(data,(self.batch_size,-1))
            yield (data, target/110.)
        self.fileindex = 0
            