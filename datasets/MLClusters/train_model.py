import numpy as np
import os
import sys
import re
import glob
import h5py
import numpy as np
#import keras
from keras.models import Sequential,Model
from keras.layers import Dense, Activation,Input, Dense, Dropout, merge
from keras.callbacks import EarlyStopping, ModelCheckpoint
class MixedGen:
    #Data generator for regression over energy 
    def __init__( self, batch_size, filepattern='/data/shared/LCD/EnergyScan_Gamma_Shuffled/GammaEscan_*GeV_fulldataset.h5'):
        self.batch_size = batch_size
        self.filelist=[]
        for i in xrange(1,6):
            for j in xrange(1,11):
                self.filelist.append('/data/shared/LCD/New_Data/GammaEscan_%d_%d.h5'%(i,j)) 
        self.train_split = 0.6 
        self.test_split = 0.2 
        self.validation_split = 0.2
        self.fileindex = 0
        self.filesize = 0
        self.position = 0
    #function to call when generating data for training  
    def train(self,cnn=False):
        length = len(self.filelist)
        #deleting the validation and test set filenames from the filelist
        del self.filelist[np.floor((1-(self.train_split))*length).astype(int):]
        return self.batches(cnn)
    #function to call when generating data for testing
    def test(self, cnn=False):
        length = len(self.filelist)
        #deleting the train and validation set filenames from the filelist
        del self.filelist[:np.floor((1-self.test_split)*length).astype(int)+1]
        return self.batches(cnn)
    #function to call when generating data for validating
    def validation(self, cnn=False):
        length = len(self.filelist)
        #modifying the filename list to only include files for validation set
        self.filelist = self.filelist[np.floor(self.train_split*length+1).astype(int):np.floor((self.train_split+self.validation_split)*length+1).astype(int)]
        return self.batches(cnn)
        
    #The function which reads files to gather data until batch size is satisfied
    def batch_helper(self, fileindex, position, batch_size):
        '''
        Yields batches of data of size N
        '''
        f = h5py.File(self.filelist[fileindex],'r')
        self.filesize = np.array(f['ECAL']).shape[0]
        #print(self.filelist[fileindex],'first')
        if (position + batch_size < self.filesize):
            data_ECAL = np.array(f['ECAL'][position : position + batch_size])
            data_HCAL = np.array(f['HCAL'][position : position + batch_size])
            target = np.array(f['target'][position : position + batch_size][:,:,0:2])
            #target = np.delete(target,0,1)

            position += batch_size
            f.close()
            #print('first position',position)
            return data_ECAL,data_HCAL, target, fileindex, position
        
        else:
            data_ECAL = np.array(f['ECAL'][position : position + batch_size])
            data_HCAL = np.array(f['HCAL'][position : position + batch_size])
            target = np.array(f['target'][position:][:,:,0:2])
            #target = np.delete(target,0,1)
            f.close()
            
            if (fileindex+1 < len(self.filelist)):
                if(self.batch_size-data_ECAL.shape[0]>0):
                    while(self.batch_size-data_ECAL.shape[0]>0):
                        if(int(np.floor((self.batch_size-data_ECAL.shape[0])/self.filesize))==0):
                            number_of_files=1
                        else:
                            number_of_files=int(np.ceil((self.batch_size-data_ECAL.shape[0])/self.filesize))
                        for i in xrange(0,number_of_files):
                            if(fileindex+i+1>len(self.filelist)):
                                fileindex=0
                                number_of_files=number_of_files-i
                                i=0
                            f = h5py.File(self.filelist[fileindex+i+1],'r')
                            #print(self.filelist[fileindex+i+1],'second')
                            if (self.batch_size-data_ECAL.shape[0]<self.filesize):
                                position = self.batch_size-data_ECAL.shape[0]
                                data_temp_ECAL = np.array(f['ECAL'][position : position + batch_size])
                                data_temp_HCAL = np.array(f['HCAL'][position : position + batch_size])
                                target_temp = np.array(f['target'][:position][:,:,0:2])
                            else:
                                data_temp_ECAL = np.array(f['ECAL'][position : position + batch_size])
                                data_temp_HCAL = np.array(f['HCAL'][position : position + batch_size])
                                target_temp = np.array(f['target'][:,:,0:2])
                            f.close()
                    #data_, target_, fileindex, position = self.batch_helper(fileindex + 1, 0, batch_size - self.filesize+position)
                            #print( data.shape,data_.shape)
                            #print( target.shape,target_.shape)
                            data_ECAL = np.concatenate((data_ECAL, data_temp_ECAL), axis=0)
                            data_HCAL = np.concatenate((data_HCAL, data_temp_HCAL), axis=0)
                            target = np.concatenate((target, target_temp), axis=0)
                    if (fileindex +i+1<len(self.filelist)):
                        fileindex = fileindex +i+1
                    else:
                        fileindex = 0
                else:
                    position = 0
                    fileindex=fileindex+1
            else:
                fileindex = 0
                position = 0
            
            return data_ECAL,data_HCAL, target, fileindex, position
    #The function which loops indefinitely and continues to return data of the specified batch size
    def batches(self, cnn):
        while (self.fileindex < len(self.filelist)):
            data_ECAL,data_HCAL, target, self.fileindex, self.position = self.batch_helper(self.fileindex, self.position, self.batch_size)
            if data_ECAL.shape[0]!=self.batch_size:
                continue
            if cnn==True:
                data_ECAL = data_ECAL.reshape((data_ECAL.shape[0],)+(1, 24, 24, 25))
                data_HCAL = data_HCAL.reshape((data_HCAL.shape[0],)+(1, 4, 4, 60))
                #data = np.swapaxes(data, 1, 3)
                #data = np.swapaxes(data, 1, 2)
                #data = np.swapaxes(data, 0, 1)
                #data=data.reshape((data.shape[0],1,20,20,25))
                
            else:
                data_ECAL= np.reshape(data_ECAL,(self.batch_size,-1))
                data_HCAL= np.reshape(data_HCAL,(self.batch_size,-1))
            yield ([data_ECAL,data_HCAL],[target[:,:,0],target[:,:,1]/500.])
        self.fileindex = 0

#The first model
input1 = Input(shape = (1, 24, 24, 25))
model1 = Convolution3D(3, 4, 4, 5, input_shape = (1, 24, 24, 25), activation='relu') (input1)
model1 = MaxPooling3D()(model1)
model1 = Flatten()(model1)

input2 = Input(shape = (1, 4, 4, 60))
model2 = Convolution3D(3, 3, 3, 4, input_shape = (1, 4, 4, 60), activation='relu')(input2)
model2 = MaxPooling3D()(model2)
model2 = Flatten()(model2)

## join the two
bmodel = merge([model1,model2], mode='concat')

## fully connected ending
bmodel = (Dense(1000, activation='sigmoid')) (bmodel)
bmodel = (Dropout(0.5)) (bmodel)

oe = Dense(1,activation='sigmoid', name='energy')(bmodel)

model = Model(input=[input1,input2], output=[oe])
model.compile(loss=['mse'], optimizer='sgd')

check = ModelCheckpoint(filepath="./bcnn.hdf5", verbose=1)
early = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')

ds1 = RegGenFull(5000)
vs1 = RegGenFull(5000)
hist = model.fit_generator(ds1.train(cnn=True), samples_per_epoch=50000, nb_epoch=30, validation_data= vs1.validation(cnn=True), nb_val_samples=50000, verbose=1, callbacks=[check,early])
           