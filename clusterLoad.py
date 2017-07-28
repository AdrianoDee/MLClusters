import argparse

import os
import sys
import gzip
import time
import pandas as pd
from pandas import DataFrame

import numpy as np

from os import listdir
from os.path import isfile, join
from sets import Set

from dataLabels import dataLab,tailLab,outPixLab,inPixLab,headLab,XYZ,inXYZ,outXYZ

from keras.utils.np_utils import to_categorical

import matplotlib.pyplot as plt

clusterSize = 15

modes = ["det","lay"]

detFlag = ["isBarrelIn","isBarrelOut"]
detCombs = np.array([[1.,1.],[1.,0.],[0.,0.]])

layFlag = ["detCounterIn","detCounterOut"]

layCombs = np.array([[0.,1.],[0.,2.],[0.,3.],[0.,4.],#barrels
            [1.,2.],[1.,3.],[1.,4.],
            [2.,3.],[3.,4.],
            [3.,4.],
            [4.,5.],[4.,6.],[4.,7.],#pos
            [5.,6.],[5.,6.],
            [5.,6.],
            [4.,5.],[4.,6.],[4.,7.],#neg
            [5.,6.],[5.,6.],
            [5.,6.]])

def dataToPandas(path,fileslimit,ext="h5"):

    start = time.time()

    allFiles = np.array([f for f in listdir(path) if (isfile(join(path, f)) and  f.lower().endswith((ext)))])

    if fileslimit < 0:
        fileslimit = len(allFiles)

    if len(allFiles) > fileslimit > 0:

        idx = np.random.randint(int(allFiles.shape[0]), size=int(fileslimit))
        allFiles = allFiles[idx]

    alldata = pd.DataFrame()

    listdata = []

    for no,f in enumerate(allFiles):
        print ("Reading clusters from csv file no."+ str(no+1) +" :" + path + f)
        if "h5" in ext:
            df = pd.read_hdf(path + f)
        else:
            df = pd.read_csv(path + f,index_col=None, header=0)
        listdata.append(df)

    alldata = pd.concat(listdata)

    end = time.time()
    print ("======================================================================")
    print ("\n - Timing : " + str(end-start))

    return alldata

def csvDumps(path,data):

    if args.mode == "det":
        choices = detCombs
        flags   = detFlag
    if args.mode == "all":
        choices = layCombs
        flags   = layFlag

    for c in choices:
        detsPath = args.read
        df = data
        for F,C in zip(flags,c):
            df = df[df[F]==C]
            detsPath += str(F) + "_" + str(int(C))

        if not os.path.exists(detsPath):
            os.makedirs(detsPath)

        df.to_csv(detsPath + "/doublets.csv",chunksize=args.split)


def filtersData(dataset,key,val):

    datafil = dataset[dataset[key] == val]

    print("========================================")
    print("Filtering")
    print(" - Filter: " + str(key))
    print(" - Values: " + str(val))
    print(" - Dropped: " + str(dataset.shape[0] - datafil.shape[0]))


    return datafil

def ladderTop(dataset):

    #datafil = dataset[dataset["detCounterIn"]==0.0]
    #datafil = dataset[dataset["detCounterOut"]==1.0]
    topLadderIn = dataset[dataset["ladderIn"]!=0.0]
    topLadderIn = dataset[["ladderIn","ladderOut"]].nlargest(1,["ladderIn","ladderOut"])[0:]

    datafil = dataset[dataset[["ladderIn","ladderOut"]]==topLadderIn]

    print("========================================")
    print("Filtering - Ladder")
    print(" - Dropped: " + str(dataset.shape[0] - datafil.shape[0]))


    return datafil

def sanitizData(dataset,ratio=0.5):

    trues = dataset[dataset["pdgId"] != 0.0]
    false = dataset[dataset["pdgId"] == 0.0]

    if trues.shape[0] < ratio * (false.shape[0] + trues.shape[0]):

        print("========================================")
        print("Balancing with true ratio: " + str(ratio))
        print(" -> trues :" + str(trues.shape[0]))
        print(" -> fakes :" + str(false.shape[0]))
        print("========================================")

        n_false = trues.shape[0] * ((1.0 - ratio) / (ratio))
        false = false.sample(n=int(n_false))
        print(" -> new fakes :" + str(false.shape[0]))

        newdata = pd.concat((trues,false))
        newdata = newdata.sample(frac=1.0)
        print(" -> data size :" + str(newdata.shape[0]))

        return newdata
    else:
        return dataset


def clusterData(dataset,chans=True,theta=True,phi=True,plots=False):

    labels = np.logical_not(dataset["pdgId"]!=0.0).astype(float)
    labels = to_categorical(labels, num_classes=2)

    #print(dataset["isBarrelIn"].head())
    #print(dataset.shape)
    inClust  = dataset[inPixLab].values
    outClust = dataset[outPixLab].values

    inClust = np.multiply(inClust, 1.0 / 65535.)
    outClust = np.multiply(outClust, 1.0 / 65535.)

    R = np.hstack((inClust,outClust))
    clustData=np.vstack((inClust,outClust))
    if theta:

        #phi correction
        cosPhiIns = np.cos(np.arctan(np.multiply(dataset["inY"],1.0/dataset["inX"])))
        cosPhiOuts = np.cos(np.arctan(np.multiply(dataset["outY"],1.0/dataset["outX"])))
        sinPhiIns = np.sin(np.arctan(np.multiply(dataset["inY"],1.0/dataset["inX"])))
        sinPhiOuts = np.sin(np.arctan(np.multiply(dataset["outY"],1.0/dataset["outX"])))

        inPhiModC = np.multiply(inClust,cosPhiIns[:,np.newaxis])
        outPhiModC = np.multiply(outClust,cosPhiOuts[:,np.newaxis])

        inPhiModS = np.multiply(inClust,sinPhiIns[:,np.newaxis])
        outPhiModS = np.multiply(outClust,sinPhiOuts[:,np.newaxis])

        inPhiC  = inClust  + inPhiModC
        outPhiC = outClust + outPhiModC

        inPhiS  = inClust  + inPhiModS
        outPhiS = outClust + outPhiModS

        # clustData = np.hstack((clustData,inPhiC,outPhiC,inPhiS,outPhiS))

        G = np.hstack((inPhiC,outPhiC))
        B = np.hstack((inPhiS,outPhiS))
	clustData=np.vstack((clustData,inPhiC,outPhiC,inPhiS,outPhiS))
    if phi:

        #theta correction
        cosThetaIns = np.cos(np.arctan(np.multiply(dataset["inY"],1.0/dataset["inZ"])))
        cosThetaOuts = np.cos(np.arctan(np.multiply(dataset["outY"],1.0/dataset["outZ"])))
        sinThetaIns = np.sin(np.arctan(np.multiply(dataset["inY"],1.0/dataset["inZ"])))
        sinThetaOuts = np.sin(np.arctan(np.multiply(dataset["outY"],1.0/dataset["outZ"])))

        inThetaModC = np.multiply(inClust,cosThetaIns[:,np.newaxis])
        outThetaModC = np.multiply(outClust,cosThetaOuts[:,np.newaxis])

        inThetaModS = np.multiply(inClust,sinThetaIns[:,np.newaxis])
        outThetaModS = np.multiply(outClust,sinThetaOuts[:,np.newaxis])

        inThetaC  = inClust  + inThetaModC
        outThetaC = outClust + outThetaModC

        inThetaS  = inClust  + inThetaModS
        outThetaS = outClust + outThetaModS

        # clustData = np.hstack((clustData,inThetaC,outThetaC,inThetaS,outThetaS))

        C = np.hstack((inThetaC,outThetaC))
        K = np.hstack((inThetaS,outThetaS))
	clustData=np.vstack((clustData,inThetaC,outThetaC,inThetaS,outThetaS))

    clustData=np.stack((inClust,outClust,inThetaC,outThetaC,inThetaS,outThetaS,inPhiC,outPhiC,inPhiS,outPhiS),axis=-1)
    clustData = clustData.reshape(labels.shape[0],clusterSize,clusterSize,-1)

    if plots:
        allDone = [False,False]

        for i in range(1000):
            if not allDone[0] and labels[i][0]!=0.0:
                continue
            else:
                allDone[0] = True
                plt.title("sample_true" + str(i) + "_clustIn")
                plt.imshow(inClust[i])
                plt.show()
                plt.clf()
                plt.cla()
                plt.clf()
                plt.title("sample_true" + str(i) + "_clustIn")
                plt.imshow(outClust[i])
                plt.show()
                plt.clf()
                plt.cla()
                plt.clf()
                continue

            if not allDone[1] and labels[i][1]!=0.0:
                continue
            else:
                plt.title("sample_fake" + str(i) + "_clustIn")
                plt.imshow(inClust[i])
                plt.show()
                plt.clf()
                plt.cla()
                plt.clf()
                plt.title("sample_fake" + str(i) + "_clustIn")
                plt.imshow(outClust[i])
                plt.show()
                plt.clf()
                plt.cla()
                plt.clf()
                continue

            if not allDone[1] and allDone[0]:
                break

    return (clustData,labels)

if __name__ == '__main__':

    if args.mode not in modes:
        print("Not a suitable choice. Choose in :")
        print(modes)
        sys.exit()

    df = csvLoad(path=args.read,fileslimit=args.flimit)
    d = sanitizData(df,ratio=0.1)
    d = filtersData(df,key="detCounterIn",val=0.0)
    c = clusterData(d)
    #csvDumps(path=args.read,data=df)
    #print (detsnum)
