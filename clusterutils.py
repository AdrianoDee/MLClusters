from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import gzip
import random
import h5py as h5

np.set_printoptions(threshold=np.nan)

import tensorflow as tf

from os import listdir
from os.path import isfile, join

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes
from sklearn.utils import shuffle


def etalabel(eta):

        if eta <= -1.0:
                return (0.0,0.0)
        if -1.0< eta <= 0.0:
                return (0.0,1.0)
        if 0.0 < eta <= 1.0:
                return (1.0,0.0)
        if eta > 1.0:
                return (1.0,1.0)

def ptlabel(pt):

        if pt < 1.5:
                return 0.0
        else:
                return 1.0

def zetalabel(zeta):
        if zeta<= 0.:
                return 0.0
        else:
                return 1.0

datalabs = ["run","evt","detSeqIn","detSeqOut","inX","inY","inZ",
"outX","outY","outZ","detCounterIn","detCounterOut","isBarrelIn","isBarrelOut",
"layerIn","ladderIn","moduleIn","sideIn","diskIn","panelIn","bladeIn",
"layerOut","ladderOut","moduleOut","sideOut","diskOut","panelOut","bladeOut",
"nId","outId","isBigIn","isEdgIn","isBadIn","isBigOut","isEdgOut","isBadOut",
"inPix1","inPix2","inPix3","inPix4","inPix5","inPix6","inPix7","inPix8",
"inPix9","inPix10","inPix11","inPix12","inPix13","inPix14","inPix15","inPix16",
"inPix17","inPix18","inPix19","inPix20","inPix21","inPix22","inPix23","inPix24",
"inPix25","inPix26","inPix27","inPix28","inPix29","inPix30","inPix31","inPix32",
"inPix33","inPix34","inPix35","inPix36","inPix37","inPix38","inPix39","inPix40",
"inPix41","inPix42","inPix43","inPix44","inPix45","inPix46","inPix47","inPix48",
"inPix49","inPix50","inPix51","inPix52","inPix53","inPix54","inPix55","inPix56",
"inPix57","inPix58","inPix59","inPix60","inPix61","inPix62","inPix63","inPix64",
"outPix1","outPix2","outPix3","outPix4","outPix5","outPix6","outPix7","outPix8",
"outPix9","outPix10","outPix11","outPix12","outPix13","outPix14","outPix15","outPix16",
"outPix17","outPix18","outPix19","outPix20","outPix21","outPix22","outPix23","outPix24",
"outPix25","outPix26","outPix27","outPix28","outPix29","outPix30","outPix31","outPix32",
"outPix33","outPix34","outPix35","outPix36","outPix37","outPix38","outPix39","outPix40",
"outPix41","outPix42","outPix43","outPix44","outPix45","outPix46","outPix47","outPix48",
"outPix49","outPix50","outPix51","outPix52","outPix53","outPix54","outPix55","outPix56",
"outPix57","outPix58","outPix59","outPix60","outPix61","outPix62","outPix63","outPix64",
"evtNum","idTrack","px","py","pz","pt","mT","eT","mSqr","rapidity","etaTrack","phi",
"pdgId","charge","noTrackerHits","noTrackerLayers","dZ","dXY","Xvertex",
"Yvertex","Zvertex","bunCross","isCosmic","chargeMatch","sigMatch"]


infolabs = ["run","evt","detSeqIn","detSeqOut","inX","inY","inZ",
"outX","outY","outZ","detCounterIn","detCounterOut","isBarrelIn","isBarrelOut",
"layerIn","ladderIn","moduleIn","sideIn","diskIn","panelIn","bladeIn",
"layerOut","ladderOut","moduleOut","sideOut","diskOut","panelOut","bladeOut",
"nId","outId","isBigIn","isEdgIn","isBadIn","isBigOut","isEdgOut","isBadOut",
"evtNum","idTrack","px","py","pz","pt","mT","eT","mSqr","rapidity","etaTrack","phi",
"pdgId","charge","noTrackerHits","noTrackerLayers","dZ","dXY","Xvertex",
"Yvertex","Zvertex","bunCross","isCosmic","chargeMatch","sigMatch"]

anylabs = ["run","evt","detSeqIn","detSeqOut","inX","inY","inZ",
"outX","outY","outZ","detCounterIn","detCounterOut","isBarrelIn","isBarrelOut",
"layerIn","ladderIn","moduleIn","sideIn","diskIn","panelIn","bladeIn",
"layerOut","ladderOut","moduleOut","sideOut","diskOut","panelOut","bladeOut",
"nId","outId","isBigIn","isEdgIn","isBadIn","isBigOut","isEdgOut","isBadOut",
"evtNum"]

singlabs = ["bunCross","sigMatch"]
multlabs = ["run","evt","detSeqIn","detSeqOut","detCounterIn","detCounterOut",
"layerIn","ladderIn","moduleIn","sideIn","diskIn","panelIn","bladeIn",
"layerOut","ladderOut","moduleOut","sideOut","diskOut","panelOut","bladeOut",
"nId","outId","evtNum","idTrack","pdgId","charge","noTrackerHits","noTrackerLayers"]

ranglabs = ["inX","inY","inZ","outX","outY","outZ","px","py","pz","pt","mT","eT","mSqr","rapidity","etaTrack","phi",
"pdgId","dZ","dXY","Xvertex","Yvertex","Zvertex"]

infoinds = [n for n in range(len(infolabs))]
datainds = [n for n in range(len(datalabs))]
infodict = dict(zip(infolabs,infoinds))
datadict = dict(zip(datalabs,datainds))


def labelstoneurons(labels_dense, num_classes):
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot

#LABEL STRUCTURE
# -m notmatched
# -e1 eta <=-1
# -e2 -1<eta<=0
# -e3 0<eta<=1
# -e4 eta>1
# -p1 pt<=1.5
# -p2 pt>1.5
# -z1 zV<=0
# -z2 zV>0
# num_classes = 1 + 4 * 2 * 2 = 17

# def clustersLabel():
#     labelstoneurons()

def neuronLabels(data,num_classes=17):
    numbers = classNumbers(data)
    return labelsToNeurons(numbers,num_classes)

def classNumbers(data):
    print(data.shape)
    ets = data[:,infodict["etaTrack"]]
    zVs = data[:,infodict["Zvertex"]]
    # print(zVs)
    pTs = data[:,infodict["pt"]]
    mat = data[:,infodict["pdgId"]]!=0.0

    label = np.zeros(data.shape[0])

    All = np.c_[ets,zVs,pTs,mat,label]


    for doublet in All:
        if bool(doublet[3]==1.0):
            etLab = np.array(etalabel(doublet[0]))
            zvLab = np.array(zetalabel(doublet[1]))
            ptLab = np.array(ptlabel(doublet[2]))

            labelBin = np.hstack((etLab,zvLab,ptLab))

            labeltwo = tuple(map(lambda (i,x):x*2**i,enumerate(labelBin)))

            doublet[4] = reduce((lambda x,y:x+y),labeltwo) + 1

    return All[:,4]

def labelsToNeurons(labels, num_classes):
  num_labels = int(labels.shape[0])

  neurons = np.zeros((num_labels, int(num_classes)))

  for l,lo in zip(labels,neurons):
      lo[int(l)] = 1.0

  return neurons

def datasetload(path='./datasets/',delimit='\t',fileslimit =-1,writetohd = False,type=""):

    datafiles = np.array([f for f in listdir(path) if (isfile(join(path, f)) and  f.lower().endswith(('.txt',".zip")))])
    print(fileslimit)
    if fileslimit > 0:
        datafiles = datafiles[:fileslimit]

    print("Loading " + str(len(datafiles)) + " dataset file(s) . . .")

    datasets  = []
    for filename in datafiles:
        if filename.lower().endswith('.txt'):
            with open(path + filename, 'rb') as f:
                print ("Reading clusters from txt :",f.name)
                f.seek(0)
                data = np.genfromtxt(f,delimiter=delimit,dtype = np.float32)
                datasets.append(data)

        if filename.lower().endswith('.zip'):
            with gzip.open(path + filename, 'rb') as f:
                print ("Reading clusters from zip :",f.name)
                f.seek(0)
                data = np.genfromtxt(f,delimiter=delimit,dtype = np.float32)
                datasets.append(data)

    data = np.vstack(datasets)

    if writetohd:
        print()

    return data

def datafiltering(filters,alldata,savetoh=False,shuffle=False,sanitize=False,sanratio=0.5):
    if not isinstance(filters, dict):
        print("Filters to the dataset must be provided in the for of a dictionary of lists: ")
        print("[ e.g. filters = {'inX':[0.4,1.2],'etaTrack':[-1.3,1.5]} ]")
        return alldata
    else:
        oldsize = alldata.shape[0]
        print(oldsize)
        for k, r in filters.iteritems():
            if k in singlabs:
                if isinstance(r,(float,int)):
                    print("Applying filter for " + str(k) + " with value = " + str(r)  + " . . .")
                    if k in anylabs:
                        alldata = alldata[alldata[:,datadict[k]]==r]
                    else:
                        alldata = alldata[np.logical_or(alldata[:,datadict["pdgId"]]==0.0 , alldata[:,datadict[k]]==r)]
                    print(" - " + str(oldsize-alldata.shape[0]) + " doublets dropped.")
                    oldsize = alldata.shape[0]
                else:
                    print("Label " + str(k) + "needs an int or a float single as filter.\n Not a " + str(type(r)) + ". Skipping.")
            if k in multlabs:
                if isinstance(r,(list,tuple)):
                    if all(isinstance(x,(int,float)) for x in r):
                        print("Applying filter for " + str(k) + " with values = " + str(r) + " . . .")
                        if k in anylabs:
                            alldata = alldata[np.logical_or.reduce([alldata[:,datadict[k]] == x for x in r])]
                        else:
                            alldata = alldata[np.logical_or(np.logical_or.reduce([alldata[:,datadict[k]] == x for x in r]), alldata[:,datadict["pdgId"]]==0.0)]
                        print(" - " + str(oldsize-alldata.shape[0]) + " doublets dropped.")
                        oldsize = alldata.shape[0]
                else:
                    print("Label " + str(k) + "needs a (int or float) list or a tuple as filter.\n Not a " + str(type(r)) + ". Skipping.")
            if k in ranglabs:
                if isinstance(r,(list,tuple)):
                    if all(isinstance(x,(int,float)) for x in r) and len(r)==2:
                        print("Applying filter for " + str(k) + " with range = [" + str(min(r)) + ";" + str(max(r)) + "] . . .")
                        if k in anylabs:
                            alldata = alldata[np.logical_and(alldata[:,datadict[k]] < max(r),alldata[:,datadict[k]] > min(r)) ]
                        else:
                            alldata = alldata[np.logical_or(alldata[:,datadict["pdgId"]]==0.0, np.logical_and(alldata[:,datadict[k]] < max(r), alldata[:,datadict[k]] > min(r))) ]
                        print(" - " + str(oldsize-alldata.shape[0]) + " doublets dropped.")
                        oldsize = alldata.shape[0]
                    else:
                        print("Label " + str(k) + "needs a (int or float) list or a tuple as filter of size two.\n Not a " + str(type(r)) + " of size " + str(len(r)) + ". Skipping.")

                else:
                    print("Label " + str(k) + "needs a (int or float) list or a tuple as filter of size two.\n Not a " + str(type(r)) + " of size " + str(len(r)) + ". Skipping.")
            if not (k in ranglabs or k in singlabs or k in multlabs):
                print(str(k) + " is not a valid label! Skipping.")

        print("All " + str(len(filters)) + " filter(s) applied. ")
        if sanitize and sanratio<=1.0:
            print("Balancing fake and true doublets content with ratio : " + str(sanratio) + " true doublets")
            oldsize = alldata.shape[0]
            print(" - tot number of doublets : " + str(oldsize))

            fakes = alldata[alldata[:,datadict["pdgId"]]==0.0]
            trues = alldata[alldata[:,datadict["pdgId"]]!=0.0]

            print(" \t -> fake = " + str(fakes.shape[0]))
            print(" \t -> true = " + str(trues.shape[0]))

            #ASSUMING FAKES ALWAYS > TRUES
            if(trues.shape[0]/oldsize>sanratio):
                idx = np.random.randint(int(trues.shape[0]), size=int(sanratio*oldsize))
                trues = trues[idx,:]
            else:
                newsize = trues.shape[0]/sanratio
                idx = np.random.randint(int(trues.shape[0]), size=int((1.0-sanratio)*newsize))
                fakes = fakes[idx,:]

            data = np.vstack((trues,fakes))

            np.take(data,np.random.rand(data.shape[0]).argsort(),axis=0,out=data)

            alldata = data

            fakes = alldata[alldata[:,datadict["pdgId"]]==0.0]
            trues = alldata[alldata[:,datadict["pdgId"]]!=0.0]

            print("After balancing: ")
            print(" \t -> fake = " + str(fakes.shape[0]))
            print(" \t -> true = " + str(trues.shape[0]))

        return alldata

def clustersInput(alldata,cols=8,rows=8,dropEdge=True,dropBad=True,dropBig=True,dropCosmic=True,dropCharge=True):

    nclusts = alldata.shape[0]

    infos=(alldata.shape[1]-2*rows*cols)

    if dropEdge:
        print("Dropping clusters with pixels on edge . . .")
        alldata = alldata[alldata[:,datadict["isEdgIn"]]==float(False)]
        alldata = alldata[alldata[:,datadict["isEdgOut"]]==float(False)]
        print(" - " + str(nclusts-alldata.shape[0]) + " dropped.")
        nclusts = alldata.shape[0]

    if dropBig:
        print("Dropping clusters with big pixels . . .")
        alldata = alldata[alldata[:,datadict["isBigIn"]]==float(False)]
        alldata = alldata[alldata[:,datadict["isBigOut"]]==float(False)]
        print(" - " + str(nclusts-alldata.shape[0]) + " dropped.")
        nclusts = alldata.shape[0]

    if dropBad:
        print("Dropping clusters with bad pixels . . .")
        alldata = alldata[alldata[:,datadict["isBadIn"]]==float(False)]
        alldata = alldata[alldata[:,datadict["isBadOut"]]==float(False)]
        print(" - " + str(nclusts-alldata.shape[0]) + " dropped.")
        nclusts = alldata.shape[0]

    if dropCosmic:
        print("Dropping clusters from cosmics . . .")
        alldata = alldata[alldata[:,datadict["isCosmic"]]==float(False)]
        print(" - " + str(nclusts-alldata.shape[0]) + " dropped.")
        nclusts = alldata.shape[0]

    if dropCharge:
        print("Dropping clusters from not qMatched tracks . . .")
        alldata = alldata[alldata[:,datadict["chargeMatch"]]==float(False)]
        print(" - " + str(nclusts-alldata.shape[0]) + " dropped.")
        nclusts = alldata.shape[0]

    # print(alldata[:,-7:])

    alldata = alldata.reshape(nclusts,2*rows*cols+infos)

    cltdata = alldata[:,datadict["inPix1"]:datadict["evtNum"]]
    cltdata = np.multiply(cltdata, 1.0 / 65535.0)
    infdata = np.append(alldata[:,0:datadict["inPix1"]],alldata[:,datadict["evtNum"]:],axis=1)

    labels = neuronLabels(infdata)
    cltdata=cltdata.reshape(nclusts,rows,cols,2,1)
    #labdata = clustersLabel(infdata)
    return (cltdata,labels)
