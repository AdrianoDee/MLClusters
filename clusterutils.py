from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import gzip
import random
import h5py as h5
import operator
#from ROOT import TH3F, TH2F, TCanvas

from math import cos,sin

from time import gmtime, strftime

np.set_printoptions(threshold=np.nan)

import tensorflow as tf

from os import listdir
from os.path import isfile, join

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes

import time


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

datalabs = ["run","evt","detSeqIn","detSeqOut","inZ","inX","inY","outZ",
"outX","outY","detCounterIn","detCounterOut","isBarrelIn","isBarrelOut",
"layerIn","ladderIn","moduleIn","sideIn","diskIn","panelIn","bladeIn",
"layerOut","ladderOut","moduleOut","sideOut","diskOut","panelOut","bladeOut",
"inId","outId","isBigIn","isEdgIn","isBadIn","isBigOut","isEdgOut","isBadOut",#"isFlippedIn","isFlippedOut",
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
"dummyFlag","idTrack","px","py","pz","pt","mT","eT","mSqr","rapidity","etaTrack","phi",
"pdgId","charge","noTrackerHits","noTrackerLayers","dZ","dXY","Xvertex",
"Yvertex","Zvertex","bunCross","isCosmic","chargeMatch","sigMatch"]


infolabs = ["run","evt","detSeqIn","detSeqOut","inZ","inX","inY",
"outZ","outX","outY","detCounterIn","detCounterOut","isBarrelIn","isBarrelOut",
"layerIn","ladderIn","moduleIn","sideIn","diskIn","panelIn","bladeIn",
"layerOut","ladderOut","moduleOut","sideOut","diskOut","panelOut","bladeOut",
"inId","outId","isBigIn","isEdgIn","isBadIn","isBigOut","isEdgOut","isBadOut",#"isFlippedIn","isFlippedOut",
"dummyFlag","idTrack","px","py","pz","pt","mT","eT","mSqr","rapidity","etaTrack","phi",
"pdgId","charge","noTrackerHits","noTrackerLayers","dZ","dXY","Xvertex",
"Yvertex","Zvertex","bunCross","isCosmic","chargeMatch","sigMatch"]

anylabs = ["run","evt","detSeqIn","detSeqOut","inZ","inX","inY",
"outX","outY","outZ","detCounterIn","detCounterOut","isBarrelIn","isBarrelOut",
"layerIn","ladderIn","moduleIn","sideIn","diskIn","panelIn","bladeIn",
"layerOut","ladderOut","moduleOut","sideOut","diskOut","panelOut","bladeOut",
"inId","outId","isBigIn","isEdgIn","isBadIn","isBigOut","isEdgOut","isBadOut",
"dummyFlag"]

singlabs = ["bunCross","sigMatch"]
multlabs = ["run","evt","detSeqIn","detSeqOut","detCounterIn","detCounterOut","isBarrelIn","isBarrelOut",
"layerIn","ladderIn","moduleIn","sideIn","diskIn","panelIn","bladeIn",
"layerOut","ladderOut","moduleOut","sideOut","diskOut","panelOut","bladeOut",
"inId","outId","dummyFlag","idTrack","pdgId","charge","noTrackerHits","noTrackerLayers"]

ranglabs = ["inZ","inX","inY","outZ","outX","outY","px","py","pz","pt","mT","eT","mSqr","rapidity","etaTrack","phi",
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

def neuronLabels(data,num_classes=2):
    numbers = classNumbers(data)
    return labelsToNeurons(numbers,num_classes)

def classNumbers(data):
    # print(data.shape)
    ets = data[:,infodict["etaTrack"]]
    zVs = data[:,infodict["Zvertex"]]
    # print(zVs)
    pTs = data[:,infodict["pt"]]
    mat = data[:,infodict["pdgId"]]!=0.0

    labels = []
    # print(mat)

    All = np.c_[ets,zVs,pTs,mat]


    for doublet in All:
        # print(doublet[3])
        if bool(doublet[3]==1.0):
            # label = np.array([0.0,1.0])
            labels.append(0.0)
        else:
            # label = np.array([1.0,0.0])
            labels.append(1.0)

        # if bool(doublet[3]==1.0):
        #     print("-")
        #     etLab = np.array(etalabel(doublet[0]))
        #     zvLab = np.array(zetalabel(doublet[1]))
        #     ptLab = np.array(ptlabel(doublet[2]))
        #
        #     labelBin = np.hstack((etLab,zvLab,ptLab))
        #
        #     labeltwo = tuple(map(lambda (i,x):x*2**i,enumerate(labelBin)))
        #
        #     doublet[4] = reduce((lambda x,y:x+y),labeltwo) + 1
    labels = np.array(labels)
    print(labels.shape)
    return labels

def labelsToNeurons(labels, num_classes):
  num_labels = int(labels.shape[0])

  neurons = np.zeros((num_labels, int(num_classes)))

  for l,lo in zip(labels,neurons):
      lo[int(l)] = 1.0

  return neurons

def datastats(data,printstat=False,mode="ladder"):

    anylabs = ["run","evt","detSeqIn","detSeqOut","inX","inY","inZ",
    "outX","outY","outZ","detCounterIn","detCounterOut","isBarrelIn","isBarrelOut",
    "layerIn","ladderIn","moduleIn","sideIn","diskIn","panelIn","bladeIn",
    "layerOut","ladderOut","moduleOut","sideOut","diskOut","panelOut","bladeOut",
    "inId","outId","isBigIn","isEdgIn","isBadIn","isBigOut","isEdgOut","isBadOut",
    "dummyFlag"]

    if mode=="ladder":
        r = np.r_[datadict["detCounterIn"]:datadict["moduleIn"],[datadict["layerOut"],datadict["ladderOut"]]]
    else:
        if mode=="disk":
            r = np.r_[datadict["detCounterIn"]:datadict["ladderIn"],[datadict["diskIn"],datadict["layerOut"],datadict["diskOut"]]]
        else:
            if mode=="ladderdisk":
                r = np.r_[datadict["detCounterIn"]:datadict["moduleIn"],[datadict["layerOut"],datadict["diskOut"]]]
            else:
                print("Not proper mode selected (ladder,disk,diskladder,ladderdisk). Autoselect: ladder.")
                mode="ladder"
                r = np.r_[datadict["detCounterIn"]:datadict["moduleIn"],[datadict["layerOut"],datadict["ladderOut"]]]

    datadets = data[:,r]

    detslabs = np.array(datalabs)[r]
    if mode=="ladder":
        datadets = datadets[np.logical_and(datadets[:,2]==1.0 , datadets[:,3]==1.0)]
    else:
        if mode=="disk":
            datadets = datadets[np.logical_and(datadets[:,2]==0.0 , datadets[:,3]==0.0)]
        else:
            if mode=="ladderdisk":
                datadets = datadets[np.logical_and(datadets[:,2]==1.0 , datadets[:,3]==0.0)]

    print("Ladders and counter stats:")

    detsdict = {}
    for k in datadets:
        kl = tuple(k)
        if kl in detsdict:
            detsdict[kl] +=1
        else:
            detsdict[kl] = 1

    sorted_detsdict = sorted(detsdict.items(), key=operator.itemgetter(1))[::-1]
    if printstat:
        for v in sorted_detsdict:
            print(str(v[0]) + "\t\t -->\t" + str(v[1]) + "/" + str(datadets.shape[0]))

    print("Most populated " + mode + " couple:\n" +str(list(sorted_detsdict[0][0])))

    topmodule = [[float(num)] for num in list(sorted_detsdict[0][0])]

    filterStat = dict(zip(detslabs,topmodule))
    return filterStat

def datamodule(data,printstat=False):

    anylabs = ["run","evt","detSeqIn","detSeqOut","inX","inY","inZ",
    "outX","outY","outZ","detCounterIn","detCounterOut","isBarrelIn","isBarrelOut",
    "layerIn","ladderIn","moduleIn","sideIn","diskIn","panelIn","bladeIn",
    "layerOut","ladderOut","moduleOut","sideOut","diskOut","panelOut","bladeOut",
    "inId","outId","isBigIn","isEdgIn","isBadIn","isBigOut","isEdgOut","isBadOut",
    "dummyFlag"]


    r = np.r_[datadict["detCounterIn"]:datadict["sideIn"],[datadict["layerOut"],datadict["layerOut"],datadict["ladderOut"],datadict["moduleOut"]]]
    datadets = data[:,r]

    detslabs = np.array(datalabs)[r]
    datadets = datadets[np.logical_and(datadets[:,2]==1.0 , datadets[:,3]==1.0)]
    print("Modules and counter stats:")

    detsdict = {}
    for k in datadets:
        kl = tuple(k)
        if kl in detsdict:
            detsdict[kl] +=1
        else:
            detsdict[kl] = 1

    sorted_detsdict = sorted(detsdict.items(), key=operator.itemgetter(1))[::-1]
    if printstat:
        for v in sorted_detsdict:
            print(str(v[0]) + "\t\t -->\t" + str(v[1]) + "/" + str(datadets.shape[0]))

    print("Most populated module couple:\n" +str(list(sorted_detsdict[0][0])))

    topmodule = [[float(num)] for num in list(sorted_detsdict[0][0])]

    filterStat = dict(zip(detslabs,topmodule))
    return filterStat

def datasetload(path='./datasets/',delimit='\t',fileslimit =-1,writetohd = False,type="",filterstats = False):

    datafiles = np.array([f for f in listdir(path) if (isfile(join(path, f)) and  f.lower().endswith(('.txt',".gz")))])

    if len(datafiles) > fileslimit > 0:

        idx = np.random.randint(int(datafiles.shape[0]), size=int(fileslimit))
        datafiles = datafiles[idx]


    print("Loading " + str(len(datafiles)) + " dataset file(s) . . .")

    datasets  = []
    for no,filename in enumerate(datafiles):
        if filename.lower().endswith('.txt'):
            with open(path + filename, 'rb') as f:
                print ("Reading clusters from txt :"+ str(no+1) +" :",f.name)
                f.seek(0)
                data = np.genfromtxt(f,delimiter=delimit,dtype = np.float32)
                datasets.append(data)

        if filename.lower().endswith('.gz'):
            with gzip.open(path + filename, 'rb') as f:
                print ("Reading clusters from zip :"+ str(no+1) +" :",f.name)
                f.seek(0)
                data = np.genfromtxt(f,delimiter=delimit,dtype = np.float32)
                datasets.append(data)

    data = np.vstack(datasets)

    # print(data[10])
    # minX = np.amin(data[:,datadict["inX"]])
    # minX = min(minX,np.amin(data[:,datadict["outX"]]))
    #
    # minY = np.amin(data[:,datadict["inY"]])
    # minY = min(minY,np.amin(data[:,datadict["outY"]]))
    #
    # minZ = np.amin(data[:,datadict["inZ"]])
    # minZ = min(minZ,np.amin(data[:,datadict["outZ"]]))
    #
    # maxX = np.amax(data[:,datadict["inX"]])
    # maxX = max(maxX,np.amax(data[:,datadict["outX"]]))
    #
    # maxY = np.amax(data[:,datadict["inY"]])
    # maxY = max(maxY,np.amax(data[:,datadict["outY"]]))
    #
    # maxZ = np.amax(data[:,datadict["inZ"]])
    # maxZ = max(maxZ,np.amax(data[:,datadict["outZ"]]))

    # print(data[np.logical_and(data[:,datadict["isFlippedIn"]]==1.,data[:,datadict["isFlippedOut"]]==1.)].shape[0])
    # print(data[np.logical_and(data[:,datadict["isFlippedIn"]]==0.,data[:,datadict["isFlippedOut"]]==1.)].shape[0])
    # print(data[np.logical_and(data[:,datadict["isFlippedIn"]]==1.,data[:,datadict["isFlippedOut"]]==0.)].shape[0])
    # print(data[np.logical_and(data[:,datadict["isFlippedIn"]]==0.,data[:,datadict["isFlippedOut"]]==0.)].shape[0])
    # data = data[np.logical_and(data[:,datadict["isFlippedIn"]]==1.,data[:,datadict["isFlippedOut"]]==1.)]

    # data = data[]

    # print(data[0].shape)
    if writetohd:
        print()

    if filterstats:
        data = datafiltering(datastats(data),data,sanitize=True)

    return data

def datafiltering(filters,alldata,savetoh=False,shuffle=False):
    if not isinstance(filters, dict):
        print("Filters to the dataset must be provided in the for of a dictionary of lists: ")
        print("[ e.g. filters = {'inX':[0.4,1.2],'etaTrack':[-1.3,1.5]} ]")
        return alldata
    else:
        print(filters)
        oldsize = alldata.shape[0]

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

        return alldata

def clustersInput(alldata,cols=8,rows=8,dropEdge=True,dropBad=True,dropBig=True,
dropCosmic=False,dropCharge=False,bAndW=False,angularcorrection=True,sanitize=False,
sanratio=0.5,writesample=False,samplesize=-1.0,writedata=False):

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

        if writesample and samplesize>0:

            fileNP = strftime("%Y-%m-%d_%H_%M_%S_sample_", gmtime())
            fileNP = "./outputs/" + fileNP + str(samplesize) +"_" + str(sanratio*100) +".npz"
            upto = max(trues.shape[0],samplesize)
            truesample = trues[:upto,:]
            upto = max(fakes.shape[0],samplesize)
            fakesample = fakes[:upto,:]

            with open(fileNP,'wb') as fnp:
                np.savez_compressed(fnp,truesample=trues,fakesample=fakes)

            # if runTSNE

        np.take(data,np.random.rand(data.shape[0]).argsort(),axis=0,out=data)

        alldata = data

        fakes = alldata[alldata[:,datadict["pdgId"]]==0.0]
        trues = alldata[alldata[:,datadict["pdgId"]]!=0.0]

        print("After balancing: ")
        print(" \t -> fake = " + str(fakes.shape[0]))
        print(" \t -> true = " + str(trues.shape[0]))

    if writesample and samplesize>0:

        fileNP = strftime("%Y-%m-%d_%H_%M_%S_sample_", gmtime())
        fileNP = "./outputs/" + fileNP + str(samplesize) +"_" + str(sanratio*100) +".npz"
        upto = max(trues.shape[0],samplesize)
        truesample = trues[:upto,:]
        upto = max(fakes.shape[0],samplesize)
        fakesample = fakes[:upto,:]

        with open(fileNP,'wb') as fnp:
            np.savez_compressed(fnp,truesample=trues,fakesample=fakes)


    nclusts = alldata.shape[0]
    alldata = alldata.reshape(nclusts,2*rows*cols+infos)

    # cltdata = alldata[:,datadict["inPix1"]:datadict["dummyFlag"]]
    cltdata = alldata

    if angularcorrection:

        # inXs =
        # inYs =
        cosIns = np.cos(np.arctan(np.multiply(alldata[:,datadict["inY"]],1.0/alldata[:,datadict["inX"]])))
        cosOuts = np.cos(np.arctan(np.multiply(alldata[:,datadict["outY"]],1.0/alldata[:,datadict["outX"]])))
        sinIns = np.sin(np.arctan(np.multiply(alldata[:,datadict["inY"]],1.0/alldata[:,datadict["inX"]])))
        sinOuts = np.sin(np.arctan(np.multiply(alldata[:,datadict["outY"]],1.0/alldata[:,datadict["outX"]])))
        cosZYIns = np.cos(np.arctan(np.multiply(alldata[:,datadict["inY"]],1.0/alldata[:,datadict["inZ"]])))
        cosZYOuts = np.cos(np.arctan(np.multiply(alldata[:,datadict["outY"]],1.0/alldata[:,datadict["outZ"]])))
        # print(cosIns[3])
        # cosIns = cos(alldata[)
        inClust  = alldata[:,datadict["inPix1"]:datadict["outPix1"]]
        outClust = alldata[:,datadict["outPix1"]:datadict["dummyFlag"]]


        # print(inClust[0])
        # print(outClust[0])

        if bAndW:
            print("Applying b&w filter (0 or 1)")
            thresh = inClust  > 0.0
            inClust[thresh]   = 1.0
            thresh = outClust > 0.0
            outClust[thresh]  = 1.0

        else:
            print("Normalising to 16bit maximum")
            inClust = np.multiply(inClust, 1.0 / 65535.0)
            outClust = np.multiply(outClust, 1.0 / 65535.0)

        # print(inClust[0])
        # print(outClust[0])

        print("Applying angular correction")
        inClustModC = np.multiply(inClust,cosIns[:,np.newaxis])
        outClustModC = np.multiply(outClust,cosOuts[:,np.newaxis])

        inClustModS = np.multiply(inClust,sinIns[:,np.newaxis])
        outClustModS = np.multiply(outClust,sinOuts[:,np.newaxis])

        # print(inClustModC[0])
        # print(inClustModS[0])
        # print(cosIns[0])
        #
        #
        # print(outClustModC[0])
        # print(outClustModS[0])
        # print(cosOuts[0])
        # print(cosZYIns)
        # print(cosZYOuts)

        # inClust = np.multiply(inClust,sinIns[:,np.newaxis])
        # outClust = np.multiply(outClust,sinOuts[:,np.newaxis])

        inClustC  = inClust  + inClustModC
        outClustC = outClust + outClustModC

        inClustS  = inClust  + inClustModS
        outClustS = outClust + outClustModS

        # print(inClust[0])
        # print(outClust[0])
        # print("=====")
        # print(inClustC[0])
        # print(outClustC[0])
        # print(inClustS[0])
        # print(outClustS[0])
        # print("=====")
        # print(inClust.shape)
        # print(outClust.shape)
        R = np.hstack((inClust,outClust))
        G = np.hstack((inClustC,outClustC))
        B = np.hstack((inClustS,outClustS))
        cltdata = []
        for r,g,b in zip(R,G,B):
            cltdata.append(r)
            cltdata.append(g)
            cltdata.append(b)

        # cltdata = np.vstack((R,G,B))
        cltdata = np.array(cltdata)
        # print(cltdata.shape)
        # print(cltdata[0])
        # print(cltdata[0].shape)
        # if angularcorrection:
        #     print(cltdata[1])
        #     print(cltdata[1].shape)
        #     print(cltdata[2])
        #     print(cltdata[2].shape)
    else:
        cltdata = cltdata[:,datadict["inPix1"]:datadict["dummyFlag"]]
        # print(cltdata[0])
        if bAndW:
            print("Applying b&w filter (0 or 1)")
            thresh = cltdata>0.0
            cltdata[thresh] = 1.0
            # print(cltdata[0])
        else:
            print("Normalising to 16bit maximum")
            cltdata = np.multiply(cltdata, 1.0 / 65535.0)
            # print(cltdata[0])

    # print(cltdata.shape)
    #

    infdata = np.append(alldata[:,0:datadict["inPix1"]],alldata[:,datadict["dummyFlag"]:],axis=1)

    labels = neuronLabels(infdata,2)

    if writedata:

        datOutput  = strftime("./outputs/%Y-%m-%d_%H_%M_%S_dat.txt", gmtime())
        labOutput  = strftime("./outputs/%Y-%m-%d_%H_%M_%S_lab.txt", gmtime())
        filOutput  = strftime("./outputs/%Y-%m-%d_%H_%M_%S_kind.txt", gmtime())

        np.savetxt(datOutput, cltdata)
        np.savetxt(labOutput, labels)

        u = "Filters \n"

        # with open(fileOuttxt + 'kind.txt', 'wb') as filtfile:
        #     for k, v in filt.iteritems():
        #         u += str(k) + " : " + str(v) + " - "
        #         filtfile.write(u)
        #
        # sys.exit()

    if angularcorrection:
        cltdata=cltdata.reshape(nclusts,rows,2*cols,3)#,1)
    else:
        cltdata=cltdata.reshape(nclusts,rows,2*cols,1)

    # print(cltdata.shape)
    # print(cltdata[0][0])

    #labdata = clustersLabel(infdata)
    return (cltdata,labels)


def barrelMap(alldata,cols=8,rows=8,dropEdge=False,dropBad=False,dropBig=False,dropCosmic=False,dropCharge=False):

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

    print("Only barrel doublets")
    alldata = alldata[alldata[:,datadict["isBarrelIn"]]==float(True)]
    alldata = alldata[alldata[:,datadict["isBarrelOut"]]==float(True)]
    print(" - " + str(nclusts-alldata.shape[0]) + " dropped.")

    mapTH   = TH2F("clusterAtlas","clusterAtlas",130,-65,65,50,-25,25)

    mapclusts = []
    for data in alldata:
        clustsIn = data[datadict["inPix1"]:datadict["outPix1"]]
        clustOut = data[datadict["outPix1"]:datadict["dummyFlag"]]

        yi = data[datadict["inY"]];
        zi = data[datadict["inZ"]];

        clustIn = alldata[0,datadict["inPix1"]:datadict["outPix1"]]

        yo = data[datadict["outY"]];
        zo = data[datadict["outZ"]];

        binIn = mapTH.FindBin(zi,yi)
        binOut = mapTH.FindBin(zo,yo)

        mapclust = np.zeros(50*130)

        pixcount=1
        for pix in clustOut:
            binPix = binIn + (pixcount%8) + int(pixcount/8)*100
            mapclust[binPix] = pix
            pixcount +=1

        pixcount=1
        for pix in clustOut:
            binPix = binOut + (pixcount%8) + int(pixcount/8)*100
            mapclust[binPix] = pix
            pixcount +=1

        mapclust = mapclust.reshape((130,50,1))
        # print(mapclust.shape)

        mapclusts.append(mapclust)

    cltdata = np.array(mapclusts)

    # sys.exit()

    # yi = alldata[0,datadict["inY"]];
    # zi = alldata[0,datadict["inZ"]];
    #
    # clustIn = alldata[0,datadict["inPix1"]:datadict["outPix1"]]
    #
    # yo = alldata[0,datadict["outY"]];
    # zo = alldata[0,datadict["outZ"]];
    #
    # clustOut = alldata[0,datadict["outPix1"]:datadict["dummyFlag"]]
    #
    # mapTH   = TH2F("clusterAtlas","clusterAtlas",100,-65,65,50,-25,25)
    # mapCIn  = TH2F("cIn","cOut",8,0,8,8,0,8)
    # mapCout = TH2F("cIn","cOut",8,0,8,8,0,8)
    #
    # can = TCanvas("c","c",1200,1000)
    #
    # print(zo)
    # print(yo)
    # print(zi)
    # print(yi)
    #
    # pixcount=1
    # for pix in clustIn:
    #
    #     mapCIn.SetBinContent(pixcount,pix)
    #     pixcount +=1
    #
    # pixcount=1
    # for pix in clustOut:
    #     mapCout.SetBinContent(pixcount,pix)
    #     pixcount +=1
    #
    # mapCIn.Draw()
    # can.SaveAs("in.png")
    # can.Clear()
    #
    # mapCout.Draw()
    # can.SaveAs("out.png")
    # can.Clear()
    #
    # binIn = mapTH.FindBin(zi,yi)# - 3*(8+1)
    #
    # pixcount=1
    # for pix in clustIn:
    #     binPix = binIn + (pixcount%8) + int(pixcount/8)*100
    #     # ypix = yo - 0.1*3 + (pixcount%8)*0.1
    #     # zpix = zo - 0.1*3 + int(pixcount/8)*0.1
    #     mapTH.SetBinContent(binPix,pix)
    #
    #     # ypix = yi - 0.1*3 + (pixcount%8)*0.1
    #     # zpix = zi - 0.1*3 + int(pixcount/8)*0.1
    #     # mapTH.SetBinContent(mapTH.FindBin(zi,yi),pix)
    #
    #     print(mapTH.GetBinContent(mapTH.FindBin(zi,yi)))
    #     print("======= In")
    #     pixcount += 1
    #
    #
    # pixcount=1
    #
    #
    # binOut = mapTH.FindBin(zo,yo)# - 3*(8+1)
    #
    # for pix in clustOut:
    #     binPix = binOut + (pixcount%8) + int(pixcount/8)*100
    #     # ypix = yo - 0.1*3 + (pixcount%8)*0.1
    #     # zpix = zo - 0.1*3 + int(pixcount/8)*0.1
    #     # mapTH.SetBinContent(binPix,pix)
    #
    #     print(pixcount)
    #     print(binPix)
    #     print(pix)
    #
    #     pixcount += 1
    #
    #     print(mapTH.GetBinContent(mapTH.FindBin(zi,yi)))
    #     print("======= Out")
    #
    #
    #
    # mapTH.Draw()
    # can.SaveAs("test.png")
    #
    # sys.exit()
    #

    alldata = alldata.reshape(alldata.shape[0],2*rows*cols+infos)

    # cltdata = alldata[:,datadict["inPix1"]:datadict["evtNum"]]
    print("Normalising to 16bit maximum")
    cltdata = np.multiply(cltdata, 1.0 / 65535.0)
    infdata = np.append(alldata[:,0:datadict["inPix1"]],alldata[:,datadict["dummyFlag"]:],axis=1)

    labels = neuronLabels(infdata,2)
    # cltdata=cltdata.reshape(nclusts,rows,cols,2,1)
    #labdata = clustersLabel(infdata)
    return (cltdata,labels)
