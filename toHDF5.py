from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import gzip

from os import listdir
from os.path import isfile, join

import numpy as np
import scipy as sp
import h5py as h
import cPickle as pkl

simLabels = ["recX","recY","recZ","isMatched","isBarrel","pdgId","simX","simY","simZ","trackId","layer","ladder","module","side","disk","panel","blade"]

parLabels = ["pCounter","px","py","pz","pt","mT","eT","mSqr","rapidity","etaTrack","phi","pdgId","charge","noTrackerHits","noTrackerLayers","simTrackId","dZ","dXY","Xvertex","Yvertex","Zvertex","bunCross","isRecoMat","isCosmic"]

simLabels = simLabels + parLabels

simDict = {}

for i in range(len(simLabels)):
        simDict[simLabels[i]] = i


douLabels = ["detCounterIn","detCounterOut","isBarrelIn","isBarrelOut","layerIn","ladderIn","moduleIn","sideIn","diskIn","panelIn","bladeIn","layerOut","ladderOut","moduleOut","sideOut","diskOut","panelOut","bladeOut","inId","outId","inX","inY","inZ","outX","outY","outZ","isBigIn","isEdgIn","isBadIn","isBigOut","isEdgOut","isBadOut","inPix1","inPix2","inPix3","inPix4","inPix5","inPix6","inPix7","inPix8","inPix9","inPix10","inPix11","inPix12","inPix13","inPix14","inPix15","inPix16","inPix17","inPix18","inPix19","inPix20","inPix21","inPix22","inPix23","inPix24","inPix25","inPix26","inPix27","inPix28","inPix29","inPix30","inPix31","inPix32","inPix33","inPix34","inPix35","inPix36","inPix37","inPix38","inPix39","inPix40","inPix41","inPix42","inPix43","inPix44","inPix45","inPix46","inPix47","inPix48","inPix49","inPix50","inPix51","inPix52","inPix53","inPix54","inPix55","inPix56","inPix57","inPix58","inPix59","inPix60","inPix61","inPix62","inPix63","inPix64","outPix1","outPix2","outPix3","outPix4","outPix5","outPix6","outPix7","outPix8","outPix9","outPix10","outPix11","outPix12","outPix13","outPix14","outPix15","outPix16","outPix17","outPix18","outPix19","outPix20","outPix21","outPix22","outPix23","outPix24","outPix25","outPix26","outPix27","outPix28","outPix29","outPix30","outPix31","outPix32","outPix33","outPix34","outPix35","outPix36","outPix37","outPix38","outPix39","outPix40","outPix41","outPix42","outPix43","outPix44","outPix45","outPix46","outPix47","outPix48","outPix49","outPix50","outPix51","outPix52","outPix53","outPix54","outPix55","outPix56","outPix57","outPix58","outPix59","outPix60","outPix61","outPix62","outPix63","outPix64","evtNum"]

douDict = {}

for i in range(len(douLabels)):
        douDict[douLabels[i]] = i

pSim = "./PKL/simData/"
pDob = "./PKL/recDoub/"

simFiles = np.array([f for f in listdir(pSim) if isfile(join(pSim, f))])
dobFiles = np.array([f for f in listdir(pDob) if isfile(join(pDob, f))])

simFiles = simFiles.reshape(simFiles.shape[0],1)
dobFiles = dobFiles.reshape(dobFiles.shape[0],1)

allFiles = np.concatenate((simFiles,dobFiles),axis=1)

pHDF = "./HDF5"

fLabel  = h.File(pHDF + "clust_0_f.hdf5", "w")
fClust  = h.File(pHDF + "label_0_f.hdf5", "w")
fInfos  = h.File(pHDF + "infos_0_.hdf5", "w")

chunks = 0
filesc = 0

for files in allFiles:
	if chunks > 10:
		fLabel.close()
		fClust.close()
		fInfos.close()

		fLabel = h.File(pHDF + "clust_"+ str(filesc) + "_f.hdf5", "w")
		fClust = h.File(pHDF + "label_"+ str(filesc) + "_f.hdf5", "w")
		fInfos = h.File(pHDF + "infos_"+ str(filesc) + "_f.hdf5", "w")

		chunks = 0
		filesc +=1
	else:
		chunks +=1

	f_pkl_Sim = gzip.open(pSim + files[0], 'rb')
	f_pkl_Rec = gzip.open(pDob + files[1], 'rb')



	#print(douDict["outPix64"]-douDict["inPix1"])

	clustShape = ((1,2,8,8))

	clustList = np.zeros(clustShape)
	labelList = np.zeros((1,10))
	infosList = np.zeros((1,douDict["isBadOut"]-douDict["detCounterIn"]+1))

	dictSim = pkl.load(f_pkl_Sim)
	dictRec = pkl.load(f_pkl_Rec)

	matchedLabel = (1.0,0.0)

	for k,hit in dictRec.iteritems():
	        cluster = np.array(hit[douDict["inPix1"]:douDict["outPix64"]+1])
        	#print(np.zeros(128))
        	#print(cluster)
        	#print((np.zeros(128).shape))
        	#print(cluster.shape)
       		cluster = cluster.reshape(clustShape)
        	info    = hit[douDict["detCounterIn"]:douDict["isBadOut"]+1]
      	  	#print(cluster)
        	#print(info)
        	#print(info.shape)
        	label   = np.zeros((1,10))
        	kIn     = (k[0],k[1],k[2],k[5],k[6],k[7])
        	kOu     = (k[0],k[1],k[2],k[8],k[9],k[10])
        	Matched = float(False)
        	notMatched = float(True)
        	#Checks if recHits are simHits
        	if ((kIn in dictSim) and (kOu in dictSim)):
                	#Checks if simHits comes from same track/trkparticle
                	pdgMatch = (dictSim.get(kIn)[simDict["pdgId"]]==dictSim.get(kOu)[simDict["pdgId"]])
                	trkMatch = (dictSim.get(kIn)[simDict["trackId"]]==dictSim.get(kOu)[simDict["trackId"]])
                	Matched = float(pdgMatch and trkMatch)
                	notMatched = float(not (pdgMatch and trkMatch))
                	if Matched == 1.0:
                        	etL = etalabel((dictSim.get(kIn))[simDict["etaTrack"]])
                        	zeL = zetalabel((dictSim.get(kIn)[simDict["Zvertex"]]))
                        	ptL = ptlabel((dictSim.get(kIn)[simDict["pt"]]))
                        	label = np.append(matchedLabel,zetalabel,ptlabel,etalabel,axis=0)
                        	#print(label)
        	infosList = np.vstack([infosList,info])
        	labelList = np.vstack([labelList,label])
        	clustList = np.vstack([clustList,cluster])

	labChunks = labelList.shape[0] // 20
	clsChunks = clustList.shape[0] // 20
	infChunks = infosList.shape[0] // 20

	labelDataset = fLabel.create_dataset("label_dataset",dtype='f',data=labelList,compression_opts = 9, chunks = (labChunks,labelList.shape[1]))
	clustDataset = fClust.create_dataset("clust_dataset",dtype='f',data=clustList,compression_opts = 9, chunks = (clsChunks,clustList.shape[1]))
	infosDataset = fInfos.create_dataset("infos_dataset",dtype='f',data=infosList,compression_opts = 9, chunks = (infChunks,infosList.shape[1]))
