#! /lustre/home/adrianodif/Tools/python273/bin/bin/python2.7
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import gzip

from os import listdir
from os.path import isfile, join
from sets import Set

import numpy as np
import scipy as sp
import h5py as h
import cPickle as pkl


args = sys.argv[1:]

if len(args)!=1:
	sys.exit("Only one and only one input: RUN number")

runKey = int(args[0])

print ("Working wiht run: " + args[0])

print ("Libraries loaded")


def merge_dicts(*dict_args):
    """
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result

def unique(a):
    order = np.lexsort(a.T)
    a = a[order]
    diff = np.diff(a, axis=0)
    ui = np.ones(len(a), 'bool')
    ui[1:] = (diff != 0).any(axis=1) 
    return a[ui]

def partial_match(key, d):
    for k, v in d.iteritems():
        if all(k1 == k2 or k2 is None  for k1, k2 in zip(k, key)):
            yield v

simLabels = ["recX","recY","recZ","isMatched","isBarrel","pdgId","simX","simY","simZ","trackId","layer","ladder","module","side","disk","panel","blade"]


simDict = {}

for i in range(len(simLabels)):
        simDict[simLabels[i]] = i

parLabels = ["pCounter","px","py","pz","pt","mT","eT","mSqr","rapidity","etaTrack","phi","pdgId","charge","noTrackerHits","noTrackerLayers","simTrackId","dZ","dXY","Xvertex","Yvertex","Zvertex","bunCross","isRecoMat","isCosmic"]

parDict = {}


for i in range(len(parLabels)):
        parDict[parLabels[i]] = i

douLabels = ["detCounterIn","detCounterOut","isBarrelIn","isBarrelOut","layerIn","ladderIn","moduleIn","sideIn","diskIn","panelIn","bladeIn","layerOut","ladderOut","moduleOut","sideOut","diskOut","panelOut","bladeOut","inId","outId","inX","inY","inZ","outX","outY","outZ","isBigIn","isEdgIn","isBadIn","isBigOut","isEdgOut","isBadOut","inPix1","inPix2","inPix3","inPix4","inPix5","inPix6","inPix7","inPix8","inPix9","inPix10","inPix11","inPix12","inPix13","inPix14","inPix15","inPix16","inPix17","inPix18","inPix19","inPix20","inPix21","inPix22","inPix23","inPix24","inPix25","inPix26","inPix27","inPix28","inPix29","inPix30","inPix31","inPix32","inPix33","inPix34","inPix35","inPix36","inPix37","inPix38","inPix39","inPix40","inPix41","inPix42","inPix43","inPix44","inPix45","inPix46","inPix47","inPix48","inPix49","inPix50","inPix51","inPix52","inPix53","inPix54","inPix55","inPix56","inPix57","inPix58","inPix59","inPix60","inPix61","inPix62","inPix63","inPix64","outPix2","outPix3","outPix4","outPix5","outPix6","outPix7","outPix8","outPix9","outPix10","outPix11","outPix12","outPix13","outPix14","outPix15","outPix16","outPix17","outPix18","outPix19","outPix20","outPix21","outPix22","outPix23","outPix24","outPix25","outPix26","outPix27","outPix28","outPix29","outPix30","outPix31","outPix32","outPix33","outPix34","outPix35","outPix36","outPix37","outPix38","outPix39","outPix40","outPix41","outPix42","outPix43","outPix44","outPix45","outPix46","outPix47","outPix48","outPix49","outPix50","outPix51","outPix52","outPix53","outPix54","outPix55","outPix56","outPix57","outPix58","outPix59","outPix60","outPix61","outPix62","outPix63","outPix64","evtNum"]

douDict = {}

for i in range(len(douLabels)):
        douDict[douLabels[i]] = i

#fSimHits = h.File("simHits0.hdf5", "w")
#fDoublets = h.File("doubClusters0.hdf5", "w")
#fParticles = h.File("tParticles0.hdf5", "w")
print("Creating new pkl files")
f_pkl_Hit = gzip.open('./PKL/simHits/simHit' + str(runKey) + '_0.pkl.gz', 'w')
f_pkl_Par = gzip.open('./PKL/trkPart/trkPar' + str(runKey) + '_0.pkl.gz', 'w')
f_pkl_Rec = gzip.open('./PKL/recDoub/recDob' + str(runKey) + '_0.pkl.gz', 'w')

print ("Files opened")

pSimHits   = "./SimHits/"
pParticles = "./TParticles/"
pDoublets  = "./Doublets/"

simHitsFiles = np.array([f for f in listdir(pSimHits) if isfile(join(pSimHits, f))])
particlFiles = np.array([f for f in listdir(pParticles) if isfile(join(pParticles, f))])
doubletFiles = np.array([f for f in listdir(pDoublets) if isfile(join(pDoublets, f))])

simHitsFilesDict = {}
particlFilesDict = {}
doubletFilesDict = {}
douboneFilesDict = {}


for files in simHitsFiles:
        #print(files)
        run = int(files.split("_")[0])
        if run==runKey: 
		lum = int(files.split("_")[1])
        	evt = int(files.split("_")[2])
        	key = (run,lum,evt)
	
        	if(key in simHitsFiles):
                	print(key)
        	simHitsFilesDict[key] = files

for files in particlFiles:
        run = int(files.split("_")[0])
        if run==runKey:
		lum = int(files.split("_")[1])
        	evt = int(files.split("_")[2])
        	key = (run,lum,evt)

        	if(key in particlFilesDict):
                	print(key)
        	particlFilesDict[key] = files

for files in doubletFiles:
        run = int(files.split("_")[0])
        if run==runKey:
                lum = int(files.split("_")[1])
                evt = int(files.split("_")[2])
                key = (run,lum,evt)

		if(key in doubletFilesDict):
	                print(key)
        	doubletFilesDict[key] = files

comm1 = (Set(simHitsFilesDict.keys()) & Set(doubletFilesDict.keys()))
comm2 = (comm1 & Set(particlFilesDict.keys()))


doubletFilesDict = dict((k,doubletFilesDict[k]) for k in comm2)
particlFilesDict = dict((k,particlFilesDict[k]) for k in comm2)
simHitsFilesDict = dict((k,simHitsFilesDict[k]) for k in comm2)

print("Joining ")
print(" - " + str(len(simHitsFilesDict)) + " sim hits  files")
print(" - " + str(len(particlFilesDict)) + " particles files")
print(" - " + str(len(doubletFilesDict)) + " doublets  files")


simHitsDictionaries = [{},{}]
parHitsDictionaries = [{},{}]
dobHitsDictionaries = [{},{}]

chunkCounter = 0
filesCounter = 0

for key,filed in doubletFilesDict.iteritems():
	
	if chunkCounter>5:
		chunkCounter = 0
		del simHitsDictionaries[0]
		del simHitsDictionaries[0]

		del parHitsDictionaries[0]
		del parHitsDictionaries[0]

		del dobHitsDictionaries[0]
		del dobHitsDictionaries[0]

		simHitsDict  =  merge_dicts(*simHitsDictionaries)
		parHitsDict  =  merge_dicts(*parHitsDictionaries)
		dobHitsDict  =  merge_dicts(*dobHitsDictionaries)

		pkl.dump(dobHitsDict, f_pkl_Rec, protocol = pkl.HIGHEST_PROTOCOL)
		pkl.dump(parHitsDict, f_pkl_Par, protocol = pkl.HIGHEST_PROTOCOL)
		pkl.dump(simHitsDict, f_pkl_Hit, protocol = pkl.HIGHEST_PROTOCOL)
		
		simHitsDictionaries = [{},{}]
		parHitsDictionaries = [{},{}]
		dobHitsDictionaries = [{},{}]

		f_pkl_Hit.close()
		f_pkl_Par.close()
		f_pkl_Rec.close()
		
		filesCounter += 1
		print("Creating new pkl files")	
		f_pkl_Hit = gzip.open('./PKL/simHits/simHit_'  + str(runKey) + '_'  + str(filesCounter) + '_f.pkl.gz', 'w')
		f_pkl_Par = gzip.open('./PKL/trkPart/trkPar_'  + str(runKey) + '_'  + str(filesCounter) + '_f.pkl.gz', 'w')
		f_pkl_Rec = gzip.open('./PKL/recDoub/recDob_'  + str(runKey) + '_'  + str(filesCounter) + '_f.pkl.gz', 'w')
	else:
		chunkCounter +=1
	
	dobFileName = filed
	simFileName = simHitsFilesDict[key]
	parFileName = particlFilesDict[key]
	
	keyEvt = key[2]
	keyRun = key[0]
	keyLum = key[1]

	print("  - Files from event " + str(keyRun) + "_" + str(keyLum) + "_" + str(keyEvt))

   	with gzip.open(pSimHits   + simFileName, 'rb') as f:		
		dataSim = np.genfromtxt(f,delimiter = "\t",)#skip_footer=len(simLabels)*10000)
		print("    - sim file loaded")
	with gzip.open(pParticles + parFileName, 'rb') as f:
		dataPar = np.genfromtxt(f,delimiter = "\t",)#,skip_footer=len(parLabels)*10000)
		print("    - par file loaded")
	with gzip.open(pDoublets  + dobFileName, 'rb') as f:
		dataDob = np.genfromtxt(f,delimiter = "\t",)#,skip_footer=len(douLabels)*10000)
		print("    - dob file loaded")
     	
	run = np.full(dataSim.shape[0],keyRun)
        lum = np.full(dataSim.shape[0],keyLum)
        evt = np.full(dataSim.shape[0],keyEvt)
	
	print("  ========>TXTs loaded")

	#print(dataSim.shape[0])
        #print(dataSim.shape[1])
	
	#print(dataDob)
	dataSim = sp.delete(dataSim,-1,1)
	dataDob = sp.delete(dataDob,32,1)
	#print(dataSim[0])
	#simChunks = dataSim.shape[0] // 20
	#parChunks = dataPar.shape[0] // 20
	#dobChunks = dataDob.shape[0] // 20
	

	simKeys = dataSim[:,[simDict["trackId"],simDict["pdgId"],simDict["recX"],simDict["recY"],simDict["recZ"]]]
	simKeys = np.c_[run,lum,evt,simKeys]
	simKeys = tuple(map(tuple,simKeys))
	simHitsDict = dict(zip(simKeys,dataSim))	
	simHitsDictionaries.append(simHitsDict)
	#print(simHitsDictionaries)
	#pkl.dump(simHitsDict, f_pkl_Sim, protocol = pkl.HIGHEST_PROTOCOL)
	
	run = np.full(dataPar.shape[0],keyRun)
        lum = np.full(dataPar.shape[0],keyLum)
        evt = np.full(dataPar.shape[0],keyEvt)

	parKeys = dataPar[:,[parDict["simTrackId"],parDict["pdgId"]]]
	parKeys = np.c_[run,lum,evt,parKeys]
	parKeys = tuple(map(tuple,parKeys))
	parHitsDict = dict(zip(parKeys,dataPar))
        parHitsDictionaries.append(parHitsDict)
	#pkl.dump(parHitsDict, f_pkl_Par, protocol = pkl.HIGHEST_PROTOCOL)
	
	run = np.full(dataDob.shape[0],keyRun)
        lum = np.full(dataDob.shape[0],keyLum)
        evt = np.full(dataDob.shape[0],keyEvt)

	dobKeys = dataDob[:,[douDict["detCounterIn"],douDict["detCounterOut"],douDict["inId"],douDict["outId"],douDict["inX"],douDict["inY"],douDict["inZ"],douDict["outX"],douDict["outY"],douDict["outZ"]]]	
	dobKeys = np.c_[run,lum,evt,dobKeys]
        dobKeys = tuple(map(tuple,dobKeys))
        dobHitsDict = dict(zip(dobKeys,dataDob))
        dobHitsDictionaries.append(dobHitsDict)

