#! /lustre/home/adrianodif/Tools/python273/bin/bin/python2.7
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

args = sys.argv[1:]

if len(args)!=1:
        sys.exit("Only one and only one input: RUN number")

runKey = int(args[0])

print ("Working wiht run: " + args[0])

print ("Libraries loaded")

pSimHits = "./PKL/simHits/"
pParticl = "./PKL/trkPart/"
pSimData = "./PKL/simData/" 
#pDoublet  = "./PKL/recDoub/"

simHitsFiles = np.array([f for f in listdir(pSimHits) if isfile(join(pSimHits, f))])
particlFiles = np.array([f for f in listdir(pParticl) if isfile(join(pParticl, f))])
#doubletFiles = np.array([f for f in listdir(pDoublet) if isfile(join(pDoublet, f))])

simHitsFiles = simHitsFiles.reshape(simHitsFiles.shape[0],1)
particlFiles = particlFiles.reshape(particlFiles.shape[0],1)
#doubletsFiles  = doubletsFiles.reshape(doubletsFiles.shape[0],1)

allFiles = np.concatenate((simHitsFiles,particlFiles),axis=1)
print(allFiles)
#allFile  = ()
#allFile += (allFiles[0])
#allFile += (allFiles[1])
#print(allFile)


for files in allFiles:
	#print(files)
	filesplit = files[0].split("_")
	simFileName = files[0]
	#print(simFileName)
	runNum = int(filesplit[0][6:])
	chkNum = int(filesplit[1][:-7])
	#print(chkNum)	
	#print(runNum)
        #if int(files[0].split("_")[1])!=runKey:
	#print(runNum)
	#print(runKey)
	if runNum==runKey:	
		#print("IN")
		#print(runNum)
		#print(simFileName)  
        	parFileName = files[1]
        	#dobFileName = files[2]
		#print(parFileName)
		
		datFileName = "simDat_" + str(runNum) + "_" + str(chkNum) + "_f.pkl.gz"
	
		f_pkl_Sim = gzip.open(pSimHits+simFileName, 'rb')
		f_pkl_Par = gzip.open(pParticl+parFileName, 'rb')

		f_pkl_Dat = gzip.open(pSimData+datFileName, 'w')
		#sys.exit()

		dictSim = pkl.load(f_pkl_Sim)
		dictPar = pkl.load(f_pkl_Par)
		
		#print(dictSim)
		#print(dictPar)
		#sys.exit()
		dictDat = {}
	
		for k,v in dictSim.iteritems():
       			kp = (k[0],k[1],k[2],k[3],k[4])
       			ks = (k[0],k[1],k[2],k[5],k[6],k[7])
       			print(kp)
			if kp in dictPar:
               			data = np.append(v,dictPar.get(kp))
               			dictDat[k] = data
				print(data)

		pkl.dump(dictDat,f_pkl_Dat, protocol = pkl.HIGHEST_PROTOCOL)



