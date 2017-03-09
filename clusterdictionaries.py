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

import cPickle as pkl


datalabs = ["run","evt","detSeqIn","detSeqOut","inX","inY","inZ",
"outX","outY","outZ","detCounterIn","detCounterOut","sBarrelIn","isBarrelOut",
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
"outX","outY","outZ","detCounterIn","detCounterOut","sBarrelIn","isBarrelOut",
"layerIn","ladderIn","moduleIn","sideIn","diskIn","panelIn","bladeIn",
"layerOut","ladderOut","moduleOut","sideOut","diskOut","panelOut","bladeOut",
"nId","outId","sBigIn","isEdgIn","isBadIn","isBigOut","isEdgOut","isBadOut",
"evtNum","idTrack","px","py","pz","pt","mT","eT","mSqr","rapidity","etaTrack","phi",
"pdgId","charge","noTrackerHits","noTrackerLayers","dZ","dXY","Xvertex",
"Yvertex","Zvertex","bunCross","isCosmic","chargeMatch","sigMatch"]


def dictionames():
    try:
        f = open("labeldict.pkl","rb")
	except IOError as e:
        with open("labeldict.pkl","w") as ff:
			print("File not found, writing dicionaries . . .")
			infoinds = [n for n in range(len(infolabs))]
			datainds = [n for n in range(len(datalabs))]
			infodict = dict(zip(infolabs,infoinds))
			datadict = dict(zip(datalabs,datainds))
    else:
        print("Loading data/infos dictionaries from -> " + str(f.name) +" . . . ")
		datadict, infodict = pkl.load(f)
	finally:
		return datadict,infodict
