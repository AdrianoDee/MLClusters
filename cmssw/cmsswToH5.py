import os
from os import listdir
from os.path import isfile, join
import sys, time
import argparse

from math import floor

headLab = ["run","evt","detSeqIn","detSeqOut","inX","inY","inZ","outX","outY","outZ",
            "inPhi","inR","outPhi","outR",
           "detCounterIn","detCounterOut","isBarrelIn","isBarrelOut",
           "layerIn","ladderIn","moduleIn","sideIn","diskIn","panelIn","bladeIn",
           "layerOut","ladderOut","moduleOut","sideOut","diskOut","panelOut","bladeOut",
           "isBigIn","isEdgIn","isBadIn","isBigOut","isEdgOut","isBadOut",
           "isFlippedIn","isFlippedOut",
           "iCSize","pixInX","pixInY","inClusterADC","iZeroADC","iCSize","iCSizeX","iCSizeY","iCSizeYy",
           "iOverFlowX","iOverFlowY",
           "oCSize","pixOutX","pixOutY","outClusterADC","oZeroADC","oCSize","oCSizeX","oCSizeY","oCSizeYy",
           "oOverFlowX","oOverFlowY",
           "diffADC"]

inPixLab = ["inPix1","inPix2","inPix3","inPix4","inPix5","inPix6","inPix7","inPix8","inPix9","inPix10",
"inPix11","inPix12","inPix13","inPix14","inPix15","inPix16","inPix17","inPix18","inPix19","inPix20",
"inPix21","inPix22","inPix23","inPix24","inPix25","inPix26","inPix27","inPix28","inPix29","inPix30",
"inPix31","inPix32","inPix33","inPix34","inPix35","inPix36","inPix37","inPix38","inPix39","inPix40",
"inPix41","inPix42","inPix43","inPix44","inPix45","inPix46","inPix47","inPix48","inPix49","inPix50",
"inPix51","inPix52","inPix53","inPix54","inPix55","inPix56","inPix57","inPix58","inPix59","inPix60",
"inPix61","inPix62","inPix63","inPix64","inPix65","inPix66","inPix67","inPix68","inPix69","inPix70",
"inPix71","inPix72","inPix73","inPix74","inPix75","inPix76","inPix77","inPix78","inPix79","inPix80",
"inPix81","inPix82","inPix83","inPix84","inPix85","inPix86","inPix87","inPix88","inPix89","inPix90",
"inPix91","inPix92","inPix93","inPix94","inPix95","inPix96","inPix97","inPix98","inPix99","inPix100",
"inPix101","inPix102","inPix103","inPix104","inPix105","inPix106","inPix107","inPix108","inPix109",
"inPix110","inPix111","inPix112","inPix113","inPix114","inPix115","inPix116","inPix117","inPix118",
"inPix119","inPix120","inPix121","inPix122","inPix123","inPix124","inPix125","inPix126","inPix127",
"inPix128","inPix129","inPix130","inPix131","inPix132","inPix133","inPix134","inPix135","inPix136",
"inPix137","inPix138","inPix139","inPix140","inPix141","inPix142","inPix143","inPix144","inPix145",
"inPix146","inPix147","inPix148","inPix149","inPix150","inPix151","inPix152","inPix153","inPix154",
"inPix155","inPix156","inPix157","inPix158","inPix159","inPix160","inPix161","inPix162","inPix163",
"inPix164","inPix165","inPix166","inPix167","inPix168","inPix169","inPix170","inPix171","inPix172",
"inPix173","inPix174","inPix175","inPix176","inPix177","inPix178","inPix179","inPix180","inPix181",
"inPix182","inPix183","inPix184","inPix185","inPix186","inPix187","inPix188","inPix189","inPix190",
"inPix191","inPix192","inPix193","inPix194","inPix195","inPix196","inPix197","inPix198","inPix199",
"inPix200","inPix201","inPix202","inPix203","inPix204","inPix205","inPix206","inPix207","inPix208",
"inPix209","inPix210","inPix211","inPix212","inPix213","inPix214","inPix215","inPix216","inPix217",
"inPix218","inPix219","inPix220","inPix221","inPix222","inPix223","inPix224","inPix225"]

outPixLab = ["outPix1","outPix2","outPix3","outPix4","outPix5","outPix6","outPix7","outPix8","outPix9",
             "outPix10","outPix11","outPix12","outPix13","outPix14","outPix15","outPix16","outPix17",
             "outPix18","outPix19","outPix20","outPix21","outPix22","outPix23","outPix24","outPix25",
             "outPix26","outPix27","outPix28","outPix29","outPix30","outPix31","outPix32","outPix33",
             "outPix34","outPix35","outPix36","outPix37","outPix38","outPix39","outPix40","outPix41",
             "outPix42","outPix43","outPix44","outPix45","outPix46","outPix47","outPix48","outPix49","outPix50","outPix51","outPix52","outPix53","outPix54","outPix55","outPix56",
"outPix57","outPix58","outPix59","outPix60","outPix61","outPix62","outPix63","outPix64","outPix65",
"outPix66","outPix67","outPix68","outPix69","outPix70","outPix71","outPix72","outPix73","outPix74",
"outPix75","outPix76","outPix77","outPix78","outPix79","outPix80","outPix81","outPix82","outPix83",
"outPix84","outPix85","outPix86","outPix87","outPix88","outPix89","outPix90","outPix91","outPix92",
"outPix93","outPix94","outPix95","outPix96","outPix97","outPix98","outPix99","outPix100","outPix101",
"outPix102","outPix103","outPix104","outPix105","outPix106","outPix107","outPix108","outPix109",
"outPix110","outPix111","outPix112","outPix113","outPix114","outPix115","outPix116","outPix117",
"outPix118","outPix119","outPix120","outPix121","outPix122","outPix123","outPix124","outPix125",
"outPix126","outPix127","outPix128","outPix129","outPix130","outPix131","outPix132","outPix133",
"outPix134","outPix135","outPix136","outPix137","outPix138","outPix139","outPix140","outPix141",
"outPix142","outPix143","outPix144","outPix145","outPix146","outPix147","outPix148","outPix149",
"outPix150","outPix151","outPix152","outPix153","outPix154","outPix155","outPix156","outPix157",
"outPix158","outPix159","outPix160","outPix161","outPix162","outPix163","outPix164","outPix165",
"outPix166","outPix167","outPix168","outPix169","outPix170","outPix171","outPix172","outPix173",
"outPix174","outPix175","outPix176","outPix177","outPix178","outPix179","outPix180","outPix181",
"outPix182","outPix183","outPix184","outPix185","outPix186","outPix187","outPix188","outPix189",
"outPix190","outPix191","outPix192","outPix193","outPix194","outPix195","outPix196","outPix197",
"outPix198","outPix199","outPix200","outPix201","outPix202","outPix203","outPix204","outPix205",
"outPix206","outPix207","outPix208","outPix209","outPix210","outPix211","outPix212","outPix213",
"outPix214","outPix215","outPix216","outPix217","outPix218","outPix219","outPix220","outPix221",
"outPix222","outPix223","outPix224","outPix225"]


tailLab = ["idTrack","px","py","pz","pt","mT","eT","mSqr","rapidity","etaTrack","phi",
"pdgId","charge","noTrackerHits","noTrackerLayers","dZ","dXY","Xvertex",
"Yvertex","Zvertex","bunCross","isCosmic","chargeMatch","sigMatch"]

XYZ = ["X","Y","Z"]
inXYZ = ["inX","inY","inZ"]
outXYZ = ["outX","outY","outZ"]

dataLab = headLab + inPixLab + outPixLab + ["dummyFlag"] + tailLab

doubLab = headLab + inPixLab + outPixLab + ["dummyFlag"]

infoLab = XYZ + tailLab

import pandas as pd
import numpy as np



def npDoubletsLoad(path,fileslimit):
    print ("======================================================================")
    
    start = time.time()
    
    datafiles = np.array([f for f in listdir(path) if (isfile(join(path, f)) and  f.lower().endswith(("txt","gz")) and "_dataset." in f)])
    
    print("Loading " + str(len(datafiles)) + " dataset file(s) . . .")
    
    idName = ""
    
    for p in path.split("/"):
        if "runs" in p:
            idName = p
    
    singlePath = path + "/singleEvts/"
    if not os.path.exists(singlePath):
        os.makedirs(singlePath)
    
    listdata = []
    for no,d in enumerate(datafiles):
        if os.stat(path + d).st_size == 0:
                print("File no." + str(no+1) + " " + d + " empty.Skipping.")
                continue
        with open(path + d, 'rb') as df:
            print("Reading file no." + str(no+1) + ": " + d)
            if d.lower().endswith(("txt")):
                dfDoublets = pd.read_table(df, sep="\t", header = None)
            if d.lower().endswith(("gz")):
                dfDoublets = pd.read_table(df, sep="\t", header = None,compression="gzip")
            dfDoublets.columns = dataLab
            #print(dfDoublets.head())
            dfDoublets.to_hdf(singlePath + idName + "_" + d.replace(".txt",".h5"),'data',append=True)
            listdata.append(dfDoublets)
            
    alldata = pd.concat(listdata)
    alldatacolumns = dataLab
    
    dfDoublets.to_hdf(path + idName + "_" + "doublets.h5",'data',append=True)
    
    end = time.time()
    print ("======================================================================")
    print ("\n - Timing : " + str(end-start))
    
    return alldata


# In[ ]:


def h5Recolumns(path,fileslimit):
    print ("======================================================================")
    
    start = time.time()
    
    print(path)
    
    datafiles = np.array([f for f in listdir(path) if (isfile(join(path, f)) and  f.lower().endswith(("h5")) and ("dataset" in f or "doublets" in f))])
    
    print("Loading " + str(len(datafiles)) + " dataset file(s) . . .")
    
    idName = ""
    
    for p in path.split("/"):
        if "runs" in p:
            idName = p.split("_")[0] + "_" + p.split("_")[1] + "_"
    
    for no,d in enumerate(datafiles):
        if os.stat(path + d).st_size == 0:
                print("File no." + str(no+1) + " " + d + " empty.Skipping.")
                continue
        with open(path + d, 'rb') as df:
            dfDoublets = pd.read_hdf(path + d)
            dfDoublets.columns = dataLab
            dfDoublets.to_hdf(path + idName + d.replace(".txt",".h5"),'data',append=True)

    alldatacolumns = dataLab
    
    dfDoublets.to_hdf(path + idName + "doublets.h5",'data',append=True)
    
    end = time.time()
    print ("======================================================================")
    print ("\n - Timing : " + str(end-start))
    
    return alldata




if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog="dataToHDF")
    parser.add_argument('--read', type=str, default="./",help='files path')
    parser.add_argument('--flimit', type=int, default=-1,help='max no. of files')
    parser.add_argument('--convert', type=bool, default=False,help='recolumn?')
    #parser.add_argument('--debug', type=bool, default=False,help='debug printouts')
    args = parser.parse_args()
    
    if args.convert:
        for path, dirs, files in os.walk(args.read):
            for f in files:
                if f.lower().endswith(("h5")) and "dataset" in f:
                    h5Recolumns(path + "/",args.flimit)
    else:
        npDoubletsLoad(args.read,args.flimit)
    




