from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy

numpy.set_printoptions(threshold=numpy.nan)

import tensorflow as tf

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes

AllClasses  = ['matching', 'notMatching','oneNotSim','twoNotSim']

def dense_to_one_hot(labels_dense, num_classes):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = numpy.arange(num_labels) * num_classes
  labels_one_hot = numpy.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot

def data_clusters(f,cols,rows,stack):

      print ("Reading clusters from ",f.name)
      data = numpy.loadtxt(f,delimiter=' ',dtype = numpy.uint16)
      print(len(data), " data size ")
      num_doub_clusters = numpy.uint64(len(data)/(rows*cols*stack))
      if stack == 2:
          print(num_doub_clusters, " cluster doublets read ")
      if stack == 4:
          print(num_doub_clusters, " cluster quadruplets read ")
      data = data.reshape(num_doub_clusters,rows,cols,stack,1)
      data = numpy.multiply(data, 1.0 / 65535.0)

      return data

def data_clusters_PU(f,cols,rows,stack):

      print ("Reading clusters from ",f.name)
      data = numpy.loadtxt(f,delimiter='\t',dtype = numpy.uint16)
      print(len(data), " data size ")
      num_doub_clusters = numpy.uint64(len(data)/(rows*cols*stack))
      if stack == 2:
          print(num_doub_clusters, " cluster doublets read ")
      data = data.reshape(num_doub_clusters,rows,cols,stack,1)
      data = numpy.multiply(data, 1.0 / 65535.0)

      return data


def data_clusterslabels(f,num_classes=2,lDelimiter=False):

    #Givin back a one dim vector of labels
    print ("Reading labels from  ",f.name)

    if not lDelimiter:

      labels = numpy.loadtxt(f,delimiter=' ',dtype = numpy.uint16)

    if lDelimiter:
      l=f.read().replace('\n','')
      labels = numpy.array(list(l), dtype = numpy.uint16)

    num_labels = labels.shape[0]
    print(num_labels, " clusters labels read ")

    index_offset = numpy.arange(num_labels) * num_classes
    labels_onedim = numpy.zeros((num_labels, num_classes))
    labels_onedim.flat[index_offset + labels.ravel()] = 1

    return labels_onedim
    #return labels

def data_clusterslabels_PU(Number,Class):

    #Matching
    if Class == AllClasses[0]:
        labels = numpy.full((Number, 2), [0. , 0.])

    #Not Matching
    if Class == AllClasses[1]:
        labels = numpy.full((Number, 2), [0. , 1.])

    #One Not Sim
    if Class == AllClasses[2]:
        labels = numpy.full((Number, 2), [1. , 0.])

    #Two Not Sim
    if Class == AllClasses[3]:
        labels = numpy.full((Number, 2), [1. , 1.])

    return labels
    #return labels

def base_read_data_sets(trainsets,testsets,
                   dtype=dtypes.uint16,train_dir='./datasets/doublets/',
                   reshape=True,cols=14,rows=14,stack=2,l_delimiter=False):

  TRAIN_DATASET = trainsets[0]
  TRAIN_LABELS  = trainsets[1]
  TEST_DATASET  = testsets[0]
  TEST_LABELS   = testsets[1]


  train_file = os.path.join(train_dir, TRAIN_DATASET)
  train_labels_file = os.path.join(train_dir, TRAIN_LABELS)

  test_file = os.path.join(train_dir, TEST_DATASET)
  test_labels_file = os.path.join(train_dir, TEST_LABELS)

  with open(train_labels_file, 'rb') as f:
      train_clusters_labels = data_clusterslabels(f,lDelimiter=l_delimiter)

  with open(train_file, 'rb') as f:
      train_clusters = data_clusters(f,cols,rows,stack)

  with open(test_labels_file, 'rb') as f:
      test_clusters_labels = data_clusterslabels(f,lDelimiter=l_delimiter)

  with open(test_file, 'rb') as f:
      test_clusters = data_clusters(f,cols,rows,stack)

  return (train_clusters, train_clusters_labels), (test_clusters, test_clusters_labels)


def accuracy_measure(y_actual, y_hat):
  TruePositive = 0
  FalsePositive = 0
  TrueNegative = 0
  FalseNegative = 0

  for i in range(0, len(y_hat)):
      if y_actual[i]==y_hat[i]==1:
          TruePositive += 1
      elif y_actual[i]==0 and y_actual[i]!=y_hat[i]:
          FalsePositive += 1
      elif y_actual[i]==y_hat[i]==0:
          TrueNegative += 1
      elif y_actual[i]==1 and y_actual[i]!=y_hat[i]:
          FalseNegative += 1
  return(TruePositive, FalsePositive, TrueNegative, FalseNegative)

def doublets_read_data_sets(trainsets,testsets,
                   dtype=dtypes.uint16,train_dir='./Hits/',
                   reshape=True,cols=8,rows=8,stack=2):

  TRAIN_DATASET = trainsets[0]
  TRAIN_LABELS  = trainsets[1]
  TEST_DATASET  = testsets[0]
  TEST_LABELS   = testsets[1]


  train_file = os.path.join(train_dir, TRAIN_DATASET)
  train_labels_file = os.path.join(train_dir, TRAIN_LABELS)

  test_file = os.path.join(train_dir, TEST_DATASET)
  test_labels_file = os.path.join(train_dir, TEST_LABELS)

  with open(train_labels_file, 'rb') as f:
      train_clusters_labels = data_clusterslabels(f,lDelimiter=l_delimiter)

  with open(train_file, 'rb') as f:
      train_clusters = data_clusters(f,cols,rows,stack)

  with open(test_file, 'rb') as f:
      test_clusters = data_clusters(f,cols,rows,stack)

  with open(test_labels_file, 'rb') as f:
      test_clusters_labels = data_clusterslabels(f,lDelimiter=l_delimiter)

  return (train_clusters, train_clusters_labels), (test_clusters, test_clusters_labels)


def doublets_read_data_sets_PU(detIn,detOu,datasets,
    dtype=dtypes.uint16,filedir='./Hits/',
    reshape=True,cols=8,rows=8,stack=2):

    DATASET = 'clusters_' + str(detIn) + '_' + str(detOu) + '.txt'

    #Pick up randomly a test dataset
    #testId = 0
    testId = numpy.random.random_integers(len(datasets)-1)
    test_dir = datasets[testId]
    datasets = numpy.delete(datasets,testId,0)

    train_clusters = numpy.array([])
    train_clusters_labels = numpy.array([])
    test_clusters = numpy.array([])
    test_clusters_labels = numpy.array([])

    train_clusters_Class = {AllClasses[0]:numpy.array([]),AllClasses[1]:numpy.array([]),AllClasses[2]:numpy.array([]),AllClasses[3]:numpy.array([])}
    train_clusters_labels_Class = {AllClasses[0]:numpy.array([]),AllClasses[1]:numpy.array([]),AllClasses[2]:numpy.array([]),AllClasses[3]:numpy.array([])}

    countClass = {AllClasses[0]:0,AllClasses[1]:0,AllClasses[2]:0,AllClasses[3]:0}

    for aClass in AllClasses:
        print("============== Reading class %s ============================"%(aClass))
        for data in datasets:
            train_file = os.path.join(filedir + data + '/' + aClass, DATASET)

            with open(train_file, 'rb') as f:
                trainC = data_clusters_PU(f,cols,rows,stack)
                # trainL = data_clusterslabels_PU(len(trainC),aClass)

                countClass[aClass] = countClass[aClass] + len(trainC)

                if(train_clusters_Class[aClass].size == 0):
                    train_clusters_Class[aClass] = trainC
                else:
                    train_clusters_Class[aClass] = numpy.append(train_clusters_Class[aClass],trainC,axis=0)
                # train_clusters_labels_Class[aClass].append(trainL)

        print("For class %s collected %g doublets.\n"%(aClass,countClass[aClass]))
        print("==================================================================")
        train_clusters_labels_Class[aClass] = data_clusterslabels_PU(countClass[aClass],aClass)
        #print(train_clusters_Class[aClass].shape[0])
        #print(train_clusters_Class[aClass])
        #print(train_clusters_labels_Class[aClass].shape[0])
        #print(train_clusters_labels_Class[aClass])
        #print(len(train_clusters_Class[aClass]))
        #print(len(train_clusters_labels_Class[aClass]))


    minimum = min(countClass['matching'],countClass['notMatching'])
    print(minimum)

    for aClass in AllClasses:
        idxs = numpy.random.randint(train_clusters_Class[aClass].shape[0], size=minimum)
        train_clusters_Class[aClass] = train_clusters_Class[aClass][idxs, :]
        train_clusters_labels_Class[aClass] = train_clusters_labels_Class[aClass][idxs, :]

    for aClass in AllClasses:
        if (train_clusters.size == 0):
            train_clusters = train_clusters_Class[aClass]
            train_clusters_labels = train_clusters_labels_Class[aClass]
        else:
            train_clusters = numpy.append(train_clusters,train_clusters_Class[aClass],axis=0)
            train_clusters_labels = numpy.append(train_clusters_labels,train_clusters_labels_Class[aClass],axis=0)

    test_clusters_Class = {AllClasses[0]:numpy.array([]),AllClasses[1]:numpy.array([]),AllClasses[2]:numpy.array([]),AllClasses[3]:numpy.array([])}
    test_clusters_labels_Class = {AllClasses[0]:numpy.array([]),AllClasses[1]:numpy.array([]),AllClasses[2]:numpy.array([]),AllClasses[3]:numpy.array([])}

    countClassTest = {AllClasses[0]:0,AllClasses[1]:0,AllClasses[2]:0,AllClasses[3]:0}

    # for aClass in AllClasses:
    #     test_file = os.path.join(filedir + test_dir + '/' + aClass, DATASET)
    #
    #     with open(test_file, 'rb') as f:
    #         test_clusters.append(data_clusters_PU(f,cols,rows,stack))
    #         test_clusters_labels.append(data_clusterslabels_PU(len(test_clusters),aClass))

    for aClass in AllClasses:
        test_file = os.path.join(filedir + test_dir + '/' + aClass, DATASET)

        with open(train_file, 'rb') as f:
            testC = data_clusters_PU(f,cols,rows,stack)
            # trainL = data_clusterslabels_PU(len(trainC),aClass)                #print(trainC)
            countClassTest[aClass] = countClass[aClass] + len(testC)

            if(test_clusters_Class[aClass].size == 0):
                test_clusters_Class[aClass] = testC
            else:
                test_clusters_Class[aClass] = numpy.append(testC,axis=0)

                # train_clusters_labels_Class[aClass].append(trainL)

        print("For class %s collected %g doublets.\n"%(aClass,countClassTest[aClass]))

        test_clusters_labels_Class[aClass] = data_clusterslabels_PU(countClassTest[aClass],aClass)
        #print(train_clusters_Class[aClass].shape[0])

    minimum = min(countClassTest['matching'],countClassTest['notMatching'])

    for aClass in AllClasses:
        idxs = numpy.random.randint(test_clusters_Class[aClass].shape[0], size=minimum)
        test_clusters_Class[aClass] = test_clusters_Class[aClass][idxs, :]
        test_clusters_labels_Class[aClass] = test_clusters_labels_Class[aClass][idxs, :]

    for aClass in AllClasses:
        if (test_clusters.size == 0):
            test_clusters = test_clusters_Class[aClass]
            test_clusters_labels =  test_clusters_labels_Class[aClass]
        else:
            test_clusters = numpy.append(test_clusters,test_clusters_Class[aClass],axis=0)
            test_clusters_labels = numpy.append(test_clusters_labels,test_clusters_labels_Class[aClass],axis=0)

    return (train_clusters,train_clusters_labels), (test_clusters, test_clusters_labels)
