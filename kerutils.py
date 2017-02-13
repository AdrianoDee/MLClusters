from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy

numpy.set_printoptions(threshold=numpy.nan)

import tensorflow as tf

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes
from sklearn.utils import shuffle

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

def data_clusterslabels_PU(Number,Class,num_classes=2):

    if(num_classes==2):
        #Matching
        if Class == AllClasses[0]:
            labels = numpy.full((Number, num_classes), [0. , 0.])

        #Not Matching
        if Class == AllClasses[1]:
            labels = numpy.full((Number, num_classes), [0. , 1.])

        #One Not Sim
        if Class == AllClasses[2]:
            labels = numpy.full((Number, num_classes), [1. , 0.])

        #Two Not Sim
        if Class == AllClasses[3]:
            labels = numpy.full((Number, num_classes), [1. , 1.])
    if(num_classes==4):
        #Matching
        if Class == AllClasses[0]:
            labels = numpy.full((Number, num_classes), [1. , 0. , 0. , 0.])

        #Not Matching
        if Class == AllClasses[1]:
            labels = numpy.full((Number, num_classes), [0. , 1. , 0. , 0.])

        #One Not Sim
        if Class == AllClasses[2]:
            labels = numpy.full((Number, num_classes), [0. , 0. , 1. , 0.])

        #Two Not Sim
        if Class == AllClasses[3]:
            labels = numpy.full((Number, num_classes), [0. , 0. , 0. , 1.])

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
                   dtype=dtypes.uint16,train_dir='./HitsPost/',
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


def doubletsReadPost(detIn,detOu,datasets,train=False,
                   dtype=dtypes.uint16,filedir='./HitsPost/',
                   cols=8,rows=8,stack=2,sanitize=False):

  DATASET = 'clusters'
  LABELS = 'clusterslabels'

  if detIn >= 0 and detOu >= 0:
      if train:
          DATASET += 'train' + str(detIn) + '_' + str(detOu) + '.txt'
          LABELS += 'train' + str(detIn) + '_' + str(detOu) + '.txt'
      else:
          DATASET += str(detIn) + '_' + str(detOu) + '.txt'
          LABELS += str(detIn) + '_' + str(detOu) + '.txt'
  else:
      DATASET += 'total.txt'
      LABELS += 'total.txt'

  testId = numpy.random.random_integers(len(datasets)-1)
  test_dir = datasets[testId]
  datasets = numpy.delete(datasets,testId,0)
  numpy.random.shuffle(datasets)

  print(" - Train Datasets : ")
  print(datasets)
  print(" - Test Dataset : ")
  print(test_dir)

  train_clusters = numpy.array([])
  train_clusters_labels = numpy.array([])
  test_clusters = numpy.array([])
  test_clusters_labels = numpy.array([])

  for data in datasets:
      print("============== Reading Train datasets in %s ============================"%(datasets))

      data_file = os.path.join(filedir + data, DATASET)
      labels_file = os.path.join(filedir + data, LABELS)

      with open(data_file, 'rb') as f:
          with open(labels_file, 'rb') as fl:
              trainC = data_clusters(f,cols,rows,stack)
              trainL = data_clusterslabels(fl)

              if(train_clusters.size == 0):
                  train_clusters = trainC
                  train_clusters_labels = trainL
              else:
                  train_clusters = numpy.append(train_clusters,trainC,axis=0)
                  train_clusters_labels = numpy.append(train_clusters_labels,trainL,axis=0)

  if sanitize:
      train_clusters_labels_one = train_clusters_labels

      truesIndex  =  numpy.where(train_clusters_labels_one[:,1]==1.0)[0]
      falsesIndex =  numpy.where(train_clusters_labels_one[:,1]==0.0)[0]
      #
      trueClusters = train_clusters[truesIndex]
      trueLabels   = train_clusters_labels[truesIndex]

      falseClusters = train_clusters[falsesIndex]
      falseLabels   = train_clusters_labels[falsesIndex]

      print("Read : %g true clusters and %g false clusters"%(trueLabels.shape[0],falseLabels.shape[0]))

      idxs = numpy.random.randint(trueClusters.shape[0], size=falseLabels.shape[0])

      trueClusters = trueClustersF[idxs]

      print(trueClusters.shape[0])
      trueLabels = trueLabels[idxs]

      train_clusters = numpy.append(trueClusters,falseClusters,axis=0)
      train_clusters_labels  = numpy.append(trueLabels,falseLabels,axis=0)

  print("============== Reading Test Dataset in %s ============================"%(test_dir))

  data_file_test = os.path.join(filedir + test_dir, DATASET)
  labels_file_test = os.path.join(filedir + test_dir, LABELS)

  with open(data_file_test, 'rb') as f:
      with open(labels_file_test, 'rb') as fl:
          test_clusters = data_clusters(f,cols,rows,stack)
          test_clusters_labels = data_clusterslabels(fl)

  if sanitize:
      truesIndex  =  numpy.where(test_clusters_labels[:,1]==1.0)[0]
      falsesIndex =  numpy.where(test_clusters_labels[:,1]==0.0)[0]

      trueClustersTest = test_clusters[truesIndex]
      trueLabelsTest   = test_clusters_labels[truesIndex]

      falseClustersTest = test_clusters[falsesIndex]
      falseLabelsTest   = test_clusters_labels[falsesIndex]

      idxs = numpy.random.randint(trueClustersTest.shape[0], size=falseLabelsTest.shape[0])

      trueClustersTest = trueClustersTest[idxs]
      trueLabelsTest = trueLabelsTest[idxs]

      test_clusters = numpy.append(trueClustersTest,falseClustersTest,axis=0)
      test_clusters_labels = numpy.append(trueLabelsTest,falseLabelsTest,axis=0)

  test_clusters, test_clusters_labels = shuffle(test_clusters, test_clusters_labels, random_state=0)
  train_clusters, train_clusters_labels = shuffle(train_clusters, train_clusters_labels, random_state=0)

  return (train_clusters, train_clusters_labels), (test_clusters, test_clusters_labels)

def doubletsReadPostMod(detIn,detOu,modIn,modOu,datasets,train=False,
                   dtype=dtypes.uint16,filedir='./HitsPost/',
                   cols=8,rows=8,stack=2,sanitize=False):

  DATASET = 'dets_'
  LABELS = 'dets_'

  if train:
      DATASET += 'train' + str(detIn) + '_' + str(detOu) + '_mods_' + str(modIn)  + '_' + str(modOu) + '.txt'
      LABELS += 'train' + str(detIn) + '_' + str(detOu) + '_mods_' + str(modIn)  + '_' + str(modOu) +  'labels.txt'
  else:
      DATASET += str(detIn) + '_' + str(detOu) + '_mods_' + str(modIn)  + '_' + str(modOu) + '.txt'
      LABELS += str(detIn) + '_' + str(detOu) + '_mods_' + str(modIn)  + '_' + str(modOu) +  'labels.txt'


  testId = numpy.random.random_integers(len(datasets)-1)
  test_dir = datasets[testId]
  datasets = numpy.delete(datasets,testId,0)
  numpy.random.shuffle(datasets)

  print(" - Train Datasets : ")
  print(datasets)
  print(" - Test Dataset : ")
  print(test_dir)

  train_clusters = numpy.array([])
  train_clusters_labels = numpy.array([])
  test_clusters = numpy.array([])
  test_clusters_labels = numpy.array([])

  for data in datasets:
      print("============== Reading Train datasets in %s ============================"%(datasets))

      data_file = os.path.join(filedir + data, DATASET)
      labels_file = os.path.join(filedir + data, LABELS)

      with open(data_file, 'rb') as f:
          with open(labels_file, 'rb') as fl:
              trainC = data_clusters(f,cols,rows,stack)
              trainL = data_clusterslabels(fl)

              if(train_clusters.size == 0):
                  train_clusters = trainC
                  train_clusters_labels = trainL
              else:
                  train_clusters = numpy.append(train_clusters,trainC,axis=0)
                  train_clusters_labels = numpy.append(train_clusters_labels,trainL,axis=0)

  if sanitize:
      train_clusters_labels_one = train_clusters_labels

      truesIndex  =  numpy.where(train_clusters_labels_one[:,1]==1.0)[0]
      falsesIndex =  numpy.where(train_clusters_labels_one[:,1]==0.0)[0]
      #
      trueClusters = train_clusters[truesIndex]
      trueLabels   = train_clusters_labels[truesIndex]

      falseClusters = train_clusters[falsesIndex]
      falseLabels   = train_clusters_labels[falsesIndex]

      print("Read : %g true clusters and %g false clusters"%(trueLabels.shape[0],falseLabels.shape[0]))

      idxs = numpy.random.randint(trueClusters.shape[0], size=falseLabels.size)

      trueClusters = trueClusters[idxs]
      trueLabels = trueLabels[idxs]

      train_clusters = numpy.append(trueClusters,falseClusters,axis=0)
      train_clusters_labels  = numpy.append(trueLabels,falseLabels,axis=0)

  print("============== Reading Test Dataset in %s ============================"%(test_dir))

  data_file_test = os.path.join(filedir + test_dir, DATASET)
  labels_file_test = os.path.join(filedir + test_dir, LABELS)

  with open(data_file_test, 'rb') as f:
      with open(labels_file_test, 'rb') as fl:
          test_clusters = data_clusters(f,cols,rows,stack)
          test_clusters_labels = data_clusterslabels(fl)

  if sanitize:
      truesIndex  =  numpy.where(test_clusters_labels[:,1]==1.0)[0]
      falsesIndex =  numpy.where(test_clusters_labels[:,1]==0.0)[0]

      trueClustersTest = test_clusters[truesIndex]
      trueLabelsTest   = test_clusters_labels[truesIndex]

      falseClustersTest = test_clusters[falsesIndex]
      falseLabelsTest   = test_clusters_labels[falsesIndex]

      idxs = numpy.random.randint(trueClustersTest.shape[0], size=falseLabelsTest.shape[0])

      trueClustersTest = trueClustersTest[idxs]
      trueLabelsTest = trueLabelsTest[idxs]

      test_clusters = numpy.append(trueClustersTest,falseClustersTest,axis=0)
      test_clusters_labels = numpy.append(trueLabelsTest,falseLabelsTest,axis=0)

  test_clusters, test_clusters_labels = shuffle(test_clusters, test_clusters_labels, random_state=0)
  train_clusters, train_clusters_labels = shuffle(train_clusters, train_clusters_labels, random_state=0)

  return (train_clusters, train_clusters_labels), (test_clusters, test_clusters_labels)


def doubletsReadPre(detIn,detOu,datasets,
    dtype=dtypes.uint16,filedir='./HitsPre/',
    cols=8,rows=8,stack=2,neurons=2):

    DATASET = 'clusters_'

    if detIn >= 0 and detOu >= 0:
        DATASET += str(detIn) + '_' + str(detOu) + '.txt'
    else:
        DATASET += 'Pre.txt'

    #Pick up randomly a test dataset
    #testId = 0
    testId = numpy.random.random_integers(len(datasets)-1)
    test_dir = datasets[testId]
    datasets = numpy.delete(datasets,testId,0)
    numpy.random.shuffle(datasets)

    print(" - Train Datasets : ")
    print(datasets)
    print(" - Test Dataset : ")
    print(test_dir)

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
            if(countClass[aClass]>((10*countClass['matching'])+1) and countClass[aClass]>((10*countClass['notMatching'])+1)):
                print("Enough doublets for this class.")
            else:
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
        train_clusters_labels_Class[aClass] = data_clusterslabels_PU(countClass[aClass],"matching",num_classes=neurons)
        #print(train_clusters_Class[aClass].shape[0])
        #print(train_clusters_Class[aClass])
        #print(train_clusters_labels_Class[aClass].shape[0])
        #print(train_clusters_labels_Class[aClass])
        #print(len(train_clusters_Class[aClass]))
        #print(len(train_clusters_labels_Class[aClass]))


    minimum = min(countClass['matching'],countClass['notMatching'])
    #print(minimum)

    for bClass in AllClasses:
        idxs = numpy.random.randint(train_clusters_Class[bClass].shape[0], size=minimum)
        train_clusters_Class[bClass] = train_clusters_Class[bClass][idxs]
        train_clusters_labels_Class[bClass] = train_clusters_labels_Class[bClass][idxs]

    for cClass in AllClasses:
        if (train_clusters.size == 0):
            train_clusters = train_clusters_Class[cClass]
            train_clusters_labels = train_clusters_labels_Class[cClass]
        else:
            train_clusters = numpy.append(train_clusters,train_clusters_Class[cClass],axis=0)
            train_clusters_labels = numpy.append(train_clusters_labels,train_clusters_labels_Class[cClass],axis=0)

    train_clusters, train_clusters_labels = shuffle(train_clusters, train_clusters_labels, random_state=0)

    test_clusters_Class = {AllClasses[0]:numpy.array([]),AllClasses[1]:numpy.array([]),AllClasses[2]:numpy.array([]),AllClasses[3]:numpy.array([])}
    test_clusters_labels_Class = {AllClasses[0]:numpy.array([]),AllClasses[1]:numpy.array([]),AllClasses[2]:numpy.array([]),AllClasses[3]:numpy.array([])}

    countClassTest = {AllClasses[0]:0,AllClasses[1]:0,AllClasses[2]:0,AllClasses[3]:0}

    # for aClass in AllClasses:
    #     test_file = os.path.join(filedir + test_dir + '/' + aClass, DATASET)
    #
    #     with open(test_file, 'rb') as f:
    #         test_clusters.append(data_clusters_PU(f,cols,rows,stack))
    #         test_clusters_labels.append(data_clusterslabels_PU(len(test_clusters),aClass))

    print("============================ Test ==================================")
    for dClass in AllClasses:
        test_file = os.path.join(filedir + test_dir + '/' + dClass, DATASET)

        with open(test_file, 'rb') as f:
            testC = data_clusters_PU(f,cols,rows,stack)
            # trainL = data_clusterslabels_PU(len(trainC),aClass)                #print(trainC)
            countClassTest[dClass] = countClassTest[dClass] + len(testC)

            if(test_clusters_Class[dClass].size == 0):
                test_clusters_Class[dClass] = testC
            else:
                test_clusters_Class[dClass] = numpy.append(testC,axis=0)


                # train_clusters_labels_Class[aClass].append(trainL)

        print("For class %s collected %g doublets.\n"%(aClass,countClassTest[dClass]))

        test_clusters_labels_Class[dClass] = data_clusterslabels_PU(countClassTest[dClass],dClass,num_classes=neurons)
        #print(train_clusters_Class[aClass].shape[0])

    minimum = min(countClassTest['matching'],countClassTest['notMatching'])

    for aClass in AllClasses:
        idxs = numpy.random.randint(test_clusters_Class[aClass].shape[0], size=minimum)
        test_clusters_Class[aClass] = test_clusters_Class[aClass][idxs]
        test_clusters_labels_Class[aClass] = test_clusters_labels_Class[aClass][idxs]

    for aClass in AllClasses:
        if (test_clusters.size == 0):
            test_clusters = test_clusters_Class[aClass]
            test_clusters_labels =  test_clusters_labels_Class[aClass]
        else:
            test_clusters = numpy.append(test_clusters,test_clusters_Class[aClass],axis=0)
            test_clusters_labels = numpy.append(test_clusters_labels,test_clusters_labels_Class[aClass],axis=0)

    test_clusters, test_clusters_labels = shuffle(test_clusters, test_clusters_labels, random_state=0)

    #print(test_clusters_labels)

    return (train_clusters,train_clusters_labels), (test_clusters, test_clusters_labels)
