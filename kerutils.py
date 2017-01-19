from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy

numpy.set_printoptions(threshold=numpy.nan)

import tensorflow as tf

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes

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

  with open(test_file, 'rb') as f:
      test_clusters = data_clusters(f,cols,rows,stack)

  with open(test_labels_file, 'rb') as f:
      test_clusters_labels = data_clusterslabels(f,lDelimiter=l_delimiter)

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
