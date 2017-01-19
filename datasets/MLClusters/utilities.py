from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy

numpy.set_printoptions(threshold=numpy.nan)

import tensorflow as tf

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes

def weight(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

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
      else:
        if stack == 4:
          print(num_doub_clusters, " cluster quadruplets read ")
        else:
            print(num_doub_clusters, " cluster blocks read ")
      data = data.reshape(num_doub_clusters,rows,cols,stack)


      return data

def data_clusterslabels(f,num_classes=2):

    #Givin back a one dim vector of labels
    print ("Reading labels from    ",f.name)
    labels = numpy.loadtxt(f,delimiter=' ',dtype = numpy.uint16)

    num_labels = labels.shape[0]

    print(num_labels, " clusters labels read ")

    #print(labels)
    print(1.0-numpy.count_nonzero(labels==0)/numpy.count_nonzero(labels==1))

    
    index_offset = numpy.arange(num_labels) * num_classes
    labels_onedim = numpy.zeros((num_labels, num_classes))
    labels_onedim.flat[index_offset + labels.ravel()] = 1

    #print(labels_onedim)
    
    return labels_onedim


def base_read_data_sets(train_dir=".",
                   dtype=dtypes.uint16,
                   reshape=True,
                   validation_fraction=0.3,cols=14,rows=14,stack=2):

  TRAIN_DATASET = 'clusterstrain.txt'
  TRAIN_LABELS  = 'clusterstrainlabels.txt'
  TEST_DATASET  = 'clusterstest.txt'
  TEST_LABELS   = 'clusterstestlabels.txt'

  test_file = os.path.join(train_dir, TEST_DATASET)
  test_labels_file = os.path.join(train_dir, TEST_LABELS)

  train_file = os.path.join(train_dir, TRAIN_DATASET)
  train_labels_file = os.path.join(train_dir, TRAIN_LABELS)

  with open(test_labels_file, 'rb') as f:
      test_clusters_labels = data_clusterslabels(f)
      
  with open(test_file, 'rb') as f:
      test_clusters = data_clusters(f,cols,rows,stack)

  with open(train_labels_file, 'rb') as f:
      train_clusters_labels = data_clusterslabels(f)

  with open(train_file, 'rb') as f:
      train_clusters = data_clusters(f,cols,rows,stack)

  validation_size = numpy.uint16(size = len(train_clusters)*0.3)

  validation_clusters = train_clusters[:validation_size]
  validation_clusters_labels = train_clusters_labels[:validation_size]
  train_clusters = train_clusters[validation_size:]
  train_clusters_labels = train_clusters_labels[validation_size:]

  train = Clusters(train_clusters,stack,train_clusters_labels,dtype=dtype)
  validation = Clusters(validation_clusters,stack, validation_clusters_labels,dtype=dtype)

  test = Clusters(test_clusters, stack, test_clusters_labels, dtype=dtype)

  return base.Datasets(train=train, validation=validation, test=test)


class Clusters(object):

  def __init__(self,
               clusters,
               stack,
               labels,
               dtype=dtypes.uint16):

    dtype = dtypes.as_dtype(dtype).base_dtype
    if dtype not in (dtypes.float32, dtypes.uint16):
      raise TypeError('dtype %r not allowed, choose uint16 or float32' %
                      dtype)

    assert clusters.shape[0] == labels.shape[0], (
    'clusters.shape: %s labels.shape: %s' % (clusters.shape, labels.shape))
    self._num_examples = clusters.shape[0]

    #assert clusters.shape[3] == stack , ('You want to stack %s but the clusters are stacked in %s !' % (stack, clusters.shape[3]))
    #clusters = clusters.reshape(clusters.shape[0],clusters.shape[3],
    #                            clusters.shape[1] * clusters.shape[2])

    if dtype == dtypes.float32:
        clusters = clusters.astype(numpy.float32)
        clusters = numpy.multiply(clusters, 1.0 / 65536)
    self._clusters = clusters
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def clusters(self):
    return self._clusters

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size):
    #"""Return the next `batch_size` examples from this data set."""

    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      if self._epochs_completed%(20) == 0:
          print("Epoch completed %d"%(self._epochs_completed))
      # Shuffle the data
      perm = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm)
      self._clusters = self._clusters[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    #print(self._labels[start:end].shape)
    #print(self._clusters[start:end].shape)
    return self._clusters[start:end], self._labels[start:end]
