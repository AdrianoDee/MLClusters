import tensorflow as tf

def convolution(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def convolution3D(x, W):
  return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')

def pool2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='VALID')
def pool3(x):
  return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                        strides=[1, 3, 3, 1], padding='VALID')
def pool7(x):
  return tf.nn.max_pool(x, ksize=[1, 7, 7, 1],
                        strides=[1, 7, 7, 1], padding='VALID')

def pool23D(x):
  return tf.nn.max_pool(x, ksize=[1,1, 2, 2, 2],
                        strides=[1,1, 2, 2, 2], padding='VALID')
