import numpy
import utilities as u
import layers as l
from io import StringIO
import tensorflow as tf
from tensorflow.python.framework import dtypes

sess = tf.InteractiveSession()

######################################################################
#Constants
size = 8
pile = 2
steps = 1000
epochs = 1000
#batchsizes = [2,5,10,15,20,25,40,50,70,100,150,200,500,1000]
#batchsizes = [20,25,40,50,70,100,150,200,500,1000]
batchsizes = [35,40,45]
######################################################################
#Results
accuracies = []
if pile == 4:
        data = u.base_read_data_sets(train_dir="./datasets/quadruplets/",cols=size,rows=size,stack=pile,dtype=dtypes.float32)
if pile == 2:
        data = u.base_read_data_sets(train_dir="./datasets/doublets/",cols=size,rows=size,stack=pile,dtype=dtypes.float32)

for batchsize in batchsizes:

    #Importing txt datasets
    #if pile == 4:
#        data = u.base_read_data_sets(train_dir="./datasets/quadruplets/",cols=size,rows=size,stack=pile,dtype=dtypes.float32)
#    if pile == 2:
#        data = u.base_read_data_sets(train_dir="./datasets/doublets/",cols=size,rows=size,stack=pile,dtype=dtypes.float32)


    #Clusters & Lables
    inputclusters = tf.placeholder(tf.float32, shape=[None,size,size,pile])
    labels = tf.placeholder(tf.float32, shape=[None, 2])

    #print(labels)

    ######################################5################################
    #First Convolutional Layer + First Pool layer

    W1 = u.weight([2, 2, pile, 32])
    b1 = u.bias([32])

    layerConvOne = tf.nn.relu(l.convolution(inputclusters, W1) + b1)
    layerPoolOne = l.pool2(layerConvOne)

    # print(layerConvOne)
    # print(layerPoolOne)

    ######################################################################
    #Second Convolutional Layer + Second Pool Layer

    W2 = u.weight([4, 4, 32, 64])
    b2 = u.bias([64])

    layerConvTwo = tf.nn.relu(l.convolution(layerPoolOne, W2) + b2)
    layerPoolTwo = l.pool2(layerConvTwo)

    # print(layerConvTwo)
    # print(layerPoolTwo)

    ######################################################################
    #First (and only) Fully Connected Layer

    WFull1 = u.weight([2 * 2 * 64, 512])
    bFull1 = u.bias([512])

    layerPoolThree = tf.reshape(layerPoolTwo, [-1, 2*2*64])
    layerFullConnOne = tf.nn.relu(tf.matmul(layerPoolThree, WFull1) + bFull1)

    ######################################################################
    #Dropout layer

    dropProbability = tf.placeholder(tf.float32)
    layerDrop = tf.nn.dropout(layerFullConnOne, dropProbability)
    # print(layerDrop)
    WFC = u.weight([512, 2])
    BFC = u.bias([2])

    ######################################################################
    #Final outcome via softmax

    finaloutput=tf.nn.softmax(tf.matmul(layerDrop, WFC) + BFC)

    # print(finaloutput)

    ######################################################################
    #Training, cross entropy, correct outcomes, accuracy
    crossH = tf.reduce_mean(-tf.reduce_sum(labels * tf.log(finaloutput), reduction_indices=[1]))

    trainStep = tf.train.AdamOptimizer(1e-4).minimize(crossH)

    correct = tf.equal(tf.argmax(labels,1),tf.argmax(finaloutput,1))

    acc = tf.reduce_mean(tf.cast(correct, tf.float32))

    ######################################################################
    #Batches

    #for batchsize in batchsizes:
        #Starting session
 
    sess.run(tf.initialize_all_variables())

    previousepoch = 0

    #while data.train.epochs_completed<epochs:
    for step in range(steps):
        batch = data.train.next_batch(batchsize)
        #if step%(steps/5) == 0:
        trainStep.run(feed_dict= {inputclusters: batch[0], labels: batch[1], dropProbability: 0.8})
        accuracy = acc.eval(feed_dict={
                 inputclusters:batch[0], labels: batch[1], dropProbability: 0.8})
        #if accuracy>=0.75 and data.train.epochs_completed%2==0 and data.train.epochs_completed!=0 and data.train.epochs_completed!=previousepoch:
        acctmp = acc.eval(feed_dict={
                inputclusters: data.test.clusters, labels: data.test.labels, dropProbability: 0.8})
        if accuracy>=.75:
 #           previousepoch = data.train.epochs_completed
            print("Step % d Epoch %d, training accuracy %g "%(step,data.train.epochs_completed,accuracy))
            print("Batchsize %d with %d epochs test accuracy %g"%(batchsize,data.train.epochs_completed,acctmp))
        

    ######################################################################
    #Test accuracy
 
    
    accuracies.append([batchsize,acctmp])

    ######################################################################
    #Accuracies results

print("#########################################################")
print("Accuracies vs batchsize : ")
for i in range(len(accuracies)):
    print("%d ----> %g "%(accuracies[i][0],accuracies[i][1]))
