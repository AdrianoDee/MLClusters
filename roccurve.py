import os
import numpy as np

from os import listdir
from os.path import isfile, join

from sklearn.metrics import roc_curve, auc
import pickle

import matplotlib.pyplot as plt


filesinput = []

colors = ["b","g","r","c","m","k","#7FFFD4","#FFA500","#800080","#A52A2A","#ADD8E6","#808000"]


path = './outputs/'

filesinput.append(np.array([f for f in listdir('./outputs/') if (isfile(join(path, f)) and  f.lower().endswith(('_acc.txt')))]))
filesinput.append(np.array([f for f in listdir('./outputs/') if (isfile(join(path, f)) and  f.lower().endswith(('_valacc.txt')))]))

filesinput.append(np.array([f for f in listdir('./outputs/') if (isfile(join(path, f)) and  f.lower().endswith(('_tralos.txt')))]))
filesinput.append(np.array([f for f in listdir('./outputs/') if (isfile(join(path, f)) and  f.lower().endswith(('_vallos.txt')))]))

filesinput.append(np.array([f for f in listdir('./outputs/') if (isfile(join(path, f)) and  f.lower().endswith(('_predtrain.txt')))]))
filesinput.append(np.array([f for f in listdir('./outputs/') if (isfile(join(path, f)) and  f.lower().endswith(('_predtest.txt')))]))
filesinput.append(np.array([f for f in listdir('./outputs/') if (isfile(join(path, f)) and  f.lower().endswith(('_labtrain.txt')))]))
filesinput.append(np.array([f for f in listdir('./outputs/') if (isfile(join(path, f)) and  f.lower().endswith(('_labtest.txt')))]))

filesinput.append(np.array([f for f in listdir('./outputs/') if (isfile(join(path, f)) and  f.lower().endswith(('_filters.txt')))]))

filesinputL = []

for fileinput in filesinput:
    fileinput = fileinput.reshape(fileinput.shape[0],1)
    filesinputL.append(fileinput)

filesinput = tuple(filesinputL)

allFiles = np.concatenate(filesinput,axis=1)

fprs = []
tprs = []
tfpr = []
ttpr = []
accs = []
vacs = []
loss = []
vlos = []
area = []
tare = []

figroc, axroc = plt.subplots(figsize=(8, 4.5))
figacc, axacc = plt.subplots(figsize=(8, 4.5))
figloss, axloss = plt.subplots(figsize=(8, 4.5))

axroc.set_xscale('log')
axroc.plot([0, 1], [0, 1], color='navy', linestyle='--')
# axroc.set_xlim([0.0, 1.0])
# axroc.set_ylim([0.0, 1.05])
axroc.set_xlabel('False Positive Rate')
axroc.set_ylabel('True Positive Rate')
axroc.set_title('ROC curve')

axacc.set_xlabel('Epochs')
axacc.set_ylabel('Accuracy')
axacc.set_title('CNNs accuracy')

axloss.set_xlabel('Epochs')
axloss.set_ylabel('Loss')
axloss.set_title('CNNs loss function')

cnnCounter = 1

filts = []
timings = []

for thisfile in allFiles:

    datas = []
    for i in range(len(thisfile)-1):
        with open(path + thisfile[i], 'rb') as f:
            data = np.genfromtxt(f,delimiter=' ',dtype = np.float16)
            datas.append(data)

    with open(path + thisfile[8], 'rb') as f:
        names=f.readlines()
        print("File : " + path + thisfile[8])
        print(names[5])

    filts.append(names[5])

    label = "cnn_"

    if "True" in names[1]:
        if "True" in names[2]:
            label += str(cnnCounter) + "_bw_a"
        if "False" in names[2]:
            label += str(cnnCounter) + "_bw"
    if "False" in names[1]:
        if "True" in names[2]:
            label += str(cnnCounter) + "_a\t"
        if "False" in names[2]:
            label += str(cnnCounter) + "\t"


    times = names[4].split()

    Label = "cnn_"

    if "True" in names[1]:
        if "True" in names[2]:
            Label += str(cnnCounter) + "_bw_a"
        if "False" in names[2]:
            Label += str(cnnCounter) + "_bw"
    if "False" in names[1]:
        if "True" in names[2]:
            Label += str(cnnCounter) + "_a"
        if "False" in names[2]:
            Label += str(cnnCounter)

    accs.append(datas[0])
    vacs.append(datas[1])


    loss.append(datas[2])
    vlos.append(datas[3])

    predtrain = datas[4]
    predtest = datas[5]
    labtrain = datas[6]
    labtest = datas[7]

    predtrain_score = np.array(predtrain[:,0])
    predtest_score  = np.array(predtest[:,0])
    labtrain_score  = np.array(labtrain[:,0])
    labtest_score  = np.array(labtest[:,0])

    #
    # print(predtrain)
    # print(predtest)
    # print(labtrain)
    # print(labtest)
    # print(predtrain.shape)
    # print(labtrain.shape)
    #
    # predtrain.astype(float)
    # labtrain.astype(float)
    # predtest.astype(float)
    # predtrain.astype(float)


    fprtrain, tprtrain, _ = roc_curve(labtrain_score,predtrain_score)
    fprtest, tprtest, _ = roc_curve(labtest_score,predtest_score)

    rejhalf = tprtest[np.where(fprtest>0.5)[0][0]]
    rejthre = tprtest[np.where(fprtest>0.25)[0][0]]
    rejnine = tprtest[np.where(fprtest>0.1)[0][0]]
    rejninn = tprtest[np.where(fprtest>0.01)[0][0]]

    labtime = [times,label,datas[0].shape[0],rejhalf,rejthre,rejnine,rejninn]
    timings.append(labtime)

    axroc.plot(fprtest, tprtest, label=' %s ROC (area = %0.6f)' % (Label,auc(fprtest, tprtest)),color=colors[cnnCounter-1])

    axacc.plot(datas[0], label='%s loss ' % (Label),color=colors[cnnCounter-1],lw=2)
    axacc.plot(datas[1], label='%s val_loss ' % (Label),ls='-.',color=colors[cnnCounter-1],lw=1)
    axloss.plot(datas[2], label='%s acc ' % (Label),color=colors[cnnCounter-1],lw=2)
    axloss.plot(datas[3], label='%s val_acc ' % (Label),ls='-.',color=colors[cnnCounter-1],lw=1)

    cnnCounter += 1

#     fprs.append(fprtrain)
#     tprs.append(tprtrain)
#     tfpr.append(fprtest)
#     ttpr.append(tprtest)
#
# fprs = np.array(fprs)
# tprs = np.array(tprs)
# tfpr = np.array(tfpr)
# ttpr = np.array(ttpr)
#
#
#
# fprs = fprs.reshape(fprs.shape[0],1)
# tprs = fprs.reshape(fprs.shape[0],1)
# tfpr = fprs.reshape(fprs.shape[0],1)
# ttpr = fprs.reshape(fprs.shape[0],1)
print ("CNN \t\t\ttraining(s) \tinf[" + str(labtrain_score.shape[0]) + "](s) \tinf[" + str(labtest_score.shape[0]) + "](s) \tepochs")
for t in timings:
    times = t[0]
    label = t[1]
    print (label + "\t\t " + times[0] + "\t" + times[1] + "\t" + times[2] + "\t" + str(t[2]))

print ("CNN eff. @ rej\t\t0.5 \t0.75 \t0.9 \t0.99 ")
for t in timings:
    label = t[1]
    rejhalf = t[3]
    rejthre = t[4]
    rejnine = t[5]
    rejninn = t[6]

    print (label + "\t\t" + str(rejhalf)[:4] + "\t" + str(rejthre)[:4] + "\t" + str(rejnine)[:4] + "\t" + str(rejninn)[:4])

cnnCounter = 1
for f in filts:
    print("================ CNN " + str(cnnCounter))
    cnnCounter += 1
    print(f)
# allROCs = np.concatenate((fprs,tprs,tfpr,ttpr),axis=1)
axroc.legend(loc="lower right")
axloss.legend(loc="upper right")
axacc.legend(loc="lower right")

plt.show()

figroc.savefig("roclog.eps")
figacc.savefig("acc.eps")
figloss.savefig("loss.eps")
axroc.set_xscale('linear')
figroc.savefig("roc.eps")
