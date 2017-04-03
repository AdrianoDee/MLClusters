import os
import numpy as np

from os import listdir
from os.path import isfile, join

from sklearn.metrics import roc_curve, auc
import pickle

import matplotlib.pyplot as plt


filesinput = []

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

figroc, axroc = plt.subplots()
figacc, axacc = plt.subplots()
figloss, axloss = plt.subplots()

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


for thisfile in allFiles:
    datas = []
    for i in range(len(thisfile)-1):
        with open(path + thisfile[i], 'rb') as f:
            data = np.genfromtxt(f,delimiter=' ',dtype = np.float16)
            datas.append(data)

    with open(path + thisfile[8], 'rb') as f:
        names=f.readlines()

    label = "cnn_"
    if "True" in names[1]:
        if "True" in names[2]:
            label += "1_bw_a"
        if "False" in names[2]:
            label += "2_bw"
    if "False" in names[1]:
        if "True" in names[2]:
            label += "3_a"
        if "False" in names[2]:
            label += "4"

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

    axroc.plot(fprtest, tprtest, label=' %s ROC (area = %0.6f)' % (label,auc(fprtest, tprtest)))

    axacc.plot(datas[0], label='%s acc ' % (label))
    axacc.plot(datas[1], label='%s val_acc ' % (label))
    axloss.plot(datas[2], label='%s acc ' % (label))
    axloss.plot(datas[3], label='%s val_acc ' % (label))

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

# allROCs = np.concatenate((fprs,tprs,tfpr,ttpr),axis=1)
axroc.legend(loc="lower right")
axloss.legend(loc="upper right")
axacc.legend(loc="lower right")

figroc.savefig("roclog.eps")
figacc.savefig("acc.eps")
figloss.savefig("loss.eps")
axroc.set_xscale('linear')
figroc.savefig("roc.eps")
