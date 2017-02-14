from os import listdir
from os.path import isfile, join

import os
import StringIO
import re
import numpy as np
from sklearn import metrics
from decimal import *

import pickle


import matplotlib.pyplot as plt

falsePaths = [f for f in listdir("./false/") if isfile(join("./false/", f))]
truesPaths = [f for f in listdir("./true/") if isfile(join("./true/", f))]

getcontext().prec = 4
fig, ax = plt.subplots()
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()
fig4, ax4 = plt.subplots()
fig5, ax5 = plt.subplots()
fig5, ax6 = plt.subplots()

collection = []

for i in range(len(falsePaths)):
    with open("./false/" + falsePaths[i], 'rb') as f:
        print(f.name)
        falses = np.loadtxt(f,delimiter='\n',dtype = np.float32)
    with open("./true/" + truesPaths[i], 'rb') as f:
        print(f.name)
        trues = np.loadtxt(f,delimiter='\n',dtype = np.float32)

    infos = re.findall(r'\d+',falsePaths[i])
    #name = "Batches \b" + infos[2] + "\b Epochs \b" + infos[3] + " \b ROC \b" + str(metrics.auc(falses,trues))
    name = f.name
    # result = name,falses,trues,infos[2],infos[3],metrics.auc(falses,trues)
    result = name,falses,trues,metrics.auc(falses,trues)
    collection.append(result)
    print(result[3])

collection = sorted(collection, key=lambda ROC:-ROC[3])
#print(len(collection))

for i in range(1,7):
    if i==1:
        ax.plot(collection[i][1],collection[i][2], label=collection[i][3],linewidth=2.0)
        ax5.plot(collection[i][1],collection[i][2], label=collection[i][3],linewidth=2.0)
    else:
        ax.plot(collection[i][1],collection[i][2], label=collection[i][3])
        ax5.plot(collection[i][1],collection[i][2], label=collection[i][3])

for i in range(18,23):
    ax1.plot(collection[i][1],collection[i][2], label=collection[i][3])
    ax.plot(collection[i][1],collection[i][2], label=collection[i][3])

for i in range(40,45):
    ax2.plot(collection[i][1],collection[i][2], label=collection[i][3])
    ax.plot(collection[i][1],collection[i][2], label=collection[i][3])

for i in range(52,64):
    ax3.plot(collection[i][1],collection[i][2], label=collection[i][3])
    ax.plot(collection[i][1],collection[i][2], label=collection[i][3])

for i in range(60,64):
    ax4.plot(collection[i][1],collection[i][2], label=collection[i][3])
    ax.plot(collection[i][1],collection[i][2], label=collection[i][3])

for i in range(len(collection)):
    if i==1:
        ax6.plot(collection[i][1],collection[i][2], label=collection[i][3],linewidth=2.0)
    else:
        ax6.plot(collection[i][1],collection[i][2], label=collection[i][3])
    print(collection[i][0])
    print(collection[i][3])

legend = ax.legend(loc='lower right', shadow=True, fontsize='x-small')
legend = ax1.legend(loc='lower right', shadow=True, fontsize='medium')
legend = ax2.legend(loc='lower right', shadow=True, fontsize='medium')
legend = ax3.legend(loc='lower right', shadow=True, fontsize='medium')
legend = ax4.legend(loc='lower right', shadow=True, fontsize='medium')
legend = ax5.legend(loc='lower right', shadow=True, fontsize='medium')
legend = ax6.legend(loc='lower right', shadow=True, fontsize='medium')

plt.show()
#fig.savefig("rocCurves.eps")

# with open('clusterstrain0_1.pik', 'rb') as f:
#     results = pickle.load(f)
#
# for j in range(len(results)):
#     cross.plot(results[j][4])
#     cross.plot(results[j][5])
#     plotname = "loss_" + str(results[j][1]) + "_" + str(results[j][0][j]) +"_plot.eps"
#     fig2.savefig(plotname)
#     cross.clear()


#plt.show()
#fig2.savefig("stuff.png")
