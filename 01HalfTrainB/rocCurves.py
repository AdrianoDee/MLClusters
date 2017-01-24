from os import listdir
from os.path import isfile, join

import os
import StringIO
import re
import numpy as np
from sklearn import metrics
from decimal import *

import matplotlib.pyplot as plt

falsePaths = [f for f in listdir("./false/") if isfile(join("./false/", f))]
truesPaths = [f for f in listdir("./true/") if isfile(join("./true/", f))]

getcontext().prec = 4
fig, ax = plt.subplots()

collection = []

for i in range(len(falsePaths)):
    with open("./false/" + falsePaths[i], 'rb') as f:
        falses = np.loadtxt(f,delimiter='\n',dtype = np.float32)

    with open("./true/" + truesPaths[i], 'rb') as f:
        trues = np.loadtxt(f,delimiter='\n',dtype = np.float32)

    infos = re.findall(r'\d+',falsePaths[i])
    name = "Batches \b" + infos[0] + "\b Epochs \b" + infos[1] + " \b ROC \b" + str(metrics.auc(falses,trues))

    result = name,falses,trues,infos[0],infos[1],metrics.auc(falses,trues)
    collection.append(result)
    print(result[5])

collection = sorted(collection, key=lambda ROC:-ROC[5])

for i in range(len(collection)):
    if i==0:
        ax.plot(collection[i][1],collection[i][2], label=collection[i][0],linewidth=2.0)
    else:
        ax.plot(collection[i][1],collection[i][2], label=collection[i][0])

legend = ax.legend(loc='lower right', shadow=True, fontsize='x-small')

plt.show()
