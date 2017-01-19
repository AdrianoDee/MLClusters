import numpy as np
x = y = z = np.arange(0.0,5.0,1.0)

size = 8
pile = 2
epochs = 2
batchsize = 30

np.savetxt("batch_%g_epoch_%g.out"%(batchsize,epochs), (x,y), delimiter=',')
