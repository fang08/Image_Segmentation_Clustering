import scipy.io as sio
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import skfuzzy as fuzz



mat_contents = sio.loadmat('ImsAndTruths2092.mat')
im = mat_contents['Im']
im2 = im.reshape(321*481,3)

imt= np.transpose(im2)

#print np.shape(imt)

#print np.shape(data)

ncenters = 8

cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(imt, ncenters, 2, error=0.005, maxiter=1000, init=None)

cluster_membership = np.argmax(u, axis=0)

#print np.shape(u)
#print u
#print np.shape(cluster_membership)
#print cluster_membership

opim = cluster_membership.reshape(321,481)
imgplot = plt.imshow(opim)
plt.colorbar()
plt.show()








