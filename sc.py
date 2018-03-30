import scipy.io as sio
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from sklearn.feature_extraction import image
from sklearn.cluster import spectral_clustering
import scipy as sp
from scipy.sparse import coo_matrix

mat_contents = sio.loadmat('ImsAndTruths2092.mat')
im = mat_contents['Im']
im2 = im.reshape(321*481,3)

img = sp.misc.imresize(im2, 0.40) / 255.
# Convert the image into a graph with the value of the gradient on the edges.
graph = image.img_to_graph(img)
#print graph.shape
#edgeMat = []
#for node in G:
#        tempNeighList = G.neighbors(node)
#        for neighbor in tempNeighList:
#            edgeMat[node][neighbor] = 1
#        edgeMat[node][node] = 1

#print edgeMat.shape

labels = spectral_clustering(graph, n_clusters=2, n_init=10, eigen_solver='arpack', assign_labels='discretize')
label_im = -np.ones(mask.shape)
label_im[mask] = labels

pl.matshow(label_im)

pl.show()


#adj_mat = nx.to_numpy_matrix(graph)
#spectral = cluster.SpectralClustering(n_clusters=2, affinity="precomputed", n_init=2000)
#spectral.fit(adj_mat)

#print(spectral.labels_)

#assign_labels = 'discretize', random_state=1
# print labels.shape
# # labels = sp.misc.imresize(321, 481)
# # imgplot = plt.imshow(labels)
# # plt.colorbar()
# # plt.show()

