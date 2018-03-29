
# import os;
# print os.getcwd(); # Prints the working directory
# os.chdir(/Users/YaoFang/Documents/pyCharm/test.py);

# sudo apt-get install python-scipy;
# pip install scipy;



### K means #####
import cv2

import numpy as np
# print np.mgrid[0:5,0:5]



import scipy.io as sio
# mat_contents = sio.loadmat('ImsAndTruths2092.mat')
# im = mat_contents['Im']  # numpy.ndarray
# seg1 = mat_contents['Seg1']  # 15
# seg2 = mat_contents['Seg2']  # 10
# seg3 = mat_contents['Seg3']  # 17
# print im.shape    # (321, 481, 3)
# print seg1.shape  # (321, 481)
# print seg2.shape  # (321, 481)
# print seg3.shape  # (321, 481)

# print im
# print seg1
# print seg2
# print seg3


# a = np.arange(18).reshape(2, 3, 3)
# b = a.reshape(6,3)
# print a
# print b


# im2 = im.reshape(321*481,3)
# print "im2:\n"
# print im2

# convert to np.float32
# im3 = np.float32(im2)

# # define criteria, number of clusters(K) and apply kmeans()
# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 1.0)
# K = 8
# compactness, label, center = cv2.kmeans(im3, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
# label1 = label.reshape(321, 481)
# # print label1.shape
#
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# imgplot = plt.imshow(label1)
# plt.colorbar()
# plt.show()


###########   SOM  #######################

import matplotlib.pyplot as plt
# from minisom import MiniSom
# som = MiniSom(4, 4, 3, sigma=1.0, learning_rate=0.5)
# som.random_weights_init(im3)
# som.train_random(im3, 100)
#
# # Plotting the response for each pattern
# plt.bone()
# plt.pcolor(som.distance_map().T)  # plotting the distance map as background
# plt.colorbar()
# plt.show()

# from pyclustering.cluster.somsc import somsc
# somsc_instance = somsc(im2, 10)
# # run cluster analysis and obtain results
# somsc_instance.process()
# somsc_instance.get_clusters()

############# FCM  ######################
#
# import matplotlib.pyplot as plt
# import skfuzzy as fuzz
#
# imt = np.transpose(im2)
# ncenters = 8
# cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(imt, ncenters, 2, error=0.005, maxiter=1000, init=None)
# # def cmeans(data, c, m, error, maxiter, metric='euclidean', init=None, seed=None)
#
# # Plot assigned clusters, for each data point in training set
# cluster_membership = np.argmax(u, axis=0)


##########  Spectral   ######
# from sklearn.feature_extraction import image
# from sklearn.cluster import spectral_clustering
# import scipy as sp
#
#
# # Resize it to 10% of the original size to speed up the processing
# img = sp.misc.imresize(im, 0.10) / 255.
# # Convert the image into a graph with the value of the gradient on the edges.
# graph = image.img_to_graph(img)
# print graph
#
# labels = spectral_clustering(graph, n_clusters = 10)
# # assign_labels = 'discretize', random_state=1
# # print labels.shape
# labels = labels.reshape(img.shape)
# # labels = sp.misc.imresize(321, 481)
# imgplot = plt.imshow(labels)
# plt.colorbar()
# plt.show()


# # load the raccoon face as a numpy array
# try:  # SciPy >= 0.16 have face in misc
#     from scipy.misc import face
#     face = face(gray=True)
# except ImportError:
#     face = sp.face(gray=True)
#
# # Resize it to 10% of the original size to speed up the processing
# face = sp.misc.imresize(face, 0.10) / 255.
#
# # Convert the image into a graph with the value of the gradient on the
# # edges.
# graph = image.img_to_graph(face)
#
# # Apply spectral clustering (this step goes much faster if you have pyamg
# # installed)
# N_REGIONS = 25
#
# labels = spectral_clustering(graph, n_clusters=N_REGIONS, assign_labels='discretize', random_state=1)
# labels = labels.reshape(face.shape)
# imgplot = plt.imshow(labels)
# plt.colorbar()
# plt.show()


####### GMM ######
from sklearn.mixture import GaussianMixture
# gmm = GaussianMixture(n_components=8, covariance_type="tied")
# gmm = gmm.fit(im2)
#
# cluster = gmm.predict(im2)
# cluster = cluster.reshape(321, 481)
# plt.imshow(cluster)
# plt.colorbar()
# plt.show()


mat_contents2 = sio.loadmat('PaviaHyperIm.mat')
imh = mat_contents2['PaviaHyperIm']
imh2 = imh.reshape(610*340, 103)
gmm2 = GaussianMixture(n_components=8, covariance_type="tied")
gmm2 = gmm2.fit(imh2)

cluster = gmm2.predict(imh2)
cluster = cluster.reshape(610, 340)
plt.imshow(cluster)
plt.colorbar()
plt.show()
# can work, but slow


###### k means hyperspectral #######
# import time
# start_time = time.time()
#
# from sklearn.cluster import KMeans
# mat_contents2 = sio.loadmat('PaviaHyperIm.mat')
# imh = mat_contents2['PaviaHyperIm']
# imh2 = imh.reshape(610*340, 103)
# kmeansh = KMeans(n_clusters=8, random_state=0).fit(imh2)
# labelsh = kmeansh.labels_.reshape(610, 340)
# print 'im here!'
# imgplot = plt.imshow(labelsh)
# plt.colorbar()
# plt.show()
# --- 91.3812999725 seconds ---
# slow but better pictures



# # convert to np.float32
# imh3 = np.float32(imh2)
#
# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 1.0)
# K = 8
# compactness, label, center = cv2.kmeans(imh3, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
# label1 = label.reshape(610, 340)
# print 'im here!'
# imgplot = plt.imshow(label1)
# plt.colorbar()
# plt.show()
# # --- 10.6716358662 seconds ---
# fast but not so good clustering

# print("--- %s seconds ---" % (time.time() - start_time))