import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA


from MyKmeans import MyKmeans
from MyGMM import MyGMM
from MyFCM import MyFCM

## load RGB images
mat_rgb = sio.loadmat('ImsAndTruths2092.mat')  # could let user input name of picture
im_rgb = mat_rgb['Im']

# load hyperspectral images
mat_hyper = sio.loadmat('PaviaHyperIm.mat')   # another one santa barbara
im_pavia = mat_hyper['PaviaHyperIm']

# result = MyKmeans(im_rgb, 'RGB', 8)
# plt.imshow(result)
# plt.colorbar()
# plt.show()

# result = MyKmeans(im_pavia, 'Hyper', 8)
# plt.imshow(result)
# plt.colorbar()
# plt.show()

# result = MyGMM(im_rgb, 'RGB', 8)
# plt.imshow(result)
# plt.colorbar()
# plt.show()

# result = MyGMM(im_pavia, 'Hyper', 8)
# plt.imshow(result)
# plt.colorbar()
# plt.show()

# result = MyFCM(im_rgb, 'RGB', 8)
# plt.imshow(result)
# plt.colorbar()
# plt.show()

result = MyFCM(im_pavia, 'Hyper', 8)
plt.imshow(result)
plt.colorbar()
plt.show()