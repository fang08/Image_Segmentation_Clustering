import numpy as np
import skfuzzy as fuzz
from sklearn.decomposition import PCA
from scipy.ndimage import median_filter
from skimage.morphology import label as cl


def MyFCM (im, imageType, numClusts):
    # check errors

    height = im.shape[0]
    width = im.shape[1]
    bands = im.shape[2]
    print 'image size is: ', height, '*', width, '*', bands

    im_change = im.reshape(height * width, bands)

    # RGB images
    if imageType == 'RGB':
        im_trans = np.transpose(im_change)
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(im_trans, numClusts, 2, error=0.005, maxiter=1000, init=None)
        clusters = np.argmax(u, axis=0)
        clusters = clusters.reshape(height, width)
        # calculate connected components
        cc_image = cl(clusters, connectivity=2)
        # add filters
        labels_filtered = median_filter(clusters, 7)
        return labels_filtered, cc_image

    # hyperspectral images
    elif imageType == 'Hyper':
        pca = PCA(n_components=3)
        im_reduced = pca.fit_transform(im_change)
        im_trans = np.transpose(im_reduced)
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(im_trans, numClusts, 2, error=0.005, maxiter=1000, init=None)
        clusters = np.argmax(u, axis=0)
        clusters = clusters.reshape(height, width)
        # add filters
        clusters_filtered = median_filter(clusters, 7)
        return clusters_filtered
