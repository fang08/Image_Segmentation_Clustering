import scipy as sp
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from skimage.transform import resize
from skimage.morphology import label as cl
from scipy.ndimage import median_filter


def MySpectral (im, imageType, numClusts):
    # check errors

    height = im.shape[0]
    width = im.shape[1]
    bands = im.shape[2]
    print 'image size is: ', height, '*', width, '*', bands

    # s_height = height / 4
    # s_width = width / 4
    # im_re = resize(im, (s_height, s_width, bands))

    # image_resized = resize(im, (im.shape[0] / 4, im.shape[1] / 4, 3))
    # im_input = im.reshape(image_resized.shape[0] * image_resized.shape[1], bands)


    # RGB images
    if imageType == 'RGB':
        org_size = im.shape
        # im_small = sp.misc.imresize(im, size=0.20, mode="RGB")
        im_small = resize(im, (org_size[0] / 4, org_size[1] / 4))
        shk_size = im_small.shape
        im_input = im_small.reshape(shk_size[0] * shk_size[1], 3)

        sc = SpectralClustering(n_clusters=numClusts, affinity='nearest_neighbors', n_jobs=-1)
        # n_clusters = 8, eigen_solver = None, random_state = None, n_init = 10, gamma = 1.0, affinity ='rbf',
        # n_neighbors = 10, eigen_tol = 0.0, assign_labels ='kmeans', degree = 3, coef0 = 1,
        # kernel_params = None, n_jobs = 1 eigen_solver='arpack',
        sc.fit(im_input)
        labels = sc.labels_  #.astype(np.int)
        labels = labels.reshape(shk_size[0], shk_size[1])
        # labels_r = sp.misc.imresize(labels, size=(org_size[0], org_size[1]))
        labels_filtered = median_filter(labels, 7)
        labels_r = resize(labels_filtered, (org_size[0], org_size[1]), preserve_range=True, clip=False)

        # labels = labels.reshape(image_resized.shape[0], image_resized.shape[1])
        # labels_r = resize(labels, (im[0], im[1]))

        # calculate connected components
        cc_image = cl(labels_r, connectivity=2)
        labels_r = cast_label(labels_r)

        return labels_r, cc_image

    # hyperspectral images
    elif imageType == 'Hyper':
        im_change = im.reshape(height * width, bands)
        pca = PCA(n_components=3)
        im_reduced = pca.fit_transform(im_change)
        print (im_reduced.shape)

        imh = im_reduced.reshape(height, width, 3)
        print (imh.shape)

        org_size = imh.shape
        # im_small = sp.misc.imresize(imh, size=0.20, mode="RGB")
        im_small = resize(imh, (122, 68))
        shk_size = im_small.shape
        im_input = im_small.reshape(shk_size[0] * shk_size[1], 3)

        sc = SpectralClustering(n_clusters=numClusts, eigen_solver='arpack', affinity='nearest_neighbors')
        sc.fit(im_input)
        labels = sc.labels_
        labels = labels.reshape(shk_size[0], shk_size[1])
        # labels_r = sp.misc.imresize(labels, size=(org_size[0], org_size[1]), )
        labels = median_filter(labels, 7)
        labels_r = resize(labels, (org_size[0], org_size[1]), preserve_range=True, clip=False)
        labels_r = cast_label(labels_r)
        return labels_r


def cast_label(img):
    for i in range(len(img)):
        for j in range(len(img[0])):
            img[i][j] = int(round(img[i][j]) + 1)
    return img
