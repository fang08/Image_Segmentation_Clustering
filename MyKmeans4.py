from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter
from scipy.ndimage import median_filter
from skimage.morphology import label as cl


def MyKmeans (im, imageType, numClusts):
    # check possible errors

    height = im.shape[0]
    width = im.shape[1]
    bands = im.shape[2]
    print 'image size is: ', height, '*', width, '*', bands

    # reshape images to (n_samples, n_features)
    im_change = im.reshape(height * width, bands)

    # RGB images
    if imageType == 'RGB':
        # how to find the best k?
        kmeans = KMeans(n_clusters=numClusts, random_state=0)
        # other parameters include: n_clusters=8, init='k-means++',
        # n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto',
        # verbose=0, random_state=None, copy_x=True, n_jobs=1, algorithm='auto'
        kmeans.fit(im_change)
        labels_k = kmeans.labels_.reshape(height, width)
        # calculate connected components
        cc_image = cl(labels_k, connectivity=2)
        # add filters
        labels_filtered = median_filter(labels_k, 7)
        return labels_filtered, cc_image

    # hyperspectral images
    elif imageType == 'Hyper':
        pca = PCA(n_components=3)
        im_reduced = pca.fit_transform(im_change)
        kmeans = KMeans(n_clusters=numClusts, random_state=0, n_jobs=-1)
        kmeans.fit(im_reduced)
        labels_k = kmeans.labels_.reshape(height, width)
        # add filters
        labels_filtered = median_filter(labels_k, 7)
        return labels_filtered
