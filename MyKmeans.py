from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def MyKmeans (im, imageType, numClusts):
    # check errors

    height = im.shape[0]
    width = im.shape[1]
    bands = im.shape[2]
    print 'image size is: ', height, '*', width, '*', bands

    im_change = im.reshape(height * width, bands)

    # RGB images
    if imageType == 'RGB':
        kmeans = KMeans(n_clusters=numClusts, random_state=0)
        # other parameters include: n_clusters=8, init='k-means++',
        # n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto',
        # verbose=0, random_state=None, copy_x=True, n_jobs=1, algorithm='auto'
        kmeans.fit(im_change)
        labels = kmeans.labels_.reshape(height, width)

    # hyperspectral images
    elif imageType == 'Hyper':
        pca = PCA(n_components=3)
        im_reduced = pca.fit_transform(im_change)
        kmeans = KMeans(n_clusters=numClusts, random_state=0)
        kmeans.fit(im_reduced)
        labels = kmeans.labels_.reshape(height, width)

    return labels
