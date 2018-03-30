from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA


def MyGMM (im, imageType, numClusts):
    # check errors

    height = im.shape[0]
    width = im.shape[1]
    bands = im.shape[2]
    print 'image size is: ', height, '*', width, '*', bands

    im_change = im.reshape(height * width, bands)

    # RGB images
    if imageType == 'RGB':
        gmm = GaussianMixture(n_components=numClusts, covariance_type="tied")
        # n_components=1, covariance_type='full', tol=0.001, reg_covar=1e-06,
        # max_iter=100, n_init=1, init_params='kmeans', weights_init=None,
        # means_init=None, precisions_init=None, random_state=None,
        # warm_start=False, verbose=0, verbose_interval=10
        gmm = gmm.fit(im_change)
        clusters = gmm.predict(im_change)
        clusters = clusters.reshape(height, width)

    # hyperspectral images
    elif imageType == 'Hyper':
        pca = PCA(n_components=3)
        im_reduced = pca.fit_transform(im_change)
        gmm = GaussianMixture(n_components=numClusts, covariance_type="tied")
        gmm = gmm.fit(im_reduced)
        clusters = gmm.predict(im_reduced)
        clusters = clusters.reshape(height, width)

    return clusters