import numpy as np
from minisom import MiniSom
from sklearn.decomposition import PCA
from scipy.ndimage import median_filter
from skimage.morphology import label as cl


def MySOM(im, imageType, numClusts):
    height = im.shape[0]
    width = im.shape[1]
    bands = im.shape[2]
    print 'image size is: ', height, '*', width, '*', bands
    im_change = im.reshape(height * width, bands)

    if imageType == 'RGB':
        im_float = np.float32(im_change)
        som = MiniSom(numClusts, 1, 3, sigma=0.1, learning_rate=0.5)
        som.random_weights_init(im_float)
        som.train_random(im_float, 100)
        qnt = som.quantization(im_float)
        z = som.get_weights().reshape(numClusts, 3)

    elif imageType == 'Hyper':
        pca = PCA(n_components=3)
        im_reduced = pca.fit_transform(im_change)
        im_float = np.float32(im_reduced)
        som = MiniSom(numClusts, 1, 3, sigma=0.1, learning_rate=0.5)
        som.random_weights_init(im_float)
        som.train_random(im_float, 100)
        qnt = som.quantization(im_float)
        z = som.get_weights().reshape(numClusts, 3)

    z = np.sum(z, axis=1)
    z = z.tolist()
    output = []
    for i, x in enumerate(qnt):
        output += [z.index(np.sum(x))]

    output = np.array(output)
    output = output.reshape(height, width)
    if imageType == 'RGB':
    	cc_image = cl(output, connectivity=2)
    	labels_filtered = median_filter(output,7)
    	return labels_filtered, cc_image
    else: 
	labels_filtered = median_filter(output,7)
    	return labels_filtered
