import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.ndimage import median_filter
import glob
import numpy as np
import sys

from MyKmeans4 import MyKmeans
from MyGMM4 import MyGMM
from MyFCM4 import MyFCM
from MySOM4 import MySOM
from MySpectral4 import MySpectral


algos = ['Kmeans', 'SOM', 'FCM', 'Spectral', 'GMM']
types = ['RGB', 'Hyper']



if __name__ == "__main__":
    assert len(sys.argv) >=5
    mat = sio.loadmat(sys.argv[1])
    img_name = sio.whosmat(sys.argv[1])[0][0] #Works for both RGB and hyperspectral images
    img = mat[img_name]
    arg_count = 2
    if (sys.argv[arg_count] == 'Algorithm'):
	arg_count += 1
	algo = sys.argv[arg_count]
    else:
	algo = sys.argv[arg_count]
    
    arg_count+=1
    if (sys.argv[arg_count] == 'ImType'):
	arg_count += 1
	im_type = sys.argv[arg_count]
    else:
	im_type = sys.argv[arg_count]

    arg_count+=1
    if (sys.argv[arg_count] == 'NumClusts'):
	arg_count += 1
	num_clusts = int(sys.argv[arg_count])
    else:
	num_clusts = int(sys.argv[arg_count])

    array_len = (img.shape[0]*img.shape[1]*img.shape[2])
    #print array_len
    if (num_clusts == 1):
	num_clusts = int(0.05*array_len)
    elif (num_clusts > array_len/4):
	num_clusts = array_len/4

    assert algo in algos
    assert im_type in types


    option = algos.index(algo)

    if (option == 0):
	result = MyKmeans(img, im_type, num_clusts)
    elif (option == 1):
	result = MySOM(img, im_type, num_clusts)
    elif (option == 2):
	result = MyFCM(img, im_type, num_clusts)
    elif (option == 3):
	result = MySpectral(img, im_type, num_clusts)
    elif (option == 4):
	result = MyGMM(img, im_type, num_clusts)

    if (im_type == 'RGB'):
	conn_comp = result[1]
	result = result[0]
	fig, (ax1, ax2) = plt.subplots(ncols = 2)
	img1 = ax1.imshow(result)
	fig.colorbar(img1, ax=ax1)
	ax1.set_title('Clustered Image')	
	ax1.set_aspect('auto')
	img2 = ax2.imshow(conn_comp)
	fig.colorbar(img2, ax=ax2)
	ax2.set_title('Connected Components')	
	ax2.set_aspect('auto')
	plt.tight_layout(h_pad = 1)
    
    else:
	plt.imshow(result)
	plt.title('Clustered Image')
        plt.colorbar()	

    plt.show()	
