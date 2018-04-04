import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.ndimage import median_filter
import glob
import numpy as np

from MyKmeans import MyKmeans
from MyGMM import MyGMM
from MyFCM import MyFCM
from MySOM import MySOM
from MySpectral import MySpectral

from MyClustEvalHyper import eval_hyper
from MyClustEvalRGB import evalRGB



def showpic(img):
    plt.imshow(img)
    plt.colorbar()
    plt.show()


def findmin(l):
    idx = 0
    min = 1;
    for i in range(len(l)):
        if l[i] < min:
            min = l[i]
            idx = i
    return idx + 5, min


## load RGB images
# mat_rgb = sio.loadmat('ImsAndTruths2092.mat')  # could let user input name of picture
# im_rgb = mat_rgb['Im']
# seg1 = mat_rgb['Seg1']
# seg2 = mat_rgb['Seg2']
# seg3 = mat_rgb['Seg3']
# plt.imshow(im_rgb)
# plt.show()

# load hyperspectral images

def rgb_eval():

    mypath = '/home/monker490/Work/ML/Project1code/ImsAndSegs/*' ##change this path to images
    dicpath = '/home/monker490/Work/ML/rgb_results/som/'
    #onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    onlyfiles = glob.glob(mypath)
    #print onlyfiles
   
    kmean_rgb = []
    gmm_rgb = []
    fcm_rgb = []
    som_rgb = []
    spec_rgb = []
   
    for i, x in enumerate(onlyfiles):
        mat_rgb = sio.loadmat(x)
	im_rgb = mat_rgb['Im']
	segs_rgb = [mat_rgb['Seg1'], mat_rgb['Seg2'], mat_rgb['Seg3']]

	(kmean_img,conn_comp) = MySOM(im_rgb, 'RGB', 8)
	conn_comp = median_filter(conn_comp,7) #KMEAN ALREADY HAS FILTER ON OUTPUT	
	filename1 = dicpath + 'som_rgb_clustered_' + "%04d" % i
	filename2 = dicpath + 'som_rgb_concomp_' + "%04d" % i
        sio.savemat(filename1, {'res': kmean_img})
	sio.savemat(filename2, {'res': conn_comp})	
	value = 1	
	for j, s in enumerate(segs_rgb):
	    temp = evalRGB(kmean_img,s)
	    if (value > temp):
		value = temp
	kmean_rgb += [value]
	
    print ('som rgb results: ')
    print (kmean_rgb) 
    print (findmin(kmean_rgb))
    sio.savemat(dicpath + 'som_rgb_results', {'som': kmean_rgb})
		


def pavia_hyper_eval():
    mat_hyper = sio.loadmat('PaviaHyperIm.mat')  # another one santa barbara
    im_pavia = mat_hyper['PaviaHyperIm']
    gt_pavia = sio.loadmat('PaviaGrTruth.mat')['PaviaGrTruth']
    mask_pavia = sio.loadmat('PaviaGrTruthMask.mat')['PaviaGrTruthMask']

    kmean_res = []
    gmm_res = []
    fcm_res = []
    som_res = []
    spec_res = []

    showpic(gt_pavia)
    dicpath = 'hyper_results/'
    for i in xrange(5, 10):
        kmean_img = median_filter(MyKmeans(im_pavia, 'Hyper', i), 7)
        filename = dicpath + 'kmean_res_' + str(i)
        sio.savemat(filename, {'res': kmean_img})
        kmean_res.append(eval_hyper(kmean_img, gt_pavia, mask_pavia))

        gmm_img = median_filter(MyGMM(im_pavia, 'Hyper', i), 7)
        filename = dicpath + 'gmm_res_' + str(i)
        sio.savemat(filename, {'res': gmm_img})
        gmm_res.append(eval_hyper(gmm_img, gt_pavia, mask_pavia))

        fcm_img = median_filter(MyFCM(im_pavia, 'Hyper', i), 7)
        filename = dicpath + 'fcm_res_' + str(i)
        sio.savemat(filename, {'res': fcm_img})
        fcm_res.append(eval_hyper(fcm_img, gt_pavia, mask_pavia))

        som_img = median_filter(MySOM(im_pavia, 'Hyper', i), 7)
        filename = dicpath + 'som_res_' + str(i)
        sio.savemat(filename, {'res': som_img})
        som_res.append(eval_hyper(som_img, gt_pavia, mask_pavia))

        spec_img = median_filter(MySpectral(im_pavia, 'Hyper', i), 7)
        filename = dicpath + 'spec_res_' + str(i)
        sio.savemat(filename, {'res': spec_img})
        spec_res.append(eval_hyper(spec_img, gt_pavia, mask_pavia))

    print("kmeans results: ")
    print(kmean_res)
    print(findmin(kmean_res))
    sio.savemat(dicpath + 'kmean_eval_results', {'kmeans': kmean_res})
    print("gmm results: ")
    print(gmm_res)
    print(findmin(gmm_res))
    sio.savemat(dicpath + 'gmm_eval_results', {'gmm': gmm_res})
    print("fcm results: ")
    print(fcm_res)
    print(findmin(fcm_res))
    sio.savemat(dicpath + 'fcm_eval_results', {'fcm': fcm_res})
    print("som results: ")
    print(som_res)
    print(findmin(som_res))
    sio.savemat(dicpath + 'som_eval_results', {'som': fcm_res})
    print("spec results: ")
    print(spec_res)
    print(findmin(spec_res))
    sio.savemat(dicpath + 'spec_eval_results', {'spec': spec_res})


if __name__ == '__main__':
    rgb_eval()
    #pavia_hyper_eval()

# for i in range(len(result)):
#     result[i] = [k + 1 for k in result[i]]

# plt.imshow(seg1)
# plt.colorbar()
# plt.show()
# result1 = MyKmeans(im_rgb, 'RGB', 17)
# plt.imshow(result1)
# plt.colorbar()
# plt.show()

# plt.imshow(seg2)
# plt.colorbar()
# plt.show()

# seg2 = relabel(seg2)
# plt.imshow(seg2)
# plt.colorbar()
# plt.show()

# result2 = MyKmeans(im_rgb, 'RGB', 10)
# plt.imshow(result2)
# plt.colorbar()
# plt.show()

# result2 = relabel(gaussian_filter(result2, sigma=3))
# plt.imshow(result2)
# plt.colorbar()
# plt.show()


# plt.imshow(seg3)
# plt.colorbar()
# plt.show()
# result3 = MyKmeans(im_rgb, 'RGB', 24)

# plt.imshow(result3)
# plt.colorbar()
# plt.show()

# print(evalRGB(result1, seg1))
# print(evalRGB(result2, seg2))
# print(evalRGB(result3, seg3))

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
# con_comp = morphology.label(result, 8)
# plt.figure(1)
# plt.subplot(121)
# plt.title('result')
# plt.imshow(result)
# plt.subplot(122)
# plt.title('connected components')
# plt.imshow(con_comp)
# plt.tight_layout()
# plt.show()

# result = MyFCM(im_pavia, 'Hyper', 8)
# plt.imshow(result)
# plt.colorbar()
# plt.show()

# result = MySOM(im_rgb, 'RGB', 8)
# plt.imshow(result)
# plt.colorbar()
# plt.show()

# result = MySOM(im_pavia, 'Hyper', 8)
# plt.imshow(result)
# plt.colorbar()
# plt.show()
