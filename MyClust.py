import scipy.io as sio
import matplotlib.pyplot as plt

from MyKmeans import MyKmeans
import matplotlib.pyplot as plt
import scipy.io as sio

from MyKmeans import MyKmeans

# from MySOM import MySOM
# from MyGMM import MyGMM
# from MyFCM import MyFCM

## load RGB images
mat_rgb = sio.loadmat('ImsAndTruths2092.mat')  # could let user input name of picture
im_rgb = mat_rgb['Im']
seg1 = mat_rgb['Seg1']
seg2 = mat_rgb['Seg2']
seg3 = mat_rgb['Seg3']
plt.imshow(im_rgb)
plt.show()

# load hyperspectral images
mat_hyper = sio.loadmat('PaviaHyperIm.mat')  # another one santa barbara
im_pavia = mat_hyper['PaviaHyperIm']

result = MyKmeans(im_rgb, 'RGB', 6)
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
