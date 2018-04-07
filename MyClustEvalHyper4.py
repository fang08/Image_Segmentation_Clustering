from scipy.io import loadmat
from MyMartinIndex4 import oce
import matplotlib.pyplot as plt


def show_pic(img):
    plt.imshow(img)
    plt.colorbar()
    plt.show()


def eval_hyper(seg, gt, mask):
    for i in xrange(len(mask)):
        for j in xrange(len(mask[0])):
            if mask[i][j] == 0:
                seg[i][j] = 0

    show_pic(seg)
    return oce(seg, gt)




if __name__ == '__main__':
    gt = loadmat('PaviaGrTruth.mat')['PaviaGrTruth']
    mask = loadmat('PaviaGrTruthMask.mat')['PaviaGrTruthMask']
    seg = loadmat('PaviaSegments.mat')

    print(eval_hyper(seg, gt, mask))
