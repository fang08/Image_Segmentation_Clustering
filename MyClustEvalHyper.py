from scipy.io import loadmat
from MyMartinIndex import oce


def eval_hyper(seg, gt, mask):
    for i in xrange(len(seg)):
        for j in xrange(len(seg[0])):
            seg[i][j] *= mask[i][j]
    return oce(seg, gt)


if __name__ == '__main__':
    gt = loadmat('PaviaGrTruth.mat')
    mask = loadmat('PaviaGrTruthMask.mat')
    seg = loadmat('PaviaSegments.mat')
    print(eval_hyper(seg, gt, mask))
