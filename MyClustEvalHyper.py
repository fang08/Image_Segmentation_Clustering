from scipy.io import loadmat
from MyMartinIndex import oce
import matplotlib.pyplot as plt


def show_pic(img):
    plt.imshow(img)
    plt.colorbar()
    plt.show()


def eval_hyper(seg, gt, mask):
    for i in xrange(len(seg)):
        for j in xrange(len(seg[0])):
            if mask[i][j] == 0:
                seg[i][j] = 0

    show_pic(seg)
    show_pic(gt)
    return oce(seg, gt)


# def relabel(seg):
#     m = len(seg)
#     n = len(seg[0])
#     lmap = {}
#     label_count = 1
#     for i in xrange(m):
#         for j in xrange(n):
#             if seg[i][j] == 0:
#                 continue
#             elif lmap.get(seg[i][j]) is None:
#                 lmap.update({seg[i][j]: label_count})
#                 label_count += 1
#     for i in xrange(m):
#         for j in xrange(n):
#             if seg[i][j] != 0:
#                 seg[i][j] = lmap.get(seg[i][j])
#
#     return seg


if __name__ == '__main__':
    gt = loadmat('PaviaGrTruth.mat')['PaviaGrTruth']
    mask = loadmat('PaviaGrTruthMask.mat')['PaviaGrTruthMask']
    seg = loadmat('PaviaSegments.mat')

    print(eval_hyper(seg, gt, mask))
