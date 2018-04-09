from MyMartinIndex4 import oce
from scipy.io import loadmat
import sys

def evalRGB(seg, gt):
    return oce(seg, gt)



if __name__ == '__main__':
    f1=sys.argv[1]
    f2=sys.argv[2]

    gt = loadmat(f1)['res']
    seg = loadmat(f2)['res']

    print(evalRGB(seg, gt))