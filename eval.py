import scipy.io
from sklearn.metrics.classification import *


def evalRGB(seg, gt):
    print OCE(seg,gt)


def PE(Ig,Is):
    #preprocessing
    m1=getPixelMap(Ig)
    m2=getPixelMap(Is)
    Wj=WVector(m1)
    Wji=WMatrix(m1,m2)

    res=0
    for j in xrange(1,len(m1)+1):
        sum=0
        for i in xrange(1,len(m2)+1):
            l1=len(m1[j].intersection(m2[i]))
            l2=(len(m1[j].union(m2[i])))
            sum+=l1*1.0/l2*Wji[j][i]
        res+=(1-sum)*Wj[j]
    return res

def OCE(seg,gt):
    e1=PE(seg,gt)
    e2=PE(gt,seg)
    print e1,e2
    return min(e1,e2)

#create dict to store cluster-index pairs
def getPixelMap(seg):
    m = len(seg)
    n = len(seg[0])
    hmap ={}

    for i in xrange(m):
        for j in xrange(n):
            if hmap.get(seg[i][j]) == None:
                hmap.update({seg[i][j]: set([])})
                hmap[seg[i][j]].add((i,j))
            else:
                hmap[seg[i][j]].add((i,j))


    return hmap

def WVector(m):
    vec=[0]
    sum=0
    for i in xrange(1,len(m)+1):
        vec.append(len(m[i]))
        sum+=len(m[i])

    vec = [k*1.0 / sum for k in vec]

    #     Sum+= len(m[i])
    return vec

def WMatrix(m1, m2):
    mat = [[0 for x in range(len(m2)+1)] for y in range(len(m1)+1)]
    for j in xrange(1,len(m1)+1):
        sum=0
        temp=[0]
        Aj=m1[j]
        for i in xrange(1,len(m2)+1):
            Bi=m2[i]
            temp.append(delta(len(Aj.intersection(Bi)))*len(Bi))
            sum+=temp[-1]
        mat[j] = [k*1.0 / sum for k in temp]
    return mat


def delta(x):
    res=0
    if x == 0:
        res = 1
    else:
        res = 0
    return 1-res

if __name__ == '__main__':
    mat = scipy.io.loadmat('ImsAndTruths2092.mat')
    seg = np.array(mat["Seg1"])
    gt = np.array(mat["Seg3"])
    evalRGB(seg,gt)
