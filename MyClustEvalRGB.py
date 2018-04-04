import scipy.io

from MyMartinIndex import oce
import os

def evalRGB(seg, gt):
    return oce(seg, gt)


if __name__ == '__main__':
    #imgs and segs
    segs = []

    for filename in os.listdir("ImsAndSegs"):
        segs.append(scipy.io.loadmat("ImsAndSegs/"+filename))

    #km
    km=[]
    res=[]
    for filename in os.listdir("km"):
        res.append(scipy.io.loadmat("km/"+filename))
    for i in xrange(len(res)):
        #res=MyKmeans.MyKmeans(res[i],"RGB",17)
        s1=evalRGB(res[i],segs[i]["Seg1"])
        s2=evalRGB(res[i],segs[i]["Seg2"])
        s3=evalRGB(res[i],segs[i]["Seg3"])
        s=min(s1,min(s2,s3))
        km.append(s)

    #fcm
    fcm = []
    res=[]
    for filename in os.listdir("fcm"):
        res.append(scipy.io.loadmat("fcm/"+filename))
    for i in xrange(len(res)):
        #res = MyFCM.MyFCM(, "RGB",17)
        s1 = evalRGB(res[i], segs[i]["Seg1"])
        s2 = evalRGB(res[i], segs[i]["Seg2"])
        s3 = evalRGB(res[i], segs[i]["Seg3"])
        s = min(s1, min(s2, s3))
        fcm.append(s)

    # SOM
    som = []
    res = []
    for filename in os.listdir("som"):
        res.append(scipy.io.loadmat("som/" + filename))
    for i in xrange(len(res)):
        # res = MyFCM.MyFCM(, "RGB",17)
        s1 = evalRGB(res[i], segs[i]["Seg1"])
        s2 = evalRGB(res[i], segs[i]["Seg2"])
        s3 = evalRGB(res[i], segs[i]["Seg3"])
        s = min(s1, min(s2, s3))
        som.append(s)

    # SC
    sc = []
    res = []
    for filename in os.listdir("som"):
        res.append(scipy.io.loadmat("som/" + filename))
    for i in xrange(len(res)):
        # res = MyFCM.MyFCM(, "RGB",17)
        s1 = evalRGB(res[i], segs[i]["Seg1"])
        s2 = evalRGB(res[i], segs[i]["Seg2"])
        s3 = evalRGB(res[i], segs[i]["Seg3"])
        s = min(s1, min(s2, s3))
        sc.append(s)

    # GMM
    gmm = []
    res = []
    for filename in os.listdir("gmm"):
        res.append(scipy.io.loadmat("gmm/" + filename))
    for i in xrange(len(res)):
        # res = MyFCM.MyFCM(, "RGB",17)
        s1 = evalRGB(res[i], segs[i]["Seg1"])
        s2 = evalRGB(res[i], segs[i]["Seg2"])
        s3 = evalRGB(res[i], segs[i]["Seg3"])
        s = min(s1, min(s2, s3))
        gmm.append(s)



    #print evalRGB(seg, gt)
