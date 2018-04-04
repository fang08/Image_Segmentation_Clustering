import scipy.io

from MyMartinIndex import oce
from os import walk
import MyFCM,MyKmeans
import os

def evalRGB(seg, gt):
    return oce(seg, gt)


if __name__ == '__main__':
    #imgs and segs
    segs = []

    for filename in os.listdir("ImsAndSegs"):
        segs.append(scipy.io.loadmat("ImsAndSegs/"+filename))

    #km
    score1=[]
    res=[]
    for filename in os.listdir("Kmeans"):
        res.append(scipy.io.loadmat("Kmeans/"+filename))
    for i in xrange(len(res)):
        #res=MyKmeans.MyKmeans(res[i],"RGB",17)
        s1=evalRGB(res[i],segs[i]["Seg1"])
        s2=evalRGB(res[i],segs[i]["Seg2"])
        s3=evalRGB(res[i],segs[i]["Seg3"])
        s=min(s1,min(s2,s3))
        score1.append(s)

    #fcm
    score2 = []
    res=[]
    for filename in os.listdir("FCM"):
        res.append(scipy.io.loadmat("FCM/"+filename))
    for i in xrange(len(res)):
        #res = MyFCM.MyFCM(, "RGB",17)
        s1 = evalRGB(res[i], segs[i]["Seg1"])
        s2 = evalRGB(res[i], segs[i]["Seg2"])
        s3 = evalRGB(res[i], segs[i]["Seg3"])
        s = min(s1, min(s2, s3))
        score2.append(s)




    #print evalRGB(seg, gt)
