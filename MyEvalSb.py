from MyKmeans import MyKmeans
from MyFCM import MyFCM
from MyGMM import MyGMM
from MySOM import MySOM
from MySpectral import MySpectral
from scipy.io import loadmat,savemat

if __name__ == '__main__':
    mat = loadmat('SanBarHyperIm.mat')
    im=mat["SanBarIm88x400"]
    km = MyKmeans(im,"RGB",5)
    fcm = MyFCM(im,"RGB",5)
    gmm = MyGMM(im,"RGB",5)
    som = MySOM(im,"RGB",5)
    sc = MySpectral(im,"RGB",5)
    savemat("km.mat",km)
    savemat("fcm.mat",fcm)
    savemat("gmm.mat",gmm)
    savemat("som.mat",som)
    savemat("sc.mat",sc)
