import MyKmeans,MyFCM,MyGMM,MySOM,MySpectral
from scipy.io import loadmat,savemat

if __name__ == '__main__':
    mat = loadmat('SanBarHyperIm.mat')
    im=mat["SanBarIm88x400"]
    km = MyKmeans.MyKmeans(im,"RGB",5)
    fcm = MyFCM.MyFCM(im,"RGB",5)
    gmm = MyGMM.MyGMM(im,"RGB",5)
    som = MySOM.MySOM(im,"RGB",5)
    sc = MySpectral.MySpectral(im,"RGB",5)
    savemat("km.mat",km)
    savemat("fcm.mat",fcm)
    savemat("gmm.mat",gmm)
    savemat("som.mat",som)
    savemat("sc.mat",sc)
