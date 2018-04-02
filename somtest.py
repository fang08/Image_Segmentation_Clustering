from mvpa2.suite import *
import scipy.io as sio
import numpy as np



mat_contents = sio.loadmat('ImsAndTruths2092.mat')
im = mat_contents['Im']
im2 = im.reshape(321*481,3)
im3 = np.float32(im2)


som = SimpleSOMMapper((321,481), 100, learning_rate=0.05)

som.train(im3)

pl.imshow(som.K[:,:,0], origin='lower')

mapped = som(im3)

pl.title('image SOM')

pl.show()
