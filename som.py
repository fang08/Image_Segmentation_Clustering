import scipy.io as sio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from minisom import MiniSom

mat_contents = sio.loadmat('ImsAndTruths2092.mat')
im = mat_contents['Im']
im2 = im.reshape(321*481,3)
im3 = np.float32(im2)

img = np.zeros([321,481,3], dtype=np.uint8)

som = MiniSom(5, 5, 3, sigma=0.1, learning_rate=0.2)
som.random_weights_init(im3)
starting_weights = som.get_weights().copy()
som.train_random(im3, 1000)

qnt = som.quantization(im3)
#print np.shape(qnt)
#print qnt



clustered = np.zeros(img.shape)
print np.shape(clustered)

for i, q in enumerate(qnt):  # place the quantized values into a new image
    clustered[np.unravel_index(i, dims=(img.shape[0],img.shape[1]))] = q
print('done.')

 #show the result
plt.figure(1)
plt.title('result')
plt.imshow(clustered)

plt.colorbar()
plt.show()

