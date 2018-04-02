import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from minisom import MiniSom

mat_contents = sio.loadmat('ImsAndTruths2092.mat')
im = mat_contents['Im']
im2 = im.reshape(321*481,3)


#def MySOM (im, imageType, numClusts):
#height = im.shape[0]
#width = im.shape[1]
#bands = im.shape[2]
#print 'image size is: ', height, '*', width, '*', bands
#im_change = im.reshape(height * width, bands)


img = np.zeros([321,481,3], dtype=np.uint8)

#	if imageType == 'RGB':		
im3 = np.float32(im2)	
som = MiniSom(7, 7, 3, sigma=0.1, learning_rate=0.5)
som.random_weights_init(im3)
starting_weights = som.get_weights().copy()
som.train_random(im3, 100)

plt.pcolor(som.distance_map().T)
plt.colorbar()

#for c,x in enumerate(im3):
#	w = som.winner(x)
#	plt.plot(w[0]+.5, w[1]+.5)

plt.show()
	

#qnt = som.quantization(im3)

#qnt = qnt.reshape((321,481,3))
#plt.imshow(qnt)
#plt.show()
#print qnt

#cluster = np.argmin(qnt, axis = 1)

#cluster = cluster.reshape((321,481))

#plt.imshow(cluster)
#plt.colorbar()
#plt.show()



#plt.pcolor(som.distance_map().T)  # plotting the distance map as background
#plt.colorbar()
#plt.show()

#print np.shape(qnt)

#t = [0,1,2]


#markers = ['o', 's', 'D']
#colors = ['r', 'g', 'b']
#w=[]
#for cnt,xx in enumerate(im3):
	#print(som.winner(xx))
	
	#plt.plot(w[0]+.5,w[1]+.5)	
	#plt.plot(w[0]+.5, w[1]+.5,markers[t[cnt]], markerfacecolor='None',
	#	markeredgecolor=colors[t[cnt]],markersize=12,markeredgewidth=12)

#plt.show()


#clustered = np.zeros(img.shape)
#print np.shape(clustered)

#for i, q in enumerate(qnt):  # place the quantized values into a new image
#    clustered[np.unravel_index(i, dims=(img.shape[0],img.shape[1]))] = q
#print('done.')

 #show the result
#plt.figure(1)
#plt.title('result')
#plt.imshow(clustered)

#plt.colorbar()
#plt.show()

