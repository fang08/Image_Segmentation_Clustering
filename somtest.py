import scipy.io as sio
import numpy as np
from matplotlib import pyplot as plt
from sompy.sompy import SOMFactory
from sompy.visualization.mapview import View2D
from sompy.visualization.hitmap import HitMapView
from sompy.visualization.umatrix import UMatrixView

mat_contents = sio.loadmat('ImsAndTruths2092.mat')

im = mat_contents['Im']
seg1 = mat_contents['Seg1']
seg2 = mat_contents['Seg2']
seg3 = mat_contents['Seg3']

im2 = im.reshape(321 * 481, 3)
seg1 = seg1.reshape(321 * 481)
seg2 = seg2.reshape(321 * 481)
seg3 = seg3.reshape(321 * 481)
# print np.shape(seg1)

data1 = np.column_stack([im2, seg1])
data2 = np.column_stack([im2, seg2])
data3 = np.column_stack([im2, seg3])

# print np.shape(data)

sm = SOMFactory().build(im2, normalization='var', initialization='random')
sm.train(n_job=1, verbose='info', train_rough_len=2, train_finetune_len=5)

# view2D  = View2D(10,10,"rand data",text_size=10)
# view2D.show(sm, col_sz=4, which_dim="all", desnormalize=True)
# sm.cluster(8)
hits = HitMapView(20, 20, "Clustering")
a = hits.show(sm)

# umat = UMatrixView(100,100,"Unified Distance Matrix", text_size=14)

# umat.show(sm)
