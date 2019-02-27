import numpy as np
from mayavi import mlab

loaddir = "/net/dataserver1/data/users/pbos/cubep3m/scratch/"

# ICs
xv0int = np.memmap(loaddir+'xv0.ic', dtype='int32', mode='r')
xv0 = np.memmap(loaddir+'xv0.ic', dtype='float32', mode='r')
xv0 = xv0[1:].reshape(xv0int[0], 6)
x0 = xv0[:,:3] # fine mesh grid units from 0 to nc
v0 = xv0[:,3:] # km/s

#~ mlab.points3d(x0[:,0], x0[:,1], x0[:,2], mode='point')

