cd dataserver/cubep3m/test1.3
ipy

from mayavi import mlab

gridsize = 64

native = np.memmap('../scratch/xv0.ic', dtype='float32')
native_x = native[1:].reshape(64**3,6)[:,:3]
mijn = np.memmap('input/xv0.ic', dtype='float32')
mijn_x = mijn[1:].reshape(64**3,6)[:,:3]

native_x = native_x.reshape(64,64,64,3)
mijn_x = mijn_x.reshape(64,64,64,3)

i = np.mgrid[1:128:2, 1:128:2, 1:128:2].reshape(3,64,64,64).T

color = np.sum((native_x - i + 0.5)**2, axis=-1)

mlab.points3d(native_x[:,:,:,0], native_x[:,:,:,1], native_x[:,:,:,2], color, mode='point')

j = np.mgrid[0:gridsize, 0:gridsize, 0:gridsize].reshape(3,gridsize,gridsize,gridsize)

mijn_x[j[2],j[1],j[0]]

# x0 = np.memmap('output/0.000xv0.dat', dtype='float32')
# x0int = np.memmap('output/0.000xv0.dat', dtype='int32')

x0 = np.memmap('3.798xv0.dat', dtype='float32')
x0int = np.memmap('3.798xv0.dat', dtype='int32')

offset = len(x0)-x0int[0]*6
x0 = x0[offset:].reshape(x0int[0],6)[:,:3]

mlab.points3d(x0[:,0], x0[:,1], x0[:,2], mode='point')
