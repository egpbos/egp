import numpy as np
from mayavi import mlab

gridsize = 64
loaddir = "/net/dataserver1/data/users/pbos/cubep3m/test1/output/"

# z = 0 snapshot
# header ifdef PPINT:
# np_local, a, t, tau, nts, dt_f_acc, dt_pp_acc, dt_c_acc, cur_checkpoint, cur_projection, cur_halofind, mass_p
# header ifndef PPINT:
# np_local, a, t, tau, nts, dt_f_acc, dt_c_acc, cur_checkpoint, cur_projection, cur_halofind, mass_p
#~ xvint = np.memmap(loaddir+'0.000xv0.dat', dtype='int32', mode='r')
#~ xv = np.memmap(loaddir+'0.000xv0.dat', dtype='float32', mode='r')
xvint = np.memmap(loaddir+'2.047xv0.dat', dtype='int32', mode='r')
xv = np.memmap(loaddir+'2.047xv0.dat', dtype='float32', mode='r')
offset = len(xvint) - xvint[0]*6 #gridsize**3*6
xv = xv[offset:].reshape(xvint[0], 6)
x = xv[:,:3] # fine mesh grid units from 0 to nc
v = xv[:,3:] # km/s

#~ mlab.points3d(x[:,0], x[:,1], x[:,2], mode='point')

# load projections:
xy = np.memmap(loaddir+'0.000proj_xy.dat', dtype='float32')
projection_size = np.sqrt(len(xy)-1)
xy = xy[1:].reshape(projection_size, projection_size)
# eerste entry is a (expansion factor)
pl.imshow(xy)

xz = np.memmap(loaddir+'0.000proj_xz.dat', dtype='float32')
xz = xz[1:].reshape(projection_size, projection_size)
pl.imshow(xz)
