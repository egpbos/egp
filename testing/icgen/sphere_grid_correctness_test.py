import numpy as np
from mayavi import mlab
from iconstrain import *

pos0 = np.array([27.,10.,70.])
gridsize = 64
boxlen = 100. # Mpc h^-1
dx = boxlen/gridsize

sphere = sphere_grid(pos0, 8*dx, boxlen, gridsize)

Xgrid = np.mgrid[dx/2:boxlen+dx/2:dx, dx/2:boxlen+dx/2:dx, dx/2:boxlen+dx/2:dx]

#~ mlab.points3d(Xgrid[0], Xgrid[1], Xgrid[2], mode='point', opacity=0.3)

mlab.points3d(pos0[0], pos0[1], pos0[2], scale_factor=2*8*dx, opacity=0.3, color=(0,0,1), resolution=32)

spherepos = Xgrid[:,sphere]

mlab.points3d(Xgrid[0, sphere], Xgrid[1, sphere], Xgrid[2, sphere], opacity=0.3, color=(1,0,0), scale_factor=0.2)#, mode='point')

mlab.axes()

# works perfectly! detects all particles in the sphere and none outside of it.
# these particles (Xgrid) are in the centers of the gridcells, just as the
# zeldovich function should give put the initial particles. Using the
# sphere_grid function you can thus find all particles that were originally (in
# Lagrangian coordinates) inside the peak sphere.