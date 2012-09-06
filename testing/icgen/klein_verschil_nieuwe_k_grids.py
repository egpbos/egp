gridsize = 64
boxlen = 100.

halfgrid = gridsize/2
dk = 2*np.pi / boxlen
kmax = gridsize*dk

k1, k2, k3 = dk*np.mgrid[0:gridsize, 0:gridsize, 0:halfgrid+1]
k1 -= kmax*(k1 > dk*(halfgrid - 1)) # shift the second half to negative k values
k2 -= kmax*(k2 > dk*(halfgrid - 1))
k = np.sqrt(k1**2 + k2**2 + k3**2)

from egp.toolbox import k_abs_grid

k2 = k_abs_grid(gridsize, boxlen)

power = ps(k)
power2 = ps(k2)
