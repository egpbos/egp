from numpy.random import random as rnd, seed
from numpy import sqrt, log, arctan2, sin, cos, exp, pi
import numpy as np

"""
Het doel hier is om uit te vinden hoe we een array van random numbers kunnen
maken in de nieuwe fieldgen die precies hetzelfde resultaat geeft als de oude
fieldgenOld.
"""

gridsize = 32
halfgrid = gridsize/2


seed(1)
# fieldgen random number arrays:
arg = rnd((gridsize,gridsize,halfgrid+1))
mod = 1 - rnd((gridsize,gridsize,halfgrid+1)) # "1 -" zodat er geen nul bij zit

seed(1)
# fieldgenOld random number arrays:
argOld = np.zeros((gridsize,gridsize,halfgrid+1))
modOld = np.zeros((gridsize,gridsize,halfgrid+1))
for k1 in range(0,gridsize):
	for k2 in range(0,gridsize):
		for k3 in range(1,halfgrid):
			argOld[k1,k2,k3] = rnd()
			modOld[k1,k2,k3] = 1 - rnd()
		for k3 in range(2):
			if k3 == 1: k3 = halfgrid
			argOld[k1,k2,k3] = rnd()
			modOld[k1,k2,k3] = 1 - rnd()

# Second try:
seed(1)
argmod = rnd((gridsize,gridsize,halfgrid+1,2))
arg2 = argmod[:,:,:,0]
arg2_ak3is0 = arg2[:,:,-2].copy()
arg2[:,:,1:-1] = arg2[:,:,:-2].copy()
arg2[:,:,0] = arg2_ak3is0

mod2 = 1 - argmod[:,:,:,1]
mod2_ak3is0 = mod2[:,:,-2].copy()
mod2[:,:,1:-1] = mod2[:,:,:-2].copy()
mod2[:,:,0] = mod2_ak3is0

