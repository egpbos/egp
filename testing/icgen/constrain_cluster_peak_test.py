#!/usr/bin/env python
# encoding: utf-8

from egp.icgen import *
from matplotlib import pyplot as pl

# ---- BASICS (cosmology, box, etc) ----
cosmo = Cosmology('wmap7')
cosmo.trans = 8

boxlen = 100/cosmo.h
gridsize = 64

dk = 2*np.pi/boxlen
kmax = gridsize*dk
halfgrid = gridsize/2

ps = CosmoPowerSpectrum(cosmo)
ps.normalize(boxlen**3)

# ---- FIELDS & CONSTRAINTS ----
# Unconstrained field
rhoU = GaussianRandomField(ps, boxlen, gridsize, seed=None)

# Constrained fields, 2 Jul 2012:
#path = "/Users/users/pbos/code/egp/testing/icgen/" # kapteyn
path = "/Users/patrick/astro/code/egp/testing/icgen/"

cluster_data = path+"MCXC+xyz_SCLx4.1.csv"

constraints = constraints_from_csv(path+"constraint_cluster_peak_test.csv", ps, boxlen)
rhoC1 = ConstrainedField(rhoU, constraints)

# ---- PLOTTING ----
# SyncMaster 2443 dpi:
y = 1200 #pixels
dInch = 24 # inch (diagonal)
ratio = 16./10 # it's not a 16/9 screen
yInch = dInch/np.sqrt(ratio**2+1)
dpi = y/yInch

fig = pl.figure(figsize=(20/2.54,24/2.54), dpi=dpi)
ax1 = fig.add_subplot(3,2,1)
ax2 = fig.add_subplot(3,2,2)
ax3 = fig.add_subplot(3,2,3)
ax4 = fig.add_subplot(3,2,4)
ax5 = fig.add_subplot(3,2,5)

ax1.imshow(rhoU.t[halfgrid], interpolation='nearest')
ax2.imshow(rhoC1.t[halfgrid], interpolation='nearest')
#~ ax3.imshow(rhoC2.t[halfgrid], interpolation='nearest')
#~ ax4.imshow(rhoC3.t[halfgrid], interpolation='nearest')
#~ ax5.imshow(rhoC4.t[halfgrid], interpolation='nearest')
pl.show()
