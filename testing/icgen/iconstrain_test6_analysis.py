#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from egp import io, toolbox
from iconstrain import sphere_grid
from mayavi import mlab

boxlen = 100. # Mpc h^-1
gridsize = 256
loaddir = '/net/dataserver1/data/users/pbos/sims/'

z06 = io.GadgetData(loaddir+'run106/snap_008')

sub = np.random.randint(0, 16777216, 100000)

pos0 = np.array([20., 40., 70.])
#posi = np.array([18.2990731, 38.69969312, 69.30830766]) # oud_oud_oud
#posi = np.array([19.57218684,  38.51409928,  68.74999988]) # oud_oud
#~ pos1 = np.array([19.9562363, 39.72440481, 69.96710087]) # oud
#~ pos2 = np.array([19.78035556, 39.29931076, 69.56125407]) # oud
#pos3 = np.array([18.27326781, 38.91134458, 69.35026985]) # oud_oud_oud
#pos3 = np.array([ 19.90462669,  38.21797494,  69.30327249]) # oud_oud
#~ pos3 = np.array([19.94143765, 39.61858504, 70.00381191]) # oud
#~ pos5 = np.array([19.62608354, 38.90870564, 69.47914189]) # oud
pos6 = np.array([19.74624532, 38.40763646, 67.7002297]) # ONGEVEER, HEB NIET GEWACHT TOT LAATSTE ITERATIE STAP, MAAR VEEL VERANDERT ER TOCH NIET

radius = 4. # Mpc h^-1 (N.B.: Gadget positions are in kpc h^-1!)
sphere6 = sphere_grid(pos6, radius, boxlen, gridsize).reshape(z06.pos.shape[0])

pos6_gadget = z06.pos[sphere6].mean(axis=0)
d6 = np.sqrt(np.sum((pos6_gadget - pos0*1000)**2))

haloes_z06 = io.SubFindHaloes(loaddir+'run106/groups_008/subhalo_tab_008.0')

# identify the SubFind halo that is most probably the one we constrained (closest in space)
halo6_index = np.sqrt(np.sum((haloes_z06.SubPos - pos6_gadget)**2, axis=1)).argmin()

halo6_mass = haloes_z06.SubTMass[halo6_index]

d6_halo = np.sqrt(np.sum((pos0 - haloes_z06.SubPos[halo6_index]/1000)**2))

# Density analysis

z6den = toolbox.TSC_density(z06.pos, gridsize, boxlen*1000, 1.)
z6den_f = np.fft.rfftn(z6den)
# smooth on 8 Mpc scale:
z6sden = np.fft.irfftn(toolbox.gaussian_smooth(z6den_f, 8000., boxlen*1000))

z6sden_max_i = np.array(np.unravel_index(z6sden.argmax(), z6sden.shape))
z6sden_max_pos = z6sden_max_i*boxlen/gridsize

d6_den = np.sqrt(np.sum((pos0-z6sden_max_pos)**2))

# Present results

print "Distance of particles in Gadget to target position, distance of most massive halo to target position, mass of this halo and distance of density maximum to target position."
print "Test 6 (64^3 CC4+1):", d6, d6_halo, halo6_mass, d6_den
