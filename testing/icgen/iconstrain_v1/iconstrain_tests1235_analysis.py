#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from egp import io, toolbox
from iconstrain import sphere_grid
from mayavi import mlab

boxlen = 100. # Mpc h^-1
gridsize = 256
loaddir = '/net/dataserver1/data/users/pbos/sims/'

z01 = io.GadgetData(loaddir+'run101_icon/snap_008')
z0no = io.GadgetData(loaddir+'run101_noicon/snap_008')
z02 = io.GadgetData(loaddir+'run102/snap_008')
z03 = io.GadgetData(loaddir+'run103/snap_008')
z05 = io.GadgetData(loaddir+'run105/snap_008')

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
pos1 = np.array(
pos2 = np.array(
pos3 = np.array(
pos5 = np.array(

radius = 8. # Mpc h^-1 (N.B.: Gadget positions are in kpc h^-1!)
sphere0 = sphere_grid(pos0, radius, boxlen, gridsize).reshape(z0no.pos.shape[0])
sphere1 = sphere_grid(pos1, radius, boxlen, gridsize).reshape(z01.pos.shape[0])
sphere2 = sphere_grid(pos2, radius, boxlen, gridsize).reshape(z02.pos.shape[0])
sphere3 = sphere_grid(pos3, radius, boxlen, gridsize).reshape(z03.pos.shape[0])
sphere5 = sphere_grid(pos5, radius, boxlen, gridsize).reshape(z05.pos.shape[0])

pos0_gadget = z0no.pos[sphere0].mean(axis=0)
pos1_gadget = z01.pos[sphere1].mean(axis=0)
pos2_gadget = z02.pos[sphere1].mean(axis=0)
pos3_gadget = z03.pos[sphere3].mean(axis=0)
pos5_gadget = z05.pos[sphere5].mean(axis=0)
d0 = np.sqrt(np.sum((pos0_gadget - pos0*1000)**2))
d1 = np.sqrt(np.sum((pos1_gadget - pos0*1000)**2))
d2 = np.sqrt(np.sum((pos2_gadget - pos0*1000)**2))
d3 = np.sqrt(np.sum((pos3_gadget - pos0*1000)**2))
d5 = np.sqrt(np.sum((pos5_gadget - pos0*1000)**2))

haloes_z01 = io.SubFindHaloes(loaddir+'run101_icon/groups_008/subhalo_tab_008.0')
haloes_z0no = io.SubFindHaloes(loaddir+'run101_noicon/groups_008/subhalo_tab_008.0')
haloes_z02 = io.SubFindHaloes(loaddir+'run102/groups_008/subhalo_tab_008.0')
haloes_z03 = io.SubFindHaloes(loaddir+'run103/groups_008/subhalo_tab_008.0')
haloes_z05 = io.SubFindHaloes(loaddir+'run105/groups_008/subhalo_tab_008.0')

# identify the SubFind halo that is most probably the one we constrained (closest in space)
halo0_index = np.sqrt(np.sum((haloes_z0no.SubPos - pos0_gadget)**2, axis=1)).argmin()
halo1_index = np.sqrt(np.sum((haloes_z01.SubPos - pos1_gadget)**2, axis=1)).argmin()
halo2_index = np.sqrt(np.sum((haloes_z02.SubPos - pos2_gadget)**2, axis=1)).argmin()
halo3_index = np.sqrt(np.sum((haloes_z03.SubPos - pos3_gadget)**2, axis=1)).argmin()
halo5_index = np.sqrt(np.sum((haloes_z05.SubPos - pos5_gadget)**2, axis=1)).argmin()

halo0_mass = haloes_z0no.SubTMass[halo0_index]
halo1_mass = haloes_z01.SubTMass[halo1_index]
halo2_mass = haloes_z02.SubTMass[halo2_index]
halo3_mass = haloes_z03.SubTMass[halo3_index]
halo5_mass = haloes_z05.SubTMass[halo5_index]

d0_halo = np.sqrt(np.sum((pos0 - haloes_z0no.SubPos[halo0_index]/1000)**2))
d1_halo = np.sqrt(np.sum((pos0 - haloes_z01.SubPos[halo1_index]/1000)**2))
d2_halo = np.sqrt(np.sum((pos0 - haloes_z02.SubPos[halo2_index]/1000)**2))
d3_halo = np.sqrt(np.sum((pos0 - haloes_z03.SubPos[halo3_index]/1000)**2))
d5_halo = np.sqrt(np.sum((pos0 - haloes_z05.SubPos[halo5_index]/1000)**2))

# Density analysis

z0den = toolbox.TSC_density(z0no.pos, gridsize, boxlen*1000, 1.)
z1den = toolbox.TSC_density(z01.pos, gridsize, boxlen*1000, 1.)
z2den = toolbox.TSC_density(z02.pos, gridsize, boxlen*1000, 1.)
z3den = toolbox.TSC_density(z03.pos, gridsize, boxlen*1000, 1.)
z5den = toolbox.TSC_density(z05.pos, gridsize, boxlen*1000, 1.)
z0den_f = np.fft.rfftn(z0den)
z1den_f = np.fft.rfftn(z1den)
z2den_f = np.fft.rfftn(z2den)
z3den_f = np.fft.rfftn(z3den)
z5den_f = np.fft.rfftn(z5den)
# smooth on 8 Mpc scale:
z0sden = np.fft.irfftn(toolbox.gaussian_smooth(z0den_f, 8000., boxlen*1000))
z1sden = np.fft.irfftn(toolbox.gaussian_smooth(z1den_f, 8000., boxlen*1000))
z2sden = np.fft.irfftn(toolbox.gaussian_smooth(z2den_f, 8000., boxlen*1000))
z3sden = np.fft.irfftn(toolbox.gaussian_smooth(z3den_f, 8000., boxlen*1000))
z5sden = np.fft.irfftn(toolbox.gaussian_smooth(z5den_f, 8000., boxlen*1000))

z0sden_max_i = np.array(np.unravel_index(z0sden.argmax(), z0sden.shape))
z1sden_max_i = np.array(np.unravel_index(z1sden.argmax(), z1sden.shape))
z2sden_max_i = np.array(np.unravel_index(z2sden.argmax(), z2sden.shape))
z3sden_max_i = np.array(np.unravel_index(z3sden.argmax(), z3sden.shape))
z5sden_max_i = np.array(np.unravel_index(z5sden.argmax(), z5sden.shape))
z0sden_max_pos = z0sden_max_i*boxlen/gridsize
z1sden_max_pos = z1sden_max_i*boxlen/gridsize
z2sden_max_pos = z2sden_max_i*boxlen/gridsize
z3sden_max_pos = z3sden_max_i*boxlen/gridsize
z5sden_max_pos = z5sden_max_i*boxlen/gridsize

d0_den = np.sqrt(np.sum((pos0-z0sden_max_pos)**2))
d1_den = np.sqrt(np.sum((pos0-z1sden_max_pos)**2))
d2_den = np.sqrt(np.sum((pos0-z2sden_max_pos)**2))
d3_den = np.sqrt(np.sum((pos0-z3sden_max_pos)**2))
d5_den = np.sqrt(np.sum((pos0-z5sden_max_pos)**2))

# Present results

print "Distance of particles in Gadget to target position, distance of most massive halo to target position, mass of this halo and distance of density maximum to target position."
print "Zonder iteratie:   ", d0, d0_halo, halo0_mass, d0_den
print "Test 1 (64^3  CC1):", d1, d1_halo, halo1_mass, d1_den
print "Test 2 (128^3 CC1):", d2, d2_halo, halo2_mass, d2_den
print "Test 3 (64^3  CC4):", d3, d3_halo, halo3_mass, d3_den
print "Test 5 (256^3 CC4):", d5, d5_halo, halo5_mass, d5_den
