#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from egp import io, toolbox
from iconstrain import sphere_grid, plot_all_plus_selection
#from mayavi import mlab

boxlen = 100. # Mpc h^-1
gridsize = 256
loaddir = '/net/dataserver1/data/users/pbos/sims/'

run_name = "run110_2522572538_gauss"

z0i = io.GadgetData(loaddir+run_name+'/snap_008')

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
#~ pos6 = np.array([19.74624532, 38.40763646, 67.7002297]) # OUD # ONGEVEER, HEB NIET GEWACHT TOT LAATSTE ITERATIE STAP, MAAR VEEL VERANDERT ER TOCH NIET
#~ pos7 = np.array([ 19.93056985,  38.59509036,  67.98152977])
#posi = np.array([ 20.26747414,  41.04179358,  69.18462654])
posi = np.array([20.3,40.,67.5]) # ONGEVEER! 21 nov 18:13

radius = 4. # Mpc h^-1 (N.B.: Gadget positions are in kpc h^-1!)
spherei = sphere_grid(posi, radius, boxlen, gridsize).reshape(z0i.pos.shape[0])

#~ plot_all_plus_selection(z0i.pos.T.reshape(3,256,256,256), pos0, sphere, radius)

posi_gadget = z0i.pos[spherei].mean(axis=0)
di = np.sqrt(np.sum((posi_gadget - pos0*1000)**2))

haloes_z0i = io.SubFindHaloes(loaddir+run_name+'/groups_008/subhalo_tab_008.0')

# identify the SubFind halo that is most probably the one we constrained (closest in space)
haloi_index = np.sqrt(np.sum((haloes_z0i.SubPos - posi_gadget)**2, axis=1)).argmin()

haloi_mass = haloes_z0i.SubTMass[haloi_index]

di_halo = np.sqrt(np.sum((pos0 - haloes_z0i.SubPos[haloi_index]/1000)**2))

# Density analysis

ziden = toolbox.TSC_density(z0i.pos, gridsize, boxlen*1000, 1.)
ziden_f = np.fft.rfftn(ziden)
# smooth on 4 Mpc scale:
zisden = np.fft.irfftn(toolbox.gaussian_smooth(ziden_f, 4000., boxlen*1000))

zisden_max_i = np.array(np.unravel_index(zisden.argmax(), zisden.shape))
zisden_max_pos = zisden_max_i*boxlen/gridsize

di_den = np.sqrt(np.sum((pos0-zisden_max_pos)**2))

# Present results

print "Distance of particles in Gadget to target position, distance of most massive halo to target position, mass of this halo and distance of density maximum to target position."
print "Test 10 (256^3 CC7):", di, di_halo, haloi_mass, di_den

