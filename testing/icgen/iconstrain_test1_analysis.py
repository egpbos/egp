#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from egp import io, toolbox
from iconstrain import sphere_grid
from mayavi import mlab

boxlen = 100. # Mpc h^-1
gridsize = 256

loaddir = '/net/dataserver1/data/users/pbos/sims/'

z0 = io.GadgetData(loaddir+'run101_icon/snap_008')
z0no = io.GadgetData(loaddir+'run101_noicon/snap_008')

z0.loadPos()
z0no.loadPos()

sub = np.random.randint(0, 16777216, 100000)

pos0 = np.array([20., 40., 70.])
#posi = np.array([18.2990731, 38.69969312, 69.30830766]) # oud
posi = np.array([19.57218684,  38.51409928,  68.74999988])

radius = 8. # Mpc h^-1 (N.B.: Gadget positions are in kpc h^-1!)
sphere0 = sphere_grid(pos0, radius, boxlen, gridsize).reshape(z0no.pos.shape[0])
spherei = sphere_grid(posi, radius, boxlen, gridsize).reshape(z0.pos.shape[0])

pos0_gadget = z0no.pos[sphere0].mean(axis=0)
posi_gadget = z0.pos[spherei].mean(axis=0)
d0 = np.sqrt(np.sum((pos0_gadget - pos0*1000)**2))
di = np.sqrt(np.sum((posi_gadget - pos0*1000)**2))

def make_movie():
    mlab.figure(1); icon = mlab.points3d(z0.pos[sub,0], z0.pos[sub,1], z0.pos[sub,2], mode='point', opacity=0.5)
    mlab.figure(1); noicon = mlab.points3d(z0no.pos[sub,0], z0no.pos[sub,1], z0no.pos[sub,2], mode='point', opacity=0.5, color=(0,0,1))
    mlab.figure(1); icon_peak = mlab.points3d(z0.pos[spherei][:,0], z0.pos[spherei][:,1], z0.pos[spherei][:,2], mode='point', color=(0,1,0))
    mlab.figure(1); noicon_peak = mlab.points3d(z0no.pos[sphere0][:,0], z0no.pos[sphere0][:,1], z0no.pos[sphere0][:,2], mode='point', color=(1,0,0))
    mlab.figure(1); cluster = mlab.points3d(1000*pos0[0], 1000*pos0[1], 1000*pos0[2], mode='sphere', color=(0.5,0.5,0), scale_factor=radius*1000, opacity=0.3)
    
    az = mlab.view()[0]
    for i in range(0, 360):
        mlab.view(az+i)
        mlab.savefig(loaddir+'fig_az%03i.png' % az+i)
    
    # ssh hathor
    # convert *.png rotate.gif

#~ make_movie()

haloes_z0 = io.SubFindHaloes(loaddir+'run101_icon/groups_008/subhalo_tab_008.0')
haloes_z0no = io.SubFindHaloes(loaddir+'run101_noicon/groups_008/subhalo_tab_008.0')

# identify the SubFind halo that is most probably the one we constrained (closest in space)
haloi_index = np.sqrt(np.sum((haloes_z0.SubPos - posi_gadget)**2, axis=1)).argmin()
halo0_index = np.sqrt(np.sum((haloes_z0no.SubPos - pos0_gadget)**2, axis=1)).argmin()

haloi_mass = haloes_z0.SubTMass[haloi_index]
halo0_mass = haloes_z0no.SubTMass[halo0_index]

d0_halo = np.sqrt(np.sum((pos0 - haloes_z0no.SubPos[halo0_index]/1000)**2))
di_halo = np.sqrt(np.sum((pos0 - haloes_z0.SubPos[haloi_index]/1000)**2))
d2_halo = np.sqrt(np.sum((pos0 - haloes_z02.SubPos[halo2_index]/1000)**2))


# Density analysis:

z0den = toolbox.TSC_density(z0no.pos, gridsize, boxlen*1000, 1.)
ziden = toolbox.TSC_density(z0.pos, gridsize, boxlen*1000, 1.)
z0den_f = np.fft.rfftn(z0den)
ziden_f = np.fft.rfftn(ziden)
# smooth on 8 Mpc scale:
z0sden = np.fft.irfftn(toolbox.gaussian_smooth(z0den_f, 8000., boxlen*1000))
zisden = np.fft.irfftn(toolbox.gaussian_smooth(ziden_f, 8000., boxlen*1000))

z0sden_max_i = np.array(np.unravel_index(z0sden.argmax(), z0sden.shape))
zisden_max_i = np.array(np.unravel_index(zisden.argmax(), zisden.shape))
z0sden_max_pos = z0sden_max_i*boxlen/gridsize
zisden_max_pos = zisden_max_i*boxlen/gridsize

d0_den = np.sqrt(np.sum((pos0-z0sden_max_pos)**2))
di_den = np.sqrt(np.sum((pos0-zisden_max_pos)**2))
