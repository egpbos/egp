#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import egp.io, egp.toolbox
from iconstrain import sphere_grid

def analyse_run(run_dir_base, run_name, pos0, boxlen, gridsize, radius, snapshot_number = '008'):
    z0 = egp.io.GadgetData(run_dir_base+'/'+run_name+'/snap_'+snapshot_number)
    with open(run_dir_base+'/'+run_name+'.log', 'r') as log_file:
        log_file.readline()
        pos_iter = np.array(log_file.readline().replace('[',"").replace(']',"").split(), dtype='float32')
    sphere = sphere_grid(pos_iter, radius, boxlen, gridsize).reshape(z0.pos.shape[0])
    
    pos_gadget = z0.pos[sphere].mean(axis=0)/1000.
    dpos = np.sqrt(np.sum((pos_gadget - pos0)**2))
    
    haloes_z0 = egp.io.SubFindHaloes(run_dir_base+'/'+run_name+'/groups_'+snapshot_number+'/subhalo_tab_'+snapshot_number+'.0')
    # identify the SubFind halo that is most probably the one we constrained (closest in space)
    halo_index = np.sqrt(np.sum((haloes_z0.SubPos/1000 - pos_gadget)**2, axis=1)).argmin()
    halo_mass = haloes_z0.SubTMass[halo_index]
    dpos_halo = np.sqrt(np.sum((pos0 - haloes_z0.SubPos[halo_index]/1000)**2))
    
    # Density analysis
    den = egp.toolbox.TSC_density(z0.pos, gridsize, boxlen, 1.)
    den_f = np.fft.rfftn(den)
    # smooth on 8 Mpc scale:
    sden = np.fft.irfftn(toolbox.gaussian_smooth(den_f, 8., boxlen))
    sden_max_i = np.array(np.unravel_index(sden.argmax(), sden.shape))
    sden_max_pos = sden_max_i*boxlen/gridsize
    dpos_den = np.sqrt(np.sum((pos0-sden_max_pos)**2))
    
    # Present results
    print "Distance of particles in Gadget to target position: ", dpos
    print "Distance of most massive halo to target position: ", dpos_halo
    print "Distance of density maximum to target position: ", dpos_den
    print "Mass of this halo:", halo_mass
    return dpos, dpos_halo, dpos_den, halo_mass
