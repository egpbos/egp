#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
iconstrain2.py
Iterative constrainer for peaks in ICs of cosmological N-body simulations.
Version 2: two-step iteration steps; Zel'dovich up to z_coll and then another.

Created by Evert Gerardus Patrick Bos.
Copyright (c) 2012. All rights reserved.
"""

"""
Todo:
- speed up constraining algorithm by rewriting elements in C
- speed up iteration by using better suited algorithm (possibly in C later on)
"""

# imports
import numpy as np
from egp.icgen import ConstraintLocation, ConstraintScale, HeightConstraint, ExtremumConstraint, ConstrainedField, DisplacementField, zeldovich_new
from matplotlib import pyplot as pl
from mayavi import mlab
import egp.toolbox
critical_density = egp.toolbox.critical_density
from iconstrain import constrain_field, sphere_grid

# Decide which one to use!
from scipy.optimize import fmin_l_bfgs_b as solve
from scipy.optimize import anneal

# constants
__version__ = "0.1.1, August 2012"

# exception classes
# interface functions
# classes
# functions

def iteration_mean(pos_i, mass_i, boxlen, gridsize, rhoU, ps, cosmo, plot=False, pos0=None):
    # N.B.: pos0 is used here for plotting only.
    rhoC_i = constrain_field(pos_i, mass_i, boxlen, rhoU, ps, cosmo)
    
    # Now, Zel'dovich it:
    psiC_i = DisplacementField(rhoC_i)
    POS_i, v_i = zeldovich(0., psiC_i, cosmo) # Mpc, not h^-1!
    
    # Find the mean position of the particles that were originally in the peak (or
    # at least in a sphere with radius of the peak scale):
    xgrid, ygrid, zgrid = np.mgrid[0:boxlen:boxlen/gridsize, 0:boxlen:boxlen/gridsize, 0:boxlen:boxlen/gridsize] + boxlen/gridsize/2 - boxlen/2
    
    # determine roll needed to get peak position back to where it should be:
    floor_cell_i = np.int32(pos_i/boxlen*gridsize) # "closest" cell (not really of course in half of the cases...)
    roll_i = floor_cell_i - gridsize/2
    # difference of roll (= integer) with real position (in cells):
    diff_i = pos_i/boxlen*gridsize - floor_cell_i
    xgrid -= diff_i[0]/gridsize*boxlen
    ygrid -= diff_i[1]/gridsize*boxlen
    zgrid -= diff_i[2]/gridsize*boxlen
    
    # (to be rolled) distance function (squared!):
    r2grid = xgrid**2 + ygrid**2 + zgrid**2
    # roll it:
    r2grid = np.roll(r2grid, -roll_i[0], axis=0) # roll negatively, because element[0,0,0]
    r2grid = np.roll(r2grid, -roll_i[1], axis=1) # is not x=0,0,0 but x=boxlen,boxlen,boxlen
    r2grid = np.roll(r2grid, -roll_i[2], axis=2) # (due to changing around in zeldovich)
    
    rhoc = critical_density(cosmo)
    scale_mpc_i = (3*(mass_i*1e14)/4/np.pi/rhoc)**(1./3) # Mpc h^-1
    spheregrid_i = r2grid < scale_mpc_i**2
    
    # finally calculate the "new position" of the peak:
    #~ POS_i = np.array([X_i,Y_i,Z_i])
    mean_peak_pos_i = POS_i[:,spheregrid_i].mean(axis=1)
    
    if plot:
        points = mlab.points3d(POS_i[0],POS_i[1],POS_i[2], mode='point', opacity=0.5)
        cluster = mlab.points3d(pos0[0], pos0[1], pos0[2], mode='sphere', color=(1,0,0), scale_factor=scale_mpc_i, opacity=0.3)
        peak_points = mlab.points3d(POS_i[0,spheregrid_i], POS_i[1,spheregrid_i], POS_i[2,spheregrid_i], opacity=0.5, mode='point', color=(0,1,0))
        mlab.show()
    
    return mean_peak_pos_i

def difference(pos_iter, pos0, mass0, boxlen, gridsize, rhoU, ps, cosmo):
    print pos_iter, "i.e.", pos_iter%boxlen, "in the box"
    pos_new = iteration_mean(pos_iter%boxlen, mass0, boxlen, gridsize, rhoU, ps, cosmo)
    print "geeft:", pos_new
    return np.sum((pos_new - pos0)**2)

def iterate(pos0, mass0, boxlen, gridsize, rhoU, ps, cosmo):
    # N.B.: eventually mass0 will have to be included in pos0 as x0 = pos0,mass0
    # to iterate over pos and mass both.
    bound_range = 0.1*boxlen
    boundaries = ((pos0[0]-bound_range, pos0[0]+bound_range), (pos0[1]-bound_range, pos0[1]+bound_range), (pos0[2]-bound_range, pos0[2]+bound_range))
    lower = np.array(boundaries)[:,0]
    upper = np.array(boundaries)[:,1]
    result = solve(difference, pos0, args=(pos0, mass0, boxlen, gridsize, rhoU, ps, cosmo), bounds = boundaries, approx_grad=True)#, epsilon=0.5)
    return result
