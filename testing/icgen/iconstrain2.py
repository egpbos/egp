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
from egp.icgen import ConstraintLocation, ConstraintScale, HeightConstraint, ExtremumConstraint, ConstrainedField, DisplacementField, zeldovich
from matplotlib import pyplot as pl
from mayavi import mlab
import egp.toolbox
critical_density = egp.toolbox.critical_density
from iconstrain import constrain_field, sphere_grid, get_peak_particle_indices

# Decide which one to use!
from scipy.optimize import fmin_l_bfgs_b as solve
from scipy.optimize import anneal

# constants
__version__ = "0.1.1, August 2012"

# exception classes
# interface functions
# classes
# functions

def iteration_mean(pos, mass, boxlen, gridsize, rhoU, ps, cosmo, plot=False, pos0=None):
    # N.B.: pos0 is used here for plotting only.
    rhoC = constrain_field(pos, mass, boxlen, rhoU, ps, cosmo)
    
    # Now, Zel'dovich it:
    psiC = DisplacementField(rhoC)
    POS, v = zeldovich(0., psiC, cosmo) # Mpc, not h^-1!
    
    # Determine peak particle indices:
    rhoc = critical_density(cosmo)
    radius = (3*(mass*1e14)/4/np.pi/rhoc)**(1./3) # Mpc h^-1
    spheregrid = get_peak_particle_indices(pos, radius, boxlen, gridsize)
    
    # finally calculate the "new position" of the peak:
    #~ POS_i = np.array([X_i,Y_i,Z_i])
    mean_peak_pos = POS[:,spheregrid].mean(axis=1)
    
    if plot:
        points = mlab.points3d(POS[0],POS[1],POS[2], mode='point', opacity=0.5)
        cluster = mlab.points3d(pos0[0], pos0[1], pos0[2], mode='sphere', color=(1,0,0), scale_factor=radius, opacity=0.3)
        peak_points = mlab.points3d(POS[0,spheregrid], POS[1,spheregrid], POS[2,spheregrid], opacity=0.5, mode='point', color=(0,1,0))
        mlab.show()
    
    return mean_peak_pos

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


# Testing

