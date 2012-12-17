#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
iconstrain2.py
Iterative constrainer for peaks in ICs of cosmological N-body simulations.
Version 2: two-step iteration steps; Zel'dovich up to z_coll and then another.

N.B.: this file doesn't actually contain the algorithm, it is in iconstrain_test4.py.
This is because the tests turned out that this method is useless, so it wasn't
really ever implemented properly after that.

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
from egp.icgen import ConstraintLocation, ConstraintScale, HeightConstraint, ExtremumConstraint, ConstrainedField, DisplacementField, zeldovich, zeldovich_step
from matplotlib import pyplot as pl
from mayavi import mlab
import egp.toolbox
critical_density = egp.toolbox.critical_density
from iconstrain import constrain_field, sphere_grid, get_peak_particle_indices, iteration_mean

# Decide which one to use!
from scipy.optimize import fmin_l_bfgs_b as solve
from scipy.optimize import anneal

# constants
__version__ = "0.1.2, August 2012"

# exception classes
# interface functions
# classes
# functions

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

