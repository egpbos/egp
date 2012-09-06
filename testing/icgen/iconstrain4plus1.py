#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
iconstrain4.py
Iterative constrainer for peaks in ICs of cosmological N-body simulations.
Version 4+1: use the result of v4 as initial guess for v1's iteration.

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
from iconstrain import constrain_field, iteration_mean, difference

# Decide which one to use!
from scipy.optimize import fmin_l_bfgs_b as solve, fmin_powell
from scipy.optimize import anneal

# constants
__version__ = "0.1, August 2012"

# exception classes
# interface functions
# classes
# functions

def initial_guess(pos0, mass0, boxlen, gridsize, rhoU, ps, cosmo):
    # N.B.: eventually mass0 will have to be included in pos0 as x0 = pos0,mass0
    # to iterate over pos and mass both.
    bound_range = 0.1*boxlen
    boundaries = ((pos0[0]-bound_range, pos0[0]+bound_range), (pos0[1]-bound_range, pos0[1]+bound_range), (pos0[2]-bound_range, pos0[2]+bound_range))
    lower = np.array(boundaries)[:,0]
    upper = np.array(boundaries)[:,1]
    #~ result = solve(difference, pos0, args=(pos0, mass0, boxlen, gridsize, rhoU, ps, cosmo), bounds = boundaries, approx_grad=True)#, epsilon=0.5)
    print pos0
    pos_new = iteration_mean(pos0%boxlen, mass0, boxlen, gridsize, rhoU, ps, cosmo)
    print "geeft:", pos_new
    result = 2*pos0 - pos_new # = pos0 + (pos0 - pos_new), mirror new pos in old
    print "dus na wat schuiven gebruiken we:", result
    return result

def iterate(pos_initial, pos0, mass0, boxlen, gridsize, rhoU, ps, cosmo, epsilon=1e-13, factr=1e11, pgtol=1e-3):
    # Machine precision = 2.220D-16 (seen when running fmin_l_bfgs_b with iprint=0)
    # N.B.: eventually mass0 will have to be included in pos0 as x0 = pos0,mass0
    # to iterate over pos and mass both.
    bound_range = 0.1*boxlen
    boundaries = ((pos0[0]-bound_range, pos0[0]+bound_range), (pos0[1]-bound_range, pos0[1]+bound_range), (pos0[2]-bound_range, pos0[2]+bound_range))
    lower = np.array(boundaries)[:,0]
    upper = np.array(boundaries)[:,1]
    result = solve(difference, pos_initial, args=(pos0, mass0, boxlen, gridsize, rhoU, ps, cosmo), bounds = boundaries, approx_grad=True, epsilon=epsilon, factr=factr, pgtol=pgtol, iprint=0)
    #~ result = fmin_powell(difference, pos_initial, args = (pos0, mass0, boxlen, gridsize, rhoU, ps, cosmo))
    #~ result = anneal(difference, pos0, args=(pos0, mass0, boxlen, gridsize, rhoU, ps, cosmo))
    #~ result = brute(difference, boundaries, args=(pos0, mass0, boxlen, gridsize, rhoU, ps, cosmo))
    return result
