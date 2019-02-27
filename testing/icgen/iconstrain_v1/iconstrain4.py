#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
iconstrain4.py
Iterative constrainer for peaks in ICs of cosmological N-body simulations.
Version 4: not really iterative, just one step.

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
#from matplotlib import pyplot as pl
import egp.toolbox
critical_density = egp.toolbox.critical_density
from iconstrain import constrain_field, iteration_mean, sphere_grid

# Decide which one to use!
from scipy.optimize import fmin_l_bfgs_b as solve
from scipy.optimize import anneal

# constants
__version__ = "0.1.1, August 2012"

# exception classes
# interface functions
# classes
# functions

def iterate(pos0, height, scale_mpc, boxlen, gridsize, rhoU, ps, cosmo, shape_constraints = []):
    # N.B.: eventually mass0 will have to be included in pos0 as x0 = pos0,mass0
    # to iterate over pos and mass both.
    bound_range = 0.1*boxlen
    boundaries = ((pos0[0]-bound_range, pos0[0]+bound_range), (pos0[1]-bound_range, pos0[1]+bound_range), (pos0[2]-bound_range, pos0[2]+bound_range))
    lower = np.array(boundaries)[:,0]
    upper = np.array(boundaries)[:,1]
    #~ result = solve(difference, pos0, args=(pos0, mass0, boxlen, gridsize, rhoU, ps, cosmo), bounds = boundaries, approx_grad=True)#, epsilon=0.5)
    print pos0
    pos_new = iteration_mean(pos0%boxlen, height, scale_mpc, boxlen, gridsize, rhoU, ps, cosmo, shape_constraints)
    print "geeft:", pos_new
    result = 2*pos0 - pos_new # = pos0 + (pos0 - pos_new), mirror new pos in old
    print "dus na wat schuiven gebruiken we:", result
    return result
