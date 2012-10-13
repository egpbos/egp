#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
iconstrain5.py
Iterative constrainer for peaks in ICs of cosmological N-body simulations.
Version 5: use the result of v4 as initial guess for a v1 type-iteration, but
           now, instead of the Zel'dovich approximation, we use 2LPT.

Created by Evert Gerardus Patrick Bos.
Copyright (c) 2012. All rights reserved.
"""

# imports
import numpy as np
from egp.icgen import ConstraintLocation, ConstraintScale, HeightConstraint, ExtremumConstraint, ConstrainedField, DisplacementField, DisplacementField2ndOrder, zeldovich, two_LPT_ICs
from matplotlib import pyplot as pl
from mayavi import mlab
import egp.toolbox
from iconstrain import sphere_grid, constrain_field

# Decide which one to use!
from scipy.optimize import fmin_l_bfgs_b as solve
from scipy.optimize import anneal, fmin_powell

# constants
__version__ = "0.1, October 2012"

# exception classes
# interface functions
# classes
# functions

# N.B.: deze functie kunnen we niet gewoon uit 4plus1 importeren, want dan
#       gebruikt ie de iteration_mean van dat script!
def initial_guess(pos0, mass0, boxlen, gridsize, rhoU, ps, cosmo, shape_constraints = []):
    # N.B.: eventually mass0 will have to be included in pos0 as x0 = pos0,mass0
    # to iterate over pos and mass both.
    bound_range = 0.1*boxlen
    boundaries = ((pos0[0]-bound_range, pos0[0]+bound_range), (pos0[1]-bound_range, pos0[1]+bound_range), (pos0[2]-bound_range, pos0[2]+bound_range))
    lower = np.array(boundaries)[:,0]
    upper = np.array(boundaries)[:,1]
    #~ result = solve(difference, pos0, args=(pos0, mass0, boxlen, gridsize, rhoU, ps, cosmo), bounds = boundaries, approx_grad=True)#, epsilon=0.5)
    print pos0
    pos_new = iteration_mean(pos0%boxlen, mass0, boxlen, gridsize, rhoU, ps, cosmo, shape_constraints)
    print "geeft:", pos_new
    result = 2*pos0 - pos_new # = pos0 + (pos0 - pos_new), mirror new pos in old
    print "dus na wat schuiven gebruiken we:", result
    return result

def iteration_mean(pos, mass, boxlen, gridsize, rhoU, ps, cosmo, shape_constraints = [], plot=False, pos0=None):
    # N.B.: pos0 is used here for plotting only.
    rhoC = constrain_field(pos, mass, boxlen, rhoU, ps, cosmo, shape_constraints)
    
    # Now, 2LPT it:
    psiC = DisplacementField(rhoC)
    psi2C = DisplacementField2ndOrder(psiC)
    POS, v = two_LPT_ICs(0., psiC, psi2C, cosmo) # Mpc, not h^-1!
    
    # Determine peak particle indices:
    rhoc = egp.toolbox.critical_density(cosmo)
    #~ radius = (3*(mass*1e14)/4/np.pi/rhoc)**(1./3) # Mpc h^-1
    radius = ((mass*1e14)/rhoc/(2*np.pi)**(3./2))**(1./3) # Mpc h^-1
    spheregrid = sphere_grid(pos, radius, boxlen, gridsize)
    
    # finally calculate the "new position" of the peak:
    #~ POS_i = np.array([X_i,Y_i,Z_i])
    mean_peak_pos = POS[:,spheregrid].mean(axis=1)
    
    if plot:
        points = mlab.points3d(POS[0],POS[1],POS[2], mode='point', opacity=0.5)
        cluster = mlab.points3d(pos0[0], pos0[1], pos0[2], mode='sphere', color=(1,0,0), scale_factor=radius, opacity=0.3)
        peak_points = mlab.points3d(POS[0,spheregrid], POS[1,spheregrid], POS[2,spheregrid], opacity=0.5, mode='point', color=(0,1,0))
        mlab.show()
    
    return mean_peak_pos

def iterate(pos_initial, pos0, mass0, boxlen, gridsize, rhoU, ps, cosmo, shape_constraints = [], epsilon=1e-13, factr=1e11, pgtol=1e-3):
    # Machine precision = 2.220D-16 (seen when running fmin_l_bfgs_b with iprint=0)
    # N.B.: eventually mass0 will have to be included in pos0 as x0 = pos0,mass0
    # to iterate over pos and mass both.
    bound_range = 0.1*boxlen
    boundaries = ((pos0[0]-bound_range, pos0[0]+bound_range), (pos0[1]-bound_range, pos0[1]+bound_range), (pos0[2]-bound_range, pos0[2]+bound_range))
    lower = np.array(boundaries)[:,0]
    upper = np.array(boundaries)[:,1]
    result = solve(difference, pos_initial, args=(pos0, mass0, boxlen, gridsize, rhoU, ps, cosmo, shape_constraints), bounds = boundaries, approx_grad=True, epsilon=epsilon, factr=factr, pgtol=pgtol, iprint=0)
    #~ result = fmin_powell(difference, pos_initial, args = (pos0, mass0, boxlen, gridsize, rhoU, ps, cosmo))
    #~ result = anneal(difference, pos0, args=(pos0, mass0, boxlen, gridsize, rhoU, ps, cosmo))
    #~ result = brute(difference, boundaries, args=(pos0, mass0, boxlen, gridsize, rhoU, ps, cosmo))
    return result

def difference(pos_iter, pos0, mass0, boxlen, gridsize, rhoU, ps, cosmo, shape_constraints = []):
    print "input:", pos_iter#, "i.e.", pos_iter%boxlen, "in the box"
    pos_new = iteration_mean(pos_iter%boxlen, mass0, boxlen, gridsize, rhoU, ps, cosmo, shape_constraints)
    print "geeft:", pos_new
    print "diff :", np.sum((pos_new - pos0)**2), "\n"
    return np.sum((pos_new - pos0)**2)
