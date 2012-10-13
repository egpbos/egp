#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
iconstrain.py
Iterative constrainer for peaks in ICs of cosmological N-body simulations.

Created by Evert Gerardus Patrick Bos.
Copyright (c) 2012. All rights reserved.
"""

# imports
import numpy as np
from egp.icgen import ConstraintLocation, ConstraintScale, HeightConstraint, ExtremumConstraint, ConstrainedField, DisplacementField, zeldovich, generate_shape_constraints
from matplotlib import pyplot as pl
from mayavi import mlab
import egp.toolbox

# Decide which one to use!
from scipy.optimize import fmin_l_bfgs_b as solve, anneal, brute

# constants
__version__ = "0.2, October 2012"

# exception classes
# interface functions
# classes
# functions

def constrain_field(pos, mass, boxlen, rhoU, ps, cosmo, shape_constraints = []):
    location = ConstraintLocation(pos)
    
    rhoc = egp.toolbox.critical_density(cosmo)
    scale_mpc = ((mass*1e14)/rhoc/(2*np.pi)**(3./2))**(1./3) # Mpc h^-1
    # using the volume of a gaussian window function, (2*pi)**(3./2) * R**3
    
    # N.B.: NU IS DIT NOG 1 ELEMENT, MAAR LATER MOET ALLES HIERONDER LOOPEN
    #       OVER MEERDERE PIEKEN!
    scale = ConstraintScale(scale_mpc)
    
    constraints = []
    
    # first guess for height: # DIT LATER OOK ITEREREN DOOR MASSA TE CHECKEN
    #~ sigma0 = ps.moment(0, scale_mpc, boxlen**3)
    height = mass*1e14/(2*np.pi)**(3./2)/scale_mpc**3/rhoc
    
    constraints.append(HeightConstraint(location, scale, height))
    
    # make it a real peak:
    constraints.append(ExtremumConstraint(location, scale, 0))
    constraints.append(ExtremumConstraint(location, scale, 1))
    constraints.append(ExtremumConstraint(location, scale, 2))
    
    # apply shape constraints to location & scale:
    if shape_constraints:
		constraints += generate_shape_constraints(location, scale, ps, boxlen, *shape_constraints)
    
    # Do the field stuff!
    rhoC = ConstrainedField(rhoU, constraints) # N.B.: rhoU stays the same!!!
    return rhoC

def get_peak_particle_indices(pos, radius, boxlen, gridsize):
    return sphere_grid(pos, radius, boxlen, gridsize)

def plot_all_plus_selection(points, pos0, selection, radius):
	points_plot = mlab.points3d(points[0],points[1],points[2], mode='point', opacity=0.5)
	cluster_plot = mlab.points3d(pos0[0], pos0[1], pos0[2], mode='sphere', color=(1,0,0), scale_factor=radius, opacity=0.3)
	peak_points_plot = mlab.points3d(points[0,selection], points[1,selection], points[2,selection], opacity=0.5, mode='sphere', scale_factor=radius/10., color=(0,1,0))
	mlab.show()

def iteration_mean(pos, mass, boxlen, gridsize, rhoU, ps, cosmo, shape_constraints = [], plot=False, pos0=None):
    # N.B.: pos0 is used here for plotting only.
    rhoC = constrain_field(pos, mass, boxlen, rhoU, ps, cosmo, shape_constraints)
    
    # Now, Zel'dovich it:
    psiC = DisplacementField(rhoC)
    POS, v = zeldovich(0., psiC, cosmo) # Mpc, not h^-1!
    
    # Determine peak particle indices:
    rhoc = egp.toolbox.critical_density(cosmo)
    #~ radius = (3*(mass*1e14)/4/np.pi/rhoc)**(1./3) # Mpc h^-1
    radius = ((mass*1e14)/rhoc/(2*np.pi)**(3./2))**(1./3) # Mpc h^-1
    spheregrid = get_peak_particle_indices(pos, radius, boxlen, gridsize)
    
    # finally calculate the "new position" of the peak:
    #~ POS_i = np.array([X_i,Y_i,Z_i])
    mean_peak_pos = POS[:,spheregrid].mean(axis=1)
    
    if plot:
        plot_all_plus_cluster(POS, pos0, spheregrid, radius)
    
    return mean_peak_pos

def difference(pos_iter, pos0, mass0, boxlen, gridsize, rhoU, ps, cosmo, shape_constraints = []):
    print "input:", pos_iter#, "i.e.", pos_iter%boxlen, "in the box"
    pos_new = iteration_mean(pos_iter%boxlen, mass0, boxlen, gridsize, rhoU, ps, cosmo, shape_constraints)
    print "geeft:", pos_new
    print "diff :", np.sum((pos_new - pos0)**2), "\n"
    return np.sum((pos_new - pos0)**2)

def iterate(pos0, mass0, boxlen, gridsize, rhoU, ps, cosmo, shape_constraints = [], epsilon=1e-13, factr=1e11, pgtol=1e-3):
    # N.B.: eventually mass0 will have to be included in pos0 as x0 = pos0,mass0
    # to iterate over pos and mass both.
    bound_range = 0.1*boxlen
    boundaries = ((pos0[0]-bound_range, pos0[0]+bound_range), (pos0[1]-bound_range, pos0[1]+bound_range), (pos0[2]-bound_range, pos0[2]+bound_range))
    lower = np.array(boundaries)[:,0]
    upper = np.array(boundaries)[:,1]
    result = solve(difference, pos0, args=(pos0, mass0, boxlen, gridsize, rhoU, ps, cosmo, shape_constraints), bounds = boundaries, approx_grad=True, epsilon=epsilon, factr=factr, pgtol=pgtol, iprint=0)
    #~ result = anneal(difference, pos0, args=(pos0, mass0, boxlen, gridsize, rhoU, ps, cosmo))
    #~ result = brute(difference, boundaries, args=(pos0, mass0, boxlen, gridsize, rhoU, ps, cosmo))
    return result

def sphere_grid(pos, radius, boxlen, gridsize):
    dx = boxlen/gridsize
    Xgrid = np.mgrid[-boxlen/2:boxlen/2:dx, -boxlen/2:boxlen/2:dx, -boxlen/2:boxlen/2:dx]
    
    # determine roll needed to get peak center back to where it should be (note
    # that we 'initially set it' at the cell at index (gridsize/2, gridsize/2, gridsize/2)):
    cell = np.int32(pos/dx) # "containing" cell
    roll = cell - gridsize/2 # gridsize/2 being the 'initial' index we roll from
    
    # difference of roll (= integer) with real original particle position:
    diff = (pos/dx - (cell+0.5)).reshape(3,1,1,1) # reshape for numpy broadcasting
    Xgrid -= diff*dx
    
    # (to be rolled) distance function (squared!):
    r2grid = np.sum(Xgrid**2, axis=0)
    # roll it:
    r2grid = np.roll(r2grid, roll[0], axis=0) # just roll, above no longer holds
    r2grid = np.roll(r2grid, roll[1], axis=1)
    r2grid = np.roll(r2grid, roll[2], axis=2)
    
    spheregrid = r2grid < radius**2
    return spheregrid
