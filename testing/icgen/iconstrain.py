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
import egp.icgen, egp.toolbox

# Decide which one to use!
from scipy.optimize import fmin_l_bfgs_b as solve, anneal, brute

# constants
__version__ = "0.3, December 2012"

# exception classes
# interface functions

# MULTI DIM!?!?!?!

def run(cosmo, ps, boxlen, gridsize, deltaU, target_pos, peak_height, scale_mpc, iterate, initial_guess = iterate_mirror_zeldovich, constrain_shape=True, shape_seed=0):
    """
    Call with /iterate/ one of the iteration functions. /initial_guess/ can be
    either an ndarray with an initial guess or a function that computes an
    initial guess (accepting the same arguments as /iterate/, except for the
    first one of course, which is the initial position). By default
    iterate_mirror_zeldovich is used as initial_guess function.
    """
    # shape / orientation constraints:
    if constrain_shape:
        shape_constraints = set_shape_constraints(ps, boxlen, peak_height, scale_mpc, shape_seed)
    else:
        shape_constraints = []
    
    if type(initial_guess) is np.ndarray:
        pos_initial = initial_guess
    else:
        pos_initial = initial_guess(target_pos, peak_height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo, shape_constraints)
    
    results_all = iterate(pos_initial, target_pos, peak_height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo, shape_constraints)
    result = results_all[0]
    
    return result

# functions

# N.B.: in iterate functions, eventually mass0 will have to be included in pos0
# as x0 = pos0,mass0 to iterate over pos and mass both.
# Machine precision = 2.220D-16 (seen when running fmin_l_bfgs_b with iprint=0)

# Mirror (old iconstrain4); not actually an iteration, just one step:
def iterate_mirror_zeldovich(pos0, height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo, shape_constraints = [], epsilon=None, factr=None, pgtol=None):
    pos_new = iteration_mean_zeldovich(pos0%boxlen, height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo, shape_constraints)
    result = 2*pos0 - pos_new # = pos0 + (pos0 - pos_new), mirror new pos in old
    return result

def iterate_mirror_2LPT(pos0, height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo, shape_constraints = [], epsilon=None, factr=None, pgtol=None):
    pos_new = iteration_mean_2LPT(pos0%boxlen, height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo, shape_constraints)
    result = 2*pos0 - pos_new # = pos0 + (pos0 - pos_new), mirror new pos in old
    return result

# Real iterations; solve pos0 == pos_i based on _x:
def iterate_zeldovich(pos_initial, pos0, height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo, shape_constraints = [], epsilon=1e-13, factr=1e11, pgtol=1e-3):
# uit iconstrain:
    bound_range = 0.1*boxlen
    boundaries = ((pos0[0]-bound_range, pos0[0]+bound_range), (pos0[1]-bound_range, pos0[1]+bound_range), (pos0[2]-bound_range, pos0[2]+bound_range))
    lower = np.array(boundaries)[:,0]
    upper = np.array(boundaries)[:,1]
    result = solve(difference, pos_initial, args=(pos0, height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo, iteration_mean_zeldovich, shape_constraints), bounds = boundaries, approx_grad=True, epsilon=epsilon, factr=factr, pgtol=pgtol, iprint=0)
    #~ result = anneal(difference, pos0, args=(pos0, height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo))
    #~ result = brute(difference, boundaries, args=(pos0, height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo))
    return result

def iterate_2LPT(pos_initial, pos0, mass0, boxlen, gridsize, deltaU, ps, cosmo, shape_constraints = [], epsilon=1e-13, factr=1e11, pgtol=1e-3):
# uit iconstrain5:
    bound_range = 0.1*boxlen
    boundaries = ((pos0[0]-bound_range, pos0[0]+bound_range), (pos0[1]-bound_range, pos0[1]+bound_range), (pos0[2]-bound_range, pos0[2]+bound_range))
    lower = np.array(boundaries)[:,0]
    upper = np.array(boundaries)[:,1]
    result = solve(difference, pos_initial, args=(pos0, height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo, iteration_mean_2LPT, shape_constraints), bounds = boundaries, approx_grad=True, epsilon=epsilon, factr=factr, pgtol=pgtol, iprint=0)
    #~ result = fmin_powell(difference, pos_initial, args = (pos0, mass0, boxlen, gridsize, deltaU, ps, cosmo))
    #~ result = anneal(difference, pos0, args=(pos0, mass0, boxlen, gridsize, deltaU, ps, cosmo))
    #~ result = brute(difference, boundaries, args=(pos0, mass0, boxlen, gridsize, deltaU, ps, cosmo))
    return result

def iterate_PM():
    pass

def iterate_P3M():
    pass


# Helper functions:
def difference(pos_iter, pos0, height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo, iteration_mean, shape_constraints = []):
    print "input:", pos_iter#, "i.e.", pos_iter%boxlen, "in the box"
    pos_new = iteration_mean(pos_iter%boxlen, height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo, shape_constraints)
    print "geeft:", pos_new
    print "diff :", np.sum((pos_new - pos0)**2), "\n"
    return np.sum((pos_new - pos0)**2)

def iteration_mean_zeldovich(pos, height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo, shape_constraints = [], plot=False, pos0=None):
    # N.B.: pos0 is used here for plotting only.
    deltaC = constrain_field(pos, height, scale_mpc, boxlen, deltaU, ps, cosmo, shape_constraints)
    
    # Now, Zel'dovich it:
    psiC = egp.icgen.DisplacementField(deltaC)
    POS, v = egp.icgen.zeldovich(0., psiC, cosmo) # Mpc, not h^-1!
    
    # Determine peak particle indices:
    radius = scale_mpc
    spheregrid = get_peak_particle_indices(pos, radius, boxlen, gridsize)
    
    # finally calculate the "new position" of the peak:
    mean_peak_pos = POS[:,spheregrid].mean(axis=1)
    
    if plot:
        plot_all_plus_cluster(POS, pos0, spheregrid, radius)
    
    return mean_peak_pos

def iteration_mean_2LPT(pos, height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo, shape_constraints = [], plot=False, pos0=None):
    # N.B.: pos0 is used here for plotting only.
    deltaC = constrain_field(pos, height, scale_mpc, boxlen, deltaU, ps, cosmo, shape_constraints)
    
    # Now, 2LPT it:
    psiC = egp.icgen.DisplacementField(deltaC)
    psi2C = egp.icgen.DisplacementField2ndOrder(psiC)
    POS, v = egp.icgen.two_LPT_ICs(0., psiC, psi2C, cosmo) # Mpc, not h^-1!
    
    # Determine peak particle indices:
    radius = scale_mpc
    spheregrid = get_peak_particle_indices(pos, radius, boxlen, gridsize)
    
    # finally calculate the "new position" of the peak:
    mean_peak_pos = POS[:,spheregrid].mean(axis=1)
    
    if plot:
        plot_all_plus_cluster(POS, pos0, spheregrid, radius)
    
    return mean_peak_pos


def constrain_field(pos, height, scale_mpc, boxlen, deltaU, ps, cosmo, shape_constraints = []):
    location = egp.icgen.ConstraintLocation(pos)
    
    # N.B.: NU IS DIT NOG 1 ELEMENT, MAAR LATER MOET ALLES HIERONDER LOOPEN
    #       OVER MEERDERE PIEKEN!
    scale = egp.icgen.ConstraintScale(scale_mpc)
    
    constraints = []
    
    constraints.append(egp.icgen.HeightConstraint(location, scale, height))
    
    # make it a real peak:
    constraints.append(egp.icgen.ExtremumConstraint(location, scale, 0))
    constraints.append(egp.icgen.ExtremumConstraint(location, scale, 1))
    constraints.append(egp.icgen.ExtremumConstraint(location, scale, 2))
    
    # apply shape constraints to location & scale:
    if shape_constraints:
		constraints += egp.icgen.generate_shape_constraints(location, scale, ps, boxlen, *shape_constraints)
    
    # Do the field stuff!
    deltaC = egp.icgen.ConstrainedField(deltaU, constraints) # N.B.: deltaU stays the same!!!
    return deltaC

def plot_all_plus_selection(points, pos0, selection, radius):
    from mayavi import mlab
    points_plot = mlab.points3d(points[0],points[1],points[2], mode='point', opacity=0.5)
    cluster_plot = mlab.points3d(pos0[0], pos0[1], pos0[2], mode='sphere', color=(1,0,0), scale_factor=radius, opacity=0.3)
    peak_points_plot = mlab.points3d(points[0,selection], points[1,selection], points[2,selection], opacity=0.5, mode='sphere', scale_factor=radius/10., color=(0,1,0))
    mlab.show()

def get_peak_particle_indices(pos, radius, boxlen, gridsize):
    return sphere_grid(pos, radius, boxlen, gridsize)
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
