#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
iconstrain.py
Iterative constrainer for peaks in ICs of cosmological N-body simulations.

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

# Decide which one to use!
from scipy.optimize import fmin_l_bfgs_b as solve, anneal, brute

# constants
__version__ = "0.1.2, August 2012"

# exception classes
# interface functions
# classes
# functions

def constrain_field(pos, mass, boxlen, rhoU, ps, cosmo):
    location = ConstraintLocation(pos)
    
    rhoc = critical_density(cosmo)
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
    
    # Do the field stuff!
    rhoC = ConstrainedField(rhoU, constraints) # N.B.: rhoU stays the same!!!
    return rhoC

def get_peak_particle_indices(pos, radius, boxlen, gridsize):
    return sphere_grid(pos, radius, boxlen, gridsize)
    #~ # Find the mean position of the particles that were originally in the peak (or
    #~ # at least in a sphere with radius of the peak scale):
    #~ xgrid, ygrid, zgrid = np.mgrid[0:boxlen:boxlen/gridsize, 0:boxlen:boxlen/gridsize, 0:boxlen:boxlen/gridsize] + boxlen/gridsize/2 - boxlen/2
    #~ 
    #~ # determine roll needed to get peak position back to where it should be:
    #~ floor_cell = np.int32(pos/boxlen*gridsize) # "closest" cell (not really of course in half of the cases...)
    #~ roll = floor_cell - gridsize/2
    #~ # difference of roll (= integer) with real position (in cells):
    #~ diff = pos/boxlen*gridsize - floor_cell
    #~ xgrid -= diff[0]/gridsize*boxlen
    #~ ygrid -= diff[1]/gridsize*boxlen
    #~ zgrid -= diff[2]/gridsize*boxlen
    #~ 
    #~ # (to be rolled) distance function (squared!):
    #~ r2grid = xgrid**2 + ygrid**2 + zgrid**2
    #~ # roll it:
    #~ r2grid = np.roll(r2grid, -roll[0], axis=0) # roll negatively, because element[0,0,0]
    #~ r2grid = np.roll(r2grid, -roll[1], axis=1) # is not x=0,0,0 but x=boxlen,boxlen,boxlen
    #~ r2grid = np.roll(r2grid, -roll[2], axis=2) # (due to changing around in zeldovich)
    #~ 
    #~ spheregrid = r2grid < radius**2
    #~ return spheregrid

def iteration_mean(pos, mass, boxlen, gridsize, rhoU, ps, cosmo, plot=False, pos0=None):
    # N.B.: pos0 is used here for plotting only.
    rhoC = constrain_field(pos, mass, boxlen, rhoU, ps, cosmo)
    
    # Now, Zel'dovich it:
    psiC = DisplacementField(rhoC)
    POS, v = zeldovich(0., psiC, cosmo) # Mpc, not h^-1!
    
    # Determine peak particle indices:
    rhoc = critical_density(cosmo)
    #~ radius = (3*(mass*1e14)/4/np.pi/rhoc)**(1./3) # Mpc h^-1
    radius = ((mass*1e14)/rhoc/(2*np.pi)**(3./2))**(1./3) # Mpc h^-1
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
    print "input:", pos_iter#, "i.e.", pos_iter%boxlen, "in the box"
    pos_new = iteration_mean(pos_iter%boxlen, mass0, boxlen, gridsize, rhoU, ps, cosmo)
    print "geeft:", pos_new
    print "diff :", np.sum((pos_new - pos0)**2), "\n"
    #~ print "Powerspectrum stats:"
    #~ print rhoU.power.__call__.cache['grid_64_box_100.0'].mean(), rhoU.power.__call__.cache['grid_64_box_100.0'].min(), rhoU.power.__call__.cache['grid_64_box_100.0'].max(), rhoU.power.__call__.cache['grid_64_box_100.0'].std()
    #~ print "k_i stats:"
    #~ print egp.toolbox.k_i_grid.cache['grid_64_box_100.0'].mean(), egp.toolbox.k_i_grid.cache['grid_64_box_100.0'].min(), egp.toolbox.k_i_grid.cache['grid_64_box_100.0'].max(), egp.toolbox.k_i_grid.cache['grid_64_box_100.0'].std()
    #~ print "k_abs stats:"
    #~ print egp.toolbox.k_abs_grid.cache['grid_64_box_100.0'].mean(), egp.toolbox.k_abs_grid.cache['grid_64_box_100.0'].min(), egp.toolbox.k_abs_grid.cache['grid_64_box_100.0'].max(), egp.toolbox.k_abs_grid.cache['grid_64_box_100.0'].std()
    #~ print "... en volgende stap.\n"
    return np.sum((pos_new - pos0)**2)

def iterate(pos0, mass0, boxlen, gridsize, rhoU, ps, cosmo, epsilon=1e-13, factr=1e11, pgtol=1e-3):
    # N.B.: eventually mass0 will have to be included in pos0 as x0 = pos0,mass0
    # to iterate over pos and mass both.
    bound_range = 0.1*boxlen
    boundaries = ((pos0[0]-bound_range, pos0[0]+bound_range), (pos0[1]-bound_range, pos0[1]+bound_range), (pos0[2]-bound_range, pos0[2]+bound_range))
    lower = np.array(boundaries)[:,0]
    upper = np.array(boundaries)[:,1]
    result = solve(difference, pos0, args=(pos0, mass0, boxlen, gridsize, rhoU, ps, cosmo), bounds = boundaries, approx_grad=True, epsilon=epsilon, factr=factr, pgtol=pgtol, iprint=0)
    #~ result = anneal(difference, pos0, args=(pos0, mass0, boxlen, gridsize, rhoU, ps, cosmo))
    #~ result = brute(difference, boundaries, args=(pos0, mass0, boxlen, gridsize, rhoU, ps, cosmo))
    return result

def sphere_grid(pos, radius, boxlen, gridsize):
    # Find the mean position of the particles that were originally in the peak (or
    # at least in a sphere with radius of the peak scale), or MEDIAN position:
    #~ xgrid, ygrid, zgrid = np.mgrid[0:boxlen:boxlen/gridsize, 0:boxlen:boxlen/gridsize, 0:boxlen:boxlen/gridsize] + boxlen/gridsize/2 - boxlen/2
    # The above previous grid was fine in the current situation, with the
    # Zel'dovich function putting particles in the center of the gridcells, but
    # at that time the code did not put them in the center, but at the lowest
    # corner! This would have caused a systematic offset in the mean peak
    # particle positions.
    # HOWEVER, the diff was wrong also; in its previous form:
    #~ diff = (pos/boxlen*gridsize - cell)
    # the distance to the lowest corner was in fact measured, not the distance
    # to the center of the cell! So, in the end, the code was fine the way it
    # was, but rather due to compensating errors than to design. In the new code
    # we do measure distance to the center of the cell, as does zeldovich.
    # The code below is also slightly faster, more compact and more similar to
    # the zeldovich code:
    dx = boxlen/gridsize
    Xgrid = np.mgrid[-boxlen/2:boxlen/2:dx, -boxlen/2:boxlen/2:dx, -boxlen/2:boxlen/2:dx]
    # Note that this is not the same as the zeldovich grid! It it the grid
    # containing in each cell the distance of that cell's center to the center
    # of the cell at index (gridsize/2, gridsize/2, gridsize/2)). This cell will
    # be rolled to the cell where the peak actually should be and after that the
    # values of the distances will be adjusted to account for the difference
    # between the peak-cell center and the peak's exact position in the cell.
    
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
    #~ r2grid = np.roll(r2grid, -roll[0], axis=0) # roll negatively, because element[0,0,0]
    #~ r2grid = np.roll(r2grid, -roll[1], axis=1) # is not x=0,0,0 but x=boxlen,boxlen,boxlen
    #~ r2grid = np.roll(r2grid, -roll[2], axis=2) # (due to changing around in zeldovich)
    r2grid = np.roll(r2grid, roll[0], axis=0) # just roll, above no longer holds
    r2grid = np.roll(r2grid, roll[1], axis=1)
    r2grid = np.roll(r2grid, roll[2], axis=2)
    
    spheregrid = r2grid < radius**2
    return spheregrid
