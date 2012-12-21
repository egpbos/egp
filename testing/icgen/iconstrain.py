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
import glob, os, subprocess, shutil

# Decide which one to use!
from scipy.optimize import fmin_l_bfgs_b as solve, anneal, brute

# constants
__version__ = "0.3, December 2012"

# functions

# N.B.: in iterate functions, eventually mass0 will have to be included in pos0
# as x0 = pos0,mass0 to iterate over pos and mass both.
# Machine precision = 2.220D-16 (seen when running fmin_l_bfgs_b with iprint=0)

# Mirror (old iconstrain4); not actually an iteration, just one step:
def iterate_mirror(iteration, peak_pos_initial, target_pos, height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo, shape_constraints = [], epsilon=None, factr=None, pgtol=None):
    peak_pos_evolved = iteration(peak_pos_initial%boxlen, height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo, shape_constraints)
    result = 2*target_pos - peak_pos_evolved # = target_pos + (target_pos - peak_pos_evolved), mirror evolved pos in target pos
    return result
    
def iterate_mirror_zeldovich(peak_pos_initial, target_pos, height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo, shape_constraints = [], epsilon=None, factr=None, pgtol=None):
    return iterate_mirror(iteration_mean_zeldovich, peak_pos_initial, target_pos, height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo, shape_constraints)

def iterate_mirror_2LPT(peak_pos_initial, target_pos, height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo, shape_constraints = [], epsilon=None, factr=None, pgtol=None):
    return iterate_mirror(iteration_mean_2LPT, peak_pos_initial, target_pos, height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo, shape_constraints)

# Real iterations; solve target_pos == pos_i based on _x:
def iterate_solve(iteration, peak_pos_initial, target_pos, height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo, shape_constraints = [], epsilon=1e-13, factr=1e11, pgtol=1e-3):
    bound_range = 0.1*boxlen
    boundaries = ((target_pos[0]-bound_range, target_pos[0]+bound_range), (target_pos[1]-bound_range, target_pos[1]+bound_range), (target_pos[2]-bound_range, target_pos[2]+bound_range))
    lower = np.array(boundaries)[:,0]
    upper = np.array(boundaries)[:,1]
    result = solve(difference, peak_pos_initial, args=(target_pos, height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo, iteration, shape_constraints), bounds = boundaries, approx_grad=True, epsilon=epsilon, factr=factr, pgtol=pgtol, iprint=0)
    #~ result = anneal(difference, peak_pos_initial, args=(target_pos, height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo))
    #~ result = brute(difference, boundaries, args=(target_pos, height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo))
    #~ result = fmin_powell(difference, peak_pos_initial, args = (target_pos, height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo))
    return result

def iterate_zeldovich(peak_pos_initial, target_pos, height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo, shape_constraints = [], epsilon=1e-13, factr=1e11, pgtol=1e-3):
    return iterate_solve(iteration_mean_zeldovich, peak_pos_initial, target_pos, height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo, shape_constraints, epsilon, factr, pgtol)

def iterate_2LPT(peak_pos_initial, target_pos, height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo, shape_constraints = [], epsilon=1e-13, factr=1e11, pgtol=1e-3):
    return iterate_solve(iteration_mean_2LPT, peak_pos_initial, target_pos, height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo, shape_constraints, epsilon, factr, pgtol)

def iterate_PM(peak_pos_initial, target_pos, height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo, shape_constraints = [], epsilon=1e-13, factr=1e11, pgtol=1e-3):
    return iterate_solve(iteration_mean_PM, peak_pos_initial, target_pos, height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo, shape_constraints, epsilon, factr, pgtol)

def iterate_P3M(peak_pos_initial, target_pos, height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo, shape_constraints = [], epsilon=1e-13, factr=1e11, pgtol=1e-3):
    return iterate_solve(iteration_mean_P3M, peak_pos_initial, target_pos, height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo, shape_constraints, epsilon, factr, pgtol)


# interface functions

def run(cosmo, ps, boxlen, gridsize, deltaU, target_pos, peak_height, scale_mpc, iterate, initial_guess = iterate_mirror_zeldovich, constrain_shape=True, shape_seed=0):
    """
    Call with /iterate/ one of the iterate_## functions. /initial_guess/ can be
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
        pos_initial = initial_guess(target_pos, target_pos, peak_height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo, shape_constraints)
    
    print pos_initial
    results_all = iterate(pos_initial, target_pos, peak_height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo, shape_constraints)
    result = results_all[0]
    
    return result


# MULTI PEAK!?!?!?!
# Wrs andere functie voor maken, of lijsten.


# Helper functions:
def difference(peak_pos_input, target_evolved_pos, height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo, iteration_mean, shape_constraints = []):
    print "input:", peak_pos_input#, "i.e.", peak_pos_input%boxlen, "in the box"
    evolved_peak_pos = iteration_mean(peak_pos_input%boxlen, height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo, shape_constraints)
    print "geeft:", evolved_peak_pos
    print "diff :", np.sum((evolved_peak_pos - target_evolved_pos)**2), "\n"
    return np.sum((evolved_peak_pos - target_evolved_pos)**2)

def iteration_mean_CubeP3M(pp_run, peak_pos, height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo, shape_constraints = [], initial_redshift = 63., run_path_base = '/Users/users/pbos/dataserver/cubep3m/iconstrain_scratch/', save_steps = False, cores = 8, nf_tile_I = 2):
    deltaC = constrain_field(peak_pos, height, scale_mpc, boxlen, deltaU, ps, cosmo, shape_constraints)
    psiC = egp.icgen.DisplacementField(deltaC)
    x, v = egp.icgen.zeldovich(initial_redshift, psiC, cosmo) # Mpc, not h^-1!
    
    # PM it (CubePM):
    snapshots = np.array([initial_redshift/2., 0.])
    # determine run_name:
    if save_steps:
        ls = glob.glob(run_path_base+'/*')
        if not ls:
            run_name = '0000'
        else:
            previous_step = int(os.path.basename(ls[-1]))
            run_name = '%04i' % previous_step+1
    else:
        run_name = 'step'
        try:
            shutil.rmtree(run_path_base+'/'+run_name)
        except OSError:
            pass
    # setup:
    run_script_path = egp.io.setup_cubep3m_run(x, v, cosmo, boxlen, gridsize, initial_redshift, snapshots, run_name, run_path_base, cores, pid_flag = True, pp_run = pp_run, nf_tile_I = nf_tile_I)
    # run:
    subprocess.call(run_script_path, shell=True)
    # load result:
    simulation = egp.io.CubeP3MData(run_path_base+run_name+'/output/0.000xv0.dat')
    
    # Determine peak particle indices:
    radius = scale_mpc
    spheregrid = get_peak_particle_indices(peak_pos, radius, boxlen, gridsize).reshape(gridsize**3)
    # Remove the spheregrid-cells of particles that were removed:
    if simulation.Ntotal < gridsize**3:
        pidarray = simulation.get_pid_array()
        np.sort(pidarray)
        all_pids = np.arange(gridsize**3)+1
        deleted = np.setdiff1d(all_pids, pidarray)
        spheregrid = spheregrid[-np.in1d(all_pids, deleted)]
    
    # finally calculate the "new position" of the peak:
    mean_evolved_peak_pos = simulation.pos[spheregrid].mean(axis=0)
    
    return mean_evolved_peak_pos

def iteration_mean_PM(peak_pos, height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo, shape_constraints = [], initial_redshift = 63., run_path_base = '/Users/users/pbos/dataserver/cubep3m/iconstrain_scratch/', save_steps = False, cores = 8, nf_tile_I = 2):
    return iteration_mean_CubeP3M(False, peak_pos, height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo, shape_constraints, initial_redshift, run_path_base, save_steps, cores, nf_tile_I = nf_tile_I)

def iteration_mean_P3M(peak_pos, height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo, shape_constraints = [], initial_redshift = 63., run_path_base = '/Users/users/pbos/dataserver/cubep3m/iconstrain_scratch/', save_steps = False, cores = 8, nf_tile_I = 2):
    return iteration_mean_CubeP3M(True, peak_pos, height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo, shape_constraints, initial_redshift, run_path_base, save_steps, cores, nf_tile_I = nf_tile_I)


def iteration_mean_zeldovich(peak_pos, height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo, shape_constraints = []):
# uit iconstrain:
    deltaC = constrain_field(peak_pos, height, scale_mpc, boxlen, deltaU, ps, cosmo, shape_constraints)
    
    # Now, Zel'dovich it:
    psiC = egp.icgen.DisplacementField(deltaC)
    x, v = egp.icgen.zeldovich(0., psiC, cosmo) # Mpc, not h^-1!
    
    # Determine peak particle indices:
    radius = scale_mpc
    spheregrid = get_peak_particle_indices(peak_pos, radius, boxlen, gridsize)
    
    # finally calculate the "new position" of the peak:
    mean_evolved_peak_pos = x[:,spheregrid].mean(axis=1)
        
    return mean_evolved_peak_pos

def iteration_mean_2LPT(peak_pos, height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo, shape_constraints = []):
# uit iconstrain5:
    deltaC = constrain_field(peak_pos, height, scale_mpc, boxlen, deltaU, ps, cosmo, shape_constraints)
    
    # Now, 2LPT it:
    psiC = egp.icgen.DisplacementField(deltaC)
    psi2C = egp.icgen.DisplacementField2ndOrder(psiC)
    x, v = egp.icgen.two_LPT_ICs(0., psiC, psi2C, cosmo) # Mpc, not h^-1!
    
    # Determine peak particle indices:
    radius = scale_mpc
    spheregrid = get_peak_particle_indices(peak_pos, radius, boxlen, gridsize)
    
    # finally calculate the "new position" of the peak:
    mean_evolved_peak_pos = x[:,spheregrid].mean(axis=1)
    
    return mean_evolved_peak_pos

 
def constrain_field(peak_pos, height, scale_mpc, boxlen, deltaU, ps, cosmo, shape_constraints = []):
    location = egp.icgen.ConstraintLocation(peak_pos)
    
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

def set_shape_constraints(ps, boxlen, peak_height, scale_mpc, shape_seed):
    sigma0 = ps.moment(0, scale_mpc, boxlen**3)
    sigma1 = ps.moment(1, scale_mpc, boxlen**3)
    sigma2 = ps.moment(2, scale_mpc, boxlen**3)
    gamma = sigma1**2/sigma0/sigma2
        
    np.random.seed(shape_seed)
    curvature = egp.icgen.random_curvature(peak_height, gamma)
    a21, a31 = egp.icgen.random_shape(curvature)
    density_phi = 180*np.random.random()
    density_theta = 180/np.pi*np.arccos(1-np.random.random())
    density_psi = 180*np.random.random()
    shape_constraints = (curvature, a21, a31, density_phi, density_theta, density_psi)
    return shape_constraints

def plot_all_plus_selection(points, peak_pos, selection, radius):
    from mayavi import mlab
    points_plot = mlab.points3d(points[0],points[1],points[2], mode='point', opacity=0.5)
    cluster_plot = mlab.points3d(peak_pos[0], peak_pos[1], peak_pos[2], mode='sphere', color=(1,0,0), scale_factor=radius, opacity=0.3)
    peak_points_plot = mlab.points3d(points[0,selection], points[1,selection], points[2,selection], opacity=0.5, mode='sphere', scale_factor=radius/10., color=(0,1,0))
    mlab.show()

def get_peak_particle_indices(peak_pos, radius, boxlen, gridsize):
    return sphere_grid(peak_pos, radius, boxlen, gridsize)
def sphere_grid(peak_pos, radius, boxlen, gridsize):
    dx = boxlen/gridsize
    Xgrid = np.mgrid[-boxlen/2:boxlen/2:dx, -boxlen/2:boxlen/2:dx, -boxlen/2:boxlen/2:dx]
    
    # determine roll needed to get peak center back to where it should be (note
    # that we 'initially set it' at the cell at index (gridsize/2, gridsize/2, gridsize/2)):
    cell = np.int32(peak_pos/dx) # "containing" cell
    roll = cell - gridsize/2 # gridsize/2 being the 'initial' index we roll from
    
    # difference of roll (= integer) with real original particle position:
    diff = (peak_pos/dx - (cell+0.5)).reshape(3,1,1,1) # reshape for numpy broadcasting
    Xgrid -= diff*dx
    
    # (to be rolled) distance function (squared!):
    r2grid = np.sum(Xgrid**2, axis=0)
    # roll it:
    r2grid = np.roll(r2grid, roll[0], axis=0) # just roll, above no longer holds
    r2grid = np.roll(r2grid, roll[1], axis=1)
    r2grid = np.roll(r2grid, roll[2], axis=2)
    
    spheregrid = r2grid < radius**2
    return spheregrid


def mass_to_peak_height(mass, cosmo):
    # Later gaan we hier een empirische fit functie in stoppen, gebaseerd op de
    # test uitkomsten.
    mass = mass*1e14 # mass given in 1e14 M_sol
    rhoc = egp.toolbox.critical_density(cosmo)
    scale_mpc = (mass/rhoc/(2*np.pi)**(3./2))**(1./3) # Mpc h^-1
    # using the volume of a gaussian window function, (2*pi)**(3./2) * R**3
    height = mass/(2*np.pi)**(3./2)/scale_mpc**3/rhoc
    return height

def mass_to_scale(mass, cosmo):
    # Later gaan we hier een empirische fit functie in stoppen, gebaseerd op de
    # test uitkomsten.
    mass = mass*1e14 # mass given in 1e14 M_sol
    rhoc = egp.toolbox.critical_density(cosmo)
    scale_mpc = (mass/rhoc/(2*np.pi)**(3./2))**(1./3) # Mpc h^-1
    # using the volume of a gaussian window function, (2*pi)**(3./2) * R**3
    return scale_mpc

