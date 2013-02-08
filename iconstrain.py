#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
iconstrain.py
Iterative constrainer for peaks in ICs of cosmological N-body simulations.

Created by Evert Gerardus Patrick Bos.
Copyright (c) 2012. All rights reserved.
"""

# imports
import numpy as np, scipy.optimize
import egp.icgen, egp.toolbox, egp.io
import glob, os, subprocess, shutil
import FuncDesigner, openopt

# Decide which one to use!
from scipy.optimize import fmin_l_bfgs_b as solve, anneal, brute

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
def iterate_solve(iteration, peak_pos_initial, target_pos, height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo, shape_constraints = [], epsilon=1e-13, factr=1e11, pgtol=1e-3, method="hybr", method_type="root"):
    bound_range = 0.1*boxlen
    boundaries = ((target_pos[0]-bound_range, target_pos[0]+bound_range), (target_pos[1]-bound_range, target_pos[1]+bound_range), (target_pos[2]-bound_range, target_pos[2]+bound_range))
    #~ result = solve(difference, peak_pos_initial, args=(target_pos, height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo, iteration, shape_constraints), bounds = boundaries, approx_grad=True, epsilon=epsilon, factr=factr, pgtol=pgtol, iprint=0)

    print "Iteration method & type: ", method, method_type
    
    options = {'disp': True,\
               'verbose': 5}
    tol = None

    if method_type == "minimize":
        if method == "Brute":
            result = scipy.optimize.brute(difference, boundaries, args=(target_pos, height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo, iteration, shape_constraints))
        else:
            result = scipy.optimize.minimize(difference, peak_pos_initial, args=(target_pos, height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo, iteration, shape_constraints), method = method, tol = tol, options = options, bounds = boundaries)
    elif method_type == "root":
        result = scipy.optimize.root(difference_root, peak_pos_initial, args=(target_pos, height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo, iteration, shape_constraints), method = method, tol = tol, options = options)

    return result

def iterate_zeldovich(peak_pos_initial, target_pos, height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo, shape_constraints = [], epsilon=1e-13, factr=1e11, pgtol=1e-3):
    return iterate_solve(iteration_mean_zeldovich, peak_pos_initial, target_pos, height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo, shape_constraints, epsilon, factr, pgtol)

def iterate_2LPT(peak_pos_initial, target_pos, height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo, shape_constraints = [], epsilon=1e-13, factr=1e11, pgtol=1e-3):
    return iterate_solve(iteration_mean_2LPT, peak_pos_initial, target_pos, height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo, shape_constraints, epsilon, factr, pgtol)

def iterate_PM(peak_pos_initial, target_pos, height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo, shape_constraints = [], epsilon=1e-6, factr=1e11, pgtol=1e-3):
    return iterate_solve(iteration_mean_PM, peak_pos_initial, target_pos, height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo, shape_constraints, epsilon, factr, pgtol)

def iterate_P3M(peak_pos_initial, target_pos, height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo, shape_constraints = [], epsilon=1e-6, factr=1e11, pgtol=1e-3):
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
    result = results_all['x']
    
    return result

# Helper functions:
def difference(peak_pos_input, target_evolved_pos, height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo, iteration_mean, shape_constraints = [], verbose = 1):
    N_peaks = len(peak_pos_input)/3
    if verbose > 1: print "input:\n", peak_pos_input.reshape(N_peaks,3)#, "i.e.", peak_pos_input%boxlen, "in the box"
    evolved_peak_pos = iteration_mean(peak_pos_input%boxlen, height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo, shape_constraints)
    if verbose > 1: print "geeft:\n", evolved_peak_pos.reshape(N_peaks,3)
    if verbose > 1: print "diff :", np.sqrt(np.sum(((evolved_peak_pos - target_evolved_pos)**2).reshape(N_peaks,3), axis=1)), "\n"
    if verbose == 1:
        printarray = np.sqrt(np.sum(((evolved_peak_pos - target_evolved_pos)**2).reshape(N_peaks,3), axis=1))
        printstring = ""
        for i in range(len(printarray)-1):
            printstring += "%f, " % printarray[i]
        printstring += "%f" % printarray[-1]
        print printstring
    return np.sum((evolved_peak_pos - target_evolved_pos)**2)

def difference_root(peak_pos_input, target_evolved_pos, height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo, iteration_mean, shape_constraints = [], verbose = 1):
    N_peaks = len(peak_pos_input)/3
    if verbose > 1: print "input:\n", peak_pos_input.reshape(N_peaks,3)#, "i.e.", peak_pos_input%boxlen, "in the box"
    evolved_peak_pos = iteration_mean(peak_pos_input%boxlen, height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo, shape_constraints)
    if verbose > 1: print "geeft:\n", evolved_peak_pos.reshape(N_peaks,3)
    if verbose > 1: print "diff :", np.sqrt(np.sum(((evolved_peak_pos - target_evolved_pos)**2).reshape(N_peaks,3), axis=1)), "\n"
    if verbose == 1:
        printarray = np.sqrt(np.sum(((evolved_peak_pos - target_evolved_pos)**2).reshape(N_peaks,3), axis=1))
        printstring = ""
        for i in range(len(printarray)-1):
            printstring += "%f, " % printarray[i]
        printstring += "%f" % printarray[-1]
        print printstring
    return evolved_peak_pos - target_evolved_pos


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
    from FuncDesigner import *

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
    mass = mass*1.e14 # mass given in 1e14 M_sol
    rhoc = egp.toolbox.critical_density(cosmo)
    scale_mpc = (mass/rhoc/(2*np.pi)**(3./2))**(1./3) # Mpc h^-1
    # using the volume of a gaussian window function, (2*pi)**(3./2) * R**3
    height = mass/(2*np.pi)**(3./2)/scale_mpc**3/rhoc
    return height # :D DIT IS ALTIJD 1! DE SCHAAL BEPAALT ALLES AL BLIJKBAAR!

def mass_to_scale(mass, cosmo):
    # Later gaan we hier een empirische fit functie in stoppen, gebaseerd op de
    # test uitkomsten.
    mass = mass*1e14 # mass given in 1e14 M_sol
    rhoc = egp.toolbox.critical_density(cosmo)
    scale_mpc = (mass/rhoc/(2*np.pi)**(3./2))**(1./3) # Mpc h^-1
    # using the volume of a gaussian window function, (2*pi)**(3./2) * R**3
    return scale_mpc


def setup_gadget_run(cosmo, ps, boxlen, gridsize, seed, peak_pos, peak_height, scale_mpc, redshift, test_id, run_dir_base, nproc, constrain_shape=True, shape_seed=0, run_name = None, location='kapteyn', save_dir = None, gadget_executable = "/net/schmidt/data/users/pbos/sw/code/gadget/gadget3Sub_512_SL6/P-Gadget3_512", time_limit_cpu = 86400):    
    # shape / orientation constraints:
    if constrain_shape:
        shape_constraints = set_shape_constraints(ps, boxlen, peak_height, scale_mpc, shape_seed)
    else:
        shape_constraints = []
    
    if not run_name:
        run_name = "run%s_%i" % (test_id, seed)
    
    ic_file = run_dir_base+"/ICs/ic_%iMpc_%i_%s.dat" % (boxlen, gridsize, run_name)
    if not save_dir:
        save_dir = run_dir_base
        ic_file_save = ic_file
    else:
        ic_file_save = "/Users/users/pbos/dataserver/sims/ICs/ic_%iMpc_%i_%s.dat" % (boxlen, gridsize, run_name)

    rhoc = egp.toolbox.critical_density(cosmo) # M_sun Mpc^-3 h^2
    particle_mass = cosmo.omegaM * rhoc * boxlen**3 / gridsize**3 / 1e10 # 10^10 M_sun h^-1
    
    deltaU = egp.icgen.GaussianRandomField(ps, boxlen, gridsize, seed=seed)
    
    print "Building %s..." % run_name
    deltaC = constrain_field(peak_pos, peak_height, scale_mpc, boxlen, deltaU, ps, cosmo, shape_constraints)
    psiC = egp.icgen.DisplacementField(deltaC)
    del deltaC
    pos, vel = egp.icgen.zeldovich(redshift, psiC, cosmo) # Mpc, not h^-1!
    del psiC
    print "Saving %s..." % run_name
    egp.io.write_gadget_ic_dm(ic_file_save, pos.reshape((3,gridsize**3)).T, vel.reshape((3,gridsize**3)).T, particle_mass, redshift, boxlen, cosmo.omegaM, cosmo.omegaL, cosmo.h)
    del pos, vel
    
    print "Preparing for gadget run %(run_name)s..." % locals()
    run_instructions = egp.io.prepare_gadget_run(boxlen, gridsize, cosmo, ic_file, redshift, run_dir_base, run_name, nproc, run_location=location, save_dir = save_dir, gadget_executable=gadget_executable, time_limit_cpu = time_limit_cpu)
    
    log_file = open(save_dir+'/'+run_name+'.log', 'w')
    log_file.write('peak position:\n')
    log_file.write(peak_pos.__str__())
    log_file.close()
    
    print "Done!"
    return run_instructions


# ANALYSE

def analyse_run(run_dir_base, run_name, target_pos, target_mass, boxlen, gridsize, radius, snapshot_number = '008'):
    z0 = egp.io.GadgetData(run_dir_base+'/'+run_name+'/snap_'+snapshot_number)
    with open(run_dir_base+'/'+run_name+'.log', 'r') as log_file:
        log_file.readline()
        pos_iter = np.array(log_file.readline().replace('[',"").replace(']',"").split(), dtype='float32')
    sphere = sphere_grid(pos_iter, radius, boxlen, gridsize).reshape(z0.pos.shape[0])
    
    pos_gadget = z0.pos[sphere].mean(axis=0)/1000.
    dpos = np.sqrt(np.sum((pos_gadget - target_pos)**2))
    
    # Subhalo analysis
    haloes_z0 = egp.io.SubFindHaloes(run_dir_base+'/'+run_name+'/groups_'+snapshot_number+'/subhalo_tab_'+snapshot_number+'.0')
    # identify the SubFind halo that is most probably the one we constrained (closest in space)
    halo_index = np.sqrt(np.sum((haloes_z0.SubPos/1000 - pos_gadget)**2, axis=1)).argmin()
    halo_mass = haloes_z0.SubTMass[halo_index]
    dpos_halo = np.sqrt(np.sum((target_pos - haloes_z0.SubPos[halo_index]/1000)**2))
    
    # Top-level parent subhalo analysis
    parent_index = np.sqrt(np.sum((haloes_z0.parent_pos/1000 - pos_gadget)**2, axis=1)).argmin()
    parent_mass = haloes_z0.parent_mass[parent_index]
    dpos_parent = np.sqrt(np.sum((target_pos - haloes_z0.parent_pos[parent_index]/1000)**2))
    parent_children = len(haloes_z0.parent_children[parent_index])
    
    # Surrounding haloes analysis
    halo_distance = np.sqrt(np.sum((haloes_z0.SubPos/1000 - target_pos)**2, axis=1))
    surrounder_filter = halo_distance < radius
    surrounder_number = np.sum(surrounder_filter) # printen
    if surrounder_number > 0:
        surrounder_relative_pos = haloes_z0.SubPos[surrounder_filter]/1000 - target_pos
        surrounder_masses = haloes_z0.SubTMass[surrounder_filter]
        surrounder_mass = np.sum(surrounder_masses) # printen
        surrounder_cm = np.sum(surrounder_masses[:,None]*surrounder_relative_pos, axis=0)/surrounder_mass
        #~ surrounder_relative_pos_mean = np.mean(surrounder_relative_pos, axis=0)
        surrounder_pos_std = np.std(surrounder_relative_pos, axis=0)
        surrounder_dpos = np.sqrt(np.sum((surrounder_cm)**2)) # printen
        surrounder_dpos_std_amplitude = np.sqrt(np.sum(surrounder_pos_std**2)) # printen
        
        # Most massive surrounding subhalo analysis
        massive_surrounder_index = surrounder_masses.argmax()
        massive_surrounder_mass = surrounder_masses[massive_surrounder_index]
        massive_surrounder_dpos = np.sqrt(np.sum((target_pos - haloes_z0.SubPos[surrounder_filter][massive_surrounder_index]/1000)**2))
    
        # Surrounding subhalo with mass closest to target_mass analysis
        closest_mass_index = np.abs(surrounder_masses - target_mass).argmin()
        closest_mass_mass = surrounder_masses[closest_mass_index]
        closest_mass_dpos = np.sqrt(np.sum((target_pos - haloes_z0.SubPos[surrounder_filter][closest_mass_index]/1000)**2))
    else:
        surrounder_dpos = -1
        surrounder_dpos_std_amplitude = -1
        surrounder_mass = -1
        massive_surrounder_dpos = -1
        massive_surrounder_mass = -1
        closest_mass_dpos = -1
        closest_mass_mass = -1
    
    #~ # Density analysis
    #~ den = egp.toolbox.TSC_density(z0.pos, gridsize, boxlen, 1.)
    #~ den_f = np.fft.rfftn(den)
    #~ # smooth on 8 Mpc scale:
    #~ sden = np.fft.irfftn(egp.toolbox.gaussian_smooth(den_f, 8., boxlen))
    #~ sden_max_i = np.array(np.unravel_index(sden.argmax(), sden.shape))
    #~ sden_max_pos = sden_max_i*boxlen/gridsize
    #~ dpos_den = np.sqrt(np.sum((target_pos-sden_max_pos)**2))
    
    # Present results
    presentation = "Distance of particles in Gadget to target position:      %f\n\
Distance of closest subhalo to target position:          %f\n\
Mass of this halo:                                       %f\n\
Distance of closest parent subhalo to target position:   %f\n\
Mass of this parent halo:                                %f\n\
Number of children of parent halo:                       %i\n\
Number of surrounding subhaloes within peak radius:             %i\n\
Distance of center of mass of surrounding subhaloes to target:  %f\n\
'Standard deviation' (spread) of surrounding subhalo positions: %f\n\
Total mass of surrounding subhaloes:                            %f\n\
Distance of most massive surrounding subhalo to target position:   %f\n\
Mass of most massive surrounding subhalo:                          %f\n\
Distance to target position of surrounding subhalo closest to target mass: %f\n\
Mass of this subhalo:                                                      %f" % (dpos, dpos_halo, halo_mass, dpos_parent, parent_mass, parent_children, surrounder_number, surrounder_dpos, surrounder_dpos_std_amplitude, surrounder_mass, massive_surrounder_dpos, massive_surrounder_mass, closest_mass_dpos, closest_mass_mass)
    #~ print "Distance of density maximum to target position: ", dpos_den
    return presentation, dpos, dpos_halo, halo_mass, dpos_parent, parent_mass, parent_children, surrounder_number, surrounder_dpos, surrounder_dpos_std_amplitude, surrounder_mass, massive_surrounder_dpos, massive_surrounder_mass, closest_mass_dpos, closest_mass_mass




###### MULTIPLE PEAKS STUFF (later herordenen) ######
# positie-arrays moeten flat zijn; ipc met shape (N,3) maar dan .flatten()'ed.
    
def iterate_mirror_zeldovich_multi(peak_pos_initial, target_pos, height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo, shape_constraints = [], epsilon=None, factr=None, pgtol=None, method=None):
    return iterate_mirror(iteration_mean_zeldovich_multi, peak_pos_initial, target_pos, height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo, shape_constraints)

def iterate_peakwise_shift_multi(iteration, peak_pos_initial, target_pos, height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo, shape_constraints = [], epsilon=None, factr=None, pgtol=None, method=None, verbose=2, alpha = 2./9, beta = 3./4, gamma = 2./9):
    """
    An evolution of the mirror algorithm (which only works for one step)
    to multiple steps. Does the same thing, but iteratively refines the
    result and adds some failsafes against unstable situations.
    
    The alpha parameter is used in the failsafe of when the next step causes
    the evolved position to be further from the target than the previous
    one. Another initial position is then tried, which is alpha parts the
    newest one and (1-alpha) parts the old one (a weighted average of both
    that is).
    
    The beta parameter has similar usage to alpha, but it is used every step
    to soften the major steps, reducing overshoot. Beta parts the newest step
    and (1-beta) parts the old one.
    
    Gamma is used when the direction of the initial peak position needs to be
    flipped because it was moving in the wrong direction in that component;
    the new initial peak position component will be the mirror image of the
    one at the beginning of the iteration step, but multiplied by gamma,
    mirrored in the initial peak position of the previous step.
    """
    N_peaks = len(target_pos)/3
    N_calculated_constrained_fields = 0
    N_refinements = 0
    N_outer_loop_steps = 0
    N_sign_flips = 0
    # Set peak_pos_evolved_previous to large value, so that first step
    # failsafes will be omitted:
    peak_pos_evolved_previous = target_pos + boxlen
    peak_pos_initial_previous = peak_pos_initial
    for i in range(10): # HIER GOEIE CONDITIES DOEN!
        if verbose: print "\nStep ", i
        N_outer_loop_steps += 1
        if verbose >= 2: print "Initial peak positions:\n", peak_pos_initial.reshape(N_peaks, 3)
        peak_pos_evolved = iteration(peak_pos_initial%boxlen, height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo, shape_constraints)
        N_calculated_constrained_fields += 1
        if verbose >= 2: print "Evolved peak positions:\n", peak_pos_evolved.reshape(N_peaks, 3)
        difference = (peak_pos_evolved - target_pos).reshape(N_peaks, 3)
        if verbose: print "Difference with target:\n", difference
        distance = np.sqrt(((peak_pos_evolved - target_pos)**2).reshape(N_peaks, 3).sum(axis=1))
        if verbose: print "Distance from target: ", distance
        
        difference_previous = (peak_pos_evolved_previous - target_pos).reshape(N_peaks, 3)
        
        distance_previous = np.sqrt(((target_pos - peak_pos_evolved_previous)**2).reshape(N_peaks,3).sum(axis=1))
        # Check for growth of absolute distance from a target.
        # If grown, reduce that part of the 
        while np.any(distance > distance_previous):
            # Check whether difference from target is in same direction as previous step *and* larger.
            # If so, change direction of that part of peak_pos_initial.
            if np.any((np.abs(difference) > np.abs(difference_previous))*(np.sign(difference) == np.sign(difference_previous))):
                N_sign_flips += 1
                change_initial_direction = (np.abs(difference) > np.abs(difference_previous))*(np.sign(difference) == np.sign(difference_previous))
                if verbose: print "Flipping signs... (", change_initial_direction, ")"
                np.atleast_2d(peak_pos_initial.reshape(N_peaks, 3))[change_initial_direction] = (1+gamma)*np.atleast_2d(peak_pos_initial_previous.reshape(N_peaks, 3))[change_initial_direction] - gamma*np.atleast_2d(peak_pos_initial.reshape(N_peaks, 3))[change_initial_direction] # mirror peak_pos_initial 
                if verbose >= 2: print "Flipped initial peak positions:\n", peak_pos_initial.reshape(N_peaks, 3)
                peak_pos_evolved = iteration(peak_pos_initial%boxlen, height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo, shape_constraints)
                N_calculated_constrained_fields += 1
                if verbose >= 2: print "Flipped evolved peak positions:\n", peak_pos_evolved.reshape(N_peaks, 3)
                difference = (peak_pos_evolved - target_pos).reshape(N_peaks, 3)
                if verbose: print "Difference with target:\n", difference
                distance = np.sqrt(((peak_pos_evolved - target_pos)**2).reshape(N_peaks, 3).sum(axis=1))
                if verbose: print "Distance from target: ", distance
            else:
                N_refinements += 1
                adjust_initial_peak = ((target_pos - peak_pos_evolved)**2).reshape(N_peaks,3).sum(axis=1) > ((target_pos - peak_pos_evolved_previous)**2).reshape(N_peaks,3).sum(axis=1)
                if verbose: print "Refining... (", adjust_initial_peak, ")"
                peak_pos_initial.reshape(N_peaks, 3)[adjust_initial_peak] = (1-alpha)*(peak_pos_initial_previous.reshape(N_peaks, 3)[adjust_initial_peak]) + alpha*peak_pos_initial.reshape(N_peaks, 3)[adjust_initial_peak]
                if verbose >= 2: print "Refined initial peak positions:\n", peak_pos_initial.reshape(N_peaks, 3)
                peak_pos_evolved = iteration(peak_pos_initial%boxlen, height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo, shape_constraints)
                N_calculated_constrained_fields += 1
                if verbose >= 2: print "Refined evolved peak positions:\n", peak_pos_evolved.reshape(N_peaks, 3)
                if verbose: print "Difference from target:\n", (peak_pos_evolved - target_pos).reshape(N_peaks, 3)
                if verbose: print "Distance from target: ", np.sqrt(((peak_pos_evolved - target_pos)**2).reshape(N_peaks, 3).sum(axis=1))
        difference = (peak_pos_evolved - target_pos).reshape(N_peaks, 3)
        peak_pos_initial_previous = peak_pos_initial
        peak_pos_evolved_previous = peak_pos_evolved
        difference_weight = np.abs(difference)/(np.abs(difference)+np.abs(difference_previous))
        difference_weight = difference_weight.flatten()
        
        peak_pos_initial = (1-difference_weight)*(target_pos - (peak_pos_evolved - peak_pos_initial)) + difference_weight*peak_pos_initial_previous # new initial
    result = [peak_pos_initial, (N_calculated_constrained_fields, N_refinements, N_outer_loop_steps)]
    return result

def iterate_peakwise_shift_zeldovich_multi(peak_pos_initial, target_pos, height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo, shape_constraints = [], epsilon=None, factr=None, pgtol=None, method=None):
    return iterate_peakwise_shift_multi(iteration_mean_zeldovich_multi, peak_pos_initial, target_pos, height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo, shape_constraints)


def iterate_solve_multi(iteration, peak_pos_initial, target_pos, height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo, shape_constraints = [], epsilon=1e-13, factr=1e11, pgtol=1e-3, method="Nelder-Mead", method_type = "minimize"):
    N_peaks = len(target_pos)/3
    bound_range = 0.1*boxlen
    boundaries = np.array([target_pos - bound_range, target_pos + bound_range]).T.tolist()
        
    print "Iteration method & type: ", method, method_type
    
    tol = None
    options = {'disp': True,\
               'verbose': 5,\
               'jac_options': {'alpha': 0.1}\
               }
    
    if method_type == "minimize":
        if method == "Brute":
            result = scipy.optimize.brute(difference, boundaries, args=(target_pos, height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo, iteration, shape_constraints))
        else:
            result = scipy.optimize.minimize(difference, peak_pos_initial, args=(target_pos, height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo, iteration, shape_constraints), method = method, tol = tol, options = options, bounds = boundaries)
    elif method_type == "root":
        result = scipy.optimize.root(difference_root, peak_pos_initial, args=(target_pos, height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo, iteration, shape_constraints), method = method, tol = tol, options = options)
    return result


def iterate_solve_multi_openopt(iteration, peak_pos_initial, target_pos, height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo, shape_constraints = [], epsilon=1e-13, factr=1e11, pgtol=1e-3, method="L-BFGS-B", method_type = "minimize"):
    N_peaks = len(target_pos)/3
    bound_range = 0.1*boxlen
    boundaries = np.array([target_pos - bound_range, target_pos + bound_range]).T.tolist()
    
    #~ target_pos_oo = FuncDesigner.oovar(size=N_peaks*3)('target_pos')
    
    peak_pos_input_oo = FuncDesigner.oovar(size=N_peaks*3)('target_pos')
    #peak_pos_input_oo /= boxlen
    #raise SystemExit
    peak_pos_evolved_oo = FuncDesigner.oofun(iteration, input=[peak_pos_input_oo], args=(height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo, shape_constraints))
    f_oo = peak_pos_evolved_oo - target_pos
    
    #~ point = {peak_pos_input_oo: target_pos}
    #~ print f_oo.D(point)
    #~ raise SystemExit
    start_oo = {peak_pos_input_oo: target_pos}
    
    problem = openopt.NLP(f_oo, start_oo)
    
    problem.constraints = []
    
    result = problem.solve('ralg')
    
    print result
    
    return result
    

class GoalReachedException(Exception):
    def __init__(self, result_peak_pos_input):
        self.result_peak_pos_input = result_peak_pos_input

class CustomResult(dict):
    pass

def iterate_solve_multi_fixed(iteration, peak_pos_initial, target_pos, height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo, shape_constraints = [], epsilon=1e-13, factr=1e11, pgtol=1e-3, method="L-BFGS-B", method_type = "minimize", goal_proximity = 0.1):
    """This multi-peak solver also allows for a few peaks to be fixed
    once they reach a proximity of /goal_proximity/ Mpc h^-1, while the
    solver keeps trying to solve for the other peaks."""
    N_peaks = len(target_pos)/3
    bound_range = 0.1*boxlen
    boundaries = np.array([target_pos - bound_range, target_pos + bound_range]).T.tolist()
    
    print "Iteration method & type: ", method, method_type
    
    tol = None
    options = {'disp': True,\
               'verbose': 5}
               #~ 'line_search': 'wolfe'}
#               'jac_options': {'alpha': 0.1}\
               #~ }
    
    def callback(x, Fx, goal_proximity=goal_proximity):
        return check_peaks_for_convergence(x, Fx, goal_proximity)
    
    try:
        if method_type == "minimize":
            if method == "Brute":
                result = scipy.optimize.brute(difference, boundaries, args=(target_pos, height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo, iteration, shape_constraints))
            else:
                result = scipy.optimize.minimize(difference, peak_pos_initial, args=(target_pos, height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo, iteration, shape_constraints), method = method, tol = tol, options = options, callback = callback, bounds = boundaries)
        elif method_type == "root":
            result = scipy.optimize.root(difference_root, peak_pos_initial, args=(target_pos, height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo, iteration, shape_constraints), method = method, tol = tol, options = options, callback = callback)
    except GoalReachedException, e:
        result = CustomResult()
        result['x'] = e.result_peak_pos_input
        result['success'] = True
        result['message'] = "Goal proximity reached, aborting solving routine."
        if options['disp'] or (options['verbose'] > 1):
            print result['message']

    return result

def check_peaks_for_convergence(peak_pos_initial, difference, goal_proximity):
    N_peaks = len(peak_pos_initial)/3
    dist = np.sqrt(np.sum(difference.reshape(N_peaks,3)**2, axis=1))
    goal_reached = dist < goal_proximity
    if np.all(goal_reached):
        raise GoalReachedException(peak_pos_initial)
    else:
        print "goal proximity not yet reached..."
        print "x:  ", peak_pos_initial
        print "Fx: ", difference

def iterate_zeldovich_multi(peak_pos_initial, target_pos, height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo, shape_constraints = [], epsilon=1e-13, factr=1e11, pgtol=1e-3, method="Nelder-Mead", method_type = "minimize"):
    return iterate_solve_multi_fixed(iteration_mean_zeldovich_multi, peak_pos_initial, target_pos, height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo, shape_constraints, epsilon, factr, pgtol, method = method, method_type = method_type)

def iteration_mean_zeldovich_multi(peak_pos, height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo, shape_constraints = []):
    N_peaks = len(peak_pos)/3
    
    deltaC = constrain_field_multi(peak_pos, height, scale_mpc, boxlen, deltaU, ps, cosmo, shape_constraints)
    
    # Now, Zel'dovich it:
    psiC = egp.icgen.DisplacementField(deltaC)
    x, v = egp.icgen.zeldovich(0., psiC, cosmo) # Mpc, not h^-1!
    del v
    
    # Calculate the "new position" of the peak:
    mean_evolved_peak_pos = np.zeros_like(peak_pos)
    # ... for which we determine peak particle indices:
    for i in range(N_peaks):
        radius = scale_mpc[i]
        spheregrid = get_peak_particle_indices(peak_pos[3*i:3*i+3], radius, boxlen, gridsize)
        mean_evolved_peak_pos[3*i:3*i+3] = x[:,spheregrid].mean(axis=1)
    
    return mean_evolved_peak_pos

def constrain_field_multi(peak_pos, height, scale_mpc, boxlen, deltaU, ps, cosmo, shape_constraints = []):
    N_peaks = len(peak_pos)/3
    constraints = []

    for i in range(N_peaks):
        location = egp.icgen.ConstraintLocation(peak_pos[3*i:3*i+3])
        scale = egp.icgen.ConstraintScale(scale_mpc[i])
        constraints.append(egp.icgen.HeightConstraint(location, scale, height[i]))
        # make it a real peak:
        constraints.append(egp.icgen.ExtremumConstraint(location, scale, 0))
        constraints.append(egp.icgen.ExtremumConstraint(location, scale, 1))
        constraints.append(egp.icgen.ExtremumConstraint(location, scale, 2))
        # apply shape constraints to location & scale:
        if shape_constraints:
            constraints += egp.icgen.generate_shape_constraints(location, scale, ps, boxlen, *shape_constraints[i])
    
    # Do the field stuff!
    deltaC = egp.icgen.ConstrainedField(deltaU, constraints) # N.B.: deltaU stays the same!!!
    return deltaC

def run_multi(cosmo, ps, boxlen, gridsize, deltaU, target_pos, peak_height, scale_mpc, iterate, initial_guess = iterate_mirror_zeldovich_multi, constrain_shape=True, shape_seed=0, epsilon=1e-13, factr=1e11, pgtol=1e-3, method="L-BFGS-B", method_type = "minimize", iteration = None):
    """
    Call with /iterate/ one of the iterate_## functions. /initial_guess/ can be
    either an ndarray with an initial guess or a function that computes an
    initial guess (accepting the same arguments as /iterate/, except for the
    first one of course, which is the initial position). By default
    iterate_mirror_zeldovich is used as initial_guess function.
    
    /target_pos/, /peak_height/ and /scale_mpc/ must be arrays with
    length equal to the number of peaks.
    
    Note that the seed for the shape constraints will be /shape_seed/+i
    where i is the index of the peak.
    """
    N_peaks = len(target_pos)/3
    # shape / orientation constraints:
    shape_constraints = []
    if constrain_shape:
        for i in range(N_peaks):
            shape_constraints.append(set_shape_constraints(ps, boxlen, peak_height[i], scale_mpc[i], shape_seed+i))
    else:
        shape_constraints = []
    
    if type(initial_guess) is np.ndarray:
        pos_initial = initial_guess
    else:
        pos_initial = initial_guess(target_pos, target_pos, peak_height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo, shape_constraints, epsilon=epsilon, factr=factr, pgtol=pgtol)
    
    #~ print pos_initial
    if not iteration:
        results_all = iterate(pos_initial, target_pos, peak_height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo, shape_constraints, epsilon=epsilon, factr=factr, pgtol=pgtol, method=method, method_type = method_type)
    else:
        results_all = iterate(iteration, pos_initial, target_pos, peak_height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo, shape_constraints, epsilon=epsilon, factr=factr, pgtol=pgtol, method=method, method_type = method_type)
        results_all = iterate(iteration, pos_initial, target_pos, peak_height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo, shape_constraints, epsilon=epsilon, factr=factr, pgtol=pgtol, method=method, method_type = method_type)
    
    if method == "Brute":
        result = results_all[0]
    else:
        result = results_all['x']
        print "Minimizer success:", results_all['success']
        print "Minimizer message:\n", results_all['message']
    
    return result

def setup_gadget_run_multi(cosmo, ps, boxlen, gridsize, seed, peak_pos, peak_height, scale_mpc, redshift, test_id, run_dir_base, nproc, constrain_shape=True, shape_seed=0, run_name = None, location='kapteyn', save_dir = None, gadget_executable = "/net/schmidt/data/users/pbos/sw/code/gadget/gadget3Sub_512_SL6/P-Gadget3_512", time_limit_cpu = 86400):
    
    N_peaks = len(peak_pos)/3
    # shape / orientation constraints:
    shape_constraints = []
    if constrain_shape:
        for i in range(N_peaks):
            shape_constraints.append(set_shape_constraints(ps, boxlen, peak_height[i], scale_mpc[i], shape_seed+i))
    else:
        shape_constraints = []
    
    if not run_name:
        run_name = "run_%s_%i" % (test_id, seed)
    
    ic_file = run_dir_base+"/ICs/ic_%iMpc_%i_%s.dat" % (boxlen, gridsize, run_name)
    if not save_dir:
        save_dir = run_dir_base
        ic_file_save = ic_file
    else:
        ic_file_save = "/Users/users/pbos/dataserver/sims/ICs/ic_%iMpc_%i_%s.dat" % (boxlen, gridsize, run_name)

    rhoc = egp.toolbox.critical_density(cosmo) # M_sun Mpc^-3 h^2
    particle_mass = cosmo.omegaM * rhoc * boxlen**3 / gridsize**3 / 1e10 # 10^10 M_sun h^-1
    
    deltaU = egp.icgen.GaussianRandomField(ps, boxlen, gridsize, seed=seed)
    
    print "Building %s..." % run_name
    deltaC = constrain_field_multi(peak_pos, peak_height, scale_mpc, boxlen, deltaU, ps, cosmo, shape_constraints)
    psiC = egp.icgen.DisplacementField(deltaC)
    del deltaC
    pos, vel = egp.icgen.zeldovich(redshift, psiC, cosmo) # Mpc, not h^-1!
    del psiC
    print "Saving %s..." % run_name
    egp.io.write_gadget_ic_dm(ic_file_save, pos.reshape((3,gridsize**3)).T, vel.reshape((3,gridsize**3)).T, particle_mass, redshift, boxlen, cosmo.omegaM, cosmo.omegaL, cosmo.h)
    del pos, vel
    
    print "Preparing for gadget run %(run_name)s..." % locals()
    run_instructions = egp.io.prepare_gadget_run(boxlen, gridsize, cosmo, ic_file, redshift, run_dir_base, run_name, nproc, run_location=location, save_dir = save_dir, gadget_executable=gadget_executable, time_limit_cpu = time_limit_cpu)
    
    log_file = open(save_dir+'/'+run_name+'.log', 'w')
    log_file.write('peak positions:\n')
    log_file.write(peak_pos.__str__())
    log_file.close()
    
    print "Done!"
    return run_instructions
