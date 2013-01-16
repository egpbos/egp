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
import egp.icgen, egp.toolbox, egp.io
import glob, os, subprocess, shutil

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


def setup_gadget_run(cosmo, ps, boxlen, gridsize, seed, peak_pos, peak_height, scale_mpc, redshift, test_id, run_dir_base, nproc, constrain_shape=True, shape_seed=0, run_name = None, location='kapteyn', save_dir = None, gadget_executable = "/net/schmidt/data/users/pbos/sw/code/gadget/gadget3Sub_512_SL6/P-Gadget3_512", time_limit_cpu = 86400):    
    # shape / orientation constraints:
    if constrain_shape:
        shape_constraints = set_shape_constraints(ps, boxlen, peak_height, scale_mpc, shape_seed)
    else:
        shape_constraints = []
    
    if not run_name:
        run_name = "run%s_%i" % (test_id, seed) # plus 100 to separate from DE+void runs
    
    ic_file = run_dir_base+"/ICs/ic_%iMpc_%i_%s.dat" % (boxlen, gridsize, run_name)
    if not save_dir:
        save_dir = run_dir_base
        ic_file_save = ic_file
    else:
        ic_file_save = "/Users/users/pbos/dataserver/sims/ICs/ic_%iMpc_%i_%s.dat" % (boxlen, gridsize, run_name)

    rhoc = egp.toolbox.critical_density(cosmo) # M_sun Mpc^-3 h^2
    particle_mass = cosmo.omegaM * rhoc * boxlen**3 / gridsize**3 / 1e10 # 10^10 M_sun h^-1
    
    rhoU_out = egp.icgen.GaussianRandomField(ps, boxlen, gridsize, seed=seed)
    
    print "Building %s..." % run_name
    irhoC = constrain_field(peak_pos, peak_height, scale_mpc, boxlen, rhoU_out, ps, cosmo, shape_constraints)
    ipsiC = egp.icgen.DisplacementField(irhoC)
    del irhoC
    ipos, ivel = egp.icgen.zeldovich(redshift, ipsiC, cosmo) # Mpc, not h^-1!
    del ipsiC
    print "Saving %s..." % run_name
    egp.io.write_gadget_ic_dm(ic_file_save, ipos.reshape((3,gridsize**3)).T, ivel.reshape((3,gridsize**3)).T, particle_mass, redshift, boxlen, cosmo.omegaM, cosmo.omegaL, cosmo.h)
    del ipos, ivel
    
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
