#!/usr/bin/env python
# -*- coding: utf-8 -*-

# MULTIPLE PEAKS
def iterate_mirror_multi(iteration, peak_pos_initial, target_pos, height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo, shape_constraints = [], epsilon=None, factr=None, pgtol=None):
    peak_pos_evolved = iteration(peak_pos_initial%boxlen, height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo, shape_constraints)
    result = 2*target_pos - peak_pos_evolved # = target_pos + (target_pos - peak_pos_evolved), mirror evolved pos in target pos
    return result
    
def iterate_mirror_zeldovich_multi(peak_pos_initial, target_pos, height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo, shape_constraints = [], epsilon=None, factr=None, pgtol=None):
    return iterate_mirror(iteration_mean_zeldovich_multi, peak_pos_initial, target_pos, height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo, shape_constraints)

def iterate_solve_multi(iteration, peak_pos_initial, target_pos, height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo, shape_constraints = [], epsilon=1e-13, factr=1e11, pgtol=1e-3):
    N_peaks = len(target_pos)
    bound_range = 0.1*boxlen
    boundaries = []
    for i in range(N_peaks):
        boundaries.append(((target_pos[0]-bound_range, target_pos[0]+bound_range), (target_pos[1]-bound_range, target_pos[1]+bound_range), (target_pos[2]-bound_range, target_pos[2]+bound_range)))
    #~ lower = np.array(boundaries)[:,0]
    #~ upper = np.array(boundaries)[:,1]
    result = solve(difference, peak_pos_initial, args=(target_pos, height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo, iteration, shape_constraints), bounds = boundaries, approx_grad=True, epsilon=epsilon, factr=factr, pgtol=pgtol, iprint=0)
    #~ result = anneal(difference, peak_pos_initial, args=(target_pos, height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo))
    #~ result = brute(difference, boundaries, args=(target_pos, height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo))
    #~ result = fmin_powell(difference, peak_pos_initial, args = (target_pos, height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo))
    return result

def iterate_zeldovich_multi(peak_pos_initial, target_pos, height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo, shape_constraints = [], epsilon=1e-13, factr=1e11, pgtol=1e-3):
    return iterate_solve_multi(iteration_mean_zeldovich_multi, peak_pos_initial, target_pos, height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo, shape_constraints, epsilon, factr, pgtol)

def iteration_mean_zeldovich_multi(peak_pos, height, scale_mpc, boxlen, gridsize, deltaU, ps, cosmo, shape_constraints = []):
    N_peaks = len(peak_pos)
    
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
        spheregrid = get_peak_particle_indices(peak_pos[i], radius, boxlen, gridsize)
        mean_evolved_peak_pos[i] = x[:,spheregrid].mean(axis=1)
    
    return mean_evolved_peak_pos

def constrain_field_multi(peak_pos, height, scale_mpc, boxlen, deltaU, ps, cosmo, shape_constraints = []):
    N_peaks = len(peak_pos)
    constraints = []

    for i in range(N_peaks):
        location = egp.icgen.ConstraintLocation(peak_pos[i])
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

def run_multi(cosmo, ps, boxlen, gridsize, deltaU, target_pos, peak_height, scale_mpc, iterate, initial_guess = iterate_mirror_zeldovich_multi, constrain_shape=True, shape_seed=0):
    """
    Call with /iterate/ one of the iterate_## functions. /initial_guess/ can be
    either an ndarray with an initial guess or a function that computes an
    initial guess (accepting the same arguments as /iterate/, except for the
    first one of course, which is the initial position). By default
    iterate_mirror_zeldovich is used as initial_guess function.
    
    /target_pos/, /peak_height/ and /scale_mpc/ must be arrays with
    length equal to the number of peaks.
    """
    N_peaks = len(target_pos)
    # shape / orientation constraints:
    shape_constraints = []
    if constrain_shape:
        for i in range(N_peaks):
            shape_constraints.append(set_shape_constraints(ps, boxlen, peak_height, scale_mpc, shape_seed))
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

def setup_gadget_run_multi(cosmo, ps, boxlen, gridsize, seed, peak_pos, peak_height, scale_mpc, redshift, test_id, run_dir_base, nproc, constrain_shape=True, shape_seed=0, run_name = None, location='kapteyn', save_dir = None, gadget_executable = "/net/schmidt/data/users/pbos/sw/code/gadget/gadget3Sub_512_SL6/P-Gadget3_512", time_limit_cpu = 86400):
    
    N_peaks = len(target_pos)
    # shape / orientation constraints:
    shape_constraints = []
    if constrain_shape:
        for i in range(N_peaks):
            shape_constraints.append(set_shape_constraints(ps, boxlen, peak_height, scale_mpc, shape_seed))
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
