#!/usr/bin/env python
# encoding: utf-8

import numpy as np
from csv import reader as csvreader
import egp.toolbox, egp.icgen, egp.io
from iconstrain import constrain_field

# iconstrain_tester.py
# Convenience functions for the iconstrain (test) runs.

def test_run(cosmo, ps, boxlen, gridsize_iter, seed, target_pos, peak_height, scale_mpc, initial_guess, iterate, constrain_shape=True, shape_seed=0):    
    # shape / orientation constraints:
    if constrain_shape:
        shape_constraints = set_shape_constraints(ps, boxlen, peak_height, scale_mpc, shape_seed)
    else:
        shape_constraints = []
    
    rhoU = egp.icgen.GaussianRandomField(ps, boxlen, gridsize_iter, seed=seed) # Unconstrained
    
    pos_initial = initial_guess(target_pos, peak_height, scale_mpc, boxlen, gridsize_iter, rhoU, ps, cosmo, shape_constraints)
    results_all = iterate(pos_initial, target_pos, peak_height, scale_mpc, boxlen, gridsize_iter, rhoU, ps, cosmo, shape_constraints)
    result = results_all[0]
    
    return result

def load_catalog(filename):
    table_file = open(filename)
    table = csvreader(table_file)
    table.next() # skip header
    catalog = []
    for entry in table:
        catalog.append(entry)
    table_file.close()
    return catalog

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

def setup_gadget_run(cosmo, ps, boxlen, gridsize, seed, peak_pos, peak_height, scale_mpc, redshift, test_id, run_dir_base, nproc, constrain_shape=True, shape_seed=0, run_name = None):    
    # shape / orientation constraints:
    # if constrain_shape:
    #     shape_constraints = set_shape_constraints(ps, boxlen, peak_height, scale_mpc, shape_seed)
    # else:
    #     shape_constraints = []
    
    if not run_name:
        run_name = "run%i_%i" % ((100+int(test_id)), seed) # plus 100 to separate from DE+void runs
    
    # rhoc = egp.toolbox.critical_density(cosmo) # M_sun Mpc^-3 h^2
    #  particle_mass = cosmo.omegaM * rhoc * boxlen**3 / gridsize**3 / 1e10 # 10^10 M_sun h^-1
    #  
    #  rhoU_out = egp.icgen.GaussianRandomField(ps, boxlen, gridsize, seed=seed)
    #  
    #  print "Building and saving %s..." % run_name
    #  irhoC = constrain_field(peak_pos, peak_height, scale_mpc, boxlen, rhoU_out, ps, cosmo, shape_constraints)
    #  ipsiC = egp.icgen.DisplacementField(irhoC)
    #  del irhoC
    #  ipos, ivel = egp.icgen.zeldovich(redshift, ipsiC, cosmo) # Mpc, not h^-1!
    #  del ipsiC
    #  ic_file = "/Users/users/pbos/dataserver/sims/ICs/ic_%iMpc_%i_%s_%i.dat" % (boxlen, gridsize, test_id, seed)
    #  egp.io.write_gadget_ic_dm(ic_file, ipos.reshape((3,gridsize**3)).T, ivel.reshape((3,gridsize**3)).T, particle_mass, redshift, boxlen, cosmo.omegaM, cosmo.omegaL, cosmo.h)
    #  del ipos, ivel
    #  
    #  print "Preparing for gadget run %(run_name)s..." % locals()
    #  egp.io.prepare_gadget_run(boxlen, gridsize, cosmo, ic_file, redshift, run_dir_base, run_name, nproc)
    
    log_file = open(run_dir_base+'/'+run_name+'.log', 'a+')
    log_file.write('peak position:\n')
    log_file.write(peak_pos.__str__())
    log_file.close()
    
    print "Done!"