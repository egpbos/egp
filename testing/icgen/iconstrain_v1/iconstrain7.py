#!/usr/bin/env python

import numpy as np
import egp.icgen, egp.toolbox
import iconstrain_tester
import scipy.interpolate, scipy.optimize, scipy.ndimage

delete_tussenstappen = True
test_id = 10

run_dir_base = "/Users/users/pbos/dataserver/sims"
nproc = 16

# ---- INPUT ----
cosmo = egp.icgen.Cosmology('wmap7')
cosmo.trans = 8
boxlen = 100. # Mpc h^-1
redshift = 63.
gridsize = 64

#gridsize_iter = 64
seed = 2522572538

ps = egp.icgen.CosmoPowerSpectrum(cosmo)
ps.normalize(boxlen**3)

# constrained peak position:
target_pos = np.array([20.,40.,70.])

# constrained peak mass:
filename = "/Users/users/pbos/code/egp/testing/icgen/MCXC+xyz_SCLx4.1.csv" # kapteyn
#filename = "/Users/patrick/astro/code/egp/testing/icgen/MCXC+xyz_SCLx4.1.csv" # mac
clusters = iconstrain_tester.load_catalog(filename)
mass0 = np.array([x[12] for x in clusters], dtype='float64')[0] # 10^14 Msun
# ... leading to the peak's height:
peak_height = iconstrain_tester.mass_to_peak_height(mass0, cosmo)
scale_mpc = iconstrain_tester.mass_to_scale(mass0, cosmo)

rhoc = egp.toolbox.critical_density(cosmo) # M_sun Mpc^-3 h^2
particle_mass = cosmo.omegaM * rhoc * boxlen**3 / gridsize**3 / 1e10 # 10^10 M_sun h^-1

delU = egp.icgen.GaussianRandomField(ps, boxlen, gridsize, seed=seed)
psiU = egp.icgen.DisplacementField(delU)
if delete_tussenstappen: del delU

x, v = egp.icgen.zeldovich(0, psiU, cosmo)
if delete_tussenstappen: del v
disp_x = egp.icgen.zeldovich_displacement(0, psiU, cosmo)
if delete_tussenstappen: del psiU

disp_x_fourier = np.fft.rfftn(disp_x, axes=(1,2,3))
if delete_tussenstappen: del disp_x

#~ kernel = egp.toolbox.tophat_kernel
kernel = egp.toolbox.gaussian_kernel
disp_x_fourier_filtered = egp.toolbox.filter_field(disp_x_fourier, boxlen, kernel, (scale_mpc,), gridsize=gridsize)
if delete_tussenstappen: del disp_x_fourier

disp_x_filtered = np.fft.irfftn(disp_x_fourier_filtered, s=(gridsize,gridsize,gridsize), axes=(1,2,3))
if delete_tussenstappen: del disp_x_fourier_filtered

dq = boxlen/gridsize
q = np.mgrid[dq/2:boxlen+dq/2:dq,dq/2:boxlen+dq/2:dq,dq/2:boxlen+dq/2:dq]
x_filtered = (q + disp_x_filtered) % boxlen # PLOT DIT! MOOI!
# pl.plot...
if delete_tussenstappen: del disp_x_filtered

# now find the position at which the peak will come closest to target_pos:
order = 3
# full x,y,z method:
#~ minimize_grid = (x_filtered - target_pos[:,None,None,None])**2
#~ minimize_interp = lambda x_interpolate: np.array([\
            #~ scipy.ndimage.map_coordinates(minimize_grid[0], x_interpolate[:,None], order=order, mode='nearest')[0],\
            #~ scipy.ndimage.map_coordinates(minimize_grid[1], x_interpolate[:,None], order=order, mode='nearest')[0],\
            #~ scipy.ndimage.map_coordinates(minimize_grid[2], x_interpolate[:,None], order=order, mode='nearest')[0]\
            #~ ])
#~ result = scipy.optimize.fsolve(minimize_interp, target_pos/boxlen*gridsize)/gridsize*boxlen
# alternative interpolator:
#~ minimize_interp = scipy.interpolate.griddata(q, minimize_grid) # DIT IS VOOR ALS GRIDDATA OOIT IN DE SCIPY VERSIE HIER KOMT
# radius-only method:
minimize_grid = ((x_filtered - target_pos[:,None,None,None])**2).sum(axis=0)
minimize_interp = lambda x_interpolate: scipy.ndimage.map_coordinates(minimize_grid, x_interpolate[:,None], order=order, mode='nearest')

#~ initial_guess = np.array([ 19.98012509,  38.26122861,  67.98546924]) # van test 9
initial_guess = target_pos/boxlen*gridsize
result = scipy.optimize.fmin(minimize_interp, initial_guess)/gridsize*boxlen

print result
# Met tophat kernel:
# 64^3 : array([ 19.77369882,  40.7561047,   68.78399267])
# 128^3: array([ 19.98743233,  41.02083634,  69.13811584])
# 256^3: array([ 20.26747414,  41.04179358,  69.18462654])
# N.B.: nogal anders dan voorgaande methodes! Test 9 was bijv np.array([ 19.98012509,  38.26122861,  67.98546924])
# Met Gaussian kernel:
# 64^3 : array([ 20.01036225,  39.52785985,  67.3660702 ]
# 128^3: array([ 20.33547656,  39.66217154,  67.58502755])
# 256^3: array(


# TEST 1: kijken of een piek in het veld zetten het resultaat verandert
import iconstrain

del_result = iconstrain.constrain_field(result, peak_height, scale_mpc, boxlen, delU, ps, cosmo)
psi_result = egp.icgen.DisplacementField(del_result)

disp_x_result = egp.icgen.zeldovich_displacement(0, psi_result, cosmo)
disp_x_result_fourier = np.fft.rfftn(disp_x_result, axes=(1,2,3))

#~ kernel = egp.toolbox.tophat_kernel
kernel = egp.toolbox.gaussian_kernel
disp_x_result_fourier_filtered = egp.toolbox.filter_field(disp_x_result_fourier, boxlen, kernel, (scale_mpc,), gridsize=gridsize)

disp_x_result_filtered = np.fft.irfftn(disp_x_result_fourier_filtered, s=(gridsize,gridsize,gridsize), axes=(1,2,3))

x_result_filtered = (q + disp_x_result_filtered) % boxlen # PLOT DIT! MOOI!

order = 3
minimize_grid_result = ((x_result_filtered - target_pos[:,None,None,None])**2).sum(axis=0)
minimize_interp_result = lambda x_interpolate: scipy.ndimage.map_coordinates(minimize_grid_result, x_interpolate[:,None], order=order, mode='nearest')

result_result = scipy.optimize.fmin(minimize_interp_result, initial_guess)/gridsize*boxlen
# DIT IS array([ 19.02927478,  34.97782867,  66.03896886])
# VOLKOMEN ANDERS DUS DAN HET EERSTE RESULTAAT.
# Achteraf gezien ook logisch: de afgeleide van het veld blijft helemaal
# niet constant, maar wordt juist op nul gezet met de ExtremumConstraints!
# Hoe kunnen we dit integreren om toch een directe methode te gebruiken?

# EINDE TEST 1


run_name = "run%i_%i_gauss" % ((100+int(test_id)), seed)

#iconstrain_tester.setup_gadget_run(cosmo, ps, boxlen, gridsize, seed, result, peak_height, scale_mpc, redshift, test_id, run_dir_base, nproc, run_name = run_name)
