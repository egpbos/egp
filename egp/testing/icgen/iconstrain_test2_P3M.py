#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import iconstrain
import egp.icgen, egp.io, egp.toolbox
import sys

test_id = "2_P3M"

run_dir_base = "/Users/users/pbos/dataserver/sims"
nproc = 2

# ---- INPUT ----
cosmo = egp.icgen.Cosmology('wmap7', trans=8)
ps = egp.icgen.CosmoPowerSpectrum(cosmo)
boxlen = 200. # Mpc h^-1
redshift = 63.
gridsize = 256
gridsize_iter = 64
nf_tile_I = 2 # zet op 3 bij 32, anders raakt ie alle particles kwijt.

seed = 2522572538
try:
    seed += int(sys.argv[1])
except:
    print "Geef een getal op om op te tellen bij de seed!"
    raise SystemExit

deltaU = egp.icgen.GaussianRandomField(ps, boxlen, gridsize_iter, seed=seed)

# constrained peak position:
target_pos = np.array([20.,40.,70.])

# constrained peak mass:
filename = "/Users/users/pbos/code/egp/testing/icgen/MCXC+xyz_SCLx4.1.csv" # kapteyn
#filename = "/Users/patrick/astro/code/egp/testing/icgen/MCXC+xyz_SCLx4.1.csv" # mac
clusters = egp.toolbox.load_csv_catalog(filename)
mass0 = np.array([x[12] for x in clusters], dtype='float64')[0] # 10^14 Msun
# ... leading to the peak's height:
peak_height = iconstrain.mass_to_peak_height(mass0, cosmo) # 1.0000000000000002
# ... and the scale of the peak in mpc:
scale_mpc = iconstrain.mass_to_scale(mass0, cosmo) # 4.0225468329142506

# Multiply height by second command line parameter:
# try:
#     height_multiplier = float(sys.argv[2])
# except:
#     print "Geef een getal op om te vermenigvuldigen met de height!"
#     raise SystemExit

# peak_height *= height_multiplier

shape_seed = 10
shape_constraints = iconstrain.set_shape_constraints(ps, boxlen, peak_height, scale_mpc, shape_seed)
one_step_result = iconstrain.iteration_mean_P3M(target_pos, peak_height, scale_mpc, boxlen, gridsize_iter, deltaU, ps, cosmo, shape_constraints, nf_tile_I = nf_tile_I)

raise SystemExit

result = iconstrain.run(cosmo, ps, boxlen, gridsize_iter, seed, target_pos, peak_height, scale_mpc, initial_guess, iconstrain.iterate_PM, constrain_shape=True)

print "Uiteindelijk gebruiken we ", result
# result = np.array([ 19.98012509,  38.26122861,  67.98546924])

# Plot
#~ iteration_mean(result, mass0, boxlen, gridsize_iter, rhoU, ps, cosmo, shape_constraints, True, pos0)

run_name = "run%i_%i_hx%f" % ((100+int(test_id)), seed, height_multiplier)

setup_gadget_run(cosmo, ps, boxlen, gridsize, seed, result, peak_height, scale_mpc, redshift, test_id, run_dir_base, nproc, run_name = run_name)

print("Saving not iterated version for comparison (run100)...")

run_name = "run100_%i_hx%f" % (seed, height_multiplier)

setup_gadget_run(cosmo, ps, boxlen, gridsize, seed, target_pos, peak_height, scale_mpc, redshift, test_id, run_dir_base, nproc, run_name = run_name)
