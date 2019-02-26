#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import egp.icgen, egp.io, egp.toolbox, egp.iconstrain
import sys

test_id = "multi_test2"

run_dir_base = "/Users/users/pbos/dataserver/sims"
nproc = 16

# ---- INPUT ----
cosmo = egp.icgen.Cosmology('wmap7', trans = 8)
ps = egp.icgen.CosmoPowerSpectrum(cosmo)
boxlen = 200. # Mpc h^-1
redshift = 63.
gridsize = 256

gridsize_iter = 64
seed = 2522572538
try:
    seed += int(sys.argv[1])
except:
    print "Geef een getal op om op te tellen bij de seed!"
    raise SystemExit

deltaU = egp.icgen.GaussianRandomField(ps, boxlen, gridsize_iter, seed=seed)

# constrained peak position:
target_pos = np.array([20.,40.,70.])
# target_pos = np.array([[20.,40.,70.], [50.,10.,20.]]).flatten()
# target_pos = np.array([[20.,40.,70.], [50.,10.,20.], [20.,40.,60.]).flatten()

# constrained peak mass:
filename = "/Users/users/pbos/code/egp/testing/icgen/MCXC+xyz_SCLx4.1.csv" # kapteyn
#filename = "/Users/patrick/astro/code/egp/testing/icgen/MCXC+xyz_SCLx4.1.csv" # mac
clusters = egp.toolbox.load_csv_catalog(filename)
target_mass = np.array([x[12] for x in clusters], dtype='float64')[0] # 10^14 Msun
# ... leading to the peak's height (in overdensity) and scale (in mpc):
peak_height = egp.iconstrain.mass_to_peak_height(target_mass, cosmo)# :D DIT IS ALTIJD 1! DE SCHAAL BEPAALT ALLES AL BLIJKBAAR!
scale_mpc = egp.iconstrain.mass_to_scale(target_mass, cosmo)

# VOOR NU EVEN ZELFDE HEIGHTS EN SCALES:
peak_height = np.zeros(len(target_pos)/3) + peak_height
scale_mpc   = np.zeros(len(target_pos)/3) + scale_mpc

result = egp.iconstrain.run_multi(cosmo, ps, boxlen, gridsize_iter, deltaU, target_pos, peak_height, scale_mpc, egp.iconstrain.iterate_peakwise_shift_zeldovich_multi, initial_guess=target_pos, constrain_shape=True)

print "Uiteindelijk gebruiken we ", result
# result = np.array([ 19.98012509,  38.26122861,  67.98546924])

run_name = "run_%s_%i" % (test_id, seed)

print "NOT SAVING, DONE FOR NOW"
# egp.iconstrain.setup_gadget_run_multi(cosmo, ps, boxlen, gridsize, seed, result, peak_height, scale_mpc, redshift, test_id, run_dir_base, nproc, run_name = run_name)
