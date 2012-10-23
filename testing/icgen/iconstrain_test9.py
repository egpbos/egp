#!/usr/bin/env python
# encoding: utf-8

from iconstrain4plus1 import initial_guess, iterate
from iconstrain_tester import *
import egp.icgen, egp.io
import sys

test_id = "9" # Test versie 4+1 met verschillende seeds en verschillende peak heights

run_dir_base = "/Users/users/pbos/dataserver/sims"
nproc = 2

# ---- INPUT ----
cosmo = egp.icgen.Cosmology('wmap7')
cosmo.trans = 8
boxlen = 100. # Mpc h^-1
redshift = 63.
gridsize = 256

gridsize_iter = 64
seed = 2522572538
try:
    seed += int(sys.argv[1])
except:
    print "Geef een getal op om op te tellen bij de seed!"
    raise SystemExit

ps = egp.icgen.CosmoPowerSpectrum(cosmo)
ps.normalize(boxlen**3)

# constrained peak position:
target_pos = np.array([20.,40.,70.])

# constrained peak mass:
filename = "/Users/users/pbos/code/egp/testing/icgen/MCXC+xyz_SCLx4.1.csv" # kapteyn
#filename = "/Users/patrick/astro/code/egp/testing/icgen/MCXC+xyz_SCLx4.1.csv" # mac
clusters = load_catalog(filename)
mass0 = np.array([x[12] for x in clusters], dtype='float64')[0] # 10^14 Msun
# ... leading to the peak's height:
peak_height = mass_to_peak_height(mass0, cosmo)
# ... and the scale of the peak in mpc:
scale_mpc = mass_to_scale(mass0, cosmo)

# Multiply height by second command line parameter:
try:
    height_multiplier = float(sys.argv[2])
except:
    print "Geef een getal op om te vermenigvuldigen met de height!"
    raise SystemExit

peak_height *= height_multiplier

result = test_run(cosmo, ps, boxlen, gridsize_iter, seed, target_pos, peak_height, scale_mpc, initial_guess, iterate, constrain_shape=True)

print "Uiteindelijk gebruiken we ", result
# result = np.array([ 19.98012509,  38.26122861,  67.98546924])

# Plot
#~ iteration_mean(result, mass0, boxlen, gridsize_iter, rhoU, ps, cosmo, shape_constraints, True, pos0)

run_name = "run%i_%i_hx%f" % ((100+int(test_id)), seed, height_multiplier)

setup_gadget_run(cosmo, ps, boxlen, gridsize, seed, result, peak_height, scale_mpc, redshift, test_id, run_dir_base, nproc, run_name = run_name)

print("Saving not iterated version for comparison (run100)...")

run_name = "run100_%i_hx%f" % (seed, height_multiplier)

setup_gadget_run(cosmo, ps, boxlen, gridsize, seed, target_pos, peak_height, scale_mpc, redshift, test_id, run_dir_base, nproc, run_name = run_name)
