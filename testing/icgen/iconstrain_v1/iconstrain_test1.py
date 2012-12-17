#!/usr/bin/env python
# encoding: utf-8

from iconstrain import initial_guess iterate
from iconstrain_tester import test_run, setup_gadget_run, load_catalog, mass_to_peak_height
import egp.toolbox, egp.icgen, egp.io

test_id = "1"

run_dir_base = "/Users/users/pbos/dataserver/sims"
nproc = 16

# ---- INPUT ----
cosmo = egp.icgen.Cosmology('wmap7')
cosmo.trans = 8
boxlen = 100. # Mpc h^-1
redshift = 63.
gridsize = 256

gridsize_iter = 64
seed = 2522572538

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

result = test_run(cosmo, boxlen, redshift, gridsize_iter, seed, target_pos, peak_height, initial_guess, iterate, constrain_shape=True)

print "Uiteindelijk gebruiken we ", result
# result = np.array([ 19.98012509,  38.26122861,  67.98546924])

# Plot
#~ iteration_mean(result, mass0, boxlen, gridsize_iter, rhoU, ps, cosmo, shape_constraints, True, pos0)

setup_gadget_run(cosmo, boxlen, redshift, gridsize_iter, seed, result, peak_height, redshift, test_id, run_dir_base, nproc)
