#!/usr/bin/env python
# encoding: utf-8

from iconstrain_tester import *
import egp.toolbox, egp.icgen, egp.io
import sys

test_id = "0"

run_dir_base = "/Users/users/pbos/dataserver/sims"
nproc = 4

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

# Plot
#~ iteration_mean(result, mass0, boxlen, gridsize_iter, rhoU, ps, cosmo, shape_constraints, True, pos0)

setup_gadget_run(cosmo, ps, boxlen, gridsize, seed, target_pos, peak_height, scale_mpc, redshift, test_id, run_dir_base, nproc)
