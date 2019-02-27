#!/usr/bin/env python
# encoding: utf-8

# --- Nutteloos:
# Newton-CG kan niet (is Jacobian bij nodig)
# COBYLA gaat ook wel de goeie kant op. Maar die was op den duur "klaar", maar niet echt. Niet bruikbaar dus.
# Anneal; die komt gewoon totaal niet in de buurt, heeft ook gigantische waarden (misschien met een kleinere temperatuur)

import numpy as np
import egp.icgen, egp.io, egp.toolbox, egp.iconstrain
import sys

test_id = "multi_test1" # Test versie 4+1 met verschillende seeds en verschillende peak heights

method = sys.argv[2]

method_type = sys.argv[3] # root of minimize
# Beste methoden: hybr en diagbroyden
# Mogelijke methoden: lm, broyden1, broyden2, anderson, linearmixing, excitingmixing, krylov.

#~ method_type = "minimize"
# Beste methoden: Nelder-Mead en L-BFGS-B
# Andere mogelijke methoden: Powell, COBYLA, SLSQP, CG, BFGS, TNC, Anneal. Newton-CG is niet mogelijk (jacobian nodig).


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
target_pos = np.array([20., 40., 70.])
#~ target_pos = np.array([[20.,40.,70.], [50.,10.,20.]]).flatten()
#~ target_pos = np.array([[20.,40.,70.], [20.,40.,60.], [20.,42.,74.]]).flatten()

initial_guess = target_pos # egp.iconstrain.iterate_mirror_zeldovich_multi

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

iterate = egp.iconstrain.iterate_solve_multi_openopt
iteration = egp.iconstrain.iteration_mean_zeldovich_multi

#result = egp.iconstrain.run_multi(cosmo, ps, boxlen, gridsize_iter, deltaU, target_pos, peak_height, scale_mpc, egp.iconstrain.iterate_zeldovich_multi, constrain_shape=True, epsilon=1e-6, factr=1e11, pgtol=1e-3)
result = egp.iconstrain.run_multi(cosmo, ps, boxlen, gridsize_iter, deltaU, target_pos, peak_height, scale_mpc, iterate, constrain_shape=True, epsilon=1e-13, factr=1e11, pgtol=1e-3, initial_guess = initial_guess, method = method, method_type = method_type, iteration = iteration)

print "Uiteindelijk gebruiken we ", result
# result = np.array([ 19.98012509,  38.26122861,  67.98546924])

run_name = "run_%s_%i" % (test_id, seed)

#egp.iconstrain.setup_gadget_run_multi(cosmo, ps, boxlen, gridsize, seed, result, peak_height, scale_mpc, redshift, test_id, run_dir_base, nproc, run_name = run_name)
