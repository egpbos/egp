#!/usr/bin/env python
# encoding: utf-8

# --- Nutteloos:
# Newton-CG kan niet (is Jacobian bij nodig)
# COBYLA gaat ook wel de goeie kant op, maar heeft een afwijkende result array blijkbaar. Maar iig, die was op den duur "klaar", maar niet echt. Niet bruikbaar dus.
# Anneal; die komt gewoon totaal niet in de buurt, heeft ook gigantische waarden (misschien met een kleinere temperatuur)

import numpy as np
import egp.icgen, egp.io, egp.toolbox, egp.iconstrain
import sys

test_id = "multi_test1" # Test versie 4+1 met verschillende seeds en verschillende peak heights

method = sys.argv[2]

method_type = "root"
# Mogelijke methoden: hybr, lm, broyden1, broyden2, anderson, linearmixing, diagbroyden, excitingmixing, krylov

#~ method_type = "minimize"
# Mogelijke methoden: Nelder-Mead, Powell, COBYLA, L-BFGS-B, SLSQP, CG, BFGS, TNC, Anneal
# Newton-CG is niet mogelijk (jacobian nodig)

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
target_pos = np.array([[20.,40.,70.], [50.,10.,20.]]).flatten()
# target_pos = np.array([[20.,40.,70.], [50.,10.,20.], [20.,40.,60.]).flatten()

# BFGS TESTS:
#~ initial_guess = np.array([19.48861907 , 37.25563749 , 70.88690692, 49.24620828,  8.95229306, 22.67281736])
#~ # Minimum dat bfgs solver met epsilon=1e-6 geeft tijdens een "bounce"
#~ # (komt vast te zitten op twee punten); hierna epsilon dus lager gezet
#~ initial_guess = np.array([ 19.06452644 , 36.1569645,   71.58331242, 49.17259507,   8.93680965 , 22.73265783])
#~ # Deze was na een bounce op epsilon=1e-9, diff = 0.81 was dat
#~ initial_guess = np.array([18.56519417, 36.21160315, 71.56745277,49.24538505,  8.94075126, 22.72700529])
#~ # Deze na bounce op epsilon=1e-11, diff = 0.54

initial_guess = egp.iconstrain.iterate_mirror_zeldovich_multi

# constrained peak mass:
filename = "/Users/users/pbos/code/egp/testing/icgen/MCXC+xyz_SCLx4.1.csv" # kapteyn
#filename = "/Users/patrick/astro/code/egp/testing/icgen/MCXC+xyz_SCLx4.1.csv" # mac
clusters = egp.toolbox.load_csv_catalog(filename)
target_mass = np.array([x[12] for x in clusters], dtype='float64')[0] # 10^14 Msun
# ... leading to the peak's height (in overdensity) and scale (in mpc):
peak_height = egp.iconstrain.mass_to_peak_height(target_mass, cosmo)
scale_mpc = egp.iconstrain.mass_to_scale(target_mass, cosmo)

# VOOR NU EVEN ZELFDE HEIGHTS EN SCALES:
peak_height = np.zeros(len(target_pos)/3) + peak_height# :D DIT IS ALTIJD 1! DE SCHAAL BEPAALT ALLES AL BLIJKBAAR!
scale_mpc   = np.zeros(len(target_pos)/3) + scale_mpc

#result = egp.iconstrain.run_multi(cosmo, ps, boxlen, gridsize_iter, deltaU, target_pos, peak_height, scale_mpc, egp.iconstrain.iterate_zeldovich_multi, constrain_shape=True, epsilon=1e-6, factr=1e11, pgtol=1e-3)
result = egp.iconstrain.run_multi(cosmo, ps, boxlen, gridsize_iter, deltaU, target_pos, peak_height, scale_mpc, egp.iconstrain.iterate_zeldovich_multi, constrain_shape=True, epsilon=1e-13, factr=1e11, pgtol=1e-3, initial_guess = initial_guess, method = method, method_type = method_type)

print "Uiteindelijk gebruiken we ", result
# result = np.array([ 19.98012509,  38.26122861,  67.98546924])

run_name = "run_%s_%i" % (test_id, seed)

#egp.iconstrain.setup_gadget_run_multi(cosmo, ps, boxlen, gridsize, seed, result, peak_height, scale_mpc, redshift, test_id, run_dir_base, nproc, run_name = run_name)
