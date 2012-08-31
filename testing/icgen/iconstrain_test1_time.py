#!/usr/bin/env ipython
# encoding: utf-8

from iconstrain import *
from egp.icgen import Cosmology, CosmoPowerSpectrum, GaussianRandomField
from csv import reader as csvreader
import egp.io as io

from egp.toolbox import KGridCache
import __builtin__
__builtin__.k_grid_cache = KGridCache() # activate cache

test_id = "1"
run_name = "run%i" % (100+int(test_id)) # plus 100 to separate from DE+void runs

run_dir_base = "/Users/users/pbos/dataserver/sims"
nproc = 16

# ---- INPUT ----
cosmo = Cosmology('wmap7')
cosmo.trans = 8
boxlen = 100. # Mpc h^-1
redshift = 63.
gridsize = 256

gridsize_iter = 64
#~ seed = np.random.randint(0x100000000)
seed = 2522572538

# constrained peak position:
pos0 = np.array([20.,40.,70.])
# constrained peak mass stuff:
path = "/Users/users/pbos/code/egp/testing/icgen/" # kapteyn
#path = "/Users/patrick/astro/code/egp/testing/icgen/" # macbook
cluster_table_file = open(path+"MCXC+xyz_SCLx4.1.csv")
cluster_table = csvreader(cluster_table_file)
cluster_table.next() # skip header
clusters = []
for cluster in cluster_table:
    clusters.append(cluster)

cluster_table_file.close()
mass0 = np.array([x[12] for x in clusters], dtype='float64')[0] # 10^14 Msun


# ---- CALCULATE STUFF ----

ps = CosmoPowerSpectrum(cosmo)
ps.normalize(boxlen**3)

# Unconstrained field
rhoU = GaussianRandomField(ps, boxlen, gridsize_iter, seed=seed)

print "Now run this command:\n%prun result = iterate(pos0, mass0, boxlen, gridsize_iter, rhoU, ps, cosmo)"
