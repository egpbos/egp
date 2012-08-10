#!/usr/bin/env python
# encoding: utf-8

from iconstrain4 import *
from egp.icgen import Cosmology, CosmoPowerSpectrum, GaussianRandomField
from csv import reader as csvreader
import egp.io as io

test_id = "3"
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
ps.normalize((boxlen/cosmo.h)**3)

# Unconstrained field
rhoU = GaussianRandomField(ps, boxlen, gridsize_iter, seed=seed)

result = iterate(pos0, mass0, boxlen, gridsize_iter, rhoU, ps, cosmo)

# Plot
iteration_mean(result, mass0, boxlen, gridsize_iter, rhoU, ps, cosmo, True, pos0)


# ---- OUTPUT to Gadget IC files ----

rhoc = critical_density(cosmo) # M_sun Mpc^-3
rhoc = rhoc/cosmo.h**2 # M_sun Mpc^-3 h^2 (boxlen is in Mpc h^-1 too, so must convert)
particle_mass = cosmo.omegaM * rhoc * boxlen**3 / gridsize**3 / 1e10 # 10^10 M_sun h^-1

rhoU_out = GaussianRandomField(ps, boxlen, gridsize, seed=seed)

# iterated version:
print "Building and saving iterated version..."
irhoC = constrain_field(result, mass0, boxlen, rhoU_out, ps, cosmo)
ipsiC = DisplacementField(irhoC)
del irhoC
ipos, ivel = zeldovich_new(redshift, ipsiC, cosmo) # Mpc, not h^-1!
del ipsiC
ic_file = "/Users/users/pbos/dataserver/sims/ICs/ic_icon_%iMpc_%i_%s_%i.dat" % (boxlen, gridsize, test_id, seed)
io.write_gadget_ic_dm(ic_file, 1000*ipos.reshape((3,gridsize**3)).T*cosmo.h, ivel.reshape((3,gridsize**3)).T, particle_mass, redshift, boxlen, cosmo.omegaM, cosmo.omegaL, cosmo.h)
del ipos, ivel

# non-iterated version (for comparison):
print "Not building and saving non-iterated version, because that's the same as in test1."

print "Preparing for gadget run %(run_name)s..."
io.prepare_gadget_run(boxlen*1000, gridsize, cosmo, ic_file, redshift, run_dir_base, run_name, nproc)

print "Done!"