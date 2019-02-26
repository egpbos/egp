#!/usr/bin/env python
# encoding: utf-8

from iconstrain5 import *
from csv import reader as csvreader
import egp.io as io
import egp.icgen as icgen

test_id = "7" # Test versie 5; !!!! 2LPT ipv Zel'dovich !!!!
run_name = "run%i" % (100+int(test_id)) # plus 100 to separate from DE+void runs

run_dir_base = "/Users/users/pbos/dataserver/sims"
nproc = 16

# ---- INPUT ----
cosmo = icgen.Cosmology('wmap7')
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

# Power spectrum et al:
ps = icgen.CosmoPowerSpectrum(cosmo)
ps.normalize(boxlen**3)

rhoc = egp.toolbox.critical_density(cosmo)
scale_mpc = ((mass0*1e14)/rhoc/(2*np.pi)**(3./2))**(1./3) # Mpc h^-1
height = mass0*1e14/(2*np.pi)**(3./2)/scale_mpc**3/rhoc

sigma0 = ps.moment(0, scale_mpc, boxlen**3)
sigma1 = ps.moment(1, scale_mpc, boxlen**3)
sigma2 = ps.moment(2, scale_mpc, boxlen**3)
gamma = sigma1**2/sigma0/sigma2

# shape / orientation constraints:
np.random.seed(0)
curvature = icgen.random_curvature(height, gamma)
a21, a31 = icgen.random_shape(curvature)
density_phi = 180*np.random.random()
density_theta = 180/np.pi*np.arccos(1-np.random.random())
density_psi = 180*np.random.random()
shape_constraints = (curvature, a21, a31, density_phi, density_theta, density_psi)

# --- CALCULATE STUFF ---

# Unconstrained field
rhoU = icgen.GaussianRandomField(ps, boxlen, gridsize_iter, seed=seed)

pos_initial = initial_guess(pos0, mass0, boxlen, gridsize_iter, rhoU, ps, cosmo, shape_constraints)
results_all = iterate(pos_initial, pos0, mass0, boxlen, gridsize_iter, rhoU, ps, cosmo, shape_constraints)
result = results_all[0]
#~ result = np.array([ 19.93056985,  38.59509036,  67.98152977]) # Zel'dovich result

print "Uiteindelijk gebruiken we ", result

#"""
# Plot
#~ iteration_mean(result, mass0, boxlen, gridsize_iter, rhoU, ps, cosmo, shape_constraints, True, pos0)


# ---- OUTPUT to Gadget IC files ----

rhoc = egp.toolbox.critical_density(cosmo) # M_sun Mpc^-3 h^2
particle_mass = cosmo.omegaM * rhoc * boxlen**3 / gridsize**3 / 1e10 # 10^10 M_sun h^-1

rhoU_out = icgen.GaussianRandomField(ps, boxlen, gridsize, seed=seed)

# iterated version:
print "Building and saving iterated version..."
irhoC = constrain_field(result, mass0, boxlen, rhoU_out, ps, cosmo, shape_constraints)
ipsiC = icgen.DisplacementField(irhoC)
ipsi2C = icgen.DisplacementField2ndOrder(ipsiC)
del irhoC
ipos, ivel = two_LPT_ICs(redshift, ipsiC, ipsi2C, cosmo) # Mpc, not h^-1!
#del ipsiC
ic_file = "/Users/users/pbos/dataserver/sims/ICs/ic_icon_%iMpc_%i_%s_%i.dat" % (boxlen, gridsize, test_id, seed)
#~ io.write_gadget_ic_dm(ic_file, 1000*ipos.reshape((3,gridsize**3)).T*cosmo.h, ivel.reshape((3,gridsize**3)).T, particle_mass, redshift, boxlen, cosmo.omegaM, cosmo.omegaL, cosmo.h)
io.write_gadget_ic_dm(ic_file, ipos.reshape((3,gridsize**3)).T, ivel.reshape((3,gridsize**3)).T, particle_mass, redshift, boxlen, cosmo.omegaM, cosmo.omegaL, cosmo.h)
del ipos, ivel

# non-iterated version (for comparison):
print "Not building and saving non-iterated version, because that's the same as in test1."

print "Preparing for gadget run %(run_name)s..." % locals()
io.prepare_gadget_run(boxlen, gridsize, cosmo, ic_file, redshift, run_dir_base, run_name, nproc)

print "Done!"
#"""
