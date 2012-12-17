#!/usr/bin/env python
# encoding: utf-8

from iconstrain2 import *
from egp.icgen import Cosmology, CosmoPowerSpectrum, GaussianRandomField, DensityField
from csv import reader as csvreader
import egp.io as io


test_id = "4"
run_name = "run%i" % (100+int(test_id)) # plus 100 to separate from DE+void runs

run_dir_base = "/Users/users/pbos/dataserver/sims"
nproc = 16

# ---- INPUT ----
cosmo = Cosmology('wmap7')
cosmo.trans = 8
boxlen = 100. # Mpc h^-1

redshift = 63.
# 2-Zel'dovich:
intermediate_redshift = 50.
# 3-Zel'dovich:
zeld3_redshift1 = 1000.
zeld3_redshift2 = 100.

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

ps.moment(

# Unconstrained field
delU = GaussianRandomField(ps, boxlen, gridsize_iter, seed=seed)

delC = constrain_field(pos0, mass0, boxlen, delU, ps, cosmo)
psiC = DisplacementField(delC)

# 2-ZEL'DOVICH:
# Zel'dovich tot z=intermediate_redshift
POS_tussen, v_tussen = zeldovich(intermediate_redshift, psiC, cosmo)

# Density & displacement field
rhoc = critical_density(cosmo)
particle_mass = rhoc*boxlen**3/gridsize**3
rho_tussen = egp.toolbox.TSC_density(POS_tussen.reshape((3,gridsize_iter**3)).T, gridsize_iter, boxlen, particle_mass)
del_tussen = DensityField(true = rho_tussen/rho_tussen.mean() - 1)
del_tussen.boxlen = boxlen
psi_tussen = DisplacementField(del_tussen)

POS, v = zeldovich_step(intermediate_redshift, 0., psi_tussen, POS_tussen, cosmo)

# 3-ZEL'DOVICH:
POS3_1, v3_1 = zeldovich(zeld3_redshift1, psiC, cosmo)

rho3_1 = egp.toolbox.TSC_density(POS3_1.reshape((3,gridsize_iter**3)).T, gridsize_iter, boxlen, particle_mass)
del3_1 = DensityField(true = rho3_1/rho3_1.mean() - 1)
del3_1.boxlen = boxlen
psi3_1 = DisplacementField(del3_1)

POS3_2, v3_2 = zeldovich_step(zeld3_redshift1, zeld3_redshift2, psi3_1, POS3_1, cosmo)

rho3_2 = egp.toolbox.TSC_density(POS3_2.reshape((3,gridsize_iter**3)).T, gridsize_iter, boxlen, particle_mass)
del3_2 = DensityField(true = rho3_2/rho3_2.mean() - 1)
del3_2.boxlen = boxlen
psi3_2 = DisplacementField(del3_2)

POS3_end, v3_end = zeldovich_step(zeld3_redshift2, 0., psi3_2, POS3_2, cosmo)


# Ter vergelijking:
POS0, v0 = zeldovich(0., psiC, cosmo)

# Vergelijk met Gadget runs (101_noicon):
#~ from mayavi import mlab
#~ 
#gadgetfile0 = "/net/dataserver1/data/users/pbos/sims/run101_noicon/snap_008"
#gadgetfile2 = "/net/dataserver1/data/users/pbos/sims/run101_noicon/snap_002"

#gadget0 = io.GadgetData(gadgetfile0)
#gadget2 = io.GadgetData(gadgetfile2)

#sub = np.random.randint(0, gadget0.Ntotal, 64**3)

# opslaan (eenmalig):
#np.save('/net/dataserver1/data/users/pbos/sims/run101_noicon/snap_008_subset.npy', gadget0.pos[sub])
#np.save('/net/dataserver1/data/users/pbos/sims/run101_noicon/snap_002_subset.npy', gadget2.pos[sub])

#~ gadget0sub = np.load('/net/dataserver1/data/users/pbos/sims/run101_noicon/snap_008_subset.npy')
#~ gadget2sub = np.load('/net/dataserver1/data/users/pbos/sims/run101_noicon/snap_002_subset.npy')
#~ 
#~ mlab.figure(1);mlab.points3d(gadget0sub[:,0], gadget0sub[:,1], gadget0sub[:,2], mode='point', opacity=0.3);mlab.axes()
#~ mlab.figure(2);mlab.points3d(POS[0], POS[1], POS[2], mode='point', opacity=0.3);mlab.axes()
#~ mlab.figure(3);mlab.points3d(POS3_end[0], POS3_end[1], POS3_end[2], mode='point', opacity=0.3);mlab.axes()
#~ mlab.figure(4);mlab.points3d(POS0[0], POS0[1], POS0[2], mode='point', opacity=0.3);mlab.axes()
#~ mlab.figure(5);mlab.points3d(POS_tussen[0], POS_tussen[1], POS_tussen[2], mode='point', opacity=0.3);mlab.axes()

#~ mlab.figure(1);mlab.points3d(gadget2sub[:,0], gadget2sub[:,1], gadget2sub[:,2], mode='point', opacity=0.3);mlab.axes()
#~ mlab.figure(2);mlab.points3d(POS_tussen[0], POS_tussen[1], POS_tussen[2], mode='point', opacity=0.3);mlab.axes()


# Test vervelend spiegel gedoe:
#~ from egp.toolbox import field_show
#~ 
#~ pl.figure(1)
#~ field_show(delU.t[0], boxlen)
#~ 
#~ pl.figure(2)
#~ field_show(delC.t[0], boxlen)
#~ 
#~ pl.figure(3)
#~ pl.subplot(221)
#~ field_show(psiC.x.t[0], boxlen)
#~ pl.subplot(222)
#~ field_show(psiC.y.t[0], boxlen)
#~ pl.subplot(223)
#~ field_show(psiC.z.t[0], boxlen)
#~ 
#~ pl.figure(4)
#~ pl.plot(POS_tussen[1,(-1,0,1)].flatten(), POS_tussen[2,(-1,0,1)].flatten(), '.')
#~ 
#~ pl.figure(5)
#~ field_show(rho_tussen[0], boxlen)
#~ 
#~ pl.figure(6)
#~ field_show(del_tussen.t[0], boxlen)
#~ 
#~ pl.figure(7)
#~ pl.subplot(221)
#~ field_show(psi_tussen.x.t[0], boxlen)
#~ pl.subplot(222)
#~ field_show(psi_tussen.y.t[0], boxlen)
#~ pl.subplot(223)
#~ field_show(psi_tussen.z.t[0], boxlen)
#~ 
#~ pl.figure(8)
#~ pl.plot(POS[1,(-1,0,1)].flatten(), POS[2,(-1,0,1)].flatten(), '.')
#~ 
#~ pl.figure(9)
#~ pl.plot(POS0[1,(-1,0,1)].flatten(), POS0[2,(-1,0,1)].flatten(), '.')
#~ 
#~ pl.plot(POS3_end[1,(-1,0,1)].flatten(), POS3_end[2,(-1,0,1)].flatten(), '.')

