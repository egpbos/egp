#/usr/bin/env python
import numpy as np
import egp.icgen, egp.io
import struct
import tarfile
import os

## cosmological simulation parameters
gridsize = 64 # amount of particles = gridsize**3
boxlen = 200.0 # Mpc/h
redshift = 63.0

seed = 2522572538

cosmo = egp.icgen.Cosmology('wmap7', trans=8)
ps = egp.icgen.CosmoPowerSpectrum(cosmo)

#snapshots = np.array([ 3.79778903, 2.98354398, 2.04688779, 1.00131278, 0.50730794, 0.2521318, 0.10412542, 0.])
snapshots = np.array([20., 5., 3., 2., 1.5, 1., 0.5, 0.])

## names, paths etc. parameters
run_name = 'test1.10_functie'
run_path_base = '/Users/users/pbos/dataserver/cubep3m/'

## grid parameters
nodes_dim = 1
tiles_node_dim = 2
cores = 8
nf_tile_I = 2

## pos en vel bepalen
delta = egp.icgen.GaussianRandomField(ps, boxlen, gridsize, seed)
psi = egp.icgen.DisplacementField(delta)
pos, vel = egp.icgen.zeldovich(redshift, psi, cosmo)

egp.io.setup_cubep3m_run(pos, vel, cosmo, boxlen, gridsize, redshift, snapshots, run_name, run_path_base, nodes_dim, tiles_node_dim, cores, nf_tile_I = nf_tile_I)
