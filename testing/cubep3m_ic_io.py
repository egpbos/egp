#/usr/bin/env python
import numpy as np
import egp.icgen, egp.io

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
run_name = 'test1.15_read_seed'
run_path_base = '/Users/users/pbos/dataserver/cubep3m/'

## grid parameters
nodes_dim = 1
tiles_node_dim = 2
cores = 8
nf_tile_I = 2

read_displacement_seed = True

## pos en vel bepalen
delta = egp.icgen.GaussianRandomField(ps, boxlen, gridsize, seed)
psi = egp.icgen.DisplacementField(delta)
pos, vel = egp.icgen.zeldovich(redshift, psi, cosmo)

egp.io.setup_cubep3m_run(pos, vel, cosmo, boxlen, gridsize, redshift, snapshots, run_name, run_path_base, cores, nodes_dim, tiles_node_dim, nf_tile_I = nf_tile_I, read_displacement_seed = read_displacement_seed)
# note that after test1.12 the default value for displace_from_mesh was changed to True! To recreate these tests you need to set it to False.
