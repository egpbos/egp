import egp.io, egp.icgen

# First run gadget comparison run:

run_dir_base = "/Users/users/pbos/dataserver/sims"
nproc = 4

run_name = 'cubep3m_vs_gadget'

cosmo = egp.icgen.Cosmology('wmap7', trans = 8)
boxlen = 200. # Mpc h^-1
redshift = 63.
gridsize = 64

seed = 2522572538

ps = egp.icgen.CosmoPowerSpectrum(cosmo)
ps.normalize(boxlen**3)

delta = egp.icgen.GaussianRandomField(ps, boxlen, gridsize, seed=seed)
psi = egp.icgen.DisplacementField(delta)
pos, vel = egp.icgen.zeldovich(redshift, psi, cosmo)

rhoc = egp.toolbox.critical_density(cosmo) # M_sun Mpc^-3 h^2
particle_mass = cosmo.omegaM * rhoc * boxlen**3 / gridsize**3 / 1e10

ic_filename = '/Users/users/pbos/dataserver/sims/ICs/%s_%iMpc_%i_%i.dat' % (run_name, boxlen, gridsize, seed)

egp.io.write_gadget_ic_dm(ic_filename, pos.reshape((3,gridsize**3)).T, vel.reshape((3,gridsize**3)).T, particle_mass, redshift, boxlen, cosmo.omegaM, cosmo.omegaL, cosmo.h)

egp.io.prepare_gadget_run(boxlen, gridsize, cosmo, ic_filename, redshift, run_dir_base, run_name+"_%i" % seed, nproc)

raise SystemExit

# Then compare:
from mayavi import mlab

filename_gadget = '/net/dataserver1/data/users/pbos/sims/cubep3m_vs_gadget_2522572538/snap_008'

filename_cubep3m = '/net/dataserver1/data/users/pbos/cubep3m/test1.12_disp_mesh/output/0.000xv0.dat'

cubep3m = egp.io.CubeP3MData(filename_cubep3m)
gadget = egp.io.GadgetData(filename_gadget)
# zeldovich:
posZ, velZ = egp.icgen.zeldovich(0., psi, cosmo)

mlab.figure(1)
mlab.points3d(gadget.pos[:,0], gadget.pos[:,1], gadget.pos[:,2], mode='point', opacity=0.5)

mlab.figure(2)
mlab.points3d(cubep3m.pos[:,0], cubep3m.pos[:,1], cubep3m.pos[:,2], mode='point', opacity=0.5)


mlab.figure(3)
mlab.points3d(posZ[0], posZ[1], posZ[2], mode='point', opacity=0.5)
