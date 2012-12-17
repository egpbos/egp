import egp.io, egp.icgen

# First run gadget comparison run:

run_dir_base = "/Users/users/pbos/dataserver/sims"
nproc = 4

run_name = 'cubep3m_vs_gadget'

cosmo = egp.icgen.Cosmology('wmap7')
cosmo.trans = 8
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

# Then run CubeP3M runs from the cubep3m_ic.py file; one with PP and one without.
# These need to be run with PIDs!

# Then compare:

fine_gridsize = 128

filename_gadget = '/Users/users/pbos/dataserver/sims/cubep3m_vs_gadget_2522572538/snap_008'
filename_cubep3m = '/net/dataserver1/data/users/pbos/cubep3m/test1.7_pid/output/0.000xv0.dat'
filename_cubepm = '/net/dataserver1/data/users/pbos/cubep3m/test1.8_pid_no_pp/output/0.000xv0.dat'
filename_cubep3m_id = '/net/dataserver1/data/users/pbos/cubep3m/test1.7_pid/output/0.000PID0.dat'
filename_cubepm_id = '/net/dataserver1/data/users/pbos/cubep3m/test1.8_pid_no_pp/output/0.000PID0.dat'
filename_cubepm9 = '/net/dataserver1/data/users/pbos/cubep3m/test1.9_+0.5_no_pp/output/0.000xv0.dat'
filename_cubepm9_id = '/net/dataserver1/data/users/pbos/cubep3m/test1.9_+0.5_no_pp/output/0.000PID0.dat'

# cubep3m
# pos
cubep3m_x = np.memmap(filename_cubep3m, dtype='float32')
cubep3m_xint = np.memmap(filename_cubep3m, dtype='int32')
offset = len(cubep3m_x)-cubep3m_xint[0]*6
cubep3m_x = cubep3m_x[offset:].reshape(cubep3m_xint[0],6)[:,:3] + 0.5
# N.B.: to compare to Gadget we have to add 0.5 again, because we subtract 0.5
# from the positions in the ICs! For unknown reasons, but this is what the original
# code did...
# id
cubep3m_id_short = np.memmap(filename_cubep3m_id, dtype='int32')
offset = len(cubep3m_id_short)-cubep3m_id_short[0]*2
cubep3m_id = np.memmap(filename_cubep3m_id, dtype='int64', offset = offset*4)
# order pos:
cubep3m_x = cubep3m_x[np.argsort(cubep3m_id)]/fine_gridsize*boxlen

# cubepm
# pos
cubepm_x = np.memmap(filename_cubepm, dtype='float32')
cubepm_xint = np.memmap(filename_cubepm, dtype='int32')
offset = len(cubepm_x)-cubepm_xint[0]*6
cubepm_x = cubepm_x[offset:].reshape(cubepm_xint[0],6)[:,:3] + 0.5
# id
cubepm_id_short = np.memmap(filename_cubepm_id, dtype='int32')
offset = len(cubepm_id_short)-cubepm_id_short[0]*2
cubepm_id = np.memmap(filename_cubepm_id, dtype='int64', offset = offset*4)
# order pos:
cubepm_x = cubepm_x[np.argsort(cubepm_id)]/fine_gridsize*boxlen

# gadget
gadget = egp.io.GadgetData(filename_gadget)

gadget_x = gadget.pos/1000

dc3c1 = cubep3m_x - cubepm_x
dc3c1 -= 200*(dc3c1 > 150)
dc3c1 += 200*(dc3c1 < -150)

dc3g = cubep3m_x - gadget_x
dc3g -= 200*(dc3g > 150)
dc3g += 200*(dc3g < -150)

dgc1 = gadget_x - cubepm_x
dgc1 -= 200*(dgc1 > 150)
dgc1 += 200*(dgc1 < -150)

# zeldovich
posZ, velZ = egp.icgen.zeldovich(0., psi, cosmo)
posZ = posZ.reshape((3,gridsize**3)).T

dgz = gadget_x - posZ
dgz -= 200*(dgz > 150)
dgz += 200*(dgz < -150)

# cubepm without -0.5
# pos
cubepm9_x = np.memmap(filename_cubepm9, dtype='float32')
cubepm9_xint = np.memmap(filename_cubepm9, dtype='int32')
offset = len(cubepm9_x)-cubepm9_xint[0]*6
cubepm9_x = cubepm9_x[offset:].reshape(cubepm9_xint[0],6)[:,:3]
# id
cubepm9_id_short = np.memmap(filename_cubepm9_id, dtype='int32')
offset = len(cubepm9_id_short)-cubepm9_id_short[0]*2
cubepm9_id = np.memmap(filename_cubepm9_id, dtype='int64', offset = offset*4)
# order pos:
cubepm9_x = cubepm9_x[np.argsort(cubepm9_id)]/fine_gridsize*boxlen

dgc9 = gadget_x - cubepm9_x
dgc9 -= 200*(dgc9 > 150)
dgc9 += 200*(dgc9 < -150)
# DIT MAAKT HET NOG SLECHTER!!

