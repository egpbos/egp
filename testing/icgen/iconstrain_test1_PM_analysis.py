import egp.io
import iconstrain

boxlen = 100.
gridsize_iter = 64
target_pos = np.array([20.,40.,70.])

run_path_base = '/Users/users/pbos/dataserver/cubep3m/iconstrain_scratch/'
run_name = 'step'
simulation = egp.io.CubeP3MData(run_path_base+run_name+'/output/0.000xv0.dat')
radius = 4.0225468329142506
spheregrid = iconstrain.get_peak_particle_indices(target_pos, radius, boxlen, gridsize_iter)

mean_peak_pos = simulation.pos.T.reshape(3,gridsize_iter, gridsize_iter, gridsize_iter)[:,spheregrid].mean(axis=1)

pid_filename = simulation.filename[:simulation.filename.find('xv')]+'PID0.dat'
idarray = np.memmap(pid_filename, dtype='int64', offset=simulation.offset)

from mayavi import mlab

sample = np.arange(64**3)
np.random.shuffle(sample)
sample = sample[:10000]

mlab.points3d(simulation.pos[sample,0], simulation.pos[sample,1], simulation.pos[sample,2], mode='point')
pl.figure(1)
pl.plot(simulation.pos[:,1], simulation.pos[:,2], ',')

gadget = egp.io.GadgetData('/Users/users/pbos/dataserver/sims/run108_2522572538/snap_009')

gsample = np.arange(256**3)
np.random.shuffle(gsample)
gsample = gsample[:64**3]

pl.figure(2)
pl.plot(gadget.pos[gsample,0], gadget.pos[gsample,1], ',')
