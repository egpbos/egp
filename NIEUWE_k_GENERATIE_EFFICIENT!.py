k12 = np.fft.fftfreq(gridsize, 1/dk/gridsize)
k3 = k12[:halfgrid+1].abs()
knew = np.sqrt(k12[:halfgrid+1]**2 + k12[:,np.newaxis]**2 + k12[:,np.newaxis,np.newaxis]**2)

a = np.arange(4)
a[:,np.newaxis,np.newaxis].reshape((4,1,1))
a[:,np.newaxis,np.newaxis].reshape((1,4,1))
a[:,np.newaxis,np.newaxis].reshape((1,1,4))

np.repeat(a[:,np.newaxis], 4, axis=1)

# CHECK OOK http://stackoverflow.com/questions/5564098/repeat-numpy-array-without-replicating-data