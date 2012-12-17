#cd dataserver/cubep3m/scratch
#ipy

import struct

gridsize = 64
test_filename = 'rewritten_xv0.ic'

native = np.memmap('xv0.ic', dtype='float32')
pos = native[1:].reshape(64**3,6)[:,:3]
vel = native[1:].reshape(64**3,6)[:,3:]

f = open(test_filename,'wb')

# header
N = len(pos) # Number of DM particles (Npart[1], i.e. the 2nd entry)
header = struct.pack("=I", N)
f.write(header)

# pos & vel
data = np.array(np.hstack((pos, vel)), dtype='float32', order='C')
f.write(data.data)

f.close()
