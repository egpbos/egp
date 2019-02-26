import os

class CubeP3MData(object):
    """
    Load a CubeP3M checkpoint file and gather related meta-data from the
    parameter files present in the run directory. The run directory is
    assumed to be one directory up from the checkpoint's location. If not,
    you need to specify the run_path in the initialization.
    
    Default instantiation argument is filename, including full path.
    """
    def __init__(self, filename, run_path = None):
        self.filename = os.path.abspath(filename)
        if not run_path:
            self.run_path = os.path.dirname(self.filename)[:-6] # cut off "output"
        self.load_metadata()
        self.Ntotal = self.metadata['N']
        self.offset = 11 + self.metadata['pp_run'] # file offset due to header
        xvint = np.memmap(self.filename, dtype='int32', mode='r')
        N = xvint[0]
        if N != self.Ntotal:
            self.Ntotal = N
            print "N.B.: particles have been deleted from the ICs!\nAdjusted particle number from %i to %i." % (self.metadata['N'], N)
        self.xv = np.memmap(self.filename, dtype='float32', mode='r', offset = self.offset*4)
    
    order = property()
    @order.setter
    def order(self, order):
        self._order = order
    @order.getter
    def order(self):
        try:
            return self._order
        except AttributeError:
            # Load particle IDs and use them to build an ordering array that
            # will be used to order the other data by ID.
            if self.metadata['pid_flag']:
                pid_filename = self.filename[:self.filename.find('xv')]+'PID0.dat'
                idarray = np.memmap(pid_filename, dtype='int64', offset=self.offset)
                self.order = np.argsort(idarray).astype('uint32')
                del idarray
            else:
                self.order = np.arange(self.Ntotal)
            return self._order
    
    pos = property()
    @pos.setter
    def pos(self, pos):
        self._pos = pos
    @pos.getter
    def pos(self):
        try:
            return self._pos
        except AttributeError:
            # Load the particle positions into a NumPy array called self._pos,
            # ordered by ID number.
            self.pos = self.xv.reshape(self.Ntotal, 6)[:,:3]
            self.pos *= self.metadata['boxlen']/self.metadata['nc'] # Mpc h^-1
            self.pos = self.pos[self.order]
            return self._pos

    vel = property()
    @vel.setter
    def vel(self, vel):
        self._vel = vel
    @vel.getter
    def vel(self):
        try:
            return self._vel
        except AttributeError:
            # Load the particle velocities into a NumPy array called self._vel,
            # ordered by ID number.
            self.vel = self.xv.reshape(self.Ntotal, 6)[:,3:]
            self.vel *= (150*(1+self.metadata['redshift']) * self.metadata['boxlen'] / self.metadata['nc'] * np.sqrt(self.metadata['omega_m'])) # km/s
            self.vel = self.vel[self.order]
            return self._vel
        
    def load_metadata(self):
        """Loads the pickled parameters. Assumes that simulation was setup with
        this code, which saves parameters as a Python pickle file."""
        self.metadata = pickle.load(open(self.run_path+'parameters.pickle', 'rb'))
