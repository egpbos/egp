class CubeP3MData(object):
    """
    Load a CubeP3M checkpoint file and gather related meta-data from the
    parameter files present in the run directory. The run directory is
    assumed to be one step up from the checkpoint's location. If not,
    you need to specify the run_path in the initialization.
    
    Default instantiation argument is filename, including full path.
    """
    def __init__(self, firstfile):
        self.filename = abspath(filename)
        self.loadHeaders()
        self.Ntotal = sum(self.header[0]['Nall'])
    
    order = property()
    @order.getter
    def order(self):
        try:
            return self._order
        except AttributeError:
            # Load particle IDs and use them to build an ordering array that
            # will be used to order the other data by ID.
            idarray = np.empty(self.Ntotal, dtype='uint32')
            
            Ncount = 0
            
            for header in self.header:
                Npart = sum(header['Npart'])
                if self.gtype == 2:
                    offset = (4*16 + (8 + 256) + (8 + Npart*3*4)*2)
                else:
                    offset = ((8 + 256) + (8 + Npart*3*4)*2)
                
                memmap = np.memmap(header['filename'], dtype='uint32', mode='r', offset=offset)
                
                idarray[Ncount:Ncount+Npart] = memmap[1:1+Npart]
                Ncount += Npart
                del memmap
            
            self.order = np.argsort(idarray).astype('uint32')
            del idarray
            return self._order
    @order.setter
    def order(self, order):
        self._order = order
    
    pos = property()
    @pos.getter
    def pos(self):
        try:
            return self._pos
        except AttributeError:
            # Load the particle positions into a NumPy array called self._pos,
            # ordered by ID number.
            self.pos = np.empty((3,self.Ntotal), dtype='float32').T
            Ncount = 0
            
            for header in self.header:
                Npart = sum(header['Npart'])
                if self.gtype == 2:
                    offset = (2*16 + (8 + 256))
                else:
                    offset = (8 + 256)
                
                memmap = np.memmap(header['filename'], dtype='float32', mode='r', offset=offset)
                
                self.pos[Ncount:Ncount+Npart] = memmap[1:1+3*Npart].reshape((Npart,3))
                Ncount += Npart
                del memmap
            
            self.pos = self.pos[self.order]
            return self._pos
    @pos.setter
    def pos(self, pos):
        self._pos = pos
    def loadPos(self, forced=False):
        pass

    vel = property()
    @vel.getter
    def vel(self):
        try:
            return self._vel
        except AttributeError:
            # Load the particle velocities into a NumPy array called self._vel,
            # ordered by ID number.
            self.vel = np.empty((3,self.Ntotal), dtype='float32').T
            Ncount = 0
            
            for header in self.header:
                Npart = sum(header['Npart'])
                if self.gtype == 2:
                    offset = 3*16 + (8 + 256) + (8 + 3*4*Npart)
                else:
                    offset = (8 + 256) + (8 + 3*4*Npart)
                
                memmap = np.memmap(header['filename'], dtype='float32', mode='r', offset=offset)
                
                self.vel[Ncount:Ncount+Npart] = memmap[1:1+3*Npart].reshape((Npart,3))
                Ncount += Npart
                del memmap
            
            self.vel = self.vel[self.order]
            return self._vel
    @vel.setter
    def vel(self, vel):
        self._vel = vel
    def loadVel(self, forced=False):
        pass
    
    def detectGType(self):
        """Detects Gadget file type (type 1 or 2; resp. without or with the 16
        byte block headers)."""
        filename = self.firstfile
        f = open(filename, 'rb')
        firstbytes = struct.unpack('I',f.read(4))
        if firstbytes[0] == 8:
            self.gtype = 2
        else:
            self.gtype = 1
        f.close()
    
    def loadHeaders(self):
        """Loads file headers of all files in the dataset into memory."""
        self.header = []
        filename = self.firstfile
        self.header.append(getheader(filename, self.gtype))
        if self.header[0]['NumFiles'] > 1:
            basename = filename[:-1]
            for filenr in range(self.header[0]['NumFiles'])[1:]:
                self.header.append(getheader(basename+str(filenr), self.gtype))
    
    def calcPosSph(self, origin=None, centerOrigin=True):
        """Calculate the positions of particles in spherical coordinates. The
        origin is by default at the center of the box, but can be specified by
        supplying an origin=(x,y,z) argument."""
        
        # Implement periodic folding around to keep the origin centered
        
        if not origin:
            center = self.header[0]['BoxSize']/2
            self.sphOrigin = np.array((center,center,center))
        else:
            self.sphOrigin = np.asarray(origin)
        
        self.originCentered = centerOrigin
        
        if not self.posloaded:
            self.loadPos()
            
        x = self.pos[:,0] - self.sphOrigin[0]
        y = self.pos[:,1] - self.sphOrigin[1]
        z = self.pos[:,2] - self.sphOrigin[2]
        
        if self.originCentered:
            box = self.header[0]['BoxSize']
            halfbox = box/2.
            np.putmask(x, x < -halfbox, x + box)
            np.putmask(y, y < -halfbox, y + box)
            np.putmask(z, z < -halfbox, z + box)
            np.putmask(x, x >= halfbox, x - box)
            np.putmask(y, y >= halfbox, y - box)
            np.putmask(z, z >= halfbox, z - box)
            self.posCO = np.vstack((x,y,z)).T
        
        xy2 = x*x + y*y
        xy = np.sqrt(xy2)
        r = np.sqrt(xy2 + z*z)
        phi = np.arctan2(y,x)        # [-180,180] angle in the (x,y) plane
                                    # counterclockwise from x-axis towards y
        theta = np.arctan2(xy,z)    # [0,180] angle from the positive towards
                                    # the negative z-axis
        
        self.posSph = np.vstack((r,phi,theta)).T
        self.posSphCalculated = True
    
    def calcRedshift(self, origin=None, centerOrigin=True):
        """Calculate the redshifts of the particles, i.e. the redshift space
        equivalents of the radial distances. The origin is by default at the
        center of the box, but can be specified by supplying an origin=(x,y,z)
        argument."""
        
        if not (self.posSphCalculated and self.velloaded):
            self.calcPosSph(origin=origin, centerOrigin=centerOrigin)
            self.loadVel()
        
        if self.originCentered:
            x = self.posCO[:,0]
            y = self.posCO[:,1]
            z = self.posCO[:,2]
        else:
            x = self.pos[:,0] - self.sphOrigin[0]
            y = self.pos[:,1] - self.sphOrigin[1]
            z = self.pos[:,2] - self.sphOrigin[2]

        r = self.posSph[:,0]
        
        unitvector_r = np.array([x/r, y/r, z/r]).T
        
        #vR_cartesian = self.vR_cartesian = unitvector_r * self.vel
        # FOUTFOUTFOUTFOUT
        # vR is zo altijd positief; richting gaat verloren
        #vR = self.vR = np.sqrt(np.sum(vR_cartesian**2, axis=1))
        # FOUTFOUTFOUTFOUT
        # Dit lijkt wel te kloppen (en is theoretisch ook correct
        # ook volgens Marsden en Tromba):
        vR = np.sum(unitvector_r * self.vel, axis=1)
        
        H = self.header[0]['HubbleParam']*100
        
        self.redshift = LOSToRedshift(r/1000, vR, H)
        self.redshiftCalculated = True
    
    def calcRedshift2(self, origin=None, centerOrigin=True):
        """Calculate the redshifts of the particles, i.e. the redshift space
        equivalents of the radial distances. The origin is by default at the
        center of the box, but can be specified by supplying an origin=(x,y,z)
        argument."""
        
        if not (self.posSphCalculated and self.velloaded):
            self.calcPosSph(origin=origin, centerOrigin=centerOrigin)
            self.loadVel()
        
        if self.originCentered:
            x = self.posCO[:,0]
            y = self.posCO[:,1]
            z = self.posCO[:,2]
        else:
            x = self.pos[:,0] - self.sphOrigin[0]
            y = self.pos[:,1] - self.sphOrigin[1]
            z = self.pos[:,2] - self.sphOrigin[2]

        r = self.posSph[:,0]
        
        unitvector_r = np.array([x/r, y/r, z/r]).T
        
        #vR_cartesian = self.vR_cartesian = unitvector_r * self.vel
        # FOUTFOUTFOUTFOUT
        # vR is zo altijd positief; richting gaat verloren
        #vR = self.vR = np.sqrt(np.sum(vR_cartesian**2, axis=1))
        # FOUTFOUTFOUTFOUT
        # Dit lijkt wel te kloppen (en is theoretisch ook correct
        # ook volgens Marsden en Tromba):
        vR = np.sum(unitvector_r * self.vel, axis=1)
        
        H = self.header[0]['HubbleParam']*100
        
        self.redshift = LOSToRedshift2(r/1000, vR, H)
        self.redshiftCalculated = True

#    def calcRedshiftSpaceC(self, origin=None, centerOrigin=True):
#        """See calcRedshiftSpace; C++ implementation"""
#
#        if not self.posloaded:
#            self.loadPos()
#        if not self.velloaded:
#            self.loadVel()
#
#        if not origin:
#            center = self.header[0]['BoxSize']/2.
#            origin = np.array((center,center,center))
#
#        self.posZC = np.empty(self.pos.shape, dtype="float64")
#
#        crunch.redshiftspace(self.pos.astype("float64"), self.vel.astype("float64"), self.posZ, len(self.pos), origin, 100*self.header[0]['HubbleParam'])
#        self.posZCCalculated = True
    
    def calcRedshiftSpace(self, origin=None, centerOrigin=True):
        """Convert particle positions to cartesian redshift space, i.e. the
        space in which the redshift is used as the radial distance. The origin
        is by default at the center of the box, but can be specified by
        supplying an origin=(x,y,z) argument."""
        
        if not self.redshiftCalculated:
            self.calcRedshift(origin=origin, centerOrigin=centerOrigin)
        
        r = 1000*redshiftToLOS(self.redshift, self.header[0]['HubbleParam']*100)
        phi = self.posSph[:,1]
        theta = self.posSph[:,2]
        
        if self.originCentered:
            box = self.header[0]['BoxSize']
            halfbox = box/2.
            xZ = r*np.sin(theta)*np.cos(phi) + halfbox
            yZ = r*np.sin(theta)*np.sin(phi) + halfbox
            zZ = r*np.cos(theta) + halfbox
        else:
            xZ = r*np.sin(theta)*np.cos(phi) + self.sphOrigin[0]
            yZ = r*np.sin(theta)*np.sin(phi) + self.sphOrigin[1]
            zZ = r*np.cos(theta) + self.sphOrigin[2]
        
        self.posZ = np.array([xZ, yZ, zZ]).T
        self.posZCalculated = True
    
    
    def calcRedshiftSpace2(self, origin=None, centerOrigin=True):
        """Convert particle positions to cartesian redshift space, i.e. the
        space in which the redshift is used as the radial distance. The origin
        is by default at the center of the box, but can be specified by
        supplying an origin=(x,y,z) argument."""
        
        if not self.redshiftCalculated:
            self.calcRedshift2(origin=origin, centerOrigin=centerOrigin)
        
        r = 1000*redshiftToLOS(self.redshift, self.header[0]['HubbleParam']*100)
        phi = self.posSph[:,1]
        theta = self.posSph[:,2]
        
        if self.originCentered:
            box = self.header[0]['BoxSize']
            halfbox = box/2.
            xZ = r*np.sin(theta)*np.cos(phi) + halfbox
            yZ = r*np.sin(theta)*np.sin(phi) + halfbox
            zZ = r*np.cos(theta) + halfbox
        else:
            xZ = r*np.sin(theta)*np.cos(phi) + self.sphOrigin[0]
            yZ = r*np.sin(theta)*np.sin(phi) + self.sphOrigin[1]
            zZ = r*np.cos(theta) + self.sphOrigin[2]
        
        self.posZ2 = np.array([xZ, yZ, zZ]).T
        self.posZCalculated = True
    
    def calcVelSph(self, origin=None):
        """Calculate the velocities of particles in spherical coordinates. The
        origin is by default at the center of the box, but can be specified by
        supplying an origin=(x,y,z) argument."""
        
        # Need both spherical positions and velocities
        if not (self.posSphCalculated and self.velloaded):
            self.calcPosSph(origin=origin)
            self.loadVel()
        
        x = self.pos[:,0]
        y = self.pos[:,1]
        z = self.pos[:,2]
        r = self.posSph[:,0]
        phi = self.posSph[:,1]
        theta = self.posSph[:,2]

        # self HIERONDER WEER WEGHALEN; HOEFT NIET WORDEN OPGESLAGEN
        self.unitvector_r = np.array([x/r, y/r, z/r]).T
        self.unitvector_phi = np.array([-np.sin(phi), np.cos(phi), np.zeros(len(phi))]).T
        self.unitvector_theta = np.array([np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi), -np.sin(theta)]).T
        
#        self.velSph = 
#        self.velSphCalculated = True
    
    def slicePos(self, origin=None, thickness=0.1, plane=None):
        """Make three slices of the particle positions around the origin
        (defaults to the center of the box). Thickness is given in fraction of
        boxsize. Origin can be changed by giving an origin=(x,y,z) argument.
        If spherical coordinates are calculated the origin of those coordinates
        will be taken.
        TO BE IMPLEMENTED:
        The three slices are taken in three perpendicular planes. If the
        plane=((phi1,theta1),(phi2,theta2)) argument is given these
        vectors will be taken to span one of the planes on, through the origin.
        The others will then again be perpendicular to this."""
        
        if not self.posloaded:
            self.loadPos()
        
        if not (origin or self.posSphCalculated):
            center = self.header[0]['BoxSize']/2
            origin = np.array((center,center,center))
        elif self.posSphCalculated:
            origin = self.sphOrigin
        else:
            origin = np.asarray(origin)
        
        self.posSlice1, self.posSlice2, self.posSlice3 = takeOrthSlices(self.pos, self.header[0]['BoxSize']*thickness, origin)
        self.posSliced = True
    
    def slicePosZ(self, origin=None, thickness=0.1, plane=None):
        """Make three slices of the particle positions around the origin
        (defaults to the center of the box). Thickness is given in fraction of
        boxsize. Origin can be changed by giving an origin=(x,y,z) argument.
        If spherical coordinates are calculated the origin of those coordinates
        will be taken.
        TO BE IMPLEMENTED:
        The three slices are taken in three perpendicular planes. If the
        plane=((phi1,theta1),(phi2,theta2)) argument is given these
        vectors will be taken to span one of the planes on, through the origin.
        The others will then again be perpendicular to this."""

        if not self.posZCalculated:
            self.calcRedshiftSpace()

        if not (origin or self.posSphCalculated):
            center = self.header[0]['BoxSize']/2
            origin = np.array((center,center,center))
        elif self.posSphCalculated:
            origin = self.sphOrigin
        else:
            origin = np.asarray(origin)
        
        self.posZSlice1, self.posZSlice2, self.posZSlice3 = takeOrthSlices(self.posZ, self.header[0]['BoxSize']*thickness, origin)
        self.posZSliced = True
        
    
    #~ def calcDensity(self, gridsize):
        #~ """Calculate density on a regular grid of gridsize cubed based on
        #~ particle positions in regular space. Uses TSC algorithm."""
        #~ 
        #~ if not self.posloaded:
            #~ self.loadPos()
        #~ 
        #~ mass = sum(self.header[0]['Massarr']) # FIX!!! Very dangerous!
        #~ 
        #~ self.rho = TSCDensity(self.pos, gridsize, self.header[0]['BoxSize'], mass)
        #~ self.densityGridsize = gridsize
    #~ 
    #~ def calcDensityOld(self, gridsize):
        #~ """Calculate density on a regular grid of gridsize cubed based on
        #~ particle positions in regular space. Uses TSC algorithm."""
#~ 
        #~ if not self.posloaded:
            #~ self.loadPos()
#~ 
        #~ mass = sum(self.header[0]['Massarr']) # FIX!!! Very dangerous!
#~ 
        #~ self.rhoOld = TSCDensityOld(self.pos, gridsize, self.header[0]['BoxSize'], mass)
        #~ self.densityGridsizeOld = gridsize
    #~ 
    #~ def calcDensityZ(self, gridsize):
        #~ """Calculate density on a regular grid of gridsize cubed based on
        #~ particle positions in redshift space. Uses TSC algorithm."""
        #~ 
        #~ if not self.posZCalculated:
            #~ self.calcRedshiftSpace()
        #~ 
        #~ mass = sum(self.header[0]['Massarr']) # FIX!!! Very dangerous!
        #~ 
        #~ self.rhoZ = TSCDensity(self.posZ, gridsize, self.header[0]['BoxSize'], mass)
        #~ self.densityZGridsize = gridsize
    #~ 
    #~ def sliceDensity(self, origin=None, thickness = 1, plane=None):
        #~ """Same as slicePos, but for density. Sums over planes in cube and
        #~ returns slices as sums. Thickness is given in absolute number of
        #~ slices."""
        #~ # N.B.: density slices seem to be transposed compared to the position
        #~ # arrays! For comparison in Gnuplot, the density slice is transposed.
        #~ if not self.densityGridsize:
            #~ print "Density needs to be calculated first!"
            #~ raise SystemExit
        #~ 
        #~ rhodim = self.densityGridsize
        #~ if not origin:
            #~ origin = np.array([rhodim/2, rhodim/2, rhodim/2])
        #~ 
        #~ self.rhoSlice1 = np.sum(self.rho[origin[0] - (thickness+1)/2 : origin[0] + thickness/2, :, :], axis=0).T
        #~ self.rhoSlice2 = np.sum(self.rho[:, origin[1] - (thickness+1)/2 : origin[1] + thickness/2, :], axis=1).T
        #~ self.rhoSlice3 = np.sum(self.rho[:, :, origin[2] - (thickness+1)/2 : origin[2] + thickness/2], axis=2).T
    #~ 
    #~ def sliceDensityZ(self, origin=None, thickness = 1, plane=None):
        #~ """Same as slicePosZ, but for density. Sums over planes in cube and
        #~ returns slices as sums. Thickness is given in absolute number of
        #~ slices."""
        #~ # N.B.: density slices seem to be transposed compared to the position
        #~ # arrays! For comparison in Gnuplot, the density slice is transposed.
        #~ if not self.densityZGridsize:
            #~ print "Density needs to be calculated first!"
            #~ raise SystemExit
#~ 
        #~ rhodim = self.densityZGridsize
        #~ if not origin:
            #~ origin = np.array([rhodim/2, rhodim/2, rhodim/2])
#~ 
        #~ self.rhoZSlice1 = np.sum(self.rhoZ[origin[0] - (thickness+1)/2 : origin[0] + thickness/2, :, :], axis=0).T
        #~ self.rhoZSlice2 = np.sum(self.rhoZ[:, origin[1] - (thickness+1)/2 : origin[1] + thickness/2, :], axis=1).T
        #~ self.rhoZSlice3 = np.sum(self.rhoZ[:, :, origin[2] - (thickness+1)/2 : origin[2] + thickness/2], axis=2).T
    
    def saveNormalPosAsErwin(self, filename):
        """Save the position data as an Erwin-type binary file (Npart|x|y|z|
        [m]; normalized coordinates)."""
        self.loadPos()
        savePosAsErwin(self.pos, self.header[0]['BoxSize'], filename)
    
    def saveNormalPosAsIfrit(self, filename, sample=None):
        """Saves as an IFrIT binary file."""
        self.loadPos()
        boxsize = self.header[0]['BoxSize']
        if sample:
            savePosAsIfrit(self.pos[np.random.randint(0,len(self.pos),sample)], boxsize, filename)
        else:
            savePosAsIfrit(self.pos, boxsize, filename)
