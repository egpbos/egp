#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
io.py
/io/ module in the /egp/ package.
  
Created by Evert Gerardus Patrick Bos.
Copyright (c) 2012. All rights reserved.
"""

# imports
import struct, numpy as np, os, stat
from os.path import abspath
import pyublas
import crunch


# constants
__version__ = "0.3, August 2012"


# exception classes
# interface functions
# classes

class GadgetData(object):
    """
    An instance of existing Gadget data, loaded from a Gadget type binary file.
    Instantiation argument is firstfile: the name of the first file of the data
    set (e.g. snapshot_000, snapshot_000.0, lcdm_gas, lcdm_gas.0, etc.),
    including path.
    """
    def __init__(self, firstfile):
        self.firstfile = abspath(firstfile)
        self.detectGType()
        self.loadHeaders()
        self.Ntotal = sum(self.header[0]['Nall'])
        self.posSphCalculated = False
        self.velSphCalculated = False
        self.redshiftCalculated = False
        self.posZCalculated = False
        self.posSliced = False
        self.posZSliced = False
        self.densityGridsize = None
        self.densityZGridsize = None
        self.originCentered = True
    
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


class SubFindHaloes(object):
    def __init__(self, firstfile):
        self.firstfile = abspath(firstfile)
        self.loadData()
    
    def loadData(self):
        """Loads file headers and data of all files in the dataset into
        memory."""
        filename = self.firstfile
        tab0l = np.memmap(filename, dtype='int32', mode='r')
        tab0file = file(filename, 'rb')
        
        self.TotNgroups = tab0l[1]
        tab0file.seek(12)
        self.TotNids = struct.unpack('=q', tab0file.read(8))
        self.Nfiles = tab0l[5]
        self.TotNsubhalos = tab0l[7]

        del tab0l
        tab0file.close()
        
        offset_fof = 0
        offset_sub = 0
        
        self.HaloLen = np.zeros(self.TotNgroups, dtype='int32')
        self.NsubPerHalo = np.zeros(self.TotNgroups, dtype='int32')
        self.FirstSubOfHalo = np.zeros(self.TotNgroups, dtype='int32')
        self.Halo_M_Mean200 = np.zeros(self.TotNgroups, dtype='float32')
        self.Halo_R_Mean200 = np.zeros(self.TotNgroups, dtype='float32')
        self.Halo_M_Crit200 = np.zeros(self.TotNgroups, dtype='float32')
        self.Halo_R_Crit200 = np.zeros(self.TotNgroups, dtype='float32')
        self.Halo_M_TopHat200 = np.zeros(self.TotNgroups, dtype='float32')
        self.Halo_R_TopHat200 = np.zeros(self.TotNgroups, dtype='float32')
        self.HaloCont = np.zeros(self.TotNgroups, dtype='float32')
        
        self.HaloPos = np.zeros((self.TotNgroups,3), dtype='float32')
        self.HaloMass = np.zeros(self.TotNgroups, dtype='float32')
        
        self.SubLen = np.zeros(self.TotNsubhalos, dtype='int32')
        self.SubOffset = np.zeros(self.TotNsubhalos, dtype='int32')
        self.SubParentHalo = np.zeros(self.TotNsubhalos, dtype='int32')
        self.SubPos = np.zeros((self.TotNsubhalos,3), dtype='float32')
        self.SubVel = np.zeros((self.TotNsubhalos,3), dtype='float32')
        self.SubVelDisp = np.zeros(self.TotNsubhalos, dtype='float32')
        self.SubVmax = np.zeros(self.TotNsubhalos, dtype='float32')
        self.SubSpin = np.zeros((self.TotNsubhalos,3), dtype='float32')
        self.SubMostBoundID = np.zeros(self.TotNsubhalos, dtype='int64') # N.B.
        self.SubHalfMass = np.zeros(self.TotNsubhalos, dtype='float32')
        self.SubMassTab = np.zeros((self.TotNsubhalos,6), dtype='float32')

        self.SubTMass = np.zeros(self.TotNsubhalos, dtype='float32')
        
        for i in range(self.Nfiles):
            ifile = str(i)
            
            tabl = np.memmap(filename[:-1]+ifile, dtype='int32', mode='r')
            tabf = np.memmap(filename[:-1]+ifile, dtype='float32', mode='r')
            tabul = np.memmap(filename[:-1]+ifile, dtype='uint32', mode='r')
            tabfile = file(filename[:-1]+ifile, 'rb')
            
            Ngroups = tabl[0]
            Nids = tabl[2]
            Nsubhalos = tabl[6]
            
            if Ngroups:
                b = 8
                f = 0
                self.HaloLen[offset_fof:offset_fof+Ngroups] = tabl[b+f*Ngroups:b+(f+1)*Ngroups] # Group Length
                f += 2 # Group member ID offset
                self.HaloMass[offset_fof:offset_fof+Ngroups] = tabf[b+f*Ngroups:b+(f+1)*Ngroups] # FoF Mass
                f += 1
                self.HaloPos[offset_fof:offset_fof+Ngroups,:] = tabf[b+f*Ngroups:b+(f+3)*Ngroups].reshape((Ngroups,3)) # FoF Position
                f += 3
                self.Halo_M_Mean200[offset_fof:offset_fof+Ngroups] = tabf[b+f*Ngroups:b+(f+1)*Ngroups]; f += 1
                self.Halo_R_Mean200[offset_fof:offset_fof+Ngroups] = tabf[b+f*Ngroups:b+(f+1)*Ngroups]; f += 1
                self.Halo_M_Crit200[offset_fof:offset_fof+Ngroups] = tabf[b+f*Ngroups:b+(f+1)*Ngroups]; f += 1
                self.Halo_R_Crit200[offset_fof:offset_fof+Ngroups] = tabf[b+f*Ngroups:b+(f+1)*Ngroups]; f += 1
                self.Halo_M_TopHat200[offset_fof:offset_fof+Ngroups] = tabf[b+f*Ngroups:b+(f+1)*Ngroups]; f += 1
                self.Halo_R_TopHat200[offset_fof:offset_fof+Ngroups] = tabf[b+f*Ngroups:b+(f+1)*Ngroups]; f += 2 # Halo contermination count
                self.HaloCont[offset_fof:offset_fof+Ngroups] = tabf[b+f*Ngroups:b+(f+1)*Ngroups]; f += 1
                self.NsubPerHalo[offset_fof:offset_fof+Ngroups] = tabl[b+f*Ngroups:b+(f+1)*Ngroups]; f += 1
                self.FirstSubOfHalo[offset_fof:offset_fof+Ngroups] = tabl[b+f*Ngroups:b+(f+1)*Ngroups]; f += 1
                offset_fof += Ngroups
            
            if Nsubhalos:
                b = 8 + f*Ngroups
                f = 0
                self.SubLen[offset_sub:offset_sub+Nsubhalos] = tabl[b+f*Nsubhalos:b+(f+1)*Nsubhalos]; f += 1
                self.SubOffset[offset_sub:offset_sub+Nsubhalos] = tabl[b+f*Nsubhalos:b+(f+1)*Nsubhalos]; f += 1
                self.SubParentHalo[offset_sub:offset_sub+Nsubhalos] = tabl[b+f*Nsubhalos:b+(f+1)*Nsubhalos]; f += 1
                self.SubTMass[offset_sub:offset_sub+Nsubhalos] = tabf[b+f*Nsubhalos:b+(f+1)*Nsubhalos]; f += 1
                self.SubPos[offset_sub:offset_sub+Nsubhalos,:] = tabf[b+f*Nsubhalos:b+(f+3)*Nsubhalos].reshape((Nsubhalos,3)); f += 3
                self.SubVel[offset_sub:offset_sub+Nsubhalos,:] = tabf[b+f*Nsubhalos:b+(f+3)*Nsubhalos].reshape((Nsubhalos,3)); f += 6 # Center of Mass
                self.SubSpin[offset_sub:offset_sub+Nsubhalos,:] = tabf[b+f*Nsubhalos:b+(f+3)*Nsubhalos].reshape((Nsubhalos,3)); f += 3
                self.SubVelDisp[offset_sub:offset_sub+Nsubhalos] = tabf[b+f*Nsubhalos:b+(f+1)*Nsubhalos]; f += 1
                self.SubVmax[offset_sub:offset_sub+Nsubhalos] = tabf[b+f*Nsubhalos:b+(f+1)*Nsubhalos]; f += 2 # Vmax Radius
                self.SubHalfMass[offset_sub:offset_sub+Nsubhalos] = tabf[b+f*Nsubhalos:b+(f+1)*Nsubhalos]; f += 1
                self.SubMostBoundID[offset_sub:offset_sub+Nsubhalos] = tabul[b+f*Nsubhalos:b+(f+1)*Nsubhalos]; f += 2 # GroupNumber
                self.SubMassTab[offset_sub:offset_sub+Nsubhalos,:] = tabf[b+f*Nsubhalos:b+(f+6)*Nsubhalos].reshape((Nsubhalos,6))
                offset_sub += Nsubhalos
            
            del tabl, tabf, tabul
            tabfile.close()


class WVFEllipses(object):
    def __init__(self, filename, boxsize, bins=None):
        self.filename = abspath(filename)
        self.boxsize = boxsize
        self.bins = bins
        self.loadData()
        self.calcShapes()
    
    def loadData(self):
        self.N = np.memmap(self.filename, dtype='int32')[0]
        self.data = np.memmap(self.filename, dtype='float32')[1:].reshape((self.N,25))
        if self.bins:
            self.pos = self.data[:,1:4]/self.bins*self.boxsize
            self.x = self.data[:,1]/self.bins*self.boxsize
            self.y = self.data[:,2]/self.bins*self.boxsize
            self.z = self.data[:,3]/self.bins*self.boxsize
        self.a = self.boxsize/np.sqrt(np.abs(self.data[:,4]))
        self.b = self.boxsize/np.sqrt(np.abs(self.data[:,5]))
        self.c = self.boxsize/np.sqrt(np.abs(self.data[:,6]))
        self.inertia = self.data[:,16:25].reshape((self.N,3,3))
    
    def calcShapes(self):
        self.vol = 4.*np.pi/3*self.a*self.b*self.c
        self.r = (self.vol*3/4/np.pi)**(1/3.)
        self.nu = self.c/self.a
        self.e = 1 - self.nu
        self.p = self.b/self.a    # "oblateness"
        self.q = self.c/self.b    # "prolateness"


# functions:
def write_gadget_ic_dm(filename, pos, vel, mass, redshift, boxsize = 0.0, om0 = 0.314, ol0 = 0.686, h0 = 0.71):
    """
    For pure dark matter simulations only, i.e. no gas! Some notes:
    - Only give boxsize if periodic boundary conditions are used; otherwise 0.0
    - The dark matter particles are assumed to have the same masses; the mass
      parameter is therefore one number, not an array of masses for each
      particle.
    - The pos and vel arrays must have (Python) shape (N,3), where pos[:,0] will
      then be all the X coordinates, pos[:,1] the Y, etc.
    - pos and boxsize are in units of Mpc h^-1, vel is in units of km/s, mass
      is in units of 10^10 M_sol h^-1
    Note: this function writes "type 2" Gadget files, i.e. with 4 character
    block names before each block (see section 6.2 of Gadget manual).
    """
    # Position is converted to kpc h^-1, the default unit in Gadget simulations.
    pos = 1000*np.array(pos, dtype='float32', order='F')
    vel = np.array(vel, dtype='float32', order='F')
    # Velocity correction, needed in GADGET for comoving (cosm.) simulation
    # See thread: http://www.mpa-garching.mpg.de/gadget/gadget-list/0111.html
    # Also, manual p. 32; what we have here is the peculiar velocity (as defined
    # here: http://www.mpa-garching.mpg.de/gadget/gadget-list/0113.html), but
    # what we want is the (old) internal Gadget velocity u (see manual p. 32).
    vel *= np.sqrt(1+redshift)
    
    f = open(filename,'wb')
    BS = {}
    BS['desc'] = '=I4sII'
    BS['HEAD'] = '=I6I6dddii6iiiddddii6ii60xI'
    
    # Values for HEAD
    N2 = len(pos) # Number of DM particles (Npart[1], i.e. the 2nd entry)
    
    # Make HEAD and write to file
    toFileDesc = struct.pack(BS['desc'], 8L, 'HEAD', 264L, 8L)
    toFile = struct.pack(BS['HEAD'], 256L, 0L, N2, 0L, 0L, 0L, 0L, 0.0, mass, 0.0, 0.0, 0.0, 0.0, 1./(1+redshift), redshift, 0, 0, 0, N2, 0, 0, 0, 0, 0, 1, boxsize, om0, ol0, h0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 256L)
    
    f.write(toFileDesc+toFile)
    
    # Make POS, VEL and ID and write to file
    toFileDescPOS = struct.pack(BS['desc'], 8L, 'POS ', 3*4*N2+8, 8L)
    toFileDescVEL = struct.pack(BS['desc'], 8L, 'VEL ', 3*4*N2+8, 8L)
    toFileDescID  = struct.pack(BS['desc'], 8L, 'ID  ', 4*N2+8, 8L)
    
    toFilePOSVELsize = struct.pack('I', 3*4*N2) # Just the block sizes
    toFileIDsize     = struct.pack('I', 4*N2)
    
    # reshape pos and vel for convenience
    pos = pos.reshape(N2*3)
    vel = vel.reshape(N2*3)
    
    # write positions:
    f.write(toFileDescPOS+toFilePOSVELsize)
    f.write(pos.data)
    f.write(toFilePOSVELsize)
    
    # write velocities:
    f.write(toFileDescVEL+toFilePOSVELsize)
    f.write(vel.data)
    f.write(toFilePOSVELsize)
    
    # write IDs:
    f.write(toFileDescID +toFileIDsize)
    IDs = np.arange(N2, dtype='int32')
    f.write(IDs.data)
    f.write(toFileIDsize)
    
    f.close()

def prepare_gadget_run(boxlen, gridsize, cosmo, ic_file, redshift_begin, run_dir_base, run_name, nproc, output_list_filename = 'outputs_main.txt', DE_file = 'wdHdGHz_LCDM_bosW7.txt', ic_format = 2, time_max = 1.0, softening_factor = 22.5*768/300000., time_limit_cpu = 864000, resubmit_on = 0, resubmit_command = '0', cpu_time_bet_restart_file = 3600, part_alloc_factor = 1.4, tree_alloc_factor = 0.8, buffer_size = 100, gadget_executable = "/net/schmidt/data/users/pbos/sw/code/gadget/gadget3Sub_512_SL6/P-Gadget3_512"):
    """Arguments:
    boxlen (Mpc h^-1)
    cosmo (Cosmology object)
    ic_file (path)
    redshift_begin
    run_dir_base (directory path)
    run_name (sub directory name)
    nproc (number of processors)
    output_list_filename (filename w.r.t. run_dir_base)
    DE_file (filename w.r.t. run_dir_base)
    ic_format (1 or 2)
    time_max (expansion factor a)
    softening_factor (fraction of mean interparticle distance)
    time_limit_cpu (seconds)
    resubmit_on (0 or 1)
    resubmit_command (path)
    cpu_time_bet_restart_file (seconds)
    part_alloc_factor
    tree_alloc_factor
    buffer_size (MB)
    gadget_executable (file path)
    
    Note that run_dir_base is not the directory where the simulation will be
    run; that is run_dir_base+run_name; the run_name directory will be created
    by this function.
    """
    # Length units are converted to kpc h^-1, the default Gadget unit.
    boxlen *= 1000
    
    output_dir = run_dir_base+'/'+run_name
    os.mkdir(output_dir)
    
    omegaM = cosmo.omegaM
    omegaL = cosmo.omegaL
    omegaB = cosmo.omegaB
    hubble = cosmo.h
    
    parameter_filename = run_dir_base+'/'+run_name+'.par'
    run_script_filename = run_dir_base+'/'+run_name+'.sh'
    restart_script_filename = run_dir_base+'/'+run_name+'_restart.sh'
    
    output_list_filename = run_dir_base+'/'+output_list_filename
    DE_file = run_dir_base+'/'+DE_file

    time_begin = 1/(redshift_begin+1)

    # Softening: based on Dolag's ratios
    # default ~ 1/17.3 of the mean ipd
    softening = softening_factor*boxlen/gridsize
    softening_max_phys = softening/3
    
    # the actual parameter file:
    par_file_text = """\
%%%%%%%%%% In-/output
InitCondFile          %(ic_file)s
OutputDir           %(output_dir)s
OutputListFilename  %(output_list_filename)s


%%%%%%%%%% Characteristics of run & Cosmology
TimeBegin             %(time_begin)f
TimeMax                  %(time_max)f
BoxSize               %(boxlen)f

Omega0                  %(omegaM)f
OmegaLambda           %(omegaL)f
OmegaBaryon           %(omegaB)f
HubbleParam           %(hubble)f   ; only needed for cooling


%%%%%%%%%% DE (GadgetXXL)
DarkEnergyFile          %(DE_file)s
%%DarkEnergyParam        -0.4
VelIniScale                1.0


%%%%%%%%%% Softening lengths
MinGasHsmlFractional     0.5  %% minimum csfc smoothing in terms of the gravitational softening length

%% ~ 1/20 of mean ipd (Dolag: 22.5 for ipd of 300000/768)
SofteningGas       %(softening)f
SofteningHalo      %(softening)f
SofteningDisk      0.0
SofteningBulge     0.0        
SofteningStars     %(softening)f
SofteningBndry     0

%% ~ 1/3 of the above values
SofteningGasMaxPhys       %(softening_max_phys)f
SofteningHaloMaxPhys      %(softening_max_phys)f
SofteningDiskMaxPhys      0.0  %% corr.to EE81_soft = 70.0
SofteningBulgeMaxPhys     0.0         
SofteningStarsMaxPhys     %(softening_max_phys)f
SofteningBndryMaxPhys     0


%%%%%%%%%% Time/restart stuff
TimeLimitCPU             %(time_limit_cpu)i
ResubmitOn               %(resubmit_on)i
ResubmitCommand          %(resubmit_command)s
CpuTimeBetRestartFile    %(cpu_time_bet_restart_file)s


%%%%%%%%%% Memory
PartAllocFactor       %(part_alloc_factor)f
TreeAllocFactor       %(tree_alloc_factor)f
BufferSize            %(buffer_size)i


ICFormat                   %(ic_format)i


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%% Usually don't edit below here %%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%% In-/output parameters
OutputListOn               1
SnapFormat                 2
NumFilesPerSnapshot        1
NumFilesWrittenInParallel  1


%%%% Default filenames
SnapshotFileBase        snap
EnergyFile        energy.txt
InfoFile          info.txt
TimingsFile       timings.txt
CpuFile           cpu.txt
RestartFile       restart


%%%% Misc. options
ComovingIntegrationOn 1
CoolingOn 0
PeriodicBoundariesOn   1


%%%% Output frequency (when not using output list)
TimeBetSnapshot        1.04912649189365
TimeOfFirstSnapshot    0.090909091
TimeBetStatistics      0.02
 

%%%% Accuracy of time integration
TypeOfTimestepCriterion 0        %% Not used option in Gadget2 (left over from G1)
ErrTolIntAccuracy       0.05    %% Accuracy of timestep criterion
MaxSizeTimestep        0.1        %% Maximum allowed timestep for cosmological simulations
                                %% as a fraction of the current Hubble time (i.e. dln(a))
MinSizeTimestep        0        %% Whatever
MaxRMSDisplacementFac  0.25        %% Something


%%%% Tree algorithm and force accuracy
ErrTolTheta            0.45
TypeOfOpeningCriterion 1
ErrTolForceAcc         0.005
TreeDomainUpdateFrequency    0.025
%% DomainUpdateFrequency   0.2


%%%%  Parameters of SPH
DesNumNgb           64
MaxNumNgbDeviation  1
ArtBulkViscConst    0.75
InitGasTemp         166.53
MinGasTemp          100.    
CourantFac          0.2


%%%% System of units
UnitLength_in_cm         3.085678e21        ;  1.0 kpc /h
UnitMass_in_g            1.989e43           ;  solar masses
UnitVelocity_in_cm_per_s 1e5                ;  1 km/sec
GravityConstantInternal  0


%%%% Quantities for star formation and feedback
StarformationOn 0
CritPhysDensity     0.     %%  critical physical density for star formation in
                            %%  hydrogen number density in cm^(-3)
MaxSfrTimescale     1.5     %% in internal time unpar_file_textits
CritOverDensity      57.7    %%  overdensity threshold value
TempSupernova        1.0e8   %%  in Kelvin
TempClouds           1000.0   %%  in Kelvin
FactorSN             0.1
FactorEVP            1000.0

WindEfficiency    2.0
WindFreeTravelLength 10.0
WindEnergyFraction  1.0
WindFreeTravelDensFac 0.5


%%%% Additional things for Gadget XXL:
%%ViscositySourceScaling 0.7
%%ViscosityDecayLength   2.0
%%ConductionEfficiency     0.33
Shock_LengthScale 2.0
Shock_DeltaDecayTimeMax  0.02
ErrTolThetaSubfind       0.1
DesLinkNgb          32
""" % locals()
    
    with open(parameter_filename, 'w') as par_file:
        par_file.write(par_file_text)
    
    run_script_text = """\
#!/bin/tcsh
# Gadget simulation %(run_name)s.

cd %(run_dir_base)s
nice +10 mpiexec -np %(nproc)i %(gadget_executable)s %(parameter_filename)s""" % locals()
    
    with open(run_script_filename, 'w') as run_script:
        run_script.write(run_script_text)
    os.chmod(run_script_filename, stat.S_IRUSR|stat.S_IWUSR|stat.S_IXUSR)
    
    with open(restart_script_filename, 'w') as restart_script:
        restart_script.write(run_script_text+" 1")
    os.chmod(restart_script_filename, stat.S_IRUSR|stat.S_IWUSR|stat.S_IXUSR)


def getheader(filename, gtype):
    """Read header data from Gadget data file 'filename' with Gadget file
    type 'gtype'. Returns a dictionary with loaded values and filename."""
    DESC = '=I4sII'                                # struct formatting string
    HEAD = '=I6I6dddii6iiiddddii6ii60xI'        # struct formatting string
    keys = ('Npart', 'Massarr', 'Time', 'Redshift', 'FlagSfr', 'FlagFeedback', 'Nall', 'FlagCooling', 'NumFiles', 'BoxSize', 'Omega0', 'OmegaLambda', 'HubbleParam', 'FlagAge', 'FlagMetals', 'NallHW', 'flag_entr_ics', 'filename')
    f = open(filename, 'rb')
    
    if gtype == 2:
        f.seek(16) # If you want to use the data: desc = struct.unpack(DESC,f.read(16))
    
    raw = struct.unpack(HEAD,f.read(264))[1:-1]
    values = (raw[:6], raw[6:12]) + raw[12:16] + (raw[16:22],) + raw[22:30] + (raw[30:36], raw[36], filename)
    header = dict(zip(keys, values))
    
    f.close()
    return header

def LOSToRedshift(xLOS, vLOS, H, split = False):
    """
    Input:  line of sight distances (Mpc), velocities (km/s) and Hubble
    constant.
    Output: relativistic (Doppler) and cosmological (Hubble) redshifts if
    split = True, otherwise the sum of these (default).
    """
    c = 3.0e5
    zREL = np.sqrt((1+vLOS/c)/(1-vLOS/c)) - 1
    zCOS = xLOS*H/c # Needs replacement for large cosmological distances
    if split:
        return np.array((zREL, zCOS)).T
    else:
        return zREL + zCOS

def LOSToRedshift2(xLOS, vLOS, H, split = False):
    """
    Input:  line of sight distances (Mpc), velocities (km/s) and Hubble
    constant.
    Output: relativistic (Doppler) and cosmological (Hubble) redshifts if
    split = True, otherwise the sum of these (default).
    """
    c = 3.0e5
    zREL = np.sqrt((1+vLOS/c)/(1-vLOS/c)) - 1
    zCOS = xLOS*H/c # Needs replacement for large cosmological distances
    if split:
        return np.array((zREL, zCOS)).T
    else:
        return zREL + zCOS +zREL*zCOS

def redshiftToLOS(redshift, H):
    """
    Convert redshifts to apparent line of sight distances, ignoring particle
    velocities.
    
    Input: redshifts and Hubble constant.
    Output: line of sight distances (Mpc).
    """
    c = 3.0e5
    return redshift*c/H

def columnize(vol):
    hist = np.histogram(vol, bins=20, range=[0,1])
    tot = float(sum(hist[0]))
    y = hist[0]/tot
    x = np.array([(hist[1][i]+hist[1][i+1])/2. for i in range(len(hist[1])-1)])
    for i in range(len(x)):
        print x[i], y[i]

def logColumnize(vol):
    vollog = np.log10(vol)
    hist = np.histogram(vollog)
    tot = float(sum(hist[0]))
    y = hist[0]/tot
    x = 10**np.array([(hist[1][i]+hist[1][i+1])/2. for i in range(len(hist[1])-1)])
    for i in range(len(x)):
        print x[i], y[i]

def savePosAsErwin(pos, boxsize, filename, weight = None):
    """Save the position data as an Erwin-type binary file (Npart|x|y|z|
    [m]; normalized coordinates)."""
    erwinX = pos[:,0]/boxsize
    erwinY = pos[:,1]/boxsize
    erwinZ = pos[:,2]/boxsize
    N = np.array([len(pos)], dtype='int32')
    
    fd = file(filename, 'wb')
    
    fd.write(struct.pack("=I", len(pos)))
    fd.write(buffer(erwinX))
    fd.write(buffer(erwinY))
    fd.write(buffer(erwinZ))
    if weight is not None:
        fd.write(buffer(weight))
    
    fd.close()

# IFRIT FUNCTIES NOG OMZETTEN NAAR BUFFER METHODE

def savePosAsIfrit(pos, boxsize, filename, sample=None):
    """Saves as an IFrIT binary file."""
    if sample:
        samp = pos[np.random.randint(0,len(pos),sample)]
    else:
        samp = pos
    x = samp[:,0]
    y = samp[:,1]
    z = samp[:,2]
    N = np.array([len(x)], dtype='int32')

    box = np.array([0.0,0.0,0.0,boxsize,boxsize,boxsize],dtype='float32')
    
    ifritfile = file(filename, 'wb')
    ifritfile.write(struct.pack("=I", 4*N.size))
    fwrite(ifritfile, N.size, N)
    ifritfile.write(struct.pack("=I", 4*N.size))
    
    ifritfile.write(struct.pack("=I", 4*box.size))
    fwrite(ifritfile, box.size, box)
    ifritfile.write(struct.pack("=I", 4*box.size))
    
    ifritfile.write(struct.pack("=I", 4*x.size))
    fwrite(ifritfile, x.size, x)
    ifritfile.write(struct.pack("=I", 4*x.size))
    
    ifritfile.write(struct.pack("=I", 4*y.size))
    fwrite(ifritfile, y.size, y)
    ifritfile.write(struct.pack("=I", 4*y.size))
    
    ifritfile.write(struct.pack("=I", 4*z.size))
    fwrite(ifritfile, z.size, z)
    ifritfile.write(struct.pack("=I", 4*z.size))
    
    ifritfile.close()


def saveRhoAsIfrit(filename, rhoin):
    """Rho needs to be transposed to save in Fortran order (otherwise x and z
    will be interchanged in IFrIT)."""
    rho = rhoin.astype('float32')
    rho = rho.T
    rho = rho/rhoin.mean()
    N = np.array(rho.shape, dtype='int32')
    ifritfile = file(filename, 'wb')
    ifritfile.write(struct.pack("=I", 4*N.size))
    fwrite(ifritfile, N.size, N)
    ifritfile.write(struct.pack("=I", 4*N.size))
    ifritfile.write(struct.pack("=I", 4*rho.size))
    fwrite(ifritfile, rho.size, rho.ravel())
    ifritfile.write(struct.pack("=I", 4*rho.size))
    ifritfile.close()
    
def saveVoidsAsIfrit(filename, rhoin):
    """Rho needs to be transposed to save in Fortran order (otherwise x and z
    will be interchanged in IFrIT)."""
    rho = rhoin.astype('float32')
    rho = rho.T
    N = np.array(rho.shape, dtype='int32')
    ifritfile = file(filename, 'wb')
    ifritfile.write(struct.pack("=I", 4*N.size))
    fwrite(ifritfile, N.size, N)
    ifritfile.write(struct.pack("=I", 4*N.size))
    ifritfile.write(struct.pack("=I", 4*rho.size))
    fwrite(ifritfile, rho.size, rho.ravel())
    ifritfile.write(struct.pack("=I", 4*rho.size))
    ifritfile.close()