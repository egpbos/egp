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
import cPickle as pickle
import egp.toolbox, egp.icgen
import tarfile


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


class SubFindHaloes(object):
    def __init__(self, firstfile):
        self.firstfile = abspath(firstfile)
        self.load_data()
    
    def save_txt(self, filenamebase):
        """Filenamebase will be appended with _groups.txt and _subhaloes.txt."""
        groups = np.array([self.HaloLen, self.NsubPerHalo, self.FirstSubOfHalo, self.Halo_M_Mean200, self.Halo_R_Mean200, self.Halo_M_Crit200, self.Halo_R_Crit200, self.Halo_M_TopHat200, self.Halo_R_TopHat200, self.HaloCont, self.HaloPos[:,0], self.HaloPos[:,1], self.HaloPos[:,2], self.HaloMass]).T
        subhaloes = np.array([self.SubLen, self.SubOffset, self.SubParentHalo, self.SubPos[:,0], self.SubPos[:,1], self.SubPos[:,2], self.SubVel[:,0], self.SubVel[:,1], self.SubVel[:,2], self.SubVelDisp, self.SubVmax, self.SubSpin[:,0], self.SubSpin[:,1], self.SubSpin[:,2], self.SubMostBoundID, self.SubHalfMass, self.SubMassTab[:,0], self.SubMassTab[:,1], self.SubMassTab[:,2], self.SubMassTab[:,3], self.SubMassTab[:,4], self.SubMassTab[:,5], self.SubTMass]).T
        np.savetxt(filenamebase+'_groups.txt', groups)
        np.savetxt(filenamebase+'_subhaloes.txt', subhaloes)
    
    def save_sub_xyzm(self, filenamebase):
        """Filenamebase will be appended with _xyzm.txt."""
        data = np.array([self.SubPos[:,0], self.SubPos[:,1], self.SubPos[:,2], self.SubTMass]).T
        np.savetxt(filenamebase+'_xyzm.txt', data)
    
    def load_data(self):
        """Loads file headers and data of all files in the dataset into
        memory."""
        filename = self.firstfile
        tab0l = np.memmap(filename, dtype='int32', mode='r')
        tab0file = file(filename, 'rb')
        
        self.TotNgroups = tab0l[1]
        tab0file.seek(12)
        self.TotNids = struct.unpack('=q', tab0file.read(8)) # double long (LL in IDL)
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


class GadgetFOFGroups(object):
    def __init__(self, firstfile):
        self.firstfile = abspath(firstfile)
        self.load_data()
    
    def save_txt(self, filenamebase):
        """Filenamebase will be appended with .txt."""
        groups = np.array([self.GroupLen, self.GroupOffset, self.GroupMass, self.GroupCM[:,0], self.GroupCM[:,1], self.GroupCM[:,2], self.GroupVel[:,0], self.GroupVel[:,1], self.GroupVel[:,2], self.GroupLenType[:,0], self.GroupLenType[:,1], self.GroupLenType[:,2], self.GroupLenType[:,3], self.GroupLenType[:,4], self.GroupLenType[:,5], self.GroupLenMass[:,0], self.GroupLenMass[:,1], self.GroupLenMass[:,2], self.GroupLenMass[:,3], self.GroupLenMass[:,4], self.GroupLenMass[:,5]]).T
        np.savetxt(filenamebase+'.txt', groups)
    
    def load_data(self):
        """Loads file headers and data of all files in the dataset into
        memory."""
        filename = self.firstfile
        tab0l = np.memmap(filename, dtype='int32', mode='r')
        tab0file = file(filename, 'rb')
        
        self.TotNgroups = tab0l[1]
        tab0file.seek(12)
        self.TotNids = struct.unpack('=q', tab0file.read(8)) # double long (LL in IDL)
        self.Nfiles = tab0l[5]

        del tab0l
        tab0file.close()
        
        offset = 0
        
        self.GroupLen = np.empty(self.TotNgroups, dtype='int32')
        self.GroupOffset = np.empty(self.TotNgroups, dtype='int32')
        self.GroupMass = np.empty(self.TotNgroups, dtype='float32')
        self.GroupCM = np.empty((self.TotNgroups,3), dtype='float32')
        self.GroupVel = np.empty((self.TotNgroups,3), dtype='float32')
        self.GroupLenType = np.empty((self.TotNgroups,6), dtype='int32')
        self.GroupLenMass = np.empty((self.TotNgroups,6), dtype='float32')
                
        for i in range(self.Nfiles):
            ifile = str(i)
            
            tabl = np.memmap(filename[:-1]+ifile, dtype='int32', mode='r')
            tabf = np.memmap(filename[:-1]+ifile, dtype='float32', mode='r')
            
            Ngroups = tabl[0]
                        
            if Ngroups:
                b = 6
                f = 0
                self.GroupLen[offset:offset+Ngroups] = tabl[b+f*Ngroups:b+(f+1)*Ngroups] # Group Length
                f += 1
                self.GroupOffset[offset:offset+Ngroups] = tabl[b+f*Ngroups:b+(f+1)*Ngroups] # Group member ID offset
                f += 1
                self.GroupMass[offset:offset+Ngroups] = tabf[b+f*Ngroups:b+(f+1)*Ngroups] # FoF Mass
                f += 1
                self.GroupCM[offset:offset+Ngroups,:] = tabf[b+f*Ngroups:b+(f+3)*Ngroups].reshape((Ngroups,3)) # FoF Position
                f += 3
                self.GroupVel[offset:offset+Ngroups,:] = tabf[b+f*Ngroups:b+(f+3)*Ngroups].reshape((Ngroups,3)) # FoF Position
                f += 3
                self.GroupLenType[offset:offset+Ngroups,:] = tabf[b+f*Ngroups:b+(f+6)*Ngroups].reshape((Ngroups,6)) # FoF Position
                f += 6
                self.GroupLenMass[offset:offset+Ngroups,:] = tabf[b+f*Ngroups:b+(f+6)*Ngroups].reshape((Ngroups,6)) # FoF Position
                f += 6
                offset += Ngroups
                        
            del tabl, tabf
        self.groups = np.array([self.GroupLen, self.GroupOffset, self.GroupMass, self.GroupCM[:,0], self.GroupCM[:,1], self.GroupCM[:,2], self.GroupVel[:,0], self.GroupVel[:,1], self.GroupVel[:,2], self.GroupLenType[:,0], self.GroupLenType[:,1], self.GroupLenType[:,2], self.GroupLenType[:,3], self.GroupLenType[:,4], self.GroupLenType[:,5], self.GroupLenMass[:,0], self.GroupLenMass[:,1], self.GroupLenMass[:,2], self.GroupLenMass[:,3], self.GroupLenMass[:,4], self.GroupLenMass[:,5]])


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
def load_rien_ic_file(filename, gridsize, header_bits=64):
    """
    Loads Riens strange output format from his (un)constrained IC codes.
    By default, on 64-bit compilers, the block headers will be 64 bits; this
    used to be 32 bits on older compilers. If you're using an older compiler,
    set header_bits to 32.
    Output: rho and psi.
    """
    head_skip = header_bits/32
    rhopsi = np.memmap(filename, dtype='float32')[7+2*head_skip:].reshape((64**3,4+2*head_skip))[:,head_skip:-head_skip]
    rho = rhopsi[:,0].reshape((gridsize,gridsize,gridsize))
    psi1 = rhopsi[:,1].reshape((gridsize,gridsize,gridsize))
    psi2 = rhopsi[:,2].reshape((gridsize,gridsize,gridsize))
    psi3 = rhopsi[:,3].reshape((gridsize,gridsize,gridsize))
    return rho, np.array([psi1,psi2,psi3])

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

# global variables (used in next function):
gadget_par_file_text = """\
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
"""

gadget_run_script_texts = {}

gadget_run_script_texts["millipede"] = """\
#!/bin/bash
#PBS -N %(run_name)s
#PBS -l nodes=%(nodes)i:ppn=%(ppn)i
#PBS -e /home/p252012/log/%(run_name)s.err
#PBS -o /home/p252012/log/%(run_name)s.out
#PBS -l walltime=%(walltime)i:00:00
#PBS -m abe
#PBS -M pbos@astro.rug.nl
# Gadget simulation %(run_name)s.

module add openmpi/gcc torque maui

cd %(run_dir_base)s
mpiexec -np %(nproc)i %(gadget_executable)s %(parameter_filename)s
"""

gadget_run_script_texts["kapteyn"] = """\
#!/bin/tcsh
# Gadget simulation %(run_name)s.

cd %(run_dir_base)s
nice %(nice)s mpiexec -np %(nproc)i %(gadget_executable)s %(parameter_filename)s
"""

def prepare_gadget_run(boxlen, gridsize, cosmo, ic_file, redshift_begin, run_dir_base, run_name, nproc, output_list_filename = 'outputs_main.txt', DE_file = 'wdHdGHz_LCDM_bosW7.txt', ic_format = 2, time_max = 1.0, softening_factor = 22.5*768/300000., time_limit_cpu = 864000, resubmit_on = 1, resubmit_command = '0', cpu_time_bet_restart_file = 3600, part_alloc_factor = 1.6, tree_alloc_factor = 0.7, buffer_size = 300, gadget_executable = "/net/schmidt/data/users/pbos/sw/code/gadget/gadget3Sub_512_SL6/P-Gadget3_512", nice = "+0", save_dir = None, run_location = 'kapteyn', mem = 23, nodes = 1, queue = 'nodes'):
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
    time_limit_cpu (seconds); also used for walltime parameter for millipede!
    resubmit_on (0 or 1)
    resubmit_command (path)
    cpu_time_bet_restart_file (seconds)
    part_alloc_factor
    tree_alloc_factor
    buffer_size (MB)
    gadget_executable (file path)
    save_dir (directory path): optional; if given, the files will be saved here
                              and no run directory will be made. This is useful
                              when you need to run on a remote location and
                              need to copy the files to there first.
    run_location: 'kapteyn' or 'millipede'
    mem (GB): required amount of memory for millipede runs.
    nodes: number of nodes (millipede).
    queue: millipede queue; choose between nodes, quads, short, etc.
    
    Note that run_dir_base is not the directory where the simulation will be
    run; that is run_dir_base+run_name; the run_name directory will be created
    by this function.
    """
    # Length units are converted to kpc h^-1, the default Gadget unit.
    boxlen *= 1000
    
    output_dir = run_dir_base+'/'+run_name
    if not save_dir:
        try:
            os.mkdir(output_dir)
        except OSError:
            print "Warning: output directory already exists. This run might be overwriting previous runs!"
    
    omegaM = cosmo.omegaM
    omegaL = cosmo.omegaL
    omegaB = cosmo.omegaB
    hubble = cosmo.h
    
    if not save_dir:
        parameter_filename = run_dir_base+'/'+run_name+'.par'
        run_script_filename = run_dir_base+'/'+run_name+'.sh'
        restart_script_filename = run_dir_base+'/'+run_name+'_restart.sh'
    else:
        parameter_filename = save_dir+'/'+run_name+'.par'
        run_script_filename = save_dir+'/'+run_name+'.sh'
        restart_script_filename = save_dir+'/'+run_name+'_restart.sh'
    
    output_list_filename = run_dir_base+'/'+output_list_filename
    DE_file = run_dir_base+'/'+DE_file
    
    time_begin = 1/(redshift_begin+1)
    
    # Softening: based on Dolag's ratios
    # default ~ 1/17.3 of the mean ipd
    softening = softening_factor*boxlen/gridsize
    softening_max_phys = softening/3
    
    # For millipede runs:
    walltime = time_limit_cpu/3600
    
    if resubmit_on and (run_location=='kapteyn'):
        resubmit_command = restart_script_filename
    if resubmit_on and (run_location=='millipede'):
        resubmit_command = run_dir_base+'/'+run_name+'_qsub_restart.sh'
        local_filename = save_dir+'/'+run_name+'_qsub_restart.sh'
        with open(local_filename, 'w') as qresub_file:
            qresub_file.write("#!/usr/bin/env bash\n")
            # HIER KUN JE DINGEN TOEVOEGEN OM BESTANDEN LOKAAL (OP DE NODE) TE rsync'EN EN rm'EN!
            qresub_file.write("qsub %(run_dir_base)s/%(run_name)s_restart.sh -q %(queue)s\n" % locals())
        os.chmod(local_filename, stat.S_IRUSR|stat.S_IWUSR|stat.S_IXUSR)
    
    ppn = nproc/nodes
    # N.B.: make sure that nproc and nodes fit with the queue!
    # 12 max. ppn for nodes queue, 24 max. ppn for quads queue.
    
    ### Open and write to files:
    global gadget_par_file_text
    par_file_text = gadget_par_file_text % locals()
    
    with open(parameter_filename, 'w') as par_file:
        par_file.write(par_file_text)
    
    if save_dir: # parameter_filename is used in the run_script, so needs to be localized!
        parameter_filename = run_dir_base+'/'+run_name+'.par'
    
    global gadget_run_script_texts
    run_script_text = gadget_run_script_texts[run_location] % locals()
    
    with open(run_script_filename, 'w') as run_script:
        run_script.write(run_script_text)
    os.chmod(run_script_filename, stat.S_IRUSR|stat.S_IWUSR|stat.S_IXUSR)
    
    run_name += "_restart"
    run_script_text = gadget_run_script_texts[run_location] % locals()
    
    with open(restart_script_filename, 'w') as restart_script:
        restart_script.write(run_script_text[:-1]+" 1\n")
    os.chmod(restart_script_filename, stat.S_IRUSR|stat.S_IWUSR|stat.S_IXUSR)

subfind_run_script_texts = {}

subfind_run_script_texts["millipede"] = """\
#!/bin/bash
#PBS -N %(run_name)s_subfind_snaps%(snapstring)s
#PBS -l nodes=%(nodes)i:ppn=%(ppn)i
#PBS -e /home/p252012/log/%(run_name)s_subfind_snaps%(snapstring)s.err
#PBS -o /home/p252012/log/%(run_name)s_subfind_snaps%(snapstring)s.out
#PBS -l walltime=%(walltime)i:00:00
#PBS -m abe
#PBS -M pbos@astro.rug.nl
# Gadget SubFind run on Gadget run %(run_name)s.

module add openmpi/gcc torque maui

cd %(run_dir_base)s
"""

subfind_run_script_texts["kapteyn"] = """\
#!/bin/tcsh
# Gadget SubFind run on Gadget run %(run_name)s.

cd %(run_dir_base)s
"""

subfind_run_script_lastline ={}

subfind_run_script_lastline["millipede"] = """\
mpiexec -np %(nproc)i %(gadget_executable)s %(parameter_filename)s 3 %(snap)s
"""

subfind_run_script_lastline["kapteyn"] = """\
nice %(nice)s mpiexec -np %(nproc)i %(gadget_executable)s %(parameter_filename)s 3 %(snap)s
"""

def prepare_gadget_subfind_run(run_dir_base, run_name, snaps, nproc, time_limit_cpu = 864000, gadget_executable = "/net/schmidt/data/users/pbos/sw/code/gadget/gadget3Sub_512_SL6/P-Gadget3_512", nice = "+0", save_dir = None, run_location = 'kapteyn', nodes = 1, queue = 'nodes'):
    """
    Use prepare_gadget_run() to prepare the parameter file. The following
    parameters in this function must be the same as what you used in
    prepare_gadget_run(): run_dir_base and run_name. It also assumes you run
    the SubFind run in the same location, so run_location must be the same
    as well and therefore save_dir as well.
    
    There is one extra argument w.r.t. prepare_gadget_run, snaps, which is a
    list of integers of the snapshots of this gadget run that you want to run
    SubFind on from the run script that this function creates.
    """    
    output_dir = run_dir_base+'/'+run_name
    parameter_filename = run_dir_base+'/'+run_name+'.par'
    
    snapstring = ""
    for snap in snaps[:-1]:
        snapstring += "%i," % snap
    snapstring += "%i" % snaps[-1]
    
    if not save_dir:
        run_script_filename = run_dir_base+'/'+run_name+'_subfind_snaps%s.sh' % snapstring
    else:
        run_script_filename = save_dir+'/'+run_name+'_subfind_snaps%s.sh' % snapstring
    
    # For millipede runs:
    walltime = time_limit_cpu/3600
    
    ppn = nproc/nodes
    # N.B.: make sure that nproc and nodes fit with the queue!
    # 12 max. ppn for nodes queue, 24 max. ppn for quads queue.
    
    ### Open and write to files:
    global subfind_run_script_texts
    global subfind_run_script_lastline
    run_script_text = subfind_run_script_texts[run_location] % locals()
    for snap in snaps:
        run_script_text += subfind_run_script_lastline[run_location] % locals()
    
    with open(run_script_filename, 'w') as run_script:
        run_script.write(run_script_text)
    os.chmod(run_script_filename, stat.S_IRUSR|stat.S_IWUSR|stat.S_IXUSR)


def setup_cubep3m_run(pos, vel, cosmo, boxlen, gridsize, redshift, snapshots, run_name, run_path_base, nodes_dim, tiles_node_dim, cores, nf_tile_I = 2, nf_cutoff = 16, pid_flag=False, pp_run=True, pp_range = 2, displace_from_mesh=False, read_displacement_seed=False, verbose=False, debug=False, chaplygin=False, dt_scale = 1.0, dt_max = 1.0, ra_max = 0.05, da_max = 0.01, cfactor = 1.05, max_nts = 4000, density_buffer = 2.0, location='kapteyn'):
    """
    Give pos and vel like they come out of egp.icgen.zeldovich.
    
    The FFLAGS parameter in the Makefile can be appended with defines:
    BINARY: output formaat, lijkt het
    NGP: iets met een kernel, wrs voor density estimation
    NGPH: ook die kernel, maar dan voor de halofinder
    LRCKCORR: geen idee, maar lijkt me nuttig, als het een CORRection is.
    PPINT: ik denk de buitenste PP loop (zie paper, figuur 2, if pp=.true.)
    PP_EXT: binnenste PP loop dan dus.
    MPI_TIME: print timings (min/max/avg) van MPI jobs.
    DIAG: print allerlei dingen, volgens mij puur voor diagnostics?
    PID_FLAG: hou particle IDs bij om ICs aan snapshots te kunnen relateren.
    READ_SEED: "This will always use the same random number at each time step. It surely introduces a bias, but is good for testing code."
    DISP_MESH: applies random offset from initial position at each timestep om anisotropie t.g.v. het grid te minimaliseren.
    DEBUG: zal wel nog meer debug info zijn. Er zijn ook andere, meer specifieke DEBUG parameters; DEBUG_LOW, DEBUG_VEL, DEBUG_CCIC, DEBUG_RHOC, DEBUG_CRHO en DEBUG_PP_EXT bijvoorbeeld.
    Chaplygin: include a Chaplygin gas cosmic component.
    
    -- Main mesh parameters:
    nodes_dim: nodes / dimension, total nodes = nodes_dim**3
    tiles_node_dim: fine mesh tiles / node / dimension
    cores: cores / node
    
    -- Fine mesh parameters:
    nf_tile_I: The size of fine mesh tile in cells / dimension must be set as:
               nf_tile = I*mesh_scale*(nodes_dim)**2 / tiles_node_dim + 2*nf_buf
               -- where I is an integer; we set this integer using nf_tile_I.
    nf_cutoff: Fine mesh force cut-off in fine mesh cells (determined by kernel)
    
    -- General code behaviour parameters:
    pp_run
    pp_range
    verbose: diagnostic info, timing, etc.
    debug: extra debugging information
    displace_from_mesh: random displacement at every timestep
    read_displacement_seed: use constant seed for displace_from_mesh
                            N.B.: must write ic_path/seed0.init containing
                            enough seeds for all timesteps!
    pid_flag
    
    -- Time-step parameters:
    dt_scale: Increase/decrease to make the timesteps larger/smaller
    dt_max
    ra_max
    da_max
    cfactor
    max_nts: max number of timesteps
    
    -- Miscellaneous:
    density_buffer: density buffer fraction (1.0 == no buffer, 2.0 == 2x avg
                    density, etc)
    
    
    """
    ### 0. Parameters
    omega_l = cosmo.omegaL
    omega_m = cosmo.omegaM
    omega_b = cosmo.omegaB
    bias = cosmo.bias
    power_index = cosmo.primn
    wde = cosmo.w_0
    w_a = cosmo.w_a
    if chaplygin:
        omega_ch = cosmo.omega_ch
        alpha_ch = cosmo.alpha_ch
        A_ch = cosmo.A_ch
    else:
        omega_ch = 0.
        alpha_ch = 0.
        A_ch = 1.
    
    num_node_compiled = nodes_dim**3
    
    # Fine mesh buffer size in fine mesh cells
    nf_buf         = nf_cutoff + 8
    # size of fine mesh tile in cells / dimension
    nf_tile = nf_tile_I*gridsize*(nodes_dim)**2 / tiles_node_dim + 2*nf_buf
    # number of cells / dimension of entire simulation (fine grid)
    nc             = (nf_tile-2*nf_buf)*tiles_node_dim*nodes_dim
    
    ### 1. Basic preparations    
    run_path     = run_path_base + run_name+'/'
    scratch_path = run_path_base + 'scratch/'
    
    ic_path      = run_path + 'input/'
    ic_filename  = ic_path + 'xv0.ic'
    output_path  = run_path + 'output/'
    cubepm_root  = '../' # relative to batch, where everything is run from
    
    # make dirs:
    egp.toolbox.mkdir(run_path)
    egp.toolbox.mkdir(output_path)

    ## vers source pakket neerzetten
    source_tar_path = egp.icgen.__file__[:egp.icgen.__file__.rfind('/')]+"/cubep3m_clean.tar"
    source_tar = tarfile.TarFile(source_tar_path)
    
    source_tar.extractall(path = run_path)
    
    ### 2. Writing IC-files
    pos = pos.reshape(3, gridsize**3).T
    vel = vel.reshape(3, gridsize**3).T
    # Position is converted to fine-grid cell units.
    pos = pos / boxlen * nc - 0.5 # -0.5 for unclear reasons, but seems to work better...
    # Velocity is also converted to internal units.
    # velunit = 150/a * L/N * sqrt(omega_m) * h
    # where L is the boxsize in Mpc/h and N is the fine grid size.
    # The h is left out in the actual number, because Hubble-units are
    # used everywhere in the code.
    vel = vel / (150*(1+redshift) * boxlen / nc * np.sqrt(omega_m))
    
    ## the actual writing (dm-only):
    f = open(ic_filename,'wb')
    # header
    N = len(pos) # Number of DM particles (Npart[1], i.e. the 2nd entry)
    header = struct.pack("=I", N)
    f.write(header)
    # pos & vel
    data = np.array(np.hstack((pos, vel)), dtype='float32', order='C')
    f.write(data.data)
    f.close()
    
    ### 3. Remaining simulation run preparations
    ## source bestanden met parameters schrijven
    FFLAGS_append = "-DBINARY -DNGP -DLRCKCORR -DNGPH"
    if pp_run:
        FFLAGS_append += " -DPPINT -DPP_EXT"
    if displace_from_mesh:
        FFLAGS_append += " -DDISP_MESH"
    if read_displacement_seed:
        FFLAGS_append += " -DREAD_SEED"
    if verbose:
        FFLAGS_append += " -DDIAG -DMPI_TIME"
    if debug:
        FFLAGS_append += " -DDEBUG -DDEBUG_LOW -DDEBUG_VEL -DDEBUG_CCIC -DDEBUG_RHOC -DDEBUG_CRHO -DDEBUG_PP_EXT"
    if pid_flag:
        FFLAGS_append += " -DPID_FLAG"
    if chaplygin:
        FFLAGS_append += " -DChaplygin"
    
    makefile_path = run_path+"source_threads/Makefile"
    parameterfile_path = run_path+"parameters"
    cubepm_par_path = run_path+"source_threads/cubepm.par"
    
    egp.toolbox.fill_template_file(makefile_path, locals())
    egp.toolbox.fill_template_file(parameterfile_path, locals())
    egp.toolbox.fill_template_file(cubepm_par_path, locals())
    
    ## input (checkpoints)
    checkpoints = open(run_path+"input/checkpoints", 'w')
    projections = open(run_path+"input/projections", 'w')
    halofinds   = open(run_path+"input/halofinds", 'w')
    for snap in snapshots:
        checkpoints.write(str(snap)+"\n")
        projections.write(str(snap)+"\n")
        halofinds.write(str(snap)+"\n")
    checkpoints.close(); projections.close(); halofinds.close()
    
    ## run-scripts schrijven
    if location == 'kapteyn':
        run_script_path = run_path+"batch/kapteyn.run"
    elif location == 'millipede':
        run_script_path = run_path+"batch/millipede.run"
        print "N.B.: MILLIPEDE SCRIPT WERKT NOG NIET!"
    else:
        print "location parameter not recognized! Either kapteyn or millipede."
        raise SystemExit
    
    egp.toolbox.fill_template_file(run_script_path, locals())
    
    # create symlink to executable in batch directory
    os.symlink(run_path+"source_threads/cubep3m", run_path+"batch/cubep3m")
    
    ### 4. Save parameters in a pickled file
    parameters = locals()
    for key in parameters:
        if type(eval(key)) in file, np.ndarray:
            del parameters[key]
    pickle.dump(parameters, open(run_path+"parameters.pickle", "wb"))
    
    print("Run with:\n%(run_script_path)s" % locals())


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
