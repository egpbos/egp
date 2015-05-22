#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
basic_types.py
/basic_types/ module in the /egp/ package.
  
Created by Evert Gerardus Patrick Bos.
Copyright (c) 2012. All rights reserved.

Contains the basic data types used in other modules.
"""

import numpy as np
import egp.toolbox
from matplotlib import pyplot as plt

class EGPException(Exception):
    def __init__(self, value):
        self.parameter = value
    def __str__(self):
        return repr(self.parameter)


class PeriodicArray(np.ndarray):
    """
    Based on http://docs.scipy.org/doc/numpy/user/basics.subclassing.html#slightly-more-realistic-example-attribute-added-to-existing-array
    """
    def __new__(cls, input_array):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # return the newly created object:
        return obj
    
    def __getitem__(self, indexing):
        try:
            # indexing contains only an integer or ndarray of integer type
            # (cases 1a and 8).
            return np.ndarray.__getitem__(self, indexing%self.shape[0])
            # Always need the 0th element of self.shape, because getitem works
            # recursively on subarrays in multiple dimensions.
        except TypeError:
            # "unsupported operand type(s) for %: 'type' and 'int'", i.e. cases
            # 3, 4, 5 and 6, i.e. contains a slice, Ellipsis or np.newaxis OR
            # "unsupported operand type(s) for %: 'tuple' and 'int'" OR
            # "unsupported operand type(s) for %: 'list' and 'int'", i.e. case
            # 1b or 7, i.e. is a tuple or list of integers.
            try:
                # if it's a /list/ of integers or other stuff, fix slices now:
                for i, index in enumerate(indexing):
                    self._doSliceChecking(i, index, indexing)
                return np.ndarray.__getitem__(self, indexing)
            except TypeError:
                # it doesn't seem to be a list (not mutable), so make it one:
                try:
                    indexing = list(indexing)
                except TypeError:
                    # "'slice' object is not iterable", indeed, so:
                    indexing = list([indexing])
                for i, index in enumerate(indexing):
                    self._doSliceChecking(i, index, indexing)
                return np.ndarray.__getitem__(self, tuple(indexing))
        except ValueError:
            # "setting an array element with a sequence.", i.e. case 9.
            # We just pass it along:
            return np.ndarray.__getitem__(self, indexing)
        except:
            # all other cases, i.e. shit hitting the fan: let Numpy handle it!
            return np.ndarray.__getitem__(self, indexing)

    def __getslice__(self, i, j):
        # This is only used for case 2 (slice without step size).
#        print "getslice called"
        indexing = [slice(i,j)]
        index = indexing[0]
        self._doSliceChecking(0,index,indexing)
        return np.ndarray.__getitem__(self,indexing)
    
    def _doSliceChecking(self, i, index, indexing):
        try:
            # try to use it as a slice...
            dimSize = self.shape[i]
            step = index.indices(dimSize)[-1] # using index.step gives
                                              # None if not specified,
                                              # but this defaults to 1
            if step*(index.start-index.stop) > 0:
                # undefined behaviour for standard python slices, so we fix it.
                if step > 0:
                    newIndex = range(index.start%dimSize, dimSize, step)\
                               + range(index.start%dimSize%step,\
                               index.stop%dimSize, step)
                else:
                    newIndex = range(index.start%dimSize, -1, step) +\
                               range(dimSize - index.start%dimSize%step - 1,\
                               index.stop%dimSize, step)
            else:
                # slice is nice, so no fixing necessary.
                newIndex = np.arange(index.start, index.stop, step)%dimSize
            indexing[i] = newIndex
        except AttributeError:
            # if it's not a slice, just continue without editing the index
            try:
                # ... well, ok, we'll only try to modulo it
                indexing[i] = index%dimSize
            except:
                pass
        except TypeError:
            # if indexing is immutable, we throw this error back
            raise TypeError


class Field(object):
    """
    Contains a field itself, given on a discrete 3D numpy.array, and the field's
    discrete fourier space representation. N.B.: the discrete fourier space
    representation is not the true fourier space representation; these differ by
    a factor proportional to dk**3 (c.f. Press+07, eqn. 12.1.8).
    If you manually set only one of the two, the other will automatically be
    calculated once it is called upon. If you set both, no automatic checks are
    done, so make sure that the fields are the correct corresponding ones!
    """
    def __init__(self, true=None, fourier=None):
        if np.any(true):
            self.t = true
        if np.any(fourier):
            self.f = fourier
    
    t, f = property(), property()
    
    @t.getter
    def t(self):
        try:
            return self._true
        except AttributeError:
            self.t = self._ifft(self.f)
            self.t *= np.size(self.t) # factor from discrete to true Fourier transform
            return self._true
    @t.setter
    def t(self, field):
        self._true = field
    @f.getter
    def f(self):
        try:
            return self._fourier
        except AttributeError:
            self._fourier = egp.toolbox.rfftn_flip(self.t)/np.size(self.t)
            #~ self._fourier = np.fft.rfftn(self.t)/np.size(self.t)
            return self._fourier
    @f.setter
    def f(self, field):
        self._fourier = field
        if field is None:
            self._ifft = egp.toolbox.irfftn_flip
            #~ self._ifft = np.fft.irfftn
        elif field.shape[0] == field.shape[2]:
            self._ifft = egp.toolbox.ifftn_flip
            #~ self._ifft = np.fft.ifftn
        elif field.shape[0] == (field.shape[2]-1)*2:
            self._ifft = egp.toolbox.irfftn_flip
            #~ self._ifft = np.fft.irfftn
    
    @property
    def periodic(self):
        """The true fields are all defined on periodic grids, so here's a
        convenience function for it."""
        # N.B.: PeriodicArray is not settable, so no need for setter (yet).
        try:
            return self._periodic
        except AttributeError:
            self._periodic = PeriodicArray(self.t)
            return self._periodic
    
    def show(self, xlabel="x (Mpc)", ylabel="y (Mpc)"):
        """
        Plot a 2D field slice with the right axes on the right side for it to make
        sense to me. In my world, the first axis of an array should represent the
        x-axis, in that if you ask for a[0] in a 2D array /a/ then you should get
        the field entries at x=0 for varying (a[0,-1] would be (x,y)=(0,boxlen)).
        
        By default matplotlib.pyplot's imshow does it the other way around, which
        could of course easily be remedied by a transpose, but this easy function
        does that all for you, and a little more.
        
        N.B.: must import pyplot as pl from matplotlib!
        N.B.2: only works for objects with a boxlen property.
        """
        plt.imshow(self.t.T, origin='bottom', interpolation='nearest', extent=(0,self.boxlen,0,self.boxlen))
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.colorbar()


class VectorField(object):
    """
    A three component vector field, containing three Field instances as
    attributes x, y and z.
    Initialization parameter true must have shape (3,N,N,N) and fourier must
    have shape (3,N,N,N/2+1).
    
    Note: We could change this to only contain one array of shape (3,N,N,N) (for
    the true component) and use egp.toolbox.rfftn_flip(psi, axes=(1,2,3)) so the
    first axis is not transformed. Might be more convenient in e.g. the
    Zel'dovich code.
    """
    def __init__(self, true=None, fourier=None):
        if np.any(true):
            self.x = Field(true=true[0])
            self.y = Field(true=true[1])
            self.z = Field(true=true[2])
        if np.any(fourier):
            self.x = Field(fourier=fourier[0])
            self.y = Field(fourier=fourier[1])
            self.z = Field(fourier=fourier[2])


class Particles(object):
    """
    A set of N particles with three-dimensional cartesian positions and velocities.
    Other properties may be added in subclasses.
    Arrays /pos/ and /vel/ must have shape (N,3) or otherwise be left empty for
    later assignment.
    """
    def __init__(self, pos = None, vel = None):
        if pos:
            self._pos = pos
        if vel:
            self._vel = vel
    pos = property()
    @pos.getter
    def pos(self):
        try:
            return self._pos
        except AttributeError:
            print("Positions not yet loaded!")
            raise EGPException
    @pos.setter
    def pos(self, pos):
        self._pos = pos

    vel = property()
    @vel.getter
    def vel(self):
        try:
            return self._vel
        except AttributeError:
            print("Velocities not yet loaded!")
            raise EGPException
    @vel.setter
    def vel(self, vel):
        self._vel = vel

    def calcPosSph(self, origin, centerOrigin):
        """Calculate the positions of particles in spherical coordinates. The
        origin is by default at the center of the box, but can be specified by
        supplying an origin=(x,y,z) argument."""

        # Implement periodic folding around to keep the origin centered

        x = self.pos[:,0] - origin[0]
        y = self.pos[:,1] - origin[1]
        z = self.pos[:,2] - origin[2]

        if centerOrigin:
            box = self.header[0]['BoxSize']
            halfbox = box/2.
            #
            # EGP (22 mei 2015):
            # IS DIT NIET PERIODIC BOUNDARY CONDITIONS?
            # WAAROM DOEN WE DIT HIER EIGENLIJK?
            #
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



class OrderedParticles(Particles):
    """
    Particles with an extra order array.
    """
    def __init__(self, pos = None, vel = None, order = None):
        super(OrderedParticles, self).__init__(pos, vel)
        if order:
            self._order = order
    order = property()
    @order.getter
    def order(self):
        try:
            return self._order
        except AttributeError:
            print("Order not yet loaded!")
            raise EGPException
    @order.setter
    def order(self, order):
        self._order = order


class MassiveParticles(Particles):
    """
    Particles with an extra mass array.
    """
    def __init__(self, pos = None, vel = None, mass = None):
        super(MassiveParticles, self).__init__(pos, vel)
        if mass:
            self._mass = mass
    mass = property()
    @mass.getter
    def mass(self):
        try:
            return self._order
        except AttributeError:
            print("Mass not yet loaded!")
            raise EGPException
    @mass.setter
    def mass(self, mass):
        self._mass = mass


class MassiveOrderedParticles(OrderedParticles, MassiveParticles):
    """
    Particles with both an order and a a mass.
    """
    def __init__(self, pos = None, vel = None, order = None, mass = None):
        super(MassiveOrderedParticles, self).__init__(pos = pos, vel = vel, order = order)
        print("NEED TO SOMEHOW ALSO REFER BACK TO MassiveParticles HERE TO INITIALIZE THE MASS!")
        raise EGPException
