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
import egp.fft
import egp.cosmology
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


# TODO: remove object superclass; no longer necessary in Python 3!
class Field(object):
    """
    Contains a field itself, given on a discrete 3D numpy.array, and the field's
    discrete fourier space representation. N.B.: the discrete fourier space
    representation is not the true fourier space representation; these differ by
    a factor proportional to dk**3 (c.f. Press+07, eqn. 12.1.8).
    If you manually set only one of the two, the other will automatically be
    calculated once it is called upon. If you set both, no automatic checks are
    done, so make sure that the fields are the correct corresponding ones!

    The class overrides __add__ and __sub__, since these are invariant to the
    Fourier transformation. Multiplication is different in true and Fourier
    space, so we have to be more careful there. We chose __mul__ to do the
    multiplication in true space and __matmul__ (the @ operator) to be in
    Fourier space. Note: __matmul__ makes the class Python 3 only.
    """
    def __init__(self, true=None, fourier=None, boxsizes=(1, 1, 1)):
        if np.any(true):
            self.t = true
        if np.any(fourier):
            self.f = fourier
        self.boxsizes = np.array(boxsizes)
        self.volume = np.prod(boxsizes)

    t, f = property(), property()

    @t.getter
    def t(self):
        try:
            return self._true
        except AttributeError:
            self.t = self._ifft(self.f)
            return self._true

    @t.setter
    def t(self, field):
        self._true = field

    @f.getter
    def f(self):
        try:
            return self._fourier
        except AttributeError:
            self._fourier = egp.fft.rfftn_flip(self.t)
            return self._fourier

    @f.setter
    def f(self, field):
        self._fourier = field
        if field is None:
            self._ifft = egp.fft.irfftn_flip
        elif field.shape[0] == field.shape[2]:
            self._ifft = egp.fft.ifftn_flip
        elif field.shape[0] == (field.shape[2] - 1) * 2:
            self._ifft = egp.fft.irfftn_flip

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

    def __add__(self, other):
        return Field(true=(self.t + other.t))

    def __sub__(self, other):
        return Field(true=(self.t - other.t))

    def __mul__(self, other):
        return Field(true=(self.t * other.t))

    def __matmul__(self, other):
        """
        Convolution of two Fields. Multiplies by volume (which must match). See
        Martel+05 or section 3.B.3 of my thesis for a derivation of this.
        """
        if np.any(self.boxsizes != other.boxsizes) or self.volume != other.volume:
            raise ValueError("boxsizes/volumes of Fields in convolution do not match!")
        return Field(fourier=(self.volume * self.f * other.f))

    def subvolve(self, other):
        """
        calculate A - A*B, i.e. subtract the convolution of A and B from A, leaving
        us with the "subvolution" of A with B
        """
        return self - (self @ other)


def test_Field_matmul():
    """
    A convolution of f(x) with a "Kronecker delta" should give f(x).
    """
    N = 10
    kronecker_raw = np.ones((N, N, N))
    a_raw = np.random.rand(N, N, N)

    kronecker = Field(true=kronecker_raw)
    a = Field(true=a_raw)

    b = a * kronecker

    # print(a.t)
    # print(b.t)

    assert(np.allclose(a.t, b.t))


# TODO: remove object superclass; no longer necessary in Python 3!
class VectorField(object):
    """
    A three component vector field, containing three Field instances as
    attributes x, y and z.
    Initialization parameter true must have shape (3,N,N,N) and fourier must
    have shape (3,N,N,N/2+1).

    Note: We could change this to only contain one array of shape (3,N,N,N) (for
    the true component) and use egp.fft.rfftn_flip(psi, axes=(1,2,3)) so the
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


# TODO: remove object superclass; no longer necessary in Python 3!
class Particles(object):
    """
    A set of N particles with three-dimensional cartesian positions and
    velocities. Other properties may be added in subclasses.
    Arrays /pos/ and /vel/ must have shape (N,3) or otherwise be left empty for
    later assignment.
    Note that pos units are kpc/h!
    """
    def __init__(self, pos=None, vel=None):
        if pos is not None:
            self._pos = pos
        if vel is not None:
            self._vel = vel
    pos = property()

    @pos.getter
    def pos(self):
        try:
            return self._pos
        except AttributeError:
            raise EGPException("Positions not yet loaded!")

    @pos.setter
    def pos(self, pos):
        self._pos = pos

    vel = property()

    @vel.getter
    def vel(self):
        try:
            return self._vel
        except AttributeError:
            raise EGPException("Velocities not yet loaded!")

    @vel.setter
    def vel(self, vel):
        self._vel = vel

    def calcPosSph(self, origin, boxsize, centerOrigin=False):
        """Calculate the positions of particles in spherical coordinates. The
        origin is by default at the center of the box, but can be specified by
        supplying an origin=(x,y,z) argument."""

        x = self.pos[:, 0] - origin[0]
        y = self.pos[:, 1] - origin[1]
        z = self.pos[:, 2] - origin[2]

        # Implement periodic folding around to center the origin
        if centerOrigin:
            halfbox = boxsize/2.
            np.putmask(x, x < -halfbox, x + boxsize)
            np.putmask(y, y < -halfbox, y + boxsize)
            np.putmask(z, z < -halfbox, z + boxsize)
            np.putmask(x, x >= halfbox, x - boxsize)
            np.putmask(y, y >= halfbox, y - boxsize)
            np.putmask(z, z >= halfbox, z - boxsize)
            self.posCO = np.vstack((x, y, z)).T

        xy2 = x*x + y*y
        xy = np.sqrt(xy2)
        r = np.sqrt(xy2 + z*z)
        # [-180,180] angle in (x,y) plane counterclockw from x-axis towards y:
        phi = np.arctan2(y, x)
        # [0,180] angle from the positive towards the negative z-axis:
        theta = np.arctan2(xy, z)

        self.posSph = np.vstack((r, phi, theta)).T
        self.posSphCalculated = True

    def calcRedshift(self, origin, boxsize, H, centerOrigin=False):
        """Calculate the redshifts of the particles, i.e. the redshift space
        equivalents of the radial distances. The origin is by default at the
        center of the box, but can be specified by supplying an origin=(x,y,z)
        argument.

        Note: centerOrigin should be the same as what was used for calcPosSph.
        """

        self.calcPosSph(origin, boxsize, centerOrigin=centerOrigin)

        if centerOrigin:
            x = self.posCO[:, 0]
            y = self.posCO[:, 1]
            z = self.posCO[:, 2]
        else:
            x = self.pos[:, 0] - origin[0]
            y = self.pos[:, 1] - origin[1]
            z = self.pos[:, 2] - origin[2]

        r = self.posSph[:, 0]

        unitvector_r = np.array([x/r, y/r, z/r]).T

        vR = np.sum(unitvector_r * self.vel, axis=1)

        self.redshift = egp.cosmology.LOSToRedshift(r/1000, vR, H)
        self.redshiftCalculated = True

    def calcRedshiftSpace(self, origin, boxsize, H, centerOrigin=False):
        """Convert particle positions to cartesian redshift space, i.e. the
        space in which the redshift is used as the radial distance. The origin
        is by default at the center of the box, but can be specified by
        supplying an origin=np.array((x,y,z)) argument."""

        self.calcRedshift(origin, boxsize, H, centerOrigin=centerOrigin)

        rZ = 1000*egp.cosmology.redshiftToLOS(self.redshift, H)
        phi = self.posSph[:, 1]
        theta = self.posSph[:, 2]

        if centerOrigin:
            # the particles were centered around 0,0,0 in the intermediate
            # coordinates, so to put them back in the box, just add halfbox
            halfbox = boxsize/2.
            xZ = rZ*np.sin(theta)*np.cos(phi) + halfbox
            yZ = rZ*np.sin(theta)*np.sin(phi) + halfbox
            zZ = rZ*np.cos(theta) + halfbox
        else:
            xZ = rZ*np.sin(theta)*np.cos(phi) + origin[0]
            yZ = rZ*np.sin(theta)*np.sin(phi) + origin[1]
            zZ = rZ*np.cos(theta) + origin[2]

        self.posZ = np.array([xZ, yZ, zZ]).T
        self.posZCalculated = True

    def calcRedshiftSpaceAllInOne(self, boxsize, H, origin=np.array((0, 0, 0)),
                                  centerOrigin=False):
        """Convert particle positions to cartesian redshift space, i.e. the
        space in which the redshift is used as the radial distance. The origin
        is by default at the bottom corner of the box, but can be specified by
        supplying an origin=np.array((x,y,z)) argument."""

        x = self.pos[:, 0] - origin[0]
        y = self.pos[:, 1] - origin[1]
        z = self.pos[:, 2] - origin[2]

        # Implement periodic folding around to center the origin (which is at
        # 0,0,0 in the x,y,z coordinates), so that all particles will be within
        # [-halfbox, halfbox]:
        if centerOrigin:
            halfbox = boxsize/2.
            np.putmask(x, x < -halfbox, x + boxsize)
            np.putmask(y, y < -halfbox, y + boxsize)
            np.putmask(z, z < -halfbox, z + boxsize)
            np.putmask(x, x >= halfbox, x - boxsize)
            np.putmask(y, y >= halfbox, y - boxsize)
            np.putmask(z, z >= halfbox, z - boxsize)

        xy2 = x*x + y*y
        xy = np.sqrt(xy2)
        r = np.sqrt(xy2 + z*z)

        unitvector_r = np.array([x/r, y/r, z/r]).T
        vR = np.sum(unitvector_r * self.vel, axis=1)

        redshift = egp.cosmology.LOSToRedshift(r/1000, vR, H)
        rZ = 1000*egp.cosmology.redshiftToLOS(redshift, H)

        # [-180,180] angle in (x,y) plane counterclockw from x-axis towards y:
        phi = np.arctan2(y, x)
        # [0,180] angle from the positive towards the negative z-axis:
        theta = np.arctan2(xy, z)

        xZ = rZ*np.sin(theta)*np.cos(phi) + origin[0]
        yZ = rZ*np.sin(theta)*np.sin(phi) + origin[1]
        zZ = rZ*np.cos(theta) + origin[2]

        # Put the particles back into [0, boxsize]:
        if centerOrigin:
            # the particles were centered around 0,0,0 in the intermediate
            # coordinates, so to put them back in the box, just add halfbox
            halfbox = boxsize/2.
            xZ = rZ*np.sin(theta)*np.cos(phi) + halfbox
            yZ = rZ*np.sin(theta)*np.sin(phi) + halfbox
            zZ = rZ*np.cos(theta) + halfbox
        else:
            xZ = rZ*np.sin(theta)*np.cos(phi) + origin[0]
            yZ = rZ*np.sin(theta)*np.sin(phi) + origin[1]
            zZ = rZ*np.cos(theta) + origin[2]

        self.posZ = np.array([xZ, yZ, zZ]).T
        self.posZCalculated = True






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
