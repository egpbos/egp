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
        pl.imshow(self.t.T, origin='bottom', interpolation='nearest', extent=(0,self.boxlen,0,self.boxlen))
        pl.xlabel(xlabel)
        pl.ylabel(ylabel)
        pl.colorbar()



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


class ParticleSet(object):
    """
    A set of N particles with three-dimensional positions and velocities. Other
    properties may be added later. Arrays /pos/ and /vel/ must have shape (N,3)
    or otherwise be left empty for later assignment.
    """
    def __init__(self, pos = None, vel = None):
        if pos:
            self.pos = pos
        if vel:
            self.vel = vel
