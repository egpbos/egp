#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  untitled.py
#  
#  Copyright 2012 E.G.P.   Bos <pbos@schmidt>
#  

import numpy as np

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
