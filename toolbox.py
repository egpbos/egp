#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
toolbox.py
/toolbox/ module in the /egp/ package.
  
Created by Evert Gerardus Patrick Bos.
Copyright (c) 2012. All rights reserved.
"""

# imports
import numpy as np
import pyublas
import crunch
from matplotlib import pyplot as pl
#~ import __builtin__
import functools
from types import MethodType
from time import time
import traceback
import sys

# constants
__version__ = "0.3, September 2012"

# exception classes
# interface functions
# classes


# decorators

def cacheable(cache_key_template = None):
    """
    Decorator that makes a function cacheable. The /cache_key_template/ argument
    needs to be a string containing the same number of string formatting
    operators (%...) as the number of arguments of the decorated function. Note
    that keyword arguments will be sorted by their keys; this is necessary for
    the keyword argument values to always be in the same order, because keyword
    arguments are given in a dictionary and so they have no defined order. If no
    /cache_key_template/ is given and caching is activated using the cache_on()
    method, then a /cache_key/ keyword argument needs to be added when calling
    the decorated function; it will be used as a keyword in the cache
    dictionary, so make sure the cache_key is something that can be used as
    such. If neither /cache_key_template/ nor a /cache_key/ are given while
    caching is activated, the decorated function will print a message to that
    effect and simply revert to a non-cached function call for that particular
    call.
    
    The /cache_key/ method (without "template") can be useful when caching a
    function that takes NumPy arrays as argument; you can use the /cache_key/ to
    describe the array. If the array is the same every time (like e.g. with the
    k_i/k_abs-arrays), but it is not exactly the same array (it is a different
    object but has the same contents as another with the same gridsize and
    boxlen) then you can manually set the cache_key and still have an efficient
    caching mechanism.
    """
    class Wrapper(object):
        def __init__(self, fct, cache_key_template):
            self.fct = fct
            self.fct_call = self.uncached_fct_call
            self.key = cache_key_template
            self.cache = {}
        def __call__(self, *args, **kwargs):
            #~ print self, self.fct_call
            return self.fct_call(*args, **kwargs)
        def uncached_fct_call(self, *args, **kwargs):
            try:
                del kwargs['cache_key']
                return self.fct(*args, **kwargs)
            except KeyError:
                return self.fct(*args, **kwargs)
        def cached_fct_call(self, *args, **kwargs):
            try:
                #~ print "1"
                kwargs_sort = kwargs.keys()
                #~ print "2"
                kwargs_sort.sort()
                #~ print "3"
                kwargs_values = tuple([kwargs[i] for i in kwargs_sort])
                #~ print "4"
                cache_key = self.key % (args + kwargs_values)
            except TypeError:
                # when self.key is None
                try:
                    #~ print "5"
                    cache_key = kwargs.pop('cache_key')
                except KeyError:
                    print "Caching activated, but no cache_key given! Will not use cache for this call."
                    return self.fct(*args, **kwargs)
            try:
                #~ print "6"
                return self.cache[cache_key]
            except KeyError:
                print "Caching result"
                #~ print "7"
                self.cache[cache_key] = self.fct(*args, **kwargs)
                #~ print "8"
                return self.cache[cache_key]
        def cache_on(self):
            self.fct_call = self.cached_fct_call
        def cache_off(self):
            self.fct_call = self.uncached_fct_call
            self.cache.clear() # maybe not necessary
        def __repr__(self):
            '''Return the function's docstring.'''
            return self.fct.__repr__()
        #~ def __get__(self, obj, objtype):
            #~ '''Support instance methods.'''
            #~ print 'get krijgt: ', obj, objtype
            #~ print 'get geeft : ', functools.partial(self.__call__, obj)
            #~ return functools.partial(self.__call__, obj)
        def __get__(self, instance, owner):
            '''Support instance methods. From:
            http://metapython.blogspot.nl/2010/11/python-instance-methods-how-are-they.html'''
            #~ print 'getting'
            #~ a = time()
            #~ instance.__dict__['__call__'] = MethodType(self, instance, owner)
            #~ print 'getting', self, instance, owner
            #~ try:
                #~ raise AssertionError
            #~ except AssertionError:
                #~ traceback.print_stack()
            #~ self.fct_call = self.fct_call.__get__(instance, owner)
            thing = MethodType(self, instance, owner)
            #~ thing = self.__get__(instance, owner) # dit zou equivalent moeten zijn aan MethodType(self, instance, owner)
            #~ thing = self.__class__(self.fct_call.__get__(instance, owner))
            #~ print time()-a
            return thing
            #~ test = self.fct_call.__get__(instance, owner)
            #~ result = self.__class__(self.fct_call.__get__(instance, owner))
            #~ print "get", self, instance, owner, MethodType(self, instance, owner)
            #~ print test#, result
            #~ return result
            #~ return self.__get__(instance, owner)
            #raise AssertionError
            #return MethodType(self, instance, owner)
            #instance.__call__ = MethodType(self, instance, owner)
            #return instance.__call__
            #~ try:
                #~ return self.as_method
            #~ except AttributeError:
                #~ print "AttributeError!"
                #~ self.as_method = 
                #~ return self.as_method
        #~ def __set__(self, value):
            #~ pass
    
    def closer(f):
        # See http://stackoverflow.com/questions/233673/lexical-closures-in-python#235764
        # on closures.
        return Wrapper(f, cache_key_template)
    
    return closer


# functions

def field_show(field, boxlen, xlabel="y (Mpc)", ylabel="z (Mpc)"):
    """
    Plot a 2D field slice with the right axes on the right side for it to make
    sense to me. In my world, the first axis of an array should represent the
    x-axis, in that if you ask for a[0] in a 2D array /a/ then you should get
    the field entries at x=0 for varying (a[0,-1] would be (x,y)=(0,boxlen)).
    
    By default matplotlib.pyplot's imshow does it the other way around, which
    could of course easily be remedied by a transpose, but this easy function
    does that all for you, and a little more.
    """
    pl.imshow(field.T, origin='bottom', interpolation='nearest', extent=(0,boxlen,0,boxlen))
    pl.xlabel(xlabel)
    pl.ylabel(ylabel)
    pl.colorbar()

# rFFTn's with flipped minus sign convention
def rfftn_flip(A, *args, **kwargs):
    """
    Real N-dimensional fast fourier transform, with flipped minus sign
    convention.
    
    The convention used by NumPy is that the FFT has a minus sign in the
    exponent and the inverse FFT has a plus. This is opposite to the convention
    used e.g. in Numerical Recipes, but, more importantly, it is opposite to the
    fourier transform convention used by Van de Weygaert & Bertschinger (1996).
    This means that if you use the NumPy FFT to compute a constrained field,
    the box will be mirrored and your constraints will not be where you expect
    them to be in the field grid. In this, we assume the field-grid-indices-to-
    physical-coordinates transformation to be simply i/gridsize*boxlen and the
    other way around, physical-coordinates-to-grid-indices transformation to be
    int(x/boxlen*gridsize).
    
    The effect of a changed sign in the FFT convention is a mirroring of your
    in- and output arrays. This is what this function and irfftn_flip thus undo.
    Try plotting np.fft.fft(np.fft.fft(A)) versus A to see for yourself.
    """
    return np.fft.rfftn(A[::-1,::-1,::-1], *args, **kwargs)

def irfftn_flip(A, *args, **kwargs):
    """
    Inverse real N-dimensional fast fourier transform, with flipped minus sign
    convention. See rfftn_flip.
    """
    
    return np.fft.irfftn(A, *args, **kwargs)[::-1,::-1,::-1]

def ifftn_flip(A, *args, **kwargs):
    """
    Inverse N-dimensional fast fourier transform, with flipped minus sign
    convention. See rfftn_flip.
    """
    return np.fft.ifftn(A, *args, **kwargs)[::-1,::-1,::-1]


# Other stuff
def TSC_density(pos, gridsize, boxsize, mass, periodic=True):
	"""Distribute particle masses on a regular grid of gridsize cubed based
	on particle positions in array pos. The masses are distributed using a
	Triangular Shaped Cloud algorithm (quadratic splines), taken from the
	P3M code of Rien van de Weygaert. By default the particle box is taken
	to be periodic; if this is not the case, you can call with argument
	periodic=False. Argument boxsize is the physical size of the box and
	defines the inter-gridpoint-distance.
	Mass of the particles is taken to be constant at the moment and is
	given by argument mass. THIS NEEDS TO BE FURTHER SPECIFIED IF OTHER
	PARTICLE TYPES ARE INCLUDED! E.g. by passing a full mass array.
	This function makes full use of Boost/PyUblas, thanks to Maarten Breddels.
	"""
	
	rho = np.zeros((gridsize,gridsize,gridsize), dtype='float64')
	
	Npart = len(pos)
	pos = np.array(pos, dtype='float64', order='C')
    
	crunch.TSCDensity(pos, rho, Npart, boxsize, gridsize, mass)
	
	return rho

def TSC_density_old(pos, gridsize, boxsize, mass, periodic=True):
	"""Distribute particle masses on a regular grid of gridsize cubed based
	on particle positions in array pos. The masses are distributed using a
	Triangular Shaped Cloud algorithm (quadratic splines), taken from the
	P3M code of Rien van de Weygaert. By default the particle box is taken
	to be periodic; if this is not the case, you can call with argument
	periodic=False. Argument boxsize is the physical size of the box and
	defines the inter-gridpoint-distance.
	Mass of the particles is taken to be constant at the moment and is
	given by argument mass. THIS NEEDS TO BE FURTHER SPECIFIED IF OTHER
	PARTICLE TYPES ARE INCLUDED! E.g. by passing a full mass array.
	This function makes full use of Boost/PyUblas, thanks to Maarten Breddels.
	"""
	
	rho = np.zeros((gridsize,gridsize,gridsize), dtype='float64')
	
	Npart = len(pos)
	
	crunch.TSCDensity(pos.astype('float64'), rho, Npart, boxsize, gridsize, mass)
	
	return rho

def filter_density(density, kernel, kernel_arguments):
    """Kind of speaks for itself, I'd say."""

def gaussian_smooth(densityFourier, r_g, boxlen):
    """Returns the fourier-space representation of the smoothed density field."""
    gridsize = len(densityFourier)
    halfgrid = gridsize/2
    dk = 2*np.pi/boxlen
    k = k_abs_grid(gridsize, boxlen)
    
    def windowGauss(ka, Rg):
        return np.exp( -ka*ka*Rg*Rg/2 ) # N.B.: de /2 factor is als je het veld smooth!
                                        # Het PowerSpec heeft deze factor niet.
    
    return densityFourier*windowGauss(k,r_g)

#~ def calc_k_abs_grid(gridsize, boxlen):
@cacheable("grid_%s_box_%s")
def k_abs_grid(gridsize, boxlen):
    halfgrid = gridsize/2
    dk = 2*np.pi/boxlen
    k12 = np.fft.fftfreq(gridsize, 1/dk/gridsize) # k3 = k12[:halfgrid+1].abs()
    return np.sqrt(k12[:halfgrid+1]**2 + k12[:,np.newaxis]**2 + k12[:,np.newaxis,np.newaxis]**2)

#~ def calc_k_i_grid(gridsize, boxlen):
@cacheable("grid_%s_box_%s")
def k_i_grid(gridsize, boxlen):
    halfgrid = gridsize/2
    dk = 2*np.pi/boxlen
    kmax = gridsize*dk
    k1, k2, k3 = dk*np.mgrid[0:gridsize, 0:gridsize, 0:halfgrid+1]
    k1 -= kmax*(k1 > dk*(halfgrid - 1)) # shift the second half to negative k values
    k2 -= kmax*(k2 > dk*(halfgrid - 1))
    return np.array((k1,k2,k3))

#~ cache_k_grids = False

#~ def k_abs_grid(gridsize, boxlen):
    #~ global cache_k_grids
    #~ if cache_k_grids:
        #~ return KGridCache.k_abs(gridsize, boxlen)
    #~ else:
        #~ return calc_k_abs_grid(gridsize, boxlen)
#~ 
#~ def k_i_grid(gridsize, boxlen):
    #~ global cache_k_grids
    #~ if cache_k_grids:
        #~ return KGridCache.k_i(gridsize, boxlen)
    #~ else:
        #~ return calc_k_i_grid(gridsize, boxlen)

#~ class KGridCache(object):
    #~ """
    #~ This class contains k_abs_grid or k_i_grid results. The functions first look
    #~ if caching is activated, which is done by setting egp.toolbox.cache_k_grids
    #~ to True. If so, they pass the arguments of their current function call to
    #~ this class. It then checks if the arguments have been used before, and if
    #~ so, returns the result. If not, the result is produced and then stored in
    #~ the cache dictionary of this class, so it needn't be calculated again.
    #~ N.B.: Do not instantiate!
    #~ """
    #~ k_abs_cache = {}
    #~ k_i_cache = {}
    #~ 
    #~ def k_abs(gridsize, boxlen):
        #~ cache_key = "%s %s" % (gridsize, boxlen)
        #~ try:
            #~ return KGridCache.k_abs_cache[cache_key]
        #~ except KeyError:
            #~ KGridCache.k_abs_cache[cache_key] = calc_k_abs_grid(gridsize, boxlen)
            #~ return KGridCache.k_abs_cache[cache_key]
    #~ 
    #~ def k_i(gridsize, boxlen):
        #~ cache_key = "%s %s" % (gridsize, boxlen)
        #~ try:
            #~ return KGridCache.k_i_cache[cache_key]
        #~ except KeyError:
            #~ KGridCache.k_i_cache[cache_key] = calc_k_i_grid(gridsize, boxlen)
            #~ return KGridCache.k_i_cache[cache_key]



# Cosmology
def critical_density(cosmo):
    """Gives the critical density, given Cosmology /cosmo/, in units of
    h^2 Msun Mpc^-3."""
    hubble_constant = 100 * 3.24077649e-20 # h s^-1
    gravitational_constant = 6.67300e-11 * (3.24077649e-23)**3 / 5.02785431e-31 # Mpc^3 Msun^-1 s^-2
    rhoc = 3.*hubble_constant**2/8/np.pi/gravitational_constant # critical density (h^2 Msun Mpc^-3)
    return rhoc
