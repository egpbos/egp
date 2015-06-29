#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
toolbox.py
/toolbox/ module in the /egp/ package.
  
Created by Evert Gerardus Patrick Bos.
Copyright (c) 2012-2015. All rights reserved.
"""

# imports
import numpy as np
try:
    import pyublas
    import crunch
except:
    print "pyublas and egp.crunch not imported!"
#~ import __builtin__

import re  # for natural sort and reglob
import types
import functools
from time import time
import traceback
import sys
import os
import csv
import egp.basic_types

import scipy.optimize # for fitting 1D functions


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
            self.__name__ = fct.__name__
            self.__doc__ = fct.__doc__
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
            thing = types.MethodType(self, instance, owner)
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
    from matplotlib import pyplot as pl
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


# Fitting 1D functions
sqrt2pi = np.sqrt(2*np.pi)
normal_fit = lambda p, x: 1./(p[1] * sqrt2pi) * np.exp(-0.5 * ((x - p[0])/p[1])**2)
powerlaw_fit = lambda p, x: p[1] * x**p[0]
powerlaw_norm_fit = lambda p, x: (x/p[1])**p[0]

def fit_1D_fct(fitfunc, p0, x, y):
    errfunc = lambda p, x, y: fitfunc(p, x) - y     # difference between fitfunc and y
    pfit, success = scipy.optimize.leastsq(errfunc, p0[:], args=(x, y))
    return pfit

def fit_2D_fct(fitfunc, p0, x1, x2, y):
    errfunc = lambda p, x1, x2, y: fitfunc(p, x1, x2) - y     # difference between fitfunc and y
    pfit, success = scipy.optimize.leastsq(errfunc, p0[:], args=(x1, x2, y))
    return pfit


# natural sorting
# from http://stackoverflow.com/a/16090640/1199693:
def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(_nsre, s)]


def natural_sorted(l):
    return sorted(l, key=natural_sort_key)


# regular expression glob
# From http://stackoverflow.com/a/17197678/1199693:
def reglob(path, exp, invert=False):
    """glob.glob() style searching which uses regex

    :param exp: Regex expression for filename
    :param invert: Invert match to non matching files
    """

    m = re.compile(exp)

    if invert is False:
        res = [f for f in os.listdir(path) if m.search(f)]
    else:
        res = [f for f in os.listdir(path) if not m.search(f)]

    res = map(lambda x: "%s/%s" % (path, x,), res)
    return res


def running_mean(arr, window_size=100):
    x = range(window_size/2, len(arr) - window_size + window_size/2 + 1)
    y = np.convolve(arr, np.ones((window_size,))/window_size, mode='valid')
    return x, y



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

def filter_Field(field, kernel, kernel_arguments, gridsize=None):
    """Returns a new Field object that is the input Field /field/
    convolved with a kernel. Kind of speaks for itself, I'd say. Use
    gaussian_kernel or tophat_kernel for /kernel/ and a list of
    appropriate kernel arguments in /kernel_arguments/. Gridsize can be
    optionally specified, otherwise the first shape element of
    fieldFourier of /field.f/ will be used."""
    field_fourier = field.f
    boxlen = field.boxlen
    if not gridsize:
        gridsize = field.f.shape[0]
    field_fourier_filtered = filter_field(field_fourier, boxlen, kernel, kernel_arguments, gridsize)
    return egp.basic_types.Field(fourier = field_fourier_filtered)

def filter_field(fieldFourier, boxlen, kernel, kernel_arguments, gridsize=None):
    """Returns the fourier-space representation of the field
    convolved with a kernel. Kind of speaks for itself, I'd say. Use
    gaussian_kernel or tophat_kernel for /kernel/ and a list of
    appropriate kernel arguments in /kernel_arguments/. Gridsize can be
    optionally specified, otherwise the first shape element of
    fieldFourier will be used."""
    if not gridsize:
        gridsize = fieldFourier.shape[0]
    k = k_abs_grid(gridsize, boxlen)
    return fieldFourier * kernel(k, *kernel_arguments)

def gaussian_kernel(k, r_g):
    return np.exp( -k*k*r_g*r_g/2 )

def tophat_kernel(k, r_th):
    x = k * r_th
    kernel = 3*(np.sin(x) - x*np.cos(x))/x**3
    return np.nan_to_num(kernel) # to get out the divisions by zero

def gaussian_smooth(densityFourier, r_g, boxlen):
    """Returns the fourier-space representation of the smoothed density field."""
    gridsize = len(densityFourier)
    k = k_abs_grid(gridsize, boxlen)
    
    def windowGauss(ka, Rg):
        return np.exp( -ka*ka*Rg*Rg/2 ) # N.B.: de /2 factor is als je het veld smooth!
                                        # Het PowerSpec heeft deze factor niet.
    
    return densityFourier*windowGauss(k,r_g)

def tophat_smooth(densityFourier, r_th, boxlen):
    """Returns the fourier-space representation of the smoothed density field."""
    gridsize = len(densityFourier)
    k = k_abs_grid(gridsize, boxlen)

#~ def calc_k_abs_grid(gridsize, boxlen):
@cacheable("grid_%s_box_%s")
def k_abs_grid(gridsize, boxlen):
    """k_abs_grid(gridsize, boxlen)"""
    halfgrid = gridsize/2
    dk = 2*np.pi/boxlen
    k12 = np.fft.fftfreq(gridsize, 1/dk/gridsize) # k3 = k12[:halfgrid+1].abs()
    return np.sqrt(k12[:halfgrid+1]**2 + k12[:,np.newaxis]**2 + k12[:,np.newaxis,np.newaxis]**2)

#~ def calc_k_i_grid(gridsize, boxlen):
@cacheable("grid_%s_box_%s")
def k_i_grid(gridsize, boxlen):
    """k_i_grid(gridsize, boxlen)"""
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
def critical_density(cosmo=None):
    """Gives the critical density in units of h^2 Msun Mpc^-3.
    N.B.: cosmo is not necessary in these units (left it in as deprecated)."""
    hubble_constant = 100 * 3.24077649e-20  # h s^-1
    gravitational_constant = 6.67300e-11 * (3.24077649e-23)**3 / 5.02785431e-31  # Mpc^3 Msun^-1 s^-2
    rhoc = 3.*hubble_constant**2/8/np.pi/gravitational_constant  # critical density (h^2 Msun Mpc^-3)
    return rhoc


def particle_mass(omegaM, boxlen, gridsize):
    rho_c = egp.toolbox.critical_density()
    mass = omegaM * rho_c * boxlen**3 / (gridsize**3) / 1e10  # 10^10 M_sol / h
    return mass


def sigma_R(field, scale):
    """Calculates the STD of the Field on a given /scale/ by first filtering
    with a top-hat filter and then just calculating the STD of the filtered
    Field. Can e.g. be used to find sigma_8 (when scale = 8)."""
    field_filtered = filter_Field(field, tophat_kernel, (scale,))
    return field_filtered.t.std()


# Statistics
def quick_power(field):
    field_f = np.fft.rfftn(field)
    return (field_f * field_f.conj()).real


def cross_correlation(field_one, field_two, gridsize, boxlen, Nbin=None):
    """
    Input fields must be Fourier transforms of the true fields.
    Note: only valid for real fields! The imaginary parts drop out due to
    the symmetry in the Fourier transforms of real fields.
    If Nbin is None, a sensible number of bins is automatically calculated.
    The returned spectrum always has Nbin + 1 elements; the last element
    contains only the Nyquist component and should be considered very noisy.

    TODO: redo this function in c++ to avoid slow Python for-loops.
    """
    if Nbin is None:
        Nbin = gridsize
        # rmax = boxlen/2 * np.sqrt(3)
        # dmin = boxlen/gridsize
        # Nbin = int(np.ceil(rmax/dmin))

    cross = field_one.real * field_two.real + field_one.imag * field_two.imag
    k = egp.toolbox.k_abs_grid(gridsize, boxlen)
    dk = k.max() / Nbin
    print k.argmax(), k.max(), k.min(), k[0,0,0]-k[0,0,1], dk

    spec = np.zeros(Nbin + 1)
    mode = np.zeros(Nbin + 1)
    for ii in range(gridsize):
        for jj in range(gridsize):
            for kk in range(gridsize/2+1):
                # try:
                    ix_bin = int(k[ii, jj, kk]/dk)
                    spec[ix_bin] += cross[ii, jj, kk]
                    mode[ix_bin] += 1
                # except IndexError as error:
                #     if ix_bin == Nbin:  # k_max, which is always out of bounds
                #         continue
                #     else:
                #         raise error

    spec /= mode

    kbins = np.arange(0, k.max()+dk, dk)

    return kbins, spec


def power_spectrum(field, gridsize, boxlen, Nbin=None):
    return cross_correlation(field, field, gridsize, boxlen, Nbin=Nbin)


def norm_cross_correlation(field_one, field_two, gridsize, boxlen, Nbin=None):
    """
    Essentially returns a cosine similarity between the two fields (in Fourier
    space).
    """
    kG, G = cross_correlation(field_one, field_two, gridsize, boxlen, Nbin=Nbin)
    kP1, P1 = power_spectrum(field_one, gridsize, boxlen, Nbin=Nbin)
    kP2, P2 = power_spectrum(field_two, gridsize, boxlen, Nbin=Nbin)
    return kG, G/np.sqrt(P1)/np.sqrt(P2)


# Other useful stuff

def x_to_xms(x_decimal):
    x = x_decimal // 1
    minutes_decimal = (x_decimal - x)*60
    minutes = minutes_decimal // 1
    seconds_decimal = (x_decimal - x - minutes/60)*3600
    return x, minutes, seconds_decimal


def hms_to_deg(h,m,s):
    h_decimal = h + m/60. + s/3600.
    degrees_decimal = h_decimal/24.*360
    return degrees_decimal


def hms_to_dms(h,m,s):
    degrees_decimal = hms_to_deg(h,m,s)
    return x_to_xms(degrees_decimal)


def dms_to_deg(d,m,s):
    degrees_decimal = d + m/60. + s/3600.
    return degrees_decimal


def dms_to_hms(d, m, s):
    degrees_decimal = dms_to_deg(d, m, s)
    hours_decimal = degrees_decimal/360.*24
    return x_to_xms(hours_decimal)


def load_csv_catalog(filename):
    table_file = open(filename)
    table = csv.reader(table_file)
    table.next() # skip header
    catalog = []
    for entry in table:
        catalog.append(entry)
    table_file.close()
    return catalog


def fill_template_file(filename, value_dict):
    """
    Open a text file with Python string replacements in it (the
    %(variable_name)s type ones) and replace them by the values of the
    corresponding keys in the /value_dict/.
    """
    f = open(filename, 'r')
    text = f.read()
    f.close()
    f = open(filename, 'w')
    f.write(text % value_dict)
    f.close()


def mkdir(path):
    try:
        os.stat(path)
    except:
        os.mkdir(path)


def binfloat(filename):
    """Open a binary file with float32 format."""
    return np.memmap(filename, dtype='float32')


def binint(filename):
    """Open a binary file with int32 format."""
    return np.memmap(filename, dtype='int32')
