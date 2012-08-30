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

# constants
__version__ = "0.2.1, August 2012"

# exception classes
# interface functions
# classes
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

def k_abs_grid(gridsize, boxlen):
    halfgrid = gridsize/2
    dk = 2*np.pi/boxlen
    k12 = np.fft.fftfreq(gridsize, 1/dk/gridsize) # k3 = k12[:halfgrid+1].abs()
    return np.sqrt(k12[:halfgrid+1]**2 + k12[:,np.newaxis]**2 + k12[:,np.newaxis,np.newaxis]**2)

def k_i_grid(gridsize, boxlen):
    halfgrid = gridsize/2
    dk = 2*np.pi/boxlen
    kmax = gridsize*dk
    k1, k2, k3 = dk*np.mgrid[0:gridsize, 0:gridsize, 0:halfgrid+1]
    k1 -= kmax*(k1 > dk*(halfgrid - 1)) # shift the second half to negative k values
    k2 -= kmax*(k2 > dk*(halfgrid - 1))
    return np.array((k1,k2,k3))


# Cosmology
def critical_density(cosmo):
    """Gives the critical density, given Cosmology /cosmo/, in units of
    h^2 Msun Mpc^-3."""
    hubble_constant = 100 * 3.24077649e-20 # h s^-1
    gravitational_constant = 6.67300e-11 * (3.24077649e-23)**3 / 5.02785431e-31 # Mpc^3 Msun^-1 s^-2
    rhoc = 3.*hubble_constant**2/8/np.pi/gravitational_constant # critical density (h^2 Msun Mpc^-3)
    return rhoc
