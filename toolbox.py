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

# constants
__version__ = "0.1.1, August 2012"

# exception classes
# interface functions
# classes
# functions
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
