#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  egpICgen.py
#  
#  Created by Evert Gerardus Patrick Bos.
#  Copyright (c) 2012. All rights reserved.

__version__ = "0.4, February 2012"

import numpy as np
from numpy.random import random as rnd, seed as setseed
from numpy import sqrt, log, sin, cos, exp, pi
from scipy.integrate import quadrature as quad

class Cosmology(object):
    """
    Object containing your cosmological parameters. It contains the following:
    * omegaM:    Cosmological matter density parameter
    * omegaB:    Cosmological baryonic matter density parameter
    * omegaL:    Cosmological dark energy density parameter
    * h:         Hubble constant in units of 100 km/s/Mpc
    * trans:     Choice of fluctuation (power) spectrum:
		    spectrum:                  function:
		    power-law                  trans1
		    CDM, adiabatic, DEFW       trans2*
		    CDM, adiabatic, BBKS       trans3
		    HDM, adiabatic, BBKS       trans4*
		    WDM, adiabatic, BBKS       trans5*
		    CDM, isocurvature, BBKS    trans6*
		    CDM, Klypin & Holtzman     trans7
		    * Not yet implemented
    * primn:     Power spectrum index n [-3,1] or primordial N [-3,2]
    * rth:       Tophat window radius in units of Mpc (N.B.: not h^{-1} Mpc!)
    * sigma0:    The galaxy number sigma at tophat radius rth
    * bias:      Bias factor b (default = 1)
    * TCMB:      Temperature of the CMB spectrum (default = 2.7 Kelvin)
    """
    def __init__(self, choice):
	"""
	Initialize a Cosmology instance with some default values.
	"""
	if not choice:
	    self.omegaM, self.omegaB, self.omegaL, self.h = 0.3, 0.04, 0.7, 0.7
	    self.rth, self.sigma0 = 8.0, 0.8
	    self.primn = 1.0
	elif choice is "WMAP3":
	    self.omegaM, self.omegaB, self.omegaL, self.h = 0.268, 0.044, 0.732, 0.704
	    self.rth, self.sigma0 = 8.0, 0.776
	    self.primn = 0.947
	elif choice is "WMAP7":
	    self.omegaM, self.omegaB, self.omegaL, self.h = 0.272, 0.0456, 0.728, 0.704
	    self.rth, self.sigma0 = 8.0, 0.809
	    self.primn = 0.963
	self.bias, self.TCMB = 1.0, 2.7
	self.trans = None

class DensityField(object):
    """
    Contains the density field itself, interpolated onto a discrete 3D grid, and
    the field's discrete fourier space representation. N.B.: the discrete
    fourier space representation is not the true fourier space representation;
    these differ by a factor proportional to dk**3 (c.f. Press+07, eqn. 12.1.8).
    """
    def __init__(self, rho=None, rhoF=None):
	self.rho = rho
	self.rhoF = rhoF
    
    rho, rhoF = property(), property()
    
    @rho.getter
    def rho(self):
	if not self._rho:
	    self.rho = self._ifft(self.rhoF)
	    self.rho *= np.size(self.rho) # factor from discrete to true Fourier transform
	return self._rho
    @rho.setter
    def rho(self, field):
	self._rho = field
    @rhoF.getter
    def rhoF(self):
	if not self._rhoF:
	    self._rhoF = np.fft.rfftn(self.rho)/np.size(self.rho)
	return self._rhoF
    @rhoF.setter
    def rhoF(self, field):
	if not field:
	    self._rhoF = field
	    self._ifft = np.fft.irfftn
	elif field.shape[0] == field.shape[2]:
	    self._rhoF = field
	    self._ifft = np.fft.ifftn
	elif field.shape[0] == (field.shape[2]-1)*2:
	    self._rhoF = field
	    self._ifft = np.fft.irfftn
	    

#class GaussianRandomField(DensityField):
    
#class MeanField(DensityField)