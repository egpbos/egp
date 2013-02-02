#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
icgen.py
/icgen/ module in the /egp/ package.
  
Created by Evert Gerardus Patrick Bos.
Copyright (c) 2012. All rights reserved.
"""

# imports
import numpy as np
import pyublas
from numpy.random import random as rnd, seed as setseed
from scipy.integrate import quadrature as integrate
from scipy.interpolate import InterpolatedUnivariateSpline as Interpolator
from scipy.special import erf
from csv import reader as csvreader

from egp.crunch import resolution_independent_random_grid
from egp.basic_types import Field, VectorField, ParticleSet, PeriodicArray
from egp import toolbox

# constants
# exception classes
# interface functions
# classes
class Cosmology(object):
    """
    Object containing your cosmological parameters. It contains the following:
    * omegaM:    Cosmological matter density parameter
    * omegaB:    Cosmological baryonic matter density parameter
    * omegaL:    Cosmological dark energy density parameter
    * h:         Hubble constant in units of 100 km/s/Mpc
    * trans:     Choice of fluctuation (power) spectrum:
            spectrum:                  function:
            power-law                  1
            CDM, adiabatic, DEFW       2*
            CDM, adiabatic, BBKS       3
            HDM, adiabatic, BBKS       4*
            WDM, adiabatic, BBKS       5*
            CDM, isocurvature, BBKS    6*
            CDM, Klypin & Holtzman     7
            CDM, Eisenstein & Hu 99    8
            * Not yet implemented
    * primn:     Power spectrum index n [-3,1] or primordial N [-3,2]
    * rth:       Tophat window radius in units of Mpc (N.B.: not h^{-1} Mpc!)
    * sigma0:    The galaxy number sigma at tophat radius rth
    * bias:      Bias factor b (default = 1)
    * TCMB:      Temperature of the CMB spectrum (default = 2.7 Kelvin)
    """
    def __init__(self, choice = None, trans = None):
        """
        Initialize a Cosmology instance with some default values.
        """
        if not choice:
            self.omegaM, self.omegaB, self.omegaL, self.omegaR, self.h = 0.3, 0.04, 0.7, 0.0, 0.7
            self.rth, self.sigma0 = 8.0, 0.8
            self.w_0, self.w_a = -1.0, 0.0
            self.primn = 1.0
        elif choice.lower() == "wmap3":
            self.omegaM, self.omegaB, self.omegaL, self.omegaR, self.h = 0.268, 0.044, 0.732, 0.0, 0.704
            self.rth, self.sigma0 = 8.0, 0.776
            self.w_0, self.w_a = -1.0, 0.0
            self.primn = 0.947
        elif choice.lower() == "wmap7":
            self.omegaM, self.omegaB, self.omegaL, self.omegaR, self.h = 0.272, 0.0456, 0.728, 0.0, 0.704
            self.rth, self.sigma0 = 8.0, 0.809
            self.w_0, self.w_a = -1.0, 0.0
            self.primn = 0.963
        self.bias, self.TCMB = 1.0, 2.7
        self.trans = trans
    
    def __str__(self):
        out = ""
        out +=   "Omega_matter:                     " + str(self.omegaM)
        out += "\nOmega_baryon:                     " + str(self.omegaB)
        out += "\nOmega_lambda:                     " + str(self.omegaL)
        out += "\nOmega_radiation:                  " + str(self.omegaR)
        out += "\nHubble constant:                  " + str(self.h     )
        out += "\nBias parameter:                   " + str(self.bias  )
        out += "\nCMB temperature:                  " + str(self.TCMB  )
        out += "\nPower spectrum index:             " + str(self.primn )
        out += "\nPower spectrum transfer function: " + str(self.trans )
        out += "\nPower spectrum top-hat radius:    " + str(self.rth   )
        out += "\nPower spectrum sigma_0:           " + str(self.sigma0)
        return out


class PowerSpectrum(object):
    """
    This meta-object can be used in the construction of Gaussian Random Fields,
    but also for just using the power spectrum as an object by itself. It
    contains all the necessary stuff to compute the power spectrum for any given
    array of k-values. It can be dependent on Cosmology or interpolated after 
    being loaded from a file or estimated.
    Before actual use, it needs to be normalized. This can be done by setting
    self.amplitude manually or by using the normalize function which normalizes
    the spectrum using the volume of the periodic box, and setting sigma0 at
    scale Rth.
    To activate caching, call the P_call.cache_on() method. You will then need
    to give a /cache_key/ keyword argument with your calls.
    #~ When a cache_key is given, the calculated power spectrum will be stored in
    #~ the self.cache dictionary under the key given by cache_key. The next time
    #~ the power spectrum is called with the same cache_key, the spectrum will
    #~ not be recalculated, but will immediately be read from memory. This can be
    #~ useful when calling the same PowerSpectrum several times, e.g. from within
    #~ a loop, without using variables outside of the loop's scope.
    """
    def __init__(self):
        self.normalized = False
        self.cache = {} # contains cached 
    
    @toolbox.cacheable()
    def __call__(self, k):#, cache_key = None):
        return self.amplitude * self.P(k) # cacheable method
        #~ return self.P_call(k) # P_call method
        #~ if not cache_key: # internal cache method
            #~ return self.amplitude * self.P(k)
        #~ else:
            #~ try:
                #~ return self.cache[cache_key]
            #~ except KeyError:
                #~ self.cache[cache_key] = self.amplitude * self.P(k)
                #~ return self.cache[cache_key]
    
    def P_call(self, k):
        """This will be returned on calling the PowerSpectrum instance. Needs to
        be a separate function from __call__ to be able to make it cacheable. To
        cache the PowerSpectrum, therefore, you need to set cache_on() on this
        function."""
        return self.amplitude * self.P(k)
    
    amplitude = property()
    @amplitude.getter
    def amplitude(self):
        try:
            return self._amplitude
        except AttributeError:
            raise AttributeError("Power Spectrum not yet normalized!")
    @amplitude.setter
    def amplitude(self, value):
        self._amplitude = value
        self.normalized = True
    
    def P(self, k):
        raise AttributeError("The PowerSpectrum class is a meta-class and not meant to actually be used! P(k) is not defined in this object. Use a subclass like CosmoPowerSpectrum or InterpolatedPowerSpectrum.")
    
    def normalize(self, volume, Rth, sigma0, maxiter=100):
        """Normalize the power spectrum for the periodic field in box of
        /volume/ that it will be used on, on a top-hat scale Rth with sigma0.
        /volume/ must have units h^-3 Mpc^3."""
        
        self.Rth = Rth
        self.sigma0 = sigma0
        
        kSwitch = 2*np.pi/Rth
        s1 = integrate(self.integrant, 0, kSwitch, maxiter=maxiter)[0]
        s2 = integrate(self.integrantInv, 1e-30, 1/kSwitch, maxiter=maxiter)[0]
        
        # The volume/(2pi)^3 is a normalization of the convolution (the
        # integral) that is used to normalize the power spectrum. sigma0 is the
        # actual value of the power spectrum that you want to give it at scale
        # Rth and (s1+s2)*volume/(2pi)^3 is the sigma0 as calculated from the
        # given power spectrum, so you divide that out to set it to one at Rth.
        self.amplitude = sigma0**2 * (2*np.pi)**3 / volume / (s1 + s2)
    
    def moment(self, order, Rg, volume, maxiter=100):
        """Calculate the spectral moment of order /order/ of the power spectrum
        by convolving with a Gaussian window of radius /Rg/ over the power
        spectrum."""
        amp = self.amplitude # we want to convolve the normalized power spectrum
        kSwitch = 2*np.pi/Rg
        s1 = integrate(self.moment_integrant, 0, kSwitch, \
             args = (order, Rg), maxiter=maxiter)[0]
        s2 = integrate(self.moment_integrantInv, 1e-30, 1/kSwitch, \
             args = (order, Rg), maxiter=maxiter)[0]
        return np.sqrt( amp * (s1+s2) * volume / (2*np.pi)**3 )
    
    def integrant(self, k):
        """Integrand used to determine power spectrum amplitude"""
        return self.P(k) * self.windowTopHat(k) * 4.0*np.pi*k**2
        
    def integrantInv(self, k):
        """Inverse of integrand used to determine power spectrum amplitude"""
        k = 1.0/k
        return self.P(k) * self.windowTopHat(k) * 4.0*np.pi*k**4
    
    def moment_integrant(self, k, order, Rg):
        """Integrand used to determine power spectrum amplitude"""
        return self.P(k) * self.windowGauss(k, Rg) * 4.0*np.pi*k**2 * k**(2*order)
        
    def moment_integrantInv(self, k, order, Rg):
        """Inverse of integrand used to determine power spectrum amplitude"""
        k = 1.0/k
        return self.P(k) * self.windowGauss(k, Rg) * 4.0*np.pi*k**4 * k**(2*order)
    
    def windowTopHat(self, k):
        """Top-hat window function"""
        kw = k*self.Rth
        return 9.0*(np.sin(kw)-kw*np.cos(kw))**2/kw**6
    
    def windowGauss(self, k, Rg):
        return np.exp( -k*k*Rg*Rg )


class CosmoPowerSpectrum(PowerSpectrum):
    """
    PowerSpectrum based on a given Cosmology. Values calculated when needed.
    """
    def __init__(self, cosmology):
        PowerSpectrum.__init__(self)
        self.cosmology = cosmology
        if self.cosmology.trans:
            exec("self.trans = self.trans"+str(self.cosmology.trans))
        else:
            raise AttributeError('No transfer function chosen in Cosmology!')
    
    def normalize(self, volume, maxiter = 200):
        PowerSpectrum.normalize(self, volume, self.cosmology.rth, self.cosmology.sigma0, maxiter)
    
    def P(self, k):
        return k**self.cosmology.primn * self.trans(k)**2
    
    def trans1(self, k):
        """Transfer function for power-law power spectrum"""
        return 1.0
        
    def trans3(self, k):
        """
        Transfer function for the Cold Dark Matter spectrum for adiabatic
        fluctuations as given by: 
            Bardeen, Bond, Kaiser and Szalay,
            Astrophys. J. 304, 15 (1986)
        """
        q = k/self.cosmology.omegaM/self.cosmology.h**2
        return np.log(1+2.34*q)/2.34/q/(1 + 3.89*q + (16.1*q)**2 + (5.46*q)**3 + (6.71*q)**4)**(1./4)

    def trans7(self, k):
        """
        Transfer function for CDM, Klypin & Holtzman power spectrum.
        Reference: astro-ph/9712217.
        """
        omega = self.cosmology.omega
        omegaB = self.cosmology.omegaB
        h = self.cosmology.h
        T27 = self.cosmology.TCMB/2.7
        
        a1 = (46.9*omega*h**2)**0.670 * (1 + (32.1*omega*h**2)**(-0.532))
        a2 = (12*omega*h**2)**0.424 * (1 + (45*omega*h**2)**(-0.582))
        alpha = a1**(-omegaB/omega)*a2**(-(omegaB/omega)**3)
        q = k*T27**2 / (omega*h**2*np.sqrt(alpha)*(1-omegaB/omega)**0.60)
        return np.log(1 + 2.34*q)/2.34/q * (1 + 13*q + (10.5*q)**2 + (10.4*q)**3 + (6.51*q)**4)**(-0.25)

    def trans8(self, k):
        """
        Transfer function for CDM with baryonic features from Eisenstein & Hu 1999.
        Reference: astro-ph/9709112.
        
        Note: this function sometimes turns off divide-by-zero warnings,
        so check for them yourself!
        """
        error_setting = np.geterr()['divide']
        
        omegaB = self.cosmology.omegaB
        omegaM = self.cosmology.omegaM
        omegaCDM = omegaM - omegaB
        h = self.cosmology.h
        T27 = self.cosmology.TCMB/2.7
        
        kEq = 7.46e-2*omegaM*h*h*T27**(-2) # Mpc^-1
        q = k/13.41/kEq
        kSilk = 1.6*(omegaB*h*h)**0.52 * (omegaM*h*h)**0.73 * \
                (1 + (10.4*omegaM*h*h)**-0.95) # Mpc^-1
        
        b1 = 0.313*(omegaM*h*h)**(-0.419) * (1 + 0.607*(omegaM*h*h)**0.674)
        b2 = 0.238*(omegaM*h*h)**0.223
        zd = 1291*(omegaM*h*h)**0.251 * (1+b1*(omegaB*h*h)**b2) / \
             (1+0.659*(omegaM*h*h)**0.828)
        zEq = 2.50e4*omegaM*h*h*T27**(-4)
            
        R = lambda z: 31.5*omegaB*h*h*T27**(-4)*(10**3/z)
        Rd = R(zd)
        Req = R(zEq)
        
        s = 2./3/kEq*np.sqrt(6/Req)*np.log( (np.sqrt(1+Rd) + np.sqrt(Rd+Req)) / (1+np.sqrt(Req)) )
        
        y = (1+zEq)/(1+zd)
        G = y*(-6*np.sqrt(1+y) + (2+3*y)*np.log((np.sqrt(1+y) + 1)/(np.sqrt(1+y)-1)))
        alphaB = 2.07*kEq*s*(1+Rd)**(-3./4) * G
        
        betaB = 0.5 + omegaB/omegaM + (3 - 2*omegaB/omegaM) * \
                np.sqrt((17.2*omegaM*h*h)**2 + 1)
        
        a1 = (46.9*omegaM*h*h)**0.670 * (1 + (32.1*omegaM*h*h)**-0.532)
        a2 = (12.0*omegaM*h*h)**0.424 * (1 + (45.0*omegaM*h*h)**-0.582)
        alphaC = a1**(-omegaB/omegaM) * a2**(-(omegaB/omegaM)**3)
        
        b1_betaC = 0.944/(1 + (458*omegaM*h*h)**-0.708)
        b2_betaC = (0.395*omegaM*h*h)**-0.0266
        betaC = 1/( 1 + b1_betaC*((omegaCDM/omegaM)**b2_betaC - 1) )
            
        betaNode = 8.41*(omegaM*h*h)**0.435
        
        np.seterr(divide = 'ignore') # Divide by zero warning off
        st = s / (1+(betaNode/k/s)**3)**(1./3)# s-tilde
        np.seterr(divide = error_setting) # Divide by zero warning back on

        C = lambda alC: 14.2/alC + 386/(1+69.9*q**1.08)
        Ttilde0 = lambda k, alC, beC: np.log(np.e + 1.8*beC*q) / \
                  ( np.log(np.e+1.8*beC*q) + C(alC)*q*q )
        
        np.seterr(divide = 'ignore') # Divide by zero warning off
        Tb = (( Ttilde0(k,1,1)/(1+(k*s/5.2)**2) + alphaB/(1+(betaB/k/s)**3) * \
             np.exp(-(k/kSilk)**1.4) )) * np.sinc(k*st/2/np.pi)
        np.seterr(divide = error_setting) # Divide by zero warning back on
        
        f = 1/(1+(k*s/5.4)**4)
        Tcdm = f*Ttilde0(k,1,betaC) + (1-f)*Ttilde0(k,alphaC,betaC)
        
        return omegaB/omegaM*Tb + omegaCDM/omegaM*Tcdm


class InterpolatedPowerSpectrum(PowerSpectrum):
    """
    PowerSpectrum based on interpolated input. Values calculated when needed.
    The interpolator is the scipy.interpolation.InterpolatedUnivariateSpline
    class, which can use several kinds of spline orders. The default order is 1.
    Possible choices range up to and including 5th order. The PowerSpectrum is
    initialized by passing the known k and P(k) values.
    For normalization, the function will always go out of the given k range.
    This means there must be some extrapolation. By default, the border values
    of pInit are used. This should be improved in a future version by being able
    to specify the extrapolation order.
    """
    def __init__(self, kInit, pInit, order=1, extrapolation_order = None):
        PowerSpectrum.__init__(self)
        # remove duplicates:
        mask = np.r_[True, (np.diff(kInit) > 0)]
        kInit = kInit[mask]
        pInit = pInit[mask]
        self.P = Interpolator(kInit, pInit, k=order)


class EstimatedPowerSpectrum(InterpolatedPowerSpectrum):
    """
    InterpolatedPowerSpectrum based on an estimation of the power spectrum of a
    given DensityField.
    Needs to be implemented to be able to build ConstrainedFields with the
    Hoffman-Ribak method from van de Weygaert & Bertschinger (1996) based on
    DensityFields that are not built up from a power spectrum like the
    GaussianRandomField is.
    """
    def __init__(self, density):
        pass


class DensityField(Field):
    """
    A single component Field representing density on a periodic grid.
    """


class DisplacementField(VectorField):
    """
    A three component VectorField representing the displacement field
    corresponding to a DensityField on a periodic grid. N.B.: the DensityField
    must have attribute /boxlen/ set. The components are stored as attributes x,
    y and z, which are Field instances.
    
    Gives the displacement field in the units as the box length, which should be
    h^{-1} Mpc (or kpc, but at least in "Hubble units", with h^{-1}!).
    """
    def __init__(self, density):
        self.density = density
        try:
            self.gridsize = self.density.gridsize
        except AttributeError:
            self.gridsize = self.density.t.shape[0]
        self.boxlen = self.density.boxlen
        self.build_fourier() # this method calls VectorField.__init__()
    
    def build_fourier(self):
        #~ halfgrid = self.gridsize/2
        #~ dk = 2*np.pi / self.boxlen
        #~ kmax = self.gridsize*dk
        
        # Initialize fourier-space k-values grid
        k1, k2, k3 = toolbox.k_i_grid(self.gridsize, self.boxlen)
        #~ k1, k2, k3 = dk*np.mgrid[0:self.gridsize, 0:self.gridsize, 0:halfgrid+1]
        #~ k1 -= kmax*(k1 > dk*(halfgrid - 1)) # shift the second half to negative k values
        #~ k2 -= kmax*(k2 > dk*(halfgrid - 1))
        k_sq = k1**2 + k2**2 + k3**2
        k_sq[0,0,0] = 1 # to avoid division by zero

        # Indices for the 7 real, independent, non-zero grid values:
        real7x,real7y,real7z = np.mgrid[0:2,0:2,0:2]*self.gridsize/2
        real7x,real7y,real7z = real7x.ravel()[1:],real7y.ravel()[1:],real7z.ravel()[1:]
        
        # Then, the actual displacement field:
        #~ Z = 1.0j/k_sq/self.boxlen * self.density.f # This was also missing a minus sign (at least in the new fourier convention)
        Z = -1.0j/k_sq * self.density.f
        Z[real7x,real7y,real7z] = 0.0 # waarom dit eigenlijk?
        Z[0,0,0] = 0.0
        VectorField.__init__(self, fourier = (k1*Z, k2*Z, k3*Z))
        
        # Finally add symmetry to the nyquist planes (so the ifft is not imaginary):
        symmetrizeMatrix(self.x.f)
        symmetrizeMatrix(self.y.f)
        symmetrizeMatrix(self.z.f)


class DisplacementField2ndOrder(VectorField):
    """
    A three component VectorField representing the 2nd order displacement field
    corresponding to a DisplacementField on a periodic grid. N.B.: the
    DisplacementField must have attribute /boxlen/ set. The components are
    stored as attributes x, y and z, which are Field instances.
    
    Gives the 2nd order displacement field in the units as the box length, which
    should be h^{-1} Mpc (or kpc, but at least in "Hubble units", with h^{-1}!).
    """
    def __init__(self, psi1):
        self.psi1 = psi1
        try:
            self.gridsize = self.psi1.gridsize
        except AttributeError:
            self.gridsize = self.psi1.x.t.shape[0]
        self.boxlen = self.psi1.boxlen
        self.build_fourier() # this method calls VectorField.__init__()
    
    def build_fourier(self):
        # Initialize fourier-space k-values grid
        k1, k2, k3 = toolbox.k_i_grid(self.gridsize, self.boxlen)
        k_sq = k1**2 + k2**2 + k3**2
        k_sq[0,0,0] = 1 # to avoid division by zero

        # Indices for the 7 real, independent, non-zero grid values:
        real7x,real7y,real7z = np.mgrid[0:2,0:2,0:2]*self.gridsize/2
        real7x,real7y,real7z = real7x.ravel()[1:],real7y.ravel()[1:],real7z.ravel()[1:]
        
        # Now, let's calculate stuff...
        # ... the second derivatives of psi(1):
        d11 = Field(fourier = self.psi1.x.f*k1)
        d22 = Field(fourier = self.psi1.y.f*k2)
        d33 = Field(fourier = self.psi1.z.f*k3)
        d12 = Field(fourier = self.psi1.x.f*k2)
        d13 = Field(fourier = self.psi1.x.f*k3)
        d23 = Field(fourier = self.psi1.y.f*k3)
        # ... nabla_q^2 phi(2):
        nabla2_phi2 = Field(true = d11.t*d22.t - d12.t**2 + d11.t*d33.t - d13.t**2 + d22.t*d33.t - d23.t**2 )
        del d11, d22, d33, d12, d13, d23
        # ... and phi(2):
        phi2 = nabla2_phi2.f/k_sq
        
        # --- Fix shit
        phi2[real7x,real7y,real7z] = 0.0 # waarom dit eigenlijk?
        phi2[0,0,0] = 0.0
        
        # And finish psi(2) (the actual second order displacement field):
        psi2 = (k1*phi2, k2*phi2, k3*phi2)
        VectorField.__init__(self, fourier = psi2)
        
        # Finally add symmetry to the nyquist planes (so the ifft is not imaginary):
        symmetrizeMatrix(self.x.f)
        symmetrizeMatrix(self.y.f)
        symmetrizeMatrix(self.z.f)


class GaussianRandomField(DensityField):
    """
    Generates a Gaussian random DensityField based on a given PowerSpectrum for
    use as initial conditions in a cosmological N-body simulation. Before use,
    the Zel'dovich approximation needs to be applied on a DisplacementField 
    (based e.g. on this DensityField) to obtain actual particle positions and
    velocities.
    
    Input for initialization:
    - power:     A PowerSpectrum, defining the Gaussian random DensityField.
    - boxlen:    Length of the sides of the box that will be produced.
                 Units of h^{-1} Mpc.
    - gridsize:  Number of grid points in one direction
    - seed:      If given, a random seed is passed to the random number
                 generator, otherwise NumPy finds its own seed somewhere.
    - ksphere:   If true, a sphere in k-space is defined with radius k_nyquist.
                 Outside of this sphere, all grid values are zero. If false,
                 these are also just random values, subject to symmetries.
    
    Computed attributes:
    - t:        The true density field.
    - f:        The density field in Fourier space.
    """
    def __init__(self, power, boxlen, gridsize, seed = None, ksphere = True):
        self.power = power
        self.boxlen = boxlen
        self.gridsize = int(gridsize)
        self.seed = seed
        if not self.seed:
            self.seed = np.random.randint(0x100000000)
        self.ksphere = ksphere
        self.build_fourier()
    
    def build_fourier(self):
        # Initialize some used constants
        halfgrid = self.gridsize/2
        dk = 2*np.pi / self.boxlen
        kmax = self.gridsize*dk
        
        #print "Begin building rho array in Fourier space..."
        
        # Initialize fourier-space k-values grid
        k = toolbox.k_abs_grid(self.gridsize, self.boxlen)
        #~ k1, k2, k3 = dk*np.mgrid[0:self.gridsize, 0:self.gridsize, 0:halfgrid+1]
        #~ k1 -= kmax*(k1 > dk*(halfgrid - 1)) # shift the second half to negative k values
        #~ k2 -= kmax*(k2 > dk*(halfgrid - 1))
        #~ k = np.sqrt(k1**2 + k2**2 + k3**2)
        #~ del(k1, k2, k3) # keep memory clean
        
        # Fill in the grid with random numbers
        # ... for the regular grid components:
        arg = resolution_independent_random_grid(self.gridsize, self.seed)
        mod = 1 - resolution_independent_random_grid(self.gridsize, self.seed+1) # "1 -" so there's no zero
        z = np.sqrt(-np.log(mod)) * np.exp(1j*2*np.pi*arg)
        # Reduce z by sqrt(2) for real f since need sum of |z|^2 over all
        # wavenumbers (positive and negative) to be Chi-square for N degrees
        # of freedom, not 2N as in the complex case.  Equivalently, the power
        # is shared with the conjugate harmonics with k1 > halfgrid (k1 < 0).
        #
        # EGP: note that this is simply the Box-Muller method for obtaining two
        # *independent* Gaussian random variables (see sect. 7.3.4 of Press+07).
        # A faster method could/should be used here (sect. 7.3.9, Press+07)!
        
        # ... for the 7 real, independent, non-zero grid values:
        real7x,real7y,real7z = np.mgrid[0:2,0:2,0:2]*halfgrid
        real7x,real7y,real7z = real7x.ravel()[1:],real7y.ravel()[1:],real7z.ravel()[1:]
        z[real7x,real7y,real7z] = np.sqrt(2) * np.sqrt(-np.log(mod[real7x,\
                                  real7y,real7z])) * np.cos(2*np.pi*arg[real7x,\
                                  real7y,real7z]) + 0j
        # Here you do still have the sqrt(2)...
        del(arg, mod) # keep memory clean
        
        # ... the 8th real, independent grid value [0,0,0] is zero:
        z[0,0,0] = 0
        
        # Then, the actual density field (in Fourier space):
        self.power.normalize(self.boxlen**3)
        ps_cache_key = "grid_%s_box_%s" % (self.gridsize, self.boxlen)
        self.f = np.sqrt(self.power(k, cache_key=ps_cache_key))*z
        
        # Finally add symmetry to the nyquist planes (so the ifft is not imaginary):
        symmetrizeMatrix(self.f)
        self.f[0,0,0] = 0.0
        # The last step is necessary because there's a 1/k**2 in z and k[0] == 0
        # It could be something different, but it's the integral over your field
        # and so then you wouldn't have a field with average value of zero
        # anymore, which is what we do want it to have.
        
        # Setting everything outside of the k-sphere with radius k_nyquist to zero.
        if self.ksphere:
            ksphere = k <= kmax/2
            self.f *= ksphere



# internal functions & classes
def symmetrizeMatrix(m):
    # Only symmetrizes the matrix in k3==(0||nyquist), because we only need to
    # fill the k3<=nyquist part for an inverse fft to a real field; so not in
    # the k1 and k2 nyquist planes and not outside the nyquist planes either.
    gridsize = m.shape[0]
    halfgrid = gridsize/2
    xi, yi, zi = np.ogrid[0:halfgrid+1,0:gridsize,0:halfgrid+1:halfgrid]
    xj, yj, zj = np.ogrid[gridsize:halfgrid-1:-1,gridsize:0:-1,0:halfgrid+1:halfgrid]
    xj[0], yj[:,0] = 0,0
    # The following must be in 2 halfs, or you won't get a symmetric matrix,
    # but just the full matrix mirrored and conjugated.
    m[xi,yi[:,:halfgrid],zi] = m[xj,yj[:,:halfgrid],zj].conj()
    m[xi,yi[:,halfgrid:],zi] = m[xj,yj[:,halfgrid:],zj].conj()

def resolution_dependent_random_grid(gridsize, seed):
    np.random.seed(seed)
    return np.random.random((gridsize,gridsize,gridsize/2+1))


# --- CONSTRAINED FIELD STUFF --- #

# In deze class gaan we er van uit dat een density een power attribute heeft.
# Dit kunnen we in het algemene DensityField (of zelfs in het Field?) als
# property implementeren die standaard een EstimatedPowerSpectrum bepaalt als
# dat nodig is en die in het GRF geval natuurlijk gewoon gegeven is.
class ConstrainedField(DensityField):
    def __init__(self, unconstrained_density, constraints, correlations = None, calculate = True):
        self.constraints = constraints
        constraint_values = []
        for constraint in self.constraints:
            constraint_values.append(constraint.value)
        self.constraint_values = np.array(constraint_values)
        self.rhoU = unconstrained_density
        self.gridsize = self.rhoU.gridsize
        self.boxlen = self.rhoU.boxlen
        self.power = self.rhoU.power
        if not np.any(correlations):
            self.correlations = ConstraintCorrelations(constraints, self.power)
        else:
            self.correlations = correlations
        if calculate:
            self.calculate_CF()
    
    def calculate_CF(self):
        self.f = self.rhoU.f + self.rhoR.f
    
    rhoR = property()
    @rhoR.getter
    def rhoR(self):
        try:
            return self._residual_field
        except AttributeError:
            k = toolbox.k_abs_grid(self.gridsize, self.boxlen)
            ki = toolbox.k_i_grid(self.gridsize, self.boxlen)
            ps_cache_key = "grid_%s_box_%s" % (self.gridsize, self.boxlen)
            self.correlations.calculate(k, ki, ps_cache_key=ps_cache_key) # P(k) H_i(k) and \xi_ij^-1
            self.calculate_unconstrained_constraints(ki) # c_j^~
            self.calculate_residual_field() # P(k) H_i(k) \xi_ij^-1 (c_j-c_j^~)
            return self._residual_field
    @rhoR.setter
    def rhoR(self, density_field):
        self._residual_field = density_field
    
    def calculate_residual_field(self):
        """Calculate the residual field."""
        delta_c_j = np.matrix(self.constraint_values - self.c_unconstrained).T
        xi_times_c = np.matrix(self.correlations.xi_ij_inverse) * delta_c_j
        PH = self.correlations.PH
        rhoR_f = np.sum(PH * np.array(xi_times_c)[:, np.newaxis, np.newaxis], axis=0)
        self.rhoR = DensityField(fourier = rhoR_f)
    
    def calculate_unconstrained_constraints(self, ki):
        """Calculate the values of the constraints in the unconstrained
        field."""
        self.c_unconstrained = np.empty((len(self.constraints),))
        for i, coni in enumerate(self.constraints):
            Hi_conj = coni.H(ki).conj()
            cU = np.sum( 2*Hi_conj[...,1:-1]*self.rhoU.f[...,1:-1] ).real
            cU += np.sum( Hi_conj[...,(0,-1)]*self.rhoU.f[...,(0,-1)] ).real # k_z = 0, nyquist
            self.c_unconstrained[i] = cU


class ConstraintCorrelations(object):
    """Contains the xi_ij constraint-correlation matrix of the given Constraint
    and PowerSpectrum objects and the correlations of the Constraints with the
    PowerSpectrum themselves."""
    def __init__(self, constraints, power):
        self.constraints = constraints
        self.power = power
    
    def calculate(self, k, ki, ps_cache_key=None):
        self.calculate_constraint_correlations(k, ki, ps_cache_key)
        self.calculate_field_constraint_correlations(k, ki, ps_cache_key)
    
    def calculate_constraint_correlations(self, k, ki, ps_cache_key=None, fast=False):
        """Calculate the matrix xi_ij containing the correlation values between
        the different constraints."""
        power = self.power(k, cache_key=ps_cache_key)
        if fast:
            # einsum has a small precision error; zie einsumPrecisie.py
            H = np.empty((len(self.constraints),) + k.shape, dtype=np.complex)
            for i, coni in enumerate(self.constraints):
                H[i] = coni.H(ki)
            xi_ij = 2*np.einsum('iklm,jklm->ij', H[...,1:-1].conj(), H[...,1:-1]*power[...,1:-1]).real
            # ... where the k,l,m indices sum over the kx,ky,kz axes.
            #     and the factor 2 is to account for the symmetry in kz.
            # Correct for k_z = 0 and nyquist planes (they count only once):
            xi_ij += np.einsum('iklm,jklm->ij', H[...,(0,-1)].conj(), H[...,(0,-1)]*power[...,(0,-1)]).real
            self.xi_ij_inverse = np.matrix(xi_ij).I
        else:
            self.xi_ij_inverse = np.empty((len(self.constraints), len(self.constraints)))
            for i, coni in enumerate(self.constraints):
                Hi = coni.H(ki)
                for j, conj in enumerate(self.constraints):
                    Hj = conj.H(ki)
                    xi_ij = np.sum( 2*Hi[...,1:-1].conj()*Hj[...,1:-1]*power[...,1:-1] ).real
                    # plus k_z = 0 and nyquist planes:
                    xi_ij += np.sum( Hi[...,(0,-1)].conj()*Hj[...,(0,-1)]*power[...,(0,-1)] ).real
                    self.xi_ij_inverse[i,j] = xi_ij # first store xi_ij
            self.xi_ij_inverse = np.matrix(self.xi_ij_inverse).I # then invert
    
    def calculate_field_constraint_correlations(self, k, ki, ps_cache_key=None):
        """Calculate the correlations between the field values and the
        constraint H matrices, i.e. P(k) H_i(k)."""
        H = np.empty((len(self.constraints),) + k.shape, dtype=np.complex)
        for i, coni in enumerate(self.constraints):
            H[i] = coni.H(ki)
        PH = self.power(k, cache_key=ps_cache_key)*H
        self.PH = PH


class ConstraintLocation(object):
    def __init__(self, location):
        self.location = location
        #self.constraints = []
        # N.B.: the constraints list will cause a reference cycle, which Python
        # is not able to handle properly! Will cause memory leak!
        # Can be manually fixed using the gc module:
        #   import gc
        #   gc.collect()
        # To see what is actually causing the memory leak:
        #   gc.set_debug(gc.DEBUG_LEAK)
        #   gc.collect()
        #   print gc.garbage
        # This way the garbage is kept in the garbage list instead of instantly
        # being deleted (as with collect() without the set_debug stuff).
        # See also Constraint.__init__()
        #
        # A better fix would be to use a "weak reference" from the weakref
        # module. See e.g. http://eli.thegreenplace.net/2009/06/12/safely-using-destructors-in-python/
        # An even better fix would be to just check for human errors in the
        # ConstrainedField upon building the field instead of doing it in the
        # ConstraintLocation object already as was the case when using this
        # constraints list.
    
    def __call__(self, k=None):
        """Return common part of the constraint kernels at this location. Once
        called with a certain k, the factor (array) is stored for that k and it
        needn't be passed as an argument again. Even if it is passed, the old
        value will still immediately be returned. The only way to reset it is to
        manually call the self.set_common_kernel_factor(k) method.
        N.B.: k must be a three component vector. The first dimension must be 3,
        i.e. k.shape[0] == 3."""
        try:
            return self._common_kernel_factor
        except AttributeError:
            self.set_common_kernel_factor(k)
            return self._common_kernel_factor
    
    def set_common_kernel_factor(self, k):
        if k is None:
            raise ValueError("No argument given; a(n initial) k must be given!")
        kr = self.location[0]*k[0] + self.location[1]*k[1] + self.location[2]*k[2]
        self._common_kernel_factor =  np.exp(1j*kr)
    
    #~ def add_constraint(self, constraint):
        #~ constraint_types = [type(con) for con in self.constraints]
        #~ if type(constraint) in constraint_types:
            #~ if type(constraint) == HeightConstraint:
                #~ raise ConstraintLocationError("There can only be one HeightConstraint at a ConstraintLocation!")
            #~ if type(constraint) == ExtremumConstraint:
                #~ extrema_directions = [con.direction for con in self.constraints if type(con) == ExtremumConstraint]
                #~ if constraint.direction in extrema_directions:
                    #~ raise ConstraintLocationError("There can only be one ExtremumConstraint in the %i direction at a ConstraintLocation!" % constraint.direction)
        #~ self.constraints.append(constraint)


class ConstraintLocationError(Exception):
    pass


class ConstraintScale(object):
    def __init__(self, scale):
        self.scale = scale
    
    def __call__(self, k=None):
        """Return common part of the constraint kernels at this scale. Once
        called with a certain k, the factor (array) is stored for that k and it
        needn't be passed as an argument again. Even if it is passed, the old
        value will still immediately be returned. The only way to reset it is to
        manually call the self.set_common_kernel_factor(k) method.
        N.B.: k must be a three component vector. The first dimension must be 3,
        i.e. k.shape[0] == 3."""
        try:
            return self._common_kernel_factor
        except AttributeError:
            self.set_common_kernel_factor(k)
            return self._common_kernel_factor
    
    def set_common_kernel_factor(self, k):
        if k is None:
            raise ValueError("No argument given; a(n initial) k must be given!")
        ksq = k[0]**2 + k[1]**2 + k[2]**2
        self._common_kernel_factor =  np.exp(-ksq * self.scale**2/2)


class Constraint(object):
    def __init__(self, location, scale):
        self.location = location
        self.scale = scale
        #~ location.add_constraint(self)
        # N.B.: this constraints list will cause a reference cycle, which Python
        # is not able to handle properly! Will cause memory leak!
        # Can be manually fixed using the gc module:
        #   import gc
        #   gc.collect()
        # To see what is actually causing the memory leak:
        #   gc.set_debug(gc.DEBUG_LEAK)
        #   gc.collect()
        #   print gc.garbage
        # This way the garbage is kept in the garbage list instead of instantly
        # being deleted (as with collect() without the set_debug stuff).
        # See also ConstraintLocation.__init__()
    
    def W_factor(self, k):
        try:
            return self._W_factor
        except AttributeError:
            self.set_W_factor(k)
            return self._W_factor
    
    def set_W_factor(self, k):
        self._W_factor = self.location(k) * self.scale(k)
    
    def H(self, k):
        try:
            return self._H
        except AttributeError:
            self.set_H(k)
            return self._H
    
    def set_H(self, k):
        """This function must be overloaded in subclasses."""
        raise AttributeError("The Constraint class is a meta-class and not meant to actually be used! H(k) is not defined in this object. Use a subclass like HeightConstraint or ExtremumConstraint.")


class HeightConstraint(Constraint):
    def __init__(self, location, scale, height):
        self.value = height
        Constraint.__init__(self, location, scale)
    
    def set_H(self, k):
        self._H = self.W_factor(k)


class ExtremumConstraint(Constraint):
    """Make the location a maximum, minimum or saddlepoint in one direction.
    Specify the direction with the direction keyword; 0,1,2 = x,y,z."""
    def __init__(self, location, scale, direction):
        self.direction = direction
        self.value = 0
        Constraint.__init__(self, location, scale)
    
    def set_H(self, k):
        self._H = 1j * k[self.direction] * self.W_factor(k)


class ShapeConstraint(Constraint):
    """Constrain the slope, shape and/or orientation of the peak by setting the
    value of the (second) derivative of the field to k[k_index1] and
    k[k_index2]. Note: only six possible combinations of k_index{1,2} are valid:
    0,0; 1,1; 2,2; 0,1; 0,2 and 1,2."""
    def __init__(self, location, scale, second_derivative, k_index1, k_index2):
        self.k_index1 = k_index1
        self.k_index2 = k_index2
        self.value = second_derivative
        Constraint.__init__(self, location, scale)
    
    def set_H(self, k):
        self._H = - k[self.k_index1] * k[self.k_index2] * self.W_factor(k)


class GravityConstraint(Constraint):
    """Constrain the gravitational acceleration on the peak in one direction.
    The cosmology needs to be given as a Cosmology object as well. N.B.: in the
    linear regime, gravity is linearly proportional to velocity."""
    def __init__(self, location, scale, gravity, direction, cosmology):
        self.direction = direction
        self.value = gravity
        self.cosmology = cosmology
        Constraint.__init__(self, location, scale)
    
    def set_H(self, k):
        C = self.cosmology
        k_squared = k[0]**2 + k[1]**2 + k[2]**2
        self._H = 3./2 * C.omegaM * 100**2 * 1j * k[self.direction] / \
                  k_squared * self.W_factor(k)
                                  # 100 = Hubble in units of h
        self._H[0,0,0] = 0 # complex division of zero by zero gives NaN


class TidalConstraint(Constraint):
    """Constrains the tidal forces on the peak. There are five independent
    eigenvalues in the tidal force tensor, leading to five valid combinations
    of k_index{1,2}: 0,0; 1,1; 0,1; 0,2 and 1,2. The cosmology needs to be given
    as a Cosmology object as well."""
    def __init__(self, location, scale, eigenvalue, k_index1, k_index2, cosmology):
        self.k_index1 = k_index1
        self.k_index2 = k_index2
        self.value = eigenvalue
        self.cosmology = cosmology
        Constraint.__init__(self, location, scale)
    
    def set_H(self, k):
        C = self.cosmology
        k_squared = k[0]**2 + k[1]**2 + k[2]**2
        if self.k_index1 == self.k_index2:
            delta = 1
        else:
            delta = 0
        self._H = 3./2 * C.omegaM * 100**2 * \
                  (k[self.k_index1] * k[self.k_index2] / k_squared - delta/3.) * \
                  self.W_factor(k)
                                  # 100 = Hubble in units of h
        self._H[0,0,0] = 0 # complex division of zero by zero gives NaN


#   CONVENIENCE FUNCTIONS

def euler_matrix(alpha, beta, psi):
    cos, sin = np.cos, np.sin
    M11 = cos(alpha)*cos(psi) - cos(beta)*sin(alpha)*sin(psi)
    M12 = sin(alpha)*cos(psi) + cos(beta)*cos(alpha)*sin(psi)
    M13 = sin(beta)*sin(psi)
    M21 = -cos(alpha)*sin(psi) - cos(beta)*sin(alpha)*cos(psi)
    M22 = -sin(alpha)*sin(psi) + cos(beta)*cos(alpha)*cos(psi)
    M23 = -sin(beta)*cos(psi)
    M31 = sin(beta)*sin(alpha)
    M32 = -sin(beta)*cos(alpha)
    M33 = cos(beta)
    return np.matrix([[M11,M12,M13],[M21,M22,M23],[M31,M32,M33]])


#   FACTORIES

def constraints_from_csv(filename, power_spectrum, boxlen):
    """Loads the table needed for constraints_from_table from a csv-file."""
    f = open(filename)
    table = csvreader(f)
    # Skip header:
    table.next()
    return constraints_from_table(table, power_spectrum, boxlen)

def constraints_from_table(table, power_spectrum, boxlen):
    """Load a table containing constraints and convert it into a list of
    Constraint objects. Also needs the power spectrum of the CRF that you're 
    going to build, because it needs to determine the spectral parameters.
    Finally, we need the boxsize (in Mpc h^-1) of the random field that we're
    going to build, also needed for the spectral parameters.
    
    The structure of each table row needs to be:
    
    1.  x-location (Mpc h^-1)
    2.  y-location (Mpc h^-1)
    3.  z-location (Mpc h^-1)
    4.  scale (Mpc h^-1)
  A 5.  peak height (density contrast units (-1-\inf))
  B 6.  x-extremum (True or False)
  B 7.  y-extremum (True or False)
  B 8.  z-extremum (True or False)
  C 9.  peak curvature (fraction of mean curvature)
  C 10. shape a_2/a_1 (0-1)
  C 11. shape a_3/a_1 (0-a_2/a_1)
  C 12. orientation (Euler) angle phi (degrees)
  C 13. orientation (Euler) angle theta (degrees)
  C 14. orientation (Euler) angle psi (degrees)
  D 15. x-velocity (km/s)
  D 16. y-velocity (km/s)
  D 17. z-velocity (km/s)
  E 18. shear eigenvalue 1* (km/s/Mpc)
  E 19. shear eigenvalue 2* (km/s/Mpc)
    * The shear eigenvalues sum to 0 (the tidal tensor is traceless) so you need
      only specify two eigenvalues. The eigenvalues determine whether you have a
      sheet (two negative eigenvalues), a filament (one negative eigenvalue) or
      a random situation (all eigenvalues zero). Could also be different, not
      sure, just guessing here, really.
  E 20. shear orientation (Euler) angle phi (degrees)
  E 21. shear orientation (Euler) angle theta (degrees)
  E 22. shear orientation (Euler) angle psi (degrees)
    
    The first four are mandatory, the others are not. If a group (A-E) is left
    empty, no constraint will be made on that group. If groups C and E are left
    partially empty, the other values will be randomly selected and constraints
    will be put on them as well (this is necessary, because they are correlated
    in some complicated way that is hard to disentangle).
    """
    cosmo = power_spectrum.cosmology
    
    locations = {}
    scales = {}
    constraints = []
    
    z = 0
    velocity_to_gravity = 2*fpeebl(z, cosmo.omegaR, cosmo.omegaM, \
                      cosmo.omegaL) / 3 / 100 / cosmo.omegaM
                                        # 100 = Hubble in units of h
    
    
    for row in table:
        # ----- Location
        pos = (float(row[0]), float(row[1]), float(row[2])) # tuple needed; lists cannot be dictionary keys
        if pos in locations:
            location = locations[pos]
        else:
            location = ConstraintLocation(pos)
            locations[pos] = location
        
        # ----- Scale
        scalestr = row[3]
        if scalestr in scales:
            scale = scales[scalestr]
        else:
            scale = ConstraintScale(float(scalestr))
            scales[scalestr] = scale
        
        # ----- Peak height
        height = row[4]
        if height:
            constraints.append(HeightConstraint(location, scale, float(height)))
        
        # ----- Extrema
        x_extremum = row[5]
        if x_extremum:
            constraints.append(ExtremumConstraint(location, scale, 0))
        y_extremum = row[6]
        if y_extremum:
            constraints.append(ExtremumConstraint(location, scale, 1))
        z_extremum = row[7]
        if z_extremum:
            constraints.append(ExtremumConstraint(location, scale, 2))
        
        # ----- Shape & orientation
        curvature = row[8]
        a21 = row[9]
        a31 = row[10]
        density_phi = row[11]
        density_theta = row[12]
        density_psi = row[13]
        
        # if one is defined, randomly define all others:
        if curvature or a21 or a31 or density_phi or density_theta or density_psi:
            shape_constraints = generate_shape_constraints(location, scale, power_spectrum, boxlen, curvature, a21, a31, density_phi, density_theta, density_psi)
            constraints.extend(shape_constraints)
        
        # ----- Velocity
        vx = row[14]
        vy = row[15]
        vz = row[16]
        
        # Convert velocity to gravity:
        if vx or vy or vz:
            # ----- Spectral parameters (for easier constraint value definition, at
            # ----- least in case of gaussian random field):
            sigma0 = power_spectrum.moment(0, scale.scale, boxlen**3)
            sigma1 = power_spectrum.moment(1, scale.scale, boxlen**3)
            #~ sigma2 = power_spectrum.moment(2, scale.scale, boxlen**3)
            sigma_min1 = power_spectrum.moment(-1, scale.scale, boxlen**3)
            gamma_nu = sigma0**2 / sigma_min1 / sigma1
            sigma_g = 3./2 * cosmo.omegaM * 100**2 * sigma_min1 # 100 = Hubble in units of h
            sigma_g_peak = sigma_g * np.sqrt(1 - gamma_nu**2)
            #~ gamma = sigma1**2/sigma0/sigma2
            #~ sigma_E = 3./2*cosmo.omegaM*100**2 * sigma0 * np.sqrt((1-gamma**2)/15)
        if vx:
            gx = velocity_to_gravity*float(vx)
            constraints.append(GravityConstraint(location, scale, gx*sigma_g_peak, 0, cosmo))
        if vy:
            gy = velocity_to_gravity*float(vy)
            constraints.append(GravityConstraint(location, scale, gy*sigma_g_peak, 1, cosmo))
        if vz:
            gz = velocity_to_gravity*float(vz)
            constraints.append(GravityConstraint(location, scale, gz*sigma_g_peak, 2, cosmo))
        
        # ----- Shear/tidal field
        ev1 = row[17]
        ev2 = row[18]
        shear_phi = row[19]
        shear_theta = row[20]
        shear_psi = row[21]
        
        if ev1 or ev2 or shear_phi or shear_theta or shear_psi:
            if not ev1 or not ev2: # No random distribution known for this
                raise Exception("Must define both shear eigenvalues or none at all!")
            ev1 = float(ev1)
            ev2 = float(ev2)
            ev3 = -(ev1 + ev2)
            
            if not shear_phi:
                shear_phi = 180*np.random.random()
            else:
                shear_phi = float(shear_phi)
            if not shear_theta:
                shear_theta = 180/np.pi*np.arccos(1-np.random.random())
            else:
                shear_theta = float(shear_theta)
            if not shear_psi:
                shear_psi = 180*np.random.random()
            else:
                shear_psi = float(shear_psi)
            
            T = euler_matrix(shear_phi, shear_theta, shear_psi)
            
            E11 = - ev1*T[0,0]*T[0,0] - ev2*T[1,0]*T[1,0] - ev3*T[2,0]*T[2,0]
            E22 = - ev1*T[0,1]*T[0,1] - ev2*T[1,1]*T[1,1] - ev3*T[2,1]*T[2,1]
            E12 = - ev1*T[0,0]*T[0,1] - ev2*T[1,0]*T[1,1] - ev3*T[2,0]*T[2,1]
            E13 = - ev1*T[0,0]*T[0,2] - ev2*T[1,0]*T[1,2] - ev3*T[2,0]*T[2,2]
            E23 = - ev1*T[0,1]*T[0,2] - ev2*T[1,1]*T[1,2] - ev3*T[2,1]*T[2,2]
            
            constraints.append(TidalConstraint(location, scale, E11, 0, 0, cosmo))
            constraints.append(TidalConstraint(location, scale, E22, 1, 1, cosmo))
            constraints.append(TidalConstraint(location, scale, E12, 0, 1, cosmo))
            constraints.append(TidalConstraint(location, scale, E13, 0, 2, cosmo))
            constraints.append(TidalConstraint(location, scale, E23, 1, 2, cosmo))
    
    return constraints

def generate_shape_constraints(location, scale, power_spectrum, boxlen, curvature="", a21="", a31="", density_phi="", density_theta="", density_psi=""):
    """The actual constraint-values should be strings and they can be empty
    strings if you don't want to constrain that particular value (it will then
    be randomly assigned according to random draws from the proper
    distributions)."""
    constraints = []
    sigma0 = power_spectrum.moment(0, scale.scale, boxlen**3)
    sigma1 = power_spectrum.moment(1, scale.scale, boxlen**3)
    sigma2 = power_spectrum.moment(2, scale.scale, boxlen**3)
    gamma = sigma1**2/sigma0/sigma2
    if not curvature:
        curvature = random_curvature(height, gamma)
    else:
        curvature = float(curvature)
    if not a21 or not a31:
		a21, a31 = random_shape(curvature)
    else:
        a21 = float(a21)
        a31 = float(a31)
    if not density_phi:
        density_phi = 180*np.random.random()
    else:
        density_phi = float(density_phi)
    if not density_theta:
        density_theta = 180/np.pi*np.arccos(1-np.random.random())
    else:
        density_theta = float(density_theta)
    if not density_psi:
        density_psi = 180*np.random.random()
    else:
        density_psi = float(density_psi)
    
    # Convert stuff and calculate matrix coefficients (= 2nd derivs)...
    a12 = 1/a21
    a13 = 1/a31
    A = euler_matrix(density_phi, density_theta, density_psi)
    lambda1 = curvature * sigma2 / (1 + a12**2 + a13**2)
    lambda2 = lambda1 * a12**2
    lambda3 = lambda1 * a13**2
    d11 = - lambda1*A[0,0]*A[0,0] - lambda2*A[1,0]*A[1,0] - lambda3*A[2,0]*A[2,0]
    d22 = - lambda1*A[0,1]*A[0,1] - lambda2*A[1,1]*A[1,1] - lambda3*A[2,1]*A[2,1]
    d33 = - lambda1*A[0,2]*A[0,2] - lambda2*A[1,2]*A[1,2] - lambda3*A[2,2]*A[2,2]
    d12 = - lambda1*A[0,0]*A[0,1] - lambda2*A[1,0]*A[1,1] - lambda3*A[2,0]*A[2,1]
    d13 = - lambda1*A[0,0]*A[0,2] - lambda2*A[1,0]*A[1,2] - lambda3*A[2,0]*A[2,2]
    d23 = - lambda1*A[0,1]*A[0,2] - lambda2*A[1,1]*A[1,2] - lambda3*A[2,1]*A[2,2]
    # ... and define Constraints:
    constraints.append(ShapeConstraint(location, scale, d11, 0, 0))
    constraints.append(ShapeConstraint(location, scale, d22, 1, 1))
    constraints.append(ShapeConstraint(location, scale, d33, 2, 2))
    constraints.append(ShapeConstraint(location, scale, d12, 0, 1))
    constraints.append(ShapeConstraint(location, scale, d13, 0, 2))
    constraints.append(ShapeConstraint(location, scale, d23, 1, 2))
    return constraints


#   FACTORY HELPER FUNCTIONS

def random_curvature(height, gamma):
	# determine the cumulative cpdf of curvature x given peak height:
	x = np.linspace(0,10,5001) # ongeveer zelfde als in Riens code
	dx = x[1]-x[0]
	cumPDF = cumulative_cpdf_curvature(x,height,gamma)
	percentile = 1 - np.random.random() # don't want zero
	ix = cumPDF.searchsorted(percentile)
	curvature = dx*((ix-1) + (percentile-cumPDF[ix-1])/(cumPDF[ix]-cumPDF[ix-1]))
	# This is basically just the ``transformation method'' of
	# drawing random variables (sect. 7.3.2 of Press+07).
	# Note: at the end of the x-range the cumulative PDF is flat,
	# within numerical precision, so no interpolation is possible. 
	# This means that the curvature will never be higher than the
	# corresponding value at which the PDF flattens.
	return curvature

def random_shape(curvature):
	# For the shape distribution we apply the rejection method (see
	# e.g. sect. 7.3.6 of Press+07) because we cannot easily
	# determine the 
	# N.B.: if only one is given, both will be randomly calculated!
	p,e = np.mgrid[-0.25:0.25:0.005, 0:0.5:0.005]
	PDF = cpdf_shape(e, p, curvature)
	# The top of the uniform distribution beneath which we draw:
	comparison_function_max = 1.01*PDF.max()
	# Drawing in 3D space beneath the comparison function until we
	# no longer reject (ugly Pythonic do-while loop equivalent):
	draw = lambda: (0.5*np.random.random(), \
					0.75*np.random.random() - 0.25, \
					comparison_function_max*(1-np.random.random()))
					# gives, respectively: e, p, something not zero
	e,p,prob = draw()
	while cpdf_shape(e,p,curvature) < prob: # rejection criterion
		e,p,prob = draw()
	# Convert to a21 and a31:
	a21 = np.sqrt((1-2*p)/(1+p-3*e))
	a31 = np.sqrt((1+p+3*e)/(1+p-3*e))
	return a21, a31

def cumulative_cpdf_curvature(x, height, gamma):
    """Determine the cumulative conditional probability distribution function of
    the curvature of peaks with height /height/ over curvature range /x/. See
    also the docstring of function cpdf_curvature()."""
    xdel = (x[-1]-x[0])/(len(x)-1)
    return np.cumsum(cpdf_curvature(x,height,gamma))*xdel

def cpdf_curvature(x, height, gamma):
    """The conditional probability distribution function of the curvature of
    peaks with height /height/ over curvature range /x/. This
    is given in Bardeen, Bond, Kaiser & Szalay (1986) by equation (7.5),
    together with (A15), (A19) and (4.4)/(4.5). /gamma/ is a fraction of
    spectral momenta."""
    w = gamma*height # xstar in Rien's code and in BBKS
    # BBKS eqn. (4.5):
    A = 2.5/(9 - 5*gamma**2)
    B = 432./np.sqrt(10*np.pi)/(9 - 5*gamma**2)**(5./2)
    C1 = 1.84 + 1.13*(1-gamma**2)**5.72
    C2 = 8.91 + 1.27*np.exp(6.51*gamma**2)
    C3 = 2.58*np.exp(1.05*gamma**2)
    # BBKS eqn. (4.4) (N.B.: this is an approximation of the real G(gamma, nu)):
    #G = ( w**3 - 3*gamma**2*w + (B*w**2 + C1)*np.exp(-A*w**2) ) \
    #    / ( 1 + C2*np.exp(-C3*w) )
    # This is a bad approximation though, it differs from the real normalizing
    # factor by up to 1.5%. Below we just normalize by the sum of all elements
    # of g.
    # BBKS eqn. (A15):
    f = f_BBKS(x)
    # BBKS eqn. (A19):
    g = f * np.exp( -(x-w)**2/2/(1-gamma**2) ) / np.sqrt( 2*np.pi*(1-gamma**2) )
    # Normalization:
    
    dx = np.r_[x[1]-x[0], x[1:]-x[:-1]] # approximate for complicated x ranges!
    G = (g*dx).sum()
    # BBKS eqn. (7.5):
    P_x_given_nu = g/G
    return P_x_given_nu

def cpdf_shape(e, p, x):
    """The conditional probability distribution function of the shape parameters
    e and p (ellipticity and prolateness) given the curvature x, as defined by
    Bardeen, Bond, Kaiser & Szalay (1986) in equation (7.6), together with
    (A15), (C4) and (C3)."""
    # BBKS eqn. (C3):
    chi = ((0 <= e)&(e <= 0.25) & (-e <= p)&(p <= e)) | ((0.25 <= e)&(e <= 0.5) & (e*3 - 1 <= p)&(p <= e))
    # Equivalent to:
    #~ if 0 <= e <= 0.25 and -e <= p <= e: chi = 1.
    #~ elif 0.25 <= e <= 0.5 and e*3 - 1 <= p <= e: chi = 1.
    #~ else: chi = 0.
    # BBKS eqn. (C4):
    W = e*(e**2 - p**2) * (1 - 2*p) * ( (1+p)**2 - 9*e**2 ) * chi
    # BBKS eqn. (A15):
    f = f_BBKS(x)
    # BBKS eqn. (7.6):
    P_e_p_given_x = 9*5**(5./2)/np.sqrt(2*np.pi) * x**8 / f * W * \
                    np.exp( -5./2*x**2 * (3*e**2 + p**2) )
    return P_e_p_given_x

def f_BBKS(x):
    """Equation (A15) from BBKS."""
    f = (x**3 - 3*x) * ( erf(np.sqrt(5./2)*x) + erf(np.sqrt(5./2)*x/2) ) / 2 \
        + np.sqrt(2./5/np.pi) * ( (31.*x**2/4 + 8./5)*np.exp(-5.*x**2/8) \
        + (x**2/2 - 8./5)*np.exp(-5*x**2/2) )
    return f


# --- END CONSTRAINED FIELD STUFF --- #



# Zel'dovich approximation    
def zeldovich(redshift, psi, cosmo, print_info=False):
    """
    Use the Zel'dovich approximation to calculate positions and velocities at
    certain /redshift/, based on the DisplacementField /psi/ of e.g. a Gaussian
    random density field and Cosmology /cosmo/.
    
    Outputs a tuple of a position and velocity vector array; positions are in
    units of h^{-1} Mpc (or in fact the same units as /psi.boxlen/) and
    velocities in km/s.
    """
    psi1 = psi.x.t
    psi2 = psi.y.t
    psi3 = psi.z.t
    
    omegaM = cosmo.omegaM
    omegaL = cosmo.omegaL
    omegaR = cosmo.omegaR
    boxlen = psi.boxlen
    
    gridsize = len(psi1)
    if print_info: print "Boxlen:    ",boxlen
    dx = boxlen/gridsize
    if print_info: print "dx:        ",dx
    f = fpeebl(redshift, omegaR, omegaM, omegaL)
    if print_info: print "fpeebl:    ",f
    D = grow(redshift, omegaR, omegaM, omegaL)
    D0 = grow(0, omegaR, omegaM, omegaL) # used for normalization of D to t = 0
    if print_info: print "D+(z):     ",D
    if print_info: print "D+(0):     ",D0
    if print_info: print "D(z)/D(0): ",D/D0
    H = hubble(redshift, omegaR, omegaM, omegaL)
    if print_info: print "H(z):      ",H
    
    xfact = D/D0
    vfact = D/D0*H*f/(1+redshift)
    
    v = vfact * np.array([psi1,psi2,psi3]) # vx,vy,vz
    # lagrangian coordinates, in the center of the gridcells:
    q = np.mgrid[dx/2:boxlen+dx/2:dx,dx/2:boxlen+dx/2:dx,dx/2:boxlen+dx/2:dx] # IN EEN CACHEABLE FUNCTIE ZETTEN
    X = (q + xfact*(v/vfact))%boxlen
    #~ # Mirror coordinates, because somehow it doesn't match the coordinates put
    #~ # into the constrained field.
    #~ X = boxlen - X # x,y,z
    #~ v = -v
    # FIXED MIRRORING: using different FFT convention now (toolbox.rfftn etc).
    
    return X,v


def zeldovich_displacement(redshift, psi, cosmo, print_info=False):
    """
    Use the Zel'dovich approximation to calculate the physical
    displacement at certain /redshift/ in the position coordinates,
    based on the DisplacementField /psi/ of e.g. a Gaussian
    random density field and Cosmology /cosmo/.
    
    Returned displacements are in units of h^{-1} Mpc (or in fact the
    same units as /psi.boxlen/).
    """
    psi1 = psi.x.t
    psi2 = psi.y.t
    psi3 = psi.z.t
    
    omegaM = cosmo.omegaM
    omegaL = cosmo.omegaL
    omegaR = cosmo.omegaR
    D = grow(redshift, omegaR, omegaM, omegaL)
    D0 = grow(0, omegaR, omegaM, omegaL) # used for normalization of D to t = 0
    
    xfact = D/D0
    
    disp_x = xfact*np.array([psi1,psi2,psi3])
    
    return disp_x


def zeldovich_step(redshift_start, redshift_end, psi, pos, cosmo):
    """
    Use the Zel'dovich approximation to calculate positions and velocities at
    certain /redshift_end/, based on the DisplacementField /psi/ and starting
    positions /pos/ at redshift /redshift_start/ and Cosmology /cosmo/. /pos/
    should have shape (3,gridsize,gridsize,gridsize).
    
    Outputs a tuple of a position and velocity vector array; positions are in
    units of h^{-1} Mpc (or in fact the same units as /psi.boxlen/) and
    velocities in km/s.
    """
    omegaM = cosmo.omegaM
    omegaL = cosmo.omegaL
    omegaR = cosmo.omegaR
    boxlen = psi.boxlen
    
    gridsize = len(psi.x.t)
    
    index = np.int32(pos/boxlen*gridsize)
    
    psi1 = psi.x.t[index[0], index[1], index[2]]
    psi2 = psi.y.t[index[0], index[1], index[2]]
    psi3 = psi.z.t[index[0], index[1], index[2]]

    dx = boxlen/gridsize
    f = fpeebl(redshift_end, omegaR, omegaM, omegaL)
    D_end = grow(redshift_end, omegaR, omegaM, omegaL)
    D_start = grow(redshift_start, omegaR, omegaM, omegaL) # used for normalization of D to t = 0
    H = hubble(redshift_end, omegaR, omegaM, omegaL)
    
    xfact = D_end/D_start
    vfact = D_end/D_start*H*f/(1+redshift_end) # KLOPT DIT WEL? CHECK EVEN WAAR AL DE FACTOREN VANDAAN KOMEN
                                               # denk het wel, snelheid heeft verder niets met vorige stappen te maken
    v = vfact * np.array([psi1,psi2,psi3]) # vx,vy,vz
    X = (pos + xfact*(v/vfact))%boxlen
    #~ # Mirror coordinates, because somehow it doesn't match the coordinates put
    #~ # into the constrained field.
    #~ X = boxlen - X # x,y,z
    #~ v = -v
    # FIXED MIRRORING: using different FFT convention now (toolbox.rfftn etc).
    
    return X,v


def two_LPT_ICs(redshift, psi1, psi2, cosmo):
    """
    Use the 2LPT approximation to calculate positions and velocities at
    certain /redshift/, based on the DisplacementField /psi1/ and second order
    DisplacementField2ndOrder /psi2/ of e.g. a Gaussian random density field and
    Cosmology /cosmo/.
    
    Outputs a tuple of a position and velocity vector array; positions are in
    units of h^{-1} Mpc (or in fact the same units as /psi.boxlen/) and
    velocities in km/s.
    """
    boxlen = psi1.boxlen
    
    gridsize = len(psi1.x.t)
    f1 = fpeebl(redshift, cosmo.omegaR, cosmo.omegaM, cosmo.omegaL)
    f2 = 2*cosmo.omegaM**(6./11) # approximation from Bernardeau+01
    D0 = grow(0, cosmo.omegaR, cosmo.omegaM, cosmo.omegaL) # used for normalization of D to t = 0
    D1 = grow(redshift, cosmo.omegaR, cosmo.omegaM, cosmo.omegaL)
    D1 = D1/D0
    D2 = -3./7*D1**2*cosmo.omegaM**(-1./143)
    
    H = hubble(redshift, cosmo.omegaR, cosmo.omegaM, cosmo.omegaL)/(1+redshift) # conformal hubble constant
    
    psi1_vec = np.array([psi1.x.t, psi1.y.t, psi1.z.t])
    psi2_vec = np.array([psi2.x.t, psi2.y.t, psi2.z.t])

    v = D1*H*f1 * psi1_vec + D2*H*f2 * psi2_vec # vx,vy,vz
    # lagrangian coordinates, in the center of the gridcells:
    dx = boxlen/gridsize
    q = np.mgrid[dx/2:boxlen+dx/2:dx,dx/2:boxlen+dx/2:dx,dx/2:boxlen+dx/2:dx]
    
    X = (q + D1*psi1_vec + D2*psi2_vec)%boxlen
    
    return X,v


#~ def zeldovich(redshift, psi1, psi2, psi3, omegaM, omegaL, omegaR, h, boxlen, print_info=False):
    #~ """
    #~ Use the Zel'dovich approximation to calculate positions and velocities at
    #~ certain redshift, based on the displacement vector field of e.g. a Gaussian
    #~ random density field. A displacement vector field can be generated using
    #~ the fieldgen function. Any other field in the same format can be used by
    #~ this function as well.
    #~ 
    #~ Input:
    #~ - redshift:  Redshift to which the Zel'dovich approximation must be made
    #~ - psiN:      Nth component of the displacement vector field.
    #~ - omegaM:    Cosmological matter density parameter
    #~ - omegaL:    Cosmological dark energy density parameter
    #~ - omegaR:    Cosmological radiation density parameter
    #~ - h:         Hubble constant in units of 100 km/s/Mpc (not used anymore)
    #~ - boxlen:    Length of the sides of the box that will be produced, in
                 #~ units of h^{-1} Mpc
                 #~ 
    #~ Output:      [x,y,z,vx,vy,vz]
    #~ - x,y,z:     Particle coordinates (h^{-1} Mpc)
    #~ - vx,vy,vz:  Particle velocities (km/s)
    #~ """
    #~ 
    #~ n1 = len(psi1)
    #~ boxlen = boxlen
    #~ if print_info: print "Boxlen:    ",boxlen
    #~ dx = boxlen/n1
    #~ if print_info: print "dx:        ",dx
    #~ f = fpeebl(redshift, omegaR, omegaM, omegaL )
    #~ if print_info: print "fpeebl:    ",f
    #~ D = grow(redshift, omegaR, omegaM, omegaL)
    #~ D0 = grow(0, omegaR, omegaM, omegaL) # used for normalization of D to t = 0
    #~ if print_info: print "D+(z):     ",D
    #~ if print_info: print "D+(0):     ",D0
    #~ if print_info: print "D(z)/D(0): ",D/D0
    #~ H = hubble(redshift, omegaR, omegaM, omegaL)
    #~ if print_info: print "H(z):      ",H
    #~ 
    #~ xfact = boxlen*D/D0
    #~ vfact = vgad*D/D0*H*f*boxlen/(1+redshift)
    #~ xfact = D/D0
    #~ vfact = D/D0*H*f/(1+redshift)
    #~ 
    #~ v = vfact * np.array([psi1,psi2,psi3]) # vx,vy,vz
    #~ X = (np.mgrid[0:boxlen:dx,0:boxlen:dx,0:boxlen:dx] + xfact*(v/vfact))%boxlen
    #~ # Mirror coordinates, because somehow it doesn't match the coordinates put
    #~ # into the constrained field.
    #~ X = boxlen - X # x,y,z
    #~ v = -v
    #~ 
    #~ return [X[0],X[1],X[2],v[0],v[1],v[2]]


# Cosmological variables
def fpeebl(z, omegaR, omegaM, omegaL):
    """
    Velocity growth factor.
    From Pablo's zeld2gadget.f code (who in turn got it from Bertschinger's).
    """
    if (omegaM == 1) and (omegaL == 0):
        return 1.0
    elif omegaM == 0:
        print "Omega_M <= 0 in fpeebl!"
        sys.exit()

    # Evaluate f := dlog[D+]/dlog[a] (logarithmic linear growth rate) for
    # lambda+matter-dominated cosmology.
    # Omega0 := Omega today (a=1) in matter only.  Omega_lambda = 1 - Omega0.
    omegaK = 1.0 - omegaM - omegaL
    eta = np.sqrt(omegaM*(1+z) + omegaL/(1+z)/(1+z) + omegaK)
    return ( 2.5/grow(z, omegaR, omegaM, omegaL) - 1.5*omegaM*(1+z) - omegaK)/eta**2

def hubble(z, omegar, omegam, omegal):
    """Hubble constant at arbitrary redshift. N.B.: this is in units of h!"""
    return 100*np.sqrt((1+z)**4*omegar + (1+z)**3*omegam + omegal + (1+z)**2*(1-omegar-omegam-omegal))

def growIntgt(a, omegar, omegam, omegal):
    """Integrand for the linear growth factor D(z) (function grow())"""
    if a == 0: return 0
    
    eta = np.sqrt(omegam/a + omegal*a*a + 1 - omegam - omegal)
    return 2.5/eta**3
  
def grow(z, omegar, omegam, omegal):
    a = 1./(1+z)
    integral = integrate(growIntgt, 0, a, args=(omegar, omegam, omegal), vec_func = False)[0]
    eta = np.sqrt(omegam/a + omegal*a*a + 1 - omegam - omegal)
    return eta/a*integral
