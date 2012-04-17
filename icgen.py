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
from numpy.random import random as rnd, seed as setseed
from scipy.integrate import quadrature as integrate
from scipy.interpolate import InterpolatedUnivariateSpline as Interpolator
from periodic import PeriodicArray


# constants
__version__ = "0.4, February 2012"


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
    def __init__(self, choice, trans = None):
        """
        Initialize a Cosmology instance with some default values.
        """
        if not choice:
            self.omegaM, self.omegaB, self.omegaL, self.h = 0.3, 0.04, 0.7, 0.7
            self.rth, self.sigma0 = 8.0, 0.8
            self.primn = 1.0
        elif choice.lower() == "wmap3":
            self.omegaM, self.omegaB, self.omegaL, self.h = 0.268, 0.044, 0.732, 0.704
            self.rth, self.sigma0 = 8.0, 0.776
            self.primn = 0.947
        elif choice.lower() == "wmap7":
            self.omegaM, self.omegaB, self.omegaL, self.h = 0.272, 0.0456, 0.728, 0.704
            self.rth, self.sigma0 = 8.0, 0.809
            self.primn = 0.963
        self.bias, self.TCMB = 1.0, 2.7
        self.trans = trans
    
    def __str__(self):
        out = ""
        out +=   "Omega_matter:                     " + str(self.omegaM)
        out += "\nOmega_baryon:                     " + str(self.omegaB)
        out += "\nOmega_lambda:                     " + str(self.omegaL)
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
    """
    def __init__(self):
        self.normalized = False
        
    def __call__(self, k):
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
        /volume/ that it will be used on, on a top-hat scale Rth with sigma0."""
        
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
        """
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
        st = s / (1+(betaNode/k/s)**3)**(1./3)# s-tilde
        
        C = lambda alC: 14.2/alC + 386/(1+69.9*q**1.08)
        Ttilde0 = lambda k, alC, beC: np.log(np.e + 1.8*beC*q) / \
                  ( np.log(np.e+1.8*beC*q) + C(alC)*q*q )
        
        Tb = (( Ttilde0(k,1,1)/(1+(k*s/5.2)**2) + alphaB/(1+(betaB/k/s)**3) * \
             np.exp(-(k/kSilk)**1.4) )) * np.sinc(k*st/2/np.pi)
        
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
            self._fourier = np.fft.rfftn(self.t)/np.size(self.t)
            return self._fourier
    @f.setter
    def f(self, field):
        self._fourier = field
        if field is None:
            self._ifft = np.fft.irfftn
        elif field.shape[0] == field.shape[2]:
            self._ifft = np.fft.ifftn
        elif field.shape[0] == (field.shape[2]-1)*2:
            self._ifft = np.fft.irfftn
    
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


class VectorField(object):
    """
    A three component vector field, containing three Field instances as
    attributes x, y and z.
    Initialization parameter true must have shape (3,N,N,N) and fourier must
    have shape (3,N,N,N/2+1).
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
        halfgrid = self.gridsize/2
        dk = 2*np.pi / self.boxlen
        kmax = self.gridsize*dk
        
        # Initialize fourier-space k-values grid
        k1, k2, k3 = dk*np.mgrid[0:self.gridsize, 0:self.gridsize, 0:halfgrid+1]
        k1 -= kmax*(k1 > dk*(halfgrid - 1)) # shift the second half to negative k values
        k2 -= kmax*(k2 > dk*(halfgrid - 1))
        k = np.sqrt(k1**2 + k2**2 + k3**2)
        k[0,0,0] = 1 # to avoid division by zero

        # Indices for the 7 real, independent, non-zero grid values:
        real7x,real7y,real7z = np.mgrid[0:2,0:2,0:2]*halfgrid
        real7x,real7y,real7z = real7x.ravel()[1:],real7y.ravel()[1:],real7z.ravel()[1:]
        
        # Then, the actual displacement field:
        Z = 1.0j/k**2/self.boxlen * self.density.f
        Z[real7x,real7y,real7z] = 0.0 # waarom dit eigenlijk?
        Z[0,0,0] = 0.0
        VectorField.__init__(self, fourier = (k1*Z, k2*Z, k3*Z))
        
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
    - boxlen:    Length of the sides of the box that will be produced. Not in
                 units of h^{-1} Mpc, so divide by h before input if necessary!
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
        self.ksphere = ksphere
        self.build_fourier()
    
    def build_fourier(self):
        # Initialize some used constants
        halfgrid = self.gridsize/2
        dk = 2*np.pi / self.boxlen
        kmax = self.gridsize*dk
        
        #print "Begin building rho array in Fourier space..."
        
        # Initialize fourier-space k-values grid
        k1, k2, k3 = dk*np.mgrid[0:self.gridsize, 0:self.gridsize, 0:halfgrid+1]
        k1 -= kmax*(k1 > dk*(halfgrid - 1)) # shift the second half to negative k values
        k2 -= kmax*(k2 > dk*(halfgrid - 1))
        k = np.sqrt(k1**2 + k2**2 + k3**2)
        del(k1, k2, k3) # keep memory clean
        
        # Fill in the grid with random numbers
        # ... for the regular grid components:
        arg = resolution_independent_random_grid(self.gridsize, self.seed)
        mod = 1 - resolution_independent_random_grid(self.gridsize, self.seed+1) # "1 -" so there's no zero
        z = np.sqrt(-np.log(mod)) * np.exp(1j*2*np.pi*arg)
        # Reduce z by sqrt(2) for real f since need sum of |z|^2 over all
        # wavenumbers (positive and negative) to be Chi-square for N degrees
        # of freedom, not 2N as in the complex case.  Equivalently, the power
        # is shared with the conjugate harmonics with k1 > halfgrid (k1 < 0).
        
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
        self.f = np.sqrt(self.power(k))*z
        
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


class Constraint(object):
    def __init__(self, sign, scale, position, height=None, gradient=None, \
                 curvature=None, shape=None, orientation=None, velocity=None, \
                 shear=None):
        """
        sign, scale, position, height, curvature: scalars
        gradient, orientation, velocity: 3 components
        shape: 2 components
        shear: 5 components
        """
        self.sign = np.sign(sign)
        self.scale = scale
        self.position = np.array(position)
        if height:
            self.height = height # IS DIT HANDIG? EERST FF BEDENKEN HOE WE DINGEN UIT GAAN REKENEN HIERNA
    
    def spectral_params(self):
        # Deze functie gaat natuurlijk heel vaak voor dezelfde scale aangeroepen
        # worden. Hij kan dan ook beter in een verzamelklasse die de integralen
        # een keer per scale doet. Misschien in de ConstraintCorrelation of
        # in een soort globale klasse die aan elke Constraint wordt meegegeven
        # als common ding; ConstraintCommonality. Aan de andere kant, als deze
        # integraal 0.1 seconde duurt, dan is dit allemaal niet nodig en is het
        # voor de duidelijkheid van de code beter om het hier te laten denk ik.
        pass
        # IDEE: een klasse die een soort dictionary met property()'s bevat; als
        # je het element met een bepaalde radius aanroept geeft ie em terug als
        # ie er al is en zo niet dan berekent ie em.
        # Dit kan dus het beste in een andere klasse geimplementeerd worden;
        # een klasse die Constraints neemt en daarmee de matrix uitrekent ofzo.
        # Daar zijn de spectral parameters toch eigenlijk pas nodig.


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

def resolution_independent_random_grid(gridsize, seed):
    np.random.seed(seed)
    return np.random.random((gridsize,gridsize,gridsize/2+1))