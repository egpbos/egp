#!/usr/bin/env python
# encoding: utf-8
"""
egpICgen module, constrained random field generation
constrainedField.py

Created by Evert Gerardus Patrick Bos.
Copyright (c) February 2012. All rights reserved.

First try for constrained code: functional version; transscription of Rien's
fortran code.

Second try will be a object oriented version.
"""

import numpy as np
from matplotlib import pyplot as pl
from egp.icgen import CosmoPowerSpectrum as PS, Cosmology, GaussianRandomField as GRF, DensityField
from matplotlib.pyplot import figure, show
from time import time

def gaussian_smooth(densityFourier, r_g, boxlen):
    gridsize = len(densityFourier)
    halfgrid = gridsize/2
    dk = 2*np.pi/boxlen
    
    k12 = np.fft.fftfreq(gridsize, 1/dk/gridsize) # k3 = k12[:halfgrid+1].abs()
    k = np.sqrt(k12[:halfgrid+1]**2 + k12[:,np.newaxis]**2 + k12[:,np.newaxis,np.newaxis]**2)
    
    def windowGauss(k, Rg):
        return np.exp( -k*k*Rg*Rg/2 ) # N.B.: de /2 factor is als je het veld smooth!
                                      # Het PowerSpec heeft deze factor niet.
    
    return DensityField(fourier=densityFourier*windowGauss(k,r_g))

# ---- PLOTTING ----
# SyncMaster 2443 dpi:
y = 1200 #pixels
dInch = 24 # inch (diagonal)
ratio = 16./10 # it's not a 16/9 screen
yInch = dInch/np.sqrt(ratio**2+1)
dpi = y/yInch

fig = figure(figsize=(20/2.54,28/2.54), dpi=dpi)
ax1 = fig.add_subplot(3,2,1)
ax2 = fig.add_subplot(3,2,2)
ax3 = fig.add_subplot(3,2,3)
ax4 = fig.add_subplot(3,2,4)
ax5 = fig.add_subplot(3,2,5)

# ---- BASICS (cosmology, box, etc) ----
cosmo = Cosmology('wmap7')
cosmo.trans = 8

boxlen = 100/cosmo.h
gridsize = 64

dk = 2*np.pi/boxlen
kmax = gridsize*dk
halfgrid = gridsize/2

ps = PS(cosmo)
ps.normalize(boxlen**3)


# ---- CONSTRAINTS ----

class ConstraintLocationError(Exception):
    pass


class ConstraintLocation(object):
    def __init__(self, location):
        # Implementeer iets als dit hieronder met een Factory
        # zie: http://stackoverflow.com/questions/674304/pythons-use-of-new-and-init
        
        #~ if len(location.shape) > 1:
            #~ objects = []
            #~ for loc in location:
                #~ objects.append(ConstraintLocation(loc))
            #~ return objects
        #~ else:
        self.location = location
        self.constraints = []
    
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
    
    def add_constraint(self, constraint):
        constraint_types = [type(con) for con in self.constraints]
        if type(constraint) in constraint_types:
            if type(constraint) == HeightConstraint:
                raise ConstraintLocationError("There can only be one HeightConstraint at a ConstraintLocation!")
            if type(constraint) == ExtremumConstraint:
                extrema_directions = [con.direction for con in self.constraints if type(con) == ExtremumConstraint]
                if constraint.direction in extrema_directions:
                    raise ConstraintLocationError("There can only be one ExtremumConstraint in the %i direction at a ConstraintLocation!" % constraint.direction)
        self.constraints.append(constraint)


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
        ka = np.sqrt(k[0]**2 + k[1]**2 + k[2]**2)
        self._common_kernel_factor =  np.exp(-ka*ka * self.scale**2/2)


#location1 = ConstraintLocation(np.array((50,50,50))/cosmo.h) # Mpc

# ConstraintScale object maken!
# Bevat gemeenschappelijk deel van die schaal (np.exp(-k*k * Rg*Rg / 2)). Bevat
# ook de spectrale parameters op die schaal (alleen berekend als ze worden
# aangeroepen)... of toch niet? Nee toch niet! Zie hieronder.
#scale1 = ConstraintScale(8/cosmo.h) # Mpc

# spectral parameters...
# AHA, HIER ZIT EEN PROBLEEM MET DE SETUP VAN DEZE METHODE!!!!
# De sigma's worden bepaald door de schaal en een integraal over het power
# spectrum. Eigenlijk zouden de sigma's echter gewoon deel uit moeten maken van
# de constraints; het zijn slechts de eenheden van de constraint waardes daar.
# MAAR, hoe je het ook wendt of keert, met deze eenheden blijf je het power
# spectrum nodig hebben. Dit is een vrij willekeurige keuze en is bovendien
# alleen nuttig bij een Gaussian random field of in elk geval een
# veld/kosmologie die door je power spectrum gedefinieerd is en ook een zinnige
# beschrijving geeft. Als je een non-gaussian field maakt is dat laatste volgens
# mij niet meer het geval!
# Oplossing: constraints invoeren in fysische eenheden, niet genormaliseerd
# met spectrale parameters.
# N.B.: aan het eind komt het probleem wel weer terug; een Hoffman-Ribak CRF
# heeft per definitie een power spectrum nodig.

# Constraint object:
# Braucht iig een locatie en een schaal.

class Constraint(object):
    def __init__(self, location, scale):
        self.location = location
        self.scale = scale
        location.add_constraint(self)
    
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
    Specify the direction with the direction keyword."""
    # To have the option to leave out direction (=> None) and get a list of
    # three ExtremumConstraint objects returned, see above Factory/__new__
    # idea in the stackoverflow url.
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
    1,1; 2,2; 3,3; 1,2; 1,3 and 2,3."""
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
        self._H = 3./2 * C.omegaM * (C.h*100)**2 * 1j * k[self.direction] / \
                  k_squared * self.W_factor(k)


class TidalConstraint(Constraint):
    """Constrains the tidal forces on the peak. There are five independent
    eigenvalues in the tidal force tensor, leading to five valid combinations
    of k_index{1,2}: 1,1; 2,2; 1,2; 1,3 and 2,3. The cosmology needs to be given
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
        if k_index1 == k_index2:
            delta = 1
        else:
            delta = 0
        self._H = 3./2 * C.omegaM * (C.h*100)**2 * \
                  (k[self.k_index1] * k[self.k_index2] / k_squared - delta/3.) * \
                  self.W_factor(k)


# In deze class gaan we er van uit dat een density een power attribute heeft.
# Dit kunnen we in het algemene DensityField (of zelfs in het Field?) als
# property implementeren die standaard een EstimatedPowerSpectrum bepaalt als
# dat nodig is en die in het GRF geval natuurlijk gewoon gegeven is.
class ConstrainedField(DensityField):
    def __init__(self, unconstrained_density, constraints, calculate = True):
        self.constraints = constraints
        constraint_values = []
        for constraint in self.constraints:
            constraint_values.append(constraint.value)
        self.constraint_values = np.array(constraint_values)
        self.rhoU = unconstrained_density
        self.gridsize = self.rhoU.gridsize
        self.boxlen = self.rhoU.boxlen
        self.power = self.rhoU.power
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
            self.calculate_constraint_correlations() # \xi_ij
            #self.calculate_field_constraint_correlations() # P(k) H_i(k) 
            self.calculate_unconstrained_constraints() # c_j^~
            self.calculate_residual_field() # P(k) H_i(k) \xi_ij^-1 (c_j-c_j^~)
            return self._residual_field
    @rhoR.setter
    def rhoR(self, densityField):
        self._residual_field = densityField
        
    def calculate_residual_field(self):
        # doe hier de dubbele sommatie over de constraints
        delta_c_j = np.matrix(self.constraint_values - self.c_unconstrained).T
        xi_times_c = np.matrix(self.xi_ij_inverse) * delta_c_j
        PH = self.calculate_field_constraint_correlations()
        rhoR_f = np.sum(PH * np.array(xi_times_c)[:, np.newaxis, np.newaxis], axis=0)
        self.rhoR = DensityField(fourier = rhoR_f)
    
    def calculate_constraint_correlations(self, fast=False):
        # Maak hier een np.matrix van ixj.
        k = k_abs_grid(self.gridsize, self.boxlen)
        ki = k_i_grid(self.gridsize, self.boxlen)
        power = self.power(k)
        constraints = self.constraints
        if fast:
            # Dit is ongeveer 3 keer sneller, dus maakt eigenlijk niet veel uit.
            # DIT GAAT NOG NIET GOED! FIXEN! Lijkt een probleem met einsum te
            # zijn... zie einsumPrecisie.py
            H = np.empty((len(constraints),) + k.shape, dtype=np.complex)
            for i, coni in enumerate(constraints):
                H[i] = coni.H(ki)
            xi_ij = 2*np.einsum('iklm,jklm->ij', H[...,1:-1].conj(), H[...,1:-1]*power[...,1:-1]).real
            # ... where the k,l,m indices sum over the kx,ky,kz axes.
            #     and the factor 2 is to account for the symmetry in kz.
            # Correct for k_z = 0 and nyquist planes (they count only once):
            xi_ij += np.einsum('iklm,jklm->ij', H[...,(0,-1)].conj(), H[...,(0,-1)]*power[...,(0,-1)]).real
            self.xi_ij_inverse = np.matrix(xi_ij).I
        else:
            self.xi_ij_inverse = np.empty((len(constraints), len(constraints)))
            t0 = time()
            for i, coni in enumerate(constraints):
                Hi = coni.H(ki)
                for j, conj in enumerate(constraints):
                    Hj = conj.H(ki)
                    xi_ij = np.sum( 2*Hi[...,1:-1].conj()*Hj[...,1:-1]*power[...,1:-1] ).real
                    # plus k_z = 0 and nyquist planes:
                    xi_ij += np.sum( Hi[...,(0,-1)].conj()*Hj[...,(0,-1)]*power[...,(0,-1)] ).real
                    self.xi_ij_inverse[i,j] = xi_ij # first store xi_ij
            print time()-t0
            self.xi_ij_inverse = np.matrix(self.xi_ij_inverse).I # then invert
    
    def calculate_field_constraint_correlations(self, store=False):
        # Maak hier een np.matrix van Nx1xi.
        k = k_abs_grid(self.gridsize, self.boxlen)
        ki = k_i_grid(self.gridsize, self.boxlen)
        H = np.empty((len(self.constraints),) + k.shape, dtype=np.complex)
        for i, coni in enumerate(self.constraints):
            H[i] = coni.H(ki)
        PH = self.power(k)*H
        if store:
            self.PH = PH
        return PH
    
    def calculate_unconstrained_constraints(self):
        # en hier een np.matrix van jx1.
        self.c_unconstrained = np.empty((len(self.constraints),))
        ki = k_i_grid(self.gridsize, self.boxlen)
        for i, coni in enumerate(self.constraints):
            Hi_conj = coni.H(ki).conj()
            cU = np.sum( 2*Hi_conj[...,1:-1]*self.rhoU.f[...,1:-1] ).real
            cU += np.sum( Hi_conj[...,(0,-1)]*self.rhoU.f[...,(0,-1)] ).real # k_z = 0, nyquist
            self.c_unconstrained[i] = cU


def k_abs_grid(gridsize, boxlen):
    halfgrid = gridsize/2
    dk = 2*np.pi/boxlen
    k12 = np.fft.fftfreq(gridsize, 1/dk/gridsize) # k3 = k12[:halfgrid+1].abs()
    return np.sqrt(k12[:halfgrid+1]**2 + k12[:,np.newaxis]**2 + k12[:,np.newaxis,np.newaxis]**2)

def k_i_grid(gridsize, boxlen):
    halfgrid = gridsize/2
    dk = 2*np.pi/boxlen
    k1, k2, k3 = dk*np.mgrid[0:gridsize, 0:gridsize, 0:halfgrid+1]
    k1 -= kmax*(k1 > dk*(halfgrid - 1)) # shift the second half to negative k values
    k2 -= kmax*(k2 > dk*(halfgrid - 1))
    return np.array((k1,k2,k3))


# ConstrainedField object:
# Braucht een lijst van Constraint objecten en een initieel DensityField.

#~ g = np.array((height,)) # constraints array g

# conkmn:
# general part of constraint kernels in fourier space (dep. on pos & scale):
#~ k1, k2, k3 = dk*np.mgrid[0:gridsize, 0:gridsize, 0:halfgrid+1]
#~ k1 -= kmax*(k1 > dk*(halfgrid - 1)) # shift the second half to negative k values
#~ k2 -= kmax*(k2 > dk*(halfgrid - 1))
#~ kSq = k1**2 + k2**2 + k3**2
#~ k = np.sqrt(kSq)
#~ kr = position[0]*k1 + position[1]*k2 + position[2]*k3 # ff nadenken over welke component met
                                                      #~ # welke vermenigvuldigd moet worden.
#~ del(k1, k2, k3) # keep memory clean
#~ gskrnc = np.exp(1j*kr) * np.exp(-kSq * scale**2/2)
# now the separate constraint matrix elements dependent on constraint choices:

#facsig = 1/sigma0 # This is to compensate for the fact that the constraint
                  # values in array g are unitless. This means that H(k) is
                  # effectively redefined to contain this factor in eqn. 38 of
                  # vdW&B96 (c_i = dimless_i * norm_fact where e.g. for height
                  # norm_fact is sigma0). This all turns out fine in the end
                  # in eqn. 35, because there we have:
                  # a. the now unitless c_j's,
                  # b. the \xi_ij^-1 which contains the unit of j (because
                  #    \xi_ij contains H_j which now contains 1/(unit of j)) and
                  #    the unit of i (same story),
                  # c. the \xi_i contains 1/(unit of i) (again, same story).
# Dit doen wij dus niet, want het is alleen handig in het geval van een random
# field dat netjes gedefinieerd is aan de hand van (alleen) een power spectrum.
# We willen hier een algemeen geldigere code maken, voor alle soorten velden.
#H1 = facsig * gskrnc
#~ H1 = gskrnc
#~ power = ps(k)
#~ H1conj = H1.conj()
#~ 
#~ # covmat:
#~ # calculate covariance matrix q (note: with one constraint it's a 1x1 matrix):
#~ q = np.sum( 2*H1conj[...,1:-1]*H1[...,1:-1]*power[...,1:-1] )
#~ # plus k_z = 0 and nyquist planes:
#~ q += np.sum( H1conj[...,(0,-1)]*H1[...,(0,-1)]*power[...,(0,-1)] )
#~ # invert it:
#~ qinv = 1/q
#~ 
#~ rhoMarray = ps(k)*H1*qinv*g # Mean field
#~ 
#~ rhoM = DensityField(fourier = rhoMarray)
#~ ax1.imshow(rhoM.t[halfgrid], interpolation='nearest')
#~ # GEEFT EEN PRACHTIGE PIEK IN HET MIDDEN! Mean field werkt dus.
#~ 
#~ # unconstrained field:
rhoU = GRF(ps, boxlen, gridsize, seed=None)
#~ 
#~ # concmp:
#~ # calculate constraint values on the unconstrained field
#~ g0 = np.sum( 2*H1conj[...,1:-1]*rhoU.f[...,1:-1] )
#~ g0 += np.sum( H1conj[...,(0,-1)]*rhoU.f[...,(0,-1)] ) # k_z = 0, nyquist
#~ 
#~ rhoCarray = rhoU.f + power*H1*qinv*(g - g0)
#~ rhoC = DensityField(fourier = rhoCarray)
#~ ax2.imshow(rhoC.t[halfgrid], interpolation='nearest')
#~ pl.show()
#~ 
#~ # unconstrained field minus values of constraints on unconstrained field
#~ rhoCMinArray = rhoU.f - power*H1*qinv*g0
#~ rhoCMin = DensityField(fourier = rhoCMinArray)
#~ ax3.imshow(rhoCMin.t[halfgrid], interpolation='nearest')
#~ pl.show()
#~ 
#~ # the minus field above:
#~ rhoMinArray = -power*H1*qinv*g0
#~ rhoMin = DensityField(fourier = rhoMinArray)
#~ ax4.imshow(rhoMin.t[halfgrid], interpolation='nearest')
#~ 
#~ ax5.imshow(rhoU.t[halfgrid], interpolation='nearest')
#~ 
#~ print rhoC.t[halfgrid,halfgrid,halfgrid]/sigma0
#~ 
#~ pl.show()
#~ 
#~ # Nu checken of het klopt in de gesmoothe versie, want daar is ie daadwerkelijk
#~ # geconstraint:
#~ rhoMs8 = gaussian_smooth(rhoM.f, scale, boxlen)
#~ rhoMs8.t[halfgrid,halfgrid,halfgrid]
#~ # Out[7]: 3.7288455976751096
#~ # Perfect! Moet zijn:
#~ sigma0*10
#~ #Out[9]: 3.7288455976751136
#~ 
#~ rhoCMins8 = gaussian_smooth(rhoCMin.f, scale, boxlen)
#~ rhoCMins8.t[halfgrid,halfgrid,halfgrid]
# Out[13]: -2.0261570199409107e-15
# Super! Moet nul zijn.

location1 = ConstraintLocation(np.array((50,25,25))/cosmo.h) # Mpc
scale1 = ConstraintScale(8/cosmo.h) # Mpc
location2 = ConstraintLocation(np.array((50,75,75))/cosmo.h) # Mpc

position = location1
scale = scale1
# height (positive for peak, negative for void)
# spectral parameters (for easier constraint value definition in case of gaussian random field):
sigma0 = ps.moment(0, scale1.scale, boxlen**3)
sigma1 = ps.moment(1, scale1.scale, boxlen**3)
sigma2 = ps.moment(2, scale1.scale, boxlen**3)
heightVal = 5*sigma0
# N.B.: 'csphgt' (Riens code) zet uiteindelijk \nu = height/sigma0 in g(i)!
#       Dat doen we hier dus niet.

height = HeightConstraint(position, scale, heightVal)
height2 = HeightConstraint(location2, scale, heightVal)
extre1 = ExtremumConstraint(position, scale, 0)
extre2 = ExtremumConstraint(position, scale, 1)
extre3 = ExtremumConstraint(position, scale, 2)


# 21-23 February 2012:
rhoC1 = ConstrainedField(rhoU, [height,])
rhoC2 = ConstrainedField(rhoU, [height, extre1, extre2, extre3])
rhoC3 = ConstrainedField(rhoU, [height, height2])

ax1.imshow(rhoU.t[halfgrid], interpolation='nearest')
ax2.imshow(rhoC1.t[halfgrid], interpolation='nearest')
ax3.imshow(rhoC2.t[halfgrid], interpolation='nearest')
ax4.imshow(rhoC3.t[halfgrid], interpolation='nearest')
pl.show()

# ------------------------------------------------------------------------------
# Older tries.
# ------------------------------------------------------------------------------

#~ def constraintCorrelation(k, Hi, xDi, Hj, xDj, power):
    #~ """
    #~ The correlation function of constraint kernels Hi and Hj, given power
    #~ spectrum power on grid k. Constraints are defined at positions xDi and
    #~ xDj.
    #~ """
    #~ 
#~ 
#~ def fieldConstraintCorrelation(k, field, Hi, xDi, power):
    #~ """
    #~ The correlation function of constraint kernel Hi with the field, given
    #~ power spectrum power on grid k. Constraints are defined at positions xDi
    #~ and xDj.
    #~ """
    #~ 
#~ 
#~ # Constraint kernels in Fourier space
#~ 
#~ def H1(k, x):
    #~ """
    #~ Constraint kernel for the height of the peak.
    #~ """
    #~ 
