#!/usr/bin/env python
# encoding: utf-8
"""
egpICgen module, constrained random field generation
constrainedField.py

Created by Evert Gerardus Patrick Bos.
Copyright (c) February 2012. All rights reserved.

First try for constrained code: functional version; transscription of Rien's
fortran code.

Second try will be an object oriented version.
"""

import numpy as np
from matplotlib import pyplot as pl
from egp.icgen import CosmoPowerSpectrum as PS, Cosmology, GaussianRandomField as GRF, DensityField
from matplotlib.pyplot import figure, show
from time import time
from csv import reader as csvreader
from scipy.special import erf
from scipy.interpolate import UnivariateSpline
import egp.zeldovich


# ---- CLASSES ----

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
            k = k_abs_grid(self.gridsize, self.boxlen)
            ki = k_i_grid(self.gridsize, self.boxlen)
            self.correlations.calculate(k, ki) # P(k) H_i(k) and \xi_ij^-1
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
    
    def calculate(self, k, ki):
        self.calculate_constraint_correlations(k, ki)
        self.calculate_field_constraint_correlations(k, ki)
    
    def calculate_constraint_correlations(self, k, ki, fast=False):
        """Calculate the matrix xi_ij containing the correlation values between
        the different constraints."""
        power = self.power(k)
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
    
    def calculate_field_constraint_correlations(self, k, ki):
        """Calculate the correlations between the field values and the
        constraint H matrices, i.e. P(k) H_i(k)."""
        H = np.empty((len(self.constraints),) + k.shape, dtype=np.complex)
        for i, coni in enumerate(self.constraints):
            H[i] = coni.H(ki)
        PH = self.power(k)*H
        self.PH = PH


class ConstraintLocation(object):
    def __init__(self, location):
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
        ka = np.sqrt(k[0]**2 + k[1]**2 + k[2]**2)
        self._common_kernel_factor =  np.exp(-ka*ka * self.scale**2/2)


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
        self._H = 3./2 * C.omegaM * (C.h*100)**2 * 1j * k[self.direction] / \
                  k_squared * self.W_factor(k)
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
        self._H = 3./2 * C.omegaM * (C.h*100)**2 * \
                  (k[self.k_index1] * k[self.k_index2] / k_squared - delta/3.) * \
                  self.W_factor(k)
        self._H[0,0,0] = 0 # complex division of zero by zero gives NaN


# --------------------------- CONVENIENCE FUNCTIONS ---------------------------

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

def gaussian_smooth(densityFourier, r_g, boxlen):
    gridsize = len(densityFourier)
    halfgrid = gridsize/2
    dk = 2*np.pi/boxlen
    k = k_abs_grid(gridsize, boxlen)
    
    def windowGauss(ka, Rg):
        return np.exp( -ka*ka*Rg*Rg/2 ) # N.B.: de /2 factor is als je het veld smooth!
                                        # Het PowerSpec heeft deze factor niet.
    
    return DensityField(fourier=densityFourier*windowGauss(k,r_g))

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


# ---------------------------- FACTORIES  ----------------------------

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
    
    fpeebl = egp.zeldovich.fpeebl
    z = 0
    velocity_to_gravity = 2*fpeebl(z, cosmo.h, cosmo.omegaR, cosmo.omegaM, \
                          cosmo.omegaL) / 3 / (cosmo.h*100) / cosmo.omegaM
    
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
        
        # ----- Spectral parameters (for easier constraint value definition, at
        # ----- least in case of gaussian random field):
        sigma0 = power_spectrum.moment(0, scale.scale, boxlen**3)
        sigma1 = power_spectrum.moment(1, scale.scale, boxlen**3)
        sigma2 = power_spectrum.moment(2, scale.scale, boxlen**3)
        sigma_min1 = power_spectrum.moment(-1, scale.scale, boxlen**3)
        gamma_nu = sigma0**2 / sigma_min1 / sigma1
        sigma_g = 3./2 * cosmo.omegaM * (cosmo.h*100)**2 * sigma_min1
        sigma_g_peak = sigma_g * np.sqrt(1 - gamma_nu**2)
        gamma = sigma1**2/sigma0/sigma2
        sigma_E = 3./2*cosmo.omegaM*(cosmo.h*100)**2 * sigma0 * np.sqrt((1-gamma**2)/15)
        
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
        # if one is defined, randomly define all:
        if curvature or a21 or a31 or density_phi or density_theta or density_psi:
            if not curvature:
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
            else:
                curvature = float(curvature)
            if not a21 or not a31:
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
        
        # ----- Velocity
        vx = row[14]
        vy = row[15]
        vz = row[16]
        
        # Convert to gravity:
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

# ------------------------- FACTORY HELPER FUNCTIONS -----------------

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

# --------------------------- TESTING CODE ---------------------------

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

# ---- FIELDS & CONSTRAINTS ----
# Unconstrained field
rhoU = GRF(ps, boxlen, gridsize, seed=None)

# Constraint Locations/Scales
#~ location1 = ConstraintLocation(np.array((50,50,50))/cosmo.h) # Mpc
#~ location2 = ConstraintLocation(np.array((50,75,75))/cosmo.h) # Mpc
#~ scale1 = ConstraintScale(8/cosmo.h) # Mpc

# Spectral parameters (for easier constraint value definition, at least in case
# of gaussian random field):
#~ sigma0 = ps.moment(0, scale1.scale, boxlen**3)
#~ sigma1 = ps.moment(1, scale1.scale, boxlen**3)
#~ sigma2 = ps.moment(2, scale1.scale, boxlen**3)
#~ 
#~ sigma_min1 = ps.moment(-1, scale1.scale, boxlen**3)
#~ gamma_nu = sigma0**2 / sigma_min1 / sigma1
#~ sigma_g = 3./2 * cosmo.omegaM * (cosmo.h*100)**2 * sigma_min1
#~ sigma_g_peak = sigma_g * np.sqrt(1 - gamma_nu**2)
#~ 
#~ gamma = sigma1**2/sigma0/sigma2
#~ sigma_E = 3./2*cosmo.omegaM*(cosmo.h*100)**2 * sigma0 * np.sqrt((1-gamma**2)/15)

# Constraints ...
# ... height (positive for peak, negative for void) ...
#~ heightVal = 5*sigma0
#~ height = HeightConstraint(location1, scale1, heightVal)
#~ height2 = HeightConstraint(location2, scale1, heightVal)

# ... extrema ...
#~ extre1 = ExtremumConstraint(location1, scale1, 0)
#~ extre2 = ExtremumConstraint(location1, scale1, 1)
#~ extre3 = ExtremumConstraint(location1, scale1, 2)

# ... gravity ...
#~ g1 = GravityConstraint(location1, scale1, 3*sigma_g_peak, 0, cosmo)
#~ g2 = GravityConstraint(location1, scale1, 3*sigma_g_peak, 1, cosmo)
#~ g3 = GravityConstraint(location1, scale1, 3*sigma_g_peak, 2, cosmo)

# ... shape ...
#~ x_d = 5. # steepness
#~ a12 = 8. # first axial ratio
#~ a13 = 1. # second axial ratio
#~ alpha = 0. # Euler angle (w.r.t box axes)
#~ beta  = 0. # Euler angle
#~ psi   = 0. # Euler angle

# BEGIN factory voor shape:
#~ A = euler_matrix(alpha, beta, psi)
#~ lambda1 = x_d * sigma2 / (1 + a12**2 + a13**2)
#~ lambda2 = lambda1 * a12**2
#~ lambda3 = lambda1 * a13**2
#~ 
#~ d11 = - lambda1*A[0,0]*A[0,0] - lambda2*A[1,0]*A[1,0] - lambda3*A[2,0]*A[2,0]
#~ d22 = - lambda1*A[0,1]*A[0,1] - lambda2*A[1,1]*A[1,1] - lambda3*A[2,1]*A[2,1]
#~ d33 = - lambda1*A[0,2]*A[0,2] - lambda2*A[1,2]*A[1,2] - lambda3*A[2,2]*A[2,2]
#~ d12 = - lambda1*A[0,0]*A[0,1] - lambda2*A[1,0]*A[1,1] - lambda3*A[2,0]*A[2,1]
#~ d13 = - lambda1*A[0,0]*A[0,2] - lambda2*A[1,0]*A[1,2] - lambda3*A[2,0]*A[2,2]
#~ d23 = - lambda1*A[0,1]*A[0,2] - lambda2*A[1,1]*A[1,2] - lambda3*A[2,1]*A[2,2]
#~ 
#~ shape1 = []
#~ shape1.append(ShapeConstraint(location1, scale1, d11, 0, 0))
#~ shape1.append(ShapeConstraint(location1, scale1, d22, 1, 1))
#~ shape1.append(ShapeConstraint(location1, scale1, d33, 2, 2))
#~ shape1.append(ShapeConstraint(location1, scale1, d12, 0, 1))
#~ shape1.append(ShapeConstraint(location1, scale1, d13, 0, 2))
#~ shape1.append(ShapeConstraint(location1, scale1, d23, 1, 2))
# END shape factory

# ... and tidal field.
#~ epsilon = 10. # strength of the tidal field
#~ pomega  = 0.25*np.pi # parameter for relative strength along axes (0, 2*pi)
#~ alphaE = 0.5*np.pi # Euler angle (w.r.t box axes)
#~ betaE  = 0.25*np.pi # Euler angle
#~ psiE   = 0. # Euler angle

# BEGIN factory voor tidal field:
#~ T = euler_matrix(alphaE, betaE, psiE)
#~ L1 = np.cos((pomega + 2*np.pi)/3)
#~ L2 = np.cos((pomega - 2*np.pi)/3)
#~ L3 = np.cos(pomega/3)
#~ fac = epsilon*sigma_E
#~ 
#~ E11 = fac*(L1*T[0,0]*T[0,0] + L2*T[1,0]*T[1,0] + L3*T[2,0]*T[2,0])
# Note that in these values the epsilon*f_G term (whatever f_G might have been)
# is neglected due to the field being linear (see vdW&B96).
#~ E22 = fac*(L1*T[0,1]*T[0,1] + L2*T[1,1]*T[1,1] + L3*T[2,1]*T[2,1])
#~ E12 = fac*(L1*T[0,0]*T[0,1] + L2*T[1,0]*T[1,1] + L3*T[2,0]*T[2,1])
#~ E13 = fac*(L1*T[0,0]*T[0,2] + L2*T[1,0]*T[1,2] + L3*T[2,0]*T[2,2])
#~ E23 = fac*(L1*T[0,1]*T[0,2] + L2*T[1,1]*T[1,2] + L3*T[2,1]*T[2,2])
#~ 
#~ tidal1 = []
#~ tidal1.append(TidalConstraint(location1, scale1, E11, 0, 0, cosmo))
#~ tidal1.append(TidalConstraint(location1, scale1, E22, 1, 1, cosmo))
#~ tidal1.append(TidalConstraint(location1, scale1, E12, 0, 1, cosmo))
#~ tidal1.append(TidalConstraint(location1, scale1, E13, 0, 2, cosmo))
#~ tidal1.append(TidalConstraint(location1, scale1, E23, 1, 2, cosmo))
# END tidal field factory


# Constrained fields, 21-23 February 2012:
#~ rhoC1 = ConstrainedField(rhoU, [height,])
#~ rhoC2 = ConstrainedField(rhoU, [height, extre1, extre2, extre3])
#~ rhoC3 = ConstrainedField(rhoU, [height, height2])


# Constrained fields, 9 March 2012:
#rhoC4 = ConstrainedField(rhoU, [height, g1, g2, g3]) # gravity/velocity werkt
#rhoC4 = ConstrainedField(rhoU, [height] + shape1) # shape werkt
#~ rhoC4 = ConstrainedField(rhoU, [height] + tidal1)

# Constrained fields, 22 May 2012:
constraints1 = constraints_from_csv("/Users/users/pbos/code/egpTesting/icgen/constraints1.csv", ps, boxlen)
rhoC1 = ConstrainedField(rhoU, constraints1)

# ---- PLOTTING ----
# SyncMaster 2443 dpi:
y = 1200 #pixels
dInch = 24 # inch (diagonal)
ratio = 16./10 # it's not a 16/9 screen
yInch = dInch/np.sqrt(ratio**2+1)
dpi = y/yInch

fig = figure(figsize=(20/2.54,24/2.54), dpi=dpi)
ax1 = fig.add_subplot(3,2,1)
ax2 = fig.add_subplot(3,2,2)
ax3 = fig.add_subplot(3,2,3)
ax4 = fig.add_subplot(3,2,4)
ax5 = fig.add_subplot(3,2,5)

ax1.imshow(rhoU.t[halfgrid], interpolation='nearest')
ax2.imshow(rhoC1.t[halfgrid], interpolation='nearest')
#~ ax3.imshow(rhoC2.t[halfgrid], interpolation='nearest')
#~ ax4.imshow(rhoC3.t[halfgrid], interpolation='nearest')
#~ ax5.imshow(rhoC4.t[halfgrid], interpolation='nearest')
pl.show()


# ------------ PROBABILITY FUNCTIONS ----------- -
# curvature
x = np.linspace(0,10,5001)

cpdf = cpdf_curvature(x, height.value, gamma)
#cumcpdf = np.cumsum(cpdf)/cpdf.sum()
cumcpdfBBKS = cumulative_cpdf_curvature(x,height.value, gamma)
invPDF = UnivariateSpline(cumcpdfBBKS, x, k=3, s=0)
curvature = invPDF(np.random.random())

pl.plot(x,cpdf,'-',x,cumcpdfBBKS,'-')#,x,cumcpdf,'-')
pl.show()

# shape
p,e = np.mgrid[-0.25:0.25:0.005, 0:0.5:0.005]
xmax = x[cpdf.argmax()]
cpdf_ep = cpdf_shape(e, p, xmax)
pl.imshow(cpdf_ep, interpolation='nearest')
pl.show()
