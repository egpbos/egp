#!/usr/bin/env python
# encoding: utf-8
"""
egpICgen module, Zel'dovich approximation
zeldovich.py

Created by Evert Gerardus Patrick Bos.
Copyright (c) January 2012. All rights reserved.

This is an extension of the cosmoICs.py code from my Master's research.
"""

import numpy as np
from scipy.integrate import quadrature as quad
from numpy import sqrt, log, sin, cos, exp, pi

def zeldovich(redshift, psi1, psi2, psi3, omegaM, omegaL, omegaR, h, boxlen):
    """
    Use the Zel'dovich approximation to calculate positions and velocities at
    certain redshift, based on the displacement vector field of e.g. a Gaussian
    random density field. A displacement vector field can be generated using
    the fieldgen function. Any other field in the same format can be used by
    this function as well.
    
    Input:
    - redshift:  Redshift to which the Zel'dovich approximation must be made
    - psiN:      Nth component of the displacement vector field.
    - omegaM:    Cosmological matter density parameter
    - omegaL:    Cosmological dark energy density parameter
    - omegaR:    Cosmological radiation density parameter
    - h:         Hubble constant in units of 100 km/s/Mpc
    - boxlen:    Length of the sides of the box that will be produced, in
                 units of h^{-1} Mpc
                 
    Output:      [x,y,z,vx,vy,vz]
    - x,y,z:     Particle coordinates (Mpc)
    - vx,vy,vz:  Particle velocities (km/s)
    """
    
    n1 = len(psi1)
    boxlen = boxlen / h
    print "Boxlen:    ",boxlen
    dx = boxlen/n1
    print "dx:        ",dx
    f = fpeebl(redshift, h, omegaR, omegaM, omegaL )
    print "fpeebl:    ",f
    D = grow(redshift, omegaR, omegaM, omegaL)
    D0 = grow(0, omegaR, omegaM, omegaL) # used for normalization of D to t = 0
    print "D+(z):     ",D
    print "D+(0):     ",D0
    print "D(z)/D(0): ",D/D0
    H = hubble(redshift, h, omegaR, omegaM, omegaL)
    print "H(z):      ",H
    
    X, Y, Z = np.zeros(n1**3), np.zeros(n1**3), np.zeros(n1**3)
    vx, vy, vz = np.zeros(n1**3), np.zeros(n1**3), np.zeros(n1**3)
    
    # Velocity correction, needed in GADGET for comoving (cosm.) simulation
    vgad = sqrt(1+redshift)
    
    xfact = boxlen*D/D0
    vfact = vgad*D/D0*H*f*boxlen/(1+redshift)
    
    for x in range(n1):
        for y in range(n1):
            for z in range(n1):
                # take modulus for periodic box
                X[x*n1**2 + y*n1 + z] = (x*dx + xfact*psi1[x][y][z])%boxlen
                Y[x*n1**2 + y*n1 + z] = (y*dx + xfact*psi2[x][y][z])%boxlen
                Z[x*n1**2 + y*n1 + z] = (z*dx + xfact*psi3[x][y][z])%boxlen
                vx[x*n1**2 + y*n1 + z] = vfact*psi1[x][y][z]
                vy[x*n1**2 + y*n1 + z] = vfact*psi2[x][y][z]
                vz[x*n1**2 + y*n1 + z] = vfact*psi3[x][y][z]
    
    return [X,Y,Z,vx,vy,vz]


# Cosmological variables
def fpeebl(z, h, omegaR, omegaM, omegaL):
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
    eta = sqrt(omegaM*(1+z) + omegaL/(1+z)/(1+z) + omegaK)
    return ( 2.5/grow(z, omegaR, omegaM, omegaL) - 1.5*omegaM*(1+z) - omegaK)/eta**2

def hubble(z, h, omegar, omegam, omegal):
    """Hubble constant at arbitrary redshift"""
    return 100*h*sqrt((1+z)**4*omegar + (1+z)**3*omegam + omegal + (1+z)**2*(1-omegar-omegam-omegal))

def growIntgt(a, omegar, omegam, omegal):
    """Integrand for the linear growth factor D(z) (function grow())"""
    if a == 0: return 0
    
    eta = sqrt(omegam/a + omegal*a*a + 1 - omegam - omegal)
    return 2.5/eta**3
  
def grow(z, omegar, omegam, omegal):
    a = 1./(1+z)
    integral = quad(growIntgt, 0, a, args=(omegar, omegam, omegal), vec_func = False)[0]
    eta = sqrt(omegam/a + omegal*a*a + 1 - omegam - omegal)
    return eta/a*integral


