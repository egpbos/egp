#!/usr/bin/env python
# encoding: utf-8
"""
egpICgen module, Gaussian random field generation
gaussianField.py

Created by Evert Gerardus Patrick Bos.
Copyright (c) January 2012. All rights reserved.

This is an extension of the cosmoICs.py code from my Master's research.
"""
# On April 27, 2010, the parts from cosmoICs.py were again tested by comparing
# output to Rien's fieldgen64.f program, and indeed these routines gave exactly
# the same results (i.e. statistically of course, not exactly because of
# randomness).
# On January 26, 2012, the renewed version was again tested and we are now very
# sure of the fact that results are the same as the fieldgen64.f ones (i.e. when
# ksphere = False, because Rien's code had no such feature).
# See testing/vergelijkNewOld.py.

import numpy as np
from numpy.random import random as rnd, seed as setseed
from numpy import sqrt, log, cos, exp, pi
from time import time
import powerSpectrum as PS

def fieldgen(cosmology, boxlen, gridsize, seed = None, ksphere = True, returnFourier = False, returnPower = False):
    """
    Fieldgen builds a Gaussian random density field (and corresponding
    displacement vector field) for use as initial conditions in a cosmological
    N-body simulation. Before use, the Zel'dovich approximation needs to be
    applied to the output of this function to obtain actual particle positions
    and velocities.
    
    Input:
    - cosmology: Dictionary with cosmological parameters. See powerspectrum for
                 a description of possible entries.
    - boxlen:    Length of the sides of the box that will be produced. Not in
                 units of h^{-1} Mpc, so divide by h before input if necessary!
    - gridsize:  Number of grid points in one direction
    - seed:      If given, a random seed is passed to the random number
                 generator, otherwise NumPy finds its own seed somewhere.
    - ksphere:   If true, a sphere in k-space is defined with radius k_nyquist.
                 Outside of this sphere, all grid values are zero. If false,
                 these are also just random values, subject to symmetries.
    - returnFourier: affects the output of the function. The output is a list
                     of either four or eight NumPy arrays; if returnFourier is
                     True, the output is [rhoR, psi1R, psi2R, psi3R, rho,
                     psi1, psi2, psi3] if it is False the output is [rhoR,
                     psi1R, psi2R, psi3R].
    - returnPower:   same story; adds the power spectrum to the end of the
                     output list if True, doesn't if False.
    
    Output:
    - rhoR:      The Gaussian random density field.
    - psiNR:     The Nth component of the displacement vector field.
    - rho:       The Gaussian random density field in Fourier space.
    - psiN:      The Nth component of the displacement vector field in Fourier
                 space.
    """
    returnFourierInput = returnFourier
    outRho = densityFieldgen(cosmology, boxlen, gridsize, seed, ksphere, True, returnPower)
    outPsi = displacementField(outRho[1], boxlen, gridsize, ksphere, returnFourierInput)
    out = [outRho[0], outPsi[0], outPsi[1], outPsi[2]] # real space
    if returnFourierInput:
        out.extend([outRho[1], outPsi[3], outPsi[4], outPsi[5]]) # Fourier space
    if returnPower:
        out.append(outRho[-1]) # Power spectrum
    return out

def densityFieldgen(cosmology, boxlen, gridsize, seed = None, ksphere = True, returnFourier = False, returnPower = False):
    """
    Part of fieldgen that builds the density field. See fieldgen for info.
    Can be run independently.
    """
    # Initialize some used constants
    halfgrid = gridsize/2
    gridcells = gridsize**3
    dk = 2*pi / boxlen
    kmax = gridsize*dk
    
    print "Begin building rho array in Fourier space..."
    
    # Initialize fourier-space k-values grid and grid indices
    ki1, ki2, ki3 = np.mgrid[0:gridsize, 0:gridsize, 0:halfgrid+1] # indices
    k1, k2, k3 = dk*np.mgrid[0:gridsize, 0:gridsize, 0:halfgrid+1] # the k values
    k1 -= kmax*(ki1 > halfgrid - 1) # shift the second half to negative k values
    k2 -= kmax*(ki2 > halfgrid - 1)
    k = sqrt(k1**2 + k2**2 + k3**2)
    
    # Compute power spectrum at all grid points
    power = PS.powerspectrum(k, dk, cosmology)
    
    # Fill in the grid with random numbers
    # ... for the regular grid components:
    if seed:
        setseed(seed)
    arg = rnd((gridsize,gridsize,halfgrid+1))
    mod = 1 - rnd((gridsize,gridsize,halfgrid+1)) # "1 -" so there's no zero
    z = sqrt(-log(mod)) * exp(1j*2*pi*arg)
    # Reduce z by sqrt(2) for real f since need sum of |z|^2 over all
    # wavenumbers (positive and negative) to be Chi-square for N degrees
    # of freedom, not 2N as in the complex case.  Equivalently, the power
    # is shared with the conjugate harmonics with k1 > halfgrid (k1 < 0).
    
    # ... for the 7 real, independent, non-zero grid values:
    real7x,real7y,real7z = np.mgrid[0:2,0:2,0:2]*halfgrid
    real7x,real7y,real7z = real7x.ravel()[1:],real7y.ravel()[1:],real7z.ravel()[1:]
    z[real7x,real7y,real7z] = sqrt(2) * sqrt(-log(mod[real7x,real7y,real7z])) * cos(2*pi*arg[real7x,real7y,real7z]) + 0j
    # Here you do still have the sqrt(2)...
    
    # ... the 8th real, independent grid value [0,0,0] is zero:
    z[0,0,0] = 0
    
    # Then, the actual density field (in Fourier space):
    rho = sqrt(power)*z
    
    # Finally add symmetry to the nyquist planes (so the ifft is not imaginary):
    symmetrizeMatrix(rho)
    rho[0,0,0] = 0.0 # Dit is nodig omdat er een 1/k**2 in Z zit, en k[0,0,0] == 0. Het kan wel iets anders zijn, maar het is de integraal over je veld en dus heb je dan geen veld meer met gemiddelde waarde nul, wat we wel willen.
    
    # Setting everything outside of the k-sphere with radius k_nyquist to zero.
    if ksphere:
        ksphere = k <= kmax/2
        rho *= ksphere
    
    print "Done building rho."
    
    print "Begin Fourier transformation on rho..."
    rhoR = gridcells*np.fft.irfftn(rho)
    print "Fourier on rho done"

    out = [rhoR,]
    if returnFourier:
        out.append(rho)
    if returnPower:
        out.append(power)
    return out

def displacementField(rho, boxlen, gridsize, ksphere = True, returnFourier = False):
    """
    Part of fieldgen that computes the displacement field belonging to a
    density field rho (in Fourier space). Can be run independently.
    """
    # Initialize some used constants
    halfgrid = gridsize/2
    gridcells = gridsize**3
    dk = 2*pi / boxlen
    kmax = gridsize*dk
    
    print "Begin building psi arrays in Fourier space..."
    
    # Initialize fourier-space k-values grid and grid indices
    ki1, ki2, ki3 = np.mgrid[0:gridsize, 0:gridsize, 0:halfgrid+1] # indices
    k1, k2, k3 = dk*np.mgrid[0:gridsize, 0:gridsize, 0:halfgrid+1] # the k values
    k1 -= kmax*(ki1 > halfgrid - 1) # shift the second half to negative k values
    k2 -= kmax*(ki2 > halfgrid - 1)
    k = sqrt(k1**2 + k2**2 + k3**2)
    
    # Indices for the 7 real, independent, non-zero grid values:
    real7x,real7y,real7z = np.mgrid[0:2,0:2,0:2]*halfgrid
    real7x,real7y,real7z = real7x.ravel()[1:],real7y.ravel()[1:],real7z.ravel()[1:]
    
    # Then, the actual displacement field:
    k[0,0,0] = 1 # to avoid division by zero
    Z = 1.0j/k**2/boxlen * rho
    Z[real7x,real7y,real7z] = 0.0 # waarom dit eigenlijk?
    Z[0,0,0] = 0.0
    psi1 = k1*Z
    psi2 = k2*Z
    psi3 = k3*Z
    
    # Finally add symmetry to the nyquist planes (so the ifft is not imaginary):
    symmetrizeMatrix(psi1)
    symmetrizeMatrix(psi2)
    symmetrizeMatrix(psi3)
    
    # Setting everything outside of the k-sphere with radius k_nyquist to zero.
    if ksphere:
        ksphere = k <= kmax/2
        psi1 *= ksphere
        psi2 *= ksphere
        psi3 *= ksphere
    # Dit is nutteloos: het hangt gewoon van rho af; als het daar is gedaan komt
    # het hiet ook automatisch op nul. Zo niet, dan hoeft het ook niet. Gefixt
    # in egp.icgen.
    
    print "Done building psi."
    
    print "Begin Fourier transformations on psi..."
    psi1R = gridcells*np.fft.irfftn(psi1)
    psi2R = gridcells*np.fft.irfftn(psi2)
    psi3R = gridcells*np.fft.irfftn(psi3)
    print "Fourier on psi done."

    if returnFourier:
        return [psi1R, psi2R, psi3R, psi1, psi2, psi3]
    else:
        return [psi1R, psi2R, psi3R]

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

def symmetrizeMatrixExplicit(m, halfgrid):
    # Only symmetrizes the matrix in k3==(0||nyquist), because we only need to
    # fill the k3<=nyquist part for an inverse fft to a real field; so not in
    # the k1 and k2 nyquist planes and not outside the nyquist planes either.
    gridsize = m.shape[0]
    halfgrid = gridsize/2
    # plane intersections
    m[0,-1:halfgrid:-1,0]           = m[0,1:halfgrid,0].conjugate()
    m[0,-1:halfgrid:-1,halfgrid]        = m[0,1:halfgrid,halfgrid].conjugate()
    m[-1:halfgrid:-1,0,0]           = m[1:halfgrid,0,0].conjugate()
    m[-1:halfgrid:-1,0,halfgrid]        = m[1:halfgrid,0,halfgrid].conjugate()
    m[halfgrid,-1:halfgrid:-1,0]        = m[halfgrid,1:halfgrid,0].conjugate()
    m[halfgrid,-1:halfgrid:-1,halfgrid] = m[halfgrid,1:halfgrid,halfgrid].conjugate()
    m[-1:halfgrid:-1,halfgrid,0]        = m[1:halfgrid,halfgrid,0].conjugate()
    m[-1:halfgrid:-1,halfgrid,halfgrid] = m[1:halfgrid,halfgrid,halfgrid].conjugate()
    # rest of the planes
    m[-1:halfgrid:-1,-1:halfgrid:-1,0]      = m[1:halfgrid,1:halfgrid,0].conjugate()
    m[-1:halfgrid:-1,-1:halfgrid:-1,halfgrid]   = m[1:halfgrid,1:halfgrid,halfgrid].conjugate()
    m[-1:halfgrid:-1,1:halfgrid,0]      = m[1:halfgrid,-1:halfgrid:-1,0].conjugate()
    m[-1:halfgrid:-1,1:halfgrid,halfgrid]   = m[1:halfgrid,-1:halfgrid:-1,halfgrid].conjugate()


