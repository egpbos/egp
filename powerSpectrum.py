#!/usr/bin/env python
# encoding: utf-8
"""
egpICgen module, power spectrum calculation/interpolation
powerSpectrum.py

Created by Evert Gerardus Patrick Bos.
Copyright (c) January 2012. All rights reserved.

This is an extension of the cosmoICs.py code from my Master's research.
"""

from scipy.integrate import quadrature as quad
from numpy import sqrt, log, sin, cos, exp, pi, e, sinc

def powerspectrum(k, dk, cosmology):
    """
    Determine the power spectrum P(k) for use in the fieldgen function.
    
    Input:
    - k:         k values to calculate the power spectrum at, units of Mpc^{-1}
    - dk:        The used spacing between gridcells in k-space. 2*pi/boxlen.
    - cosmology: Dictionary containing any of the following parameters:
     . omegaM:    Cosmological matter density parameter
     . omegaB:    Cosmological baryonic matter density parameter
     . omegaL:    Cosmological dark energy density parameter
     . h:         Hubble constant in units of 100 km/s/Mpc
     . trans:     Choice of fluctuation (power) spectrum:
                      spectrum:                  function:
                      power-law                  trans1
                      CDM, adiabatic, DEFW       trans2*
                      CDM, adiabatic, BBKS       trans3
                      HDM, adiabatic, BBKS       trans4*
                      WDM, adiabatic, BBKS       trans5*
                      CDM, isocurvature, BBKS    trans6*
                      CDM, Klypin & Holtzman     trans7
                  * Not yet implemented
     . primn:     Power spectrum index n [-3,1] or primordial N [-3,2]
     . rth:       Tophat window radius in units of Mpc (N.B.: not h^{-1} Mpc!)
     . sigma0:    The galaxy number sigma at tophat radius rth
     . bias:      Bias factor b (default = 1)
     . TCMB:      Temperature of the CMB spectrum (default = 2.7 Kelvin)
    
    Output: the power spectrum P(k).
    """
    # Set default values
    if 'bias' not in cosmology:
        cosmology['bias'] = 1
    if 'TCMB' not in cosmology:
        cosmology['TCMB'] = 2.7
    
    cosmology['omega'] = cosmology['omegaM'] + cosmology['omegaL']
    d3k = dk**3
    
    # Set up unconstrained field sample
    kSwitch = 2*pi/cosmology['rth']
    s1 = quad(intgt, 0, kSwitch, args=(cosmology['rth'], cosmology['primn'], cosmology['trans'], cosmology), maxiter=200)[0]
    s2 = quad(intgtinv, 1e-30, 1/kSwitch, args=(cosmology['rth'], cosmology['primn'], cosmology['trans'], cosmology), maxiter=200)[0]
    
    amplit = cosmology['sigma0']**2/(cosmology['bias']**2*(s1+s2))
    return amplit * k**cosmology['primn'] * cosmology['trans'](k, cosmology)**2 * d3k


def intgt(xk, rth, primn, trans, transargs):
    """Integrand used to determine power spectrum amplitude"""
    return xk**primn * windth(xk, rth) * trans(xk, transargs)**2 * 4.0*pi*xk**2

def intgtinv(xk, rth, primn, trans, transargs):
    """Inverse of integrand used to determine power spectrum amplitude"""
    # Met dank aan Wendy die de fout verbeterde die ik had gemaakt
    # (xk = 1.0/xk**3 en dan de hele formule nog maal xk**2).
    xk = 1.0/xk
    #return xk**primn * windth(xk, rth) * trans(xk, transargs)**2 * 4.0*pi
    # 26 januari 2012: klopt bovenstaande wel though? Volgens mij moet het
    # juist extra erbij staan (zie ook nieuwe berekening in logboek):
    return xk**primn * windth(xk, rth) * trans(xk, transargs)**2 * 4.0*pi*xk**4
    # Dit geeft als resultaat een exacte overeenkomst met de Fortran berekening.

def windth(xk, rth):
    """Tophat window function"""
    xkw = xk*rth
    return 9.0*(sin(xkw)-xkw*cos(xkw))**2/xkw**6

def trans1(xk, args):
    """Transfer function for power-law power spectrum"""
    return 1.0
    
def trans3(xk, args):
    """
    Transfer function for the Cold Dark Matter spectrum for adiabatic
    fluctuations as given by: 
        Bardeen, Bond, Kaiser and Szalay,
        Astrophys. J. 304, 15 (1986)
    """
    omegaM = args['omegaM']
    h = args['h']
    
    q = xk/omegaM/h**2
    return log(1+2.34*q)/2.34/q/(1 + 3.89*q + (16.1*q)**2 + (5.46*q)**3 + (6.71*q)**4)**(1./4)

def trans7(xk, args):
    """
    Transfer function for CDM, Klypin & Holtzman power spectrum.
    Reference: astro-ph/9712217.
    """
    omega = args['omega']
    omegaB = args['omegaB']
    h = args['h']
    TCMB = args['TCMB']
    a1 = (46.9*omega*h**2)**0.670 * (1 + (32.1*omega*h**2)**(-0.532))
    a2 = (12*omega*h**2)**0.424 * (1 + (45*omega*h**2)**(-0.582))
    alpha = a1**(-omegaB/omega)*a2**(-(omegaB/omega)**3)
    q = xk*(TCMB/2.7)**2 / (omega*h**2*sqrt(alpha)*(1-omegaB/omega)**0.60)
    return log(1 + 2.34*q)/2.34/q * (1 + 13*q + (10.5*q)**2 + (10.4*q)**3 + (6.51*q)**4)**(-0.25)

def trans8(k, args):
    """
    Transfer function for CDM with baryonic features from Eisenstein & Hu 1999.
    Reference: astro-ph/9709112.
    """
    omegaB = args['omegaB']
    omegaM = args['omegaM']
    omegaCDM = omegaM - omegaB
    h = args['h']
    T27 = args['TCMB']/2.7
    
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
    
    s = 2./3/kEq*sqrt(6/Req)*log( (sqrt(1+Rd) + sqrt(Rd+Req)) / (1+sqrt(Req)) )
    
    y = (1+zEq)/(1+zd)
    G = y*(-6*sqrt(1+y) + (2+3*y)*log((sqrt(1+y) + 1)/(sqrt(1+y)-1)))
    alphaB = 2.07*kEq*s*(1+Rd)**(-3./4) * G
    
    betaB = 0.5 + omegaB/omegaM + (3 - 2*omegaB/omegaM) * \
            sqrt((17.2*omegaM*h*h)**2 + 1)
    
    a1 = (46.9*omegaM*h*h)**0.670 * (1 + (32.1*omegaM*h*h)**-0.532)
    a2 = (12.0*omegaM*h*h)**0.424 * (1 + (45.0*omegaM*h*h)**-0.582)
    alphaC = a1**(-omegaB/omegaM) * a2**(-(omegaB/omegaM)**3)
    
    b1_betaC = 0.944/(1 + (458*omegaM*h*h)**-0.708)
    b2_betaC = (0.395*omegaM*h*h)**-0.0266
    betaC = 1/( 1 + b1_betaC*((omegaCDM/omegaM)**b2_betaC - 1) )
        
    betaNode = 8.41*(omegaM*h*h)**0.435
    st = s / (1+(betaNode/k/s)**3)**(1./3)# s-tilde
    
    C = lambda alC: 14.2/alC + 386/(1+69.9*q**1.08)
    Ttilde0 = lambda k, alC, beC: log(e + 1.8*beC*q) / \
              ( log(e+1.8*beC*q) + C(alC)*q*q )
    
    Tb = (( Ttilde0(k,1,1)/(1+(k*s/5.2)**2) + alphaB/(1+(betaB/k/s)**3) * \
         exp(-(k/kSilk)**1.4) )) * sinc(k*st/2/pi)
    
    f = 1/(1+(k*s/5.4)**4)
    Tcdm = f*Ttilde0(k,1,betaC) + (1-f)*Ttilde0(k,alphaC,betaC)
    
    return omegaB/omegaM*Tb + omegaCDM/omegaM*Tcdm

#def trans9(k, args):
    #"""
    #Transfer function including massive neutrinos from Eisenstein & Hu 1997.
    #Reference: astro-ph/9710252.
    #"""
    
    #q = 
    #qeff = 
    #fNu = 
    #Nnu = 
    #fNuB = 
    #alphaNu = 
    
    #betaC = 1/(1-0.949*fNuB)
    #L = log(e + 1.84*betaC*sqrt(alphaNu)*qeff
    #C = 14.4 + 325/(1+60.5*qeff**1.08)
    #Tsup = L/(L+C*qeff*qeff)
    
    #qNu = 3.92*q*sqrt(Nnu/fNu)
    #Bk = 1 + 1.24 * fNu**0.64 * Nnu**(0.3+0.6*fNu) / (qNu**-1.6 + qNu**0.8)
    #return Tsup*Bk # N.B.: to include massive neutrinos, this should be modified
                   ## by a growth factor D_cb or D_cb\nu.

