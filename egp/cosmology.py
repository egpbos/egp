#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
cosmology.py
/cosmology/ module in the /egp/ package.

Created by Evert Gerardus Patrick Bos.
Copyright (c) 2015. All rights reserved.
"""

import numpy as np

# constants in cosmologically convenient units
hubble_constant = 100 * 3.24077649e-20  # h s^-1
meter_to_megaparsec = 3.24077649e-23  # Mpc
kg_to_solar_mass = 5.02785431e-31  # Msun
gravitational_constant_SI = 6.67300e-11  # m^3 kg^-1 s^-2
gravitational_constant = gravitational_constant_SI * (meter_to_megaparsec)**3 / kg_to_solar_mass  # Mpc^3 Msun^-1 s^-2


def critical_density(G=gravitational_constant):
    """
    Gives the critical density in units of h^2 Msun Mpc^-3.
    The units can be changed by using a value of `G` in different units, e.g.
    by passing `gravitational_constant_SI`.
    """
    return 3. * hubble_constant**2 / 8 / np.pi / G


def LOSToRedshift_wrong(xLOS, vLOS, H, split=False):
    """
    Input:  line of sight distances (Mpc), velocities (km/s) and Hubble
    constant.
    Output: relativistic (Doppler) and cosmological (Hubble) redshifts if
    split = True, otherwise the sum of these (default).
    """
    c = 3.0e5
    zREL = np.sqrt((1+vLOS/c)/(1-vLOS/c)) - 1
    zCOS = xLOS*H/c  # Needs replacement for large cosmological distances
    if split:
        return np.array((zREL, zCOS)).T
    else:
        return zREL + zCOS


def LOSToRedshift(xLOS, vLOS, H, split=False):
    """
    Input:  line of sight distances (Mpc), velocities (km/s) and Hubble
    constant.
    Output: relativistic (Doppler) and cosmological (Hubble) redshifts if
    split = True, otherwise the sum of these (default).
    """
    c = 3.0e5
    zREL = np.sqrt((1+vLOS/c)/(1-vLOS/c)) - 1
    zCOS = xLOS*H/c  # Needs replacement for large cosmological distances
    if split:
        return np.array((zREL, zCOS)).T
    else:
        return zREL + zCOS + zREL*zCOS


def redshiftToLOS(redshift, H):
    """
    Convert redshifts to apparent line of sight distances, ignoring particle
    velocities.

    Input: redshifts and Hubble constant.
    Output: line of sight distances (Mpc).
    """
    c = 3.0e5
    return redshift*c/H
