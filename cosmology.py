#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
cosmology.py
/cosmology/ module in the /egp/ package.

Created by Evert Gerardus Patrick Bos.
Copyright (c) 2015. All rights reserved.
"""

import numpy as np


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
