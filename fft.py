# -*- coding: utf-8 -*-
# @Author: pbos
# @Date:   2018-01-08 16:22:07
# @Last Modified by:   pbos
# @Last Modified time: 2018-01-09 14:08:56

"""
fft.py
/fft/ module in the /egp/ package.
Defines custom DFT conventions.

Created by Evert Gerardus Patrick Bos.
Copyright (c) 2012-2018. All rights reserved.
"""

import numpy as np


# (r)FFTn's with flipped minus sign convention

def rfftn(A, *args, **kwargs):
    """
    Real N-dimensional fast fourier transform, with flipped minus sign
    convention.

    The convention used by NumPy is that the FFT has a minus sign in the
    exponent and the inverse FFT has a plus. This is opposite to the convention
    used e.g. in Numerical Recipes, but, more importantly, it is opposite to the
    fourier transform convention used by Van de Weygaert & Bertschinger (1996).
    This means that if you use the NumPy FFT to compute a constrained field,
    the box will be mirrored and your constraints will not be where you expect
    them to be in the field grid. In this, we assume the field-grid-indices-to-
    physical-coordinates transformation to be simply i/gridsize*boxlen and the
    other way around, physical-coordinates-to-grid-indices transformation to be
    int(x/boxlen*gridsize).

    The effect of a changed sign in the FFT convention is a mirroring of your
    in- and output arrays. This is what this function and irfftn thus undo.
    Try plotting np.fft.fft(np.fft.fft(A)) versus A to see for yourself.
    """
    norm = 1 / np.size(A)
    return norm * np.fft.rfftn(A[::-1, ::-1, ::-1], *args, **kwargs)


def irfftn(A, *args, **kwargs):
    """
    Inverse real N-dimensional fast fourier transform, with flipped minus sign
    convention. See rfftn.
    """
    real_space = np.fft.irfftn(A, *args, **kwargs)[::-1, ::-1, ::-1]
    norm = np.size(real_space)  # factor from discrete to true Fourier transform

    return norm * real_space


def irfftn_odd(A, *args, **kwargs):
    """
    Inverse real N-dimensional fast fourier transform, with flipped minus sign
    convention and odd number of grid cells in the third dimension. See rfftn.
    """
    if 's' in kwargs:
        print("Warning: popping shape parameter 's' from kwargs in irfftn_odd!")
        kwargs.pop('s')

    s = A.shape
    s = (s[0], s[1], s[2] * 2 - 1)

    return irfftn(A, *args, s=s, **kwargs)


def ifftn(A, *args, **kwargs):
    """
    Inverse N-dimensional fast fourier transform, with flipped minus sign
    convention. See rfftn.
    """
    norm = np.size(A)  # factor from discrete to true Fourier transform
    return norm * np.fft.ifftn(A, *args, **kwargs)[::-1, ::-1, ::-1]
