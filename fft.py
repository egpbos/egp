# -*- coding: utf-8 -*-
# @Author: pbos
# @Date:   2018-01-08 16:22:07
# @Last Modified by:   pbos
# @Last Modified time: 2018-01-08 16:31:47

"""
fft.py
/fft/ module in the /egp/ package.
Defines custom DFT conventions.

Created by Evert Gerardus Patrick Bos.
Copyright (c) 2012-2018. All rights reserved.
"""

import numpy as np


# rFFTn's with flipped minus sign convention
def rfftn_flip(A, *args, **kwargs):
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
    in- and output arrays. This is what this function and irfftn_flip thus undo.
    Try plotting np.fft.fft(np.fft.fft(A)) versus A to see for yourself.
    """
    norm = 1 / np.size(A)
    return norm * np.fft.rfftn(A[::-1, ::-1, ::-1], *args, **kwargs)


def irfftn_flip(A, *args, **kwargs):
    """
    Inverse real N-dimensional fast fourier transform, with flipped minus sign
    convention. See rfftn_flip.
    """
    real_space = np.fft.irfftn(A, *args, **kwargs)[::-1, ::-1, ::-1]
    norm = np.size(real_space)  # factor from discrete to true Fourier transform

    return norm * real_space


def ifftn_flip(A, *args, **kwargs):
    """
    Inverse N-dimensional fast fourier transform, with flipped minus sign
    convention. See rfftn_flip.
    """
    norm = np.size(A)  # factor from discrete to true Fourier transform
    return norm * np.fft.ifftn(A, *args, **kwargs)[::-1, ::-1, ::-1]
