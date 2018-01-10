# -*- coding: utf-8 -*-

"""
field_filtering.py
/field_filtering/ module in the /egp/ package.
Filter Fields using convolution kernels.

Created by Evert Gerardus Patrick Bos.
Copyright (c) 2012-2018. All rights reserved.
"""
import numpy as np
import basic_types
import toolbox
import numbers


def gaussian_3d_kernel(gridsize, boxsize, gaussian_scale):
    """
    Build a three dimensional Gaussian kernel on a Field grid. Since we treat
    Fields as periodic, we have to truncate somewhere; we just do this at
    half the grid for now. The kernel is normalized so that the integral over
    the box's volume is one.

    This kernel can be used to smooth other Fields by simply convolving using
    Field.convolve or the @ operator.

    kernelcomp in patchy/barcode do the same as this for kerneltype gauss,
    except that they use a different DFT convention, so we have to compensate
    for that in the Fourier space construction with an extra factor N.
    """
    if not isinstance(gridsize, numbers.Integral):
        raise TypeError("gridsize must be integer!")

    k_squared = toolbox.k_abs_grid(gridsize, boxsize)**2

    kernel_f = np.exp(-k_squared * gaussian_scale**2 / 2) + 0j
    # since we were too lazy to calculate the true normalization factor, we just force it to one:
    kernel_t = basic_types.Field(fourier=kernel_f, odd_3rd_dim=(gridsize % 2 != 0)).t
    kernel_f /= kernel_t.sum()
    # then, dividing by dV is necessary to make sure the integral over V is 1
    dV = boxsize**3 / gridsize**3
    kernel_f /= dV
    kernel = basic_types.Field(fourier=kernel_f, boxsize=boxsize, odd_3rd_dim=(gridsize % 2 != 0))

    return kernel


def gaussian_3d_kernel_real_space(gridsize, boxsize, gaussian_scale):
    """
    Build a three dimensional Gaussian kernel on a Field grid. Since we treat
    Fields as periodic, we have to truncate somewhere; we just do this at
    half the grid for now. The kernel is normalized so that the integral over
    the box's volume is one.

    This kernel can be used to smooth other Fields by simply convolving using
    Field.convolve or the @ operator.

    This kernel differs slightly from the Fourier-space defined one in
    gaussian_3d_kernel due to the effects of the Fourier transform.
    """
    if not isinstance(gridsize, numbers.Integral):
        raise TypeError("gridsize must be integer!")

    half = boxsize / 2
    dx = boxsize / gridsize
    x = np.mgrid[-half:half:dx, -half:half:dx, -half:half:dx]
    x_sq = np.fft.fftshift((x**2).sum(axis=0))[::-1, ::-1, ::-1]

    kernel = np.exp(- x_sq / gaussian_scale**2 / 2) / (np.sqrt(2 * np.pi) * gaussian_scale)**3
    kernel = basic_types.Field(true=kernel, boxsize=boxsize)

    return kernel


def test_gaussian_3d_kernel():
    gridsize = 32
    boxsize = 10.
    gaussian_scale = 0.8  # magic number, it is not at all equal for other numbers!
    assert np.allclose(gaussian_3d_kernel(gridsize, boxsize, gaussian_scale).t,
                       gaussian_3d_kernel_real_space(gridsize, boxsize, gaussian_scale).t)

##
# old stuff previously in toolbox
##


def filter_Field(field, kernel, kernel_arguments, gridsize=None):
    """Returns a new Field object that is the input Field /field/
    convolved with a kernel. Kind of speaks for itself, I'd say. Use
    gaussian_kernel or tophat_kernel for /kernel/ and a list of
    appropriate kernel arguments in /kernel_arguments/. Gridsize can be
    optionally specified, otherwise the first shape element of
    fieldFourier of /field.f/ will be used."""
    field_fourier = field.f
    boxlen = field.boxlen
    if not gridsize:
        gridsize = field.f.shape[0]
    field_fourier_filtered = filter_field(field_fourier, boxlen, kernel, kernel_arguments, gridsize)
    return basic_types.Field(fourier=field_fourier_filtered)


def filter_field(fieldFourier, boxlen, kernel, kernel_arguments, gridsize=None):
    """Returns the fourier-space representation of the field
    convolved with a kernel. Kind of speaks for itself, I'd say. Use
    gaussian_kernel or tophat_kernel for /kernel/ and a list of
    appropriate kernel arguments in /kernel_arguments/. Gridsize can be
    optionally specified, otherwise the first shape element of
    fieldFourier will be used."""
    if not gridsize:
        gridsize = fieldFourier.shape[0]
    k = toolbox.k_abs_grid(gridsize, boxlen)
    return fieldFourier * kernel(k, *kernel_arguments)


def gaussian_kernel(k, r_g):
    return np.exp(-k * k * r_g * r_g / 2)


def tophat_kernel(k, r_th):
    x = k * r_th
    kernel = 3 * (np.sin(x) - x * np.cos(x)) / x**3
    return np.nan_to_num(kernel)  # to get out the divisions by zero


def gaussian_smooth(densityFourier, r_g, boxlen):
    """Returns the fourier-space representation of the smoothed density field."""
    gridsize = len(densityFourier)
    k = toolbox.k_abs_grid(gridsize, boxlen)

    def windowGauss(ka, Rg):
        return np.exp(-ka * ka * Rg * Rg / 2)   # N.B.: de /2 factor is als je het veld smooth!
                                                # Het PowerSpec heeft deze factor niet.

    return densityFourier * windowGauss(k, r_g)


# def tophat_smooth(densityFourier, r_th, boxlen):
#     """Returns the fourier-space representation of the smoothed density field."""
#     gridsize = len(densityFourier)
#     k = k_abs_grid(gridsize, boxlen)
