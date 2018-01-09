# -*- coding: utf-8 -*-

"""
cosmography.py
/cosmography/ module in the /egp/ package.
Methods for describing the large-scale matter distribution and kinematics of the
Universe and simulations thereof.

Created by Evert Gerardus Patrick Bos.
Copyright (c) 2018. All rights reserved.
"""

import numpy as np

import egp.basic_types
import egp.cosmology
import warnings


class DensityField(egp.basic_types.Field):
    """
    A single component Field representing density on a periodic grid. It can be
    initialized in several ways, but by default it is stored as an array with
    units of the critical density,
    $\rho_c = \frac{3 H^2}{8 \pi G} = 1.88 \times 10^{-26} h^2 kg m^{-3} $.
    To convert to physical density, multiply the raw field by the critical
    density (e.g. using `cosmology.critical_density()`).

    :param boxsize: A density field must have a physical size to be meaningful.
    It can be given as a single number (for cubic boxes) or as a sequence of
    three numbers. Note that this is not an explicit keyword argument of this
    class's initializer, but rather of the Field class.

    :param input_type: string that specifies the type of input density given to
    initialize the DensityField. This allows conversion to the correct internal
    units. Options are:
    - 'overdensity' (default): input in units of the critical density, but minus
      one, so that empty space has value -1 and 0 is the mean density.
    - 'SI': input in SI units, h^2 kg m^{-3}.
    - 'cosmology': input in cosmological density units, h^2 Msun Mpc^-3.
    - 'critical': input already in internal units, no conversion.

    :param **kwargs: Remaining kwargs are passed on to the Field initializer.
    """

    def __init__(self, input_type='overdensity', **kwargs):
        raw_field = egp.basic_types.Field(**kwargs)
        field_t = raw_field.t
        if input_type == 'overdensity':
            # sanity checks:
            if np.any(field_t < -1):
                raise Exception("overdensity fields should not have values below -1!")
            if np.all(field_t > 0):
                warnings.warn("overdensity field has no values below zero, which is highly unlikely (unless the field is zero everywhere)")
            if not np.allclose(field_t.mean(), 0):
                warnings.warn("overdensity field mean is {}, should be close to zero".format(field_t.mean()))

            # transform to internal units:
            field_t += 1
        elif input_type == 'SI':
            # sanity checks:
            if np.any(field_t < 0):
                raise Exception("density fields should not have values below 0!")
            # transform to internal units:
            rho_c = egp.cosmology.critical_density(G=egp.cosmology.gravitational_constant_SI)
            field_t /= rho_c
        elif input_type == 'cosmology':
            # sanity checks:
            if np.any(raw_field.t < 0):
                raise Exception("density fields should not have values below 0!")
            # transform to internal units:
            rho_c = egp.cosmology.critical_density()
            field_t /= rho_c
        elif input_type == 'critical':
            pass
        else:
            raise ValueError(f'input_type "{input_type}" not valid!')

        if 'boxsize' not in kwargs:
            warnings.warn("The boxsize keyword argument was not given in DensityField.__init__, so it will be set to the default value for Field.__init__. It is highly unlikely that this is what you intended!")

        kwargs.pop('true', None)
        super().__init__(true=field_t, **kwargs)
