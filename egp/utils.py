#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
utils.py
/utils/ module in the /egp/ package.

Created by Evert Gerardus Patrick Bos.
Copyright (c) 2018. All rights reserved.

Utility functions used internally in multiple modules.
"""

import numpy as np
import warnings
import collections
import numbers


def boxsize_tuple(boxsize):
    """
    :param boxsize: When a single number, it will be converted to a np.array
    with three times that same number. When a sequence of size 3 it will convert
    that to the output np.array.
    """
    if isinstance(boxsize, (collections.abc.Sequence, np.ndarray)) and len(boxsize) == 3:
        boxtuple = np.array(boxsize)
        if boxsize[0] != boxsize[1] or boxsize[1] != boxsize[2]:
            warnings.warn("unequal box sizes are not taken into account in most functions",
                          category=RuntimeWarning)
    elif isinstance(boxsize, numbers.Real):
        boxtuple = np.array([boxsize, boxsize, boxsize])
    else:
        raise ValueError("boxsize must either be a single number (for cubic boxes) or a sequence of 3 numbers!")

    return boxtuple
