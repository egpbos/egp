#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  cython_test.py
#  
#  Copyright 2012 E.G.P.   Bos <pbos@schmidt>

import numpy as np
import pyximport
pyximport.install(setup_args={'include_dirs':[np.get_include()]})
from array_multiplication_cython import cmultiply

def pmultiply(array, factor):
    return factor*array

gridsize = 64

x = np.random.random((3,gridsize,gridsize,gridsize))

factor = 30.2

