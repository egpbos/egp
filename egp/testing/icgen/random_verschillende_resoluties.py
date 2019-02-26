#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from numpy.random import random as rnd, seed as setseed

seed = 0

# Zo werd het gedaan; dit levert dus simulaties op die verschillend zijn voor
# verschillende gridsizes!
gridsize = 4
np.random.seed(seed)
arg4 = np.random.random((gridsize,gridsize,gridsize/2+1))
mod4 = 1 - np.random.random((gridsize,gridsize,gridsize/2+1)) # "1 -" so there's no zero
z4 = np.sqrt(-np.log(mod4)) * np.exp(1j*2*np.pi*arg4)

gridsize = 8
np.random.seed(seed)
arg8 = np.random.random((gridsize,gridsize,gridsize/2+1))
mod8 = 1 - np.random.random((gridsize,gridsize,gridsize/2+1)) # "1 -" so there's no zero
z8 = np.sqrt(-np.log(mod8)) * np.exp(1j*2*np.pi*arg8)


# Is dit een betere manier? Neen helaas:
gridsize = 4
np.random.seed(seed)
argmod4 = np.random.random((2,gridsize,gridsize,gridsize/2+1))
argmod4[1] = 1 - argmod4[1] # "1 -" so there's no zero
z4 = np.sqrt(-np.log(argmod4[1])) * np.exp(1j*2*np.pi*argmod4[0])

gridsize = 8
np.random.seed(seed)
argmod8 = np.random.random((2,gridsize,gridsize,gridsize/2+1))
argmod8[1] = 1 - argmod8[1] # "1 -" so there's no zero
z8 = np.sqrt(-np.log(argmod8[1])) * np.exp(1j*2*np.pi*argmod8[0])


#hmm hoe doet gadget het eigenlijk?
