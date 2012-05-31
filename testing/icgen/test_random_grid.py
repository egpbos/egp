#!/usr/bin/env python
# RESULTAAT TESTEN VOOR GAUSSIAN RANDOM FIELD
from egp.icgen import GaussianRandomField as GF, CosmoPowerSpectrum as PS, Cosmology as Cosmo

cosmo = Cosmo()
cosmo.trans = 8
power = PS(cosmo)

out00 = GF(power, 100./0.7, 64, seed=2)
out01 = GF(power, 100./0.7, 128, seed=2)
out10 = GF(power, 200./0.7, 64, seed=2)
out11 = GF(power, 200./0.7, 128, seed=2)

# WERKT!!!

##

# OPTIMALISATIE
import pyublas, egp.crunch as c
from timeit import Timer

t1 = Timer("array=c.resolution_independent_random_grid(64,0)", "import pyublas; import egp.crunch as c")
t1.timeit(1000)
#Out[4]: 1.1734099388122559

t2 = Timer("array=np.random.random((64,64,33))", "import numpy as np")
t2.timeit(1000)
#Out[8]: 1.9942600727081299

t3 = Timer("array=c.naive_random_grid(64,0)", "import pyublas; import egp.crunch as c")
t3.timeit(1000)
#Out[13]: 0.70445013046264648

t4 = Timer("array=c.resolution_independent_random_grid_seedtable(64,0)", "import pyublas; import egp.crunch as c")
t4.timeit(1000)
#Out[16]: 12.354748964309692

# Python tests of indexed versus iterator access:
# from timeit import Timer
# t1 = Timer("array=c.resolution_independent_random_grid(16,0)", "import pyublas; import egp.crunch as c")
# t2 = Timer("array=np.random.random((16,16,9))", "import numpy as np")
# t1.timeit(10000)
# Out[108]: 40.59319806098938 # FOR INDEXED ACCESS
# Out[4]: 32.884864091873169 # FOR ITERATOR ACCESS => almost no difference!
# t2.timeit(10000)
# Out[5]: 0.35071587562561035 # So yeah, this is the goal...
# DIT GEBRUIKTE IK HIERVOOR IN DE TWEEDE LOOP:
"""
//                std::cout << typeid(rngf()).name() << "\n";
//                std::cout << i << "\t" << j << "\t" << k << "\n";
                // Using indexed access (slow, see test_random_grid.py):
                //out[i_stride*i + j_stride*j + k] = rngf();
                // Using breddels-iterator access (can't get other types working):
//                outpoint[i_stride*i + j_stride*j + k] = uni_dist(engine); // rngf();
"""

# The problem here seems to be mainly due to all the seeding every loop.

# Aha! Het grootste probleem bleek de uniform_real functie te zijn!
# Nu dus gewoon delen door het grootst mogelijke getal (+1 om een half open dist te krijgen)

array=c.resolution_independent_random_grid(16,0)

# STRANGE: the first inner loop in crunch::resolution_independent_random_grid is
# now 1.6 times faster than what you would have with the following code (which
# should be equivalent):
"""         for (j = 0; j < i; j++) {
                outpoint[i * i_stride + j * j_stride + k] = (double) ((unsigned long) engine()) / 0x100000000;
                outpoint[(gridsize - 1 - i) * i_stride + j * j_stride + k] = (double) ((unsigned long) engine()) / 0x100000000;
                outpoint[i * i_stride + (gridsize - 1 - j) * j_stride + k] = (double) ((unsigned long) engine()) / 0x100000000;
                outpoint[(gridsize - 1 - i) * i_stride + (gridsize - 1 - j) * j_stride + k] = (double) ((unsigned long) engine()) / 0x100000000;
                outpoint[j * i_stride + i * j_stride + k] = (double) ((unsigned long) engine()) / 0x100000000;
                outpoint[(gridsize - 1 - j) * i_stride + i * j_stride + k] = (double) ((unsigned long) engine()) / 0x100000000;
                outpoint[j * i_stride + (gridsize - 1 - i) * j_stride + k] = (double) ((unsigned long) engine()) / 0x100000000;
                outpoint[(gridsize - 1 - j) * i_stride + (gridsize - 1 - i) * j_stride + k] = (double) ((unsigned long) engine()) / 0x100000000;
			}
			outpoint[i * i_stride + i * j_stride + k] = (double) ((unsigned long) engine()) / 0x100000000;
			outpoint[(gridsize - 1 - i) * i_stride + i * j_stride + k] = (double) ((unsigned long) engine()) / 0x100000000;
			outpoint[i * i_stride + (gridsize - 1 - i) * j_stride + k] = (double) ((unsigned long) engine()) / 0x100000000;
			outpoint[(gridsize - 1 - i) * i_stride + (gridsize - 1 - i) * j_stride + k] = (double) ((unsigned long) engine()) / 0x100000000;
"""
