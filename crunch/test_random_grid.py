#!/usr/bin/env python
import pyublas, egp.crunch as c
from timeit import Timer

t1 = Timer("array=c.resolution_independent_random_grid(16,0)", "import pyublas; import egp.crunch as c")
t1.timeit(1000)

array=c.resolution_independent_random_grid(16,0)