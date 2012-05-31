#!/bin/bash
module load openmpi-x86_64

cd /Users/users/pbos/code/egpTesting/icgen/random_verschillende_resoluties_N-GenIC
mpiexec -np 8 gadget3_128/P-Gadget3_128 run0_128.par
