#!/bin/bash
module load openmpi-x86_64

cd /Users/users/pbos/code/egpTesting/icgen/random_verschillende_resoluties_N-GenIC
mpiexec -np 2 gadget3_64/P-Gadget3_64 run0_64.par
