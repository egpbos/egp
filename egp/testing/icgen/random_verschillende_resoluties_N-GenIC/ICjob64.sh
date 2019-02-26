#!/bin/tcsh
#$ -j n
#$ -cwd
#$ -pe mvapich2 4
#$ -m be
#$ -M volker@mpa-garching.mpg.de
#$ -N ics
#

#module load mvapich2-1.2-sdr-gnu/4.1.2
# EGP:
module load openmpi-x86_64
mpiexec -np 1 N-GenIC/N-GenIC ic64.param
#N-GenIC/N-GenIC ic64.param

 
