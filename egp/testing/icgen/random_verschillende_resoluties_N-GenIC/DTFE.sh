#!/usr/bin/env bash

#export OMP_NUM_THREADS=4

DTFE run0_64/snap_007 DTFE/run0_64_007_rho256 -g 256 -p --partition 2 1 1
DTFE run0_128/snap_007 DTFE/run0_128_007_rho256 -g 256 -p --partition 2 1 1

DTFE run0_64/snap_002 DTFE/run0_64_002_rho256 -g 256 -p --partition 2 1 1
DTFE run0_128/snap_002 DTFE/run0_128_002_rho256 -g 256 -p --partition 2 1 1

