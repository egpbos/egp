#!/usr/bin/env python
# encoding: utf-8
"""
egp package, main initialization script
__init__.py

Created by Evert Gerardus Patrick Bos.
Copyright (c) January 2012. All rights reserved.

Conventions:
- The values in Fields correspond to physical coordinates. The boxlen size spans the full cells on both ends, i.e. it begins at the far end of the [0,0,0] cell and ends at the far end of the [gridsize,gridsize,gridsize] cell. Each cell value corresponds to the coordinates at the center of the cell. This means that the coordinates of cell [0,0,0] are [dx/2,dx/2,dx/2] where dx is the linear size of one cell.
"""

__all__ = ['icgen']

# Mathematical
def skew(a):
	m1 = a.mean()
	m2 = a.std()
	m3 = 0.0
	for i in range(len(a)):
		m3 += (a[i]-m1)**3.0
	return m3/m2**3/len(a)

def kurtosis(a):
	m1 = a.mean()
	m2 = a.std()
	m4 = 0.0
	for i in range(len(a)):
		m4 += (a[i]-m1)**4.0
	return m4/m2**4/len(a) - 3.0

