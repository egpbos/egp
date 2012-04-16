#!/usr/bin/env python
# encoding: utf-8
"""
egp package, main initialization script
__init__.py

Created by Evert Gerardus Patrick Bos.
Copyright (c) January 2012. All rights reserved.
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

