#pragma once /* Dit is om ervoor te zorgen dat crunch.hpp maar 1 keer wordt
              * geinclude in de module; dus over de verschillende .cpp files. */

// Boost Includes ==============================================================
#include <boost/python.hpp>
#include <boost/python/numeric.hpp>
#include <boost/cstdint.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
// PyUblas
#include <pyublas/numpy.hpp>
// List
#include <list>

namespace crunch {
	typedef pyublas::numpy_vector<int> int_vector;
	typedef pyublas::numpy_vector<float> float_vector;
	typedef pyublas::numpy_vector<double> double_vector;
	typedef std::complex<double> complexdouble;
	typedef pyublas::numpy_vector<complexdouble> complex_vector;
	typedef std::list<long> long_list;
    
//	Gaussian filter; apply on Fourier space components. hk is the output vector
    extern void gaussian_filter(complex_vector rhok, complex_vector hk, int n, float dk, float akmax, float smooth);
//  TSC: Density estimator using Triangular Shaped Cloud algorithm
    extern void TSCDensity(double_vector pos, double_vector rho, int Npart, double boxsize, int gridsize, double mass);
    extern void fillrho(int_vector k, float_vector w, float_vector rho, int Npart, float mass);
//	zspace: transformations to and from redshift space
    extern void redshiftspace(double_vector pos, double_vector vel, double_vector posZ, int Npart, double_vector origin, int centerOrigin, double boxsize, double H);
//  gridsearch: functions for the grid searching method for statistics
    extern void gridSearch(float_vector pos, long Npart, float_vector ball, long Nball, int_vector count, float cellsize, int bins, float radius);
    extern void gridSearchOrder(float_vector pos, long Npart, float cellsize, int bins, std::vector<long_list> &grid);
//	twopcf: two point correlation function calculation, using gridsearch
    extern void twopcf(float_vector pos, int_vector Rcount, float dr, long Npart, float boxsize, float rmin, float rmax);
    extern void twopcfnull(float_vector pos, int_vector Rcount, float dr, long Npart, float boxsize, float rmax);
    // Octreesearch
    extern int_vector octreeSectorSearch(float_vector pos, long Npart, int_vector sectors, long Nsec, int bins, float cellsize);
    // Testspul
    extern int_vector test3(int n);
    extern void test2(float_vector v);
    
    // Resolution independent random grid (for gaussian random field generation)
    extern void resolution_independent_random_grid(long gridsize, unsigned int seed);
}
