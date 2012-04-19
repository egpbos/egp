/* pyublas/numpy.hpp:
 * Defines pyublas::numpy_vector<type> */
#include <pyublas/numpy.hpp>
/* stdlib.h:
 * Numeric conversions, rand/srand (pseudo random number generators),
 * {m,re,c}alloc/free and process control (exit/abort/getenv/system). */
#include <stdlib.h>
/* iostream:
 * For std::cout <<, std::cin << functionality */
#include <iostream>
/* time.h:
 * For timing (time()) */
//#include <time.h>
/* list:
 * For list types like std::list<long> */
//#include <list>
/* vector:
 * Vector types, obviously; e.g. a vector of long lists:
 *     std::vector<std::list<long>>
 * Vectors are like arrays (performance-wise), but clean up after themselves. */
#include <vector>
/* math.h:
 * Mathematical functions like pow, exp, log, sin, etc.
 * N.B.: need to include this in the CMakeLists target_link_libraries (as m) */
//#include <math.h>
/* functional:
 * Needed for std::bind. */
#include <functional>

// For testing types of variables (typeid(var).name())
#include <typeinfo>

// For defaulting to indexed stuff
//#define BOOST_UBLAS_USE_ITERATING

//#define C11_RNG

// RANDOM STUFF
#ifdef C11_RNG
// C++11
#include <tr1/random>
#else
// BOOST (http://www.boost.org/libs/random/random-generators.html)
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>
#endif

#include "crunch.hpp"

using namespace crunch;

#ifdef C11_RNG
typedef std::tr1::mt19937 rng_type;
typedef std::tr1::uniform_int<> uniform_int;
typedef std::tr1::uniform_real<> uniform_real;
#else
typedef boost::mt19937 rng_type;
typedef boost::uniform_int<> uniform_int;
typedef boost::uniform_real<> uniform_real;
#endif

double_vector crunch::resolution_independent_random_grid(int gridsize, unsigned int seed) {
    rng_type engine; // only initialize an RNG once! http://tinyurl.com/cwbqqvg
    uniform_real uni_dist(0,1);
    int i, j, k;
    
    // Create the output array:
    double_vector out(gridsize * gridsize * (gridsize/2+1));
    double* outpoint = out.data().begin();
    int i_stride = gridsize*(gridsize/2+1);
    int j_stride = (gridsize/2+1);

    // Seed random number generator:
    engine.seed(seed); // don't seed too much! http://tinyurl.com/cwbqqvg
    
    // Fill out:
    for (i = 0; i < gridsize / 2; i++) {
        for (k = 0; k < i+1; k++) {
            for (j = 0; j < i; j++) {
                outpoint[i * i_stride + j * j_stride + k] = uni_dist(engine);
                outpoint[(gridsize - 1 - i) * i_stride + j * j_stride + k] = uni_dist(engine);
                outpoint[i * i_stride + (gridsize - 1 - j) * j_stride + k] = uni_dist(engine);
                outpoint[(gridsize - 1 - i) * i_stride + (gridsize - 1 - j) * j_stride + k] = uni_dist(engine);
                outpoint[j * i_stride + i * j_stride + k] = uni_dist(engine);
                outpoint[(gridsize - 1 - j) * i_stride + i * j_stride + k] = uni_dist(engine);
                outpoint[j * i_stride + (gridsize - 1 - i) * j_stride + k] = uni_dist(engine);
                outpoint[(gridsize - 1 - j) * i_stride + (gridsize - 1 - i) * j_stride + k] = uni_dist(engine);
			}
			outpoint[i * i_stride + i * j_stride + k] = uni_dist(engine);
			outpoint[(gridsize - 1 - i) * i_stride + i * j_stride + k] = uni_dist(engine);
			outpoint[i * i_stride + (gridsize - 1 - i) * j_stride + k] = uni_dist(engine);
			outpoint[(gridsize - 1 - i) * i_stride + (gridsize - 1 - i) * j_stride + k] = uni_dist(engine);
        }
        for (j = 0; j < i; j++) {
            for (k = 0; k < i; k++) {
                outpoint[j * i_stride + k * j_stride + i] = uni_dist(engine);
                outpoint[(gridsize - 1 - j) * i_stride + k * j_stride + i] = uni_dist(engine);
                outpoint[j * i_stride + (gridsize - 1 - k) * j_stride + i] = uni_dist(engine);
                outpoint[(gridsize - 1 - j) * i_stride + (gridsize - 1 - k) * j_stride + i] = uni_dist(engine);
            }
        }
    }
    
    // Fill out's nyquist plane:
    for (i = 0; i < gridsize; i++) {
        for (j = 0; j < gridsize; j++) {
            outpoint[i_stride*i + j_stride*j + gridsize/2] = uni_dist(engine); // rngf();
        }
    }
    
    const npy_intp dims[3] = {(npy_intp) gridsize, (npy_intp) gridsize, (npy_intp) gridsize/2+1};
    out.reshape(3, dims);
    return out;
}

// EXTRA TEST FUNCTIE VOOR VERGELIJKING MET numpy.random.random
double_vector crunch::naive_random_grid(int gridsize, unsigned int seed) {
    rng_type engine; // only initialize an RNG once! http://tinyurl.com/cwbqqvg
    uniform_real uni_dist(0,1);
    int i, j, k;
    
    // Create the output array:
    double_vector out(gridsize * gridsize * (gridsize/2+1));
    double* outpoint = out.data().begin();
    int i_stride = gridsize*(gridsize/2+1);
    int j_stride = (gridsize/2+1);

    // Seed random number generator:
    engine.seed(seed); // don't seed too much! http://tinyurl.com/cwbqqvg
    
    // Fill out:
    for (i = 0; i < gridsize; i++) {
        for (j = 0; j < gridsize; j++) {
			for (k = 0; k < gridsize/2+1; k++) {
            	outpoint[i_stride*i + j_stride*j + k] = uni_dist(engine);
			}
		}
    }
    
    const npy_intp dims[3] = {(npy_intp) gridsize, (npy_intp) gridsize, (npy_intp) gridsize/2+1};
    out.reshape(3, dims);
    return out;
}

// OLD VERSION (using seedtable as in N-GenIC -> slow!)
/* Based on:
 * http://en.wikipedia.org/wiki/C%2B%2B11#Extensible_random_number_facility
 * http://www.boost.org/doc/libs/1_40_0/libs/random/random_demo.cpp
 * and N-GenIC.
 * N.B.: gridsize must be an even number.*/
double_vector crunch::resolution_independent_random_grid_seedtable(int gridsize, unsigned int seed) {
    rng_type engine; // only initialize an RNG once! http://tinyurl.com/cwbqqvg
    uniform_real uni_dist(0,1);
    uniform_int uni_int(0,0x7fffffff);
    std::vector<unsigned int> seedtable(gridsize * gridsize);
    int i, j, k;
    
    // Seed for the random integers:
    engine.seed(seed); // don't seed too much! http://tinyurl.com/cwbqqvg
    auto rngi = std::bind(uni_int, engine);
    
    // Filling seedtable (cell 0 == "lower left" and count left-right, down-up):
    for (i = 0; i < gridsize / 2; i++) {
        for (j = 0; j < i; j++) /* i==0 -> nothing
                                * i!=0 -> the edge above cell 0 */
            seedtable[i * gridsize + j] = rngi();
        for (j = 0; j < i + 1; j++) /* i==0 -> cell 0
                                    * i!=0 -> the edge to the right of cell 0 */
            seedtable[j * gridsize + i] = rngi();
        for (j = 0; j < i; j++) /* i==0 -> nothing
                                * i!=0 -> the edge below cell y */
            seedtable[(gridsize - 1 - i) * gridsize + j] = rngi();
        for (j = 0; j < i + 1; j++) /* i==0 -> cell y ("top left")
                                    * i!=0 -> the edge to the right of cell y */
            seedtable[(gridsize - 1 - j) * gridsize + i] = rngi();
        for (j = 0; j < i; j++) /* i==0 -> nothing
                                * i!=0 -> the edge above cell x*/
            seedtable[i * gridsize + (gridsize - 1 - j)] = rngi();
        for (j = 0; j < i + 1; j++) /* i==0 -> cell x ("bottom right")
                                    * i!=0 -> the edge to the left of cell x */
            seedtable[j * gridsize + (gridsize - 1 - i)] = rngi();
        for (j = 0; j < i; j++) /* i==0 -> nothing
                                * i!=0 -> the edge below cell xy */
            seedtable[(gridsize - 1 - i) * gridsize + (gridsize - 1 - j)] = rngi();
        for (j = 0; j < i + 1; j++) /* i==0 -> cell xy ("top right")
                                    * i!=0 -> the edge to the left of cell xy */
            seedtable[(gridsize - 1 - j) * gridsize + (gridsize - 1 - i)] = rngi();
    }
    
    // Create the output array:
    double_vector out(gridsize * gridsize * (gridsize/2+1));
    double* outpoint = out.data().begin();
    int i_stride = gridsize*(gridsize/2+1);
    int j_stride = (gridsize/2+1);
    
    for (i = 0; i < gridsize; i++) {
        for (j = 0; j < gridsize; j++) {
            // Reseed using seedtable:
            engine.seed(seedtable[i*gridsize + j]); // too much seeding really..
//            auto rngf = std::bind(uni_dist, engine);
            for (k = 0; k < gridsize/2+1; k++) {
//                std::cout << typeid(rngf()).name() << "\n";
//                std::cout << i << "\t" << j << "\t" << k << "\n";
                // Using indexed access (slow, see python tests below):
                //out[i_stride*i + j_stride*j + k] = rngf();
                // Using breddels-iterator access (can't get other types working):
                outpoint[i_stride*i + j_stride*j + k] = uni_dist(engine); // rngf();
            }
        }
    }
    
    const npy_intp dims[3] = {(npy_intp) gridsize, (npy_intp) gridsize, (npy_intp) gridsize/2+1};
    out.reshape(3, dims);
//    std::cout << "Done here.\n";
    return out;
}

// Python tests of indexed versus iterator access:
// from timeit import Timer
// t1 = Timer("array=c.resolution_independent_random_grid(16,0)", "import pyublas; import egp.crunch as c")
// t2 = Timer("array=np.random.random((16,16,9))", "import numpy as np")
// t1.timeit(10000)
// Out[108]: 40.59319806098938 # FOR INDEXED ACCESS
// Out[4]: 32.884864091873169 # FOR ITERATOR ACCESS => almost no difference!
// t2.timeit(10000)
// Out[5]: 0.35071587562561035 # So yeah, this is the goal...
// The problem here seems to be mainly due to all the seeding every loop.
