/* pyublas/numpy.hpp:
 * Defines pyublas::numpy_vector<type> */
#include <pyublas/numpy.hpp>
/* stdlib.h:
 * Numeric conversions, rand/srand (pseudo random number generators),
 * {m,re,c}alloc/free and process control (exit/abort/getenv/system). */
//#include <stdlib.h>
/* iostream:
 * For std::cout <<, std::cin << functionality */
//#include <iostream>
/* time.h:
 * For timing (time()) */
//#include <time.h>
/* list:
 * For list types like std::list<long> */
//#include <list>
/* vector:
 * Vector types, obviously; e.g. a vector of long lists:
 *     std::vector<std::list<long>>  */
//#include <vector>
/* math.h:
 * Mathematical functions like pow, exp, log, sin, etc.
 * N.B.: need to include this in the CMakeLists target_link_libraries (as m) */
//#include <math.h>
/* functional:
 * Needed for std::bind. */
#include <functional>

//#define C11_RNG

// RANDOM STUFF
#ifdef C11_RNG
// C++11
#include <random>
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
typedef std::mt19937 rng_type;
typedef std::uniform_real_distribution<> uniform_real;
#else
typedef boost::mt19937 rng_type;
typedef boost::uniform_real<> uniform_real;
#endif

/* Based on:
 * http://en.wikipedia.org/wiki/C%2B%2B11#Extensible_random_number_facility
 * http://www.boost.org/doc/libs/1_40_0/libs/random/random_demo.cpp */
void crunch::resolution_independent_random_grid(long gridsize, unsigned int seed) {
    rng_type engine(); // only initialize an RNG once! http://tinyurl.com/cwbqqvg
    engine.seed(seed); // don't seed too much either (same reference as above)
    uniform_real uni_dist(0,1);
    
    auto rng = std::bind(uni_dist, engine); // Johan denkt dat dit gaat werken!
                                            // en dus hetzelfde doet als hieronder
    //boost::variate_generator<rng_type&, uniform_real<> > rng(engine, uni_dist);
    printf("%f %f %f %f", rng(), rng(), rng(), rng());
}

