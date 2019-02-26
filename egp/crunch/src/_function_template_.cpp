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
#include "crunch.hpp"

using namespace crunch;

void crunch::fillrho_template(int_vector k, float_vector w, float_vector rho, int Npart, float mass) {
    int ksize1 = k.strides()[0]/k.itemsize();
    int ksize2 = k.strides()[1]/k.itemsize();
    int ksize3 = k.strides()[2]/k.itemsize();
    int* kptr = k.data().begin();
    
    int wsize1 = w.strides()[0]/w.itemsize();
    int wsize2 = w.strides()[1]/w.itemsize();
    int wsize3 = w.strides()[2]/w.itemsize();
    float* wptr = w.data().begin();
    
    int rhosize1 = rho.strides()[0]/rho.itemsize();
    int rhosize2 = rho.strides()[1]/rho.itemsize();
    int rhosize3 = rho.strides()[2]/rho.itemsize();
    float* rhoptr = rho.data().begin();
    
    for(int j = 0; j < Npart; j++)
        for(int i3 = 0; i3 < 3; i3++)
            for(int i2 = 0; i2 < 3; i2++)
                for(int i1 = 0; i1 < 3; i1++) {
                    //rho.sub(k.sub(i1,j,0), k.sub(i2,j,1), k.sub(i3,j,2)) += mass*w.sub(i1,j,0)*w.sub(i2,j,1)*w.sub(i3,j,2);
                    //rho.sub(kptr[i1*ksize1 + j*ksize2 + 0*ksize3] , kptr[i2*ksize1 + j*ksize2 + 1*ksize3], kptr[i3*ksize1 + j*ksize2 + 2*ksize3]) += mass*w.sub(i1,j,0)*w.sub(i2,j,1)*w.sub(i3,j,2);
                    rhoptr[kptr[i1*ksize1 + j*ksize2 + 0*ksize3]*rhosize1 + kptr[i2*ksize1 + j*ksize2 + 1*ksize3]*rhosize2 + kptr[i3*ksize1 + j*ksize2 + 2*ksize3]*rhosize3] += mass*wptr[i1*wsize1 + j*wsize2 + 0*wsize3] * wptr[i2*wsize1 + j*wsize2 + 1*wsize3] * wptr[i3*wsize1 + j*wsize2 + 2*wsize3];
                    
                }
}

