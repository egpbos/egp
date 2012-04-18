/* pyublas/numpy.hpp:
 * Defines pyublas::numpy_vector<type> */
#include <pyublas/numpy.hpp>
/* math.h:
 * Mathematical functions like pow, exp, log, sin, etc.
 * N.B.: need to include this in the CMakeLists target_link_libraries (as m) */
#include <math.h>
#include "crunch.hpp"

using namespace crunch;

//	Gaussian filter; apply on Fourier space components. hk is the output vector
void crunch::gaussian_filter(complex_vector rhok, complex_vector hk, int n, float dk, float akmax, float smooth) {
    long rhoksize1 = rhok.strides()[0]/rhok.itemsize();
    long rhoksize2 = rhok.strides()[1]/rhok.itemsize();
    long rhoksize3 = rhok.strides()[2]/rhok.itemsize();
    complexdouble* rhokptr = rhok.data().begin();
    
    long hksize1 = hk.strides()[0]/hk.itemsize();
    long hksize2 = hk.strides()[1]/hk.itemsize();
    long hksize3 = hk.strides()[2]/hk.itemsize();
    complexdouble* hkptr = hk.data().begin();
    
    float ak1, ak2, ak3, aksq;
    complexdouble f;
    
    float ssq = smooth*smooth;
    float pi = 3.1415926535897931;
    int n2 = n/2;
    
    for (int k1 = 0; k1 < n; k1++) {
        ak1 = k1*dk;
        if (k1 > n2 - 1) ak1 = ak1 - akmax;
        for (int k2 = 0; k2 < n; k2++) {
            ak2 = k2*dk;
            if (k2 > n2 - 1) ak2 = ak2 - akmax;
            for (int k3 = 0; k3 < n2+1; k3++) {
                ak3 = k3*dk;
                if (k3 > n2 - 1) ak3 = ak3 - akmax;
                aksq = ak1*ak1 + ak2*ak2 + ak3*ak3;
        
                f = rhokptr[k1*rhoksize1 + k2*rhoksize2 + k3*rhoksize3]/pow(2*pi,1.5)*exp(-0.5*aksq*ssq);
        
                hkptr[k1*hksize1 + k2*hksize2 + k3*hksize3] = f;
            }
        }
    }
}
