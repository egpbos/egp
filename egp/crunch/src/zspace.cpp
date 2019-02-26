#include <pyublas/numpy.hpp>
#include <math.h>
#include "crunch.hpp"

using namespace crunch;

void crunch::redshiftspace(double_vector pos, double_vector vel, double_vector posZ, int Npart, double_vector origin, int centerOrigin, double boxsize, double H) {
    long ps1 = pos.strides()[0]/pos.itemsize();
    long ps2 = pos.strides()[1]/pos.itemsize();
    long ps2x2 = 2*ps2;
    double* pp = pos.data().begin();
    
    long vs1 = vel.strides()[0]/vel.itemsize();
    long vs2 = vel.strides()[1]/vel.itemsize();
    long vs2x2 = 2*vs2;
    double* vp = vel.data().begin();
    
    double* op = origin.data().begin();
    
    long zs1 = posZ.strides()[0]/posZ.itemsize();
    long zs2 = posZ.strides()[1]/posZ.itemsize();
    long zs2x2 = 2*zs2;
    double* zp = posZ.data().begin();
    
    double x, y, z, vx, vy, vz; // Base coordinates
    double xy2, r, ph, th, beta, redshift; // Derived
    double c = 3.0e5; // Speed of light
    double halfbox = boxsize/2;
    
    for(long j = 0; j < Npart; j++) {
        // put origin at (0,0,0)
        x = pp[j*ps1] - op[0];
        y = pp[j*ps1 + ps2] - op[1];
        z = pp[j*ps1 + ps2x2] - op[2];
        
        // center (if wanted)
        if (centerOrigin) {
            if (x < -halfbox) x = x + boxsize;
            if (x >= halfbox) x = x - boxsize;
            if (y < -halfbox) y = y + boxsize;
            if (y >= halfbox) y = y - boxsize;
            if (z < -halfbox) z = z + boxsize;
            if (z >= halfbox) z = z - boxsize;
        }
        
        vx = vp[j*vs1];
        vy = vp[j*vs1 + vs2];
        vz = vp[j*vs1 + vs2x2];
        
        xy2 = x*x + y*y;
        r = sqrt(xy2 + z*z);
        ph = atan2(y,x);
        th = atan2(sqrt(xy2),z);
        
        beta = (vx*x/r + vy*y/r + vz*z/r)/c; // v_rad / c
        
        // Redshift = (1 + relativistic rs) * (1 + cosmological (Hubble) rs) - 1
        redshift = sqrt((1+beta)/(1-beta)) * (1 + r/1000*H/c) - 1;
        
        // New redshift-space radial distance
        r = 1000*redshift*c/H;
        
        if (centerOrigin) {
            zp[j*zs1] = r*sin(th)*cos(ph) + halfbox;
            zp[j*zs1 + zs2] = r*sin(th)*sin(ph) + halfbox;
            zp[j*zs1 + zs2x2] = r*cos(th) + halfbox;
        } else {
            zp[j*zs1] = r*sin(th)*cos(ph) + op[0];
            zp[j*zs1 + zs2] = r*sin(th)*sin(ph) + op[1];
            zp[j*zs1 + zs2x2] = r*cos(th) + op[2];
        }
    }
}
