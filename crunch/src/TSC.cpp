#include <pyublas/numpy.hpp>
#include "crunch.hpp"

using namespace crunch;

void crunch::TSCDensity(double_vector pos, double_vector rho, int Npart, double boxsize, int gridsize, double mass) {
    long possize1 = pos.strides()[0]/pos.itemsize();
    long possize2 = pos.strides()[1]/pos.itemsize();
    long possize2x2 = 2*possize2;
    double* posptr = pos.data().begin();
    
    long rhosize1 = rho.strides()[0]/rho.itemsize();
    long rhosize2 = rho.strides()[1]/rho.itemsize();
    long rhosize3 = rho.strides()[2]/rho.itemsize();
    double* rhoptr = rho.data().begin();
    
    double x, y, z, dxp, dyp, dzp, dxm, dym, dzm;
    long ix, iy, iz, kx1, kx2, kx3, ky1, ky2, ky3, kz1, kz2, kz3, index;
    double wx1, wx2, wx3, wy1, wy2, wy3, wz1, wz2, wz3;
    
    double dGrid = boxsize/gridsize;
    
    for(long j = 0; j < Npart; j++) {
        x = posptr[j*possize1] / dGrid;
        y = posptr[j*possize1 + possize2] / dGrid;
        z = posptr[j*possize1 + possize2x2] / dGrid;
        
        //~ ix = x + 0.5; // N.B.: this is wrong! The 0.5 should be added as float.
        //~ iy = y + 0.5;
        //~ iz = z + 0.5;
        //~ 
        //~ x = x - ix; // positions themselves are no longer needed
        //~ y = y - iy;
        //~ z = z - iz;
        ix = x;
        iy = y;
        iz = z;
        
        x = x - (ix + 0.5); // distance from center of cell
        y = y - (iy + 0.5);
        z = z - (iz + 0.5);
        
        dxp = (1.5 - fabs(1.0 + x));
        dyp = (1.5 - fabs(1.0 + y));
        dzp = (1.5 - fabs(1.0 + z));
        dxm = (1.5 - fabs(1.0 - x));
        dym = (1.5 - fabs(1.0 - y));
        dzm = (1.5 - fabs(1.0 - z));
        
        ix = ix % gridsize;
        iy = iy % gridsize;
        iz = iz % gridsize;
        
        kx1 = (ix-1 + gridsize) % gridsize;
        ky1 = (iy-1 + gridsize) % gridsize;
        kz1 = (iz-1 + gridsize) % gridsize;
        kx2 = ix;
        ky2 = iy;
        kz2 = iz;
        kx3 = (ix+1) % gridsize;
        ky3 = (iy+1) % gridsize;
        kz3 = (iz+1) % gridsize;
        
        wx1 = 0.5*dxp*dxp;
        wy1 = 0.5*dyp*dyp;
        wz1 = 0.5*dzp*dzp;
        wx2 = 0.75 - x*x;
        wy2 = 0.75 - y*y;
        wz2 = 0.75 - z*z;
        wx3 = 0.5*dxm*dxm;
        wy3 = 0.5*dym*dym;
        wz3 = 0.5*dzm*dzm;
        
        rhoptr[kx1*rhosize1 + ky1*rhosize2 + kz1*rhosize3] += mass*wx1*wy1*wz1;
        rhoptr[kx2*rhosize1 + ky1*rhosize2 + kz1*rhosize3] += mass*wx2*wy1*wz1;
        rhoptr[kx3*rhosize1 + ky1*rhosize2 + kz1*rhosize3] += mass*wx3*wy1*wz1;
        rhoptr[kx1*rhosize1 + ky2*rhosize2 + kz1*rhosize3] += mass*wx1*wy2*wz1;
        rhoptr[kx2*rhosize1 + ky2*rhosize2 + kz1*rhosize3] += mass*wx2*wy2*wz1;
        rhoptr[kx3*rhosize1 + ky2*rhosize2 + kz1*rhosize3] += mass*wx3*wy2*wz1;
        rhoptr[kx1*rhosize1 + ky3*rhosize2 + kz1*rhosize3] += mass*wx1*wy3*wz1;
        rhoptr[kx2*rhosize1 + ky3*rhosize2 + kz1*rhosize3] += mass*wx2*wy3*wz1;
        rhoptr[kx3*rhosize1 + ky3*rhosize2 + kz1*rhosize3] += mass*wx3*wy3*wz1;
        rhoptr[kx1*rhosize1 + ky1*rhosize2 + kz2*rhosize3] += mass*wx1*wy1*wz2;
        rhoptr[kx2*rhosize1 + ky1*rhosize2 + kz2*rhosize3] += mass*wx2*wy1*wz2;
        rhoptr[kx3*rhosize1 + ky1*rhosize2 + kz2*rhosize3] += mass*wx3*wy1*wz2;
        rhoptr[kx1*rhosize1 + ky2*rhosize2 + kz2*rhosize3] += mass*wx1*wy2*wz2;
        rhoptr[kx2*rhosize1 + ky2*rhosize2 + kz2*rhosize3] += mass*wx2*wy2*wz2;
        rhoptr[kx3*rhosize1 + ky2*rhosize2 + kz2*rhosize3] += mass*wx3*wy2*wz2;
        rhoptr[kx1*rhosize1 + ky3*rhosize2 + kz2*rhosize3] += mass*wx1*wy3*wz2;
        rhoptr[kx2*rhosize1 + ky3*rhosize2 + kz2*rhosize3] += mass*wx2*wy3*wz2;
        rhoptr[kx3*rhosize1 + ky3*rhosize2 + kz2*rhosize3] += mass*wx3*wy3*wz2;
        rhoptr[kx1*rhosize1 + ky1*rhosize2 + kz3*rhosize3] += mass*wx1*wy1*wz3;
        rhoptr[kx2*rhosize1 + ky1*rhosize2 + kz3*rhosize3] += mass*wx2*wy1*wz3;
        rhoptr[kx3*rhosize1 + ky1*rhosize2 + kz3*rhosize3] += mass*wx3*wy1*wz3;
        rhoptr[kx1*rhosize1 + ky2*rhosize2 + kz3*rhosize3] += mass*wx1*wy2*wz3;
        rhoptr[kx2*rhosize1 + ky2*rhosize2 + kz3*rhosize3] += mass*wx2*wy2*wz3;
        rhoptr[kx3*rhosize1 + ky2*rhosize2 + kz3*rhosize3] += mass*wx3*wy2*wz3;
        rhoptr[kx1*rhosize1 + ky3*rhosize2 + kz3*rhosize3] += mass*wx1*wy3*wz3;
        rhoptr[kx2*rhosize1 + ky3*rhosize2 + kz3*rhosize3] += mass*wx2*wy3*wz3;
        rhoptr[kx3*rhosize1 + ky3*rhosize2 + kz3*rhosize3] += mass*wx3*wy3*wz3;
    }
}

void crunch::fillrho(int_vector k, float_vector w, float_vector rho, int Npart, float mass) {
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
