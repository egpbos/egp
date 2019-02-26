#include <pyublas/numpy.hpp>
#include <iostream>
#include <list>
#include <vector>
#include <time.h>
#include <stdlib.h>
#include "crunch.hpp"

using namespace crunch;

/*
Don't give twopcf an rmax larger than 0.5*boxsize!
This is a non-monte carlo implementation; the Rivolo estimator, which uses
simple volume division instead of comparing to a uniform random sample.
This means that this method can only be used for periodic volumes.
The C++ part will not normalize, i.e. you still need to divide by the
volume of the shell at r, etc.
*/
void crunch::twopcf(float_vector pos, int_vector Rcount, float dr, long Npart, float boxsize, float rmin, float rmax) {
    if (rmax > boxsize/2) {
        printf("rmax is larger than boxsize/2!");
        return;
    }
    long possize1 = pos.strides()[0]/pos.itemsize();
    long possize2 = pos.strides()[1]/pos.itemsize();
    long possize2x2 = 2*possize2;
    float* posptr = pos.data().begin();
    
    long Rcountsize1 = Rcount.strides()[0]/Rcount.itemsize();
    int* Rcountptr = Rcount.data().begin();
    
//		std::cout << "vars and constants..\n";
//		long binssqr = bins*bins;
    
    // The diagonal of a gridcell must be smaller than rmin
    float gridrib = rmin/1.7320508075688772;
    
    long bins = boxsize/gridrib + 1; // makes boxes slightly smaller than gridrib
    float cellsize = boxsize/bins;
    long binssqr = bins*bins;
    long binscubed = binssqr*bins;
    
    std::vector<long_list> grid (binscubed);
    
    long_list cell, searchcell;
    float x, y, z, rsq;
    long index, ic, isc, ir;
    time_t tb, te;
    long_list::iterator lic, lisc;
    
    gridSearchOrder(pos, Npart, cellsize, bins, grid);
    
    long searchbins = bins*rmax/boxsize + 1;
    long sbinssqr = searchbins*searchbins;
//		long sbinscubed = sbinssqr*searchbins;
    // We will only be needing searchbins^3 - 1, so we start one index
    // later when looping over the searchcells vector.
//		long searchbinsrib = searchbins*2 + 1;
//		std::vector<long> searchcells (searchbinsrib*searchbinsrib*searchbinsrib);
    
    for (long g1=0; g1<bins; g1++)
//			std::cout << g1 << '\n';
    for (long g2=0; g2<bins; g2++)
    for (long g3=0; g3<bins; g3++) {
//			std::cout << g1 << '\t' << g2 << '\t' << g3 << '\n';
        cell.swap(grid[g1 + bins*g2 + binssqr*g3]);
//			std::cout << cell.size() << '\n';
        if (!cell.empty()) {
            for (long sg1=-searchbins; sg1<=searchbins; sg1++)
            for (long sg2=-searchbins; sg2<=searchbins; sg2++)
            for (long sg3=-searchbins; sg3<=searchbins; sg3++) {
//					printf("in de loep\n");
                index = (g1 + sg1 + bins)%bins + ((g2 + sg2 + bins)%bins)*bins + ((g3 + sg3 + bins)%bins)*binssqr;
//					std::cout << "index: " << index << '\t' << "cell size: ";
                searchcell = grid[index];
//					std::cout << searchcell.size() << '\n';
            
                for (lisc = searchcell.begin(); lisc != searchcell.end(); lisc++)
                for (lic = cell.begin(); lic != cell.end(); lic++) {
//						std::cout << "index: " << index << '\t' << "cell size: ";
//						std::cout << searchcell.size() << '\n';
                    ic = *lic;
                    isc = *lisc;
                    x = posptr[ic*possize1] - posptr[isc*possize1];
                    x = x-round(x/boxsize)*boxsize; // periodic
                    y = posptr[ic*possize1 + possize2] - posptr[isc*possize1 + possize2];
                    y = y-round(y/boxsize)*boxsize; // periodic
                    z = posptr[ic*possize1 + possize2x2] - posptr[isc*possize1 + possize2x2];
                    z = z-round(z/boxsize)*boxsize; // periodic
                    ir = sqrt(x*x + y*y + z*z)/dr;
//						std::cout << "r = " << sqrt(x*x + y*y + z*z) << '\t' << "ir = " << ir << '\n';
                    // Take care when initializing the Rcount array in Python, don't make it too short!
                    (Rcountptr[ir*Rcountsize1])++;
                }
            }
//				printf("clearing cell...");
            cell.clear();
//				printf("... cleared\n");
        }
    }
}

/*
This function, twopcfnull, calculates the two-point-correlation function
from r=0 up to r=rmax. Do not use this to calculate the entire range you
want! This function works best for rather small rmax (~ 1% of the boxsize).
For a larger range of radii, use this function first and then extend your
range with the twopcf function that takes an rmin as well.
*/
void crunch::twopcfnull(float_vector pos, int_vector Rcount, float dr, long Npart, float boxsize, float rmax) {
    if (rmax > boxsize/50) {
        printf("Warning: rmax is larger than 2% of boxsize; for large radii, use twopcf!");
    }
    long possize1 = pos.strides()[0]/pos.itemsize();
    long possize2 = pos.strides()[1]/pos.itemsize();
    long possize2x2 = 2*possize2;
    float* posptr = pos.data().begin();
    
    long Rcountsize1 = Rcount.strides()[0]/Rcount.itemsize();
    int* Rcountptr = Rcount.data().begin();
    
    long bins = boxsize/rmax; // makes cells slightly larger than rmax
    float cellsize = boxsize/bins;
    long binssqr = bins*bins;
    long binscubed = binssqr*bins;
    
    std::vector<long_list> grid (binscubed);
    
    long_list cell, searchcell;
    float x, y, z, rsq;
    long index, ic, isc, ir;
    time_t tb, te;
    long_list::iterator lic, lisc;
    
    gridSearchOrder(pos, Npart, cellsize, bins, grid);
    
    for (long g1=0; g1<bins; g1++)
    for (long g2=0; g2<bins; g2++)
    for (long g3=0; g3<bins; g3++) {
        cell.swap(grid[g1 + bins*g2 + binssqr*g3]);
        if (!cell.empty()) {
            for (long sg1=-1; sg1<=1; sg1++)
            for (long sg2=-1; sg2<=1; sg2++)
            for (long sg3=-1; sg3<=1; sg3++) {
                index = (g1 + sg1 + bins)%bins + ((g2 + sg2 + bins)%bins)*bins + ((g3 + sg3 + bins)%bins)*binssqr;
                searchcell = grid[index];
                
                for (lisc = searchcell.begin(); lisc != searchcell.end(); lisc++)
                for (lic = cell.begin(); lic != cell.end(); lic++) {
                    ic = *lic;
                    isc = *lisc;
                    x = posptr[ic*possize1] - posptr[isc*possize1];
                    x = x-round(x/boxsize)*boxsize; // periodic
                    y = posptr[ic*possize1 + possize2] - posptr[isc*possize1 + possize2];
                    y = y-round(y/boxsize)*boxsize; // periodic
                    z = posptr[ic*possize1 + possize2x2] - posptr[isc*possize1 + possize2x2];
                    z = z-round(z/boxsize)*boxsize; // periodic
                    ir = sqrt(x*x + y*y + z*z)/dr;
                    // Take care when initializing the Rcount array in Python, don't make it too short!
                    (Rcountptr[ir*Rcountsize1])++;
                }
            }
            lic = cell.begin();
            while (lic != cell.end()) {
                ic = *lic;
                lic = cell.erase(lic);
                for (lisc = cell.begin(); lisc != cell.end(); lisc++) {
                    isc = *lisc;
                    x = posptr[ic*possize1] - posptr[isc*possize1];
                    x = x-round(x/boxsize)*boxsize; // periodic
                    y = posptr[ic*possize1 + possize2] - posptr[isc*possize1 + possize2];
                    y = y-round(y/boxsize)*boxsize; // periodic
                    z = posptr[ic*possize1 + possize2x2] - posptr[isc*possize1 + possize2x2];
                    z = z-round(z/boxsize)*boxsize; // periodic
                    ir = sqrt(x*x + y*y + z*z)/dr;
                    // Take care when initializing the Rcount array in Python, don't make it too short!
                    (Rcountptr[ir*Rcountsize1])++;
                }
            }
            cell.clear();
        }
    }
}
