#include <pyublas/numpy.hpp>
#include <iostream>
#include <list>
#include <vector>
#include <time.h>
#include <stdlib.h>
#include "crunch.hpp"

using namespace crunch;

void crunch::gridSearchOrder(float_vector pos, long Npart, float cellsize, int bins, std::vector<long_list> &grid) {
//  std::cout << "pos pointers..\n";
    long possize1 = pos.strides()[0]/pos.itemsize();
    long possize2 = pos.strides()[1]/pos.itemsize();
    long possize2x2 = 2*possize2;
    float* posptr = pos.data().begin();
    
//  std::cout << "vars and constants..\n";
    long binssqr = bins*bins;
    
    long ix, iy, iz;
    time_t tb, te;
//  long_list cell;
    
//  printf("Ordering particles... ");
//  tb = time(NULL);
    for(long j = 0; j < Npart; j++) {
//      std::cout << j << ' ';
        ix = posptr[j*possize1] / cellsize;
        iy = posptr[j*possize1 + possize2] / cellsize;
        iz = posptr[j*possize1 + possize2x2] / cellsize;
        
//      std::cout << ix << '\t' << iy << '\t' << iz << '\n';
        
        (grid[ix + bins*iy + binssqr*iz]).push_back(j);
//      if (j >= 49699000)
//          std::cout << j << ' ' << ix << '\t' << iy << '\t' << iz << '\n';
//          std::cout << (grid[ix + bins*iy + binssqr*iz]).back();

    }
//  te = time(NULL);
//  printf("... done ordering (took %ld seconds).\n", te-tb);
}

void crunch::gridSearch(float_vector pos, long Npart, float_vector ball, long Nball, int_vector count, float cellsize, int bins, float radius) {
//  std::cout << "Pos pointers\n";
    long possize1 = pos.strides()[0]/pos.itemsize();
    long possize2 = pos.strides()[1]/pos.itemsize();
    long possize2x2 = 2*possize2;
    float* posptr = pos.data().begin();
    
//  std::cout << "Ball pointers\n";
    long ballsize1 = ball.strides()[0]/ball.itemsize();
    long ballsize2 = ball.strides()[1]/ball.itemsize();
    long ballsize2x2 = 2*ballsize2;
    float* ballptr = ball.data().begin();
    
//  std::cout << "Count pointers\n";
    long countsize1 = count.strides()[0]/count.itemsize();
    int* countptr = count.data().begin();
    
//  std::cout << "Init constants\n";
    long binssqr = bins*bins;
    long binscubed = binssqr*bins;
//  std::cout << binscubed;
    float r2 = radius*radius;
    
//  std::cout << "Init grid list\n";
    std::vector<long_list> grid (binscubed);
//  long_list* grid = (long_list*) malloc(binscubed * sizeof(long_list));
    // VUL GRID MET LEGE LIJSTEN
//  for (long index = 0; index < binscubed; index++)
//      grid[index] = new long_list();
    
//  std::cout << "Init cell list\n";
    long_list cell;
    
//  std::cout << "Init vars\n";
    float x, y, z, xb, yb, zb;
    long in, kx, ky, kz, ixb, iyb, izb;
    time_t tb, te;
    long_list::iterator i;
    
//  std::cout << "Running gridSearchOrder\n";
    gridSearchOrder(pos, Npart, cellsize, bins, grid);
    
    printf("Iterating over balls...");
    tb = time(NULL);
    
    for(long j = 0; j < Nball; j++) {
        xb = ballptr[j*ballsize1];
        yb = ballptr[j*ballsize1 + ballsize2];
        zb = ballptr[j*ballsize1 + ballsize2x2];
        
//      std::cout << "Bal coordinaten:\t";
//      std::cout << xb << '\t' << yb << '\t' << zb << '\t';
        
        ixb = xb/cellsize;
        iyb = yb/cellsize;
        izb = zb/cellsize;
        
//      std::cout << ixb << '\t' << iyb << '\t' << izb << '\n';
        
        for(long i1 = -1; i1 < 2; i1++)
            for(long i2 = -1; i2 < 2; i2++)
                for(long i3 = -1; i3 < 2; i3++) {
                    kx = (ixb + i1 + bins) % bins;
                    ky = (iyb + i2 + bins) % bins;
                    kz = (izb + i3 + bins) % bins;
                    
//                  std::cout << kx << '\t' << ky << '\t' << kz << '\n';
                    
                    cell = grid[kx + bins*ky + binssqr*kz];
//                  cell = grid[kx][ky][kz];
                    
//                  if (cell.empty()) {
//                      std::cout << "LEGE CEL: ";
//                      std::cout << kx + bins*ky + binssqr*kz << '\n';
//                  }
                    
                    for (i = cell.begin(); i != cell.end(); i++) {
                        in = *i;
                        x = posptr[in*possize1] - xb;
                        y = posptr[in*possize1 + possize2] - yb;
                        z = posptr[in*possize1 + possize2x2] - zb;
                        
//                      std::cout << x << '\t' << y << '\t' << z << '\n';
                        
                        if (x*x + y*y + z*z < r2)
                            (countptr[j*countsize1])++;
                    }
                }
    }
//  free(grid);
    te = time(NULL);
    printf("... done iterating (took %ld seconds).\n", te-tb);
}

