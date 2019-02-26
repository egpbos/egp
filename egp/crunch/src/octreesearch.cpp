#include <pyublas/numpy.hpp>
#include <iostream>
#include <list>
#include <vector>
#include <time.h>
#include <stdlib.h>
#include <math.h>

#include "octree/octree.h"

#include "crunch.hpp"

using namespace crunch;

int_vector crunch::octreeSectorSearch(float_vector pos, long Npart, int_vector sectors, long Nsec, int bins, float cellsize) {
//		std::cout << "Pos pointers\n";
    long possize1 = pos.strides()[0]/pos.itemsize();
    long possize2 = pos.strides()[1]/pos.itemsize();
    long possize2x2 = 2*possize2;
    float* posptr = pos.data().begin();
    
//		std::cout << "Ball pointers\n";
    long secsize1 = sectors.strides()[0]/sectors.itemsize();
    long secsize2 = sectors.strides()[1]/sectors.itemsize();
    long secsize2x2 = 2*secsize2;
    int* secptr = sectors.data().begin();
    
//		std::cout << "Init constants\n";		
    int treesize = log(bins)/log(2);
    if (pow(2,treesize) < bins) treesize++;
    Octree<int> tree(pow(2,treesize));
    
    // Init variables
    int xs, ys, zs;
    time_t tb, te;
    
    tb = time(NULL);
    for (int s = 0; s < Nsec; s++) {
        xs = secptr[s*secsize1];
        ys = secptr[s*secsize1 + secsize2];
        zs = secptr[s*secsize1 + secsize2x2];
        
        tree.set(xs,ys,zs,1);
    }
    te = time(NULL);
    printf("Built octree in %ld seconds.\n", te-tb);
    
    long_list ids;
    
    for (int j = 0; j < Npart; j++) {
        xs = posptr[j*possize1] / cellsize;
        ys = posptr[j*possize1 + possize2] / cellsize;
        zs = posptr[j*possize1 + possize2x2] / cellsize;
        
        if (tree.at(xs,ys,zs)) {
            ids.push_back(j);
        }
    }
    
    int_vector pyids(ids.size());
    
    int i = 0;
    for (long_list::iterator li = ids.begin(); li != ids.end(); li++) {
        pyids[i++] = *li;
    }
    
    return pyids;
}
