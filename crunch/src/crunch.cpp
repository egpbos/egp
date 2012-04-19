#include "crunch.hpp"

using namespace crunch;

int_vector crunch::test3(int n) {
    //const int dims [] = {n,d};
    int_vector out(n);
    for(int i=0; i<n; i++){
        out[i] = n-i;
    }
    return out;
}

void crunch::test2(float_vector v) {
    for(int i = 0; i < v.size(); i++) {
        printf("v[%d] = %f\n", i, v(i));
    }
    if (v.ndim() == 3) {
        v.sub(1,1,2) = 43.69;
        printf("v[%d] = %f\n", 0, v.sub(1,1,2));
    } else {
        printf("2e testje gaat niet door, want dimensie is... ");
    }
    printf("%d\n", v.ndim());
}

// Module ======================================================================
BOOST_PYTHON_MODULE(crunch)
{
	boost::python::def("TSCDensity", TSCDensity);
	boost::python::def("test2", test2);
	boost::python::def("fillrho", fillrho);
	boost::python::def("redshiftspace", redshiftspace);
	boost::python::def("gridSearch", gridSearch);
	boost::python::def("twopcf", twopcf);
	boost::python::def("twopcfnull", twopcfnull);
	boost::python::def("gaussian_filter", gaussian_filter);
	boost::python::def("test3", test3);
	boost::python::def("octreeSectorSearch", octreeSectorSearch);
	boost::python::def("resolution_independent_random_grid", resolution_independent_random_grid);
	boost::python::def("naive_random_grid", naive_independent_random_grid);
	boost::python::def("resolution_independent_random_grid_seedtable", resolution_independent_random_grid_seedtable);
}
