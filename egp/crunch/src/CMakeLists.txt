project(crunch)

set(SOURCE crunch.cpp TSC.cpp zspace.cpp gridsearch.cpp twopcf.cpp octreesearch.cpp gaussian_filter.cpp resolution_independent_random_grid.cpp)

add_library(crunch SHARED ${SOURCE})

#find_package(PythonLibs REQUIRED)
#include_directories(${PYTHON_INCLUDE_PATH})

include_directories(/opt/local/Library/Frameworks/Python.framework/Versions/2.7/include/python2.7)

#find_package(Boost REQUIRED)
#include_directories(${BOOST_INCLUDEDIR})

include_directories(/opt/local/include)

# NumPy includes:
#set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})
#find_package(NumPy REQUIRED)
#include_directories(${NUMPY_INCLUDE_DIRS})

include_directories(/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/numpy/core/include)

#include_directories(/Users/patrick/astro/software/lib/python2.7/site-packages/PyUblas-2011.1-py2.7-macosx-10.5-x86_64.egg/include)
include_directories(/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/PyUblas-2013.1-py2.7-macosx-10.9-x86_64.egg/pyublas/include)
include_directories(/Users/patrick/astro/software/include)

#link_directories($ENV{HOME}/astro/software/lib/python2.7/site-packages/PyUblas-2011.1-py2.7-macosx-10.5-x86_64.egg/pyublas)
#link_directories(/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/PyUblas-2013.1-py2.7-macosx-10.9-x86_64.egg/pyublas)
#link_directories(/opt/local/lib)

# m -> math.h (voor octreesearch en zspace)
target_link_libraries(crunch /opt/local/lib/libboost_python-mt.dylib m /opt/local/lib/libpython2.7.dylib)

set_target_properties(crunch PROPERTIES PREFIX "")
set_target_properties(crunch PROPERTIES LIBRARY_OUTPUT_DIRECTORY $ENV{HOME}/astro/code/egp)

# Add C++11 support (used in resolution_independent_random_grid):
# And add massive local(!!) optimization with -march=native
#set_target_properties(crunch PROPERTIES COMPILE_FLAGS "-std=c++0x -march=native")
set_target_properties(crunch PROPERTIES COMPILE_FLAGS "-std=c++0x")# -DNPY_NO_DEPRECATED_API=NPY_1_8_API_VERSION")
