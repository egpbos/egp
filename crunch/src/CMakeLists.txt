project(crunch)

set(SOURCE crunch.cpp TSC.cpp zspace.cpp gridsearch.cpp twopcf.cpp octreesearch.cpp gaussian_filter.cpp resolution_independent_random_grid.cpp)

add_library(crunch SHARED ${SOURCE})

find_package(PythonLibs)
include_directories(${PYTHON_INCLUDE_PATH})

find_package(Boost)
include_directories(${BOOST_INCLUDEDIR})
#~ include_directories(BEFORE /Users/users/pbos/sw/include/boost)
#~ include_directories(BEFORE /Users/users/pbos/sw/include)

# NumPy includes:
include_directories(BEFORE /usr/lib64/python2.6/site-packages/numpy/core/include)
include_directories(BEFORE /usr/lib64/python2.6/site-packages/numpy/numarray/include) # is deze wel nodig?
include_directories(BEFORE $ENV{HOME}/data/sw/pythonSL6/lib64/python2.6/site-packages/PyUblas-0.93.1-py2.6-linux-x86_64.egg/include)

link_directories(/Users/users/pbos/sw/lib)
link_directories(BEFORE /usr/lib64)
link_directories(BEFORE $ENV{HOME}/data/sw/pythonSL6/lib64/python2.6/site-packages/PyUblas-0.93.1-py2.6-linux-x86_64.egg/pyublas)

# m -> math.h (voor octreesearch en zspace)
target_link_libraries(crunch boost_python m)

set_target_properties(crunch PROPERTIES PREFIX "")
set_target_properties(crunch PROPERTIES LIBRARY_OUTPUT_DIRECTORY $ENV{HOME}/code/egp)

# Add C++11 support (used in resolution_independent_random_grid):
# And add massive local(!!) optimization with -march=native
set_target_properties(crunch PROPERTIES COMPILE_FLAGS "-std=c++0x -march=native")
