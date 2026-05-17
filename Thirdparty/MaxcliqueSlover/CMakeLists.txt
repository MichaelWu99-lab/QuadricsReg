cmake_minimum_required(VERSION 3.10)
project(MaxCliqueSolver VERSION 1.0)
set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# pybind11 path: pass -Dpybind11_DIR=... via cmake command line or build.sh
find_package(pybind11 REQUIRED)

# MaxCliqueSolver
add_library(MaxCliqueSolver SHARED maxcliquesolver.cpp)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

add_subdirectory(clique_solver)
target_link_libraries(MaxCliqueSolver PRIVATE CLIQUE_SOLVER::mac_solver Eigen3::Eigen)

pybind11_add_module(maxclique_solver_bindings maxclique_solver_bindings.cpp)

target_link_libraries(maxclique_solver_bindings PRIVATE MaxCliqueSolver Eigen3::Eigen)
