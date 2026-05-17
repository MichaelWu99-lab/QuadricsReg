cmake_minimum_required(VERSION 3.15)
set(CMAKE_BUILD_TYPE Release)
project(ElementsExtractor)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)


set(ALL_TARGET_LIBRARIES "")

# pybind11 path: pass -Dpybind11_DIR=... via cmake command line or build.sh
include(cmake/boost.cmake)
include(cmake/eigen.cmake)
include(cmake/yaml.cmake)
# include(cmake/glog.cmake)
include(cmake/pcl.cmake)
include(cmake/openmp.cmake)
include(cmake/global_definition.cmake)

include_directories(
        ${PROJECT_SOURCE_DIR}/include
)

find_package(pybind11 REQUIRED)

file(GLOB_RECURSE ALL_SRCS "src/*.c*" "include/*.h*")
list(APPEND ALL_TARGET_LIBRARIES stdc++fs tbb)
add_library(${PROJECT_NAME} STATIC ${ALL_SRCS})
target_include_directories(${PROJECT_NAME} PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

add_subdirectory(Thirdparty/robot_utils)
target_link_libraries(${PROJECT_NAME}
        ${ALL_TARGET_LIBRARIES}
        ROBOT_UTILS::robot_utils
)

add_executable(main main.cpp)
target_link_libraries(main
${PROJECT_NAME}
)

target_link_libraries(main ${PCL_LIBRARIES})

add_library(elements_extractor_bindings MODULE example_binding.cpp)
target_link_libraries(elements_extractor_bindings PRIVATE ${PROJECT_NAME} ${PCL_LIBRARIES} pybind11::module)
set_target_properties(elements_extractor_bindings PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}" SUFFIX "${PYTHON_MODULE_EXTENSION}")
