#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "maxcliquesolver.h"  // 包含头文件，声明你要暴露的函数

namespace py = pybind11;

PYBIND11_MODULE(maxclique_solver_bindings, m) {
    m.def("process_data_matrix", &process_data_matrix, "Process data matrix and return the max clique",
          py::arg("data_matrix"),
          py::arg("max_clique_time_limit"),
          py::arg("kcore_heuristic_threshold"),
          py::arg("prune_level"));
}
