// maxcliquesolver.h

#ifndef MAXCLIQUESOLVER_H
#define MAXCLIQUESOLVER_H

#include <Eigen/Dense>
#include <vector>
#include <map>
#include <string>

// 声明 process_data_matrix 函数
std::vector<int> process_data_matrix(
    const Eigen::MatrixXd& data_matrix,
    double max_clique_time_limit,
    double kcore_heuristic_threshold,
    int prune_level);

#endif // MAXCLIQUESOLVER_H
