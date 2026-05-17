#pragma once

#include <Eigen/Dense>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <algorithm>
#include <iostream>
#include <clique_solver/pmc_solver.h>

/**
 * @brief 对 pair<int, int> 的哈希结构体
 * 
 * 这里简单地将两个 int 拼装到一个 64-bit 整数上，然后用 std::hash<long long>。
 * 也可以使用 boost::hash_combine 等其它方法，只要保证分布尽量均匀即可。
 */
struct PairHash {
    std::size_t operator()(const std::pair<int, int>& p) const {
        // 注意：如果你的 int 是 32 位，则可以安全地左移 32 位并异或
        // 让 (p.first, p.second) 尽量映射到独一无二的 long long。
        // 也可以使用其他哈希组合方式。
        auto h = static_cast<unsigned long long>(static_cast<long long>(p.first)) << 32;
        h ^= static_cast<unsigned long long>(p.second) & 0xffffffffULL;
        return std::hash<unsigned long long>()(h);
    }
};

/**
 * @brief 利用 PMC Solver 寻找最大团的示例函数
 * 
 * @param data_matrix   输入数据矩阵，假设每行描述一条“无向边”，前两列表示顶点1，后两列表示顶点2
 * @param max_clique_time_limit 最大团搜索的时间限制
 * @param kcore_heuristic_threshold k-core 启发式阈值
 * @param prune_level   剪枝等级
 * @return std::vector<std::pair<int, int>> 返回最大团对应的顶点（以 pair<int,int> 的形式）
 */
std::vector<std::pair<int, int>> process_data_matrix(
    const Eigen::MatrixXd& data_matrix,
    double max_clique_time_limit,
    double kcore_heuristic_threshold,
    int prune_level)
{
    // 1) 收集所有顶点，构建 “pair<int,int> -> index” 的映射
    std::unordered_map<std::pair<int, int>, int, PairHash> vertex_to_index;
    vertex_to_index.reserve(static_cast<size_t>(data_matrix.rows() * 2));  // 预留空间，避免反复扩容

    int current_index = 0;
    for (int i = 0; i < data_matrix.rows(); ++i) {
        // 从矩阵中读取两个顶点（pair<int,int>）
        std::pair<int, int> v1 {
            static_cast<int>(data_matrix(i, 0)),
            static_cast<int>(data_matrix(i, 1))
        };
        std::pair<int, int> v2 {
            static_cast<int>(data_matrix(i, 2)),
            static_cast<int>(data_matrix(i, 3))
        };

        // 如果不存在就插入，并分配一个新的 index
        if (vertex_to_index.find(v1) == vertex_to_index.end()) {
            vertex_to_index[v1] = current_index++;
        }
        if (vertex_to_index.find(v2) == vertex_to_index.end()) {
            vertex_to_index[v2] = current_index++;
        }
    }

    // 2) 构建边列表（无向图），为避免重复边，我们先收集到 vector，再排序去重
    std::vector<std::pair<int, int>> edges;
    edges.reserve(static_cast<size_t>(data_matrix.rows() * 2)); // 每行对应两条有向边 (v1->v2 和 v2->v1)

    for (int i = 0; i < data_matrix.rows(); ++i) {
        std::pair<int, int> v1 {
            static_cast<int>(data_matrix(i, 0)),
            static_cast<int>(data_matrix(i, 1))
        };
        std::pair<int, int> v2 {
            static_cast<int>(data_matrix(i, 2)),
            static_cast<int>(data_matrix(i, 3))
        };

        int idx1 = vertex_to_index[v1];
        int idx2 = vertex_to_index[v2];

        // 对于无向图，一般会插入两条有向边 (idx1->idx2) & (idx2->idx1)。
        // 也可以只插入一次，后面构造邻接表时再做对称，但这里保留双向方便后续统一处理。
        edges.emplace_back(idx1, idx2);
        edges.emplace_back(idx2, idx1);
    }

    // 3) 对 edges 排序并去重，避免 (i -> j) 重复插入
    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());

    // 4) 利用 clique_solver::Graph 来创建图
    clique_solver::Graph inlier_graph;
    inlier_graph.populateVertices(current_index); // 告诉图结构，顶点从 [0, current_index-1]

    for (auto& e : edges) {
        // e.first, e.second 都是顶点 index
        inlier_graph.addEdge(e.first, e.second);
    }

    // 5) 配置 PMC Solver 参数
    clique_solver::MaxCliqueSolver::Params clique_params;
    // 这里直接以 PMC_EXACT 为例
    clique_params.solver_mode = clique_solver::MaxCliqueSolver::CLIQUE_SOLVER_MODE::PMC_EXACT;
    clique_params.time_limit = max_clique_time_limit;
    clique_params.kcore_heuristic_threshold = kcore_heuristic_threshold;

    // 6) 调用 PMC Solver 寻找最大团
    clique_solver::MaxCliqueSolver solver(clique_params);
    std::vector<int> max_clique_indices = solver.findMaxClique(inlier_graph, prune_level);

    if (max_clique_indices.size() <= 1) {
        std::cout << "Clique size too small. Abort." << std::endl;
        return {};
    }

    // 7) 还原 “index -> pair<int,int>” 的映射
    //    因为要返回原始数据表示的顶点
    std::vector<std::pair<int, int>> index_to_vertex(static_cast<size_t>(current_index));
    for (auto& kv : vertex_to_index) {
        index_to_vertex[kv.second] = kv.first;
    }

    // 8) 将最大团索引对应到真实的顶点 pair<int,int>
    std::vector<std::pair<int, int>> result;
    result.reserve(max_clique_indices.size());
    for (int idx : max_clique_indices) {
        result.push_back(index_to_vertex[idx]);
    }

    return result;
}
