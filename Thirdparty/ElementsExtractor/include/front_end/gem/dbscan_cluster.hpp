#ifndef DBSCAN_CLUSTER_H
#define DBSCAN_CLUSTER_H

#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <yaml-cpp/yaml.h>
#include <unordered_map>

#include <memory>
#include <vector>
#include <cmath>
#include <iostream>

template <typename PointT>
class DBSCANCluster
{
public:
    struct DBSCANParam {
        double eps = 1.0;       // 邻域半径
        int minPts = 5;         // 最少点数
        double max_range = 100; // 与DCVC示例保持一致
        double min_range = 0.5;
        int min_cluster_size= 20;

    };

    using PointCloudPtr = boost::shared_ptr<pcl::PointCloud<PointT>>;

private:
    DBSCANParam params_;
    PointCloudPtr cloud_;               // 输入点云
    std::vector<PointCloudPtr> clusters_;  // 聚类结果

public:

    template<typename T>
    T get(const YAML::Node &node, const std::string &father_key, const std::string &key, const T &default_value) {
        if (!node[father_key] || !node[father_key][key]) {
//        std::cout << "Key " << father_key << "/" << key << " not found, using default value: " << default_value << std::endl;
            return default_value;
        }
        T value = node[father_key][key].as<T>();
//    std::cout << "Key " << father_key << "/" << key << " found, using value: " << value << std::endl;
        return value;
    }

    // 读取配置文件版本
    DBSCANCluster(std::string &config_path)
    {

        YAML::Node config_node = YAML::LoadFile(config_path);

        params_.eps = get(config_node, "dbscan", "eps", 1.0);
        params_.minPts = get(config_node, "dbscan", "deltaR", 5);
        params_.max_range = get(config_node, "dbscan", "max_range", 120.0);
        params_.min_range = get(config_node, "dbscan", "min_range", 0.5);
        params_.min_cluster_size = get(config_node, "dbscan", "min_cluster_size", 20);

        cloud_.reset(new pcl::PointCloud<PointT>);
        clusters_.clear();
    }

    // 手动指定参数版本
    DBSCANCluster(DBSCANParam &params)
    {
        params_ = params;
        cloud_.reset(new pcl::PointCloud<PointT>);
        clusters_.clear();
    }

    // 析构
    ~DBSCANCluster()
    {
        cloud_.reset();
    }

    /**
     * @brief 调用 DBSCAN 来分割点云
     * @param input_cloud 输入点云
     * @param clusters 输出分割结果(多个PointCloud)
     * @return 是否成功
     */
    bool segmentPointCloud(PointCloudPtr input_cloud, std::vector<PointCloudPtr> &clusters)
    {
        if (!input_cloud || input_cloud->points.empty()) {
            std::cerr << "[DBSCAN] Input cloud is empty\n";
            return false;
        }

        // 1) 复制点云
        pcl::copyPointCloud(*input_cloud, *cloud_);

        // 2) 可选: 过滤掉 < min_range 或 > max_range 的点
        PointCloudPtr filtered(new pcl::PointCloud<PointT>);
        filterByRange(cloud_, filtered);

        // 3) 构建 kdtree 以加速邻域搜索
        pcl::KdTreeFLANN<PointT> kdtree;
        kdtree.setInputCloud(filtered);

        // 4) 调用 DBSCAN 核心
        std::vector<int> labels;
        if (!dbscanCluster(filtered, kdtree, labels)) {
            std::cerr << "[DBSCAN] Clustering failed.\n";
            return false;
        }

        // 5) 根据 labels 生成输出 clusters
        labelAnalysis(filtered, labels);

        // 6) 将结果返回
        clusters = clusters_;
        return true;
    }

private:

    /**
     * @brief 将点云过滤到[min_range, max_range]之间
     */
    void filterByRange(PointCloudPtr in, PointCloudPtr out)
    {
        out->points.reserve(in->points.size());
        for (auto &p : in->points) {
            double range = std::sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
            if (range >= params_.min_range && range <= params_.max_range) {
                out->points.push_back(p);
            }
        }
        out->width  = out->points.size();
        out->height = 1;
        out->is_dense = true;
    }

    /**
     * @brief DBSCAN算法核心
     * @param cloud 过滤后的点云
     * @param kdtree 用于搜索邻域
     * @param labels 输出每个点的标签，-1表示噪声
     * @return 是否成功
     */
    bool dbscanCluster(PointCloudPtr cloud,
                       pcl::KdTreeFLANN<PointT> &kdtree,
                       std::vector<int> &labels)
    {
        size_t N = cloud->points.size();
        if (N == 0) return false;

        labels.resize(N, -1);  // 初始全为-1(未分类)
        int clusterID = 0;

        // 记录已访问的点
        std::vector<bool> visited(N, false);

        // 对每个点检查
        for (size_t i = 0; i < N; ++i) {
            if (visited[i]) continue;

            visited[i] = true;

            // 找到 i 的邻域
            std::vector<int> neighbors;
            regionQuery(kdtree, cloud->points[i], neighbors);

            // 如果邻居数量 < minPts，则标记为噪声(-1)
            if (neighbors.size() < (size_t)params_.minPts) {
                labels[i] = -1; // 噪声
            } else {
                // 创建新簇
                clusterID++;
                // 将 i 加入该簇
                expandCluster(cloud, kdtree, i, neighbors, clusterID, visited, labels);
            }
        }

        return true;
    }

    /**
     * @brief 搜索点 p 的邻域(半径 = eps)
     */
    void regionQuery(pcl::KdTreeFLANN<PointT> &kdtree,
                     const PointT &p,
                     std::vector<int> &neighbors)
    {
        // 利用FLANN搜索半径内点
        std::vector<float> sqrDist;
        kdtree.radiusSearch(p, params_.eps, neighbors, sqrDist, 0);
    }

    /**
     * @brief 扩展聚类
     * @param i 当前核心点索引
     * @param neighbors 当前核心点邻居索引
     * @param clusterID 当前簇标签
     * @param visited 标记是否访问
     * @param labels 输出标签数组
     */
    void expandCluster(PointCloudPtr cloud,
                       pcl::KdTreeFLANN<PointT> &kdtree,
                       size_t i,
                       std::vector<int> &neighbors,
                       int clusterID,
                       std::vector<bool> &visited,
                       std::vector<int> &labels)
    {
        // 将 i 和它的邻居都标记为 clusterID
        labels[i] = clusterID;

        // 遍历 neighbors
        for (size_t idx = 0; idx < neighbors.size(); ++idx) {
            int pid = neighbors[idx];
            if (!visited[pid]) {
                visited[pid] = true;

                // 再找 pid 的邻居
                std::vector<int> neighborPts;
                regionQuery(kdtree, cloud->points[pid], neighborPts);

                // 若邻居点数 >= minPts，则合并
                if (neighborPts.size() >= (size_t)params_.minPts) {
                    // 将neighborPts合并到neighbors
                    neighbors.insert(neighbors.end(), neighborPts.begin(), neighborPts.end());
                }
            }
            // 如果还没标记，则标记
            if (labels[pid] == -1) {
                labels[pid] = clusterID;
            }
        }
    }

    /**
     * @brief 根据标签合并点云
     */
    void labelAnalysis(PointCloudPtr cloud, std::vector<int> &labels)
    {
        // label -> 对应的点索引
        std::unordered_map<int, std::vector<int>> label2indices;
        for (size_t i = 0; i < labels.size(); ++i) {
            if (labels[i] == -1) continue; // 噪声不进入任何簇
            label2indices[labels[i]].push_back(i);
        }

        // 逐簇生成点云
        clusters_.clear();
        for (auto &kv : label2indices) {
            if (kv.second.size() < (size_t)params_.min_cluster_size) continue;
            typename pcl::PointCloud<PointT>::Ptr ccloud(new pcl::PointCloud<PointT>);
            for (auto &idx : kv.second) {
                ccloud->points.push_back(cloud->points[idx]);
            }
            ccloud->width = ccloud->points.size();
            ccloud->height = 1;
            ccloud->is_dense = true;
            clusters_.push_back(ccloud);
        }
    }
};

#endif // DBSCAN_CLUSTER_H
