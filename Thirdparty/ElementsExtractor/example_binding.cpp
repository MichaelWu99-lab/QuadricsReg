#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <filesystem>
#include "include/utils/config.h"
#include "global_definition/global_definition.h"
#include "front_end/gem/lineplane_extractor.h"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace py = pybind11;
using namespace g3reg;

// 将 pcl::PointCloud 转为 pybind11 的 numpy 数组
py::array_t<float> pointcloud_to_numpy(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) {
    // 创建一个形状为 [N,3] 的 array
    py::array_t<float> result({(py::ssize_t)cloud->size(), (py::ssize_t)3});
    auto r = result.mutable_unchecked<2>();
    for (size_t i = 0; i < cloud->size(); ++i) {
        r(i, 0) = cloud->points[i].x;
        r(i, 1) = cloud->points[i].y;
        r(i, 2) = cloud->points[i].z;
    }
    return result;
}

// 将输入的 numpy 点云转为 pcl::PointCloud
pcl::PointCloud<pcl::PointXYZ>::Ptr numpy_to_pointcloud(py::array_t<float> points) {
    if (points.ndim() != 2 || points.shape(1) != 3) {
        throw std::runtime_error("Input numpy array must be of shape [N,3]");
    }
    auto r = points.unchecked<2>();
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    cloud->resize(r.shape(0));
    for (py::ssize_t i = 0; i < r.shape(0); i++) {
        cloud->points[i].x = r(i, 0);
        cloud->points[i].y = r(i, 1);
        cloud->points[i].z = r(i, 2);
    }
    return cloud;
}

py::dict extract_features(py::array_t<float> input_points, const std::string &config_path,const string &seg_type="all", double z_up=0.0) {
    // 加载配置文件
    config.load_config(config_path, nullptr);
    config.z_up = z_up;

    if (config.save_ground == 1 || config.save_plane == 1 || config.save_line == 1 || config.save_cluster == 1)
    {
        // 如果文件夹存在则删除
        if (filesystem::exists(config.output_file)) {
            std::error_code ec;
            filesystem::remove_all(config.output_file, ec);
            if (ec) {
                throw std::runtime_error("Failed to remove directory: " + ec.message());
            }
        }
        filesystem::create_directories(config.output_file);
        string path_plane = config.output_file + "/plane";
        string path_line = config.output_file + "/line";
        string path_cluster = config.output_file + "/cluster";
        string path_ground = config.output_file + "/ground";
        filesystem::create_directories(path_plane);
        filesystem::create_directories(path_line);
        filesystem::create_directories(path_cluster);
        filesystem::create_directories(path_ground);
    }

    // 将 numpy -> pcl
    auto source = numpy_to_pointcloud(input_points);
    
    // 初始化提取器
    g3reg::PLCExtractor plc_extractor;
    plc_extractor.z_up = z_up;

    // 提取特征
    g3reg::FeatureSet feature_set;
    plc_extractor.ExtractFeature(source, feature_set.ground, feature_set.lines, feature_set.planes, feature_set.clusters,seg_type);

    std::cout << "elements_extrated G/L/P/C: " << feature_set.ground.size() << "/" << feature_set.lines.size() << "/"
              << feature_set.planes.size() << "/" << feature_set.clusters.size() << std::endl;

    // 构建返回的dict
    py::dict result;
    {
        py::dict ground_dict;
        for (size_t i = 0; i < feature_set.ground.size(); ++i) {
            auto &gf = feature_set.ground[i];
            auto cloud_out = gf->cloud();
            for (auto &p : cloud_out->points) {
                p.z += z_up;
            }
            ground_dict[py::int_(i)] = pointcloud_to_numpy(cloud_out);
        }
        result["ground"] = ground_dict;
    }

    {
        py::dict line_dict;
        for (size_t i = 0; i < feature_set.lines.size(); ++i) {
            auto &lf = feature_set.lines[i];
            // lf->cloud() 返回 PointCloud
            auto cloud_out = lf->cloud();
            // 将 z 加回来
            for (auto &p : cloud_out->points) {
                p.z += z_up;
            }
            line_dict[py::int_(i)] = pointcloud_to_numpy(cloud_out);
        }
        result["line"] = line_dict;
    }

    {
        py::dict plane_dict;
        for (size_t i = 0; i < feature_set.planes.size(); ++i) {
            auto &pf = feature_set.planes[i];
            auto cloud_out = pf->cloud();
            for (auto &p : cloud_out->points) {
                p.z += z_up;
            }
            plane_dict[py::int_(i)] = pointcloud_to_numpy(cloud_out);
        }
        result["plane"] = plane_dict;
    }

    {
        py::dict cluster_dict;
        for (size_t i = 0; i < feature_set.clusters.size(); ++i) {
            auto &cf = feature_set.clusters[i];
            auto cloud_out = cf->cloud();
            for (auto &p : cloud_out->points) {
                p.z += z_up;
            }
            cluster_dict[py::int_(i)] = pointcloud_to_numpy(cloud_out);
        }
        result["cluster"] = cluster_dict;
    }

    return result;
}

PYBIND11_MODULE(elements_extractor_bindings, m) {
    m.doc() = "Python binding for elements_extractor_bindings feature extraction using pybind11";

    m.def("extract_features", &extract_features,
          py::arg("points"), py::arg("config_path"), py::arg("seg_type")="all", py::arg("z_up")=0.0,
          "Extract line/plane/cluster features from given point cloud (Nx3 numpy), "
          "returning a dictionary {\"lines\":{...}, \"planes\":{...}, \"clusters\":{...}}");
}