#include "include/utils/config.h"
#include <iostream>
#include <filesystem>
#include <string>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "global_definition/global_definition.h"
#include <pcl/common/transforms.h>
#include "front_end/gem/lineplane_extractor.h"
#include <pcl/io/pcd_io.h> // 用于 loadPCDFile
#include <pcl/point_types.h> // 定义 pcl::PointXYZ
// #include <glog/logging.h>

using namespace std;
using namespace g3reg;

int main(int argc, char **argv) {
    if (argc < 4) {
        std::cout << "Usage: reg_bm config_file input_file output_file" << std::endl;
        return -1;
    }
    string config_path = argv[1];
    // InitGLOG(config_path, argv);
    config.load_config(config_path, argv);
    config.pcd_file = argv[2];
    config.output_file = argv[3];
    double z_ = config.z_up;
    string file_path = config.pcd_file;
    cout<< "Input file: " << file_path << endl;
    size_t dot_position = file_path.rfind('.');
    std::string type_part = file_path.substr(dot_position + 1);

    string path_plane = config.output_file + "/plane";
    string path_line = config.output_file + "/line";
    string path_cluster = config.output_file + "/cluster";
    string path_ground = config.output_file + "/ground";
    // 检查目录是否存在
    std::cout << "Output file exists: " << std::filesystem::exists(config.output_file) << std::endl;
    if (filesystem::exists(path_plane)) {
        std::error_code ec;
        filesystem::remove_all(path_plane, ec);
        if (ec) {
            throw std::runtime_error("Failed to remove directory: " + ec.message());
        }
    }

    if (filesystem::exists(path_line)) {
        std::error_code ec;
        filesystem::remove_all(path_line, ec);
        if (ec) {
            throw std::runtime_error("Failed to remove directory: " + ec.message());
        }
    }

    if (filesystem::exists(path_cluster)) {
        std::error_code ec;
        filesystem::remove_all(path_cluster, ec);
        if (ec) {
            throw std::runtime_error("Failed to remove directory: " + ec.message());
        }
    }

    if (filesystem::exists(path_ground)) {
        std::error_code ec;
        filesystem::remove_all(path_ground, ec);
        if (ec) {
            throw std::runtime_error("Failed to remove directory: " + ec.message());
        }
    }
    std::filesystem::create_directory(path_plane);
    std::filesystem::create_directory(path_line);
    std::filesystem::create_directory(path_cluster);
    std::filesystem::create_directory(path_ground);

    pcl::PointCloud<pcl::PointXYZ>::Ptr source(new pcl::PointCloud<pcl::PointXYZ>);
    if (type_part == "pcd"){
        if (pcl::io::loadPCDFile<pcl::PointXYZ>(file_path, *source) == -1) //* load the file
        {
            PCL_ERROR ("Couldn't read file source.pcd \n");
            return (-1);
        }
    }
    else if (type_part == "ply"){
        if (pcl::io::loadPLYFile<pcl::PointXYZ>(file_path, *source) == -1) //* load the file
        {
            PCL_ERROR ("Couldn't read file source.pcd \n");
            return (-1);
        }
    }
    else {
        PCL_ERROR ("Couldn't read file source \n");
        return (-1);
    }

    for (size_t i = 0; i < source->points.size(); ++i) {
        source->points[i].z -= z_;
    }
    g3reg::FeatureSet feature_set;
    g3reg::PLCExtractor plc_extractor;
    plc_extractor.z_up = z_;
    // config.save_shape = false;
    // config.num_clusters = 1000;
    // config.num_lines = 1000;
    // config.num_planes = 1000;
    plc_extractor.ExtractFeature(source, feature_set.ground,feature_set.lines, feature_set.planes, feature_set.clusters);
    // cout << "Num of clusters：" << (feature_set.clusters.size()) << endl;
    // cout << "Num of planes" << (feature_set.planes.size()) << endl;
    // cout << "Num of lines" << (feature_set.lines.size()) << endl;

    return 0;
}