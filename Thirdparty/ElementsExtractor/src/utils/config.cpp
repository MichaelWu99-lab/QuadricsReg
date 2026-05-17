/**
** Created by Zhijian QIAO.
** UAV Group, Hong Kong University of Science and Technology
** email: zqiaoac@connect.ust.hk
**/

#include "utils/config.h"
// #include <glog/logging.h>
#include "global_definition/global_definition.h"
#include <filesystem>

namespace g3reg {

    Config config;

    Config::Config() {
        reset_config();
    }

    void Config::reset_config() {
        verbose = false;

        label_dir = "labels";
        sensor_dir = "";
        test_file = "";
        pcd_file = "";
        output_file ="";
        z_up = 0;
        project_path = "./";
        std::string abs_sensor_dir = project_path + "/" + sensor_dir;
        dcvc_file = abs_sensor_dir + "/dcvc.yaml";
        travel_file = abs_sensor_dir + "/travel.yaml";
        dbscan_file = abs_sensor_dir + "/dbscan.yaml";

        min_range = 0.5;
        max_range = 120.0;
        min_cluster_size = 20;
        ds_resolution = 0.5;

        front_end = "gem";
        cluster_mtd = "dcvc";

        plane_aided = true;
        volume_chi2 = 7.815;

        // save plane,line or cluster. remove ground
        save_ground = true;
        save_plane = true;
        save_line = true;
        save_cluster = true;
        remove_ground = true;

        fpfh_radius = 2.5;

        plane_resolution = 1.0;
        plane_distance_thresh = 0.2;
        plane_normal_thresh = 0.95;
        eigenvalue_thresh = 30;
        eigenvalue_thresh_line = 0.0;

        keep_horizontal_plane = false; 
        keep_horizontal_line = false; 
    }

    void Config::load_config(const std::string &config_file, char **argv) {

        YAML::Node config_node;
        if (!config_file.empty()) {
            // LOG(INFO) << "Config_path: " << config_file;
            std::ifstream fin(config_file);
            if (!fin) {
                std::cout << "Config_file: " << config_file << " not found." << std::endl;
                return;
            }
            config_node = YAML::LoadFile(config_file);
        }

        verbose = get(config_node, "verbose", false);
        sensor_dir = get(config_node, "dataset", "sensor_dir", sensor_dir);
        test_file = argv != nullptr ? config.pcd_file : "";
        pcd_file = get(config_node, "dataset", "pcd_file", pcd_file);
        output_file = get(config_node, "dataset", "output_file", output_file);
        z_up = get(config_node, "dataset", "z_up", z_up);

        // save plane,line or cluster. remove ground
        save_ground = get(config_node, "save", "save_ground", save_ground);
        save_plane = get(config_node, "save", "save_plane", save_plane);
        save_line = get(config_node, "save", "save_line", save_line);
        save_cluster = get(config_node, "save", "save_cluster", save_cluster);
        remove_ground = get(config_node, "save", "remove_ground", remove_ground);

        std::string abs_sensor_dir = sensor_dir;
        dcvc_file = abs_sensor_dir + "/dcvc.yaml";
        travel_file = abs_sensor_dir + "/travel.yaml";
        dbscan_file = abs_sensor_dir + "/dbscan.yaml";
        std::string fpfh_file = abs_sensor_dir + "/fpfh.yaml";
        std::string plane_file = abs_sensor_dir + "/plane.yaml";

        min_range = get(config_node, "dataset", "min_range", min_range);
        max_range = get(config_node, "dataset", "max_range", max_range);
        min_cluster_size = get(config_node, "dataset", "min_cluster_size", min_cluster_size);
        ds_resolution = get(config_node, "dataset", "ds_resolution", ds_resolution);

        front_end = get(config_node, "front_end", front_end);
        cluster_mtd = get(config_node, "cluster_mtd", cluster_mtd);

        // association parameter
        num_clusters = get(config_node, "extraction", "num_clusters", num_clusters);
        num_planes = get(config_node, "extraction", "num_planes", num_planes);
        num_lines = get(config_node, "extraction", "num_lines", num_lines);

        use_pseudo_cov = get(config_node, "use_pseudo_cov", use_pseudo_cov);
        use_bbox_center = get(config_node, "use_bbox_center", use_bbox_center);
        plane_aided = get(config_node, "plane_aided", plane_aided);
        grad_pmc = get(config_node, "grad_pmc", grad_pmc);
        volume_chi2 = get(config_node, "volume_chi2", volume_chi2);

        keep_horizontal_plane = get(config_node, "extraction", "keep_horizontal_plane", keep_horizontal_plane);
        keep_horizontal_line = get(config_node, "extraction", "keep_horizontal_line", keep_horizontal_line);

        if (std::ifstream(fpfh_file)) {
            config_node = YAML::LoadFile(fpfh_file);
        }
        normal_radius = get(config_node, "fpfh", "normal_radius", normal_radius);
        fpfh_radius = get(config_node, "fpfh", "fpfh_radius", fpfh_radius);

        if (std::ifstream(plane_file)) {
            config_node = YAML::LoadFile(plane_file);
        }
        plane_resolution = get(config_node, "plane_extraction", "resolution", plane_resolution);
        plane_distance_thresh = get(config_node, "plane_extraction", "distance_thresh", plane_distance_thresh);
        plane_normal_thresh = get(config_node, "plane_extraction", "normal_thresh", plane_normal_thresh);
        eigenvalue_thresh = get(config_node, "plane_extraction", "eigenvalue_thresh", eigenvalue_thresh);
        eigenvalue_thresh_line = get(config_node, "line_extraction", "eigenvalue_thresh_line", eigenvalue_thresh_line);
    }
}