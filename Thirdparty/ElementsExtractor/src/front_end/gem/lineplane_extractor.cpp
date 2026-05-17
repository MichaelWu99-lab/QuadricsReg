/**
** Created by Zhijian QIAO.
** UAV Group, Hong Kong University of Science and Technology
** email: zqiaoac@connect.ust.hk
**/
#include "front_end/gem/lineplane_extractor.h"
// #include "front_end/graph_vertex.h"
#include "robot_utils/tic_toc.h"
#include "front_end/gem/clustering.h"
#include <pcl/io/pcd_io.h>
#include <unsupported/Eigen/MatrixFunctions>

namespace g3reg {

    void PLCExtractor::ExtractFeature(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_xyz,
                                      std::vector<SurfaceFeature::Ptr> &ground_features,
                                      std::vector<LineFeature::Ptr> &line_features,
                                      std::vector<SurfaceFeature::Ptr> &surface_features,
                                      std::vector<ClusterFeature::Ptr> &cluster_features,
                                      const string &seg_type) {
        reset();

        // 单独提取平面、线、聚类
        if(seg_type == "plane"){
            cutCloud(*cloud_xyz, FeatureType::None, config.plane_resolution, voxel_map);
            for (auto voxel_iter = voxel_map.begin(); voxel_iter != voxel_map.end(); ++voxel_iter) {
                Voxel::Ptr voxel = voxel_iter->second;
                if (voxel->parse()) {
                    if (config.plane_aided) {
                        voxel->setSemanticType(FeatureType::Plane);
                    }
                }
            }
            MergePlanes(surface_features);
            FilterSurface(surface_features, config.min_cluster_size);
            if(config.save_plane){
                SaveSurface(surface_features);
            }

            return;
        }
        else if(seg_type == "line"){
            travel::Cluster(cloud_xyz, cluster_features, false);
            ExtractPole(cluster_features, line_features);
            cluster_features.clear();
            if(config.save_line){
                SaveLine(line_features);
            }
            return;
        }
        else if(seg_type == "cluster"){
            if (config.cluster_mtd == "travel") {
                travel::Cluster(cloud_xyz, cluster_features, false);
            } else if (config.cluster_mtd == "dcvc") {
                DCVC::Cluster(cloud_xyz, cluster_features, false);
            } else if (config.cluster_mtd == "dbscan") {
                DBSCAN::Cluster(cloud_xyz, cluster_features, false);
            } else {
                std::cerr << "Unknown cluster method: " << config.cluster_mtd << std::endl;
            }
            // travel::Cluster(cloud_xyz, cluster_features, false);
            if(config.save_cluster){
                SaveCluster(cluster_features);
            }
            return;
        }

        // 统一提取
        robot_utils::TicToc t;
        double tSrc, ground_time, plane_time, cluster_time, line_time;
        pcl::PointCloud<pcl::PointXYZ> cloud_ground;
        pcl::PointCloud<pcl::PointXYZ> cloud_nonground;
        if(config.remove_ground == true){
            travel::estimateGround(*cloud_xyz, cloud_ground, cloud_nonground, tSrc);
            ground_features.emplace_back(SurfaceFeature::Ptr(new SurfaceFeature(cloud_ground.makeShared())));
        } else {
            cloud_nonground = *cloud_xyz;
        }

        cutCloud(cloud_nonground, FeatureType::None, config.plane_resolution, voxel_map);

        if (config.plane_aided) {
            for (auto voxel_iter = voxel_map.begin(); voxel_iter != voxel_map.end(); ++voxel_iter) {
                Voxel::Ptr voxel = voxel_iter->second;
                if (voxel->parse()) {
                    if (config.plane_aided) {
                        voxel->setSemanticType(FeatureType::Plane);
                    }
                }
            }
            MergePlanes(surface_features);
            FilterSurface(surface_features, config.min_cluster_size);
            plane_time = t.toc();
        } else {
            for (auto voxel_iter = voxel_map.begin(); voxel_iter != voxel_map.end(); ++voxel_iter) {
                voxel_iter->second->solveCenter();
            }
        }

        auto voxel_iter = voxel_map.begin();
        pcl::PointCloud<pcl::PointXYZ>::Ptr other_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        for (; voxel_iter != voxel_map.end(); ++voxel_iter) {
            if (voxel_iter->second->type() != FeatureType::Plane) {
                *other_cloud += *(voxel_iter->second->cloud());
            }
        }

        // 测速
        if (config.cluster_mtd == "travel") {
            travel::Cluster(other_cloud, cluster_features, false);
        } else if (config.cluster_mtd == "dcvc") {
            DCVC::Cluster(other_cloud, cluster_features, false);
        } else if (config.cluster_mtd == "dbscan") {
            DBSCAN::Cluster(other_cloud, cluster_features, false);
        } else {
            std::cerr << "Unknown cluster method: " << config.cluster_mtd << std::endl;
        }
        cluster_time = t.toc();
        // std:cout << "time: cluster: " << cluster_time << "ms" <<std::endl;

        if (config.num_lines > 0) {
            ExtractPole(cluster_features, line_features);
            line_time = t.toc();
        }

        if (!config.plane_aided) {
            ExtractPlanes(cluster_features, surface_features);
            plane_time = t.toc();
        }

        // std::cout << "[Extracted Feature Counts]" << std::endl;
        // std::cout << "  Planes : " << surface_features.size() << std::endl;
        // std::cout << "  Lines  : " << line_features.size() << std::endl;
        // std::cout << "  Clusters: " << cluster_features.size() << std::endl;

        if(config.save_ground){
            SaveGround(cloud_ground.makeShared());
        }
        if(config.save_cluster){
            SaveCluster(cluster_features);
        }
        if(config.save_plane){
            SaveSurface(surface_features);
        }
        if(config.save_line){
            SaveLine(line_features);
        }
//    LOG(INFO) << "ExtractFeature time: ground/planar/cluster/line: " << ground_time << "/" << plane_time << "/" << cluster_time << "/" << line_time;
    }

    void PLCExtractor::ExtractPlanes(std::vector<ClusterFeature::Ptr> &cluster_features,
                                     std::vector<SurfaceFeature::Ptr> &plane_features) {
        plane_features.clear();
        std::vector<int> remove_index;
        for (int i = 0; i < cluster_features.size(); ++i) {

            ClusterFeature::Ptr cluster_feature = cluster_features[i];
            const Eigen::Vector3d &eigen_values = cluster_feature->eigen_values();

            if (eigen_values(1) / eigen_values(0) < config.eigenvalue_thresh) {
                continue;
            }

            SurfaceFeature::Ptr feature = SurfaceFeature::Ptr(new SurfaceFeature(cluster_feature->cloud()));
            remove_index.push_back(i);
            plane_features.emplace_back(feature);
        }
        // remove clusters from cluster_features
        for (int i = remove_index.size() - 1; i >= 0; --i) {
            cluster_features.erase(cluster_features.begin() + remove_index[i]);
        }
    }

    void PLCExtractor::ExtractPole(std::vector<ClusterFeature::Ptr> &cluster_features,
                                   std::vector<LineFeature::Ptr> &line_features) {
        line_features.clear();
        std::vector<int> remove_index;
        //保存线段类别的物体
        for (int i = 0; i < cluster_features.size(); ++i) {

            ClusterFeature::Ptr cluster_feature = cluster_features[i];
            Eigen::Vector3d eigen_values = cluster_feature->eigen_values();
            
            // if (eigen_values[2] / eigen_values[0] < config.eigenvalue_thresh / 2.0) {
            //     continue;
            // }

            if (config.eigenvalue_thresh_line != 0.0) {
                // 因为是3维线，所以eigen_values[2] / eigen_values[0]
                if (eigen_values[2] / eigen_values[0] < config.eigenvalue_thresh_line) {
                    continue;
                }
            } else {
                if (eigen_values[2] / eigen_values[0] < config.eigenvalue_thresh / 2.0) {
                    continue;
                }
            }

            double angle = abs(cluster_feature->direction().dot(Eigen::Vector3d(0, 0, 1)));
            // if (angle < 0.707) {
            //     continue;
            // }

            if (!config.keep_horizontal_line && angle < 0.707) {
                // 如果不保留水平线，且方向接近水平（夹角大于 45°），跳过
                continue;
            }

            pcl::PointCloud<pcl::PointXYZ>::Ptr cluster_cloud = cluster_feature->cloud();
            // using RANSAC 拟合直线，判断内点率和内点个数是否满足条件
            pcl::SampleConsensusModelLine<pcl::PointXYZ>::Ptr model_line(
                    new pcl::SampleConsensusModelLine<pcl::PointXYZ>(cluster_cloud)
            );
            pcl::RandomSampleConsensus<pcl::PointXYZ> ransac(model_line);
            ransac.setDistanceThreshold(0.5);
            ransac.setMaxIterations(10);
            ransac.computeModel();
            std::vector<int> inliers;
            ransac.getInliers(inliers);
            double inlier_ratio = (double) inliers.size() / cluster_cloud->size();
            if (inlier_ratio < 0.5 || inliers.size() < 5) {
                continue;
            }

            pcl::PointCloud<pcl::PointXYZ> pole_line;
            pcl::copyPointCloud<pcl::PointXYZ>(*cluster_cloud, inliers, pole_line);
            //添加局部点云到整体线段点云当中
            LineFeature::Ptr feature = LineFeature::Ptr(new LineFeature());
            if (feature->Init(pole_line.makeShared())) {
                remove_index.push_back(i);
                line_features.emplace_back(feature);
            }
        }

        // remove clusters from cluster_features
        for (int i = remove_index.size() - 1; i >= 0; --i) {
            cluster_features.erase(cluster_features.begin() + remove_index[i]);
        }
    }

    void PLCExtractor::MergePlanes(std::vector<SurfaceFeature::Ptr> &surface_features) {

        //    merge surface
        std::map<int, SurfaceFeature::Ptr> surface_map;
        int global_id = 0;
        for (auto voxel_iter = voxel_map.begin(); voxel_iter != voxel_map.end(); ++voxel_iter) {
            const VoxelKey &loc = voxel_iter->first;
            Voxel::Ptr cur_voxel = voxel_iter->second;
            if (cur_voxel->type() != FeatureType::Plane) continue;

            if (cur_voxel->instance_id < 0) {
                SurfaceFeature::Ptr surface(
                        new SurfaceFeature(cur_voxel->center(), cur_voxel->normal(), cur_voxel->sigma()));
                surface->setCloud(cur_voxel->cloud());
                surface->push_back(loc);
                surface_map[global_id] = surface;
                cur_voxel->instance_id = global_id;
                global_id++;
            }
            SurfaceFeature::Ptr surface = surface_map[cur_voxel->instance_id];
            // if (!surface) {
            //     LOG(INFO) << "surface is null";
            // }
            std::vector<VoxelKey> neighbors;
            cur_voxel->getNeighbors(neighbors);
            for (VoxelKey &neighbor: neighbors) {
                auto neighbor_iter = voxel_map.find(neighbor);
                // if neighbor is not in voxel_map
                if (neighbor_iter == voxel_map.end()) continue;
                Voxel &neighbor_voxel = *neighbor_iter->second;
                // if neighbor is not a surface voxel, continue
                if (neighbor_voxel.type() == FeatureType::None) continue;
                // if neighbor has not been assigned to a surface
                if (neighbor_voxel.instance_id < 0) {
                    if (surface->consistent(neighbor_voxel)) {
                        surface->merge(neighbor_voxel);
                        neighbor_voxel.instance_id = cur_voxel->instance_id;
                    }
                } else {
                    // if neighbor has been assigned to a surface, try to merge
                    if (neighbor_voxel.instance_id == cur_voxel->instance_id) continue;
                    SurfaceFeature::Ptr neighbor_surface = surface_map[neighbor_voxel.instance_id];
                    if (surface->consistent(*neighbor_surface)) {
                        surface->merge(*neighbor_surface);
                        surface_map.erase(neighbor_voxel.instance_id);
                        for (auto &voxel_loc: neighbor_surface->voxels()) {
                            voxel_map[voxel_loc]->instance_id = cur_voxel->instance_id;
                        }
                    }
                }
            }
        }

        surface_features.clear();
        for (auto &surface_pair: surface_map) {
            surface_features.emplace_back(surface_pair.second);
        }
    }

    void PLCExtractor::FilterSurface(std::vector<SurfaceFeature::Ptr> &surface_features, 
                                 int min_points) {
        std::vector<SurfaceFeature::Ptr> filtered_surface_features;

        for (auto &surface_feature : surface_features) {
            double angle = abs(surface_feature->normal().dot(Eigen::Vector3d(0, 0, 1)));

            bool enough_points = surface_feature->cloud()->size() > min_points;
            bool satisfy_angle = false;
            if (config.keep_horizontal_plane) {
                // 保留水平 + 垂直
                // 夹角小于 30°（cosθ > 0.866） → 平行于地面,夹角大于 75°（cosθ < 0.2588）→ 垂直于地面
                satisfy_angle = (angle > 0.866 || angle < 0.2588);
            } else {
                // 排除水平，只保留非水平
                satisfy_angle = angle < 0.707;
            }

            if (enough_points && satisfy_angle) {
                filtered_surface_features.emplace_back(surface_feature);
            } else {
                for (VoxelKey &loc : surface_feature->voxels()) {
                    voxel_map[loc]->setSemanticType(FeatureType::None);
                }
            }
        }

        surface_features = filtered_surface_features;
    }

    // void PLCExtractor::FilterSurface(std::vector<SurfaceFeature::Ptr> &surface_features, int min_points) {

    //     // only save the surfaces with more than min_points
    //     std::vector<SurfaceFeature::Ptr> filtered_surface_features;
    //     for (auto &surface_feature: surface_features) {
    //         double angle = abs(surface_feature->normal().dot(Eigen::Vector3d(0, 0, 1)));
    //         // if (surface_feature->cloud()->size() > min_points && angle < 0.707) 
    //         // 夹角小于 30°（cosθ > 0.866） → 平行于地面,夹角大于 75°（cosθ < 0.2588）→ 垂直于地面
    //         if (surface_feature->cloud()->size() > min_points && (angle > 0.866 || angle < 0.2588)) 
    //         {
    //             filtered_surface_features.emplace_back(surface_feature);
    //         } else {
    //             for (VoxelKey &loc: surface_feature->voxels()) {
    //                 voxel_map[loc]->setSemanticType(FeatureType::None);
    //             }
    //         }
    //     }
    //     surface_features = filtered_surface_features;
    // }

    // void PLCExtractor::FilterSurface(std::vector<SurfaceFeature::Ptr> &surface_features,
    //                              int min_points) {
    // // Only keep planes satisfying Manhattan-world alignment and point-count threshold
    // std::vector<SurfaceFeature::Ptr> filtered;
    // // Alignment threshold: e.g., config.axis_align_thresh = 0.95 (≈18° tolerance)
    // double axis_thresh = 0.95;

    // for (auto &sf : surface_features) {
    //     size_t pts = sf->cloud()->size();
    //     Eigen::Vector3d n = sf->normal().normalized();
    //     // Absolute cosines with X, Y, Z axes
    //     double ax = std::abs(n.dot(Eigen::Vector3d::UnitX()));
    //     double ay = std::abs(n.dot(Eigen::Vector3d::UnitY()));
    //     double az = std::abs(n.dot(Eigen::Vector3d::UnitZ()));

    //     // Keep if above point threshold and aligned to any principal axis
    //     if (pts > min_points && (ax > axis_thresh || ay > axis_thresh || az > axis_thresh)) {
    //         filtered.emplace_back(sf);
    //     } else {
    //         // Reset semantic type for removed voxels
    //         for (auto &loc : sf->voxels()) {
    //             voxel_map[loc]->setSemanticType(FeatureType::None);
    //         }
    //     }
    // }
    // surface_features.swap(filtered);
    // }


    void PLCExtractor::reset() {
        voxel_map.clear();
    }

    void TopKEllipse(std::vector<g3reg::QuadricFeature::Ptr> &ellipsoids, int k) {
        ellipsoids.erase(std::remove_if(ellipsoids.begin(), ellipsoids.end(), [](g3reg::QuadricFeature::Ptr ellipsoid) {
            const double &norm = ellipsoid->center().norm();
            return (norm > config.max_range || norm < config.min_range);
        }), ellipsoids.end());
        if (ellipsoids.size() < k)
            return;
        std::vector<std::pair<double, g3reg::QuadricFeature::Ptr>> ellipsoids_score;
        for (auto &ellipsoid: ellipsoids) {
            ellipsoids_score.emplace_back(std::make_pair(ellipsoid->score(), ellipsoid));
        }
        std::sort(ellipsoids_score.begin(), ellipsoids_score.end(),
                  [](std::pair<double, g3reg::QuadricFeature::Ptr> p1,
                     std::pair<double, g3reg::QuadricFeature::Ptr> p2) {
                      return p1.first > p2.first;
                  });
        ellipsoids.clear();
        for (int i = 0; i < k; i++) {
            ellipsoids.push_back(ellipsoids_score[i].second);
        }
    }

    void TransformToEllipsoid(const FeatureSet &featureSet, std::vector<std::vector<QuadricFeature::Ptr>> &ellipsoids) {
        ellipsoids.clear();
        std::vector<g3reg::QuadricFeature::Ptr> ellipsoid_lines;
        for (auto &line: featureSet.lines) {
            g3reg::QuadricFeature::Ptr quadric_feature = std::dynamic_pointer_cast<g3reg::QuadricFeature>(line);
            ellipsoid_lines.push_back(quadric_feature);
        }
        TopKEllipse(ellipsoid_lines, config.num_lines);
        ellipsoids.push_back(ellipsoid_lines);

        std::vector<g3reg::QuadricFeature::Ptr> ellipsoid_planes;
        for (auto &plane: featureSet.planes) {
            g3reg::QuadricFeature::Ptr quadric_feature = std::dynamic_pointer_cast<g3reg::QuadricFeature>(plane);
            ellipsoid_planes.push_back(quadric_feature);
        }
        TopKEllipse(ellipsoid_planes, config.num_planes);
        ellipsoids.push_back(ellipsoid_planes);

        std::vector<g3reg::QuadricFeature::Ptr> ellipsoid_clusters;
        for (auto &cluster: featureSet.clusters) {
            g3reg::QuadricFeature::Ptr quadric_feature = std::dynamic_pointer_cast<g3reg::QuadricFeature>(cluster);
            ellipsoid_clusters.push_back(quadric_feature);
        }
        TopKEllipse(ellipsoid_clusters, config.num_clusters);
        ellipsoids.push_back(ellipsoid_clusters);

        if (config.use_pseudo_cov) {
            for (int i = 0; i < ellipsoids.size(); ++i) {
                for (int j = 0; j < ellipsoids[i].size(); ++j) {
                    ellipsoids[i][j]->fitting();
                }
            }
        }
//        LOG(INFO) << "TransformToEllipsoid: " << ellipsoids[0].size() << " lines, " << ellipsoids[1].size() << " planes, " << ellipsoids[2].size() << " clusters" << std::endl;
    }

    void PLCExtractor::SaveCluster(const std::vector<ClusterFeature::Ptr> &cluster_features){
        int fileIndex = 0;
        pcl::PLYWriter writer;
        for (int i = 0; i < cluster_features.size(); ++i) {
            ClusterFeature::Ptr cluster_feature = cluster_features[i];
            pcl::PointCloud<pcl::PointXYZ>::Ptr saved_cloud(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::copyPointCloud(*cluster_feature->cloud(), *saved_cloud);
            saved_cloud->width=saved_cloud->points.size();
            saved_cloud->height=1;
            for (size_t i = 0; i < saved_cloud->points.size(); ++i) {
                saved_cloud->points[i].z += z_up;
            }
            std::string fileName = generateFileName(2, fileIndex);
            writer.write(config.output_file + fileName, *saved_cloud, true,false);
            fileIndex++;
        }
        std::cout<<"Cluster: "<<fileIndex<<std::endl;
        // std::cout<<"保存了"<<fileIndex<<"个物体"<<std::endl;
    }

    void PLCExtractor::SaveSurface(const std::vector<SurfaceFeature::Ptr> &surface_features){
        int fileIndex = 0;
        pcl::PLYWriter writer;
        for (int i = 0; i < surface_features.size(); ++i) {
            SurfaceFeature::Ptr surface_feature = surface_features[i];
            pcl::PointCloud<pcl::PointXYZ>::Ptr saved_cloud(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::copyPointCloud(*surface_feature->cloud(), *saved_cloud);
            for (size_t i = 0; i < saved_cloud->points.size(); ++i) {
                saved_cloud->points[i].z += z_up;
            }
            std::string fileName = generateFileName(3, fileIndex);
            writer.write(config.output_file + fileName, *saved_cloud, true,false);
            fileIndex++;
        }
        std::cout<<"Plane: "<<fileIndex<<std::endl;
        // std::cout<<"保存了"<<fileIndex<<"个平面"<<std::endl;
    }

    void PLCExtractor::SaveLine(const std::vector<LineFeature::Ptr> &line_features){
        int fileIndex = 0;
        pcl::PLYWriter writer;
        for (int i = 0; i < line_features.size(); ++i) {
            LineFeature::Ptr line_feature = line_features[i];
            pcl::PointCloud<pcl::PointXYZ>::Ptr saved_cloud(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::copyPointCloud(*line_feature->cloud(), *saved_cloud);
            saved_cloud->width=saved_cloud->points.size();
            saved_cloud->height=1;
            for (size_t i = 0; i < saved_cloud->points.size(); ++i) {
                saved_cloud->points[i].z += z_up;
            }
            std::string fileName = generateFileName(1, fileIndex);
            pcl::PLYWriter writer;
            writer.write(config.output_file+fileName, *saved_cloud, true,false);
            fileIndex++;
        }
        std::cout<<"Line: "<<fileIndex<<std::endl;
    }

    void PLCExtractor::SaveGround(const pcl::PointCloud<pcl::PointXYZ>::Ptr &ground_cloud){
        pcl::PLYWriter writer;
        writer.write(config.output_file+"/ground/ground.ply", *ground_cloud, true,false);
        if (ground_cloud->points.size() > 0) {
            std::cout<<"Ground: "<<ground_cloud->points.size()<<" points"<<std::endl;
        } else {
            std::cout<<"Ground: 0"<<std::endl;
        }
    }
}