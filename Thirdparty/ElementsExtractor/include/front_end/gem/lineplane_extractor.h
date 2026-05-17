/**
** Created by Zhijian QIAO.
** UAV Group, Hong Kong University of Science and Technology
** email: zqiaoac@connect.ust.hk
**/

#ifndef SRC_LINEPLANE_EXTRACTOR_H
#define SRC_LINEPLANE_EXTRACTOR_H

#include <vector>
#include <memory>
#include <mutex>
#include <iomanip>

#include <Eigen/Core>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_line.h>
#include <pcl/sample_consensus/sac_model_cylinder.h>
#include <pcl/io/ply_io.h>

#include <pcl/segmentation/extract_clusters.h>
// #include "dataset/kitti_utils.h"
#include "gemodel.h"
#include "voxel.h"
#include "front_end/gem/clustering.h"

namespace g3reg {

    typedef struct {
        std::vector<LineFeature::Ptr> lines;
        std::vector<SurfaceFeature::Ptr> planes;
        std::vector<ClusterFeature::Ptr> clusters;
        std::vector<SurfaceFeature::Ptr> ground;    
    } FeatureSet;

    class PLCExtractor {
    public:
        typedef std::shared_ptr<PLCExtractor> Ptr;

        PLCExtractor() = default;

        ~PLCExtractor() = default;

        std::string generateFileName(int a, int index){        
            std::ostringstream oss;
            std::string s;
            if(a == 1)
                s = "line/line(";
            else if (a == 2)
                s = "cluster/cluster(";
            else if (a == 3)
                s = "plane/plane(";
            oss << index << ")" << ".ply";
            return "/" + s + oss.str();
        };

        void ExtractFeature(const pcl::PointCloud<pcl::PointXYZ>::Ptr &sem_cloud,
                            std::vector<SurfaceFeature::Ptr> &ground_features,
                            std::vector<LineFeature::Ptr> &line_features,
                            std::vector<SurfaceFeature::Ptr> &surface_features,
                            std::vector<ClusterFeature::Ptr> &cluster_features,
                            const string &seg_type="all");

        void ExtractPole(std::vector<ClusterFeature::Ptr> &cluster_features,
                         std::vector<LineFeature::Ptr> &line_features);

        void ExtractPlanes(std::vector<ClusterFeature::Ptr> &cluster_features,
                           std::vector<SurfaceFeature::Ptr> &plane_features);

        void SaveCluster(const std::vector<ClusterFeature::Ptr> &cluster_features);

        void SaveSurface(const std::vector<SurfaceFeature::Ptr> &surface_features);

        void SaveLine(const std::vector<LineFeature::Ptr> &line_features);

        void SaveGround(const pcl::PointCloud<pcl::PointXYZ>::Ptr &ground_cloud);

        void MergePlanes(std::vector<SurfaceFeature::Ptr> &surface_features);

        void FilterSurface(std::vector<SurfaceFeature::Ptr> &surface_features, int min_points);

        void reset();

        const VoxelMap &getVoxels() const {
            return voxel_map;
        }
        string num;
        float z_up;
    private:
        VoxelMap voxel_map;
    };

    void TransformToEllipsoid(const FeatureSet &featureSet, std::vector<std::vector<QuadricFeature::Ptr>> &ellipsoids);

} // namespace g3reg

#endif //SRC_LINEPLANE_EXTRACTOR_H
