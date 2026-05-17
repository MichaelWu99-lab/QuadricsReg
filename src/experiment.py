from cProfile import label
import shutil
import numpy as np
import os
import yaml
import open3d as o3d
import random as randg
from scipy.spatial.transform import Rotation
from src.experiment_utils import *
from src.quadarics_transformation_estimation_utils import judge_Manhattan
from src.quadrics_modeling import quadrics_info_extration

KITTI_360_number_sequence = {
    0: "2013_05_28_drive_0000_sync",
    2: "2013_05_28_drive_0002_sync",
    3: "2013_05_28_drive_0003_sync",
    4: "2013_05_28_drive_0004_sync",
    5: "2013_05_28_drive_0005_sync",
    6: "2013_05_28_drive_0006_sync",
    7: "2013_05_28_drive_0007_sync",
    9: "2013_05_28_drive_0009_sync",
    10: "2013_05_28_drive_0010_sync",
}

Apollo_SouthBay_number_sequence = {
    # 1: "HighWay237/Apollo-SourthBay/MapData/HighWay237/2018-10-05",
    # 2: "SunnyvaleBigloop/Apollo-SourthBay/MapData/SunnyvaleBigloop/Bordeaux/2017-12-13",
    # 3: "SunnyvaleBigloop/Apollo-SourthBay/MapData/SunnyvaleBigloop/Caspian_and_Geneva/2017-12-13",
    # 4: "SunnyvaleBigloop/Apollo-SourthBay/MapData/SunnyvaleBigloop/Crossman/2017-12-13",
    # 5: "SunnyvaleBigloop/Apollo-SourthBay/MapData/SunnyvaleBigloop/Java/2017-12-13",
    # 6: "SunnyvaleBigloop/Apollo-SourthBay/MapData/SunnyvaleBigloop/Mathilda_Carribean/2017-12-14",
    # 7: "SunnyvaleBigloop/Apollo-SourthBay/MapData/SunnyvaleBigloop/Mathilda_Moffet/2017-12-28",
    # 8: "MathildaAVE/Apollo-SourthBay/MapData/MathildaAVE/2018-09-25",
    # 9: "SanJoseDowntown/Apollo-SourthBay/MapData/SanJoseDowntown/2018-10-02",
    # 10: "BaylandsToSeafood/Apollo-SourthBay/MapData/BaylandsToSeafood/2018-09-26",
    # 11: "ColumbiaPark/Apollo-SourthBay/MapData/ColumbiaPark/2018-09-21/1",
    # 12: "ColumbiaPark/Apollo-SourthBay/MapData/ColumbiaPark/2018-09-21/2",
    # 13: "ColumbiaPark/Apollo-SourthBay/MapData/ColumbiaPark/2018-09-21/3",
    # 14: "ColumbiaPark/Apollo-SourthBay/MapData/ColumbiaPark/2018-09-21/4",

    # 15: "HighWay237/Apollo-SourthBay/TrainData/HighWay237/2018-10-12",
    # 16: "MathildaAVE/Apollo-SourthBay/TrainData/MathildaAVE/2018-10-04",
    # 17: "SanJoseDowntown/Apollo-SourthBay/TrainData/SanJoseDowntown/2018-10-11",
    # 18: "BaylandsToSeafood/Apollo-SourthBay/TrainData/BaylandsToSeafood/2018-10-05",
    # 19: "ColumbiaPark/Apollo-SourthBay/TrainData/ColumbiaPark/2018-10-03",
    
    20: "HighWay237/Apollo-SourthBay/TestData/HighWay237/2018-10-12",
    21: "SunnyvaleBigloop/Apollo-SourthBay/TestData/SunnyvaleBigloop/2018-10-03",
    22: "MathildaAVE/Apollo-SourthBay/TestData/MathildaAVE/2018-10-12",
    23: "SanJoseDowntown/Apollo-SourthBay/TestData/SanJoseDowntown/2018-10-11/1",
    24: "SanJoseDowntown/Apollo-SourthBay/TestData/SanJoseDowntown/2018-10-11/2",
    25: "BaylandsToSeafood/Apollo-SourthBay/TestData/BaylandsToSeafood/2018-10-12",
    26: "ColumbiaPark/Apollo-SourthBay/TestData/ColumbiaPark/2018-10-11",
}

wayamo_data_indicator = '0'

self_collected_number_sequence = {
    0: "cscarvlp",
    1: "zhuoervlp",
    2: "leijunmid",
    3: "csindooravia",
    4: "yugangF6mid360",
    5: "hillmid360",
    6: "yugangUnderGroundmid360"
}

randg.seed(0)
def sample_random_yaw_trans(pcd, rotation_range=360,z_trans=0):
    T = np.eye(4)

    yaw_angle = rotation_range * np.pi / 180.0 * (randg.random() - 0.5) * 2

    R = np.array([
        [np.cos(yaw_angle), -np.sin(yaw_angle), 0],
        [np.sin(yaw_angle),  np.cos(yaw_angle), 0],
        [0,                 0,                1]
    ])

    T[:3, :3] = R

    T[:3, 3] = R.dot(-np.mean(pcd, axis=0))

    T[2, 3] += z_trans

    return T

def transform_point_cloud_withIndensity(point_cloud, transformation_matrix):
    """
    Apply a transformation matrix to the point cloud coordinates.
    Ignore the intensity in the transformation.
    """
    # Extract xyz coordinates (3xN) and append 1 for homogeneous coordinates (4xN)
    xyz = point_cloud[:, :3]
    ones = np.ones((xyz.shape[0], 1))
    xyz_homogeneous = np.hstack((xyz, ones))
    
    # Apply the transformation
    transformed_xyz = (transformation_matrix @ xyz_homogeneous.T).T[:, :3]
    
    # Combine transformed coordinates with intensity
    transformed_point_cloud = np.hstack((transformed_xyz, point_cloud[:, 3:4]))
    return transformed_point_cloud

def R2Angle(R, is_degree=True):
    """Compute the rotation angle of a rotation matrix."""
    a = (R[0, 0] + R[1, 1] + R[2, 2] - 1) * 0.5
    a = max(-1.0, min(a, 1.0))
    angle = np.arccos(a)
    if is_degree:
        angle = np.degrees(angle)
    return angle

def semantic_extraction(label_path, points, semantic_yaml_path, label_file_type='GT_semantics', DATA_SET_name='KITTI', dropout_ratio=0.0):

    semantic_yaml = yaml.safe_load(open(semantic_yaml_path, 'r'))
    semantic_dict = {}

    if label_file_type == 'Pre_semantics' or DATA_SET_name == 'KITTI':
        labels = load_data_labels(label_path)
        semi_labels,_ = set_label(labels,points)
    elif DATA_SET_name == 'KITTI-360':
        labels = load_data_labels_KITTI360(label_path)
        semi_labels = labels
    elif DATA_SET_name == 'nuScenes':
        labels = load_data_labels_nuSences(label_path)
        semi_labels,_ = set_label(labels,points)
    elif DATA_SET_name == 'waymo':
        labels = load_data_labels_waymo(label_path)
        semi_labels = labels

    # Robustness test: randomly drop a portion of semantic labels
    DROPOUT_LABEL = 999999
    if dropout_ratio > 0.0:
        num_points = len(semi_labels)
        num_drop = int(num_points * dropout_ratio)
        drop_indices = np.random.choice(num_points, size=num_drop, replace=False)
        semi_labels[drop_indices] = DROPOUT_LABEL

    semi_labels_unique = np.unique(semi_labels)
    
    for i,v in enumerate(semi_labels_unique):
        # Skip dropped labels
        if v == DROPOUT_LABEL:
            continue
            
        semi_index = semi_labels==v
        semi_points = points[semi_index]
        if semi_points.shape[0] < 20:
            continue

        semantic_dict[semantic_yaml["labels"][v]] = semi_points

    return semantic_dict

def semantic_extraction_(label_path, points, semantic_yaml_path, label_file_type='GT_semantics', DATA_SET_name='KITTI'):

    semantic_yaml = yaml.safe_load(open(semantic_yaml_path, 'r'))
    semantic_dict = {}

    if label_file_type == 'Pre_semantics' or DATA_SET_name == 'KITTI':
        labels = load_data_labels(label_path)
        semi_labels,_ = set_label(labels,points)
    elif DATA_SET_name == 'KITTI-360':
        labels = load_data_labels_KITTI360(label_path)
        semi_labels = labels
    elif DATA_SET_name == 'nuScenes':
        labels = load_data_labels_nuSences(label_path)
        semi_labels,_ = set_label(labels,points)
    elif DATA_SET_name == 'waymo':
        labels = load_data_labels_waymo(label_path)
        semi_labels = labels

    semi_labels_unique = np.unique(semi_labels)
    
    for i,v in enumerate(semi_labels_unique):
        semi_index = semi_labels==v
        semi_points = points[semi_index]
        if semi_points.shape[0] < 20:
            continue

        semantic_dict[semantic_yaml["labels"][v]] = semi_points

    return semantic_dict

class test_data:
    def __init__(self, DATA_SET_config,DATA_SET_prepare_type,mode,extraction_method,if_eval,dropout_ratio=0.0):
        self.DATA_SET_config = DATA_SET_config
        self.if_eval = if_eval
        self.extraction_method = extraction_method
        self.dropout_ratio = dropout_ratio
        self.DATA_SET_name = DATA_SET_config['DATA_SET']['name']
        self.DATA_SET_root = DATA_SET_config['DATA_SET']['data_root']
        self.label_dir = DATA_SET_config['DATA_SET']['label_dir']
        
        self.DATA_SET_prepare = DATA_SET_prepare_type
        self.mode = mode

        self.save_dir = f'./results/{self.DATA_SET_name}_{DATA_SET_prepare_type}/{mode}'
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.log_dir = f'./result_logs/{self.DATA_SET_name}_{DATA_SET_prepare_type}/{mode}'
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.benchmark_file = f'./benchmarks/{self.DATA_SET_name}_{DATA_SET_prepare_type}/test_{mode}.txt'
        with open(self.benchmark_file, 'r') as file:
            file.readline()  # skip header line
            self.benchmark = [
                [int(values[i]) if i < 4 else float(values[i]) for i in range(len(values))]
                for values in (line.strip().split() for line in file)
            ]

        self.index_load = 0

        self.success_flag_dict = {}
        self.time_reg_dict = {}
        self.representation_storage_dict = {}
        self.rot_error_dict = {}
        self.time_total_dict = {}
        self.time_extract_dict = {}
        self.time_model_dict = {}
        self.time_reconstruction_dict = {}
        self.time_prematch_dict = {}
        self.time_compatibility_check_dict = {}
        self.time_maxclique_dict = {}
        self.time_estimation_dict = {}
        self.trans_error_dict = {}
        self.correspondence_num_dict = {}
        self.correspondence_inlier_ratio_dict = {}
        self.correspondence_inlier_num_dict = {}

        self.sample_random_yaw_list = []

    def load_data(self):
        if self.index_load+1 > len(self.benchmark):
            return None, None
        if self.DATA_SET_name == 'KITTI':
            [sequence_source,timestamp_source,source_xyz,source_pcd,source_xyz_semantic_dict],[sequence_target,timestamp_target,target_xyz,target_pcd,target_xyz_semantic_dict],T_icp = self.test_KITTI(self.index_load)
        elif self.DATA_SET_name == 'KITTI-360':
            [sequence_source,timestamp_source,source_xyz,source_pcd,source_xyz_semantic_dict],[sequence_target,timestamp_target,target_xyz,target_pcd,target_xyz_semantic_dict],T_icp = self.test_KITTI_360(self.index_load)
        elif self.DATA_SET_name == 'Apollo-SouthBay':
            [sequence_source,timestamp_source,source_xyz,source_pcd,source_xyz_semantic_dict],[sequence_target,timestamp_target,target_xyz,target_pcd,target_xyz_semantic_dict],T_icp = self.test_Apollo_SouthBay(self.index_load)
        elif self.DATA_SET_name == 'nuScenes':
            [sequence_source,timestamp_source,source_xyz,source_pcd,source_xyz_semantic_dict],[sequence_target,timestamp_target,target_xyz,target_pcd,target_xyz_semantic_dict],T_icp = self.test_nuSences(self.index_load)
        elif self.DATA_SET_name == 'waymo':
            [sequence_source,timestamp_source,source_xyz,source_pcd,source_xyz_semantic_dict],[sequence_target,timestamp_target,target_xyz,target_pcd,target_xyz_semantic_dict],T_icp = self.test_waymo(self.index_load)
        elif self.DATA_SET_name == 'HeteroReg':
            [sequence_source,timestamp_source,source_xyz,source_pcd,source_xyz_semantic_dict],[sequence_target,timestamp_target,target_xyz,target_pcd,target_xyz_semantic_dict],T_icp = self.test_self_collected(self.index_load)

        self.sequence_current = sequence_source
        self.timestamp_source_current = timestamp_source
        self.source_pcd_current = source_pcd
        self.source_xyz = source_xyz
        self.source_xyz_semantic_dict = source_xyz_semantic_dict
        self.timestamp_target_current = timestamp_target
        self.target_pcd_current = target_pcd
        self.target_xyz = target_xyz
        self.target_xyz_semantic_dict = target_xyz_semantic_dict
        self.T_estimation = np.eye(4)
        self.augment_points_in_estimation = []
        if self.if_eval:
            self.T_gt = T_icp
        else:
            self.T_gt = []

        self.save_dir_current =  f'{self.save_dir}/{self.sequence_current}/{self.index_load}'
        if os.path.exists(self.save_dir_current):
            shutil.rmtree(self.save_dir_current)
        os.makedirs(self.save_dir_current)

        # result analysis
        self.time_reg = np.nan
        self.time_total = np.nan
        self.time_extract = np.nan
        self.time_model = np.nan
        self.time_reconstruction = np.nan
        self.time_prematch = np.nan
        self.time_compatibility_check = np.nan
        self.time_maxclique = np.nan
        self.time_estimation = np.nan
        self.rot_error = np.nan
        self.trans_error = np.nan
        self.correspondence_num = np.nan
        self.correspondence_inlier_ratio = np.nan
        self.correspondence_inlier_num = np.nan
        self.representation_storage = np.nan
        self.success_flag = np.nan

        print('##############################################')
        print(f"Load data: {self.index_load}/{len(self.benchmark)}, seq: {self.sequence_current}")

        self.index_load += 1
        
        return source_xyz,target_xyz

    def test_KITTI(self,index_benchmark):

        test_sample = self.benchmark[index_benchmark]
        sequence_source = str(test_sample[0]).zfill(2)
        timestamp_source = str(test_sample[1]).zfill(6)
        sequence_target = str(test_sample[2]).zfill(2)
        timestamp_target =  str(test_sample[3]).zfill(6)
        T = np.array(test_sample[4:]).reshape(4, 4)

        # source sequence must equal target sequence
        assert sequence_source == sequence_target

        label_source_path = f"{self.label_dir}/{sequence_source}/labels/{timestamp_source}.label"
        label_target_path = f"{self.label_dir}/{sequence_target}/labels/{timestamp_target}.label"

        # Check file existence
        if not os.path.exists(label_source_path) or not os.path.exists(label_target_path):
            return None, None, None

        pcd_source_file_bin = f"{self.DATA_SET_root}/sequences/{sequence_source}/velodyne/{str(timestamp_source).zfill(6)}.bin"
        pcd_target_file_bin = f"{self.DATA_SET_root}/sequences/{sequence_target}/velodyne/{str(timestamp_target).zfill(6)}.bin"

        if 'ode' in self.DATA_SET_prepare:
            # Apply yaw rotation to ODE data
            source_xyzi = np.fromfile(pcd_source_file_bin, dtype=np.float32).reshape(-1, 4)
            T_aug = sample_random_yaw_trans(source_xyzi[:, :3], rotation_range=45.0)
            source_xyzi = transform_point_cloud_withIndensity(source_xyzi, T_aug)
            source_xyzi = source_xyzi.astype(np.float32)

            source_pcd = o3d.geometry.PointCloud()
            source_xyz = source_xyzi[:, :3].astype(np.float32)
            source_pcd.points = o3d.utility.Vector3dVector(source_xyz)

            T = np.dot(T, np.linalg.inv(T_aug))

        else:
            source_xyz = np.fromfile(pcd_source_file_bin, dtype=np.float32).reshape(-1, 4)[:, :3]
            source_xyz = source_xyz.astype(np.float32)
            source_pcd = o3d.geometry.PointCloud()
            source_pcd.points = o3d.utility.Vector3dVector(source_xyz)

        target_xyz = np.fromfile(pcd_target_file_bin, dtype=np.float32).reshape(-1, 4)[:, :3]
        target_pcd = o3d.geometry.PointCloud()
        target_xyz = target_xyz.astype(np.float32)
        target_pcd.points = o3d.utility.Vector3dVector(target_xyz)

        # Voxel downsampling
        voxel_size = 0.05
        source_pcd_down = source_pcd.voxel_down_sample(voxel_size)
        target_pcd_down = target_pcd.voxel_down_sample(voxel_size)         
        reg = o3d.pipelines.registration.registration_icp(source_pcd_down, target_pcd_down, 0.2, T,
                                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=500))
        T_icp = reg.transformation

        if self.extraction_method in ['GT_semantics','Pre_semantics']:
            source_xyz_semantic_dict = semantic_extraction(label_source_path, source_xyz, self.DATA_SET_config['DATA_SET']['semantic_yaml'],label_file_type=self.extraction_method, DATA_SET_name=self.DATA_SET_name, dropout_ratio=self.dropout_ratio)
            target_xyz_semantic_dict = semantic_extraction(label_target_path, target_xyz, self.DATA_SET_config['DATA_SET']['semantic_yaml'],label_file_type=self.extraction_method, DATA_SET_name=self.DATA_SET_name, dropout_ratio=self.dropout_ratio)
        else:
            source_xyz_semantic_dict = {}
            target_xyz_semantic_dict = {}

        return [sequence_source,timestamp_source,source_xyz,source_pcd,source_xyz_semantic_dict],[sequence_target,timestamp_target,target_xyz,target_pcd,target_xyz_semantic_dict],T_icp

    def test_KITTI_360(self,index_benchmark):
        test_sample = self.benchmark[index_benchmark]
        sequence_source = str(test_sample[0])
        timestamp_source = str(test_sample[1]).zfill(10)
        sequence_target = str(test_sample[2])
        timestamp_target =  str(test_sample[3]).zfill(10)
        T = np.array(test_sample[4:]).reshape(4, 4)

        # source sequence must equal target sequence
        assert sequence_source == sequence_target

        sequence_source_name = KITTI_360_number_sequence[int(sequence_source)]
        sequence_target_name = KITTI_360_number_sequence[int(sequence_target)]

        label_source_path = f"{self.label_dir}/{sequence_source_name}/predictions/data/{timestamp_source}.label"
        label_target_path = f"{self.label_dir}/{sequence_target_name}/predictions/data/{timestamp_target}.label"

        if not os.path.exists(label_source_path) or not os.path.exists(label_target_path):
            return None, None, None

        pcd_source_file_bin = f"{self.DATA_SET_root}/data_3d_raw/{sequence_source_name}/velodyne_points/data/{timestamp_source}.bin"
        pcd_target_file_bin = f"{self.DATA_SET_root}/data_3d_raw/{sequence_target_name}/velodyne_points/data/{timestamp_target}.bin"
        if 'ode' in self.DATA_SET_prepare:
            # Apply yaw rotation to ODE data
            source_xyzi = np.fromfile(pcd_source_file_bin, dtype=np.float32).reshape(-1, 4)
            T_aug = sample_random_yaw_trans(source_xyzi[:, :3], rotation_range=45.0)
            source_xyzi = transform_point_cloud_withIndensity(source_xyzi, T_aug)
            source_xyzi = source_xyzi.astype(np.float32)

            source_pcd = o3d.geometry.PointCloud()
            source_xyz = source_xyzi[:, :3].astype(np.float32)
            source_pcd.points = o3d.utility.Vector3dVector(source_xyz)

            T = np.dot(T, np.linalg.inv(T_aug))

        else:
            source_xyz = np.fromfile(pcd_source_file_bin, dtype=np.float32).reshape(-1, 4)[:, :3]
            source_xyz = source_xyz.astype(np.float32)
            source_pcd = o3d.geometry.PointCloud()
            source_pcd.points = o3d.utility.Vector3dVector(source_xyz)

        target_xyz = np.fromfile(pcd_target_file_bin, dtype=np.float32).reshape(-1, 4)[:, :3]
        target_pcd = o3d.geometry.PointCloud()
        target_xyz = target_xyz.astype(np.float32)
        target_pcd.points = o3d.utility.Vector3dVector(target_xyz)

        # Voxel downsampling
        voxel_size = 0.05
        source_pcd_down = source_pcd.voxel_down_sample(voxel_size)
        target_pcd_down = target_pcd.voxel_down_sample(voxel_size)
        reg = o3d.pipelines.registration.registration_icp(source_pcd_down, target_pcd_down, 0.2, T,
                                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=500))

        T_icp = reg.transformation
        if self.extraction_method in ['Pre_semantics']:
            source_xyz_semantic_dict = semantic_extraction(label_source_path, source_xyz, self.DATA_SET_config['DATA_SET']['semantic_yaml'],label_file_type=self.extraction_method, DATA_SET_name=self.DATA_SET_name, dropout_ratio=self.dropout_ratio)
            target_xyz_semantic_dict = semantic_extraction(label_target_path, target_xyz, self.DATA_SET_config['DATA_SET']['semantic_yaml'],label_file_type=self.extraction_method, DATA_SET_name=self.DATA_SET_name, dropout_ratio=self.dropout_ratio)
        else:
            source_xyz_semantic_dict = {}
            target_xyz_semantic_dict = {}

        return [sequence_source,timestamp_source,source_xyz,source_pcd,source_xyz_semantic_dict],[sequence_target,timestamp_target,target_xyz,target_pcd,target_xyz_semantic_dict],T_icp

    def test_Apollo_SouthBay(self,index_benchmark):

        test_sample = self.benchmark[index_benchmark]
        sequence_source = str(test_sample[0])
        timestamp_source = str(test_sample[1])
        sequence_target = str(test_sample[2])
        timestamp_target =  str(test_sample[3])
        T = np.array(test_sample[4:]).reshape(4, 4)

        # source sequence must equal target sequence
        assert sequence_source == sequence_target

        sequence_source_name = Apollo_SouthBay_number_sequence[int(sequence_source)]
        sequence_target_name = Apollo_SouthBay_number_sequence[int(sequence_target)]

        label_source_path = f"{self.label_dir}/{sequence_source_name}/predictions/{timestamp_source}.label"
        label_target_path = f"{self.label_dir}/{sequence_target_name}/predictions/{timestamp_target}.label"

        if not os.path.exists(label_source_path) or not os.path.exists(label_target_path):
            return None, None, None

        pcd_source_file = f"{self.DATA_SET_root}/{sequence_source_name}/pcds/{timestamp_source}.pcd"
        pcd_target_file = f"{self.DATA_SET_root}/{sequence_target_name}/pcds/{timestamp_target}.pcd"

        source_pcd = o3d.t.io.read_point_cloud(pcd_source_file)
        source_xyz = source_pcd.point.positions.numpy().astype(np.float32)
        source_intensity = source_pcd.point.intensity.numpy().reshape(-1, 1)
        # Normalize intensity to [0,1]
        source_intensity = source_intensity / 255.0
        source_xyzi = np.hstack((source_xyz, source_intensity)).astype(np.float32)

        if 'ode' in self.DATA_SET_prepare:
            # Apply yaw rotation to ODE data
            T_aug = sample_random_yaw_trans(source_xyzi[:, :3], rotation_range=45.0)
            source_xyzi = transform_point_cloud_withIndensity(source_xyzi, T_aug)
            source_xyzi = source_xyzi.astype(np.float32)

            source_pcd = o3d.geometry.PointCloud()
            source_xyz = source_xyzi[:, :3].astype(np.float32)
            source_pcd.points = o3d.utility.Vector3dVector(source_xyz)

            T = np.dot(T, np.linalg.inv(T_aug))
        
        else:
            source_xyz = source_xyzi[:, :3].astype(np.float32)
            source_pcd = o3d.geometry.PointCloud()
            source_pcd.points = o3d.utility.Vector3dVector(source_xyz)

        target_pcd = o3d.t.io.read_point_cloud(pcd_target_file)
        target_xyz = target_pcd.point.positions.numpy().astype(np.float32)
        target_intensity = target_pcd.point.intensity.numpy().reshape(-1, 1)
        # Normalize intensity to [0,1]
        target_intensity = target_intensity / 255.0
        target_xyzi = np.hstack((target_xyz, target_intensity)).astype(np.float32)
        target_xyz = target_xyzi[:, :3].astype(np.float32)
        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points = o3d.utility.Vector3dVector(target_xyz)

        # Voxel downsampling
        voxel_size = 0.05
        source_pcd_down = source_pcd.voxel_down_sample(voxel_size)
        target_pcd_down = target_pcd.voxel_down_sample(voxel_size)
        reg = o3d.pipelines.registration.registration_icp(source_pcd_down, target_pcd_down, 0.2, T,
                                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=500))
        T_icp = reg.transformation

        if self.extraction_method in ['Pre_semantics']:
            source_xyz_semantic_dict = semantic_extraction(label_source_path, source_xyz, self.DATA_SET_config['DATA_SET']['semantic_yaml'],label_file_type=self.extraction_method, DATA_SET_name=self.DATA_SET_name, dropout_ratio=self.dropout_ratio)
            target_xyz_semantic_dict = semantic_extraction(label_target_path, target_xyz, self.DATA_SET_config['DATA_SET']['semantic_yaml'],label_file_type=self.extraction_method, DATA_SET_name=self.DATA_SET_name, dropout_ratio=self.dropout_ratio)
        else:
            source_xyz_semantic_dict = {}
            target_xyz_semantic_dict = {}

        return [sequence_source,timestamp_source,source_xyz,source_pcd,source_xyz_semantic_dict],[sequence_target,timestamp_target,target_xyz,target_pcd,target_xyz_semantic_dict],T_icp

    def test_nuSences(self,index_benchmark):

        test_sample = self.benchmark[index_benchmark]
        sequence_source = str(test_sample[0])
        sequence_source_name = sequence_source.zfill(6)
        timestamp_source = str(test_sample[1]).zfill(6)
        timestamp_source_name = sequence_source_name+'_'+timestamp_source
        sequence_target = str(test_sample[2])
        sequence_target_name = sequence_target.zfill(6)
        timestamp_target =  str(test_sample[3]).zfill(6)
        timestamp_target_name = sequence_target_name+'_'+timestamp_target
        T = np.array(test_sample[4:]).reshape(4, 4)

        # source sequence must equal target sequence
        assert sequence_source == sequence_target

        label_source_path = f"{self.label_dir}/{timestamp_source_name}.label"
        label_target_path = f"{self.label_dir}/{timestamp_target_name}.label"

        # Check file existence
        if not os.path.exists(label_source_path) or not os.path.exists(label_target_path):
            return None, None, None

        pcd_source_file_pcd = f"{self.DATA_SET_root}/pcd/{timestamp_source_name}.pcd"
        pcd_target_file_pcd = f"{self.DATA_SET_root}/pcd/{timestamp_target_name}.pcd"

        source_pcd = o3d.t.io.read_point_cloud(pcd_source_file_pcd)
        source_xyz = source_pcd.point.positions.numpy().astype(np.float32)
        source_intensity = source_pcd.point.intensity.numpy().reshape(-1, 1)
        source_xyzi = np.hstack((source_xyz, source_intensity)).astype(np.float32)

        if 'ode' in self.DATA_SET_prepare:
            # Apply yaw rotation to ODE data
            T_aug = sample_random_yaw_trans(source_xyzi[:, :3], rotation_range=45.0)
            source_xyzi = transform_point_cloud_withIndensity(source_xyzi, T_aug)
            source_xyzi = source_xyzi.astype(np.float32)

            source_pcd = o3d.geometry.PointCloud()
            source_xyz = source_xyzi[:, :3].astype(np.float32)
            source_pcd.points = o3d.utility.Vector3dVector(source_xyz)

            T = np.dot(T, np.linalg.inv(T_aug))

        else:
            source_xyz = source_xyzi[:, :3].astype(np.float32)
            source_pcd = o3d.geometry.PointCloud()
            source_pcd.points = o3d.utility.Vector3dVector(source_xyz)

        target_pcd = o3d.t.io.read_point_cloud(pcd_target_file_pcd)
        target_xyz = target_pcd.point.positions.numpy().astype(np.float32)
        target_intensity = target_pcd.point.intensity.numpy().reshape(-1, 1)
        target_xyzi = np.hstack((target_xyz, target_intensity)).astype(np.float32)
        target_xyz = target_xyzi[:, :3].astype(np.float32)
        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points = o3d.utility.Vector3dVector(target_xyz)

        # Voxel downsampling
        voxel_size = 0.05
        source_pcd_down = source_pcd.voxel_down_sample(voxel_size)
        target_pcd_down = target_pcd.voxel_down_sample(voxel_size)
        reg = o3d.pipelines.registration.registration_icp(source_pcd_down, target_pcd_down, 0.2, T,
                                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=500))
        T_icp = reg.transformation

        if self.extraction_method in ['GT_semantics','Pre_semantics']:
            source_xyz_semantic_dict = semantic_extraction(label_source_path, source_xyz, self.DATA_SET_config['DATA_SET']['semantic_yaml'],label_file_type=self.extraction_method, DATA_SET_name=self.DATA_SET_name, dropout_ratio=self.dropout_ratio)
            target_xyz_semantic_dict = semantic_extraction(label_target_path, target_xyz, self.DATA_SET_config['DATA_SET']['semantic_yaml'],label_file_type=self.extraction_method, DATA_SET_name=self.DATA_SET_name, dropout_ratio=self.dropout_ratio)
        else:
            source_xyz_semantic_dict = {}
            target_xyz_semantic_dict = {}

        return [sequence_source,timestamp_source,source_xyz,source_pcd,source_xyz_semantic_dict],[sequence_target,timestamp_target,target_xyz,target_pcd,target_xyz_semantic_dict],T_icp

    def test_waymo(self,index_benchmark):

        test_sample = self.benchmark[index_benchmark]
        sequence_source = str(test_sample[0])
        sequence_source_name = sequence_source.zfill(3)
        timestamp_source = str(test_sample[1]).zfill(3)
        timestamp_source_name = wayamo_data_indicator + sequence_source_name + timestamp_source
        sequence_target = str(test_sample[2])
        sequence_target_name = sequence_target.zfill(3)
        timestamp_target =  str(test_sample[3]).zfill(3)
        timestamp_target_name = wayamo_data_indicator + sequence_target_name + timestamp_target

        T = np.array(test_sample[4:]).reshape(4, 4)

        # source sequence must equal target sequence
        assert sequence_source == sequence_target

        label_source_path = f"{self.label_dir}/{timestamp_source_name}.label"
        label_target_path = f"{self.label_dir}/{timestamp_source_name}.label"

        # Check file existence
        if not os.path.exists(label_source_path) or not os.path.exists(label_target_path):
            return None, None, None

        pcd_source_file_pcd = f"{self.DATA_SET_root}/velodyne/{timestamp_source_name}.pcd"
        pcd_target_file_pcd = f"{self.DATA_SET_root}/velodyne/{timestamp_source_name}.pcd"

        source_pcd = o3d.t.io.read_point_cloud(pcd_source_file_pcd)
        source_xyz = source_pcd.point.positions.numpy().astype(np.float32)
        source_intensity = source_pcd.point.intensity.numpy().reshape(-1, 1)
        source_xyzi = np.hstack((source_xyz, source_intensity)).astype(np.float32)

        if 'ode' in self.DATA_SET_prepare:
            # Apply yaw rotation to ODE data
            T_aug = sample_random_yaw_trans(source_xyzi[:, :3], rotation_range=45.0)
            source_xyzi = transform_point_cloud_withIndensity(source_xyzi, T_aug)
            source_xyzi = source_xyzi.astype(np.float32)

            source_pcd = o3d.geometry.PointCloud()
            source_xyz = source_xyzi[:, :3].astype(np.float32)
            source_pcd.points = o3d.utility.Vector3dVector(source_xyz)

            T = np.dot(T, np.linalg.inv(T_aug))

        else:
            source_xyz = source_xyzi[:, :3].astype(np.float32)
            source_pcd = o3d.geometry.PointCloud()
            source_pcd.points = o3d.utility.Vector3dVector(source_xyz)

        target_pcd = o3d.t.io.read_point_cloud(pcd_target_file_pcd)
        target_xyz = target_pcd.point.positions.numpy().astype(np.float32)
        target_intensity = target_pcd.point.intensity.numpy().reshape(-1, 1)
        target_xyzi = np.hstack((target_xyz, target_intensity)).astype(np.float32)
        target_xyz = target_xyzi[:, :3].astype(np.float32)
        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points = o3d.utility.Vector3dVector(target_xyz)
        
        # Voxel downsampling
        voxel_size = 0.05
        source_pcd_down = source_pcd.voxel_down_sample(voxel_size)
        target_pcd_down = target_pcd.voxel_down_sample(voxel_size)
        reg = o3d.pipelines.registration.registration_icp(source_pcd_down, target_pcd_down, 0.2, T,
                                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=500))
        T_icp = reg.transformation
        
        if self.extraction_method in ['GT_semantics','Pre_semantics']:
            source_xyz_semantic_dict = semantic_extraction(label_source_path, source_xyz, self.DATA_SET_config['DATA_SET']['semantic_yaml'],label_file_type=self.extraction_method, DATA_SET_name=self.DATA_SET_name, dropout_ratio=self.dropout_ratio)
            target_xyz_semantic_dict = semantic_extraction(label_target_path, target_xyz, self.DATA_SET_config['DATA_SET']['semantic_yaml'],label_file_type=self.extraction_method, DATA_SET_name=self.DATA_SET_name, dropout_ratio=self.dropout_ratio)
        else:
            source_xyz_semantic_dict = {}
            target_xyz_semantic_dict = {}

        return [sequence_source,timestamp_source,source_xyz,source_pcd,source_xyz_semantic_dict],[sequence_target,timestamp_target,target_xyz,target_pcd,target_xyz_semantic_dict],T_icp

    def test_self_collected(self,index_benchmark):

        test_sample = self.benchmark[index_benchmark]
        sequence_source = str(test_sample[0])
        sequence_source_name = self_collected_number_sequence[int(sequence_source)]
        timestamp_source = str(test_sample[1])
        sequence_target = str(test_sample[2])
        sequence_target_name = self_collected_number_sequence[int(sequence_target)]
        timestamp_target =  str(test_sample[3])
        T = np.array(test_sample[4:]).reshape(4, 4)

        # source sequence must equal target sequence
        assert sequence_source == sequence_target

        pcd_source_file = f"{self.DATA_SET_root}/point_cloud/{sequence_source_name}/{timestamp_source}.pcd"
        pcd_target_file = f"{self.DATA_SET_root}/point_cloud/{sequence_target_name}/{timestamp_target}.pcd"
    
        if not os.path.exists(pcd_source_file) or not os.path.exists(pcd_target_file):
            return None, None, None

        source_pcd = o3d.t.io.read_point_cloud(pcd_source_file)
        source_xyz = source_pcd.point.positions.numpy().astype(np.float32)
        if 'intensity' not in source_pcd.point:
            print('intensity not in source_pcd.point')
            source_intensity = np.ones((source_pcd.point.positions.shape[0], 1), dtype=np.float32)
            source_xyzi = np.hstack((source_xyz, source_intensity)).astype(np.float32)
        else:
            source_intensity = source_pcd.point.intensity.numpy().reshape(-1, 1)
            # Normalize source intensity to [0,1]
            source_intensity = source_intensity / 255.0
            source_xyzi = np.hstack((source_xyz, source_intensity)).astype(np.float32)

        if 'ode' in self.DATA_SET_prepare:
            # Apply yaw rotation to ODE data
            T_aug = sample_random_yaw_trans(source_xyzi[:, :3], rotation_range=45.0)
            source_xyzi = transform_point_cloud_withIndensity(source_xyzi, T_aug)
            source_xyzi = source_xyzi.astype(np.float32)

            source_pcd = o3d.geometry.PointCloud()
            source_xyz = source_xyzi[:, :3].astype(np.float32)
            source_pcd.points = o3d.utility.Vector3dVector(source_xyz)

            T = np.dot(T, np.linalg.inv(T_aug))
            sample_random_yaw_info_temp = (
                list(map(int, test_sample[:4])) +         # four integer index values
                T_aug.flatten().tolist()          # flattened 4x4 matrix as length-16 list
            )
            self.sample_random_yaw_list.append(sample_random_yaw_info_temp)
        
        else:
            source_xyz = source_xyzi[:, :3].astype(np.float32)
            source_pcd = o3d.geometry.PointCloud()
            source_pcd.points = o3d.utility.Vector3dVector(source_xyz)

        target_pcd = o3d.t.io.read_point_cloud(pcd_target_file)
        target_xyz = target_pcd.point.positions.numpy().astype(np.float32)
        if 'intensity' not in target_pcd.point:
            print('intensity not in target.point')
            target_intensity = np.ones((target_pcd.point.positions.shape[0], 1), dtype=np.float32)
            target_xyzi = np.hstack((target_xyz, target_intensity)).astype(np.float32)
        else:
            target_intensity = target_pcd.point.intensity.numpy().reshape(-1, 1)
            # Normalize target intensity to [0,1]
            target_intensity = target_intensity / 255.0
            target_xyzi = np.hstack((target_xyz, target_intensity)).astype(np.float32)
        target_xyz = target_xyzi[:, :3].astype(np.float32)
        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points = o3d.utility.Vector3dVector(target_xyz)

        # Voxel downsampling
        voxel_size = 0.05
        source_pcd_down = source_pcd.voxel_down_sample(voxel_size)
        target_pcd_down = target_pcd.voxel_down_sample(voxel_size)
        reg = o3d.pipelines.registration.registration_icp(source_pcd_down, target_pcd_down, 0.2, T,
                                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=500))
        T_icp = reg.transformation

        source_xyz_semantic_dict = {}
        target_xyz_semantic_dict = {}

        return [sequence_source,timestamp_source,source_xyz,source_pcd,source_xyz_semantic_dict],[sequence_target,timestamp_target,target_xyz,target_pcd,target_xyz_semantic_dict],T_icp

    def save_representation(self,key_elements_quadrics_data_dict_source, key_elements_quadrics_data_point_augmented_dict_source,key_elements_quadrics_data_dict_target, key_elements_quadrics_data_point_augmented_dict_target):
        os.makedirs(f"{self.save_dir_current}/source")
        os.makedirs(f"{self.save_dir_current}/target")

        representation_storage_source = self.representation_save(key_elements_quadrics_data_dict_source, key_elements_quadrics_data_point_augmented_dict_source,'source')
        representation_storage_target = self.representation_save(key_elements_quadrics_data_dict_target, key_elements_quadrics_data_point_augmented_dict_target,'target')
        self.representation_storage = (representation_storage_source + representation_storage_target) / 2   
        
        self.reconstruction_save(key_elements_quadrics_data_dict_source, key_elements_quadrics_data_point_augmented_dict_source,'source')
        self.reconstruction_save(key_elements_quadrics_data_dict_target, key_elements_quadrics_data_point_augmented_dict_target,'target')

    def reconstruction_save(self,key_elements_quadrics_data_dict, key_elements_quadrics_data_point_augmented_dict,tag='source'):

        key_elements_mesh = o3d.geometry.TriangleMesh()
        key_elements_point = o3d.geometry.PointCloud()
        for element_semantic_type in key_elements_quadrics_data_dict.keys():
            if element_semantic_type == 'ground':
                continue
            key_elements_mesh_semantic_type = o3d.geometry.TriangleMesh()
            key_elements_point_semantic_type = o3d.geometry.PointCloud()
            for element_index in key_elements_quadrics_data_dict[element_semantic_type].keys():
                key_elements_mesh += key_elements_quadrics_data_dict[element_semantic_type][element_index]['mesh']
                key_elements_mesh_semantic_type += key_elements_quadrics_data_dict[element_semantic_type][element_index]['mesh']

                key_elements_quadrics_data_dict[element_semantic_type][element_index].pop("mesh")
            write_mesh_as_ply(f"{self.save_dir_current}/{tag}/reconstruction_mesh_{element_semantic_type}.ply", key_elements_mesh_semantic_type)
        write_mesh_as_ply(f"{self.save_dir_current}/{tag}/reconstruction_merge_mesh.ply", key_elements_mesh)

        ############################################
        point_augmented_pcd = o3d.geometry.PointCloud()
        if len(key_elements_quadrics_data_point_augmented_dict.keys()) > 0:
            for element_semantic_type in key_elements_quadrics_data_point_augmented_dict.keys():
                for element_index in key_elements_quadrics_data_point_augmented_dict[element_semantic_type].keys():
                    point_augmented_pcd.points.extend(o3d.utility.Vector3dVector(key_elements_quadrics_data_point_augmented_dict[element_semantic_type][element_index]['full_center'].reshape(1,3)))

            o3d.io.write_point_cloud(f"{self.save_dir_current}/{tag}/point_augmented.ply", point_augmented_pcd)

        return key_elements_mesh, point_augmented_pcd

    def reconstruction_save_with_manhattan_filter(self,key_elements_quadrics_data_dict, key_elements_quadrics_data_point_augmented_dict,tag='source', threshold_Manhattan=10):
        """Save meshes filtered by the Manhattan world assumption."""
        key_elements_quadrics_data_dict = quadrics_info_extration(key_elements_quadrics_data_dict.copy(), self.DATA_SET_config)

        ground_normal = key_elements_quadrics_data_dict.get('ground', {}).get('ground_normal', np.array([0, 0, 1]))

        key_elements_mesh = o3d.geometry.TriangleMesh()
        key_elements_point = o3d.geometry.PointCloud()
        for element_semantic_type in key_elements_quadrics_data_dict.keys():
            if element_semantic_type == 'ground':
                continue
            key_elements_mesh_semantic_type = o3d.geometry.TriangleMesh()
            key_elements_point_semantic_type = o3d.geometry.PointCloud()
            for element_index in key_elements_quadrics_data_dict[element_semantic_type].keys():
                element_data = key_elements_quadrics_data_dict[element_semantic_type][element_index]
                
                # Manhattan filter
                if 'quadrics_type' in element_data and 'decomposition_rotation' in element_data:
                    if not judge_Manhattan(ground_normal, element_data['quadrics_type'], element_data['decomposition_rotation'], threshold_Manhattan):
                        if 'mesh' in element_data:
                            element_data.pop("mesh")
                            print(f"judge_Manhattan: {element_data['quadrics_type']}")
                        continue

                if 'mesh' in element_data:
                    key_elements_mesh += element_data['mesh']
                    key_elements_mesh_semantic_type += element_data['mesh']

                if 'mesh' in element_data:
                    element_data.pop("mesh")
            write_mesh_as_ply(f"{self.save_dir_current}/{tag}/reconstruction_mesh_{element_semantic_type}.ply", key_elements_mesh_semantic_type)
        write_mesh_as_ply(f"{self.save_dir_current}/{tag}/reconstruction_merge_mesh.ply", key_elements_mesh)

        ############################################
        point_augmented_pcd = o3d.geometry.PointCloud()
        if len(key_elements_quadrics_data_point_augmented_dict.keys()) > 0:
            for element_semantic_type in key_elements_quadrics_data_point_augmented_dict.keys():
                for element_index in key_elements_quadrics_data_point_augmented_dict[element_semantic_type].keys():
                    point_augmented_pcd.points.extend(o3d.utility.Vector3dVector(key_elements_quadrics_data_point_augmented_dict[element_semantic_type][element_index]['full_center'].reshape(1,3)))

            o3d.io.write_point_cloud(f"{self.save_dir_current}/{tag}/point_augmented.ply", point_augmented_pcd)

        return key_elements_mesh, point_augmented_pcd

    def representation_save(self,key_elements_quadrics_data_dict, key_elements_quadrics_data_point_augmented_dict,tag='source'):

        representation_path = f"{self.save_dir_current}/{tag}/key_elements_quadrics_data.csv"
        with open(f"{representation_path}", 'w') as f:
            f.write("semantic,quadric_coeff,full_scale,full_rotation,full_center\n")
            for element_semantic_type in key_elements_quadrics_data_dict.keys():
                if element_semantic_type == 'ground':
                    f.write(f"{element_semantic_type},{key_elements_quadrics_data_dict[element_semantic_type]['ground_normal']}\n")
                    continue
                for element_index in key_elements_quadrics_data_dict[element_semantic_type].keys():
                    data = key_elements_quadrics_data_dict[element_semantic_type][element_index]
                    # transform rotation matrix to quaternion, [w, x, y, w]
                    full_quaternion = Rotation.from_matrix(data['full_rotation']).as_quat()
                    f.write(f"{element_semantic_type},{data['quadrics_coeff']},{data['full_scale']},{full_quaternion},{data['full_center']}\n")
            for element_semantic_type in key_elements_quadrics_data_point_augmented_dict.keys():
                for element_index in key_elements_quadrics_data_point_augmented_dict[element_semantic_type].keys():
                    data = key_elements_quadrics_data_point_augmented_dict[element_semantic_type][element_index]
                    full_quaternion = np.zeros(4)
                    f.write(f"{element_semantic_type},{data['quadrics_coeff']},{data['full_scale']},{full_quaternion},{data['full_center']}\n")

        representation_storage = os.path.getsize(representation_path)
        return representation_storage
        
    def save_registration(self):

        o3d.io.write_point_cloud(f"{self.save_dir_current}/source.ply", self.source_pcd_current)
        o3d.io.write_point_cloud(f"{self.save_dir_current}/target.ply", self.target_pcd_current)

        np.savetxt(f"{self.save_dir_current}/T_estimation.txt",self.T_estimation, fmt='%f', delimiter=' ')
        self.source_pcd_current.transform(self.T_estimation)
        self.source_pcd_current.paint_uniform_color([1, 0.706, 0])
        self.target_pcd_current.paint_uniform_color([0, 0.651, 0.929])
        self.pcd_combined = self.source_pcd_current + self.target_pcd_current
        o3d.io.write_point_cloud(f"{self.save_dir_current}/pcd_registration.ply", self.pcd_combined)

        self.augment_pcd_in_estimation = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(self.augment_points_in_estimation))
        o3d.io.write_point_cloud(f"{self.save_dir_current}/augment_src_pcd_in_estimation.ply",self.augment_pcd_in_estimation)

        print(f'    time_total: {self.time_total:4f}, time_reg: {self.time_reg:4f}, storage: {self.representation_storage/1024:0f}KB')

    def compute_pose_error(self):

        rot_error = np.dot(self.T_gt[:3, :3].T, self.T_estimation[:3, :3])
        trans_error = np.dot(self.T_gt[:3, :3].T, (self.T_estimation[:3, 3] - self.T_gt[:3, 3]))

        self.rot_error = abs(R2Angle(rot_error, True))
        self.trans_error = np.linalg.norm(trans_error)

        if self.rot_error < 5 and self.trans_error < 2:
            self.success_flag = 1
        else:
            self.success_flag = 0

    def computer_correspondence_inlier(self,quadrics_correspondences_with_info_all, threshold_correspondence_inlier_distance):
        correspondence_list = list(quadrics_correspondences_with_info_all.keys())
        self.correspondence_num = len(correspondence_list)

        centers_source = np.array([
            quadrics_correspondences_with_info_all[correspondence][0]["full_center"]
            for correspondence in correspondence_list
        ])
        centers_target = np.array([
            quadrics_correspondences_with_info_all[correspondence][1]["full_center"]
            for correspondence in correspondence_list
        ])

        centers_transformed = (self.T_gt[0:3,0:3]@ centers_source.T).T + self.T_gt[0:3,3].T

        distances = np.linalg.norm(centers_transformed - centers_target, axis=1)

        inliers_mask = distances < threshold_correspondence_inlier_distance
        self.correspondence_inlier_num = np.sum(inliers_mask)

        total_points = len(centers_source)
        self.correspondence_inlier_ratio = self.correspondence_inlier_num / total_points if total_points > 0 else 0

        print(f"    corr: {self.correspondence_num}, corr_inlier_ratio: {self.correspondence_inlier_ratio:4f}, corr_inlier_num: {self.correspondence_inlier_num}")

    def log_analysis_results(self):

        np.savetxt(f"{self.save_dir_current}/T_gt.txt",self.T_gt, fmt='%f', delimiter=' ')

        self.compute_pose_error()

        print(f'  {self.success_flag}, rot_error: {self.rot_error:4f}, trans_error: {self.trans_error:4f}')

        if self.sequence_current not in self.time_reg_dict.keys():
            
            self.success_flag_dict[self.sequence_current] = []
            self.rot_error_dict[self.sequence_current] = []
            self.trans_error_dict[self.sequence_current] = []
            self.time_total_dict[self.sequence_current] = []
            self.time_reg_dict[self.sequence_current] = []
            self.time_extract_dict[self.sequence_current] = []
            self.time_model_dict[self.sequence_current] = []
            self.time_reconstruction_dict[self.sequence_current] = []
            self.time_prematch_dict[self.sequence_current] = []
            self.time_compatibility_check_dict[self.sequence_current] = []
            self.time_maxclique_dict[self.sequence_current] = []
            self.time_estimation_dict[self.sequence_current] = []
            self.representation_storage_dict[self.sequence_current] = []
            self.correspondence_num_dict[self.sequence_current] = []
            self.correspondence_inlier_ratio_dict[self.sequence_current] = []
            self.correspondence_inlier_num_dict[self.sequence_current] = []

        self.success_flag_dict[self.sequence_current].append(self.success_flag)
        self.rot_error_dict[self.sequence_current].append(self.rot_error)
        self.trans_error_dict[self.sequence_current].append(self.trans_error)
        self.time_total_dict[self.sequence_current].append(self.time_total)
        self.time_reg_dict[self.sequence_current].append(self.time_reg)
        self.time_extract_dict[self.sequence_current].append(self.time_extract)
        self.time_model_dict[self.sequence_current].append(self.time_model)
        self.time_reconstruction_dict[self.sequence_current].append(self.time_reconstruction)
        self.time_prematch_dict[self.sequence_current].append(self.time_prematch)
        self.time_compatibility_check_dict[self.sequence_current].append(self.time_compatibility_check)
        self.time_maxclique_dict[self.sequence_current].append(self.time_maxclique)
        self.time_estimation_dict[self.sequence_current].append(self.time_estimation)
        self.representation_storage_dict[self.sequence_current].append(self.representation_storage)
        self.correspondence_num_dict[self.sequence_current].append(self.correspondence_num)
        self.correspondence_inlier_ratio_dict[self.sequence_current].append(self.correspondence_inlier_ratio)
        self.correspondence_inlier_num_dict[self.sequence_current].append(self.correspondence_inlier_num)

        print(f"seq {self.sequence_current} mean metrics, success_rate: {np.mean(self.success_flag_dict[self.sequence_current]):4f}, rot_error: {np.mean(self.rot_error_dict[self.sequence_current]):4f}, trans_error: {np.mean(self.trans_error_dict[self.sequence_current]):4f}, time_total: {np.mean(self.time_total_dict[self.sequence_current]):4f}, time_reg: {np.mean(self.time_reg_dict[self.sequence_current]):4f}, time-e/m/r/p/c/m/e: {np.mean(self.time_extract_dict[self.sequence_current]):4f}, {np.mean(self.time_model_dict[self.sequence_current]):4f}, {np.mean(self.time_reconstruction_dict[self.sequence_current]):4f}, {np.mean(self.time_prematch_dict[self.sequence_current]):4f}, {np.mean(self.time_compatibility_check_dict[self.sequence_current]):4f}, {np.mean(self.time_maxclique_dict[self.sequence_current]):4f}, {np.mean(self.time_estimation_dict[self.sequence_current]):4f}, storage: {np.mean(self.representation_storage_dict[self.sequence_current])/1024:0f}KB, corr: {np.mean(self.correspondence_num_dict[self.sequence_current]):4f}, corr_inlier_ratio: {np.mean(self.correspondence_inlier_ratio_dict[self.sequence_current]):4f}, corr_inlier_num: {np.mean(self.correspondence_inlier_num_dict[self.sequence_current]):4f}")

    def save_analysis_results(self):

        print('##############################################')
        for sequence_current in self.time_reg_dict.keys():
            # log
            self.log_dir_current =  f'{self.log_dir}/{sequence_current}'
            if os.path.exists(self.log_dir_current):
                shutil.rmtree(self.log_dir_current)
            os.makedirs(self.log_dir_current)

            np.savetxt(f"{self.log_dir_current}/success_flag.txt",self.success_flag_dict[sequence_current], fmt='%f', delimiter=' ')
            np.savetxt(f"{self.log_dir_current}/rot_error.txt",self.rot_error_dict[sequence_current], fmt='%f', delimiter=' ')
            np.savetxt(f"{self.log_dir_current}/trans_error.txt",self.trans_error_dict[sequence_current], fmt='%f', delimiter=' ')
            np.savetxt(f"{self.log_dir_current}/time_total.txt",self.time_total_dict[sequence_current], fmt='%f', delimiter=' ')
            np.savetxt(f"{self.log_dir_current}/time_reg.txt",self.time_reg_dict[sequence_current], fmt='%f', delimiter=' ')
            np.savetxt(f"{self.log_dir_current}/time_extract.txt",self.time_extract_dict[sequence_current], fmt='%f', delimiter=' ')
            np.savetxt(f"{self.log_dir_current}/time_model.txt",self.time_model_dict[sequence_current], fmt='%f', delimiter=' ')
            np.savetxt(f"{self.log_dir_current}/time_reconstruction.txt",self.time_reconstruction_dict[sequence_current], fmt='%f', delimiter=' ')
            np.savetxt(f"{self.log_dir_current}/time_prematch.txt",self.time_prematch_dict[sequence_current], fmt='%f', delimiter=' ')
            np.savetxt(f"{self.log_dir_current}/time_compatibility_check.txt",self.time_compatibility_check_dict[sequence_current], fmt='%f', delimiter=' ')
            np.savetxt(f"{self.log_dir_current}/time_maxclique.txt",self.time_maxclique_dict[sequence_current], fmt='%f', delimiter=' ')
            np.savetxt(f"{self.log_dir_current}/time_estimation.txt",self.time_estimation_dict[sequence_current], fmt='%f', delimiter=' ')
            np.savetxt(f"{self.log_dir_current}/representation_storage.txt",self.representation_storage_dict[sequence_current], fmt='%f', delimiter=' ')
            np.savetxt(f"{self.log_dir_current}/correspondence_num.txt",self.correspondence_num_dict[sequence_current], fmt='%f', delimiter=' ')
            np.savetxt(f"{self.log_dir_current}/correspondence_inlier_ratio.txt",self.correspondence_inlier_ratio_dict[sequence_current], fmt='%f', delimiter=' ')
            np.savetxt(f"{self.log_dir_current}/correspondence_inlier_num.txt",self.correspondence_inlier_num_dict[sequence_current], fmt='%f', delimiter=' ')
            
            success_rate = np.mean(self.success_flag_dict[sequence_current])
            rot_error_mean = np.mean(self.rot_error_dict[sequence_current])
            trans_error_mean = np.mean(self.trans_error_dict[sequence_current])
            time_total_mean = np.mean(self.time_total_dict[sequence_current])
            time_reg_mean = np.mean(self.time_reg_dict[sequence_current])
            time_extract_mean = np.mean(self.time_extract_dict[sequence_current])
            time_model_mean = np.mean(self.time_model_dict[sequence_current])
            time_reconstruction_mean = np.mean(self.time_reconstruction_dict[sequence_current])
            time_prematch_mean = np.mean(self.time_prematch_dict[sequence_current])
            time_compatibility_check_mean = np.mean(self.time_compatibility_check_dict[sequence_current])
            time_maxclique_mean = np.mean(self.time_maxclique_dict[sequence_current])
            time_estimation_mean = np.mean(self.time_estimation_dict[sequence_current])
            representation_storage_mean = np.mean(self.representation_storage_dict[sequence_current])
            correspondence_num_mean = np.mean(self.correspondence_num_dict[sequence_current])
            correspondence_inlier_ratio_mean = np.mean(self.correspondence_inlier_ratio_dict[sequence_current])
            correspondence_inlier_num_mean = np.mean(self.correspondence_inlier_num_dict[sequence_current])

            print(f"seq {sequence_current} mean metrics, sample_num: {len(self.rot_error_dict[sequence_current])}, success_rate: {success_rate:4f}, rot_error: {rot_error_mean:4f}, trans_error: {trans_error_mean:4f}, time_total: {time_total_mean:4f}, time_reg: {time_reg_mean:4f}, time-e/m/r/p/c/m/e: {time_extract_mean:4f}, {time_model_mean:4f}, {time_reconstruction_mean:4f}, {time_prematch_mean:4f}, {time_compatibility_check_mean:4f}, {time_maxclique_mean:4f}, {time_estimation_mean:4f}, storage: {representation_storage_mean/1024:0f}KB, corr: {correspondence_num_mean:4f}, corr_inlier_ratio: {correspondence_inlier_ratio_mean:4f}, corr_inlier_num: {correspondence_inlier_num_mean:4f}")

        if self.sample_random_yaw_list != []:
            header_str = 'seq i seq_db j ' + ' '.join([f'mot{i+1}' for i in range(16)])
            np.savetxt(
                f"{self.save_dir}/ode_T_aug.txt",
                np.array(self.sample_random_yaw_list),
                fmt=['%d', '%d', '%d', '%d'] + ['%.6f'] * 16,
                delimiter=' ',
                header=header_str,
                comments=''  # suppress default '#' comment prefix
            )
            print(f"save ode_T_aug to {self.save_dir}/ode_T_aug.txt")
        

def write_obj_with_normals(filename, mesh):
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    with open(filename, "w") as f:
        for v in np.asarray(mesh.vertices):
            f.write("v {} {} {}\n".format(v[0], v[1], v[2]))

        for vn in np.asarray(mesh.vertex_normals):
            f.write("vn {} {} {}\n".format(vn[0], vn[1], vn[2]))

        for face in np.asarray(mesh.triangles):
            f.write("f {0}//{0} {1}//{1} {2}//{2}\n".format(face[0] + 1, face[1] + 1, face[2] + 1))

def write_mesh_as_ply(filename, mesh):
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    if not mesh.has_vertex_colors():
        mesh.paint_uniform_color([1.0, 1.0, 1.0])

    o3d.io.write_triangle_mesh(
        filename,
        mesh,
        write_ascii=True,
        write_vertex_normals=True,
        write_vertex_colors=True
    )

