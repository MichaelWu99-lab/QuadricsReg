import numpy as np
import open3d as o3d
import gtsam
from gtsam import Pose3,Point3
from scipy.linalg import svd
from scipy.spatial.transform import Rotation as R
from src.data_processing import quadrics_scale_identification,q_Q

def add_noise_to_rotation_matrix(R_init, noise_level_degrees=0.01):
    noise_level_radians = np.deg2rad(noise_level_degrees)
    noise_vector = noise_level_radians * np.random.randn(3)
    noise_rotation = R.from_rotvec(noise_vector)
    R_noise = noise_rotation.as_matrix()
    R_noisy = R_noise @ R_init
    return R_noisy

def add_noise_to_translation_vector(t_init, noise_level=0.01):
    t_noisy = t_init + noise_level * np.random.randn(3)
    return t_noisy

def compute_pose_error_2(T_true, T_est):
    T_err = np.linalg.inv(T_true) @ T_est
    trace = np.trace(T_err)
    input_value = trace/ 2 -1
    input_value = max(-1.0, min(1.0, input_value))

    rotation_error = np.arccos(input_value)
    rotation_error = np.degrees(rotation_error)

    translation_error = np.linalg.norm(T_err[:3, 3])

    return rotation_error, translation_error

def R2Angle(R, is_degree=True):
    """Compute the rotation angle of a rotation matrix."""
    a = (R[0, 0] + R[1, 1] + R[2, 2] - 1) * 0.5
    a = max(-1.0, min(a, 1.0))
    angle = np.arccos(a)
    if is_degree:
        angle = np.degrees(angle)
    return angle

def compute_pose_error(T_true, T_est):
    """Compute rotation (degrees) and translation error between two transforms."""
    rot_err = np.dot(T_true[:3, :3].T, T_est[:3, :3])
    trans_err = np.dot(T_true[:3, :3].T, (T_est[:3, 3] - T_true[:3, 3]))

    rot_err = abs(R2Angle(rot_err, True))
    trans_err = np.linalg.norm(trans_err)

    return rot_err,trans_err

def svdSE3(src, tgt):
    src_mean = np.mean(src, axis=1, keepdims=True)
    tgt_mean = np.mean(tgt, axis=1, keepdims=True)
    src_centered = src - src_mean
    tgt_centered = tgt - tgt_mean
    H = src_centered @ tgt_centered.T
    U, S, Vt = svd(H)
    R_ = Vt.T @ U.T
    if np.linalg.det(R_) < 0:
        Vt[2, :] *= -1
        R_ = Vt.T @ U.T
    T = np.eye(4)
    T[:3, :3] = R_
    T[:3, 3] = (tgt_mean - R_ @ src_mean).flatten()
    return T

def skew_symmetric(v):
    """ Returns the skew symmetric matrix of a vector """
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

def judge_Manhattan(ground_normal,semantic,tgt_rotation_decompsition,tolerance=10):
    if semantic in ["plane","elliptic_cylinder","cylinder","line"]:
    # if semantic in ["plane","line"]:
        if semantic == "plane":
            # Normal perpendicular or parallel to Z-axis
            normal = tgt_rotation_decompsition[:,0]
        elif semantic in ["elliptic_cylinder","cylinder"]:
            normal = tgt_rotation_decompsition[:,2]
        elif semantic == "line":
            normal = tgt_rotation_decompsition[:,2]

        normal = np.real(normal)
        # Angle with Z-axis
        angle = np.arccos(np.dot(normal, np.array([0,0,1])))
        # angle = np.arccos(np.dot(normal, np.array(ground_normal)))

        angle = np.degrees(angle)
        # Check if parallel or perpendicular to Z-axis
        if (angle < tolerance or \
        np.abs(angle - 180) < tolerance or \
        np.abs(angle - 90) < tolerance):
            return True
        else:
            return False
    
    return True

class Quadrics2QuadricsFactor(gtsam.CustomFactor):
    def __init__(self, key, src_center, tgt_center, src_quadrics_info,tgt_quadrics_info,noise_model,if_error_R=True,if_error_t=True):
        self.src_center = Point3(src_center)
        self.tgt_center = Point3(tgt_center)
        self.src_quadrics_info = src_quadrics_info
        self.tgt_quadrics_info = tgt_quadrics_info

        self.tgt_Q = quadrics_scale_identification(q_Q(self.tgt_quadrics_info['quadrics_coeff']), self.tgt_quadrics_info['quadrics_type'])[0]
        self.tgt_E = self.tgt_Q[0:3,0:3]
        self.tgt_l = self.tgt_Q[0:3,3]
        self.tgt_c_decompsition,self.tgt_rotation_decompsition = np.linalg.eig(self.tgt_E)
        # Sort descending
        index_descending = np.argsort(-self.tgt_c_decompsition)
        self.tgt_c_decompsition = self.tgt_c_decompsition[index_descending]
        self.tgt_C_decompsition = np.diag(self.tgt_c_decompsition)
        self.tgt_rotation_decompsition = self.tgt_rotation_decompsition[:,index_descending]
        self.tgt_It = self.tgt_quadrics_info['decomposition_It']
        
        self.tgt_rotation = self.tgt_quadrics_info['full_rotation'].reshape(3,3)

        # self.tgt_Ir = self.tgt_quadrics_info['decomposition_Ir']
        if self.tgt_quadrics_info['quadrics_type'] in ["ellipsoid","sphere"]:
            # self.tgt_Ir = np.array([0,0,1]) * self.tgt_quadrics_info['decomposition_Ir']
            self.tgt_Ir = np.array([0,0,0])
        elif self.tgt_quadrics_info['quadrics_type'] in ["elliptic_cylinder","cylinder","elliptic_cone","cone"]:
            self.tgt_Ir = np.array([0,0,1]) * self.tgt_quadrics_info['decomposition_Ir']
        elif self.tgt_quadrics_info['quadrics_type'] in ["line"]:
            self.tgt_Ir = self.tgt_quadrics_info['decomposition_Ir']
        elif self.tgt_quadrics_info['quadrics_type'] in ["plane"]:
            self.tgt_Ir = self.tgt_quadrics_info['decomposition_Ir']
            # self.tgt_rotation_decompsition[:,1:] = self.tgt_quadrics_info['full_rotation'].reshape(3,3)[:,0:2]
            # self.tgt_It = np.array([1,0.1,0.1])

        # self.tgt_Ir = np.array([0,0,0])

        self.src_Q = quadrics_scale_identification(q_Q(src_quadrics_info['quadrics_coeff']), src_quadrics_info['quadrics_type'])[0]
        self.src_E = self.src_Q[0:3,0:3]
        self.src_l = self.src_Q[0:3,3]
        self.src_c_decompsition,self.src_rotation_decompsition = np.linalg.eig(self.src_E)
        # Sort descending
        index_descending = np.argsort(-self.src_c_decompsition)
        # self.src_c_decompsition = self.src_c_decompsition[index_descending]
        # self.src_C_decompsition = np.diag(self.src_c_decompsition)
        self.src_rotation_decompsition = self.src_rotation_decompsition[:,index_descending]

        # Pseudo-inverse E based on quadrics type
        if self.tgt_quadrics_info['quadrics_type'] in ["ellipsoid","sphere"]:
            self.tgt_E_pseudo = self.tgt_quadrics_info['full_E']
            self.src_E_pseudo = self.src_quadrics_info['full_E']
        elif self.tgt_quadrics_info['quadrics_type'] in ["elliptic_cylinder","cylinder","elliptic_cone","cone"]:
            self.tgt_E_pseudo = self.tgt_quadrics_info['full_E']
            self.src_E_pseudo = self.src_quadrics_info['full_E']
        elif self.tgt_quadrics_info['quadrics_type'] in ["line"]:
            self.tgt_E_pseudo = self.tgt_quadrics_info['E']
            self.src_E_pseudo = self.src_quadrics_info['E']
        elif self.tgt_quadrics_info['quadrics_type'] in ["plane"]:
            self.tgt_E_pseudo = self.tgt_quadrics_info['E']
            self.src_E_pseudo = self.src_quadrics_info['E']

        # Error function
        def error_function(this: gtsam.CustomFactor, values: gtsam.Values, jacobians):

            # if self.tgt_quadrics_info['quadrics_type'] in ["line"]:
            #     print(1)

            pose = values.atPose3(key)
            R = pose.rotation().matrix()
            t = pose.translation()

            # Use tgt_center instead of tgt_l
            error_t = (self.tgt_rotation_decompsition.T @ (R @self.src_center + t) -self.tgt_rotation_decompsition.T@self.tgt_center)
            error_t = np.expand_dims(error_t*self.tgt_It,1)

            error_R = np.zeros(9)
            for i in range(3):
                if self.tgt_Ir[i] == 0:
                    continue
                idx = 3*i
                error_R[idx:idx+3] = np.cross(self.tgt_rotation_decompsition[:,i],R @ self.src_rotation_decompsition[:,i])
                # error_R[idx:idx+3] = np.cross(R @ self.src_rotation_decompsition[:,i],self.tgt_rotation_decompsition[:,i])
            error_R = np.expand_dims(error_R,1)

            error = np.concatenate((error_R,error_t),axis=0)

            if if_error_R == False:
                error = np.concatenate((np.zeros(error_R.shape),error_t),axis=0)

            if if_error_t == False:
                error = np.concatenate((error_R,np.zeros(error_t.shape)),axis=0)

            if jacobians is not None :
                H_pose = np.zeros((12, 6))

                for i in range(3):
                    if self.tgt_Ir[i] == 0:
                        continue
                    idx = 3*i
                    H_pose[idx:idx+3, :3] = (-skew_symmetric(self.tgt_rotation_decompsition[:,i]) @ R @ skew_symmetric(self.src_rotation_decompsition[:,i]))
                    # H_pose[idx:idx+3, :3] = - R @(skew_symmetric(np.cross(self.tgt_rotation_decompsition[:,i], self.src_rotation_decompsition[:,i])))

                if if_error_R == False:
                    H_pose[:9, :3] = np.zeros((9,3))

                H_pose[9:12, :3] = (-self.tgt_rotation_decompsition.T @ R @ skew_symmetric(self.src_center)).T @ np.diag(self.tgt_It)
                H_pose[9:12, 3:6] = (self.tgt_rotation_decompsition.T).T @ np.diag(self.tgt_It)

                jacobians[0] = H_pose
                
                if if_error_R == False:
                    jacobians[0][:9, :3] = np.zeros((9,3))

            return error

        super(Quadrics2QuadricsFactor, self).__init__(noise_model, [key], error_function)

class Ground2GroundFactor(gtsam.CustomFactor):
    def __init__(self, key, src_center, tgt_center,src_ground_normal,tgt_ground_normal,noise_model,if_error_R=True,if_error_t=True):

        self.src_center = Point3(src_center)
        self.tgt_center = Point3(tgt_center)

        src_ground_ratation = np.zeros((3,3))
        tgt_ground_ratation = np.zeros((3,3))
        src_ground_ratation[:,0] = src_ground_normal
        tgt_ground_ratation[:,0] = tgt_ground_normal

        self.tgt_It = np.array([1,0,0])
        self.tgt_rotation_decompsition = tgt_ground_ratation
        self.tgt_Ir = np.array([1,0,0])

        self.src_It = np.array([1,0,0])
        self.src_rotation_decompsition = src_ground_ratation
        self.src_Ir = np.array([1,0,0])

        # Error function
        def error_function(this: gtsam.CustomFactor, values: gtsam.Values, jacobians):

            pose = values.atPose3(key)
            R = pose.rotation().matrix()
            t = pose.translation()

            error_t = (self.tgt_rotation_decompsition.T @ (R @self.src_center + t) - self.tgt_rotation_decompsition.T @ self.tgt_center)
            error_t = np.expand_dims(error_t*self.tgt_It,1)

            error_R = np.zeros(9)
            for i in range(3):
                if self.tgt_Ir[i] == 0:
                    continue
                idx = 3*i
                error_R[idx:idx+3] = np.cross(self.tgt_rotation_decompsition[:,i],R @ self.src_rotation_decompsition[:,i])
            error_R = np.expand_dims(error_R,1)

            error = np.concatenate((error_R,error_t),axis=0)

            if if_error_R == False:
                error = np.concatenate((np.zeros(error_R.shape),error_t),axis=0)

            if if_error_t == False:
                error = np.concatenate((error_R,np.zeros(error_t.shape)),axis=0)

            # if self.src_quadrics_info['quadrics_type'] == "elliptic_cylinder":
            #     print(error)

            if jacobians is not None :
                H_pose = np.zeros((12, 6))

                for i in range(3):
                    if self.tgt_Ir[i] == 0:
                        continue
                    idx = 3*i
                    H_pose[idx:idx+3, :3] = (-skew_symmetric(self.tgt_rotation_decompsition[:,i]) @ R @ skew_symmetric(self.src_rotation_decompsition[:,i]))

                if if_error_R == False:
                    H_pose[:9, :3] = np.zeros((9,3))

                H_pose[9:12, :3] = (-self.tgt_rotation_decompsition.T @ R @ skew_symmetric(self.src_center)).T @ np.diag(self.tgt_It)
                H_pose[9:12, 3:6] = (self.tgt_rotation_decompsition.T).T @ np.diag(self.tgt_It)

                if if_error_R == False:
                    H_pose[:9, :3] = np.zeros((9,3))

                jacobians[0] = H_pose

            return error

        super(Ground2GroundFactor, self).__init__(noise_model, [key], error_function)

from scipy.spatial import cKDTree
def nearest_neighbors(source, target):
    tree = cKDTree(target)
    min_dist, indices = tree.query(source, k=1)
    return indices, min_dist

def error_vertification(vertification_transform,source_seg_label_list,target_seg_label_list,quadrics_info_source_stack,quadrics_info_target_stack,center_type_vertification,degenerate_quadrics_aid_vertification,ground_aid,source_ground_points,target_ground_points,source_ground_normal,target_ground_normal,threshold_Manhattan=10):

    error_R_list = []
    error_t_list = []
    error_vertification_quadrics_list = []
    
    for i in range(len(source_seg_label_list)):
        src_seg_label = source_seg_label_list[i]
        tgt_seg_label = target_seg_label_list[i]

        src_quadrics_info = quadrics_info_source_stack[src_seg_label]
        tgt_quadrics_info = quadrics_info_target_stack[tgt_seg_label]

        src_center = src_quadrics_info[center_type_vertification]
        tgt_center = tgt_quadrics_info[center_type_vertification]

        if src_quadrics_info['quadrics_type'] != "point" and tgt_quadrics_info['quadrics_type'] != "point":

            if (judge_Manhattan(target_ground_normal,tgt_quadrics_info['quadrics_type'],tgt_quadrics_info['decomposition_rotation'],threshold_Manhattan)==False) or (judge_Manhattan(source_ground_normal,src_quadrics_info['quadrics_type'],src_quadrics_info['decomposition_rotation'],threshold_Manhattan)==False):
                continue

        if src_quadrics_info['quadrics_type'] in ["point"] or tgt_quadrics_info['quadrics_type'] in ["point"]:
            error_t = np.linalg.norm(vertification_transform[:3,:3] @ src_center + vertification_transform[:3,3] - tgt_center) / 3
            error_t_list.append(error_t)
            error_quadrics = error_t
            error_vertification_quadrics_list.append(error_quadrics)
        elif degenerate_quadrics_aid_vertification:
            if src_quadrics_info['quadrics_type'] in ["plane"]:
            
                plane_points_augment_src = generate_square_on_plane(src_center,src_quadrics_info['decomposition_rotation'][:,0],np.sqrt(src_quadrics_info["volume"]))

                for i in range(plane_points_augment_src.shape[0]):

                    error_R,error_t = quadrics_error_computation(tgt_quadrics_info,src_quadrics_info,vertification_transform,src_center=plane_points_augment_src[i],tgt_center=tgt_center)
                    error_quadrics = error_R + error_t
                    error_t_list.append(error_t)
                    error_R_list.append(error_R)
                    error_quadrics = error_R + error_t
                    error_vertification_quadrics_list.append(error_quadrics)

            elif src_quadrics_info['quadrics_type'] in ["line"]:
                                        
                line_points_augment_src = generate_segment_on_line(src_center,src_quadrics_info['decomposition_rotation'][:,2],src_quadrics_info["scale"][2])
                for i in range(line_points_augment_src.shape[0]):
                    error_R,error_t = quadrics_error_computation(tgt_quadrics_info,src_quadrics_info,vertification_transform,src_center=line_points_augment_src[i],tgt_center=tgt_center)
                    error_quadrics = error_R + error_t
                    error_t_list.append(error_t)
                    error_R_list.append(error_R)
                    error_quadrics = error_R + error_t
                    error_vertification_quadrics_list.append(error_quadrics)

            elif src_quadrics_info['quadrics_type'] in ["cylinder","elliptic_cylinder"]:
                
                cylinder_points_augment_src = generate_segment_on_line(src_center,src_quadrics_info['decomposition_rotation'][:,2],src_quadrics_info["scale"][2])
                for i in range(cylinder_points_augment_src.shape[0]):
                    error_R,error_t = quadrics_error_computation(tgt_quadrics_info,src_quadrics_info,vertification_transform,src_center=cylinder_points_augment_src[i],tgt_center=tgt_center)
                    error_quadrics = error_R + error_t
                    error_t_list.append(error_t)
                    error_R_list.append(error_R)
                    error_quadrics = error_R + error_t
                    error_vertification_quadrics_list.append(error_quadrics)
            else:
                error_R,error_t = quadrics_error_computation(tgt_quadrics_info,src_quadrics_info,vertification_transform,src_center=src_center,tgt_center=tgt_center)
                error_t_list.append(error_t)
                error_R_list.append(error_R)
                error_quadrics = error_R + error_t
                error_vertification_quadrics_list.append(error_quadrics)
        else:
            error_R,error_t = quadrics_error_computation(tgt_quadrics_info,src_quadrics_info,vertification_transform,src_center=src_center,tgt_center=tgt_center)
            error_t_list.append(error_t)
            error_R_list.append(error_R)
            error_quadrics = error_R + error_t
            error_vertification_quadrics_list.append(error_quadrics)

    if ground_aid and tgt_quadrics_info['quadrics_type'] in ["plane"]:
        # Ground points
        tgt_Ir = np.array([1,0,0])
        tgt_It = np.array([1,0,0])
        source_ground_rotation = np.zeros((3,3))
        target_ground_rotation = np.zeros((3,3))
        source_ground_rotation[:,0] = source_ground_normal
        target_ground_rotation[:,0] = target_ground_normal

        for i in range(len(source_ground_points)):
            src_center = source_ground_points[i]
            tgt_center = target_ground_points[i]

            error_R = np.zeros(9)
            for i in range(3):
                if tgt_Ir[i] == 0:
                    continue
                idx = 3*i
                error_R[idx:idx+3] = np.cross(target_ground_rotation[:,i],vertification_transform[:3,:3] @ source_ground_rotation[:,i])
            error_R = np.expand_dims(error_R,1) / np.sum(tgt_Ir) if np.sum(tgt_Ir) != 0 else 0
            error_R = np.linalg.norm(error_R)

            src_center_trans = vertification_transform[:3,:3] @ src_center + vertification_transform[:3,3]
            
            error_t = target_ground_rotation.T @ src_center_trans - target_ground_rotation.T @ tgt_center
            error_t = np.expand_dims(error_t*tgt_It,1) / np.sum(tgt_It)
            
            error_t = np.linalg.norm(error_t)

            error_quadrics = error_R + error_t

            error_t_list.append(error_t)
            error_R_list.append(error_R)

            error_vertification_quadrics_list.append(error_quadrics)
    
    return error_vertification_quadrics_list,error_R_list,error_t_list

def quadrics_error_computation(tgt_quadrics_info,src_quadrics_info,vertification_transform,src_center='',tgt_center='',center_type=''):

    if src_center == '' or tgt_center == '':
        assert center_type != ''
        src_center = src_quadrics_info[center_type]
        tgt_center = tgt_quadrics_info[center_type]

    tgt_Q = quadrics_scale_identification(q_Q(tgt_quadrics_info['quadrics_coeff']), tgt_quadrics_info['quadrics_type'])[0]
    tgt_E = tgt_Q[0:3,0:3]
    tgt_l = tgt_Q[0:3,3]
    tgt_c_decompsition,tgt_rotation_decompsition = np.linalg.eig(tgt_E)
    index_descending = np.argsort(-tgt_c_decompsition)
    # tgt_c_decompsition = tgt_c_decompsition[index_descending]
    # tgt_C_decompsition = np.diag(tgt_c_decompsition)
    tgt_rotation_decompsition = tgt_rotation_decompsition[:,index_descending]
    tgt_It = tgt_quadrics_info['decomposition_It']

    tgt_rotation = tgt_quadrics_info['full_rotation'].reshape(3,3)

    src_Q = quadrics_scale_identification(q_Q(src_quadrics_info['quadrics_coeff']), src_quadrics_info['quadrics_type'])[0]
    src_E = src_Q[0:3,0:3]
    src_c_decompsition,src_rotation_decompsition = np.linalg.eig(src_E)
    index_descending = np.argsort(-src_c_decompsition)
    # src_c_decompsition = src_c_decompsition[index_descending]
    src_rotation_decompsition = src_rotation_decompsition[:,index_descending]

    if tgt_quadrics_info['quadrics_type'] in ["ellipsoid","sphere"]:
        # tgt_Ir = np.array([0,0,1]) * tgt_quadrics_info['decomposition_Ir']
        tgt_Ir = np.array([0,0,0])
    elif tgt_quadrics_info['quadrics_type'] in ["elliptic_cylinder","cylinder","elliptic_cone","cone"]:
        tgt_Ir = np.array([0,0,1]) * tgt_quadrics_info['decomposition_Ir']
    elif tgt_quadrics_info['quadrics_type'] in ["plane","line"]:
        tgt_Ir = tgt_quadrics_info['decomposition_Ir']

    # tgt_Ir = np.array([0,0,0])

    error_R = np.zeros(9)
    for i in range(3):
        if tgt_Ir[i] == 0:
            continue
        idx = 3*i
        error_R[idx:idx+3] = np.cross(tgt_rotation_decompsition[:,i],vertification_transform[:3,:3] @ src_rotation_decompsition[:,i])
    
    error_R = np.expand_dims(error_R,1) / np.sum(tgt_Ir) if np.sum(tgt_Ir) != 0 else 0
    error_R = np.linalg.norm(error_R)

    # Quadrics distance
    src_center_trans = vertification_transform[:3,:3] @ src_center + vertification_transform[:3,3]

    error_t = tgt_rotation_decompsition.T @ src_center_trans - tgt_rotation_decompsition.T@ tgt_center
    error_t = np.expand_dims(error_t*tgt_It,1) / np.sum(tgt_It)

    error_t = np.linalg.norm(error_t)

    return error_R,error_t

def trimmed_mean(arr):
    """
    Calculate the mean of an array after removing the maximum and minimum values.
    
    Parameters:
    arr (numpy.ndarray): The input array.
    
    Returns:
    float: The trimmed mean of the array.
    """
    if len(arr) <= 2:
        raise ValueError("Array must contain more than two elements")
    
    # Convert to numpy array if not already
    arr = np.asarray(arr)
    
    # Remove the maximum and minimum values
    trimmed_arr = arr[arr != arr.max()]
    trimmed_arr = trimmed_arr[trimmed_arr != arr.min()]
    
    # Check if trimming removed all elements
    if len(trimmed_arr) == 0:
        raise ValueError("All elements are removed after trimming")
    
    # Calculate the mean of the trimmed array
    mean_value = np.mean(trimmed_arr)
    
    return mean_value

def ground_info_gen(size,normal,groud_z=0):
    ground_point_0 = np.array([0,0,0])
    ground_point_1 = np.array([1,1,0])
    ground_point_2 = np.array([-1,1,0])
    ground_point_3 = np.array([1,-1,0])
    ground_point_4 = np.array([-1,-1,0])
    ground_points = np.array([ground_point_0,ground_point_1,ground_point_2,ground_point_3,ground_point_4]) *size
    ground_points = ground_points + np.array([0,0,groud_z])

    rotation = np.zeros((3,3))
    rotation[:,0] = np.array(normal)

    # ground_info = {}
    # ground_info["ground_points"] = ground_points
    # ground_info["rotation"] = rotation

    return ground_points,rotation


def robust_kernel(residual, hyper_parameter, kernel_type="dcs"):
    kernel_type_lower = kernel_type.lower()

    if kernel_type_lower == "huber":
        if np.abs(residual) <= hyper_parameter:
            return 0.5 * residual * residual
        else:
            return hyper_parameter * (np.abs(residual) - 0.5 * hyper_parameter)
    elif kernel_type_lower == "cauchy":
        return hyper_parameter * hyper_parameter * np.log(1.0 + (residual * residual) / (hyper_parameter * hyper_parameter))
    elif kernel_type_lower == "tukey":
        if np.abs(residual) <= hyper_parameter:
            rotation_temp = 1 - (residual / hyper_parameter) ** 2
            return (hyper_parameter * hyper_parameter / 6.0) * (1 - rotation_temp ** 3)
        else:
            return hyper_parameter * hyper_parameter / 6.0
    elif kernel_type_lower == "geman_mcclure":
        return (residual * residual) / (2.0 * (hyper_parameter * hyper_parameter + residual * residual))
    elif kernel_type_lower == "tls":
        return min(0.5 * residual * residual, 0.5 * hyper_parameter * hyper_parameter)
    elif kernel_type_lower == "dcs":
        return hyper_parameter * hyper_parameter * (1 - np.exp(-residual * residual / (2 * hyper_parameter * hyper_parameter)))
    else:
        print(f"Unknown robust kernel type: {kernel_type}")
        exit(-1)

def project_point_cloud_to_plane(point_cloud, point_on_plane,normal_vector):
    """Project a point cloud onto a plane and center the projection at a given point."""
    normal_vector = normal_vector / np.linalg.norm(normal_vector)

    def project_point_to_plane(point, normal_vector, point_on_plane):
        point = np.array(point)
        vector_to_plane = point - point_on_plane
        distance_to_plane = np.dot(vector_to_plane, normal_vector)
        projection = point - distance_to_plane * normal_vector
        return projection

    projected_points = np.array([project_point_to_plane(point, normal_vector, point_on_plane) for point in point_cloud])
    center_of_projected_points = np.mean(projected_points, axis=0)
    translation_vector = point_on_plane - center_of_projected_points
    aligned_projected_points = projected_points + translation_vector

    return aligned_projected_points

def generate_square_on_plane(point_on_plane,normal_vector, side_length):
    """Generate four vertices of a square centered at a point on a plane."""
    normal_vector = normal_vector / np.linalg.norm(normal_vector)

    if normal_vector[0] != 0 or normal_vector[1] != 0:
        v1 = np.array([-normal_vector[1], normal_vector[0], 0])
    else:
        v1 = np.array([0, -normal_vector[2], normal_vector[1]])

    v1 = v1 / np.linalg.norm(v1)
    v2 = np.cross(normal_vector, v1)
    v2 = v2 / np.linalg.norm(v2)

    half_side = side_length / 2

    p1 = point_on_plane + half_side * (v1 + v2)
    p2 = point_on_plane + half_side * (v1 - v2)
    p3 = point_on_plane + half_side * (-v1 + v2)
    p4 = point_on_plane + half_side * (-v1 - v2)

    return np.array([p1, p2, p3, p4,point_on_plane])

def generate_rectangle_on_plane(point_center, point_on_plane, normal_vector, length_dir, width_dir, length, width):
    """Generate four vertices of a rectangle centered at a point on a plane."""
    normal_vector = normal_vector / np.linalg.norm(normal_vector)

    length_dir = length_dir / np.linalg.norm(length_dir)
    width_dir = width_dir / np.linalg.norm(width_dir)

    half_length = length
    half_width = width

    p1 = point_center + half_length * length_dir + half_width * width_dir
    p2 = point_center + half_length * length_dir - half_width * width_dir
    p3 = point_center - half_length * length_dir + half_width * width_dir
    p4 = point_center - half_length * length_dir - half_width * width_dir

    return np.array([p1, p2, p3, p4,point_on_plane])

def generate_segment_on_line(point_on_line, direction_vector, segment_length):
    """Generate two endpoints of a line segment centered at a point."""
    direction_vector = direction_vector / np.linalg.norm(direction_vector)

    half_length = segment_length

    endpoint1 = point_on_line + half_length * direction_vector
    endpoint2 = point_on_line - half_length * direction_vector

    return np.array([endpoint1, endpoint2,point_on_line])


def compute_average_nearest_neighbor_distance(pcd1, pcd2,kernel="",hyper_parameter=1):
    pcd2_tree = o3d.geometry.KDTreeFlann(pcd2)

    distances = []
    for point in pcd1.points:
        [_, idx, _] = pcd2_tree.search_knn_vector_3d(point, 1)
        nearest_point = np.asarray(pcd2.points)[idx[0]]
        if kernel == "":
            distance = np.linalg.norm(point - nearest_point)
            distances.append(distance)
        else:
            distance = np.linalg.norm(point - nearest_point)
            distances.append(robust_kernel(distance,hyper_parameter,kernel))
    # return np.mean(distances)
    return distances

class Points2LineFactor(gtsam.CustomFactor):
    def __init__(self, key, src_center, tgt_center, src_quadrics_info,tgt_quadrics_info,noise_model):
        self.src_center = Point3(src_center)
        self.tgt_center = Point3(tgt_center)

        self.tgt_quadrics_info = tgt_quadrics_info

        self.tgt_Q = quadrics_scale_identification(q_Q(tgt_quadrics_info['quadrics_coeff']), tgt_quadrics_info['quadrics_type'])[0]
        self.tgt_E = self.tgt_Q[0:3,0:3]

        self.tgt_c_decompsition,self.tgt_rotation_decompsition = np.linalg.eig(self.tgt_E)
        # Sort descending
        index_descending = np.argsort(-self.tgt_c_decompsition)
        self.tgt_c_decompsition = self.tgt_c_decompsition[index_descending]
        self.tgt_rotation_decompsition = self.tgt_rotation_decompsition[:,index_descending]

        self.tgt_line_direction = np.expand_dims(self.tgt_rotation_decompsition[:,2],1)

        # Error function
        def error_function(this: gtsam.CustomFactor, values: gtsam.Values, jacobians):
            pose = values.atPose3(key)
            R = pose.rotation().matrix()
            t = pose.translation()
            error = (np.eye(3) - self.tgt_line_direction@self.tgt_line_direction.T)@np.expand_dims((R@self.src_center+t - self.tgt_center),1)

            if jacobians is not None :
                H_pose = np.zeros((3, 6))

                H_pose[:3, :3] = -R@skew_symmetric(self.src_center)
                H_pose[:3, 3:6] = np.eye(3)
                H_pose = (np.eye(3) - self.tgt_line_direction@self.tgt_line_direction.T)@H_pose

                jacobians[0] = H_pose

            return error

        super(Points2LineFactor, self).__init__(noise_model, [key], error_function)

class Points2PlaneFactor(gtsam.CustomFactor):
    def __init__(self, key, src_center, tgt_center, src_quadrics_info,tgt_quadrics_info,noise_model):
        self.src_center = Point3(src_center)
        self.tgt_center = Point3(tgt_center)

        self.tgt_quadrics_info = tgt_quadrics_info

        self.tgt_Q = quadrics_scale_identification(q_Q(tgt_quadrics_info['quadrics_coeff']), tgt_quadrics_info['quadrics_type'])[0]
        self.tgt_E = self.tgt_Q[0:3,0:3]

        self.tgt_c_decompsition,self.tgt_rotation_decompsition = np.linalg.eig(self.tgt_E)
        # Sort descending
        index_descending = np.argsort(-self.tgt_c_decompsition)
        self.tgt_c_decompsition = self.tgt_c_decompsition[index_descending]
        self.tgt_rotation_decompsition = self.tgt_rotation_decompsition[:,index_descending]

        self.tgt_plane_normal = np.expand_dims(self.tgt_rotation_decompsition[:,0],1)

        # Error function
        def error_function(this: gtsam.CustomFactor, values: gtsam.Values, jacobians):
            pose = values.atPose3(key)
            R = pose.rotation().matrix()
            t = pose.translation()
            error = np.expand_dims((R@self.src_center+t - self.tgt_center),1).T @ self.tgt_plane_normal

            if jacobians is not None :
                H_pose = np.zeros((3, 6))

                H_pose[:3, :3] = -R@skew_symmetric(self.src_center)
                H_pose[:3, 3:6] = np.eye(3)
                H_pose = self.tgt_plane_normal.T @ H_pose

                jacobians[0] = H_pose

            return error

        super(Points2PlaneFactor, self).__init__(noise_model, [key], error_function)

class Points2PointsFactor(gtsam.CustomFactor):
    def __init__(self, key, src_center, tgt_center, src_quadrics_info,tgt_quadrics_info,noise_model,if_error_R=True,if_error_t=True):
        self.src_center = Point3(src_center)
        self.tgt_center = Point3(tgt_center)

        # Error function
        def error_function(this: gtsam.CustomFactor, values: gtsam.Values, jacobians):
            pose = values.atPose3(key)
            R = pose.rotation().matrix()
            t = pose.translation()
            error = np.expand_dims((R@self.src_center+t - self.tgt_center),1)

            if jacobians is not None :
                H_pose = np.zeros((3, 6))

                H_pose[:3, :3] = -R@skew_symmetric(self.src_center)
                H_pose[:3, 3:6] = np.eye(3)

                jacobians[0] = H_pose

            return error

        super(Points2PointsFactor, self).__init__(noise_model, [key], error_function)

def chamfer_distance(points1, points2):

    # Calculate pairwise distances
    diff1 = np.expand_dims(points1, axis=1) - np.expand_dims(points2, axis=0)
    diff2 = np.expand_dims(points2, axis=1) - np.expand_dims(points1, axis=0)
    
    dist1 = np.min(np.sum(np.square(diff1), axis=-1), axis=1)
    dist2 = np.min(np.sum(np.square(diff2), axis=-1), axis=1)
    
    chamfer_dist = np.mean(dist1) + np.mean(dist2)
    
    return chamfer_dist

def info_resacle_pca(info,quadrics_type=['plane','sphere','ellipsoid','line']):
    # Remove 3-sigma factor, keep only sigma

    if info['quadrics_type'] in quadrics_type:
        info['decomposition_scale'] = info['decomposition_scale'] / np.sqrt(3)
        info['full_scale'] = info['full_scale'] / np.sqrt(3)
        info['full_E'] = info['full_E'] / 3
        
        if info['quadrics_type'] in ['plane']:
            info['volume'] = info['volume'] / 3
        elif info['quadrics_type'] in ['sphere','ellipsoid']:
            info['volume'] = info['volume'] / (3*np.sqrt(3))
        elif info['quadrics_type'] in ['line']:
            info['volume'] = info['volume'] / np.sqrt(3)
    return info