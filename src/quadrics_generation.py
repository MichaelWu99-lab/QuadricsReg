import numpy as np
import open3d as o3d
import os
from scipy.spatial import ConvexHull
from src.data_processing import *


def quadrics_generation_from_pca(points, shape):

    points = np.squeeze(points)

    points = points.T  # shape (3, N)

    T = np.eye(4)

    # translation
    center = np.mean(points, axis=1)  # mean over N, result shape (3,)
    T_d = np.eye(4)
    T_d[0:3, 3] = center
    T = T_d @ T

    # PCA
    points_centered = points - center[:, None]
    cov = (points_centered @ points_centered.T) / points_centered.shape[1]
    U, S, Vt = np.linalg.svd(cov)

    # scales
    # sqrt(3*S)
    scales = np.sqrt(3 * S)
    # scales = S
    
    T_s = np.eye(4)
    T_s[0,0] = scales[0]
    T_s[1,1] = scales[1]
    T_s[2,2] = scales[2]
    T = T_s @ T

    # rotation
    T_r = np.eye(4)
    T_r[0:3,0:3] = U
    T = T_r @ T

    if shape in ["sphere","ellipsoid"]:
        I = np.diag([1.0,1.0,1.0,-1.0])
    elif shape in ["cylinder","elliptic_cylinder"]:
        I = np.diag([0.0,1.0,1.0,-1.0])
    elif shape in ["cone","elliptic_cone"]:
        I = np.diag([1.0,1.0,-1.0,0.0])
    elif shape in ["plane"]:
        I = np.diag([0.0,0.0,1.0,0.0])
    else:
        raise ValueError(f"Unsupported shape: {shape}")

    T_inv = np.linalg.inv(T)
    Q = T_inv.T @ I @ T_inv

    q = np.array([Q[0,0], Q[1,1], Q[2,2], Q[0,1], Q[0,2], Q[1,2], Q[0,3], Q[1,3], Q[2,3], Q[3,3]])

    T_s_inv = np.linalg.inv(T_s)
    C = T_s_inv.T @ I @ T_s_inv
    c = np.array([C[0,0], C[1,1], C[2,2], C[0,1], C[0,2], C[1,2], C[0,3], C[1,3], C[2,3], C[3,3]])

    q = np.expand_dims(q, axis=0)
    c = np.expand_dims(c, axis=0)
    C = np.expand_dims(C, axis=0)
    T_inv = np.expand_dims(T_inv, axis=0)

    return q, c, C, T_inv


def project_to_2d(point_cloud, normal):
    centroid = np.mean(point_cloud, axis=0)

    z_values = point_cloud @ normal
    z_centroid = (np.max(z_values) + np.min(z_values))/2
    centroid[2] = z_centroid

    centered_point_cloud = point_cloud - centroid

    normal = normal / np.linalg.norm(normal)
    projection_matrix = np.eye(3) - np.outer(normal, normal)
    points_2d = centered_point_cloud @ projection_matrix.T
    return points_2d[:, :2], centroid, normal

def compute_2d_obb(points_2d):
    hull = ConvexHull(points_2d)
    hull_points = points_2d[hull.vertices]
    
    min_area = np.inf
    best_box = None
    for i in range(len(hull_points)):
        p1 = hull_points[i]
        p2 = hull_points[(i + 1) % len(hull_points)]
        edge = p2 - p1
        edge_unit = edge / np.linalg.norm(edge)
        normal_unit = np.array([-edge_unit[1], edge_unit[0]])
        
        rotation_matrix = np.array([edge_unit, normal_unit])
        rotated_points = points_2d @ rotation_matrix.T
        min_x, min_y = rotated_points.min(axis=0)
        max_x, max_y = rotated_points.max(axis=0)
        
        area = (max_x - min_x) * (max_y - min_y)
        if area < min_area:
            min_area = area
            best_box = (min_x, min_y, max_x, max_y, rotation_matrix)
    
    min_x, min_y, max_x, max_y, rotation_matrix = best_box
    obb_corners_2d = np.array([
        [min_x, min_y],
        [max_x, min_y],
        [max_x, max_y],
        [min_x, max_y]
    ])
    obb_corners_2d = obb_corners_2d @ rotation_matrix
    scales_2d = np.array([max_x - min_x, max_y - min_y])
    
    return obb_corners_2d, scales_2d

def create_3d_obb(centroid, obb_corners_2d, normal, original_point_cloud, scales_2d,shape):
    z_values = original_point_cloud @ normal
    if shape == "plane":
        height = 0
    else:
        height = np.max(z_values) - np.min(z_values)
    min_z = -height / 2
    max_z = height / 2
    
    normal_unit = normal / np.linalg.norm(normal)
    obb_corners_2d_expanded = np.hstack([obb_corners_2d, np.zeros((obb_corners_2d.shape[0], 1))])
    obb_corners_3d_bottom = obb_corners_2d_expanded + min_z * normal_unit
    obb_corners_3d_top = obb_corners_2d_expanded + max_z * normal_unit
    
    obb_corners_3d = np.vstack([obb_corners_3d_bottom, obb_corners_3d_top])
    obb_corners_3d_bottom += centroid
    obb_corners_3d += centroid

    if shape in ["cone","elliptic_cone"]:
        obb_centroid = np.mean(obb_corners_3d_bottom, axis=0)
    else:
        obb_centroid = np.mean(obb_corners_3d, axis=0)

    # Calculate 3D rotation matrix directly from the OBB corners
    x_axis = obb_corners_3d_top[1] - obb_corners_3d_top[0]
    y_axis = obb_corners_3d_top[2] - obb_corners_3d_top[1]
    z_axis = normal_unit
    rotation_matrix_3d = np.vstack([x_axis / np.linalg.norm(x_axis), 
                                    y_axis / np.linalg.norm(y_axis), 
                                    z_axis]).T
    
    scales_3d = np.hstack([scales_2d, height])
    
    return obb_centroid, obb_corners_3d,rotation_matrix_3d, scales_3d

def quadrics_generation_from_obb(points, shape, axis_original):

    # Rotate points so that axis_original aligns with Z-axis
    normal_original = axis_original
    normal_z = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    R_originalToZ = rotation_matrix_a_to_b(normal_original, normal_z)  # must be in NumPy

    # Rotate the point cloud to align with Z
    points_numpy_z = points @ R_originalToZ.T  # shape: (N, 3)

    # Define the canonical quadric matrix I based on `shape`
    if shape in ["sphere", "ellipsoid"]:
        # Q for sphere/ellipsoid: diag(1,1,1,-1)
        I = np.diag([1.0, 1.0, 1.0, -1.0]).astype(np.float32)
    elif shape in ["cylinder", "elliptic_cylinder"]:
        # cylinder main axis is the largest eigenvalue direction => diag(1,1,0,-1)
        I = np.diag([1.0, 1.0, 0.0, -1.0]).astype(np.float32)
    elif shape in ["cone", "elliptic_cone"]:
        # cone => diag(1,1,-1,0)
        I = np.diag([1.0, 1.0, -1.0, 0.0]).astype(np.float32)
    elif shape in ["plane"]:
        # plane => diag(0,0,1,0)
        I = np.diag([0.0, 0.0, 1.0, 0.0]).astype(np.float32)
    else:
        raise ValueError(f"Unknown shape type: {shape}")

    # Compute 2D OBB in the Z-aligned space and then reconstruct the 3D OBB
    points_2d, centroid_2d, normal_2d = project_to_2d(points_numpy_z, normal_z)
    obb_corners_2d, scales_2d = compute_2d_obb(points_2d)
    (obb_centroid_z, 
     obb_corners_3d_z, 
     rotation_matrix_3d_z, 
     scales_3d_z) = create_3d_obb(
        centroid_2d, 
        obb_corners_2d, 
        normal_2d, 
        points_numpy_z, 
        scales_2d, 
        shape
    )

    # Transform back from Z-aligned space to the original space
    obb_centroid = obb_centroid_z @ R_originalToZ  # shape: (3,)
    obb_corners_3d = obb_corners_3d_z @ R_originalToZ  # shape: (8, 3) or similar

    # Combine rotation from Z-aligned to the original space
    rotation_matrix_3d = R_originalToZ.T @ rotation_matrix_3d_z
    scales_3d = scales_3d_z.copy()

    # Build the transformation T = T_d @ T_r @ T_s so that the final quadric is Q = T^-T @ I @ T^-1
    # Possibly adjust scales for certain shapes
    if shape == "plane":
        # in the original code: keep T_s full rank by forcing scales_3d[2] = 2
        scales_3d[2] = 2.0
    elif shape in ["cone", "elliptic_cone"]:
        # double the z-scale
        scales_3d[2] = scales_3d[2] * 2.0

    # T_s
    scales_half = scales_3d / 2.0
    T_s = np.eye(4, dtype=np.float32)
    T_s[0, 0] = scales_half[0]
    T_s[1, 1] = scales_half[1]
    T_s[2, 2] = scales_half[2]

    # T_r
    T_r = np.eye(4, dtype=np.float32)
    T_r[0:3, 0:3] = rotation_matrix_3d

    # T_d
    T_d = np.eye(4, dtype=np.float32)
    T_d[0:3, 3] = obb_centroid

    # Compose T
    # NOTE: the original code does T = T_s @ T, then T_r @ T, then T_d @ T
    # but ends up with the final T = T_d @ T_r @ T_s
    # We'll do the same order as the original final formula: T_d @ T_r @ T_s
    T = T_d @ T_r @ T_s  # shape: (4,4)

    # Compute quadric Q = T^-T * I * T^-1
    T_inv = np.linalg.inv(T)
    # T^-T is simply (T_inv).T
    Q = T_inv.T @ I @ T_inv
    # Force Q to be symmetric
    Q = 0.5 * (Q + Q.T)

    # Flatten out Q -> q = (Q00, Q11, Q22, Q01, Q02, Q12, Q03, Q13, Q23, Q33)
    q = np.array([
        Q[0, 0], Q[1, 1], Q[2, 2],
        Q[0, 1], Q[0, 2], Q[1, 2],
        Q[0, 3], Q[1, 3], Q[2, 3],
        Q[3, 3]
    ], dtype=np.float32)

    # Compute C = T_s^-T * I * T_s^-1  (the shape-only component)
    T_s_inv = np.linalg.inv(T_s)
    C_ = T_s_inv.T @ I @ T_s_inv
    # Flatten C the same way:
    c_ = np.array([
        C_[0, 0], C_[1, 1], C_[2, 2],
        C_[0, 1], C_[0, 2], C_[1, 2],
        C_[0, 3], C_[1, 3], C_[2, 3],
        C_[3, 3]
    ], dtype=np.float32)

    # Expand dims to match original return shapes (1, 10), etc.
    q = np.expand_dims(q, axis=0)      # shape => (1, 10)
    c_ = np.expand_dims(c_, axis=0)    # shape => (1, 10)
    C_ = np.expand_dims(C_, axis=0)    # shape => (1, 4, 4)
    T_inv = np.expand_dims(T_inv, axis=0)  # shape => (1, 4, 4)

    return q, c_, C_, T_inv


def quadrics_generation_line_from_pca(points, shape, axis_original):
    # If the original points shape is (1, N, 3), after squeeze it becomes (N,3)
    points = np.squeeze(points)  # Removes batch dimension if present
    # The original code does permute(1,0), which corresponds to transpose in NumPy
    # From (N,3) to (3,N)
    points = points.T  # shape now: (3, N)

    # Create a 4x4 identity matrix T
    T = np.eye(4)

    # Compute the centroid and apply translation
    center = np.mean(points, axis=1)  # mean along N, result shape (3,)
    T_d = np.eye(4)
    T_d[0:3, 3] = center
    T = T_d @ T

    # PCA
    # Center the points
    points_centered = points - center[:, None]  # (3,N) - (3,1) -> (3,N)
    N = points_centered.shape[1]
    # Compute covariance matrix (3x3)
    cov = (points_centered @ points_centered.T) / N

    # SVD decomposition
    U, S, Vt = np.linalg.svd(cov)
    # Sort singular values in ascending order
    index = np.argsort(S)
    U = U[:, index]
    S = S[index]

    # Scales
    scales = np.array([1.0, 1.0, 1.0])

    # Define I matrix based on shape
    I = np.diag([1.0,1.0,0.0,0.0])

    # Scale matrix T_s
    T_s = np.eye(4)
    T_s[0,0], T_s[1,1], T_s[2,2] = scales[0], scales[1], scales[2]
    T = T_s @ T

    # Rotation matrix T_r
    T_r = np.eye(4)
    T_r[0:3, 0:3] = U
    T = T_r @ T

    # Apply translation again (as in the original code)
    T_d = np.eye(4)
    T_d[0:3, 3] = center
    T = T_d @ T

    # Compute inverse of T
    T_inv = np.linalg.inv(T)

    # Compute Q
    Q = T_inv.T @ I @ T_inv
    # Make Q symmetric (to avoid numerical asymmetry)
    Q = (Q.T + Q) / 2.0

    # Extract q parameters from Q
    q = np.array([Q[0,0], Q[1,1], Q[2,2], Q[0,1], Q[0,2], Q[1,2], Q[0,3], Q[1,3], Q[2,3], Q[3,3]])

    # Compute C using the inverse of T_s
    T_s_inv = np.linalg.inv(T_s)
    C = T_s_inv.T @ I @ T_s_inv
    c = np.array([C[0,0], C[1,1], C[2,2], C[0,1], C[0,2], C[1,2], C[0,3], C[1,3], C[2,3], C[3,3]])

    # Add batch dimension as (1,10) similar to unsqueeze(0) in PyTorch
    q = np.expand_dims(q, axis=0)
    c = np.expand_dims(c, axis=0)
    C = np.expand_dims(C, axis=0)
    T_inv = np.expand_dims(T_inv, axis=0)

    return q, c, C, T_inv

def quadrics_generation_point(points):
    num = points.shape[0]
    quadrics_list = []
    centers_list = []
    for i in range(num):
        T = np.eye(4)
        T[0:3,3] = points[i,:]
        C = np.diag([1,1,1,0])
        T_inv = np.linalg.inv(T)
        Q = T_inv.T@C@T_inv
        q = Q_q(Q)
        quadrics_list.append(q)
        centers_list.append(points[i,:])
    return quadrics_list,centers_list

def quadrics_decomposition_info_extracting_pca(points, output, element_quadrics_type, trans_inv, C, mode="not"):
    # Cylinders, cones, and lines not supported in original comment.

    points = np.expand_dims(points.T, axis=0)

    # Check if points is (1, N, 3)
    assert points.shape[1] == 3 and points.shape[0] == 1
    
    for i in range(1):
        # Construct Q_each from output
        Q_each = np.array([[output[i, 0], output[i, 3], output[i, 4], output[i, 6]],
                           [output[i, 3], output[i, 1], output[i, 5], output[i, 7]],
                           [output[i, 4], output[i, 5], output[i, 2], output[i, 8]],
                           [output[i, 6], output[i, 7], output[i, 8], output[i, 9]]], dtype=np.float64)

        Q_each, _ = quadrics_scale_identification(Q_each, element_quadrics_type)

        if mode == "decomposition":
            # Decompose Q_each
            E_each = Q_each[0:3, 0:3]
            # np.linalg.eig returns eigenvalues as 1D array and eigenvectors as columns in V
            value_each, vector_pre_each = np.linalg.eig(E_each)
            # Assume eigenvalues are real; if not, take real parts:
            value_each = np.real(value_each)
            
            # Sort eigenvalues in descending order
            idx_pre_each = np.argsort(value_each)[::-1]  
            value_each_sorted = value_each[idx_pre_each]
            vector_each_sorted = vector_pre_each[:, idx_pre_each]

            Is_each, Ir_each, It_each = quadrics_judgment(value_each_sorted)

            if element_quadrics_type in ["cone", "elliptic_cone"]:
                scale_each_sorted = np.sqrt(np.abs(1.0 / (value_each_sorted + 1e-8)))
                # Normalize height
                scale_each_sorted = (scale_each_sorted / scale_each_sorted[2]) * Is_each
            else:
                scale_each_sorted = np.sqrt(np.abs(1.0 / (value_each_sorted + 1e-8))) * Is_each

            # Normalize rotation vectors
            norm_factor = np.linalg.norm(vector_each_sorted, axis=0, keepdims=True) + 1e-8
            rotation_each_sorted = (vector_each_sorted / norm_factor) * Ir_each

            scale_pre_each_sorted = scale_each_sorted
            rotation_pre_each_sorted = rotation_each_sorted
            Is_pre_each = Is_each
            Ir_pre_each = Ir_each
            It_pre_each = It_each

        else:
            # Using C[i]
            C_pre_each = C[i].copy()
            if element_quadrics_type in ["sphere", "ellipsoid", "cylinder", "elliptic_cylinder"]:
                C_pre_each = C_pre_each / (-C_pre_each[3,3])

            trans_r = trans_inv[i][0:3,0:3].T

            # Eigen decomposition on C_pre_each
            diag_vals = np.diag(C_pre_each)[:3]
            idx_pre_each = np.argsort(diag_vals)[::-1]
            value_pre_each_sorted = diag_vals[idx_pre_each]
            vector_pre_each_sorted = trans_r[:, idx_pre_each]

            Is_pre_each, Ir_pre_each, It_pre_each = quadrics_judgment(value_pre_each_sorted)

            if element_quadrics_type in ["cone", "elliptic_cone"]:
                scale_pre_each_sorted = np.sqrt(np.abs(1.0 / (value_pre_each_sorted + 1e-8)))
                # Normalize height
                scale_pre_each_sorted = (scale_pre_each_sorted / scale_pre_each_sorted[2]) * Is_pre_each
            else:
                scale_pre_each_sorted = np.sqrt(np.abs(1.0 / (value_pre_each_sorted + 1e-8))) * Is_pre_each

            norm_factor = np.linalg.norm(vector_pre_each_sorted, axis=0, keepdims=True) + 1e-8
            rotation_pre_each_sorted = (vector_pre_each_sorted / norm_factor) * Ir_pre_each

        # PCA
        points_numpy = points[i].transpose(1,0)  # shape (N,3)
        pca_centroid = np.mean(points_numpy, axis=0)
        points_centered = points_numpy - pca_centroid
        S, U = pca_numpy(points_centered)
        
        # Sort ascending by S
        idx_sort_S = np.argsort(S)
        S = S[idx_sort_S]
        U = U[:, idx_sort_S]
        pca_scales = np.sqrt(3 * S)
        # pca_scales = S
        pca_rotations = U
        pca_scales_torch = pca_scales
        pca_centroid_torch = pca_centroid
        pca_rotations_torch = pca_rotations

        # If shape is plane
        if element_quadrics_type in ["plane"]:
            center = pca_centroid_torch
            scale_pre_each_sorted = pca_scales_torch
            # Flip scales
            scale_pre_each_sorted = scale_pre_each_sorted[::-1]

            # Is_pre_each = [1,1,1]
            Is_pre_each = np.array([1,1,1], dtype=np.float64)

            # Flip rotation and assign normal vector
            rotation_pre_each_sorted_ = pca_rotations_torch[:, ::-1].copy()
            rotation_pre_each_sorted_[:,2] = rotation_pre_each_sorted[:,0]
            rotation_pre_each_sorted = rotation_pre_each_sorted_
            Ir_pre_each = np.array([1,1,1], dtype=np.float64)

        else:
            # For other shapes, try to extract center from Q
            Q = q_Q(output[i])
            A = Q[0:3, 0:3]
            b = -Q[0:3, 3]
            try:
                center_system, center_particular = solveLS(A, b)
                if center_system.size == 0 and center_particular.size != 0:
                    # sphere, ellipsiod, cone
                    center_numpy = center_particular
                elif center_system.shape == (3,1) and center_particular.size != 0:
                    # cylinder
                    proj = np.dot(points_numpy - center_particular, center_system)
                    center_project_length = (proj.max() + proj.min()) / 2.0
                    center_numpy = center_particular + (center_project_length * center_system).squeeze()
                else:
                    center_numpy = pca_centroid_torch

                if np.linalg.norm(center_numpy - pca_centroid) > 2:
                    print("center too far from pca center")
                    center_numpy = pca_centroid_torch
                center = center_numpy
            except:
                print("center extraction failed")
                center = pca_centroid_torch

            # Adjust scales and rotations based on conditions
            if Is_pre_each[0] == 0 or Is_pre_each[1] == 0:
                # ellipsoid and sphere
                scale_pre_each_sorted = pca_scales_torch

            # For cylinder and cone, keep the z-scale
            if Is_pre_each[2] == 0:
                if element_quadrics_type in ["cone","elliptic_cone"]:
                    # Estimate height
                    # project points and center on the main axis (Z direction)
                    height = ((points_numpy @ rotation_pre_each_sorted[:,2]) - (center @ rotation_pre_each_sorted[:,2])).max()
                    scale_pre_each_sorted = scale_pre_each_sorted * height
                    scale_pre_each_sorted[2] = height
                else:
                    scale_pre_each_sorted[2] = pca_scales_torch[2]

                Is_pre_each = np.array([1,1,1], dtype=np.float64)

            if Ir_pre_each[0] == 0 or Ir_pre_each[1] == 0:
                # for cone, cylinder, sphere, ellipsoid
                rotation_pre_each_sorted[:,0] = pca_rotations_torch[:,0]
                rotation_pre_each_sorted[:,1] = pca_rotations_torch[:,1]

            if Ir_pre_each[2] == 0:
                # sphere
                rotation_pre_each_sorted[:,2] = pca_rotations_torch[:,2]

            Ir_pre_each = np.array([1,1,1], dtype=np.float64)

        center_statistics = np.mean(points[i].transpose(1,0), axis=0)

    return center, center_statistics, scale_pre_each_sorted, rotation_pre_each_sorted, Is_pre_each, Ir_pre_each

def quadrics_decomposition_info_extracting_obb(points,axis_original,output,element_quadrics_type,trans_inv,C,mode="not"):
    
    points = np.expand_dims(points.T, axis=0)

    # Check if points is (1, N, 3)
    assert points.shape[1] == 3 and points.shape[0] == 1
    
    for i in range(1):

        Q_each = np.array([
            [output[i, 0], output[i, 3], output[i, 4], output[i, 6]],
            [output[i, 3], output[i, 1], output[i, 5], output[i, 7]],
            [output[i, 4], output[i, 5], output[i, 2], output[i, 8]],
            [output[i, 6], output[i, 7], output[i, 8], output[i, 9]]
        ], dtype=np.float32)

        Q_each, _ = quadrics_scale_identification(Q_each, element_quadrics_type)

        if mode == "decomposition":

            E_each = Q_each[0:3, 0:3]

            val, vec = np.linalg.eig(E_each)
            val_real = val.real

            idx_pre_each = np.argsort(val_real)[::-1]
            value_each_sorted = val_real[idx_pre_each]
            vector_each_sorted = vec[:, idx_pre_each].real

            Is_each, Ir_each, It_each = quadrics_judgment(value_each_sorted)

            if element_quadrics_type in ["cone", "elliptic_cone"]:
                scale_each_sorted = np.sqrt(np.abs(1.0 / (value_each_sorted + 1e-8)))
                scale_each_sorted = (scale_each_sorted / scale_each_sorted[2]) * Is_each
            else:
                scale_each_sorted = np.sqrt(np.abs(1.0 / (value_each_sorted + 1e-8))) * Is_each

            col_norms = np.linalg.norm(vector_each_sorted, axis=0) + 1e-8
            rotation_each_sorted = (vector_each_sorted / col_norms) * Ir_each

            scale_pre_each_sorted = scale_each_sorted
            rotation_pre_each_sorted = rotation_each_sorted
            Is_pre_each = Is_each
            Ir_pre_each = Ir_each
            It_pre_each = It_each

        else:
            C_pre_each = C[i]

            if element_quadrics_type in ["sphere", "ellipsoid", "cylinder", "elliptic_cylinder"]:
                denom = -C_pre_each[3, 3] if abs(C_pre_each[3, 3]) > 1e-8 else -1e-8
                C_pre_each = C_pre_each / denom

            trans_r = trans_inv[i][0:3, 0:3].T

            # ---------------------- pre eigen ------------------------------
            diag_vals = np.diag(C_pre_each[0:3, 0:3])
            idx_pre_each = np.argsort(diag_vals)[::-1]
            value_pre_each_sorted = diag_vals[idx_pre_each]

            vector_pre_each_sorted = trans_r[:, idx_pre_each]

            Is_pre_each, Ir_pre_each, It_pre_each = quadrics_judgment(value_pre_each_sorted)

            if element_quadrics_type in ["cone", "elliptic_cone"]:
                scale_pre_each_sorted = np.sqrt(np.abs(1.0 / (value_pre_each_sorted + 1e-8)))
                scale_pre_each_sorted = (scale_pre_each_sorted / scale_pre_each_sorted[2]) * Is_pre_each
            else:
                scale_pre_each_sorted = np.sqrt(np.abs(1.0 / (value_pre_each_sorted + 1e-8))) * Is_pre_each

            col_norms = np.linalg.norm(vector_pre_each_sorted, axis=0) + 1e-8
            rotation_pre_each_sorted = (vector_pre_each_sorted / col_norms) * Ir_pre_each

        if element_quadrics_type in ["plane"]:
            normal_original_ = rotation_pre_each_sorted[:, 0]
        else:
            normal_original_ = rotation_pre_each_sorted[:, 2]
            if np.sum(np.abs(normal_original_)) == 0:
                normal_original_ = axis_original

        points_numpy = points[i].T

        normal_z = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        R_originalToZ = rotation_matrix_a_to_b(normal_original_, normal_z)

        points_numpy_z = np.dot(points_numpy, R_originalToZ.T)

        points_2d, centroid, normal = project_to_2d(points_numpy_z, normal_z)
        obb_corners_2d, scales_2d = compute_2d_obb(points_2d)

        obb_centroid_z, obb_corners_3d_z, rotation_matrix_3d_z, scales_3d_z = create_3d_obb(
            centroid, obb_corners_2d, normal, points_numpy_z, scales_2d, element_quadrics_type
        )

        obb_centroid = np.dot(obb_centroid_z, R_originalToZ)
        rotation_matrix_3d = np.dot(R_originalToZ.T, rotation_matrix_3d_z)
        scales_3d = scales_3d_z

        obb_centroid_torch = obb_centroid
        rotation_matrix_3d_torch = rotation_matrix_3d
        scales_3d_torch = scales_3d

        if element_quadrics_type in ["plane"]:
            Q_each_numpy = Q_each

            T_Z_inv_temp = np.eye(4, dtype=np.float32)
            T_Z_inv_temp[0:3, 0:3] = R_originalToZ.T
            Q_Z_temp = T_Z_inv_temp.T @ Q_each_numpy @ T_Z_inv_temp
            Q_Z_temp = 0.5 * (Q_Z_temp + Q_Z_temp.T)

            eigen_value_Q_Z_temp, eigen_vector_Q_Z_temp = np.linalg.eig(Q_Z_temp[0:3, 0:3])
            idx = np.argsort(eigen_value_Q_Z_temp)
            d = np.dot(eigen_vector_Q_Z_temp[:, idx[2]], Q_Z_temp[0:3, 3]) / eigen_value_Q_Z_temp[idx[2]]
            obb_centroid_z[2] = -d
            obb_centroid = np.dot(obb_centroid_z, R_originalToZ)
            obb_centroid_torch = obb_centroid

            center = obb_centroid_torch
            scale_pre_each_sorted = scales_3d_torch / 2.0
            Is_pre_each = np.array([1, 1, 0], dtype=np.float32)

            rotation_pre_each_sorted_ = rotation_matrix_3d_torch.copy()
            rotation_pre_each_sorted_[:, 2] = rotation_pre_each_sorted[:, 0]
            rotation_pre_each_sorted = rotation_pre_each_sorted_
            Ir_pre_each = np.array([1, 1, 1], dtype=np.float32)

        else:
            Q = q_Q(output[i])
            A = Q[0:3, 0:3]
            b = -Q[0:3, 3]
            try:
                center_system, center_particular = solveLS(A, b)

                if len(center_system) == 0 and len(center_particular) != 0:
                    # sphere, ellipsoid, cone
                    center_numpy = center_particular
                elif center_system.shape == (3, 1) and len(center_particular) != 0:
                    # cylinder
                    center_project_length = (
                        np.matmul(points_numpy - center_particular, center_system).max() +
                        np.matmul(points_numpy - center_particular, center_system).min()
                    ) / 2.0
                    center_numpy = center_particular + (center_project_length * center_system).squeeze()

                center = center_numpy
                if np.linalg.norm(center_numpy - obb_centroid) > 2.0:
                    print("center too far from obb center (2)")
                    center = obb_centroid_torch.astype(np.float32)

            except:
                print("center extraction failed")
                center = obb_centroid_torch.astype(np.float32)

            if Is_pre_each[0] == 0 or Is_pre_each[1] == 0:
                # ellipsoid / sphere
                sorted_scales = np.sort(scales_3d_torch)
                scale_pre_each_sorted = sorted_scales / 2.0

            if Is_pre_each[2] == 0:
                if element_quadrics_type in ["cone", "elliptic_cone"]:
                    height = (
                        (points[i].T @ rotation_pre_each_sorted[:, 2]) -
                        (center.reshape(1, -1) @ rotation_pre_each_sorted[:, 2])
                    ).max()
                    scale_pre_each_sorted = scale_pre_each_sorted * height
                    scale_pre_each_sorted[2] = height
                else:
                    scale_pre_each_sorted[2] = scales_3d_torch[2] / 2.0
                Is_pre_each = np.array([1, 1, 1], dtype=np.float32)

            if Ir_pre_each[0] == 0 or Ir_pre_each[1] == 0:
                rotation_pre_each_sorted[:, 0] = rotation_matrix_3d_torch[:, 0]
                rotation_pre_each_sorted[:, 1] = rotation_matrix_3d_torch[:, 1]

            if Ir_pre_each[2] == 0:
                rotation_pre_each_sorted[:, 2] = rotation_matrix_3d_torch[:, 2]
            Ir_pre_each = np.array([1, 1, 1], dtype=np.float32)

        center_statistics = points[i].T.mean(axis=0)

    return (
        center,
        center_statistics,
        scale_pre_each_sorted,
        rotation_pre_each_sorted,
        Is_pre_each,
        Ir_pre_each
    )

def quadrics_decomposition_info_extracting_line_pca(points, output, element_quadrics_type, trans_inv, C, mode="not"):

    points = np.expand_dims(points.T, axis=0)
    # Check if points is (1, N, 3)
    assert points.shape[1] == 3 and points.shape[0] == 1

    # For each batch element
    for i in range(1):
        # Construct Q_each from output
        Q_each = np.array([[output[i, 0], output[i, 3], output[i, 4], output[i, 6]],
                           [output[i, 3], output[i, 1], output[i, 5], output[i, 7]],
                           [output[i, 4], output[i, 5], output[i, 2], output[i, 8]],
                           [output[i, 6], output[i, 7], output[i, 8], output[i, 9]]], dtype=np.float64)

        # Identify scale
        Q_each, _ = quadrics_scale_identification(Q_each, element_quadrics_type)

        if mode == "decomposition":
            # Decompose the top-left 3x3 part of Q_each
            E_each = Q_each[0:3,0:3]
            # np.linalg.eig returns eigenvalues and eigenvectors
            val_each, vec_each = np.linalg.eig(E_each)
            val_each = np.real(val_each)  # ensure real if needed
            
            # Sort eigenvalues in descending order
            idx_sorted = np.argsort(val_each)[::-1]
            vector_pre_each_sorted = vec_each[:, idx_sorted]

        else:
            C_pre_each = C[i].copy()
            # Normalize C_pre_each if shape is sphere/ellipsoid/cylinder/elliptic_cylinder
            if element_quadrics_type in ["sphere", "ellipsoid", "cylinder", "elliptic_cylinder"]:
                C_pre_each = C_pre_each / (-C_pre_each[3,3])

            trans_r = trans_inv[i][0:3,0:3].T

            # Sort diagonal of C_pre_each
            diag_vals = np.diag(C_pre_each)[:3]
            idx_pre_each = np.argsort(diag_vals)[::-1]
            vector_pre_each_sorted = trans_r[:, idx_pre_each]

        # Normalize rotation vectors
        rotation_pre_each_sorted = vector_pre_each_sorted / (np.linalg.norm(vector_pre_each_sorted, axis=0, keepdims=True) + 1e-8)
        Ir_pre_each = np.array([1,1,1], dtype=np.float64)

        # Perform PCA on points
        # points[i] has shape (3,N), transpose to (N,3)
        points_numpy = points[i].T
        pca_centroid = np.mean(points_numpy, axis=0)
        points_centered = points_numpy - pca_centroid
        S, U = pca_numpy(points_centered)

        # Sort eigenvalues of PCA in ascending order
        index_sorted = np.argsort(S)
        S = S[index_sorted]
        U = U[:, index_sorted]

        pca_scales = np.sqrt(3 * S)
        # pca_scales =S
        scale_pre_each_sorted = pca_scales

        # For line, we just set Is_pre_each = [1,1,1]
        Is_pre_each = np.array([1,1,1], dtype=np.float64)

        # Keep the last axis from previous rotation, replace first two with PCA directions
        rotation_pre_each_sorted[:,0:2] = U[:,0:2]

        center_statistics = np.mean(points[i].T, axis=0)
        center = center_statistics

    return center, center_statistics, scale_pre_each_sorted, rotation_pre_each_sorted, Is_pre_each, Ir_pre_each


def genrate_mesh_open3d(shape,scale,center,rotation):
    T_t = np.eye(4)
    T_t[0:3,3] = center
    T_R = np.eye(4)
    T_R[0:3,0:3] = rotation
    T_s = np.eye(4)
    if shape in ["line"]:
        scale[0:2] = 0.1
    T_s[0:3,0:3] = np.diag(scale)
    T = T_t @ T_R @ T_s
    
    # T[0:3,0:3] = T[0:3,0:3] * scale
    # T[0:3,0:3] = rotation
    # T[0:3,3] = center
    # # T[0:3,0:3] = T[0:3,0:3]*scale
    
    if shape in ["plane"]:
        mesh = o3d.geometry.TriangleMesh.create_box(width=2.0, height=2.0, depth=2.0)
        mesh.translate((-2 / 2, -2 / 2, -2 / 2))
        color = [246,230,203]
        color = [c / 255.0 for c in color]
        mesh.paint_uniform_color(color)
    elif shape in ["shape","ellipsoid"]:
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
        color = [102,123,198]
        color = [c / 255.0 for c in color]
        mesh.paint_uniform_color(color)
    elif shape in ["cylinder","elliptic_cylinder"]:
        mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=1.0,height=2.0)
        color = [151,49,49]
        color = [c / 255.0 for c in color]
        mesh.paint_uniform_color(color)
        # o3d.io.write_triangle_mesh("cylinder.ply", mesh)
    elif shape in ["cone","elliptic_cone"]:
        mesh = o3d.geometry.TriangleMesh.create_cone(radius=1.0,height=1.0)
        color = [89,116,69]
        color = [c / 255.0 for c in color]
        mesh.paint_uniform_color(color)

        T_temp = np.eye(4)
        R_temp = np.eye(3)
        R_temp[0:3, 0:3] = np.array([
            [1,  0,  0],
            [0, -1,  0],
            [0,  0, -1]
        ])
        T_temp[0:3,0:3] = R_temp
        T_temp[2,3] = 1.0
        mesh.transform(T_temp)
    elif shape in ["line"]:
        mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=1.0,height=2.0)
        color = [8,131,149]
        color = [c / 255.0 for c in color]
        mesh.paint_uniform_color(color)

    mesh.transform(T)
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    return mesh

def ground_normal_estimate(points):
    """

        points (array-like): Point cloud data, shape (N, 3).

        normal (numpy.ndarray): Ground normal vector, shape (3,).
    """
    points = np.asarray(points, dtype=np.float64)

    center = np.mean(points, axis=0)

    centered_points = points - center

    cov = np.dot(centered_points.T, centered_points) / len(points)

    eig_val, eig_vec = np.linalg.eigh(cov)

    normal = eig_vec[:, 0]

    return normal

def generate_ellipsoid_coeff(center, scale, rotation):
    """
    center: Ellipsoid center (x0, y0, z0)
    scale: Ellipsoid semi-axis lengths (a, b, c)
    rotation: Ellipsoid rotation matrix (3x3)
    """

    T_s = np.eye(4)
    T_s[:3, :3] = np.diag(scale)
    T_r = np.eye(4)
    T_r[:3, :3] = rotation
    T_t = np.eye(4)
    T_t[:3, 3] = center
    C = np.diag([1, 1, 1, -1])
    T = T_t @ T_r @ T_s
    T_inv = np.linalg.inv(T)
    Q = T_inv.T @ C @ T_inv

    q = Q_q(Q)

    return q
