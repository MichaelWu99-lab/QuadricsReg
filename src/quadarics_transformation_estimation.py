import numpy as np
import warnings
import open3d as o3d
import time
import os
import gtsam
from gtsam import NonlinearFactorGraph, Values, Pose3, noiseModel,Rot3
from gtsam.symbol_shorthand import X
from src.quadarics_transformation_estimation_utils import *

warnings.filterwarnings("ignore")

def solve_gtsam_multi(test_data_experiment,correspondence_max_clique_tuple_list,quadrics_info_source_stack,quadrics_info_target_stack,quadricsReg_config,T_gt=[]):

    quadricsEstimation_config = quadricsReg_config["quadricsEstimation"]
    if_trans_refine = quadricsEstimation_config["if_trans_refine"]
    if_est_augment = quadricsEstimation_config["if_est_augment"]
    if_refine_errorR = quadricsEstimation_config["if_refine_errorR"]
    if_refine_errort = quadricsEstimation_config["if_refine_errort"]
    optimization_inerations = quadricsEstimation_config["optimization_inerations"]
    threshold_Manhattan_gtsam = quadricsEstimation_config["threshold_Manhattan_gtsam"]
    threshold_Manhattan_vertification = quadricsEstimation_config["threshold_Manhattan_vertification"]
    degenerate_quadrics_aid_gtsam = quadricsEstimation_config["degenerate_quadrics_aid_gtsam"]
    degenerate_quadrics_aid_vertification = quadricsEstimation_config["degenerate_quadrics_aid_vertification"]
    ground_aid = quadricsEstimation_config["ground_aid"]
    vertification_method = quadricsEstimation_config["vertification_method"]
    voxel_size_KNN_vertification = quadricsEstimation_config["voxel_size_KNN_vertification"]
    hyper_parameter = quadricsEstimation_config["hyper_parameter"]
    radius_vertification = quadricsEstimation_config["radius_vertification"]
    reference_origin_source = quadricsEstimation_config["reference_origin_source"]
    ground_z = quadricsEstimation_config["ground_z"]
    center_type_SVD = quadricsEstimation_config["center_type_SVD"]
    center_type_gtsam = quadricsEstimation_config["center_type_gtsam"]
    center_type_nn = quadricsEstimation_config["center_type_nn"]
    center_type_vertification = quadricsEstimation_config["center_type_vertification"]

    #  ablation study
    if if_trans_refine == False:
        optimization_inerations = 0

    if if_est_augment == False:
        degenerate_quadrics_aid_gtsam = False
        degenerate_quadrics_aid_vertification = False
        ground_aid = False
    else:
        degenerate_quadrics_aid_gtsam = True
        degenerate_quadrics_aid_vertification = False
        ground_aid = True
    if if_refine_errort == False:
        optimization_inerations = 20 # Accelerate

    quadrics_info_source_stack_copy = quadrics_info_source_stack.copy()
    quadrics_info_target_stack_copy = quadrics_info_target_stack.copy()

    src_ground_normal = quadrics_info_source_stack_copy["ground_normal"]
    tgt_ground_normal = quadrics_info_target_stack_copy["ground_normal"]
    src_ground_points = generate_square_on_plane(np.array([0,0,ground_z]),src_ground_normal,5)
    tgt_ground_points = generate_square_on_plane(np.array([0,0,ground_z]),tgt_ground_normal,5)
    quadrics_info_source_stack_copy.pop("ground_normal")
    quadrics_info_target_stack_copy.pop("ground_normal")

    for key in quadrics_info_source_stack_copy.keys():
        quadrics_info_source_stack_copy[key] = info_resacle_pca(quadrics_info_source_stack_copy[key])
    for key in quadrics_info_target_stack_copy.keys():
        quadrics_info_target_stack_copy[key] = info_resacle_pca(quadrics_info_target_stack_copy[key])

    time_0 = time.time()
        
    source_seg_label_list = []
    target_seg_label_list = []
    source_centers_SVD_list = []
    target_centers_SVD_list = []
    maxclique_count = 0
    SVD_count = 0
    vertification_count = 0
    for corr_index in range(len(correspondence_max_clique_tuple_list)):
        source_seg_label = correspondence_max_clique_tuple_list[corr_index][0]
        target_seg_label = correspondence_max_clique_tuple_list[corr_index][1]

        src_quadrics_info = quadrics_info_source_stack_copy[source_seg_label]
        tgt_quadrics_info = quadrics_info_target_stack_copy[target_seg_label]

        source_center_SVD = src_quadrics_info[center_type_SVD]
        target_center_SVD = tgt_quadrics_info[center_type_SVD]

        maxclique_count += + 1

        SVD_count += 1

        source_centers_SVD_list.append(source_center_SVD)
        target_centers_SVD_list.append(target_center_SVD)

        if tgt_quadrics_info['quadrics_type']=="point" or src_quadrics_info['quadrics_type']=="point":
            continue

        # if tgt_quadrics_info['quadrics_type']!="ellipsoid" or src_quadrics_info['quadrics_type']=="ellipsoid":
        #     tgt_quadrics_info['quadrics_type']="ellipsoid"
        #     src_quadrics_info['quadrics_type']="ellipsoid"

        source_seg_label_list.append(source_seg_label)
        target_seg_label_list.append(target_seg_label)

        # line = create_line_with_auxiliary_points(source_center,target_center,100)
        # o3d.io.write_point_cloud(f"{fitting_result_dir_sequence}/{testsample_index}/maxclique_Pmc_{source_seg_label}-{target_seg_label}.ply",line)

    source_centers_SVD = np.array(source_centers_SVD_list)
    target_centers_SVD = np.array(target_centers_SVD_list)

    if source_centers_SVD.shape[0] < 3:
        T_solved_SVD = np.eye(4)
        print(f"svdSE3 error")
    else:
        try:
            T_solved_SVD = svdSE3(source_centers_SVD.T,target_centers_SVD.T)
        except:
            T_solved_SVD = np.eye(4)
            print(f"svdSE3 error")

    time_1 = time.time()

    graph = NonlinearFactorGraph()
    initial = Values()
    R_init = T_solved_SVD[:3,:3]
    t_init = T_solved_SVD[:3,3]

    initial.insert(X(0), Pose3(Rot3(R_init),t_init))

    noise12 = noiseModel.Diagonal.Sigmas(np.ones(12))

    factor_count = 0
    augment_points_src = []
    for i in range(len(source_seg_label_list)):
        src_seg_label = source_seg_label_list[i]
        tgt_seg_label = target_seg_label_list[i]
        src_quadrics_info = quadrics_info_source_stack_copy[src_seg_label]
        tgt_quadrics_info = quadrics_info_target_stack_copy[tgt_seg_label]

        if (judge_Manhattan(tgt_ground_normal,tgt_quadrics_info['quadrics_type'],tgt_quadrics_info['decomposition_rotation'],threshold_Manhattan_gtsam)==False) or (judge_Manhattan(src_ground_normal,src_quadrics_info['quadrics_type'],src_quadrics_info['decomposition_rotation'],threshold_Manhattan_gtsam)==False):
            continue

        factor_count = factor_count+1

        src_center = src_quadrics_info[center_type_gtsam]
        tgt_center = tgt_quadrics_info[center_type_gtsam]

        if degenerate_quadrics_aid_gtsam:
            if src_quadrics_info['quadrics_type'] in ["plane"]:
                
                plane_points_augment_src = generate_square_on_plane(src_center,src_quadrics_info['decomposition_rotation'][:,0],np.sqrt(src_quadrics_info["volume"]))
                # No need to generate augmented points for target
                for i in range(plane_points_augment_src.shape[0]):
                    graph.add(Quadrics2QuadricsFactor(X(0), plane_points_augment_src[i],tgt_center, src_quadrics_info, tgt_quadrics_info, noise12,if_refine_errorR,if_refine_errort))

                augment_points_src.append(plane_points_augment_src)

            elif src_quadrics_info['quadrics_type'] in ["line"]:

                line_points_augment_src = generate_segment_on_line(src_center,src_quadrics_info['decomposition_rotation'][:,2],src_quadrics_info["volume"])
                for i in range(line_points_augment_src.shape[0]):
                    graph.add(Quadrics2QuadricsFactor(X(0), line_points_augment_src[i], tgt_center, src_quadrics_info, tgt_quadrics_info, noise12,if_refine_errorR,if_refine_errort))

                augment_points_src.append(line_points_augment_src)

            elif src_quadrics_info['quadrics_type'] in ["cylinder","elliptic_cylinder"]:

                cylinder_points_augment_src = generate_segment_on_line(src_center,src_quadrics_info['decomposition_rotation'][:,2],src_quadrics_info["full_scale"][2])
                for i in range(cylinder_points_augment_src.shape[0]):
                    graph.add(Quadrics2QuadricsFactor(X(0), cylinder_points_augment_src[i], tgt_center, src_quadrics_info, tgt_quadrics_info, noise12,if_refine_errorR,if_refine_errort))

                augment_points_src.append(cylinder_points_augment_src)

            else:
                graph.add(Quadrics2QuadricsFactor(X(0), src_center,tgt_center, src_quadrics_info, tgt_quadrics_info, noise12,if_refine_errorR,if_refine_errort))
        
        else:
            graph.add(Quadrics2QuadricsFactor(X(0), src_center,tgt_center, src_quadrics_info, tgt_quadrics_info, noise12,if_refine_errorR,if_refine_errort))
    
    augment_points_src.append(src_ground_points)

    # Save augmented points
    augment_points_src = np.concatenate(augment_points_src,axis=0)
    test_data_experiment.augment_points_in_estimation = augment_points_src

    if ground_aid:
        for src_point, tgt_point in zip(src_ground_points, tgt_ground_points):
            graph.add(Ground2GroundFactor(X(0), src_point, tgt_point, src_ground_normal, tgt_ground_normal, noise12,if_refine_errorR,if_refine_errort))
            
    if factor_count <= 3:
        print(f"        factor_count={factor_count} <= 3, no optimized")
        optimized_transform = T_solved_SVD
        iterations = 0
    else:
        # Create Levenberg-Marquardt optimizer
        params = gtsam.LevenbergMarquardtParams()
        params.setRelativeErrorTol(1e-8)
        params.setAbsoluteErrorTol(1e-8)
        params.setErrorTol(0)
        params.setMaxIterations(optimization_inerations)
        params.setVerbosityLM("TERMINATION")
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial,params)
        result = optimizer.optimize()
        iterations = optimizer.iterations()

        optimized_transform = result.atPose3(X(0)).matrix()

    if T_gt != []:
        rotation_error_init,translation_error_init = compute_pose_error(T_gt,T_solved_SVD)
        rotation_error,translation_error = compute_pose_error(T_gt,optimized_transform)
    else:
        rotation_error_init = np.nan
        translation_error_init = np.nan
        rotation_error = np.nan
        translation_error = np.nan

    time_2 = time.time()

    # ################vertification################
    vertification_transform = optimized_transform

    if vertification_method == "KNN":
        # Transform source to target frame and compute KNN distance
        source_pcd = test_data_experiment.source_pcd_current
        target_pcd = test_data_experiment.target_pcd_current
        source_pcd.transform(vertification_transform)
        source_pcd = source_pcd.voxel_down_sample(voxel_size_KNN_vertification)
        target_pcd = target_pcd.voxel_down_sample(voxel_size_KNN_vertification)
        error_vertification_quadrics = compute_average_nearest_neighbor_distance(source_pcd,target_pcd,kernel="dcs",hyper_parameter=hyper_parameter)

        error_vertification_quadrics= error_vertification_quadrics

        # print(f'    maxclique_c: {maxclique_count}, SVD_c: {SVD_count}, factor_c: {factor_count}, iter: {iterations}, rot_error: {rotation_error_init:4f}-{rotation_error:4f}, tran_error: {translation_error_init:4f}-{translation_error:4f}, error_vert_KNN: {error_vertification_KNN_mean:4f}')

    elif vertification_method == "quadrics_inliers":
        # Compute correspondence errors
        error_vertification_quadrics = []
        error_vertification_R = []
        error_vertification_t = []

        if len(source_seg_label_list) > 0:
            error_vertification_quadrics,error_vertification_R,error_vertification_t = error_vertification(vertification_transform,source_seg_label_list,target_seg_label_list,quadrics_info_source_stack_copy,quadrics_info_target_stack_copy,center_type_vertification,degenerate_quadrics_aid_vertification,ground_aid,src_ground_points,tgt_ground_points,src_ground_normal,tgt_ground_normal,threshold_Manhattan_vertification)

    else:
        # Extract all source centers
        source_centers_all = []
        source_seg_label_all = []
        source_semantic_all = []
        for key in quadrics_info_source_stack_copy.keys():
            source_centers_all.append(quadrics_info_source_stack_copy[key][center_type_nn])
            source_seg_label_all.append(key)
            source_semantic_all.append(quadrics_info_source_stack_copy[key]["semantic"])
        source_centers_all = np.array(source_centers_all)
        source_seg_label_all = np.array(source_seg_label_all)
        source_semantic_all = np.array(source_semantic_all)

        # Extract all target centers
        target_centers_all = []
        target_seg_label_all = []
        target_semantic_all = []
        for key in quadrics_info_target_stack_copy.keys():
            target_centers_all.append(quadrics_info_target_stack_copy[key][center_type_nn])
            target_seg_label_all.append(key)
            target_semantic_all.append(quadrics_info_target_stack_copy[key]["semantic"])
        target_centers_all = np.array(target_centers_all)                
        target_seg_label_all = np.array(target_seg_label_all)
        target_semantic_all = np.array(target_semantic_all)
        
        # Filter points outside radius_vertification from reference origin
        source_radius_selected_index = (np.linalg.norm(source_centers_all - reference_origin_source,axis=1) < radius_vertification[1]) & (np.linalg.norm(source_centers_all - reference_origin_source,axis=1) > radius_vertification[0])
        source_seg_label_all = source_seg_label_all[source_radius_selected_index]
        source_centers_all = source_centers_all[source_radius_selected_index]
        source_semantic_all = source_semantic_all[source_radius_selected_index]

        target_radius_selected_index = (np.linalg.norm(target_centers_all,axis=1) < radius_vertification[1]) & (np.linalg.norm(target_centers_all,axis=1) > radius_vertification[0])
        target_seg_label_all = target_seg_label_all[target_radius_selected_index]
        target_centers_all = target_centers_all[target_radius_selected_index]
        target_semantic_all = target_semantic_all[target_radius_selected_index]

        # Transform source to target frame and find nearest neighbors
        source_centers_transformed_all = (vertification_transform[:3,:3] @ source_centers_all.T + vertification_transform[:3,3].reshape(-1,1)).T

        if vertification_method == 'quadrics_all':
            nearest_neighbors_indices,nearest_neighbors_dists = nearest_neighbors(source_centers_transformed_all,target_centers_all)

            source_seg_label_all = source_seg_label_all
            target_seg_label_all = list(np.array(target_seg_label_all)[nearest_neighbors_indices])

            error_vertification_quadrics,error_vertification_R,error_vertification_t = error_vertification(vertification_transform,source_seg_label_all,target_seg_label_all,quadrics_info_source_stack_copy,quadrics_info_target_stack_copy,center_type_vertification,degenerate_quadrics_aid_vertification,ground_aid,src_ground_points,tgt_ground_points,src_ground_normal,tgt_ground_normal,threshold_Manhattan_vertification)
        
        elif vertification_method == 'quadrics_semantic':
            # Compute distance per semantic label
            error_vertification_quadrics = []
            error_vertification_R = []
            error_vertification_t = []
            for semantic_current in set(source_semantic_all):
                source_semantic_index = source_semantic_all == semantic_current
                target_semantic_index = target_semantic_all == semantic_current
                source_centers_transformed_semantic = source_centers_transformed_all[source_semantic_index]
                target_centers_semantic = target_centers_all[target_semantic_index]
                source_seg_semantic = source_seg_label_all[source_semantic_index]
                target_seg_semantic = target_seg_label_all[target_semantic_index]

                if source_centers_transformed_semantic.shape[0] == 0 or target_centers_semantic.shape[0] == 0:
                    continue
                else:
                    nearest_neighbors_indices,nearest_neighbors_dists = nearest_neighbors(source_centers_transformed_semantic,target_centers_semantic)

                source_seg_semantic = source_seg_semantic
                target_seg_semantic = list(np.array(target_seg_semantic)[nearest_neighbors_indices])

                error_vertification_quadrics_semantic,error_vertification_R_semantic,error_vertification_t_semantic = error_vertification(vertification_transform,source_seg_semantic,target_seg_semantic,quadrics_info_source_stack_copy,quadrics_info_target_stack_copy,center_type_vertification,degenerate_quadrics_aid_vertification,ground_aid,src_ground_points,tgt_ground_points,src_ground_normal,tgt_ground_normal,threshold_Manhattan_vertification)
                error_vertification_quadrics.extend(error_vertification_quadrics_semantic)
                error_vertification_R.extend(error_vertification_R_semantic)
                error_vertification_t.extend(error_vertification_t_semantic)

    vertification_count = len(error_vertification_quadrics)

    try:
        # error_vertification_quadrics_mean = trimmed_mean(error_vertification_quadrics)
        # error_vertification_R_mean = trimmed_mean(error_vertification_R)
        # error_vertification_t_mean = trimmed_mean(error_vertification_t)
        
        # Apply robust kernel to errors
        error_list = []
        for error in error_vertification_quadrics:
            error_list.append(robust_kernel(error,hyper_parameter,"dcs"))
        error_vertification_quadrics_mean = np.mean(error_list)

        # error_vertification_quadrics_mean = np.mean(error_vertification_quadrics)

    except:

        error_vertification_quadrics_mean = np.mean(error_vertification_quadrics)

    error_vertification_mean = error_vertification_quadrics_mean
    
    print(f'    maxclique_c: {maxclique_count}, SVD_c: {SVD_count}, factor_c: {factor_count}, vertification_c: {vertification_count}, iter: {iterations}, rot_error: {rotation_error_init:4f}-{rotation_error:4f}, tran_error: {translation_error_init:4f}-{translation_error:4f}, error_vert_quadrics: {error_vertification_quadrics_mean:4f}')
    
    time_3 = time.time()

    # print(f"        time SVD-gtsam-ver: {time_1-time_0:4f}-{time_2-time_1:4f}-{time_3-time_2:4f}")

    return error_vertification_mean,{"maxclique_count":maxclique_count,"SVD_count":SVD_count,"factor_count":factor_count,"iterations":iterations,"rotation_error_init":rotation_error_init,"rotation_error":rotation_error,"translation_error_init":translation_error_init,"translation_error":translation_error,"error_vertification":error_vertification_mean,"T_optimized":optimized_transform}


def solve_gtsam_multi_point(test_data_experiment,correspondence_max_clique_tuple_list,quadrics_info_source_stack,quadrics_info_target_stack,quadricsReg_config,T_gt=[]):

    quadricsEstimation_config = quadricsReg_config["quadricsEstimation"]
    if_trans_refine = quadricsEstimation_config["if_trans_refine"]
    if_est_augment = quadricsEstimation_config["if_est_augment"]
    if_refine_errorR = quadricsEstimation_config["if_refine_errorR"]
    if_refine_errort = quadricsEstimation_config["if_refine_errort"]
    optimization_inerations = quadricsEstimation_config["optimization_inerations"]
    threshold_Manhattan_gtsam = quadricsEstimation_config["threshold_Manhattan_gtsam"]
    threshold_Manhattan_vertification = quadricsEstimation_config["threshold_Manhattan_vertification"]
    degenerate_quadrics_aid_gtsam = quadricsEstimation_config["degenerate_quadrics_aid_gtsam"]
    degenerate_quadrics_aid_vertification = quadricsEstimation_config["degenerate_quadrics_aid_vertification"]
    ground_aid = quadricsEstimation_config["ground_aid"]
    vertification_method = quadricsEstimation_config["vertification_method"]
    voxel_size_KNN_vertification = quadricsEstimation_config["voxel_size_KNN_vertification"]
    hyper_parameter = quadricsEstimation_config["hyper_parameter"]
    radius_vertification = quadricsEstimation_config["radius_vertification"]
    reference_origin_source = quadricsEstimation_config["reference_origin_source"]
    ground_z = quadricsEstimation_config["ground_z"]
    center_type_SVD = quadricsEstimation_config["center_type_SVD"]
    center_type_gtsam = quadricsEstimation_config["center_type_gtsam"]
    center_type_nn = quadricsEstimation_config["center_type_nn"]
    center_type_vertification = quadricsEstimation_config["center_type_vertification"]

    #  ablation study
    if if_trans_refine == False:
        optimization_inerations = 0

    if if_est_augment == False:
        degenerate_quadrics_aid_gtsam = False
        degenerate_quadrics_aid_vertification = False
        ground_aid = False
    else:
        degenerate_quadrics_aid_gtsam = True
        degenerate_quadrics_aid_vertification = False
        ground_aid = True
    if if_refine_errort == False:
        optimization_inerations = 20 # Accelerate

    quadrics_info_source_stack_copy = quadrics_info_source_stack.copy()
    quadrics_info_target_stack_copy = quadrics_info_target_stack.copy()

    src_ground_normal = quadrics_info_source_stack_copy["ground_normal"]
    tgt_ground_normal = quadrics_info_target_stack_copy["ground_normal"]
    src_ground_points = generate_square_on_plane(np.array([0,0,ground_z]),src_ground_normal,5)
    tgt_ground_points = generate_square_on_plane(np.array([0,0,ground_z]),tgt_ground_normal,5)
    quadrics_info_source_stack_copy.pop("ground_normal")
    quadrics_info_target_stack_copy.pop("ground_normal")

    for key in quadrics_info_source_stack_copy.keys():
        quadrics_info_source_stack_copy[key] = info_resacle_pca(quadrics_info_source_stack_copy[key])
    for key in quadrics_info_target_stack_copy.keys():
        quadrics_info_target_stack_copy[key] = info_resacle_pca(quadrics_info_target_stack_copy[key])

    time_0 = time.time()
        
    source_seg_label_list = []
    target_seg_label_list = []
    source_centers_SVD_list = []
    target_centers_SVD_list = []
    maxclique_count = 0
    SVD_count = 0
    vertification_count = 0
    for corr_index in range(len(correspondence_max_clique_tuple_list)):
        source_seg_label = correspondence_max_clique_tuple_list[corr_index][0]
        target_seg_label = correspondence_max_clique_tuple_list[corr_index][1]

        src_quadrics_info = quadrics_info_source_stack_copy[source_seg_label]
        tgt_quadrics_info = quadrics_info_target_stack_copy[target_seg_label]

        source_center_SVD = src_quadrics_info[center_type_SVD]
        target_center_SVD = tgt_quadrics_info[center_type_SVD]

        maxclique_count += + 1

        SVD_count += 1

        source_centers_SVD_list.append(source_center_SVD)
        target_centers_SVD_list.append(target_center_SVD)

        if tgt_quadrics_info['quadrics_type']=="point" or src_quadrics_info['quadrics_type']=="point":
            continue

        if tgt_quadrics_info['quadrics_type']!="point" or src_quadrics_info['quadrics_type']=="point":
            tgt_quadrics_info['quadrics_type']="point"
            src_quadrics_info['quadrics_type']="point"

        source_seg_label_list.append(source_seg_label)
        target_seg_label_list.append(target_seg_label)

        # line = create_line_with_auxiliary_points(source_center,target_center,100)
        # o3d.io.write_point_cloud(f"{fitting_result_dir_sequence}/{testsample_index}/maxclique_Pmc_{source_seg_label}-{target_seg_label}.ply",line)

    source_centers_SVD = np.array(source_centers_SVD_list)
    target_centers_SVD = np.array(target_centers_SVD_list)

    if source_centers_SVD.shape[0] < 3:
        T_solved_SVD = np.eye(4)
        print(f"svdSE3 error")
    else:
        try:
            T_solved_SVD = svdSE3(source_centers_SVD.T,target_centers_SVD.T)
        except:
            T_solved_SVD = np.eye(4)
            print(f"svdSE3 error")

    time_1 = time.time()

    graph = NonlinearFactorGraph()
    initial = Values()
    R_init = T_solved_SVD[:3,:3]
    t_init = T_solved_SVD[:3,3]

    initial.insert(X(0), Pose3(Rot3(R_init),t_init))

    noise3 = noiseModel.Diagonal.Sigmas(np.ones(3))

    factor_count = 0
    augment_points_src = []
    for i in range(len(source_seg_label_list)):
        src_seg_label = source_seg_label_list[i]
        tgt_seg_label = target_seg_label_list[i]
        src_quadrics_info = quadrics_info_source_stack_copy[src_seg_label]
        tgt_quadrics_info = quadrics_info_target_stack_copy[tgt_seg_label]

        if (judge_Manhattan(tgt_ground_normal,tgt_quadrics_info['quadrics_type'],tgt_quadrics_info['decomposition_rotation'],threshold_Manhattan_gtsam)==False) or (judge_Manhattan(src_ground_normal,src_quadrics_info['quadrics_type'],src_quadrics_info['decomposition_rotation'],threshold_Manhattan_gtsam)==False):
            continue

        factor_count = factor_count+1

        src_center = src_quadrics_info[center_type_gtsam]
        tgt_center = tgt_quadrics_info[center_type_gtsam]

        if degenerate_quadrics_aid_gtsam:
            if src_quadrics_info['quadrics_type'] in ["plane"]:
                
                plane_points_augment_src = generate_square_on_plane(src_center,src_quadrics_info['decomposition_rotation'][:,0],np.sqrt(src_quadrics_info["volume"]))
                # No need to generate augmented points for target
                for i in range(plane_points_augment_src.shape[0]):
                    graph.add(Points2PointsFactor(X(0), plane_points_augment_src[i],tgt_center, src_quadrics_info, tgt_quadrics_info, noise3,if_refine_errorR,if_refine_errort))

                augment_points_src.append(plane_points_augment_src)

            elif src_quadrics_info['quadrics_type'] in ["line"]:

                line_points_augment_src = generate_segment_on_line(src_center,src_quadrics_info['decomposition_rotation'][:,2],src_quadrics_info["volume"])
                for i in range(line_points_augment_src.shape[0]):
                    graph.add(Points2PointsFactor(X(0), line_points_augment_src[i], tgt_center, src_quadrics_info, tgt_quadrics_info, noise3,if_refine_errorR,if_refine_errort))

                augment_points_src.append(line_points_augment_src)

            elif src_quadrics_info['quadrics_type'] in ["cylinder","elliptic_cylinder"]:

                cylinder_points_augment_src = generate_segment_on_line(src_center,src_quadrics_info['decomposition_rotation'][:,2],src_quadrics_info["full_scale"][2])
                for i in range(cylinder_points_augment_src.shape[0]):
                    graph.add(Points2PointsFactor(X(0), cylinder_points_augment_src[i], tgt_center, src_quadrics_info, tgt_quadrics_info, noise3,if_refine_errorR,if_refine_errort))

                augment_points_src.append(cylinder_points_augment_src)

            else:
                graph.add(Points2PointsFactor(X(0), src_center,tgt_center, src_quadrics_info, tgt_quadrics_info, noise3,if_refine_errorR,if_refine_errort))
        
        else:
            graph.add(Points2PointsFactor(X(0), src_center,tgt_center, src_quadrics_info, tgt_quadrics_info, noise3,if_refine_errorR,if_refine_errort))
    
    augment_points_src.append(src_ground_points)

    # Save augmented points
    augment_points_src = np.concatenate(augment_points_src,axis=0)
    test_data_experiment.augment_points_in_estimation = augment_points_src

    if ground_aid:
        for src_point, tgt_point in zip(src_ground_points, tgt_ground_points):
            graph.add(Points2PointsFactor(X(0), src_point, tgt_point, src_ground_normal, tgt_ground_normal, noise3,if_refine_errorR,if_refine_errort))
            
    if factor_count <= 3:
        print(f"        factor_count={factor_count} <= 3, no optimized")
        optimized_transform = T_solved_SVD
        iterations = 0
    else:
        # Create Levenberg-Marquardt optimizer
        params = gtsam.LevenbergMarquardtParams()
        params.setRelativeErrorTol(1e-8)
        params.setAbsoluteErrorTol(1e-8)
        params.setErrorTol(0)
        params.setMaxIterations(optimization_inerations)
        params.setVerbosityLM("TERMINATION")
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial,params)
        result = optimizer.optimize()
        iterations = optimizer.iterations()

        optimized_transform = result.atPose3(X(0)).matrix()

    if T_gt != []:
        rotation_error_init,translation_error_init = compute_pose_error(T_gt,T_solved_SVD)
        rotation_error,translation_error = compute_pose_error(T_gt,optimized_transform)
    else:
        rotation_error_init = np.nan
        translation_error_init = np.nan
        rotation_error = np.nan
        translation_error = np.nan

    time_2 = time.time()

    # ################vertification################
    vertification_transform = optimized_transform

    if vertification_method == "KNN":
        # Transform source to target frame and compute KNN distance
        source_pcd = test_data_experiment.source_pcd_current
        target_pcd = test_data_experiment.target_pcd_current
        source_pcd.transform(optimized_transform)
        source_pcd = source_pcd.voxel_down_sample(voxel_size_KNN_vertification)
        target_pcd = target_pcd.voxel_down_sample(voxel_size_KNN_vertification)
        error_vertification_quadrics = compute_average_nearest_neighbor_distance(source_pcd,target_pcd,kernel="dcs",hyper_parameter=hyper_parameter)

        error_vertification_quadrics= error_vertification_quadrics

        # print(f'    maxclique_c: {maxclique_count}, SVD_c: {SVD_count}, factor_c: {factor_count}, iter: {iterations}, rot_error: {rotation_error_init:4f}-{rotation_error:4f}, tran_error: {translation_error_init:4f}-{translation_error:4f}, error_vert_KNN: {error_vertification_KNN_mean:4f}')

    elif vertification_method == "quadrics_inliers":
        # Compute correspondence errors
        error_vertification_quadrics = []
        error_vertification_R = []
        error_vertification_t = []

        if len(source_seg_label_list) > 0:
            error_vertification_quadrics,error_vertification_R,error_vertification_t = error_vertification(vertification_transform,source_seg_label_list,target_seg_label_list,quadrics_info_source_stack_copy,quadrics_info_target_stack_copy,center_type_vertification,degenerate_quadrics_aid_vertification,ground_aid,src_ground_points,tgt_ground_points,src_ground_normal,tgt_ground_normal,threshold_Manhattan_vertification)

    else:
        # Extract all source centers
        source_centers_all = []
        source_seg_label_all = []
        source_semantic_all = []
        for key in quadrics_info_source_stack_copy.keys():
            source_centers_all.append(quadrics_info_source_stack_copy[key][center_type_nn])
            source_seg_label_all.append(key)
            source_semantic_all.append(quadrics_info_source_stack_copy[key]["semantic"])
        source_centers_all = np.array(source_centers_all)
        source_seg_label_all = np.array(source_seg_label_all)
        source_semantic_all = np.array(source_semantic_all)

        # Extract all target centers
        target_centers_all = []
        target_seg_label_all = []
        target_semantic_all = []
        for key in quadrics_info_target_stack_copy.keys():
            target_centers_all.append(quadrics_info_target_stack_copy[key][center_type_nn])
            target_seg_label_all.append(key)
            target_semantic_all.append(quadrics_info_target_stack_copy[key]["semantic"])
        target_centers_all = np.array(target_centers_all)                
        target_seg_label_all = np.array(target_seg_label_all)
        target_semantic_all = np.array(target_semantic_all)
        
        # Filter points outside radius_vertification from reference origin
        source_radius_selected_index = (np.linalg.norm(source_centers_all - reference_origin_source,axis=1) < radius_vertification[1]) & (np.linalg.norm(source_centers_all - reference_origin_source,axis=1) > radius_vertification[0])
        source_seg_label_all = source_seg_label_all[source_radius_selected_index]
        source_centers_all = source_centers_all[source_radius_selected_index]
        source_semantic_all = source_semantic_all[source_radius_selected_index]

        target_radius_selected_index = (np.linalg.norm(target_centers_all,axis=1) < radius_vertification[1]) & (np.linalg.norm(target_centers_all,axis=1) > radius_vertification[0])
        target_seg_label_all = target_seg_label_all[target_radius_selected_index]
        target_centers_all = target_centers_all[target_radius_selected_index]
        target_semantic_all = target_semantic_all[target_radius_selected_index]

        # Transform source to target frame and find nearest neighbors
        source_centers_transformed_all = (vertification_transform[:3,:3] @ source_centers_all.T + vertification_transform[:3,3].reshape(-1,1)).T

        if vertification_method == 'quadrics_all':
            nearest_neighbors_indices,nearest_neighbors_dists = nearest_neighbors(source_centers_transformed_all,target_centers_all)

            source_seg_label_all = source_seg_label_all
            target_seg_label_all = list(np.array(target_seg_label_all)[nearest_neighbors_indices])

            error_vertification_quadrics,error_vertification_R,error_vertification_t = error_vertification(vertification_transform,source_seg_label_all,target_seg_label_all,quadrics_info_source_stack_copy,quadrics_info_target_stack_copy,center_type_vertification,degenerate_quadrics_aid_vertification,ground_aid,src_ground_points,tgt_ground_points,src_ground_normal,tgt_ground_normal,threshold_Manhattan_vertification)
        
        elif vertification_method == 'quadrics_semantic':
            # Compute distance per semantic label
            error_vertification_quadrics = []
            error_vertification_R = []
            error_vertification_t = []
            for semantic_current in set(source_semantic_all):
                source_semantic_index = source_semantic_all == semantic_current
                target_semantic_index = target_semantic_all == semantic_current
                source_centers_transformed_semantic = source_centers_transformed_all[source_semantic_index]
                target_centers_semantic = target_centers_all[target_semantic_index]
                source_seg_semantic = source_seg_label_all[source_semantic_index]
                target_seg_semantic = target_seg_label_all[target_semantic_index]

                if source_centers_transformed_semantic.shape[0] == 0 or target_centers_semantic.shape[0] == 0:
                    continue
                else:
                    nearest_neighbors_indices,nearest_neighbors_dists = nearest_neighbors(source_centers_transformed_semantic,target_centers_semantic)

                source_seg_semantic = source_seg_semantic
                target_seg_semantic = list(np.array(target_seg_semantic)[nearest_neighbors_indices])

                error_vertification_quadrics_semantic,error_vertification_R_semantic,error_vertification_t_semantic = error_vertification(vertification_transform,source_seg_semantic,target_seg_semantic,quadrics_info_source_stack_copy,quadrics_info_target_stack_copy,center_type_vertification,degenerate_quadrics_aid_vertification,ground_aid,src_ground_points,tgt_ground_points,src_ground_normal,tgt_ground_normal,threshold_Manhattan_vertification)
                error_vertification_quadrics.extend(error_vertification_quadrics_semantic)
                error_vertification_R.extend(error_vertification_R_semantic)
                error_vertification_t.extend(error_vertification_t_semantic)

    vertification_count = len(error_vertification_quadrics)

    try:
        # error_vertification_quadrics_mean = trimmed_mean(error_vertification_quadrics)
        # error_vertification_R_mean = trimmed_mean(error_vertification_R)
        # error_vertification_t_mean = trimmed_mean(error_vertification_t)
        
        # Apply robust kernel to errors
        error_list = []
        for error in error_vertification_quadrics:
            error_list.append(robust_kernel(error,hyper_parameter,"dcs"))
        error_vertification_quadrics_mean = np.mean(error_list)

        # error_vertification_quadrics_mean = np.mean(error_vertification_quadrics)

    except:

        error_vertification_quadrics_mean = np.mean(error_vertification_quadrics)

    error_vertification_mean = error_vertification_quadrics_mean
    
    print(f'    maxclique_c: {maxclique_count}, SVD_c: {SVD_count}, factor_c: {factor_count}, vertification_c: {vertification_count}, iter: {iterations}, rot_error: {rotation_error_init:4f}-{rotation_error:4f}, tran_error: {translation_error_init:4f}-{translation_error:4f}, error_vert_quadrics: {error_vertification_quadrics_mean:4f}')
    
    time_3 = time.time()

    # print(f"        time SVD-gtsam-ver: {time_1-time_0:4f}-{time_2-time_1:4f}-{time_3-time_2:4f}")

    return error_vertification_mean,{"maxclique_count":maxclique_count,"SVD_count":SVD_count,"factor_count":factor_count,"iterations":iterations,"rotation_error_init":rotation_error_init,"rotation_error":rotation_error,"translation_error_init":translation_error_init,"translation_error":translation_error,"error_vertification":error_vertification_mean,"T_optimized":optimized_transform}