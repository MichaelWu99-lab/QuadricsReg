import numpy as np
import open3d as o3d
import os
import time
import yaml
import argparse

from src.quadrics_modeling import *
from src.quadrics_matching import *
from src.quadarics_transformation_estimation_demo import *

import elements_extractor_bindings
import maxclique_solver_bindings

if __name__ == "__main__":

    testsample_dir = 'demo/result'
    os.makedirs(testsample_dir, exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--pcd_source_path', type=str, default='demo/pcd/vlp64_source.ply')
    parser.add_argument('--pcd_target_path', type=str, default='demo/pcd/vlp64_target.ply')
    parser.add_argument('--voxel_size', type=float, default=0.00)
    parser.add_argument('--if_centerize', type=bool, default=False)
    parser.add_argument('--config_quadricsReg', type=str, default='configs/quadricsReg_demo.yaml')
    parser.add_argument('--extraction_config_source', type=str, default='configs/extraction/elements_extractor_vlp64.yaml')
    parser.add_argument('--extraction_config_target', type=str, default='configs/extraction/elements_extractor_vlp64.yaml')

    args = parser.parse_args()

    pcd_source_path = args.pcd_source_path
    pcd_target_path = args.pcd_target_path
    voxel_size = args.voxel_size
    if_centerize = args.if_centerize
    config_quadricsReg = args.config_quadricsReg
    extraction_config_source = args.extraction_config_source
    extraction_config_target = args.extraction_config_target

    with open(config_quadricsReg, 'r', encoding='utf-8') as file:
        quadricsReg_config = yaml.safe_load(file)

    if_eval = quadricsReg_config["if_eval"]
    if_save_representation = quadricsReg_config["if_save_representation"]

    extraction_method = quadricsReg_config["elementExtraction"]["extraction_method"]

    topVolume_k_save = quadricsReg_config["quadricsFitting"]["topVolume_k_save"]
    on_point_threshold_all = quadricsReg_config["quadricsFitting"]["on_point_threshold_all"]
    topVolume_k_point = quadricsReg_config["quadricsFitting"]["topVolume_k_point"]
    VOXEL_SIZE_point = quadricsReg_config["quadricsFitting"]["VOXEL_SIZE_point"]

    topK_correspondence_used = quadricsReg_config["quadricsMatching"]["topK_correspondence_used"]
    threshold_distance_list = quadricsReg_config["quadricsMatching"]["threshold_distance_list"]
    threshold_correspondence_inlier_distance = quadricsReg_config["quadricsMatching"]["threshold_correspondence_inlier_distance"]
    threshold_correspondence_inlier_num = quadricsReg_config["quadricsMatching"]["threshold_correspondence_inlier_num"]
    grad_pmc = quadricsReg_config["quadricsMatching"]["grad_pmc"]
    max_clique_time_limit = quadricsReg_config["quadricsMatching"]["max_clique_time_limit"]
    kcore_heuristic_threshold = quadricsReg_config["quadricsMatching"]["kcore_heuristic_threshold"]

    instanceType_used_dict = quadricsReg_config["semantics_quadrics"]
    DATA_SET_config = quadricsReg_config

    # key elements extraction
    pcd_source = o3d.io.read_point_cloud(pcd_source_path)
    pcd_target = o3d.io.read_point_cloud(pcd_target_path)

    if voxel_size > 0:
        pcd_source = pcd_source.voxel_down_sample(voxel_size=voxel_size)
        pcd_target = pcd_target.voxel_down_sample(voxel_size=voxel_size)

    points_source = np.asarray(pcd_source.points).astype(np.float32)
    points_target = np.asarray(pcd_target.points).astype(np.float32)

    if if_centerize:
        points_source -= np.mean(points_source, axis=0, keepdims=True)
        points_target -= np.mean(points_target, axis=0, keepdims=True)

    time_start = time.time()
    key_elements_dict_source = elements_extractor_bindings.extract_features(points_source, extraction_config_source, z_up=0.0)
    key_elements_dict_target = elements_extractor_bindings.extract_features(points_target, extraction_config_target, z_up=0.0)
    time_extract = time.time()

    key_elements_quadrics_data_dict_source, key_elements_save_num_source = quadrics_fitting(key_elements_dict_source, instanceType_used_dict, topVolume_k_save, if_save_representation)
    key_elements_quadrics_data_point_augmented_dict_source = point_augmentation_for_modeling(key_elements_quadrics_data_dict_source, on_point_threshold_all, key_elements_save_num_source, topVolume_k_point, VOXEL_SIZE_point)
    key_elements_quadrics_data_dict_target, key_elements_save_num_target = quadrics_fitting(key_elements_dict_target, instanceType_used_dict, topVolume_k_save, if_save_representation)
    key_elements_quadrics_data_point_augmented_dict_target = point_augmentation_for_modeling(key_elements_quadrics_data_dict_target, on_point_threshold_all, key_elements_save_num_target, topVolume_k_point, VOXEL_SIZE_point)

    time_model = time.time()

    # save quadrics representation
    if if_save_representation:
        reconstruction_save_demo(key_elements_quadrics_data_dict_source, key_elements_quadrics_data_point_augmented_dict_source, save_dir=testsample_dir, tag="source")
        reconstruction_save_demo(key_elements_quadrics_data_dict_target, key_elements_quadrics_data_point_augmented_dict_target, save_dir=testsample_dir, tag="target")
    time_reconstruction = time.time()

    # matching
    key_elements_quadrics_data_dict_source = quadrics_info_extration(key_elements_quadrics_data_dict_source, DATA_SET_config)
    key_elements_quadrics_data_dict_target = quadrics_info_extration(key_elements_quadrics_data_dict_target, DATA_SET_config)

    quadrics_correspondences = {}
    quadrics_correspondences_with_info_all = {}
    for element_semantic_type in key_elements_quadrics_data_dict_source.keys():
        if element_semantic_type == "ground":
            continue
        if element_semantic_type not in key_elements_quadrics_data_dict_target.keys():
            continue

        print(f"    {element_semantic_type:15s}, Num_seg_source: {len(key_elements_quadrics_data_dict_source[element_semantic_type])}, Num_seg_target: {len(key_elements_quadrics_data_dict_target[element_semantic_type])}")

        quadrics_correspondences[element_semantic_type] = quadrics_artribution_matching_realsence_mutual(key_elements_quadrics_data_dict_source[element_semantic_type], key_elements_quadrics_data_dict_target[element_semantic_type], topK_correspondence_used)

        for correspondence in quadrics_correspondences[element_semantic_type]:
            quadrics_correspondences_with_info_all[correspondence] = [key_elements_quadrics_data_dict_source[element_semantic_type][correspondence[0]], key_elements_quadrics_data_dict_target[element_semantic_type][correspondence[1]]]

    # match augmented points
    point_correspondences_with_info_all, key_elements_quadrics_data_point_augmented_dict_source, key_elements_quadrics_data_point_augmented_dict_target = process_point_correspondences_with_info(key_elements_quadrics_data_point_augmented_dict_source, key_elements_quadrics_data_point_augmented_dict_target, DATA_SET_config)

    time_match = time.time()

    # merge correspondence info
    quadrics_correspondences_with_info_all.update(point_correspondences_with_info_all)
    quadrics_info_source_stack = {}
    quadrics_info_target_stack = {}
    quadrics_info_source_stack.update(stack_dict(key_elements_quadrics_data_dict_source))
    quadrics_info_source_stack.update(stack_dict(key_elements_quadrics_data_point_augmented_dict_source))
    quadrics_info_target_stack.update(stack_dict(key_elements_quadrics_data_dict_target))
    quadrics_info_target_stack.update(stack_dict(key_elements_quadrics_data_point_augmented_dict_target))

    # compatibility check
    correspondence_consistency_dict = distance_consistency_check(quadrics_correspondences_with_info_all, threshold_distance_list)
    time_compatibility_check = time.time()

    # feature matching evaluation
    if if_eval:
        T_gt = np.loadtxt(f'{testsample_dir}/T_icp.txt').reshape(-1).reshape(4, 4)
        correspondence_num, correspondence_inlier_ratio, correspondence_inlier_num = computer_correspondence_inlier(quadrics_correspondences_with_info_all, T_gt, threshold_correspondence_inlier_distance=threshold_correspondence_inlier_distance)
    else:
        T_gt = []
        correspondence_num = np.nan
        correspondence_inlier_ratio = np.nan
        correspondence_inlier_num = np.nan

    # find max cliques and solve 6-DoF transformation
    prune_level = 0
    vertification_info_dict = {}
    error_vertification_list = []
    for threshold_distance, correspondence_consistency in correspondence_consistency_dict.items():

        pair_index = 0
        list_all = []

        for pair in correspondence_consistency:
            list_all.append([pair[0][0], pair[0][1], pair[1][0], pair[1][1]])

        data_matrix = np.array(list_all)
        try:
            correspondence_max_clique = maxclique_solver_bindings.process_data_matrix(
                data_matrix,
                max_clique_time_limit,
                kcore_heuristic_threshold,
                prune_level
            )
        except:
            print(f"maxclique_solver_bindings error!")
            continue
        correspondence_max_clique_tuple_list = [(correspondence_max_clique[i], correspondence_max_clique[i+1]) for i in range(0, len(correspondence_max_clique), 2)]
        prune_level = len(correspondence_max_clique_tuple_list) if grad_pmc else 0

        # solve 6-DoF transformation
        error_vertification_mean, vertification_info_dict[threshold_distance] = solve_gtsam_multi_demo(pcd_source, pcd_target, correspondence_max_clique_tuple_list, quadrics_info_source_stack, quadrics_info_target_stack, quadricsReg_config=quadricsReg_config, T_gt=T_gt)

        error_vertification_list.append(error_vertification_mean)

    error_min_index = np.argmin(error_vertification_list)

    selected_info = vertification_info_dict[threshold_distance_list[error_min_index]]

    selected_rotation_error = selected_info["rotation_error"]
    selected_translation_error = selected_info["translation_error"]
    selected_translation_error_init = selected_info["translation_error_init"]
    selected_rotation_error_init = selected_info["rotation_error_init"]
    selected_T_optimized = selected_info["T_optimized"]

    time_maxclique_and_estimation = time.time()

    print("  Time, extract: {:.4f}, model: {:.4f}, reconstruction: {:.4f}, match: {:.4f}, compatibility_check: {:.4f}, maxclique_and_estimation: {:.4f}".format(
        time_extract-time_start,
        time_model-time_extract,
        time_reconstruction-time_model,
        time_match-time_reconstruction,
        time_compatibility_check-time_match,
        time_maxclique_and_estimation-time_compatibility_check))
    print("  Time, total: {:.4f}, registration: {:.4f}".format(
        time_maxclique_and_estimation-time_start,
        time_maxclique_and_estimation-time_reconstruction))

    success_flag = '-'
    if if_eval:
        if selected_rotation_error < 5 and selected_translation_error < 2:
            success_flag = '+'
    else:
        success_flag = '?'

    print(f'  {success_flag}, select {error_min_index}, rot_error: {selected_info["rotation_error_init"]:4f}-{selected_info["rotation_error"]:4f}, tran_error: {selected_info["translation_error_init"]:4f}-{selected_info["translation_error"]:4f}, error_vert: {selected_info["error_vertification"]:4f}')

    # save registration result
    T_path = os.path.join(testsample_dir, 'T_optimized.txt')
    pcd_path = os.path.join(testsample_dir, 'pcd_combined.ply')
    np.savetxt(T_path, selected_T_optimized, fmt='%f', delimiter=' ')
    pcd_source.transform(selected_T_optimized)
    pcd_source.paint_uniform_color([1, 0.706, 0])
    pcd_target.paint_uniform_color([0, 0.651, 0.929])
    pcd_combined = pcd_source + pcd_target
    o3d.io.write_point_cloud(pcd_path, pcd_combined)

    print(f"  Estimated transformation saved to: {os.path.abspath(T_path)}")
    print(f"  Registered point cloud saved to:   {os.path.abspath(pcd_path)}")
