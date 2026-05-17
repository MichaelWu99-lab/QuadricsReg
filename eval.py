import numpy as np
import open3d as o3d
import time
import yaml
import argparse
import shutil

from src.quadrics_modeling import *
from src.quadrics_matching import *
from src.quadarics_transformation_estimation import *
from src.experiment import *

if __name__ == "__main__":

    testsample_dir = './'

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_quadricsReg_path', type=str, default='configs/quadricsReg.yaml')
    parser.add_argument('--DATA_SET_name', type=str, default='KITTI')
    parser.add_argument('--DATA_SET_prepare_type', type=str, default='lc')
    parser.add_argument('--mode', type=str, default='20_30')
    parser.add_argument('--extraction_config_source', type=str, default='configs/elements_extractor.yaml')
    parser.add_argument('--extraction_config_target', type=str, default='configs/elements_extractor.yaml')
    args = parser.parse_args()

    print(f"args: {args}")

    config_quadricsReg_path = args.config_quadricsReg_path
    DATA_SET_name = args.DATA_SET_name
    DATA_SET_prepare_type = args.DATA_SET_prepare_type
    mode = args.mode
    extraction_config_source = args.extraction_config_source
    extraction_config_target = args.extraction_config_target

    with open(config_quadricsReg_path, 'r', encoding='utf-8') as file:
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

    with open(f'./configs/data_set/{DATA_SET_name}.yaml', 'r', encoding='utf-8') as file:
        DATA_SET_config = yaml.safe_load(file)
    instanceType_used_dict = DATA_SET_config["semantics_quadrics"]

    # load dataset
    test_data_experiment = test_data(DATA_SET_config, DATA_SET_prepare_type, mode, extraction_method, if_eval)

    while True:

        points_source, points_target = test_data_experiment.load_data()

        if points_source is None:
            break

        time_start = time.time()

        key_elements_dict_source, key_elements_dict_target = elements_extractor(points_source, points_target, DATA_SET_config, test_data_experiment, extraction_method, extraction_config_source=extraction_config_source, extraction_config_target=extraction_config_target)

        time_extract = time.time()

        key_elements_quadrics_data_dict_source, key_elements_save_num_source = quadrics_fitting(key_elements_dict_source, instanceType_used_dict, topVolume_k_save, if_save_representation)
        key_elements_quadrics_data_point_augmented_dict_source = point_augmentation_for_modeling(key_elements_quadrics_data_dict_source, on_point_threshold_all, key_elements_save_num_source, topVolume_k_point, VOXEL_SIZE_point)
        key_elements_quadrics_data_dict_target, key_elements_save_num_target = quadrics_fitting(key_elements_dict_target, instanceType_used_dict, topVolume_k_save, if_save_representation)
        key_elements_quadrics_data_point_augmented_dict_target = point_augmentation_for_modeling(key_elements_quadrics_data_dict_target, on_point_threshold_all, key_elements_save_num_target, topVolume_k_point, VOXEL_SIZE_point)
        time_model = time.time()

        # save quadrics representation
        if if_save_representation:
            test_data_experiment.save_representation(key_elements_quadrics_data_dict_source, key_elements_quadrics_data_point_augmented_dict_source, key_elements_quadrics_data_dict_target, key_elements_quadrics_data_point_augmented_dict_target)
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

            quadrics_correspondences[element_semantic_type] = quadrics_artribution_matching_realsence_mutual(key_elements_quadrics_data_dict_source[element_semantic_type], key_elements_quadrics_data_dict_target[element_semantic_type], topK_correspondence_used)

            for correspondence in quadrics_correspondences[element_semantic_type]:
                quadrics_correspondences_with_info_all[correspondence] = [key_elements_quadrics_data_dict_source[element_semantic_type][correspondence[0]], key_elements_quadrics_data_dict_target[element_semantic_type][correspondence[1]]]

        output_line = " | ".join(
            f"{element_semantic_type:8s}: {len(key_elements_quadrics_data_dict_source[element_semantic_type])}/{len(key_elements_quadrics_data_dict_target[element_semantic_type])}"
            for element_semantic_type in key_elements_quadrics_data_dict_source.keys()
            if element_semantic_type != "ground" and element_semantic_type in key_elements_quadrics_data_dict_target
        )
        print("    elements_semantic_used src/tag ->", output_line)

        # match augmented points
        quadrics_info_source_stack_point = stack_dict(key_elements_quadrics_data_point_augmented_dict_source)
        quadrics_info_target_stack_point = stack_dict(key_elements_quadrics_data_point_augmented_dict_target)
        point_correspondences_with_info_all = process_point_correspondences_with_info_simplified(quadrics_info_source_stack_point, quadrics_info_target_stack_point, DATA_SET_config)

        time_prematch = time.time()

        # merge correspondence info
        quadrics_correspondences_with_info_all.update(point_correspondences_with_info_all)
        quadrics_info_source_stack = {}
        quadrics_info_target_stack = {}
        quadrics_info_source_stack.update(stack_dict(key_elements_quadrics_data_dict_source))
        quadrics_info_source_stack.update(quadrics_info_source_stack_point)
        quadrics_info_target_stack.update(stack_dict(key_elements_quadrics_data_dict_target))
        quadrics_info_target_stack.update(quadrics_info_target_stack_point)

        if if_save_representation:
            with open(f"{test_data_experiment.save_dir_current}/correspondences_init.csv", "w") as f:
                for pair_save in quadrics_correspondences_with_info_all.keys():
                    source_index = pair_save[0]
                    target_index = pair_save[1]
                    source_center = quadrics_info_source_stack[source_index]['full_center']
                    target_center = quadrics_info_target_stack[target_index]['full_center']
                    f.write(f"{source_index},{target_index},{source_center[0]},{source_center[1]},{source_center[2]},{target_center[0]},{target_center[1]},{target_center[2]}\n")

        # compatibility check
        correspondence_consistency_dict = distance_consistency_check(quadrics_correspondences_with_info_all, threshold_distance_list)
        time_compatibility_check = time.time()

        # find max cliques and solve 6-DoF transformation
        prune_level = 0
        vertification_info_dict = {}
        error_vertification_list = []
        time_maxclique_list = []
        time_estimation_list = []
        for threshold_distance, correspondence_consistency in correspondence_consistency_dict.items():

            time_maxclique_start_temp = time.time()

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

            time_estimation_start_temp = time.time()
            time_maxclique_list.append(time_estimation_start_temp-time_maxclique_start_temp)

            # solve 6-DoF transformation
            error_vertification_mean, vertification_info_dict[threshold_distance] = solve_gtsam_multi(test_data_experiment, correspondence_max_clique_tuple_list, quadrics_info_source_stack, quadrics_info_target_stack, quadricsReg_config=quadricsReg_config, T_gt=test_data_experiment.T_gt)

            error_vertification_list.append(error_vertification_mean)

            time_maxclique_and_estimation_temp = time.time()
            time_estimation_list.append(time_maxclique_and_estimation_temp-time_estimation_start_temp)

            if if_save_representation:
                with open(f"{test_data_experiment.save_dir_current}/correspondences_{threshold_distance}.csv", "w") as f:
                    for pair_save in correspondence_max_clique_tuple_list:
                        source_index = pair_save[0]
                        target_index = pair_save[1]
                        source_center = quadrics_info_source_stack[source_index]['full_center']
                        target_center = quadrics_info_target_stack[target_index]['full_center']
                        f.write(f"{source_index},{target_index},{source_center[0]},{source_center[1]},{source_center[2]},{target_center[0]},{target_center[1]},{target_center[2]}\n")

        error_min_index = np.argmin(error_vertification_list)

        if if_save_representation:
            shutil.copyfile(f"{test_data_experiment.save_dir_current}/correspondences_{threshold_distance_list[error_min_index]}.csv", f"{test_data_experiment.save_dir_current}/correspondences_optimal.csv")

        selected_info = vertification_info_dict[threshold_distance_list[error_min_index]]
        selected_T_optimized = selected_info["T_optimized"]
        print(f'    select {error_min_index}')

        time_maxclique_and_estimation = time.time()
        print(f"    Time, ext: {time_extract-time_start:4f}, model: {time_model-time_extract:4f}, recons: {time_reconstruction-time_model:4f}, match: {time_prematch-time_reconstruction:4f}, comp_check: {time_compatibility_check-time_prematch:4f}, maxclique: {np.sum(time_maxclique_list):4f}, estimation: {np.sum(time_estimation_list):4f}")

        # save registration result and analysis
        test_data_experiment.T_estimation = selected_T_optimized
        test_data_experiment.time_total = time_maxclique_and_estimation-time_start
        test_data_experiment.time_reg = time_maxclique_and_estimation-time_reconstruction
        test_data_experiment.time_extract = time_extract-time_start
        test_data_experiment.time_model = time_model-time_extract
        test_data_experiment.time_reconstruction = time_reconstruction-time_model
        test_data_experiment.time_prematch = time_prematch-time_reconstruction
        test_data_experiment.time_compatibility_check = time_compatibility_check-time_prematch
        test_data_experiment.time_maxclique = np.sum(time_maxclique_list)
        test_data_experiment.time_estimation = np.sum(time_estimation_list)
        test_data_experiment.save_registration()

        if if_eval:
            test_data_experiment.computer_correspondence_inlier(quadrics_correspondences_with_info_all, threshold_correspondence_inlier_distance=threshold_correspondence_inlier_distance)
            test_data_experiment.log_analysis_results()

    if if_eval:
        test_data_experiment.save_analysis_results()
