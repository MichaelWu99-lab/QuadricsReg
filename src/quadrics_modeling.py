import numpy as np
import open3d as o3d
import os
import shutil

from src.data_processing import *
from src.quadrics_generation import *
from src.quadrics_matching import matching_info_dict_prune

import elements_extractor_bindings
import maxclique_solver_bindings

def elements_extractor(points_source, points_target, DATA_SET_config, test_data_experiment, extraction_method, extraction_config_source="./configs/elements_extractor.yaml", extraction_config_target="./configs/elements_extractor.yaml",VOXEL_SIZE_quadrics=0):
    if VOXEL_SIZE_quadrics > 0:
        pcd_source = o3d.geometry.PointCloud()
        pcd_source.points = o3d.utility.Vector3dVector(points_source)
        pcd_source_down = pcd_source.voxel_down_sample(VOXEL_SIZE_quadrics)
        points_source = np.asarray(pcd_source_down.points)

        pcd_target = o3d.geometry.PointCloud()
        pcd_target.points = o3d.utility.Vector3dVector(points_target)
        pcd_target_down = pcd_target.voxel_down_sample(VOXEL_SIZE_quadrics)
        points_target = np.asarray(pcd_target_down.points)

    if extraction_method == "cluster":
        key_elements_dict_source = elements_extractor_bindings.extract_features(points_source, extraction_config_source, seg_type="all")
        key_elements_dict_target = elements_extractor_bindings.extract_features(points_target, extraction_config_target, seg_type="all")
    else:
        key_elements_dict_source = {}
        # extract key structures
        key_elements_dict_source_temp = elements_extractor_bindings.extract_features(points_source, extraction_config_source, seg_type='all')
        key_elements_dict_source['plane'] = key_elements_dict_source_temp['plane']
        key_elements_dict_source['line'] = key_elements_dict_source_temp['line']
        key_elements_dict_source['ground'] = key_elements_dict_source_temp['ground']
        # extract key objects
        for semantic in test_data_experiment.source_xyz_semantic_dict.keys():
            if semantic not in DATA_SET_config["semantics_seg"].keys():
                continue
            seg_type = DATA_SET_config["semantics_seg"][semantic]
            key_elements_dict_source_temp = elements_extractor_bindings.extract_features(test_data_experiment.source_xyz_semantic_dict[semantic], extraction_config_source, seg_type=seg_type)
            key_elements_dict_source[semantic] = key_elements_dict_source_temp[seg_type]

        key_elements_dict_target = {}
        # extract key structures
        key_elements_dict_target_temp = elements_extractor_bindings.extract_features(points_target, extraction_config_target, seg_type='all')
        key_elements_dict_target['plane'] = key_elements_dict_target_temp['plane']
        key_elements_dict_target['line'] = key_elements_dict_target_temp['line']
        key_elements_dict_target['ground'] = key_elements_dict_target_temp['ground']
        # extract key objects
        for semantic in test_data_experiment.target_xyz_semantic_dict.keys():
            if semantic not in DATA_SET_config["semantics_seg"].keys():
                continue
            seg_type = DATA_SET_config["semantics_seg"][semantic]
            key_elements_dict_target_temp = elements_extractor_bindings.extract_features(test_data_experiment.target_xyz_semantic_dict[semantic], extraction_config_target, seg_type=seg_type)
            key_elements_dict_target[semantic] = key_elements_dict_target_temp[seg_type]
            
    return key_elements_dict_source, key_elements_dict_target

def quadrics_fitting(key_elements_dict, instanceType_used_dict, topVolume_k_save, if_save_representation):
    key_elements_quadrics_data_dict = {}
    key_elements_save_num = 0
    key_elements_save_num_val = 0

    for element_semantic_type in key_elements_dict.keys():
        if element_semantic_type not in instanceType_used_dict:
            continue

        if element_semantic_type == 'ground':
            normal_ground = ground_normal_estimate(key_elements_dict[element_semantic_type][0])
            key_elements_quadrics_data_dict['ground'] = {'ground_normal': normal_ground}
            continue

        element_quadrics_type = instanceType_used_dict[element_semantic_type]
        elements_semantic_type_quadrics_data_dict = {}

        for element_index in key_elements_dict[element_semantic_type].keys():
            element_quadrics_data_dict = {}
            element_points = key_elements_dict[element_semantic_type][element_index]

            points_scaled, _, points_raw_scaled, points_raw, normals__temp, T, T_s_t, axis_original = process_data_all(element_points, d_scale=True)

            if element_quadrics_type in ["sphere", "ellipsoid"]:
                output, output_c, C, trans_inv = quadrics_generation_from_pca(points_raw_scaled, element_quadrics_type)
            elif element_quadrics_type == "plane":
                output, output_c, C, trans_inv = quadrics_generation_from_pca(points_raw_scaled, element_quadrics_type)
            elif element_quadrics_type in ["cylinder", "elliptic_cylinder", "cone", "elliptic_cone"]:
                output, output_c, C, trans_inv = quadrics_generation_from_obb(points_raw_scaled, element_quadrics_type, axis_original)
            elif element_quadrics_type == "line":
                output, output_c, C, trans_inv = quadrics_generation_line_from_pca(points_raw_scaled, element_quadrics_type, axis_original)
            else:
                print("Error: element_quadrics_type not supported")
                continue

            # Decomposition: extract geometric info
            if element_quadrics_type in ["sphere", "ellipsoid", "plane"]:
                center, center_statistics, scale, rotation, Is, Ir = quadrics_decomposition_info_extracting_pca(points_raw_scaled, output, element_quadrics_type, trans_inv, C)
            elif element_quadrics_type in ["cylinder", "elliptic_cylinder", "cone", "elliptic_cone"]:
                center, center_statistics, scale, rotation, Is, Ir = quadrics_decomposition_info_extracting_obb(points_raw_scaled, axis_original, output, element_quadrics_type, trans_inv, C)
            else:
                center, center_statistics, scale, rotation, Is, Ir = quadrics_decomposition_info_extracting_line_pca(points_raw_scaled, output, element_quadrics_type, trans_inv, C)

            T, T_s_t, output, output_c, center, center_c, center_statistics, center_statistics_c, scale, rotation = rescale_input_outputs_quadrics_test_anything(T, T_s_t, output, output_c, center, center_statistics, scale, rotation, Is, Ir)

            element_quadrics_data_dict['semantic'] = element_semantic_type
            element_quadrics_data_dict['points_raw'] = element_points
            element_quadrics_data_dict['quadrics_coeff'] = output
            element_quadrics_data_dict['full_center'] = center_statistics_c
            element_quadrics_data_dict['full_scale'] = scale
            element_quadrics_data_dict['full_rotation'] = rotation
            element_volume = volume_shape(scale, element_quadrics_type)
            element_volume_norm = volume_shape_normlize(scale, element_quadrics_type)
            element_quadrics_data_dict['volume'] = element_volume
            element_quadrics_data_dict['volume_norm'] = element_volume_norm
            if if_save_representation:
                mesh = genrate_mesh_open3d(element_quadrics_type, scale, center, rotation)
                element_quadrics_data_dict['mesh'] = mesh

            elements_semantic_type_quadrics_data_dict[element_index] = element_quadrics_data_dict
            key_elements_save_num += 1

        dict_temp = dict(sorted(elements_semantic_type_quadrics_data_dict.items(), key=lambda item: -item[1]['volume_norm'])[:topVolume_k_save])
        dict_temp = dict(sorted(dict_temp.items(), key=lambda item: -item[1]['volume']))
        elements_semantic_type_quadrics_data_dict = {i+key_elements_save_num_val: value for i, (key, value) in enumerate(dict_temp.items())}
        key_elements_save_num_val = key_elements_save_num_val + len(elements_semantic_type_quadrics_data_dict.keys())

        key_elements_quadrics_data_dict[element_semantic_type] = elements_semantic_type_quadrics_data_dict

    return key_elements_quadrics_data_dict, key_elements_save_num

def reconstruction_save_demo(key_elements_quadrics_data_dict, key_elements_quadrics_data_point_augmented_dict,save_dir,tag='source'):

    if os.path.exists(f"{save_dir}/{tag}"):
        shutil.rmtree(f"{save_dir}/{tag}")
    os.makedirs(f"{save_dir}/{tag}")

    key_elements_mesh = o3d.geometry.TriangleMesh()
    for element_semantic_type in key_elements_quadrics_data_dict.keys():
        if element_semantic_type == 'ground':
            continue
        for element_index in key_elements_quadrics_data_dict[element_semantic_type].keys():
            key_elements_mesh += key_elements_quadrics_data_dict[element_semantic_type][element_index]['mesh']
            # delete mesh key
            key_elements_quadrics_data_dict[element_semantic_type][element_index].pop("mesh")
    o3d.io.write_triangle_mesh(f"{save_dir}/{tag}/reconstruction_merge_mesh.ply", key_elements_mesh)

    ############################################
    point_augmented_pcd = o3d.geometry.PointCloud()
    if len(key_elements_quadrics_data_point_augmented_dict.keys()) > 0:
        for element_semantic_type in key_elements_quadrics_data_point_augmented_dict.keys():
            for element_index in key_elements_quadrics_data_point_augmented_dict[element_semantic_type].keys():
                point_augmented_pcd.points.extend(o3d.utility.Vector3dVector(key_elements_quadrics_data_point_augmented_dict[element_semantic_type][element_index]['full_center'].reshape(1,3)))

        o3d.io.write_point_cloud(f"{save_dir}/{tag}/point_augmented.ply", point_augmented_pcd)

def point_augmentation_for_modeling(key_elements_quadrics_data_dict, on_point_threshold_all, key_elements_save_num, topVolume_k_point, VOXEL_SIZE_point,point_augmentation=False):

    # point augmentation for topVolume_k_point elements
    if key_elements_save_num < on_point_threshold_all or point_augmentation:
        key_elements_quadrics_data_point_augmented_dict = {}

        point_count = key_elements_save_num # name from key_elements_save_num
        for element_semantic_type in key_elements_quadrics_data_dict.keys():
            if element_semantic_type == 'ground':
                continue

            element_point_augment_count = 0 # no mare than topVolume_k_point
            element_semantic_type_point = element_semantic_type + "_point"
            key_elements_quadrics_data_point_augmented_dict[element_semantic_type_point] = {}

            for element_index in key_elements_quadrics_data_dict[element_semantic_type].keys():

                if element_point_augment_count >= topVolume_k_point:
                    # delete points_raw key
                    key_elements_quadrics_data_dict[element_semantic_type][element_index].pop("points_raw")
                    continue

                # voxel down sample for topVolume_k_point elements
                element_points = key_elements_quadrics_data_dict[element_semantic_type][element_index]["points_raw"]
                # delete points_raw key
                key_elements_quadrics_data_dict[element_semantic_type][element_index].pop("points_raw")
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(element_points)
                pcd_down = pcd.voxel_down_sample(VOXEL_SIZE_point)
                element_points_down = np.asarray(pcd_down.points)
                output,centers_points = quadrics_generation_point(element_points_down)

                # save element data
                for i in range(element_points_down.shape[0]):
                    j = point_count + i
                    key_elements_quadrics_data_point_augmented_dict[element_semantic_type_point][j] = {}

                    key_elements_quadrics_data_point_augmented_dict[element_semantic_type_point][j]['semantic'] = element_semantic_type_point
                    key_elements_quadrics_data_point_augmented_dict[element_semantic_type_point][j]['quadrics_type'] = 'point'
                    key_elements_quadrics_data_point_augmented_dict[element_semantic_type_point][j]['quadrics_coeff'] = output[i]
                    key_elements_quadrics_data_point_augmented_dict[element_semantic_type_point][j]['full_center'] = centers_points[i]
                    key_elements_quadrics_data_point_augmented_dict[element_semantic_type_point][j]['full_scale'] = np.zeros(3).astype(np.int8)
                    key_elements_quadrics_data_point_augmented_dict[element_semantic_type_point][j]['full_rotation'] = np.eye(3).astype(np.int8)
                
                point_count += element_points_down.shape[0]

                element_point_augment_count += 1

        return key_elements_quadrics_data_point_augmented_dict
    else:
        return {}

def quadrics_artributions_extracting(q,prim_current):

    Q = q_Q(q)
    Q,_ = quadrics_scale_identification(Q,prim_current)

    E = Q[:3, :3]
    value, vector = np.linalg.eig(E)
    idx = np.argsort(-value)
    value = value[idx]
    vector = vector[:, idx]
    Is,Ir,It = quadrics_judgment(value)

    if prim_current in ["cone","elliptic_cone"]:
        if sum(value < 0) == 1:
            factor_gt = -value[value < 0]
            value = value / factor_gt
            scale = scale * np.sqrt(factor_gt)
    
    if prim_current in ["line"]:
        Is = Is * 0
    
    scale = np.sqrt(1 / (np.abs(value)+ 1e-8)) * Is
    
    for index_column in range(vector.shape[1]):
        if vector[:,index_column][np.where(np.abs(vector[:,index_column]) > 0)][0]<0:
            vector[:,index_column] = vector[:,index_column] * -1
    vector = vector * Ir
    
    return scale,vector,Is,Ir,It,E

def quadrics_info_extration(key_elements_quadrics_data_dict,DATA_SET_config):
    key_elements_quadrics_data_dict = matching_info_dict_prune(key_elements_quadrics_data_dict,DATA_SET_config)
    for element_semantic_type in key_elements_quadrics_data_dict.keys():
        if element_semantic_type == 'ground':
            continue
        if element_semantic_type not in DATA_SET_config['semantics_quadrics_reg'].keys():
            key_elements_quadrics_data_dict.pop(element_semantic_type)
            continue
                    
        for element_index in key_elements_quadrics_data_dict[element_semantic_type].keys():

            element_quadrics_data_dict = key_elements_quadrics_data_dict[element_semantic_type][element_index]

            element_quadrics_data_dict['quadrics_type'] = DATA_SET_config['semantics_quadrics_reg'][element_semantic_type]

            quadrics_artibutions_temp = quadrics_artributions_extracting(element_quadrics_data_dict['quadrics_coeff'],element_quadrics_data_dict['quadrics_type'])

            element_quadrics_data_dict["decomposition_scale"] = quadrics_artibutions_temp[0]
            element_quadrics_data_dict["decomposition_rotation"] = quadrics_artibutions_temp[1]
            element_quadrics_data_dict["decomposition_Is"] = quadrics_artibutions_temp[2]
            element_quadrics_data_dict["decomposition_Ir"] = quadrics_artibutions_temp[3]
            element_quadrics_data_dict["decomposition_It"] = quadrics_artibutions_temp[4]
            element_quadrics_data_dict["E"] = quadrics_artibutions_temp[5]

            element_quadrics_type = element_quadrics_data_dict['quadrics_type']
            if element_quadrics_type in ["plane"]:
                element_quadrics_data_dict["full_Is"] = np.array([1,1,1])
                element_quadrics_data_dict["full_Ir"] = np.array([1,1,1])
            elif element_quadrics_type in ["cylinder","elliptic_cylinder"]:
                element_quadrics_data_dict["full_Is"] = np.array([1,1,1])
                element_quadrics_data_dict["full_Ir"] = np.array([1,1,1])
            elif element_quadrics_type in ["cone","elliptic_cone"]:
                element_quadrics_data_dict["full_Is"] = np.array([1,1,1])
                element_quadrics_data_dict["full_Ir"] = np.array([1,1,1])
            elif element_quadrics_type in ["sphere","ellipsoid"]:
                element_quadrics_data_dict["full_Is"] = np.array([1,1,1])
                element_quadrics_data_dict["full_Ir"] = np.array([1,1,1])
            elif element_quadrics_type in ["line"]:
                element_quadrics_data_dict["full_Is"] = np.array([1,1,1])
                element_quadrics_data_dict["full_Ir"] = np.array([1,1,1])

            element_quadrics_data_dict["full_E"] = element_quadrics_data_dict["full_rotation"].reshape(3,3) @ (np.diag(element_quadrics_data_dict["full_scale"])** 2) @ element_quadrics_data_dict["full_rotation"].reshape(3,3).T

            element_quadrics_data_dict["full_quadrics_coeff"] = generate_ellipsoid_coeff(element_quadrics_data_dict["full_center"], element_quadrics_data_dict["full_scale"], element_quadrics_data_dict["full_rotation"])

            key_elements_quadrics_data_dict[element_semantic_type][element_index] = element_quadrics_data_dict
    return key_elements_quadrics_data_dict