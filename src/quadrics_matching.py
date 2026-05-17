import time
import numpy as np
import open3d as o3d
from scipy.spatial import distance,cKDTree
from scipy.spatial.distance import pdist, squareform

def quadrics_artribution_matching_realsence_mutual(quadrics_info_source,quadrics_info_target,topK_correspondence_used=20):

    quadrics_correspondences_selected = []
    quadrics_correspondences_selected_source = []
    for quadrics_segment_source_index in quadrics_info_source.keys():
        quadrics_source_info_seg = quadrics_info_source[quadrics_segment_source_index]
        source_scales = np.sqrt(quadrics_source_info_seg["full_scale"]) * quadrics_source_info_seg["full_Is"]

        quadrics_correspondences_selected_current = []
        quadrics_correspondences_selected_similarity_current = []
        for quadrics_segment_target_index in quadrics_info_target.keys():
            quadrics_target_info_seg = quadrics_info_target[quadrics_segment_target_index]
            
            # Scale comparison (sorted to handle unordered plane scales)
            target_scales = np.sqrt(quadrics_target_info_seg["full_scale"]) * quadrics_target_info_seg["full_Is"]
            similarity_volume = np.linalg.norm(np.sort(target_scales) - np.sort(source_scales))

            quadrics_correspondences_selected_current.append(quadrics_segment_target_index)
            quadrics_correspondences_selected_similarity_current.append(similarity_volume)

        quadrics_correspondences_selected_current_index = np.argsort(quadrics_correspondences_selected_similarity_current)[0:topK_correspondence_used]
        quadrics_correspondences_selected_current = [(quadrics_segment_source_index,quadrics_correspondences_selected_current[i]) for i in quadrics_correspondences_selected_current_index]
        quadrics_correspondences_selected_source = quadrics_correspondences_selected_source + quadrics_correspondences_selected_current
    

    quadrics_correspondences_selected_target = []
    for quadrics_segment_target_index in quadrics_info_target.keys():
        quadrics_target_info_seg = quadrics_info_target[quadrics_segment_target_index]
        target_scales = np.sqrt(quadrics_target_info_seg["full_scale"]) * quadrics_target_info_seg["full_Is"]

        quadrics_correspondences_selected_current = []
        quadrics_correspondences_selected_similarity_current = []
        for quadrics_segment_source_index in quadrics_info_source.keys():
            quadrics_source_info_seg = quadrics_info_source[quadrics_segment_source_index]
            
            # Scale comparison (sorted to handle unordered plane scales)
            source_scales = np.sqrt(quadrics_source_info_seg["full_scale"]) * quadrics_source_info_seg["full_Is"]
            similarity_volume = np.linalg.norm(np.sort(target_scales) - np.sort(source_scales))

            quadrics_correspondences_selected_current.append(quadrics_segment_source_index)
            quadrics_correspondences_selected_similarity_current.append(similarity_volume)

        quadrics_correspondences_selected_current_index = np.argsort(quadrics_correspondences_selected_similarity_current)[0:topK_correspondence_used]
        quadrics_correspondences_selected_current = [(quadrics_correspondences_selected_current[i],quadrics_segment_target_index) for i in quadrics_correspondences_selected_current_index]
        quadrics_correspondences_selected_target = quadrics_correspondences_selected_target + quadrics_correspondences_selected_current

    # Mutual intersection of source-to-target and target-to-source correspondences
    quadrics_correspondences_selected = list(set(quadrics_correspondences_selected_source) & set(quadrics_correspondences_selected_target))

    return quadrics_correspondences_selected

def find_knn_cpu(feat0, feat1, knn=1, return_distance=False):
  feat1tree = cKDTree(feat1)
  dists, nn_inds = feat1tree.query(feat0, k=knn)
  if return_distance:
    return nn_inds, dists
  else:
    return nn_inds

def find_correspondences(feats0, feats1, mutual_filter=True):
  nns01 = find_knn_cpu(feats0, feats1, knn=1, return_distance=False)
  corres01_idx0 = np.arange(len(nns01))
  corres01_idx1 = nns01

  if not mutual_filter:
    return corres01_idx0, corres01_idx1

  nns10 = find_knn_cpu(feats1, feats0, knn=1, return_distance=False)
  corres10_idx1 = np.arange(len(nns10))
  corres10_idx0 = nns10

  mutual_filter = (corres10_idx0[corres01_idx1] == corres01_idx0)
  corres_idx0 = corres01_idx0[mutual_filter]
  corres_idx1 = corres01_idx1[mutual_filter]

  return corres_idx0, corres_idx1

def extract_fpfh(pcd, voxel_size=0.5):
  radius_normal = voxel_size * 2
  pcd.estimate_normals(
      o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

  radius_feature = voxel_size * 5
  fpfh = o3d.pipelines.registration.compute_fpfh_feature(
      pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
  return np.array(fpfh.data).T

def stack_dict(nested_dict):
    return {subkey: subvalue
            for parent_key, subdict in nested_dict.items()
            for subkey, subvalue in subdict.items()}

def matching_info_dict_prune(info,DATA_SET_config):
    for semantic_current in list(info.keys()):
        if semantic_current == 'ground':
            continue
        if semantic_current not in DATA_SET_config['semantics_quadrics_reg'].keys():
                try:
                    info.pop(semantic_current)
                except:
                    pass
    return info

def process_point_correspondences_with_info(quadrics_info_source,quadrics_info_target,DATA_SET_config):

    quadrics_info_source_stack = stack_dict(quadrics_info_source)
    quadrics_info_target_stack = stack_dict(quadrics_info_target)

    points_point_source = []
    semantic_point_source = []
    seg_label_point_source = []
    points_point_target = []
    semantic_point_target = []
    seg_label_point_target = []

    points_point_source = []
    semantic_point_source = []
    seg_label_point_source = []
    points_point_target = []
    semantic_point_target = []
    seg_label_point_target = []

    for semantic_current in list(quadrics_info_source.keys()) + list(quadrics_info_target.keys()):

        if semantic_current not in quadrics_info_target.keys():
            try:
                quadrics_info_source.pop(semantic_current)
            except:
                pass
            continue
        if semantic_current not in quadrics_info_source.keys():
            try:
                quadrics_info_target.pop(semantic_current)
            except:
                pass
            continue
        if semantic_current not in DATA_SET_config['semantics_quadrics_reg'].keys():
            try:
                quadrics_info_source.pop(semantic_current)
                quadrics_info_target.pop(semantic_current)
            except:
                pass
            continue

        print(f"    {semantic_current:15s}, Num_seg_source: {len(quadrics_info_source[semantic_current])}, Num_seg_target: {len(quadrics_info_target[semantic_current])}")
    
        for key, item in quadrics_info_source[semantic_current].items():
            quadrics_info_source[semantic_current][key]['quadrics_type'] = DATA_SET_config['semantics_quadrics_reg'][semantic_current]
            points_point_source.append(np.array(item["full_center"]))
            semantic_point_source.append(np.array(item["semantic"]))
            seg_label_point_source.append(key)

        for key, item in quadrics_info_target[semantic_current].items():
            quadrics_info_target[semantic_current][key]['quadrics_type'] = DATA_SET_config['semantics_quadrics_reg'][semantic_current]
            points_point_target.append(np.array(item["full_center"]))
            semantic_point_target.append(np.array(item["semantic"]))
            seg_label_point_target.append(key)

    if len(points_point_source) == 0 or len(points_point_target) == 0:
        return {},quadrics_info_source,quadrics_info_target
        
    points_point_source = np.array(points_point_source)
    semantic_point_source = np.array(semantic_point_source)
    seg_label_point_source = np.array(seg_label_point_source)
    points_point_target = np.array(points_point_target)
    semantic_point_target = np.array(semantic_point_target)
    seg_label_point_target = np.array(seg_label_point_target)

    pcd_point_source = o3d.geometry.PointCloud()
    pcd_point_source.points = o3d.utility.Vector3dVector(points_point_source)
    feats_FPFH_source = extract_fpfh(pcd_point_source,voxel_size=0.5)
    pcd_point_target = o3d.geometry.PointCloud()
    pcd_point_target.points = o3d.utility.Vector3dVector(points_point_target)
    feats_FPFH_target = extract_fpfh(pcd_point_target,voxel_size=0.5)

    index_source_FPFH_preMatch, index_target_FPFH_preMatch = find_correspondences(feats_FPFH_source,feats_FPFH_target, mutual_filter=True)

    semantic_source_FPFH_preMatch = semantic_point_source[index_source_FPFH_preMatch]
    semantic_target_FPFH_preMatch = semantic_point_target[index_target_FPFH_preMatch]
    # Require matching semantics between source and target
    index_fitterSemantic = semantic_source_FPFH_preMatch == semantic_target_FPFH_preMatch
    index_source_FPFH_preMatch_fitterSemantic = index_source_FPFH_preMatch[index_fitterSemantic]
    index_target_FPFH_preMatch_fitterSemantic = index_target_FPFH_preMatch[index_fitterSemantic]

    FPFH_correspondences_with_info_all = {}
    for index_source,index_target in zip(index_source_FPFH_preMatch_fitterSemantic,index_target_FPFH_preMatch_fitterSemantic):
        seg_index_source = seg_label_point_source[index_source]
        seg_index_target = seg_label_point_target[index_target]

        correspondence_FPFH = (seg_index_source,seg_index_target)
        FPFH_correspondences_with_info_all[correspondence_FPFH] = [quadrics_info_source_stack[seg_index_source],quadrics_info_target_stack[seg_index_target]]

    return FPFH_correspondences_with_info_all,quadrics_info_source,quadrics_info_target

def process_point_correspondences_with_info_simplified(quadrics_info_source_stack,quadrics_info_target_stack,DATA_SET_config):

    points_point_source = []
    semantic_point_source = []
    seg_label_point_source = []
    points_point_target = []
    semantic_point_target = []
    seg_label_point_target = []

    points_point_source = []
    semantic_point_source = []
    seg_label_point_source = []
    points_point_target = []
    semantic_point_target = []
    seg_label_point_target = []

    for key in quadrics_info_source_stack:
        points_point_source.append(np.array(quadrics_info_source_stack[key]["full_center"]))
        semantic_point_source.append(np.array(quadrics_info_source_stack[key]["semantic"]))
        seg_label_point_source.append(key)

    for key in quadrics_info_target_stack:
        points_point_target.append(np.array(quadrics_info_target_stack[key]["full_center"]))
        semantic_point_target.append(np.array(quadrics_info_target_stack[key]["semantic"]))
        seg_label_point_target.append(key)

    if len(points_point_source) == 0 or len(points_point_target) == 0:
        return {}

    points_point_source = np.array(points_point_source)
    semantic_point_source = np.array(semantic_point_source)
    seg_label_point_source = np.array(seg_label_point_source)
    points_point_target = np.array(points_point_target)
    semantic_point_target = np.array(semantic_point_target)
    seg_label_point_target = np.array(seg_label_point_target)

    pcd_point_source = o3d.geometry.PointCloud()
    pcd_point_source.points = o3d.utility.Vector3dVector(points_point_source)
    feats_FPFH_source = extract_fpfh(pcd_point_source,voxel_size=0.5)
    pcd_point_target = o3d.geometry.PointCloud()
    pcd_point_target.points = o3d.utility.Vector3dVector(points_point_target)
    feats_FPFH_target = extract_fpfh(pcd_point_target,voxel_size=0.5)

    index_source_FPFH_preMatch, index_target_FPFH_preMatch = find_correspondences(feats_FPFH_source,feats_FPFH_target, mutual_filter=True)

    semantic_source_FPFH_preMatch = semantic_point_source[index_source_FPFH_preMatch]
    semantic_target_FPFH_preMatch = semantic_point_target[index_target_FPFH_preMatch]
    # Require matching semantics
    index_fitterSemantic = semantic_source_FPFH_preMatch == semantic_target_FPFH_preMatch
    index_source_FPFH_preMatch_fitterSemantic = index_source_FPFH_preMatch[index_fitterSemantic]
    index_target_FPFH_preMatch_fitterSemantic = index_target_FPFH_preMatch[index_fitterSemantic]

    FPFH_correspondences_with_info_all = {}
    for index_source,index_target in zip(index_source_FPFH_preMatch_fitterSemantic,index_target_FPFH_preMatch_fitterSemantic):
        seg_index_source = seg_label_point_source[index_source]
        seg_index_target = seg_label_point_target[index_target]

        correspondence_FPFH = (seg_index_source,seg_index_target)
        FPFH_correspondences_with_info_all[correspondence_FPFH] = [quadrics_info_source_stack[seg_index_source],quadrics_info_target_stack[seg_index_target]]

    return FPFH_correspondences_with_info_all

def compute_distances_and_consistency(centers_source, centers_target, eps=1e-6):
    dist_source = pdist(centers_source)
    dist_target = pdist(centers_target)

    # Mask: both source and target distances must exceed eps
    valid_mask = (dist_source >= eps) & (dist_target >= eps)

    # Invalid entries set to inf so they are never selected
    dist_diff = np.full_like(dist_source, np.inf)
    dist_diff[valid_mask] = np.abs(dist_source[valid_mask] - dist_target[valid_mask])

    distance_consistency = squareform(dist_diff)

    return distance_consistency


def distance_consistency_check(quadrics_correspondences_with_info_all, threshold_distance_list):
    correspondence_list = list(quadrics_correspondences_with_info_all.keys())

    if len(correspondence_list) < 2:
        print("correspondence < 2")
        return {}

    centers_source = np.array([
        quadrics_correspondences_with_info_all[correspondence][0]["full_center"]
        for correspondence in correspondence_list
    ])
    centers_target = np.array([
        quadrics_correspondences_with_info_all[correspondence][1]["full_center"]
        for correspondence in correspondence_list
    ])

    distance_consistency = compute_distances_and_consistency(centers_source, centers_target)

    n = len(correspondence_list)
    i_indices, j_indices = np.triu_indices(n, k=1)
    scores = distance_consistency[i_indices, j_indices]

    correspondence_consistency_dict = {threshold: [] for threshold in threshold_distance_list}

    for threshold_distance in threshold_distance_list:
        mask = scores < threshold_distance
        selected_i = i_indices[mask]
        selected_j = j_indices[mask]
        selected_pairs = [
            (correspondence_list[i], correspondence_list[j])
            for i, j in zip(selected_i, selected_j)
        ]
        correspondence_consistency_dict[threshold_distance] = selected_pairs

    line = " | ".join(
        f"{threshold}:{len(correspondence_consistency_dict[threshold])}"
        for threshold in threshold_distance_list
    )
    print(f"  threshold_distance:num -> {line}")

    return correspondence_consistency_dict

def computer_correspondence_inlier(quadrics_correspondences_with_info_all, T, threshold_correspondence_inlier_distance):
    correspondence_list = list(quadrics_correspondences_with_info_all.keys())
    correspondence_num = len(correspondence_list)

    centers_source = np.array([
        quadrics_correspondences_with_info_all[correspondence][0]["full_center"]
        for correspondence in correspondence_list
    ])
    centers_target = np.array([
        quadrics_correspondences_with_info_all[correspondence][1]["full_center"]
        for correspondence in correspondence_list
    ])

    centers_transformed = (T[0:3,0:3]@ centers_source.T).T + T[0:3,3].T

    distances = np.linalg.norm(centers_transformed - centers_target, axis=1)

    inliers_mask = distances < threshold_correspondence_inlier_distance
    inlier_num = np.sum(inliers_mask)

    total_points = len(centers_source)
    inlier_ratio = inlier_num / total_points if total_points > 0 else 0

    return correspondence_num,inlier_ratio, inlier_num