import time
import numpy as np
from sklearn.decomposition import PCA

EPS = np.finfo(np.float32).eps
def process_data_all(test_points, test_normals=[],d_mean=True, d_scale=True, d_rotation=True,shape='',num=1100,if_normals=False):

    if test_normals == []:
        test_normals = np.zeros(test_points.shape, dtype=float)

    points_raw = test_points
    normals_ = test_normals

    points = test_points
    normals = test_normals


    T = np.diag([1.0, 1.0, 1.0, 1.0])
    T_s_t = np.diag([1.0, 1.0, 1.0, 1.0])


    points_raw_scaled = points_raw

    if d_mean:
        mean = np.mean(points, 0)
        points = points - mean
        points_raw_scaled = points_raw_scaled - mean
        T_d = np.diag([1.0, 1.0, 1.0, 1.0])
        T_d[0:3, 3] = -mean
        T = np.dot(T_d, T)
        T_s_t = np.dot(T_d, T_s_t)

    axis_original = np.array([1.0, 1.0,1.0])
    if d_rotation:
        S, U = pca_numpy(points)
        index_sorted = np.argsort(-S)
        S_sorted = S[index_sorted]
        U_sorted = U[:,index_sorted]

        if "plane" in shape:
            smallest_ev = U_sorted[:,2]
            R = rotation_matrix_a_to_b(smallest_ev, axis_original)
        elif "line" in shape:
            largest_ev = U_sorted[:,0]
            R = rotation_matrix_a_to_b(largest_ev, axis_original)
        else:
            # Gravity direction
            # axis_shape = U_sorted[:,pca_judgment_numpy(S_sorted,shape)]
            axis_shape = np.array([0, 0, 1.0])
            R = rotation_matrix_a_to_b(axis_shape, axis_original)

        points = (R @ points.T).T
        points_raw_scaled = (R @ points_raw_scaled.T).T
        T_r = np.diag([1.0, 1.0, 1.0, 1.0])
        T_r[0:3,0:3] = R
        T = np.dot(T_r, T)

    if d_scale:
        # Max Euclidean distance from each point to the origin
        # std = np.max(np.max(points, 0) - np.min(points, 0))
        std = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
        points = points / (std + EPS)
        points_raw_scaled = points_raw_scaled / (std + EPS)
        T_s = np.diag([1 / (std + EPS), 1 / (std + EPS), 1 / (std + EPS), 1.0])
        T = np.dot(T_s, T)
        T_s_t = np.dot(T_s, T_s_t)

    normals = (np.linalg.inv(T[0:3,0:3]).T @ normals.T).T

    if if_normals:
        # normalization
        normals = np.divide(normals,np.expand_dims(np.linalg.norm(normals,axis=1),axis=1))
    
        for index_normals in range(normals.shape[0]):
            if normals[index_normals][np.where(np.abs(normals[index_normals]) > 1e-8)][0]<0:
                normals[index_normals] = normals[index_normals] * -1

    return [points,0,points_raw_scaled,points_raw,normals,T,T_s_t,axis_original]

def estimate_cylinder_properties_noNoramls_numpy(points,k=6):
    points_mean = points.mean(axis=0)
    points_centered = points - points_mean

    pca = PCA(n_components=3)
    pca.fit(points_centered)

    axis_direction = find_axis_numpy(points,"cylinder")

    projected_points = np.dot(points_centered, axis_direction)
    height = projected_points.max() - projected_points.min()

    distances = np.linalg.norm(points_centered - projected_points[:, np.newaxis] * axis_direction, axis=1)
    radius = np.mean(distances)

    # Crop points exceeding k * radius in height
    max_height = k * radius
    if height > max_height:
        height_limit = max_height
        valid_indices = (projected_points >= projected_points.min() + (height - height_limit) / 2) & \
                        (projected_points <= projected_points.max() - (height - height_limit) / 2)
        points_cropped = points[valid_indices]
    else:
        points_cropped = points

    return axis_direction, height, radius, points_cropped

def estimate_plane_properties_numpy(points,normals,k=2):

    points_mean = points.mean(axis=0)
    points_centered = points - points_mean
    cov_matrix = np.cov(points_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    sorted_indices = np.argsort(eigenvalues)[-2:]
    principal_components = eigenvectors[:, sorted_indices]
    points_projected = points_centered @ principal_components
    max_values = points_projected.max(axis=0)
    min_values = points_projected.min(axis=0)

    lengths = max_values - min_values

    long_dim = 0 if lengths[0] > lengths[1] else 1
    short_dim = 1 - long_dim

    # Crop the long side if aspect ratio exceeds k
    if lengths[long_dim] / lengths[short_dim] > k:
        desired_length = lengths[short_dim] * k
        center_long = (max_values[long_dim] + min_values[long_dim]) / 2
        new_min = center_long - desired_length / 2
        new_max = center_long + desired_length / 2

        clipping_component = principal_components[:, long_dim]
        index_cropped = np.all([
            points_centered @ clipping_component >= new_min,
            points_centered @ clipping_component <= new_max
        ], axis=0)
        points_cropped = points[index_cropped]
        nomrlas_cropped = normals[index_cropped]
    else:
        points_cropped = points
        nomrlas_cropped = normals

    return points_cropped, nomrlas_cropped

def pca_numpy(X):
    X = X - np.mean(X, axis=0)
    S, U = np.linalg.eig(X.T @ X)
    S = S / X.shape[0]
    S = np.maximum(S, 1e-6)
    return S, U

def rotation_matrix_a_to_b(A, B):
    """
    Finds rotation matrix from vector A in 3d to vector B
    in 3d.
    B = R @ A
    """
    A = np.divide(A,np.linalg.norm(A))
    B = np.divide(B,np.linalg.norm(B))

    cos = np.dot(A, B)
    sin = np.linalg.norm(np.cross(B, A))
    u = A
    v = B - np.dot(A, B) * A
    v = v / (np.linalg.norm(v) + EPS)
    w = np.cross(B, A)
    w = w / (np.linalg.norm(w) + EPS)
    F = np.stack([u, v, w], 1)
    G = np.array([[cos, -sin, 0],
                    [sin, cos, 0],
                    [0, 0, 1]])

    # B = R @ A
    try:
        R = F @ G @ np.linalg.inv(F)
    except:
        R = np.eye(3, dtype=np.float32)
    return R

def find_axis_numpy(points, shape="cone"):

    mean = np.mean(points, axis=0)
    points = points - mean

    S, U = pca_numpy(points)
    index_sorted = np.argsort(-S)
    S_sorted = S[index_sorted]
    U_sorted = U[:, index_sorted]

    if "plane" in shape:
        smallest_ev = U_sorted[:, 2]
        axis_shape = smallest_ev
    elif "cone" in shape:
        axis_shape = U_sorted[:, pca_judgment_numpy(S_sorted, shape)]
    else:
        axis_shape = U_sorted[:, pca_judgment_numpy(S_sorted, shape)]

    return axis_shape

def pca_judgment_numpy(S,shape):

    x = S[1] / S[0]
    y = S[2] / S[0]

    margin = 0.1
    # Find the principal axis whose eigenvalue differs from the other two
    if abs(x-1)<margin:
        shape_axis_index = 2
    elif abs(y-1)<margin:
        shape_axis_index = 1
    elif abs(x-y)<margin:
        shape_axis_index = 0
    else:
        if "cone" in shape:
            # cone
            shape_axis_index = 2
        else:
            # cylinder and sphere
            shape_axis_index = 0
    return shape_axis_index

def quadrics_scale_identification(Q, prim):
    """
    Identifies and scales the quadrics based on the given primitive type `prim`.
    Q: 4x4 NumPy array
    prim: string, one of ["plane", "cone", "elliptic_cone", "line", "cylinder", "elliptic_cylinder", ...]
    """

    # Compute eigenvalues of E = Q[0:3,0:3]
    E = Q[0:3, 0:3]
    eigenvalue_E = np.linalg.eigvals(E)

    if prim in ["plane"]:
        scale_identification = 1.0 / np.max(np.abs(eigenvalue_E))
    elif prim in ["cone", "elliptic_cone"]:
        # We expect exactly one negative eigenvalue
        neg_vals = eigenvalue_E[eigenvalue_E < 0]
        if neg_vals.size == 1:
            factor = -neg_vals[0]
            scale_identification = 1.0 / factor
        else:
            raise ValueError("cone eigenvalue_E mistake: expected exactly one negative eigenvalue")
    elif prim in ["line"]:
        scale_identification = 1.0 / np.max(np.abs(eigenvalue_E))
    else:
        # For cylinder or ellipsoid etc.
        if prim in ["cylinder", "elliptic_cylinder"]:
            # remove the eigenvalue with smallest absolute value
            min_abs_index = np.argmin(np.abs(eigenvalue_E))
            eigenvalue_E = np.concatenate((eigenvalue_E[:min_abs_index], eigenvalue_E[min_abs_index+1:]))

        # Compute scale_E as product of eigenvalues of E
        scale_E = np.prod(eigenvalue_E)

        # Compute eigenvalues of Q
        eigenvalue_Q = np.linalg.eigvals(Q)
        if prim in ["cylinder", "elliptic_cylinder"]:
            # also remove the eigenvalue with smallest abs value in Q
            min_abs_index = np.argmin(np.abs(eigenvalue_Q))
            eigenvalue_Q = np.concatenate((eigenvalue_Q[:min_abs_index], eigenvalue_Q[min_abs_index+1:]))

        scale_Q = np.prod(eigenvalue_Q)

        # scale_identification = abs(scale_E / scale_Q)
        scale_identification = np.abs(scale_E / (scale_Q + EPS))

    Q = scale_identification * Q
    return Q, np.squeeze(1.0 / (scale_identification))

def quadrics_judgment(eigenvalue):
    """
    Judges the scale and rotation degeneracies based on eigenvalues.
    eigenvalue: 1D NumPy array of length 3, sorted in descending order as per original code.
    Returns: Is, Ir, It as NumPy arrays.
    """

    # margin_0 for scale and translation, margin_1 for rotation
    margin_0 = 1e-3
    margin_1 = 1e-3

    # eigenvalue[0] is the largest (since sorted descending)
    # Compute ratios
    x = eigenvalue[1] / eigenvalue[0] if eigenvalue[0] != 0 else 0.0
    y = eigenvalue[2] / eigenvalue[0] if eigenvalue[0] != 0 else 0.0

    # It - translation degeneration
    # It = 1 if |eigenvalue| > margin_0, else 0
    It = (np.abs(eigenvalue) > margin_0).astype(float)

    # Is - scale degeneration (initially same as It)
    Is = It.copy()

    # Special cases
    # Plane: if |x|<margin_0 and |y|<margin_0 -> Is = [0,0,0]
    if (abs(x) < margin_0) and (abs(y) < margin_0):
        Is = np.array([0.0,0.0,0.0], dtype=float)

    # Cylinder: if x>margin_0 and |y|<margin_0 -> Is = [1,1,0]
    if (x > margin_0) and (abs(y) < margin_0):
        Is = np.array([1.0,1.0,0.0], dtype=float)

    # Cone: if x>margin_0 and y<-margin_0 -> Is = [1,1,0]
    if (x > margin_0) and (y < -margin_0):
        Is = np.array([1.0,1.0,0.0], dtype=float)

    # Ir - rotation degeneration
    Ir = np.ones(3, dtype=float)

    # Check rotation degeneracies
    # If |x-1|<margin_1: Ir[1]=0 and Ir[0]=0
    if abs(x - 1) < margin_1:
        Ir[0] = 0
        Ir[1] = 0

    # If |x - y|<margin_1: Ir[1]=0 and Ir[2]=0
    if abs(x - y) < margin_1:
        Ir[1] = 0
        Ir[2] = 0

    return Is, Ir, It

def null_space(A, tol=1e-3):
    """
    Compute the null space of matrix A using SVD.
    A: (m,n) array
    tol: tolerance for considering singular values as zero.

    Returns:
        null_space: A (n,k) array whose columns form a basis for the null space of A.
    """
    U, S, Vh = np.linalg.svd(A)
    # Identify singular values that are effectively zero
    null_mask = (S <= tol)
    null_space = Vh[null_mask].T
    return null_space

def solveLS(A, b):
    """
    Solve the linear system A x = b for x.

    A: coefficient matrix of size (m,n)
    b: constant terms vector of size (m,)
    
    Returns:
        S_H: A basis for the homogeneous solution space of A x = 0.
        S_P: A particular solution of A x = b.

    Cases:
    1) If rank(A) < rank([A|b]), no solution exists:
       S_H = empty, S_P = empty
    2) If rank(A) = rank([A|b]) = n (the number of unknowns):
       Unique solution:
       S_P: the unique solution vector
       S_H: empty
    3) If rank(A) = rank([A|b]) < n:
       Infinite solutions:
       S_P: one particular solution
       S_H: basis of the null space of A (homogeneous solutions)
    """
    if A.shape[0] != len(b):
        raise ValueError('Input dimension mismatch: A and b must have compatible shapes.')

    # Form augmented matrix [A|b]
    B = np.column_stack((A, b))
    rank_A = np.linalg.matrix_rank(A, tol=1e-3)
    rank_B = np.linalg.matrix_rank(B, tol=1e-3)

    if rank_A != rank_B:
        # No solution
        S_H = np.array([])
        S_P = np.array([])
    elif rank_B == A.shape[1]:
        # Unique solution
        S_P = np.linalg.lstsq(A, b, rcond=None)[0]
        S_H = np.array([])
    else:
        # Infinite solutions
        S_H = null_space(A)
        S_P = np.linalg.lstsq(A, b, rcond=None)[0]

    return S_H, S_P

def q_Q(q):
    Q = np.array([[q[0], q[3], q[4], q[6]],
                [q[3], q[1], q[5], q[7]],
                [q[4], q[5], q[2], q[8]],
                [q[6], q[7], q[8], q[9]]])
    return Q

def Q_q(Q):
    Q = (Q.T + Q)/2
    q = np.array([Q[0, 0], Q[1, 1], Q[2, 2], Q[0, 1], Q[0, 2], Q[1, 2], Q[0, 3], Q[1, 3], Q[2, 3],Q[3, 3]])
    return q

def rescale_input_outputs_quadrics_test_anything(T, T_s_t, quadrics, quadrics_c, center, center_statistics, scale, rotation, Is, Ir):
    """
    Rescales input and output quadrics using given transformation matrices.
    """

    # Stack T and T_s_t to form arrays
    T = np.stack(T, axis=0).astype(np.float32)
    T_s_t = np.stack(T_s_t, axis=0).astype(np.float32)

    # Convert from shape (batch_size, ...) to a single matrix if needed
    # Assuming input was a single transformation: if multiple, handle indexing as required
    # If T and T_s_t were single transformations (not lists), no need to stack.
    # Adjust code accordingly if you really need batch handling.

    # For simplicity, assume single transformation:
    if T.shape[0] == 1:
        T = T[0]
    if T_s_t.shape[0] == 1:
        T_s_t = T_s_t[0]

    # Convert quadrics to Q matrices
    quadrics = quadrics.squeeze()
    Q_quadrics = q_Q(quadrics)  # q_Q should return a 4x4 matrix

    # Convert to float64 for better precision
    Q_quadrics = Q_quadrics.astype(np.float64)
    T = T.astype(np.float64)

    # Apply transformation: Q' = T^T Q T
    Q_quadrics = T.T @ Q_quadrics @ T
    Q_quadrics = (Q_quadrics.T + Q_quadrics) / 2  # symmetrize

    # Convert back to float32 if desired
    T = T.astype(np.float32)

    quadrics_d_T = Q_q(Q_quadrics)  # Convert back to quadric vector

    # Convert center to homogeneous coordinates
    center = center.astype(np.float32)
    center_h = np.concatenate([center, np.array([1.0], dtype=np.float32)])
    T_s_t_inv = np.linalg.inv(T_s_t)
    T_inv = np.linalg.inv(T.astype(np.float64))  # T converted to float64 for inverse
    center_c_d_T = (T_s_t_inv @ center_h)[0:3]
    center_d_T = (T_inv @ center_h.astype(np.float64))[0:3]

    center_statistics = center_statistics.astype(np.float32)
    center_statistics_h = np.concatenate([center_statistics, np.array([1.0], dtype=np.float32)])
    center_statistics_c_d_T = (T_s_t_inv @ center_statistics_h)[0:3]
    center_statistics_d_T = (T_inv @ center_statistics_h.astype(np.float64))[0:3]

    # scale_d_s = scale * diag(T_s_t^-1)[0:3] * Is
    # diag of T_s_t_inv:
    T_s_t = T_s_t.astype(np.float64)
    T_s_t_inv = np.linalg.inv(T_s_t)
    diag_T_s_t_inv = np.diag(T_s_t_inv)[0:3]
    scale_d_s = scale * diag_T_s_t_inv * Is

    rotation = rotation.astype(np.float32)
    rotation_d_r = (T_inv[0:3,0:3] @ rotation)
    # Normalize columns
    col_norms = np.linalg.norm(rotation_d_r, axis=0) + 1e-8
    rotation_d_r = (rotation_d_r / col_norms) * Ir

    quadrics_c = quadrics_c.squeeze()
    Q_quadrics_c = q_Q(quadrics_c).astype(np.float64)

    # Apply T_s_t transformation: Q' = T_s_t^T Q T_s_t
    Q_quadrics_c = (T_s_t.T @ Q_quadrics_c @ T_s_t)
    Q_quadrics_c = (Q_quadrics_c.T + Q_quadrics_c) / 2
    quadrics_c_d_T = Q_q(Q_quadrics_c)

    return (T, T_s_t, quadrics_d_T, quadrics_c_d_T, 
            center_d_T, center_c_d_T, 
            center_statistics_d_T, center_statistics_c_d_T, 
            scale_d_s, rotation_d_r)

def volume_shape(scales,prim):
    if prim in ["plane"]:
        volume = scales[0] * scales[1]
    elif prim in ["cone","elliptic_cone"]:
        volume = scales[0] * scales[1] * scales[2] * np.pi / 3
    elif prim in ["cylinder","elliptic_cylinder"]:
        volume = scales[0] * scales[1] * scales[2] * np.pi
    elif prim in ["sphere","ellipsoid"]:
        volume = scales[0] * scales[1] * scales[2]
    elif prim in ["line"]:
        volume = scales[2]
    return volume

def volume_shape_normlize(scales,prim):
    if prim in ["plane"]:
        volume = scales[0] * scales[1]
    elif prim in ["cone","elliptic_cone"]:
        volume = scales[0] * scales[1] * scales[2] * np.pi / 3
    elif prim in ["cylinder","elliptic_cylinder"]:
        volume = scales[0] * scales[1] * scales[2] * np.pi
    elif prim in ["sphere","ellipsoid"]:
        # scales = scales / np.linalg.norm(scales)
        scales = scales / np.sum(scales) # shape structure difference
        volume = scales[0] * scales[1] * scales[2]
    elif prim in ["line"]:
        volume = scales[2]
    return volume