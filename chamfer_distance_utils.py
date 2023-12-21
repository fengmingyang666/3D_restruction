import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree
import scipy

def load_mesh(file_name):
    try:
        mesh = o3d.io.read_triangle_mesh(file_name)
        mesh.compute_vertex_normals() if not mesh.has_vertex_normals() else None
        mesh.compute_triangle_normals() if not mesh.has_triangle_normals() else None
        return mesh
    except Exception as e:
        print(f"Error loading mesh from {file_name}: {e}")
        return None

def kabsch_umeyama(P, Q):
    """
    The Kabsch-Umeyama algorithm to find the optimal rotation and translation
    to align two sets of points P and Q.
    
    BUG It is not suitable for our case because our points are not correspondences.

    Parameters:
    P (numpy.ndarray): First set of points (Nx3).
    Q (numpy.ndarray): Second set of points (Nx3), to which P is to be aligned.

    Returns:
    numpy.ndarray: Rotated and translated version of P.
    numpy.ndarray: Rotation matrix.
    numpy.ndarray: Translation vector.
    """

    # Step 1: Calculate the centroids of P and Q
    centroid_P = np.mean(P, axis=0)
    centroid_Q = np.mean(Q, axis=0)

    # Step 2: Translate points to the centroid
    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q

    # Step 3: Compute the cross-covariance matrix
    H = P_centered.T @ Q_centered

    # Step 4: Compute the optimal rotation matrix using SVD
    U, _, Vt = scipy.linalg.svd(H)
    R = Vt.T @ U.T

    # Ensure a right-handed coordinate system
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Step 5: Compute the optimal translation
    t = centroid_Q - R @ centroid_P

    # Step 6: Apply the rotation and translation to P
    P_aligned = P @ R + t

    return P_aligned, R, t  
    
def transfer_normals_from_mesh_to_point_cloud(mesh, point_cloud):
    # Ensure the mesh has triangle normals
    # BUG
    if not mesh.has_triangle_normals():
        mesh.compute_triangle_normals()

    # Create a KDTree for the mesh triangles (using triangle centroids)
    triangle_centroids = np.mean(np.asarray(mesh.vertices)[np.asarray(mesh.triangles), :], axis=1)
    mesh_tree = o3d.geometry.KDTreeFlann(triangle_centroids)

    # Preallocate array for normals
    normals = np.zeros((len(point_cloud.points), 3))

    # Find the closest triangle in the mesh for each point in the point cloud
    for i, point in enumerate(point_cloud.points):
        _, idx, _ = mesh_tree.search_knn_vector_3d(point, 1)
        normals[i] = mesh.triangle_normals[idx[0]]

    # Assign computed normals to the point cloud
    point_cloud.normals = o3d.utility.Vector3dVector(normals)


def align_meshes(source_mesh, target_mesh, max_correspondence_distance=0.5, initial_transformation=np.eye(4), use_point_to_plane=True, relative_fitness=0.000001, relative_rmse=0.000001, max_iteration=3000, number_of_points=1000):
    """
    Aligns the source mesh to the target mesh using the ICP algorithm.

    :param source_mesh: Open3D mesh object of source.
    :param target_mesh: Open3D mesh object of target.
    :param max_correspondence_distance: Maximum correspondence points-pair distance.
    :param initial_transformation: Initial transformation matrix (4x4 numpy array).
    :param use_point_to_plane: Use point-to-plane ICP (more accurate but requires normals) or point-to-point.
    :return: Aligned source point cloud as a numpy array and the final transformation matrix.
    """
    # Sample points from the meshes
    source = source_mesh.sample_points_poisson_disk(number_of_points)
    target = target_mesh.sample_points_poisson_disk(number_of_points)

    # # Transfer normals from the meshes to the sampled point clouds
    # transfer_normals_from_mesh_to_point_cloud(source_mesh, source)
    # transfer_normals_from_mesh_to_point_cloud(target_mesh, target)
    
    # Estimate normals for the sampled point clouds
    source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # Choose the ICP method: point-to-plane or point-to-point
    registration_method = o3d.pipelines.registration.TransformationEstimationPointToPlane() if use_point_to_plane else o3d.pipelines.registration.TransformationEstimationPointToPoint()

    # Set the ICP convergence criteria
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=relative_fitness, relative_rmse=relative_rmse, max_iteration=max_iteration)

    # Apply ICP to align the point clouds
    icp_result = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance, initial_transformation,
        registration_method, criteria
    )

    return icp_result.transformation

def chamfer_distance(set1, set2):
    tree1, tree2 = cKDTree(set1), cKDTree(set2)
    distances1, distances2 = tree1.query(set2)[0], tree2.query(set1)[0]
    return np.mean(np.square(distances1)) + np.mean(np.square(distances2))


def visualize_meshes(mesh1, mesh2, title="Meshes"):
    """
    Visualize two Open3D meshes side by side.

    :param mesh1: Open3D mesh object.
    :param mesh2: Open3D mesh object.
    :param title: Title of the visualization window.
    """
    # Visualize the meshes
    o3d.visualization.draw_geometries([mesh1, mesh2], window_name=title)
    
def transform_mesh(mesh,transformation_matirx):
    mesh.transform(transformation_matirx)
    return mesh

def sample_points_from_mesh(mesh, number_of_points=1000, sampling_method='poisson_disk'):
    """
    Sample points from an Open3D mesh.

    :param mesh: Open3D mesh object.
    :param number_of_points: Number of points to sample.
    :param sampling_method: Sampling method to use. Options: 'uniform' or 'poisson_disk'.
    :return: Open3D point cloud object.
    """
    if sampling_method == 'uniform':
        return mesh.sample_points_uniformly(number_of_points)
    elif sampling_method == 'poisson_disk':
        return mesh.sample_points_poisson_disk(number_of_points)
    else:
        raise ValueError(f"Sampling method {sampling_method} not supported.")

def normalize_mesh_scale(mesh):
    """
    Normalize the scale of an Open3D mesh to fit within a unit cube centered at the origin.

    :param mesh: Open3D mesh object.
    :return: Scaled Open3D mesh object.
    """
    # Calculate the bounding box of the mesh
    min_bound = np.asarray(mesh.get_min_bound())
    max_bound = np.asarray(mesh.get_max_bound())

    # Calculate the scale factor
    scale = np.max(max_bound - min_bound)
    if scale == 0:
        return mesh

    # Normalize the mesh scale and center it at the origin
    mesh.scale(1 / scale, center=min_bound)
    mesh.translate(-mesh.get_center())

    return mesh




