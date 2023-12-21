import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree
import random

def load_and_normalize_mesh(file_name, number_of_points=10000):
    try:
        mesh = o3d.io.read_triangle_mesh(file_name)
        mesh.compute_vertex_normals() if not mesh.has_vertex_normals() else None
        mesh.compute_triangle_normals() if not mesh.has_triangle_normals() else None

        scale_factor = 1 / np.max(mesh.get_max_bound() - mesh.get_min_bound())
        mesh.scale(scale_factor, center=mesh.get_center())

        # Sample points from the mesh surface using Poisson Disk Sampling
        pcd = mesh.sample_points_poisson_disk(number_of_points)

        return np.asarray(pcd.points)
    except Exception as e:
        print(f"Error loading mesh from {file_name}: {e}")
        return None

def align_meshes(source_points, target_points, threshold=0.02, max_iterations=2000, initial_transformation=None, use_point_to_plane=False):
    """
    Aligns the source points to the target points using the ICP algorithm.

    :param source_points: NumPy array of source points.
    :param target_points: NumPy array of target points.
    :param threshold: Threshold for ICP convergence.
    :param max_iterations: Maximum number of iterations for ICP.
    :param initial_transformation: Initial transformation matrix (4x4 numpy array) or None for identity matrix.
    :param use_point_to_plane: Use point-to-plane ICP (more accurate but requires normals) or point-to-point.
    :return: Aligned source point cloud as a numpy array.
    """
    # Create Open3D point cloud objects from the numpy arrays
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(source_points)
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(target_points)

    # Use identity matrix as the initial transformation if none provided
    if initial_transformation is None:
        initial_transformation = np.eye(4)

    # Choose the ICP method: point-to-plane or point-to-point
    if use_point_to_plane:
        # Compute normals for the point clouds for point-to-plane ICP
        source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=threshold * 2, max_nn=30))
        target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=threshold * 2, max_nn=30))
        registration_method = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    else:
        registration_method = o3d.pipelines.registration.TransformationEstimationPointToPoint()

    # Apply ICP to align the point clouds
    icp_result = o3d.pipelines.registration.registration_icp(
        source, target, threshold, initial_transformation,
        registration_method,
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations)
    )

    # Apply the final transformation to the source points
    transformed_points = np.dot(np.hstack((source_points, np.ones((source_points.shape[0], 1)))), icp_result.transformation.T)[:, :3]
    
    return transformed_points

def chamfer_distance(set1, set2):
    tree1, tree2 = cKDTree(set1), cKDTree(set2)
    distances1, distances2 = tree1.query(set2)[0], tree2.query(set1)[0]
    return np.mean(np.square(distances1)) + np.mean(np.square(distances2))

def rotate_mesh_points(mesh_points, angle, axis):
    """
    Rotate mesh points around a given axis by a specified angle.

    :param mesh_points: NumPy array of mesh points.
    :param angle: Rotation angle in radians.
    :param axis: Axis of rotation, should be a 3D vector.
    :return: Rotated mesh points as a NumPy array.
    """
    # Normalize the axis to make it a unit vector
    axis = axis / np.linalg.norm(axis)

    # Create a rotation matrix
    rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)

    # Apply rotation
    rotated_points = np.dot(mesh_points, rotation_matrix.T)
    return rotated_points

def visualize_meshes(mesh1_points, mesh2_points, title="Mesh Visualization"):
    """
    Visualize two sets of mesh points.

    :param mesh1_points: First set of mesh points.
    :param mesh2_points: Second set of mesh points.
    :param title: Window title.
    """
    # Create point clouds
    mesh1_cloud = o3d.geometry.PointCloud()
    mesh1_cloud.points = o3d.utility.Vector3dVector(mesh1_points)
    mesh1_cloud.paint_uniform_color([1, 0, 0])  # Red color

    mesh2_cloud = o3d.geometry.PointCloud()
    mesh2_cloud.points = o3d.utility.Vector3dVector(mesh2_points)
    mesh2_cloud.paint_uniform_color([0, 1, 0])  # Green color
    
    # Red is prediction, green is ground truth
    title = f"{title} (Red: Prediction, Green: Ground Truth)"

    # Visualize
    o3d.visualization.draw_geometries([mesh1_cloud, mesh2_cloud], window_name=title)

def normalize_mesh_scale(mesh_points):
    """
    Normalize the scale of mesh points to fit within a unit cube centered at the origin.

    :param mesh_points: NumPy array of mesh points.
    :return: Scaled mesh points as a NumPy array.
    """
    # Calculate the bounding box of the points
    min_bound = np.min(mesh_points, axis=0)
    max_bound = np.max(mesh_points, axis=0)

    # Calculate the scale factor
    scale = np.max(max_bound - min_bound)
    if scale == 0:
        return mesh_points

    # Normalize the mesh points
    normalized_points = (mesh_points - min_bound) / scale
    return normalized_points - np.mean(normalized_points, axis=0)



