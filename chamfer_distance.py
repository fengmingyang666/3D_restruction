import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree
import random
from chamfer_distance_utils import *
def main():
    filename = 'good_phone'
    
    file1, file2 = f'./data/prediction/{filename}_pre.ply', f'./data/ground_truth/{filename}_gt.ply'
    mesh1,mesh2 = load_mesh(file1),load_mesh(file2)
    print(f"Loaded {file1} and {file2}")
    if mesh1 is None or mesh2 is None:
        print("Error loading mesh files.")
        return
    
    # Normalize meshes
    mesh1, mesh2 = normalize_mesh_scale(mesh1), normalize_mesh_scale(mesh2)


    # Visualize original meshes
    visualize_meshes(mesh1, mesh2, "Original Meshes")
    # sample points from meshes
    mesh1_points, mesh2_points = sample_points_from_mesh(mesh1), sample_points_from_mesh(mesh2)
    # get points from point cloud objects
    mesh1_points, mesh2_points = np.asarray(mesh1_points.points), np.asarray(mesh2_points.points)
    distance_original = chamfer_distance(mesh1_points, mesh2_points)
    print(f"Chamfer Distance before random rotation: {distance_original}")


    # Align meshes
    transformation_matrix = align_meshes( mesh1, mesh2, number_of_points=5000)
    # Apply the transformation to mesh1
    aligned_mesh1 = transform_mesh(mesh1, transformation_matrix)
    # output the transformation matrix as well
    print(f"Transformation Matrix:\n{transformation_matrix}")
    # Visualize after alignment
    visualize_meshes(aligned_mesh1, mesh2, "Meshes After Alignment")

    # sample points from meshes
    aligned_mesh1_points, mesh2_points = sample_points_from_mesh(aligned_mesh1), sample_points_from_mesh(mesh2)
    # get points from point cloud objects
    aligned_mesh1_points, mesh2_points = np.asarray(aligned_mesh1_points.points), np.asarray(mesh2_points.points)
    # Chamfer Distance after alignment
    distance_post_align = chamfer_distance( aligned_mesh1_points, mesh2_points)
    print(f"Chamfer Distance after alignment: {distance_post_align}")

if __name__ == "__main__":
    main()
