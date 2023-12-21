import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree
import random
from chamfer_distance_utils import load_and_normalize_mesh, align_meshes, rotate_mesh_points, visualize_meshes, normalize_mesh_scale, chamfer_distance

def main():
    filename = 'polaroid_99'
    
    file1, file2 = f'./data/prediction/{filename}.ply', f'./data/ground_truth/{filename}.ply'
    mesh1_points, mesh2_points = load_and_normalize_mesh(file1,number_of_points=5000), load_and_normalize_mesh(file2,number_of_points=5000)
    print(f"Loaded {file1} and {file2}")
    if mesh1_points is None or mesh2_points is None:
        print("Error loading mesh files.")
        return

    # Normalize mesh scales
    mesh1_points, mesh2_points = normalize_mesh_scale(mesh1_points), normalize_mesh_scale(mesh2_points)

    # Visualize original meshes
    visualize_meshes(mesh1_points, mesh2_points, "Original Meshes")
    distance_original = chamfer_distance(mesh1_points, mesh2_points)
    print(f"Chamfer Distance before random rotation: {distance_original}")
    
    
    # Apply random rotation
    random_angle, random_axis = random.uniform(0, 2 * np.pi), np.random.rand(3)
    rotated_mesh1_points = rotate_mesh_points(mesh1_points, random_angle, random_axis)

    # Visualize after rotation
    visualize_meshes(rotated_mesh1_points, mesh2_points, "Meshes After Random Rotation")

    # Chamfer Distance before alignment
    distance_pre_align = chamfer_distance(rotated_mesh1_points, mesh2_points)
    print(f"Chamfer Distance after random rotation: {distance_pre_align}")

    # Align meshes
    aligned_mesh1_points, transformation_matrix = align_meshes(rotated_mesh1_points, mesh2_points)
    # output the transformation matrix as well
    print(f"Transformation Matrix:\n{transformation_matrix}")
    # Visualize after alignment
    visualize_meshes(aligned_mesh1_points, mesh2_points, "Meshes After Alignment")

    # Chamfer Distance after alignment
    distance_post_align = chamfer_distance(aligned_mesh1_points, mesh2_points)
    print(f"Chamfer Distance after alignment: {distance_post_align}")

if __name__ == "__main__":
    main()
