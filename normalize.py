import open3d as o3d
import numpy as np
from pathlib import Path
import pandas as pd

from warnings import filterwarnings
filterwarnings("ignore")

from get_stats import extract_stats

# 1. Translate mesh to origin
def translate_mesh(mesh):
    """
    Translate mesh so its barycenter coincides with the origin.
    """
    vertices = np.asarray(mesh.vertices)
    barycenter = vertices.mean(axis=0)
    mesh.translate(-barycenter)
    return mesh

# 2. Scale mesh to fit unit cube
def scale_mesh(mesh, target_size=1.0):
    """
    Scale mesh uniformly so it fits into a unit cube (or target_size cube).
    """
    bbox = mesh.get_axis_aligned_bounding_box()
    extent = bbox.get_extent()  # x, y, z lengths
    max_dim = max(extent)
    if max_dim > 0:
        scale_factor = target_size / max_dim
        mesh.scale(scale_factor, center=(0,0,0))
    return mesh

# 3. Save normalized mesh
def save_normalized_mesh(mesh, original_path, output_dir):
    """
    Save normalized mesh to output directory, keeping folder structure.
    """
    original_path = Path(original_path)
    output_path = Path(output_dir) / original_path.parent.name / original_path.name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(output_path), mesh)
    return output_path

# 4. Normalize all meshes in database
def normalize_database(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    for class_dir in [d for d in input_dir.iterdir() if d.is_dir()]:
        for mesh_file in class_dir.glob("*.obj"):
            mesh = o3d.io.read_triangle_mesh(str(mesh_file))
            mesh.compute_vertex_normals()
            
            mesh = translate_mesh(mesh)
            mesh = scale_mesh(mesh)
            save_normalized_mesh(mesh, mesh_file, output_dir)
            print(f"Normalized: {mesh_file}")
        # break



if __name__ == "__main__":
    input_database = "resampled_data"  # original/resampled database
    output_database = "normalized_data"  # where normalized meshes will be saved
    normalize_database(input_database, output_database)
    print("Normalization complete.")

    all_data = extract_stats(folder_path=output_database, logs=False)
    df = pd.DataFrame(all_data)
    df.to_csv("normalized_stats.csv", index=False)