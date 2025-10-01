import open3d as o3d
import numpy as np
from pathlib import Path
import pandas as pd

from warnings import filterwarnings
filterwarnings("ignore")

from get_stats import extract_stats

# Step 2.5 -Translate mesh to origin
def translate_mesh(mesh, logs=0):
    """Translate mesh so its barycenter coincides with the origin."""
    vertices = np.asarray(mesh.vertices)
    barycenter = vertices.mean(axis=0)
    mesh.translate(-barycenter)
    if logs:
        print(f"Translated mesh to origin. Barycenter: {barycenter}")
    return mesh

# Step 2.5 - Scale mesh to fit unit cube
def scale_mesh(mesh, target_size=1.0, logs=0):
    """Scale mesh uniformly so it fits into a unit cube (or target_size cube)."""
    bbox = mesh.get_axis_aligned_bounding_box()
    extent = bbox.get_extent()  # x, y, z lengths
    max_dim = max(extent)
    if max_dim > 0:
        scale_factor = target_size / max_dim
        mesh.scale(scale_factor, center=(0,0,0))
        if logs:
            print(f"Scaled mesh by factor {scale_factor}. Extent: {extent}")
    return mesh

# Step 2.5 - Save normalized mesh
def save_normalized_mesh(mesh, original_path, output_dir, logs=0):
    """Save normalized mesh to output directory, keeping folder structure."""
    original_path = Path(original_path)
    output_path = Path(output_dir) / original_path.parent.name / original_path.name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(output_path), mesh)
    if logs:
        print(f"Saved normalized mesh to {output_path}")
    return output_path

# Step 2.5 - Normalize all meshes in database
def normalize_database(input_dir, output_dir, logs=0):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Collect all mesh files first
    mesh_files = []
    for class_dir in [d for d in input_dir.iterdir() if d.is_dir()]:
        for mesh_file in class_dir.glob("*.obj"):
            mesh_files.append((class_dir, mesh_file))

    # Progress bar
    try:
        from tqdm import tqdm
        iterator = tqdm(mesh_files, desc="Normalizing meshes")
    except ImportError:
        iterator = mesh_files

    for class_dir, mesh_file in iterator:
        mesh = o3d.io.read_triangle_mesh(str(mesh_file))
        mesh.compute_vertex_normals()

        mesh = translate_mesh(mesh, logs=logs)
        mesh = scale_mesh(mesh, logs=logs)
        save_normalized_mesh(mesh, mesh_file, output_dir, logs=logs)
        if logs:
            print(f"Normalized: {mesh_file}")


if __name__ == "__main__":
    mode = 'stats'  # 'normalize' or 'stats'
    logs = 0  # set to 1 or 2 for more verbose output
    if mode == 'normalize':
        input_database = "resampled_data"  # original/resampled database
        output_database = "normalized_data"  # where normalized meshes will be saved
        normalize_database(input_database, output_database, logs=logs)
        print("Normalization complete.")
        all_data = extract_stats(folder_path=output_database, logs=logs)
        df = pd.DataFrame(all_data)
        df.to_csv("normalized_stats.csv", index=False)
    elif mode == 'stats':
        from get_stats import plot_histograms
        df = pd.read_csv("stats/normalized_stats.csv")
        plot_histograms(df)