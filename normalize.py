import open3d as o3d
import numpy as np
from pathlib import Path
import pandas as pd
import trimesh

from warnings import filterwarnings
filterwarnings("ignore")

from get_stats import extract_stats
from plots import visualize_normalized_shape
import tempfile

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



def center_mesh(mesh, verbose=False):
    """
    Translate mesh so its barycenter is at the origin.
    """
    vertices = np.asarray(mesh.vertices)
    barycenter = vertices.mean(axis=0)
    mesh.apply_translation(-barycenter)
    
    if verbose:
        print(f"[Center] Barycenter shifted from {barycenter} to origin.")
    
    return mesh


def scale_to_unit(mesh, target_size=1.0, verbose=False):
    """
    Uniformly scale mesh so that its largest bounding box dimension = target_size.
    """
    bbox = mesh.bounds  # returns (min, max) coordinates
    extents = bbox[1] - bbox[0]
    max_dim = np.max(extents)
    
    if max_dim > 0:
        scale_factor = target_size / max_dim
        mesh.apply_scale(scale_factor)
        if verbose:
            print(f"[Scale] Scaled mesh by factor {scale_factor}. Extents before scaling: {extents}")
    
    return mesh


def pca_align(mesh, verbose=False):
    """
    Rotate mesh so its principal axes align with X,Y,Z axes.
    Largest variance → X axis, second → Y, smallest → Z.
    """
    vertices = np.asarray(mesh.vertices)
    covariance = np.cov(vertices, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    
    # Sort eigenvectors by descending eigenvalues
    idx = np.argsort(eigenvalues)[::-1]
    axes = eigenvectors[:, idx]
    
    # Project vertices onto principal axes
    aligned_vertices = vertices @ axes
    mesh.vertices = aligned_vertices
    
    if verbose:
        print(f"[PCA Align] Mesh aligned to principal axes.")
    
    return mesh


def moment_flip(mesh, verbose=False):
    """
    Flip mesh along axes so the 'heavier' side is on positive side.
    Heaviness is estimated from vertex distribution.
    """
    vertices = np.asarray(mesh.vertices)
    signs = np.sign(vertices.sum(axis=0))
    signs[signs == 0] = 1  # avoid zero
    
    mesh.vertices *= signs  # element-wise multiplication
    if verbose:
        print(f"[Flip] Mesh flipped with signs {signs}.")
    
    return mesh


def full_normalization(mesh, verbose=False):
    """
    Apply full 4-step normalization: center, scale, PCA align, flip.
    """
    mesh = center_mesh(mesh, verbose=verbose)
    mesh = scale_to_unit(mesh, target_size=1.0, verbose=verbose)
    mesh = pca_align(mesh, verbose=verbose)
    mesh = moment_flip(mesh, verbose=verbose)
    
    if verbose:
        print("[Normalization] Full normalization complete.")
    
    return mesh


if __name__ == "__main__":
    # mode = 'stats'  # 'normalize' or 'stats'
    # logs = 0  # set to 1 or 2 for more verbose output
    # if mode == 'normalize':
    #     input_database = "resampled_data"  # original/resampled database
    #     output_database = "normalized_data"  # where normalized meshes will be saved
    #     normalize_database(input_database, output_database, logs=logs)
    #     print("Normalization complete.")
    #     all_data = extract_stats(folder_path=output_database, logs=logs)
    #     df = pd.DataFrame(all_data)
    #     df.to_csv("normalized_stats.csv", index=False)
    # elif mode == 'stats':
    #     from get_stats import plot_histograms
    #     df = pd.read_csv("stats/normalized_stats.csv")
    #     plot_histograms(df)

    from read_data import get_random_data_from_directory
    mesh = trimesh.load(get_random_data_from_directory(parent_directory="resampled_data"))
    mesh = full_normalization(mesh, verbose=True)

    # Save normalized mesh to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".obj", delete=False) as tmp:
        mesh.export(tmp.name)
        temp_path = tmp.name

    # Visualize using Open3D
    visualize_normalized_shape(temp_path)
