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
    """
    Translate mesh so barycenter coincides with origin.
    Mathematical formula: barycenter = (1/N) * Σvi
    Translation: vi' = vi - barycenter
    """
    centroid = mesh.get_center()
    mesh.translate(-centroid)
    if logs:
        print(f"Translated by: {-centroid}")
    return mesh

# Step 2.5 - Scale mesh to fit unit cube
def scale_mesh(mesh, target_size=1.0, logs=0):
    """
    Scale mesh uniformly to fit in unit cube.
    Mathematical formula: s = target_size / max(bbox_dimensions)
    Scaling: vi' = s * vi (applied after translation to origin)
    """
    bbox = mesh.get_axis_aligned_bounding_box()
    bbox_extent = bbox.get_extent()
    max_extent = max(bbox_extent)
    scale_factor = target_size / max_extent
    mesh.scale(scale_factor, center=(0, 0, 0))
    if logs:
        print(f"Scaled by factor: {scale_factor:.6f}")
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

    # Add verification at the end
    if logs:
        print("\nVerifying normalized meshes...")
        verification_passed = verify_normalized_database(output_dir, logs=logs)
        if verification_passed:
            print("✅ All meshes properly normalized!")
        else:
            print("⚠️ Some meshes may not be properly normalized")


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


# Add this verification function
def verify_normalization(mesh, logs=0):
    """Verify that mesh is properly normalized"""
    centroid = mesh.get_center()
    bbox = mesh.get_axis_aligned_bounding_box()
    bbox_extent = bbox.get_extent()
    max_extent = max(bbox_extent)
    
    if logs:
        print(f"Centroid: {centroid}")
        print(f"Max extent: {max_extent:.6f}")
    
    # Check if properly normalized (with small tolerance)
    centroid_ok = np.allclose(centroid, [0, 0, 0], atol=1e-6)
    scale_ok = abs(max_extent - 1.0) < 1e-6
    
    return centroid_ok and scale_ok


def verify_normalized_database(normalized_dir, logs=0):
    """Check that all normalized meshes are properly centered and scaled"""
    from get_stats import extract_stats
    
    stats = extract_stats(normalized_dir, logs=False)
    issues = []
    
    for stat in stats:
        from get_stats import read_data
        file_path = stat['file']
        mesh = read_data(file_path)
        
        if not verify_normalization(mesh, logs=0):
            issues.append(file_path)
    
    if logs:
        print(f"Normalization verification: {len(stats) - len(issues)}/{len(stats)} passed")
        if issues:
            print("Issues found in:", issues[:5])  # Show first 5
    
    return len(issues) == 0


def normalize_filtered_database(input_dir, output_dir, filter_csv, logs=0):
    """
    Normalize only meshes that passed the resampling threshold.
    
    Args:
        input_dir: Directory containing resampled meshes  
        output_dir: Directory to save normalized meshes
        filter_csv: Path to CSV file with resampling flags
        logs: Logging level
    """
    import pandas as pd
    
    # Read the CSV with flags
    df = pd.read_csv(filter_csv)
    
    # Filter to only files within threshold
    valid_files = df[df['within_threshold'] == True]
    
    if logs:
        total_files = len(df)
        filtered_files = len(valid_files)
        skipped_files = total_files - filtered_files
        print(f"Resampling filter results:")
        print(f"  Total files: {total_files}")
        print(f"  Within threshold (5K-10K vertices): {filtered_files}")
        print(f"  Skipped (outside threshold): {skipped_files}")
        print(f"  Success rate: {filtered_files/total_files*100:.1f}%")

    # Progress bar
    try:
        from tqdm import tqdm
        iterator = tqdm(valid_files.iterrows(), total=len(valid_files), desc="Normalizing valid meshes")
    except ImportError:
        iterator = valid_files.iterrows()

    normalized_count = 0
    skipped_count = 0
    
    for _, row in iterator:
        mesh_file = Path(row['file'])
        
        # Check if file exists
        if not mesh_file.exists():
            if logs >= 2:
                print(f"File not found: {mesh_file}")
            skipped_count += 1
            continue
            
        try:
            mesh = o3d.io.read_triangle_mesh(str(mesh_file))
            mesh.compute_vertex_normals()

            mesh = translate_mesh(mesh, logs=logs if logs >= 2 else 0)
            mesh = scale_mesh(mesh, logs=logs if logs >= 2 else 0)
            save_normalized_mesh(mesh, mesh_file, output_dir, logs=logs if logs >= 2 else 0)
            
            normalized_count += 1
            if logs >= 2:
                print(f"Normalized: {mesh_file}")
                
        except Exception as e:
            if logs:
                print(f"Error processing {mesh_file}: {e}")
            skipped_count += 1

    if logs:
        print(f"\nNormalization complete:")
        print(f"  Successfully normalized: {normalized_count}")
        print(f"  Skipped/failed: {skipped_count}")


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
