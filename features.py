import numpy as np
import trimesh
import random
import math
from collections import defaultdict

def compute_surface_area(mesh):
    return mesh.area

def compute_volume(mesh):
    return mesh.volume

def compute_aabb_volume(mesh):
    return mesh.bounding_box_oriented.volume

def compute_compactness(mesh):
    # compactness = (surface area)^3 / (volume)^2
    if mesh.volume == 0:
        return 0
    return (mesh.area ** 3) / (mesh.volume ** 2)

def compute_diameter(mesh, sample_limit=200):
    """
    Estimate the diameter as the maximum distance between points on convex hull.
    Use random sampling if too many vertices.
    """
    hull_vertices = list(mesh.convex_hull.vertices)
    if len(hull_vertices) > sample_limit:
        hull_vertices = random.sample(hull_vertices, sample_limit)
    
    max_dist = 0
    for i in range(len(hull_vertices)):
        for j in range(i+1, len(hull_vertices)):
            d = np.linalg.norm(hull_vertices[i] - hull_vertices[j])
            if d > max_dist:
                max_dist = d
    return max_dist

def compute_convexity(mesh):
    # convexity = mesh volume / convex hull volume
    hull_vol = mesh.convex_hull.volume
    return mesh.volume / hull_vol if hull_vol != 0 else 0

def compute_eccentricity(mesh):
    # ratio of largest to smallest eigenvalues of covariance matrix
    vertices = np.asarray(mesh.vertices)
    covariance = np.cov(vertices, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(covariance)
    # avoid division by zero
    if np.min(eigenvalues) == 0:
        return 0
    return np.max(eigenvalues) / np.min(eigenvalues)

def extract_scalar_features(mesh):
    """
    Returns a dictionary of all scalar features for a mesh.
    """
    features = {}
    features['area'] = np.round(compute_surface_area(mesh), 3)
    features['volume'] = np.round(compute_volume(mesh), 3)
    features['aabb_volume'] = np.round(compute_aabb_volume(mesh), 3)
    features['compactness'] = np.round(compute_compactness(mesh), 3)
    features['diameter'] = np.round(compute_diameter(mesh), 3)
    features['convexity'] = np.round(compute_convexity(mesh), 3)
    features['eccentricity'] = np.round(compute_eccentricity(mesh), 3)

    return features


# Histogram configuration
SAMPLE_N = 2000
BINS = 10
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Ranges for normalization of histogram features
HIST_RANGES = {
    'A3': (0, math.pi),
    'D1': (0, 1.0),
    'D2': (0, 1.4),
    'D3': (0, 0.7),
    'D4': (0, 0.5)
}

def histogram(values, bins=BINS, value_range=(0,1)):
    """
    Compute normalized histogram vector for a list of values.
    """
    hist, _ = np.histogram(values, bins=bins, range=value_range)
    hist = hist / np.sum(hist)  # normalize to sum=1
    return hist

def compute_a3(mesh):
    """Angles between 3 random vertices."""
    vertices = list(mesh.vertices)
    angles = []
    for _ in range(SAMPLE_N):
        a, b, c = random.sample(vertices, 3)
        ab = b - a
        ac = c - a
        cos_angle = np.dot(ab, ac) / (np.linalg.norm(ab) * np.linalg.norm(ac))
        cos_angle = np.clip(cos_angle, -1, 1)  # avoid numerical errors
        angles.append(math.acos(cos_angle))
    return histogram(angles, bins=BINS, value_range=HIST_RANGES['A3'])

def compute_d1(mesh):
    """Distance from barycenter to random vertex."""
    vertices = list(mesh.vertices)
    center = mesh.centroid
    distances = [np.linalg.norm(random.choice(vertices) - center) for _ in range(SAMPLE_N)]
    return histogram(distances, bins=BINS, value_range=HIST_RANGES['D1'])

def compute_d2(mesh):
    """Distance between 2 random vertices."""
    vertices = list(mesh.vertices)
    distances = []
    for _ in range(SAMPLE_N):
        v1, v2 = random.sample(vertices, 2)
        distances.append(np.linalg.norm(v1 - v2))
    return histogram(distances, bins=BINS, value_range=HIST_RANGES['D2'])

def compute_d3(mesh):
    """Square root of triangle area from 3 random vertices."""
    vertices = list(mesh.vertices)
    sqr_areas = []
    for _ in range(SAMPLE_N):
        p1, p2, p3 = random.sample(vertices, 3)
        a = p2 - p1
        b = p3 - p1
        cross_prod = np.cross(a, b)
        area = 0.5 * np.linalg.norm(cross_prod)
        sqr_areas.append(math.sqrt(area))
    return histogram(sqr_areas, bins=BINS, value_range=HIST_RANGES['D3'])

def compute_d4(mesh):
    """Cube root of tetrahedron volume from 4 random vertices."""
    vertices = list(mesh.vertices)
    volumes = []
    for _ in range(SAMPLE_N):
        p1, p2, p3, p4 = random.sample(vertices, 4)
        mat = np.stack([p1-p4, p2-p4, p3-p4], axis=1)
        vol = abs(np.linalg.det(mat)) / 6
        volumes.append(vol ** (1/3))
    return histogram(volumes, bins=BINS, value_range=HIST_RANGES['D4'])

def extract_histogram_features(mesh):
    """
    Returns a dictionary with histogram vectors for all local descriptors.
    Keys: A3_0..9, D1_0..9, D2_0..9, D3_0..9, D4_0..9
    """
    features = {}
    features['A3'] = compute_a3(mesh)
    features['D1'] = compute_d1(mesh)
    features['D2'] = compute_d2(mesh)
    features['D3'] = compute_d3(mesh)
    features['D4'] = compute_d4(mesh)
    
    # flatten into BINS-length keys
    flat_features = {}
    for key, vec in features.items():
        for i, val in enumerate(vec):
            flat_features[f"{key}_{i}"] = val
    return flat_features

def standardize_single_value(value, mean, std):
    """
    Standardizes a single value:
    Centers at 0.5, scales so most values fall in [0,1].
    """
    return 0.5 + (value - mean) / (7 * std)

def standardize_column(column, mean=None, std=None):
    """
    Standardizes a list/array of values.
    If mean/std not provided, compute from column.
    Returns standardized column, mean, std.
    """
    if mean is None or std is None:
        mean = np.mean(column)
        std = np.std(column)

    standardized = [standardize_single_value(v, mean, std) for v in column]
    return standardized, mean, std

def standardize_features_db(features_df, columns_to_standardize):
    """
    Standardizes selected scalar columns of a dataframe.
    Returns new dataframe and a dict of standardization params.
    """
    std_params = {}
    for col in columns_to_standardize:
        features_df[col], mean, std = standardize_column(features_df[col])
        std_params[col] = {"mean": mean, "std": std}
    return features_df, std_params


# --- Geometric alignment utilities ---
def align_mesh_pca(mesh, verbose=False):
    """
    Align mesh so its principal axes coincide with coordinate axes.
    Largest variance → X, smallest → Z.
    """
    vertices = np.asarray(mesh.vertices)
    centered = vertices - np.mean(vertices, axis=0)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    rotation = eigvecs[:, order]
    mesh.vertices = np.dot(centered, rotation)
    if verbose:
        print("PCA alignment complete.")
    return mesh

def flip_mesh_orientation(mesh, verbose=False):
    """
    Flip mesh along axes so major mass lies on positive side.
    """
    centers = mesh.triangles_center
    inertia = np.sum(np.sign(centers) * (centers ** 2), axis=0)
    flips = np.sign(inertia)
    mesh.vertices *= flips
    if verbose:
        print(f"Applied flipping along axes {flips}")
    return mesh


# if __name__ == "__main__":
#     from read_data import get_random_data_from_directory
#     mesh = trimesh.load(get_random_data_from_directory(parent_directory="normalized_data"))

#     if not mesh.is_watertight:
#         mesh.fill_holes()

#     # Ensure alignment and flipping before feature extraction
#     mesh = align_mesh_pca(mesh)
#     mesh = flip_mesh_orientation(mesh)

#     scalars = extract_scalar_features(mesh)
#     print(scalars)

#     hist_feats = extract_histogram_features(mesh)
#     print(hist_feats)

#     from plots import show_mesh_simple
#     show_mesh_simple(mesh)


if __name__ == "__main__":
    from read_data import get_random_data_from_directory, get_data_from_directory
    from plots import show_mesh_simple
    from read_data import read_data
    import trimesh  # Add this import

    # mesh_path = get_data_from_directory(parent_directory="normalized_data", directory_name="Wheel")
    random.seed()
    mesh_path = get_random_data_from_directory(parent_directory="normalized_data")
    # mesh_path = "normalized_data/Truck/D00010_6939.obj"

    # Load as Open3D for visualization
    mesh_o3d = read_data(mesh_path)
    
    # Load as Trimesh for feature extraction
    mesh = trimesh.load(mesh_path)
    
    print("Before processing:")
    print(f"  Centroid: {mesh.centroid}")
    show_mesh_simple(mesh_o3d)  # Show original using Open3D mesh
    
    # Process mesh (Trimesh)
    mesh = align_mesh_pca(mesh, verbose=True)
    mesh = flip_mesh_orientation(mesh, verbose=True)
    
    print("After PCA + flipping:")
    print(f"  Centroid: {mesh.centroid}")
    
    # Convert back to Open3D for visualization
    import open3d as o3d
    mesh_o3d_processed = o3d.geometry.TriangleMesh()
    mesh_o3d_processed.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    mesh_o3d_processed.triangles = o3d.utility.Vector3iVector(mesh.faces)
    mesh_o3d_processed.compute_vertex_normals()
    
    show_mesh_simple(mesh_o3d_processed)  # Show processed using Open3D mesh
    
    # Extract features
    scalars = extract_scalar_features(mesh)
    hist_feats = extract_histogram_features(mesh)
    print("Features extracted successfully!")
    print("Scalar features:", scalars)
    print("First 10 histogram features:", {k: v for k, v in list(hist_feats.items())[:10]})