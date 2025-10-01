import numpy as np
import trimesh
import random

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


if __name__ == "__main__":
    from read_data import get_random_data_from_directory
    mesh = trimesh.load(get_random_data_from_directory(parent_directory="normalized_data"))
    scalars = extract_scalar_features(mesh)
    print(scalars)
