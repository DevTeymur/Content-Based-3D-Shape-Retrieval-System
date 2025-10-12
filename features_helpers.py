import trimesh
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

# Import your existing feature functions
from features import extract_scalar_features, extract_histogram_features
from normalize_utils import full_normalization

def load_and_normalize_mesh(mesh_path):
    """
    Load mesh and apply full normalization if needed.
    
    Args:
        mesh_path: Path to mesh file
        
    Returns:
        trimesh.Trimesh: Loaded and normalized mesh
    """
    try:
        # Load mesh using trimesh
        mesh = trimesh.load(str(mesh_path))
        
        # Check if mesh needs normalization (simple heuristic)
        centroid = mesh.centroid
        bbox_extent = mesh.bounding_box.extents
        max_extent = np.max(bbox_extent)
        
        # If not centered or not unit scale, apply normalization
        if np.linalg.norm(centroid) > 0.01 or abs(max_extent - 1.0) > 0.01:
            mesh = full_normalization(mesh, verbose=False)
            
        return mesh
        
    except Exception as e:
        print(f"Error loading mesh {mesh_path}: {e}")
        return None

def extract_all_features_single_mesh(mesh_path, standardize=False, logs=False):
    """
    Extract both scalar and histogram features for one mesh.
    
    Args:
        mesh_path: Path to mesh file
        standardize: Whether to apply feature standardization
        logs: Print progress information
        
    Returns:
        dict: Combined feature dictionary with metadata
    """
    mesh_path = Path(mesh_path)
    
    if logs:
        print(f"Extracting features from: {mesh_path.name}")
    
    # Load and normalize mesh
    mesh = load_and_normalize_mesh(mesh_path)
    if mesh is None:
        return None
    
    # Initialize feature dictionary
    all_features = {}
    
    # Add metadata
    all_features['filename'] = mesh_path.name
    all_features['filepath'] = str(mesh_path)
    all_features['category'] = mesh_path.parent.name
    
    try:
        # Extract scalar features
        if logs:
            print("  Computing scalar features...")
        scalar_features = extract_scalar_features(mesh)
        all_features.update(scalar_features)
        
        # Extract histogram features
        if logs:
            print("  Computing histogram features...")
        histogram_features = extract_histogram_features(mesh)
        all_features.update(histogram_features)
        
        if logs:
            total_features = len(scalar_features) + len(histogram_features)
            print(f"  Extracted {total_features} features total")
            print(f"    - {len(scalar_features)} scalar features")
            print(f"    - {len(histogram_features)} histogram features")
        
        # Apply standardization if requested
        if standardize:
            # TODO: Implement standardization using your existing functions
            if logs:
                print("  Applying standardization...")
            # all_features = apply_standardization(all_features)
        
        return all_features
        
    except Exception as e:
        print(f"Error extracting features from {mesh_path}: {e}")
        return None

def extract_features_from_mesh_object(mesh, filename=None, category=None):
    """
    Extract features from an already loaded mesh object.
    
    Args:
        mesh: trimesh.Trimesh object
        filename: Optional filename for metadata
        category: Optional category for metadata
        
    Returns:
        dict: Feature dictionary
    """
    all_features = {}
    
    # Add metadata if provided
    if filename:
        all_features['filename'] = filename
    if category:
        all_features['category'] = category
    
    # Extract features
    scalar_features = extract_scalar_features(mesh)
    histogram_features = extract_histogram_features(mesh)
    
    all_features.update(scalar_features)
    all_features.update(histogram_features)
    
    return all_features

def test_single_mesh_features(mesh_path=None, logs=True):
    """
    Test feature extraction on one mesh to verify Step E works.
    
    Args:
        mesh_path: Path to test mesh (if None, will try to find one)
        logs: Print detailed information
        
    Returns:
        dict: Extracted features or None if failed
    """
    if mesh_path is None:
        # Try to find a test mesh
        test_dirs = ["normalized_data", "resampled_data", "data"]
        for test_dir in test_dirs:
            test_path = Path(test_dir)
            if test_path.exists():
                # Find first .obj file
                for class_dir in test_path.iterdir():
                    if class_dir.is_dir():
                        for mesh_file in class_dir.iterdir():
                            if mesh_file.suffix.lower() == '.obj':
                                mesh_path = mesh_file
                                break
                        if mesh_path:
                            break
                if mesh_path:
                    break
    
    if mesh_path is None:
        print("No test mesh found. Please provide mesh_path.")
        return None
    
    print("=" * 60)
    print("TESTING SINGLE MESH FEATURE EXTRACTION (Step E)")
    print("=" * 60)
    print(f"Test mesh: {mesh_path}")
    
    # Extract features
    features = extract_all_features_single_mesh(mesh_path, logs=logs)
    
    if features:
        print("\n" + "=" * 60)
        print("FEATURE EXTRACTION RESULTS")
        print("=" * 60)
        
        # Separate scalar and histogram features
        scalar_features = {}
        histogram_features = {}
        metadata = {}
        
        for key, value in features.items():
            if key in ['filename', 'filepath', 'category']:
                metadata[key] = value
            elif any(hist_name in key for hist_name in ['A3_', 'D1_', 'D2_', 'D3_', 'D4_']):
                histogram_features[key] = value
            else:
                scalar_features[key] = value
        
        # Print results
        print(f"Metadata:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")
        
        print(f"\nScalar Features ({len(scalar_features)}):")
        for key, value in scalar_features.items():
            print(f"  {key}: {value:.6f}")
        
        print(f"\nHistogram Features ({len(histogram_features)}):")
        print("  A3 histogram:", [f"{features.get(f'A3_{i}', 0):.3f}" for i in range(10)])
        print("  D1 histogram:", [f"{features.get(f'D1_{i}', 0):.3f}" for i in range(10)])
        print("  D2 histogram:", [f"{features.get(f'D2_{i}', 0):.3f}" for i in range(10)])
        print("  D3 histogram:", [f"{features.get(f'D3_{i}', 0):.3f}" for i in range(10)])
        print("  D4 histogram:", [f"{features.get(f'D4_{i}', 0):.3f}" for i in range(10)])
        
        print(f"\nTotal features extracted: {len(features) - 3}")  # Exclude metadata
        print("=" * 60)
        print("✅ Step E: Single mesh feature extraction SUCCESSFUL!")
        print("=" * 60)
        
        return features
    else:
        print("❌ Feature extraction failed!")
        return None

def get_feature_names():
    """
    Get list of all feature names that will be extracted.
    Useful for creating DataFrame columns.
    
    Returns:
        list: List of feature names
    """
    feature_names = ['filename', 'filepath', 'category']
    
    # Scalar feature names (from your features.py)
    scalar_names = ['surface_area', 'volume', 'aabb_volume', 'compactness', 
                   'diameter', 'convexity', 'eccentricity']
    feature_names.extend(scalar_names)
    
    # Histogram feature names
    histogram_descriptors = ['A3', 'D1', 'D2', 'D3', 'D4']
    bins = 10  # From your BINS constant
    
    for descriptor in histogram_descriptors:
        for i in range(bins):
            feature_names.append(f'{descriptor}_{i}')
    
    return feature_names

if __name__ == "__main__":
    # Test the implementation
    print("Testing Step E: Single mesh feature extraction")
    test_single_mesh_features()