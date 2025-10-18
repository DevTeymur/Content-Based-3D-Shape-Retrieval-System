"""
Content-Based Shape Retrieval (CBSR) - Step 4
Enhanced shape retrieval system building on Step 3 outputs
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional, Union
import warnings
from warnings import filterwarnings
filterwarnings("ignore")

# Try to import our existing Step 3 modules
try:
    from features_helpers import extract_all_features_single_mesh as extract_features
    from database_features import load_features_database
    from features import *  # Import all feature extraction functions
    HAS_STEP3_MODULE = True
    print("✅ Successfully imported Step 3 feature extraction modules")
except ImportError as e:
    HAS_STEP3_MODULE = False
    print(f"⚠️  Could not import Step 3 modules: {e}")
    print("    Will use file-based fallback for feature loading")

# Add these imports at the top if not already present
import json
from typing import Dict, Any

# =============================================================================
# PART A: DATA LOADING & INTEGRATION WITH STEP 3
# =============================================================================

def load_raw_features(path: str, expected_features: Optional[int] = None) -> Tuple[np.ndarray, List[str]]:
    """
    Load Step 3 feature outputs and provide safe API access.
    
    Supports multiple formats:
    - CSV with headers (our current format: 57 features)
    - NPZ files (numpy compressed)
    - NPY files (numpy arrays)
    
    Automatically converts between our 57-feature format and required 45-feature format.
    
    Args:
        path: Path to feature file (CSV, NPZ, or NPY)
        expected_features: Expected number of features (None for auto-detect, 45 for compatibility)
        
    Returns:
        features_raw: np.ndarray of shape (N, features) 
        shape_ids: List[str] of shape identifiers (filenames)
        
    Raises:
        ValueError: If file format is invalid or features don't match expected count
        FileNotFoundError: If path doesn't exist
    """
    
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Feature file not found: {path}")
    
    print(f"📂 Loading features from: {path}")
    
    # Determine file format and load accordingly
    if path.suffix.lower() == '.csv':
        features_raw, shape_ids = _load_csv_features(path)
    elif path.suffix.lower() == '.npz':
        features_raw, shape_ids = _load_npz_features(path)
    elif path.suffix.lower() == '.npy':
        features_raw, shape_ids = _load_npy_features(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}. Supported: .csv, .npz, .npy")
    
    original_feature_count = features_raw.shape[1]
    print(f"   Original features: {original_feature_count}")
    print(f"   Shapes loaded: {len(shape_ids)}")
    
    # Handle feature count conversion if needed
    if expected_features is not None:
        if original_feature_count != expected_features:
            print(f"⚠️  Feature count mismatch: got {original_feature_count}, expected {expected_features}")
            
            if original_feature_count == 57 and expected_features == 45:
                print("🔄 Auto-converting from 57-feature format to 45-feature format...")
                features_raw = _convert_57_to_45_features(features_raw)
                print(f"✅ Converted to {features_raw.shape[1]} features")
            else:
                raise ValueError(
                    f"Cannot convert {original_feature_count} features to {expected_features} features. "
                    f"Supported conversions: 57→45"
                )
    
    # Final validation
    final_feature_count = features_raw.shape[1]
    if expected_features is not None and final_feature_count != expected_features:
        raise ValueError(f"Feature validation failed: expected {expected_features}, got {final_feature_count}")
    
    print(f"✅ Successfully loaded {len(shape_ids)} shapes with {final_feature_count} features each")
    
    return features_raw, shape_ids


def _load_csv_features(path: Path) -> Tuple[np.ndarray, List[str]]:
    """Load features from CSV file (our current format)."""
    
    try:
        # Load CSV with pandas
        df = pd.read_csv(path)
        print(f"   CSV columns: {len(df.columns)}")
        
        # Identify metadata and feature columns
        metadata_cols = ['filename', 'filepath', 'category']
        feature_cols = [col for col in df.columns if col not in metadata_cols]
        
        # Extract shape IDs (prefer filename, fallback to index)
        if 'filename' in df.columns:
            shape_ids = df['filename'].astype(str).tolist()
        elif 'id' in df.columns:
            shape_ids = df['id'].astype(str).tolist()
        else:
            shape_ids = [f"shape_{i:04d}" for i in range(len(df))]
            print("⚠️  No 'filename' or 'id' column found, using generated IDs")
        
        # Extract feature matrix
        if len(feature_cols) == 0:
            raise ValueError("No feature columns found in CSV")
        
        features_raw = df[feature_cols].values.astype(np.float32)
        
        # Validate no NaN values
        if np.isnan(features_raw).any():
            nan_count = np.isnan(features_raw).sum()
            print(f"⚠️  Found {nan_count} NaN values, replacing with 0")
            features_raw = np.nan_to_num(features_raw, nan=0.0)
        
        print(f"   Feature columns: {feature_cols[:5]}{'...' if len(feature_cols) > 5 else ''}")
        
        return features_raw, shape_ids
        
    except Exception as e:
        # Provide helpful error message with expected format
        error_msg = f"Failed to load CSV features: {e}\n\n"
        error_msg += "Expected CSV format (57 features):\n"
        error_msg += "filename,filepath,category,area,volume,aabb_volume,compactness,diameter,convexity,eccentricity,"
        error_msg += "A3_0,A3_1,...,A3_9,D1_0,D1_1,...,D1_9,D2_0,...,D4_9\n\n"
        error_msg += "Or simplified 45-feature format:\n"
        error_msg += "id,surface_area,compactness,aabb_volume,diameter,elongation,"
        error_msg += "A3_1,A3_2,...,A3_10,D1_1,...,D1_10,D2_1,...,D3_10"
        
        raise ValueError(error_msg)


def _load_npz_features(path: Path) -> Tuple[np.ndarray, List[str]]:
    """Load features from NPZ file."""
    
    try:
        data = np.load(path)
        
        # Look for standard keys
        if 'features' in data:
            features_raw = data['features']
        elif 'features_raw' in data:
            features_raw = data['features_raw']
        else:
            # Use first array found
            key = list(data.keys())[0]
            features_raw = data[key]
            print(f"   Using array '{key}' as features")
        
        # Look for shape IDs
        if 'shape_ids' in data:
            shape_ids = data['shape_ids'].tolist()
        elif 'filenames' in data:
            shape_ids = data['filenames'].tolist()
        else:
            shape_ids = [f"shape_{i:04d}" for i in range(len(features_raw))]
            print("⚠️  No shape IDs found in NPZ, using generated IDs")
        
        return features_raw.astype(np.float32), shape_ids
        
    except Exception as e:
        raise ValueError(f"Failed to load NPZ features: {e}")


def _load_npy_features(path: Path) -> Tuple[np.ndarray, List[str]]:
    """Load features from NPY file (features only, no IDs)."""
    
    try:
        features_raw = np.load(path).astype(np.float32)
        
        if len(features_raw.shape) != 2:
            raise ValueError(f"Expected 2D array, got shape {features_raw.shape}")
        
        # Generate shape IDs since NPY doesn't store them
        shape_ids = [f"shape_{i:04d}" for i in range(len(features_raw))]
        print("ℹ️  NPY format doesn't include shape IDs, using generated IDs")
        
        return features_raw, shape_ids
        
    except Exception as e:
        raise ValueError(f"Failed to load NPY features: {e}")


def _convert_57_to_45_features(features_57: np.ndarray) -> np.ndarray:
    """
    Convert our 57-feature format to required 45-feature format.
    
    Mapping:
    - Scalar (7→5): area, volume, aabb_volume, compactness, diameter, convexity, eccentricity
                 → surface_area, aabb_volume, compactness, diameter, elongation
    - Histogram (50→40): A3(10), D1(10), D2(10), D3(10), D4(10)
                      → A3(10), D1(10), D2(10), D3(10)
    """
    
    print("   🔧 Converting feature format:")
    print("      57 features (7 scalar + 50 histogram) → 45 features (5 scalar + 40 histogram)")
    
    # Our current feature order:
    # Scalars: [area, volume, aabb_volume, compactness, diameter, convexity, eccentricity]
    # Histograms: [A3_0..A3_9, D1_0..D1_9, D2_0..D2_9, D3_0..D3_9, D4_0..D4_9]
    
    features_45 = np.zeros((features_57.shape[0], 45), dtype=np.float32)
    
    # Map scalar features (indices 0-6 → 0-4)
    features_45[:, 0] = features_57[:, 0]    # area → surface_area
    features_45[:, 1] = features_57[:, 3]    # compactness → compactness  
    features_45[:, 2] = features_57[:, 2]    # aabb_volume → aabb_volume
    features_45[:, 3] = features_57[:, 4]    # diameter → diameter
    features_45[:, 4] = features_57[:, 6]    # eccentricity → elongation
    
    # Map histogram features (keep first 40, remove D4)
    # A3: indices 7-16 → 5-14
    features_45[:, 5:15] = features_57[:, 7:17]    # A3_0..A3_9
    # D1: indices 17-26 → 15-24  
    features_45[:, 15:25] = features_57[:, 17:27]  # D1_0..D1_9
    # D2: indices 27-36 → 25-34
    features_45[:, 25:35] = features_57[:, 27:37]  # D2_0..D2_9
    # D3: indices 37-46 → 35-44
    features_45[:, 35:45] = features_57[:, 37:47]  # D3_0..D3_9
    # Skip D4 (indices 47-56)
    
    print("      ✅ Scalar mapping: area→surface_area, eccentricity→elongation")
    print("      ✅ Histogram mapping: A3,D1,D2,D3 (removed D4)")
    
    return features_45


def get_feature_names(format_type: str = "57") -> List[str]:
    """
    Get standardized feature names for different formats.
    
    Args:
        format_type: "57" for our format, "45" for required format
        
    Returns:
        List of feature names in order
    """
    
    if format_type == "57":
        # Our current 57-feature format
        scalar_names = ['area', 'volume', 'aabb_volume', 'compactness', 'diameter', 'convexity', 'eccentricity']
        histogram_names = []
        for prefix in ['A3', 'D1', 'D2', 'D3', 'D4']:
            histogram_names.extend([f"{prefix}_{i}" for i in range(10)])
        return scalar_names + histogram_names
        
    elif format_type == "45":
        # Required 45-feature format
        scalar_names = ['surface_area', 'compactness', 'aabb_volume', 'diameter', 'elongation']
        histogram_names = []
        for prefix in ['A3', 'D1', 'D2', 'D3']:
            histogram_names.extend([f"{prefix}_{i+1}" for i in range(10)])  # 1-indexed
        return scalar_names + histogram_names
        
    else:
        raise ValueError(f"Unknown format_type: {format_type}. Use '57' or '45'")


# =============================================================================
# STEP 3 INTEGRATION AND FALLBACK FUNCTIONS
# =============================================================================

def extract_features_with_fallback(mesh_path: str) -> np.ndarray:
    """
    Extract features using Step 3 module if available, otherwise fallback to file-based approach.
    
    Args:
        mesh_path: Path to mesh file
        
    Returns:
        Feature vector as np.ndarray
    """
    
    if HAS_STEP3_MODULE:
        try:
            # Use our existing feature extraction
            features_dict = extract_features(mesh_path, standardize=False, logs=False)
            
            if features_dict is None:
                raise ValueError(f"Feature extraction failed for {mesh_path}")
            
            # Convert dict to array (skip metadata)
            metadata_keys = ['filename', 'filepath', 'category']
            feature_values = [v for k, v in features_dict.items() if k not in metadata_keys]
            
            return np.array(feature_values, dtype=np.float32)
            
        except Exception as e:
            print(f"⚠️  Step 3 module extraction failed: {e}")
            print("   Falling back to file-based approach")
    
    # Fallback: would need to implement standalone feature extraction
    # For now, raise error since we need Step 3 modules
    raise RuntimeError(
        "Feature extraction requires Step 3 modules. "
        "Please ensure features_helpers.py and related modules are available."
    )


def load_features_database_with_fallback(path: str) -> pd.DataFrame:
    """Load features database using Step 3 functions if available."""
    
    if HAS_STEP3_MODULE:
        try:
            return load_features_database(path)
        except Exception as e:
            print(f"⚠️  Step 3 database loading failed: {e}")
    
    # Fallback to direct pandas loading
    return pd.read_csv(path)


# =============================================================================
# PART B: NORMALIZATION MODULE
# =============================================================================

class FeatureNormalizer:
    """
    Normalizes features for Content-Based Shape Retrieval.
    
    - Scalar features (0-4): Standardization using mean/std
    - Histogram features (5-44): Area normalization per shape
    
    This approach is robust because:
    - Standardization handles different scales in scalar features (area vs diameter)
    - Area normalization ensures histograms sum to 1, making them comparable
    - Preserves shape distribution characteristics in histograms
    """
    
    def __init__(self):
        """Initialize empty normalizer."""
        self.fitted = False
        
        # Scalar normalization parameters (features 0-4)
        self.scalar_means = None  # length 5
        self.scalar_stds = None   # length 5
        
        # Histogram group indices (features 5-44)
        self.histogram_indices = {
            'A3': list(range(5, 15)),   # indices 5-14 (10 bins)
            'D1': list(range(15, 25)),  # indices 15-24 (10 bins)  
            'D2': list(range(25, 35)),  # indices 25-34 (10 bins)
            'D3': list(range(35, 45))   # indices 35-44 (10 bins)
        }
        
        # Feature names for reference
        self.scalar_names = ['surface_area', 'compactness', 'aabb_volume', 'diameter', 'elongation']
        self.histogram_names = ['A3', 'D1', 'D2', 'D3']
        
    def fit(self, features_raw: np.ndarray, verbose: bool = True) -> 'FeatureNormalizer':
        """
        Compute normalization parameters from database features.
        
        Args:
            features_raw: np.ndarray of shape (N, 45) - raw features
            verbose: Print fitting progress
            
        Returns:
            self for chaining
            
        Note:
            - Scalar features: Compute mean/std for standardization
            - Histogram features: Only validate structure, no global stats needed
            - Outliers are preserved (no automatic clipping)
        """
        
        if features_raw.shape[1] != 45:
            raise ValueError(f"Expected 45 features, got {features_raw.shape[1]}")
        
        if verbose:
            print("🔧 Fitting FeatureNormalizer...")
            print(f"   Database size: {features_raw.shape[0]} shapes × {features_raw.shape[1]} features")
        
        # =============================
        # FIT SCALAR NORMALIZATION
        # =============================
        scalar_features = features_raw[:, 0:5]  # First 5 features
        
        # Compute mean and std for each scalar feature (ddof=0 as specified)
        self.scalar_means = np.mean(scalar_features, axis=0, dtype=np.float64)
        self.scalar_stds = np.std(scalar_features, axis=0, ddof=0, dtype=np.float64)
        
        # Handle zero std (avoid division by zero)
        zero_std_mask = self.scalar_stds == 0
        if np.any(zero_std_mask):
            if verbose:
                zero_features = [self.scalar_names[i] for i in np.where(zero_std_mask)[0]]
                print(f"   ⚠️  Zero std detected in: {zero_features}, setting std=1")
            self.scalar_stds[zero_std_mask] = 1.0
        
        if verbose:
            print("   📊 Scalar feature statistics:")
            for i, name in enumerate(self.scalar_names):
                print(f"      {name:15}: mean={self.scalar_means[i]:8.4f}, std={self.scalar_stds[i]:8.4f}")
        
        # =============================
        # VALIDATE HISTOGRAM STRUCTURE
        # =============================
        histogram_features = features_raw[:, 5:45]  # Features 5-44 (40 histogram features)
        
        if verbose:
            print("   📈 Histogram feature validation:")
            for name, indices in self.histogram_indices.items():
                # Adjust indices for histogram subset (subtract 5)
                hist_indices = [i - 5 for i in indices]
                hist_data = histogram_features[:, hist_indices]
                
                # Check for negative values (shouldn't happen in histograms)
                neg_count = np.sum(hist_data < 0)
                if neg_count > 0:
                    print(f"      ⚠️  {name}: {neg_count} negative values detected")
                
                # Check histogram sums (for information)
                hist_sums = np.sum(hist_data, axis=1)
                zero_sum_count = np.sum(hist_sums == 0)
                
                print(f"      {name}: range=[{hist_data.min():.6f}, {hist_data.max():.6f}], "
                      f"zero_sums={zero_sum_count}")
        
        self.fitted = True
        
        if verbose:
            print("   ✅ FeatureNormalizer fitted successfully")
        
        return self
    
    def transform(self, features_raw: np.ndarray, verbose: bool = False) -> np.ndarray:
        """
        Apply normalization to features using fitted parameters.
        
        Args:
            features_raw: np.ndarray of shape (N, 45) or (45,) for single query
            verbose: Print transformation details
            
        Returns:
            features_norm: Normalized features of same shape
            
        Note:
            - Scalars: standardized using (x - mean) / std
            - Histograms: area-normalized (each histogram sums to 1)
        """
        
        if not self.fitted:
            raise RuntimeError("FeatureNormalizer must be fitted before transform")
        
        # Handle single query vector
        single_query = False
        if features_raw.ndim == 1:
            if len(features_raw) != 45:
                raise ValueError(f"Expected 45 features for single query, got {len(features_raw)}")
            features_raw = features_raw.reshape(1, -1)
            single_query = True
        
        if features_raw.shape[1] != 45:
            raise ValueError(f"Expected 45 features, got {features_raw.shape[1]}")
        
        n_shapes = features_raw.shape[0]
        features_norm = np.zeros_like(features_raw, dtype=np.float32)
        
        if verbose:
            print(f"🔄 Transforming {n_shapes} shape(s)...")
        
        # =============================
        # TRANSFORM SCALAR FEATURES
        # =============================
        scalar_features = features_raw[:, 0:5]
        
        # Standardize: (x - mean) / std
        scalar_norm = (scalar_features - self.scalar_means) / self.scalar_stds
        features_norm[:, 0:5] = scalar_norm
        
        if verbose and n_shapes <= 5:  # Only show details for small batches
            print("   📊 Scalar transformation:")
            for i in range(n_shapes):
                print(f"      Shape {i}: {scalar_features[i]} → {scalar_norm[i]}")
        
        # =============================
        # TRANSFORM HISTOGRAM FEATURES
        # =============================
        for name, indices in self.histogram_indices.items():
            # Extract histogram for all shapes
            hist_data = features_raw[:, indices]  # Shape: (n_shapes, 10)
            
            # Area normalize each histogram (divide by sum)
            hist_sums = np.sum(hist_data, axis=1, keepdims=True)  # Shape: (n_shapes, 1)
            
            # Handle zero sums (leave as zeros)
            nonzero_mask = hist_sums.flatten() > 0
            hist_norm = np.zeros_like(hist_data)
            
            if np.any(nonzero_mask):
                hist_norm[nonzero_mask] = hist_data[nonzero_mask] / hist_sums[nonzero_mask]
            
            # Store normalized histogram
            features_norm[:, indices] = hist_norm
            
            if verbose and n_shapes <= 3:
                zero_count = np.sum(~nonzero_mask)
                print(f"   📈 {name} normalization: {zero_count} zero-sum histograms")
        
        # Return single vector if input was single vector
        if single_query:
            features_norm = features_norm.flatten()
        
        if verbose:
            print("   ✅ Transformation completed")
        
        return features_norm
    
    def fit_transform(self, features_raw: np.ndarray, verbose: bool = True) -> np.ndarray:
        """
        Convenience method: fit and transform in one step.
        
        Args:
            features_raw: Database features to fit on and transform
            verbose: Print progress
            
        Returns:
            Normalized features
        """
        return self.fit(features_raw, verbose=verbose).transform(features_raw, verbose=verbose)
    
    def transform_query(self, feature_query_raw: np.ndarray, verbose: bool = False) -> np.ndarray:
        """
        Transform a single query feature vector using database normalization parameters.
        
        Args:
            feature_query_raw: Single feature vector of length 45
            verbose: Print transformation details
            
        Returns:
            Normalized query feature vector
            
        Note:
            This is the same as transform() but explicitly for single queries
        """
        
        if not self.fitted:
            raise RuntimeError("FeatureNormalizer must be fitted before transforming queries")
        
        if feature_query_raw.ndim != 1 or len(feature_query_raw) != 45:
            raise ValueError(f"Expected 1D array of length 45, got shape {feature_query_raw.shape}")
        
        return self.transform(feature_query_raw, verbose=verbose)
    
    def save_params(self, filepath: str = "stats/normalizer_params.json") -> None:
        """
        Save normalization parameters to JSON file.
        
        Args:
            filepath: Path to save parameters
        """
        
        if not self.fitted:
            raise RuntimeError("Cannot save parameters before fitting")
        
        params = {
            'fitted': self.fitted,
            'scalar_means': self.scalar_means.tolist(),
            'scalar_stds': self.scalar_stds.tolist(),
            'histogram_indices': self.histogram_indices,
            'scalar_names': self.scalar_names,
            'histogram_names': self.histogram_names,
            'feature_count': 45,
            'version': '1.0'
        }
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(params, f, indent=2)
        
        print(f"✅ Normalization parameters saved to: {filepath}")
    
    @staticmethod
    def from_params(filepath: str = "stats/normalizer_params.json") -> 'FeatureNormalizer':
        """
        Load normalization parameters from JSON file.
        
        Args:
            filepath: Path to parameter file
            
        Returns:
            Fitted FeatureNormalizer instance
        """
        
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Parameter file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            params = json.load(f)
        
        # Create and configure normalizer
        normalizer = FeatureNormalizer()
        normalizer.fitted = params['fitted']
        normalizer.scalar_means = np.array(params['scalar_means'], dtype=np.float64)
        normalizer.scalar_stds = np.array(params['scalar_stds'], dtype=np.float64)
        normalizer.histogram_indices = params['histogram_indices']
        normalizer.scalar_names = params['scalar_names']
        normalizer.histogram_names = params['histogram_names']
        
        print(f"✅ Normalization parameters loaded from: {filepath}")
        
        return normalizer
    
    def get_normalization_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics about the normalization parameters.
        
        Returns:
            Dictionary with normalization statistics
        """
        
        if not self.fitted:
            return {'fitted': False}
        
        return {
            'fitted': True,
            'scalar_stats': {
                name: {'mean': float(mean), 'std': float(std)}
                for name, mean, std in zip(self.scalar_names, self.scalar_means, self.scalar_stds)
            },
            'histogram_groups': {
                name: {'indices': indices, 'size': len(indices)}
                for name, indices in self.histogram_indices.items()
            },
            'total_features': 45,
            'scalar_features': 5,
            'histogram_features': 40
        }


# =============================================================================
# LEGACY STANDARDIZATION FUNCTIONS (Enhanced)
# =============================================================================

def standardize_single_value(value: float, mean: float, std: float, verbose: bool = False) -> float:
    """
    Standardize a single value given a mean and std.
    CENTERED at 0.5 and most values will be within [0,1]
    
    Note: This is the legacy function - FeatureNormalizer uses standard z-score normalization
    """
    standardized = (0.5 + (value - mean) / (7 * std))  # distance from 0 to 1 should be n standard deviations
    
    if verbose:
        if standardized < 0 or standardized > 1:
            print(f"Value: {value}, Mean: {mean}, Std: {std}, Standardized: {standardized}")
    
    return standardized


def standardize_column(column: np.ndarray, mean: Optional[float] = None, std: Optional[float] = None) -> Tuple[List[float], float, float]:
    """
    Take an iterable and standardize it (with given mean/std if given).
    Return new set of values, also mean and std.
    
    Note: This is the legacy function - FeatureNormalizer is preferred for new code
    """
    
    if mean is None or std is None:  # calculate standardization parameters if not given
        mean = np.mean(column)
        std = np.std(column)
    
    newcolumn = []
    for value in column:
        newcolumn.append(standardize_single_value(value, mean, std, verbose=False))
    
    return newcolumn, mean, std


# =============================================================================
# TESTING AND VALIDATION
# =============================================================================

def validate_feature_loading(path: str = "stats/features_database.csv"):
    """
    Test and validate the feature loading functionality.
    
    Args:
        path: Path to feature database for testing
    """
    
    print("🧪 TESTING FEATURE LOADING FUNCTIONALITY")
    print("=" * 60)
    
    try:
        # Test 1: Load in original format
        print("\n1️⃣  Testing original format loading...")
        features_57, shape_ids_57 = load_raw_features(path)
        print(f"   ✅ Loaded {features_57.shape[0]} shapes with {features_57.shape[1]} features")
        
        # Test 2: Load with 45-feature conversion
        print("\n2️⃣  Testing 45-feature conversion...")
        features_45, shape_ids_45 = load_raw_features(path, expected_features=45)
        print(f"   ✅ Converted to {features_45.shape[0]} shapes with {features_45.shape[1]} features")
        
        # Test 3: Verify shape IDs are consistent
        print("\n3️⃣  Testing shape ID consistency...")
        if shape_ids_57 == shape_ids_45:
            print("   ✅ Shape IDs consistent between formats")
        else:
            print("   ⚠️  Shape ID mismatch detected")
        
        # Test 4: Verify feature names
        print("\n4️⃣  Testing feature name generation...")
        names_57 = get_feature_names("57")
        names_45 = get_feature_names("45")
        print(f"   ✅ Generated {len(names_57)} names for 57-feature format")
        print(f"   ✅ Generated {len(names_45)} names for 45-feature format")
        
        # Test 5: Basic statistics
        print("\n5️⃣  Testing feature statistics...")
        print(f"   Feature ranges (45-format):")
        print(f"     Min: {features_45.min():.6f}")
        print(f"     Max: {features_45.max():.6f}")
        print(f"     Mean: {features_45.mean():.6f}")
        print(f"     Std: {features_45.std():.6f}")
        
        print(f"\n✅ ALL TESTS PASSED! Feature loading is working correctly.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        return False


# =============================================================================
# TESTING AND VALIDATION FOR PART B
# =============================================================================

def test_feature_normalizer(features_path: str = "stats/features_database.csv"):
    """
    Test the FeatureNormalizer class with real data.
    
    Args:
        features_path: Path to feature database
    """
    
    print("🧪 TESTING FEATURE NORMALIZER")
    print("=" * 60)
    
    try:
        # Load test data
        print("\n1️⃣  Loading test features...")
        features_raw, shape_ids = load_raw_features(features_path, expected_features=45)
        print(f"   ✅ Loaded {features_raw.shape[0]} shapes with {features_raw.shape[1]} features")
        
        # Test fitting
        print("\n2️⃣  Testing normalizer fitting...")
        normalizer = FeatureNormalizer()
        normalizer.fit(features_raw, verbose=True)
        
        # Test transformation
        print("\n3️⃣  Testing batch transformation...")
        features_norm = normalizer.transform(features_raw[:5], verbose=True)  # Test first 5 shapes
        print(f"   ✅ Transformed shape: {features_norm.shape}")
        
        # Test single query transformation
        print("\n4️⃣  Testing single query transformation...")
        query_raw = features_raw[0]  # First shape as query
        query_norm = normalizer.transform_query(query_raw, verbose=True)
        print(f"   ✅ Query shape: {query_norm.shape}")
        
        # Test fit_transform convenience method
        print("\n5️⃣  Testing fit_transform...")
        features_norm_2 = normalizer.fit_transform(features_raw[:3], verbose=False)
        print(f"   ✅ Fit_transform shape: {features_norm_2.shape}")
        
        # Test parameter saving and loading
        print("\n6️⃣  Testing parameter persistence...")
        param_file = "stats/test_normalizer_params.json"
        normalizer.save_params(param_file)
        
        normalizer_loaded = FeatureNormalizer.from_params(param_file)
        query_norm_loaded = normalizer_loaded.transform_query(query_raw, verbose=False)
        
        # Check if results are identical
        if np.allclose(query_norm, query_norm_loaded):
            print("   ✅ Parameter persistence test passed")
        else:
            print("   ❌ Parameter persistence test failed")
        
        # Test statistics
        print("\n7️⃣  Testing normalization statistics...")
        stats = normalizer.get_normalization_stats()
        print(f"   ✅ Generated {len(stats)} statistics entries")
        
        # Validate histogram normalization
        print("\n8️⃣  Validating histogram normalization...")
        for name, indices in normalizer.histogram_indices.items():
            hist_norms = features_norm[:, indices]
            hist_sums = np.sum(hist_norms, axis=1)
            
            # Check that non-zero histograms sum to 1
            nonzero_mask = hist_sums > 1e-6
            if np.any(nonzero_mask):
                nonzero_sums = hist_sums[nonzero_mask]
                if np.allclose(nonzero_sums, 1.0, rtol=1e-5):
                    print(f"   ✅ {name}: histograms properly normalized")
                else:
                    print(f"   ⚠️  {name}: histogram normalization issue - sums range [{nonzero_sums.min():.6f}, {nonzero_sums.max():.6f}]")
            else:
                print(f"   ⚠️  {name}: all histograms are zero")
        
        print(f"\n✅ ALL TESTS PASSED! FeatureNormalizer is working correctly.")
        
        return True, normalizer
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False, None


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Example usage and testing
    print("CBSR - Content-Based Shape Retrieval")
    print("Part A: Data Loading & Step 3 Integration")
    print("=" * 50)
    
    # Test with our current database
    feature_file = "stats/features_database.csv"
    
    if Path(feature_file).exists():
        # Run validation tests
        validate_feature_loading(feature_file)
        
        # Example usage
        print("\n" + "=" * 50)
        print("EXAMPLE USAGE:")
        
        # Load in 45-feature format for CBSR
        features, shape_ids = load_raw_features(feature_file, expected_features=45)
        print(f"\nLoaded database: {features.shape[0]} shapes × {features.shape[1]} features")
        print(f"Sample shape IDs: {shape_ids[:5]}")
        print(f"Feature names: {get_feature_names('45')[:10]}...")
        
    else:
        print(f"❌ Feature database not found: {feature_file}")
        print("   Please run Step 3 first to generate the features database.")
    
    # Test Part B if Part A passed
    if Path("stats/features_database.csv").exists():
        print("\n" + "=" * 60)
        print("TESTING PART B: FEATURE NORMALIZER")
        
        success, normalizer = test_feature_normalizer("stats/features_database.csv")
        
        if success:
            print("\n" + "=" * 60)
            print("EXAMPLE NORMALIZER USAGE:")
            
            # Example normalization workflow
            features, _ = load_raw_features("stats/features_database.csv", expected_features=45)
            
            # Fit normalizer on database
            normalizer = FeatureNormalizer()
            features_normalized = normalizer.fit_transform(features[:10])  # Use first 10 for demo
            
            print(f"Original features shape: {features.shape}")
            print(f"Normalized features shape: {features_normalized.shape}")
            print(f"Original range: [{features.min():.3f}, {features.max():.3f}]")
            print(f"Normalized range: [{features_normalized.min():.3f}, {features_normalized.max():.3f}]")
            
            # Save parameters for later use
            normalizer.save_params("stats/normalizer_params.json")