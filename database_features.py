import os
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import trimesh

# Import your existing functions
from features_helpers import extract_all_features_single_mesh, get_feature_names
from features import standardize_column

def extract_features_database(input_dir="normalized_data", 
                            output_csv="stats/features_database.csv",
                            filter_csv=None,
                            standardize=False,
                            standardization_csv="stats/standardization_parameters.csv",
                            logs=True):
    """
    Extract features from all meshes in database and save to CSV.
    
    Args:
        input_dir: Directory containing normalized meshes
        output_csv: Path to save features CSV
        filter_csv: Optional CSV with flags to filter meshes (from resampling step)
        standardize: Whether to apply standardization to scalar features
        standardization_csv: Path to save standardization parameters
        logs: Print progress information
        
    Returns:
        pd.DataFrame: Features matrix
    """
    
    if logs:
        print("=" * 60)
        print("STEP F: DATABASE-WIDE FEATURE EXTRACTION")
        print("=" * 60)
    
    # Initialize feature aggregation dictionary
    feature_aggregation = defaultdict(list)
    
    # Get list of files to process
    mesh_files_to_process = []
    
    if filter_csv:
        # Filter based on resampling flags
        if logs:
            print(f"Loading filter from: {filter_csv}")
        
        filter_df = pd.read_csv(filter_csv)
        valid_files = filter_df[filter_df['within_threshold'] == True]['file'].tolist()
        
        if logs:
            total_in_filter = len(filter_df)
            valid_count = len(valid_files)
            print(f"Filter results: {valid_count}/{total_in_filter} meshes passed resampling threshold")
        
        # Convert to normalized data paths
        for file_path in valid_files:
            # Convert resampled path to normalized path
            file_path = Path(file_path)
            normalized_path = Path(input_dir) / file_path.parent.name / file_path.name
            if normalized_path.exists():
                mesh_files_to_process.append(normalized_path)
        
    else:
        # Process all files in input directory
        input_path = Path(input_dir)
        for category_dir in input_path.iterdir():
            if category_dir.is_dir() and category_dir.name != ".DS_Store":
                for mesh_file in category_dir.iterdir():
                    if mesh_file.suffix.lower() in ['.obj', '.ply', '.stl'] and mesh_file.name != ".DS_Store":
                        mesh_files_to_process.append(mesh_file)
    
    if logs:
        print(f"Found {len(mesh_files_to_process)} meshes to process")
        print("Starting feature extraction...")
    
    # Process meshes with progress bar
    processed_count = 0
    skipped_count = 0
    
    progress_bar = tqdm(mesh_files_to_process, desc="Extracting features") if logs else mesh_files_to_process
    
    for mesh_file in progress_bar:
        try:
            # Extract features using Step E function
            mesh_features = extract_all_features_single_mesh(
                mesh_path=mesh_file, 
                standardize=False,  # We'll standardize later if needed
                logs=False
            )
            
            if mesh_features is None:
                skipped_count += 1
                continue
            
            # Add features to aggregation dictionary
            for feature_name, feature_value in mesh_features.items():
                feature_aggregation[feature_name].append(feature_value)
            
            processed_count += 1
            
            if logs and processed_count % 50 == 0:
                print(f"Processed {processed_count} meshes...")
                
        except Exception as e:
            if logs:
                print(f"Error processing {mesh_file}: {e}")
            skipped_count += 1
            continue
    
    if logs:
        print(f"\nFeature extraction complete:")
        print(f"  Successfully processed: {processed_count}")
        print(f"  Skipped/failed: {skipped_count}")
    
    # Convert aggregated features to DataFrame
    if logs:
        print("Converting to DataFrame...")
    
    features_matrix = pd.DataFrame.from_dict(feature_aggregation)
    
    # Apply standardization if requested
    standardization_params = None
    if standardize and processed_count > 0:
        if logs:
            print("Applying standardization to scalar features...")
        
        features_matrix, standardization_params = apply_standardization_to_database(
            features_matrix, 
            save_params=True,
            params_csv=standardization_csv,
            logs=logs
        )
    
    # Save features to CSV
    if logs:
        print(f"Saving features to: {output_csv}")
    
    features_matrix.to_csv(output_csv, index=False)
    
    if logs:
        print(f"Features database saved with shape: {features_matrix.shape}")
        print("Feature columns:", list(features_matrix.columns))
    
    return features_matrix

def apply_standardization_to_database(features_df, save_params=True, params_csv="standardization_parameters.csv", logs=True):
    """
    Apply standardization to scalar features in the database.
    
    Args:
        features_df: DataFrame with extracted features
        save_params: Whether to save standardization parameters
        params_csv: Path to save parameters
        logs: Print information
        
    Returns:
        tuple: (standardized_dataframe, standardization_parameters_dict)
    """
    
    # Define scalar features to standardize (exclude metadata and histogram features)
    scalar_features_to_standardize = [
        'surface_area', 'volume', 'aabb_volume', 'compactness', 
        'diameter', 'convexity', 'eccentricity'
    ]
    
    standardization_parameters = {
        "feature": [],
        "mean": [],
        "std": []
    }
    
    # Create copy to avoid modifying original
    standardized_df = features_df.copy()
    
    for feature in scalar_features_to_standardize:
        if feature in standardized_df.columns:
            if logs:
                print(f"  Standardizing {feature}...")
            
            # Apply standardization using your existing function
            standardized_column, mean_val, std_val = standardize_column(standardized_df[feature])
            
            # Update DataFrame
            standardized_df[feature] = standardized_column
            
            # Save parameters
            standardization_parameters["feature"].append(feature)
            standardization_parameters["mean"].append(mean_val)
            standardization_parameters["std"].append(std_val)
    
    # Save standardization parameters if requested
    if save_params:
        params_df = pd.DataFrame.from_dict(standardization_parameters)
        params_df.to_csv(params_csv, index=False)
        if logs:
            print(f"Standardization parameters saved to: {params_csv}")
    
    return standardized_df, standardization_parameters

def load_features_database(features_csv="features_database.csv", logs=True):
    """
    Load features database from CSV.
    
    Args:
        features_csv: Path to features CSV file
        logs: Print information
        
    Returns:
        pd.DataFrame: Loaded features database
    """
    if logs:
        print(f"Loading features database from: {features_csv}")
    
    try:
        features_df = pd.read_csv(features_csv)
        
        if logs:
            print(f"Loaded features database with shape: {features_df.shape}")
            print(f"Categories: {features_df['category'].unique()}")
            print(f"Total meshes: {len(features_df)}")
        
        return features_df
        
    except Exception as e:
        print(f"Error loading features database: {e}")
        return None

def get_database_statistics(features_df, logs=True):
    """
    Get statistics about the features database.
    
    Args:
        features_df: Features DataFrame
        logs: Print detailed statistics
        
    Returns:
        dict: Database statistics
    """
    stats = {}
    
    # Basic statistics
    stats['total_meshes'] = len(features_df)
    stats['total_features'] = len(features_df.columns) - 3  # Exclude metadata
    stats['categories'] = features_df['category'].unique().tolist()
    stats['meshes_per_category'] = features_df['category'].value_counts().to_dict()
    
    # Feature statistics
    scalar_features = [col for col in features_df.columns 
                      if col not in ['filename', 'filepath', 'category'] 
                      and not any(hist in col for hist in ['A3_', 'D1_', 'D2_', 'D3_', 'D4_'])]
    
    histogram_features = [col for col in features_df.columns 
                         if any(hist in col for hist in ['A3_', 'D1_', 'D2_', 'D3_', 'D4_'])]
    
    stats['scalar_features_count'] = len(scalar_features)
    stats['histogram_features_count'] = len(histogram_features)
    
    if logs:
        print("=" * 60)
        print("DATABASE STATISTICS")
        print("=" * 60)
        print(f"Total meshes: {stats['total_meshes']}")
        print(f"Total features: {stats['total_features']}")
        print(f"  - Scalar features: {stats['scalar_features_count']}")
        print(f"  - Histogram features: {stats['histogram_features_count']}")
        print(f"Categories ({len(stats['categories'])}):")
        for category, count in stats['meshes_per_category'].items():
            print(f"  - {category}: {count} meshes")
        print("=" * 60)
    
    return stats

def test_database_extraction(input_dir="normalized_data", sample_size=5):
    """
    Test database feature extraction on a small sample.
    
    Args:
        input_dir: Directory with normalized meshes
        sample_size: Number of meshes to test
    """
    print("=" * 60)
    print("TESTING DATABASE FEATURE EXTRACTION")
    print("=" * 60)
    
    # Get sample files
    sample_files = []
    input_path = Path(input_dir)
    
    for category_dir in input_path.iterdir():
        if category_dir.is_dir() and len(sample_files) < sample_size:
            for mesh_file in category_dir.iterdir():
                if mesh_file.suffix.lower() == '.obj' and len(sample_files) < sample_size:
                    sample_files.append(mesh_file)
    
    print(f"Testing with {len(sample_files)} sample meshes...")
    
    # Test feature extraction
    feature_list = defaultdict(list)
    
    for mesh_file in sample_files:
        print(f"Processing: {mesh_file.parent.name}/{mesh_file.name}")
        
        features = extract_all_features_single_mesh(mesh_file, logs=False)
        if features:
            for key, value in features.items():
                feature_list[key].append(value)
    
    # Create test DataFrame
    test_df = pd.DataFrame.from_dict(feature_list)
    print(f"\nTest extraction complete!")
    print(f"DataFrame shape: {test_df.shape}")
    print(f"Columns: {list(test_df.columns)}")
    
    return test_df

def get_comprehensive_database_statistics(features_csv="stats/features_database.csv", 
                                        save_stats=True, 
                                        output_file="stats/database_statistics.txt",
                                        logs=True):
    """
    Get comprehensive statistics about the features database.
    
    Args:
        features_csv: Path to features CSV file
        save_stats: Whether to save statistics to file
        output_file: Path to save statistics
        logs: Print detailed statistics
        
    Returns:
        dict: Comprehensive database statistics
    """
    # Load the database
    features_df = pd.read_csv(features_csv)
    
    stats = {}
    
    # =========================
    # BASIC STATISTICS
    # =========================
    stats['total_meshes'] = len(features_df)
    stats['categories'] = sorted(features_df['category'].unique().tolist())
    stats['total_categories'] = len(stats['categories'])
    stats['meshes_per_category'] = features_df['category'].value_counts().to_dict()
    
    # =========================
    # FEATURE ANALYSIS
    # =========================
    # Identify feature types
    metadata_cols = ['filename', 'filepath', 'category']
    scalar_features = [col for col in features_df.columns 
                      if col not in metadata_cols 
                      and not any(hist in col for hist in ['A3_', 'D1_', 'D2_', 'D3_', 'D4_'])]
    
    histogram_features = [col for col in features_df.columns 
                         if any(hist in col for hist in ['A3_', 'D1_', 'D2_', 'D3_', 'D4_'])]
    
    stats['total_features'] = len(features_df.columns) - len(metadata_cols)
    stats['scalar_features'] = scalar_features
    stats['scalar_features_count'] = len(scalar_features)
    stats['histogram_features_count'] = len(histogram_features)
    
    # Histogram feature breakdown
    stats['histogram_breakdown'] = {}
    for prefix in ['A3', 'D1', 'D2', 'D3', 'D4']:
        count = len([col for col in histogram_features if col.startswith(prefix + '_')])
        stats['histogram_breakdown'][prefix] = count
    
    # =========================
    # SCALAR FEATURE STATISTICS
    # =========================
    stats['scalar_statistics'] = {}
    for feature in scalar_features:
        if feature in features_df.columns:
            stats['scalar_statistics'][feature] = {
                'min': features_df[feature].min(),
                'max': features_df[feature].max(),
                'mean': features_df[feature].mean(),
                'std': features_df[feature].std(),
                'median': features_df[feature].median()
            }
    
    # =========================
    # CATEGORY DISTRIBUTION ANALYSIS
    # =========================
    stats['category_distribution'] = {}
    stats['category_distribution']['min_meshes'] = min(stats['meshes_per_category'].values())
    stats['category_distribution']['max_meshes'] = max(stats['meshes_per_category'].values())
    stats['category_distribution']['avg_meshes'] = np.mean(list(stats['meshes_per_category'].values()))
    stats['category_distribution']['categories_with_min'] = [cat for cat, count in stats['meshes_per_category'].items() 
                                                           if count == stats['category_distribution']['min_meshes']]
    stats['category_distribution']['categories_with_max'] = [cat for cat, count in stats['meshes_per_category'].items() 
                                                           if count == stats['category_distribution']['max_meshes']]
    
    # =========================
    # DATA QUALITY METRICS
    # =========================
    stats['data_quality'] = {}
    
    # Check for missing values
    stats['data_quality']['missing_values'] = {}
    for col in features_df.columns:
        missing_count = features_df[col].isnull().sum()
        if missing_count > 0:
            stats['data_quality']['missing_values'][col] = missing_count
    
    # Check for duplicate filenames
    stats['data_quality']['duplicate_filenames'] = features_df['filename'].duplicated().sum()
    
    # Check for outliers in scalar features (values beyond 3 std from mean)
    stats['data_quality']['outliers'] = {}
    for feature in scalar_features:
        if feature in features_df.columns:
            mean_val = features_df[feature].mean()
            std_val = features_df[feature].std()
            outliers = ((features_df[feature] - mean_val).abs() > 3 * std_val).sum()
            if outliers > 0:
                stats['data_quality']['outliers'][feature] = outliers
    
    # =========================
    # FEATURE RANGE ANALYSIS
    # =========================
    stats['feature_ranges'] = {}
    for feature in scalar_features:
        if feature in features_df.columns:
            feature_range = features_df[feature].max() - features_df[feature].min()
            stats['feature_ranges'][feature] = feature_range
    
    # =========================
    # PRINT AND SAVE RESULTS
    # =========================
    if logs or save_stats:
        report = generate_statistics_report(stats)
        
        if logs:
            print(report)
        
        if save_stats:
            with open(output_file, 'w') as f:
                f.write(report)
            print(f"\nStatistics saved to: {output_file}")
    
    return stats

def generate_statistics_report(stats):
    """Generate a formatted statistics report."""
    
    report = []
    report.append("=" * 80)
    report.append("COMPREHENSIVE DATABASE STATISTICS")
    report.append("=" * 80)
    
    # Basic Statistics
    report.append(f"\n📊 BASIC STATISTICS:")
    report.append(f"   Total meshes: {stats['total_meshes']:,}")
    report.append(f"   Total categories: {stats['total_categories']}")
    report.append(f"   Total features: {stats['total_features']}")
    report.append(f"   - Scalar features: {stats['scalar_features_count']}")
    report.append(f"   - Histogram features: {stats['histogram_features_count']}")
    
    # Feature Breakdown
    report.append(f"\n  FEATURE BREAKDOWN:")
    report.append(f"   Scalar features: {', '.join(stats['scalar_features'])}")
    report.append(f"   Histogram features:")
    for prefix, count in stats['histogram_breakdown'].items():
        report.append(f"     - {prefix}: {count} bins")
    
    # Category Distribution
    report.append(f"\n  CATEGORY DISTRIBUTION:")
    report.append(f"   Average meshes per category: {stats['category_distribution']['avg_meshes']:.1f}")
    report.append(f"   Min meshes per category: {stats['category_distribution']['min_meshes']} "
                 f"({', '.join(stats['category_distribution']['categories_with_min'][:3])}{'...' if len(stats['category_distribution']['categories_with_min']) > 3 else ''})")
    report.append(f"   Max meshes per category: {stats['category_distribution']['max_meshes']} "
                 f"({', '.join(stats['category_distribution']['categories_with_max'][:3])}{'...' if len(stats['category_distribution']['categories_with_max']) > 3 else ''})")
    
    # Scalar Feature Statistics
    report.append(f"\n  SCALAR FEATURE STATISTICS:")
    for feature, stats_dict in stats['scalar_statistics'].items():
        report.append(f"   {feature}:")
        report.append(f"     Range: [{stats_dict['min']:.3f}, {stats_dict['max']:.3f}]")
        report.append(f"     Mean: {stats_dict['mean']:.3f} ± {stats_dict['std']:.3f}")
        report.append(f"     Median: {stats_dict['median']:.3f}")
    
    # Data Quality
    report.append(f"\n  DATA QUALITY METRICS:")
    if not stats['data_quality']['missing_values']:
        report.append("   ✅ No missing values detected")
    else:
        report.append("   ❌ Missing values found:")
        for col, count in stats['data_quality']['missing_values'].items():
            report.append(f"     - {col}: {count} missing")
    
    if stats['data_quality']['duplicate_filenames'] == 0:
        report.append("   ✅ No duplicate filenames")
    else:
        report.append(f"   ❌ {stats['data_quality']['duplicate_filenames']} duplicate filenames")
    
    if not stats['data_quality']['outliers']:
        report.append("   ✅ No extreme outliers detected (>3σ)")
    else:
        report.append("   ⚠️  Outliers detected:")
        for feature, count in stats['data_quality']['outliers'].items():
            report.append(f"     - {feature}: {count} outliers")
    
    report.append("=" * 80)
    
    return "\n".join(report)

# Add this convenience function
def run_database_analysis(features_csv="stats/features_database.csv"):
    """
    Quick function to run complete database analysis.
    
    Usage:
        from database_features import run_database_analysis
        run_database_analysis()
    """
    print("Running comprehensive database analysis...")
    
    stats = get_comprehensive_database_statistics(
        features_csv=features_csv,
        save_stats=True,
        output_file="stats/database_statistics.txt",
        logs=True
    )
    
    return stats

if __name__ == "__main__":
    # Test the implementation
    # print("Testing Step F: Database-wide feature extraction")
    
    # Small test first
    # test_df = test_database_extraction(sample_size=3)

    # test_df.to_csv("stats/features_test_sample.csv", index=False)
    
    # Full extraction (uncomment when ready)
    # features_db = extract_features_database(
    #     input_dir="normalized_data",
    #     output_csv="stats/features_database.csv",
    #     filter_csv="stats/resampled_stats_with_flags.csv",
    #     standardize=True,
    #     logs=True
    # )

    features_db = load_features_database(features_csv="stats/features_database.csv", logs=True)
    
    # Get statistics
    stats = get_database_statistics(features_db)

    stats = run_database_analysis("stats/features_database.csv")