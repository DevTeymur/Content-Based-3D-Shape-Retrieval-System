"""
Step 5 Feature Preparation
- Load feature database from Step 3
- Apply proper normalization for KNN indexing
- Separate scalar vs histogram features
- Save processed features for ANN index building
"""

import numpy as np
import pandas as pd
from pathlib import Path
import pickle

class FeaturePreparation:
    def __init__(self, features_csv_path="stats/features_database.csv"):
        """
        Initialize feature preparation system
        
        Args:
            features_csv_path: Path to the features database CSV file
        """
        self.features_csv_path = features_csv_path
        self.features_df = None
        self.feature_columns = None
        self.scalar_features = None
        self.histogram_features = None
        self.X_scaled = None
        self.metadata_mapping = None
        
    def load_features_database(self):
        """Load feature database and identify feature columns"""
        try:
            # Load the features database
            self.features_df = pd.read_csv(self.features_csv_path)
            print(f"✅ Loaded {len(self.features_df)} shapes from {self.features_csv_path}")
            
            # Handle missing values
            inf_mask = np.isinf(self.features_df.select_dtypes(include=[np.number]))
            if inf_mask.any().any():
                print("⚠️  Found infinite values, replacing with 0")
                self.features_df = self.features_df.replace([np.inf, -np.inf], 0)
            
            # Fill NaN values
            numeric_columns = self.features_df.select_dtypes(include=[np.number]).columns
            self.features_df[numeric_columns] = self.features_df[numeric_columns].fillna(0)
            
            # Identify feature columns (exclude metadata)
            metadata_columns = ['filename', 'category', 'filepath']
            self.feature_columns = [col for col in self.features_df.columns 
                                  if col not in metadata_columns]
            
            print(f"✅ Identified {len(self.feature_columns)} feature columns")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading features database: {e}")
            return False
    
    def separate_feature_types(self):
        """Separate scalar features from histogram features"""
        if self.feature_columns is None:
            print("❌ Please load features database first")
            return False
        
        # Identify histogram features (A3, D1, D2, D3, D4 distributions)
        histogram_prefixes = ['A3_', 'D1_', 'D2_', 'D3_', 'D4_']
        
        self.histogram_features = [col for col in self.feature_columns 
                                 if any(prefix in col for prefix in histogram_prefixes)]
        
        self.scalar_features = [col for col in self.feature_columns 
                              if col not in self.histogram_features]
        
        print(f"✅ Separated features:")
        print(f"   📊 Scalar features: {len(self.scalar_features)}")
        print(f"   📈 Histogram features: {len(self.histogram_features)}")
        
        return True
    
    def normalize_features(self, method='zscore', histogram_weight=0.6, scalar_weight=0.4):
        """
        Normalize features using specified method
        
        Args:
            method: 'zscore' or 'minmax'
            histogram_weight: Weight for histogram features in final combination
            scalar_weight: Weight for scalar features in final combination
        """
        if self.scalar_features is None or self.histogram_features is None:
            print("❌ Please separate feature types first")
            return False
        
        try:
            # Get feature matrices
            scalar_matrix = self.features_df[self.scalar_features].values
            histogram_matrix = self.features_df[self.histogram_features].values
            
            print(f"🔄 Normalizing features using {method} method...")
            
            # Normalize scalar features
            if method == 'zscore':
                scalar_means = np.mean(scalar_matrix, axis=0)
                scalar_stds = np.std(scalar_matrix, axis=0)
                scalar_stds[scalar_stds == 0] = 1  # Avoid division by zero
                scalar_normalized = (scalar_matrix - scalar_means) / scalar_stds
                
                # Normalize histogram features
                hist_means = np.mean(histogram_matrix, axis=0)
                hist_stds = np.std(histogram_matrix, axis=0)
                hist_stds[hist_stds == 0] = 1
                hist_normalized = (histogram_matrix - hist_means) / hist_stds
                
            elif method == 'minmax':
                # Min-max normalization [0, 1]
                scalar_mins = np.min(scalar_matrix, axis=0)
                scalar_maxs = np.max(scalar_matrix, axis=0)
                scalar_ranges = scalar_maxs - scalar_mins
                scalar_ranges[scalar_ranges == 0] = 1
                scalar_normalized = (scalar_matrix - scalar_mins) / scalar_ranges
                
                hist_mins = np.min(histogram_matrix, axis=0)
                hist_maxs = np.max(histogram_matrix, axis=0)
                hist_ranges = hist_maxs - hist_mins
                hist_ranges[hist_ranges == 0] = 1
                hist_normalized = (histogram_matrix - hist_mins) / hist_ranges
            
            else:
                raise ValueError(f"Unknown normalization method: {method}")
            
            # Weight the feature types to prevent one type from dominating
            scalar_weighted = scalar_normalized * scalar_weight
            hist_weighted = hist_normalized * histogram_weight
            
            # Combine into final feature matrix
            n_shapes = len(self.features_df)
            n_scalar = len(self.scalar_features)
            n_hist = len(self.histogram_features)
            
            self.X_scaled = np.zeros((n_shapes, n_scalar + n_hist))
            self.X_scaled[:, :n_scalar] = scalar_weighted
            self.X_scaled[:, n_scalar:] = hist_weighted
            
            print(f"✅ Feature normalization complete")
            print(f"   📐 Final matrix shape: {self.X_scaled.shape}")
            print(f"   📊 Scalar weight: {scalar_weight}, Histogram weight: {histogram_weight}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error during normalization: {e}")
            return False
    
    def create_metadata_mapping(self):
        """Create index to metadata mapping for ANN results"""
        if self.features_df is None:
            print("❌ Please load features database first")
            return False
        
        self.metadata_mapping = []
        
        for idx, row in self.features_df.iterrows():
            metadata = {
                'index': idx,
                'filename': row['filename'],
                'category': row['category'],
                'filepath': row['filepath']
            }
            self.metadata_mapping.append(metadata)
        
        print(f"✅ Created metadata mapping for {len(self.metadata_mapping)} shapes")
        return True
    
    def save_processed_features(self, output_dir="step5_data"):
        """Save processed features and metadata to disk"""
        if self.X_scaled is None or self.metadata_mapping is None:
            print("❌ Please process features and create metadata mapping first")
            return False
        
        try:
            # Create output directory
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            # Save normalized feature matrix
            features_file = output_path / "features_normalized.npy"
            np.save(features_file, self.X_scaled)
            print(f"✅ Saved normalized features to {features_file}")
            
            # Save labels (categories)
            labels = self.features_df['category'].values
            labels_file = output_path / "labels.npy"
            np.save(labels_file, labels)
            print(f"✅ Saved labels to {labels_file}")
            
            # Save metadata mapping
            metadata_file = output_path / "metadata.pkl"
            with open(metadata_file, 'wb') as f:
                pickle.dump(self.metadata_mapping, f)
            print(f"✅ Saved metadata mapping to {metadata_file}")
            
            # Save feature column information
            feature_info = {
                'all_features': self.feature_columns,
                'scalar_features': self.scalar_features,
                'histogram_features': self.histogram_features,
                'n_shapes': len(self.features_df),
                'n_features': self.X_scaled.shape[1]
            }
            
            info_file = output_path / "feature_info.pkl"
            with open(info_file, 'wb') as f:
                pickle.dump(feature_info, f)
            print(f"✅ Saved feature info to {info_file}")
            
            # Save processing summary
            summary = {
                'n_shapes': len(self.features_df),
                'n_total_features': len(self.feature_columns),
                'n_scalar_features': len(self.scalar_features),
                'n_histogram_features': len(self.histogram_features),
                'final_matrix_shape': self.X_scaled.shape,
                'data_files': {
                    'features': str(features_file),
                    'labels': str(labels_file),
                    'metadata': str(metadata_file),
                    'feature_info': str(info_file)
                }
            }
            
            summary_file = output_path / "processing_summary.txt"
            with open(summary_file, 'w') as f:
                f.write("STEP 5 FEATURE PREPARATION SUMMARY\n")
                f.write("=" * 50 + "\n\n")
                for key, value in summary.items():
                    f.write(f"{key}: {value}\n")
            
            print(f"✅ Saved processing summary to {summary_file}")
            print(f"\n🎯 All processed data saved to: {output_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving processed features: {e}")
            return False
    
    def get_feature_statistics(self):
        """Print detailed feature statistics"""
        if self.X_scaled is None:
            print("❌ Please normalize features first")
            return
        
        print(f"\n📊 FEATURE STATISTICS")
        print("=" * 50)
        print(f"Dataset shape: {self.X_scaled.shape}")
        print(f"Feature range: [{self.X_scaled.min():.3f}, {self.X_scaled.max():.3f}]")
        print(f"Feature mean: {self.X_scaled.mean():.3f}")
        print(f"Feature std: {self.X_scaled.std():.3f}")
        
        # Category distribution
        if self.features_df is not None:
            category_counts = self.features_df['category'].value_counts()
            print(f"\n📁 CATEGORY DISTRIBUTION")
            print(f"Number of categories: {len(category_counts)}")
            print("Top 5 categories:")
            for cat, count in category_counts.head().items():
                print(f"  {cat}: {count} shapes")

def main():
    """Main function to run feature preparation"""
    print("🚀 Starting Step 5 Feature Preparation...")
    
    # Initialize preparation system
    prep = FeaturePreparation()
    
    # Step 1: Load features database
    if not prep.load_features_database():
        print("❌ Failed to load features database")
        return
    
    # Step 2: Separate feature types
    if not prep.separate_feature_types():
        print("❌ Failed to separate feature types")
        return
    
    # Step 3: Normalize features
    if not prep.normalize_features(method='zscore', histogram_weight=0.6, scalar_weight=0.4):
        print("❌ Failed to normalize features")
        return
    
    # Step 4: Create metadata mapping
    if not prep.create_metadata_mapping():
        print("❌ Failed to create metadata mapping")
        return
    
    # Step 5: Save processed features
    if not prep.save_processed_features():
        print("❌ Failed to save processed features")
        return
    
    # Step 6: Show statistics
    prep.get_feature_statistics()
    
    print("\n✅ Feature preparation completed successfully!")
    print("📁 Ready for KNN indexing and dimensionality reduction")

if __name__ == "__main__":
    main()