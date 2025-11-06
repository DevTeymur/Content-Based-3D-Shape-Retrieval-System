"""
Step 5 Part 2 - KNN Engine
- Load normalized features from Part 1
- Build efficient KNN index for fast similarity search
- Implement K-nearest and R-range query modes
- Provide integration interface for GUI
"""

import numpy as np
import pandas as pd
import pickle
import time
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import pairwise_distances

class KNNEngine:
    def __init__(self, data_dir="step5_data"):
        """
        Initialize KNN Engine with processed features from Part 1
        
        Args:
            data_dir: Directory containing processed features from preparation step
        """
        self.data_dir = Path(data_dir)
        self.X_features = None
        self.labels = None
        self.metadata = None
        self.feature_info = None
        self.nn_model = None
        self.is_fitted = False
        
    def load_processed_features(self):
        """Load processed features and metadata from Part 1"""
        try:
            # Load normalized features
            features_file = self.data_dir / "features_normalized.npy"
            self.X_features = np.load(features_file)
            print(f"✅ Loaded features matrix: {self.X_features.shape}")
            
            # Load labels (categories) - FIX: Add allow_pickle=True
            labels_file = self.data_dir / "labels.npy"
            self.labels = np.load(labels_file, allow_pickle=True)
            print(f"✅ Loaded labels: {len(self.labels)} categories")
            
            # Load metadata mapping
            metadata_file = self.data_dir / "metadata.pkl"
            with open(metadata_file, 'rb') as f:
                self.metadata = pickle.load(f)
            print(f"✅ Loaded metadata: {len(self.metadata)} shapes")
            
            # Load feature info
            info_file = self.data_dir / "feature_info.pkl"
            with open(info_file, 'rb') as f:
                self.feature_info = pickle.load(f)
            print(f"✅ Loaded feature info: {self.feature_info['n_features']} features")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading processed features: {e}")
            return False
    
    def build_index(self, n_neighbors=50, metric='euclidean', algorithm='auto', use_step4_normalization=False):
        """
        Build KNN index for fast similarity search
        
        Args:
            n_neighbors: Maximum number of neighbors to pre-compute
            metric: Distance metric ('euclidean', 'cosine', 'manhattan', 'chebyshev')
            algorithm: Algorithm for neighbor search ('auto', 'ball_tree', 'kd_tree', 'brute')
            use_step4_normalization: Use same normalization as Step 4 for fair comparison
        """
        if self.X_features is None:
            print("❌ Please load processed features first")
            return False
        
        try:
            print(f"🔄 Building KNN index with {metric} distance...")
            start_time = time.time()
            
            # CHECK IF FEATURES ARE ALREADY NORMALIZED
            feature_range = self.X_features.max() - self.X_features.min()
            feature_mean = abs(self.X_features.mean())
            
            print(f"📊 Feature statistics:")
            print(f"   Range: {feature_range:.3f}")
            print(f"   Mean: {feature_mean:.3f}")
            print(f"   Min: {self.X_features.min():.3f}")
            print(f"   Max: {self.X_features.max():.3f}")
            
            # APPLY STEP 4 STYLE NORMALIZATION ONLY IF REQUESTED
            if use_step4_normalization:
                print("🔧 Applying feature-type-specific normalization (MATCHING STEP 4)...")
                
                # MATCH STEP 4 EXACTLY: 7 scalar + 50 histogram
                n_scalar = 7  # Only basic scalar features
                n_histogram = 50  # 10 bins × 5 descriptors
                
                print(f"   📊 Feature split: {n_scalar} scalar + {n_histogram} histogram = {n_scalar + n_histogram} total")
                
                # Split features
                scalar_features = self.X_features[:, :n_scalar]
                histogram_features = self.X_features[:, n_scalar:]
                
                # Z-SCORE NORMALIZE ONLY SCALAR FEATURES
                scalar_means = np.mean(scalar_features, axis=0)
                scalar_stds = np.std(scalar_features, axis=0)
                scalar_stds[scalar_stds == 0] = 1  # Avoid division by zero
                scalar_normalized = (scalar_features - scalar_means) / scalar_stds
                
                # KEEP HISTOGRAM FEATURES AS-IS (unchanged!)
                histogram_normalized = histogram_features
                
                # CONCATENATE BACK
                self.X_features_normalized = np.hstack([scalar_normalized, histogram_normalized])
                
                print(f"   ✅ Scalar features z-normalized: {n_scalar} features")
                print(f"   ✅ Histogram features unchanged: {n_histogram} bins")
                
                features_for_knn = self.X_features_normalized
            else:
                # Use original features (already normalized from preparation)
                print("📏 Using original normalized features from preparation")
                features_for_knn = self.X_features
                self.X_features_normalized = self.X_features  # Store reference
            
            # Initialize NearestNeighbors model
            self.nn_model = NearestNeighbors(
                n_neighbors=min(n_neighbors, len(features_for_knn)),
                metric=metric,
                algorithm=algorithm,
                n_jobs=-1  # Use all CPU cores
            )
            
            # Fit the model
            self.nn_model.fit(features_for_knn)
            
            build_time = time.time() - start_time
            self.is_fitted = True
            
            print(f"✅ KNN index built successfully!")
            print(f"   ⏱️  Build time: {build_time:.2f} seconds")
            print(f"   📊 Index size: {len(features_for_knn)} shapes")
            print(f"   🎯 Distance metric: {metric}")
            print(f"   🔧 Algorithm: {algorithm}")
            print(f"   📏 Normalization: {'Step 4 z-score' if use_step4_normalization else 'Original preparation'}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error building KNN index: {e}")
            return False

    def query_knn(self, query_shape_index, k=10):
        """
        Perform K-nearest neighbors search - DEBUG VERSION
        """
        # print(f"\n🔍 === KNN SEARCH DEBUG ===")
        # print(f"Query Index: {query_shape_index}")
        # print(f"Requested K: {k}")
        
        if not self.is_fitted:
            print("❌ KNN index not built. Please call build_index() first")
            return None
        
        try:
            # Use the same normalized features that were used for building the index
            if hasattr(self, 'X_features_normalized'):
                query_vector = self.X_features_normalized[query_shape_index].reshape(1, -1)
            else:
                query_vector = self.X_features[query_shape_index].reshape(1, -1)
            
            # Perform KNN search
            start_time = time.time()
            distances, indices = self.nn_model.kneighbors(query_vector, n_neighbors=k)
            query_time = time.time() - start_time
            
            # sklearn returns sorted results, but let's ensure proper order
            distances = distances[0]  # Remove batch dimension
            indices = indices[0]      # Remove batch dimension
            
            # print(f"✅ KNN returned exactly {len(distances)} results")
            # print(f"   Distance range: [{distances.min():.4f}, {distances.max():.4f}]")
            # print(f"   Query time: {query_time:.4f}s")
            
            # Convert to results DataFrame (already sorted by sklearn)
            results = self._create_results_dataframe(indices, distances, query_time)
            
            # print(f"🔍 K-NN search completed: {k} neighbors in {query_time:.4f}s")
            # print(f"=== END KNN SEARCH DEBUG ===\n")
            return results
            
        except Exception as e:
            print(f"❌ Error in KNN query: {e}")
            return None
    
    def query_range(self, query_shape_index, radius=1.0):
        """
        Perform range (radius) search - ENHANCED DEBUG VERSION
        """
        print(f"\n🔍 === RANGE SEARCH DEBUG ===")
        print(f"Query Index: {query_shape_index}")
        print(f"Requested Radius: {radius}")
        print(f"Database Size: {len(self.X_features)}")
        
        if not self.is_fitted:
            print("❌ KNN index not built. Please call build_index() first")
            return None
        
        try:
            # Use the same normalized features that were used for building the index
            if hasattr(self, 'X_features_normalized'):
                query_vector = self.X_features_normalized[query_shape_index].reshape(1, -1)
                print(f"✅ Using normalized features")
            else:
                query_vector = self.X_features[query_shape_index].reshape(1, -1)
                print(f"✅ Using original features")
            
            print(f"Query vector shape: {query_vector.shape}")
            
            # Perform range search
            start_time = time.time()
            
            # CHECK WHICH METHOD IS BEING USED
            print(f"\n🔧 Checking available methods:")
            print(f"   nn_model type: {type(self.nn_model)}")
            print(f"   Has radius_neighbors: {hasattr(self.nn_model, 'radius_neighbors')}")
            
            # Use radius_neighbors if available
            if hasattr(self.nn_model, 'radius_neighbors'):
                print(f"🎯 Using radius_neighbors method")
                try:
                    distances, indices = self.nn_model.radius_neighbors(query_vector, radius=radius)
                    distances = distances[0]
                    indices = indices[0]
                    print(f"✅ radius_neighbors SUCCESS: {len(distances)} results")
                    print(f"   Distance range: [{distances.min():.4f}, {distances.max():.4f}]")
                    method_used = "radius_neighbors"
                except Exception as e:
                    print(f"❌ radius_neighbors FAILED: {e}")
                    print(f"🔄 Falling back to KNN method...")
                    # Fall through to KNN method
                    method_used = "knn_fallback"
            else:
                print(f"🔄 radius_neighbors not available, using KNN fallback")
                method_used = "knn_fallback"
            
            # KNN Fallback method (if radius_neighbors failed or unavailable)
            if method_used == "knn_fallback" or 'distances' not in locals():
                print(f"\n🔧 Using KNN fallback method:")
                
                # INCREASE the fallback limit significantly
                max_neighbors = min(1000, len(self.X_features))  # Increased from 100 to 1000
                print(f"   Max neighbors to query: {max_neighbors}")
                
                all_distances, all_indices = self.nn_model.kneighbors(query_vector, n_neighbors=max_neighbors)
                print(f"   KNN returned {len(all_distances[0])} neighbors")
                print(f"   KNN distance range: [{all_distances[0].min():.4f}, {all_distances[0].max():.4f}]")
                
                # Filter by radius
                mask = all_distances[0] <= radius
                distances = all_distances[0][mask]
                indices = all_indices[0][mask]
                
                print(f"✅ After radius filter ({radius}): {len(distances)} results")
                if len(distances) > 0:
                    print(f"   Filtered distance range: [{distances.min():.4f}, {distances.max():.4f}]")
                
                # CHECK IF WE HIT THE LIMIT
                if len(distances) == max_neighbors:
                    print(f"⚠️  WARNING: Hit max_neighbors limit! Might be missing results.")
                elif all_distances[0][-1] <= radius:
                    print(f"⚠️  WARNING: Last KNN neighbor ({all_distances[0][-1]:.4f}) is within radius. Increase max_neighbors!")
            
            query_time = time.time() - start_time
            
            print(f"\n📊 FINAL RESULTS:")
            print(f"   Method used: {method_used}")
            print(f"   Results found: {len(distances)}")
            print(f"   Query time: {query_time:.4f}s")
            
            # Sort by distance before creating DataFrame
            if len(distances) > 0:
                print(f"🔧 Sorting {len(distances)} results...")
                # Create pairs and sort by distance
                distance_index_pairs = list(zip(distances, indices))
                distance_index_pairs.sort(key=lambda x: x[0])  # Sort by distance
                
                # Unzip back to separate arrays
                distances = np.array([pair[0] for pair in distance_index_pairs])
                indices = np.array([pair[1] for pair in distance_index_pairs])
                
                print(f"✅ Sorted - Distance range: [{distances.min():.4f}, {distances.max():.4f}]")
                print(f"   First 5 distances: {distances[:5]}")
                print(f"   Last 5 distances: {distances[-5:] if len(distances) > 5 else distances}")
            
            # Convert to results DataFrame
            results = self._create_results_dataframe(indices, distances, query_time)
            
            print(f"📋 Created DataFrame with {len(results)} rows")
            print(f"🔍 Range search completed: {len(results)} neighbors within radius {radius} in {query_time:.4f}s")
            print(f"=== END RANGE SEARCH DEBUG ===\n")
            
            return results
            
        except Exception as e:
            print(f"❌ Error in range query: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def query_by_filename(self, filename, k=10, search_type='knn', radius=1.0):
        """
        Query by filename instead of index
        
        Args:
            filename: Name of the query shape file
            k: Number of neighbors (for KNN)
            search_type: 'knn' or 'range'
            radius: Search radius (for range search)
            
        Returns:
            DataFrame with search results
        """
        # Find index by filename
        query_index = None
        for i, meta in enumerate(self.metadata):
            if meta['filename'] == filename:
                query_index = i
                break
        
        if query_index is None:
            print(f"❌ Shape '{filename}' not found in database")
            return None
        
        print(f"🎯 Query shape: {filename} (index: {query_index})")
        
        if search_type == 'knn':
            return self.query_knn(query_index, k)
        elif search_type == 'range':
            return self.query_range(query_index, radius)
        else:
            print(f"❌ Unknown search type: {search_type}")
            return None
    
    def _create_results_dataframe(self, indices, distances, query_time):
        """Create formatted results DataFrame from indices and distances"""
        results_data = []
        
        for i, (idx, dist) in enumerate(zip(indices, distances)):
            meta = self.metadata[idx]
            results_data.append({
                'rank': i + 1,  # This should already be correct since we sort before calling this
                'database_index': idx,
                'filename': meta['filename'],
                'category': meta['category'],
                'filepath': meta['filepath'],
                'distance': dist,
                'query_time': query_time if i == 0 else 0  # Only store query time once
            })
        
        # Create DataFrame and ensure it's sorted by distance (double-check)
        df = pd.DataFrame(results_data)
        df = df.sort_values('distance').reset_index(drop=True)
        
        # Re-assign correct ranks after sorting
        df['rank'] = range(1, len(df) + 1)
        
        return df
    
    def get_shape_info(self, shape_index):
        """Get detailed information about a shape by index"""
        if shape_index < 0 or shape_index >= len(self.metadata):
            return None
        
        meta = self.metadata[shape_index]
        features = self.X_features[shape_index]
        
        info = {
            'index': shape_index,
            'filename': meta['filename'],
            'category': meta['category'],
            'filepath': meta['filepath'],
            'label': self.labels[shape_index],
            'feature_vector': features,
            'feature_stats': {
                'min': float(features.min()),
                'max': float(features.max()),
                'mean': float(features.mean()),
                'std': float(features.std())
            }
        }
        
        return info
    
    def evaluate_performance(self, n_test_queries=100, k_values=[1, 5, 10, 20]):
        """
        Evaluate KNN performance with different parameters
        
        Args:
            n_test_queries: Number of random queries to test
            k_values: List of k values to test
            
        Returns:
            Performance statistics
        """
        if not self.is_fitted:
            print("❌ KNN index not built. Please call build_index() first")
            return None
        
        print(f"🔬 Evaluating KNN performance with {n_test_queries} test queries...")
        
        # Random test indices
        test_indices = np.random.choice(len(self.X_features), size=n_test_queries, replace=False)
        
        performance_stats = {}
        
        for k in k_values:
            print(f"   Testing k={k}...")
            query_times = []
            
            for query_idx in test_indices:
                start_time = time.time()
                self.query_knn(query_idx, k)
                query_time = time.time() - start_time
                query_times.append(query_time)
            
            performance_stats[k] = {
                'avg_query_time': np.mean(query_times),
                'std_query_time': np.std(query_times),
                'min_query_time': np.min(query_times),
                'max_query_time': np.max(query_times)
            }
        
        return performance_stats
    
    def get_statistics(self):
        """Get comprehensive statistics about the KNN engine"""
        if not self.is_fitted:
            print("❌ KNN index not built")
            return None
        
        stats = {
            'database_size': len(self.X_features),
            'feature_dimensions': self.X_features.shape[1],
            'n_categories': len(set(self.labels)),
            'categories': list(set(self.labels)),
            'feature_range': {
                'min': float(self.X_features.min()),
                'max': float(self.X_features.max()),
                'mean': float(self.X_features.mean()),
                'std': float(self.X_features.std())
            },
            'model_params': self.nn_model.get_params() if self.nn_model else None
        }
        
        return stats
    
    def print_statistics(self):
        """Print detailed statistics about the KNN engine"""
        stats = self.get_statistics()
        if stats is None:
            return
        
        print(f"\n📊 KNN ENGINE STATISTICS")
        print("=" * 50)
        print(f"Database size: {stats['database_size']} shapes")
        print(f"Feature dimensions: {stats['feature_dimensions']}")
        print(f"Number of categories: {stats['n_categories']}")
        print(f"Feature range: [{stats['feature_range']['min']:.3f}, {stats['feature_range']['max']:.3f}]")
        print(f"Feature mean: {stats['feature_range']['mean']:.3f}")
        print(f"Feature std: {stats['feature_range']['std']:.3f}")
        
        if stats['model_params']:
            print(f"\n🔧 MODEL PARAMETERS")
            for param, value in stats['model_params'].items():
                print(f"  {param}: {value}")

def main():
    """Main function to test KNN engine"""
    print("🚀 Starting KNN Engine Testing...")
    
    # Initialize KNN engine
    knn = KNNEngine()
    
    # Step 1: Load processed features
    if not knn.load_processed_features():
        print("❌ Failed to load processed features")
        return
    
    # Step 2: Build KNN index
    if not knn.build_index(n_neighbors=50, metric='euclidean'):
        print("❌ Failed to build KNN index")
        return
    
    # Step 3: Test KNN search
    print(f"\n🔍 Testing KNN search...")
    test_index = 0  # First shape in database
    results = knn.query_knn(test_index, k=10)
    
    if results is not None:
        print(f"\nTop 5 similar shapes to {knn.metadata[test_index]['filename']}:")
        print(results[['rank', 'filename', 'category', 'distance']].head().to_string(index=False))
    
    # Step 4: Test range search
    print(f"\n🎯 Testing range search...")
    range_results = knn.query_range(test_index, radius=2.0)
    
    if range_results is not None:
        print(f"Found {len(range_results)} shapes within radius 2.0")
        print(range_results[['filename', 'category', 'distance']].head().to_string(index=False))
    
    # Step 5: Test filename query
    print(f"\n📁 Testing filename query...")
    filename = knn.metadata[5]['filename']  # Test with 6th shape
    filename_results = knn.query_by_filename(filename, k=5)
    
    if filename_results is not None:
        print(f"Results for '{filename}':")
        print(filename_results[['rank', 'filename', 'category', 'distance']].to_string(index=False))
    
    # Step 6: Performance evaluation
    print(f"\n⚡ Performance evaluation...")
    perf_stats = knn.evaluate_performance(n_test_queries=20, k_values=[1, 5, 10])
    
    if perf_stats:
        print("Average query times:")
        for k, stats in perf_stats.items():
            print(f"  k={k}: {stats['avg_query_time']:.4f}s (±{stats['std_query_time']:.4f}s)")
    
    # Step 7: Print statistics
    knn.print_statistics()
    
    print("\n✅ KNN Engine testing completed!")

if __name__ == "__main__":
    main()