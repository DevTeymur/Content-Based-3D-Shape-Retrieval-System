import pandas as pd
import numpy as np
from pathlib import Path
import os
from scipy.spatial.distance import euclidean, cosine, cityblock
from scipy.stats import wasserstein_distance

# Import your existing functions
from features_helpers import extract_all_features_single_mesh
from database_features import load_features_database
from features import standardize_column

from visualization import visualize_shape_retrieval, quick_retrieval_visualization

class ShapeRetrieval:
    """3D Shape Retrieval System"""
    
    def __init__(self, features_database_path="stats/features_database.csv"):
        """
        Initialize retrieval system with features database.
        
        Args:
            features_database_path: Path to features CSV file
        """
        self.features_db = load_features_database(features_database_path, logs=True)
        self.feature_columns = self._identify_feature_columns()
        
    def _identify_feature_columns(self):
        """Identify scalar and histogram feature columns"""
        if self.features_db is None:
            return None, None
            
        all_columns = list(self.features_db.columns)
        metadata_columns = ['filename', 'filepath', 'category']
        
        # Scalar features (not metadata, not histogram bins)
        scalar_features = [col for col in all_columns 
                          if col not in metadata_columns 
                          and not any(hist in col for hist in ['A3_', 'D1_', 'D2_', 'D3_', 'D4_'])]
        
        # Histogram features (A3_0, A3_1, ..., D4_9)
        histogram_features = [col for col in all_columns 
                             if any(hist in col for hist in ['A3_', 'D1_', 'D2_', 'D3_', 'D4_'])]
        
        print(f"Feature identification:")
        print(f"  Scalar features ({len(scalar_features)}): {scalar_features}")
        print(f"  Histogram features ({len(histogram_features)}): first 10: {histogram_features[:10]}...")
        
        return scalar_features, histogram_features
    
    def process_query_mesh(self, mesh_path, logs=False):
        """
        Process query mesh: load, normalize, extract features.
        
        Args:
            mesh_path: Path to query mesh file
            logs: Print processing information
            
        Returns:
            dict: Extracted features or None if failed
        """
        if logs:
            print(f"Processing query mesh: {mesh_path}")
        
        
        # Check if query is from normalized_data_test
        if "normalized_data_test" not in str(mesh_path):
            print("❌ WARNING: Query not from test normalized data!")
        
        try:
            # Extract features using Step E function (includes normalization)
            query_features = extract_all_features_single_mesh(
                mesh_path=mesh_path,
                standardize=False,  # We'll handle standardization separately
                logs=logs
            )
            
            if query_features is None:
                print(f"Failed to extract features from {mesh_path}")
                return None
                
            if logs:
                print(f"Successfully extracted {len(query_features)} features from query")
                
            # Debug: Print first few scalar features
            scalar_features = ['surface_area', 'volume', 'diameter', 'compactness']
            # print("DEBUG QUERY FEATURES:")
            # for feat in scalar_features:
            #     if feat in query_features:
            #         print(f"  {feat}: {query_features[feat]}")
                
            return query_features
            
        except Exception as e:
            print(f"Error processing query mesh {mesh_path}: {e}")
            return None
    
    def compute_feature_distances(self, query_features, database_row, distance_metrics=None):
        """
        Compute distances between query and database row.
        
        Args:
            query_features: Query feature dictionary
            database_row: Single row from features database
            distance_metrics: Dict with 'scalar' and 'histogram' metrics
            
        Returns:
            tuple: (scalar_distance, histogram_distance)
        """
        if distance_metrics is None:
            distance_metrics = {'scalar': 'euclidean', 'histogram': 'earth_movers'}
        
        scalar_features, histogram_features = self.feature_columns
        
        # Extract scalar feature vectors
        query_scalars = np.array([query_features.get(feat, 0) for feat in scalar_features])
        db_scalars = np.array([database_row[feat] for feat in scalar_features])
        
        # Extract histogram feature vectors  
        query_histograms = np.array([query_features.get(feat, 0) for feat in histogram_features])
        db_histograms = np.array([database_row[feat] for feat in histogram_features])
        
        # Compute scalar distance
        scalar_distance = self._compute_distance(
            query_scalars, db_scalars, distance_metrics['scalar']
        )
        
        # Compute histogram distance
        histogram_distance = self._compute_distance(
            query_histograms, db_histograms, distance_metrics['histogram']
        )
        
        return scalar_distance, histogram_distance
    
    def _compute_distance(self, vector1, vector2, metric):
        """Compute distance between two vectors using specified metric"""
        try:
            if metric == 'euclidean':
                return euclidean(vector1, vector2)
            elif metric == 'cosine':
                return cosine(vector1, vector2)
            elif metric == 'manhattan':
                return cityblock(vector1, vector2)
            elif metric == 'earth_movers':
                # For histograms - use Earth Mover's Distance
                bins1 = np.arange(len(vector1))
                bins2 = np.arange(len(vector2))
                return wasserstein_distance(bins1, bins2, vector1, vector2)
            else:
                raise ValueError(f"Unknown distance metric: {metric}")
        except Exception as e:
            print(f"Error computing {metric} distance: {e}")
            return float('inf')
    
    def search_similar_shapes(self, query_mesh_path, k=5, scalar_weight=0.5, 
                             distance_metrics=None, exclude_self=False, logs=True):
        """
        Find k most similar shapes to query mesh.
        
        Args:
            query_mesh_path: Path to query mesh
            k: Number of similar shapes to return
            scalar_weight: Weight for scalar features (0-1)
            distance_metrics: Distance metrics to use
            exclude_self: Exclude query mesh from results if it's in database
            logs: Print progress information
            
        Returns:
            pd.DataFrame: Results with similarity rankings
        """
        if self.features_db is None:
            print("Features database not loaded!")
            return None
            
        if logs:
            print("=" * 60)
            print("SHAPE RETRIEVAL QUERY")
            print("=" * 60)
        
        # Process query mesh
        query_features = self.process_query_mesh(query_mesh_path, logs=logs)
        if query_features is None:
            return None

        # ADD THIS DEBUG CODE HERE - right after query processing
        # Debug: Check database features
        db_sample = self.features_db.head(3)
        # print("DEBUG DATABASE SAMPLE:")
        # print(db_sample[['filename', 'area', 'volume', 'diameter']].to_string())
    
        # Initialize results
        similarity_results = []
        
        if logs:
            print(f"Computing distances to {len(self.features_db)} database shapes...")
        
        # Compute distances to each database entry
        for index, db_row in self.features_db.iterrows():
            try:
                scalar_dist, hist_dist = self.compute_feature_distances(
                    query_features, db_row, distance_metrics
                )
                
                similarity_results.append({
                    'database_index': index,
                    'filename': db_row['filename'],
                    'filepath': db_row['filepath'], 
                    'category': db_row['category'],
                    'scalar_distance': scalar_dist,
                    'histogram_distance': hist_dist
                })
                
            except Exception as e:
                if logs:
                    print(f"Error computing distance for {db_row['filename']}: {e}")
                continue
        
        # Convert to DataFrame
        results_df = pd.DataFrame(similarity_results)
        
        if len(results_df) == 0:
            print("No valid distance computations!")
            return None
        
        # Standardize distances before combining
        if logs:
            print("Standardizing distances...")
            
        results_df['scalar_dist_standardized'] = standardize_column(results_df['scalar_distance'])[0]
        results_df['histogram_dist_standardized'] = standardize_column(results_df['histogram_distance'])[0]
        
        # Compute combined distance
        histogram_weight = 1.0 - scalar_weight
        results_df['combined_distance'] = (
            scalar_weight * results_df['scalar_dist_standardized'] + 
            histogram_weight * results_df['histogram_dist_standardized']
        )
        
        # Sort by combined distance
        results_df = results_df.sort_values('combined_distance')
        
        # Exclude self if requested
        if exclude_self:
            query_filename = Path(query_mesh_path).name
            results_df = results_df[results_df['filename'] != query_filename]
        
        # Return top k results
        top_results = results_df.head(k)
        
        if logs:
            print(f"Retrieval complete! Found {len(top_results)} similar shapes.")
            self._print_results(top_results, query_mesh_path)
        
        return top_results
    
    def _print_results(self, results_df, query_path):
        """Print formatted retrieval results"""
        print("\n" + "=" * 80)
        print(f"TOP {len(results_df)} SIMILAR SHAPES TO: {Path(query_path).name} {Path(query_path).parent.name}")
        print("=" * 80)
        
        for i, (_, row) in enumerate(results_df.iterrows(), 1):
            print(f"{i:2d}. {row['filename']:<25} | {row['category']:<15} | "
                  f"Distance: {row['combined_distance']:.4f}")
            print(f"    Scalar: {row['scalar_distance']:.4f} | "
                  f"Histogram: {row['histogram_distance']:.4f}")
            print(f"    Path: {row['filepath']}")
            print()

    def search_and_visualize(self, query_mesh_path, k=5, scalar_weight=0.5, 
                        distance_metrics=None, exclude_self=False, 
                        save_path=None, logs=True):
        """
        Search for similar shapes and visualize results.
        
        Args:
            query_mesh_path: Path to query mesh
            k: Number of similar shapes to return
            scalar_weight: Weight for scalar features (0-1)
            distance_metrics: Distance metrics to use
            exclude_self: Exclude query mesh from results
            save_path: Optional path to save visualization
            logs: Print progress information
            
        Returns:
            tuple: (results_dataframe, matplotlib_figure)
        """
        # Get retrieval results
        results = self.search_similar_shapes(
            query_mesh_path=query_mesh_path,
            k=k,
            scalar_weight=scalar_weight,
            distance_metrics=distance_metrics,
            exclude_self=exclude_self,
            logs=logs
        )
        
        if results is None or len(results) == 0:
            print("No results to visualize!")
            return None, None
        
        # Create visualization
        fig = visualize_shape_retrieval(
            query_mesh_path=query_mesh_path,
            results_df=results,
            max_display=k,
            save_path=save_path
        )
        
        return results, fig


def run_retrieval_query(query_mesh_path, features_db_path="stats/features_database.csv", 
                       k=5, scalar_weight=0.5, show_results=True):
    """
    Convenient function to run a single retrieval query.
    
    Args:
        query_mesh_path: Path to query mesh
        features_db_path: Path to features database  
        k: Number of results
        scalar_weight: Weight for scalar vs histogram features
        show_results: Print detailed results
        
    Returns:
        pd.DataFrame: Retrieval results
    """
    retrieval_system = ShapeRetrieval(features_db_path)
    
    results = retrieval_system.search_similar_shapes(
        query_mesh_path=query_mesh_path,
        k=k,
        scalar_weight=scalar_weight,
        exclude_self=True,
        logs=show_results
    )
    
    return results

if __name__ == "__main__":
    # Test the retrieval system

    from visualization import quick_retrieval_visualization
    from read_data import get_random_data_from_directory
    # results, fig = quick_retrieval_visualization(
    #     query_mesh_path=get_random_data_from_directory(parent_directory="normalized_data"),
    #     k=6
    # )

    # from retrieval import ShapeRetrieval
    # retrieval_system = ShapeRetrieval()
    # results, fig = retrieval_system.search_and_visualize(
    #     # query_mesh_path=get_random_data_from_directory(parent_directory="normalized_data"),
    #     query_mesh_path=get_random_data_from_directory(parent_directory="normalized_data"),
    #     k=5,
    #     save_path="img/my_retrieval_result.png"
    # )

