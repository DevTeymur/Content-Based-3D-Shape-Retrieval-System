import random
from plots import show_mesh_simple, plot_histograms
from read_data import read_data

import numpy as np
import open3d as o3d
import pandas as pd



def step1(display=True, mode='original'):
    from read_data import get_random_data_from_directory
    
    if mode == 'original':
        # Test original mesh
        mesh_path = 'data/Tool/m1106_8392.obj'  # Use path instead of mesh object
        show_mesh_simple(mesh_path) if display else None 
    elif mode == 'normalized':
        from plots import visualize_normalized_shape
        # Test normalized mesh
        normalized_path = get_random_data_from_directory(parent_directory="normalized_data")
        normalized_path = "normalized_data_test/Spoon/D00669_5895.obj"
        visualize_normalized_shape(normalized_path, axes_size=0.5)
    else:
        raise ValueError("Mode must be 'original' or 'normalized'")


def step2(logs=0, display=True):
    # Step 2.1 - this step done and correct no need to change
    from get_stats import extract_stats
    process_meshes = False
    if process_meshes:
        print("Extracting stats...", end=" ")
        all_data = extract_stats(logs=False) 
        print("done")
        print(f"Total files processed: {len(all_data)}")
        print("Saving to original_stats.csv", end=" ")
        df = pd.DataFrame(all_data)
        df.to_csv("stats/original_stats.csv", index=False)
    else:
        print("Loading from original_stats.csv", end=" ")
        df = pd.read_csv("stats/original_stats.csv")
    print("done")

    # Step 2.2 - same for this step, no further changes needed
    from get_stats import compute_averages, detect_outliers
    print("Computing averages for original...", end=" ")
    avg_vertices, avg_faces = compute_averages(df)
    print("done")
    print(f"Averages - Vertices: {avg_vertices:.0f}, Faces: {avg_faces:.0f}")

    # Draw average shape in database
    df["dist_to_avg"] = np.sqrt(
        (df["num_vertices"] - avg_vertices)**2 +
        (df["num_faces"] - avg_faces)**2
    )
    avg_shape_row = df.loc[df["dist_to_avg"].idxmin()]
    # Print how many vertices and faces the average shape has
    print(f"Average shape - Vertices: {avg_shape_row['num_vertices']}, Faces: {avg_shape_row['num_faces']}")
    avg_shape_path = avg_shape_row["file"] 

    # Load and visualize with Open3D
    # show_mesh_simple(read_data(avg_shape_path)) if display else None

    print("Detecting outliers...", end=" ")
    outliers = detect_outliers(df, avg_vertices, avg_faces)
    print("done")
    print(f"Total outliers found: {len(outliers)}")

    # Outlier with fewest and most vertices
    min_shape, max_shape = df.loc[df["num_vertices"].idxmin()], df.loc[df["num_vertices"].idxmax()]
    # Print how many vertices and faces the min and max shapes have
    print(f"Min shape - Vertices: {min_shape['num_vertices']}, Faces: {min_shape['num_faces']}")
    print(f"Max shape - Vertices: {max_shape['num_vertices']}, Faces: {max_shape['num_faces']}")    
    # Visualize outliers
    # for shape in [min_shape, max_shape]:
    #     show_mesh_simple(read_data(shape["file"])) if display else None

    print("Plotting histograms...", end=" ")
    # plot_histograms(df, step='2_2') if display else None
    print("done")
    exit()

    # Step 2.3
    print("---------Resampling step---------")
    resample_meshes = False
    if resample_meshes:
        from resample_utils import resample_all
        print("Resampling meshes...")
        resample_all(database_dir="data", target=7500, margin=2500, logs=1)
        print("done")

    
    # Step 2.4
    process_resampled_meshes = False
    if process_resampled_meshes:
        print("Extracting stats from resampled meshes...", end=" ")
        all_data = extract_stats(folder_path="resampled_data", logs=False) 
        print("done")
        print(f"Total files processed: {len(all_data)}")
        print("Saving to resampled_stats.csv", end=" ")
        resampled_df = pd.DataFrame(all_data)
        resampled_df.to_csv("stats/resampled_stats.csv", index=False)
    else:
        print("Loading from resampled_stats.csv", end=" ")
        resampled_df = pd.read_csv("stats/resampled_stats.csv")
        print("done")

    print("Computing averages for resampled...", end=" ")
    avg_vertices, avg_faces = compute_averages(resampled_df)
    print("done")
    print(f"Averages - Vertices: {avg_vertices:.0f}, Faces: {avg_faces:.0f}")

    # Histograms after resampling
    print("Plotting histograms...", end=" ")
    # plot_histograms(resampled_df, step='2_4') 
    print("done")


    # Step 2.5
    print("---------Normalization step (filtered)---------")
    normalize_meshes = True
    if normalize_meshes:
        from normalize_utils import normalize_filtered_database_trimesh  # Use new function
        input_database = "resampled_data"
        output_database = "normalized_data"
        filter_csv = "stats/resampled_stats_with_flags.csv"  # CSV with flags from draft.py
        
        # OLD - uses Open3D functions (incomplete)
        # normalize_filtered_database(input_database, output_database, filter_csv, logs=logs)

        # NEW - uses Trimesh functions (complete)
        normalize_filtered_database_trimesh(input_database, output_database, filter_csv, logs=logs)
        print("Filtered normalization complete.")

    process_normalized_meshes = True
    if process_normalized_meshes:
        print("Extracting stats from normalized meshes...", end=" ")
        all_data = extract_stats(folder_path="normalized_data", logs=False) 
        print("done")
        print(f"Total files processed: {len(all_data)}")
        print("Saving to normalized_stats.csv", end=" ")
        normalized_df = pd.DataFrame(all_data)
        normalized_df.to_csv("stats/normalized_stats.csv", index=False)
    else:
        print("Loading from normalized_stats.csv", end=" ")
        normalized_df = pd.read_csv("stats/normalized_stats.csv")
        print("done")

    print("Computing averages for normalized data...", end=" ")
    avg_vertices, avg_faces = compute_averages(normalized_df)
    print("done")
    print(f"Averages - Vertices: {avg_vertices:.0f}, Faces: {avg_faces:.0f}")

    plot_histograms(normalized_df, step='2_5') 

    
# Step A. Normalization pipeline (mesh centering + scaling + resampling).
# Step B. Core feature functions (diameter, convexity, eccentricity, etc.).
# Step C. Histogram-based local features (A3, D1–D4).
# Step D. Standardization helpers.
# Step E. Single-mesh feature integration.
# Step F. Database-wide feature extraction (CSV export).
# Step G. Query + distance computation + retrieval.
# Step H. Final integration in app.py.
def step3(logs=0, display=True):
    """Step 3: Features extraction and retrieval system"""
    
    # Step F: Build features database (FULL DATABASE)
    build_features_db = False
    if build_features_db:
        # First, normalize FULL database with PCA + flipping
        print("Normalizing FULL database with PCA + flipping...", end=" ")
        from normalize_utils import normalize_filtered_database_trimesh
        normalize_filtered_database_trimesh(
            input_dir="resampled_data",
            output_dir="normalized_data",  # ← Back to full database
            filter_csv="stats/resampled_stats_with_flags.csv",
            logs=logs
        )
        print("done")
        
        # Build features database from FULL normalized data
        print("Building features database from FULL dataset...", end=" ")
        from database_features import extract_features_database
        features_df = extract_features_database(
            input_dir="normalized_data",  # ← Full database
            output_csv="stats/features_database.csv",  # ← Main database
            filter_csv=None,  # Process all normalized meshes
            standardize=False,  # ← Keep this False (we learned it works better)
            logs=(logs > 0)
        )
        print("done")
        print(f"Features database created with {len(features_df)} meshes")
    
    # Add after features_df is created:
    if build_features_db:
        print("Analyzing feature distributions...", end=" ")
        
        # Feature distribution analysis
        scalar_features = ['area', 'volume', 'diameter', 'compactness', 'convexity', 'eccentricity']
        
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, feature in enumerate(scalar_features):
            if feature in features_df.columns:
                axes[i].hist(features_df[feature], bins=30, alpha=0.7)
                axes[i].set_title(f'{feature.title()} Distribution')
                axes[i].set_xlabel(feature)
                axes[i].set_ylabel('Count')
        
        plt.tight_layout()
        plt.savefig('img/feature_distributions.png', dpi=300, bbox_inches='tight')
        if display:
            plt.show()
        print("done")
    
    # Step G: Test retrieval system on FULL database
    test_retrieval = True
    if test_retrieval:
        print("Testing shape retrieval system on FULL database...", end=" ")
        
        # Use existing ShapeRetrieval class instead of missing test_shape_retrieval
        from retrieval import ShapeRetrieval
        from read_data import get_random_data_from_directory
        
        try:
            # Initialize retrieval system
            retrieval_system = ShapeRetrieval("stats/features_database.csv")
            
            # Test with a random query
            test_query = get_random_data_from_directory(parent_directory="normalized_data")
            
            # Run a quick test query
            results = retrieval_system.search_similar_shapes(
                query_mesh_path=test_query,
                k=5,
                logs=False  # No detailed logs for test
            )
            
            if results and len(results) > 0:
                print("done")
                print(f"Test successful: Found {len(results)} similar shapes")
            else:
                print("failed - no results returned")
                
        except Exception as e:
            print(f"failed - {str(e)}")
    
    # Step H: Query interface on FULL data
    run_simple_query = True  # ← Set to True when you want to test specific queries
    if run_simple_query:
        print("Running query on FULL database...")
        from retrieval import ShapeRetrieval
        from read_data import get_random_data_from_directory
        
        # Initialize retrieval system with FULL database
        retrieval_system = ShapeRetrieval("stats/features_database.csv")
        
        # Get random query mesh from normalized data (NO SEED!)
        query_path = get_random_data_from_directory(parent_directory="normalized_data")
        print(f"Query mesh: {query_path}")
        
        # Optional: Show the query mesh first
        # from plots import show_mesh_simple
        # print("Query mesh visualization:")
        # show_mesh_simple(query_path)
        
        # Run query with visualization
        results, fig = retrieval_system.search_and_visualize(
            query_mesh_path=query_path,
            k=6,
            scalar_weight=0.6,
            exclude_self=True,
            logs=(logs > 0),
            save_path=f"img/step3_2.png",
        )
        print("Query completed")


logs = 1  # 0: no logs, 1: some logs, 2: detailed logs
step = 2
display = True  # Whether to display meshes or not


# step1(display=display, mode='normalized') if step==1 else None
step2(logs=logs, display=False) if step==2 else None
step3(logs=logs, display=display) if step==3 else None  # Add this line