from plots import show_mesh_simple, plot_histograms
from read_data import read_data

import numpy as np
import open3d as o3d
import pandas as pd

logs = 1  # 0: no logs, 1: some logs, 2: detailed logs
step = 1
display = True  # Whether to display meshes or not


def step1(logs=0, display=True):
    from read_data import get_random_data_from_directory
    # Example 1
    # mesh = read_data('resampled_data/Car/m1518.obj')
    # Example 2 
    # mesh = read_data('resampled_data/Hand/D01172_9178.obj')
    # Example 3 - Random
    mesh = read_data(get_random_data_from_directory(parent_directory="data"))
    show_mesh_simple(mesh) if display else None 
    # from plots import visualize_normalized_shape
    # visualize_normalized_shape(get_random_data_from_directory(parent_directory="normalized_data"), axes_size=.5)

def step2(logs=0, display=True):
    # Step 2.1
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

    # Step 2.2
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
    avg_shape_path = avg_shape_row["file"] 

    # Load and visualize with Open3D
    show_mesh_simple(read_data(avg_shape_path)) if display else None

    print("Detecting outliers...", end=" ")
    outliers = detect_outliers(df, avg_vertices, avg_faces)
    print("done")
    print(f"Total outliers found: {len(outliers)}")

    # Outlier with fewest and most vertices
    min_shape, max_shape = df.loc[df["num_vertices"].idxmin()], df.loc[df["num_vertices"].idxmax()]

    # Visualize outliers
    for shape in [min_shape, max_shape]:
        show_mesh_simple(read_data(shape["file"])) if display else None

    print("Plotting histograms...", end=" ")
    plot_histograms(df, step='2_2') if display else None
    print("done")

    # Step 2.3
    resample_meshes = False
    if resample_meshes:
        from new_resample import resample_all
        print("Resampling meshes...")
        resample_all(database_dir="data", target=7000, margin=2000, logs=logs)
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
    plot_histograms(resampled_df, step='2_4') 
    print("done")

    # Step 2.5
    normalize_meshes = False
    if normalize_meshes == True:
        from normalize import normalize_database
        input_database = "resampled_data"  # original/resampled database
        output_database = "normalized_data"  # where normalized meshes will be saved
        normalize_database(input_database, output_database, logs=logs)
        print("Normalization complete.")

    process_normalized_meshes = False
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
    pass



step1(logs=logs, display=display) if step==1 else None
