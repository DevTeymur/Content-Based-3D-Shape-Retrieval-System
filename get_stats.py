import os
from read_data import read_data
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt


# Step 2.1
def get_object_info(file_path, logs=False):
    # 1. Reading file
    mesh = read_data(file_path)

    # 2. Extract class name from parent folder
    class_name = os.path.basename(os.path.dirname(file_path))
    
    # 3. Number of vertices and faces
    num_vertices = len(mesh.vertices)

    face_types = {"triangles": 0, "quads": 0, "other": 0}
    faces = []
    if file_path.endswith('.obj'):
        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith('f '):
                    parts = line.strip().split()[1:]
                    faces.append(parts)
                    if len(parts) == 3:
                        face_types["triangles"] += 1
                    elif len(parts) == 4:
                        face_types["quads"] += 1
                    else:
                        face_types["other"] += 1
    else:
        # For .ply/.stl, Open3D loads only triangles
        face_types["triangles"] = len(mesh.triangles)

    num_faces = sum(face_types.values())

    # 4. Bounding box
    bbox = mesh.get_axis_aligned_bounding_box()
    min_bound = np.round(bbox.get_min_bound(), 2)
    max_bound = np.round(bbox.get_max_bound(), 2)

    # Face type summary
    if face_types["triangles"] and not face_types["quads"] and not face_types["other"]:
        face_type_str = "only triangles"
    elif face_types["quads"] and not face_types["triangles"] and not face_types["other"]:
        face_type_str = "only quads"
    elif face_types["triangles"] and face_types["quads"] and not face_types["other"]:
        face_type_str = "mix of triangles and quads"
    else:
        face_type_str = "mixed or other types"

    if logs:
        print("-----Stats of current file-----")
        print(f"Current file: {file_path}")
        print(f"- Class: {class_name}")
        print(f"- Number of vertices: {num_vertices}")
        print(f"- Number of faces: {num_faces}")
        print(f"- Face types: {face_type_str}")
        print(f"- Bounding box: {min_bound.tolist()}, {max_bound.tolist()}")
        print()

    # 5. Return all information
    return {
        "file": file_path,
        "class": class_name,
        "num_vertices": num_vertices,
        "num_faces": num_faces,
        "face_types": face_type_str,
        "bounding_box": {
            "min": min_bound.tolist(),
            "max": max_bound.tolist()
        }
    }

# Step 2.1
def extract_stats(folder_path="data", logs=False):
    all_stats = []
    mesh_files = []
    # Collect all mesh files first
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(('.obj', '.ply', '.stl')):
                mesh_files.append(os.path.join(root, file))

    try:
        iterator = tqdm(mesh_files, desc="Processing meshes")
    except ImportError:
        iterator = mesh_files

    for idx, file_path in enumerate(iterator, 1):
        # print(f"[{idx}/{len(mesh_files)}] Processing: {file_path}") if 'tqdm' not in globals() else None
        stats = get_object_info(file_path, logs=logs)
        all_stats.append(stats)
    return all_stats

# Step 2.2
def compute_averages(df):
    avg_vertices = df["num_vertices"].mean()
    avg_faces = df["num_faces"].mean()
    print(f"Average vertices: {avg_vertices:.2f}")
    print(f"Average faces: {avg_faces:.2f}")
    return avg_vertices, avg_faces

# Step 2.2
def detect_outliers(df, avg_vertices, avg_faces, logs=0):
    std_vertices = df["num_vertices"].std()
    std_faces = df["num_faces"].std()

    outliers = df[
        (np.abs(df["num_vertices"] - avg_vertices) > 2*std_vertices) |
        (np.abs(df["num_faces"] - avg_faces) > 2*std_faces)
    ]
    print("Outliers:") if logs else None
    print(outliers) if logs else None
    return outliers


# Step 2.2
def calculate_avg_without_outliers(df, column="num_vertices"):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    avg_vertices_no_outliers = filtered_df["vertices"].mean()
    avg_faces_no_outliers = filtered_df["faces"].mean()
    print(f"Average vertices without outliers: {avg_vertices_no_outliers:.0f}")
    print(f"Average faces without outliers: {avg_faces_no_outliers:.0f}")
    return avg_vertices_no_outliers, avg_faces_no_outliers



if __name__ == "__main__":
    all_data = extract_stats(logs=False)
    df = pd.DataFrame(all_data)
    print(df.head())
    df.to_csv("stats.csv", index=False)
    