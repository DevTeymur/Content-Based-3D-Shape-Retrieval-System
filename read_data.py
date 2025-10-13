import os
import random
import pandas as pd
import open3d as o3d
from pathlib import Path
import matplotlib.pyplot as plt

from plots import show_mesh_simple

DIR_NAME = "data"
RESAMPLED_DIR_NAME = "resampled_data"
NORMALIZED_DIR_NAME = "normalized_data"

# Step 1
def get_classdirs(directory_name=DIR_NAME):
    directory_name = Path(directory_name)
    classes = [(directory_name / f) for f in os.listdir(directory_name) if (directory_name / f).is_dir()]
    return classes

# Step 1
def get_objects(classdir):
    filenames = [(classdir / f) for f in os.listdir(classdir) if f.endswith(('.obj', '.ply', '.stl'))] 
    return filenames

# Step 2.3 and 2.4
def get_resampled_filenames(object_path):
    resampled_dir = "data_resampled"
    object_path = Path(object_path)
    filenames = [(object_path / f) for f in os.listdir(resampled_dir) if f.endswith(('.obj', '.ply', '.stl'))] 

    return filenames

# Step 1
def get_data_from_directory(parent_directory = 'data', directory_name=None, logs=True):
    if directory_name is None:
        # Pick a random folder from data/
        directory_name = random.choice([
            os.path.join(parent_directory, d)
            for d in os.listdir(parent_directory)
            if os.path.isdir(os.path.join(parent_directory, d))
        ])
        print(f"Selected random directory: {directory_name}") if logs else None

    if not os.path.exists(parent_directory + '/' + directory_name):
        raise ValueError(f"Directory {directory_name} does not exist.")
    
    # Pick a random file from the chosen directory
    files = [f for f in os.listdir(parent_directory + '/' + directory_name) if f.endswith(('.obj', '.ply', '.stl'))]
    if not files:
        raise ValueError(f"No .obj, .ply, or .stl files found in {directory_name}.")
    
    return os.path.join(directory_name, random.choice(files))

# Step 1
def get_random_data_from_directory(parent_directory="data", directory_name=None):
    """
    Get a random file from specified directory or any subdirectory.
    """
    import random
    from pathlib import Path
    
    # Don't set seed here - let it be truly random
    parent_path = Path(parent_directory)
    
    if directory_name:
        # Get random file from specific category
        target_dir = parent_path / directory_name
        if not target_dir.exists():
            raise ValueError(f"Directory {target_dir} does not exist")
        files = list(target_dir.glob("*.obj"))
    else:
        # Get random file from any subdirectory
        files = list(parent_path.glob("**/*.obj"))
    
    if not files:
        raise ValueError(f"No .obj files found in {parent_directory}")
    
    # Use current time for randomness
    import time
    random.seed(int(time.time() * 1000000) % 2**32)
    
    return str(random.choice(files))

# Step 1
def read_data(file_path):
    if file_path.endswith('.obj') or file_path.endswith('.ply') or file_path.endswith('.stl'):
        mesh = o3d.io.read_triangle_mesh(file_path).compute_vertex_normals()
        return mesh
    else:
        raise ValueError("Unsupported file format. Please provide a .obj, .ply, or .stl file.")

# Step 2.2
def read_stats(stats_path="stats",files=["original_stats.csv","resampled_stats.csv"]):
    stats_path = Path(stats_path)
    for f in files:
        f = Path(f)
        df = pd.read_csv(stats_path / f)
        s = df['num_vertices']

        s.plot(kind="hist", bins=100, edgecolor="black")
        plt.savefig(stats_path / f"histogram_{f.name}.png", dpi=300, bbox_inches="tight")
        plt.clf()


if __name__ == "__main__":
    # for i in range(25):
    #     mesh = read_data(get_random_data_from_directory())
    #     print(mesh)
    #     print("__"*10)
    mesh = read_data('resampled_data/Car/m1518.obj')
    mesh = read_data('resampled_data/Hand/D01172_9178.obj')
    # mesh = read_data('data/Car/D00236.obj')
    show_mesh_simple(mesh)    

