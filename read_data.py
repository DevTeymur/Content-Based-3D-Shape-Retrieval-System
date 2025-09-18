import open3d as o3d
import os
import random


def get_random_data_from_directory(directory_name=None, logs=True):
    if directory_name is None:
        # Pick a random folder from data/
        directory_name = random.choice([
            os.path.join("data", d)
            for d in os.listdir("data")
            if os.path.isdir(os.path.join("data", d))
        ])
        print(f"Selected random directory: {directory_name}") if logs else None
    
    if not os.path.exists(directory_name):
        raise ValueError(f"Directory {directory_name} does not exist.")
    
    # Pick a random file from the chosen directory
    files = [f for f in os.listdir(directory_name) if f.endswith(('.obj', '.ply', '.stl'))]
    if not files:
        raise ValueError(f"No .obj, .ply, or .stl files found in {directory_name}.")
    
    return os.path.join(directory_name, random.choice(files))


def read_data(file_path):
    if file_path.endswith('.obj') or file_path.endswith('.ply') or file_path.endswith('.stl'):
        mesh = o3d.io.read_triangle_mesh(file_path).compute_vertex_normals()
        return mesh
    else:
        raise ValueError("Unsupported file format. Please provide a .obj, .ply, or .stl file.")


def show_mesh_simple(mesh):
    o3d.visualization.draw_geometries(
    [mesh],
    window_name="3D Shape Viewer",
    width=1500,
    height=1500,
    mesh_show_back_face=True  # show both sides of faces
    )

if __name__ == "__main__":
    for i in range(25):
        mesh = read_data(get_random_data_from_directory())
        print(mesh)
        print("__"*10)
    # show_mesh_simple(mesh)    

