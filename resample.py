import read_data as rd
import pymeshlab
from pathlib import Path
import os
import math

resampled_database_dir = Path("resampled_data")
original_database_dir = Path("data")

# Steps 2.3 and 2.4
def print_classname(classname):
    classname = str(classname)
    width=50
    side_length = (width - 2 - len(classname))//2
    width = width - (len(classname)%2)
    print()
    print("-"*width)
    print("-"*side_length,classname.upper(),"-"*side_length)
    print("-"*width)
    print()

def resample_all(target=7500, margin=2500, logs=1, class_name=None):
    resample_logs = True if logs >= 2 else False

    if class_name:
        classdirs = [Path(original_database_dir) / class_name]
    else:
        classdirs = rd.get_classdirs(original_database_dir)

    # Collect all objects to process
    all_objs = []
    for classdir in classdirs:
        for obj in rd.get_objects(classdir):
            if obj.name.endswith(".obj"):
                all_objs.append((classdir, obj))

    try:
        from tqdm import tqdm
        iterator = tqdm(all_objs, desc="Resampling meshes")
    except ImportError:
        iterator = all_objs

    for classdir, obj in iterator:
        print_classname(classdir.name) if logs>=1 else None
        obj_string = str(obj)
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(obj_string)

        n_vertices = ms.current_mesh().vertex_number()
        n_gap = n_vertices - target

        if logs>=2:print(f"# vertices: {n_vertices}. ", end ="")
        if n_gap > margin:
            if logs>=2:print(f"simplifying {obj.name}")
            simplify(obj,logs=resample_logs)
        elif n_gap < (-margin):
            if logs>=2:print(f"refining {obj.name}")
            refine(obj,logs=resample_logs)
        else:
            if logs>=2: print(f"not altering {obj.name}")
            save_path = Path(resampled_database_dir) / classdir.name / obj.name
            save_path.parent.mkdir(parents=True, exist_ok=True)
            ms.save_current_mesh(str(save_path))


def clean_mesh(ms):
    ms.apply_filter("meshing_remove_duplicate_vertices")
    ms.apply_filter("meshing_remove_duplicate_faces")
    ms.apply_filter("meshing_remove_unreferenced_vertices")
    ms.apply_filter("meshing_remove_null_faces")
    ms.apply_filter("meshing_repair_non_manifold_edges")
    ms.apply_filter("meshing_repair_non_manifold_vertices")
    ms.apply_filter("meshing_close_holes")
    return ms


def refine(mesh_path, target=5000, logs=1, save_original=False):
    mesh_path = Path(mesh_path)
    mesh_path_string = str(mesh_path)
    ms = pymeshlab.MeshSet()
    
    # Load the mesh
    ms.load_new_mesh(mesh_path_string)
    
    # Extract information
    obj_name = mesh_path.name
    n_vertices = ms.current_mesh().vertex_number()
    class_name = os.path.basename(os.path.dirname(mesh_path))
    
    # Save original mesh if not already saved
    save_to_og = resampled_database_dir / class_name / f"{obj_name.split('.')[0]}_{n_vertices}.obj"
    if save_original:
        save_to_og.parent.mkdir(parents=True, exist_ok=True)
        ms.save_current_mesh(str(save_to_og))
    
    if logs >= 1:
        print(f"[INFO] Loaded: {mesh_path} (class={class_name})")
        print(f"[INFO] Old #vertices: {n_vertices}")
    
    ms = clean_mesh(ms)

    c = round(math.log(target/n_vertices, 2) - 1)

    if logs >= 2:
        print(f"    Formula check: n_v * 2^c = {n_vertices} * 2^{c} = {n_vertices*2**(c+1)} (target:{target})")
        print(f"    Initial c = {c}")

    margin = 2500
    diminished_return = 50
    last_n = ms.current_mesh().vertex_number()

    while ms.current_mesh().vertex_number() < target:
        ms.apply_filter(
            "meshing_isotropic_explicit_remeshing",
            iterations=c
        )
        c += 1

        if logs >= 2:
            print(f"    Iter step: #vertices={ms.current_mesh().vertex_number()}  -  c={c}")

        if abs(ms.current_mesh().vertex_number() - last_n) < diminished_return:
            if logs >= 1:
                print("[INFO] Refinement stopped: not improving anymore")
            break
        last_n = ms.current_mesh().vertex_number()
    
    # Extract info after refinement
    n_vertices_resampled = ms.current_mesh().vertex_number()
    new_obj_name = f"{obj_name.split('.')[0]}_{n_vertices_resampled}.obj"
    
    save_to = resampled_database_dir / class_name / new_obj_name
    save_to.parent.mkdir(parents=True, exist_ok=True)

    if logs >= 1:
        print(f"[INFO] Saving to: {save_to}")
        print(f"[INFO] New #vertices: {n_vertices_resampled}")
    
    ms.save_current_mesh(str(save_to))

    if logs >= 1:
        print("-" * 40)  # separator between meshes


def simplify(mesh_path, target=10_000, percentage_value=0.1, logs=1, save_original=False):
    mesh_path = Path(mesh_path)
    mesh_path_string = str(mesh_path)
    ms = pymeshlab.MeshSet()
     
    # Load the given mesh_path
    ms.load_new_mesh(mesh_path_string)

    # Extract information about the original object
    obj_name = mesh_path.name
    n_vertices = ms.current_mesh().vertex_number()
    n_faces = ms.current_mesh().face_number()
    class_name = os.path.basename(os.path.dirname(mesh_path))

    # Save the original file if requested
    save_to_og = resampled_database_dir / class_name / f"{obj_name.split('.')[0]}_{n_vertices}.obj"
    if save_original:
        save_to_og.parent.mkdir(parents=True, exist_ok=True)
        ms.save_current_mesh(str(save_to_og))

    if logs >= 1:
        print(f"[INFO] Loaded: {mesh_path} (class={class_name})")
        print(f"[INFO] Old #vertices: {n_vertices}")

    # Clean the mesh
    ms = clean_mesh(ms)

    # Keep a copy of the original (index 1) and work on index 0
    ms.add_mesh(ms.current_mesh(), 'original_mesh_copy')
    ms.set_current_mesh(0)

    margin = 2500

    while ms.current_mesh().vertex_number() > target:
        ms.apply_filter(
            "meshing_decimation_clustering",
            threshold=pymeshlab.PercentageValue(percentage_value)
        )
        percentage_value += 0.1

        if logs >= 2:
            print(f"    Iter step: #vertices={ms.current_mesh().vertex_number()}")

    # Extract info about new object
    n_vertices_resampled = ms.current_mesh().vertex_number()
    n_faces_resampled = ms.current_mesh().face_number()
    new_obj_name = f"{obj_name.split('.')[0]}_{n_vertices_resampled}.obj"

    save_to = resampled_database_dir / class_name / new_obj_name
    save_to_string = str(save_to)

    save_to.parent.mkdir(parents=True, exist_ok=True)
    ms.save_current_mesh(save_to_string)

    if logs >= 1:
        print(f"[INFO] Saving to: {save_to}")
        print(f"[INFO] New #vertices: {n_vertices_resampled}")
        print("-" * 40)  # separator

    return {
        "file": str(save_to),
        "class": class_name,
        "old_vertices": n_vertices,
        "new_vertices": n_vertices_resampled,
        "old_faces": n_faces,
        "new_faces": n_faces_resampled
    }


if __name__=="__main__":
    resample_all()