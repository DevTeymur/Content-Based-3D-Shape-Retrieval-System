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


def refine(mesh_path, target=7500, margin=1000, logs=1, save_original=False):
    mesh_path = Path(mesh_path)
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(mesh_path))

    obj_name = mesh_path.name
    class_name = os.path.basename(os.path.dirname(mesh_path))
    n_vertices = ms.current_mesh().vertex_number()

    # Save original mesh if requested
    if save_original:
        save_to_og = resampled_database_dir / class_name / f"{obj_name.split('.')[0]}_{n_vertices}.obj"
        save_to_og.parent.mkdir(parents=True, exist_ok=True)
        ms.save_current_mesh(str(save_to_og))

    if logs >= 1:
        print(f"[INFO] Loaded: {mesh_path} (class={class_name})")
        print(f"[INFO] Old #vertices: {n_vertices}")

    ms = clean_mesh(ms)

    # Initial iteration factor, roughly proportional to log2(target/current)
    c = max(1, round(math.log(target/n_vertices, 2) - 1))

    diminished_return = 50
    last_n = n_vertices

    while ms.current_mesh().vertex_number() < target:
        current_vertices = ms.current_mesh().vertex_number()
        # Adaptive iterations: smaller meshes use fewer iterations
        factor = max(1, int((target - current_vertices) / 1000))
        ms.apply_filter("meshing_isotropic_explicit_remeshing", iterations=factor)
        
        if logs >= 2:
            print(f"    Iter step: #vertices={ms.current_mesh().vertex_number()} - iterations={factor}")

        # Stop if increase is too small
        if abs(ms.current_mesh().vertex_number() - last_n) < diminished_return:
            if logs >= 1:
                print("[INFO] Refinement stopped: not improving anymore")
            break

        last_n = ms.current_mesh().vertex_number()

        # Stop if we exceed target + margin
        if last_n >= target + margin:
            if logs >= 1:
                print("[INFO] Refinement stopped: reached target + margin")
            break

    # Save result
    n_vertices_resampled = ms.current_mesh().vertex_number()
    new_obj_name = f"{obj_name.split('.')[0]}_{n_vertices_resampled}.obj"
    save_to = resampled_database_dir / class_name / new_obj_name
    save_to.parent.mkdir(parents=True, exist_ok=True)
    ms.save_current_mesh(str(save_to))

    if logs >= 1:
        print(f"[INFO] Saving to: {save_to}")
        print(f"[INFO] New #vertices: {n_vertices_resampled}")
        print("-" * 40)

    return {
        "file": str(save_to),
        "class": class_name,
        "old_vertices": n_vertices,
        "new_vertices": n_vertices_resampled
    }



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
        new_n = ms.current_mesh().vertex_number()
        if logs >= 2:
            print(f"    Iter step: #vertices={new_n}")

        if new_n <= target:
            break  # Stop if we reach target

        percentage_value = min(percentage_value + 0.05, 0.9)  # prevent too high

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


def resample_one(file_path, target=7500, margin=2500, logs=1):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(file_path))
    n_vertices = ms.current_mesh().vertex_number()
    n_gap = n_vertices - target

    if logs >= 2:
        print(f"# vertices: {n_vertices}. ", end="")
    if n_gap > margin:
        if logs >= 2: print(f"simplifying {Path(file_path).name}")
        return simplify(file_path, logs=logs)
    elif n_gap < (-margin):
        if logs >= 2: print(f"refining {Path(file_path).name}")
        return refine(file_path, logs=logs)
    else:
        if logs >= 2: print(f"not altering {Path(file_path).name}")
        class_name = os.path.basename(os.path.dirname(file_path))
        save_path = resampled_database_dir / class_name / Path(file_path).name
        save_path.parent.mkdir(parents=True, exist_ok=True)
        ms.save_current_mesh(str(save_path))
        return {"file": str(save_path), "class": class_name, "old_vertices": n_vertices, "new_vertices": n_vertices, "old_faces": ms.current_mesh().face_number(), "new_faces": ms.current_mesh().face_number()}


if __name__ == "__main__":
    from read_data import get_random_data_from_directory
    result = resample_one(get_random_data_from_directory())
    # from get_stats import get_object_info

    # resampled_file = result["file"]
    # print(get_object_info(resampled_file))