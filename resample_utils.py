import pymeshlab
from pathlib import Path
import os
import math

resampled_database_dir = Path("resampled_data")

# ---------- Helper: clean mesh ----------
def clean_mesh(ms: pymeshlab.MeshSet, logs=1):
    """Cleans a mesh: remove duplicates, repair, close holes"""
    ms.apply_filter("meshing_remove_duplicate_vertices")
    ms.apply_filter("meshing_remove_duplicate_faces")
    ms.apply_filter("meshing_remove_unreferenced_vertices")
    ms.apply_filter("meshing_remove_null_faces")
    ms.apply_filter("meshing_repair_non_manifold_edges")
    ms.apply_filter("meshing_repair_non_manifold_vertices")
    ms.apply_filter("meshing_close_holes")
    if logs >= 2:
        print("[INFO] Mesh cleaned")
    return ms

# ---------- Helper: refine mesh ----------
def refine_mesh(mesh_path: str, target=7500, logs=1):
    """
    Refines small meshes by subdividing faces until target number of vertices is reached.
    Returns info dictionary.
    """
    mesh_path = Path(mesh_path)
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(mesh_path))
    old_vertices = ms.current_mesh().vertex_number()
    old_faces = ms.current_mesh().face_number() 
    class_name = mesh_path.parent.name
    obj_name = mesh_path.stem

    ms = clean_mesh(ms, logs)

    iteration = 0
    while ms.current_mesh().vertex_number() < target:
        ms.apply_filter("meshing_isotropic_explicit_remeshing", iterations=1)
        iteration += 1
        if logs >= 2:
            print(f"Iteration {iteration}: {ms.current_mesh().vertex_number()} vertices")
        # safety check to prevent infinite loop
        if iteration > 15:  
            if logs >= 1:
                print("[WARN] Refinement stopped: max iterations reached")
            break

    new_vertices = ms.current_mesh().vertex_number()
    new_faces = ms.current_mesh().face_number()  
    save_path = resampled_database_dir / class_name / f"{obj_name}_{new_vertices}.obj"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    ms.save_current_mesh(str(save_path))

    if logs >= 1:
        print(f"[INFO] Refined {obj_name}: {old_vertices} -> {new_vertices} vertices")

    return {
        "file": str(save_path),
        "class": class_name,
        "old_vertices": old_vertices,
        "new_vertices": new_vertices,
        "old_faces": old_faces,     
        "new_faces": new_faces       
    }

# ---------- Helper: simplify mesh ----------
def simplify_mesh(mesh_path: str, target=7500, logs=1):
    """
    Simplifies large meshes using clustering decimation until target number of vertices is reached.
    Returns info dictionary.
    """
    mesh_path = Path(mesh_path)
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(mesh_path))
    old_vertices = ms.current_mesh().vertex_number()
    old_faces = ms.current_mesh().face_number()  
    class_name = mesh_path.parent.name
    obj_name = mesh_path.stem

    ms = clean_mesh(ms, logs)

    iteration = 0
    percentage_value = 0.1
    while ms.current_mesh().vertex_number() > target:
        ms.apply_filter("meshing_decimation_clustering",
                        threshold=pymeshlab.PercentageValue(percentage_value))
        iteration += 1
        if logs >= 2:
            print(f"Iteration {iteration}: {ms.current_mesh().vertex_number()} vertices")
        percentage_value += 0.05
        if iteration > 15:  
            if logs >= 1:
                print("[WARN] Simplification stopped: max iterations reached")
            break

    new_vertices = ms.current_mesh().vertex_number()
    new_faces = ms.current_mesh().face_number()  
    save_path = resampled_database_dir / class_name / f"{obj_name}_{new_vertices}.obj"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    ms.save_current_mesh(str(save_path))

    if logs >= 1:
        print(f"[INFO] Simplified {obj_name}: {old_vertices} -> {new_vertices} vertices")

    return {
        "file": str(save_path),
        "class": class_name,
        "old_vertices": old_vertices,
        "new_vertices": new_vertices,
        "old_faces": old_faces,      
        "new_faces": new_faces   
    }


# ---------- Resample one mesh ----------
def resample_one(mesh_path: str, target=7500, margin=2500, logs=1):
    """
    Resample a single mesh:
    - If vertices < target - margin: refine
    - If vertices > target + margin: simplify
    - Else: save as-is in resampled_data
    Returns info dictionary.
    """
    mesh_path = Path(mesh_path)
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(mesh_path))
    n_vertices = ms.current_mesh().vertex_number()
    class_name = mesh_path.parent.name
    obj_name = mesh_path.stem

    if n_vertices < (target - margin):
        if logs >= 1:
            print(f"[INFO] Refining {obj_name} ({n_vertices} vertices < {target - margin})")
        return refine_mesh(mesh_path, target=target, logs=logs)

    elif n_vertices > (target + margin):
        if logs >= 1:
            print(f"[INFO] Simplifying {obj_name} ({n_vertices} vertices > {target + margin})")
        return simplify_mesh(mesh_path, target=target, logs=logs)

    else:
        # within acceptable range, save as-is
        old_faces = ms.current_mesh().face_number()  
        save_path = resampled_database_dir / class_name / f"{obj_name}_{n_vertices}.obj"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        ms.save_current_mesh(str(save_path))
        if logs >= 1:
            print(f"[INFO] Mesh {obj_name} is within target range ({n_vertices} vertices). Saved as-is.")
        return {
            "file": str(save_path),
            "class": class_name,
            "old_vertices": n_vertices,
            "new_vertices": n_vertices,
            "old_faces": old_faces,      
            "new_faces": old_faces       
        }

# ---------- Resample all meshes in a dataset ----------
def resample_all(database_dir: str, target=7500, margin=2500, logs=1):
    """
    Iterate over all meshes in the database and resample them using resample_one().
    Saves resampled meshes to resampled_data/category/
    Returns a list of dictionaries with resampling info.
    """
    database_dir = Path(database_dir)
    all_mesh_files = []
    for class_dir in database_dir.iterdir():
        if class_dir.is_dir():
            for mesh_file in class_dir.iterdir():
                if mesh_file.suffix.lower() in [".obj", ".ply", ".off"]:
                    all_mesh_files.append(mesh_file)

    # Progress bar
    try:
        from tqdm import tqdm
        iterator = tqdm(all_mesh_files, desc="Resampling meshes")
    except ImportError:
        iterator = all_mesh_files

    all_results = []
    for mesh_file in iterator:
        result = resample_one(mesh_file, target=target, margin=margin, logs=logs)
        all_results.append(result)

    return all_results


# if __name__ == "__main__":
#     database_path = "data"  # path to original database
#     results = resample_all(database_path, target=7500, margin=2500, logs=0)

#     # optional: print summary
#     print("\nResampling complete. Summary of results:")
#     for r in results[:5]:  # show first 5 for brevity
#         print(r)

if __name__ == "__main__":
    from read_data import get_random_data_from_directory, read_data
    random_mesh = get_random_data_from_directory()
    random_mesh = "data/Bookset/D01096.obj"
    print(f"Selected random mesh: {random_mesh}")

    # Call resample_one to test
    result = resample_one(random_mesh, target=7500, margin=2500, logs=1)

    print("\nResample result:")
    print(result)
