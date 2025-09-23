import read_data as rd
import pymeshlab
from pathlib import Path
import os
import math

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


resampled_database_dir = Path("data") / "ShapeDatabase_resampled"
original_database_dir = Path("data") / "ShapeDatabase_INFOMR"

def resample_all(target=7500,margin=2500,logs=1):
    resample_logs = True if logs>=2 else False

    for classdir in rd.get_classdirs(original_database_dir):
        print_classname(classdir.name)
        for obj in rd.get_objects(classdir):
            if not obj.name.endswith(".obj"): continue
            obj_string = str(obj)
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(obj_string)

            n_vertices = ms.current_mesh().vertex_number()
            
            n_gap = n_vertices - target

            if logs>=1:print(f"# vertices: {n_vertices}. ", end ="")
            if n_gap > margin:
                if logs>=1:print(f"simplifying {obj.name}")
                simplify(obj,logs=resample_logs)
            elif n_gap < (-margin):
                if logs>=1:print(f"refining {obj.name}")
                refine(obj,logs=resample_logs)
            else:
                if logs>=1: print(f"not altering {obj.name}")
                ms.save_current_mesh(str(resampled_database_dir / classdir.name / obj.name))


def clean_mesh(ms):
    ms.apply_filter("meshing_remove_duplicate_vertices")
    ms.apply_filter("meshing_remove_duplicate_faces")
    ms.apply_filter("meshing_remove_unreferenced_vertices")
    ms.apply_filter("meshing_remove_null_faces")
    ms.apply_filter("meshing_repair_non_manifold_edges")
    ms.apply_filter("meshing_repair_non_manifold_vertices")
    ms.apply_filter("meshing_close_holes")

    return ms




def refine(mesh_path, target=5000, logs=True,save_original=False):
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
    

    if logs:
        print(f"loaded {mesh_path} from {class_name}")
        print(f"old # vertices: {n_vertices}")
    
    ms = clean_mesh(ms)

    c = round(math.log(target/n_vertices, 2) - 1)

    if logs:
        print(f"n_v * 2^c = target = {n_vertices} * 2^{c} = {n_vertices*2**(c+1)} (target:{target})")
        print(f"c = {c}")

    #ms.add_mesh(ms.current_mesh(), 'original_mesh_coy')
    #ms.set_current_mesh(0)

    margin = 2500
    diminished_return = 50
    last_n = ms.current_mesh().vertex_number()

    while ms.current_mesh().vertex_number() < target:
        # quite nice
        ms.apply_filter(
            "meshing_isotropic_explicit_remeshing",
            iterations=c
        )
        c+=1

        if logs:
            print(f"# vertices: {ms.current_mesh().vertex_number()}  -  c={c}")

        if abs(ms.current_mesh().vertex_number() - last_n) < diminished_return:
            if logs: print("not improving anymore")
            break
        last_n = ms.current_mesh().vertex_number()
    
    # Extract info after refinement
    n_vertices_resampled = ms.current_mesh().vertex_number()
    new_obj_name = f"{obj_name.split('.')[0]}_{n_vertices_resampled}.obj"
    
    
    save_to = resampled_database_dir / class_name / new_obj_name
    save_to.parent.mkdir(parents=True, exist_ok=True)
    
    if logs:
        print(f"saving {save_to}")
        print(f"new # vertices: {n_vertices_resampled}")
    ms.save_current_mesh(str(save_to))
    print("new # vertices:",n_vertices_resampled)
    print()



def simplify(mesh_path,target=10_000,percentage_value=0.1,logs=True,save_original=False):
    # make a nice path
    mesh_path = Path(mesh_path)
    mesh_path_string = str(mesh_path)
    ms = pymeshlab.MeshSet()
     
    # load the given mesh_path
    ms.load_new_mesh(mesh_path_string)

    # extract information about the original object
    obj_name = mesh_path.name
    n_vertices = ms.current_mesh().vertex_number()
    n_faces = ms.current_mesh().face_number()
    class_name = os.path.basename(os.path.dirname(mesh_path))
    # save the original file to the new location
    save_to_og = resampled_database_dir / class_name / f"{obj_name.split('.')[0]}_{n_vertices}.obj"

    if save_original:
        save_to_og.parent.mkdir(parents=True, exist_ok=True)
        ms.save_current_mesh(str(save_to_og))

    if logs:
        print(f"loaded {mesh_path} from {class_name}")
        print(f"old # vertices: {n_vertices}")
        #print("old # faces:", n_faces)

    # clean the mesh
    ms = clean_mesh(ms)


    ms.add_mesh(ms.current_mesh(), 'original_mesh_copy')

    # Apply a filter to the first mesh
    ms.set_current_mesh(0)

    margin = 2500

    while ms.current_mesh().vertex_number() > target:
        ms.apply_filter(
                "meshing_decimation_clustering",
                threshold=pymeshlab.PercentageValue(percentage_value)
            )
        percentage_value += 0.1
        
        if logs:
            print(f"# vertices: {ms.current_mesh().vertex_number()}")



    
    # extract info about new object
    n_vertices_resampled = ms.current_mesh().vertex_number()
    n_faces_resampled = ms.current_mesh().face_number()
    new_obj_name = f"{obj_name.split(".")[0]}_{str(n_vertices_resampled)}.obj"

    save_to = resampled_database_dir / class_name / new_obj_name
    save_to_string = str(save_to)

    if logs:
        print(f"saving {save_to}")
        print(f"new # vertices: {n_vertices_resampled}")
        #print("new # faces:", n_faces_resampled)

    save_to.parent.mkdir(parents=True, exist_ok=True)
    ms.save_current_mesh(save_to_string)
    print("new # vertices:",n_vertices_resampled)
    print()



if __name__=="__main__":
    resample_all()
