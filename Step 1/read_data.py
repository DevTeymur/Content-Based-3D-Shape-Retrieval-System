import open3d as o3d

file_path = "data/Bicycle/D00033.obj"

mesh = o3d.io.read_triangle_mesh(file_path)


mesh.compute_vertex_normals()

# Create visualizer
o3d.visualization.draw_geometries(
    [mesh],
    window_name="3D Shape Viewer",
    width=1500,
    height=1500,
    mesh_show_back_face=True  # show both sides of faces
)

