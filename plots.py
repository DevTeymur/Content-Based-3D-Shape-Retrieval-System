import open3d as o3d
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Step 1
def show_mesh_simple(mesh):
    o3d.visualization.draw_geometries(
    [mesh],
    window_name="3D Shape Viewer",
    width=1500,
    height=1500,
    mesh_show_back_face=True  # show both sides of faces
    )

# Step 2.5
def visualize_normalized_shape(mesh_path, show_plot=True, axes_size=0.5, show_bbox=True):
    """
    Visualize a normalized 3D shape with optional coordinate axes and bounding box.

    Parameters:
    -----------
    mesh_path : str or Path
        Path to the normalized mesh file (.obj, .ply, etc.).
    show_plot : bool, optional
        If True, displays the visualization. Default is True.
    axes_size : float, optional
        Size of the coordinate axes. Default is 0.5.
    show_bbox : bool, optional
        If True, draws the axis-aligned bounding box around the shape. Default is True.
    """
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    mesh.compute_vertex_normals()
    
    geometries = [mesh]

    if axes_size > 0:
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axes_size, origin=[0, 0, 0])
        geometries.append(axes)

    if show_bbox:
        bbox = mesh.get_axis_aligned_bounding_box()
        bbox.color = (1, 0, 0)  # Red bounding box
        geometries.append(bbox)

    if show_plot:
        o3d.visualization.draw_geometries(
            geometries,
            window_name="Normalized Shape Viewer",
            width=1000,
            height=800,
            mesh_show_back_face=True
        )

    # Return geometries for optional further processing
    return geometries


# Step 2.2
def avg_shape_plot(df, avg_vertices, avg_faces):
    # === 5. Average shape + outliers visualization ===
    # Find the shape closest to the average
    df["dist_to_avg"] = np.sqrt(
        (df["num_vertices"] - avg_vertices)**2 +
        (df["num_faces"] - avg_faces)**2
    )
    avg_shape_row = df.loc[df["dist_to_avg"].idxmin()]
    avg_shape_path = avg_shape_row["file"]  # assuming "path" column in CSV

    # Load and visualize with Open3D
    mesh_avg = o3d.io.read_triangle_mesh(avg_shape_path)
    mesh_avg.compute_vertex_normals()
    print("Showing average shape...")
    o3d.visualization.draw_geometries([mesh_avg])

# Step 2.2
def plot_outlier_shapes(df):
    # Outlier with fewest vertices
    min_shape = df.loc[df["num_vertices"].idxmin()]

    # Outlier with most vertices
    max_shape = df.loc[df["num_vertices"].idxmax()]

    for shape in [min_shape, max_shape]:
        mesh_out = o3d.io.read_triangle_mesh(shape["file"])
        mesh_out.compute_vertex_normals()
        print(f"Showing outlier: {shape['file']}")
        o3d.visualization.draw_geometries([mesh_out])

# Step 2.2
def plot_boxplots(df):
    df[["vertices", "faces"]].plot(kind="box", subplots=True, layout=(1,2), figsize=(10,5))
    plt.suptitle("Boxplots with Outliers")
    plt.show()

# Step 2.2
def plot_histograms(df, step="2"):
    plt.figure()
    df["num_vertices"].hist(bins=30)
    plt.xlabel("Number of vertices")
    plt.ylabel("Number of shapes")
    plt.title("Histogram of vertices")
    plt.savefig(f"img/hist_vertices_step_{step}.png")
    plt.show()

    plt.figure()
    df["num_faces"].hist(bins=30)
    plt.xlabel("Number of faces")
    plt.ylabel("Number of shapes")
    plt.title("Histogram of faces")
    plt.savefig(f"img/hist_faces_step_{step}.png")
    plt.show()

    if False:
        plt.figure(figsize=(12,6))
        df["class"].value_counts().plot(kind="bar")
        plt.xlabel("Shape class")
        plt.ylabel("Count")
        plt.title("Class distribution")
        plt.savefig(f"img/class_distribution_step_{step}.png")
        plt.show()



# Class not working for now
import open3d as o3d
import open3d.visualization.gui as gui  # type: ignore
import open3d.visualization.rendering as rendering # type: ignore

class NormalizedShapeViewer:
    def __init__(self, mesh_path, axes_size=0.5):
        self.mesh_path = mesh_path
        self.axes_size = axes_size
        self.show_bbox = True

        # Load mesh and normals
        self.mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        self.mesh.compute_vertex_normals()

        # Create window
        self.window = gui.Application.instance.create_window(
            "Normalized Shape Viewer", 1000, 800
        )
        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.window.renderer)
        self.window.add_child(self.scene)

        # Create materials
        self.mesh_material = rendering.MaterialRecord()
        self.mesh_material.shader = "defaultLit"

        # Add mesh
        self.scene.scene.add_geometry("mesh", self.mesh, self.mesh_material)

        # Add axes
        self.axes = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=self.axes_size, origin=[0, 0, 0]
        )
        self.scene.scene.add_geometry("axes", self.axes, self.mesh_material)

        # Add bounding box
        self.bbox = self.mesh.get_axis_aligned_bounding_box()
        self.bbox.color = (1, 0, 0)
        self.scene.scene.add_geometry("bbox", self.bbox, self.mesh_material)

        # Button panel
        em = gui.Vert(0.5)
        self.button = gui.Button("Toggle Bounding Box")
        self.button.horizontal_padding_em = 0.5
        self.button.vertical_padding_em = 0.5
        self.button.set_on_clicked(self.toggle_bbox)
        em.add_child(self.button)
        self.window.add_child(em)

        # Set camera
        bounds = self.mesh.get_axis_aligned_bounding_box()
        self.scene.setup_camera(60, bounds, bounds.get_center())

    def toggle_bbox(self):
        self.show_bbox = not self.show_bbox
        if self.show_bbox:
            self.scene.scene.add_geometry("bbox", self.bbox, self.mesh_material)
        else:
            self.scene.scene.remove_geometry("bbox")


if __name__=="__main__":
    # visualize_normalized_shape("normalized_data/Insect/D00291_5256.obj", axes_size=.5)
    df = pd.read_csv("stats/original_stats.csv")
    df = df.drop(columns=["file"])
    df = df.rename(columns={"num_vertices": "vertices", "num_faces": "faces"})
    df = df[["vertices", "faces"]]
    df = df.astype(float)
    plot_boxplots(df)


