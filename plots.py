from pathlib import Path
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


def plot_mesh_comparison(mesh1_path, mesh2_path=None, labels=None, title=None, save_path=None, show_stats=True, interactive=False):
    """
    Plot mesh(es) for comparison. Can show single mesh or side-by-side comparison.
    
    Args:
        mesh1_path: Path to first mesh file
        mesh2_path: Path to second mesh file (optional, if None shows only first mesh)
        labels: List of labels for the meshes (e.g., ['Original', 'Resampled'])
        title: Overall title for the plot
        save_path: Path to save the plot image (optional)
        show_stats: Whether to print mesh statistics
        interactive: Whether to show interactive Open3D visualization after matplotlib
    """
    from read_data import read_data
    from plots import show_mesh_simple
    import matplotlib.pyplot as plt
    from pathlib import Path
    import numpy as np
    
    # Load first mesh
    mesh1 = read_data(mesh1_path)
    mesh1_name = Path(mesh1_path).stem
    
    # Get mesh1 statistics
    vertices1 = np.asarray(mesh1.vertices)
    triangles1 = np.asarray(mesh1.triangles)
    n_vertices1 = len(vertices1)
    n_faces1 = len(triangles1)
    
    # Set default labels
    if labels is None:
        labels = [mesh1_name]
        if mesh2_path:
            labels.append(Path(mesh2_path).stem)
    
    # Print statistics
    if show_stats:
        print("=" * 60)
        print(f"MESH VISUALIZATION: {labels[0]}")
        if mesh2_path:
            print(f" vs {labels[1]}")
        print("=" * 60)
        print(f"{labels[0]}: {n_vertices1:,} vertices, {n_faces1:,} faces")
    
    # Single mesh plot
    if mesh2_path is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot mesh
        ax.plot_trisurf(vertices1[:, 0], vertices1[:, 1], vertices1[:, 2], 
                       triangles=triangles1, alpha=0.8, cmap='viridis')
        
        ax.set_title(f'{labels[0]}\n{n_vertices1:,} vertices, {n_faces1:,} faces', fontsize=14)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elev=20, azim=45)
        
        if title:
            fig.suptitle(title, fontsize=16)
        
    # Side-by-side comparison
    else:
        # Load second mesh
        mesh2 = read_data(mesh2_path)
        vertices2 = np.asarray(mesh2.vertices)
        triangles2 = np.asarray(mesh2.triangles)
        n_vertices2 = len(vertices2)
        n_faces2 = len(triangles2)
        
        if show_stats:
            print(f"{labels[1]}: {n_vertices2:,} vertices, {n_faces2:,} faces")
            if n_vertices1 != n_vertices2:
                change = n_vertices2 - n_vertices1
                percentage = (n_vertices2 / n_vertices1) * 100
                print(f"Change: {change:+,} vertices ({percentage:.1f}%)")
            print("=" * 60)
        
        # Create side-by-side plot
        fig = plt.figure(figsize=(16, 8))
        
        # First mesh
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.plot_trisurf(vertices1[:, 0], vertices1[:, 1], vertices1[:, 2], 
                        triangles=triangles1, alpha=0.8, cmap='viridis')
        ax1.set_title(f'{labels[0]}\n{n_vertices1:,} vertices, {n_faces1:,} faces', fontsize=12)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.view_init(elev=20, azim=45)
        
        # Second mesh
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.plot_trisurf(vertices2[:, 0], vertices2[:, 1], vertices2[:, 2], 
                        triangles=triangles2, alpha=0.8, cmap='viridis')
        ax2.set_title(f'{labels[1]}\n{n_vertices2:,} vertices, {n_faces2:,} faces', fontsize=12)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.view_init(elev=20, azim=45)
        
        # Overall title
        if title is None:
            title = f'Mesh Comparison: {labels[0]} vs {labels[1]}'
        fig.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()
    
    # Interactive visualization if requested
    if interactive:
        print("\nShowing interactive Open3D visualizations...")
        print("Close the window to continue.")
        
        print(f"Showing {labels[0]}...")
        show_mesh_simple(mesh1)
        
        if mesh2_path:
            print(f"Showing {labels[1]}...")
            mesh2 = read_data(mesh2_path)
            show_mesh_simple(mesh2)

# Convenience wrapper functions for common use cases
def plot_single_mesh(mesh_path, title=None, save_path=None, interactive=False):
    """Plot a single mesh"""
    plot_mesh_comparison(mesh_path, title=title, save_path=save_path, interactive=interactive)

def plot_original_vs_resampled(original_path, resampled_path=None, class_name=None, save_path=None, interactive=False):
    """Plot original vs resampled mesh comparison"""
    # Auto-find resampled file if not provided
    if resampled_path is None and class_name:
        obj_name = Path(original_path).stem
        resampled_dir = Path("resampled_data") / class_name
        resampled_files = list(resampled_dir.glob(f"{obj_name}_*.obj"))
        if resampled_files:
            resampled_path = str(resampled_files[0])
    
    if resampled_path:
        plot_mesh_comparison(
            original_path, 
            resampled_path, 
            labels=['Original', 'Resampled'],
            title=f'Original vs Resampled: {Path(original_path).stem}',
            save_path=save_path,
            interactive=interactive
        )
    else:
        print(f"No resampled file found for {original_path}")



if __name__ == "__main__":
    from read_data import get_random_data_from_directory
    path = get_random_data_from_directory(parent_directory="normalized_data")
    # Single mesh
    plot_single_mesh(path, title=" ")

    # Original vs Resampled (auto-find resampled file)
    # plot_original_vs_resampled("data/Bed/D00031.obj", class_name="Bed")

    # Custom comparison
    # plot_mesh_comparison(
    #     "data/Bed/D00031.obj", 
    #     "resampled_data/Bed/D00031_7500.obj",
    #     labels=['Before', 'After'],
    #     title="Resampling Result",
    #     save_path="img/bed_comparison.png",
    #     interactive=True
    # )

    # Just one mesh with interactive view
    # plot_mesh_comparison(path, interactive=True)


# if __name__=="__main__":
#     # visualize_normalized_shape("normalized_data/Insect/D00291_5256.obj", axes_size=.5)
#     df = pd.read_csv("stats/original_stats.csv")
#     df = df.drop(columns=["file"])
#     df = df.rename(columns={"num_vertices": "vertices", "num_faces": "faces"})
#     df = df[["vertices", "faces"]]
#     df = df.astype(float)
#     plot_boxplots(df)