import matplotlib.pyplot as plt
import numpy as np
import trimesh
from pathlib import Path
from read_data import read_data
import open3d as o3d

def visualize_shape_retrieval(query_mesh_path, results_df, max_display=8, 
                             figsize=(12, 9), save_path=None, show_scores=True):
    """
    Visualize query mesh + top similar results in adaptive grid layout.
    
    Args:
        query_mesh_path: Path to query mesh file
        results_df: DataFrame with retrieval results from search_similar_shapes()
        max_display: Maximum number of result meshes to display (excluding query)
        figsize: Figure size tuple (made smaller)
        save_path: Optional path to save the visualization
        show_scores: Whether to show similarity scores in titles
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    
    # Determine number of results to show
    num_results = min(len(results_df), max_display)
    total_plots = num_results + 1  # +1 for query mesh
    
    # Determine grid layout adaptively
    if total_plots <= 2:
        rows, cols = 1, 2
    elif total_plots <= 4:
        rows, cols = 2, 2
    elif total_plots <= 6:
        rows, cols = 2, 3
    elif total_plots <= 9:
        rows, cols = 3, 3
    else:
        rows, cols = 3, 4  # Maximum layout
    
    print(f"Creating {rows}x{cols} grid for {total_plots} meshes (1 query + {num_results} results)")
    
    # Create figure and subplots
    fig = plt.figure(figsize=figsize)
    fig.suptitle('3D Shape Retrieval Results', fontsize=14, fontweight='bold')
    
    # Plot query mesh (always first position) - Show class name in red
    ax_query = fig.add_subplot(rows, cols, 1, projection='3d')
    query_category = Path(query_mesh_path).parent.name
    plot_single_mesh_in_subplot(ax_query, query_mesh_path, 
                                title=query_category.upper(),  # Class name instead of "QUERY MESH"
                                subtitle="",  # No filename
                                highlight=True)
    
    # Plot result meshes
    for i, (_, result_row) in enumerate(results_df.head(num_results).iterrows()):
        subplot_idx = i + 2  # Start from position 2 (after query)
        ax = fig.add_subplot(rows, cols, subplot_idx, projection='3d')
        
        # Create title with rank and similarity info
        rank = i + 1
        category = result_row['category']
        
        if show_scores:
            distance = result_row['combined_distance']
            title = f"#{rank}. {category}"
            subtitle = f"Distance: {distance:.3f}"  # Only distance, no filename
        else:
            title = f"#{rank}. {category}"
            subtitle = ""  # No subtitle if not showing scores
        
        plot_single_mesh_in_subplot(ax, result_row['filepath'], 
                                    title=title, 
                                    subtitle=subtitle,
                                    highlight=False)
    
    # Hide unused subplots
    total_subplots = rows * cols
    for i in range(total_plots + 1, total_subplots + 1):
        ax = fig.add_subplot(rows, cols, i)
        ax.axis('off')
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.show()
    return fig

def plot_single_mesh_in_subplot(ax, mesh_path, title="", subtitle="", highlight=False):
    """
    Plot a single mesh in a matplotlib 3D subplot using SAME method as visualize_normalized_shape.
    """
    try:
        # Use SAME loading method as visualize_normalized_shape
        import trimesh
        mesh = trimesh.load(str(mesh_path))
        
        # Get vertices and faces (SAME as visualize_normalized_shape)
        vertices = mesh.vertices
        faces = mesh.faces
        
        # Create surface plot (SAME method)
        ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                       triangles=faces, 
                       alpha=0.9, 
                       cmap='plasma' if highlight else 'viridis',
                       linewidth=0,
                       shade=True)
        
        # CRITICAL: Set equal aspect ratio (SAME as visualize_normalized_shape)
        max_range = max(
            vertices[:, 0].max() - vertices[:, 0].min(),
            vertices[:, 1].max() - vertices[:, 1].min(),
            vertices[:, 2].max() - vertices[:, 2].min()
        ) / 2.0
        
        mid_x = (vertices[:, 0].max() + vertices[:, 0].min()) * 0.5
        mid_y = (vertices[:, 1].max() + vertices[:, 1].min()) * 0.5
        mid_z = (vertices[:, 2].max() + vertices[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # Styling
        if highlight:
            title_color = 'red'
            title_weight = 'bold'
        else:
            title_color = 'black'
            title_weight = 'normal'
        
        # Set title
        ax.set_title(title, fontsize=10, fontweight=title_weight, color=title_color, pad=10)
        
        # Set subtitle (only for non-query meshes and only distance)
        if subtitle and not highlight:
            ax.text2D(0.5, -0.05, subtitle, transform=ax.transAxes, 
                     ha='center', va='top', fontsize=8)
        
        # Remove axes and grid (SAME as visualize_normalized_shape)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_zlabel('')
        ax.grid(False)
        
        # Set consistent viewing angle
        ax.view_init(elev=20, azim=45)
        
        # Remove axis panes for cleaner look
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')
        
    except Exception as e:
        # If mesh loading fails, show error message
        ax.text(0.5, 0.5, 0.5, f'Error loading\nmesh\n{str(e)[:30]}...', 
               ha='center', va='center', transform=ax.transAxes, fontsize=8, color='red')
        ax.set_title(title, fontsize=10, color='red')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

def create_retrieval_summary(query_mesh_path, results_df, save_path=None):
    """
    Create a text summary of retrieval results.
    
    Args:
        query_mesh_path: Path to query mesh
        results_df: Retrieval results DataFrame
        save_path: Optional path to save summary
        
    Returns:
        str: Formatted summary text
    """
    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append("3D SHAPE RETRIEVAL SUMMARY")
    summary_lines.append("=" * 80)
    summary_lines.append(f"Query Mesh: {Path(query_mesh_path).name}")
    summary_lines.append(f"Query Category: {Path(query_mesh_path).parent.name}")
    summary_lines.append(f"Number of Results: {len(results_df)}")
    summary_lines.append("")
    summary_lines.append("Top Similar Shapes:")
    summary_lines.append("-" * 50)
    
    for i, (_, row) in enumerate(results_df.iterrows(), 1):
        mesh_name = Path(row['filepath']).stem
        category = row['category']
        distance = row['combined_distance']
        scalar_dist = row['scalar_distance']
        hist_dist = row['histogram_distance']
        
        summary_lines.append(f"{i:2d}. {mesh_name:<20} ({category:<12})")
        summary_lines.append(f"    Combined Distance: {distance:.4f}")
        summary_lines.append(f"    Scalar: {scalar_dist:.4f} | Histogram: {hist_dist:.4f}")
        summary_lines.append("")
    
    summary_text = "\n".join(summary_lines)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(summary_text)
        print(f"Summary saved to: {save_path}")
    
    return summary_text

def quick_retrieval_visualization(query_mesh_path, features_db_path="stats/features_database.csv", 
                                 k=5, scalar_weight=0.5, save_dir=None):
    """
    Convenient function to run retrieval + visualization in one call.
    
    Args:
        query_mesh_path: Path to query mesh
        features_db_path: Path to features database
        k: Number of results to retrieve
        scalar_weight: Weight for scalar vs histogram features
        save_dir: Directory to save results (optional)
        
    Returns:
        tuple: (results_df, figure)
    """
    # Import here to avoid circular imports
    from retrieval import run_retrieval_query
    
    print("Running shape retrieval...")
    results = run_retrieval_query(
        query_mesh_path=query_mesh_path,
        features_db_path=features_db_path,
        k=k,
        scalar_weight=scalar_weight,
        show_results=False
    )
    
    if results is None or len(results) == 0:
        print("No results found!")
        return None, None
    
    # Create visualization
    save_img_path = None
    save_summary_path = None
    
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        query_name = Path(query_mesh_path).stem
        save_img_path = save_dir / f"retrieval_{query_name}.png"
        save_summary_path = save_dir / f"summary_{query_name}.txt"
    
    print("Creating visualization...")
    fig = visualize_shape_retrieval(
        query_mesh_path=query_mesh_path,
        results_df=results,
        max_display=k,
        save_path=save_img_path
    )
    
    # Create summary
    summary = create_retrieval_summary(
        query_mesh_path=query_mesh_path,
        results_df=results,
        save_path=save_summary_path
    )
    
    print(summary)
    return results, fig

if __name__ == "__main__":
    # Test visualization with path checking
    print("Testing retrieval visualization...")
    
    # Check if the test query exists
    test_query = "normalized_data/Tool/m1106_8392.obj"
    
    if not Path(test_query).exists():
        print(f"Test file {test_query} not found!")
        # Try to find any mesh file
        normalized_dir = Path("normalized_data")
        if normalized_dir.exists():
            for category_dir in normalized_dir.iterdir():
                if category_dir.is_dir():
                    for mesh_file in category_dir.iterdir():
                        if mesh_file.suffix.lower() == '.obj':
                            test_query = str(mesh_file)
                            print(f"Using alternative test file: {test_query}")
                            break
                    if test_query != "normalized_data/Bed/D00031.obj":
                        break
    
    if Path(test_query).exists():
        print(f"Testing with: {test_query}")
        results, fig = quick_retrieval_visualization(
            query_mesh_path=test_query,
            k=6,
            scalar_weight=0.6,
            save_dir="retrieval_results"
        )
        
        if results is not None:
            print("Visualization working correctly!")
        else:
            print("Visualization test failed!")
    else:
        print("No test mesh file found!")