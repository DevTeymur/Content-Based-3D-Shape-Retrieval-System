import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


def analyze_resampled_stats(csv_path="stats/resampled_stats.csv", target=7500, margin=2500):
    """
    Analyze resampled statistics and check how many files are within threshold.
    Add flag column and plot histograms.
    """
    # Define thresholds
    # min_threshold = target - margin  # 5000
    # max_threshold = target + margin  # 10000
    min_threshold, max_threshold = 4500, 12000  # stricter lower bound
    
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Add threshold flag column
    df['within_threshold'] = (
        (df['num_vertices'] >= min_threshold) & 
        (df['num_vertices'] <= max_threshold)
    )
    
    # Print statistics
    total_files = len(df)
    within_threshold = df['within_threshold'].sum()
    below_threshold = (df['num_vertices'] < min_threshold).sum()
    above_threshold = (df['num_vertices'] > max_threshold).sum()
    
    print("=" * 60)
    print("RESAMPLED MESH ANALYSIS")
    print("=" * 60)
    print(f"Target range: {min_threshold:,} - {max_threshold:,} vertices")
    print(f"Total files analyzed: {total_files:,}")
    print("-" * 40)
    print(f"Within threshold ({min_threshold:,}-{max_threshold:,}): {within_threshold:,} ({within_threshold/total_files*100:.1f}%)")
    print(f"Below threshold (<{min_threshold:,}): {below_threshold:,} ({below_threshold/total_files*100:.1f}%)")
    print(f"Above threshold (>{max_threshold:,}): {above_threshold:,} ({above_threshold/total_files*100:.1f}%)")
    print("=" * 60)
    
    # Show some examples of problematic files
    if below_threshold > 0:
        print(f"\nFiles below threshold (showing first 10):")
        below_files = df[df['num_vertices'] < min_threshold].head(10)
        for _, row in below_files.iterrows():
            print(f"  {row['file']}: {row['num_vertices']} vertices")
    
    if above_threshold > 0:
        print(f"\nFiles above threshold (showing first 10):")
        above_files = df[df['num_vertices'] > max_threshold].head(10)
        for _, row in above_files.iterrows():
            print(f"  {row['file']}: {row['num_vertices']} vertices")
    
    # Save updated CSV with flag column
    output_csv = csv_path.replace('.csv', '_with_flags.csv')
    df.to_csv(output_csv, index=False)
    print(f"\nSaved updated CSV with flags to: {output_csv}")
    
    # Plot histograms - Before filtering
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Resampled Mesh Analysis', fontsize=16)
    
    # Histogram 1: All vertices (before filtering)
    axes[0, 0].hist(df['num_vertices'], bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].axvline(min_threshold, color='red', linestyle='--', label=f'Min threshold ({min_threshold:,})')
    axes[0, 0].axvline(max_threshold, color='red', linestyle='--', label=f'Max threshold ({max_threshold:,})')
    axes[0, 0].set_xlabel('Number of Vertices')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('All Resampled Meshes - Vertices')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Histogram 2: All faces (before filtering)
    axes[0, 1].hist(df['num_faces'], bins=50, alpha=0.7, color='green', edgecolor='black')
    axes[0, 1].set_xlabel('Number of Faces')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('All Resampled Meshes - Faces')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Filter data to only include meshes within threshold
    df_filtered = df[df['within_threshold']]
    
    # Histogram 3: Filtered vertices (within threshold only)
    if len(df_filtered) > 0:
        axes[1, 0].hist(df_filtered['num_vertices'], bins=30, alpha=0.7, color='blue', edgecolor='black')
        axes[1, 0].axvline(min_threshold, color='red', linestyle='--', label=f'Min threshold ({min_threshold:,})')
        axes[1, 0].axvline(max_threshold, color='red', linestyle='--', label=f'Max threshold ({max_threshold:,})')
        axes[1, 0].set_xlabel('Number of Vertices')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title(f'Filtered Meshes - Vertices (n={len(df_filtered)})')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Histogram 4: Filtered faces (within threshold only)
        axes[1, 1].hist(df_filtered['num_faces'], bins=30, alpha=0.7, color='green', edgecolor='black')
        axes[1, 1].set_xlabel('Number of Faces')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title(f'Filtered Meshes - Faces (n={len(df_filtered)})')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'No meshes within threshold!', ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 1].text(0.5, 0.5, 'No meshes within threshold!', ha='center', va='center', transform=axes[1, 1].transAxes)
    
    plt.tight_layout()
    plt.savefig('img/resampled_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Summary statistics for filtered data
    if len(df_filtered) > 0:
        print(f"\nFiltered Data Statistics (within threshold):")
        print(f"Vertices - Mean: {df_filtered['num_vertices'].mean():.0f}, Std: {df_filtered['num_vertices'].std():.0f}")
        print(f"Faces - Mean: {df_filtered['num_faces'].mean():.0f}, Std: {df_filtered['num_faces'].std():.0f}")
        print(f"Target achievement rate: {len(df_filtered)/total_files*100:.1f}%")
    
    return df, df_filtered

def plot_class_distribution(df, column='num_vertices'):
    """Plot distribution by class to see which classes have issues"""
    plt.figure(figsize=(12, 8))
    
    # Box plot by class
    plt.subplot(2, 1, 1)
    df.boxplot(column=column, by='class', ax=plt.gca(), rot=45)
    plt.title(f'{column.replace("_", " ").title()} Distribution by Class')
    plt.suptitle('')  # Remove default title
    
    # Count of problematic meshes by class
    plt.subplot(2, 1, 2)
    class_issues = df.groupby('class')['within_threshold'].agg(['count', 'sum']).reset_index()
    class_issues['problematic'] = class_issues['count'] - class_issues['sum']
    class_issues['success_rate'] = class_issues['sum'] / class_issues['count'] * 100
    
    bars = plt.bar(range(len(class_issues)), class_issues['success_rate'])
    plt.xlabel('Class')
    plt.ylabel('Success Rate (%)')
    plt.title('Resampling Success Rate by Class')
    plt.xticks(range(len(class_issues)), class_issues['class'], rotation=45)
    plt.ylim(0, 100)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.0f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('img/class_analysis.png', dpi=300, bbox_inches='tight')
    # plt.show()

def compare_original_vs_resampled(original_file, resampled_file=None, class_name=None):
    """
    Compare original and resampled mesh visualizations side by side.
    
    Args:
        original_file: Path to original mesh file
        resampled_file: Path to resampled mesh file (optional, will auto-find if None)
        class_name: Class name (optional, will extract from path if None)
    """
    from read_data import read_data
    from plots import show_mesh_simple
    import matplotlib.pyplot as plt
    from pathlib import Path
    
    # Extract class name and object name
    if class_name is None:
        class_name = Path(original_file).parent.name
    obj_name = Path(original_file).stem
    
    # Auto-find resampled file if not provided
    if resampled_file is None:
        resampled_dir = Path("resampled_data") / class_name
        if resampled_dir.exists():
            # Look for files starting with the object name
            resampled_files = list(resampled_dir.glob(f"{obj_name}_*.obj"))
            if resampled_files:
                resampled_file = str(resampled_files[0])  # Take the first match
            else:
                print(f"No resampled file found for {original_file}")
                return
        else:
            print(f"Resampled directory not found: {resampled_dir}")
            return
    
    try:
        # Load meshes
        print(f"Loading original: {original_file}")
        original_mesh = read_data(original_file)
        
        print(f"Loading resampled: {resampled_file}")
        resampled_mesh = read_data(resampled_file)
        
        # Get statistics
        orig_vertices = len(original_mesh.vertices)
        orig_faces = len(original_mesh.triangles)
        
        resamp_vertices = len(resampled_mesh.vertices)
        resamp_faces = len(resampled_mesh.triangles)
        
        # Print comparison stats
        print("=" * 60)
        print(f"MESH COMPARISON: {class_name}/{obj_name}")
        print("=" * 60)
        print(f"Original:   {orig_vertices:,} vertices, {orig_faces:,} faces")
        print(f"Resampled:  {resamp_vertices:,} vertices, {resamp_faces:,} faces")
        print(f"Change:     {resamp_vertices - orig_vertices:+,} vertices ({(resamp_vertices/orig_vertices)*100:.1f}%)")
        print(f"Target:     5,000 - 10,000 vertices")
        print(f"In range:   {'✅ YES' if 5000 <= resamp_vertices <= 10000 else '❌ NO'}")
        print("=" * 60)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 8))
        
        # Original mesh subplot
        ax1 = fig.add_subplot(121, projection='3d')
        vertices = np.asarray(original_mesh.vertices)
        triangles = np.asarray(original_mesh.triangles)
        
        ax1.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                        triangles=triangles, alpha=0.8, cmap='viridis')
        ax1.set_title(f'Original\n{orig_vertices:,} vertices, {orig_faces:,} faces', fontsize=12)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        
        # Resampled mesh subplot
        ax2 = fig.add_subplot(122, projection='3d')
        vertices2 = np.asarray(resampled_mesh.vertices)
        triangles2 = np.asarray(resampled_mesh.triangles)
        
        ax2.plot_trisurf(vertices2[:, 0], vertices2[:, 1], vertices2[:, 2], 
                        triangles=triangles2, alpha=0.8, cmap='viridis')
        ax2.set_title(f'Resampled\n{resamp_vertices:,} vertices, {resamp_faces:,} faces', fontsize=12)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        
        # Set same viewing angle for both
        ax1.view_init(elev=20, azim=45)
        ax2.view_init(elev=20, azim=45)
        
        plt.suptitle(f'Mesh Resampling Comparison: {class_name}/{obj_name}', fontsize=16)
        plt.tight_layout()
        
        # Save comparison image
        comparison_filename = f'img/comparison_{class_name}_{obj_name}.png'
        plt.savefig(comparison_filename, dpi=300, bbox_inches='tight')
        print(f"Saved comparison to: {comparison_filename}")
        plt.show()
        
        # Also show using Open3D for interactive viewing
        print("\nShowing interactive Open3D visualizations...")
        print("Close the Open3D window to continue to the next mesh.")
        
        # Show original
        print("Showing original mesh...")
        show_mesh_simple(original_mesh)
        
        # Show resampled
        print("Showing resampled mesh...")
        show_mesh_simple(resampled_mesh)
        
    except Exception as e:
        print(f"Error comparing meshes: {e}")

def compare_random_meshes(n_samples=3):
    """
    Compare n random original vs resampled meshes.
    """
    import pandas as pd
    import random
    
    # Read resampled stats to get available files
    df = pd.read_csv("stats/resampled_stats.csv")
    
    # Sample random files
    sampled_files = df.sample(n=min(n_samples, len(df)))
    
    for _, row in sampled_files.iterrows():
        # Extract original file path from resampled file path
        resampled_path = row['file']
        class_name = row['class']
        
        # Reconstruct original path
        resampled_filename = Path(resampled_path).name
        # Remove the vertex count suffix (e.g., "_7500" from "D00031_7500.obj")
        original_filename = resampled_filename.split('_')[0] + '.obj'
        original_path = f"data/{class_name}/{original_filename}"
        
        # Check if original file exists
        if Path(original_path).exists():
            print(f"\n{'='*60}")
            print(f"Comparing mesh {sampled_files.index[0] + 1} of {n_samples}")
            compare_original_vs_resampled(original_path, resampled_path, class_name)
        else:
            print(f"Original file not found: {original_path}")

# Add to your if __name__ == "__main__": section
if __name__ == "__main__":
    # Analyze resampled stats
    df_all, df_filtered = analyze_resampled_stats("stats/resampled_stats.csv")
    
    # Plot class distribution if data exists
    if len(df_all) > 0:
        plot_class_distribution(df_all)
    
    print("\nAnalysis complete! Check 'resampled_analysis.png' and 'class_analysis.png'")
    
    # # Compare some random meshes
    # print("\n" + "="*60)
    # print("COMPARING ORIGINAL VS RESAMPLED MESHES")
    # print("="*60)
    # compare_random_meshes(n_samples=3)