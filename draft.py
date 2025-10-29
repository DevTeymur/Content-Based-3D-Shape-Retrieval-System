# import pandas as pd
# import matplotlib.pyplot as plt
# from pathlib import Path
# import numpy as np


# def analyze_resampled_stats(csv_path="stats/resampled_stats.csv", target=7500, margin=2500):
#     """
#     Analyze resampled statistics and check how many files are within threshold.
#     Add flag column and plot histograms.
#     """
#     # Define thresholds
#     # min_threshold = target - margin  # 5000
#     # max_threshold = target + margin  # 10000
#     min_threshold, max_threshold = 4500, 12000  # stricter lower bound
    
#     # Read the CSV file
#     df = pd.read_csv(csv_path)
    
#     # Add threshold flag column
#     df['within_threshold'] = (
#         (df['num_vertices'] >= min_threshold) & 
#         (df['num_vertices'] <= max_threshold)
#     )
    
#     # Print statistics
#     total_files = len(df)
#     within_threshold = df['within_threshold'].sum()
#     below_threshold = (df['num_vertices'] < min_threshold).sum()
#     above_threshold = (df['num_vertices'] > max_threshold).sum()
    
#     print("=" * 60)
#     print("RESAMPLED MESH ANALYSIS")
#     print("=" * 60)
#     print(f"Target range: {min_threshold:,} - {max_threshold:,} vertices")
#     print(f"Total files analyzed: {total_files:,}")
#     print("-" * 40)
#     print(f"Within threshold ({min_threshold:,}-{max_threshold:,}): {within_threshold:,} ({within_threshold/total_files*100:.1f}%)")
#     print(f"Below threshold (<{min_threshold:,}): {below_threshold:,} ({below_threshold/total_files*100:.1f}%)")
#     print(f"Above threshold (>{max_threshold:,}): {above_threshold:,} ({above_threshold/total_files*100:.1f}%)")
#     print("=" * 60)
    
#     # Show some examples of problematic files
#     if below_threshold > 0:
#         print(f"\nFiles below threshold (showing first 10):")
#         below_files = df[df['num_vertices'] < min_threshold].head(10)
#         for _, row in below_files.iterrows():
#             print(f"  {row['file']}: {row['num_vertices']} vertices")
    
#     if above_threshold > 0:
#         print(f"\nFiles above threshold (showing first 10):")
#         above_files = df[df['num_vertices'] > max_threshold].head(10)
#         for _, row in above_files.iterrows():
#             print(f"  {row['file']}: {row['num_vertices']} vertices")
    
#     # Save updated CSV with flag column
#     output_csv = csv_path.replace('.csv', '_with_flags.csv')
#     df.to_csv(output_csv, index=False)
#     print(f"\nSaved updated CSV with flags to: {output_csv}")
    
#     # Plot histograms - Before filtering
#     fig, axes = plt.subplots(2, 2, figsize=(15, 10))
#     fig.suptitle('Resampled Mesh Analysis', fontsize=16)
    
#     # Histogram 1: All vertices (before filtering)
#     axes[0, 0].hist(df['num_vertices'], bins=50, alpha=0.7, color='blue', edgecolor='black')
#     axes[0, 0].axvline(min_threshold, color='red', linestyle='--', label=f'Min threshold ({min_threshold:,})')
#     axes[0, 0].axvline(max_threshold, color='red', linestyle='--', label=f'Max threshold ({max_threshold:,})')
#     axes[0, 0].set_xlabel('Number of Vertices')
#     axes[0, 0].set_ylabel('Frequency')
#     axes[0, 0].set_title('All Resampled Meshes - Vertices')
#     axes[0, 0].legend()
#     axes[0, 0].grid(True, alpha=0.3)
    
#     # Histogram 2: All faces (before filtering)
#     axes[0, 1].hist(df['num_faces'], bins=50, alpha=0.7, color='green', edgecolor='black')
#     axes[0, 1].set_xlabel('Number of Faces')
#     axes[0, 1].set_ylabel('Frequency')
#     axes[0, 1].set_title('All Resampled Meshes - Faces')
#     axes[0, 1].grid(True, alpha=0.3)
    
#     # Filter data to only include meshes within threshold
#     df_filtered = df[df['within_threshold']]
    
#     # Histogram 3: Filtered vertices (within threshold only)
#     if len(df_filtered) > 0:
#         axes[1, 0].hist(df_filtered['num_vertices'], bins=30, alpha=0.7, color='blue', edgecolor='black')
#         axes[1, 0].axvline(min_threshold, color='red', linestyle='--', label=f'Min threshold ({min_threshold:,})')
#         axes[1, 0].axvline(max_threshold, color='red', linestyle='--', label=f'Max threshold ({max_threshold:,})')
#         axes[1, 0].set_xlabel('Number of Vertices')
#         axes[1, 0].set_ylabel('Frequency')
#         axes[1, 0].set_title(f'Filtered Meshes - Vertices (n={len(df_filtered)})')
#         axes[1, 0].legend()
#         axes[1, 0].grid(True, alpha=0.3)
        
#         # Histogram 4: Filtered faces (within threshold only)
#         axes[1, 1].hist(df_filtered['num_faces'], bins=30, alpha=0.7, color='green', edgecolor='black')
#         axes[1, 1].set_xlabel('Number of Faces')
#         axes[1, 1].set_ylabel('Frequency')
#         axes[1, 1].set_title(f'Filtered Meshes - Faces (n={len(df_filtered)})')
#         axes[1, 1].grid(True, alpha=0.3)
#     else:
#         axes[1, 0].text(0.5, 0.5, 'No meshes within threshold!', ha='center', va='center', transform=axes[1, 0].transAxes)
#         axes[1, 1].text(0.5, 0.5, 'No meshes within threshold!', ha='center', va='center', transform=axes[1, 1].transAxes)
    
#     plt.tight_layout()
#     plt.savefig('img/resampled_analysis.png', dpi=300, bbox_inches='tight')
#     plt.show()
    
#     # Summary statistics for filtered data
#     if len(df_filtered) > 0:
#         print(f"\nFiltered Data Statistics (within threshold):")
#         print(f"Vertices - Mean: {df_filtered['num_vertices'].mean():.0f}, Std: {df_filtered['num_vertices'].std():.0f}")
#         print(f"Faces - Mean: {df_filtered['num_faces'].mean():.0f}, Std: {df_filtered['num_faces'].std():.0f}")
#         print(f"Target achievement rate: {len(df_filtered)/total_files*100:.1f}%")
    
#     return df, df_filtered

# def plot_class_distribution(df, column='num_vertices'):
#     """Plot distribution by class to see which classes have issues"""
#     plt.figure(figsize=(12, 8))
    
#     # Box plot by class
#     plt.subplot(2, 1, 1)
#     df.boxplot(column=column, by='class', ax=plt.gca(), rot=45)
#     plt.title(f'{column.replace("_", " ").title()} Distribution by Class')
#     plt.suptitle('')  # Remove default title
    
#     # Count of problematic meshes by class
#     plt.subplot(2, 1, 2)
#     class_issues = df.groupby('class')['within_threshold'].agg(['count', 'sum']).reset_index()
#     class_issues['problematic'] = class_issues['count'] - class_issues['sum']
#     class_issues['success_rate'] = class_issues['sum'] / class_issues['count'] * 100
    
#     bars = plt.bar(range(len(class_issues)), class_issues['success_rate'])
#     plt.xlabel('Class')
#     plt.ylabel('Success Rate (%)')
#     plt.title('Resampling Success Rate by Class')
#     plt.xticks(range(len(class_issues)), class_issues['class'], rotation=45)
#     plt.ylim(0, 100)
    
#     # Add value labels on bars
#     for i, bar in enumerate(bars):
#         height = bar.get_height()
#         plt.text(bar.get_x() + bar.get_width()/2., height + 1,
#                 f'{height:.0f}%', ha='center', va='bottom')
    
#     plt.tight_layout()
#     plt.savefig('img/class_analysis.png', dpi=300, bbox_inches='tight')
#     # plt.show()

# def compare_original_vs_resampled(original_file, resampled_file=None, class_name=None):
#     """
#     Compare original and resampled mesh visualizations side by side.
    
#     Args:
#         original_file: Path to original mesh file
#         resampled_file: Path to resampled mesh file (optional, will auto-find if None)
#         class_name: Class name (optional, will extract from path if None)
#     """
#     from read_data import read_data
#     from plots import show_mesh_simple
#     import matplotlib.pyplot as plt
#     from pathlib import Path
    
#     # Extract class name and object name
#     if class_name is None:
#         class_name = Path(original_file).parent.name
#     obj_name = Path(original_file).stem
    
#     # Auto-find resampled file if not provided
#     if resampled_file is None:
#         resampled_dir = Path("resampled_data") / class_name
#         if resampled_dir.exists():
#             # Look for files starting with the object name
#             resampled_files = list(resampled_dir.glob(f"{obj_name}_*.obj"))
#             if resampled_files:
#                 resampled_file = str(resampled_files[0])  # Take the first match
#             else:
#                 print(f"No resampled file found for {original_file}")
#                 return
#         else:
#             print(f"Resampled directory not found: {resampled_dir}")
#             return
    
#     try:
#         # Load meshes
#         print(f"Loading original: {original_file}")
#         original_mesh = read_data(original_file)
        
#         print(f"Loading resampled: {resampled_file}")
#         resampled_mesh = read_data(resampled_file)
        
#         # Get statistics
#         orig_vertices = len(original_mesh.vertices)
#         orig_faces = len(original_mesh.triangles)
        
#         resamp_vertices = len(resampled_mesh.vertices)
#         resamp_faces = len(resampled_mesh.triangles)
        
#         # Print comparison stats
#         print("=" * 60)
#         print(f"MESH COMPARISON: {class_name}/{obj_name}")
#         print("=" * 60)
#         print(f"Original:   {orig_vertices:,} vertices, {orig_faces:,} faces")
#         print(f"Resampled:  {resamp_vertices:,} vertices, {resamp_faces:,} faces")
#         print(f"Change:     {resamp_vertices - orig_vertices:+,} vertices ({(resamp_vertices/orig_vertices)*100:.1f}%)")
#         print(f"Target:     5,000 - 10,000 vertices")
#         print(f"In range:   {'✅ YES' if 5000 <= resamp_vertices <= 10000 else '❌ NO'}")
#         print("=" * 60)
        
#         # Create figure with subplots
#         fig = plt.figure(figsize=(16, 8))
        
#         # Original mesh subplot
#         ax1 = fig.add_subplot(121, projection='3d')
#         vertices = np.asarray(original_mesh.vertices)
#         triangles = np.asarray(original_mesh.triangles)
        
#         ax1.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
#                         triangles=triangles, alpha=0.8, cmap='viridis')
#         ax1.set_title(f'Original\n{orig_vertices:,} vertices, {orig_faces:,} faces', fontsize=12)
#         ax1.set_xlabel('X')
#         ax1.set_ylabel('Y')
#         ax1.set_zlabel('Z')
        
#         # Resampled mesh subplot
#         ax2 = fig.add_subplot(122, projection='3d')
#         vertices2 = np.asarray(resampled_mesh.vertices)
#         triangles2 = np.asarray(resampled_mesh.triangles)
        
#         ax2.plot_trisurf(vertices2[:, 0], vertices2[:, 1], vertices2[:, 2], 
#                         triangles=triangles2, alpha=0.8, cmap='viridis')
#         ax2.set_title(f'Resampled\n{resamp_vertices:,} vertices, {resamp_faces:,} faces', fontsize=12)
#         ax2.set_xlabel('X')
#         ax2.set_ylabel('Y')
#         ax2.set_zlabel('Z')
        
#         # Set same viewing angle for both
#         ax1.view_init(elev=20, azim=45)
#         ax2.view_init(elev=20, azim=45)
        
#         plt.suptitle(f'Mesh Resampling Comparison: {class_name}/{obj_name}', fontsize=16)
#         plt.tight_layout()
        
#         # Save comparison image
#         comparison_filename = f'img/comparison_{class_name}_{obj_name}.png'
#         plt.savefig(comparison_filename, dpi=300, bbox_inches='tight')
#         print(f"Saved comparison to: {comparison_filename}")
#         plt.show()
        
#         # Also show using Open3D for interactive viewing
#         print("\nShowing interactive Open3D visualizations...")
#         print("Close the Open3D window to continue to the next mesh.")
        
#         # Show original
#         print("Showing original mesh...")
#         show_mesh_simple(original_mesh)
        
#         # Show resampled
#         print("Showing resampled mesh...")
#         show_mesh_simple(resampled_mesh)
        
#     except Exception as e:
#         print(f"Error comparing meshes: {e}")

# def compare_random_meshes(n_samples=3):
#     """
#     Compare n random original vs resampled meshes.
#     """
#     import pandas as pd
#     import random
    
#     # Read resampled stats to get available files
#     df = pd.read_csv("stats/resampled_stats.csv")
    
#     # Sample random files
#     sampled_files = df.sample(n=min(n_samples, len(df)))
    
#     for _, row in sampled_files.iterrows():
#         # Extract original file path from resampled file path
#         resampled_path = row['file']
#         class_name = row['class']
        
#         # Reconstruct original path
#         resampled_filename = Path(resampled_path).name
#         # Remove the vertex count suffix (e.g., "_7500" from "D00031_7500.obj")
#         original_filename = resampled_filename.split('_')[0] + '.obj'
#         original_path = f"data/{class_name}/{original_filename}"
        
#         # Check if original file exists
#         if Path(original_path).exists():
#             print(f"\n{'='*60}")
#             print(f"Comparing mesh {sampled_files.index[0] + 1} of {n_samples}")
#             compare_original_vs_resampled(original_path, resampled_path, class_name)
#         else:
#             print(f"Original file not found: {original_path}")

# # Add to your if __name__ == "__main__": section
# if __name__ == "__main__":
#     # Analyze resampled stats
#     df_all, df_filtered = analyze_resampled_stats("stats/resampled_stats.csv")
    
#     # Plot class distribution if data exists
#     if len(df_all) > 0:
#         plot_class_distribution(df_all)
    
#     print("\nAnalysis complete! Check 'resampled_analysis.png' and 'class_analysis.png'")
    
#     # # Compare some random meshes
#     print("\n" + "="*60)
#     print("COMPARING ORIGINAL VS RESAMPLED MESHES")
#     print("="*60)
#     compare_random_meshes(n_samples=1)

# import pandas as pd
# df = pd.read_csv("stats/features_database.csv")
# print(len(df.columns))

# # In Python console after running step6_analysis.py:
# from step6_analysis import Step6Analyzer
# analyzer = Step6Analyzer("step5_data")
# analyzer.evaluator.initialize()

# # Load existing results
# import json
# with open("step5_data/step6_results/evaluation_results.json", 'r') as f:
#     analyzer.results = json.load(f)
# with open("step5_data/step6_results/summary_statistics.json", 'r') as f:
#     analyzer.summary = json.load(f)

# # Generate category analysis
# analyzer.create_category_ranking(k=5, metric='precision', top_n=15)
# analyzer.create_category_ranking(k=1, metric='precision', top_n=10)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import trimesh
from features import extract_scalar_features
import random

def analyze_elementary_features():
    """Validate elementary features across the database"""
    
    print("🔍 ELEMENTARY FEATURES VALIDATION")
    print("=" * 50)
    
    # Load sample of shapes from different categories
    feature_data = []
    normalized_dir = Path("normalized_data")
    
    # Get representative shapes from different categories
    categories = ['Airplane', 'Car', 'Chair', 'Table', 'Human', 'Bird', 'Fish']
    
    for category in categories:
        cat_dir = normalized_dir / category
        if cat_dir.exists():
            obj_files = list(cat_dir.glob("*.obj"))
            if obj_files:
                # Take first few shapes from each category
                for obj_file in obj_files[:5]:
                    try:
                        mesh = trimesh.load(str(obj_file))
                        features = extract_scalar_features(mesh)
                        features['category'] = category
                        features['filename'] = obj_file.name
                        feature_data.append(features)
                    except:
                        continue
    
    df = pd.DataFrame(feature_data)
    
    # 1. Check value ranges
    print("\n📊 FEATURE VALUE RANGES:")
    feature_cols = ['area', 'volume', 'aabb_volume', 'compactness', 'diameter', 'convexity', 'eccentricity']
    
    for col in feature_cols:
        if col in df.columns:
            print(f"{col:15} | Min: {df[col].min():.6f} | Max: {df[col].max():.6f} | Mean: {df[col].mean():.6f}")
    
    # 2. Expected ranges for normalized shapes
    print(f"\n✅ VALIDATION CHECKS:")
    print(f"Volume range 0-1: {'✅' if df['volume'].min() >= 0 and df['volume'].max() <= 1.5 else '❌'}")
    print(f"Area reasonable: {'✅' if df['area'].min() >= 0 and df['area'].max() <= 10 else '❌'}")
    print(f"Convexity 0-1: {'✅' if df['convexity'].min() >= 0 and df['convexity'].max() <= 1 else '❌'}")
    print(f"Diameter reasonable: {'✅' if df['diameter'].min() >= 0 and df['diameter'].max() <= 2 else '❌'}")
    
    # 3. Show examples of different shape types
    print(f"\n🎭 SHAPE TYPE EXAMPLES:")
    
    # Find most/least compact, eccentric, etc.
    most_compact = df.loc[df['compactness'].idxmin()]
    least_compact = df.loc[df['compactness'].idxmax()]
    most_eccentric = df.loc[df['eccentricity'].idxmax()]
    least_eccentric = df.loc[df['eccentricity'].idxmin()]
    
    examples = [
        ("Most Compact (sphere-like)", most_compact),
        ("Least Compact (elongated)", least_compact), 
        ("Most Eccentric (flat/long)", most_eccentric),
        ("Least Eccentric (round)", least_eccentric)
    ]
    
    for desc, row in examples:
        print(f"\n{desc}:")
        print(f"  File: {row['filename']} ({row['category']})")
        print(f"  Volume: {row['volume']:.3f}, Compactness: {row['compactness']:.3f}")
        print(f"  Eccentricity: {row['eccentricity']:.3f}, Convexity: {row['convexity']:.3f}")
    
    return df

def create_feature_distribution_plots(df):
    """Create distribution plots for elementary features"""
    
    feature_cols = ['area', 'volume', 'compactness', 'diameter', 'convexity', 'eccentricity']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, col in enumerate(feature_cols):
        if col in df.columns:
            # Distribution plot
            axes[i].hist(df[col], bins=20, alpha=0.7, edgecolor='black')
            axes[i].set_title(f'{col.title()} Distribution')
            axes[i].set_xlabel(col.title())
            axes[i].set_ylabel('Frequency')
            axes[i].grid(True, alpha=0.3)
            
            # Add statistics
            mean_val = df[col].mean()
            axes[i].axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.3f}')
            axes[i].legend()
    
    plt.tight_layout()
    plt.savefig('img/elementary_features_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📈 Feature distribution plots saved to: elementary_features_distributions.png")

# if __name__ == "__main__":
#     df = analyze_elementary_features()
#     create_feature_distribution_plots(df)

import numpy as np
import matplotlib.pyplot as plt
from features import extract_histogram_features
from scipy.spatial.distance import euclidean
import trimesh
from pathlib import Path

def get_available_categories():
    """Get list of available categories in normalized_data"""
    normalized_dir = Path("normalized_data")
    categories = [d.name for d in normalized_dir.iterdir() if d.is_dir()]
    print(f"Available categories: {categories[:10]}...")  # Show first 10
    return categories

def analyze_histogram_discriminativeness():
    """Analyze if histogram features are discriminative between classes"""
    
    print("📊 HISTOGRAM FEATURES DISCRIMINATIVE ANALYSIS")
    print("=" * 60)
    
    # Get available categories and select first 4 with enough shapes
    available_categories = get_available_categories()
    test_categories = []
    
    normalized_dir = Path("normalized_data")
    
    for category in available_categories:
        cat_dir = normalized_dir / category
        if cat_dir.exists():
            obj_files = list(cat_dir.glob("*.obj"))
            if len(obj_files) >= 3:  # Need at least 3 shapes for analysis
                test_categories.append(category)
                if len(test_categories) >= 4:  # Take first 4 suitable categories
                    break
    
    print(f"Selected categories for analysis: {test_categories}")
    
    histogram_data = {}
    
    for category in test_categories:
        cat_dir = normalized_dir / category
        category_histograms = []
        obj_files = list(cat_dir.glob("*.obj"))[:3]  # Take 3 shapes per category
        
        print(f"Processing {category}: {len(obj_files)} shapes")
        
        for obj_file in obj_files:
            try:
                mesh = trimesh.load(str(obj_file))
                hist_features = extract_histogram_features(mesh)
                category_histograms.append(hist_features)
                print(f"  ✅ {obj_file.name}")
            except Exception as e:
                print(f"  ❌ {obj_file.name}: {e}")
                continue
        
        if category_histograms:
            histogram_data[category] = category_histograms
    
    # 1. Compute intra-class vs inter-class distances
    print("\n🔍 INTRA-CLASS vs INTER-CLASS ANALYSIS:")
    
    # For each descriptor type (A3, D1, D2, D3, D4)
    descriptor_types = ['A3', 'D1', 'D2', 'D3', 'D4']
    
    for desc_type in descriptor_types:
        print(f"\n📈 {desc_type} Descriptor Analysis:")
        
        intra_distances = []
        inter_distances = []
        
        # Intra-class distances (within same category)
        for category, histograms in histogram_data.items():
            if len(histograms) >= 2:
                for i in range(len(histograms)):
                    for j in range(i+1, len(histograms)):
                        try:
                            # Extract histogram bins for this descriptor
                            hist1 = [histograms[i][f"{desc_type}_{k}"] for k in range(10)]
                            hist2 = [histograms[j][f"{desc_type}_{k}"] for k in range(10)]
                            
                            distance = euclidean(hist1, hist2)
                            intra_distances.append(distance)
                        except KeyError as e:
                            print(f"    Warning: Missing key {e} for {category}")
                            continue
        
        # Inter-class distances (between different categories)
        categories = list(histogram_data.keys())
        for i in range(len(categories)):
            for j in range(i+1, len(categories)):
                cat1_histograms = histogram_data[categories[i]]
                cat2_histograms = histogram_data[categories[j]]
                
                for hist1_data in cat1_histograms:
                    for hist2_data in cat2_histograms:
                        try:
                            hist1 = [hist1_data[f"{desc_type}_{k}"] for k in range(10)]
                            hist2 = [hist2_data[f"{desc_type}_{k}"] for k in range(10)]
                            
                            distance = euclidean(hist1, hist2)
                            inter_distances.append(distance)
                        except KeyError:
                            continue
        
        # Statistics
        if intra_distances and inter_distances:
            intra_mean = np.mean(intra_distances)
            inter_mean = np.mean(inter_distances)
            separation_ratio = inter_mean / intra_mean if intra_mean > 0 else float('inf')
            
            print(f"  Intra-class distance (mean): {intra_mean:.4f}")
            print(f"  Inter-class distance (mean): {inter_mean:.4f}")
            print(f"  Separation ratio: {separation_ratio:.2f} {'✅' if separation_ratio > 1.2 else '❌'}")
            print(f"  Discriminative power: {'Good' if separation_ratio > 1.5 else 'Moderate' if separation_ratio > 1.2 else 'Poor'}")
        else:
            print(f"  ❌ Insufficient data for analysis")

def visualize_histogram_comparison():
    """Visualize histogram shapes for same vs different classes"""
    
    print("\n🎨 HISTOGRAM SHAPE VISUALIZATION")
    
    # Get available categories
    normalized_dir = Path("normalized_data")
    available_categories = [d.name for d in normalized_dir.iterdir() if d.is_dir()]
    
    # Find two categories with shapes
    selected_files = []
    selected_categories = []
    
    for category in available_categories:
        cat_dir = normalized_dir / category
        obj_files = list(cat_dir.glob("*.obj"))
        if obj_files:
            selected_files.append(obj_files[0])
            selected_categories.append(category)
            if len(selected_files) >= 2:
                break
    
    if len(selected_files) < 2:
        print("❌ Not enough categories with shapes found for comparison")
        return
    
    print(f"Comparing: {selected_categories[0]} vs {selected_categories[1]}")
    
    try:
        mesh1 = trimesh.load(str(selected_files[0]))
        mesh2 = trimesh.load(str(selected_files[1]))
        
        hist1 = extract_histogram_features(mesh1)
        hist2 = extract_histogram_features(mesh2)
        
        # Plot A3 and D2 histograms for comparison
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        descriptor_types = ['A3', 'D2']
        
        for i, desc_type in enumerate(descriptor_types):
            try:
                # Extract histogram values
                values1 = [hist1[f"{desc_type}_{j}"] for j in range(10)]
                values2 = [hist2[f"{desc_type}_{j}"] for j in range(10)]
                
                # Plot first shape
                axes[i, 0].bar(range(10), values1, alpha=0.7, color='blue', edgecolor='black')
                axes[i, 0].set_title(f'{desc_type} - {selected_categories[0]}')
                axes[i, 0].set_xlabel('Bin')
                axes[i, 0].set_ylabel('Normalized Frequency')
                axes[i, 0].grid(True, alpha=0.3)
                
                # Plot second shape
                axes[i, 1].bar(range(10), values2, alpha=0.7, color='red', edgecolor='black')
                axes[i, 1].set_title(f'{desc_type} - {selected_categories[1]}')
                axes[i, 1].set_xlabel('Bin')
                axes[i, 1].set_ylabel('Normalized Frequency')
                axes[i, 1].grid(True, alpha=0.3)
                
            except KeyError as e:
                print(f"❌ Missing histogram data for {desc_type}: {e}")
                continue
        
        plt.tight_layout()
        plt.savefig('histogram_comparison_different_classes.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📊 Histogram comparison plots saved")
        
    except Exception as e:
        print(f"❌ Error in visualization: {e}")

if __name__ == "__main__":
    analyze_histogram_discriminativeness()
    visualize_histogram_comparison()

# Add this to validate your sampling approach
def analyze_sampling_coverage():
    """Analyze if 2000 samples provide good coverage"""
    
    print("🎯 SAMPLING COVERAGE ANALYSIS")
    print("=" * 40)
    
    # Test with different sample sizes
    sample_sizes = [500, 1000, 2000, 5000, 10000]
    
    # Load a test mesh
    test_file = list(Path("normalized_data").rglob("*.obj"))[0]
    mesh = trimesh.load(str(test_file))
    
    print(f"Test mesh: {test_file.name}")
    print(f"Vertices: {len(mesh.vertices)}")
    
    # For A3 descriptor, test different sample sizes
    from features import BINS, HIST_RANGES
    import random
    
    results = {}
    
    for sample_size in sample_sizes:
        # Compute A3 with this sample size
        vertices = list(mesh.vertices)
        angles = []
        
        for _ in range(sample_size):
            a, b, c = random.sample(vertices, 3)
            ab = b - a
            ac = c - a
            cos_angle = np.dot(ab, ac) / (np.linalg.norm(ab) * np.linalg.norm(ac))
            cos_angle = np.clip(cos_angle, -1, 1)
            angles.append(np.arccos(cos_angle))
        
        # Compute histogram
        hist, _ = np.histogram(angles, bins=BINS, range=HIST_RANGES['A3'])
        hist = hist / np.sum(hist)
        
        results[sample_size] = hist
    
    # Compare stability of histograms
    print(f"\n📊 HISTOGRAM STABILITY ANALYSIS:")
    base_hist = results[2000]  # Your current setting
    
    for sample_size, hist in results.items():
        if sample_size != 2000:
            difference = euclidean(base_hist, hist)
            print(f"Sample size {sample_size:5d}: Distance from 2000-sample = {difference:.4f}")
    
    print(f"\n💡 INTERPRETATION:")
    print(f"  - Small distances (<0.1) indicate stable histograms")
    print(f"  - Large distances (>0.2) suggest insufficient sampling")
    print(f"  - Your 2000 samples: {'Adequate' if max([euclidean(base_hist, h) for h in results.values()]) < 0.15 else 'May need increase'}")

# if __name__ == "__main__":
#     analyze_sampling_coverage()