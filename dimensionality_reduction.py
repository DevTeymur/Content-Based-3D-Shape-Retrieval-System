"""
Step 5 Part 3 - Dimensionality Reduction & Interactive Visualization
- Load normalized features from Part 1
- Perform t-SNE to create 2D embedding
- Interactive scatterplot with category coloring
- Brushing functionality to inspect shapes
- Integration with KNN engine for similarity highlighting
"""

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.manifold import TSNE
from pathlib import Path
import time
import mplcursors
from knn_engine import KNNEngine

class DimensionalityReducer:
    def __init__(self, data_dir="step5_data"):
        """
        Initialize Dimensionality Reducer with processed features from Part 1
        
        Args:
            data_dir: Directory containing processed features from preparation step
        """
        self.data_dir = Path(data_dir)
        self.X_features = None
        self.labels = None
        self.metadata = None
        self.feature_info = None
        self.X_2d = None
        self.tsne_model = None
        self.knn_engine = None
        
        # Visualization components
        self.fig = None
        self.ax = None
        self.scatter = None
        self.category_colors = {}
        self.unique_categories = None
        
    def load_processed_features(self):
        """Load processed features and metadata from Part 1"""
        try:
            # Load normalized features
            features_file = self.data_dir / "features_normalized.npy"
            self.X_features = np.load(features_file)
            print(f"✅ Loaded features matrix: {self.X_features.shape}")
            
            # Load labels (categories)
            labels_file = self.data_dir / "labels.npy"
            self.labels = np.load(labels_file, allow_pickle=True)
            print(f"✅ Loaded labels: {len(self.labels)} categories")
            
            # Load metadata mapping
            metadata_file = self.data_dir / "metadata.pkl"
            with open(metadata_file, 'rb') as f:
                self.metadata = pickle.load(f)
            print(f"✅ Loaded metadata: {len(self.metadata)} shapes")
            
            # Load feature info
            info_file = self.data_dir / "feature_info.pkl"
            with open(info_file, 'rb') as f:
                self.feature_info = pickle.load(f)
            print(f"✅ Loaded feature info: {self.feature_info['n_features']} features")
            
            # Get unique categories for color mapping
            self.unique_categories = sorted(list(set(self.labels)))
            print(f"✅ Found {len(self.unique_categories)} unique categories")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading processed features: {e}")
            return False
    
    def compute_tsne(self, n_components=2, perplexity=30, n_iter=1000, random_state=42, learning_rate='auto'):
        """
        Compute t-SNE dimensionality reduction
        
        Args:
            n_components: Target dimensions (should be 2 for visualization)
            perplexity: Balance between local and global structure (5-50)
            n_iter: Maximum iterations
            random_state: Random seed for reproducibility
            learning_rate: Learning rate ('auto' or float)
        """
        if self.X_features is None:
            print("❌ Please load processed features first")
            return False
        
        try:
            print(f"🔄 Computing t-SNE embedding...")
            print(f"   📊 Input shape: {self.X_features.shape}")
            print(f"   🎯 Target dimensions: {n_components}")
            print(f"   🔧 Perplexity: {perplexity}")
            print(f"   🔄 Iterations: {n_iter}")
            
            start_time = time.time()
            
            # Initialize t-SNE
            self.tsne_model = TSNE(
                n_components=n_components,
                perplexity=perplexity,
                n_iter=n_iter,
                random_state=random_state,
                learning_rate=learning_rate,
                verbose=1  # Show progress
            )
            
            # Fit and transform
            self.X_2d = self.tsne_model.fit_transform(self.X_features)
            
            computation_time = time.time() - start_time
            
            print(f"✅ t-SNE computation completed!")
            print(f"   ⏱️  Computation time: {computation_time:.2f} seconds")
            print(f"   📐 Output shape: {self.X_2d.shape}")
            print(f"   📊 2D range: X[{self.X_2d[:, 0].min():.2f}, {self.X_2d[:, 0].max():.2f}], Y[{self.X_2d[:, 1].min():.2f}, {self.X_2d[:, 1].max():.2f}]")
            
            return True
            
        except Exception as e:
            print(f"❌ Error computing t-SNE: {e}")
            return False
    
    def _create_category_colors(self):
        """Create color mapping for categories"""
        n_categories = len(self.unique_categories)
        
        # Use tab20 colormap for up to 20 categories, then cycle through
        if n_categories <= 20:
            colors = plt.cm.tab20(np.linspace(0, 1, n_categories))
        else:
            # For more categories, use a combination of colormaps
            colors1 = plt.cm.tab20(np.linspace(0, 1, 20))
            colors2 = plt.cm.Set3(np.linspace(0, 1, min(12, n_categories - 20)))
            remaining = max(0, n_categories - 32)
            if remaining > 0:
                colors3 = plt.cm.Pastel1(np.linspace(0, 1, min(9, remaining)))
                colors = np.vstack([colors1, colors2, colors3])
            else:
                colors = np.vstack([colors1, colors2])
        
        # Create category to color mapping
        for i, category in enumerate(self.unique_categories):
            self.category_colors[category] = colors[i % len(colors)]
        
        print(f"✅ Created color mapping for {len(self.category_colors)} categories")
    
    def create_interactive_plot(self, figsize=(12, 10), point_size=20, alpha=0.7):
        """
        Create interactive scatterplot with category coloring
        
        Args:
            figsize: Figure size tuple
            point_size: Size of scatter points
            alpha: Transparency of points
        """
        if self.X_2d is None:
            print("❌ Please compute t-SNE embedding first")
            return False
        
        try:
            print("🎨 Creating interactive scatterplot...")
            
            # Create color mapping
            self._create_category_colors()
            
            # Create color array for all points
            colors = [self.category_colors[label] for label in self.labels]
            
            # Create the plot
            self.fig, self.ax = plt.subplots(figsize=figsize)
            
            # Create scatter plot
            self.scatter = self.ax.scatter(
                self.X_2d[:, 0], 
                self.X_2d[:, 1],
                c=colors,
                s=point_size,
                alpha=alpha,
                edgecolors='black',
                linewidth=0.5
            )
            
            # Style the plot
            self.ax.set_title('t-SNE Visualization of 3D Shape Features\n(Click points for details)', 
                             fontsize=16, fontweight='bold')
            self.ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
            self.ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
            self.ax.grid(True, alpha=0.3)
            
            # Create legend with top categories
            self._create_legend()
            
            # Add interactive cursors
            self._setup_interactivity()
            
            print("✅ Interactive plot created successfully!")
            print("   🖱️  Click on points to see shape details")
            print("   🎯 Right-click to highlight similar shapes")
            
            return True
            
        except Exception as e:
            print(f"❌ Error creating interactive plot: {e}")
            return False
    
    def _create_legend(self, max_legend_items=15):
        """Create legend with most frequent categories"""
        # Count category frequencies
        category_counts = {}
        for label in self.labels:
            category_counts[label] = category_counts.get(label, 0) + 1
        
        # Sort by frequency
        sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Create legend for top categories
        legend_categories = sorted_categories[:max_legend_items]
        legend_elements = []
        
        for category, count in legend_categories:
            color = self.category_colors[category]
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                           markerfacecolor=color, markersize=8, 
                                           label=f'{category} ({count})'))
        
        # Add legend
        legend = self.ax.legend(handles=legend_elements, loc='center left', 
                              bbox_to_anchor=(1, 0.5), title='Top Categories',
                              title_fontsize=12, fontsize=10)
        legend.get_title().set_fontweight('bold')
    
    def _setup_interactivity(self):
        """Setup interactive cursors for point inspection"""
        cursor = mplcursors.cursor(self.scatter, hover=True)
        
        @cursor.connect("add")
        def on_add(sel):
            # Get point index
            index = sel.index
            
            # Get shape information
            meta = self.metadata[index]
            label = self.labels[index]
            
            # Create annotation text
            annotation_text = (
                f"{meta['filename']}\n"
                f"Category: {label}\n"
                f"Index: {index}\n"
                f"Position: ({self.X_2d[index, 0]:.2f}, {self.X_2d[index, 1]:.2f})"
            )
            
            sel.annotation.set_text(annotation_text)
            sel.annotation.get_bbox_patch().set(boxstyle="round,pad=0.5", 
                                              facecolor="yellow", alpha=0.8)
        
        # Connect mouse click for KNN highlighting
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
    
    def _on_click(self, event):
        """Handle mouse click events for KNN highlighting"""
        if event.inaxes != self.ax or event.button != 3:  # Right click only
            return
        
        # Find closest point to click
        distances = np.sqrt((self.X_2d[:, 0] - event.xdata)**2 + (self.X_2d[:, 1] - event.ydata)**2)
        closest_index = np.argmin(distances)
        
        # Highlight similar shapes using KNN
        self.highlight_similar_shapes(closest_index, k=10)
    
    def highlight_similar_shapes(self, query_index, k=10):
        """
        Highlight K nearest neighbors of a selected shape
        
        Args:
            query_index: Index of the query shape
            k: Number of similar shapes to highlight
        """
        if self.knn_engine is None:
            print("🔄 Initializing KNN engine for similarity search...")
            self.knn_engine = KNNEngine(self.data_dir)
            if not self.knn_engine.load_processed_features():
                print("❌ Failed to load KNN engine")
                return
            if not self.knn_engine.build_index():
                print("❌ Failed to build KNN index")
                return
        
        try:
            # Get similar shapes
            results = self.knn_engine.query_knn(query_index, k=k)
            if results is None:
                return
            
            # Get indices of similar shapes
            similar_indices = results['database_index'].values
            
            # Clear previous highlights
            self.ax.collections.clear()
            
            # Recreate base scatter plot
            colors = [self.category_colors[label] for label in self.labels]
            self.scatter = self.ax.scatter(
                self.X_2d[:, 0], 
                self.X_2d[:, 1],
                c=colors,
                s=20,
                alpha=0.7,
                edgecolors='black',
                linewidth=0.5
            )
            
            # Highlight similar shapes
            similar_x = self.X_2d[similar_indices, 0]
            similar_y = self.X_2d[similar_indices, 1]
            
            # Highlight with larger, brighter points
            self.ax.scatter(similar_x, similar_y, 
                          c='red', s=100, alpha=0.9, 
                          edgecolors='black', linewidth=2,
                          marker='o', label=f'Top {k} Similar')
            
            # Highlight query shape specially
            query_x, query_y = self.X_2d[query_index]
            self.ax.scatter(query_x, query_y, 
                          c='blue', s=150, alpha=1.0,
                          edgecolors='white', linewidth=3,
                          marker='*', label='Query Shape')
            
            # Add legend for highlights
            self.ax.legend(loc='upper right')
            
            # Update plot
            self.fig.canvas.draw()
            
            # Print information
            query_meta = self.metadata[query_index]
            print(f"\n🎯 Highlighted {k} shapes similar to:")
            print(f"   📄 {query_meta['filename']} ({self.labels[query_index]})")
            print("   Top 5 similar shapes:")
            for i, (_, row) in enumerate(results.head().iterrows()):
                print(f"   {i+1}. {row['filename']} ({row['category']}) - distance: {row['distance']:.3f}")
            
        except Exception as e:
            print(f"❌ Error highlighting similar shapes: {e}")
    
    def show_plot(self):
        """Display the interactive plot"""
        if self.fig is None:
            print("❌ Please create interactive plot first")
            return
        
        plt.tight_layout()
        plt.show()
    
    def save_embedding(self, filename="tsne_embedding_2d.npy"):
        """Save 2D embedding to file"""
        if self.X_2d is None:
            print("❌ No embedding to save. Please compute t-SNE first")
            return False
        
        try:
            output_file = self.data_dir / filename
            np.save(output_file, self.X_2d)
            print(f"✅ Saved 2D embedding to {output_file}")
            
            # Also save a CSV with metadata for external analysis
            embedding_df = pd.DataFrame({
                'filename': [meta['filename'] for meta in self.metadata],
                'category': self.labels,
                'x': self.X_2d[:, 0],
                'y': self.X_2d[:, 1]
            })
            
            csv_file = self.data_dir / "tsne_embedding_with_metadata.csv"
            embedding_df.to_csv(csv_file, index=False)
            print(f"✅ Saved embedding with metadata to {csv_file}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving embedding: {e}")
            return False
    
    def analyze_clusters(self):
        """Analyze clustering quality by categories"""
        if self.X_2d is None:
            print("❌ Please compute t-SNE embedding first")
            return
        
        print(f"\n📊 CLUSTER ANALYSIS")
        print("=" * 50)
        
        # Calculate intra-category distances
        category_stats = {}
        
        for category in self.unique_categories:
            # Get points of this category
            category_mask = np.array(self.labels) == category
            category_points = self.X_2d[category_mask]
            
            if len(category_points) > 1:
                # Calculate pairwise distances within category
                distances = []
                for i in range(len(category_points)):
                    for j in range(i+1, len(category_points)):
                        dist = np.linalg.norm(category_points[i] - category_points[j])
                        distances.append(dist)
                
                category_stats[category] = {
                    'n_shapes': len(category_points),
                    'avg_intra_distance': np.mean(distances),
                    'std_intra_distance': np.std(distances)
                }
        
        # Sort by clustering quality (lower intra-distance = better clustering)
        sorted_categories = sorted(category_stats.items(), 
                                 key=lambda x: x[1]['avg_intra_distance'])
        
        print("Categories with best clustering (lowest intra-category distance):")
        for i, (category, stats) in enumerate(sorted_categories[:10]):
            print(f"{i+1:2d}. {category:20s} - {stats['n_shapes']:3d} shapes, "
                  f"avg dist: {stats['avg_intra_distance']:.2f} ± {stats['std_intra_distance']:.2f}")
    
    def get_statistics(self):
        """Get comprehensive statistics about the embedding"""
        if self.X_2d is None:
            print("❌ No embedding available")
            return None
        
        stats = {
            'embedding_shape': self.X_2d.shape,
            'n_categories': len(self.unique_categories),
            'embedding_range': {
                'x_min': float(self.X_2d[:, 0].min()),
                'x_max': float(self.X_2d[:, 0].max()),
                'y_min': float(self.X_2d[:, 1].min()),
                'y_max': float(self.X_2d[:, 1].max())
            },
            'tsne_params': self.tsne_model.get_params() if self.tsne_model else None
        }
        
        return stats

def main():
    """Main function to test dimensionality reduction"""
    print("🚀 Starting Dimensionality Reduction & Visualization...")
    
    # Initialize DR system
    dr = DimensionalityReducer()
    
    # Step 1: Load processed features
    if not dr.load_processed_features():
        print("❌ Failed to load processed features")
        return
    
    # Step 2: Compute t-SNE embedding
    if not dr.compute_tsne(perplexity=30, n_iter=1000):
        print("❌ Failed to compute t-SNE embedding")
        return
    
    # Step 3: Analyze clusters
    dr.analyze_clusters()
    
    # Step 4: Save embedding
    if not dr.save_embedding():
        print("❌ Failed to save embedding")
        return
    
    # Step 5: Create interactive plot
    if not dr.create_interactive_plot():
        print("❌ Failed to create interactive plot")
        return
    
    # Step 6: Show statistics
    stats = dr.get_statistics()
    if stats:
        print(f"\n📊 EMBEDDING STATISTICS")
        print("=" * 50)
        print(f"Embedding shape: {stats['embedding_shape']}")
        print(f"Categories: {stats['n_categories']}")
        print(f"X range: [{stats['embedding_range']['x_min']:.2f}, {stats['embedding_range']['x_max']:.2f}]")
        print(f"Y range: [{stats['embedding_range']['y_min']:.2f}, {stats['embedding_range']['y_max']:.2f}]")
    
    print("\n✅ Dimensionality reduction completed!")
    print("🎨 Interactive plot created - use mouse to explore:")
    print("   • Hover over points to see details")
    print("   • Right-click points to highlight similar shapes")
    
    # Step 7: Display the plot
    dr.show_plot()

if __name__ == "__main__":
    main()