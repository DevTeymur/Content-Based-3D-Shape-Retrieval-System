"""
Simple 3D Shape Retrieval GUI Tool
Loads Step 3 features and provides distance-based similarity search
"""

# Step 1 — Import and Load Processed Data
import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean, cosine, cityblock
from tkinter import *
from tkinter import filedialog, messagebox, ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pathlib import Path
import time

# Add these imports after your existing imports
from knn_engine import KNNEngine
from dimensionality_reduction import DimensionalityReducer
import threading

class ShapeRetrievalGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("3D Shape Retrieval Tool - Step 4 & 5")  # Updated title
        self.root.geometry("1200x700")  # Increased width for new controls
        
        # Data storage (existing)
        self.features_df = None
        self.feature_columns = None
        self.current_results = None
        
        # NEW: Step 5 components
        self.knn_engine = None
        self.dimensionality_reducer = None
        self.step5_initialized = False
        
        # Create GUI layout
        self.create_widgets()
        
        # Try to auto-load default data
        self.auto_load_default_data()
    
    def create_widgets(self):
        """Create the GUI layout"""
        
        # Main frame
        main_frame = Frame(self.root, padx=10, pady=10)
        main_frame.pack(fill=BOTH, expand=True)
        
        # Title
        title_label = Label(main_frame, text="3D Shape Retrieval Tool", 
                        font=("Arial", 16, "bold"))
        title_label.pack(pady=(0, 10))
        
        # Control buttons frame
        control_frame = Frame(main_frame)
        control_frame.pack(fill=X, pady=(0, 10))
        
        # Load Data button
        self.load_btn = Button(control_frame, text="Load Data", 
                            command=self.load_data, bg="lightblue")
        self.load_btn.pack(side=LEFT, padx=(0, 5))
        
        # Compute Stats button  
        self.stats_btn = Button(control_frame, text="Compute Stats",
                            command=self.compute_stats, bg="lightgreen")
        self.stats_btn.pack(side=LEFT, padx=5)
        
        # Mesh Info button
        self.mesh_info_btn = Button(control_frame, text="Mesh Info",
                                command=self.show_mesh_info, bg="lightcyan")
        self.mesh_info_btn.pack(side=LEFT, padx=5)
        
        # NEW: Visualize Mesh button (single mesh)
        self.viz_mesh_btn = Button(control_frame, text="View Mesh",
                                command=self.visualize_single_mesh, bg="lightpink")
        self.viz_mesh_btn.pack(side=LEFT, padx=5)
        
        # Compute Distances button
        self.distance_btn = Button(control_frame, text="Compute Distances",
                                command=self.compute_distances, bg="lightyellow")
        self.distance_btn.pack(side=LEFT, padx=5)
        
        # Visualize Similar button (renamed for clarity)
        self.viz_similar_btn = Button(control_frame, text="View Similar",
                                    command=self.visualize_results, bg="lightcoral")
        self.viz_similar_btn.pack(side=LEFT, padx=5)
        
        # Export button
        self.export_btn = Button(control_frame, text="Export Results",
                                command=self.export_results, bg="lightgray")
        self.export_btn.pack(side=LEFT, padx=(5, 0))
        

        
        # Status label
        self.status_label = Label(main_frame, text="Ready to load data...", 
                                 fg="blue")
        self.status_label.pack(fill=X, pady=(0, 5))
        
        # Options frame
        options_frame = Frame(main_frame)
        options_frame.pack(fill=X, pady=(0, 10))
        
        # Distance metric selection
        Label(options_frame, text="Distance Metric:").pack(side=LEFT)
        self.distance_var = StringVar(value="euclidean")
        distance_menu = ttk.Combobox(options_frame, textvariable=self.distance_var,
                                    values=["euclidean", "cosine", "manhattan", "advanced_combined"],
                                    state="readonly", width=15)  # Increased width
        distance_menu.pack(side=LEFT, padx=(5, 20))
        
        # Reference mesh selection
        Label(options_frame, text="Reference Mesh:").pack(side=LEFT)
        self.mesh_var = StringVar(value="Select mesh...")
        self.mesh_menu = ttk.Combobox(options_frame, textvariable=self.mesh_var,
                                     state="readonly", width=20)
        self.mesh_menu.pack(side=LEFT, padx=(5, 5))
        
        # Browse by Category button
        self.browse_btn = Button(options_frame, text="Browse by Category",
                            command=self.browse_by_category, bg="lightsteelblue",
                            font=("Arial", 9))
        self.browse_btn.pack(side=LEFT, padx=(5, 20))
        
        # Selected mesh info
        self.selected_info = Label(options_frame, text="", fg="darkgreen", 
                                  font=("Arial", 9))
        self.selected_info.pack(side=LEFT)
        
        # Results display area (REMOVED visualization panel)
        results_frame = Frame(main_frame)
        results_frame.pack(fill=BOTH, expand=True)
        
        # Text area for results (now takes full width)
        text_frame = Frame(results_frame)
        text_frame.pack(fill=BOTH, expand=True)
        
        Label(text_frame, text="Results:", font=("Arial", 12, "bold")).pack(anchor=W)
        
        self.results_text = Text(text_frame, wrap=WORD, font=("Courier", 9))
        scrollbar = Scrollbar(text_frame, orient=VERTICAL, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)
        
        self.results_text.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar.pack(side=RIGHT, fill=Y)
        
        # NEW: Step 5 Controls Frame
        step5_frame = Frame(main_frame)
        step5_frame.pack(fill=X, pady=(5, 10))

        # Step 5 section label
        step5_label = Label(step5_frame, text="Step 5 - Fast KNN Search & t-SNE Visualization:", 
                           font=("Arial", 10, "bold"), fg="darkblue")
        step5_label.pack(anchor=W, pady=(0, 5))

        # Step 5 controls sub-frame
        step5_controls = Frame(step5_frame)
        step5_controls.pack(fill=X)

        # K-value selection
        Label(step5_controls, text="K neighbors:", font=("Arial", 9)).pack(side=LEFT)
        self.k_var = StringVar(value="10")
        k_spinbox = Spinbox(step5_controls, from_=1, to=50, textvariable=self.k_var, 
                           width=5, font=("Arial", 9))
        k_spinbox.pack(side=LEFT, padx=(5, 15))

        # Radius selection for range search
        Label(step5_controls, text="Radius:", font=("Arial", 9)).pack(side=LEFT)
        self.radius_var = StringVar(value="2.0")
        radius_entry = Entry(step5_controls, textvariable=self.radius_var, 
                            width=8, font=("Arial", 9))
        radius_entry.pack(side=LEFT, padx=(5, 15))

        # Search type selection
        Label(step5_controls, text="Search:", font=("Arial", 9)).pack(side=LEFT)
        self.search_type_var = StringVar(value="knn")
        search_type_menu = ttk.Combobox(step5_controls, textvariable=self.search_type_var,
                                       values=["knn", "range"], state="readonly", width=8)
        search_type_menu.pack(side=LEFT, padx=(5, 15))

        # Step 5 action buttons
        knn_btn = Button(step5_controls, text="Search", command=self.knn_search, 
                        bg="lightgreen", font=("Arial", 9, "bold"))
        knn_btn.pack(side=LEFT, padx=5)

        tsne_btn = Button(step5_controls, text="Show t-SNE Plot", command=self.show_tsne, 
                         bg="lightblue", font=("Arial", 9, "bold"))
        tsne_btn.pack(side=LEFT, padx=5)

        compare_btn = Button(step5_controls, text="Compare Methods", command=self.compare_methods, 
                            bg="lightyellow", font=("Arial", 9, "bold"))
        compare_btn.pack(side=LEFT, padx=5)

        # Performance info label
        self.performance_label = Label(step5_frame, text="", fg="darkgreen", font=("Arial", 8))
        self.performance_label.pack(anchor=W, pady=(2, 0))
    
    def auto_load_default_data(self):
        """Try to automatically load the default features database"""
        default_path = "stats/features_database.csv"
        
        if Path(default_path).exists():
            try:
                self.load_features_file(default_path)
                self.status_label.config(text=f"✅ Auto-loaded: {default_path}", fg="green")
            except Exception as e:
                self.status_label.config(text=f"⚠️ Auto-load failed: {str(e)}", fg="orange")
        else:
            self.status_label.config(text="📁 Default data not found. Use 'Load Data' button.", fg="blue")
    
    def load_data(self):
        """Load feature data from file dialog"""
        file_path = filedialog.askopenfilename(
            title="Select Feature Database",
            filetypes=[("CSV files", "*.csv"), ("Pickle files", "*.pkl"), ("All files", "*.*")],
            initialdir="stats"
        )
        
        if file_path:
            try:
                self.load_features_file(file_path)
                self.status_label.config(text=f"✅ Loaded: {Path(file_path).name}", fg="green")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file:\n{str(e)}")
                self.status_label.config(text="❌ Load failed", fg="red")
    
    def load_features_file(self, file_path):
        """Load features from specified file path"""
        file_path = Path(file_path)
        
        if file_path.suffix.lower() == '.csv':
            self.features_df = pd.read_csv(file_path)
        elif file_path.suffix.lower() == '.pkl':
            self.features_df = pd.read_pickle(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        # Identify feature columns (exclude metadata)
        metadata_cols = ['filename', 'filepath', 'category']
        self.feature_columns = [col for col in self.features_df.columns 
                               if col not in metadata_cols]
        
        # Handle missing values
        if self.features_df[self.feature_columns].isnull().any().any():
            self.features_df[self.feature_columns] = self.features_df[self.feature_columns].fillna(0)
            print("⚠️ Filled missing values with 0")
        
        # Handle infinite values  
        inf_mask = np.isinf(self.features_df[self.feature_columns]).any(axis=1)
        if inf_mask.any():
            print(f"⚠️ Found {inf_mask.sum()} rows with infinite values")
            # Replace inf with column max (finite values only)
            for col in self.feature_columns:
                col_data = self.features_df[col]
                if np.isinf(col_data).any():
                    finite_values = col_data[np.isfinite(col_data)]
                    if len(finite_values) > 0:
                        replacement = finite_values.max()
                    else:
                        replacement = 0.0
                    self.features_df.loc[np.isinf(col_data), col] = replacement
        
        # UPDATE: Better mesh selection setup
        if 'filename' in self.features_df.columns:
            # Group by categories for better organization
            categories = sorted(self.features_df['category'].unique())
            
            # Update mesh dropdown with first few meshes as examples
            sample_meshes = self.features_df['filename'].head(20).tolist()
            self.mesh_menu['values'] = sample_meshes
            if sample_meshes:
                self.mesh_var.set(sample_meshes[0])
                
            # Update info about categories
            self.selected_info.config(text=f"📊 {len(categories)} categories, {len(self.features_df)} meshes")        
        # Display basic info
        self.display_data_info()
        
        print(f"✅ Loaded {len(self.features_df)} shapes with {len(self.feature_columns)} features")

    def display_data_info(self):
        """Display basic information about loaded data"""
        if self.features_df is None:
            return
        
        # Group by categories for better display
        category_counts = self.features_df['category'].value_counts()
        
        info_text = f"""DATA SUMMARY
================
Total shapes: {len(self.features_df)}
Features: {len(self.feature_columns)}
Categories: {len(category_counts)}

TOP CATEGORIES:
{category_counts.head(10).to_string()}

FEATURE COLUMNS:
{', '.join(self.feature_columns[:10])}
{'...' if len(self.feature_columns) > 10 else ''}

💡 TIP: Use "Browse by Category" to select meshes by category
"""
        
        self.results_text.delete(1.0, END)
        self.results_text.insert(1.0, info_text)
    
    def compute_stats(self):
        """Compute and display statistical summaries per feature"""
        if self.features_df is None:
            messagebox.showwarning("No Data", "Please load data first!")
            return
        
        try:
            self.status_label.config(text="🔄 Computing statistics...", fg="blue")
            
            # Compute stats for feature columns
            feature_data = self.features_df[self.feature_columns]
            
            stats_summary = {
                'mean': feature_data.mean(),
                'std': feature_data.std(),
                'min': feature_data.min(),
                'max': feature_data.max(),
                'median': feature_data.median()
            };
            
            # Format results - UPDATED: More compact display
            stats_text = "FEATURE STATISTICS\n"
            stats_text += "=" * 85 + "\n"
            stats_text += f"{'Feature':<15} {'Mean':<8} {'Std':<8} {'Min':<8} {'Max':<8} {'Median':<8} {'Range':<12}\n"
            stats_text += "-" * 85 + "\n"
            
            # Show ALL features in compact format
            for feature in self.feature_columns:
                mean_val = stats_summary['mean'][feature]
                std_val = stats_summary['std'][feature]
                min_val = stats_summary['min'][feature]
                max_val = stats_summary['max'][feature]
                median_val = stats_summary['median'][feature]
                range_val = max_val - min_val
                
                # Format with 3 decimal places, but handle very large numbers
                def format_number(val):
                    if abs(val) >= 1e6:  # Very large numbers
                        return f"{val:.2e}"  # Scientific notation
                    elif abs(val) >= 1000:  # Large numbers
                        return f"{val:.0f}"  # No decimals
                    else:  # Normal numbers
                        return f"{val:.3f}"  # 3 decimals
                
                stats_text += f"{feature:<15} "
                stats_text += f"{format_number(mean_val):<8} "
                stats_text += f"{format_number(std_val):<8} "
                stats_text += f"{format_number(min_val):<8} "
                stats_text += f"{format_number(max_val):<8} "
                stats_text += f"{format_number(median_val):<8} "
                stats_text += f"{format_number(range_val):<12}\n"
            
            # Add compact overall summary
            stats_text += "-" * 85 + "\n"
            stats_text += f"SUMMARY: {len(self.feature_columns)} features, {len(self.features_df)} shapes\n"
            stats_text += f"Data range: [{format_number(feature_data.min().min())} to {format_number(feature_data.max().max())}]\n"
            
            # Identify problematic features (infinite, very large values)
            problematic_features = []
            for feature in self.feature_columns:
                max_val = stats_summary['max'][feature]
                if abs(max_val) > 1e6:  # Very large values
                    problematic_features.append(f"{feature} (max: {format_number(max_val)})")
            
            if problematic_features:
                stats_text += f"\n⚠️  Features with extreme values:\n"
                for i, feat in enumerate(problematic_features[:5]):  # Show first 5
                    stats_text += f"   • {feat}\n"
                if len(problematic_features) > 5:
                    stats_text += f"   ... and {len(problematic_features) - 5} more\n"
            
            # Display results
            self.results_text.delete(1.0, END)
            self.results_text.insert(1.0, stats_text)
            
            self.status_label.config(text="✅ Statistics computed successfully", fg="green")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to compute statistics:\n{str(e)}")
            self.status_label.config(text="❌ Statistics computation failed", fg="red")
    
    def compute_distances(self):
        """Compute distances using selected method from dropdown"""
        if self.features_df is None:
            messagebox.showwarning("No Data", "Please load data first!")
            return
        
        if self.mesh_var.get() == "Select mesh...":
            messagebox.showwarning("No Reference", "Please select a reference mesh!")
            return
        
        try:
            # Show selected mesh info first
            self.status_label.config(text="🔄 Analyzing selected mesh...", fg="blue")
            
            info = self.get_selected_mesh_info()
            self.results_text.delete(1.0, END)
            self.results_text.insert(1.0, info + "\n\n🔄 Computing distances...\n")
            self.root.update()
            
            # Get selected distance method
            distance_method = self.distance_var.get()
            
            if distance_method == "advanced_combined":
                # Use advanced retrieval system
                self._compute_advanced_distances()
            else:
                # Use simple GUI methods
                self._compute_simple_distances(distance_method)
                
            self.status_label.config(text="✅ Distances computed successfully", fg="green")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to compute distances:\n{str(e)}")
            self.status_label.config(text="❌ Distance computation failed", fg="red")

    def _compute_advanced_distances(self):
        """Compute distances using advanced retrieval system"""
        query_mesh_path = self.get_query_mesh_path()
        if query_mesh_path is None:
            messagebox.showerror("Error", "Could not find query mesh file!")
            return
        
        # Use advanced retrieval system
        from retrieval import ShapeRetrieval
        retrieval_system = ShapeRetrieval("stats/features_database.csv")
        
        results_df = retrieval_system.search_similar_shapes(
            query_mesh_path=query_mesh_path,
            k=10,
            scalar_weight=0.5,
            exclude_self=False,
            logs=True
        )
        
        if results_df is None:
            messagebox.showerror("Error", "Failed to compute distances!")
            return
        
        # Store results in GUI format WITH detailed distance info
        self.current_results = []
        for i, (_, row) in enumerate(results_df.iterrows()):
            self.current_results.append({
                'filename': row['filename'],
                'category': row['category'],
                'distance': row['combined_distance'],
                'index': row['database_index'],
                'method': 'advanced_combined',
                'scalar_distance': row['scalar_distance'],  # Store detailed distances
                'histogram_distance': row['histogram_distance']
            })
        
        # Display results
        self._display_advanced_results(results_df)

    def _compute_simple_distances(self, distance_method):
        """Compute distances using simple methods with feature standardization"""
        # Get reference mesh
        ref_filename = self.mesh_var.get()
        ref_idx = self.features_df[self.features_df['filename'] == ref_filename].index[0]
        
        # STANDARDIZE FEATURES FIRST
        feature_matrix = self.features_df[self.feature_columns].values
        
        # Z-score standardization
        means = np.mean(feature_matrix, axis=0)
        stds = np.std(feature_matrix, axis=0)
        stds[stds == 0] = 1  # Avoid division by zero
        
        standardized_matrix = (feature_matrix - means) / stds
        ref_features_std = standardized_matrix[ref_idx]
        
        # Compute distances on standardized features
        distances = []
        
        for i, features_std in enumerate(standardized_matrix):
            if distance_method == "euclidean":
                dist = euclidean(ref_features_std, features_std)
            elif distance_method == "cosine":
                # Handle zero vectors
                ref_norm = np.linalg.norm(ref_features_std)
                feat_norm = np.linalg.norm(features_std)
                if ref_norm == 0 or feat_norm == 0:
                    dist = 1.0
                else:
                    dist = cosine(ref_features_std, features_std)
            elif distance_method == "manhattan":
                dist = cityblock(ref_features_std, features_std)
            else:
                dist = euclidean(ref_features_std, features_std)
        
            distances.append({
                'filename': self.features_df.iloc[i]['filename'],
                'category': self.features_df.iloc[i]['category'] if 'category' in self.features_df.columns else 'Unknown',
                'distance': dist,
                'index': i,
                'method': distance_method
            })
    
        # Sort and store results
        distances.sort(key=lambda x: x['distance'])
        self.current_results = distances
        self._display_simple_results(distances, distance_method)

    def _display_simple_results(self, distances, method):
        """Display results from simple distance methods"""
        mesh_info = self.get_selected_mesh_info()
        
        results_text = mesh_info + "\n\n"
        results_text += "🎯 DISTANCE COMPUTATION RESULTS\n"
        results_text += "=" * 70 + "\n"
        results_text += f"Distance Method: {method.upper()}\n"
        results_text += f"Total Comparisons: {len(distances)}\n\n"
        
        results_text += f"TOP 10 MOST SIMILAR MESHES:\n"
        results_text += f"{'-' * 70}\n"
        results_text += f"{'Rank':<4} {'Distance':<15} {'Category':<15} {'Filename':<20}\n"
        results_text += f"{'-' * 70}\n"
        
        for i, result in enumerate(distances[:10]):
            results_text += f"{i+1:<4} {result['distance']:<15.6f} {result['category']:<15} {result['filename']:<20}\n"
        
        # Add statistics
        all_distances = [r['distance'] for r in distances]
        results_text += f"\nDISTANCE STATISTICS:\n"
        results_text += f"Min: {min(all_distances):.6f}\n"
        results_text += f"Max: {max(all_distances):.6f}\n"
        results_text += f"Mean: {np.mean(all_distances):.6f}\n"
        
        self.results_text.delete(1.0, END)
        self.results_text.insert(1.0, results_text)

    def _display_advanced_results(self, results_df):
        """Display results from advanced retrieval system"""
        mesh_info = self.get_selected_mesh_info()
        
        results_text = mesh_info + "\n\n"
        results_text += "🎯 DISTANCE COMPUTATION RESULTS\n"
        results_text += "=" * 80 + "\n"
        results_text += f"Distance Method: ADVANCED COMBINED (Scalar + Histogram)\n"
        results_text += f"Total Comparisons: {len(results_df)}\n\n"
        
        results_text += f"TOP 10 MOST SIMILAR MESHES:\n"
        results_text += f"{'-' * 80}\n"
        results_text += f"{'Rank':<4} {'Distance':<12} {'Scalar':<12} {'Histogram':<12} {'Category':<15} {'Filename':<20}\n"
        results_text += f"{'-' * 80}\n"
        
        for i, (_, row) in enumerate(results_df.head(10).iterrows()):
            results_text += f"{i+1:<4} {row['combined_distance']:<12.4f} {row['scalar_distance']:<12.4f} {row['histogram_distance']:<12.4f} {row['category']:<15} {row['filename']:<20}\n"
        
        self.results_text.delete(1.0, END)
        self.results_text.insert(1.0, results_text)
    
    def visualize_results(self):
        """Visualize results using visualize_shape_retrieval for ALL distance methods"""
        # Auto-compute distances if not done yet
        if self.current_results is None:
            print("No distances computed yet. Computing automatically...")
            self.compute_distances()
            
            # Check if computation was successful
            if self.current_results is None:
                messagebox.showerror("Error", "Failed to compute distances automatically!")
                return
        
        try:
            self.status_label.config(text="🔄 Loading 3D visualization...", fg="blue")
            
            query_mesh_path = self.get_query_mesh_path()
            if query_mesh_path is None:
                messagebox.showerror("Error", "Could not find query mesh file!")
                return
            
            # Get selected distance method
            distance_method = self.distance_var.get()
            
            # Convert current_results to DataFrame format expected by visualize_shape_retrieval
            results_df = self._convert_results_to_dataframe()
            
            if results_df is None:
                messagebox.showerror("Error", "Failed to prepare visualization data!")
                return
            
            # Use the unified visualization function for ALL methods
            print(f"🎯 Visualizing with {distance_method} method: {query_mesh_path}")
            
            from visualization import visualize_shape_retrieval
            
            # Create visualization using the same function for all methods
            fig = visualize_shape_retrieval(
                query_mesh_path=query_mesh_path,
                results_df=results_df,
                max_display=6,  # Show top 6 similar shapes
                save_path=f"img/gui_{distance_method}_result.png",
                show_scores=True
            )
            
            if fig is not None:
                self.status_label.config(text=f"✅ {distance_method.upper()} visualization opened!", fg="green")
                self._update_viz_results_display(results_df, distance_method.upper(), query_mesh_path)
            else:
                self.status_label.config(text="❌ Visualization failed", fg="red")
            
        except Exception as e:
            error_msg = f"Failed to create visualization:\n{str(e)}"
            messagebox.showerror("Error", error_msg)
            self.status_label.config(text="❌ Visualization failed", fg="red")
            print(f"Visualization error: {e}")

    def _convert_results_to_dataframe(self):
        """Convert current_results to DataFrame format expected by visualize_shape_retrieval"""
        if not self.current_results:
            return None
        
        try:
            distance_method = self.distance_var.get();
            
            # Check if this is from KNN search
            if self.current_results and 'method' in self.current_results[0] and 'knn' in self.current_results[0]['method']:
                # KNN results - convert to expected format
                data = []
                for result in self.current_results:
                    # Find the original row to get filepath
                    orig_row = self.features_df[self.features_df['filename'] == result['filename']].iloc[0]
                    
                    data.append({
                        'database_index': result['index'],
                        'filename': result['filename'],
                        'filepath': orig_row['filepath'],
                        'category': result['category'],
                        'combined_distance': result['distance'],
                        'scalar_distance': result['distance'],  # KNN uses Euclidean distance
                        'histogram_distance': 0.0  # KNN doesn't separate histogram
                    })
        
            elif distance_method == "advanced_combined":
                # Existing advanced method code...
                data = []
                for result in self.current_results:
                    orig_row = self.features_df[self.features_df['filename'] == result['filename']].iloc[0]
                    
                    data.append({
                        'database_index': result['index'],
                        'filename': result['filename'],
                        'filepath': orig_row['filepath'],
                        'category': result['category'],
                        'combined_distance': result['distance'],
                        'scalar_distance': result.get('scalar_distance', result['distance']),
                        'histogram_distance': result.get('histogram_distance', 0.0)
                    })
            else:
                # Existing simple method code...
                data = []
                for result in self.current_results:
                    orig_row = self.features_df[self.features_df['filename'] == result['filename']].iloc[0]
                    
                    data.append({
                        'database_index': result['index'],
                        'filename': result['filename'],
                        'filepath': orig_row['filepath'],
                        'category': result['category'],
                        'combined_distance': result['distance'],
                        'scalar_distance': result['distance'],
                        'histogram_distance': 0.0
                    })
        
            results_df = pd.DataFrame(data)
            return results_df
            
        except Exception as e:
            print(f"Error converting results to DataFrame: {e}")
            return None

    def _update_viz_results_display(self, results_df, method_name, query_path):
        """Update text area with visualization results"""
        results_text = f"3D SHAPE RETRIEVAL VISUALIZATION\n"
        results_text += "=" * 60 + "\n\n"
        results_text += f"Query Mesh: {Path(query_path).name}\n"
        results_text += f"Query Category: {Path(query_path).parent.name}\n"
        results_text += f"Distance Method: {method_name}\n"
        results_text += f"Visualization: External window (using visualize_shape_retrieval)\n\n"
        
        results_text += "TOP SIMILAR SHAPES VISUALIZED:\n"
        results_text += "-" * 50 + "\n"
        
        for i, (_, row) in enumerate(results_df.head(6).iterrows(), 1):
            results_text += f"{i}. {row['filename']} ({row['category']})\n"
            if method_name == "ADVANCED_COMBINED":
                results_text += f"   Combined: {row['combined_distance']:.4f} | "
                results_text += f"Scalar: {row['scalar_distance']:.4f} | "
                results_text += f"Histogram: {row['histogram_distance']:.4f}\n"
            else:
                results_text += f"   Distance: {row['combined_distance']:.4f}\n"
        
        results_text += f"\n💡 Close the visualization window when done viewing"
        
        self.results_text.delete(1.0, END)
        self.results_text.insert(1.0, results_text)

    def show_mesh_info(self):
        """Display current mesh information and all its features"""
        if self.features_df is None:
            messagebox.showwarning("No Data", "Please load data first!")
            return
        
        if self.mesh_var.get() == "Select mesh...":
            messagebox.showwarning("No Selection", "Please select a mesh first!")
            return
        
        try:
            self.status_label.config(text="📋 Displaying mesh information...", fg="blue")
            
            selected_filename = self.mesh_var.get()
            mesh_row = self.features_df[self.features_df['filename'] == selected_filename].iloc[0]
            
            # Basic mesh info
            info_text = f"""CURRENT MESH INFORMATION
{'=' * 60}
📄 Filename: {mesh_row['filename']}
📁 Category: {mesh_row['category']}
📂 Path: {mesh_row['filepath']}
🎯 Database Index: {mesh_row.name}

ALL 57 FEATURES:
{'=' * 60}
"""
            
            # Display all features in compact format
            feature_count = 0
            for feature in self.feature_columns:
                value = mesh_row[feature]
                
                # Format the value
                if abs(value) >= 1e6 or abs(value) <= 1e-6:
                    formatted_value = f"{value:.3e}"
                elif abs(value) >= 1000:
                    formatted_value = f"{value:.0f}"
                else:
                    formatted_value = f"{value:.4f}"
                
                # Print 3 features per line for compact display
                if feature_count % 3 == 0:
                    info_text += f"\n{feature:<18}: {formatted_value:<12}"
                else:
                    info_text += f" | {feature:<18}: {formatted_value:<12}"
                
                feature_count += 1
            
            info_text += f"\n\n{'=' * 60}"
            info_text += f"\nTotal Features: {len(self.feature_columns)}"
            info_text += f"\n💡 Use 'Compute Distances' to find similar shapes"
            
            # Display the results
            self.results_text.delete(1.0, END)
            self.results_text.insert(1.0, info_text)
            
            self.status_label.config(text="✅ Mesh information displayed", fg="green")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to display mesh info:\n{str(e)}")
            self.status_label.config(text="❌ Failed to display mesh info", fg="red")

    def visualize_single_mesh(self):
        """Visualize the selected mesh in a new window"""
        if self.features_df is None:
            messagebox.showwarning("No Data", "Please load data first!")
            return
        
        if self.mesh_var.get() == "Select mesh...":
            messagebox.showwarning("No Selection", "Please select a mesh first!")
            return
        
        try:
            self.status_label.config(text="🔄 Loading mesh visualization...", fg="blue")
            
            # Get query mesh path
            query_mesh_path = self.get_query_mesh_path()
            
            if query_mesh_path is None:
                messagebox.showerror("Error", "Could not find mesh file!")
                return
            
            # Import your existing visualization function
            from plots import show_mesh_simple  # or whatever function you have
            
            print(f"🎯 Visualizing mesh: {query_mesh_path}")
            
            # Show the single mesh
            show_mesh_simple(query_mesh_path)
            
            self.status_label.config(text="✅ Mesh visualization opened!", fg="green")
            
            # Update text area with mesh info
            selected_filename = self.mesh_var.get()
            mesh_row = self.features_df[self.features_df['filename'] == selected_filename].iloc[0]
            
            viz_text = f"""MESH VISUALIZATION
{'=' * 40}
📄 Filename: {mesh_row['filename']}
📁 Category: {mesh_row['category']}
📂 Path: {query_mesh_path}

✅ Mesh displayed in new matplotlib window
💡 Close the window when done viewing
"""
            
            self.results_text.delete(1.0, END)
            self.results_text.insert(1.0, viz_text)
            
        except Exception as e:
            error_msg = f"Failed to visualize mesh:\n{str(e)}"
            messagebox.showerror("Error", error_msg)
            self.status_label.config(text="❌ Mesh visualization failed", fg="red")
            print(f"Visualization error: {e}")

    def browse_by_category(self):
        """Open category browser window"""
        if self.features_df is None:
            messagebox.showwarning("No Data", "Please load data first!")
            return
            
        # Create category browser window
        browser_window = Toplevel(self.root)
        browser_window.title("Browse Meshes by Category")
        browser_window.geometry("600x500")
        browser_window.transient(self.root)
        browser_window.grab_set()
        
        # Create treeview for hierarchical display
        tree_frame = Frame(browser_window)
        tree_frame.pack(fill=BOTH, expand=True, padx=10, pady=10)
        
        # Treeview with scrollbar
        tree_scroll = Scrollbar(tree_frame)
        tree_scroll.pack(side=RIGHT, fill=Y)
        
        tree = ttk.Treeview(tree_frame, yscrollcommand=tree_scroll.set)
        tree_scroll.config(command=tree.yview)
        
        # Configure treeview columns
        tree['columns'] = ("count", "filename")
        tree.column("#0", width=200, minwidth=150)  # Category column
        tree.column("count", width=80, minwidth=50)
        tree.column("filename", width=300, minwidth=200)
        
        tree.heading("#0", text="Category", anchor=W)
        tree.heading("count", text="Count", anchor=CENTER)
        tree.heading("filename", text="Selected Mesh", anchor=W)
        
        tree.pack(fill=BOTH, expand=True)
        
        # Populate tree with categories
        categories = self.features_df.groupby('category')
        
        category_items = {}  # Store category tree items
        
        for category_name, category_data in categories:
            # Add category parent node
            category_item = tree.insert("", "end", text=category_name, 
                                      values=(len(category_data), ""), open=False)
            category_items[category_name] = category_item
            
            # Add mesh files under category
            for idx, row in category_data.iterrows():
                mesh_item = tree.insert(category_item, "end", text="  " + row['filename'], 
                                      values=("", row['filename']))
        
        # Selection handler
        def on_tree_select(event):
            selection = tree.selection()
            if selection:
                item = tree.item(selection[0])
                # Check if it's a mesh file (has filename in values)
                if item['values'] and len(item['values']) > 1 and item['values'][1]:
                    selected_filename = item['values'][1]
                    
                    # Update main GUI
                    self.mesh_var.set(selected_filename)
                    
                    # Update info label
                    mesh_row = self.features_df[self.features_df['filename'] == selected_filename].iloc[0]
                    self.selected_info.config(text=f"📁 {mesh_row['category']} | 📄 {selected_filename}")
                    
                    # Close browser window
                    browser_window.destroy()
                    
                    # Show success message
                    self.status_label.config(text=f"✅ Selected: {selected_filename}", fg="green")
        
        tree.bind("<<TreeviewSelect>>", on_tree_select)
        
        # Instructions
        instructions = Label(browser_window, 
                           text="🔍 Expand categories and click on a mesh file to select it",
                           font=("Arial", 10), fg="darkblue")
        instructions.pack(pady=5)
        
        # Buttons frame
        btn_frame = Frame(browser_window)
        btn_frame.pack(fill=X, padx=10, pady=5)
        
        # Close button
        close_btn = Button(btn_frame, text="Close", command=browser_window.destroy)
        close_btn.pack(side=RIGHT)
        
        # Expand all button
        def expand_all():
            for category_item in category_items.values():
                tree.item(category_item, open=True)
        
        expand_btn = Button(btn_frame, text="Expand All", command=expand_all)
        expand_btn.pack(side=LEFT)

    def get_selected_mesh_info(self):
        """Get detailed information about the selected mesh"""
        selected_filename = self.mesh_var.get()
        
        if selected_filename == "Select mesh..." or not selected_filename:
            return "No mesh selected"
            
        # Find the mesh in the database
        mesh_row = self.features_df[self.features_df['filename'] == selected_filename]
        
        if len(mesh_row) == 0:
            return f"Mesh {selected_filename} not found in database"
        
        row = mesh_row.iloc[0]
        info = f"""SELECTED MESH INFORMATION
{'=' * 40}
📄 Filename: {row['filename']}
📁 Category: {row['category']}
📂 Path: {row['filepath']}
🎯 Database Index: {mesh_row.index[0]}

FEATURE PREVIEW:
area: {row.get('area', 'N/A'):.3f}
volume: {row.get('volume', 'N/A'):.3f}
compactness: {row.get('compactness', 'N/A'):.2e}
diameter: {row.get('diameter', 'N/A'):.3f}

"""
        return info

    def get_query_mesh_path(self):
        """Convert selected filename to full mesh path"""
        selected_filename = self.mesh_var.get()
        
        if selected_filename == "Select mesh..." or not selected_filename:
            return None
            
        # Find the mesh in the database
        mesh_row = self.features_df[self.features_df['filename'] == selected_filename]
        
        if len(mesh_row) == 0:
            print(f"Mesh {selected_filename} not found in database")
            return None
            
        # Get filepath from database
        filepath = mesh_row.iloc[0]['filepath']
        
        # Convert to Path object and check if exists
        mesh_path = Path(filepath)
        
        if not mesh_path.exists():
            # Try to construct path using normalized_data folder
            category = mesh_row.iloc[0]['category']
            alternative_path = Path("normalized_data") / category / selected_filename
            
            if alternative_path.exists():
                return str(alternative_path)
            else:
                print(f"Mesh file not found: {mesh_path}")
                print(f"Also tried: {alternative_path}")
                return None
        
        return str(mesh_path)

    def export_results(self):
        """Export current results to file"""
        if self.current_results is None:
            messagebox.showwarning("No Results", "Please compute distances first!")
            return
        
        try:
            file_path = filedialog.asksaveasfilename(
                title="Save Results",
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("Text files", "*.txt"), ("All files", "*.*")]
            )
            
            if file_path:
                self.status_label.config(text="🔄 Exporting results...", fg="blue")
                
                if file_path.endswith('.csv'):
                    results_df = pd.DataFrame(self.current_results)
                    results_df.to_csv(file_path, index=False)
                else:
                    with open(file_path, 'w') as f:
                        f.write(f"Distance Computation Results\n")
                        f.write(f"Generated by 3D Shape Retrieval Tool\n")
                        f.write(f"{'=' * 50}\n\n")
                        f.write(f"Reference Mesh: {self.mesh_var.get()}\n")
                        f.write(f"Distance Metric: {self.distance_var.get().upper()}\n")
                        f.write(f"Total Comparisons: {len(self.current_results)}\n\n")
                        
                        f.write(f"{'Rank':<6} {'Distance':<15} {'Category':<15} {'Filename':<30}\n")
                        f.write(f"{'-' * 70}\n")
                        
                        for i, result in enumerate(self.current_results):
                            f.write(f"{i+1:<6} {result['distance']:<15.6f} {result['category']:<15} {result['filename']:<30}\n")
                
                self.status_label.config(text=f"✅ Results exported to: {Path(file_path).name}", fg="green")
                messagebox.showinfo("Export Complete", f"Results saved to:\n{file_path}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export results:\n{str(e)}")
            self.status_label.config(text="❌ Export failed", fg="red")

    def initialize_step5_engines(self):
        """Initialize KNN engine and dimensionality reducer"""
        if self.step5_initialized:
            return True
        
        try:
            self.status_label.config(text="🔄 Initializing Step 5 engines...", fg="blue")
            
            # Initialize KNN engine
            self.knn_engine = KNNEngine("step5_data")
            if not self.knn_engine.load_processed_features():
                self.status_label.config(text="❌ Failed to load KNN features", fg="red")
                return False
            
            # BUILD INDEX WITHOUT DOUBLE NORMALIZATION
            if not self.knn_engine.build_index(n_neighbors=50, metric='euclidean', use_step4_normalization=False):
                self.status_label.config(text="❌ Failed to build KNN index", fg="red")
                return False
            
            # Initialize dimensionality reducer
            self.dimensionality_reducer = DimensionalityReducer("step5_data")
            if not self.dimensionality_reducer.load_processed_features():
                self.status_label.config(text="❌ Failed to load DR features", fg="red")
                return False
            
            self.step5_initialized = True
            self.performance_label.config(text="✅ Step 5 engines ready - KNN index built with original normalization")
            return True
            
        except Exception as e:
            self.status_label.config(text=f"❌ Step 5 initialization failed", fg="red")
            print(f"Step 5 initialization error: {e}")
            return False

    def knn_search(self):
        """Perform KNN or Range search using Step 5 engine"""
        if self.features_df is None:
            messagebox.showwarning("No Data", "Please load data first!")
            return
        
        if self.mesh_var.get() == "Select mesh...":
            messagebox.showwarning("No Reference", "Please select a reference mesh!")
            return
        
        # Initialize Step 5 engines if needed
        if not self.initialize_step5_engines():
            return
        
        try:
            # Get search type FIRST
            search_type = self.search_type_var.get()
            
            self.status_label.config(text=f"🔄 Performing {search_type.upper()} search...", fg="blue")
            
            # Get query filename and find its index in metadata
            query_filename = self.mesh_var.get()
            query_index = None
            
            for i, meta in enumerate(self.knn_engine.metadata):
                if meta['filename'] == query_filename:
                    query_index = i
                    break
            
            if query_index is None:
                messagebox.showerror("Error", f"Query mesh '{query_filename}' not found in KNN database!")
                return
            
            # FIX: Perform DIFFERENT searches based on type
            if search_type == "knn":
                k_value = int(self.k_var.get())
                results_df = self.knn_engine.query_knn(query_index, k=k_value)
                search_info = f"K-NN (K={k_value})"
                print(f"🔍 Performed KNN search with K={k_value}")
            elif search_type == "range":
                radius_value = float(self.radius_var.get())
                results_df = self.knn_engine.query_range(query_index, radius=radius_value)
                search_info = f"Range (R={radius_value})"
                print(f"🎯 Performed Range search with R={radius_value}")
            else:
                messagebox.showerror("Error", f"Unknown search type: {search_type}")
                return
            
            if results_df is None:
                messagebox.showerror("Error", f"{search_type.upper()} search failed!")
                return
            
            # Store results for visualization
            self.current_results = []
            for _, row in results_df.iterrows():
                self.current_results.append({
                    'filename': row['filename'],
                    'category': row['category'],
                    'distance': row['distance'],
                    'index': row['database_index'],
                    'method': f'knn_{search_type}'  # This will be 'knn_knn' or 'knn_range'
                })
            
            # Display results
            self._display_knn_results(results_df, search_info, query_filename)
            
            # Update performance info
            query_time = results_df.iloc[0]['query_time'] if len(results_df) > 0 else 0
            self.performance_label.config(text=f"⚡ {search_info} completed in {query_time:.4f}s - {len(results_df)} results")
            
            self.status_label.config(text=f"✅ {search_type.upper()} search completed successfully", fg="green")
            
        except Exception as e:
            messagebox.showerror("Error", f"{search_type.upper()} search failed:\n{str(e)}")
            self.status_label.config(text=f"❌ {search_type.upper()} search failed", fg="red")

    def _display_knn_results(self, results_df, search_info, query_filename):
        """Display KNN/Range search results"""
        mesh_info = self.get_selected_mesh_info()
        
        # ENSURE RESULTS ARE SORTED BY DISTANCE
        results_df = results_df.sort_values('distance').reset_index(drop=True)
        results_df['rank'] = range(1, len(results_df) + 1)
        
        # UPDATE: Better title based on search type
        search_type = self.search_type_var.get().upper()
        
        results_text = mesh_info + "\n\n"
        results_text += f"🚀 {search_type} SEARCH RESULTS (Step 5)\n"
        results_text += "=" * 70 + "\n"
        results_text += f"Search Type: {search_info}\n"
        results_text += f"Query: {query_filename}\n"
        results_text += f"Results Found: {len(results_df)}\n"
        
        if len(results_df) > 0:
            query_time = results_df.iloc[0]['query_time']
            results_text += f"Query Time: {query_time:.4f} seconds\n\n"
        
        # ADD EXPLANATION OF DIFFERENCE
        if search_type == "KNN":
            results_text += f"📋 K-NN finds exactly {self.k_var.get()} most similar shapes\n\n"
        else:  # RANGE
            results_text += f"📋 Range search finds ALL shapes within distance {self.radius_var.get()}\n\n"
        
        results_text += f"TOP RESULTS:\n"
        results_text += f"{'-' * 70}\n"
        results_text += f"{'Rank':<4} {'Distance':<12} {'Category':<15} {'Filename':<25}\n"
        results_text += f"{'-' * 70}\n"
        
        # Display sorted results
        for _, row in results_df.head(15).iterrows():
            results_text += f"{row['rank']:<4} {row['distance']:<12.4f} {row['category']:<15} {row['filename']:<25}\n"
        
        if len(results_df) > 15:
            results_text += f"... and {len(results_df) - 15} more results\n"
        
        results_text += f"\n💡 Use 'View Similar' to visualize these results in 3D"
        
        self.results_text.delete(1.0, END)
        self.results_text.insert(1.0, results_text)

    def show_tsne(self):
        """Show t-SNE visualization - Fixed for macOS threading"""
        if self.features_df is None:
            messagebox.showwarning("No Data", "Please load data first!")
            return
        
        def update_progress(message, step=0, total=5):
            """Update progress in results text"""
            progress_bar = "█" * step + "░" * (total - step)
            progress_text = f"""t-SNE VISUALIZATION PROGRESS
{'=' * 40}
{message}

Progress: [{progress_bar}] {step}/{total}

Status: {"Computing..." if step < total else "Complete!"}
"""
            self.results_text.delete(1.0, END)
            self.results_text.insert(1.0, progress_text)
            self.root.update()  # Force GUI update
        
        try:
            update_progress("Initializing Step 5 engines...", 1)
            self.status_label.config(text="🔄 Loading t-SNE visualization...", fg="blue")
            
            # Initialize dimensionality reducer if needed
            if not self.initialize_step5_engines():
                return
            
            update_progress("Checking for existing embedding...", 2)
            
            # Check if embedding already exists
            from pathlib import Path
            embedding_file = Path("step5_data/tsne_embedding_2d.npy")
            
            if not embedding_file.exists():
                update_progress("Computing t-SNE embedding (1-2 minutes)...", 3)
                self.status_label.config(text="🔄 Computing t-SNE embedding...", fg="blue")
                
                # Compute t-SNE in main thread (blocking but safe)
                if not self.dimensionality_reducer.compute_tsne(perplexity=30, n_iter=1000):
                    self.status_label.config(text="❌ t-SNE computation failed", fg="red")
                    return
                
                self.dimensionality_reducer.save_embedding()
            else:
                # Load existing embedding
                import numpy as np
                self.dimensionality_reducer.X_2d = np.load(embedding_file)
                print("✅ Loaded existing t-SNE embedding")
            
            update_progress("Creating interactive plot...", 4)
            
            # Create interactive plot (on main thread)
            if not self.dimensionality_reducer.create_interactive_plot():
                self.status_label.config(text="❌ t-SNE plot creation failed", fg="red")
                return
            
            update_progress("Opening visualization...", 5)
            
            # Show the plot (on main thread)
            self.dimensionality_reducer.show_plot()
            
            self.status_label.config(text="✅ t-SNE visualization opened!", fg="green")
            self.performance_label.config(text="🎨 t-SNE plot: Hover points for details, right-click for KNN highlighting")
            
            # Get actual statistics from your dimensionality reducer
            if hasattr(self.dimensionality_reducer, 'X_2d') and self.dimensionality_reducer.X_2d is not None:
                embedding_shape = self.dimensionality_reducer.X_2d.shape
                n_points = embedding_shape[0]
                n_dims = embedding_shape[1]
            else:
                n_points = len(self.features_df) if self.features_df is not None else 0
                n_dims = 2
            
            # Get actual category count
            if self.features_df is not None:
                n_categories = len(self.features_df['category'].unique())
                category_counts = self.features_df['category'].value_counts()
                top_categories = category_counts.head(5)
            else:
                n_categories = 0
                top_categories = []
            
            # Get actual feature count
            n_features = len(self.feature_columns) if self.feature_columns else 0
            
            # Build dynamic results text
            results_text = f"""t-SNE DIMENSIONALITY REDUCTION VISUALIZATION
======================================================

🎨 Interactive 2D scatterplot opened in new window
📊 {n_points} shapes reduced from {n_features}D to {n_dims}D using t-SNE
🌈 Points colored by category ({n_categories} categories)

CURRENT DATABASE STATISTICS:
• Total shapes: {n_points:,}
• Feature dimensions: {n_features}
• Categories: {n_categories}
• Selected mesh: {self.mesh_var.get()}

TOP CATEGORIES IN DATABASE:
"""
            
            # Add top categories dynamically
            for i, (category, count) in enumerate(top_categories.items(), 1):
                results_text += f"{i}. {category} - {count} shapes\n"
            
            results_text += f"""
INTERACTION:
• Hover over points to see shape details
• Right-click points to highlight K nearest neighbors
• Close window when done exploring

EMBEDDING QUALITY:
• Perplexity: 30 (controls local neighborhood size)
• Iterations: 1000 (convergence parameter)
• Embedding range: X[{self.dimensionality_reducer.X_2d[:, 0].min():.1f}, {self.dimensionality_reducer.X_2d[:, 0].max():.1f}], Y[{self.dimensionality_reducer.X_2d[:, 1].min():.1f}, {self.dimensionality_reducer.X_2d[:, 1].max():.1f}]

💡 Well-separated clusters indicate good feature quality!
Currently viewing: {self.mesh_var.get() if self.mesh_var.get() != "Select mesh..." else "No mesh selected"}
"""
            
            self.results_text.delete(1.0, END)
            self.results_text.insert(1.0, results_text)
            
        except Exception as e:
            self.status_label.config(text="❌ t-SNE visualization failed", fg="red")
            print(f"t-SNE error: {e}")
            messagebox.showerror("Error", f"t-SNE visualization failed:\n{str(e)}")

    def compare_methods(self):
        """Compare Step 4 vs Step 5 methods side by side"""
        if self.features_df is None:
            messagebox.showwarning("No Data", "Please load data first!")
            return
        
        if self.mesh_var.get() == "Select mesh...":
            messagebox.showwarning("No Reference", "Please select a reference mesh!")
            return
        
        try:
            # SET DETERMINISTIC BEHAVIOR
            import numpy as np
            np.random.seed(42)  # Fixed seed for consistency
            
            self.status_label.config(text="🔄 Comparing Step 4 vs Step 5 methods...", fg="blue")
            
            query_filename = self.mesh_var.get()
            print(f"🎯 Deterministic comparison for: {query_filename}")
            
            # Step 4: Compute using your custom distance (advanced_combined)
            old_distance_var = self.distance_var.get()
            self.distance_var.set("advanced_combined")
            
            step4_start = time.time()
            self._compute_advanced_distances()
            step4_time = time.time() - step4_start
            step4_results = self.current_results.copy() if self.current_results else []
            
            # Step 5: Compute using KNN WITH STEP 4 NORMALIZATION
            if not self.initialize_step5_engines():
                return
            
            # Find query index
            query_index = None
            for i, meta in enumerate(self.knn_engine.metadata):
                if meta['filename'] == query_filename:
                    query_index = i
                    break
            
            if query_index is None:
                messagebox.showerror("Error", "Query mesh not found in KNN database!")
                return
            
            step5_start = time.time()
            knn_results_df = self.knn_engine.query_knn(query_index, k=10)
            step5_time = time.time() - step5_start
            
            # Restore original distance setting
            self.distance_var.set(old_distance_var)
            
            # Display comparison
            self._display_comparison_results(step4_results, knn_results_df, step4_time, step5_time, query_filename)
            
            self.status_label.config(text="✅ Method comparison completed", fg="green")
            speedup = step4_time / step5_time if step5_time > 0 else float('inf')
            self.performance_label.config(text=f"⚡ Speedup: {speedup:.1f}x faster (Step 4: {step4_time:.3f}s vs Step 5: {step5_time:.4f}s)")
            
            # ADD DEBUG INFO
            print(f"🔍 Debug Info:")
            print(f"  Step 4 distance range: {min([r['distance'] for r in step4_results]):.4f} - {max([r['distance'] for r in step4_results]):.4f}")
            if len(knn_results_df) > 0:
                print(f"  Step 5 distance range: {knn_results_df['distance'].min():.4f} - {knn_results_df['distance'].max():.4f}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Method comparison failed:\n{str(e)}")
            self.status_label.config(text="❌ Method comparison failed", fg="red")

    def _display_comparison_results(self, step4_results, step5_results_df, step4_time, step5_time, query_filename):
        """Display side-by-side comparison of Step 4 vs Step 5 results"""
        results_text = f"""METHOD COMPARISON: Step 4 vs Step 5
{'=' * 80}
Query: {query_filename}
Step 4 Time: {step4_time:.3f} seconds (Custom Advanced Distance)
Step 5 Time: {step5_time:.4f} seconds (KNN Euclidean Distance)
Speedup: {step4_time/step5_time:.1f}x faster with Step 5

TOP 10 RESULTS COMPARISON:
{'=' * 80}
{'Rank':<4} {'Step 4 (Advanced)':<35} {'Step 5 (KNN)':<35}
{'    ':<4} {'Distance':<12} {'Category':<15} {'Distance':<12} {'Category':<15}
{'-' * 80}
"""
        
        # Compare top 10 results
        for i in range(min(10, len(step4_results), len(step5_results_df))):
            step4_result = step4_results[i] if i < len(step4_results) else None
            step5_row = step5_results_df.iloc[i] if i < len(step5_results_df) else None
            
            rank = i + 1
            
            if step4_result:
                step4_dist = f"{step4_result['distance']:.4f}"
                step4_cat = step4_result['category'][:13]  # Truncate long categories
            else:
                step4_dist = "N/A"
                step4_cat = "N/A"
            
            if step5_row is not None:
                step5_dist = f"{step5_row['distance']:.4f}"
                step5_cat = step5_row['category'][:13]
            else:
                step5_dist = "N/A"
                step5_cat = "N/A"
            
            results_text += f"{rank:<4} {step4_dist:<12} {step4_cat:<15} {step5_dist:<12} {step5_cat:<15}\n"
        
        results_text += f"\nASSESSMENT:\n"
        results_text += f"{'-' * 40}\n"
        
        # Simple overlap analysis
        if step4_results and len(step5_results_df) > 0:
            step4_filenames = {r['filename'] for r in step4_results[:10]}
            step5_filenames = set(step5_results_df.head(10)['filename'])
            overlap = len(step4_filenames.intersection(step5_filenames))
            
            results_text += f"Top 10 Result Overlap: {overlap}/10 shapes in common\n"
            results_text += f"Performance Gain: {step4_time/step5_time:.1f}x faster with KNN\n"
            
            if overlap >= 7:
                results_text += "✅ High similarity: Methods agree well on shape similarity\n"
            elif overlap >= 4:
                results_text += "⚠️  Moderate similarity: Some differences in ranking\n"
            else:
                results_text += "❌ Low similarity: Significant differences between methods\n"
        
        results_text += f"\n💡 Use 'View Similar' to visualize results from either method"
        
        self.results_text.delete(1.0, END)
        self.results_text.insert(1.0, results_text)


def main():
    """Main function to run the GUI application"""
    root = Tk()
    app = ShapeRetrievalGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()