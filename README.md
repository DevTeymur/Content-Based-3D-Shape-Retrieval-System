# 3D Shape Retrieval System

A content-based 3D shape retrieval system using geometric features and shape descriptors.

## Overview

This system extracts geometric features from 3D meshes and enables similarity-based retrieval using k-nearest neighbors (KNN). It includes preprocessing, feature extraction, normalization, dimensionality reduction, and evaluation components.

## Requirements

```bash
pip install trimesh numpy pandas scikit-learn matplotlib open3d pymeshlab
```

## Quick Start - Testing the System

**If you want to quickly test the system with pre-processed data:**

```bash
python3 shape_retrieval_gui.py
```

The GUI will automatically load the feature database and allow you to:
- Search meshes by filename
- View mesh statistics and visualizations
- Query similar shapes using KNN
- Explore retrieval results interactively

**Note:** This assumes the feature database has already been created. If you get errors, follow the full pipeline below.

---

## Full Pipeline - Processing from Scratch

If you need to process the data from the beginning or want to understand the complete workflow:

### 1. Data Preprocessing

**Resample meshes to uniform vertex count (~7,500 vertices):**
```bash
python3 resample.py
```
- Input: data folder with original .obj files
- Output: resampled_data folder with normalized meshes

### 2. Feature Extraction

**Extract geometric features and shape descriptors:**
```bash
python3 step4_extraction.py
```
- Extracts 7 scalar features (area, volume, compactness, etc.)
- Extracts 5 shape descriptors (A3, D1, D2, D3, D4) with 10-bin histograms
- Output: features_database.csv (57 features per shape)

### 3. Feature Normalization

**Apply z-score normalization to scalar features only:**
```bash
python3 create_npy_from_csv.py
```
- Normalizes only the 7 scalar features
- Keeps histogram bins unchanged (preserves proportions)
- Output: features_normalized.npy and metadata files

### 4. Test the System

**Launch the GUI to verify everything works:**
```bash
python3 shape_retrieval_gui.py
```

### 5. System Evaluation

**Run comprehensive evaluation on all shapes:**
```bash
python3 step6_analysis.py
```
- Evaluates on all 2,006 shapes in database
- Computes standard and class-balanced metrics
- Generates performance plots and statistics
- Output: step6_results folder

**Evaluation Metrics:**
- Precision@K and Recall@K (K=1,5,10)
- Per-category performance analysis
- Class-balanced metrics accounting for dataset imbalance

### 6. Dimensionality Reduction (Optional)

**Generate t-SNE visualization:**
```bash
python3 dimensionality_reduction.py
```
- Creates 2D t-SNE projection of feature space
- Visualizes category clustering
- Output: Interactive plot and saved figures

---

## Project Structure

```
Multimedia-Retrieval/
├── data/                    # Original mesh database
├── resampled_data/          # Preprocessed meshes (7,500 vertices)
├── stats/                   # Feature database CSV
├── step5_data/             # Normalized features (NPY format)
│   └── step6_results/      # Evaluation results and plots
├── resample.py             # Step 1: Mesh preprocessing
├── step4_extraction.py     # Step 2: Feature extraction
├── create_npy_from_csv.py  # Step 3: Feature normalization
├── knn_engine.py           # KNN index building (used by GUI)
├── shape_retrieval_gui.py  # Interactive retrieval tool
├── step6_analysis.py       # System evaluation
└── dimensionality_reduction.py  # t-SNE visualization
```

## GUI Features

**Search and Browse:**
- Search meshes by filename using the search box
- Browse available meshes from dropdown menu
- View mesh metadata and statistics

**Visualization:**
- View Plot: Static Matplotlib visualization
- View 3D: Interactive Open3D viewer (rotate, zoom, pan)

**Retrieval:**
- Compute Distances: Find K nearest neighbors
- View Similar: Visualize top matching shapes
- Adjustable K value (number of results)

**Results Display:**
- Detailed distance metrics (scalar and histogram)
- Category information
- Ranked results table

---

## Key Features

**Geometric Features (7):**
- Surface area, Volume, AABB volume
- Compactness, Diameter, Convexity, Eccentricity

**Shape Descriptors (50 bins):**
- A3: Angle between 3 random points (10 bins)
- D1: Distance from centroid to surface (10 bins)
- D2: Distance between 2 random points (10 bins)
- D3: Square root of area of triangle (10 bins)
- D4: Cube root of volume of tetrahedron (10 bins)

**Total: 57 features per shape**

## Normalization Strategy

- **Scalar features**: Z-score normalization (mean=0, std=1)
- **Histogram features**: No normalization (preserves bin proportions)

This approach maintains histogram shape information while standardizing scalar measurements.

## System Performance

Evaluated on 2,006 shapes across 69 categories:

- **Precision@1**: 100% (perfect top match)
- **Precision@10**: 32.0% (standard) / 26.9% (class-balanced)
- **22x better than random** (baseline: 1.4%)

Top performing categories: ComputerKeyboard, Jet, Humanoid  
Performance correlates moderately with category size (5.1% imbalance impact)

## Notes

- Run all scripts from the project root directory
- For quick testing, just run the GUI
- For full processing, run steps 1-3 in order before launching GUI
- Evaluation can take 10-15 minutes for full database

## Troubleshooting

**"File not found" errors**: Ensure you run scripts from project root  
**"No features found"**: Run steps 1-3 (resample → extract → normalize)  
**Inf/NaN values**: Re-run `create_npy_from_csv.py` (includes automatic cleaning)  
**GUI won't start**: Check that features_normalized.npy exists
