"""
Step 6 Analysis and Visualization
Provides detailed analysis and plotting for CBSR evaluation results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from cbsr_evaluator import CBSREvaluator

class Step6Analyzer:
    def __init__(self, data_path="step5_data"):
        """Initialize analyzer with evaluator"""
        self.data_path = Path(data_path)
        self.evaluator = CBSREvaluator(data_path)
        self.results = None
        self.summary = None
        
    def run_full_evaluation(self, max_queries=None, k_values=[1, 5, 10]):
        """Run comprehensive evaluation on all shapes"""
        print("  Running comprehensive CBSR evaluation...")
        
        # Initialize evaluator
        if not self.evaluator.initialize():
            print("❌ Failed to initialize evaluator")
            return False
        
        # Determine number of queries
        total_shapes = len(self.evaluator.metadata)
        num_queries = total_shapes if max_queries is None else min(max_queries, total_shapes)
        
        print(f"  Evaluating {num_queries} queries out of {total_shapes} total shapes")
        
        # Run evaluation
        self.results = self.evaluator.evaluate_subset(
            max_queries=num_queries, 
            k_values=k_values,
            random_seed=42
        )
        
        if not self.results:
            print("❌ Evaluation failed")
            return False
        
        # Compute regular summary statistics
        self.summary = self.evaluator.compute_summary_statistics()
        
        # ✅ NEW: Compute class-balanced metrics
        self.balanced_metrics = self.evaluator.compute_class_balanced_metrics()
        
        # Save results
        self._save_results()
        
        print("✅ Comprehensive evaluation completed!")
        return True
    
    def _save_results(self):
        """Save evaluation results to files"""
        results_dir = self.data_path / "step6_results"
        results_dir.mkdir(exist_ok=True)
        
        # Save raw results
        with open(results_dir / "evaluation_results.json", 'w') as f:
            json_results = self._convert_numpy_types(self.results)
            json.dump(json_results, f, indent=2)
        
        # Save summary statistics
        with open(results_dir / "summary_statistics.json", 'w') as f:
            json_summary = self._convert_numpy_types(self.summary)
            json.dump(json_summary, f, indent=2)
        
        # ✅ NEW: Save class-balanced metrics
        if hasattr(self, 'balanced_metrics'):
            with open(results_dir / "class_balanced_metrics.json", 'w') as f:
                json_balanced = self._convert_numpy_types(self.balanced_metrics)
                json.dump(json_balanced, f, indent=2)
        
        print(f"  Results saved to {results_dir}")
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to native Python types for JSON serialization"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    def create_performance_table(self):
        """Create performance summary table"""
        if not self.summary:
            print("❌ No summary statistics available")
            return None
        
        # Extract overall performance
        data = []
        for k in [1, 5, 10]:
            precision_stats = self.summary[f'overall_precision@{k}']
            recall_stats = self.summary[f'overall_recall@{k}']
            
            data.append({
                'K': k,
                'Precision (Mean)': f"{precision_stats['mean']:.3f}",
                'Precision (Std)': f"{precision_stats['std']:.3f}",
                'Recall (Mean)': f"{recall_stats['mean']:.3f}",
                'Recall (Std)': f"{recall_stats['std']:.3f}",
                'Precision (Range)': f"[{precision_stats['min']:.3f}, {precision_stats['max']:.3f}]"
            })
        
        df = pd.DataFrame(data)
        print("\n  OVERALL PERFORMANCE SUMMARY")
        print("=" * 80)
        print(df.to_string(index=False))
        
        return df
    
    def create_category_ranking(self, k=10, metric='precision', top_n=10):
        """Create ranking of best/worst performing categories"""
        if not self.summary:
            print("❌ No summary statistics available")
            return None
        
        category_data = []
        category_summary = self.summary['category_summary']
        
        for category, stats in category_summary.items():
            metric_key = f'{metric}@{k}'
            if metric_key in stats:
                category_data.append({
                    'Category': category,
                    f'{metric.title()}@{k}': stats[metric_key]['mean'],
                    'Query Count': stats[metric_key]['count']
                })
        
        if not category_data:
            print(f"❌ No data available for {metric}@{k}")
            return None
        
        df = pd.DataFrame(category_data)
        df_sorted = df.sort_values(f'{metric.title()}@{k}', ascending=False)
        
        print(f"\n  TOP {top_n} CATEGORIES BY {metric.upper()}@{k}")
        print("=" * 60)
        print(df_sorted.head(top_n).to_string(index=False))
        
        print(f"\n  BOTTOM {top_n} CATEGORIES BY {metric.upper()}@{k}")
        print("=" * 60)
        print(df_sorted.tail(top_n).to_string(index=False))
        
        return df_sorted
    
    def plot_performance_distribution(self, k=10):
        """Create distribution plots for precision and recall"""
        if not self.results:
            print("❌ No evaluation results available")
            return
        
        # Extract data for plotting
        overall_data = self.results['overall_results'][k]
        precisions = [item['precision'] for item in overall_data]
        recalls = [item['recall'] for item in overall_data]
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Precision distribution
        ax1.hist(precisions, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel(f'Precision@{k}')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'Distribution of Precision@{k}')
        ax1.axvline(np.mean(precisions), color='red', linestyle='--', 
                    label=f'Mean: {np.mean(precisions):.3f}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Recall distribution
        ax2.hist(recalls, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        ax2.set_xlabel(f'Recall@{k}')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'Distribution of Recall@{k}')
        ax2.axvline(np.mean(recalls), color='red', linestyle='--', 
                    label=f'Mean: {np.mean(recalls):.3f}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        results_dir = self.data_path / "step6_results"
        results_dir.mkdir(exist_ok=True)
        plt.savefig(results_dir / f"performance_distribution_k{k}.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"  Performance distribution plot saved")
    
    def plot_category_performance(self, k=10, metric='precision', top_n=15):
        """Create bar plot of category performance"""
        if not self.summary:
            print("❌ No summary statistics available")
            return
        
        # Get category ranking
        df_sorted = self.create_category_ranking(k=k, metric=metric, top_n=len(self.summary['category_summary']))
        
        if df_sorted is None:
            return
        
        # Plot top categories
        top_categories = df_sorted.head(top_n)
        
        plt.figure(figsize=(12, 8))
        bars = plt.bar(range(len(top_categories)), 
                      top_categories[f'{metric.title()}@{k}'],
                      color='steelblue', alpha=0.8)
        
        plt.xlabel('Category')
        plt.ylabel(f'{metric.title()}@{k}')
        plt.title(f'Top {top_n} Categories by {metric.title()}@{k}')
        plt.xticks(range(len(top_categories)), top_categories['Category'], 
                  rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        # Save plot
        results_dir = self.data_path / "step6_results"
        results_dir.mkdir(exist_ok=True)
        plt.savefig(results_dir / f"category_{metric}_k{k}.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"  Category {metric} plot saved")
    
    def generate_report_summary(self):
        """Generate text summary for report"""
        if not self.summary:
            print("❌ No summary statistics available")
            return
        
        print("\n  STEP 6 EVALUATION REPORT SUMMARY")
        print("=" * 80)
        
        # Overall performance (STANDARD - all queries equally weighted)
        print("\n  OVERALL PERFORMANCE (All queries equally weighted):")
        for k in [1, 5, 10]:
            precision = self.summary[f'overall_precision@{k}']['mean']
            recall = self.summary[f'overall_recall@{k}']['mean']
            print(f"   Precision@{k}: {precision:.3f} | Recall@{k}: {recall:.3f}")
        
        # ✅ NEW: Class-balanced performance
        if hasattr(self, 'balanced_metrics') and self.balanced_metrics:
            print("\n⚖️  CLASS-BALANCED PERFORMANCE (Each category weighted equally):")
            for k in [1, 5, 10]:
                precision = self.balanced_metrics[k]['balanced_precision']
                recall = self.balanced_metrics[k]['balanced_recall']
                num_categories = self.balanced_metrics[k]['num_categories']
                print(f"   Precision@{k}: {precision:.3f} | Recall@{k}: {recall:.3f} ({num_categories} categories)")
        
        # Best performing categories
        print("\n  TOP PERFORMING CATEGORIES (Precision@10):")
        df_precision = self.create_category_ranking(k=10, metric='precision', top_n=5)
        if df_precision is not None:
            top_5 = df_precision.head(5)
            for _, row in top_5.iterrows():
                print(f"   {row['Category']}: {row['Precision@10']:.3f}")
        
        # Key insights
        precision_1 = self.summary['overall_precision@1']['mean']
        precision_10 = self.summary['overall_precision@10']['mean']
        
        print(f"\n  KEY INSIGHTS:")
        print(f"   • Perfect top-1 accuracy: {precision_1:.1%}")
        print(f"   • Strong top-10 relevance: {precision_10:.1%}")
        
        # ✅ NEW: Compare standard vs balanced
        if hasattr(self, 'balanced_metrics'):
            balanced_p10 = self.balanced_metrics[10]['balanced_precision']
            diff = abs(precision_10 - balanced_p10)
            print(f"   • Class imbalance impact: {diff:.3f} difference (standard vs balanced)")
        
        print(f"   • Evaluated on {self.results['metadata']['num_queries']} queries")
        print(f"   • Database contains {len(self.evaluator.metadata)} shapes in 69 categories")
    
    def plot_tp_fp_fn_trends(self):
        """Create TP/FP/FN trends plot across K values"""
        if not self.results:
            print("❌ No evaluation results available")
            return
        
        # Debug first to see what we have
        self.debug_results_structure()
        
        k_values = [1, 5, 10]
        tp_means = []
        fp_means = []
        fn_means = []
        tp_stds = []
        fp_stds = []
        fn_stds = []
        
        # Extract TP/FP/FN data for each K
        for k in k_values:
            overall_data = self.results['overall_results'][k]
            
            # Try different possible key names
            tps = []
            fps = []
            fns = []
            
            for item in overall_data:
                # Check what keys are available
                available_keys = list(item.keys())
                # print(f"Available keys for K={k}: {available_keys}")
                
                # Calculate TP/FP/FN from precision/recall if direct values not available
                if 'tp' in item:
                    tps.append(item['tp'])
                    fps.append(item['fp'])
                    fns.append(item['fn'])
                elif 'precision' in item and 'recall' in item:
                    # Calculate from precision and recall
                    precision = item['precision']
                    recall = item['recall']
                    
                    # TP = precision * k (retrieved items that are relevant)
                    # FP = k - TP (retrieved items that are not relevant)
                    # Total relevant items for this query = TP / recall
                    # FN = total_relevant - TP
                    
                    tp = precision * k
                    fp = k - tp
                    
                    if recall > 0:
                        total_relevant = tp / recall
                        fn = total_relevant - tp
                    else:
                        fn = 0  # If recall is 0, no relevant items exist
                    
                    tps.append(tp)
                    fps.append(fp)
                    fns.append(fn)
                else:
                    print(f"❌ Cannot calculate TP/FP/FN for item: {item}")
                    return
            
            if not tps:
                print(f"❌ No TP/FP/FN data available for K={k}")
                return
            
            tp_means.append(np.mean(tps))
            fp_means.append(np.mean(fps))
            fn_means.append(np.mean(fns))
            
            tp_stds.append(np.std(tps))
            fp_stds.append(np.std(fps))
            fn_stds.append(np.std(fns))
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        
        x = np.array(k_values)
        width = 0.25
        
        # Plot bars with error bars
        plt.bar(x - width, tp_means, width, label='True Positives (TP)', 
                color='green', alpha=0.7, yerr=tp_stds, capsize=5)
        plt.bar(x, fp_means, width, label='False Positives (FP)', 
                color='red', alpha=0.7, yerr=fp_stds, capsize=5)
        plt.bar(x + width, fn_means, width, label='False Negatives (FN)', 
                color='orange', alpha=0.7, yerr=fn_stds, capsize=5)
        
        # Customize plot
        plt.xlabel('K Value')
        plt.ylabel('Count (Mean ± Std)')
        plt.title('Trends of True Positives, False Positives, and False Negatives across K values')
        plt.xticks(k_values)
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, k in enumerate(k_values):
            plt.text(k - width, tp_means[i] + tp_stds[i] + 0.1, 
                    f'{tp_means[i]:.2f}', ha='center', va='bottom', fontsize=9)
            plt.text(k, fp_means[i] + fp_stds[i] + 0.1, 
                    f'{fp_means[i]:.2f}', ha='center', va='bottom', fontsize=9)
            plt.text(k + width, fn_means[i] + fn_stds[i] + 0.1, 
                    f'{fn_means[i]:.2f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        # Save plot
        results_dir = self.data_path / "step6_results"
        results_dir.mkdir(exist_ok=True)
        plt.savefig(results_dir / "tp_fp_fn_trends.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print("  TP/FP/FN trends plot saved")
    
    def plot_all_categories_precision10(self):
        """Create comprehensive category-wise Precision@10 plot for all 69 categories"""
        if not self.summary:
            print("❌ No summary statistics available")
            return
        
        # Get all category data
        category_data = []
        category_summary = self.summary['category_summary']
        
        for category, stats in category_summary.items():
            if 'precision@10' in stats:
                category_data.append({
                    'Category': category,
                    'Precision@10': stats['precision@10']['mean'],
                    'Query Count': stats['precision@10']['count']
                })
        
        if not category_data:
            print("❌ No precision@10 data available")
            return
        
        df = pd.DataFrame(category_data)
        df_sorted = df.sort_values('Precision@10', ascending=False)
        
        # Create the plot
        plt.figure(figsize=(16, 10))
        
        # Color-code bars based on performance
        colors = []
        for precision in df_sorted['Precision@10']:
            if precision >= 0.8:
                colors.append('darkgreen')  # High performance
            elif precision >= 0.5:
                colors.append('orange')     # Medium performance
            else:
                colors.append('red')        # Low performance
        
        bars = plt.bar(range(len(df_sorted)), df_sorted['Precision@10'], 
                       color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Customize plot
        plt.xlabel('Category')
        plt.ylabel('Precision@10')
        plt.title('Category-wise Precision@10 across all 69 categories')
        plt.xticks(range(len(df_sorted)), df_sorted['Category'], 
                   rotation=90, ha='right', fontsize=8)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add horizontal reference lines
        plt.axhline(y=0.8, color='darkgreen', linestyle='--', alpha=0.5, label='High (≥0.8)')
        plt.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Medium (≥0.5)')
        plt.axhline(y=1/69, color='black', linestyle=':', alpha=0.7, label='Random (1/69)')
        
        # Add legend for performance levels
        plt.legend(loc='upper right')
        
        # Add summary statistics as text
        high_count = sum(1 for p in df_sorted['Precision@10'] if p >= 0.8)
        medium_count = sum(1 for p in df_sorted['Precision@10'] if p >= 0.5 and p < 0.8)
        low_count = len(df_sorted) - high_count - medium_count
        
        plt.text(0.02, 0.98, f'High: {high_count} categories\nMedium: {medium_count} categories\nLow: {low_count} categories', 
                 transform=plt.gca().transAxes, verticalalignment='top', 
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        results_dir = self.data_path / "step6_results"
        results_dir.mkdir(exist_ok=True)
        plt.savefig(results_dir / "categorywise_precision10.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print("  Category-wise Precision@10 plot saved")
        
        return df_sorted
    
    def generate_tp_fp_fn_table_data(self):
        """Generate TP/FP/FN statistics table data"""
        if not self.results:
            print("❌ No evaluation results available")
            return None
        
        k_values = [1, 5, 10]
        table_data = []
        
        for k in k_values:
            overall_data = self.results['overall_results'][k]
            
            tps = []
            fps = []
            fns = []
            
            for item in overall_data:
                if 'tp' in item:
                    tps.append(item['tp'])
                    fps.append(item['fp']) 
                    fns.append(item['fn'])
                elif 'precision' in item and 'recall' in item:
                    # Calculate from precision and recall
                    precision = item['precision']
                    recall = item['recall']
                    
                    tp = precision * k
                    fp = k - tp
                    
                    if recall > 0:
                        total_relevant = tp / recall
                        fn = total_relevant - tp
                    else:
                        fn = 0
                    
                    tps.append(tp)
                    fps.append(fp)
                    fns.append(fn)
            
            if tps:
                table_data.append({
                    'K': k,
                    'TP (Mean ± Std)': f"{np.mean(tps):.2f} ± {np.std(tps):.2f}",
                    'FP (Mean ± Std)': f"{np.mean(fps):.2f} ± {np.std(fps):.2f}",
                    'FN (Mean ± Std)': f"{np.mean(fns):.2f} ± {np.std(fns):.2f}"
                })
        
        if table_data:
            df = pd.DataFrame(table_data)
            print("\n  TP/FP/FN STATISTICS TABLE")
            print("=" * 80)
            print(df.to_string(index=False))
            return df
        else:
            print("❌ No TP/FP/FN data could be calculated")
            return None
    
    def debug_results_structure(self):
        """Debug the structure of results to understand available keys"""
        if not self.results:
            print("❌ No evaluation results available")
            return
        
        print("  DEBUGGING RESULTS STRUCTURE")
        print("=" * 50)
        
        # Check top-level keys
        print("Top-level keys:", list(self.results.keys()))
        
        # Check what's in overall_results
        if 'overall_results' in self.results:
            print("overall_results keys:", list(self.results['overall_results'].keys()))
            
            # Check structure of first K value
            k = list(self.results['overall_results'].keys())[0]
            print(f"Structure for K={k}:")
            
            first_item = self.results['overall_results'][k][0]
            print("First item keys:", list(first_item.keys()))
            print("First item sample:", first_item)
        
        # Check if there's category_results
        if 'category_results' in self.results:
            print("category_results keys:", list(self.results['category_results'].keys()))

def main():
    """Run full Step 6 analysis with all figures"""
    analyzer = Step6Analyzer("step5_data")
    
    # Run evaluation
    if analyzer.run_full_evaluation(max_queries=None, k_values=[1, 5, 10]):
        
        # Generate existing analysis
        analyzer.create_performance_table()
        analyzer.create_category_ranking(k=10, metric='precision', top_n=10)
        analyzer.plot_performance_distribution(k=10)
        analyzer.plot_category_performance(k=10, metric='precision', top_n=15)
        
        # Generate missing figures for your report
        print("\n  Generating missing figures for report...")
        
        # This one should work
        analyzer.plot_all_categories_precision10()  # For categorywise_precision10.png
        
        # Debug and try TP/FP/FN
        print("\n  Attempting TP/FP/FN analysis...")
        analyzer.plot_tp_fp_fn_trends()             # For tp_fp_fn_trends.png
        analyzer.generate_tp_fp_fn_table_data()     # For table data verification
        
        analyzer.generate_report_summary()
        
        print("\n✅ Step 6 analysis completed!")
        print("  Check step5_data/step6_results/ for generated plots")
    else:
        print("❌ Analysis failed")

if __name__ == "__main__":
    main()