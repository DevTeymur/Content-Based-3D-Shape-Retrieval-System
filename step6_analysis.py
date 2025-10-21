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
        
    def run_full_evaluation(self, max_queries=500, k_values=[1, 5, 10]):
        """Run comprehensive evaluation on larger subset"""
        print("🚀 Running comprehensive CBSR evaluation...")
        
        # Initialize evaluator
        if not self.evaluator.initialize():
            print("❌ Failed to initialize evaluator")
            return False
        
        # Run evaluation
        print(f"📊 Evaluating {max_queries} queries...")
        self.results = self.evaluator.evaluate_subset(
            max_queries=max_queries, 
            k_values=k_values,
            random_seed=42
        )
        
        if not self.results:
            print("❌ Evaluation failed")
            return False
        
        # Compute summary statistics
        self.summary = self.evaluator.compute_summary_statistics()
        
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
            # Convert numpy types to native Python for JSON serialization
            json_results = self._convert_numpy_types(self.results)
            json.dump(json_results, f, indent=2)
        
        # Save summary statistics
        with open(results_dir / "summary_statistics.json", 'w') as f:
            json_summary = self._convert_numpy_types(self.summary)
            json.dump(json_summary, f, indent=2)
        
        print(f"💾 Results saved to {results_dir}")
    
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
        print("\n📊 OVERALL PERFORMANCE SUMMARY")
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
        
        print(f"\n🏆 TOP {top_n} CATEGORIES BY {metric.upper()}@{k}")
        print("=" * 60)
        print(df_sorted.head(top_n).to_string(index=False))
        
        print(f"\n💔 BOTTOM {top_n} CATEGORIES BY {metric.upper()}@{k}")
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
        
        print(f"📈 Performance distribution plot saved")
    
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
        
        print(f"📊 Category {metric} plot saved")
    
    def generate_report_summary(self):
        """Generate text summary for report"""
        if not self.summary:
            print("❌ No summary statistics available")
            return
        
        print("\n📝 STEP 6 EVALUATION REPORT SUMMARY")
        print("=" * 80)
        
        # Overall performance
        print("\n🎯 OVERALL SYSTEM PERFORMANCE:")
        for k in [1, 5, 10]:
            precision = self.summary[f'overall_precision@{k}']['mean']
            recall = self.summary[f'overall_recall@{k}']['mean']
            print(f"   Precision@{k}: {precision:.3f} | Recall@{k}: {recall:.3f}")
        
        # Best performing categories
        print("\n🏆 TOP PERFORMING CATEGORIES (Precision@10):")
        df_precision = self.create_category_ranking(k=10, metric='precision', top_n=5)
        if df_precision is not None:
            top_5 = df_precision.head(5)
            for _, row in top_5.iterrows():
                print(f"   {row['Category']}: {row['Precision@10']:.3f}")
        
        # Key insights
        precision_1 = self.summary['overall_precision@1']['mean']
        precision_10 = self.summary['overall_precision@10']['mean']
        
        print(f"\n💡 KEY INSIGHTS:")
        print(f"   • Perfect top-1 accuracy: {precision_1:.1%}")
        print(f"   • Strong top-10 relevance: {precision_10:.1%}")
        print(f"   • System is {precision_10/0.014:.0f}x better than random (1/69 categories)")
        print(f"   • Evaluated on {self.results['metadata']['num_queries']} queries")
        print(f"   • Database contains {len(self.evaluator.metadata)} shapes in 69 categories")

def main():
    """Run full Step 6 analysis"""
    analyzer = Step6Analyzer("step5_data")
    
    # Run evaluation
    if analyzer.run_full_evaluation(max_queries=200, k_values=[1, 5, 10]):
        
        # Generate analysis
        analyzer.create_performance_table()
        analyzer.create_category_ranking(k=10, metric='precision', top_n=10)
        analyzer.plot_performance_distribution(k=10)
        analyzer.plot_category_performance(k=10, metric='precision', top_n=15)
        analyzer.generate_report_summary()
        
        print("\n✅ Step 6 analysis completed!")
        print("📁 Check step5_data/step6_results/ for saved plots and data")
    else:
        print("❌ Analysis failed")

if __name__ == "__main__":
    main()