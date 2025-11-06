"""
CBSR Evaluation Engine for Step 6
Implements Precision@K and Recall metrics for shape retrieval system
"""

import numpy as np
import pandas as pd
from pathlib import Path
import time
from collections import defaultdict
from knn_engine import KNNEngine

class CBSREvaluator:
    def __init__(self, data_path="step5_data"):
        """Initialize CBSR evaluator with KNN engine"""
        self.data_path = Path(data_path)
        self.knn_engine = None
        self.metadata = None
        self.categories = None
        self.category_counts = None
        self.evaluation_results = {}
        
    def initialize(self):
        """Initialize KNN engine and load metadata"""
        try:
            print("  Initializing CBSR Evaluator...")
            
            # Initialize KNN engine
            self.knn_engine = KNNEngine(self.data_path)
            
            # Load features and build index
            if not self.knn_engine.load_processed_features():
                print("❌ Failed to load processed features")
                return False
            
            if not self.knn_engine.build_index(n_neighbors=50, metric='euclidean'):
                print("❌ Failed to build KNN index")
                return False
            
            # Extract metadata
            self.metadata = self.knn_engine.metadata
            self.categories = [meta['category'] for meta in self.metadata]
            
            # Compute category statistics
            unique_categories = list(set(self.categories))
            self.category_counts = {cat: self.categories.count(cat) for cat in unique_categories}
            
            print(f"✅ Evaluator initialized successfully!")
            print(f"     Database size: {len(self.metadata)} shapes")
            print(f"     Categories: {len(unique_categories)}")
            print(f"     KNN index ready")
            
            return True
            
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            return False
    
    def precision_at_k(self, query_category, result_categories, k):
        """
        Compute Precision@K for a single query
        
        Args:
            query_category: Category of the query shape
            result_categories: List of categories from KNN results
            k: Number of top results to consider
            
        Returns:
            float: Precision@K value (0.0 to 1.0)
        """
        if k <= 0 or len(result_categories) == 0:
            return 0.0
        
        # Count relevant items in top-K results
        top_k_categories = result_categories[:k]
        relevant_count = sum(1 for cat in top_k_categories if cat == query_category)
        
        return relevant_count / k
    
    def recall_at_k(self, query_category, result_categories, k):
        """
        Compute Recall@K for a single query
        
        Args:
            query_category: Category of the query shape
            result_categories: List of categories from KNN results  
            k: Number of top results to consider
            
        Returns:
            float: Recall@K value (0.0 to 1.0)
        """
        if k <= 0 or len(result_categories) == 0:
            return 0.0
        
        # Total relevant items in database for this category
        total_relevant = self.category_counts.get(query_category, 0)
        
        if total_relevant <= 1:  # Only the query itself
            return 1.0 if k > 0 else 0.0
        
        # Count relevant items found in top-K results
        top_k_categories = result_categories[:k]
        relevant_found = sum(1 for cat in top_k_categories if cat == query_category)
        
        # Recall = (relevant found) / (total relevant in database)
        return min(1.0, relevant_found / total_relevant)
    
    def evaluate_single_shape(self, query_index, k_values=[1, 5, 10]):
        """
        Evaluate Precision@K and Recall@K for a single shape
        
        Args:
            query_index: Index of query shape in database
            k_values: List of K values to evaluate
            
        Returns:
            dict: Results for this query
        """
        if self.knn_engine is None:
            raise ValueError("KNN engine not initialized. Call initialize() first.")
        
        # Get query category
        query_category = self.metadata[query_index]['category']
        
        # Perform KNN search with max K
        max_k = max(k_values)
        knn_results = self.knn_engine.query_knn(query_index, k=max_k)
        
        if knn_results is None or len(knn_results) == 0:
            return {k: {'precision': 0.0, 'recall': 0.0} for k in k_values}
        
        # Extract result categories
        result_categories = knn_results['category'].tolist()
        
        # Compute metrics for each K
        results = {}
        for k in k_values:
            precision = self.precision_at_k(query_category, result_categories, k)
            recall = self.recall_at_k(query_category, result_categories, k)
            
            results[k] = {
                'precision': precision,
                'recall': recall,
                'query_category': query_category,
                'result_categories': result_categories[:k]
            }
        
        return results
    
    def evaluate_subset(self, max_queries=100, k_values=[1, 5, 10], random_seed=42):
        """
        Evaluate a subset of database for faster testing
        
        Args:
            max_queries: Maximum number of queries to test
            k_values: List of K values to evaluate
            random_seed: Random seed for reproducible results
            
        Returns:
            dict: Evaluation results
        """
        if self.knn_engine is None:
            raise ValueError("KNN engine not initialized. Call initialize() first.")
        
        np.random.seed(random_seed)
        
        # Select random subset of queries
        total_shapes = len(self.metadata)
        num_queries = min(max_queries, total_shapes)
        
        query_indices = np.random.choice(total_shapes, size=num_queries, replace=False)
        
        print(f"  Evaluating {num_queries} random queries...")
        print(f"  K values: {k_values}")
        
        # Store results by category and K value
        category_results = defaultdict(lambda: defaultdict(list))
        overall_results = defaultdict(list)
        
        start_time = time.time()
        
        for i, query_idx in enumerate(query_indices):
            if (i + 1) % 20 == 0:
                print(f"   Progress: {i+1}/{num_queries} queries completed...")
            
            # Evaluate this query
            query_results = self.evaluate_single_shape(query_idx, k_values)
            query_category = self.metadata[query_idx]['category']
            
            # Store results by category
            for k in k_values:
                if k in query_results:
                    precision = query_results[k]['precision']
                    recall = query_results[k]['recall']
                    
                    category_results[query_category][k].append({
                        'precision': precision,
                        'recall': recall,
                        'query_index': query_idx
                    })
                    
                    overall_results[k].append({
                        'precision': precision,
                        'recall': recall,
                        'category': query_category,
                        'query_index': query_idx
                    })
        
        evaluation_time = time.time() - start_time
        
        print(f"✅ Evaluation completed in {evaluation_time:.2f} seconds")
        
        # Store results
        self.evaluation_results = {
            'category_results': dict(category_results),
            'overall_results': dict(overall_results),
            'metadata': {
                'num_queries': num_queries,
                'k_values': k_values,
                'evaluation_time': evaluation_time,
                'random_seed': random_seed
            }
        }
        
        return self.evaluation_results
    
    def compute_summary_statistics(self):
        """Compute summary statistics from evaluation results"""
        if not self.evaluation_results:
            raise ValueError("No evaluation results available. Run evaluate_subset() first.")
        
        summary = {}
        
        # Overall statistics
        for k in self.evaluation_results['metadata']['k_values']:
            overall_data = self.evaluation_results['overall_results'][k]
            
            precisions = [item['precision'] for item in overall_data]
            recalls = [item['recall'] for item in overall_data]
            
            summary[f'overall_precision@{k}'] = {
                'mean': np.mean(precisions),
                'std': np.std(precisions),
                'min': np.min(precisions),
                'max': np.max(precisions)
            }
            
            summary[f'overall_recall@{k}'] = {
                'mean': np.mean(recalls),
                'std': np.std(recalls),
                'min': np.min(recalls),
                'max': np.max(recalls)
            }
        
        # Category-wise statistics
        category_summary = {}
        for category, k_results in self.evaluation_results['category_results'].items():
            category_summary[category] = {}
            
            for k, results_list in k_results.items():
                if results_list:  # Check if category has results for this K
                    precisions = [item['precision'] for item in results_list]
                    recalls = [item['recall'] for item in results_list]
                    
                    category_summary[category][f'precision@{k}'] = {
                        'mean': np.mean(precisions),
                        'count': len(precisions)
                    }
                    
                    category_summary[category][f'recall@{k}'] = {
                        'mean': np.mean(recalls),
                        'count': len(recalls)
                    }
        
        summary['category_summary'] = category_summary
        
        return summary
    
    def compute_detailed_metrics(self, k=10):
        """
        Compute detailed TP/FP/FN breakdown following Step 6 guidelines
        """
        if not self.evaluation_results:
            raise ValueError("No evaluation results available")
        
        detailed_metrics = {}
        overall_data = self.evaluation_results['overall_results'][k]
        
        # Per-category analysis
        for category, results_list in self.evaluation_results['category_results'].items():
            if k in results_list:
                tp_total = 0
                fp_total = 0
                fn_total = 0
                queries_count = len(results_list[k])
                
                total_in_category = self.category_counts[category]
                
                for result in results_list[k]:
                    # For this specific query
                    precision = result['precision']
                    recall = result['recall']
                    
                    # Calculate TP, FP, FN for this query
                    tp = int(precision * k)  # True positives in top-k
                    fp = k - tp  # False positives in top-k
                    fn = total_in_category - tp  # False negatives (not found in top-k)
                    
                    tp_total += tp
                    fp_total += fp
                    fn_total += fn
                
                # Average per query in this category
                detailed_metrics[category] = {
                    'avg_tp_per_query': tp_total / queries_count,
                    'avg_fp_per_query': fp_total / queries_count,
                    'avg_fn_per_query': fn_total / queries_count,
                    'queries_evaluated': queries_count,
                    'total_in_category': total_in_category,
                    'avg_precision': np.mean([r['precision'] for r in results_list[k]]),
                    'avg_recall': np.mean([r['recall'] for r in results_list[k]])
                }
        
        return detailed_metrics
    
    def print_technical_analysis(self, k=10):
        """Print analysis following Step 6 technical guidelines"""
        detailed = self.compute_detailed_metrics(k)
        
        print(f"\n  TECHNICAL EVALUATION ANALYSIS (K={k})")
        print("=" * 80)
        print(f"Following Step 6 evaluation guidelines:")
        print(f"• Database size |DB| = {len(self.metadata)}")
        print(f"• Query function: KNN with Euclidean distance")
        print(f"• Quality metric M: Precision@{k} and Recall@{k}")
        print(f"• Class labels C(Q) ∈ {{C1,...,C{len(self.category_counts)}}} (69 categories)")
        
        print(f"\nPER-CLASS ANALYSIS M(C):")
        print("-" * 80)
        print(f"{'Category':<20} {'M(C)':<8} {'K(C)':<6} {'Queries':<8} {'Avg TP':<7} {'Avg FP':<7} {'Avg FN':<7}")
        print("-" * 80)
        
        # Sort by precision for better presentation
        sorted_categories = sorted(detailed.items(), 
                                  key=lambda x: x[1]['avg_precision'], reverse=True)
        
        for category, metrics in sorted_categories:
            print(f"{category:<20} {metrics['avg_precision']:<8.3f} "
                  f"{metrics['total_in_category']:<6} {metrics['queries_evaluated']:<8} "
                  f"{metrics['avg_tp_per_query']:<7.1f} {metrics['avg_fp_per_query']:<7.1f} "
                  f"{metrics['avg_fn_per_query']:<7.1f}")
        
        # Overall average M_avg
        overall_precision = self.evaluation_results['overall_results'][k]
        mavg = np.mean([item['precision'] for item in overall_precision])
        
        print(f"\nOVERALL AVERAGE M_avg = {mavg:.3f}")
        print(f"Computed as: Σ M(all queries) / |DB| = sum of all precisions / total queries")
    
    def compute_class_balanced_metrics(self):
        """
        Compute metrics with class balancing
        Each category gets equal weight regardless of size
        """
        if not self.evaluation_results:  
            return None
        
        print("\n  Computing class-balanced metrics...")
        
        # Group results by category
        category_metrics = {}
        
        k_values = self.evaluation_results['metadata']['k_values']
        
        for k in k_values:
            category_precisions = {}
            category_recalls = {}
            
            # For each query, group by its category
            for result in self.evaluation_results['overall_results'][k]:  
                query_category = result['category']  
                if query_category not in category_precisions:
                    category_precisions[query_category] = []
                    category_recalls[query_category] = []
                
                category_precisions[query_category].append(result['precision'])
                category_recalls[query_category].append(result['recall'])
            
            # Compute mean per category (each category weighted equally)
            category_mean_precisions = []
            category_mean_recalls = []
            
            for category in category_precisions.keys():
                category_mean_precisions.append(np.mean(category_precisions[category]))
                category_mean_recalls.append(np.mean(category_recalls[category]))
            
            # Overall class-balanced metrics (mean of category means)
            balanced_precision = np.mean(category_mean_precisions)
            balanced_recall = np.mean(category_mean_recalls)
            
            category_metrics[k] = {
                'balanced_precision': balanced_precision,
                'balanced_recall': balanced_recall,
                'num_categories': len(category_precisions),
                'per_category_precision': dict(zip(category_precisions.keys(), category_mean_precisions)),
                'per_category_recall': dict(zip(category_recalls.keys(), category_mean_recalls))
            }
            
            print(f"   K={k}: Balanced Precision={balanced_precision:.3f}, Balanced Recall={balanced_recall:.3f}")
        
        return category_metrics
        

# Test function to verify the implementation
def test_evaluator():
    """Test the CBSR evaluator with a small subset"""
    print("  Testing CBSR Evaluator...")
    
    evaluator = CBSREvaluator("step5_data")
    
    if not evaluator.initialize():
        print("❌ Test failed: Could not initialize evaluator")
        return False
    
    # Test with small subset
    results = evaluator.evaluate_subset(max_queries=10, k_values=[1, 5, 10])
    
    if results:
        summary = evaluator.compute_summary_statistics()
        
        print("\n  TEST RESULTS:")
        print(f"   Queries tested: {results['metadata']['num_queries']}")
        print(f"   Categories found: {len(results['category_results'])}")
        
        for k in [1, 5, 10]:
            precision_mean = summary[f'overall_precision@{k}']['mean']
            recall_mean = summary[f'overall_recall@{k}']['mean']
            print(f"   Precision@{k}: {precision_mean:.3f}")
            print(f"   Recall@{k}: {recall_mean:.3f}")
        
        print("✅ Test completed successfully!")
        return True
    else:
        print("❌ Test failed: No results generated")
        return False

if __name__ == "__main__":
    test_evaluator()