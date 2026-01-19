"""
compare_overlap_experiments.py

Compare overlap results from multiple experiments.

Usage:
    python compare_overlap_experiments.py \
        --exp_dirs ./experiments/exp1 ./experiments/exp2 \
        --labels "Alpaca" "OpenAssistant" \
        --output_path ./comparison.png
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any

import matplotlib.pyplot as plt
import numpy as np


def load_experiment_results(exp_dir: str) -> Dict[str, Any]:
    """Load results from an experiment directory."""
    results_path = os.path.join(exp_dir, "overlap_results.json")
    args_path = os.path.join(exp_dir, "args.json")
    
    if not Path(results_path).exists():
        raise FileNotFoundError(f"Results not found: {results_path}")
    
    with open(results_path, "r") as f:
        results = json.load(f)
    
    # Load args if available
    args = {}
    if Path(args_path).exists():
        with open(args_path, "r") as f:
            args = json.load(f)
    
    return {
        "results": results,
        "args": args,
        "dir": exp_dir,
    }


def plot_comparison(experiments: List[Dict[str, Any]], labels: List[str], 
                   output_path: str = None, show: bool = True):
    """Plot comparison of multiple experiments."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#BC4B51']
    
    # Plot 1: Overlap vs Steps (all experiments)
    ax = axes[0, 0]
    for i, (exp, label) in enumerate(zip(experiments, labels)):
        results = exp["results"]
        steps = [r["step"] for r in results]
        overlaps = [r["avg_overlap_ratio"] for r in results]
        
        color = colors[i % len(colors)]
        ax.plot(steps, overlaps, marker='o', linewidth=2, markersize=6, 
                color=color, label=label, alpha=0.8)
    
    ax.set_xlabel("Training Steps", fontsize=12, fontweight='bold')
    ax.set_ylabel("Average Overlap Ratio", fontsize=12, fontweight='bold')
    ax.set_title("Overlap vs Training Steps (Comparison)", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=10)
    
    # Plot 2: Overlap Decrease Rate
    ax = axes[0, 1]
    for i, (exp, label) in enumerate(zip(experiments, labels)):
        results = exp["results"]
        if len(results) < 2:
            continue
        
        steps = [r["step"] for r in results]
        overlaps = [r["avg_overlap_ratio"] for r in results]
        
        # Calculate decrease rate
        decrease_rates = []
        step_intervals = []
        for j in range(1, len(results)):
            step_diff = steps[j] - steps[j-1]
            overlap_diff = overlaps[j-1] - overlaps[j]
            rate = overlap_diff / step_diff if step_diff > 0 else 0
            decrease_rates.append(rate)
            step_intervals.append((steps[j-1] + steps[j]) / 2)
        
        color = colors[i % len(colors)]
        ax.plot(step_intervals, decrease_rates, marker='s', linewidth=2, 
                markersize=6, color=color, label=label, alpha=0.8)
    
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel("Training Steps", fontsize=12, fontweight='bold')
    ax.set_ylabel("Overlap Decrease Rate (per step)", fontsize=12, fontweight='bold')
    ax.set_title("Decrease Rate Comparison", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=10)
    
    # Plot 3: Bar chart of final overlap
    ax = axes[1, 0]
    final_overlaps = []
    for exp in experiments:
        results = exp["results"]
        if results:
            final_overlaps.append(results[-1]["avg_overlap_ratio"])
        else:
            final_overlaps.append(0)
    
    bars = ax.bar(range(len(labels)), final_overlaps, 
                  color=[colors[i % len(colors)] for i in range(len(labels))],
                  alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel("Final Overlap Ratio", fontsize=12, fontweight='bold')
    ax.set_title("Final Overlap Comparison", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 4: Statistics table
    ax = axes[1, 1]
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    table_data = [["Experiment", "Steps", "Initial", "Final", "Decrease", "Rate/step"]]
    
    for exp, label in zip(experiments, labels):
        results = exp["results"]
        if len(results) < 2:
            continue
        
        steps_range = f"{results[0]['step']}-{results[-1]['step']}"
        initial = results[0]["avg_overlap_ratio"]
        final = results[-1]["avg_overlap_ratio"]
        decrease = initial - final
        total_steps = results[-1]["step"] - results[0]["step"]
        rate = decrease / total_steps if total_steps > 0 else 0
        
        table_data.append([
            label[:20],  # Truncate long labels
            steps_range,
            f"{initial:.4f}",
            f"{final:.4f}",
            f"{decrease:.4f}",
            f"{rate:.6f}"
        ])
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                    colWidths=[0.25, 0.15, 0.12, 0.12, 0.12, 0.14])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header row
    for i in range(len(table_data[0])):
        cell = table[(0, i)]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data)):
        for j in range(len(table_data[0])):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#E7E6E6')
    
    ax.set_title("Experiment Statistics", fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {output_path}")
    
    if show:
        plt.show()
    
    plt.close()


def print_comparison_summary(experiments: List[Dict[str, Any]], labels: List[str]):
    """Print a text summary comparing experiments."""
    
    print("\n" + "=" * 100)
    print("EXPERIMENT COMPARISON SUMMARY")
    print("=" * 100)
    
    for exp, label in zip(experiments, labels):
        results = exp["results"]
        args = exp["args"]
        
        print(f"\n{label}")
        print("-" * 100)
        print(f"  Directory: {exp['dir']}")
        
        if args:
            print(f"  Base model: {args.get('base_model_name', 'N/A')}")
            print(f"  Dataset: {args.get('dataset_name', 'N/A')}")
            print(f"  Bottom-k vocab: {args.get('bottom_k_vocab', 'N/A')}")
            print(f"  Num fingerprints: {args.get('num_fingerprints', 'N/A')}")
        
        if len(results) >= 2:
            steps = [r["step"] for r in results]
            overlaps = [r["avg_overlap_ratio"] for r in results]
            
            print(f"\n  Results:")
            print(f"    Checkpoints: {len(results)}")
            print(f"    Steps range: {steps[0]} - {steps[-1]}")
            print(f"    Initial overlap: {overlaps[0]:.6f}")
            print(f"    Final overlap: {overlaps[-1]:.6f}")
            print(f"    Total decrease: {overlaps[0] - overlaps[-1]:.6f}")
            print(f"    Relative decrease: {(overlaps[0] - overlaps[-1]) / overlaps[0] * 100:.2f}%")
            
            # Linear fit
            z = np.polyfit(steps, overlaps, 1)
            print(f"    Average rate: {-z[0]:.8f} per step")
            
            # R-squared
            y_pred = np.poly1d(z)(steps)
            ss_res = np.sum((np.array(overlaps) - y_pred) ** 2)
            ss_tot = np.sum((np.array(overlaps) - np.mean(overlaps)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            print(f"    R-squared: {r_squared:.6f}")
    
    print("\n" + "=" * 100)
    
    # Ranking
    if len(experiments) > 1:
        print("\nRANKINGS")
        print("=" * 100)
        
        # Rank by final overlap (higher = more similar to base)
        final_overlaps = [(i, exp["results"][-1]["avg_overlap_ratio"] if exp["results"] else 0) 
                         for i, exp in enumerate(experiments)]
        final_overlaps.sort(key=lambda x: x[1], reverse=True)
        
        print("\nBy Final Overlap (higher = more similar to base model):")
        for rank, (idx, overlap) in enumerate(final_overlaps, 1):
            print(f"  {rank}. {labels[idx]}: {overlap:.6f}")
        
        # Rank by total decrease (higher = more change)
        decreases = []
        for i, exp in enumerate(experiments):
            if len(exp["results"]) >= 2:
                initial = exp["results"][0]["avg_overlap_ratio"]
                final = exp["results"][-1]["avg_overlap_ratio"]
                decreases.append((i, initial - final))
        decreases.sort(key=lambda x: x[1], reverse=True)
        
        print("\nBy Total Overlap Decrease (higher = more change from base):")
        for rank, (idx, decrease) in enumerate(decreases, 1):
            print(f"  {rank}. {labels[idx]}: {decrease:.6f}")
        
        print("=" * 100 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Compare overlap experiments")
    parser.add_argument("--exp_dirs", nargs="+", required=True,
                        help="Directories containing experiment results")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Labels for each experiment (default: use directory names)")
    parser.add_argument("--output_path", type=str, default="overlap_comparison.png",
                        help="Path to save comparison plot")
    parser.add_argument("--no_show", action="store_true",
                        help="Don't display plot interactively")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.exp_dirs:
        raise ValueError("Must provide at least one experiment directory")
    
    # Generate labels if not provided
    if args.labels is None:
        args.labels = [Path(d).name for d in args.exp_dirs]
    elif len(args.labels) != len(args.exp_dirs):
        raise ValueError(f"Number of labels ({len(args.labels)}) must match number of directories ({len(args.exp_dirs)})")
    
    # Load all experiments
    print(f"Loading {len(args.exp_dirs)} experiments...")
    experiments = []
    for exp_dir in args.exp_dirs:
        try:
            exp = load_experiment_results(exp_dir)
            experiments.append(exp)
            print(f"  ✓ Loaded: {exp_dir}")
        except Exception as e:
            print(f"  ✗ Failed to load {exp_dir}: {e}")
    
    if not experiments:
        raise ValueError("No valid experiments loaded")
    
    # Print comparison summary
    print_comparison_summary(experiments, args.labels[:len(experiments)])
    
    # Plot comparison
    print("\nGenerating comparison plot...")
    plot_comparison(
        experiments,
        args.labels[:len(experiments)],
        output_path=args.output_path,
        show=not args.no_show
    )
    
    print("\n✓ Comparison completed!")


if __name__ == "__main__":
    main()
