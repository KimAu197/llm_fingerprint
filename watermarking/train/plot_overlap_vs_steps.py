"""
plot_overlap_vs_steps.py

Visualize the relationship between training steps and overlap ratio.

Usage:
    python plot_overlap_vs_steps.py --result_dir ./ft_overlap_experiment
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_overlap_results(result_dir: str):
    """Load overlap results from experiment directory."""
    results_path = os.path.join(result_dir, "overlap_results.json")
    
    if not Path(results_path).exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")
    
    with open(results_path, "r") as f:
        results = json.load(f)
    
    return results


def plot_overlap_vs_steps(results, output_path: str = None, show: bool = True):
    """Plot overlap ratio vs training steps."""
    steps = [r["step"] for r in results]
    overlaps = [r["avg_overlap_ratio"] for r in results]
    
    plt.figure(figsize=(10, 6))
    plt.plot(steps, overlaps, marker='o', linewidth=2, markersize=8, color='#2E86AB')
    plt.xlabel("Training Steps", fontsize=14, fontweight='bold')
    plt.ylabel("Average Overlap Ratio", fontsize=14, fontweight='bold')
    plt.title("Wordlist Overlap vs Training Steps", fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    
    if show:
        plt.show()
    
    plt.close()


def plot_overlap_decrease_rate(results, output_path: str = None, show: bool = True):
    """Plot the rate of overlap decrease between consecutive checkpoints."""
    if len(results) < 2:
        print("Need at least 2 checkpoints to plot decrease rate")
        return
    
    steps = [r["step"] for r in results]
    overlaps = [r["avg_overlap_ratio"] for r in results]
    
    # Calculate decrease rate between consecutive points
    decrease_rates = []
    step_intervals = []
    
    for i in range(1, len(results)):
        step_diff = steps[i] - steps[i-1]
        overlap_diff = overlaps[i-1] - overlaps[i]  # positive if decreasing
        rate = overlap_diff / step_diff if step_diff > 0 else 0
        
        decrease_rates.append(rate)
        step_intervals.append((steps[i-1] + steps[i]) / 2)  # midpoint
    
    plt.figure(figsize=(10, 6))
    plt.plot(step_intervals, decrease_rates, marker='s', linewidth=2, markersize=8, color='#A23B72')
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    plt.xlabel("Training Steps", fontsize=14, fontweight='bold')
    plt.ylabel("Overlap Decrease Rate (per step)", fontsize=14, fontweight='bold')
    plt.title("Rate of Overlap Decrease vs Training Steps", fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    
    if show:
        plt.show()
    
    plt.close()


def plot_combined_analysis(results, output_path: str = None, show: bool = True):
    """Create a combined plot with overlap and decrease rate."""
    steps = [r["step"] for r in results]
    overlaps = [r["avg_overlap_ratio"] for r in results]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Overlap vs Steps
    ax1.plot(steps, overlaps, marker='o', linewidth=2, markersize=8, color='#2E86AB', label='Overlap Ratio')
    ax1.set_xlabel("Training Steps", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Average Overlap Ratio", fontsize=12, fontweight='bold')
    ax1.set_title("Wordlist Overlap vs Training Steps", fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend()
    
    # Add trend line
    if len(steps) > 1:
        z = np.polyfit(steps, overlaps, 1)
        p = np.poly1d(z)
        ax1.plot(steps, p(steps), "--", color='red', alpha=0.5, label=f'Trend: y={z[0]:.6f}x+{z[1]:.4f}')
        ax1.legend()
    
    # Plot 2: Decrease Rate
    if len(results) >= 2:
        decrease_rates = []
        step_intervals = []
        
        for i in range(1, len(results)):
            step_diff = steps[i] - steps[i-1]
            overlap_diff = overlaps[i-1] - overlaps[i]
            rate = overlap_diff / step_diff if step_diff > 0 else 0
            
            decrease_rates.append(rate)
            step_intervals.append((steps[i-1] + steps[i]) / 2)
        
        ax2.plot(step_intervals, decrease_rates, marker='s', linewidth=2, markersize=8, color='#A23B72', label='Decrease Rate')
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax2.set_xlabel("Training Steps", fontsize=12, fontweight='bold')
        ax2.set_ylabel("Overlap Decrease Rate (per step)", fontsize=12, fontweight='bold')
        ax2.set_title("Rate of Overlap Decrease", fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.legend()
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Combined plot saved to: {output_path}")
    
    if show:
        plt.show()
    
    plt.close()


def print_statistics(results):
    """Print statistical summary of overlap changes."""
    if len(results) < 2:
        print("Need at least 2 checkpoints for statistics")
        return
    
    steps = [r["step"] for r in results]
    overlaps = [r["avg_overlap_ratio"] for r in results]
    
    print("\n" + "=" * 80)
    print("Statistical Summary")
    print("=" * 80)
    
    # Basic stats
    print(f"Number of checkpoints: {len(results)}")
    print(f"Training steps range: {steps[0]} - {steps[-1]}")
    print(f"Initial overlap: {overlaps[0]:.6f}")
    print(f"Final overlap: {overlaps[-1]:.6f}")
    print(f"Total overlap decrease: {overlaps[0] - overlaps[-1]:.6f}")
    print(f"Relative decrease: {(overlaps[0] - overlaps[-1]) / overlaps[0] * 100:.2f}%")
    
    # Linear regression
    if len(steps) > 1:
        z = np.polyfit(steps, overlaps, 1)
        print(f"\nLinear fit: overlap = {z[0]:.8f} * steps + {z[1]:.6f}")
        print(f"Average decrease rate: {-z[0]:.8f} per step")
        
        # R-squared
        y_pred = np.poly1d(z)(steps)
        ss_res = np.sum((np.array(overlaps) - y_pred) ** 2)
        ss_tot = np.sum((np.array(overlaps) - np.mean(overlaps)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        print(f"R-squared: {r_squared:.6f}")
    
    # Per-interval statistics
    print("\n" + "-" * 80)
    print("Per-interval statistics:")
    print("-" * 80)
    print(f"{'Interval':<20} {'Steps':<10} {'Overlap Δ':<15} {'Rate/step':<15}")
    print("-" * 80)
    
    for i in range(1, len(results)):
        step_diff = steps[i] - steps[i-1]
        overlap_diff = overlaps[i-1] - overlaps[i]
        rate = overlap_diff / step_diff if step_diff > 0 else 0
        
        interval = f"{steps[i-1]}-{steps[i]}"
        print(f"{interval:<20} {step_diff:<10} {overlap_diff:<15.6f} {rate:<15.8f}")
    
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Visualize overlap vs training steps")
    parser.add_argument("--result_dir", type=str, required=True,
                        help="Directory containing overlap_results.json")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save plots (default: same as result_dir)")
    parser.add_argument("--no_show", action="store_true",
                        help="Don't display plots interactively")
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from: {args.result_dir}")
    results = load_overlap_results(args.result_dir)
    print(f"Loaded {len(results)} checkpoints")
    
    # Set output directory
    output_dir = args.output_dir if args.output_dir else args.result_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Print statistics
    print_statistics(results)
    
    # Create plots
    show = not args.no_show
    
    print("\nGenerating plots...")
    
    # Plot 1: Overlap vs Steps
    plot_overlap_vs_steps(
        results,
        output_path=os.path.join(output_dir, "overlap_vs_steps.png"),
        show=show
    )
    
    # Plot 2: Decrease Rate
    plot_overlap_decrease_rate(
        results,
        output_path=os.path.join(output_dir, "overlap_decrease_rate.png"),
        show=show
    )
    
    # Plot 3: Combined Analysis
    plot_combined_analysis(
        results,
        output_path=os.path.join(output_dir, "overlap_analysis_combined.png"),
        show=show
    )
    
    print("\n✓ Visualization completed!")
    print(f"Plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
