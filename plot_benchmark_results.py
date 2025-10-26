#!/usr/bin/env python3
"""
Benchmark Results Visualization Script

This script generates publication-quality plots for analyzing MCNF solver benchmark results.
It creates focused plots emphasizing successful runs and solution quality.

Usage:
    python plot_benchmark_results.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')

# Configuration
RESULTS_DIR = Path("benchmark_results")
FIGURES_DIR = RESULTS_DIR / "figures"
DETAILED_CSV = RESULTS_DIR / "detailed_results.csv"
SUMMARY_CSV = RESULTS_DIR / "summary_results.csv"

# Plot styling
ALGORITHMS = ['simplex', 'column_generation', 'dual', 'interior_point']
ALGORITHM_LABELS = {
    'simplex': 'Simplex',
    'column_generation': 'Column Gen',
    'dual': 'Dual Decomp',
    'interior_point': 'Interior Point'
}

# Colorblind-friendly palette
COLORS = {
    'simplex': '#1f77b4',           # blue
    'column_generation': '#ff7f0e', # orange
    'dual': '#2ca02c',              # green
    'interior_point': '#d62728'     # red
}

# Best known objective values from LMCF benchmark instances
# Source: tests/LMCF/LMCF_Instances.md
BEST_OBJECTIVES = {
    # Grid Demands
    'GridDemands/gd1': 8.27323e5,
    'GridDemands/gd2': 1.70538e6,
    'GridDemands/gd3': 1.52464e6,
    'GridDemands/gd4': 3.03170e6,
    'GridDemands/gd5': 5.04970e6,
    'GridDemands/gd6': 1.04007e7,
    'GridDemands/gd7': 2.58641e7,
    'GridDemands/gd8': 4.17113e7,
    'GridDemands/gd9': 8.26533e7,

    # Planar Networks
    'PlanarNetworks/pl30': 4.43508e7,
    'PlanarNetworks/pl50': 1.22200e8,
    'PlanarNetworks/pl80': 1.82438e8,
    'PlanarNetworks/pl100': 2.31340e8,
    'PlanarNetworks/pl150': 5.48089e8,

    # Traffic Networks
    'TrafficNetworks/siou': 3.20184e5,
    'TrafficNetworks/win': 2.94065e7,
    'TrafficNetworks/chica': 5.49053e1,

    # Other Demands
    'OtherDemands/22': 1.88237e3,
    'OtherDemands/148': 1.39500e5,
    'OtherDemands/904': 1.37850e7,
}

# Plot configuration
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'


class BenchmarkVisualizer:
    """Class for creating benchmark visualization plots."""

    def __init__(self):
        """Initialize the visualizer and load data."""
        self.detailed_df = pd.read_csv(DETAILED_CSV)
        self.summary_df = pd.read_csv(SUMMARY_CSV)

        # Create figures directory
        FIGURES_DIR.mkdir(exist_ok=True)

        print(f"Loaded {len(self.detailed_df)} detailed results")
        print(f"Loaded {len(self.summary_df)} summary results")
        print(f"Figures will be saved to: {FIGURES_DIR}")

    def save_figure(self, filename: str, fmt: str = 'png'):
        """Save the current figure."""
        filepath = FIGURES_DIR / f"{filename}.{fmt}"
        plt.savefig(filepath, format=fmt, bbox_inches='tight')
        print(f"  Saved: {filepath}")

    def plot_overall_success_rate(self):
        """Plot 1: Overall success rate across all instances."""
        print("\n[1/5] Generating overall success rate...")

        # Calculate overall success rate for each algorithm
        success_data = []

        for algo in ALGORITHMS:
            algo_df = self.detailed_df[self.detailed_df['implementation'] == algo]
            total = len(algo_df)

            if total > 0:
                success_count = len(algo_df[algo_df['status'] == 'success'])
                success_rate = (success_count / total) * 100
            else:
                success_rate = 0

            success_data.append({
                'algorithm': ALGORITHM_LABELS[algo],
                'success_rate': success_rate
            })

        df = pd.DataFrame(success_data)

        # Create bar chart
        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(df))
        bars = ax.bar(x, df['success_rate'], width=0.6,
                      color=[COLORS[algo] for algo in ALGORITHMS],
                      alpha=0.8, edgecolor='black', linewidth=1.2)

        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, df['success_rate'])):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.1f}%',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax.set_xlabel('Algorithm', fontsize=13)
        ax.set_ylabel('Success Rate (%)', fontsize=13)
        ax.set_title('Overall Success Rate Across All Instances', fontsize=15, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(df['algorithm'])
        ax.set_ylim([0, 105])
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        plt.tight_layout()
        self.save_figure('01_overall_success_rate')
        plt.close()

    def plot_average_time(self):
        """Plot 2: Average computation time for successful runs."""
        print("[2/5] Generating average computation time...")

        # Get successful runs only
        success_df = self.detailed_df[self.detailed_df['status'] == 'success'].copy()

        if len(success_df) == 0:
            print("  Skipped: No successful runs")
            return

        # Calculate mean and std for each algorithm
        time_stats = []

        for algo in ALGORITHMS:
            algo_data = success_df[success_df['implementation'] == algo]['time']

            if len(algo_data) > 0:
                mean_time = algo_data.mean()
                std_time = algo_data.std()
                time_stats.append({
                    'algorithm': ALGORITHM_LABELS[algo],
                    'mean': mean_time,
                    'std': std_time if not pd.isna(std_time) else 0
                })

        if len(time_stats) == 0:
            print("  Skipped: No data available")
            return

        df = pd.DataFrame(time_stats)

        # Create bar chart with error bars
        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(df))
        bars = ax.bar(x, df['mean'], width=0.6,
                     yerr=df['std'], capsize=5,
                     color=[COLORS[algo] for algo in ALGORITHMS[:len(df)]],
                     alpha=0.8, edgecolor='black', linewidth=1.2,
                     error_kw={'linewidth': 2, 'ecolor': 'black'})

        ax.set_xlabel('Algorithm', fontsize=13)
        ax.set_ylabel('Average Time (seconds)', fontsize=13)
        ax.set_title('Average Computation Time', fontsize=15, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(df['algorithm'])
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        plt.tight_layout()
        self.save_figure('02_average_time')
        plt.close()

    def plot_time_distribution(self):
        """Plot 3: Time distribution for successful runs."""
        print("[3/5] Generating time distribution...")

        # Get successful runs only
        success_df = self.detailed_df[self.detailed_df['status'] == 'success'].copy()

        if len(success_df) == 0:
            print("  Skipped: No successful runs")
            return

        # Prepare data for violin plot (need data for each algorithm)
        plot_data = []
        for algo in ALGORITHMS:
            algo_data = success_df[success_df['implementation'] == algo]
            if len(algo_data) > 0:
                for _, row in algo_data.iterrows():
                    plot_data.append({
                        'Algorithm': ALGORITHM_LABELS[algo],
                        'Time': row['time']
                    })

        if len(plot_data) == 0:
            print("  Skipped: No data available")
            return

        # Create DataFrame for seaborn
        plot_df = pd.DataFrame(plot_data)

        # Create violin plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # Create violin plot with seaborn
        parts = ax.violinplot(
            [plot_df[plot_df['Algorithm'] == ALGORITHM_LABELS[algo]]['Time'].values
             for algo in ALGORITHMS if ALGORITHM_LABELS[algo] in plot_df['Algorithm'].values],
            positions=range(len([algo for algo in ALGORITHMS if ALGORITHM_LABELS[algo] in plot_df['Algorithm'].values])),
            widths=0.7,
            showmeans=False,
            showmedians=False,
            showextrema=False
        )

        # Color the violins
        active_algos = [algo for algo in ALGORITHMS if ALGORITHM_LABELS[algo] in plot_df['Algorithm'].values]
        for i, (pc, algo) in enumerate(zip(parts['bodies'], active_algos)):
            pc.set_facecolor(COLORS[algo])
            pc.set_alpha(0.7)
            pc.set_edgecolor('black')
            pc.set_linewidth(1.5)

        # Add median lines and quartiles
        for i, algo in enumerate(active_algos):
            algo_times = plot_df[plot_df['Algorithm'] == ALGORITHM_LABELS[algo]]['Time'].values
            median = np.median(algo_times)
            q1 = np.percentile(algo_times, 25)
            q3 = np.percentile(algo_times, 75)

            # Draw median line
            ax.plot([i-0.15, i+0.15], [median, median], color='white', linewidth=3, zorder=10)
            ax.plot([i-0.15, i+0.15], [median, median], color='red', linewidth=2, zorder=11)

            # Draw quartile markers
            ax.scatter([i], [q1], color='white', s=100, zorder=10, edgecolors='black', linewidths=1.5)
            ax.scatter([i], [q3], color='white', s=100, zorder=10, edgecolors='black', linewidths=1.5)

        # Set labels
        ax.set_xticks(range(len(active_algos)))
        ax.set_xticklabels([ALGORITHM_LABELS[algo] for algo in active_algos])
        ax.set_ylabel('Time (seconds, log scale)', fontsize=13)
        ax.set_xlabel('Algorithm', fontsize=13)
        ax.set_title('Computation Time Distribution (Successful Runs)', fontsize=15, fontweight='bold')
        ax.set_yscale('log')
        ax.grid(axis='y', alpha=0.3, linestyle='--', which='both')

        plt.tight_layout()
        self.save_figure('03_time_distribution')
        plt.close()

    def plot_solution_precision(self):
        """Plot 4: Solution precision compared to best known objectives."""
        print("[4/5] Generating solution precision...")

        # Get successful runs only
        success_df = self.detailed_df[
            (self.detailed_df['status'] == 'success') &
            (self.detailed_df['objective'].notna())
        ].copy()

        if len(success_df) == 0:
            print("  Skipped: No successful runs with objectives")
            return

        # Calculate relative errors
        errors_by_algo = {algo: [] for algo in ALGORITHMS}

        for _, row in success_df.iterrows():
            instance = row['instance_name']
            achieved = row['objective']
            algo = row['implementation']

            if instance in BEST_OBJECTIVES:
                best = BEST_OBJECTIVES[instance]
                # Calculate relative error as percentage
                rel_error = abs(achieved - best) / best * 100
                errors_by_algo[algo].append(rel_error)

        # Prepare data for plotting
        plot_data = []
        labels = []
        colors_list = []

        for algo in ALGORITHMS:
            if len(errors_by_algo[algo]) > 0:
                plot_data.append(errors_by_algo[algo])
                labels.append(ALGORITHM_LABELS[algo])
                colors_list.append(COLORS[algo])

        if len(plot_data) == 0:
            print("  Skipped: No data with known best objectives")
            return

        # Create box plot
        fig, ax = plt.subplots(figsize=(10, 6))

        bp = ax.boxplot(plot_data, labels=labels, patch_artist=True,
                       widths=0.6, showfliers=True,
                       boxprops=dict(linewidth=1.5),
                       whiskerprops=dict(linewidth=1.5),
                       capprops=dict(linewidth=1.5),
                       medianprops=dict(linewidth=2, color='darkred'))

        # Color the boxes
        for patch, color in zip(bp['boxes'], colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_ylabel('Relative Error (%)', fontsize=13)
        ax.set_xlabel('Algorithm', fontsize=13)
        ax.set_title('Solution Precision vs Best Known Objectives', fontsize=15, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # Add horizontal line at 0 for reference
        ax.axhline(y=0, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Perfect')
        ax.legend()

        plt.tight_layout()
        self.save_figure('04_solution_precision')
        plt.close()

    def plot_iteration_distribution(self):
        """Plot 5: Iteration count distribution for successful runs."""
        print("[5/5] Generating iteration count distribution...")

        # Get successful runs with iteration data
        success_df = self.detailed_df[
            (self.detailed_df['status'] == 'success') &
            (self.detailed_df['iterations'].notna())
        ].copy()

        if len(success_df) == 0:
            print("  Skipped: No successful runs with iteration data")
            return

        # Prepare data for box plot
        plot_data = []
        labels = []
        colors_list = []

        for algo in ALGORITHMS:
            algo_data = success_df[success_df['implementation'] == algo]['iterations']

            if len(algo_data) > 0:
                plot_data.append(algo_data)
                labels.append(ALGORITHM_LABELS[algo])
                colors_list.append(COLORS[algo])

        if len(plot_data) == 0:
            print("  Skipped: No data available")
            return

        # Create box plot
        fig, ax = plt.subplots(figsize=(10, 6))

        bp = ax.boxplot(plot_data, labels=labels, patch_artist=True,
                       widths=0.6, showfliers=True,
                       boxprops=dict(linewidth=1.5),
                       whiskerprops=dict(linewidth=1.5),
                       capprops=dict(linewidth=1.5),
                       medianprops=dict(linewidth=2, color='red'))

        # Color the boxes
        for patch, color in zip(bp['boxes'], colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_ylabel('Number of Iterations (log scale)', fontsize=13)
        ax.set_xlabel('Algorithm', fontsize=13)
        ax.set_title('Iteration Count Distribution (Successful Runs)', fontsize=15, fontweight='bold')
        ax.set_yscale('log')
        ax.grid(axis='y', alpha=0.3, linestyle='--', which='both')

        plt.tight_layout()
        self.save_figure('05_iteration_distribution')
        plt.close()

    def generate_all_plots(self):
        """Generate all plots."""
        print("\n" + "="*60)
        print("Generating Benchmark Visualization Plots")
        print("="*60)

        self.plot_overall_success_rate()
        self.plot_average_time()
        self.plot_time_distribution()
        self.plot_solution_precision()
        self.plot_iteration_distribution()

        print("\n" + "="*60)
        print(f"All plots generated successfully!")
        print(f"Figures saved to: {FIGURES_DIR.absolute()}")
        print("="*60)


def main():
    """Main function."""
    visualizer = BenchmarkVisualizer()
    visualizer.generate_all_plots()


if __name__ == "__main__":
    main()
