#!/usr/bin/env python3
"""
Visualization script for Levenberg-Marquardt implementation comparison results.
Creates bar charts comparing execution time, memory usage, and other metrics.
"""

import json
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# Color scheme for different implementations
COLORS = {
    'C++ (No Eigen)': '#1f77b4',
    'C++ (Eigen)': '#ff7f0e',
    'Rust': '#2ca02c',
    'Python (NumPy)': '#d62728',
    'Python (Pure)': '#9467bd'
}

def load_results():
    """Load all result JSON files."""
    results = {}
    base_dir = Path(__file__).parent.parent
    
    result_files = {
        'C++ (No Eigen)': base_dir / 'results_cpp_no_eigen.json',
        'C++ (Eigen)': base_dir / 'results_cpp_eigen.json',
        'Rust': base_dir / 'results_rust.json',
        'Python (NumPy)': base_dir / 'results_python_numpy.json',
        'Python (Pure)': base_dir / 'results_python_pure.json'
    }
    
    for name, filepath in result_files.items():
        if filepath.exists():
            with open(filepath, 'r') as f:
                results[name] = json.load(f)
        else:
            print(f"Warning: {filepath} not found, skipping {name}")
    
    return results

def create_execution_time_chart(results, output_dir):
    """Create bar chart comparing execution times."""
    implementations = []
    times = []
    colors_list = []
    
    for impl, data in results.items():
        implementations.append(impl)
        times.append(data['execution_time_ms'])
        colors_list.append(COLORS.get(impl, '#808080'))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(implementations, times, color=colors_list, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, time in zip(bars, times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{time:.2f} ms',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('Execution Time (ms)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Implementation', fontsize=12, fontweight='bold')
    ax.set_title('Levenberg-Marquardt: Execution Time Comparison', fontsize=14, fontweight='bold', pad=20)
    ax.set_yscale('log')  # Use log scale due to large differences
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'execution_time_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'execution_time_comparison.png'}")
    plt.close()

def create_memory_usage_chart(results, output_dir):
    """Create bar chart comparing memory usage."""
    implementations = []
    memory = []
    colors_list = []
    
    for impl, data in results.items():
        implementations.append(impl)
        memory.append(data['peak_memory_kb'])
        colors_list.append(COLORS.get(impl, '#808080'))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(implementations, memory, color=colors_list, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, mem in zip(bars, memory):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mem:.2f} KB',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('Peak Memory Usage (KB)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Implementation', fontsize=12, fontweight='bold')
    ax.set_title('Levenberg-Marquardt: Memory Usage Comparison', fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'memory_usage_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'memory_usage_comparison.png'}")
    plt.close()

def create_combined_comparison(results, output_dir):
    """Create a combined comparison chart with multiple metrics."""
    implementations = list(results.keys())
    
    # Extract metrics
    times = [results[impl]['execution_time_ms'] for impl in implementations]
    memory = [results[impl]['peak_memory_kb'] for impl in implementations]
    errors = [results[impl]['final_error'] for impl in implementations]
    colors_list = [COLORS.get(impl, '#808080') for impl in implementations]
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Execution Time
    bars1 = ax1.bar(implementations, times, color=colors_list, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Execution Time (ms)', fontsize=11, fontweight='bold')
    ax1.set_title('Execution Time', fontsize=12, fontweight='bold')
    ax1.set_yscale('log')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)
    ax1.tick_params(axis='x', rotation=15)
    for bar, time in zip(bars1, times):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{time:.1f}',
                ha='center', va='bottom', fontsize=8)
    
    # Memory Usage
    bars2 = ax2.bar(implementations, memory, color=colors_list, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Peak Memory (KB)', fontsize=11, fontweight='bold')
    ax2.set_title('Memory Usage', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_axisbelow(True)
    ax2.tick_params(axis='x', rotation=15)
    for bar, mem in zip(bars2, memory):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{mem:.1f}',
                ha='center', va='bottom', fontsize=8)
    
    # Final Error (should be similar)
    bars3 = ax3.bar(implementations, errors, color=colors_list, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax3.set_ylabel('Final Error', fontsize=11, fontweight='bold')
    ax3.set_title('Final Error (Convergence)', fontsize=12, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    ax3.set_axisbelow(True)
    ax3.tick_params(axis='x', rotation=15)
    for bar, err in zip(bars3, errors):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{err:.6f}',
                ha='center', va='bottom', fontsize=8, rotation=90)
    
    plt.suptitle('Levenberg-Marquardt Implementation Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'combined_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'combined_comparison.png'}")
    plt.close()

def create_performance_radar_chart(results, output_dir):
    """Create a radar chart comparing normalized performance metrics."""
    implementations = list(results.keys())
    colors_list = [COLORS.get(impl, '#808080') for impl in implementations]
    
    # Normalize metrics (lower is better for time and memory, error should be similar)
    times = np.array([results[impl]['execution_time_ms'] for impl in implementations])
    memory = np.array([results[impl]['peak_memory_kb'] for impl in implementations])
    
    # Normalize: best = 1.0, worst = 0.0 (inverted since lower is better)
    times_norm = 1.0 - (times - times.min()) / (times.max() - times.min() + 1e-10)
    memory_norm = 1.0 - (memory - memory.min()) / (memory.max() - memory.min() + 1e-10)
    
    # Create radar chart
    categories = ['Speed\n(Lower Time)', 'Memory\n(Lower Usage)']
    N = len(categories)
    
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the circle
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    for i, impl in enumerate(implementations):
        values = [times_norm[i], memory_norm[i]]
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, label=impl, color=colors_list[i])
        ax.fill(angles, values, alpha=0.25, color=colors_list[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.set_title('Normalized Performance Comparison\n(Higher is Better)', 
                 size=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_radar.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'performance_radar.png'}")
    plt.close()

def create_summary_table(results, output_dir):
    """Create a summary table visualization."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data
    table_data = []
    headers = ['Implementation', 'Time (ms)', 'Memory (KB)', 'Iterations', 'Final Error']
    
    for impl, data in results.items():
        row = [
            impl,
            f"{data['execution_time_ms']:.2f}",
            f"{data['peak_memory_kb']:.2f}",
            str(data['iterations']),
            f"{data['final_error']:.6f}"
        ]
        table_data.append(row)
    
    # Create table
    table = ax.table(cellText=table_data,
                    colLabels=headers,
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # Color header row
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color implementation rows
    implementations = list(results.keys())
    for i, impl in enumerate(implementations, start=1):
        color = COLORS.get(impl, '#f0f0f0')
        for j in range(len(headers)):
            table[(i, j)].set_facecolor(color)
            table[(i, j)].set_alpha(0.7)
    
    plt.title('Levenberg-Marquardt Implementation Summary', 
              fontsize=16, fontweight='bold', pad=20)
    plt.savefig(output_dir / 'summary_table.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'summary_table.png'}")
    plt.close()

def main():
    """Main function to generate all visualizations."""
    print("Loading results...")
    results = load_results()
    
    if not results:
        print("Error: No results found. Please ensure result JSON files exist.")
        sys.exit(1)
    
    # Create output directory
    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / 'visualizations'
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nGenerating visualizations in {output_dir}...")
    print("=" * 60)
    
    # Generate all visualizations
    create_execution_time_chart(results, output_dir)
    create_memory_usage_chart(results, output_dir)
    create_combined_comparison(results, output_dir)
    create_performance_radar_chart(results, output_dir)
    create_summary_table(results, output_dir)
    
    print("=" * 60)
    print(f"\nAll visualizations saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - execution_time_comparison.png")
    print("  - memory_usage_comparison.png")
    print("  - combined_comparison.png")
    print("  - performance_radar.png")
    print("  - summary_table.png")

if __name__ == '__main__':
    main()

