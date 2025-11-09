#!/usr/bin/env python3
"""
Compare benchmark results from all implementations.
Generates a summary report with performance metrics.
"""

import json
import os
import glob
from pathlib import Path

def load_json_results(filename):
    """Load results from JSON file."""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        print(f"Warning: Could not parse {filename}")
        return None

def find_result_files():
    """Find all result JSON files."""
    result_files = {
        'C++ (Eigen)': 'results_cpp_eigen.json',
        'C++ (No Eigen)': 'results_cpp_no_eigen.json',
        'Python (NumPy)': 'results_python_numpy.json',
        'Python (Pure)': 'results_python_pure.json',
        'Rust': 'results_rust.json',
        'CUDA (Custom)': 'results_cuda_custom.json',
        'CUDA (cuBLAS)': 'results_cuda_cublas.json',
    }
    return result_files

def generate_markdown_report(results_dict):
    """Generate markdown comparison report."""
    report = []
    report.append("# Levenberg-Marquardt Performance Comparison\n")
    report.append("## Summary\n")
    
    # Table header
    report.append("| Implementation | Time (ms) | Memory (KB) | Iterations | Final Error |")
    report.append("|----------------|-----------|-------------|------------|-------------|")
    
    # Sort by execution time
    sorted_results = sorted(
        [(name, data) for name, data in results_dict.items() if data is not None],
        key=lambda x: x[1]['execution_time_ms']
    )
    
    for name, data in sorted_results:
        report.append(
            f"| {name} | {data['execution_time_ms']:.4f} | "
            f"{data['peak_memory_kb']:.2f} | {data['iterations']} | "
            f"{data['final_error']:.6e} |"
        )
    
    report.append("\n## Detailed Results\n")
    
    for name, data in sorted_results:
        report.append(f"### {name}\n")
        report.append(f"- **Execution Time**: {data['execution_time_ms']:.4f} ms")
        report.append(f"- **Peak Memory**: {data['peak_memory_kb']:.2f} KB")
        report.append(f"- **Iterations**: {data['iterations']}")
        report.append(f"- **Final Error**: {data['final_error']:.6e}")
        report.append(f"- **Initial Parameters**: {data['initial_params']}")
        report.append(f"- **Final Parameters**: {data['final_params']}")
        report.append("")
    
    # Performance analysis
    report.append("## Performance Analysis\n")
    
    if sorted_results:
        fastest = sorted_results[0]
        report.append(f"**Fastest Implementation**: {fastest[0]} ({fastest[1]['execution_time_ms']:.4f} ms)\n")
        
        # Find most memory efficient
        most_memory_efficient = min(
            sorted_results,
            key=lambda x: x[1]['peak_memory_kb']
        )
        report.append(f"**Most Memory Efficient**: {most_memory_efficient[0]} ({most_memory_efficient[1]['peak_memory_kb']:.2f} KB)\n")
        
        # Find most accurate (lowest error)
        most_accurate = min(
            sorted_results,
            key=lambda x: x[1]['final_error']
        )
        report.append(f"**Most Accurate**: {most_accurate[0]} (error: {most_accurate[1]['final_error']:.6e})\n")
    
    return "\n".join(report)

def generate_csv_report(results_dict):
    """Generate CSV comparison report."""
    lines = []
    lines.append("Implementation,Execution Time (ms),Peak Memory (KB),Iterations,Final Error,Initial Params,Final Params")
    
    sorted_results = sorted(
        [(name, data) for name, data in results_dict.items() if data is not None],
        key=lambda x: x[1]['execution_time_ms']
    )
    
    for name, data in sorted_results:
        init_params = f"[{data['initial_params'][0]},{data['initial_params'][1]}]"
        final_params = f"[{data['final_params'][0]},{data['final_params'][1]}]"
        lines.append(
            f"{name},{data['execution_time_ms']:.6f},{data['peak_memory_kb']:.2f},"
            f"{data['iterations']},{data['final_error']:.6e},{init_params},{final_params}"
        )
    
    return "\n".join(lines)

def main():
    """Main comparison function."""
    result_files = find_result_files()
    results_dict = {}
    
    print("Loading results...")
    for name, filename in result_files.items():
        result = load_json_results(filename)
        if result:
            results_dict[name] = result
            print(f"  + Loaded {name}")
        else:
            print(f"  X Missing {name} ({filename})")
    
    if not results_dict:
        print("No results found! Please run the implementations first.")
        return
    
    # Generate reports
    print("\nGenerating reports...")
    
    markdown_report = generate_markdown_report(results_dict)
    with open('comparison_report.md', 'w') as f:
        f.write(markdown_report)
    print("  + Generated comparison_report.md")
    
    csv_report = generate_csv_report(results_dict)
    with open('comparison_report.csv', 'w') as f:
        f.write(csv_report)
    print("  + Generated comparison_report.csv")
    
    # Print summary to console
    print("\n" + "="*80)
    print(markdown_report)
    print("="*80)

if __name__ == '__main__':
    main()

