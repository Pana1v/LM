#!/usr/bin/env python3
"""
Levenberg-Marquardt algorithm implementation using NumPy.
"""

import numpy as np
import json
import time
import tracemalloc
import csv
import sys
import os

# Add parent directory to path to import data generator
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../scripts'))
from generate_data import generate_data

def levenberg_marquardt(data, initial_params, max_iterations, epsilon, lambda_init):
    """
    Levenberg-Marquardt algorithm for non-linear least squares.
    
    Parameters:
    - data: numpy array of shape (n, 2) with columns [x, y]
    - initial_params: numpy array of initial parameter values
    - max_iterations: maximum number of iterations
    - epsilon: convergence threshold
    - lambda_init: initial damping parameter
    
    Returns:
    - final_params: optimized parameters
    - iterations: number of iterations
    - final_error: final sum of squared residuals
    """
    num_params = len(initial_params)
    params = initial_params.copy()
    error = float('inf')
    iteration = 0
    lambda_val = lambda_init
    
    while error > epsilon and iteration < max_iterations:
        x = data[:, 0]
        y = data[:, 1]
        
        # Compute residuals and Jacobian
        f = np.exp(-params[0] * x) * np.cos(params[1] * x)
        r = y - f
        
        # Jacobian matrix
        J = np.zeros((len(x), num_params))
        J[:, 0] = x * f
        J[:, 1] = -x * np.exp(-params[0] * x) * np.sin(params[1] * x)
        
        # Normal equations: (J^T * J + lambda * I) * delta = J^T * r
        A = J.T @ J + lambda_val * np.eye(num_params)
        b = J.T @ r
        
        # Solve for parameter update
        delta = np.linalg.solve(A, b)
        new_params = params + delta
        
        # Compute new error
        new_f = np.exp(-new_params[0] * x) * np.cos(new_params[1] * x)
        new_r = y - new_f
        new_error = np.sum(new_r ** 2)
        
        if new_error < error:
            lambda_val /= 10.0
            error = new_error
            params = new_params
        else:
            lambda_val *= 10.0
        
        iteration += 1
    
    return params, iteration, error

def load_data_from_csv(filename='data/test_data.csv'):
    """Load data from CSV file."""
    x, y = [], []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            x.append(float(row[0]))
            y.append(float(row[1]))
    return np.array([x, y]).T

def benchmark():
    """Run benchmark and save results."""
    # Load data
    try:
        data = load_data_from_csv()
    except FileNotFoundError:
        print("Data file not found. Generating data...")
        x, y = generate_data()
        data = np.array([x, y]).T
    
    initial_params = np.array([0.5, 1.0])
    max_iterations = 1000
    epsilon = 0.000001
    lambda_init = 0.01
    
    # Start memory tracking
    tracemalloc.start()
    start_time = time.perf_counter()
    
    # Run algorithm
    final_params, iterations, final_error = levenberg_marquardt(
        data, initial_params, max_iterations, epsilon, lambda_init
    )
    
    end_time = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    execution_time_ms = (end_time - start_time) * 1000.0
    peak_memory_kb = peak / 1024.0
    
    # Prepare results
    results = {
        "execution_time_ms": execution_time_ms,
        "peak_memory_kb": peak_memory_kb,
        "iterations": iterations,
        "final_error": float(final_error),
        "initial_params": initial_params.tolist(),
        "final_params": final_params.tolist()
    }
    
    # Print results
    print("Python (NumPy) Results:")
    print(f"Initial parameters: {initial_params}")
    print(f"Final parameters: {final_params}")
    print(f"Iterations: {iterations}")
    print(f"Execution time: {execution_time_ms:.4f} ms")
    print(f"Peak memory: {peak_memory_kb:.2f} KB")
    print(f"Final error: {final_error:.10f}")
    
    # Save to JSON
    with open('results_python_numpy.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

if __name__ == '__main__':
    benchmark()

