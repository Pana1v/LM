#!/usr/bin/env python3
"""
Levenberg-Marquardt algorithm implementation in pure Python (no NumPy).
"""

import json
import time
import csv
import sys
import os
import math
import random
import psutil
import threading

# Add parent directory to path to import data generator
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../scripts'))
from generate_data import generate_data

def matmul(A, B):
    """Matrix multiplication: C = A * B"""
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])
    
    if cols_A != rows_B:
        raise ValueError("Matrix dimensions incompatible for multiplication")
    
    C = [[0.0] * cols_B for _ in range(rows_A)]
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                C[i][j] += A[i][k] * B[k][j]
    return C

def matvec(A, v):
    """Matrix-vector multiplication: y = A * v"""
    rows_A = len(A)
    cols_A = len(A[0])
    
    if cols_A != len(v):
        raise ValueError("Matrix and vector dimensions incompatible")
    
    y = [0.0] * rows_A
    for i in range(rows_A):
        for j in range(cols_A):
            y[i] += A[i][j] * v[j]
    return y

def transpose(A):
    """Matrix transpose"""
    rows = len(A)
    cols = len(A[0])
    return [[A[j][i] for j in range(rows)] for i in range(cols)]

def eye(n):
    """Identity matrix of size n x n"""
    I = [[0.0] * n for _ in range(n)]
    for i in range(n):
        I[i][i] = 1.0
    return I

def add_scaled_identity(A, lambda_val):
    """Add scaled identity: A + lambda * I"""
    n = len(A)
    result = [row[:] for row in A]  # Deep copy
    for i in range(n):
        result[i][i] += lambda_val
    return result

def ldlt_solve(A, b):
    """Solve Ax = b using LDLT decomposition"""
    n = len(A)
    if n != len(b):
        raise ValueError("Matrix must be square and match vector size")
    
    # LDLT decomposition: A = L * D * L^T
    L = eye(n)
    D = [0.0] * n
    
    for j in range(n):
        sum_val = 0.0
        for k in range(j):
            sum_val += L[j][k] * L[j][k] * D[k]
        D[j] = A[j][j] - sum_val
        
        if abs(D[j]) < 1e-10:
            raise ValueError("Matrix is singular or near-singular")
        
        for i in range(j + 1, n):
            sum_val = 0.0
            for k in range(j):
                sum_val += L[i][k] * L[j][k] * D[k]
            L[i][j] = (A[i][j] - sum_val) / D[j]
    
    # Forward substitution: L * y = b
    y = [0.0] * n
    for i in range(n):
        sum_val = 0.0
        for k in range(i):
            sum_val += L[i][k] * y[k]
        y[i] = (b[i] - sum_val) / L[i][i]
    
    # Diagonal solve: D * z = y
    z = [y[i] / D[i] for i in range(n)]
    
    # Backward substitution: L^T * x = z
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        sum_val = 0.0
        for k in range(i + 1, n):
            sum_val += L[k][i] * x[k]
        x[i] = z[i] - sum_val
    
    return x

def vec_add(a, b):
    """Vector addition"""
    if len(a) != len(b):
        raise ValueError("Vector dimensions incompatible")
    return [a[i] + b[i] for i in range(len(a))]

def sum_squares(v):
    """Sum of squares of vector elements"""
    return sum(x * x for x in v)

def levenberg_marquardt(data, initial_params, max_iterations, epsilon, lambda_init):
    """
    Levenberg-Marquardt algorithm for non-linear least squares.
    
    Parameters:
    - data: list of [x, y] pairs
    - initial_params: list of initial parameter values
    - max_iterations: maximum number of iterations
    - epsilon: convergence threshold
    - lambda_init: initial damping parameter
    
    Returns:
    - final_params: optimized parameters
    - iterations: number of iterations
    - final_error: final sum of squared residuals
    """
    num_params = len(initial_params)
    params = initial_params[:]  # Copy
    error = float('inf')
    iteration = 0
    lambda_val = lambda_init
    
    while error > epsilon and iteration < max_iterations:
        n = len(data)
        J = [[0.0] * num_params for _ in range(n)]
        r = [0.0] * n
        
        for i in range(n):
            x = data[i][0]
            y = data[i][1]
            f = math.exp(-params[0] * x) * math.cos(params[1] * x)
            
            r[i] = y - f
            
            J[i][0] = x * f
            J[i][1] = -x * math.exp(-params[0] * x) * math.sin(params[1] * x)
        
        # Normal equations: (J^T * J + lambda * I) * delta = J^T * r
        Jt = transpose(J)
        A = matmul(Jt, J)
        A = add_scaled_identity(A, lambda_val)
        b = matvec(Jt, r)
        
        # Solve for parameter update
        delta = ldlt_solve(A, b)
        new_params = vec_add(params, delta)
        
        # Compute new error
        new_r = [0.0] * n
        for i in range(n):
            x = data[i][0]
            y = data[i][1]
            new_f = math.exp(-new_params[0] * x) * math.cos(new_params[1] * x)
            new_r[i] = y - new_f
        
        new_error = sum_squares(new_r)
        
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
    data = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            data.append([float(row[0]), float(row[1])])
    return data

def benchmark():
    """Run benchmark and save results."""
    # Load data
    try:
        data = load_data_from_csv()
    except FileNotFoundError:
        print("Data file not found. Generating data...")
        x, y = generate_data()
        data = [[xi, yi] for xi, yi in zip(x, y)]
    
    initial_params = [0.5, 1.0]
    max_iterations = 1000
    epsilon = 0.000001
    lambda_init = 0.01
    
    # Memory tracking
    process = psutil.Process(os.getpid())
    peak_memory_kb = [0.0]  # Use list to allow modification in nested function
    
    def track_memory():
        """Track peak memory during execution."""
        while not stop_tracking[0]:
            current_memory = process.memory_info().rss / 1024.0  # KB
            if current_memory > peak_memory_kb[0]:
                peak_memory_kb[0] = current_memory
            time.sleep(0.001)  # Sample every 1ms
    
    stop_tracking = [False]
    memory_tracker = threading.Thread(target=track_memory, daemon=True)
    memory_tracker.start()
    
    start_time = time.perf_counter()
    
    # Run algorithm
    final_params, iterations, final_error = levenberg_marquardt(
        data, initial_params, max_iterations, epsilon, lambda_init
    )
    
    end_time = time.perf_counter()
    stop_tracking[0] = True
    memory_tracker.join(timeout=0.1)
    
    # Final check
    final_memory = process.memory_info().rss / 1024.0
    if final_memory > peak_memory_kb[0]:
        peak_memory_kb[0] = final_memory
    
    execution_time_ms = (end_time - start_time) * 1000.0
    
    # Prepare results
    results = {
        "execution_time_ms": execution_time_ms,
        "peak_memory_kb": peak_memory_kb[0],
        "iterations": iterations,
        "final_error": float(final_error),
        "initial_params": initial_params,
        "final_params": final_params
    }
    
    # Print results
    print("Python (Pure) Results:")
    print(f"Initial parameters: {initial_params}")
    print(f"Final parameters: {final_params}")
    print(f"Iterations: {iterations}")
    print(f"Execution time: {execution_time_ms:.4f} ms")
    print(f"Peak memory: {peak_memory_kb[0]:.2f} KB")
    print(f"Final error: {final_error:.10f}")
    
    # Save to JSON
    with open('results_python_pure.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

if __name__ == '__main__':
    benchmark()

