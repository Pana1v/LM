#!/usr/bin/env python3
"""
Generate test data for Levenberg-Marquardt algorithm.
Saves data to CSV format for cross-language compatibility.
"""

import numpy as np
import csv
import os

def generate_data(num_data_points=100, true_params=[0.5, 1.0], noise_std=0.1, seed=42):
    """Generate synthetic data matching the C++ implementation."""
    np.random.seed(seed)
    
    # Generate x values
    x = np.linspace(0, 1, num_data_points)
    
    # Generate y values with noise
    y_true = np.exp(-true_params[0] * x) * np.cos(true_params[1] * x)
    noise = np.random.normal(0.0, noise_std, num_data_points)
    y = y_true + noise
    
    return x, y

def save_to_csv(x, y, filename='data/test_data.csv'):
    """Save data to CSV file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['x', 'y'])  # Header
        for xi, yi in zip(x, y):
            writer.writerow([xi, yi])
    
    print(f"Data saved to {filename}")
    print(f"Generated {len(x)} data points")

if __name__ == '__main__':
    x, y = generate_data()
    save_to_csv(x, y)

