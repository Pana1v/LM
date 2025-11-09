# Levenberg-Marquardt Algorithm: Multi-Language Implementation & Performance Analysis

## Executive Summary

This project implements the Levenberg-Marquardt (LM) algorithm for non-linear least squares optimization across multiple programming languages and frameworks. The algorithm is used to fit parameters to a model function `f(x) = exp(-a*x) * cos(b*x)` using 100 data points with Gaussian noise.

**Key Findings:**
- **Fastest Implementation**: C++ (No Eigen) - 6.71 ms
- **Most Memory Efficient (Compiled)**: Rust - 3,920 KB
- **All implementations converge to identical parameters**: [0.398125, 0.912147]
- **All implementations produce identical final error**: 1.209082

---

## Table of Contents

1. [Approach & Methodology](#approach--methodology)
2. [Implementation Details](#implementation-details)
3. [Performance Results](#performance-results)
4. [Metric Verification](#metric-verification)
5. [Analysis & Insights](#analysis--insights)
6. [Code Quality & Consistency](#code-quality--consistency)
7. [Conclusion](#conclusion)

---

## Approach & Methodology

### Algorithm Overview

The Levenberg-Marquardt algorithm solves non-linear least squares problems by iteratively updating parameters:

1. **Model Function**: `f(x) = exp(-a*x) * cos(b*x)`
2. **Parameters**: `[a, b]` (initial: `[0.5, 1.0]`)
3. **Objective**: Minimize sum of squared residuals `Σ(y_i - f(x_i))²`

### Algorithm Steps

For each iteration:
1. Compute residuals `r = y - f(x, params)`
2. Compute Jacobian matrix `J` (partial derivatives)
3. Form normal equations: `(J^T * J + λ*I) * δ = J^T * r`
4. Solve for parameter update `δ`
5. Compute new error with updated parameters
6. If error decreases: accept update, decrease damping `λ`
7. If error increases: reject update, increase damping `λ`

### Test Data

- **Points**: 100 data points
- **X range**: [0, 1] (uniformly spaced)
- **Noise**: Gaussian N(0, 0.1)
- **Seed**: 42 (for reproducibility)
- **Format**: CSV file shared across all implementations

### Convergence Criteria

- **Maximum iterations**: 1,000
- **Convergence threshold**: ε = 1e-6
- **Initial damping**: λ = 0.01

---

## Implementation Details

### 1. C++ with Eigen (`cpp/eigen/`)

**Library**: Eigen3 (header-only linear algebra library)

**Key Features**:
- Uses Eigen's optimized matrix operations
- LDLT decomposition for solving linear systems
- High-performance BLAS/LAPACK backend

**Build System**: CMake with Eigen3 detection

**Code Structure**:
```cpp
// Matrix operations using Eigen
Eigen::MatrixXd J(data.rows(), num_params);
Eigen::VectorXd r(data.rows());
Eigen::MatrixXd A = J.transpose() * J + lambda * Identity;
Eigen::VectorXd delta = A.ldlt().solve(b);
```

### 2. C++ without Eigen (`cpp/no_eigen/`)

**Approach**: Custom matrix operations implementation

**Custom Implementations**:
- Matrix multiplication (`matmul`)
- Matrix transpose (`transpose`)
- LDLT decomposition (`ldlt_solve`)
- Vector operations (`vec_add`, `sum_squares`)

**Data Structure**: `std::vector<double>` with custom Matrix/Vector classes

**Advantages**: No external dependencies, full control over operations

### 3. Rust (`rust/`)

**Library**: `ndarray` crate for multi-dimensional arrays

**Key Features**:
- Memory-safe implementation
- Zero-cost abstractions
- Efficient array operations

**Code Structure**:
```rust
let J = Array2::<f64>::zeros((n, num_params));
let A = Jt.dot(&J) + lambda_val * Array2::<f64>::eye(num_params);
let delta = solve_ldlt(&A, &b);
```

### 4. Python with NumPy (`python/with_numpy/`)

**Library**: NumPy for numerical computations

**Key Features**:
- High-level array operations
- Optimized C implementations under the hood
- `numpy.linalg.solve` for linear system solving

**Code Structure**:
```python
J = np.zeros((len(x), num_params))
A = J.T @ J + lambda_val * np.eye(num_params)
delta = np.linalg.solve(A, b)
```

### 5. Python Pure (`python/pure_python/`)

**Approach**: Pure Python with manual matrix operations

**Custom Implementations**:
- Matrix multiplication using nested loops
- Manual LDLT decomposition
- List-based data structures

**Purpose**: Baseline comparison without optimized libraries

### 6. CUDA Implementations (`cpp/cuda/`)

**Status**: Implemented but not tested (CUDA Toolkit requires admin installation)

**Two Variants**:
1. **Custom Kernels**: Manual CUDA kernel implementations
2. **cuBLAS/cuSOLVER**: Using NVIDIA's optimized libraries

---

## Performance Results

### Summary Table

| Implementation | Time (ms) | Memory (KB) | Iterations | Final Error | Final Parameters |
|----------------|-----------|-------------|------------|-------------|------------------|
| **C++ (No Eigen)** | **6.71** | 4,172 | 1,000 | 1.209082 | [0.398125, 0.912147] |
| **C++ (Eigen)** | 6.87 | 4,196 | 1,000 | 1.209082 | [0.398125, 0.912147] |
| **Rust** | 8.71 | 3,920 | 1,000 | 1.209082 | [0.398125, 0.912147] |
| **Python (NumPy)** | 243.18 | 37.55* | 1,000 | 1.209082 | [0.398125, 0.912147] |
| **Python (Pure)** | 1,860.68 | 32.74* | 1,000 | 1.209082 | [0.398125, 0.912147] |

*Python memory measurements track only Python heap allocations, not full process memory

### Detailed Results

#### C++ (No Eigen)
- **Execution Time**: 6.7130 ms
- **Peak Memory**: 4,172 KB
- **Iterations**: 1,000 (max iterations reached)
- **Final Error**: 1.209082e+00
- **Final Parameters**: [0.3981248183, 0.9121467429]

#### C++ (Eigen)
- **Execution Time**: 6.8650 ms
- **Peak Memory**: 4,196 KB
- **Iterations**: 1,000 (max iterations reached)
- **Final Error**: 1.209082e+00
- **Final Parameters**: [0.3981248183, 0.9121467429]

#### Rust
- **Execution Time**: 8.7065 ms
- **Peak Memory**: 3,920 KB
- **Iterations**: 1,000 (max iterations reached)
- **Final Error**: 1.209082e+00
- **Final Parameters**: [0.3981248183300331, 0.9121467428795254]

#### Python (NumPy)
- **Execution Time**: 243.1786 ms
- **Peak Memory**: 37.55 KB (Python heap only)
- **Iterations**: 1,000 (max iterations reached)
- **Final Error**: 1.209082e+00
- **Final Parameters**: [0.3981248183300332, 0.9121467428795255]

#### Python (Pure)
- **Execution Time**: 1,860.6785 ms
- **Peak Memory**: 32.74 KB (Python heap only)
- **Iterations**: 1,000 (max iterations reached)
- **Final Error**: 1.209082e+00
- **Final Parameters**: [0.3981248183300332, 0.9121467428795255]

### Performance Comparison

#### Speed Comparison (Relative to Python NumPy)

| Implementation | Speedup | Time (ms) |
|----------------|--------|-----------|
| C++ (No Eigen) | **36.2x** | 6.71 |
| C++ (Eigen) | 35.4x | 6.87 |
| Rust | 27.9x | 8.71 |
| Python (NumPy) | 1.0x (baseline) | 243.18 |
| Python (Pure) | 0.13x | 1,860.68 |

#### Memory Comparison (Compiled Languages Only)

| Implementation | Memory (KB) | Relative to Rust |
|----------------|-------------|------------------|
| Rust | 3,920 | 1.0x (baseline) |
| C++ (No Eigen) | 4,172 | 1.06x |
| C++ (Eigen) | 4,196 | 1.07x |

**Note**: Python memory measurements are not comparable (see Metric Verification section)

---

## Metric Verification

### Execution Time Measurement

All implementations use high-resolution timers:

| Implementation | Method | Precision |
|----------------|--------|-----------|
| C++ (Eigen) | `std::chrono::high_resolution_clock` | Microseconds |
| C++ (No Eigen) | `std::chrono::high_resolution_clock` | Microseconds |
| Rust | `std::time::Instant` | Nanoseconds |
| Python (NumPy) | `time.perf_counter()` | Nanoseconds |
| Python (Pure) | `time.perf_counter()` | Nanoseconds |

**Status**: ✅ All measurements are accurate and comparable

### Memory Measurement

| Implementation | Method | What It Measures |
|----------------|--------|------------------|
| C++ (Eigen) | `GetProcessMemoryInfo` (Windows) | Peak Working Set Size |
| C++ (No Eigen) | `GetProcessMemoryInfo` (Windows) | Peak Working Set Size |
| Rust | `GetProcessMemoryInfo` (Windows) | Peak Working Set Size |
| Python (NumPy) | `tracemalloc.get_traced_memory()` | Python heap allocations only |
| Python (Pure) | `tracemalloc.get_traced_memory()` | Python heap allocations only |

**Status**: ⚠️ **Python measurements are not comparable** - they only track Python heap, not interpreter overhead

**Compiled Languages**: All use the same Windows API, measurements are directly comparable (~3,900-4,200 KB)

### Error Calculation

All implementations compute the sum of squared residuals using **new parameters**:

```cpp
// Correct approach (all implementations)
new_params = params + delta;
new_r = y - f(x, new_params);  // Compute residuals with new params
new_error = sum(new_r²);
```

**Status**: ✅ All implementations verified to use correct approach

### Iteration Counting

All implementations:
- Start at iteration 0
- Increment each loop iteration
- Stop when `error <= epsilon` OR `iteration >= max_iterations`

**Status**: ✅ Consistent across all implementations

### Parameter Convergence

All implementations converge to:
- **Initial**: [0.5, 1.0]
- **Final**: [0.398125, 0.912147] (within floating-point precision)

**Status**: ✅ Identical results confirm algorithm correctness

---

## Analysis & Insights

### Performance Insights

1. **C++ Performance**: Both Eigen and No-Eigen versions are nearly identical in speed (~6.7-6.9 ms), suggesting:
   - Eigen's optimizations are effective
   - Custom implementation is well-optimized
   - Matrix size (2x2) is too small to see significant Eigen advantages

2. **Rust Performance**: Slightly slower than C++ (~8.7 ms) but:
   - Memory usage is lowest among compiled languages
   - Safety guarantees come with minimal overhead
   - Performance is still excellent (27.9x faster than Python NumPy)

3. **Python Performance**:
   - **NumPy**: 243 ms (36x slower than C++) - overhead from Python interpreter
   - **Pure Python**: 1,861 ms (277x slower than C++) - demonstrates importance of optimized libraries

4. **Memory Usage**:
   - Compiled languages: ~4,000 KB (includes runtime, libraries, stack)
   - Python: ~30-40 KB (only Python objects, excludes interpreter)
   - Rust uses slightly less memory than C++ implementations

### Algorithm Behavior

- **Convergence**: All implementations reach max iterations (1,000) before convergence
- **Final Error**: Identical across all implementations (1.209082)
- **Parameters**: All converge to same values, confirming correctness

### Implementation Quality

**Strengths**:
- ✅ All implementations produce identical results
- ✅ Consistent error calculation (fixed bugs during verification)
- ✅ Proper memory measurement (fixed Rust estimate)
- ✅ Same test data across all implementations
- ✅ Comprehensive benchmarking metrics

**Areas for Improvement**:
- Python memory measurement could use process-level APIs for fair comparison
- CUDA implementations not tested (requires admin installation)
- Could add more data points or different problem sizes for scalability analysis

---

## Code Quality & Consistency

### Code Structure

All implementations follow the same algorithm structure:

1. **Data Loading**: Load from CSV or generate with same seed
2. **Initialization**: Same initial parameters [0.5, 1.0]
3. **Main Loop**:
   - Compute residuals and Jacobian
   - Form normal equations
   - Solve for parameter update
   - Compute new error with new parameters
   - Update or reject based on error
4. **Output**: JSON format with consistent fields

### Consistency Checks

- ✅ Same test data (CSV file)
- ✅ Same initial parameters
- ✅ Same convergence criteria
- ✅ Same algorithm logic
- ✅ Same output format (JSON)

### Bugs Fixed During Development

1. **C++ (Eigen)**: Fixed error calculation to use new_params instead of old residuals
2. **C++ (No Eigen)**: Fixed error calculation to use new_params
3. **Rust**: Fixed memory measurement from estimate to actual peak memory
4. **All**: Ensured CSV data loading for consistency

---

## Conclusion

### Key Takeaways

1. **Performance**: Compiled languages (C++, Rust) are 27-36x faster than Python NumPy
2. **Consistency**: All implementations produce identical results, confirming correctness
3. **Memory**: Compiled languages use ~4,000 KB; Python measurements are not comparable
4. **Eigen vs Custom**: For small matrices, custom implementation performs similarly to Eigen
5. **Rust**: Excellent performance with memory safety, only slightly slower than C++

### Recommendations

1. **For Production**: Use C++ (Eigen) for best performance with library support
2. **For Safety**: Use Rust for memory safety with excellent performance
3. **For Prototyping**: Use Python (NumPy) for rapid development
4. **For Learning**: Pure Python provides clear algorithm understanding

### Future Work

1. Test CUDA implementations when CUDA Toolkit is available
2. Scale to larger problems (more data points, more parameters)
3. Add more sophisticated convergence criteria
4. Implement adaptive damping strategies
5. Add parallel implementations for comparison

---

## Appendix

### Project Structure

```
LM/
├── data/
│   └── test_data.csv          # Shared test data
├── cpp/
│   ├── eigen/                  # C++ with Eigen
│   ├── no_eigen/               # C++ without Eigen
│   └── cuda/                   # CUDA implementations
├── python/
│   ├── with_numpy/             # Python with NumPy
│   └── pure_python/            # Pure Python
├── rust/                       # Rust implementation
├── scripts/
│   ├── generate_data.py        # Data generation
│   └── compare_results.py      # Results comparison
└── README.md                   # Build instructions
```

### Build Instructions

See `README.md` for detailed build instructions for each implementation.

### Data Files

- `data/test_data.csv`: 100 data points with header (x, y)
- Generated with seed=42 for reproducibility

### Result Files

Each implementation outputs JSON results:
- `results_cpp_eigen.json`
- `results_cpp_no_eigen.json`
- `results_rust.json`
- `results_python_numpy.json`
- `results_python_pure.json`

### Comparison Reports

- `comparison_report.md`: Markdown comparison table
- `comparison_report.csv`: CSV data for analysis
- `METRIC_VERIFICATION.md`: Detailed metric verification

---

**Report Generated**: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")
**Test System**: Windows 10, Visual Studio 2022, Rust 1.70+, Python 3.12

2025-11-09 21:09:49
