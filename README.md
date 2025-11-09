# Levenberg-Marquardt Multi-Language Implementation

This project implements the Levenberg-Marquardt algorithm for non-linear least squares optimization in multiple programming languages and frameworks, with comprehensive performance benchmarking.

## Project Structure

```
LM/
├── data/
│   └── test_data.csv          # Generated test data (shared across implementations)
├── cpp/
│   ├── eigen/                  # C++ with Eigen
│   │   ├── main.cpp
│   │   └── CMakeLists.txt
│   ├── no_eigen/               # C++ without Eigen
│   │   ├── main.cpp
│   │   ├── matrix_ops.hpp
│   │   └── CMakeLists.txt
│   └── cuda/                   # CUDA implementations
│       ├── custom_kernels/     # Custom CUDA kernels
│       │   ├── main.cpp
│       │   ├── lm_kernels.cu
│       │   └── CMakeLists.txt
│       └── cublas/             # cuBLAS/cuSOLVER version
│           ├── lm_cublas.cu
│           └── CMakeLists.txt
├── python/
│   ├── with_numpy/
│   │   └── lm_numpy.py
│   └── pure_python/
│       └── lm_pure.py
├── rust/
│   ├── src/
│   │   ├── main.rs
│   │   └── lib.rs
│   └── Cargo.toml
├── scripts/
│   ├── generate_data.py        # Generate and save test data
│   └── compare_results.py     # Summary comparison script
└── README.md
```

## Algorithm

The Levenberg-Marquardt algorithm solves non-linear least squares problems by iteratively updating parameters:

- **Model**: `f(x) = exp(-a * x) * cos(b * x)`
- **Parameters**: `[a, b]` (initial: `[0.5, 1.0]`)
- **Data**: 100 points with Gaussian noise (σ = 0.1)

## Prerequisites

### C++ Implementations
- **C++17** compatible compiler (GCC, Clang, or MSVC)
- **CMake** 3.10 or higher
- **Eigen3** (for Eigen version)
- **CUDA Toolkit** (for CUDA versions, optional)

### Python Implementations
- **Python 3.7+**
- **NumPy** (for NumPy version only, install via `pip install -r python/requirements.txt`)

### Rust Implementation
- **Rust** 1.70+ (install from [rustup.rs](https://rustup.rs/))
- **Cargo** (comes with Rust)

## Building and Running

### 1. Generate Test Data

First, generate the test data that will be used by all implementations:

```bash
python scripts/generate_data.py
```

This creates `data/test_data.csv` with 100 data points.

### 2. C++ with Eigen

```bash
cd cpp/eigen
mkdir build && cd build
cmake ..
cmake --build . --config Release
./lm_eigen  # On Windows: .\lm_eigen.exe
```

### 3. C++ without Eigen

```bash
cd cpp/no_eigen
mkdir build && cd build
cmake ..
cmake --build . --config Release
./lm_no_eigen  # On Windows: .\lm_no_eigen.exe
```

### 4. CUDA Custom Kernels

```bash
cd cpp/cuda/custom_kernels
mkdir build && cd build
cmake ..
cmake --build . --config Release
./lm_cuda_custom  # On Windows: .\lm_cuda_custom.exe
```

### 5. CUDA cuBLAS/cuSOLVER

```bash
cd cpp/cuda/cublas
mkdir build && cd build
cmake ..
cmake --build . --config Release
./lm_cuda_cublas  # On Windows: .\lm_cuda_cublas.exe
```

### 6. Python with NumPy

```bash
# Install dependencies
pip install -r python/requirements.txt

# Run
cd python/with_numpy
python lm_numpy.py
```

### 7. Python Pure

```bash
cd python/pure_python
python lm_pure.py
```

### 8. Rust

```bash
cd rust
cargo build --release
cargo run --release
```

## Benchmarking

Each implementation outputs a JSON file with performance metrics:
- `results_cpp_eigen.json`
- `results_cpp_no_eigen.json`
- `results_python_numpy.json`
- `results_python_pure.json`
- `results_rust.json`
- `results_cuda_custom.json`
- `results_cuda_cublas.json`

### Compare Results

After running all implementations, generate a comparison report:

```bash
python scripts/compare_results.py
```

This generates:
- `comparison_report.md` - Markdown report with detailed analysis
- `comparison_report.csv` - CSV file for spreadsheet analysis

## Performance Metrics

Each implementation measures:
- **Execution Time**: Wall-clock time in milliseconds
- **Peak Memory**: Peak memory consumption in KB
- **Iterations**: Number of iterations to convergence
- **Final Error**: Sum of squared residuals
- **Final Parameters**: Optimized parameter values

## Notes

- All implementations use the same test data (generated with seed=42 for reproducibility)
- Convergence threshold: `epsilon = 1e-6`
- Maximum iterations: `1000`
- Initial damping parameter: `lambda = 0.01`

## Troubleshooting

### Eigen3 Not Found
On Linux, install Eigen3:
```bash
sudo apt-get install libeigen3-dev  # Debian/Ubuntu
```

On macOS:
```bash
brew install eigen
```

### CUDA Not Available
CUDA implementations require an NVIDIA GPU with CUDA support. If CUDA is not available, you can skip these implementations.

### Memory Tracking
Memory tracking may vary between platforms:
- Linux: Uses `getrusage` (accurate)
- Windows: Uses `GetProcessMemoryInfo` (accurate)
- Other platforms: May use approximations

## License

This project is provided as-is for educational and benchmarking purposes.

