# Installing Eigen and CUDA for Windows

## Installing Eigen3

Eigen is a header-only C++ template library for linear algebra.

### Method 1: Using vcpkg (Recommended)

```powershell
# Install vcpkg if not already installed
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat

# Install Eigen3
.\vcpkg install eigen3

# Integrate with Visual Studio
.\vcpkg integrate install
```

Then update CMakeLists.txt to use vcpkg:
```cmake
find_package(Eigen3 REQUIRED)
```

### Method 2: Manual Installation

1. Download Eigen from: https://eigen.tuxfamily.org/index.php?title=Main_Page
2. Extract to a location like `C:\Eigen3`
3. Update CMakeLists.txt to point to Eigen:

```cmake
set(EIGEN3_INCLUDE_DIR "C:/Eigen3")
include_directories(${EIGEN3_INCLUDE_DIR})
```

Or set environment variable:
```powershell
$env:EIGEN3_ROOT = "C:\Eigen3"
```

## Installing CUDA Toolkit

### Prerequisites
- NVIDIA GPU with CUDA support (check: https://developer.nvidia.com/cuda-gpus)
- Visual Studio 2019 or 2022

### Installation Steps

1. **Download CUDA Toolkit**
   - Visit: https://developer.nvidia.com/cuda-downloads
   - Select Windows → x86_64 → 10/11 → exe (local)
   - Download the installer

2. **Run Installer**
   - Run the downloaded `.exe` file
   - Choose "Express" installation (recommended)
   - The installer will:
     - Install CUDA Toolkit
     - Install NVIDIA GPU drivers (if needed)
     - Set up environment variables

3. **Verify Installation**
   ```powershell
   nvcc --version
   ```
   Should show CUDA version (e.g., 12.x)

4. **Set Environment Variables** (if not automatic)
   ```powershell
   $env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x"
   $env:PATH += ";$env:CUDA_PATH\bin"
   ```

5. **Verify GPU Detection**
   ```powershell
   nvidia-smi
   ```

### CMake Configuration

The CMakeLists.txt files should automatically detect CUDA if installed correctly. If not found:

```cmake
set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.x")
```

## Testing After Installation

### Test Eigen
```powershell
cd cpp\eigen
mkdir build
cd build
cmake ..
cmake --build . --config Release
.\Release\lm_eigen.exe
```

### Test CUDA
```powershell
cd cpp\cuda\custom_kernels
mkdir build
cd build
cmake ..
cmake --build . --config Release
.\Release\lm_cuda_custom.exe
```

## Troubleshooting

### Eigen Not Found
- Ensure Eigen headers are in include path
- Check CMakeLists.txt has correct path
- Try setting `EIGEN3_INCLUDE_DIR` explicitly

### CUDA Not Found
- Verify `nvcc --version` works
- Check Visual Studio integration
- Ensure CUDA architecture matches your GPU (check CMakeLists.txt CUDA_ARCHITECTURES)

### CUDA Compilation Errors
- Ensure Visual Studio C++ build tools are installed
- Check CUDA version compatibility with Visual Studio
- Verify GPU compute capability matches CUDA_ARCHITECTURES setting

