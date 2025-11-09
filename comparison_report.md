# Levenberg-Marquardt Performance Comparison

## Summary

| Implementation | Time (ms) | Memory (KB) | Iterations | Final Error |
|----------------|-----------|-------------|------------|-------------|
| C++ (No Eigen) | 6.7130 | 4172.00 | 1000 | 1.209082e+00 |
| C++ (Eigen) | 6.8650 | 4196.00 | 1000 | 1.209082e+00 |
| Rust | 8.7065 | 3920.00 | 1000 | 1.209082e+00 |
| Python (NumPy) | 243.1786 | 37.55 | 1000 | 1.209082e+00 |
| Python (Pure) | 1860.6785 | 32.74 | 1000 | 1.209082e+00 |

## Detailed Results

### C++ (No Eigen)

- **Execution Time**: 6.7130 ms
- **Peak Memory**: 4172.00 KB
- **Iterations**: 1000
- **Final Error**: 1.209082e+00
- **Initial Parameters**: [0.5, 1.0]
- **Final Parameters**: [0.3981248183, 0.9121467429]

### C++ (Eigen)

- **Execution Time**: 6.8650 ms
- **Peak Memory**: 4196.00 KB
- **Iterations**: 1000
- **Final Error**: 1.209082e+00
- **Initial Parameters**: [0.5, 1.0]
- **Final Parameters**: [0.3981248183, 0.9121467429]

### Rust

- **Execution Time**: 8.7065 ms
- **Peak Memory**: 3920.00 KB
- **Iterations**: 1000
- **Final Error**: 1.209082e+00
- **Initial Parameters**: [0.5, 1.0]
- **Final Parameters**: [0.3981248183300331, 0.9121467428795254]

### Python (NumPy)

- **Execution Time**: 243.1786 ms
- **Peak Memory**: 37.55 KB
- **Iterations**: 1000
- **Final Error**: 1.209082e+00
- **Initial Parameters**: [0.5, 1.0]
- **Final Parameters**: [0.3981248183300332, 0.9121467428795255]

### Python (Pure)

- **Execution Time**: 1860.6785 ms
- **Peak Memory**: 32.74 KB
- **Iterations**: 1000
- **Final Error**: 1.209082e+00
- **Initial Parameters**: [0.5, 1.0]
- **Final Parameters**: [0.3981248183300332, 0.9121467428795255]

## Performance Analysis

**Fastest Implementation**: C++ (No Eigen) (6.7130 ms)

**Most Memory Efficient**: Python (Pure) (32.74 KB)

**Most Accurate**: Python (NumPy) (error: 1.209082e+00)
