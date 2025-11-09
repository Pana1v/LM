# Metric Calculation Verification

## Summary of Metric Calculation Methods

### 1. Execution Time

All implementations measure wall-clock time for the algorithm execution:

| Implementation | Method | Unit Conversion |
|----------------|--------|-----------------|
| **C++ (Eigen)** | `std::chrono::high_resolution_clock` | microseconds → ms (÷1000) |
| **C++ (No Eigen)** | `std::chrono::high_resolution_clock` | microseconds → ms (÷1000) |
| **Rust** | `std::time::Instant` | seconds → ms (×1000) |
| **Python (NumPy)** | `time.perf_counter()` | seconds → ms (×1000) |
| **Python (Pure)** | `time.perf_counter()` | seconds → ms (×1000) |

**Status**: ✅ Consistent - All measure algorithm execution time only

---

### 2. Peak Memory Usage

| Implementation | Method | What It Measures |
|----------------|--------|------------------|
| **C++ (Eigen)** | `GetProcessMemoryInfo` (Windows) / `getrusage` (Linux) | Peak Working Set Size (actual process memory) |
| **C++ (No Eigen)** | `GetProcessMemoryInfo` (Windows) / `getrusage` (Linux) | Peak Working Set Size (actual process memory) |
| **Rust** | `GetProcessMemoryInfo` (Windows) / `getrusage` (Linux) | Peak Working Set Size (actual process memory) |
| **Python (NumPy)** | `tracemalloc.get_traced_memory()` | Python heap allocations only |
| **Python (Pure)** | `tracemalloc.get_traced_memory()` | Python heap allocations only |

**Status**: ⚠️ **Inconsistent** - Python measures only heap, not full process memory
- C++/Rust: ~3,900-4,200 KB (includes runtime, libraries, stack)
- Python: ~30-40 KB (only Python objects, excludes interpreter overhead)

**Note**: Python memory measurements are not directly comparable to compiled languages.

---

### 3. Iterations

All implementations count iterations identically:
- Start at 0
- Increment each loop iteration
- Stop when `error <= epsilon` OR `iteration >= max_iterations`

**Status**: ✅ Consistent

---

### 4. Final Error (Sum of Squared Residuals)

**Formula**: `Σ(y_i - f(x_i, params))²` where `f(x, params) = exp(-a*x) * cos(b*x)`

| Implementation | Calculation | Status |
|----------------|-------------|--------|
| **C++ (Eigen)** | ✅ Computes `new_r` with `new_params`, then `sum(new_r²)` | Fixed |
| **C++ (No Eigen)** | ✅ Computes `new_r` with `new_params`, then `sum(new_r²)` | Correct |
| **Rust** | ✅ Computes `new_r` with `new_params`, then `sum(new_r²)` | Correct |
| **Python (NumPy)** | ✅ Computes `new_r` with `new_params`, then `np.sum(new_r²)` | Correct |
| **Python (Pure)** | ✅ Computes `new_r` with `new_params`, then `sum(new_r²)` | Correct |

**Status**: ✅ **Now Consistent** - All compute error with new parameters

**Bug Fixed**: C++ (Eigen) was previously using old residuals `r` instead of computing new residuals with `new_params`.

---

### 5. Final Parameters

All implementations converge to the same parameters:
- **Initial**: `[0.5, 1.0]`
- **Final**: `[0.398125, 0.912147]` (within floating-point precision)

**Status**: ✅ Consistent across all implementations

---

## Current Results (After Fixes)

| Implementation | Time (ms) | Memory (KB) | Error | Parameters Match |
|----------------|-----------|-------------|-------|------------------|
| C++ (No Eigen) | 6.71 | 4,172 | 1.209082 | ✅ |
| C++ (Eigen) | 6.87 | 4,196 | 1.209082 | ✅ |
| Rust | 8.71 | 3,920 | 1.209082 | ✅ |
| Python (NumPy) | 243.18 | 37.55* | 1.209082 | ✅ |
| Python (Pure) | 1,860.68 | 32.74* | 1.209082 | ✅ |

*Python memory is Python heap only, not comparable to compiled languages

---

## Verification Checklist

- [x] Execution time: All use high-resolution timers
- [x] Memory: C++/Rust use actual process memory (consistent)
- [x] Memory: Python uses tracemalloc (noted as different metric)
- [x] Iterations: All count identically
- [x] Final error: All compute with new_params (FIXED)
- [x] Final parameters: All converge to same values

---

## Recommendations

1. **Memory Measurement**: Python's `tracemalloc` only tracks Python heap, not interpreter overhead. For fair comparison, consider using `psutil` or process memory APIs, but note this adds overhead.

2. **Execution Time**: All measurements are accurate and comparable.

3. **Error Calculation**: Now consistent - all use new_params for error computation.

4. **Parameters**: All implementations produce identical results, confirming algorithm correctness.

