#include <iostream>
#include <random>
#include <cmath>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <vector>
#include <limits>
#include <cuda_runtime.h>

// CUDA kernel declarations (defined in lm_kernels.cu)
extern void compute_residuals_and_jacobian(
    const double* data_x, const double* data_y,
    const double* params, double* residuals, double* jacobian,
    int num_points);

extern void matrix_multiply_AtA(
    const double* A, double* AtA, int rows, int cols);

extern void matrix_vector_multiply_At(
    const double* A, const double* v, double* result,
    int rows, int cols);

extern void add_scaled_identity(
    double* A, double lambda, int n);

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#endif

struct BenchmarkResults {
    double execution_time_ms;
    size_t peak_memory_kb;
    int iterations;
    double final_error;
    std::vector<double> final_params;
    std::vector<double> initial_params;
};

size_t get_peak_memory_kb() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS_EX pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc))) {
        return pmc.PeakWorkingSetSize / 1024;
    }
    return 0;
#else
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        return usage.ru_maxrss;
    }
    return 0;
#endif
}

// Simple CPU-based LDLT solve (for now)
std::vector<double> ldlt_solve_cpu(const std::vector<std::vector<double>>& A, const std::vector<double>& b) {
    int n = A.size();
    std::vector<std::vector<double>> L(n, std::vector<double>(n, 0.0));
    std::vector<double> D(n, 0.0);
    
    for (int i = 0; i < n; i++) L[i][i] = 1.0;
    
    for (int j = 0; j < n; j++) {
        double sum = 0.0;
        for (int k = 0; k < j; k++) {
            sum += L[j][k] * L[j][k] * D[k];
        }
        D[j] = A[j][j] - sum;
        
        for (int i = j + 1; i < n; i++) {
            sum = 0.0;
            for (int k = 0; k < j; k++) {
                sum += L[i][k] * L[j][k] * D[k];
            }
            L[i][j] = (A[i][j] - sum) / D[j];
        }
    }
    
    // Forward substitution
    std::vector<double> y(n, 0.0);
    for (int i = 0; i < n; i++) {
        double sum = 0.0;
        for (int k = 0; k < i; k++) {
            sum += L[i][k] * y[k];
        }
        y[i] = (b[i] - sum) / L[i][i];
    }
    
    // Diagonal solve
    std::vector<double> z(n);
    for (int i = 0; i < n; i++) {
        z[i] = y[i] / D[i];
    }
    
    // Backward substitution
    std::vector<double> x(n, 0.0);
    for (int i = n - 1; i >= 0; i--) {
        double sum = 0.0;
        for (int k = i + 1; k < n; k++) {
            sum += L[k][i] * x[k];
        }
        x[i] = z[i] - sum;
    }
    
    return x;
}

void LevenbergMarquardt(const std::vector<double>& data_x,
                        const std::vector<double>& data_y,
                        const std::vector<double>& initial_params,
                        int max_iterations,
                        double epsilon,
                        double lambda,
                        std::vector<double>& final_params,
                        BenchmarkResults& results) {
    
    int num_points = data_x.size();
    int num_params = initial_params.size();
    std::vector<double> params = initial_params;
    double error = std::numeric_limits<double>::max();
    int iteration = 0;
    
    // Allocate device memory
    double *d_data_x, *d_data_y, *d_params, *d_residuals, *d_jacobian;
    double *d_AtA, *d_Atb;
    
    cudaMalloc(&d_data_x, num_points * sizeof(double));
    cudaMalloc(&d_data_y, num_points * sizeof(double));
    cudaMalloc(&d_params, num_params * sizeof(double));
    cudaMalloc(&d_residuals, num_points * sizeof(double));
    cudaMalloc(&d_jacobian, num_points * num_params * sizeof(double));
    cudaMalloc(&d_AtA, num_params * num_params * sizeof(double));
    cudaMalloc(&d_Atb, num_params * sizeof(double));
    
    // Copy initial data to device
    cudaMemcpy(d_data_x, data_x.data(), num_points * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_data_y, data_y.data(), num_points * sizeof(double), cudaMemcpyHostToDevice);
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    while (error > epsilon && iteration < max_iterations) {
        // Copy current parameters to device
        cudaMemcpy(d_params, params.data(), num_params * sizeof(double), cudaMemcpyHostToDevice);
        
        // Compute residuals and Jacobian
        int threadsPerBlock = 256;
        int blocksPerGrid = (num_points + threadsPerBlock - 1) / threadsPerBlock;
        compute_residuals_and_jacobian<<<blocksPerGrid, threadsPerBlock>>>(
            d_data_x, d_data_y, d_params, d_residuals, d_jacobian, num_points);
        cudaDeviceSynchronize();
        
        // Compute J^T * J
        dim3 blockSize(16, 16);
        dim3 gridSize((num_params + blockSize.x - 1) / blockSize.x,
                      (num_params + blockSize.y - 1) / blockSize.y);
        matrix_multiply_AtA<<<gridSize, blockSize>>>(
            d_jacobian, d_AtA, num_points, num_params);
        cudaDeviceSynchronize();
        
        // Add lambda * I
        add_scaled_identity<<<1, num_params>>>(d_AtA, lambda, num_params);
        cudaDeviceSynchronize();
        
        // Compute J^T * r
        matrix_vector_multiply_At<<<1, num_params>>>(
            d_jacobian, d_residuals, d_Atb, num_points, num_params);
        cudaDeviceSynchronize();
        
        // Copy back to host for solving
        std::vector<double> AtA(num_params * num_params);
        std::vector<double> Atb(num_params);
        std::vector<double> residuals(num_points);
        
        cudaMemcpy(AtA.data(), d_AtA, num_params * num_params * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(Atb.data(), d_Atb, num_params * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(residuals.data(), d_residuals, num_points * sizeof(double), cudaMemcpyDeviceToHost);
        
        // Convert to 2D for LDLT solve
        std::vector<std::vector<double>> A_mat(num_params, std::vector<double>(num_params));
        for (int i = 0; i < num_params; i++) {
            for (int j = 0; j < num_params; j++) {
                A_mat[i][j] = AtA[i * num_params + j];
            }
        }
        
        // Solve for delta
        std::vector<double> delta = ldlt_solve_cpu(A_mat, Atb);
        
        // Update parameters
        std::vector<double> new_params(num_params);
        for (int i = 0; i < num_params; i++) {
            new_params[i] = params[i] + delta[i];
        }
        
        // Compute new error
        double new_error = 0.0;
        for (int i = 0; i < num_points; i++) {
            new_error += residuals[i] * residuals[i];
        }
        
        if (new_error < error) {
            lambda /= 10.0;
            error = new_error;
            params = new_params;
        } else {
            lambda *= 10.0;
        }
        
        iteration++;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    final_params = params;
    results.execution_time_ms = duration.count() / 1000.0;
    results.iterations = iteration;
    results.final_error = error;
    results.peak_memory_kb = get_peak_memory_kb();
    
    // Free device memory
    cudaFree(d_data_x);
    cudaFree(d_data_y);
    cudaFree(d_params);
    cudaFree(d_residuals);
    cudaFree(d_jacobian);
    cudaFree(d_AtA);
    cudaFree(d_Atb);
}

void save_results_json(const BenchmarkResults& results, const std::string& filename) {
    std::ofstream file(filename);
    file << std::fixed << std::setprecision(10);
    file << "{\n";
    file << "  \"execution_time_ms\": " << results.execution_time_ms << ",\n";
    file << "  \"peak_memory_kb\": " << results.peak_memory_kb << ",\n";
    file << "  \"iterations\": " << results.iterations << ",\n";
    file << "  \"final_error\": " << results.final_error << ",\n";
    file << "  \"initial_params\": [" << results.initial_params[0] << ", " << results.initial_params[1] << "],\n";
    file << "  \"final_params\": [" << results.final_params[0] << ", " << results.final_params[1] << "]\n";
    file << "}\n";
    file.close();
}

int main() {
    const int num_data_points = 100;
    const int max_iterations = 1000;
    double lambda = 0.01;
    const double epsilon = 0.000001;
    
    std::vector<double> initial_params = {0.5, 1.0};
    std::vector<double> final_params(2);
    
    // Generate data
    std::default_random_engine gen(42);
    std::normal_distribution<double> dist(0.0, 0.1);
    std::vector<double> data_x(num_data_points);
    std::vector<double> data_y(num_data_points);
    
    for (int i = 0; i < num_data_points; ++i) {
        data_x[i] = i / static_cast<double>(num_data_points - 1);
        data_y[i] = std::exp(-initial_params[0] * data_x[i]) * 
                    std::cos(initial_params[1] * data_x[i]) + dist(gen);
    }
    
    BenchmarkResults results;
    results.initial_params = initial_params;
    
    // Run Levenberg-Marquardt
    LevenbergMarquardt(data_x, data_y, initial_params, max_iterations, epsilon, lambda, final_params, results);
    
    results.final_params = final_params;
    
    // Print results
    std::cout << "CUDA (Custom Kernels) Results:\n";
    std::cout << "Initial parameters: [" << initial_params[0] << ", " << initial_params[1] << "]\n";
    std::cout << "Final parameters: [" << final_params[0] << ", " << final_params[1] << "]\n";
    std::cout << "Iterations: " << results.iterations << std::endl;
    std::cout << "Execution time: " << results.execution_time_ms << " ms\n";
    std::cout << "Peak memory: " << results.peak_memory_kb << " KB\n";
    std::cout << "Final error: " << results.final_error << std::endl;
    
    // Save to JSON
    save_results_json(results, "results_cuda_custom.json");
    
    return 0;
}

