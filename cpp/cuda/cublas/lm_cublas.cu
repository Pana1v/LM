#include <iostream>
#include <random>
#include <cmath>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

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

__global__ void compute_residuals_and_jacobian(
    const double* data_x, const double* data_y,
    const double* params, double* residuals, double* jacobian,
    int num_points) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_points) {
        double x = data_x[idx];
        double y = data_y[idx];
        
        double exp_term = exp(-params[0] * x);
        double cos_term = cos(params[1] * x);
        double sin_term = sin(params[1] * x);
        double f = exp_term * cos_term;
        
        residuals[idx] = y - f;
        
        jacobian[idx * 2 + 0] = x * f;
        jacobian[idx * 2 + 1] = -x * exp_term * sin_term;
    }
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
    
    // Initialize cuBLAS and cuSOLVER
    cublasHandle_t cublas_handle;
    cusolverDnHandle_t cusolver_handle;
    cublasCreate(&cublas_handle);
    cusolverDnCreate(&cusolver_handle);
    
    // Allocate device memory
    double *d_data_x, *d_data_y, *d_params, *d_residuals, *d_jacobian;
    double *d_AtA, *d_Atb, *d_work;
    int *d_info;
    
    cudaMalloc(&d_data_x, num_points * sizeof(double));
    cudaMalloc(&d_data_y, num_points * sizeof(double));
    cudaMalloc(&d_params, num_params * sizeof(double));
    cudaMalloc(&d_residuals, num_points * sizeof(double));
    cudaMalloc(&d_jacobian, num_points * num_params * sizeof(double));
    cudaMalloc(&d_AtA, num_params * num_params * sizeof(double));
    cudaMalloc(&d_Atb, num_params * sizeof(double));
    cudaMalloc(&d_info, sizeof(int));
    
    // Copy initial data to device
    cudaMemcpy(d_data_x, data_x.data(), num_points * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_data_y, data_y.data(), num_points * sizeof(double), cudaMemcpyHostToDevice);
    
    // Query workspace size for cuSOLVER
    int lwork = 0;
    cusolverDnDpotrf_bufferSize(cusolver_handle, CUBLAS_FILL_MODE_LOWER, 
                                 num_params, d_AtA, num_params, &lwork);
    cudaMalloc(&d_work, lwork * sizeof(double));
    
    const double alpha = 1.0;
    const double beta = 0.0;
    
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
        
        // Compute J^T * J using cuBLAS (gemm)
        cublasDgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                    num_params, num_params, num_points,
                    &alpha, d_jacobian, num_points,
                    d_jacobian, num_points,
                    &beta, d_AtA, num_params);
        
        // Add lambda * I (add lambda to diagonal elements)
        std::vector<double> identity_diag(num_params, lambda);
        double *d_identity_diag;
        cudaMalloc(&d_identity_diag, num_params * sizeof(double));
        cudaMemcpy(d_identity_diag, identity_diag.data(), num_params * sizeof(double), cudaMemcpyHostToDevice);
        // Add lambda to diagonal using a simple kernel or manual addition
        // For simplicity, we'll do this on host and copy back
        std::vector<double> AtA_host(num_params * num_params);
        cudaMemcpy(AtA_host.data(), d_AtA, num_params * num_params * sizeof(double), cudaMemcpyDeviceToHost);
        for (int i = 0; i < num_params; i++) {
            AtA_host[i * num_params + i] += lambda;
        }
        cudaMemcpy(d_AtA, AtA_host.data(), num_params * num_params * sizeof(double), cudaMemcpyHostToDevice);
        cudaFree(d_identity_diag);
        
        // Compute J^T * r using cuBLAS (gemv)
        cublasDgemv(cublas_handle, CUBLAS_OP_T,
                    num_points, num_params,
                    &alpha, d_jacobian, num_points,
                    d_residuals, 1,
                    &beta, d_Atb, 1);
        
        // Solve using Cholesky decomposition (cuSOLVER)
        cusolverDnDpotrf(cusolver_handle, CUBLAS_FILL_MODE_LOWER,
                         num_params, d_AtA, num_params, d_work, lwork, d_info);
        cusolverDnDpotrs(cusolver_handle, CUBLAS_FILL_MODE_LOWER,
                         num_params, 1, d_AtA, num_params, d_Atb, num_params, d_info);
        
        // Copy delta back to host
        std::vector<double> delta(num_params);
        cudaMemcpy(delta.data(), d_Atb, num_params * sizeof(double), cudaMemcpyDeviceToHost);
        
        // Update parameters
        std::vector<double> new_params(num_params);
        for (int i = 0; i < num_params; i++) {
            new_params[i] = params[i] + delta[i];
        }
        
        // Compute new error (copy residuals to host)
        std::vector<double> residuals(num_points);
        cudaMemcpy(residuals.data(), d_residuals, num_points * sizeof(double), cudaMemcpyDeviceToHost);
        
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
    
    // Cleanup
    cudaFree(d_data_x);
    cudaFree(d_data_y);
    cudaFree(d_params);
    cudaFree(d_residuals);
    cudaFree(d_jacobian);
    cudaFree(d_AtA);
    cudaFree(d_Atb);
    cudaFree(d_work);
    cudaFree(d_info);
    cublasDestroy(cublas_handle);
    cusolverDnDestroy(cusolver_handle);
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
    std::cout << "CUDA (cuBLAS/cuSOLVER) Results:\n";
    std::cout << "Initial parameters: [" << initial_params[0] << ", " << initial_params[1] << "]\n";
    std::cout << "Final parameters: [" << final_params[0] << ", " << final_params[1] << "]\n";
    std::cout << "Iterations: " << results.iterations << std::endl;
    std::cout << "Execution time: " << results.execution_time_ms << " ms\n";
    std::cout << "Peak memory: " << results.peak_memory_kb << " KB\n";
    std::cout << "Final error: " << results.final_error << std::endl;
    
    // Save to JSON
    save_results_json(results, "results_cuda_cublas.json");
    
    return 0;
}

