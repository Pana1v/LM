#include <cuda_runtime.h>
#include <cmath>

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

__global__ void matrix_multiply_AtA(
    const double* A, double* AtA, int rows, int cols) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < cols && j < cols) {
        double sum = 0.0;
        for (int k = 0; k < rows; k++) {
            sum += A[k * cols + i] * A[k * cols + j];
        }
        AtA[i * cols + j] = sum;
    }
}

__global__ void matrix_vector_multiply_At(
    const double* A, const double* v, double* result,
    int rows, int cols) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < cols) {
        double sum = 0.0;
        for (int k = 0; k < rows; k++) {
            sum += A[k * cols + i] * v[k];
        }
        result[i] = sum;
    }
}

__global__ void add_scaled_identity(
    double* A, double lambda, int n) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        A[idx * n + idx] += lambda;
    }
}

