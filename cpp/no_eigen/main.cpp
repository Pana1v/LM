#include <iostream>
#include <random>
#include <cmath>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <limits>
#include <vector>
#include <sstream>
#include <string>
#include "matrix_ops.hpp"

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

void LevenbergMarquardt(const Matrix& data,
                        const Vector& initial_params,
                        const int max_iterations,
                        const double epsilon,
                        double lambda,
                        Vector& final_params,
                        BenchmarkResults& results) {

    const int num_params = initial_params.size();
    Vector params = initial_params;
    double error = (std::numeric_limits<double>::max)();
    int iteration = 0;

    auto start_time = std::chrono::high_resolution_clock::now();

    while (error > epsilon && iteration < max_iterations) {
        Matrix J(data.rows(), num_params);
        Vector r(data.rows());

        for (int i = 0; i < data.rows(); i++) {
            const double x = data(i, 0);
            const double y = data(i, 1);
            const double f = std::exp(-params(0) * x) * std::cos(params(1) * x);
            
            r(i) = y - f;

            J(i, 0) = x * f;
            J(i, 1) = -x * std::exp(-params(0) * x) * std::sin(params(1) * x);
        }

        Matrix Jt = J.transpose();
        Matrix A = matmul(Jt, J);
        A = add_scaled_identity(A, lambda);
        Vector b = matvec(Jt, r);

        Vector delta = ldlt_solve(A, b);
        Vector new_params = vec_add(params, delta);

        // Compute new error with new_params
        Vector new_r(data.rows());
        for (int i = 0; i < data.rows(); i++) {
            const double x = data(i, 0);
            const double y = data(i, 1);
            const double new_f = std::exp(-new_params(0) * x) * std::cos(new_params(1) * x);
            new_r(i) = y - new_f;
        }
        const double new_error = sum_squares(new_r);

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
    
    Vector initial_params(2);
    initial_params(0) = 0.5;
    initial_params(1) = 1.0;
    Vector final_params(2);

    // Load data from CSV
    Matrix data(num_data_points, 2);
    std::ifstream file("data/test_data.csv");
    if (file.is_open()) {
        std::string line;
        std::getline(file, line); // Skip header
        int i = 0;
        while (std::getline(file, line) && i < num_data_points) {
            std::stringstream ss(line);
            std::string token;
            std::getline(ss, token, ',');
            data(i, 0) = std::stod(token);
            std::getline(ss, token, ',');
            data(i, 1) = std::stod(token);
            i++;
        }
        file.close();
    } else {
        // Fallback: Generate data if CSV not found
        std::default_random_engine gen(42);
        std::normal_distribution<double> dist(0.0, 0.1);
        for (int i = 0; i < num_data_points; ++i) {
            data(i, 0) = i / static_cast<double>(num_data_points - 1);
            data(i, 1) = std::exp(-initial_params(0) * data(i, 0)) * 
                         std::cos(initial_params(1) * data(i, 0)) + dist(gen);
        }
    }

    BenchmarkResults results;
    results.initial_params = {initial_params(0), initial_params(1)};

    // Run Levenberg-Marquardt
    LevenbergMarquardt(data, initial_params, max_iterations, epsilon, lambda, final_params, results);

    results.final_params = {final_params(0), final_params(1)};

    // Print results
    std::cout << "C++ (No Eigen) Results:\n";
    std::cout << "Initial parameters: [" << initial_params(0) << ", " << initial_params(1) << "]\n";
    std::cout << "Final parameters: [" << final_params(0) << ", " << final_params(1) << "]\n";
    std::cout << "Iterations: " << results.iterations << std::endl;
    std::cout << "Execution time: " << results.execution_time_ms << " ms\n";
    std::cout << "Peak memory: " << results.peak_memory_kb << " KB\n";
    std::cout << "Final error: " << results.final_error << std::endl;

    // Save to JSON
    save_results_json(results, "results_cpp_no_eigen.json");

    return 0;
}

