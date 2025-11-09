#ifndef MATRIX_OPS_HPP
#define MATRIX_OPS_HPP

#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>

class Matrix {
public:
    Matrix(size_t rows, size_t cols) : rows_(rows), cols_(cols), data_(rows * cols, 0.0) {}
    
    double& operator()(size_t i, size_t j) { return data_[i * cols_ + j]; }
    const double& operator()(size_t i, size_t j) const { return data_[i * cols_ + j]; }
    
    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    
    void set_identity() {
        std::fill(data_.begin(), data_.end(), 0.0);
        for (size_t i = 0; i < std::min(rows_, cols_); ++i) {
            (*this)(i, i) = 1.0;
        }
    }
    
    Matrix transpose() const {
        Matrix result(cols_, rows_);
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < cols_; ++j) {
                result(j, i) = (*this)(i, j);
            }
        }
        return result;
    }
    
    std::vector<double> get_data() const { return data_; }
    void set_data(const std::vector<double>& data) { data_ = data; }

private:
    size_t rows_, cols_;
    std::vector<double> data_;
};

class Vector {
public:
    Vector(size_t size) : data_(size, 0.0) {}
    
    double& operator()(size_t i) { return data_[i]; }
    const double& operator()(size_t i) const { return data_[i]; }
    
    size_t size() const { return data_.size(); }
    
    std::vector<double> get_data() const { return data_; }
    void set_data(const std::vector<double>& data) { data_ = data; }

private:
    std::vector<double> data_;
};

// Matrix multiplication: C = A * B
Matrix matmul(const Matrix& A, const Matrix& B) {
    if (A.cols() != B.rows()) {
        throw std::runtime_error("Matrix dimensions incompatible for multiplication");
    }
    
    Matrix C(A.rows(), B.cols());
    for (size_t i = 0; i < A.rows(); ++i) {
        for (size_t j = 0; j < B.cols(); ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < A.cols(); ++k) {
                sum += A(i, k) * B(k, j);
            }
            C(i, j) = sum;
        }
    }
    return C;
}

// Matrix-vector multiplication: y = A * x
Vector matvec(const Matrix& A, const Vector& x) {
    if (A.cols() != x.size()) {
        throw std::runtime_error("Matrix and vector dimensions incompatible");
    }
    
    Vector y(A.rows());
    for (size_t i = 0; i < A.rows(); ++i) {
        double sum = 0.0;
        for (size_t j = 0; j < A.cols(); ++j) {
            sum += A(i, j) * x(j);
        }
        y(i) = sum;
    }
    return y;
}

// Vector dot product
double dot(const Vector& a, const Vector& b) {
    if (a.size() != b.size()) {
        throw std::runtime_error("Vector dimensions incompatible for dot product");
    }
    
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        sum += a(i) * b(i);
    }
    return sum;
}

// LDLT decomposition and solve
// Returns solution x to Ax = b
Vector ldlt_solve(const Matrix& A, const Vector& b) {
    size_t n = A.rows();
    if (n != A.cols() || n != b.size()) {
        throw std::runtime_error("Matrix must be square and match vector size");
    }
    
    // LDLT decomposition: A = L * D * L^T
    Matrix L(n, n);
    Vector D(n);
    L.set_identity();
    
    for (size_t j = 0; j < n; ++j) {
        double sum = 0.0;
        for (size_t k = 0; k < j; ++k) {
            sum += L(j, k) * L(j, k) * D(k);
        }
        D(j) = A(j, j) - sum;
        
        if (std::abs(D(j)) < 1e-10) {
            throw std::runtime_error("Matrix is singular or near-singular");
        }
        
        for (size_t i = j + 1; i < n; ++i) {
            sum = 0.0;
            for (size_t k = 0; k < j; ++k) {
                sum += L(i, k) * L(j, k) * D(k);
            }
            L(i, j) = (A(i, j) - sum) / D(j);
        }
    }
    
    // Forward substitution: L * y = b
    Vector y(n);
    for (size_t i = 0; i < n; ++i) {
        double sum = 0.0;
        for (size_t k = 0; k < i; ++k) {
            sum += L(i, k) * y(k);
        }
        y(i) = (b(i) - sum) / L(i, i);
    }
    
    // Diagonal solve: D * z = y
    Vector z(n);
    for (size_t i = 0; i < n; ++i) {
        z(i) = y(i) / D(i);
    }
    
    // Backward substitution: L^T * x = z
    Vector x(n);
    for (int i = n - 1; i >= 0; --i) {
        double sum = 0.0;
        for (size_t k = i + 1; k < n; ++k) {
            sum += L(k, i) * x(k);
        }
        x(i) = z(i) - sum;
    }
    
    return x;
}

// Add scaled identity: A + lambda * I
Matrix add_scaled_identity(const Matrix& A, double lambda) {
    Matrix result = A;
    for (size_t i = 0; i < std::min(result.rows(), result.cols()); ++i) {
        result(i, i) += lambda;
    }
    return result;
}

// Vector addition: c = a + b
Vector vec_add(const Vector& a, const Vector& b) {
    if (a.size() != b.size()) {
        throw std::runtime_error("Vector dimensions incompatible for addition");
    }
    
    Vector c(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        c(i) = a(i) + b(i);
    }
    return c;
}

// Vector subtraction: c = a - b
Vector vec_sub(const Vector& a, const Vector& b) {
    if (a.size() != b.size()) {
        throw std::runtime_error("Vector dimensions incompatible for subtraction");
    }
    
    Vector c(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        c(i) = a(i) - b(i);
    }
    return c;
}

// Sum of squares of vector elements
double sum_squares(const Vector& v) {
    double sum = 0.0;
    for (size_t i = 0; i < v.size(); ++i) {
        sum += v(i) * v(i);
    }
    return sum;
}

#endif // MATRIX_OPS_HPP

