#include <iostream>
#include <random>
#include "lm/lm.hpp"

#include <cmath>
#include "eigen3/Eigen3/Dense"

void LevenbergMarquardt(const Eigen::MatrixXd& data,
const Eigen::VectorXd& initial_params, const int max_iterations, const double epsilon, double lambda, Eigen::VectorXd& final_params) {

    std::cout << "LevenbergMarquardt" <