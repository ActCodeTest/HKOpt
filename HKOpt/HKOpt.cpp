#include <iostream>
#include <Eigen/dense>
#include "TrustRegion.h"
#include "DenseNonLinearProblem.h"  // or whatever your problem header is

int main() {
    // Define a simple quadratic problem:
    // f(x) = 0.5 * x^T A x - b^T x
    // A is a positive-definite matrix, and b is a vector.

    // Problem data
    Eigen::MatrixXd A(2, 2);
    A << 2, 0,
        0, 2; // A positive definite matrix

    Eigen::VectorXd b(2);
    b << 1, 1;  // Vector b

    Eigen::VectorXd x0(2);
    x0 << 0, 0;  // Initial guess for x

    // Create the problem
    auto x_ptr = std::make_shared<Eigen::VectorXd>(x0);
    auto A_ptr = std::make_shared<Eigen::MatrixXd>(A);
    auto b_ptr = std::make_shared<Eigen::VectorXd>(b);

    // Define the linear objective function: f(x) = 0.5 * x^T A x - b^T x
    auto f = [&A_ptr, &b_ptr](const Eigen::VectorXd& x) -> double {
        return 0.5 * x(0);
        };

    // Create the dense linear problem
    NLP::DenseNonLinearProblem problem(f, x_ptr, A_ptr, b_ptr);


    //std::cout << *problem.hessian();
    // Trust Region Solver setup
    NLP::TrustRegion<NLP::DenseNonLinearProblem> solver(10000, 1.0);  // Max 100 iterations and initial delta = 1.0

    // Solve the problem
    solver.solve(problem);

    // Output the results
    //std::cout << "Solution found after " << solver.iteration_ << " iterations." << std::endl;
    std::cout << "Optimal x: " << (*x_ptr).transpose() << std::endl;

    return 0;
}