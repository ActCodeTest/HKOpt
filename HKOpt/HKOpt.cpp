#include <iostream>
#include <Eigen/dense>
#include "Optimizer.h"
#include "Objective.h"
#include "Constraint.h"

using namespace HKOpt;
using namespace Eigen;

int main() {
    // Define a simple objective function f(x) = x^2
    auto objective_function = [](const VectorXd& x) -> double {
        return x.squaredNorm(); // f(x) = x^2
        };

    // Define the gradient of the objective function f'(x) = 2x
    auto gradient_function = [](const VectorXd& x) -> VectorXd {
        return 2 * x; // f'(x) = 2x
        };

    // Define the Hessian of the objective function f''(x) = 2
    auto hessian_function = [](const VectorXd& x) -> MatrixXd {
        return 2 * MatrixXd::Identity(x.size(), x.size()); // f''(x) = 2
        };

    // Create the objective
    std::shared_ptr<Objective<double>> objective = std::make_shared<Objective<double>>(
        objective_function, gradient_function, hessian_function);

    // Define no constraints for simplicity in this example
    Eigen::VectorXd constraint_coeff(2);
    constraint_coeff <<  1, 1;
    auto constraint = std::make_shared<Constraint<double>>(constraint_coeff, 10, ConstraintType::GEQ);

    Constraints<double> constraints({ constraint });

    // Instantiate the optimizer (GradientDescent)
    GradientDescent<double> optimizer(objective, constraints, 0.1, 10000);

    // Initial guess (starting point)
    VectorXd x0(2);
    x0 << 100.0, 100.0; // Start at x = 10

    // Optimize
    OptimizerStatus status = optimizer.optimize(x0);

    // Print results
    std::cout << "Optimization Status: ";
    switch (status) {
    case OptimizerStatus::NOT_STARTED: std::cout << "NOT_STARTED"; break;
    case OptimizerStatus::SUCCEEDED: std::cout << "SUCCEEDED"; break;
    case OptimizerStatus::FAILED: std::cout << "FAILED"; break;
    }
    std::cout << std::endl;

    //std::cout << "Iteration: " << optimizer.iteration() << std::endl;

    std::cout << "Final x: " << optimizer.x().transpose() << std::endl;
    
    return 0;
}
