#pragma once
#include <Eigen/dense>
#include "DenseSolverBase.h"
namespace NLP {
	template<typename _DenseProblem>
	class TrustRegion : public Core::DenseSolverBase<TrustRegion<_DenseProblem>, _DenseProblem> {
		friend class Core::DenseSolverBase<TrustRegion<_DenseProblem>, _DenseProblem>;
	public:
		typedef Core::DenseSolverBase<TrustRegion<_DenseProblem>, _DenseProblem> DenseSolverBase;
		typedef typename DenseSolverBase::Vector Vector;
		typedef typename DenseSolverBase::Matrix Matrix;
		typedef typename DenseSolverBase::VectorPtr VectorPtr;
		typedef typename DenseSolverBase::MatrixPtr MatrixPtr;

		TrustRegion(
			const int& max_iteration,
			const double& delta = 1) :
			iteration_(0), max_iteration_(max_iteration), delta_(delta), max_delta_(10 * delta),
			rho_upper_(0.75), rho_lower_(0.25), gamma_expand_(2), gamma_shrink_(0.5) {}

	protected:
		VectorPtr solveImpl(_DenseProblem& problem) {

			VectorPtr x = problem.x();
			VectorPtr gradient = problem.gradient();
			MatrixPtr hessian = problem.hessian();

			for (iteration_ = 0; iteration_ < max_iteration_; iteration_++) {
				problem.update();

				Vector step = dogLegStep(*gradient, *hessian, delta_);
				double predicted_reduction = predictedReduction(step, *gradient, *hessian);
				double actual_reduction = actualReduction(problem, step);

				double rho = actual_reduction / predicted_reduction;


				// Adjust the radius
				if (rho < rho_lower_) {
					delta_ *= gamma_shrink_;  // Shrink trust region
				}
				else if (rho > rho_upper_) {
					delta_ = std::min(gamma_expand_ * delta_, max_delta_);  // Expand trust region but limit max size
				}

				if (rho > 0) {
					(*problem.x()) += enforceFeasibility(problem, step);  // Accept the step and update x
					for (size_t i = 0; i < problem.x()->size(); i++) {
						if ((*problem.x())(i) < 0) {
							(*problem.x())(i) = 0;
						}
					}
				}

				if (step.norm() < 1e-6 || gradient->norm() < 1e-6) {
					break;
				}
			}

			return problem.x();
		}

	private:
		int iteration_;
		int max_iteration_;

		double delta_;
		double max_delta_;

		double rho_upper_;
		double rho_lower_;

		double gamma_shrink_;
		double gamma_expand_;

		Vector dogLegStep(const Vector& gradient, const Matrix& hessian, const double& delta) const {
			Vector cauchy_step = cauchyStep(gradient, hessian, delta);
			if (cauchy_step.norm() >= delta) {
				return cauchy_step;
			}

			Vector newton_step = newtonStep(gradient, hessian);
			if (newton_step.norm() <= delta) {
				return newton_step;
			}

			Vector diff = newton_step - cauchy_step;
			double a = diff.squaredNorm();
			double b = 2.0 * cauchy_step.dot(diff);
			double c = cauchy_step.squaredNorm() - delta_ * delta_;

			double discriminant = b * b - 4 * a * c;
			if (discriminant < 0) return cauchy_step;

			double tau = (-b + std::sqrt(discriminant)) / (2.0 * a);
			return cauchy_step + tau * diff;

		}

		Vector cauchyStep(const Vector& gradient, const Matrix& hessian, const double& delta) const {
			double gHg = gradient.dot(hessian * gradient);
			if (gHg <= 0) return -delta * gradient.normalized();
			double alpha = gradient.dot(gradient) / gHg;
			Vector step = -alpha * gradient;
			return (step.norm() >= delta) ? (delta * step.normalized()) : step;
		}

		Vector newtonStep(const Vector& gradient, const Matrix& hessian) const {
			return hessian.ldlt().solve(-gradient);
		}

		double predictedReduction(const Vector& step, const Vector& gradient, const Matrix& hessian) const {
			return gradient.dot(step) + 0.5 * step.dot(hessian * step);
		}

		double actualReduction(_DenseProblem& problem, const Vector& step) const {
			double f = problem.evaluate();
			Vector x = *problem.x();
			*problem.x() += step;
			double f1 = problem.evaluate();

			*problem.x() = x;

			return f1 - f;
		}

		Vector& enforceFeasibility(_DenseProblem& problem, Vector& step) {
			Matrix A = *problem.A();
			Vector b = *problem.b();
			Vector x = *problem.x();

			Vector Ax = A * x;
			Vector r = b - Ax;
			Vector As = A * step;

			double alpha = 1.0;
			for (size_t i = 0; i < As.size(); i++) {
				if (As(i) > r(i)) {
					alpha = std::min(alpha, r(i) / As(i));
				}
			}

			step *= alpha;
			return step;
		}
	};

}