#pragma once
#include <vector>
#include <memory>

#include <Eigen/dense>

#include "Constraint.h"
#include "Objective.h"

namespace HKOpt {
	enum class OptimizerStatus { NOT_STARTED = -1, SUCCEEDED = 0, FAILED = 1};

	template<typename _Derived, typename _Scalar>
	class OptimizerBase {
	public:
		using Vector = Eigen::VectorX<_Scalar>;
		using Matrix = Eigen::MatrixX<_Scalar>;
		
		using Objective = Objective<_Scalar>;
		using ObjectivePtr = std::shared_ptr<Objective>;
		using Constraints = Constraints<_Scalar>;

		OptimizerBase(const ObjectivePtr& objective, const Constraints& constraints) :
			objective_(objective), constraints_(constraints) {}

		inline const Vector& x() { return x_;  }

		OptimizerStatus optimize(const Vector& x0) { 
			x_ = x0;
			return static_cast<_Derived*>(this)->optimizeImpl();
		}
		
	protected:
		Vector x_;

		ObjectivePtr objective_;
		Constraints constraints_;

	};

	template<typename _Scalar>
	class GradientDescent : public OptimizerBase<GradientDescent<_Scalar>, _Scalar> {
		friend class OptimizerBase<GradientDescent<_Scalar>, _Scalar>;

	public:
		using OptimizerBase = OptimizerBase<GradientDescent<_Scalar>, _Scalar>;

		using Vector = Eigen::VectorX<_Scalar>;
		using Matrix = Eigen::MatrixX<_Scalar>;

		using Objective = Objective<_Scalar>;
		using ObjectivePtr = std::shared_ptr<Objective>;

		using Constraints = Constraints<_Scalar>;

		GradientDescent(
			const ObjectivePtr& objective,
			const Constraints& constraints,
			const _Scalar& learning_rate = 0.01,
			const int& max_iteration = 1000,
			const _Scalar& tolerance = 1e-4,
			const _Scalar& learning_rate_shrink_factor = 2,
			const _Scalar& min_learning_rate = 1e-4,
			const size_t& cycling_window = 2,
			const _Scalar& cycling_tolerance = 1e-6) :
			OptimizerBase(objective, constraints), learning_rate_(learning_rate), max_iteration_(max_iteration), 
			tolerance_(tolerance), learning_rate_shrink_factor_(learning_rate_shrink_factor), 
			min_learning_rate_(min_learning_rate), cycling_window_(cycling_window), cycling_tolerance_(cycling_tolerance) { }

	protected:
		OptimizerStatus optimizeImpl() {
			for (auto iteration = 0; iteration < max_iteration_; iteration++) {
				gradient_ = this->objective_->gradient(this->x_);
				std::cout << gradient_.transpose() << std::endl;

				if (gradient_.norm() < tolerance_) {
					return OptimizerStatus::SUCCEEDED;
				}

				Vector step = gradient_ / gradient_.norm() * learning_rate_;

				this->x_ -= step;

				this->constraints_.projectOntoConstraints(this->x_);

				bool is_cycling = false;
				for (const auto& x_prev : history_) {
					if ((this->x_ - x_prev).norm() < cycling_tolerance_) {
						is_cycling = true;
						break;
					}
				}
				if (is_cycling) {
					learning_rate_ /= learning_rate_shrink_factor_;
					if (learning_rate_ < min_learning_rate_) {
						return OptimizerStatus::SUCCEEDED;
					}
				}


				if (history_.size() > cycling_window_) {
					history_.erase(history_.begin());
				}
				history_.push_back(this->x_);
			}
			return OptimizerStatus::FAILED;
		}
	private:
		Vector gradient_;
		_Scalar learning_rate_;
		int max_iteration_;
		_Scalar tolerance_;

		_Scalar learning_rate_shrink_factor_;
		_Scalar min_learning_rate_;

		size_t cycling_window_;
		_Scalar cycling_tolerance_;
		std::vector<Vector> history_;
	};
}