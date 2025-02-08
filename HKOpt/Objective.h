#pragma once
#include <Eigen/dense>
#include <functional>

namespace HKOpt {

	template<typename _Derived, typename _Scalar>
	class ObjectiveBase {
	public:
		using Vector = Eigen::VectorX<_Scalar>;
		using Matrix = Eigen::MatrixX<_Scalar>;

		inline _Scalar evaluate(const Vector& x) const { return static_cast<const _Derived*>(this)->evaluateImpl(x); }
		inline Vector gradient(const Vector& x) const { return static_cast<const _Derived*>(this)->gradientImpl(x); }
		inline Matrix hessian(const Vector& x) const { return static_cast<const _Derived*>(this)->hessianImpl(x); }
	};

	template<typename _Scalar> 
	class Objective : public ObjectiveBase<Objective<_Scalar>, _Scalar> {
		friend class ObjectiveBase<Objective<_Scalar>, _Scalar>;
	public:
		using ObjectiveBase = ObjectiveBase<Objective<_Scalar>, _Scalar>;
		using Vector = ObjectiveBase::Vector;
		using Matrix = ObjectiveBase::Matrix;
		using Function = std::function<_Scalar(const Vector&)>;
		using GradientFunction = std::function<Vector(const Vector&)>;
		using HessianFunction = std::function<Matrix(const Vector&)>;

		Objective(const Function& function, const GradientFunction& gradient_function = nullptr, const HessianFunction& hessian_function = nullptr) :
			function_(function), gradient_function_(gradient_function), hessian_function_(hessian_function) { }

	protected:
		_Scalar evaluateImpl(const Vector& x) const { 
			return function_(x); 
		}

		Vector gradientImpl(const Vector& x) const {
			if (!gradient_function_) {
				return Vector::Zero(x.size());
			}
			return gradient_function_(x);
		}

		Matrix hessianImpl(const Vector& x) const {
			if (!hessian_function_) {
				return Matrix::Zero(x.size(), x.size());
			}
			return hessian_function_(x);
		}

	private:
		Function function_;
		GradientFunction gradient_function_;
		HessianFunction hessian_function_;
	};
}