#pragma once
#include <Eigen/dense>
namespace Core {
	template<typename _Derived, typename _DenseProblem>
	class DenseSolverBase {
	public:
		typedef Eigen::MatrixXd Matrix;
		typedef Eigen::VectorXd Vector;
		typedef std::shared_ptr<Eigen::MatrixXd> MatrixPtr;
		typedef std::shared_ptr<Eigen::VectorXd> VectorPtr;

		VectorPtr solve(_DenseProblem& problem) {
			return static_cast<_Derived*>(this)->solveImpl(problem);
		}
		
	};
}