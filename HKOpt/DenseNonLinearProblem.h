#include <limits>
#include <functional>

#include <Eigen/dense>
#include "DenseProblemBase.h"


namespace NLP {

    class DenseNonLinearProblem : public Core::DenseProblemBase<DenseNonLinearProblem> {
        friend class Core::DenseProblemBase<DenseNonLinearProblem>;
    public:
        typedef Core::DenseProblemBase<DenseNonLinearProblem> DenseProblemBase;
        typedef typename DenseProblemBase::Vector Vector;
        typedef typename DenseProblemBase::Matrix Matrix;
        typedef typename DenseProblemBase::VectorPtr VectorPtr;
        typedef typename DenseProblemBase::MatrixPtr MatrixPtr;
        typedef std::function<double(const Vector&)> Function;

        DenseNonLinearProblem() = default;
        DenseNonLinearProblem(
            const Function& f,
            const VectorPtr& x,
            const MatrixPtr& A,
            const VectorPtr& b) :
            f_(f), DenseProblemBase(x, A, b) {
        }

    protected:

        double evaluateImpl() const {
            return f_(*x_);
        }

        Vector gradientImpl() const {
            const double e = std::pow(std::numeric_limits<double>::epsilon(), 1.0 / 3.0);
            Vector gradient = Vector::Zero(x_->size());

            for (Eigen::Index i = 0; i < x_->size(); i++) {
                Vector x_up = *x_, x_down = *x_;
                x_up(i) += e;
                x_down(i) -= e;
                gradient(i) = (f_(x_up) - f_(x_down)) / (2 * e);
            }

            return gradient;
        }


        Matrix hessianImpl() const {
            const double e = std::pow(std::numeric_limits<double>::epsilon(), 1.0 / 3.0);
            Matrix hessian = Matrix::Zero(x_->size(), x_->size());

            double f_x = f_(*x_);  // Store base function value

            for (Eigen::Index i = 0; i < x_->size(); i++) {
                for (Eigen::Index j = 0; j < x_->size(); j++) {
                    Vector x_ij = *x_, x_i = *x_, x_j = *x_;
                    x_ij(i) += e; x_ij(j) += e;
                    x_i(i) += e;
                    x_j(j) += e;

                    double f_ij = f_(x_ij);
                    double f_i = f_(x_i);
                    double f_j = f_(x_j);

                    hessian(i, j) = (f_ij - f_i - f_j + f_x) / (4 * e * e);
                }
            }

            return hessian;
        }

    private:
        Function f_;

    };
}
