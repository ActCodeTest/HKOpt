#include <Eigen/dense>
#include "DenseProblemBase.h"

namespace NLP {
    class DenseLinearProblem : public Core::DenseProblemBase<DenseLinearProblem> {
        friend class Core::DenseProblemBase<DenseLinearProblem>;
    public:
        typedef Core::DenseProblemBase<DenseLinearProblem> DenseProblemBase;
        typedef typename DenseProblemBase::Vector Vector;
        typedef typename DenseProblemBase::Matrix Matrix;
        typedef typename DenseProblemBase::VectorPtr VectorPtr;
        typedef typename DenseProblemBase::MatrixPtr MatrixPtr;

        DenseLinearProblem() = default;
        DenseLinearProblem(
            const VectorPtr& c,
            const VectorPtr& x,
            const MatrixPtr& A,
            const VectorPtr& b) :
            c_(c), DenseProblemBase(x, A, b) { }
        
        // Getter
        inline VectorPtr& c() { return c_; }

    protected:
        double evaluateImpl() const {
            return c_->dot(*x_);
        }

        Vector gradientImpl() const {
            return *c_;
        }

        Matrix hessianImpl() const {
            return Matrix::Zero(x_->size(), x_->size());
        }

    private:
        VectorPtr c_;
    };
}
