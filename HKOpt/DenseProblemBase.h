#pragma once

#include <memory>
#include <Eigen/dense>

namespace Core {

    template<typename _Derived>
    class DenseProblemBase {
    public:
        typedef Eigen::MatrixXd Matrix;
        typedef Eigen::VectorXd Vector;
        typedef std::shared_ptr<Eigen::MatrixXd> MatrixPtr;
        typedef std::shared_ptr<Eigen::VectorXd> VectorPtr;

        DenseProblemBase(
            const VectorPtr& x, 
            const MatrixPtr& A, 
            const VectorPtr& b) :
            x_(x), A_(A), b_(b), 
            gradient_(std::make_shared<Vector>(Vector::Zero(x->size()))), 
            gradient_initialized_(false),
            hessian_(std::make_shared<Matrix>(Matrix::Zero(x->size(), x->size()))),
            hessian_initialized_(false) {}

        void update() {
            gradient_initialized_ = false;
            hessian_initialized_ = false;
        }
        
        inline VectorPtr x() const { return x_; }
        inline MatrixPtr A() const { return A_; }
        inline VectorPtr b() const { return b_; }

        inline double evaluate() const {
            return static_cast<const _Derived*>(this)->evaluateImpl();
        }

        inline VectorPtr gradient() { 
            if (!gradient_initialized_) {
                *gradient_ = static_cast<const _Derived*>(this)->gradientImpl();
                gradient_initialized_ = true;
            }
            return gradient_;
        
        }
        inline MatrixPtr hessian() { 
            if (!hessian_initialized_) {
                *hessian_ = static_cast<const _Derived*>(this)->hessianImpl();
                hessian_initialized_ = true;
            }
            return hessian_;
        }

    protected:
        MatrixPtr A_;
        VectorPtr b_;
        VectorPtr x_;

        bool gradient_initialized_;
        VectorPtr gradient_;

        bool hessian_initialized_;
        MatrixPtr hessian_;
    };

}
