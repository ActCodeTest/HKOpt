/*
MIT License

Copyright (c) 2024 Harold James Krause

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#pragma once
#include <map>
#include <vector>
#include <Eigen/dense>

namespace HKOpt {

	enum class ConstraintType {GEQ, EQ, LEQ};

    template<typename _Scalar>
    class Constraint {
    public:
        using Vector = Eigen::VectorX<_Scalar>;
        Constraint(const Vector& A, _Scalar b, ConstraintType type) : A_(A), b_(b), type_(type) {}
        inline const Vector& A() const { return A_; }
        inline _Scalar b() const { return b_; }
        inline ConstraintType type() const { return type_; }
    private:
        const Vector A_;
        const _Scalar b_;
        const ConstraintType type_;
    };


    template<typename _Scalar> 
    class Constraints {
    public:
        using Vector = Eigen::VectorX<_Scalar>;
        using Matrix = Eigen::MatrixX<_Scalar>;

        using Constraint = Constraint<_Scalar>;
        using ConstraintPtr = std::shared_ptr<Constraint>;

        Constraints(const std::vector<ConstraintPtr>& constraints) {
            size_t required_size = constraints[0]->A().size();
            for (const auto& constraint : constraints) {
                if (constraint->A().size() != required_size) {
                    throw std::runtime_error("Constraint coefficients must be the same size.");
                }
                constraints_[constraint->type()].push_back(constraint);
            }
        }
        
        void projectOntoConstraints(Vector& x) {
            // Project onto equality constraints 
            
            std::vector<ConstraintPtr> eq_constraints = constraints_[ConstraintType::EQ];
            if (eq_constraints.size() != 0) {
                Matrix A(eq_constraints.size(), x.size());
                Vector b(eq_constraints.size());
                for (auto i = 0; i < eq_constraints.size(); i++) {
                    A.row(i) = eq_constraints[i]->A();
                    b(i) = eq_constraints[i]->b();
                }
                x = x - A.transpose() * (A * A.transpose()).inverse() * (A * x - b);
            }

            std::vector<ConstraintPtr> leq_constraints = constraints_[ConstraintType::LEQ];
            for (const auto& constraint : leq_constraints) {
                Vector A = constraint->A();
                _Scalar b = constraint->b();
                if (A.dot(x) > b) {
                    x = x - (A.dot(x) - b) * A / A.squaredNorm();
                }
            }

            std::vector<ConstraintPtr> geq_constraints = constraints_[ConstraintType::GEQ];
            for (const auto& constraint : geq_constraints) {
                Vector A = constraint->A();
                _Scalar b = constraint->b();
                if (A.dot(x) < b) {
                    x = x + (b - A.dot(x)) * A / A.squaredNorm();
                }
            }
        }

    private:
        std::map<ConstraintType, std::vector<ConstraintPtr>> constraints_;
    };
}