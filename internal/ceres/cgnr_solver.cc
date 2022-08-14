// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2015 Google Inc. All rights reserved.
// http://ceres-solver.org/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of Google Inc. nor the names of its contributors may be
//   used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: keir@google.com (Keir Mierle)

#include "ceres/cgnr_solver.h"

#include <memory>
#include <utility>

#include "ceres/block_jacobi_preconditioner.h"
#include "ceres/conjugate_gradients_solver.h"
#include "ceres/internal/eigen.h"
#include "ceres/linear_solver.h"
#include "ceres/subset_preconditioner.h"
#include "ceres/wall_time.h"
#include "glog/logging.h"

#ifndef CERES_NO_CUDA
#include "ceres/cuda_sparse_matrix.h"
#include "ceres/cuda_vector.h"
#endif  // CERES_NO_CUDA

namespace ceres::internal {

// A linear operator which takes a matrix A and a diagonal vector D and
// performs products of the form
//
//   (A^T A + D^T D)x
//
// This is used to implement iterative general sparse linear solving with
// conjugate gradients, where A is the Jacobian and D is a regularizing
// parameter. A brief proof that D^T D is the correct regularizer:
//
// Given a regularized least squares problem:
//
//   min  ||Ax - b||^2 + ||Dx||^2
//    x
//
// First expand into matrix notation:
//
//   (Ax - b)^T (Ax - b) + xD^TDx
//
// Then multiply out to get:
//
//   = xA^TAx - 2b^T Ax + b^Tb + xD^TDx
//
// Take the derivative:
//
//   0 = 2A^TAx - 2A^T b + 2 D^TDx
//   0 = A^TAx - A^T b + D^TDx
//   0 = (A^TA + D^TD)x - A^T b
//
// Thus, the symmetric system we need to solve for CGNR is
//
//   Sx = z
//
// with S = A^TA + D^TD
//  and z = A^T b
//
// Note: This class is not thread safe, since it uses some temporary storage.
class CERES_NO_EXPORT CgnrLinearOperator final
    : public ConjugateGradientsLinearOperator<Vector> {
 public:
  CgnrLinearOperator(const LinearOperator& A, const double* D)
      : A_(A), D_(D), z_(Vector::Zero(A.num_rows())) {}

  void RightMultiplyAndAccumulate(const Vector& x, Vector& y) final {
    // z = Ax
    // y = y + Atz
    z_.setZero();
    A_.RightMultiplyAndAccumulate(x, z_);
    A_.LeftMultiplyAndAccumulate(z_, y);

    // y = y + DtDx
    if (D_ != nullptr) {
      int n = A_.num_cols();
      y.array() += ConstVectorRef(D_, n).array().square() * x.array();
    }
  }

 private:
  const LinearOperator& A_;
  const double* D_;
  Vector z_;
};

CgnrSolver::CgnrSolver(LinearSolver::Options options)
    : options_(std::move(options)) {
  if (options_.preconditioner_type != JACOBI &&
      options_.preconditioner_type != IDENTITY &&
      options_.preconditioner_type != SUBSET) {
    LOG(FATAL)
        << "Preconditioner = "
        << PreconditionerTypeToString(options_.preconditioner_type) << ". "
        << "Congratulations, you found a bug in Ceres. Please report it.";
  }
}

CgnrSolver::~CgnrSolver() = default;

LinearSolver::Summary CgnrSolver::SolveImpl(
    BlockSparseMatrix* A,
    const double* b,
    const LinearSolver::PerSolveOptions& per_solve_options,
    double* x) {
  EventLogger event_logger("CgnrSolver::Solve");
  if (!preconditioner_) {
    if (options_.preconditioner_type == JACOBI) {
      preconditioner_ = std::make_unique<BlockJacobiPreconditioner>(*A);
    } else if (options_.preconditioner_type == SUBSET) {
      Preconditioner::Options preconditioner_options;
      preconditioner_options.type = SUBSET;
      preconditioner_options.subset_preconditioner_start_row_block =
          options_.subset_preconditioner_start_row_block;
      preconditioner_options.sparse_linear_algebra_library_type =
          options_.sparse_linear_algebra_library_type;
      preconditioner_options.ordering_type = options_.ordering_type;
      preconditioner_options.num_threads = options_.num_threads;
      preconditioner_options.context = options_.context;
      preconditioner_ =
          std::make_unique<SubsetPreconditioner>(preconditioner_options, *A);
    } else {
      preconditioner_ = std::make_unique<IdentityPreconditioner>(A->num_cols());
    }
  }

  preconditioner_->Update(*A, per_solve_options.D);

  ConjugateGradientsSolverOptions cg_options;
  cg_options.min_num_iterations = options_.min_num_iterations;
  cg_options.max_num_iterations = options_.max_num_iterations;
  cg_options.residual_reset_period = options_.residual_reset_period;
  cg_options.q_tolerance = per_solve_options.q_tolerance;
  cg_options.r_tolerance = per_solve_options.r_tolerance;

  // lhs = AtA + DtD
  CgnrLinearOperator lhs(*A, per_solve_options.D);
  // rhs = Atb.
  Vector rhs(A->num_cols());
  rhs.setZero();
  A->LeftMultiplyAndAccumulate(b, rhs.data());

  cg_solution_ = Vector::Zero(A->num_cols());
  for (int i = 0; i < 4; ++i) {
    scratch_[i] = Vector::Zero(A->num_cols());
  }
  event_logger.AddEvent("Setup");

  Vector* scratch_ptr[4] = {
      &scratch_[0], &scratch_[1], &scratch_[2], &scratch_[3],
  };

  LinearOperatorAdapter preconditioner(*preconditioner_);
  auto summary = ConjugateGradientsSolver(
      cg_options, lhs, rhs, preconditioner, scratch_ptr, cg_solution_);
  VectorRef(x, A->num_cols()) = cg_solution_;
  event_logger.AddEvent("Solve");
  return summary;
}

#ifndef CERES_NO_CUDA

// A linear operator which takes a matrix A and a diagonal vector D and
// performs products of the form
//
//   (A^T A + D^T D)x
//
// This is used to implement iterative general sparse linear solving with
// conjugate gradients, where A is the Jacobian and D is a regularizing
// parameter. A brief proof is included in cgnr_linear_operator.h.
class CERES_NO_EXPORT CudaCgnrLinearOperator final : public
    ConjugateGradientsLinearOperator<CudaVector> {
 public:
  CudaCgnrLinearOperator(CudaSparseMatrix& A,
                         CudaVector* D,
                         CudaVector* z) : A_(A), D_(D), z_(z) {}

  void RightMultiplyAndAccumulate(const CudaVector& x, CudaVector& y) final {
    // z = Ax
    z_->SetZero();
    A_.RightMultiplyAndAccumulate(x, z_);

    // y = y + Atz
    //   = y + AtAx
    A_.LeftMultiplyAndAccumulate(*z_, &y);

    // y = y + DtDx
    if (D_ != nullptr) {
      y.DtDxpy(*D_, x);
    }
  }

 private:
  CudaSparseMatrix& A_;
  CudaVector* D_ = nullptr;
  CudaVector* z_ = nullptr;
};

class CERES_NO_EXPORT CudaIdentityPreconditioner final : public
    ConjugateGradientsLinearOperator<CudaVector> {
 public:
  void RightMultiplyAndAccumulate(const CudaVector& x, CudaVector& y) final {
    y.Axpby(1.0, x, 1.0);
  }
};

CudaCgnrSolver::CudaCgnrSolver() = default;

CudaCgnrSolver::~CudaCgnrSolver() = default;

bool CudaCgnrSolver::Init(
    const LinearSolver::Options& options, std::string* error) {
  options_ = options;
  context_ = options.context;
  return true;
}

std::unique_ptr<CudaCgnrSolver> CudaCgnrSolver::Create(
      LinearSolver::Options options, std::string* error) {
  CHECK(error != nullptr);
  if (options.preconditioner_type != IDENTITY) {
    *error = "CudaCgnrSolver does not support preconditioner type " +
        std::string(PreconditionerTypeToString(options.preconditioner_type)) + ". ";
    return nullptr;
  }
  if (!options.context->InitCUDA(error)) {
    return nullptr;
  }
  std::unique_ptr<CudaCgnrSolver> solver(new CudaCgnrSolver());
  if (!solver->Init(options, error)) {
    return nullptr;
  }
  solver->options_ = options;
  return solver;
}

LinearSolver::Summary CudaCgnrSolver::SolveImpl(
    CompressedRowSparseMatrix* A,
    const double* b,
    const LinearSolver::PerSolveOptions& per_solve_options,
    double* x) {
  static const bool kDebug = false;
  EventLogger event_logger("CudaCgnrSolver::Solve");
  LinearSolver::Summary summary;
  summary.num_iterations = 0;
  summary.termination_type = LinearSolverTerminationType::FATAL_ERROR;

  if (A_ == nullptr) {
    // Assume structure is not cached, do an initialization and structural copy.
    A_ = std::make_unique<CudaSparseMatrix>(options_.context, *A);
    b_ = std::make_unique<CudaVector>(options_.context, A->num_rows());
    x_ = std::make_unique<CudaVector>(options_.context, A->num_cols());
    Atb_ = std::make_unique<CudaVector>(options_.context, A->num_cols());
    Ax_ = std::make_unique<CudaVector>(options_.context, A->num_rows());
    D_ = std::make_unique<CudaVector>(options_.context, A->num_cols());
    if (A_ == nullptr ||
        b_ == nullptr ||
        x_ == nullptr ||
        Atb_ == nullptr ||
        Ax_ == nullptr ||
        D_ == nullptr) {
      summary.message = "Cuda Matrix or Vector initialization failed.";
      return summary;
    }
    for (int i = 0; i < 4; ++i) {
      scratch_[i] = std::make_unique<CudaVector>(
          options_.context, A->num_cols());
      if (scratch_[i] == nullptr) {
        summary.message = "CudaVector::Create failed.";
        return summary;
      }
    }
    event_logger.AddEvent("Initialize");
  } else {
    // Assume structure is cached, do a value copy.
    A_->CopyValues(*A);
    event_logger.AddEvent("A CPU to GPU Transfer");
  }
  b_->CopyFromCpu(ConstVectorRef(b, A->num_rows()));
  D_->CopyFromCpu(ConstVectorRef(per_solve_options.D, A->num_cols()));
  event_logger.AddEvent("b CPU to GPU Transfer");

  // Form z = Atb.
  Atb_->SetZero();
  A_->LeftMultiplyAndAccumulate(*b_, Atb_.get());
  if (kDebug) printf("z = Atb\n");

  LinearSolver::PerSolveOptions cg_per_solve_options = per_solve_options;
  cg_per_solve_options.preconditioner = nullptr;

  // Solve (AtA + DtD)x = z (= Atb).
  x_->SetZero();
  CudaCgnrLinearOperator lhs(*A_, D_.get(), Ax_.get());

  event_logger.AddEvent("Setup");
  if (kDebug) printf("Solve (AtA + DtD)x = z (= Atb)\n");

  ConjugateGradientsSolverOptions cg_options;
  cg_options.min_num_iterations = options_.min_num_iterations;
  cg_options.max_num_iterations = options_.max_num_iterations;
  cg_options.residual_reset_period = options_.residual_reset_period;
  cg_options.q_tolerance = per_solve_options.q_tolerance;
  cg_options.r_tolerance = per_solve_options.r_tolerance;

  CudaVector* scratch_ptr[4] = {
    scratch_[0].get(), scratch_[1].get(), scratch_[2].get(), scratch_[3].get()
  };

  CudaIdentityPreconditioner preconditioner;
  summary = ConjugateGradientsSolver(
      cg_options, lhs, *Atb_, preconditioner, scratch_ptr, *x_);
  x_->CopyTo(x);
  event_logger.AddEvent("Solve");
  return summary;
}

#endif  // CERES_NO_CUDA

}  // namespace ceres::internal
