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
// Author: joydeepb@cs.utexas.edu (Joydeep Biswas)

#include "ceres/experimental_custom_solver.h"


#include <iostream>
#include <memory>

#include "Eigen/Sparse"

#include "ceres/cgnr_solver.h"
#include "ceres/block_sparse_matrix.h"
#include "ceres/triplet_sparse_matrix.h"
#include "ceres/internal/export.h"
#include "ceres/linear_solver.h"

namespace ceres::internal {

LinearSolver::Summary ExperimentalCustomSolver::SolveImpl(
    BlockSparseMatrix* A,
    const double* b_ptr,
    const LinearSolver::PerSolveOptions& per_solve_options,
    double* x)  {
  LinearSolver::Summary summary;
  summary.termination_type = LinearSolverTerminationType::FATAL_ERROR;
  summary.message = "ExperimentalCustomSolver::SolveImpl Not Implemented.";
  summary.num_iterations = 0;
  ConstVectorRef b(b_ptr, A->num_cols());

  TripletSparseMatrix A_ts;
  A->ToTripletSparseMatrix(&A_ts);
  std::vector<Eigen::Triplet<double>> triplets(A_ts.num_nonzeros());
  for (int i = 0; i < A_ts.num_nonzeros(); ++i) {
    triplets[i] = Eigen::Triplet<double>(
        A_ts.rows()[i], A_ts.cols()[i], A_ts.values()[i]);
  }
  Eigen::SparseMatrix<double, Eigen::RowMajor> A_eigen(
      A->num_rows(), A->num_cols());
  A_eigen.setFromTriplets(triplets.begin(), triplets.end());
  Eigen::SparseMatrix<float, Eigen::RowMajor> A_eigen_float(
    A->num_rows(), A->num_cols());
  A_eigen_float = A_eigen.cast<float>();
  if (false) {
    // Just to test that the plumbing works, you can try this.
    CgnrSolver solver(options_);
    return solver.SolveImpl(A, b_ptr, per_solve_options, x);
  }
  // std::cout << "A = \n" << A_eigen << "\n";

  // TODO: Actually solve the problem, and set summary.termination_type to
  // something that is not FATAL_ERROR.
  return summary;

}

}  // namespace ceres::internal

