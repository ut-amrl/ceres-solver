// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2022 Google Inc. All rights reserved.
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
//
// A CUDA-accelerated incomplete Cholesky preconditioner.

#include "ceres/cuda_incomplete_cholesky_preconditioner.h"
#include "ceres/wall_time.h"

#ifndef CERES_NO_CUDA
#include "cusparse.h"

#include "ceres/cuda_cgnr_linear_operator.h"
#include "ceres/cuda_conjugate_gradients_solver.h"
#include "ceres/cuda_linear_operator.h"
#include "ceres/cuda_preconditioner.h"
#include "ceres/cuda_sparse_matrix.h"
#include "ceres/cuda_vector.h"

namespace ceres::internal {

// Compute M = A'A, where A is a CudaSparseMatrix.
void CudaSpMtM(const CudaSparseMatrix& A,
               CudaSparseMatrix* M,
               ContextImpl* context) {
  CudaSparseMatrix A_transpose;
  std::string message;
  CHECK(A_transpose.Init(context, &message));
  A_transpose.CopyFromTranspose(A);
  M->Multiply(A_transpose, A);
}

bool CudaIncompleteCholeskyPreconditioner::Update(
    const CudaSparseMatrix& A, const CudaVector& D) {
  EventLogger event_logger("CudaIncompleteCholeskyPreconditioner::Update");
  CHECK_NE(context_, nullptr);
  if (!H_.Init(context_, nullptr) || !A_transpose_.Init(context_, nullptr)) {
    printf("Failed to initialize H or A_transpose.\n");
    return false;
  }
  // Copy A to A_transpose.
  // A_transpose_.CopyFromTranspose(A);
  // Compute the Hessian.
  CudaSpMtM(A, &H_, context_);
  cudaDeviceSynchronize();
  event_logger.AddEvent("HessianComputation");
  // Todo(Joydeep): Add D to diagonal to ensure positive-definiteness.

  // Compute incomplete Cholesky factorization.
  return true;
}

void CudaIncompleteCholeskyPreconditioner::Apply(
    const CudaVector& x, CudaVector* y) {
}

} // namespace ceres::internal

#endif  // CERES_NO_CUDA