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

#ifndef CERES_NO_CUDA

#include <memory>
#include <string>
#include <vector>

#include "ceres/crs_matrix.h"
#include "ceres/cuda_helpers.h"
#include "ceres/cuda_sparse.h"
#include "ceres/execution_summary.h"
#include "ceres/linear_solver.h"
#include "ceres/sparse_cholesky.h"
#include "ceres/compressed_row_sparse_matrix.h"

#include "cuda_runtime.h"
#include "cusolverSp.h"
#include "glog/logging.h"

namespace ceres {
namespace internal {

std::unique_ptr<SparseCholesky> CudaSparseCholesky::Create(
    OrderingType ordering_type) {
  if (ordering_type != NATURAL) {
    LOG(FATAL) << "Ordering type other than NATURAL is not supported by "
               << "CudaSparseCholesky.";
  }
  return std::unique_ptr<SparseCholesky>(new CudaSparseCholesky(ordering_type));
}

CudaSparseCholesky::CudaSparseCholesky(const OrderingType ordering_type) {
  if (ordering_type != NATURAL) {
    LOG(FATAL) << "Ordering type other than NATURAL is not supported by "
               << "CudaSparseCholesky.";
  }
  CHECK_EQ(cusolverSpCreate(&cusolver_handle_), CUSOLVER_STATUS_SUCCESS);
  CHECK_EQ(cudaStreamCreate(&stream_), cudaSuccess);
  CHECK_EQ(cusolverSpSetStream(cusolver_handle_, stream_),
      CUSOLVER_STATUS_SUCCESS);
}

CudaSparseCholesky::~CudaSparseCholesky() {
  CHECK_EQ(cusolverSpDestroy(cusolver_handle_), CUSOLVER_STATUS_SUCCESS);
  CHECK_EQ(cudaStreamDestroy(stream_), cudaSuccess);
  printf("CudaSparseCholesky:\n");
  execution_summary_.Print("ToCRSMatrix");
  execution_summary_.Print("MemcpyHostToGPU");
  execution_summary_.Print("MemcpyGPUToHost");
  execution_summary_.Print("SpDcsrlsvchol");
}

CompressedRowSparseMatrix::StorageType CudaSparseCholesky::StorageType() const {
  return CompressedRowSparseMatrix::LOWER_TRIANGULAR;
}


LinearSolverTerminationType CudaSparseCholesky::Factorize(
    CompressedRowSparseMatrix* lhs,
    std::string* message) {
  {
    ScopedExecutionTimer timer("ToCRSMatrix", &execution_summary_);
    lhs->ToCRSMatrix(&lhs_);
  }
  {
    ScopedExecutionTimer timer("MemcpyHostToGPU", &execution_summary_);
    cuda_lhs_csr_cols_.CopyToGpu(lhs_.cols);
    cuda_lhs_csr_rows_.CopyToGpu(lhs_.rows);
    cuda_lhs_csr_vals_.CopyToGpu(lhs_.values);
  }
  return LinearSolverTerminationType::LINEAR_SOLVER_SUCCESS;
}

LinearSolverTerminationType CudaSparseCholesky::Solve(const double* rhs,
                                                      double* solution,
                                                      std::string* message) {
  {
    ScopedExecutionTimer timer("MemcpyHostToGPU", &execution_summary_);
    CHECK_EQ(lhs_.num_cols, lhs_.num_rows);
    cuda_rhs_.CopyToGpu(rhs, lhs_.num_rows);
  }
  cusparseMatDescr_t lhs_descr;
  CHECK_EQ(cusparseCreateMatDescr(&lhs_descr), CUSPARSE_STATUS_SUCCESS);
  CHECK_EQ(cusparseSetMatType(lhs_descr, CUSPARSE_MATRIX_TYPE_GENERAL),
      CUSPARSE_STATUS_SUCCESS);
  {
    ScopedExecutionTimer timer("SpXcsrsymamd", &execution_summary_);
    int* permutation_vector_ = new int[lhs_.num_rows];
    CHECK_EQ(cusolverSpXcsrsymamdHost(cusolver_handle_,
                                      lhs_.num_rows,
                                      lhs_.values.size(),
                                      lhs_descr,
                                      lhs_.rows.data(),
                                      lhs_.cols.data(),
                                      permutation_vector_),
        CUSOLVER_STATUS_SUCCESS);

    delete [] permutation_vector_;
  }
  {
    ScopedExecutionTimer timer("SpDcsrlsvchol", &execution_summary_);
    int singularity = 0;
    cuda_solution_.Reserve(lhs_.num_cols);
    CHECK_EQ(cusolverSpDcsrlsvchol(cusolver_handle_,
                                  lhs_.num_rows,
                                  lhs_.values.size(),
                                  lhs_descr,
                                  cuda_lhs_csr_vals_.data(),
                                  cuda_lhs_csr_rows_.data(),
                                  cuda_lhs_csr_cols_.data(),
                                  cuda_rhs_.data(),
                                  0,
                                  0,
                                  cuda_solution_.data(),
                                  &singularity),
        CUSOLVER_STATUS_SUCCESS);
  }
  {
    ScopedExecutionTimer timer("MemcpyGPUToHost", &execution_summary_);
    cuda_solution_.CopyToHost(solution, lhs_.num_rows);
  }
  return LinearSolverTerminationType::LINEAR_SOLVER_SUCCESS;
}

}  // namespace internal
}  // namespace ceres

#endif  // CERES_NO_CUDA