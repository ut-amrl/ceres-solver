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

#ifndef CERES_INTERNAL_CUDA_SPARSE_H_
#define CERES_INTERNAL_CUDA_SPARSE_H_

#include "ceres/internal/config.h"

#ifndef CERES_NO_CUDA

#include <memory>
#include <string>
#include <vector>

#include "ceres/cuda_helpers.h"
#include "ceres/crs_matrix.h"
#include "ceres/execution_summary.h"
#include "ceres/linear_solver.h"
#include "ceres/sparse_cholesky.h"
#include "cusolverSp.h"

namespace ceres {
namespace internal {

class CompressedRowSparseMatrix;
class TripletSparseMatrix;

// An implementation of SparseCholesky interface using the CUDA CuSparse
// library.
class CudaSparseCholesky : public SparseCholesky {
 public:
  // Factory
  static std::unique_ptr<SparseCholesky> Create(OrderingType ordering_type);

  // SparseCholesky interface.
  virtual ~CudaSparseCholesky() override;
  CompressedRowSparseMatrix::StorageType StorageType() const final;
  LinearSolverTerminationType Factorize(CompressedRowSparseMatrix* lhs,
                                        std::string* message) final;
  LinearSolverTerminationType Solve(const double* rhs,
                                    double* solution,
                                    std::string* message) final;

 private:
  explicit CudaSparseCholesky(const OrderingType ordering_type);

  // Handle to the cuSOLVER context.
  cusolverSpHandle_t cusolver_handle_;
  // CUDA device stream.
  cudaStream_t stream_;
  // Execution summary.
  ExecutionSummary execution_summary_;

  CRSMatrix lhs_;
  CudaBuffer<int> cuda_lhs_csr_rows_;
  CudaBuffer<int> cuda_lhs_csr_cols_;
  CudaBuffer<double> cuda_lhs_csr_vals_;
  CudaBuffer<double> cuda_rhs_;
  CudaBuffer<double> cuda_solution_;
};

}  // namespace internal
}  // namespace ceres

#endif  // CERES_NO_CUDA

#endif  // CERES_INTERNAL_CUDA_SPARSE_H_
