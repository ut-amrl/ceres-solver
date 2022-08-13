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
// A simple CUDA vector class.

#ifndef CERES_INTERNAL_CUDA_VECTOR_H_
#define CERES_INTERNAL_CUDA_VECTOR_H_

// This include must come before any #ifndef check on Ceres compile options.
// clang-format off
#include "ceres/internal/config.h"
// clang-format on

#include <math.h>
#include <memory>
#include <string>

#include "ceres/internal/export.h"
#include "ceres/types.h"
#include "ceres/context_impl.h"

#ifndef CERES_NO_CUDA

#include "ceres/cuda_buffer.h"
#include "ceres/ceres_cuda_kernels.h"
#include "ceres/internal/eigen.h"
#include "cublas_v2.h"
#include "cusparse.h"

namespace ceres::internal {

// An Nx1 vector, denoted y hosted on the GPU, with CUDA-accelerated operations.
class CERES_NO_EXPORT CudaVector {
 public:
  CudaVector() = default;
  ~CudaVector() = default;

  static std::unique_ptr<CudaVector> Create(ContextImpl* context, int size);

  CudaVector(ContextImpl* context, int size);

  bool Init(ContextImpl* context, std::string* message);

  void resize(int size);

  // Return the inner product x' * y.
  double dot(const CudaVector& x) const;

  // Return the L2 norm of the vector (||y||_2).
  double norm() const;

  // Set all elements to zero.
  void setZero();

  // Set y = x.
  void CopyFromCpu(const CudaVector& x);

  // Copy from Eigen vector.
  void CopyFromCpu(const Vector& x);

  // Copy from CPU memory array.
  void CopyFromCpu(const double* x, int size);

  // Copy to Eigen vector.
  void CopyTo(Vector* x) const;

  // Copy to CPU memory array. It is the caller's responsibility to ensure
  // that the array is large enough.
  void CopyTo(double* x) const;

  // y = a * x + y.
  void Axpy(double a, const CudaVector& x);

  // y = a * x + b * y.
  void Axpby(double a, const CudaVector& x, double b);

  // y = diag(d)' * diag(d) * x + y.
  void DtDxpy(const CudaVector& D, const CudaVector& x);

  int num_rows() const { return num_rows_; }
  int num_cols() const { return 1; }

  // Return the pointer to the GPU buffer.
  const CudaBuffer<double>& data() const { return data_; }

  const cusparseDnVecDescr_t& descr() const { return cusparse_descr_; }

 private:
  CudaVector(const CudaVector&) = delete;
  CudaVector& operator=(const CudaVector&) = delete;

  int num_rows_ = 0;
  ContextImpl* context_ = nullptr;
  CudaBuffer<double> data_;
  // CuSparse object that describes this dense vector.
  cusparseDnVecDescr_t cusparse_descr_ = nullptr;
};

// Convenience template initializer.
template<typename T>
bool InitCudaObjects(ContextImpl* context,
                     std::string* error,
                     T obj) {
  return obj->Init(context, error);
}

template<typename T, typename... Targs>
bool InitCudaObjects(ContextImpl* context,
                     std::string* error,
                     T obj1,
                     Targs... objN) {
  return (obj1->Init(context, error) &&
          InitCudaObjects(context, error, objN...));
}

}  // namespace ceres::internal

#endif  // CERES_NO_CUDA
#endif  // CERES_INTERNAL_CUDA_SPARSE_LINEAR_OPERATOR_H_
