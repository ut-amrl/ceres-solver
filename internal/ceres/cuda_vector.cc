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

// This include must come before any #ifndef check on Ceres compile options.
// clang-format off
#include "ceres/internal/config.h"
// clang-format on

#include <math.h>

#include "ceres/internal/export.h"
#include "ceres/types.h"
#include "ceres/context_impl.h"

#ifndef CERES_NO_CUDA

#include "ceres/cuda_buffer.h"
#include "ceres/cuda_vector.h"
#include "ceres/ceres_cuda_kernels.h"
#include "cublas_v2.h"

namespace ceres::internal {

std::unique_ptr<CudaVector> CudaVector::Create(ContextImpl* context, int size) {
  if (context == nullptr || !context->InitCUDA(nullptr)) {
    return nullptr;
  }
  return std::unique_ptr<CudaVector>(new CudaVector(context, size));
}

CudaVector::CudaVector(ContextImpl* context, int size) :
    context_(context) {
  resize(size);
}

bool CudaVector::Init(ContextImpl* context, std::string* message) {
  CHECK(message != nullptr);
  if (context == nullptr) {
    *message = "CudaVector::Init: context is nullptr";
    return false;
  }
  if (!context->InitCUDA(message)) {
    *message = "CudaVector::Init: context->InitCUDA() failed";
    return false;
  }
  context_ = context;
  return true;
}

void CudaVector::resize(int size) {
  data_.Reserve(size);
  num_rows_ = size;
  cusparseCreateDnVec(&cusparse_descr_,
                      num_rows_,
                      data_.data(),
                      CUDA_R_64F);
}

double CudaVector::dot(const CudaVector& x) const {
  double result = 0;
  CHECK_EQ(cublasDdot(context_->cublas_handle_,
                      num_rows_,
                      data_.data(),
                      1,
                      x.data().data(),
                      1, &result),
           CUBLAS_STATUS_SUCCESS) << "CuBLAS cublasDdot failed.";
  return result;
}

double CudaVector::norm() const {
  double result = 0;
  CHECK_EQ(cublasDnrm2(context_->cublas_handle_,
                       num_rows_,
                       data_.data(),
                       1,
                       &result),
           CUBLAS_STATUS_SUCCESS) << "CuBLAS cublasDnrm2 failed.";
  return result;
}

void CudaVector::CopyFromCpu(const Vector& x) {
  data_.Reserve(x.rows());
  data_.CopyFromCpu(x.data(), x.rows(), context_->stream_);
  num_rows_ = x.rows();
  cusparseCreateDnVec(&cusparse_descr_,
                      num_rows_,
                      data_.data(),
                      CUDA_R_64F);
}

void CudaVector::CopyFromCpu(const double* x, int size) {
  data_.Reserve(size);
  data_.CopyFromCpu(x, size, context_->stream_);
  num_rows_ = size;
  cusparseCreateDnVec(&cusparse_descr_,
                      num_rows_,
                      data_.data(),
                      CUDA_R_64F);
}

void CudaVector::CopyTo(Vector* x) const {
  CHECK(x != nullptr);
  x->resize(num_rows_);
  // Need to synchronize with any GPU kernels that may be writing to the
  // buffer before the transfer happens.
  CHECK_EQ(cudaStreamSynchronize(context_->stream_), cudaSuccess);
  data_.CopyToCpu(x->data(), num_rows_);
}

void CudaVector::CopyTo(double* x) const {
  CHECK(x != nullptr);
  // Need to synchronize with any GPU kernels that may be writing to the
  // buffer before the transfer happens.
  CHECK_EQ(cudaStreamSynchronize(context_->stream_), cudaSuccess);
  data_.CopyToCpu(x, num_rows_);
}

void CudaVector::CopyFromCpu(const CudaVector& x) {
  data_.CopyNItemsFrom(x.num_rows_, x.data(), context_->stream_);
  num_rows_ = x.num_rows_;
  cusparseCreateDnVec(&cusparse_descr_,
                      num_rows_,
                      data_.data(),
                      CUDA_R_64F);
}

void CudaVector::setZero() {
  CHECK(data_.data() != nullptr);
  CudaSetZeroFP64(data_.data(), num_rows_, context_->stream_);
}

void CudaVector::Axpy(double a, const CudaVector& x) {
  CHECK_EQ(num_rows_, x.num_rows_);
  CHECK_EQ(cublasDaxpy(context_->cublas_handle_,
                       num_rows_,
                       &a,
                       x.data().data(),
                       1,
                       data_.data(),
                       1),
           CUBLAS_STATUS_SUCCESS) << "CuBLAS cublasDaxpy failed.";
}

void CudaVector::Axpby(double a, const CudaVector& x, double b) {
  CHECK_EQ(num_rows_, x.num_rows_);
  // First scale y by b.
  CHECK_EQ(cublasDscal(context_->cublas_handle_,
                       num_rows_,
                       &b,
                       data_.data(),
                       1),
           CUBLAS_STATUS_SUCCESS) << "CuBLAS cublasDscal failed.";
  // Then add a * x to y.
  CHECK_EQ(cublasDaxpy(context_->cublas_handle_,
                       num_rows_,
                       &a,
                       x.data().data(),
                       1,
                       data_.data(),
                       1),
           CUBLAS_STATUS_SUCCESS) << "CuBLAS cublasDaxpy failed.";
}

void CudaVector::DtDxpy(const CudaVector& D, const CudaVector& x) {
  CudaDtDxpy(data_.data(),
             D.data().data(),
             x.data().data(),
             num_rows_,
             context_->stream_);
}

}  // namespace ceres::internal

#endif  // CERES_NO_CUDA