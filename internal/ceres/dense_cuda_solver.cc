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
// A C++ interface to dense CUDA solvers.

#ifndef CERES_NO_CUDASOLVER

#include <cstring>
#include <string>
#include <vector>

#include "ceres/dense_cuda_solver.h"
#include "ceres/execution_summary.h"
#include "ceres/linear_solver.h"
#include "ceres/map_util.h"

#include "cuda_runtime.h"
#include "cusolverDn.h"
#include "glog/logging.h"

namespace {
void CudaRealloc(void** ptr, size_t size) {
  if (*ptr != nullptr) {
    CHECK_EQ(cudaFree(*ptr), cudaSuccess);
  }
  CHECK_EQ(cudaMalloc(ptr, size), cudaSuccess);
}
}  // namespace

namespace ceres {
namespace internal {

DenseCudaSolver::DenseCudaSolver() :
    cusolver_handle_(nullptr),
    stream_(nullptr),
    num_cols_(0),
    gpu_a_(nullptr),
    gpu_b_(nullptr),
    max_num_cols_(0),
    gpu_scratch_(nullptr),
    gpu_scratch_size_(0),
    host_scratch_(nullptr),
    host_scratch_size_(0),
    gpu_error_(nullptr) {
  CHECK_EQ(cusolverDnCreate(&cusolver_handle_), CUSOLVER_STATUS_SUCCESS);
  CHECK_EQ(cudaStreamCreate(&stream_), cudaSuccess);
  CHECK_EQ(cusolverDnSetStream(cusolver_handle_, stream_),
      CUSOLVER_STATUS_SUCCESS);
  CHECK_EQ(cudaMalloc(&gpu_error_, sizeof(int)), cudaSuccess);
}

DenseCudaSolver::~DenseCudaSolver() {
  CHECK_EQ(cudaFree(gpu_error_), cudaSuccess);
  if (gpu_scratch_) {
    CHECK_EQ(cudaFree(gpu_scratch_), cudaSuccess);
  }
  if (host_scratch_) {
    free(host_scratch_);
  }
  if (gpu_a_) {
    CHECK_EQ(cudaFree(gpu_a_), cudaSuccess);
  }
  if (gpu_b_) {
    CHECK_EQ(cudaFree(gpu_b_), cudaSuccess);
  }
  CHECK_EQ(cusolverDnDestroy(cusolver_handle_), CUSOLVER_STATUS_SUCCESS);
  CHECK_EQ(cudaStreamDestroy(stream_), cudaSuccess);
  if (true) {
    printf("DenseCudaSolver:\n");
    execution_summary_.Print("AllocateGPUMemory");
    execution_summary_.Print("MemcpyHostToGPU");
    execution_summary_.Print("MemcpyGPUToHost");
    execution_summary_.Print("potrf");
    execution_summary_.Print("potrs");
  }
}

void DenseCudaSolver::AllocateGPUMemory(size_t num_cols) {
  ScopedExecutionTimer timer("AllocateGPUMemory", &execution_summary_);
  if (num_cols <= max_num_cols_) {
    return;
  }
  const size_t sizeof_a = num_cols * num_cols * sizeof(double);
  const size_t sizeof_b = num_cols * sizeof(double);
  CudaRealloc(reinterpret_cast<void**>(&gpu_a_), sizeof_a);
  CudaRealloc(reinterpret_cast<void**>(&gpu_b_), sizeof_b);
  max_num_cols_ = num_cols;
}

// Perform Cholesky factorization of a symmetric matrix A.
LinearSolverTerminationType DenseCudaSolver::CholeskyFactorize(
    int num_cols, double* A, std::string* message) {
  // Allocate GPU memory if necessary.
  AllocateGPUMemory(num_cols);
  num_cols_ = num_cols;
  {
    ScopedExecutionTimer timer("MemcpyHostToGPU", &execution_summary_);
    // Copy A to GPU.
    CHECK_EQ(cudaMemcpy(gpu_a_,
                        A,
                        num_cols * num_cols * sizeof(double),
                        cudaMemcpyHostToDevice),
            cudaSuccess);
  }

  {
    ScopedExecutionTimer timer("potrf", &execution_summary_);
    size_t device_scratch_size = 0;
    size_t host_scratch_size = 0;
    CHECK_EQ(cusolverDnXpotrf_bufferSize(cusolver_handle_,
                                        nullptr,
                                        CUBLAS_FILL_MODE_LOWER,
                                        num_cols,
                                        CUDA_R_64F,
                                        gpu_a_,
                                        num_cols,
                                        CUDA_R_64F,
                                        &device_scratch_size,
                                        &host_scratch_size),
            CUSOLVER_STATUS_SUCCESS);

    // ALlocate GPU scratch memory.
    CudaRealloc(reinterpret_cast<void**>(&gpu_scratch_), device_scratch_size);
    gpu_scratch_size_ = std::max(gpu_scratch_size_, device_scratch_size);

    // Allocate host scratch memory.
    if (host_scratch_size > host_scratch_size_) {
      CHECK_NOTNULL(realloc(
          reinterpret_cast<void**>(&host_scratch_), host_scratch_size));
      host_scratch_size_ = host_scratch_size;
    }

    // Perform Cholesky factorization.
    CHECK_EQ(cusolverDnXpotrf(cusolver_handle_,
                              nullptr,
                              CUBLAS_FILL_MODE_LOWER,
                              num_cols,
                              CUDA_R_64F,
                              gpu_a_,
                              num_cols,
                              CUDA_R_64F,
                              gpu_scratch_,
                              gpu_scratch_size_,
                              host_scratch_,
                              host_scratch_size_,
                              gpu_error_),
            CUSOLVER_STATUS_SUCCESS);
    CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
    CHECK_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
  }

  int error = 0;
  {
    ScopedExecutionTimer timer("MemcpyGPUToHost", &execution_summary_);
    // Check for errors.
    CHECK_EQ(cudaMemcpy(&error,
                        gpu_error_,
                        sizeof(int),
                        cudaMemcpyDeviceToHost),
            cudaSuccess);
  }
  if (error != 0) {
    *message = "cuSOLVER Cholesky factorization failed.";
    return LinearSolverTerminationType::LINEAR_SOLVER_FATAL_ERROR;
  }
  return LinearSolverTerminationType::LINEAR_SOLVER_SUCCESS;
}

LinearSolverTerminationType DenseCudaSolver::CholeskySolve(
    const double* B, double* X, std::string* message) {

  {
    ScopedExecutionTimer timer("MemcpyHostToGPU", &execution_summary_);
    // Copy B to GPU.
    CHECK_EQ(cudaMemcpy(gpu_b_,
                        B,
                        num_cols_ * sizeof(double),
                        cudaMemcpyHostToDevice),
            cudaSuccess);
  }
  {
    ScopedExecutionTimer timer("potrs", &execution_summary_);
    // Solve the system.
    CHECK_EQ(cusolverDnXpotrs(cusolver_handle_,
                              nullptr,
                              CUBLAS_FILL_MODE_LOWER,
                              num_cols_,
                              1,
                              CUDA_R_64F,
                              gpu_a_,
                              num_cols_,
                              CUDA_R_64F,
                              gpu_b_,
                              num_cols_,
                              gpu_error_),
            CUSOLVER_STATUS_SUCCESS);
    CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
    CHECK_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
  }
  // Check for errors.
  int error = 0;
  {
    ScopedExecutionTimer timer("MemcpyGPUToHost", &execution_summary_);
    // Copy error variable from GPU to host.
    CHECK_EQ(cudaMemcpy(&error,
                        gpu_error_,
                        sizeof(int),
                        cudaMemcpyDeviceToHost),
            cudaSuccess);
    // Copy X from GPU to host.
    CHECK_EQ(cudaMemcpy(X,
                        gpu_b_,
                        num_cols_ * sizeof(double),
                        cudaMemcpyDeviceToHost),
            cudaSuccess);
  }
  if (error != 0) {
    *message = "cuSOLVER Cholesky solve failed.";
    return LinearSolverTerminationType::LINEAR_SOLVER_FATAL_ERROR;
  }
  return LinearSolverTerminationType::LINEAR_SOLVER_SUCCESS;
}


}  // namespace internal
}  // namespace ceres

#endif  // CERES_NO_CUDASOLVER
