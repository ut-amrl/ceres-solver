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

#ifndef CERES_INTERNAL_CUDA_HELPERS_H_
#define CERES_INTERNAL_CUDA_HELPERS_H_

#ifndef CERES_NO_CUDA

#include <vector>

#include "cuda_runtime.h"
#include "glog/logging.h"

template <typename T>
class CudaBuffer {
 public:
  CudaBuffer() : data_(nullptr), size_(0) {}
  CudaBuffer(const CudaBuffer&) = delete;
  CudaBuffer& operator=(const CudaBuffer&) = delete;

  ~CudaBuffer() {
    if (data_ != nullptr) {
      CHECK_EQ(cudaFree(data_), cudaSuccess);
    }
  }

  void Reserve(const size_t size) {
    if (size > size_) {
      if (data_ != nullptr) {
        CHECK_EQ(cudaFree(data_), cudaSuccess);
      }
      CHECK_EQ(cudaMalloc(&data_, size * sizeof(T)), cudaSuccess);
      CHECK_NOTNULL(data_);
      size_ = size;
    }
  }

  void CopyToGpu(const T* data, const size_t size) {
    Reserve(size);
    CHECK_EQ(cudaMemcpy(data_, data, size * sizeof(T), cudaMemcpyHostToDevice),
             cudaSuccess);
  }

  void CopyToHost(T* data, const size_t size) {
    CHECK_NOTNULL(data_);
    CHECK_EQ(cudaMemcpy(data, data_, size * sizeof(T), cudaMemcpyDeviceToHost),
             cudaSuccess);
  }

  void CopyToGpu(const std::vector<T>& data) {
    Reserve(data.size());
    CHECK_EQ(cudaMemcpy(data_,
                        data.data(),
                        data.size() * sizeof(T),
                        cudaMemcpyHostToDevice),
             cudaSuccess);
  }

  T* data() { return data_; }
  size_t size() const { return size_; }

 private:
  T* data_;
  size_t size_;
};

#endif  // CERES_NO_CUDA

#endif  // CERES_INTERNAL_CUDA_HELPERS_H_