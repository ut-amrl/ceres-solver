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

#include <iostream>
#include <string>

#include "ceres/dense_cuda_solver.h"
#include "ceres/internal/eigen.h"

#include "glog/logging.h"
#include "gtest/gtest.h"

using std::cout;
using std::endl;
using std::string;

namespace ceres {
namespace internal {

// Tests the CUDA Cholesky solver with a simple 4x4 matrix.
TEST(DenseCholeskyCudaSolver, Cholesky4x4Matrix) {
  Eigen::Matrix4d A;
  A <<  4,  12, -16, 0,
       12,  37, -43, 0,
      -16, -43,  98, 0,
        0,   0,   0, 1;
  const Eigen::Vector4d b = Eigen::Vector4d::Ones();
  string error_string;
  DenseCudaSolver dense_cuda_solver;
  ASSERT_EQ(dense_cuda_solver.CholeskyFactorize(A.cols(),
                                                A.data(),
                                                &error_string),
            LinearSolverTerminationType::LINEAR_SOLVER_SUCCESS);
  Eigen::Vector4d x = Eigen::Vector4d::Zero();
  ASSERT_EQ(dense_cuda_solver.CholeskySolve(b.data(), x.data(), &error_string),
            LinearSolverTerminationType::LINEAR_SOLVER_SUCCESS);
  EXPECT_DOUBLE_EQ(x(0), 113.75 / 3.0);
  EXPECT_DOUBLE_EQ(x(1), -31.0 / 3.0);
  EXPECT_DOUBLE_EQ(x(2), 5.0 / 3.0);
  EXPECT_DOUBLE_EQ(x(3), 1.0000);
}

}  // namespace internal
}  // namespace ceres
