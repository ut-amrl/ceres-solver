// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2016 Google Inc. All rights reserved.
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
// Author: vitus@google.com (Michael Vitus)

#include <fstream>
#include <iostream>
#include <string>

#include "ceres/ceres.h"
#include "common/read_g2o.h"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "pose_graph_3d_error_term.h"
#include "types.h"

DEFINE_string(input, "", "The pose graph definition filename in g2o format.");

DEFINE_string(linear_solver, "sparse_schur", "Options are: "
              "sparse_schur, dense_schur, iterative_schur, sparse_normal_cholesky, "
              "dense_qr, dense_normal_cholesky, cgnr, and "
              "experimental_custom_solver");
DEFINE_string(preconditioner, "jacobi", "Options are: "
              "identity, jacobi, schur_jacobi, cluster_jacobi, "
              "cluster_tridiagonal.");
DEFINE_int32(num_iterations, 5, "Number of iterations.");
DEFINE_bool(mixed_precision_solves, false, "Use mixed precision solves.");
DEFINE_bool(line_search, false, "Use a line search instead of trust region "
            "algorithm.");
DEFINE_int32(num_threads, 1, "Number of threads.");
DEFINE_double(max_solver_time, 1e32, "Maximum solve time in seconds.");
DEFINE_bool(nonmonotonic_steps, false, "Trust region algorithm can use"
            " nonmonotic steps.");
DEFINE_int32(max_num_refinement_iterations, 0,
             "Iterative refinement iterations");
DEFINE_string(trust_region_strategy, "levenberg_marquardt",
              "Options are: levenberg_marquardt, dogleg.");
DEFINE_string(dogleg, "traditional_dogleg", "Options are: traditional_dogleg,"
              "subspace_dogleg.");
DEFINE_bool(inner_iterations, false, "Use inner iterations to non-linearly "
            "refine each successful trust region step.");
namespace ceres::examples {
namespace {

void SetOptions(Solver::Options* options) {
  CHECK(StringToLinearSolverType(CERES_GET_FLAG(FLAGS_linear_solver),
                                 &options->linear_solver_type));
  CHECK(StringToPreconditionerType(CERES_GET_FLAG(FLAGS_preconditioner),
                                   &options->preconditioner_type));
  // CHECK(StringToVisibilityClusteringType(
  //     CERES_GET_FLAG(FLAGS_visibility_clustering),
  //     &options->visibility_clustering_type));
  // CHECK(StringToSparseLinearAlgebraLibraryType(
  //     CERES_GET_FLAG(FLAGS_sparse_linear_algebra_library),
  //     &options->sparse_linear_algebra_library_type));
  // CHECK(StringToDenseLinearAlgebraLibraryType(
  //     CERES_GET_FLAG(FLAGS_dense_linear_algebra_library),
  //     &options->dense_linear_algebra_library_type));
  // CHECK(
  //     StringToLinearSolverOrderingType(CERES_GET_FLAG(FLAGS_ordering_type),
  //                                      &options->linear_solver_ordering_type));
  // options->use_explicit_schur_complement =
  //     CERES_GET_FLAG(FLAGS_explicit_schur_complement);
  options->use_mixed_precision_solves =
      CERES_GET_FLAG(FLAGS_mixed_precision_solves);
  options->max_num_refinement_iterations =
      CERES_GET_FLAG(FLAGS_max_num_refinement_iterations);


  options->max_num_iterations = CERES_GET_FLAG(FLAGS_num_iterations);
  options->minimizer_progress_to_stdout = true;
  options->num_threads = CERES_GET_FLAG(FLAGS_num_threads);
  // options->eta = CERES_GET_FLAG(FLAGS_eta);
  // options->max_solver_time_in_seconds = CERES_GET_FLAG(FLAGS_max_solver_time);
  // options->use_nonmonotonic_steps = CERES_GET_FLAG(FLAGS_nonmonotonic_steps);
  if (CERES_GET_FLAG(FLAGS_line_search)) {
    options->minimizer_type = ceres::LINE_SEARCH;
  }

  CHECK(StringToTrustRegionStrategyType(
      CERES_GET_FLAG(FLAGS_trust_region_strategy),
      &options->trust_region_strategy_type));
  CHECK(
      StringToDoglegType(CERES_GET_FLAG(FLAGS_dogleg), &options->dogleg_type));
  options->use_inner_iterations = CERES_GET_FLAG(FLAGS_inner_iterations);
}

// Constructs the nonlinear least squares optimization problem from the pose
// graph constraints.
void BuildOptimizationProblem(const VectorOfConstraints& constraints,
                              MapOfPoses* poses,
                              ceres::Problem* problem) {
  CHECK(poses != nullptr);
  CHECK(problem != nullptr);
  if (constraints.empty()) {
    LOG(INFO) << "No constraints, no problem to optimize.";
    return;
  }

  ceres::LossFunction* loss_function = nullptr;
  ceres::Manifold* quaternion_manifold = new EigenQuaternionManifold;

  for (const auto& constraint : constraints) {
    auto pose_begin_iter = poses->find(constraint.id_begin);
    CHECK(pose_begin_iter != poses->end())
        << "Pose with ID: " << constraint.id_begin << " not found.";
    auto pose_end_iter = poses->find(constraint.id_end);
    CHECK(pose_end_iter != poses->end())
        << "Pose with ID: " << constraint.id_end << " not found.";

    const Eigen::Matrix<double, 6, 6> sqrt_information =
        constraint.information.llt().matrixL();
    // Ceres will take ownership of the pointer.
    ceres::CostFunction* cost_function =
        PoseGraph3dErrorTerm::Create(constraint.t_be, sqrt_information);

    problem->AddResidualBlock(cost_function,
                              loss_function,
                              pose_begin_iter->second.p.data(),
                              pose_begin_iter->second.q.coeffs().data(),
                              pose_end_iter->second.p.data(),
                              pose_end_iter->second.q.coeffs().data());

    problem->SetManifold(pose_begin_iter->second.q.coeffs().data(),
                         quaternion_manifold);
    problem->SetManifold(pose_end_iter->second.q.coeffs().data(),
                         quaternion_manifold);
  }

  // The pose graph optimization problem has six DOFs that are not fully
  // constrained. This is typically referred to as gauge freedom. You can apply
  // a rigid body transformation to all the nodes and the optimization problem
  // will still have the exact same cost. The Levenberg-Marquardt algorithm has
  // internal damping which mitigates this issue, but it is better to properly
  // constrain the gauge freedom. This can be done by setting one of the poses
  // as constant so the optimizer cannot change it.
  auto pose_start_iter = poses->begin();
  CHECK(pose_start_iter != poses->end()) << "There are no poses.";
  problem->SetParameterBlockConstant(pose_start_iter->second.p.data());
  problem->SetParameterBlockConstant(pose_start_iter->second.q.coeffs().data());
}

// Returns true if the solve was successful.
bool SolveOptimizationProblem(ceres::Problem* problem) {
  CHECK(problem != nullptr);

  ceres::Solver::Options options;
  options.max_num_iterations = 200;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;

  SetOptions(&options);
  ceres::Solver::Summary summary;
  ceres::Solve(options, problem, &summary);

  std::cout << summary.FullReport() << '\n';

  return summary.IsSolutionUsable();
}

// Output the poses to the file with format: id x y z q_x q_y q_z q_w.
bool OutputPoses(const std::string& filename, const MapOfPoses& poses) {
  std::fstream outfile;
  outfile.open(filename.c_str(), std::istream::out);
  if (!outfile) {
    LOG(ERROR) << "Error opening the file: " << filename;
    return false;
  }
  for (const auto& pair : poses) {
    outfile << pair.first << " " << pair.second.p.transpose() << " "
            << pair.second.q.x() << " " << pair.second.q.y() << " "
            << pair.second.q.z() << " " << pair.second.q.w() << '\n';
  }
  return true;
}

}  // namespace
}  // namespace ceres::examples

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);

  CHECK(FLAGS_input != "") << "Need to specify the filename to read.";

  ceres::examples::MapOfPoses poses;
  ceres::examples::VectorOfConstraints constraints;

  CHECK(ceres::examples::ReadG2oFile(FLAGS_input, &poses, &constraints))
      << "Error reading the file: " << FLAGS_input;

  std::cout << "Number of poses: " << poses.size() << '\n';
  std::cout << "Number of constraints: " << constraints.size() << '\n';

  CHECK(ceres::examples::OutputPoses("poses_original.txt", poses))
      << "Error outputting to poses_original.txt";

  ceres::Problem problem;
  ceres::examples::BuildOptimizationProblem(constraints, &poses, &problem);

  CHECK(ceres::examples::SolveOptimizationProblem(&problem))
      << "The solve was not successful, exiting.";

  CHECK(ceres::examples::OutputPoses("poses_optimized.txt", poses))
      << "Error outputting to poses_original.txt";

  return 0;
}
