/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdlib>
#include <string>
#include <sys/stat.h>

#include <cuda_runtime.h>
#include <nccl.h>

#include "gpu_test_utils.h"

namespace cudecomp_test {
namespace {

bool envFlagEnabled(const char* name) {
  const char* value = std::getenv(name);
  if (!value || value[0] == '\0') return false;

  const std::string flag(value);
  return flag != "0" && flag != "false" && flag != "False" && flag != "FALSE";
}

bool pathExists(const std::string& path) {
  struct stat info;
  return stat(path.c_str(), &info) == 0;
}

std::string mpsPipeDirectory() {
  const char* pipe_dir = std::getenv("CUDA_MPS_PIPE_DIRECTORY");
  if (pipe_dir && pipe_dir[0] != '\0') return pipe_dir;
  return "/tmp/nvidia-mps";
}

bool ranksShareGpu(int local_ranks, int visible_devices) {
  return visible_devices > 0 && local_ranks > visible_devices;
}

TestSetupDecision queryCudaDeviceCount(int& count) {
  cudaError_t status = cudaGetDeviceCount(&count);
  if (status == cudaSuccess) return {};

  count = 0;
  return {false, true, std::string("unable to query CUDA device count: ") + cudaGetErrorString(status)};
}

GpuTestRuntime queryGpuTestRuntime() {
  GpuTestRuntime runtime;
  runtime.mps_active = pathExists(mpsPipeDirectory() + "/nvidia-cuda-mps-control.pid");
  if (ncclGetVersion(&runtime.nccl_version) != ncclSuccess) { runtime.nccl_version = 0; }
  runtime.nccl_multi_rank_gpu_enabled = envFlagEnabled("NCCL_MULTI_RANK_GPU_ENABLE");
  return runtime;
}

TestSetupDecision checkGpuSetupGlobal(const MpiTestComm& comm, const TestSetupDecision& local_decision) {
  const int local_state = local_decision.fail ? 2 : local_decision.skip ? 1 : 0;
  int global_state = 0;
  MPI_Allreduce(&local_state, &global_state, 1, MPI_INT, MPI_MAX, comm.mpiComm());

  if (global_state == 0) return {};
  if (local_decision.skip || local_decision.fail) return local_decision;
  if (global_state == 2) return {false, true, "GPU setup failed on another rank"};
  return {true, false, "GPU setup skipped on another rank"};
}

TestSetupDecision checkGpuTestRequirementsLocal(const MpiTestComm& comm) {
  int device_count = 0;
  const TestSetupDecision device_count_decision = queryCudaDeviceCount(device_count);
  if (device_count_decision.skip || device_count_decision.fail) return device_count_decision;

  if (device_count <= 0) { return {false, true, "GPU tests require at least one visible CUDA device"}; }

  const int local_ranks = comm.localSize();
  if (ranksShareGpu(local_ranks, device_count)) {
    if (!mpsActive()) {
      return {false, true,
              std::to_string(local_ranks) + " local ranks require GPU sharing with only " +
                  std::to_string(device_count) +
                  " visible CUDA device(s); enable CUDA MPS or provide at least one visible GPU per local rank"};
    }
  }

  const int device = comm.localRank() % device_count;
  cudaError_t status = cudaSetDevice(device);
  if (status != cudaSuccess) {
    return {false, true, "unable to set CUDA device " + std::to_string(device) + ": " + cudaGetErrorString(status)};
  }

  return {};
}

TestSetupDecision initializeGpuForTestLocal(const MpiTestComm& comm, bool check_nccl) {
  TestSetupDecision decision = checkGpuTestRequirementsLocal(comm);
  if (decision.skip || decision.fail) return decision;

  if (check_nccl) {
    int device_count = 0;
    decision = queryCudaDeviceCount(device_count);
    if (decision.skip || decision.fail) return decision;

    if (ranksShareGpu(comm.localSize(), device_count)) {
      if (!ncclSupportsMultiRankPerGpu()) {
        return {true, false,
                "NCCL multi-rank-per-GPU testing with MPS requires NCCL 2.30 or newer; runtime reports NCCL version " +
                    std::to_string(ncclVersion())};
      }

      if (!ncclMultiRankGpuEnabled()) {
        return {true, false, "NCCL multi-rank-per-GPU testing with MPS requires NCCL_MULTI_RANK_GPU_ENABLE=1"};
      }
    }
  }

  return {};
}

} // namespace

void initializeGpuTestRuntime() { (void)gpuTestRuntime(); }

const GpuTestRuntime& gpuTestRuntime() {
  static const GpuTestRuntime runtime = queryGpuTestRuntime();
  return runtime;
}

bool mpsActive() { return gpuTestRuntime().mps_active; }

int ncclVersion() { return gpuTestRuntime().nccl_version; }

bool ncclSupportsMultiRankPerGpu() { return ncclVersion() >= 23000; }

bool ncclMultiRankGpuEnabled() { return gpuTestRuntime().nccl_multi_rank_gpu_enabled; }

TestSetupDecision checkGpuTestRequirements(const MpiTestComm& comm) {
  if (!comm.valid()) { return {true, false, "inactive MPI communicator"}; }

  return checkGpuSetupGlobal(comm, checkGpuTestRequirementsLocal(comm));
}

TestSetupDecision initializeGpuForTest(const MpiTestComm& comm, bool check_nccl) {
  if (!comm.valid()) { return {true, false, "inactive MPI communicator"}; }

  return checkGpuSetupGlobal(comm, initializeGpuForTestLocal(comm, check_nccl));
}

} // namespace cudecomp_test
