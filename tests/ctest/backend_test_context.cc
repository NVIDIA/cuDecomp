/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "backend_test_context.h"

#include <cstdlib>
#include <memory>
#include <string>
#include <utility>

#include "test_utils.h"

namespace cudecomp_test {
namespace {

constexpr const char* kKeepaliveBackendEnv = "CUDECOMP_TEST_KEEPALIVE_BACKEND";

class SharedBackendTestContextState {
public:
  SharedBackendTestContextState(std::string backend_label, int active_ranks, MpiTestComm active_comm,
                                cudecompHandle_t handle)
      : backend_label_(std::move(backend_label)), active_ranks_(active_ranks), active_comm_(std::move(active_comm)),
        handle_(handle), handle_guard_(handle) {}

  const MpiTestComm& comm() const { return active_comm_; }
  cudecompHandle_t handle() const { return handle_; }

  bool compatibleWith(const char* backend_label, int active_ranks) const {
    return backend_label_ == backend_label && active_ranks_ == active_ranks;
  }

  testing::AssertionResult ensureKeepaliveGridDesc(const cudecompGridDescConfig_t& config) {
    if (keepalive_grid_desc_guard_) return testing::AssertionSuccess();

    cudecompGridDescConfig_t keepalive_config = config;
    cudecompGridDesc_t grid_desc = nullptr;
    const cudecompResult_t result = cudecompGridDescCreate(handle_, &grid_desc, &keepalive_config, nullptr);
    keepalive_grid_desc_guard_ = std::make_unique<gridDescGuard>(handle_, grid_desc);
    return checkCudecompGlobal(active_comm_, result, __FILE__, __LINE__);
  }

private:
  std::string backend_label_;
  int active_ranks_ = 0;
  MpiTestComm active_comm_;
  cudecompHandle_t handle_ = nullptr;
  cudecompHandleGuard handle_guard_;
  std::unique_ptr<gridDescGuard> keepalive_grid_desc_guard_;
};

std::unique_ptr<SharedBackendTestContextState> shared_context;

std::string requestedKeepaliveBackend() {
  const char* value = std::getenv(kKeepaliveBackendEnv);
  if (!value || value[0] == '\0') return {};

  const std::string backend(value);
  if (backend == "nccl" || backend == "nvshmem") return backend;
  return {};
}

} // namespace

testing::AssertionResult BackendTestContext::initialize(const MpiTestComm& world_comm, int active_ranks,
                                                        const char* backend_label, bool check_nccl,
                                                        const cudecompGridDescConfig_t& config,
                                                        TestSetupDecision* setup_decision) {
  if (!setup_decision) { return testing::AssertionFailure() << "test setup decision argument cannot be null"; }

  *setup_decision = {};
  active_comm_ = nullptr;
  handle_ = nullptr;

  const std::string keepalive_backend = requestedKeepaliveBackend();
  const bool use_shared_context =
      !keepalive_backend.empty() && keepalive_backend == backend_label && active_ranks == world_comm.size();

  if (use_shared_context) {
    if (shared_context && !shared_context->compatibleWith(backend_label, active_ranks)) {
      resetSharedBackendTestContext();
    }

    if (!shared_context) {
      auto active_comm = MpiTestComm::split(world_comm, active_ranks);
      if (!active_comm.valid()) {
        *setup_decision = {true, false, "inactive rank for shared backend test context"};
        return testing::AssertionSuccess();
      }

      *setup_decision = initializeGpuForTest(active_comm, check_nccl);
      if (setup_decision->skip || setup_decision->fail) return testing::AssertionSuccess();

      cudecompHandle_t handle = nullptr;
      const cudecompResult_t init_result = cudecompInit(&handle, active_comm.mpiComm());
      auto state =
          std::make_unique<SharedBackendTestContextState>(backend_label, active_ranks, std::move(active_comm), handle);

      testing::AssertionResult init_status = checkCudecompGlobal(state->comm(), init_result, __FILE__, __LINE__);
      if (!init_status) return init_status;

      testing::AssertionResult keepalive_status = state->ensureKeepaliveGridDesc(config);
      if (!keepalive_status) return keepalive_status;

      shared_context = std::move(state);
    }

    active_comm_ = &shared_context->comm();
    handle_ = shared_context->handle();
    return testing::AssertionSuccess();
  }

  resetSharedBackendTestContext();

  local_active_comm_ = MpiTestComm::split(world_comm, active_ranks);
  if (!local_active_comm_.valid()) {
    *setup_decision = {true, false,
                       std::string("inactive rank for ") + std::to_string(active_ranks) + "-rank " + backend_label +
                           " test case"};
    return testing::AssertionSuccess();
  }

  *setup_decision = initializeGpuForTest(local_active_comm_, check_nccl);
  if (setup_decision->skip || setup_decision->fail) return testing::AssertionSuccess();

  const cudecompResult_t init_result = cudecompInit(&local_handle_, local_active_comm_.mpiComm());
  local_handle_guard_ = std::make_unique<cudecompHandleGuard>(local_handle_);
  testing::AssertionResult init_status = checkCudecompGlobal(local_active_comm_, init_result, __FILE__, __LINE__);
  if (!init_status) return init_status;

  active_comm_ = &local_active_comm_;
  handle_ = local_handle_;
  return testing::AssertionSuccess();
}

void resetSharedBackendTestContext() { shared_context.reset(); }

} // namespace cudecomp_test
