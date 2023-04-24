/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <algorithm>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>
#ifdef ENABLE_NVSHMEM
#include <nvshmem.h>
#include <nvshmemx.h>
#endif

#include "cudecomp.h"
#include "internal/autotune.h"
#include "internal/checks.h"
#include "internal/common.h"
#include "internal/exceptions.h"
#include "internal/halo.h"
#include "internal/transpose.h"

namespace cudecomp {
namespace {

// Static flag to disable multiple handle creation
static bool cudecomp_initialized = false;

static ncclComm_t ncclCommFromMPIComm(MPI_Comm mpi_comm) {
  int rank, nranks;
  CHECK_MPI(MPI_Comm_rank(mpi_comm, &rank));
  CHECK_MPI(MPI_Comm_size(mpi_comm, &nranks));

  ncclUniqueId id;
  if (rank == 0) CHECK_NCCL(ncclGetUniqueId(&id));
  CHECK_MPI(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, mpi_comm));
  ncclComm_t nccl_comm;
  CHECK_NCCL(ncclCommInitRank(&nccl_comm, nranks, id, rank));

  return nccl_comm;
}

static void checkTransposeCommBackend(cudecompTransposeCommBackend_t comm_backend) {
  switch (comm_backend) {
  case CUDECOMP_TRANSPOSE_COMM_NCCL:
  case CUDECOMP_TRANSPOSE_COMM_NCCL_PL:
  case CUDECOMP_TRANSPOSE_COMM_MPI_P2P:
  case CUDECOMP_TRANSPOSE_COMM_MPI_P2P_PL:
  case CUDECOMP_TRANSPOSE_COMM_MPI_A2A:
#ifdef ENABLE_NVSHMEM
  case CUDECOMP_TRANSPOSE_COMM_NVSHMEM:
  case CUDECOMP_TRANSPOSE_COMM_NVSHMEM_PL: return;
#else
    return;
  case CUDECOMP_TRANSPOSE_COMM_NVSHMEM:
  case CUDECOMP_TRANSPOSE_COMM_NVSHMEM_PL: THROW_NOT_SUPPORTED("transpose communication type unsupported");
#endif

  default: THROW_INVALID_USAGE("unknown transpose communication type");
  }
}

static void checkHaloCommBackend(cudecompHaloCommBackend_t comm_backend) {
  switch (comm_backend) {
  case CUDECOMP_HALO_COMM_NCCL:
  case CUDECOMP_HALO_COMM_MPI:
  case CUDECOMP_HALO_COMM_MPI_BLOCKING:
#ifdef ENABLE_NVSHMEM
  case CUDECOMP_HALO_COMM_NVSHMEM:
  case CUDECOMP_HALO_COMM_NVSHMEM_BLOCKING: return;
#else
    return;
  case CUDECOMP_HALO_COMM_NVSHMEM:
  case CUDECOMP_HALO_COMM_NVSHMEM_BLOCKING: THROW_NOT_SUPPORTED("halo communication type unsupported");
#endif
  default: THROW_INVALID_USAGE("unknown halo communication type");
  }
}

static void checkDataType(cudecompDataType_t dtype) {
  switch (dtype) {
  case CUDECOMP_FLOAT:
  case CUDECOMP_FLOAT_COMPLEX:
  case CUDECOMP_DOUBLE:
  case CUDECOMP_DOUBLE_COMPLEX: return;
  default: THROW_INVALID_USAGE("unknown data type");
  }
}

static void checkHandle(cudecompHandle_t handle) {
  if (!handle || !handle->initialized) { THROW_INVALID_USAGE("invalid handle"); }
}

static void checkGridDesc(cudecompGridDesc_t grid_desc) {
  if (!grid_desc || !grid_desc->initialized) { THROW_INVALID_USAGE("invalid grid descriptor"); }
}

static void checkConfig(cudecompHandle_t handle, const cudecompGridDescConfig_t* config, bool autotune_transpose,
                        bool autotune_halos) {
  if (!autotune_transpose) { checkTransposeCommBackend(config->transpose_comm_backend); }
  if (!autotune_halos) { checkHaloCommBackend(config->halo_comm_backend); }

  int pdims_prod = config->pdims[0] * config->pdims[1];
  if (pdims_prod == 0) {
    if (config->pdims[0] != 0 || config->pdims[1] != 0) { THROW_INVALID_USAGE("pdims values are invalid"); }
  } else if (pdims_prod != handle->nranks) {
    THROW_INVALID_USAGE("product of pdims values must equal number of ranks");
  }
}

static void gatherGlobalMPIInfo(cudecompHandle_t& handle) {
  handle->hostnames.resize(handle->nranks);
  int resultlen;
  CHECK_MPI(MPI_Get_processor_name(handle->hostnames[handle->rank].data(), &resultlen));
  CHECK_MPI(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, handle->hostnames.data(), MPI_MAX_PROCESSOR_NAME,
                          MPI_CHAR, handle->mpi_comm));

  handle->rank_to_local_rank.resize(handle->nranks);
  handle->rank_to_local_rank[handle->rank] = handle->local_rank;
  CHECK_MPI(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, handle->rank_to_local_rank.data(), 1, MPI_INT,
                          handle->mpi_comm));
}

#ifdef ENABLE_NVSHMEM
static void inspectNvshmemEnvVars(cudecompHandle_t& handle) {
  // Check NVSHMEM_DISABLE_CUDA_VMM
  handle->nvshmem_vmm = true;
  char* vmm_str = std::getenv("NVSHMEM_DISABLE_CUDA_VMM");
  if (vmm_str) { handle->nvshmem_vmm = std::strtol(vmm_str, nullptr, 10) == 0; }

  if (handle->rank == 0 && handle->nvshmem_vmm) {
    printf("CUDECOMP:WARN: NVSHMEM_DISABLE_CUDA_VMM is unset. We currently recommend setting it "
           "(i.e. NVSHMEM_DISABLE_CUDA_VMM=1) for best compatibility with MPI libraries. See the documentation "
           "for more details.\n");
  }

  // Check NVSHMEM_SYMMETRIC_SIZE
  char* symmetric_size_str = std::getenv("NVSHMEM_SYMMETRIC_SIZE");
  if (symmetric_size_str) {
    int scale;
    if (std::strchr(symmetric_size_str, 'k') || std::strchr(symmetric_size_str, 'K')) {
      scale = 10;
    } else if (std::strchr(symmetric_size_str, 'm') || std::strchr(symmetric_size_str, 'M')) {
      scale = 20;
    } else if (std::strchr(symmetric_size_str, 'g') || std::strchr(symmetric_size_str, 'G')) {
      scale = 30;
    } else if (std::strchr(symmetric_size_str, 't') || std::strchr(symmetric_size_str, 'T')) {
      scale = 40;
    } else {
      scale = 0;
    }
    handle->nvshmem_symmetric_size = std::ceil(std::strtod(symmetric_size_str, nullptr) * (1ull << scale));
  } else {
    // NVSHMEM symmetric size defaults to 1 GiB
    handle->nvshmem_symmetric_size = 1ull << 30;
  }
}
#endif

} // namespace
} // namespace cudecomp

cudecompResult_t cudecompInit(cudecompHandle_t* handle_in, MPI_Comm mpi_comm) {
  using namespace cudecomp;
  cudecompHandle_t handle = nullptr;
  try {
    if (cudecomp_initialized) {
      THROW_INVALID_USAGE("cuDecomp already initialized and multiple handles are not supported.");
    }
    handle = new cudecompHandle;
    handle->mpi_comm = mpi_comm;
    CHECK_MPI(MPI_Comm_rank(mpi_comm, &handle->rank));
    CHECK_MPI(MPI_Comm_size(mpi_comm, &handle->nranks));

    CHECK_MPI(MPI_Comm_split_type(handle->mpi_comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &handle->mpi_local_comm));
    CHECK_MPI(MPI_Comm_rank(handle->mpi_local_comm, &handle->local_rank));
    CHECK_MPI(MPI_Comm_size(handle->mpi_local_comm, &handle->local_nranks));

    // Initialize cuTENSOR library
    CHECK_CUTENSOR(cutensorInit(&handle->cutensor_handle));

    // Gather extra MPI info from all communicator ranks
    gatherGlobalMPIInfo(handle);

    handle->initialized = true;
    cudecomp_initialized = true;

    *handle_in = handle;

  } catch (const BaseException& e) {
    if (handle) { delete handle; }
    std::cerr << e.what();
    return e.getResult();
  }
  return CUDECOMP_RESULT_SUCCESS;
};

cudecompResult_t cudecompFinalize(cudecompHandle_t handle) {
  using namespace cudecomp;
  try {
    checkHandle(handle);

    if (handle->nccl_comm) { CHECK_NCCL(ncclCommDestroy(handle->nccl_comm)); }
    if (handle->nccl_local_comm) { CHECK_NCCL(ncclCommDestroy(handle->nccl_local_comm)); }
    if (handle->pl_stream) { CHECK_CUDA(cudaStreamDestroy(handle->pl_stream)); }
#ifdef ENABLE_NVSHMEM
    if (handle->nvshmem_initialized) {
      nvshmem_finalize();
      handle->nvshmem_initialized = false;
      handle->nvshmem_allocations.clear();
      handle->nvshmem_allocation_size = 0;
    }
#endif
    CHECK_MPI(MPI_Comm_free(&handle->mpi_local_comm));

    handle = nullptr;
    delete handle;

    cudecomp_initialized = false;

  } catch (const cudecomp::BaseException& e) {
    std::cerr << e.what();
    return e.getResult();
  }
  return CUDECOMP_RESULT_SUCCESS;
}

cudecompResult_t cudecompGridDescCreate(cudecompHandle_t handle, cudecompGridDesc_t* grid_desc_in,
                                        cudecompGridDescConfig_t* config,
                                        const cudecompGridDescAutotuneOptions_t* options) {

  using namespace cudecomp;
  cudecompGridDesc_t grid_desc = nullptr;
  try {
    checkHandle(handle);
    if (!config) { THROW_INVALID_USAGE("config argument cannot be null"); }

    // Check some autotuning options
    bool autotune_transpose_backend = false;
    bool autotune_halo_backend = false;
    bool autotune_disable_nccl_backends = false;
    bool autotune_disable_nvshmem_backends = false;
    if (options) {
      autotune_transpose_backend = options->autotune_transpose_backend;
      autotune_halo_backend = options->autotune_halo_backend;
      autotune_disable_nccl_backends = options->disable_nccl_backends;
      autotune_disable_nvshmem_backends = options->disable_nvshmem_backends;
    }

    checkConfig(handle, config, autotune_transpose_backend, autotune_halo_backend);

    bool autotune_pdims = (config->pdims[0] == 0 && config->pdims[1] == 0);
    if (autotune_pdims && !options) { THROW_INVALID_USAGE("options argument cannot be null if autotuning pdims"); }

    grid_desc = new cudecompGridDesc;
    grid_desc->initialized = true;
    grid_desc->config = *config;
    auto comm_backend = grid_desc->config.transpose_comm_backend;
    auto halo_comm_backend = grid_desc->config.halo_comm_backend;

    grid_desc->config.transpose_axis_contiguous[0] = false; // For x-axis, always set to false.

    if (grid_desc->config.gdims_dist[0] > grid_desc->config.gdims[0] ||
        grid_desc->config.gdims_dist[1] > grid_desc->config.gdims[1] ||
        grid_desc->config.gdims_dist[2] > grid_desc->config.gdims[2]) {
      THROW_INVALID_USAGE("gdims_dist entries must be less than or equal to gdims entries");
    }
    if (grid_desc->config.gdims_dist[0] != 0 && grid_desc->config.gdims_dist[1] != 0 &&
        grid_desc->config.gdims_dist[2] != 0) {
      grid_desc->config.gdims_dist[0] = grid_desc->config.gdims_dist[0];
      grid_desc->config.gdims_dist[1] = grid_desc->config.gdims_dist[1];
      grid_desc->config.gdims_dist[2] = grid_desc->config.gdims_dist[2];
      grid_desc->gdims_dist_set = true;
    } else {
      grid_desc->config.gdims_dist[0] = grid_desc->config.gdims[0];
      grid_desc->config.gdims_dist[1] = grid_desc->config.gdims[1];
      grid_desc->config.gdims_dist[2] = grid_desc->config.gdims[2];
    }

    // Initialize NCCL communicator if needed
    if (transposeBackendRequiresNccl(comm_backend) || haloBackendRequiresNccl(halo_comm_backend) ||
        ((autotune_transpose_backend || autotune_halo_backend) && !autotune_disable_nccl_backends)) {
      if (!handle->nccl_comm) { handle->nccl_comm = ncclCommFromMPIComm(handle->mpi_comm); }
      if (!handle->nccl_local_comm) { handle->nccl_local_comm = ncclCommFromMPIComm(handle->mpi_local_comm); }
      if (!handle->pl_stream) {
        int greatest_priority;
        CHECK_CUDA(cudaDeviceGetStreamPriorityRange(nullptr, &greatest_priority));
        CHECK_CUDA(cudaStreamCreateWithPriority(&handle->pl_stream, cudaStreamNonBlocking, greatest_priority));
      }
    }

    // Initialize NVSHMEM if needed
    if (transposeBackendRequiresNvshmem(comm_backend) || haloBackendRequiresNvshmem(halo_comm_backend) ||
        ((autotune_transpose_backend || autotune_halo_backend) && !autotune_disable_nvshmem_backends)) {
#ifdef ENABLE_NVSHMEM
      if (!handle->nvshmem_initialized) {
        inspectNvshmemEnvVars(handle);
        nvshmemx_init_attr_t attr;
        attr.mpi_comm = &handle->mpi_comm;
        nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
        handle->nvshmem_initialized = true;
        handle->nvshmem_allocation_size = 0;
      }
      if (!handle->pl_stream) {
        int greatest_priority;
        CHECK_CUDA(cudaDeviceGetStreamPriorityRange(nullptr, &greatest_priority));
        CHECK_CUDA(cudaStreamCreateWithPriority(&handle->pl_stream, cudaStreamNonBlocking, greatest_priority));
      }
#endif
    }

    // Create CUDA events for scheduling
    grid_desc->events.resize(handle->nranks);
    for (auto& event : grid_desc->events) { CHECK_CUDA(cudaEventCreateWithFlags(&event, cudaEventDisableTiming)); }

    // Disable decompositions with empty pencils
    if (!autotune_pdims &&
        (grid_desc->config.pdims[0] > std::min(grid_desc->config.gdims_dist[0], grid_desc->config.gdims_dist[1]) ||
         grid_desc->config.pdims[1] > std::min(grid_desc->config.gdims_dist[1], grid_desc->config.gdims_dist[2]))) {
      THROW_NOT_SUPPORTED("grid descriptor settings yields a distribution with empty pencils");
    }

    // Run autotuning if requested
    if (options) {
      if (options->grid_mode == CUDECOMP_AUTOTUNE_GRID_TRANSPOSE) {
        if (autotune_transpose_backend || autotune_pdims) { autotuneTransposeBackend(handle, grid_desc, options); }
        if (autotune_halo_backend) { autotuneHaloBackend(handle, grid_desc, options); }
      } else if (options->grid_mode == CUDECOMP_AUTOTUNE_GRID_HALO) {
        if (autotune_halo_backend || autotune_pdims) { autotuneHaloBackend(handle, grid_desc, options); }
        if (autotune_transpose_backend) { autotuneTransposeBackend(handle, grid_desc, options); }
      } else {
        THROW_INVALID_USAGE("unknown value of autotune_grid_mode encountered.");
      }
    }

    if (grid_desc->config.pdims[0] == 0 || grid_desc->config.pdims[1] == 0) {
      THROW_NOT_SUPPORTED("No valid decomposition found during autotuning with provided arguments.");
    }

    grid_desc->pidx[0] = handle->rank / grid_desc->config.pdims[1];
    grid_desc->pidx[1] = handle->rank % grid_desc->config.pdims[1];

    // Setup final row and column communicators
    int color_row = grid_desc->pidx[0];
    MPI_Comm row_comm;
    CHECK_MPI(MPI_Comm_split(handle->mpi_comm, color_row, handle->rank, &row_comm));
    setCommInfo(handle, grid_desc, row_comm, CUDECOMP_COMM_ROW);

    int color_col = grid_desc->pidx[1];
    MPI_Comm col_comm;
    CHECK_MPI(MPI_Comm_split(handle->mpi_comm, color_col, handle->rank, &col_comm));
    setCommInfo(handle, grid_desc, col_comm, CUDECOMP_COMM_COL);
#ifdef ENABLE_NVSHMEM
    if (transposeBackendRequiresNvshmem(grid_desc->config.transpose_comm_backend) ||
        haloBackendRequiresNvshmem(grid_desc->config.halo_comm_backend)) {
      nvshmem_team_config_t tmp;
      nvshmem_team_split_2d(NVSHMEM_TEAM_WORLD, grid_desc->config.pdims[1], &tmp, 0,
                            &grid_desc->row_comm_info.nvshmem_team, &tmp, 0, &grid_desc->col_comm_info.nvshmem_team);
      handle->n_grid_descs_using_nvshmem++;
    } else {
      // Finalize nvshmem to reclaim symmetric heap memory if not used
      if (handle->nvshmem_initialized && handle->n_grid_descs_using_nvshmem == 0) {
        nvshmem_finalize();
        handle->nvshmem_initialized = false;
        handle->nvshmem_allocations.clear();
        handle->nvshmem_allocation_size = 0;
      }
    }
#endif
    if (transposeBackendRequiresNccl(grid_desc->config.transpose_comm_backend) ||
        haloBackendRequiresNccl(grid_desc->config.halo_comm_backend)) {
      handle->n_grid_descs_using_nccl++;
    } else {
      // Destroy NCCL communicator to reclaim resources if not used
      if (handle->nccl_comm && handle->nccl_local_comm && handle->n_grid_descs_using_nccl == 0) {
        CHECK_NCCL(ncclCommDestroy(handle->nccl_comm));
        handle->nccl_comm = nullptr;
        CHECK_NCCL(ncclCommDestroy(handle->nccl_local_comm));
        handle->nccl_local_comm = nullptr;
      }
    }

    *grid_desc_in = grid_desc;
    *config = grid_desc->config;
    // If gdims_dist was not set, return config with default values
    if (!grid_desc->gdims_dist_set) {
      config->gdims_dist[0] = 0;
      config->gdims_dist[1] = 0;
      config->gdims_dist[2] = 0;
    }

  } catch (const cudecomp::BaseException& e) {
    if (grid_desc) { delete grid_desc; }
    std::cerr << e.what();
    return e.getResult();
  }
  return CUDECOMP_RESULT_SUCCESS;
}

cudecompResult_t cudecompGridDescDestroy(cudecompHandle_t handle, cudecompGridDesc_t grid_desc) {
  using namespace cudecomp;
  try {
    checkGridDesc(grid_desc);

    if (grid_desc->row_comm_info.mpi_comm != MPI_COMM_NULL) {
      CHECK_MPI(MPI_Comm_free(&grid_desc->row_comm_info.mpi_comm));
    }
    if (grid_desc->col_comm_info.mpi_comm != MPI_COMM_NULL) {
      CHECK_MPI(MPI_Comm_free(&grid_desc->col_comm_info.mpi_comm));
    }

    for (auto e : grid_desc->events) {
      if (e) { CHECK_CUDA(cudaEventDestroy(e)); }
    }

    if (transposeBackendRequiresNccl(grid_desc->config.transpose_comm_backend) ||
        haloBackendRequiresNccl(grid_desc->config.halo_comm_backend)) {
      handle->n_grid_descs_using_nccl--;

      // Destroy NCCL communicator to reclaim resources if not used
      if (handle->nccl_comm && handle->nccl_local_comm && handle->n_grid_descs_using_nccl == 0) {
        CHECK_NCCL(ncclCommDestroy(handle->nccl_comm));
        handle->nccl_comm = nullptr;
        CHECK_NCCL(ncclCommDestroy(handle->nccl_local_comm));
        handle->nccl_local_comm = nullptr;
      }
    }

#ifdef ENABLE_NVSHMEM
    if (transposeBackendRequiresNvshmem(grid_desc->config.transpose_comm_backend) ||
        haloBackendRequiresNvshmem(grid_desc->config.halo_comm_backend)) {
      if (grid_desc->row_comm_info.nvshmem_team != NVSHMEM_TEAM_INVALID) {
        nvshmem_team_destroy(grid_desc->row_comm_info.nvshmem_team);
      }
      if (grid_desc->col_comm_info.nvshmem_team != NVSHMEM_TEAM_INVALID) {
        nvshmem_team_destroy(grid_desc->col_comm_info.nvshmem_team);
      }
      handle->n_grid_descs_using_nvshmem--;

      // Finalize nvshmem to reclaim symmetric heap memory if not used
      if (handle->nvshmem_initialized && handle->n_grid_descs_using_nvshmem == 0) {
        nvshmem_finalize();
        handle->nvshmem_initialized = false;
        handle->nvshmem_allocations.clear();
        handle->nvshmem_allocation_size = 0;
      }
    }
#endif

    delete grid_desc;
    grid_desc = nullptr;

  } catch (const cudecomp::BaseException& e) {
    std::cerr << e.what();
    return e.getResult();
  }
  return CUDECOMP_RESULT_SUCCESS;
}

cudecompResult_t cudecompGridDescConfigSetDefaults(cudecompGridDescConfig_t* config) {
  using namespace cudecomp;
  try {
    if (!config) { THROW_INVALID_USAGE("config argument cannot be null"); }

    // Grid Information
    for (int i = 0; i < 2; ++i) { config->pdims[i] = 0; }
    for (int i = 0; i < 3; ++i) {
      config->gdims[i] = 0;
      config->gdims_dist[i] = 0;
    }

    // Transpose Options
    config->transpose_comm_backend = CUDECOMP_TRANSPOSE_COMM_MPI_P2P;
    for (int i = 0; i < 3; ++i) { config->transpose_axis_contiguous[i] = false; }

    // Halo Options
    config->halo_comm_backend = CUDECOMP_HALO_COMM_MPI;

  } catch (const cudecomp::BaseException& e) {
    std::cerr << e.what();
    return e.getResult();
  }
  return CUDECOMP_RESULT_SUCCESS;
}

cudecompResult_t cudecompGridDescAutotuneOptionsSetDefaults(cudecompGridDescAutotuneOptions_t* options) {
  using namespace cudecomp;
  try {
    if (!options) { THROW_INVALID_USAGE("options argument cannot be null"); }

    // General options
    options->grid_mode = CUDECOMP_AUTOTUNE_GRID_TRANSPOSE;
    options->dtype = CUDECOMP_DOUBLE;
    options->allow_uneven_decompositions = true;
    options->disable_nccl_backends = false;
    options->disable_nvshmem_backends = false;

    // Transpose-specific options
    options->autotune_transpose_backend = false;
    options->transpose_use_inplace_buffers = false;

    // Halo-specific options
    options->autotune_halo_backend = false;
    options->halo_axis = 0;
    for (int i = 0; i < 3; ++i) {
      options->halo_extents[i] = 0;
      options->halo_periods[i] = false;
    }

  } catch (const cudecomp::BaseException& e) {
    std::cerr << e.what();
    return e.getResult();
  }
  return CUDECOMP_RESULT_SUCCESS;
}

cudecompResult_t cudecompGetPencilInfo(cudecompHandle_t handle, cudecompGridDesc_t grid_desc,
                                       cudecompPencilInfo_t* pencil_info, int32_t axis, const int32_t halo_extents[]) {
  using namespace cudecomp;
  try {
    checkHandle(handle);
    checkGridDesc(grid_desc);
    if (!pencil_info) { THROW_INVALID_USAGE("pencil_info argument cannot be null."); }
    if (axis < 0 || axis > 2) { THROW_INVALID_USAGE("axis argument out of range"); }

    std::array<int32_t, 3> invorder;

    // Setup order (and inverse)
    for (int i = 0; i < 3; ++i) {
      if (grid_desc->config.transpose_axis_contiguous[axis]) {
        pencil_info->order[i] = (axis + i) % 3;
      } else {
        pencil_info->order[i] = i;
      }
      invorder[pencil_info->order[i]] = i;
    }

    int j = 0;
    pencil_info->size = 1;
    for (int i = 0; i < 3; ++i) {
      int ord = invorder[i];
      if (i != axis) {
        int64_t d = grid_desc->config.gdims_dist[i] / grid_desc->config.pdims[j];
        int64_t mod = grid_desc->config.gdims_dist[i] % grid_desc->config.pdims[j];
        pencil_info->shape[ord] = d;
        pencil_info->shape[ord] += (grid_desc->pidx[j] < mod) ? 1 : 0;
        if (grid_desc->pidx[j] == std::min(grid_desc->config.pdims[j], grid_desc->config.gdims_dist[i]) - 1) {
          // Tack any difference in gdim and gdims_dist to last pencil in gdims_dist decomposition
          pencil_info->shape[ord] += (grid_desc->config.gdims[i] - grid_desc->config.gdims_dist[i]);
        }
        pencil_info->lo[ord] = (grid_desc->pidx[j] * d + std::min((int64_t)grid_desc->pidx[j], mod));
        j++;
      } else {
        pencil_info->shape[ord] = grid_desc->config.gdims[i];
        pencil_info->lo[ord] = 0;
      }
      pencil_info->hi[ord] = pencil_info->lo[ord] + pencil_info->shape[ord] - 1;

      if (halo_extents) {
        pencil_info->shape[ord] += 2 * halo_extents[i];
        pencil_info->halo_extents[i] = halo_extents[i];
      } else {
        pencil_info->halo_extents[i] = 0;
      }
      pencil_info->size *= pencil_info->shape[ord];
    }

  } catch (const cudecomp::BaseException& e) {
    std::cerr << e.what();
    return e.getResult();
  }
  return CUDECOMP_RESULT_SUCCESS;
}

cudecompResult_t cudecompGetGridDescConfig(cudecompHandle_t handle, cudecompGridDesc_t grid_desc,
                                           cudecompGridDescConfig_t* config) {
  using namespace cudecomp;
  try {
    checkHandle(handle);
    checkGridDesc(grid_desc);
    if (!config) { THROW_INVALID_USAGE("config argument cannot be null."); }

    *config = grid_desc->config;
    // If gdims_dist was not set, return config with default values
    if (!grid_desc->gdims_dist_set) {
      config->gdims_dist[0] = 0;
      config->gdims_dist[1] = 0;
      config->gdims_dist[2] = 0;
    }

  } catch (const cudecomp::BaseException& e) {
    std::cerr << e.what();
    return e.getResult();
  }
  return CUDECOMP_RESULT_SUCCESS;
}

cudecompResult_t cudecompGetTransposeWorkspaceSize(cudecompHandle_t handle, cudecompGridDesc_t grid_desc,
                                                   int64_t* workspace_size) {
  using namespace cudecomp;
  try {
    checkHandle(handle);
    checkGridDesc(grid_desc);
    if (!workspace_size) { THROW_INVALID_USAGE("workspace_size argument cannot be null."); }
    int64_t max_pencil_size_x = getGlobalMaxPencilSize(handle, grid_desc, 0);
    int64_t max_pencil_size_y = getGlobalMaxPencilSize(handle, grid_desc, 1);
    int64_t max_pencil_size_z = getGlobalMaxPencilSize(handle, grid_desc, 2);
    *workspace_size = std::max(max_pencil_size_x + max_pencil_size_y, max_pencil_size_y + max_pencil_size_z);

  } catch (const cudecomp::BaseException& e) {
    std::cerr << e.what();
    return e.getResult();
  }
  return CUDECOMP_RESULT_SUCCESS;
}

cudecompResult_t cudecompGetHaloWorkspaceSize(cudecompHandle_t handle, cudecompGridDesc_t grid_desc, int32_t axis,
                                              const int32_t halo_extents[], int64_t* workspace_size) {
  using namespace cudecomp;
  try {
    checkHandle(handle);
    checkGridDesc(grid_desc);
    if (axis < 0 || axis > 2) { THROW_INVALID_USAGE("axis argument out of range"); }
    if (!halo_extents) { THROW_INVALID_USAGE("halo_extents argument cannot be null."); }
    if (!workspace_size) { THROW_INVALID_USAGE("workspace_size argument cannot be null."); }
    cudecompPencilInfo_t pinfo;
    CHECK_CUDECOMP(cudecompGetPencilInfo(handle, grid_desc, &pinfo, axis, halo_extents));
    auto shape_g = getShapeG(pinfo);
    size_t halo_size_x = 4 * shape_g[1] * shape_g[2] * pinfo.halo_extents[0];
    size_t halo_size_y = 4 * shape_g[0] * shape_g[2] * pinfo.halo_extents[1];
    size_t halo_size_z = 4 * shape_g[0] * shape_g[1] * pinfo.halo_extents[2];

    *workspace_size = std::max(halo_size_x, std::max(halo_size_y, halo_size_z));
  } catch (const cudecomp::BaseException& e) {
    std::cerr << e.what();
    return e.getResult();
  }
  return CUDECOMP_RESULT_SUCCESS;
}

cudecompResult_t cudecompMalloc(cudecompHandle_t handle, cudecompGridDesc_t grid_desc, void** buffer,
                                size_t buffer_size_bytes) {
  using namespace cudecomp;
  try {
    checkHandle(handle);
    checkGridDesc(grid_desc);

    if (transposeBackendRequiresNvshmem(grid_desc->config.transpose_comm_backend) ||
        haloBackendRequiresNvshmem(grid_desc->config.halo_comm_backend)) {
#ifdef ENABLE_NVSHMEM
      // NVSHMEM requires allocations to be the same size for all ranks. Find maximum.
      CHECK_MPI(MPI_Allreduce(MPI_IN_PLACE, &buffer_size_bytes, 1, MPI_LONG_LONG_INT, MPI_MAX, handle->mpi_comm));

      size_t nvshmem_free_size = handle->nvshmem_symmetric_size - handle->nvshmem_allocation_size;
      if (!handle->nvshmem_vmm && handle->rank == 0 && buffer_size_bytes > nvshmem_free_size) {
        fprintf(stderr,
                "CUDECOMP:WARN: Attempting an NVSHMEM allocation of %lld bytes but *approximately* "
                "%zu free bytes of %zu total bytes of symmetric heap space available. If the allocation fails, "
                "set NVSHMEM_SYMMETRIC_SIZE >= %zu and try again.\n",
                buffer_size_bytes, nvshmem_free_size, handle->nvshmem_symmetric_size,
                handle->nvshmem_symmetric_size + (buffer_size_bytes - nvshmem_free_size));
      }

      *buffer = nvshmem_malloc(buffer_size_bytes);
      if (buffer_size_bytes != 0 && *buffer == nullptr) { THROW_NVSHMEM_ERROR("nvshmem_malloc failed"); }
      // Record NVSHMEM allocation details
      handle->nvshmem_allocations[*buffer] = buffer_size_bytes;
      handle->nvshmem_allocation_size += buffer_size_bytes;
#else
      THROW_NOT_SUPPORTED("build does not support NVSHMEM communication backends.");
#endif
    } else {
      CHECK_CUDA(cudaMalloc(buffer, buffer_size_bytes));
    }

  } catch (const cudecomp::BaseException& e) {
    std::cerr << e.what();
    return e.getResult();
  }
  return CUDECOMP_RESULT_SUCCESS;
}

cudecompResult_t cudecompFree(cudecompHandle_t handle, cudecompGridDesc_t grid_desc, void* buffer) {
  using namespace cudecomp;
  try {
    checkHandle(handle);
    checkGridDesc(grid_desc);

    if (transposeBackendRequiresNvshmem(grid_desc->config.transpose_comm_backend) ||
        haloBackendRequiresNvshmem(grid_desc->config.halo_comm_backend)) {
#ifdef ENABLE_NVSHMEM
      if (buffer) {
        nvshmem_free(buffer);

        // Record NVSHMEM deallocation details
        size_t buffer_size_bytes = handle->nvshmem_allocations[buffer];
        handle->nvshmem_allocation_size -= buffer_size_bytes;
        handle->nvshmem_allocations.erase(buffer);
      }
#else
      THROW_NOT_SUPPORTED("build does not support NVSHMEM communication backends.");
#endif

    } else {
      if (buffer) { CHECK_CUDA(cudaFree(buffer)); }
    }
  } catch (const cudecomp::BaseException& e) {
    std::cerr << e.what();
    return e.getResult();
  }
  return CUDECOMP_RESULT_SUCCESS;
}

const char* cudecompTransposeCommBackendToString(cudecompTransposeCommBackend_t comm_backend) {
  switch (comm_backend) {
  case CUDECOMP_TRANSPOSE_COMM_NCCL: return "NCCL";
  case CUDECOMP_TRANSPOSE_COMM_NCCL_PL: return "NCCL (pipelined)";
  case CUDECOMP_TRANSPOSE_COMM_MPI_P2P: return "MPI_P2P";
  case CUDECOMP_TRANSPOSE_COMM_MPI_P2P_PL: return "MPI_P2P (pipelined)";
  case CUDECOMP_TRANSPOSE_COMM_MPI_A2A: return "MPI_A2A";
  case CUDECOMP_TRANSPOSE_COMM_NVSHMEM: return "NVSHMEM";
  case CUDECOMP_TRANSPOSE_COMM_NVSHMEM_PL: return "NVSHMEM (pipelined)";
  default: return "ERROR";
  }
}

const char* cudecompHaloCommBackendToString(cudecompHaloCommBackend_t comm_backend) {
  switch (comm_backend) {
  case CUDECOMP_HALO_COMM_NCCL: return "NCCL";
  case CUDECOMP_HALO_COMM_MPI: return "MPI";
  case CUDECOMP_HALO_COMM_MPI_BLOCKING: return "MPI (blocking)";
  case CUDECOMP_HALO_COMM_NVSHMEM: return "NVSHMEM";
  case CUDECOMP_HALO_COMM_NVSHMEM_BLOCKING: return "NVSHMEM (blocking)";
  default: return "ERROR";
  }
}

cudecompResult_t cudecompGetDataTypeSize(cudecompDataType_t dtype, int64_t* dtype_size) {
  using namespace cudecomp;
  try {
    checkDataType(dtype);
    if (!dtype_size) { THROW_INVALID_USAGE("dtype_size cannot be null."); }
    switch (dtype) {
    case CUDECOMP_FLOAT: *dtype_size = 4; break;
    case CUDECOMP_DOUBLE:
    case CUDECOMP_FLOAT_COMPLEX: *dtype_size = 8; break;
    case CUDECOMP_DOUBLE_COMPLEX: *dtype_size = 16; break;
    }

  } catch (const cudecomp::BaseException& e) {
    std::cerr << e.what();
    return e.getResult();
  }
  return CUDECOMP_RESULT_SUCCESS;
}

cudecompResult_t cudecompGetShiftedRank(cudecompHandle_t handle, cudecompGridDesc_t grid_desc, int32_t axis,
                                        int32_t dim, int32_t displacement, bool periodic, int32_t* shifted_rank) {
  using namespace cudecomp;

  try {
    checkHandle(handle);
    checkGridDesc(grid_desc);
    if (axis < 0 || axis > 2) { THROW_INVALID_USAGE("axis argument out of range"); }
    if (dim < 0 || dim > 2) { THROW_INVALID_USAGE("dim argument out of range"); }
    if (!shifted_rank) { THROW_INVALID_USAGE("shifted_rank argument cannot be null."); }

    if (dim == axis) {
      *shifted_rank = periodic ? handle->rank : -1;
      return CUDECOMP_RESULT_SUCCESS;
    }

    cudecompPencilInfo_t pinfo;
    CHECK_CUDECOMP(cudecompGetPencilInfo(handle, grid_desc, &pinfo, axis, nullptr));

    int count = 0;
    for (int i = 0; i < 3; ++i) {
      if (i == axis) continue;
      if (i == dim) break;
      count++;
    }

    auto comm_axis = (count == 0) ? CUDECOMP_COMM_COL : CUDECOMP_COMM_ROW;
    int comm_rank = (comm_axis == CUDECOMP_COMM_COL) ? grid_desc->col_comm_info.rank : grid_desc->row_comm_info.rank;
    int shifted = comm_rank + displacement;

    if (!periodic && (shifted < 0 || shifted >= grid_desc->config.pdims[comm_axis])) {
      *shifted_rank = -1; // "null" case
    } else {
      int comm_peer = (shifted + grid_desc->config.pdims[comm_axis]) % grid_desc->config.pdims[comm_axis];
      int global_peer = getGlobalRank(grid_desc, comm_axis, comm_peer);
      *shifted_rank = global_peer;
    }

  } catch (const cudecomp::BaseException& e) {
    std::cerr << e.what();
    return e.getResult();
  }
  return CUDECOMP_RESULT_SUCCESS;
}

cudecompResult_t cudecompTransposeXToY(cudecompHandle_t handle, cudecompGridDesc_t grid_desc, void* input, void* output,
                                       void* work, cudecompDataType_t dtype, const int32_t input_halo_extents[],
                                       const int32_t output_halo_extents[], cudaStream_t stream) {
  using namespace cudecomp;
  try {
    checkHandle(handle);
    checkGridDesc(grid_desc);
    checkDataType(dtype);
    if (!input) { THROW_INVALID_USAGE("input argument cannot be null"); }
    if (!output) { THROW_INVALID_USAGE("output argument cannot be null"); }
    if (!work) { THROW_INVALID_USAGE("work argument cannot be null"); }
    switch (dtype) {
    case CUDECOMP_FLOAT:
      cudecompTransposeXToY(handle, grid_desc, reinterpret_cast<float*>(input), reinterpret_cast<float*>(output),
                            reinterpret_cast<float*>(work), input_halo_extents, output_halo_extents, stream);
      break;
    case CUDECOMP_DOUBLE:
      cudecompTransposeXToY(handle, grid_desc, reinterpret_cast<double*>(input), reinterpret_cast<double*>(output),
                            reinterpret_cast<double*>(work), input_halo_extents, output_halo_extents, stream);
      break;
    case CUDECOMP_FLOAT_COMPLEX:
      cudecompTransposeXToY(handle, grid_desc, reinterpret_cast<cuda::std::complex<float>*>(input),
                            reinterpret_cast<cuda::std::complex<float>*>(output),
                            reinterpret_cast<cuda::std::complex<float>*>(work), input_halo_extents, output_halo_extents,
                            stream);
      break;
    case CUDECOMP_DOUBLE_COMPLEX:
      cudecompTransposeXToY(handle, grid_desc, reinterpret_cast<cuda::std::complex<double>*>(input),
                            reinterpret_cast<cuda::std::complex<double>*>(output),
                            reinterpret_cast<cuda::std::complex<double>*>(work), input_halo_extents,
                            output_halo_extents, stream);
      break;
    }
  } catch (const cudecomp::BaseException& e) {
    std::cerr << e.what();
    return e.getResult();
  }
  return CUDECOMP_RESULT_SUCCESS;
}

cudecompResult_t cudecompTransposeYToZ(cudecompHandle_t handle, cudecompGridDesc_t grid_desc, void* input, void* output,
                                       void* work, cudecompDataType_t dtype, const int32_t input_halo_extents[],
                                       const int32_t output_halo_extents[], cudaStream_t stream) {
  using namespace cudecomp;
  try {
    checkHandle(handle);
    checkGridDesc(grid_desc);
    checkDataType(dtype);
    if (!input) { THROW_INVALID_USAGE("input argument cannot be null"); }
    if (!output) { THROW_INVALID_USAGE("output argument cannot be null"); }
    if (!work) { THROW_INVALID_USAGE("work argument cannot be null"); }
    switch (dtype) {
    case CUDECOMP_FLOAT:
      cudecompTransposeYToZ(handle, grid_desc, reinterpret_cast<float*>(input), reinterpret_cast<float*>(output),
                            reinterpret_cast<float*>(work), input_halo_extents, output_halo_extents, stream);
      break;
    case CUDECOMP_DOUBLE:
      cudecompTransposeYToZ(handle, grid_desc, reinterpret_cast<double*>(input), reinterpret_cast<double*>(output),
                            reinterpret_cast<double*>(work), input_halo_extents, output_halo_extents, stream);
      break;
    case CUDECOMP_FLOAT_COMPLEX:
      cudecompTransposeYToZ(handle, grid_desc, reinterpret_cast<cuda::std::complex<float>*>(input),
                            reinterpret_cast<cuda::std::complex<float>*>(output),
                            reinterpret_cast<cuda::std::complex<float>*>(work), input_halo_extents, output_halo_extents,
                            stream);
      break;
    case CUDECOMP_DOUBLE_COMPLEX:
      cudecompTransposeYToZ(handle, grid_desc, reinterpret_cast<cuda::std::complex<double>*>(input),
                            reinterpret_cast<cuda::std::complex<double>*>(output),
                            reinterpret_cast<cuda::std::complex<double>*>(work), input_halo_extents,
                            output_halo_extents, stream);
      break;
    }

  } catch (const cudecomp::BaseException& e) {
    std::cerr << e.what();
    return e.getResult();
  }
  return CUDECOMP_RESULT_SUCCESS;
}

cudecompResult_t cudecompTransposeZToY(cudecompHandle_t handle, cudecompGridDesc_t grid_desc, void* input, void* output,
                                       void* work, cudecompDataType_t dtype, const int32_t input_halo_extents[],
                                       const int32_t output_halo_extents[], cudaStream_t stream) {
  using namespace cudecomp;
  try {
    checkHandle(handle);
    checkGridDesc(grid_desc);
    checkDataType(dtype);
    if (!input) { THROW_INVALID_USAGE("input argument cannot be null"); }
    if (!output) { THROW_INVALID_USAGE("output argument cannot be null"); }
    if (!work) { THROW_INVALID_USAGE("work argument cannot be null"); }
    switch (dtype) {
    case CUDECOMP_FLOAT:
      cudecompTransposeZToY(handle, grid_desc, reinterpret_cast<float*>(input), reinterpret_cast<float*>(output),
                            reinterpret_cast<float*>(work), input_halo_extents, output_halo_extents, stream);
      break;
    case CUDECOMP_DOUBLE:
      cudecompTransposeZToY(handle, grid_desc, reinterpret_cast<double*>(input), reinterpret_cast<double*>(output),
                            reinterpret_cast<double*>(work), input_halo_extents, output_halo_extents, stream);
      break;
    case CUDECOMP_FLOAT_COMPLEX:
      cudecompTransposeZToY(handle, grid_desc, reinterpret_cast<cuda::std::complex<float>*>(input),
                            reinterpret_cast<cuda::std::complex<float>*>(output),
                            reinterpret_cast<cuda::std::complex<float>*>(work), input_halo_extents, output_halo_extents,
                            stream);
      break;
    case CUDECOMP_DOUBLE_COMPLEX:
      cudecompTransposeZToY(handle, grid_desc, reinterpret_cast<cuda::std::complex<double>*>(input),
                            reinterpret_cast<cuda::std::complex<double>*>(output),
                            reinterpret_cast<cuda::std::complex<double>*>(work), input_halo_extents,
                            output_halo_extents, stream);
      break;
    }

  } catch (const cudecomp::BaseException& e) {
    std::cerr << e.what();
    return e.getResult();
  }
  return CUDECOMP_RESULT_SUCCESS;
}

cudecompResult_t cudecompTransposeYToX(cudecompHandle_t handle, cudecompGridDesc_t grid_desc, void* input, void* output,
                                       void* work, cudecompDataType_t dtype, const int32_t input_halo_extents[],
                                       const int32_t output_halo_extents[], cudaStream_t stream) {
  using namespace cudecomp;
  try {
    checkHandle(handle);
    checkGridDesc(grid_desc);
    checkDataType(dtype);
    if (!input) { THROW_INVALID_USAGE("input argument cannot be null"); }
    if (!output) { THROW_INVALID_USAGE("output argument cannot be null"); }
    if (!work) { THROW_INVALID_USAGE("work argument cannot be null"); }
    switch (dtype) {
    case CUDECOMP_FLOAT:
      cudecompTransposeYToX(handle, grid_desc, reinterpret_cast<float*>(input), reinterpret_cast<float*>(output),
                            reinterpret_cast<float*>(work), input_halo_extents, output_halo_extents, stream);
      break;
    case CUDECOMP_DOUBLE:
      cudecompTransposeYToX(handle, grid_desc, reinterpret_cast<double*>(input), reinterpret_cast<double*>(output),
                            reinterpret_cast<double*>(work), input_halo_extents, output_halo_extents, stream);
      break;
    case CUDECOMP_FLOAT_COMPLEX:
      cudecompTransposeYToX(handle, grid_desc, reinterpret_cast<cuda::std::complex<float>*>(input),
                            reinterpret_cast<cuda::std::complex<float>*>(output),
                            reinterpret_cast<cuda::std::complex<float>*>(work), input_halo_extents, output_halo_extents,
                            stream);
      break;
    case CUDECOMP_DOUBLE_COMPLEX:
      cudecompTransposeYToX(handle, grid_desc, reinterpret_cast<cuda::std::complex<double>*>(input),
                            reinterpret_cast<cuda::std::complex<double>*>(output),
                            reinterpret_cast<cuda::std::complex<double>*>(work), input_halo_extents,
                            output_halo_extents, stream);
      break;
    }

  } catch (const cudecomp::BaseException& e) {
    std::cerr << e.what();
    return e.getResult();
  }
  return CUDECOMP_RESULT_SUCCESS;
}

cudecompResult_t cudecompUpdateHalosX(cudecompHandle_t handle, cudecompGridDesc_t grid_desc, void* input, void* work,
                                      cudecompDataType_t dtype, const int32_t halo_extents[], const bool halo_periods[],
                                      int32_t dim, cudaStream_t stream) {
  using namespace cudecomp;
  try {
    checkHandle(handle);
    checkGridDesc(grid_desc);
    checkDataType(dtype);
    if (!halo_extents) { THROW_INVALID_USAGE("halo_extents argument cannot be null"); }
    if (halo_extents[0] == 0 && halo_extents[1] == 0 && halo_extents[2] == 0) {
      // No halos, quick return.
      return CUDECOMP_RESULT_SUCCESS;
    }
    if (!halo_periods) { THROW_INVALID_USAGE("halo_periods argument cannot be null"); }
    if (!input) { THROW_INVALID_USAGE("input argument cannot be null"); }
    if (!work) { THROW_INVALID_USAGE("work argument cannot be null"); }
    if (dim < 0 || dim > 2) { THROW_INVALID_USAGE("dim argument out of range"); }

    switch (dtype) {
    case CUDECOMP_FLOAT:
      cudecompUpdateHalosX(handle, grid_desc, reinterpret_cast<float*>(input), reinterpret_cast<float*>(work),
                           halo_extents, halo_periods, dim, stream);
      break;
    case CUDECOMP_DOUBLE:
      cudecompUpdateHalosX(handle, grid_desc, reinterpret_cast<double*>(input), reinterpret_cast<double*>(work),
                           halo_extents, halo_periods, dim, stream);
      break;
    case CUDECOMP_FLOAT_COMPLEX:
      cudecompUpdateHalosX(handle, grid_desc, reinterpret_cast<cuda::std::complex<float>*>(input),
                           reinterpret_cast<cuda::std::complex<float>*>(work), halo_extents, halo_periods, dim, stream);
      break;
    case CUDECOMP_DOUBLE_COMPLEX:
      cudecompUpdateHalosX(handle, grid_desc, reinterpret_cast<cuda::std::complex<double>*>(input),
                           reinterpret_cast<cuda::std::complex<double>*>(work), halo_extents, halo_periods, dim,
                           stream);
      break;
    }

  } catch (const cudecomp::BaseException& e) {
    std::cerr << e.what();
    return e.getResult();
  }
  return CUDECOMP_RESULT_SUCCESS;
}

cudecompResult_t cudecompUpdateHalosY(cudecompHandle_t handle, cudecompGridDesc_t grid_desc, void* input, void* work,
                                      cudecompDataType_t dtype, const int32_t halo_extents[], const bool halo_periods[],
                                      int32_t dim, cudaStream_t stream) {
  using namespace cudecomp;
  try {
    checkHandle(handle);
    checkGridDesc(grid_desc);
    checkDataType(dtype);
    if (!halo_extents) { THROW_INVALID_USAGE("halo_extents argument cannot be null"); }
    if (halo_extents[0] == 0 && halo_extents[1] == 0 && halo_extents[2] == 0) {
      // No halos, quick return.
      return CUDECOMP_RESULT_SUCCESS;
    }
    if (!halo_periods) { THROW_INVALID_USAGE("halo_periods argument cannot be null"); }
    if (!input) { THROW_INVALID_USAGE("input argument cannot be null"); }
    if (!work) { THROW_INVALID_USAGE("work argument cannot be null"); }
    if (dim < 0 || dim > 2) { THROW_INVALID_USAGE("dim argument out of range"); }

    switch (dtype) {
    case CUDECOMP_FLOAT:
      cudecompUpdateHalosY(handle, grid_desc, reinterpret_cast<float*>(input), reinterpret_cast<float*>(work),
                           halo_extents, halo_periods, dim, stream);
      break;
    case CUDECOMP_DOUBLE:
      cudecompUpdateHalosY(handle, grid_desc, reinterpret_cast<double*>(input), reinterpret_cast<double*>(work),
                           halo_extents, halo_periods, dim, stream);
      break;
    case CUDECOMP_FLOAT_COMPLEX:
      cudecompUpdateHalosY(handle, grid_desc, reinterpret_cast<cuda::std::complex<float>*>(input),
                           reinterpret_cast<cuda::std::complex<float>*>(work), halo_extents, halo_periods, dim, stream);
      break;
    case CUDECOMP_DOUBLE_COMPLEX:
      cudecompUpdateHalosY(handle, grid_desc, reinterpret_cast<cuda::std::complex<double>*>(input),
                           reinterpret_cast<cuda::std::complex<double>*>(work), halo_extents, halo_periods, dim,
                           stream);
      break;
    }

  } catch (const cudecomp::BaseException& e) {
    std::cerr << e.what();
    return e.getResult();
  }
  return CUDECOMP_RESULT_SUCCESS;
}

cudecompResult_t cudecompUpdateHalosZ(cudecompHandle_t handle, cudecompGridDesc_t grid_desc, void* input, void* work,
                                      cudecompDataType_t dtype, const int32_t halo_extents[], const bool halo_periods[],
                                      int32_t dim, cudaStream_t stream) {
  using namespace cudecomp;
  try {
    checkHandle(handle);
    checkGridDesc(grid_desc);
    checkDataType(dtype);
    if (!halo_extents) { THROW_INVALID_USAGE("halo_extents argument cannot be null"); }
    if (halo_extents[0] == 0 && halo_extents[1] == 0 && halo_extents[2] == 0) {
      // No halos, quick return.
      return CUDECOMP_RESULT_SUCCESS;
    }
    if (!halo_periods) { THROW_INVALID_USAGE("halo_periods argument cannot be null"); }
    if (!input) { THROW_INVALID_USAGE("input argument cannot be null"); }
    if (!work) { THROW_INVALID_USAGE("work argument cannot be null"); }
    if (dim < 0 || dim > 2) { THROW_INVALID_USAGE("dim argument out of range"); }

    switch (dtype) {
    case CUDECOMP_FLOAT:
      cudecompUpdateHalosZ(handle, grid_desc, reinterpret_cast<float*>(input), reinterpret_cast<float*>(work),
                           halo_extents, halo_periods, dim, stream);
      break;
    case CUDECOMP_DOUBLE:
      cudecompUpdateHalosZ(handle, grid_desc, reinterpret_cast<double*>(input), reinterpret_cast<double*>(work),
                           halo_extents, halo_periods, dim, stream);
      break;
    case CUDECOMP_FLOAT_COMPLEX:
      cudecompUpdateHalosZ(handle, grid_desc, reinterpret_cast<cuda::std::complex<float>*>(input),
                           reinterpret_cast<cuda::std::complex<float>*>(work), halo_extents, halo_periods, dim, stream);
      break;
    case CUDECOMP_DOUBLE_COMPLEX:
      cudecompUpdateHalosZ(handle, grid_desc, reinterpret_cast<cuda::std::complex<double>*>(input),
                           reinterpret_cast<cuda::std::complex<double>*>(work), halo_extents, halo_periods, dim,
                           stream);
      break;
    }

  } catch (const cudecomp::BaseException& e) {
    std::cerr << e.what();
    return e.getResult();
  }
  return CUDECOMP_RESULT_SUCCESS;
}
