/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <algorithm>
#include <cstring>
#include <iostream>
#include <numeric>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include <hip/hip_runtime.h>
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
#include "internal/cuda_wrap.h"
#include "internal/exceptions.h"
#include "internal/halo.h"
#include "internal/hashes.h"
#include "internal/nvml_wrap.h"
#include "internal/transpose.h"

namespace cudecomp {
namespace {

// Static flag to disable multiple handle creation
static bool cudecomp_initialized = false;

static cudecomp::ncclComm ncclCommFromMPIComm(MPI_Comm mpi_comm) {
  int rank, nranks;
  CHECK_MPI(MPI_Comm_rank(mpi_comm, &rank));
  CHECK_MPI(MPI_Comm_size(mpi_comm, &nranks));

  ncclUniqueId id;
  if (rank == 0) CHECK_NCCL(ncclGetUniqueId(&id));
  CHECK_MPI(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, mpi_comm));
  CHECK_MPI(MPI_Barrier(mpi_comm));
  ncclComm_t nccl_comm;
  CHECK_NCCL(ncclCommInitRank(&nccl_comm, nranks, id, rank));

  return cudecomp::createNcclComm(nccl_comm);
}

#ifdef ENABLE_NVSHMEM
static void initNvshmemFromMPIComm(MPI_Comm mpi_comm) {
  int rank, nranks;
  CHECK_MPI(MPI_Comm_rank(mpi_comm, &rank));
  CHECK_MPI(MPI_Comm_size(mpi_comm, &nranks));

  nvshmemx_init_attr_t attr;
#if NVSHMEM_VENDOR_MAJOR_VERSION >= 3
  nvshmemx_uniqueid_t id;
  if (rank == 0) nvshmemx_get_uniqueid(&id);
  CHECK_MPI(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, mpi_comm));
  CHECK_MPI(MPI_Barrier(mpi_comm));
  nvshmemx_set_attr_uniqueid_args(rank, nranks, &id, &attr);
  nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr);
#else
  attr.mpi_comm = &mpi_comm;
  nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
#endif
}
#endif

static bool checkEnvVar(const char* env_var_str) {
  const char* env_var_val_str = std::getenv(env_var_str);
  bool result = false;
  if (env_var_val_str) { result = std::strtol(env_var_val_str, nullptr, 10) == 1; }
  return result;
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

  bool mem_order_set = (config->transpose_mem_order[0][0] >= 0);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      if ((mem_order_set && config->transpose_mem_order[i][j] < 0) ||
          (!mem_order_set && config->transpose_mem_order[i][j] >= 0)) {
        THROW_INVALID_USAGE("transpose_mem_order only partially set");
      }
    }
  }

  if (mem_order_set) {
    for (int i = 0; i < 3; ++i) {
      std::set<int32_t> order_vals;
      for (int j = 0; j < 3; ++j) {
        order_vals.insert(config->transpose_mem_order[i][j]);
      }
      if (order_vals.size() != 3 || *order_vals.begin() != 0 || *order_vals.rbegin() != 2) {
        THROW_INVALID_USAGE("transpose_mem_order setting is invalid");
      }
    }
  }
}

static void gatherGlobalMPIInfo(cudecompHandle_t& handle) {
  // Gather hostnames by rank
  handle->hostnames.resize(handle->nranks);
  int resultlen;
  CHECK_MPI(MPI_Get_processor_name(handle->hostnames[handle->rank].data(), &resultlen));
  CHECK_MPI(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, handle->hostnames.data(), MPI_MAX_PROCESSOR_NAME,
                          MPI_CHAR, handle->mpi_comm));

  // Gather rank to local rank mappings
  handle->rank_to_local_rank.resize(handle->nranks);
  handle->rank_to_local_rank[handle->rank] = handle->local_rank;
  CHECK_MPI(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, handle->rank_to_local_rank.data(), 1, MPI_INT,
                          handle->mpi_comm));

  // Gather MNNVL clique related structures (if supported)
  int dev;
  CHECK_CUDA(hipGetDevice(&dev));
  char pciBusId[] = "00000000:00:00.0";
  CHECK_CUDA(hipDeviceGetPCIBusId(pciBusId, sizeof(pciBusId), dev));
  nvmlDevice_t nvml_dev;
  CHECK_NVML(nvmlDeviceGetHandleByPciBusId(pciBusId, &nvml_dev));
#if NVML_API_VERSION >= 12 && CUDART_VERSION >= 12040
  nvmlGpuFabricInfoV_t fabricInfo = {.version = nvmlGpuFabricInfo_v2};

  // Check CUDECOMP_DISABLE_MNNVL (debug setting to disable MNNVL topology detection)
  bool disable_mnnvl = checkEnvVar("CUDECOMP_DISABLE_MNNVL");

  if (nvmlHasFabricSupport() && !disable_mnnvl) {
    handle->rank_to_mnnvl_info.resize(handle->nranks);

    // Gather MNNVL information (clusterUuid, cliqueId) by rank
    CHECK_NVML(nvmlDeviceGetGpuFabricInfoV(nvml_dev, &fabricInfo));
    unsigned char zeros[NVML_GPU_FABRIC_UUID_LEN]{};
    std::vector<std::array<unsigned char, NVML_GPU_FABRIC_UUID_LEN>> clusterUuids(handle->nranks);
    std::vector<unsigned int> cliqueIds(handle->nranks);

    if (std::memcmp(fabricInfo.clusterUuid, zeros, NVML_GPU_FABRIC_UUID_LEN) != 0) {
      std::memcpy(clusterUuids[handle->rank].data(), fabricInfo.clusterUuid, NVML_GPU_FABRIC_UUID_LEN);
      cliqueIds[handle->rank] = fabricInfo.cliqueId;
    } else {
      std::memcpy(clusterUuids[handle->rank].data(), zeros, NVML_GPU_FABRIC_UUID_LEN);
      cliqueIds[handle->rank] = 0;
    }

    CHECK_MPI(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, clusterUuids.data(), NVML_GPU_FABRIC_UUID_LEN, MPI_CHAR,
                            handle->mpi_comm));
    CHECK_MPI(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, cliqueIds.data(), 1, MPI_INT, handle->mpi_comm));

    for (int i = 0; i < handle->nranks; ++i) {
      if (std::memcmp(clusterUuids[i].data(), zeros, NVML_GPU_FABRIC_UUID_LEN) == 0) {
        // If any rank has a zero cluster UUID, disable MNNVL.
        handle->rank_to_mnnvl_info.resize(0);
        break;
      }
      handle->rank_to_mnnvl_info[i].first = clusterUuids[i];
      handle->rank_to_mnnvl_info[i].second = cliqueIds[i];
    }
  }

  if (handle->rank_to_mnnvl_info.size() != 0) {
    // cliqueIds returned in fabricInfo are not unique across MNNVL domains (defined by clusterUuid).
    // Define a modified clique id that assigns a unique index to each (clusterUuid, cliqueId) pair.
    std::unordered_map<mnnvl_info, unsigned int> clique_ids;
    clique_ids = getUniqueIds(handle->rank_to_mnnvl_info);

    // Populate rank to MNNVL clique mappings
    handle->rank_to_clique.resize(handle->nranks);
    for (int i = 0; i < handle->nranks; ++i) {
      handle->rank_to_clique[i] = clique_ids[handle->rank_to_mnnvl_info[i]];
    }

    // Create clique local MPI communicator
    CHECK_MPI(MPI_Comm_split(handle->mpi_comm, static_cast<int>(handle->rank_to_clique[handle->rank]), handle->rank,
                             &handle->mpi_clique_comm));
    CHECK_MPI(MPI_Comm_rank(handle->mpi_clique_comm, &handle->clique_rank));
    CHECK_MPI(MPI_Comm_size(handle->mpi_clique_comm, &handle->clique_nranks));

    // Gather rank to clique rank mappings
    handle->rank_to_clique_rank.resize(handle->nranks);
    handle->rank_to_clique_rank[handle->rank] = handle->clique_rank;
    CHECK_MPI(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, handle->rank_to_clique_rank.data(), 1, MPI_INT,
                            handle->mpi_comm));
  }
#endif

  // Copy local rank mapping to clique rank mapping in general case
  if (handle->rank_to_clique_rank.size() == 0) { handle->rank_to_clique_rank = handle->rank_to_local_rank; }
}

static void getCudecompEnvVars(cudecompHandle_t& handle) {
  // Check CUDECOMP_ENABLE_NCCL_UBR (NCCL user buffer registration)
  handle->nccl_enable_ubr = checkEnvVar("CUDECOMP_ENABLE_NCCL_UBR");

  // Check CUDECOMP_ENABLE_CUMEM (CUDA VMM allocations for work buffers)
  handle->cuda_cumem_enable = checkEnvVar("CUDECOMP_ENABLE_CUMEM");
  if (handle->cuda_cumem_enable) {
#if CUDART_VERSION < 12030
    if (handle->rank == 0) {
      printf("CUDECOMP:WARN: CUDECOMP_ENABLE_CUMEM is set but CUDA version used for compilation does not "
             "support fabric allocations. Disabling this feature.\n");
    }
    handle->cuda_cumem_enable = false;
#else
    int driverVersion;
    CHECK_CUDA(hipDriverGetVersion(&driverVersion));
    if (driverVersion < 12030) {
      if (handle->rank == 0) {
        printf("CUDECOMP:WARN: CUDECOMP_ENABLE_CUMEM is set but installed driver does not "
               "support fabric allocations. Disabling this feature.\n");
      }
      handle->cuda_cumem_enable = false;
    } else {
      // Check if fabric allocation type is supported
      int dev;
      hipDevice_t cu_dev;
      CHECK_CUDA(hipGetDevice(&dev));
      CHECK_CUDA_DRV(hipDeviceGet(&cu_dev, dev));
      int flag = 0;
      CHECK_CUDA_DRV(hipDeviceGetAttribute(&flag, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, cu_dev));
      if (!flag) {
        if (handle->rank == 0) {
          printf("CUDECOMP:WARN: CUDECOMP_ENABLE_CUMEM is set but device does not "
                 "support fabric allocations. Disabling this feature.\n");
        }
        handle->cuda_cumem_enable = false;
      }
    }
#endif
  }

  // Check CUDECOMP_ENABLE_CUDA_GRAPHS (CUDA Graphs usage in pipelined backends)
  handle->cuda_graphs_enable = checkEnvVar("CUDECOMP_ENABLE_CUDA_GRAPHS");
  if (handle->cuda_graphs_enable) {
#if CUDART_VERSION < 11010
    if (handle->rank == 0) {
      printf("CUDECOMP:WARN: CUDECOMP_ENABLE_CUDA_GRAPHS is set but CUDA version used for compilation does not "
             "support hipEventRecordWithFlags which is required. Disabling this feature.\n");
    }
    handle->cuda_graphs_enable = false;
#endif
  }

  // Check CUDECOMP_ENABLE_PERFORMANCE_REPORT (Performance reporting)
  handle->performance_report_enable = checkEnvVar("CUDECOMP_ENABLE_PERFORMANCE_REPORT");

  // Check CUDECOMP_PERFORMANCE_REPORT_DETAIL (Performance report detail level)
  const char* performance_detail_str = std::getenv("CUDECOMP_PERFORMANCE_REPORT_DETAIL");
  if (performance_detail_str) {
    int32_t detail = std::strtol(performance_detail_str, nullptr, 10);
    if (detail >= 0 && detail <= 2) {
      handle->performance_report_detail = detail;
    } else if (handle->rank == 0) {
      printf("CUDECOMP:WARN: Invalid CUDECOMP_PERFORMANCE_REPORT_DETAIL value (%d). Using default (0).\n", detail);
    }
  }

  // Check CUDECOMP_PERFORMANCE_REPORT_SAMPLES (Number of performance samples to keep)
  const char* performance_samples_str = std::getenv("CUDECOMP_PERFORMANCE_REPORT_SAMPLES");
  if (performance_samples_str) {
    int32_t samples = std::strtol(performance_samples_str, nullptr, 10);
    if (samples > 0) { // Only require positive values
      handle->performance_report_samples = samples;
    } else if (handle->rank == 0) {
      printf("CUDECOMP:WARN: Invalid CUDECOMP_PERFORMANCE_REPORT_SAMPLES value (%d). Using default (%d).\n", samples,
             handle->performance_report_samples);
    }
  }

  // Check CUDECOMP_PERFORMANCE_REPORT_WARMUP_SAMPLES (Number of initial samples to ignore)
  const char* performance_warmup_str = std::getenv("CUDECOMP_PERFORMANCE_REPORT_WARMUP_SAMPLES");
  if (performance_warmup_str) {
    int32_t warmup_samples = std::strtol(performance_warmup_str, nullptr, 10);
    if (warmup_samples >= 0) { // Only require non-negative values
      handle->performance_report_warmup_samples = warmup_samples;
    } else if (handle->rank == 0) {
      printf("CUDECOMP:WARN: Invalid CUDECOMP_PERFORMANCE_REPORT_WARMUP_SAMPLES value (%d). Using default (%d).\n",
             warmup_samples, handle->performance_report_warmup_samples);
    }
  }

  // Check CUDECOMP_PERFORMANCE_REPORT_WRITE_DIR (Directory for CSV performance reports)
  const char* performance_write_dir_str = std::getenv("CUDECOMP_PERFORMANCE_REPORT_WRITE_DIR");
  if (performance_write_dir_str) { handle->performance_report_write_dir = std::string(performance_write_dir_str); }

  // Check CUDECOMP_USE_COL_MAJOR_RANK_ORDER (Column-major rank assignment)
  handle->use_col_major_rank_order = checkEnvVar("CUDECOMP_USE_COL_MAJOR_RANK_ORDER");
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

    // Load CUDA driver symbols into table
    initCuFunctionTable();

    // Load NVML symbols into table and initialize NVML
    initNvmlFunctionTable();

    CHECK_NVML(nvmlInit());

    // Initialize cuTENSOR library
#if CUTENSOR_MAJOR >= 2
    CHECK_CUTENSOR(hiptensorCreate(&handle->cutensor_handle));
    CHECK_CUTENSOR(hiptensorCreatePlanPreference(handle->cutensor_handle, &handle->cutensor_plan_pref,
                                                 HIPTENSOR_ALGO_DEFAULT, HIPTENSOR_JIT_MODE_NONE));
#else
    CHECK_CUTENSOR(cutensorInit(&handle->cutensor_handle));
#endif

    // Gather cuDecomp environment variable settings
    getCudecompEnvVars(handle);

    // Gather extra MPI info from all communicator ranks
    gatherGlobalMPIInfo(handle);

    // Determine P2P CE count
    int dev;
    hipDevice_t cu_dev;
    CHECK_CUDA(hipGetDevice(&dev));
    CHECK_CUDA_DRV(hipDeviceGet(&cu_dev, dev));
    char pciBusId[] = "00000000:00:00.0";
    CHECK_CUDA(hipDeviceGetPCIBusId(pciBusId, sizeof(pciBusId), dev));
    nvmlDevice_t nvml_dev;
    CHECK_NVML(nvmlDeviceGetHandleByPciBusId(pciBusId, &nvml_dev));

    // Check if NVSwitch is present
    bool has_nvswitch = false;
    nvmlFieldValue_t fv;
    fv.fieldId = NVML_FI_DEV_NVSWITCH_CONNECTED_LINK_COUNT;
    CHECK_NVML(nvmlDeviceGetFieldValues(nvml_dev, 1, &fv));
    if (fv.nvmlReturn == NVML_SUCCESS) { has_nvswitch = fv.value.uiVal > 0; }

    // If NVSwitch is not present, determine number of NVLink connected peers
    int num_nvlink_peers = 0;
    if (!has_nvswitch) {
      std::set<std::string> remote_bus_ids;
      for (int i = 0; i < NVML_NVLINK_MAX_LINKS; ++i) {
        unsigned int p2p_supported = 0;
        nvmlReturn_t ret =
            nvmlFnTable.pfn_nvmlDeviceGetNvLinkCapability(nvml_dev, i, NVML_NVLINK_CAP_P2P_SUPPORTED, &p2p_supported);
        if (ret != NVML_SUCCESS || !p2p_supported) continue;

        nvmlEnableState_t isActive;
        ret = nvmlFnTable.pfn_nvmlDeviceGetNvLinkState(nvml_dev, i, &isActive);
        if (ret != NVML_SUCCESS || isActive != NVML_FEATURE_ENABLED) continue;

        nvmlPciInfo_t pciInfo = {};
        CHECK_NVML(nvmlDeviceGetNvLinkRemotePciInfo(nvml_dev, i, &pciInfo));
        std::string busId = std::string(pciInfo.busId);
        if (busId.empty()) {
          // Fall back to legacy bus ID
          busId = std::string(pciInfo.busIdLegacy);
        }
        remote_bus_ids.insert(busId);
      }
      num_nvlink_peers = static_cast<int>(remote_bus_ids.size());
    }

    // Set P2P CE count
    if (has_nvswitch) {
      handle->device_p2p_ce_count = 1; // 1 P2P CE for NVSwitch
    } else if (num_nvlink_peers > 0) {
      handle->device_p2p_ce_count = num_nvlink_peers; // Assume each NVLink peer has a P2P CE available
    } else {
      handle->device_p2p_ce_count = 2; // Assume 2 P2P CE otherwise (shared D2H/H2D CE)
    }

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

    handle->nccl_comm.reset();
    handle->nccl_local_comm.reset();

    for (auto& stream : handle->streams) {
      CHECK_CUDA(hipStreamDestroy(stream));
    }
#ifdef ENABLE_NVSHMEM
    if (handle->nvshmem_initialized) {
      nvshmem_finalize();
      handle->nvshmem_initialized = false;
      handle->nvshmem_allocations.clear();
      handle->nvshmem_allocation_size = 0;
    }
#endif
    CHECK_MPI(MPI_Comm_free(&handle->mpi_local_comm));

#if CUTENSOR_MAJOR >= 2
    CHECK_CUTENSOR(hiptensorDestroy(handle->cutensor_handle));
    CHECK_CUTENSOR(hiptensorDestroyPlanPreference(handle->cutensor_plan_pref));
#endif

    CHECK_NVML(nvmlShutdown());

    handle = nullptr;
    delete handle;

    cudecomp_initialized = false;

  } catch (const cudecomp::BaseException& e) {
    std::cerr << e.what();
    return e.getResult();
  }
  return CUDECOMP_RESULT_SUCCESS;
}

cudecompResult_t cudecompInit_F(cudecompHandle_t* handle_in, MPI_Fint mpi_comm_f) {
  using namespace cudecomp;
  try {
    MPI_Comm mpi_comm = MPI_Comm_f2c(mpi_comm_f);
    cudecompInit(handle_in, mpi_comm);
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

    // If transpose_mem_order not used, set based on transpose_axis_contiguous settings)
    grid_desc->transpose_mem_order_set = (config->transpose_mem_order[0][0] >= 0);

    if (!grid_desc->transpose_mem_order_set) {
      for (int axis = 0; axis < 3; ++axis) {
        for (int i = 0; i < 3; ++i) {
          if (grid_desc->config.transpose_axis_contiguous[axis]) {
            grid_desc->config.transpose_mem_order[axis][i] = (axis + i) % 3;
          } else {
            grid_desc->config.transpose_mem_order[axis][i] = i;
          }
        }
      }
    }

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
      if (!handle->nccl_local_comm) {
        if (grid_desc->config.pdims[0] > 0 && grid_desc->config.pdims[1] > 0) {
          // If pdims are set, temporarily set up comm info stuctures to determine if we need to create a local NCCL
          // communicator
          if (handle->use_col_major_rank_order) {
            grid_desc->pidx[0] = handle->rank % grid_desc->config.pdims[0];
            grid_desc->pidx[1] = handle->rank / grid_desc->config.pdims[0];
          } else {
            grid_desc->pidx[0] = handle->rank / grid_desc->config.pdims[1];
            grid_desc->pidx[1] = handle->rank % grid_desc->config.pdims[1];
          }
          int color_row = grid_desc->pidx[0];
          MPI_Comm row_comm;
          CHECK_MPI(MPI_Comm_split(handle->mpi_comm, color_row, handle->rank, &row_comm));
          setCommInfo(handle, grid_desc, row_comm, CUDECOMP_COMM_ROW);

          int color_col = grid_desc->pidx[1];
          MPI_Comm col_comm;
          CHECK_MPI(MPI_Comm_split(handle->mpi_comm, color_col, handle->rank, &col_comm));
          setCommInfo(handle, grid_desc, col_comm, CUDECOMP_COMM_COL);

          // Create local NCCL communicator if row or column communicator uses it
          int need_local_nccl_comm =
              static_cast<int>((grid_desc->row_comm_info.ngroups == 1 && grid_desc->row_comm_info.nranks > 1) ||
                               (grid_desc->col_comm_info.ngroups == 1 && grid_desc->col_comm_info.nranks > 1));

          // Local comm can include ranks in other rows/columns, need additional check for those cases.
          CHECK_MPI(MPI_Allreduce(MPI_IN_PLACE, &need_local_nccl_comm, 1, MPI_INT, MPI_LOR,
                                  handle->mpi_clique_comm != MPI_COMM_NULL ? handle->mpi_clique_comm
                                                                           : handle->mpi_local_comm));

          if (need_local_nccl_comm) {
            handle->nccl_local_comm = ncclCommFromMPIComm(
                handle->mpi_clique_comm != MPI_COMM_NULL ? handle->mpi_clique_comm : handle->mpi_local_comm);
          }
          CHECK_MPI(MPI_Comm_free(&row_comm));
          CHECK_MPI(MPI_Comm_free(&col_comm));
        } else {
          // If pdims are not set, set up local NCCL communicator for use during autotuning
          handle->nccl_local_comm = ncclCommFromMPIComm(
              handle->mpi_clique_comm != MPI_COMM_NULL ? handle->mpi_clique_comm : handle->mpi_local_comm);
        }
      }

      // Set grid descriptor references to NCCL communicators
      grid_desc->nccl_comm = handle->nccl_comm;
      grid_desc->nccl_local_comm = handle->nccl_local_comm;
    }

    // Initialize NVSHMEM if needed
    if (transposeBackendRequiresNvshmem(comm_backend) || haloBackendRequiresNvshmem(halo_comm_backend) ||
        ((autotune_transpose_backend || autotune_halo_backend) && !autotune_disable_nvshmem_backends)) {
#ifdef ENABLE_NVSHMEM
      if (!handle->nvshmem_initialized) {
        inspectNvshmemEnvVars(handle);
        initNvshmemFromMPIComm(handle->mpi_comm);
        handle->nvshmem_initialized = true;
        handle->nvshmem_allocation_size = 0;
      }
#endif
    }

    if (handle->streams.empty()) {
      handle->streams.resize(handle->device_p2p_ce_count + 1);
      int greatest_priority;
      CHECK_CUDA(hipDeviceGetStreamPriorityRange(nullptr, &greatest_priority));
      for (auto& stream : handle->streams) {
        CHECK_CUDA(hipStreamCreateWithPriority(&stream, hipStreamNonBlocking, greatest_priority));
      }
    }

    // Create CUDA events for scheduling
    grid_desc->events.resize(handle->nranks);
    for (auto& event : grid_desc->events) {
      CHECK_CUDA(hipEventCreateWithFlags(&event, hipEventDisableTiming));
    }
#ifdef ENABLE_NVSHMEM
    CHECK_CUDA(hipEventCreateWithFlags(&grid_desc->nvshmem_sync_event, hipEventDisableTiming));
#endif

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

    if (handle->use_col_major_rank_order) {
      grid_desc->pidx[0] = handle->rank % grid_desc->config.pdims[0];
      grid_desc->pidx[1] = handle->rank / grid_desc->config.pdims[0];
    } else {
      grid_desc->pidx[0] = handle->rank / grid_desc->config.pdims[1];
      grid_desc->pidx[1] = handle->rank % grid_desc->config.pdims[1];
    }

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
      // If this grid descriptor initialized the group local NCCL communicator but does not need it, release reference
      // to it
      if (grid_desc->nccl_local_comm) {
        if ((grid_desc->row_comm_info.ngroups > 1 || grid_desc->row_comm_info.nranks == 1) &&
            (grid_desc->col_comm_info.ngroups > 1 || grid_desc->col_comm_info.nranks == 1)) {
          grid_desc->nccl_local_comm.reset();

          // If handle has the only remaining reference to the local NCCL communicator, destroy it to reclaim resources
          if (handle->nccl_local_comm.use_count() == 1) { handle->nccl_local_comm.reset(); }
        }
      }
    } else {
      // Release grid descriptor references to NCCL communicators
      grid_desc->nccl_comm.reset();
      grid_desc->nccl_local_comm.reset();

      // Destroy NCCL communicators to reclaim resources if not used by other grid descriptors
      if (handle->nccl_comm && handle->nccl_comm.use_count() == 1) { handle->nccl_comm.reset(); }
      if (handle->nccl_local_comm && handle->nccl_local_comm.use_count() == 1) { handle->nccl_local_comm.reset(); }
    }

    *grid_desc_in = grid_desc;
    *config = grid_desc->config;
    // If gdims_dist was not set, return config with default values
    if (!grid_desc->gdims_dist_set) {
      config->gdims_dist[0] = 0;
      config->gdims_dist[1] = 0;
      config->gdims_dist[2] = 0;
    }

    // If transpose_mem_order was not set, return config with default values
    if (!grid_desc->transpose_mem_order_set) {
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
          config->transpose_mem_order[i][j] = -1;
        }
      }
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
      if (e) { CHECK_CUDA(hipEventDestroy(e)); }
    }

#ifdef ENABLE_NVSHMEM
    if (grid_desc->nvshmem_sync_event) { CHECK_CUDA(hipEventDestroy(grid_desc->nvshmem_sync_event)); }
#endif

    if (handle->performance_report_enable) {
      // Print performance report before destroying events
      printPerformanceReport(handle, grid_desc);

      // Destroy all transpose performance sample events in the map
      for (auto& entry : grid_desc->transpose_perf_samples_map) {
        auto& collection = entry.second;
        for (auto& sample : collection.samples) {
          CHECK_CUDA(hipEventDestroy(sample.transpose_start_event));
          CHECK_CUDA(hipEventDestroy(sample.transpose_end_event));
          for (auto& event : sample.alltoall_start_events) {
            CHECK_CUDA(hipEventDestroy(event));
          }
          for (auto& event : sample.alltoall_end_events) {
            CHECK_CUDA(hipEventDestroy(event));
          }
        }
      }

      // Destroy all halo performance sample events in the map
      for (auto& entry : grid_desc->halo_perf_samples_map) {
        auto& collection = entry.second;
        for (auto& sample : collection.samples) {
          CHECK_CUDA(hipEventDestroy(sample.halo_start_event));
          CHECK_CUDA(hipEventDestroy(sample.halo_end_event));
          CHECK_CUDA(hipEventDestroy(sample.sendrecv_start_event));
          CHECK_CUDA(hipEventDestroy(sample.sendrecv_end_event));
        }
      }
    }

    if (transposeBackendRequiresNccl(grid_desc->config.transpose_comm_backend) ||
        haloBackendRequiresNccl(grid_desc->config.halo_comm_backend)) {
      // Release grid descriptor references to NCCL communicators
      grid_desc->nccl_comm.reset();
      grid_desc->nccl_local_comm.reset();

      // Destroy NCCL communicators to reclaim resources if not used by other grid descriptors
      if (handle->nccl_comm && handle->nccl_comm.use_count() == 1) { handle->nccl_comm.reset(); }
      if (handle->nccl_local_comm && handle->nccl_local_comm.use_count() == 1) { handle->nccl_local_comm.reset(); }
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
    for (int i = 0; i < 2; ++i) {
      config->pdims[i] = 0;
    }
    for (int i = 0; i < 3; ++i) {
      config->gdims[i] = 0;
      config->gdims_dist[i] = 0;
    }

    // Transpose Options
    config->transpose_comm_backend = CUDECOMP_TRANSPOSE_COMM_MPI_P2P;
    for (int i = 0; i < 3; ++i) {
      config->transpose_axis_contiguous[i] = false;
      for (int j = 0; j < 3; ++j) {
        config->transpose_mem_order[i][j] = -1;
      }
    }

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
    options->n_warmup_trials = 3;
    options->n_trials = 5;
    options->grid_mode = CUDECOMP_AUTOTUNE_GRID_TRANSPOSE;
    options->dtype = CUDECOMP_DOUBLE;
    options->allow_uneven_decompositions = true;
    options->disable_nccl_backends = false;
    options->disable_nvshmem_backends = false;
    options->skip_threshold = 0.0;

    // Transpose-specific options
    options->autotune_transpose_backend = false;
    for (int i = 0; i < 4; ++i) {
      options->transpose_use_inplace_buffers[i] = false;
      options->transpose_op_weights[i] = 1.0;
      for (int j = 0; j < 3; ++j) {
        options->transpose_input_halo_extents[i][j] = 0;
        options->transpose_output_halo_extents[i][j] = 0;
        options->transpose_input_padding[i][j] = 0;
        options->transpose_output_padding[i][j] = 0;
      }
    }

    // Halo-specific options
    options->autotune_halo_backend = false;
    options->halo_axis = 0;
    for (int i = 0; i < 3; ++i) {
      options->halo_extents[i] = 0;
      options->halo_periods[i] = false;
      options->halo_padding[i] = 0;
    }

  } catch (const cudecomp::BaseException& e) {
    std::cerr << e.what();
    return e.getResult();
  }
  return CUDECOMP_RESULT_SUCCESS;
}

cudecompResult_t cudecompGetPencilInfo(cudecompHandle_t handle, cudecompGridDesc_t grid_desc,
                                       cudecompPencilInfo_t* pencil_info, int32_t axis, const int32_t halo_extents[],
                                       const int32_t padding[]) {
  using namespace cudecomp;
  try {
    checkHandle(handle);
    checkGridDesc(grid_desc);
    if (!pencil_info) { THROW_INVALID_USAGE("pencil_info argument cannot be null."); }
    if (axis < 0 || axis > 2) { THROW_INVALID_USAGE("axis argument out of range"); }

    std::array<int32_t, 3> invorder;

    // Setup order (and inverse)
    for (int i = 0; i < 3; ++i) {
      pencil_info->order[i] = grid_desc->config.transpose_mem_order[axis][i];
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

      if (padding) {
        pencil_info->shape[ord] += padding[i];
        pencil_info->padding[i] = padding[i];
      } else {
        pencil_info->padding[i] = 0;
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

    // If transpose_mem_order was not set, return config with default values
    if (!grid_desc->transpose_mem_order_set) {
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
          config->transpose_mem_order[i][j] = -1;
        }
      }
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
    CHECK_CUDECOMP(cudecompGetPencilInfo(handle, grid_desc, &pinfo, axis, halo_extents, nullptr));
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
      if (handle->cuda_cumem_enable) {
#if CUDART_VERSION >= 12030
        int dev;
        hipDevice_t cu_dev;
        CHECK_CUDA(hipGetDevice(&dev));
        CHECK_CUDA_DRV(hipDeviceGet(&cu_dev, dev));

        hipMemAllocationProp prop = {};
        prop.type = hipMemAllocationTypePinned;
        prop.location.type = hipMemLocationTypeDevice;
        prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
        prop.location.id = cu_dev;

        // Check for RDMA support
        int flag;
        CHECK_CUDA_DRV(
            hipDeviceGetAttribute(&flag, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED, cu_dev));
        if (flag) prop.allocFlags.gpuDirectRDMACapable = 1;

        // Align allocation size to required granularity
        size_t granularity;
        CHECK_CUDA_DRV(hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum));
        buffer_size_bytes = (buffer_size_bytes + granularity - 1) / granularity * granularity;

        // Allocate memory
        hipMemGenericAllocationHandle_t cumem_handle;
        CHECK_CUDA_DRV(hipMemCreate(&cumem_handle, buffer_size_bytes, &prop, 0));
        CHECK_CUDA_DRV(hipMemAddressReserve((hipDeviceptr_t*)buffer, buffer_size_bytes, granularity, 0, 0));
        CHECK_CUDA_DRV(hipMemMap((hipDeviceptr_t)*buffer, buffer_size_bytes, 0, cumem_handle, 0));

        // Set read/write access
        hipMemAccessDesc accessDesc = {};
        accessDesc.location.type = hipMemLocationTypeDevice;
        accessDesc.location.id = cu_dev;
        accessDesc.flags = hipMemAccessFlagsProtReadWrite;
        CHECK_CUDA_DRV(hipMemSetAccess((hipDeviceptr_t)*buffer, buffer_size_bytes, &accessDesc, 1));
#endif
      } else {
        CHECK_CUDA(hipMalloc(buffer, buffer_size_bytes));
      }
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 19, 0)
      if (transposeBackendRequiresNccl(grid_desc->config.transpose_comm_backend) ||
          haloBackendRequiresNccl(grid_desc->config.halo_comm_backend)) {

        if (handle->nccl_enable_ubr) {
          void* nccl_ubr_handle;
          if (grid_desc->nccl_comm) {
            CHECK_NCCL(ncclCommRegister(*grid_desc->nccl_comm, buffer, buffer_size_bytes, &nccl_ubr_handle));
            handle->nccl_ubr_handles[*buffer].push_back(std::make_pair(*grid_desc->nccl_comm, nccl_ubr_handle));
          }
          if (grid_desc->nccl_local_comm) {
            CHECK_NCCL(ncclCommRegister(*grid_desc->nccl_local_comm, buffer, buffer_size_bytes, &nccl_ubr_handle));
            handle->nccl_ubr_handles[*buffer].push_back(std::make_pair(*grid_desc->nccl_local_comm, nccl_ubr_handle));
          }
        }
      }
#endif
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

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 19, 0)
    if (transposeBackendRequiresNccl(grid_desc->config.transpose_comm_backend) ||
        haloBackendRequiresNccl(grid_desc->config.halo_comm_backend)) {

      if (handle->nccl_ubr_handles.count(buffer) != 0) {
        for (const auto& entry : handle->nccl_ubr_handles[buffer]) {
          CHECK_NCCL(ncclCommDeregister(entry.first, entry.second));
        }
        handle->nccl_ubr_handles.erase(buffer);
      }
    }
#endif

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
      if (handle->cuda_cumem_enable) {
#if CUDART_VERSION >= 12030
        if (buffer) {
          hipMemGenericAllocationHandle_t cumem_handle;
          CHECK_CUDA_DRV(hipMemRetainAllocationHandle(&cumem_handle, buffer));
          CHECK_CUDA_DRV(hipMemRelease(cumem_handle));
          size_t size = 0;
          CHECK_CUDA_DRV(hipMemGetAddressRange(NULL, &size, (hipDeviceptr_t)buffer));
          CHECK_CUDA_DRV(hipMemUnmap((hipDeviceptr_t)buffer, size));
          CHECK_CUDA_DRV(hipMemRelease(cumem_handle));
          CHECK_CUDA_DRV(hipMemAddressFree((hipDeviceptr_t)buffer, size));
        }
#endif
      } else {
        if (buffer) { CHECK_CUDA(hipFree(buffer)); }
      }
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
    CHECK_CUDECOMP(cudecompGetPencilInfo(handle, grid_desc, &pinfo, axis, nullptr, nullptr));

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
      int global_peer = getGlobalRank(handle, grid_desc, comm_axis, comm_peer);
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
                                       const int32_t output_halo_extents[], const int32_t input_padding[],
                                       const int32_t output_padding[], hipStream_t stream) {
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
                            reinterpret_cast<float*>(work), input_halo_extents, output_halo_extents, input_padding,
                            output_padding, stream);
      break;
    case CUDECOMP_DOUBLE:
      cudecompTransposeXToY(handle, grid_desc, reinterpret_cast<double*>(input), reinterpret_cast<double*>(output),
                            reinterpret_cast<double*>(work), input_halo_extents, output_halo_extents, input_padding,
                            output_padding, stream);
      break;
    case CUDECOMP_FLOAT_COMPLEX:
      cudecompTransposeXToY(handle, grid_desc, reinterpret_cast<cuda::std::complex<float>*>(input),
                            reinterpret_cast<cuda::std::complex<float>*>(output),
                            reinterpret_cast<cuda::std::complex<float>*>(work), input_halo_extents, output_halo_extents,
                            input_padding, output_padding, stream);
      break;
    case CUDECOMP_DOUBLE_COMPLEX:
      cudecompTransposeXToY(handle, grid_desc, reinterpret_cast<cuda::std::complex<double>*>(input),
                            reinterpret_cast<cuda::std::complex<double>*>(output),
                            reinterpret_cast<cuda::std::complex<double>*>(work), input_halo_extents,
                            output_halo_extents, input_padding, output_padding, stream);
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
                                       const int32_t output_halo_extents[], const int32_t input_padding[],
                                       const int32_t output_padding[], hipStream_t stream) {
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
                            reinterpret_cast<float*>(work), input_halo_extents, output_halo_extents, input_padding,
                            output_padding, stream);
      break;
    case CUDECOMP_DOUBLE:
      cudecompTransposeYToZ(handle, grid_desc, reinterpret_cast<double*>(input), reinterpret_cast<double*>(output),
                            reinterpret_cast<double*>(work), input_halo_extents, output_halo_extents, input_padding,
                            output_padding, stream);
      break;
    case CUDECOMP_FLOAT_COMPLEX:
      cudecompTransposeYToZ(handle, grid_desc, reinterpret_cast<cuda::std::complex<float>*>(input),
                            reinterpret_cast<cuda::std::complex<float>*>(output),
                            reinterpret_cast<cuda::std::complex<float>*>(work), input_halo_extents, output_halo_extents,
                            input_padding, output_padding, stream);
      break;
    case CUDECOMP_DOUBLE_COMPLEX:
      cudecompTransposeYToZ(handle, grid_desc, reinterpret_cast<cuda::std::complex<double>*>(input),
                            reinterpret_cast<cuda::std::complex<double>*>(output),
                            reinterpret_cast<cuda::std::complex<double>*>(work), input_halo_extents,
                            output_halo_extents, input_padding, output_padding, stream);
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
                                       const int32_t output_halo_extents[], const int32_t input_padding[],
                                       const int32_t output_padding[], hipStream_t stream) {
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
                            reinterpret_cast<float*>(work), input_halo_extents, output_halo_extents, input_padding,
                            output_padding, stream);
      break;
    case CUDECOMP_DOUBLE:
      cudecompTransposeZToY(handle, grid_desc, reinterpret_cast<double*>(input), reinterpret_cast<double*>(output),
                            reinterpret_cast<double*>(work), input_halo_extents, output_halo_extents, input_padding,
                            output_padding, stream);
      break;
    case CUDECOMP_FLOAT_COMPLEX:
      cudecompTransposeZToY(handle, grid_desc, reinterpret_cast<cuda::std::complex<float>*>(input),
                            reinterpret_cast<cuda::std::complex<float>*>(output),
                            reinterpret_cast<cuda::std::complex<float>*>(work), input_halo_extents, output_halo_extents,
                            input_padding, output_padding, stream);
      break;
    case CUDECOMP_DOUBLE_COMPLEX:
      cudecompTransposeZToY(handle, grid_desc, reinterpret_cast<cuda::std::complex<double>*>(input),
                            reinterpret_cast<cuda::std::complex<double>*>(output),
                            reinterpret_cast<cuda::std::complex<double>*>(work), input_halo_extents,
                            output_halo_extents, input_padding, output_padding, stream);
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
                                       const int32_t output_halo_extents[], const int32_t input_padding[],
                                       const int32_t output_padding[], hipStream_t stream) {
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
                            reinterpret_cast<float*>(work), input_halo_extents, output_halo_extents, input_padding,
                            output_padding, stream);
      break;
    case CUDECOMP_DOUBLE:
      cudecompTransposeYToX(handle, grid_desc, reinterpret_cast<double*>(input), reinterpret_cast<double*>(output),
                            reinterpret_cast<double*>(work), input_halo_extents, output_halo_extents, input_padding,
                            output_padding, stream);
      break;
    case CUDECOMP_FLOAT_COMPLEX:
      cudecompTransposeYToX(handle, grid_desc, reinterpret_cast<cuda::std::complex<float>*>(input),
                            reinterpret_cast<cuda::std::complex<float>*>(output),
                            reinterpret_cast<cuda::std::complex<float>*>(work), input_halo_extents, output_halo_extents,
                            input_padding, output_padding, stream);
      break;
    case CUDECOMP_DOUBLE_COMPLEX:
      cudecompTransposeYToX(handle, grid_desc, reinterpret_cast<cuda::std::complex<double>*>(input),
                            reinterpret_cast<cuda::std::complex<double>*>(output),
                            reinterpret_cast<cuda::std::complex<double>*>(work), input_halo_extents,
                            output_halo_extents, input_padding, output_padding, stream);
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
                                      int32_t dim, const int32_t padding[], hipStream_t stream) {
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
                           halo_extents, halo_periods, dim, padding, stream);
      break;
    case CUDECOMP_DOUBLE:
      cudecompUpdateHalosX(handle, grid_desc, reinterpret_cast<double*>(input), reinterpret_cast<double*>(work),
                           halo_extents, halo_periods, dim, padding, stream);
      break;
    case CUDECOMP_FLOAT_COMPLEX:
      cudecompUpdateHalosX(handle, grid_desc, reinterpret_cast<cuda::std::complex<float>*>(input),
                           reinterpret_cast<cuda::std::complex<float>*>(work), halo_extents, halo_periods, dim, padding,
                           stream);
      break;
    case CUDECOMP_DOUBLE_COMPLEX:
      cudecompUpdateHalosX(handle, grid_desc, reinterpret_cast<cuda::std::complex<double>*>(input),
                           reinterpret_cast<cuda::std::complex<double>*>(work), halo_extents, halo_periods, dim,
                           padding, stream);
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
                                      int32_t dim, const int32_t padding[], hipStream_t stream) {
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
                           halo_extents, halo_periods, dim, padding, stream);
      break;
    case CUDECOMP_DOUBLE:
      cudecompUpdateHalosY(handle, grid_desc, reinterpret_cast<double*>(input), reinterpret_cast<double*>(work),
                           halo_extents, halo_periods, dim, padding, stream);
      break;
    case CUDECOMP_FLOAT_COMPLEX:
      cudecompUpdateHalosY(handle, grid_desc, reinterpret_cast<cuda::std::complex<float>*>(input),
                           reinterpret_cast<cuda::std::complex<float>*>(work), halo_extents, halo_periods, dim, padding,
                           stream);
      break;
    case CUDECOMP_DOUBLE_COMPLEX:
      cudecompUpdateHalosY(handle, grid_desc, reinterpret_cast<cuda::std::complex<double>*>(input),
                           reinterpret_cast<cuda::std::complex<double>*>(work), halo_extents, halo_periods, dim,
                           padding, stream);
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
                                      int32_t dim, const int32_t padding[], hipStream_t stream) {
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
                           halo_extents, halo_periods, dim, padding, stream);
      break;
    case CUDECOMP_DOUBLE:
      cudecompUpdateHalosZ(handle, grid_desc, reinterpret_cast<double*>(input), reinterpret_cast<double*>(work),
                           halo_extents, halo_periods, dim, padding, stream);
      break;
    case CUDECOMP_FLOAT_COMPLEX:
      cudecompUpdateHalosZ(handle, grid_desc, reinterpret_cast<cuda::std::complex<float>*>(input),
                           reinterpret_cast<cuda::std::complex<float>*>(work), halo_extents, halo_periods, dim, padding,
                           stream);
      break;
    case CUDECOMP_DOUBLE_COMPLEX:
      cudecompUpdateHalosZ(handle, grid_desc, reinterpret_cast<cuda::std::complex<double>*>(input),
                           reinterpret_cast<cuda::std::complex<double>*>(work), halo_extents, halo_periods, dim,
                           padding, stream);
      break;
    }

  } catch (const cudecomp::BaseException& e) {
    std::cerr << e.what();
    return e.getResult();
  }
  return CUDECOMP_RESULT_SUCCESS;
}
