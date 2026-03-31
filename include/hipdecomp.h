/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 The Authors.
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

/**
 * @file
 * @brief This file contains all public function and type declarations
 * used in the hipDecomp library.
 **/

#ifndef HIPDECOMP_H
#define HIPDECOMP_H

#include <hip/hip_runtime.h>
#include <mpi.h>

#define HIPDECOMP_MAJOR 0
#define HIPDECOMP_MINOR 6
#define HIPDECOMP_PATCH 1

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stdint.h>

/**
 * @brief This enum lists the different available transpose backend options.
 */
typedef enum {
  HIPDECOMP_TRANSPOSE_COMM_MPI_P2P = 1,    ///< MPI backend using peer-to-peer algorithm (i.e.,MPI_Isend/MPI_Irecv)
  HIPDECOMP_TRANSPOSE_COMM_MPI_P2P_PL = 2, ///< MPI backend using peer-to-peer algorithm with pipelining
  HIPDECOMP_TRANSPOSE_COMM_MPI_A2A = 3,    ///< MPI backend using MPI_Alltoallv
  HIPDECOMP_TRANSPOSE_COMM_NCCL = 4,       ///< NCCL backend
  HIPDECOMP_TRANSPOSE_COMM_NCCL_PL = 5,    ///< NCCL backend with pipelining
  HIPDECOMP_TRANSPOSE_COMM_NVSHMEM = 6,    ///< NVSHMEM backend
  HIPDECOMP_TRANSPOSE_COMM_NVSHMEM_PL = 7  ///< NVSHMEM backend with pipelining
} hipdecompTransposeCommBackend_t;

/**
 * @brief This enum lists the different available halo backend options.
 */
typedef enum {
  HIPDECOMP_HALO_COMM_MPI = 1,             ///< MPI backend
  HIPDECOMP_HALO_COMM_MPI_BLOCKING = 2,    ///< MPI backend with blocking between each peer transfer
  HIPDECOMP_HALO_COMM_NCCL = 3,            ///< NCCL backend
  HIPDECOMP_HALO_COMM_NVSHMEM = 4,         ///< NVSHMEM backend
  HIPDECOMP_HALO_COMM_NVSHMEM_BLOCKING = 5 ///< NVSHMEM backend with blocking between each peer transfer
} hipdecompHaloCommBackend_t;

/**
 * @brief This enum defines the data types supported.
 */
typedef enum {
  HIPDECOMP_FLOAT = -1,         ///< Single-precision real
  HIPDECOMP_DOUBLE = -2,        ///< Double-precision real
  HIPDECOMP_FLOAT_COMPLEX = -3, ///< Single-precision complex (interleaved)
  HIPDECOMP_DOUBLE_COMPLEX = -4 ///< Double-precision complex (interleaved)
} hipdecompDataType_t;

/**
 * @brief This enum defines the modes available for process grid autotuning.
 */
typedef enum {
  HIPDECOMP_AUTOTUNE_GRID_TRANSPOSE = 0, ///< Use transpose communication to autotune process grid dimensions
  HIPDECOMP_AUTOTUNE_GRID_HALO = 1       ///< Use halo communication to autotune process grid dimensions
} hipdecompAutotuneGridMode_t;

/**
 * @brief This enum defines the possible values return values from hipDecomp. Most functions in the hipDecomp library
 * will return one of these values to indicate if an operation has completed successfully or an error occured.
 */
typedef enum {
  HIPDECOMP_RESULT_SUCCESS = 0,         ///< The operation completed successfully
  HIPDECOMP_RESULT_INVALID_USAGE = 1,   ///< A user error, typically an invalid argument
  HIPDECOMP_RESULT_NOT_SUPPORTED = 2,   ///< A user error, requesting an invalid or unsupported operation configuration
  HIPDECOMP_RESULT_INTERNAL_ERROR = 3,  ///< An internal library error, should be reported
  HIPDECOMP_RESULT_HIP_ERROR = 4,       ///< An error occured in the HIP Runtime
  HIPDECOMP_RESULT_HIPTENSOR_ERROR = 5, ///< An error occured in the hipTENSOR library
  HIPDECOMP_RESULT_MPI_ERROR = 6,       ///< An error occurred in the MPI library
  HIPDECOMP_RESULT_NCCL_ERROR = 7,      ///< An error occured in the NCCL library
  HIPDECOMP_RESULT_NVSHMEM_ERROR = 8,   ///< An error occured in the NVSHMEM library
} hipdecompResult_t;

/**
 * @brief A pointer to a hipDecomp internal handle structure.
 */
typedef struct hipdecompHandle* hipdecompHandle_t;

/**
 * @brief A pointer to a hipDecomp internal grid descriptor structure.
 */
typedef struct hipdecompGridDesc* hipdecompGridDesc_t;

/**
 * @brief A data structure defining configuration options for grid descriptor creation.
 */
typedef struct {
  // Grid information
  int32_t gdims[3];      ///< dimensions of global data grid
  int32_t gdims_dist[3]; ///< dimensions of global data grid to use for distribution
  int32_t pdims[2];      ///< dimensions of process grid

  // Transpose settings
  hipdecompTransposeCommBackend_t transpose_comm_backend; ///< communication backend to use for transpose communication
                                                          ///< (default: HIPDECOMP_TRANSPOSE_COMM_MPI_P2P)
  bool transpose_axis_contiguous[3]; ///< flag (by axis) indicating if memory should be contiguous along pencil axis
                                     ///< (default: [false, false, false])
  int32_t transpose_mem_order[3][3]; ///< user-specified memory ordering by axis, overrides transpose_axis_contiguous
                                     ///< setting; first index specifies axis, second index specifies memory order
                                     ///< setting (default: unset)

  // Halo settings
  hipdecompHaloCommBackend_t
      halo_comm_backend; ///< communication backend to use for halo communication (default: HIPDECOMP_HALO_COMM_MPI)

} hipdecompGridDescConfig_t;

/**
 * @brief A data structure defining autotuning options for grid descriptor creation.
 */
typedef struct {
  // General options
  int32_t n_warmup_trials; ///< number of warmup trials to run for each tested configuration during autotuning
                           ///< (default: 3)
  int32_t n_trials;        ///< number of timed trials to run for each tested configuration during autotuning
                           ///< (default: 5)
  hipdecompAutotuneGridMode_t grid_mode; ///< which communication (transpose/halo) to use to autotune process grid
                                         ///< (default: HIPDECOMP_AUTOTUNE_GRID_TRANSPOSE)
  hipdecompDataType_t dtype;             ///< datatype to use during autotuning (default: HIPDECOMP_DOUBLE)
  bool allow_uneven_decompositions; ///< flag to control whether autotuning allows process grids that result in uneven
                                    ///< distributions of elements across processes (default: true)
  bool disable_nccl_backends;       ///< flag to disable NCCL backend options during autotuning (default: false)
  bool disable_nvshmem_backends;    ///< flag to disable NVSHMEM backend options during autotuning (default: false)
  double skip_threshold;            ///< threshold used to skip testing slow configurations; skip configuration
                         ///< if `skip_threshold * t > t_best`, where `t` is the duration of the first timed trial
                         ///< for the configuration and `t_best` is the average trial time of the current best
                         ///< configuration (default: 0.0)

  // Transpose-specific options
  bool autotune_transpose_backend;       ///< flag to enable transpose backend autotuning (default: false)
  bool transpose_use_inplace_buffers[4]; ///< flag to control whether transpose autotuning uses in-place or out-of-place
                                         ///< buffers during autotuning by transpose operation, considering
                                         ///< the following order: X-to-Y, Y-to-Z, Z-to-Y, Y-to-X
                                         ///< (default: [false, false, false, false])
  double transpose_op_weights[4]; ///< multiplicative weight to apply to trial time contribution by transpose operation
                                  ///< in the following order: X-to-Y, Y-to-Z, Z-to-Y, Y-to-X
                                  ///< (default: [1.0, 1.0, 1.0, 1.0])

  int32_t transpose_input_halo_extents[4][3];  ///< input_halo_extents argument to use during autotuning by transpose
                                               ///< operation; first index specifies operation in the following order:
                                               ///< X-to-Y, Y-to-Z, Z-to-Y, Y-to-X, second index specifies halo_extent
                                               ///< argument (default: all zeros, no halos)
  int32_t transpose_output_halo_extents[4][3]; ///< output_halo_extents argument to use during autotuning by transpose
                                               ///< operation; first index specifies operation in the following order:
                                               ///< X-to-Y, Y-to-Z, Z-to-Y, Y-to-X, second index specifies halo_extent
                                               ///< argument (default: all zeros, no halos)

  int32_t transpose_input_padding[4][3];  ///< input_padding argument to use during autotuning by transpose operation;
                                          ///< first index specifies operation in the following order: X-to-Y, Y-to-Z,
                                          ///< Z-to-Y, Y-to-X, second index specifies input_padding argument (default:
                                          ///< all zeros, no padding)
  int32_t transpose_output_padding[4][3]; ///< output_padding argument to use during autotuning by transpose operation;
                                          ///< first index specifies operation in the following order: X-to-Y, Y-to-Z,
                                          ///< Z-to-Y, Y-to-X, second index specifies input_padding argument (default:
                                          ///< all zeros, no padding)

  // Halo-specific options
  bool autotune_halo_backend; ///< flag to enable halo backend autotuning (default: false)
  int32_t halo_extents[3];    ///< extents for halo autotuning (default: [0, 0, 0])
  bool halo_periods[3];       ///< periodicity for halo autotuning (default: [false, false, false])
  int32_t halo_axis;          ///< which axis pencils to use for halo autotuning (default: 0, X-pencils)
  int32_t halo_padding[3];    ///< padding argument for halo autotuning (default: [0, 0, 0])
} hipdecompGridDescAutotuneOptions_t;

/**
 * @brief A data structure containing geometry information about a pencil data buffer.
 */
typedef struct {
  int32_t shape[3];        ///< pencil shape (in local order, including halo and padding elements)
  int32_t lo[3];           ///< lower bound coordinates (in local order, excluding halo and padding elements)
  int32_t hi[3];           ///< upper bound coordinates (in local order, excluding halo and padding elements)
  int32_t order[3];        ///< data layout order (e.g. 2,1,0 means memory is ordered Z,Y,X)
  int32_t halo_extents[3]; ///< halo extents by dimension (in global order)
  int32_t padding[3];      ///< padding by dimension (in global order)
  int64_t size;            ///< number of elements in pencil (including halo and padding elements)
} hipdecompPencilInfo_t;

// hipDecomp initialization/finalization functions
/**
 * @brief Initializes the hipDecomp library from an existing MPI communicator
 *
 * @param[out] handle A pointer to an uninitialized hipdecompHandle_t
 * @param[in] mpi_comm MPI communicator containing ranks to use with hipDecomp
 *
 * @return HIPDECOMP_RESULT_SUCCESS on success or error code on failure.
 */
hipdecompResult_t hipdecompInit(hipdecompHandle_t* handle, MPI_Comm mpi_comm);

/**
 * @brief Initializes the hipDecomp library from an existing MPI communicator
 *
 * @param[out] handle A pointer to an uninitialized hipdecompHandle_t
 * @param[in] mpi_comm_f MPI communicator, in Fortran integer format, containing ranks to use with hipDecomp
 *
 * @return HIPDECOMP_RESULT_SUCCESS on success or error code on failure.
 */
hipdecompResult_t hipdecompInit_F(hipdecompHandle_t* handle, MPI_Fint mpi_comm_f);

/**
 * @brief Finalizes the hipDecomp library and frees associated resources
 *
 * @param[in] handle The initialized hipDecomp library handle
 *
 * @return HIPDECOMP_RESULT_SUCCESS on success or error code on failure.
 */
hipdecompResult_t hipdecompFinalize(hipdecompHandle_t handle);

// hipdecompGridDesc_t creation/manipulation functions
/**
 * @brief Creates a hipDecomp grid descriptor for use with hipDecomp functions.
 * @details This function creates a grid descriptor that hipDecomp requires for most library operations that perform
 * communication or query decomposition information. This grid descriptor contains information about how
 * the global data grid is distributed and other internal resources to facilitate communication.
 * @param[in] handle The initialized hipDecomp library handle
 * @param[out] grid_desc A pointer to an uninitialized hipdecompGridDesc_t
 * @param[in,out] config A pointer to a populated hipdecompGridDescConfig_t structure. This config structure defines
 * the required attributes of the decomposition. On successful exit, fields in this structure may be updated to reflect
 * autotuning results.
 * @param[in] options A pointer to hipdecompGridDescAutotuneOptions_t structure. This options structure is used
 * to control the behavior of the process grid and communication backend autotuning. If autotuning is not desired, a
 * NULL pointer can be passed in for this argument.
 *
 * @return HIPDECOMP_RESULT_SUCCESS on success or error code on failure.
 */
hipdecompResult_t hipdecompGridDescCreate(hipdecompHandle_t handle, hipdecompGridDesc_t* grid_desc,
                                          hipdecompGridDescConfig_t* config,
                                          const hipdecompGridDescAutotuneOptions_t* options);
/**
 * @brief Destroys a hipDecomp grid descriptor and frees associated resources.
 * @param[in] handle The initialized hipDecomp library handle
 * @param[in] grid_desc A hipDecomp grid descriptor
 *
 * @return HIPDECOMP_RESULT_SUCCESS on success or error code on failure.
 */
hipdecompResult_t hipdecompGridDescDestroy(hipdecompHandle_t handle, hipdecompGridDesc_t grid_desc);

// hipdecompGridDescConfig_t creation/manipulation functions
/**
 * @brief Initializes a hipdecompGridDescConfig_t structure with default values
 * @details This function initializes entries in a hipDecomp grid descriptor configuration structure to default
 * values.
 * @param[in,out] config A pointer to hipdecompGridDescConfig_t structure
 *
 * @return HIPDECOMP_RESULT_SUCCESS on success or error code on failure.
 */
hipdecompResult_t hipdecompGridDescConfigSetDefaults(hipdecompGridDescConfig_t* config);

// hipdecompGridDescAutotuneOptions_t creation/manipulation functions
/**
 * @brief Initializes a hipdecompGridDescAutotuneOptions_t structure with default values
 * @details This function initializes entries in a hipDecomp grid descriptor autotune options structure to default
 * values.
 * @param[in,out] options A pointer to hipdecompGridDescAutotuneOptions_t structure
 *
 * @return HIPDECOMP_RESULT_SUCCESS on success or error code on failure.
 */
hipdecompResult_t hipdecompGridDescAutotuneOptionsSetDefaults(hipdecompGridDescAutotuneOptions_t* options);

// General functions
/**
 * @brief Collects geometry information about assigned pencils, by domain axis
 * @details This function queries information about the pencil assigned to the calling worker for the given axis.
 * This information is collected in a hipdecompPencilInfo_t structure, which can be used to access and manipuate
 * data within the user-allocated memory buffer.
 * @param[in] handle The initialized hipDecomp library handle
 * @param[in] grid_desc A created hipDecomp grid descriptor
 * @param[out] pencil_info A pointer to a hipDecompPencilInfo_t structure
 * @param[in] axis The domain axis the desired pencil is aligned with
 * @param[in] halo_extents An array of three integers to define halo region extents of the pencil, in global order. The
 * i-th entry in this array should contain the number of halo elements (per direction) expected in the along the i-th
 * global domain axis. Symmetric halos are assumed (e.g. a value of one in halo_extents means there are 2 halo elements,
 * one element on each side). If no halo regions are necessary, a NULL pointer can be provided in place of this array.
 * @param[in] padding An array of three integers to define padding of the pencil, in global order. The i-th entry
 * in this array should contain the number of elements to treat as padding in the i-th global domain axis. If no padding
 * is necesary, a NULL pointer can be provided in place of this array.
 *
 * @return HIPDECOMP_RESULT_SUCCESS on success or error code on failure.
 */
hipdecompResult_t hipdecompGetPencilInfo(hipdecompHandle_t handle, hipdecompGridDesc_t grid_desc,
                                         hipdecompPencilInfo_t* pencil_info, int32_t axis, const int32_t halo_extents[],
                                         const int32_t padding[]);

/**
 * @brief Queries the required transpose workspace size, in elements, for a provided grid descriptor.
 * @details This function queries the required workspace size, in elements, for transposition communication using
 * a provided grid descriptor. This workspace is required to faciliate local transposition/packing/unpacking operations,
 * or for use as a staging buffer.
 * @param[in] handle The initialized hipDecomp library handle
 * @param[in] grid_desc A hipDecomp grid descriptor
 * @param[out] workspace_size A pointer to a 64-bit integer to write the workspace size
 *
 * @return HIPDECOMP_RESULT_SUCCESS on success or error code on failure.
 */
hipdecompResult_t hipdecompGetTransposeWorkspaceSize(hipdecompHandle_t handle, hipdecompGridDesc_t grid_desc,
                                                     int64_t* workspace_size);

/**
 * @brief Queries the required halo workspace size, in elements, for a provided grid descriptor.
 * @details This function queries the required workspace size, in elements, for halo communication using
 * a provided grid descriptor. This workspace is required to faciliate local packing operations for halo regions that
 * are not contiguous in memory, or for use as a staging buffer.
 * @param[in] handle The initialized hipDecomp library handle
 * @param[in] grid_desc A hipDecomp grid descriptor
 * @param[in] axis The domain axis the desired pencil is aligned with
 * @param[in] halo_extents An array of three integers to define halo region extents of the pencil, in global order. The
 * i-th entry in this array should contain the number of halo elements (per direction) expected in the along the i-th
 * global domain axis. Symmetric halos are assumed (e.g. a value of one in halo_extents means there are 2 halo elements,
 * one element on each side).
 * @param[out] workspace_size A pointer to a 64-bit integer to write the workspace size
 *
 * @return HIPDECOMP_RESULT_SUCCESS on success or error code on failure.
 */
hipdecompResult_t hipdecompGetHaloWorkspaceSize(hipdecompHandle_t handle, hipdecompGridDesc_t grid_desc, int32_t axis,
                                                const int32_t halo_extents[], int64_t* workspace_size);

/**
 * @brief Function to get size (in bytes) of a hipDecomp data type
 * @param[in] dtype A hipdecompDataType_t value
 * @param[out] dtype_size A pointer to a 64-bit integer to write the data type size
 *
 * @return HIPDECOMP_RESULT_SUCCESS on success or error code on failure.
 */
hipdecompResult_t hipdecompGetDataTypeSize(hipdecompDataType_t dtype, int64_t* dtype_size);

/**
 * @brief Allocation function for hipDecomp workspaces
 * @details This function should be used to allocate hipDecomp workspaces. It will select an appropriate allocator
 * based on the communication backend information found in the provided grid descriptor. At the current time, only
 * NVSHMEM-enabled backends require a special allocation (using nvshmem_malloc). This function is collective and should
 * be called on all workers to avoid deadlocks. Additionally, any memory allocated using this function is invalidated
 * if the provided grid descriptor is destroyed and care are should be taken free memory allocated using this function
 * before the provided grid descriptor is destroyed.
 * @param[in] handle The initialized hipDecomp library handle
 * @param[in] grid_desc A hipDecomp grid descriptor
 * @param[out] buffer A pointer to the allocated memory
 * @param[out] buffer_size_bytes The size of requested allocation, in bytes
 *
 * @return HIPDECOMP_RESULT_SUCCESS on success or error code on failure.
 */
hipdecompResult_t hipdecompMalloc(hipdecompHandle_t handle, hipdecompGridDesc_t grid_desc, void** buffer,
                                  size_t buffer_size_bytes);

/**
 * @brief Deallocation function for hipDecomp workspaces
 * @details This function should be used to deallocate memory allocate with hipdecompMalloc. It will select an
 * appropriate deallocation function based on the communication backend information found in the provided grid
 * descriptor. At the current time, only NVSHMEM-enabled backends require a special deallocation (using nvshmem_free).
 * This function is collective and should be called on all workers to avoid deadlocks.
 * @param[in] handle The initialized hipDecomp library handle
 * @param[in] grid_desc A hipDecomp grid descriptor
 * @param[in] buffer A pointer to the memory to be deallocated
 *
 * @return HIPDECOMP_RESULT_SUCCESS on success or error code on failure.
 */
hipdecompResult_t hipdecompFree(hipdecompHandle_t handle, hipdecompGridDesc_t grid_desc, void* buffer);

// Convenience functions
/**
 * @brief Function to get string name of transpose communication backend.
 * @param[in] comm_backend A hipdecompTransposeCommBackend_t value
 *
 * @return A string representation of the transpose communication backend. Will return string "ERROR" if
 * invalid backend value is provided.
 */
const char* hipdecompTransposeCommBackendToString(hipdecompTransposeCommBackend_t comm_backend);

/**
 * @brief Function to get string name of halo communication backend.
 * @param[in] comm_backend A hipdecompHaloCommBackend_t value
 *
 * @return A string representation of the halo communication backend. Will return string "ERROR" if
 * invalid backend value is provided.
 */
const char* hipdecompHaloCommBackendToString(hipdecompHaloCommBackend_t comm_backend);

/**
 * @brief Queries the configuration used to create a grid descriptor.
 * @param[in] handle The initialized hipDecomp library handle
 * @param[in] grid_desc hipDecomp grid descriptor
 * @param[out] config A pointer to a hipDecompGridDescConfig_t structure.
 *
 * @return HIPDECOMP_RESULT_SUCCESS on success or error code on failure.
 */
hipdecompResult_t hipdecompGetGridDescConfig(hipdecompHandle_t handle, hipdecompGridDesc_t grid_desc,
                                             hipdecompGridDescConfig_t* config);

/**
 * @brief Function to retrieve the global rank of neighboring processes
 * @param[in] handle The initialized hipDecomp library handle
 * @param[in] grid_desc A hipDecomp grid descriptor
 * @param[in] axis The domain axis the pencil is aligned with
 * @param[in] dim Which pencil dimension (global indexed) to retrieve neighboring rank
 * @param[in] displacement Displacement of neighboring rank to retrieve. For example, 1 will retrieve the +1-th neighbor
 * rank along dim, while -1 will retrieve the -1-th neighbor rank.
 * @param[in] periodic A boolean flag to indicate whether dim should be treated periodically
 * @param[out] shifted_rank A pointer to an integer to write the global rank of the requested neighbor. For non-periodic
 * cases, a value of -1 will be written if the displacement results in a position outside the global domain.
 *
 * @return HIPDECOMP_RESULT_SUCCESS on success or error code on failure.
 */
hipdecompResult_t hipdecompGetShiftedRank(hipdecompHandle_t handle, hipdecompGridDesc_t grid_desc, int32_t axis,
                                          int32_t dim, int32_t displacement, bool periodic, int32_t* shifted_rank);

// Transpose functions
/**
 * @brief Function to transpose data from X-axis aligned pencils to a Y-axis aligned pencils.
 * @param[in] handle The initialized hipDecomp library handle
 * @param[in] grid_desc A hipDecomp grid descriptor
 * @param[in] input A pointer to the memory buffer to read input X-axis aligned pencil data
 * @param[out] output A pointer to the memory buffer to write output Y-axis aligned pencil data. If input and output are
 * the same, operation is performed in-place
 * @param[in] work A pointer to the transpose workspace memory
 * @param[in] dtype The hipDecomp datatype to use for the transpose operation
 * @param[in] input_halo_extents An array of three integers to define halo region extents of the input data, in global
 * order. The i-th entry in this array should contain the number of halo elements (per direction) expected in the along
 * the i-th global domain axis. Symmetric halos are assumed (e.g. a value of one in halo_extents means there are 2 halo
 * elements, one element on each side). If the input has no halo regions, a NULL pointer can be provided.
 * @param[in] output_halo_extents Similar to input_halo_extents, but for the output data. If the output has no halo
 * regions, a NULL pointer can be provided.
 * @param[in] input_padding An array of three integers to define padding of the input data, in global order. The i-th
 * entry in this array should contain the number of elements to treat as padding in the i-th global domain axis. If the
 * input has no padding, a NULL pointer can be provided.
 * @param[in] output_padding Similar to input_padding, but for the output data. If the output has no padding, a NULL
 * pointer can be provided.
 * @param[in] stream HIP stream to enqueue GPU operations into
 *
 * @return HIPDECOMP_RESULT_SUCCESS on success or error code on failure.
 */
hipdecompResult_t hipdecompTransposeXToY(hipdecompHandle_t handle, hipdecompGridDesc_t grid_desc, void* input,
                                         void* output, void* work, hipdecompDataType_t dtype,
                                         const int32_t input_halo_extents[], const int32_t output_halo_extents[],
                                         const int32_t input_padding[], const int32_t output_padding[],
                                         hipStream_t stream);

/**
 * @brief Function to transpose data from Y-axis aligned pencils to a Z-axis aligned pencils.
 * @param[in] handle The initialized hipDecomp library handle
 * @param[in] grid_desc A hipDecomp grid descriptor
 * @param[in] input A pointer to the memory buffer to read input Y-axis aligned pencil data
 * @param[out] output A pointer to the memory buffer to write output Z-axis aligned pencil data. If input and output are
 * the same, operation is performed in-place
 * @param[in] work A pointer to the transpose workspace memory
 * @param[in] dtype The hipDecomp datatype to use for the transpose operation
 * @param[in] input_halo_extents An array of three integers to define halo region extents of the input data, in global
 * order. The i-th entry in this array should contain the number of halo elements (per direction) expected in the along
 * the i-th global domain axis. Symmetric halos are assumed (e.g. a value of one in halo_extents means there are 2 halo
 * elements, one element on each side). If the input has no halo regions, a NULL pointer can be provided.
 * @param[in] output_halo_extents Similar to input_halo_extents, but for the output data. If the output has no halo
 * regions, a NULL pointer can be provided.
 * @param[in] input_padding An array of three integers to define padding of the input data, in global order. The i-th
 * entry in this array should contain the number of elements to treat as padding in the i-th global domain axis. If the
 * input has no padding, a NULL pointer can be provided.
 * @param[in] output_padding Similar to input_padding, but for the output data. If the output has no padding, a NULL
 * pointer can be provided.
 * @param[in] stream HIP stream to enqueue GPU operations into
 *
 * @return HIPDECOMP_RESULT_SUCCESS on success or error code on failure.
 */
hipdecompResult_t hipdecompTransposeYToZ(hipdecompHandle_t handle, hipdecompGridDesc_t grid_desc, void* input,
                                         void* output, void* work, hipdecompDataType_t dtype,
                                         const int32_t input_halo_extents[], const int32_t output_halo_extents[],
                                         const int32_t input_padding[], const int32_t output_padding[],
                                         hipStream_t stream);

/**
 * @brief Function to transpose data from Z-axis aligned pencils to a Y-axis aligned pencils.
 * @param[in] handle The initialized hipDecomp library handle
 * @param[in] grid_desc A hipDecomp grid descriptor
 * @param[in] input A pointer to the memory buffer to read input Z-axis aligned pencil data
 * @param[out] output A pointer to the memory buffer to write output Y-axis aligned pencil data. If input and output are
 * the same, operation is performed in-place
 * @param[in] work A pointer to the transpose workspace memory
 * @param[in] dtype The hipDecomp datatype to use for the transpose operation
 * @param[in] input_halo_extents An array of three integers to define halo region extents of the input data, in global
 * order. The i-th entry in this array should contain the number of halo elements (per direction) expected in the along
 * the i-th global domain axis. Symmetric halos are assumed (e.g. a value of one in halo_extents means there are 2 halo
 * elements, one element on each side). If the input has no halo regions, a NULL pointer can be provided.
 * @param[in] output_halo_extents Similar to input_halo_extents, but for the output data. If the output has no halo
 * regions, a NULL pointer can be provided.
 * @param[in] input_padding An array of three integers to define padding of the input data, in global order. The i-th
 * entry in this array should contain the number of elements to treat as padding in the i-th global domain axis. If the
 * input has no padding, a NULL pointer can be provided.
 * @param[in] output_padding Similar to input_padding, but for the output data. If the output has no padding, a NULL
 * pointer can be provided.
 * @param[in] stream HIP stream to enqueue GPU operations into
 *
 * @return HIPDECOMP_RESULT_SUCCESS on success or error code on failure.
 */
hipdecompResult_t hipdecompTransposeZToY(hipdecompHandle_t handle, hipdecompGridDesc_t grid_desc, void* input,
                                         void* output, void* work, hipdecompDataType_t dtype,
                                         const int32_t input_halo_extents[], const int32_t output_halo_extents[],
                                         const int32_t input_padding[], const int32_t output_padding[],
                                         hipStream_t stream);

/**
 * @brief Function to transpose data from Y-axis aligned pencils to a X-axis aligned pencils.
 * @param[in] handle The initialized hipDecomp library handle
 * @param[in] grid_desc A hipDecomp grid descriptor
 * @param[in] input A pointer to the memory buffer to read input Y-axis aligned pencil data
 * @param[out] output A pointer to the memory buffer to write output X-axis aligned pencil data. If input and output are
 * the same, operation is performed in-place
 * @param[in] work A pointer to the transpose workspace memory
 * @param[in] dtype The hipDecomp datatype to use for the transpose operation
 * @param[in] input_halo_extents An array of three integers to define halo region extents of the input data, in global
 * order. The i-th entry in this array should contain the number of halo elements (per direction) expected in the along
 * the i-th global domain axis. Symmetric halos are assumed (e.g. a value of one in halo_extents means there are 2 halo
 * elements, one element on each side). If the input has no halo regions, a NULL pointer can be provided.
 * @param[in] output_halo_extents Similar to input_halo_extents, but for the output data. If the output has no halo
 * regions, a NULL pointer can be provided.
 * @param[in] input_padding An array of three integers to define padding of the input data, in global order. The i-th
 * entry in this array should contain the number of elements to treat as padding in the i-th global domain axis. If the
 * input has no padding, a NULL pointer can be provided.
 * @param[in] output_padding Similar to input_padding, but for the output data. If the output has no padding, a NULL
 * pointer can be provided.
 * @param[in] stream HIP stream to enqueue GPU operations into
 *
 * @return HIPDECOMP_RESULT_SUCCESS on success or error code on failure.
 */
hipdecompResult_t hipdecompTransposeYToX(hipdecompHandle_t handle, hipdecompGridDesc_t grid_desc, void* input,
                                         void* output, void* work, hipdecompDataType_t dtype,
                                         const int32_t input_halo_extents[], const int32_t output_halo_extents[],
                                         const int32_t input_padding[], const int32_t output_padding[],
                                         hipStream_t stream);

// Halo functions
/**
 * @brief Function to perform halo communication of X-axis aligned pencil data
 * @param[in] handle The initialized hipDecomp library handle
 * @param[in] grid_desc A hipDecomp grid descriptor
 * @param[in,out] input A pointer to the memory buffer to read input X-axis aligned pencil data. On successful
 * completion, this buffer will contain the input X-axis aligned pencil data with the specified halo regions updated.
 * @param[in] work A pointer to the halo workspace memory
 * @param[in] dtype The hipDecomp datatype to use for the halo operation
 * @param[in] halo_extents An array of three integers to define halo region extents of the input data, in global order.
 * The i-th entry in this array should contain the number of halo elements (per direction) expected in the along the
 * i-th global domain axis. Symmetric halos are assumed (e.g. a value of one in halo_extents means there are 2 halo
 * elements, one element on each side)
 * @param[in] halo_periods An array of three booleans to define halo periodicity of the input data, in global order.
 * If the i-th entry in this array is true, the domain is treated periodically along the i-th global domain axis. A NULL
 * pointer can be provided if none of the domain axes are periodic.
 * @param[in] dim Which pencil dimension (global indexed) to perform the halo update
 * @param[in] padding An array of three integers to define padding of the input data, in global order. The i-th entry
 * in this array should contain the number of elements to treat as padding in the i-th global domain axis. If the input
 * has no padding, a NULL pointer can be provided.
 * @param[in] stream HIP stream to enqueue GPU operations into
 *
 * @return HIPDECOMP_RESULT_SUCCESS on success or error code on failure.
 */
hipdecompResult_t hipdecompUpdateHalosX(hipdecompHandle_t handle, hipdecompGridDesc_t grid_desc, void* input,
                                        void* work, hipdecompDataType_t dtype, const int32_t halo_extents[],
                                        const bool halo_periods[], int32_t dim, const int32_t padding[],
                                        hipStream_t stream);

/**
 * @brief Function to perform halo communication of Y-axis aligned pencil data
 * @param[in] handle The initialized hipDecomp library handle
 * @param[in] grid_desc A hipDecomp grid descriptor
 * @param[in,out] input A pointer to the memory buffer to read input Y-axis aligned pencil data. On successful
 * completion, this buffer will contain the input Y-axis aligned pencil data with the specified halo regions updated.
 * @param[in] work A pointer to the halo workspace memory
 * @param[in] dtype The hipDecomp datatype to use for the halo operation
 * @param[in] halo_extents An array of three integers to define halo region extents of the input data, in global order.
 * The i-th entry in this array should contain the number of halo elements (per direction) expected in the along the
 * i-th global domain axis. Symmetric halos are assumed (e.g. a value of one in halo_extents means there are 2 halo
 * elements, one element on each side)
 * @param[in] halo_periods An array of three booleans to define halo periodicity of the input data, in global order.
 * If the i-th entry in this array is true, the domain is treated periodically along the i-th global domain axis.
 * @param[in] dim Which pencil dimension (global indexed) to perform the halo update
 * @param[in] padding An array of three integers to define padding of the input data, in global order. The i-th entry
 * in this array should contain the number of elements to treat as padding in the i-th global domain axis. If the input
 * has no padding, a NULL pointer can be provided.
 * @param[in] stream HIP stream to enqueue GPU operations into
 *
 * @return HIPDECOMP_RESULT_SUCCESS on success or error code on failure.
 */
hipdecompResult_t hipdecompUpdateHalosY(hipdecompHandle_t handle, hipdecompGridDesc_t grid_desc, void* input,
                                        void* work, hipdecompDataType_t dtype, const int32_t halo_extents[],
                                        const bool halo_periods[], int32_t dim, const int32_t padding[],
                                        hipStream_t stream);

/**
 * @brief Function to perform halo communication of Z-axis aligned pencil data
 * @param[in] handle The initialized hipDecomp library handle
 * @param[in] grid_desc A hipDecomp grid descriptor
 * @param[in,out] input A pointer to the memory buffer to read input Z-axis aligned pencil data. On successful
 * completion, this buffer will contain the input Z-axis aligned pencil data with the specified halo regions updated.
 * @param[in] work A pointer to the halo workspace memory
 * @param[in] dtype The hipDecomp datatype to use for the halo operation
 * @param[in] halo_extents An array of three integers to define halo region extents of the input data, in global order.
 * The i-th entry in this array should contain the number of halo elements (per direction) expected in the along the
 * i-th global domain axis. Symmetric halos are assumed (e.g. a value of one in halo_extents means there are 2 halo
 * elements, one element on each side)
 * @param[in] halo_periods An array of three booleans to define halo periodicity of the input data, in global order.
 * If the i-th entry in this array is true, the domain is treated periodically along the i-th global domain axis.
 * @param[in] dim Which pencil dimension (global indexed) to perform the halo update
 * @param[in] padding An array of three integers to define padding of the input data, in global order. The i-th entry
 * in this array should contain the number of elements to treat as padding in the i-th global domain axis. If the input
 * has no padding, a NULL pointer can be provided.
 * @param[in] stream HIP stream to enqueue GPU operations into
 *
 * @return HIPDECOMP_RESULT_SUCCESS on success or error code on failure.
 */
hipdecompResult_t hipdecompUpdateHalosZ(hipdecompHandle_t handle, hipdecompGridDesc_t grid_desc, void* input,
                                        void* work, hipdecompDataType_t dtype, const int32_t halo_extents[],
                                        const bool halo_periods[], int32_t dim, const int32_t padding[],
                                        hipStream_t stream);

#ifdef __cplusplus
}
#endif

#endif // HIPDECOMP_H
