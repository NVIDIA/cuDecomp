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

/**
 * @file
 * @brief This file contains all public function and type declarations
 * used in the cuDecomp library.
 **/

#ifndef CUDECOMP_H
#define CUDECOMP_H

#include <string>

#include <cuda_runtime.h>
#include <mpi.h>

#define CUDECOMP_MAJOR 0
#define CUDECOMP_MINOR 3
#define CUDECOMP_PATCH 1

/**
 * @brief This enum lists the different available transpose backend options.
 */
enum cudecompTransposeCommBackend_t {
  CUDECOMP_TRANSPOSE_COMM_MPI_P2P = 1,    ///< MPI backend using peer-to-peer algorithm (i.e.,MPI_Isend/MPI_Irecv)
  CUDECOMP_TRANSPOSE_COMM_MPI_P2P_PL = 2, ///< MPI backend using peer-to-peer algorithm with pipelining
  CUDECOMP_TRANSPOSE_COMM_MPI_A2A = 3,    ///< MPI backend using MPI_Alltoallv
  CUDECOMP_TRANSPOSE_COMM_NCCL = 4,       ///< NCCL backend
  CUDECOMP_TRANSPOSE_COMM_NCCL_PL = 5,    ///< NCCL backend with pipelining
  CUDECOMP_TRANSPOSE_COMM_NVSHMEM = 6,    ///< NVSHMEM backend
  CUDECOMP_TRANSPOSE_COMM_NVSHMEM_PL = 7  ///< NVSHMEM backend with pipelining
};

/**
 * @brief This enum lists the different available halo backend options.
 */
enum cudecompHaloCommBackend_t {
  CUDECOMP_HALO_COMM_MPI = 1,             ///< MPI backend
  CUDECOMP_HALO_COMM_MPI_BLOCKING = 2,    ///< MPI backend with blocking between each peer transfer
  CUDECOMP_HALO_COMM_NCCL = 3,            ///< NCCL backend
  CUDECOMP_HALO_COMM_NVSHMEM = 4,         ///< NVSHMEM backend
  CUDECOMP_HALO_COMM_NVSHMEM_BLOCKING = 5 ///< NVSHMEM backend with blocking between each peer transfer
};

/**
 * @brief This enum defines the data types supported.
 */
enum cudecompDataType_t {
  CUDECOMP_FLOAT = -1,         ///< Single-precision real
  CUDECOMP_DOUBLE = -2,        ///< Double-precision real
  CUDECOMP_FLOAT_COMPLEX = -3, ///< Single-precision complex (interleaved)
  CUDECOMP_DOUBLE_COMPLEX = -4 ///< Double-precision complex (interleaved)
};

/**
 * @brief This enum defines the modes available for process grid autotuning.
 */
enum cudecompAutotuneGridMode_t {
  CUDECOMP_AUTOTUNE_GRID_TRANSPOSE = 0, ///< Use transpose communication to autotune process grid dimensions
  CUDECOMP_AUTOTUNE_GRID_HALO = 1       ///< Use halo communication to autotune process grid dimensions
};

/**
 * @brief This enum defines the possible values return values from cuDecomp. Most functions in the cuDecomp library
 * will return one of these values to indicate if an operation has completed successfully or an error occured.
 */
enum cudecompResult_t {
  CUDECOMP_RESULT_SUCCESS = 0,        ///< The operation completed successfully
  CUDECOMP_RESULT_INVALID_USAGE = 1,  ///< A user error, typically an invalid argument
  CUDECOMP_RESULT_NOT_SUPPORTED = 2,  ///< A user error, requesting an invalid or unsupported operation configuration
  CUDECOMP_RESULT_INTERNAL_ERROR = 3, ///< An internal library error, should be reported
  CUDECOMP_RESULT_CUDA_ERROR = 4,     ///< An error occured in the CUDA Runtime
  CUDECOMP_RESULT_CUTENSOR_ERROR = 5, ///< An error occured in the cuTENSOR library
  CUDECOMP_RESULT_MPI_ERROR = 6,      ///< An error occurred in the MPI library
  CUDECOMP_RESULT_NCCL_ERROR = 7,     ///< An error occured in the NCCL library
  CUDECOMP_RESULT_NVSHMEM_ERROR = 8   ///< An error occured in the NVSHMEM library
};

/**
 * @brief A pointer to a cuDecomp internal handle structure.
 */
typedef struct cudecompHandle* cudecompHandle_t;

/**
 * @brief A pointer to a cuDecomp internal grid descriptor structure.
 */
typedef struct cudecompGridDesc* cudecompGridDesc_t;

/**
 * @brief A data structure defining configuration options for grid descriptor creation.
 */
typedef struct {
  // Grid information
  int32_t gdims[3];      ///< dimensions of global data grid
  int32_t gdims_dist[3]; ///< dimensions of global data grid to use for distribution
  int32_t pdims[2];      ///< dimensions of process grid

  // Transpose settings
  cudecompTransposeCommBackend_t transpose_comm_backend; ///< communication backend to use for transpose communication
                                                         ///< (default: CUDECOMP_TRANSPOSE_COMM_MPI_P2P)
  bool transpose_axis_contiguous[3]; ///< flag (by axis) indicating if memory should be contiguous along pencil axis
                                     ///< (default: [false, false, false])

  // Halo settings
  cudecompHaloCommBackend_t
      halo_comm_backend; ///< communication backend to use for halo communication (default: CUDECOMP_HALO_COMM_MPI)

} cudecompGridDescConfig_t;

/**
 * @brief A data structure defining autotuning options for grid descriptor creation.
 */
typedef struct {
  // General options
  cudecompAutotuneGridMode_t grid_mode; ///< which communication (transpose/halo) to use to autotune process grid
                                        ///< (default: CUDECOMP_AUTOTUNE_GRID_TRANSPOSE)
  cudecompDataType_t dtype;             ///< datatype to use during autotuning (default: CUDECOMP_DOUBLE)
  bool allow_uneven_decompositions; ///< flag to control whether autotuning allows process grids that result in uneven
                                    ///< distributions of elements across processes (default: true)
  bool disable_nccl_backends;       ///< flag to disable NCCL backend options during autotuning (default: false)
  bool disable_nvshmem_backends;    ///< flag to disable NVSHMEM backend options during autotuning (default: false)

  // Transpose-specific options
  bool autotune_transpose_backend;    ///< flag to enable transpose backend autotuning (default: false)
  bool transpose_use_inplace_buffers; ///< flag to control whether transpose autotuning uses in-place or out-of-place
                                      ///< buffers (default: false)

  // Halo-specific options
  bool autotune_halo_backend; ///< flag to enable halo backend autotuning (default: false)
  int32_t halo_extents[3];    ///< extents for halo autotuning (default: [0, 0, 0])
  bool halo_periods[3];       ///< periodicity for halo autotuning (default: [false, false, false])
  int32_t halo_axis;          ///< which axis pencils to use for halo autotuning (default: 0, X-pencils)
} cudecompGridDescAutotuneOptions_t;

/**
 * @brief A data structure containing geometry information about a pencil data buffer.
 */
typedef struct {
  int32_t shape[3];        ///< pencil shape (in local order, including halo elements)
  int32_t lo[3];           ///< lower bound coordinates (in local order, excluding halo elements)
  int32_t hi[3];           ///< upper bound coordinates (in local order, excluding halo elements)
  int32_t order[3];        ///< data layout order (e.g. 2,1,0 means memory is ordered Z,Y,X)
  int32_t halo_extents[3]; ///< halo extents by dimension (in global order)
  int64_t size;            ///< number of elements in pencil (including halo elements)
} cudecompPencilInfo_t;

#ifdef __cplusplus
extern "C" {
#endif
// cuDecomp initialization/finalization functions
/**
 * @brief Initializes the cuDecomp library from an existing MPI communicator
 *
 * @param[out] handle A pointer to an uninitialized cutensorHandle_t
 * @param[in] mpi_comm MPI communicator containing ranks to use with cuDecomp
 *
 * @return CUDECOMP_RESULT_SUCCESS on success or error code on failure.
 */
cudecompResult_t cudecompInit(cudecompHandle_t* handle, MPI_Comm mpi_comm);

/**
 * @brief Finalizes the cuDecomp library and frees associated resources
 *
 * @param[in] handle The initialized cuDecomp library handle
 *
 * @return CUDECOMP_RESULT_SUCCESS on success or error code on failure.
 */
cudecompResult_t cudecompFinalize(cudecompHandle_t handle);

// cudecompGridDesc_t creation/manipulation functions
/**
 * @brief Creates a cuDecomp grid descriptor for use with cuDecomp functions.
 * @details This function creates a grid descriptor that cuDecomp requires for most library operations that perform
 * communication or query decomposition information. This grid descriptor contains information about how
 * the global data grid is distributed and other internal resources to facilitate communication.
 * @param[in] handle The initialized cuDecomp library handle
 * @param[out] grid_desc A pointer to an uninitialized cudecompGridDesc_t
 * @param[in,out] config A pointer to a populated cudecompGridDescConfig_t structure. This config structure defines
 * the required attributes of the decomposition. On successful exit, fields in this structure may be updated to reflect
 * autotuning results.
 * @param[in] options A pointer to cudecompGridDescAutotuneOptions_t structure. This options structure is used
 * to control the behavior of the process grid and communication backend autotuning. If autotuning is not desired, a
 * NULL pointer can be passed in for this argument.
 *
 * @return CUDECOMP_RESULT_SUCCESS on success or error code on failure.
 */
cudecompResult_t cudecompGridDescCreate(cudecompHandle_t handle, cudecompGridDesc_t* grid_desc,
                                        cudecompGridDescConfig_t* config,
                                        const cudecompGridDescAutotuneOptions_t* options);
/**
 * @brief Destroys a cuDecomp grid descriptor and frees associated resources.
 * @param[in] handle The initialized cuDecomp library handle
 * @param[in] grid_desc A cuDecomp grid descriptor
 *
 * @return CUDECOMP_RESULT_SUCCESS on success or error code on failure.
 */
cudecompResult_t cudecompGridDescDestroy(cudecompHandle_t handle, cudecompGridDesc_t grid_desc);

// cudecompGridDescConfig_t creation/manipulation functions
/**
 * @brief Initializes a cudecompGridDescConfig_t structure with default values
 * @details This function initializes entries in a cuDecomp grid descriptor configuration structure to default
 * values.
 * @param[in,out] config A pointer to cudecompGridDescConfig_t structure
 *
 * @return CUDECOMP_RESULT_SUCCESS on success or error code on failure.
 */
cudecompResult_t cudecompGridDescConfigSetDefaults(cudecompGridDescConfig_t* config);

// cudecompGridDescAutotuneOptions_t creation/manipulation functions
/**
 * @brief Initializes a cudecompGridDescAutotuneOptions_t structure with default values
 * @details This function initializes entries in a cuDecomp grid descriptor autotune options structure to default
 * values.
 * @param[in,out] options A pointer to cudecompGridDescAutotuneOptions_t structure
 *
 * @return CUDECOMP_RESULT_SUCCESS on success or error code on failure.
 */
cudecompResult_t cudecompGridDescAutotuneOptionsSetDefaults(cudecompGridDescAutotuneOptions_t* options);

// General functions
/**
 * @brief Collects geometry information about assigned pencils, by domain axis
 * @details This function queries information about the pencil assigned to the calling worker for the given axis.
 * This information is collected in a cudecompPencilInfo_t structure, which can be used to access and manipuate
 * data within the user-allocated memory buffer.
 * @param[in] handle The initialized cuDecomp library handle
 * @param[in] grid_desc A created cuDecomp grid descriptor
 * @param[out] pencil_info A pointer to a cuDecompPencilInfo_t structure
 * @param[in] axis The domain axis the desired pencil is aligned with
 * @param[in] halo_extents An array of three integers to define halo region extents of the pencil, in global order. The
 * i-th entry in this array should contain the number of halo elements (per direction) expected in the along the i-th
 * global domain axis. Symmetric halos are assumed (e.g. a value of one in halo_extents means there are 2 halo elements,
 * one element on each side). If no halo regions are necessary, a NULL pointer can be provided in place of this array.
 *
 * @return CUDECOMP_RESULT_SUCCESS on success or error code on failure.
 */
cudecompResult_t cudecompGetPencilInfo(cudecompHandle_t handle, cudecompGridDesc_t grid_desc,
                                       cudecompPencilInfo_t* pencil_info, int32_t axis, const int32_t halo_extents[]);

/**
 * @brief Queries the required transpose workspace size, in elements, for a provided grid descriptor.
 * @details This function queries the required workspace size, in elements, for transposition communication using
 * a provided grid descriptor. This workspace is required to faciliate local transposition/packing/unpacking operations,
 * or for use as a staging buffer.
 * @param[in] handle The initialized cuDecomp library handle
 * @param[in] grid_desc A cuDecomp grid descriptor
 * @param[out] workspace_size A pointer to a 64-bit integer to write the workspace size
 *
 * @return CUDECOMP_RESULT_SUCCESS on success or error code on failure.
 */
cudecompResult_t cudecompGetTransposeWorkspaceSize(cudecompHandle_t handle, cudecompGridDesc_t grid_desc,
                                                   int64_t* workspace_size);

/**
 * @brief Queries the required halo workspace size, in elements, for a provided grid descriptor.
 * @details This function queries the required workspace size, in elements, for halo communication using
 * a provided grid descriptor. This workspace is required to faciliate local packing operations for halo regions that
 * are not contiguous in memory, or for use as a staging buffer.
 * @param[in] handle The initialized cuDecomp library handle
 * @param[in] grid_desc A cuDecomp grid descriptor
 * @param[in] axis The domain axis the desired pencil is aligned with
 * @param[in] halo_extents An array of three integers to define halo region extents of the pencil, in global order. The
 * i-th entry in this array should contain the number of halo elements (per direction) expected in the along the i-th
 * global domain axis. Symmetric halos are assumed (e.g. a value of one in halo_extents means there are 2 halo elements,
 * one element on each side).
 * @param[out] workspace_size A pointer to a 64-bit integer to write the workspace size
 *
 * @return CUDECOMP_RESULT_SUCCESS on success or error code on failure.
 */
cudecompResult_t cudecompGetHaloWorkspaceSize(cudecompHandle_t handle, cudecompGridDesc_t grid_desc, int32_t axis,
                                              const int32_t halo_extents[], int64_t* workspace_size);

/**
 * @brief Function to get size (in bytes) of a cuDecomp data type
 * @param[in] dtype A cudecompDataType_t value
 * @param[out] dtype_size A pointer to a 64-bit integer to write the data type size
 *
 * @return CUDECOMP_RESULT_SUCCESS on success or error code on failure.
 */
cudecompResult_t cudecompGetDataTypeSize(cudecompDataType_t dtype, int64_t* dtype_size);

/**
 * @brief Allocation function for cuDecomp workspaces
 * @details This function should be used to allocate cuDecomp workspaces. It will select an appropriate allocator
 * based on the communication backend information found in the provided grid descriptor. At the current time, only
 * NVSHMEM-enabled backends require a special allocation (using nvshmem_malloc). This function is collective and should
 * be called on all workers to avoid deadlocks. Additionally, any memory allocated using this function is invalidated
 * if the provided grid descriptor is destroyed and care are should be taken free memory allocated using this function
 * before the provided grid descriptor is destroyed.
 * @param[in] handle The initialized cuDecomp library handle
 * @param[in] grid_desc A cuDecomp grid descriptor
 * @param[out] buffer A pointer to the allocated memory
 * @param[out] buffer_size_bytes The size of requested allocation, in bytes
 *
 * @return CUDECOMP_RESULT_SUCCESS on success or error code on failure.
 */
cudecompResult_t cudecompMalloc(cudecompHandle_t handle, cudecompGridDesc_t grid_desc, void** buffer,
                                size_t buffer_size_bytes);

/**
 * @brief Deallocation function for cuDecomp workspaces
 * @details This function should be used to deallocate memory allocate with cudecompMalloc. It will select an
 * appropriate deallocation function based on the communication backend information found in the provided grid
 * descriptor. At the current time, only NVSHMEM-enabled backends require a special deallocation (using nvshmem_free).
 * This function is collective and should be called on all workers to avoid deadlocks.
 * @param[in] handle The initialized cuDecomp library handle
 * @param[in] grid_desc A cuDecomp grid descriptor
 * @param[in] buffer A pointer to the memory to be deallocated
 *
 * @return CUDECOMP_RESULT_SUCCESS on success or error code on failure.
 */
cudecompResult_t cudecompFree(cudecompHandle_t handle, cudecompGridDesc_t grid_desc, void* buffer);

// Convenience functions
/**
 * @brief Function to get string name of transpose communication backend.
 * @param[in] comm_backend A cudecompTransposeCommBackend_t value
 *
 * @return A string representation of the transpose communication backend. Will return string "ERROR" if
 * invalid backend value is provided.
 */
const char* cudecompTransposeCommBackendToString(cudecompTransposeCommBackend_t comm_backend);

/**
 * @brief Function to get string name of halo communication backend.
 * @param[in] comm_backend A cudecompHaloCommBackend_t value
 *
 * @return A string representation of the halo communication backend. Will return string "ERROR" if
 * invalid backend value is provided.
 */
const char* cudecompHaloCommBackendToString(cudecompHaloCommBackend_t comm_backend);

/**
 * @brief Queries the configuration used to create a grid descriptor.
 * @param[in] handle The initialized cuDecomp library handle
 * @param[in] grid_desc cuDecomp grid descriptor
 * @param[out] config A pointer to a cuDecompGridDescConfig_t structure.
 *
 * @return CUDECOMP_RESULT_SUCCESS on success or error code on failure.
 */
cudecompResult_t cudecompGetGridDescConfig(cudecompHandle_t handle, cudecompGridDesc_t grid_desc,
                                           cudecompGridDescConfig_t* config);

/**
 * @brief Function to retrieve the global rank of neighboring processes
 * @param[in] handle The initialized cuDecomp library handle
 * @param[in] grid_desc A cuDecomp grid descriptor
 * @param[in] axis The domain axis the pencil is aligned with
 * @param[in] dim Which pencil dimension (global indexed) to retrieve neighboring rank
 * @param[in] displacement Displacement of neighboring rank to retrieve. For example, 1 will retrieve the +1-th neighbor
 * rank along dim, while -1 will retrieve the -1-th neighbor rank.
 * @param[in] periodic A boolean flag to indicate whether dim should be treated periodically
 * @param[out] shifted_rank A pointer to an integer to write the global rank of the requested neighbor. For non-periodic
 * cases, a value of -1 will be written if the displacement results in a position outside the global domain.
 *
 * @return CUDECOMP_RESULT_SUCCESS on success or error code on failure.
 */
cudecompResult_t cudecompGetShiftedRank(cudecompHandle_t handle, cudecompGridDesc_t grid_desc, int32_t axis,
                                        int32_t dim, int32_t displacement, bool periodic, int32_t* shifted_rank);

// Transpose functions
/**
 * @brief Function to transpose data from X-axis aligned pencils to a Y-axis aligned pencils.
 * @param[in] handle The initialized cuDecomp library handle
 * @param[in] grid_desc A cuDecomp grid descriptor
 * @param[in] input A pointer to the memory buffer to read input X-axis aligned pencil data
 * @param[out] output A pointer to the memory buffer to write output Y-axis aligned pencil data. If input and output are
 * the same, operation is performed in-place
 * @param[in] work A pointer to the transpose workspace memory
 * @param[in] dtype The cuDecomp datatype to use for the transpose operation
 * @param[in] input_halo_extents An array of three integers to define halo region extents of the input data, in global
 * order. The i-th entry in this array should contain the number of halo elements (per direction) expected in the along
 * the i-th global domain axis. Symmetric halos are assumed (e.g. a value of one in halo_extents means there are 2 halo
 * elements, one element on each side). If the input has no halo regions, a NULL pointer can be provided.
 * @param[in] output_halo_extents Similar to input_halo_extents, but for the output data. If the output has no halo
 * regions, a NULL pointer can be provided.
 * @param[in] stream CUDA stream to enqueue GPU operations into
 *
 * @return CUDECOMP_RESULT_SUCCESS on success or error code on failure.
 */
cudecompResult_t cudecompTransposeXToY(cudecompHandle_t handle, cudecompGridDesc_t grid_desc, void* input, void* output,
                                       void* work, cudecompDataType_t dtype, const int32_t input_halo_extents[],
                                       const int32_t output_halo_extents[], cudaStream_t stream);

/**
 * @brief Function to transpose data from Y-axis aligned pencils to a Z-axis aligned pencils.
 * @param[in] handle The initialized cuDecomp library handle
 * @param[in] grid_desc A cuDecomp grid descriptor
 * @param[in] input A pointer to the memory buffer to read input Y-axis aligned pencil data
 * @param[out] output A pointer to the memory buffer to write output Z-axis aligned pencil data. If input and output are
 * the same, operation is performed in-place
 * @param[in] work A pointer to the transpose workspace memory
 * @param[in] dtype The cuDecomp datatype to use for the transpose operation
 * @param[in] input_halo_extents An array of three integers to define halo region extents of the input data, in global
 * order. The i-th entry in this array should contain the number of halo elements (per direction) expected in the along
 * the i-th global domain axis. Symmetric halos are assumed (e.g. a value of one in halo_extents means there are 2 halo
 * elements, one element on each side). If the input has no halo regions, a NULL pointer can be provided.
 * @param[in] output_halo_extents Similar to input_halo_extents, but for the output data. If the output has no halo
 * regions, a NULL pointer can be provided.
 * @param[in] stream CUDA stream to enqueue GPU operations into
 *
 * @return CUDECOMP_RESULT_SUCCESS on success or error code on failure.
 */
cudecompResult_t cudecompTransposeYToZ(cudecompHandle_t handle, cudecompGridDesc_t grid_desc, void* input, void* output,
                                       void* work, cudecompDataType_t dtype, const int32_t input_halo_extents[],
                                       const int32_t output_halo_extents[], cudaStream_t stream);

/**
 * @brief Function to transpose data from Z-axis aligned pencils to a Y-axis aligned pencils.
 * @param[in] handle The initialized cuDecomp library handle
 * @param[in] grid_desc A cuDecomp grid descriptor
 * @param[in] input A pointer to the memory buffer to read input Z-axis aligned pencil data
 * @param[out] output A pointer to the memory buffer to write output Y-axis aligned pencil data. If input and output are
 * the same, operation is performed in-place
 * @param[in] work A pointer to the transpose workspace memory
 * @param[in] dtype The cuDecomp datatype to use for the transpose operation
 * @param[in] input_halo_extents An array of three integers to define halo region extents of the input data, in global
 * order. The i-th entry in this array should contain the number of halo elements (per direction) expected in the along
 * the i-th global domain axis. Symmetric halos are assumed (e.g. a value of one in halo_extents means there are 2 halo
 * elements, one element on each side). If the input has no halo regions, a NULL pointer can be provided.
 * @param[in] output_halo_extents Similar to input_halo_extents, but for the output data. If the output has no halo
 * regions, a NULL pointer can be provided.
 * @param[in] stream CUDA stream to enqueue GPU operations into
 *
 * @return CUDECOMP_RESULT_SUCCESS on success or error code on failure.
 */
cudecompResult_t cudecompTransposeZToY(cudecompHandle_t handle, cudecompGridDesc_t grid_desc, void* input, void* output,
                                       void* work, cudecompDataType_t dtype, const int32_t input_halo_extents[],
                                       const int32_t output_halo_extents[], cudaStream_t stream);

/**
 * @brief Function to transpose data from Y-axis aligned pencils to a X-axis aligned pencils.
 * @param[in] handle The initialized cuDecomp library handle
 * @param[in] grid_desc A cuDecomp grid descriptor
 * @param[in] input A pointer to the memory buffer to read input Y-axis aligned pencil data
 * @param[out] output A pointer to the memory buffer to write output X-axis aligned pencil data. If input and output are
 * the same, operation is performed in-place
 * @param[in] work A pointer to the transpose workspace memory
 * @param[in] dtype The cuDecomp datatype to use for the transpose operation
 * @param[in] input_halo_extents An array of three integers to define halo region extents of the input data, in global
 * order. The i-th entry in this array should contain the number of halo elements (per direction) expected in the along
 * the i-th global domain axis. Symmetric halos are assumed (e.g. a value of one in halo_extents means there are 2 halo
 * elements, one element on each side). If the input has no halo regions, a NULL pointer can be provided.
 * @param[in] output_halo_extents Similar to input_halo_extents, but for the output data. If the output has no halo
 * regions, a NULL pointer can be provided.
 * @param[in] stream CUDA stream to enqueue GPU operations into
 *
 * @return CUDECOMP_RESULT_SUCCESS on success or error code on failure.
 */
cudecompResult_t cudecompTransposeYToX(cudecompHandle_t handle, cudecompGridDesc_t grid_desc, void* input, void* output,
                                       void* work, cudecompDataType_t dtype, const int32_t input_halo_extents[],
                                       const int32_t output_halo_extents[], cudaStream_t stream);

// Halo functions
/**
 * @brief Function to perform halo communication of X-axis aligned pencil data
 * @param[in] handle The initialized cuDecomp library handle
 * @param[in] grid_desc A cuDecomp grid descriptor
 * @param[in,out] input A pointer to the memory buffer to read input X-axis aligned pencil data. On successful
 * completion, this buffer will contain the input X-axis aligned pencil data with the specified halo regions updated.
 * @param[in] work A pointer to the halo workspace memory
 * @param[in] dtype The cuDecomp datatype to use for the halo operation
 * @param[in] halo_extents An array of three integers to define halo region extents of the input data, in global order.
 * The i-th entry in this array should contain the number of halo elements (per direction) expected in the along the
 * i-th global domain axis. Symmetric halos are assumed (e.g. a value of one in halo_extents means there are 2 halo
 * elements, one element on each side)
 * @param[in] halo_periods An array of three booleans to define halo periodicity of the input data, in global order.
 * If the i-th entry in this array is true, the domain is treated periodically along the i-th global domain axis. A NULL
 * pointer can be provided if none of the domain axes are periodic.
 * @param[in] dim Which pencil dimension (global indexed) to perform the halo update
 * @param[in] stream CUDA stream to enqueue GPU operations into
 *
 * @return CUDECOMP_RESULT_SUCCESS on success or error code on failure.
 */
cudecompResult_t cudecompUpdateHalosX(cudecompHandle_t handle, cudecompGridDesc_t grid_desc, void* input, void* work,
                                      cudecompDataType_t dtype, const int32_t halo_extents[], const bool halo_periods[],
                                      int32_t dim, cudaStream_t stream);

/**
 * @brief Function to perform halo communication of Y-axis aligned pencil data
 * @param[in] handle The initialized cuDecomp library handle
 * @param[in] grid_desc A cuDecomp grid descriptor
 * @param[in,out] input A pointer to the memory buffer to read input Y-axis aligned pencil data. On successful
 * completion, this buffer will contain the input Y-axis aligned pencil data with the specified halo regions updated.
 * @param[in] work A pointer to the halo workspace memory
 * @param[in] dtype The cuDecomp datatype to use for the halo operation
 * @param[in] halo_extents An array of three integers to define halo region extents of the input data, in global order.
 * The i-th entry in this array should contain the number of halo elements (per direction) expected in the along the
 * i-th global domain axis. Symmetric halos are assumed (e.g. a value of one in halo_extents means there are 2 halo
 * elements, one element on each side)
 * @param[in] halo_periods An array of three booleans to define halo periodicity of the input data, in global order.
 * If the i-th entry in this array is true, the domain is treated periodically along the i-th global domain axis.
 * @param[in] dim Which pencil dimension (global indexed) to perform the halo update
 * @param[in] stream CUDA stream to enqueue GPU operations into
 *
 * @return CUDECOMP_RESULT_SUCCESS on success or error code on failure.
 */
cudecompResult_t cudecompUpdateHalosY(cudecompHandle_t handle, cudecompGridDesc_t grid_desc, void* input, void* work,
                                      cudecompDataType_t dtype, const int32_t halo_extents[], const bool halo_periods[],
                                      int32_t dim, cudaStream_t stream);

/**
 * @brief Function to perform halo communication of Z-axis aligned pencil data
 * @param[in] handle The initialized cuDecomp library handle
 * @param[in] grid_desc A cuDecomp grid descriptor
 * @param[in,out] input A pointer to the memory buffer to read input Z-axis aligned pencil data. On successful
 * completion, this buffer will contain the input Z-axis aligned pencil data with the specified halo regions updated.
 * @param[in] work A pointer to the halo workspace memory
 * @param[in] dtype The cuDecomp datatype to use for the halo operation
 * @param[in] halo_extents An array of three integers to define halo region extents of the input data, in global order.
 * The i-th entry in this array should contain the number of halo elements (per direction) expected in the along the
 * i-th global domain axis. Symmetric halos are assumed (e.g. a value of one in halo_extents means there are 2 halo
 * elements, one element on each side)
 * @param[in] halo_periods An array of three booleans to define halo periodicity of the input data, in global order.
 * If the i-th entry in this array is true, the domain is treated periodically along the i-th global domain axis.
 * @param[in] dim Which pencil dimension (global indexed) to perform the halo update
 * @param[in] stream CUDA stream to enqueue GPU operations into
 *
 * @return CUDECOMP_RESULT_SUCCESS on success or error code on failure.
 */
cudecompResult_t cudecompUpdateHalosZ(cudecompHandle_t handle, cudecompGridDesc_t grid_desc, void* input, void* work,
                                      cudecompDataType_t dtype, const int32_t halo_extents[], const bool halo_periods[],
                                      int32_t dim, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif // CUDECOMP_H
