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

#ifndef CUDECOMP_EXCEPTIONS_H
#define CUDECOMP_EXCEPTIONS_H

#include <exception>
#include <iostream>
#include <string>

#include "cudecomp.h"

// Useful defines for throwing with line/file info
#define THROW_INVALID_USAGE(msg)                                                                                       \
  do {                                                                                                                 \
    throw cudecomp::InvalidUsage(__FILE__, __LINE__, msg);                                                             \
  } while (false)
#define THROW_NOT_SUPPORTED(msg)                                                                                       \
  do {                                                                                                                 \
    throw cudecomp::NotSupported(__FILE__, __LINE__, msg);                                                             \
  } while (false)
#define THROW_INTERNAL_ERROR(msg)                                                                                      \
  do {                                                                                                                 \
    throw cudecomp::InternalError(__FILE__, __LINE__, msg);                                                            \
  } while (false)
#define THROW_CUDA_ERROR(msg)                                                                                          \
  do {                                                                                                                 \
    throw cudecomp::CudaError(__FILE__, __LINE__, msg);                                                                \
  } while (false)
#define THROW_CUTENSOR_ERROR(msg)                                                                                      \
  do {                                                                                                                 \
    throw cudecomp::CutensorError(__FILE__, __LINE__, msg);                                                            \
  } while (false)
#define THROW_MPI_ERROR(msg)                                                                                           \
  do {                                                                                                                 \
    throw cudecomp::MpiError(__FILE__, __LINE__, msg);                                                                 \
  } while (false)
#define THROW_NCCL_ERROR(msg)                                                                                          \
  do {                                                                                                                 \
    throw cudecomp::NcclError(__FILE__, __LINE__, msg);                                                                \
  } while (false)
#define THROW_NVSHMEM_ERROR(msg)                                                                                       \
  do {                                                                                                                 \
    throw cudecomp::NvshmemError(__FILE__, __LINE__, msg);                                                             \
  } while (false)

namespace cudecomp {

class BaseException : public std::exception {
public:
  BaseException(const char* file, int line, const char* generic_info, const char* extra_info = nullptr) {
    s = "CUDECOMP:ERROR: ";
    s += std::string(file) + std::string(":") + std::to_string(line) + std::string(" ");
    s += std::string(generic_info);
    if (extra_info) {
      s += std::string(" (") + std::string(extra_info) + std::string(")\n");
    } else {
      s += std::string("\n");
    }
  }

  const char* what() const throw() { return s.c_str(); }

  virtual cudecompResult_t getResult() const = 0;

private:
  std::string s;
};

class InvalidUsage : public BaseException {
public:
  InvalidUsage(const char* file, int line, const char* extra_info = nullptr)
      : BaseException(file, line, "Invalid usage.", extra_info) {};
  cudecompResult_t getResult() const override { return CUDECOMP_RESULT_INVALID_USAGE; }
};

class NotSupported : public BaseException {
public:
  NotSupported(const char* file, int line, const char* extra_info = nullptr)
      : BaseException(file, line, "Not supported.", extra_info) {};
  cudecompResult_t getResult() const override { return CUDECOMP_RESULT_NOT_SUPPORTED; }
};

class InternalError : public BaseException {
public:
  InternalError(const char* file, int line, const char* extra_info = nullptr)
      : BaseException(file, line, "Internal error.", extra_info) {};
  cudecompResult_t getResult() const override { return CUDECOMP_RESULT_INTERNAL_ERROR; }
};

class CudaError : public BaseException {
public:
  CudaError(const char* file, int line, const char* extra_info = nullptr)
      : BaseException(file, line, "CUDA error.", extra_info) {};
  cudecompResult_t getResult() const override { return CUDECOMP_RESULT_CUDA_ERROR; }
};

class CutensorError : public BaseException {
public:
  CutensorError(const char* file, int line, const char* extra_info = nullptr)
      : BaseException(file, line, "cuTENSOR error.", extra_info) {};
  cudecompResult_t getResult() const override { return CUDECOMP_RESULT_CUTENSOR_ERROR; }
};

class MpiError : public BaseException {
public:
  MpiError(const char* file, int line, const char* extra_info = nullptr)
      : BaseException(file, line, "MPI error.", extra_info) {};
  cudecompResult_t getResult() const override { return CUDECOMP_RESULT_MPI_ERROR; }
};

class NcclError : public BaseException {
public:
  NcclError(const char* file, int line, const char* extra_info = nullptr)
      : BaseException(file, line, "NCCL error.", extra_info) {};
  cudecompResult_t getResult() const override { return CUDECOMP_RESULT_NCCL_ERROR; }
};

class NvshmemError : public BaseException {
public:
  NvshmemError(const char* file, int line, const char* extra_info = nullptr)
      : BaseException(file, line, "NVSHMEM error.", extra_info) {};
  cudecompResult_t getResult() const override { return CUDECOMP_RESULT_NVSHMEM_ERROR; }
};

class NvmlError : public BaseException {
public:
  NvmlError(const char* file, int line, const char* extra_info = nullptr)
      : BaseException(file, line, "NVML error.", extra_info) {};
  cudecompResult_t getResult() const override { return CUDECOMP_RESULT_NVML_ERROR; }
};

} // namespace cudecomp

#endif // CUDECOMP_EXCEPTIONS_H
