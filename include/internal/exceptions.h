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

#ifndef CUDECOMP_EXCEPTIONS_H
#define CUDECOMP_EXCEPTIONS_H

#include <exception>
#include <iostream>
#include <string>

#include "cudecomp.h"

// Useful defines for throwing with line/file info
#define THROW_INVALID_USAGE(msg)                                                                                       \
  do { throw cudecomp::InvalidUsage(__FILE__, __LINE__, msg); } while (false)
#define THROW_NOT_SUPPORTED(msg)                                                                                       \
  do { throw cudecomp::NotSupported(__FILE__, __LINE__, msg); } while (false)
#define THROW_INTERNAL_ERROR(msg)                                                                                      \
  do { throw cudecomp::InternalError(__FILE__, __LINE__, msg); } while (false)
#define THROW_CUDA_ERROR(msg)                                                                                          \
  do { throw cudecomp::CudaError(__FILE__, __LINE__, msg); } while (false)
#define THROW_CUTENSOR_ERROR(msg)                                                                                      \
  do { throw cudecomp::CutensorError(__FILE__, __LINE__, msg); } while (false)
#define THROW_MPI_ERROR(msg)                                                                                           \
  do { throw cudecomp::MpiError(__FILE__, __LINE__, msg); } while (false)
#define THROW_NCCL_ERROR(msg)                                                                                          \
  do { throw cudecomp::NcclError(__FILE__, __LINE__, msg); } while (false)
#define THROW_NVSHMEM_ERROR(msg)                                                                                       \
  do { throw cudecomp::NvshmemError(__FILE__, __LINE__, msg); } while (false)

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
      : BaseException(file, line, "Invalid usage.", extra_info){};
  cudecompResult_t getResult() const override { return CUDECOMP_RESULT_INVALID_USAGE; }
};

class NotSupported : public BaseException {
public:
  NotSupported(const char* file, int line, const char* extra_info = nullptr)
      : BaseException(file, line, "Not supported.", extra_info){};
  cudecompResult_t getResult() const override { return CUDECOMP_RESULT_NOT_SUPPORTED; }
};

class InternalError : public BaseException {
public:
  InternalError(const char* file, int line, const char* extra_info = nullptr)
      : BaseException(file, line, "Internal error.", extra_info){};
  cudecompResult_t getResult() const override { return CUDECOMP_RESULT_INTERNAL_ERROR; }
};

class CudaError : public BaseException {
public:
  CudaError(const char* file, int line, const char* extra_info = nullptr)
      : BaseException(file, line, "CUDA error.", extra_info){};
  cudecompResult_t getResult() const override { return CUDECOMP_RESULT_CUDA_ERROR; }
};

class CutensorError : public BaseException {
public:
  CutensorError(const char* file, int line, const char* extra_info = nullptr)
      : BaseException(file, line, "cuTENSOR error.", extra_info){};
  cudecompResult_t getResult() const override { return CUDECOMP_RESULT_CUTENSOR_ERROR; }
};

class MpiError : public BaseException {
public:
  MpiError(const char* file, int line, const char* extra_info = nullptr)
      : BaseException(file, line, "MPI error.", extra_info){};
  cudecompResult_t getResult() const override { return CUDECOMP_RESULT_MPI_ERROR; }
};

class NcclError : public BaseException {
public:
  NcclError(const char* file, int line, const char* extra_info = nullptr)
      : BaseException(file, line, "NCCL error.", extra_info){};
  cudecompResult_t getResult() const override { return CUDECOMP_RESULT_NCCL_ERROR; }
};

class NvshmemError : public BaseException {
public:
  NvshmemError(const char* file, int line, const char* extra_info = nullptr)
      : BaseException(file, line, "NVSHMEM error.", extra_info){};
  cudecompResult_t getResult() const override { return CUDECOMP_RESULT_NVSHMEM_ERROR; }
};

} // namespace cudecomp

#endif // CUDECOMP_EXCEPTIONS_H
