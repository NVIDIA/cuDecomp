/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef CUDECOMP_TEST_BACKEND_UTILS_H
#define CUDECOMP_TEST_BACKEND_UTILS_H

#include <vector>

#include "cudecomp.h"

namespace cudecomp_test {

struct TransposeBackend {
  cudecompTransposeCommBackend_t backend;
  const char* name;
  const char* label;
};

struct HaloBackend {
  cudecompHaloCommBackend_t backend;
  const char* name;
  const char* label;
};

std::vector<TransposeBackend> transposeBackends();
std::vector<HaloBackend> haloBackends();

} // namespace cudecomp_test

#endif
