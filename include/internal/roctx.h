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

#ifndef HIPDECOMP_ROCTX_H
#define HIPDECOMP_ROCTX_H

#include <string>

#ifdef ENABLE_ROCTX
#include <rocprofiler-sdk-roctx/roctx.h>
#endif

namespace hipdecomp {

// Helper class for ROCTx ranges
class roctx {
public:
  static void rangePush(const std::string& range_name) {
#ifdef ENABLE_ROCTX
    roctxRangePush(range_name.c_str());
#endif
  }

  static void rangePop() {
#ifdef ENABLE_ROCTX
    roctxRangePop();
#endif
  }
};

} // namespace hipdecomp

#endif // HIPDECOMP_ROCTX_H
