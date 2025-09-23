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

#ifndef CUDECOMP_NVTX_H
#define CUDECOMP_NVTX_H

#include <string>

#ifdef ENABLE_NVTX
#include <nvtx3/nvToolsExt.h>
#endif

namespace cudecomp {

// Helper class for NVTX ranges
class nvtx {
public:
  static void rangePush(const std::string& range_name) {
#ifdef ENABLE_NVTX
    static constexpr int ncolors_ = 8;
    static constexpr int colors_[ncolors_] = {0x3366CC, 0xDC3912, 0xFF9900, 0x109618,
                                              0x990099, 0x3B3EAC, 0x0099C6, 0xDD4477};
    std::hash<std::string> hash_fn;
    int color = colors_[hash_fn(range_name) % ncolors_];
    nvtxEventAttributes_t ev = {0};
    ev.version = NVTX_VERSION;
    ev.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    ev.colorType = NVTX_COLOR_ARGB;
    ev.color = color;
    ev.messageType = NVTX_MESSAGE_TYPE_ASCII;
    ev.message.ascii = range_name.c_str();
    nvtxRangePushEx(&ev);
#endif
  }

  static void rangePop() {
#ifdef ENABLE_NVTX
    nvtxRangePop();
#endif
  }
};

} // namespace cudecomp

#endif // CUDECOMP_NVTX_H
