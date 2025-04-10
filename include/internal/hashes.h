/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef CUDECOMP_HASHES_H
#define CUDECOMP_HASHES_H

#include <array>
#include <functional>
#include <tuple>
#include <utility>

#include "cudecomp.h"

#define MAGIC 0x9e3779b9

template <typename T>
inline void hash_combine(size_t& hash_value, const T& val) {
  hash_value ^= std::hash<T>{}(val) + MAGIC + (hash_value << 6) + (hash_value >> 2);
}

template <typename T, size_t N> struct std::hash<std::array<T, N>> {
  size_t operator()(const std::array<T, N>& array) const {
    size_t hash_value = 0;
    for (const auto& val : array) {
      hash_combine(hash_value, val);
    }
    return hash_value;
  }
};

template <typename T, size_t N> struct std::hash<T[N]> {
  size_t operator()(const T(&array)[N]) const {
    size_t hash_value = 0;
    for (size_t i = 0; i < N; ++i) {
      hash_combine(hash_value, array[i]);
    }
    return hash_value;
  }
};

template <typename U, typename V> struct std::hash<std::pair<U, V>> {
  size_t operator()(const std::pair<U, V>& pair) const {
    size_t hash_value = 0;
    hash_combine(hash_value, pair.first);
    hash_combine(hash_value, pair.second);
    return hash_value;
  }
};

template<> struct std::hash<cudecompPencilInfo_t> {
  size_t operator()(const cudecompPencilInfo_t& info) const {
    size_t hash_value = 0;
    hash_combine(hash_value, info.shape);
    hash_combine(hash_value, info.order);
    hash_combine(hash_value, info.halo_extents);
    hash_combine(hash_value, info.padding);
    return hash_value;
  }
};

template <typename Tuple, std::size_t Index = std::tuple_size<Tuple>::value - 1>
struct tuple_hasher {
  static void apply(std::size_t& hash_value, const Tuple& tuple) {
    tuple_hasher<Tuple, Index - 1>::apply(hash_value, tuple);
    hash_combine(hash_value, std::get<Index>(tuple));
  }
};

template <typename Tuple>
struct tuple_hasher<Tuple, 0> {
  static void apply(std::size_t& hash_value, const Tuple& tuple) {
    hash_combine(hash_value, std::get<0>(tuple));
  }
};

template <typename... Types> struct std::hash<std::tuple<Types...>> {
  size_t operator()(const std::tuple<Types...>& tuple) const {
    size_t hash_value = 0;
    tuple_hasher<std::tuple<Types...>>::apply(hash_value, tuple);
    return hash_value;
  }
};

#undef MAGIC

#endif // CUDECOMP_HASHES_H
