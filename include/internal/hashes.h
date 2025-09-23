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

#ifndef CUDECOMP_HASHES_H
#define CUDECOMP_HASHES_H

#include <array>
#include <functional>
#include <tuple>
#include <utility>

#include "cudecomp.h"

#define MAGIC 0x9e3779b9

template <typename T> inline void hash_combine(size_t& hash_value, const T& val) {
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
  size_t operator()(const T (&array)[N]) const {
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

template <> struct std::hash<cudecompPencilInfo_t> {
  size_t operator()(const cudecompPencilInfo_t& info) const {
    size_t hash_value = 0;
    hash_combine(hash_value, info.shape);
    hash_combine(hash_value, info.order);
    hash_combine(hash_value, info.halo_extents);
    hash_combine(hash_value, info.padding);
    return hash_value;
  }
};

template <typename Tuple, std::size_t Index = std::tuple_size<Tuple>::value - 1> struct tuple_hasher {
  static void apply(std::size_t& hash_value, const Tuple& tuple) {
    tuple_hasher<Tuple, Index - 1>::apply(hash_value, tuple);
    hash_combine(hash_value, std::get<Index>(tuple));
  }
};

template <typename Tuple> struct tuple_hasher<Tuple, 0> {
  static void apply(std::size_t& hash_value, const Tuple& tuple) { hash_combine(hash_value, std::get<0>(tuple)); }
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
