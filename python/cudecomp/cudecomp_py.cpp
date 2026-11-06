/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <cctype>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cudecomp.h"

namespace py = pybind11;

namespace {

struct PyHandle {
  cudecompHandle_t handle = nullptr;
};

struct PyGridDesc {
  explicit PyGridDesc(std::shared_ptr<PyHandle> parent) : handle(std::move(parent)) {}

  std::shared_ptr<PyHandle> handle;
  cudecompGridDesc_t grid_desc = nullptr;
};

struct CudaStreamPointer {
  std::uintptr_t ptr;
};

struct ArrayInfo {
  std::uintptr_t ptr = 0;
  bool read_only = false;
  int64_t itemsize = 0;
  bool c_contiguous = true;
  bool has_numeric_dtype = false;
  std::optional<cudecompDataType_t> dtype;
  std::vector<int64_t> shape;
};

struct PyPencilInfo {
  std::array<int32_t, 3> shape{};
  std::array<int32_t, 3> lo{};
  std::array<int32_t, 3> hi{};
  std::array<int32_t, 3> order{};
  std::array<int32_t, 3> halo_extents{};
  std::array<int32_t, 3> padding{};
  int64_t size = 0;
};

struct ExceptionTypes {
  py::object cudecomp_error;
  py::object invalid_usage_error;
  py::object not_supported_error;
  py::object internal_error;
  py::object cuda_error;
  py::object cutensor_error;
  py::object mpi_error;
  py::object nccl_error;
  py::object nvshmem_error;
  py::object nvml_error;
};

py::object make_exception(py::module_& module, const std::string& module_name, const char* name, const char* doc,
                          PyObject* base) {
  const std::string qualified_name = module_name + "." + name;
  PyObject* exception = PyErr_NewExceptionWithDoc(qualified_name.c_str(), doc, base, nullptr);
  if (exception == nullptr) { throw py::error_already_set(); }

  py::object exception_obj = py::reinterpret_steal<py::object>(exception);
  module.attr(name) = exception_obj;
  return exception_obj;
}

ExceptionTypes register_exceptions(py::module_& module) {
  std::string module_name = py::str(module.attr("__name__"));
  const std::string extension_suffix = "._cudecomp";
  if (module_name.size() >= extension_suffix.size() &&
      module_name.compare(module_name.size() - extension_suffix.size(), extension_suffix.size(), extension_suffix) ==
          0) {
    module_name.resize(module_name.size() - extension_suffix.size());
  }

  ExceptionTypes exceptions;
  exceptions.cudecomp_error = make_exception(module, module_name, "CuDecompError",
                                             "Base class for cuDecomp Python errors.", PyExc_RuntimeError);
  exceptions.invalid_usage_error =
      make_exception(module, module_name, "InvalidUsageError", "Raised for invalid cuDecomp API usage.",
                     exceptions.cudecomp_error.ptr());
  exceptions.not_supported_error =
      make_exception(module, module_name, "NotSupportedError", "Raised for unsupported cuDecomp configurations.",
                     exceptions.cudecomp_error.ptr());
  exceptions.internal_error = make_exception(module, module_name, "InternalError",
                                             "Raised for internal cuDecomp errors.", exceptions.cudecomp_error.ptr());
  exceptions.cuda_error =
      make_exception(module, module_name, "CudaError", "Raised for CUDA Runtime errors reported by cuDecomp.",
                     exceptions.cudecomp_error.ptr());
  exceptions.cutensor_error =
      make_exception(module, module_name, "CutensorError", "Raised for cuTENSOR errors reported by cuDecomp.",
                     exceptions.cudecomp_error.ptr());
  exceptions.mpi_error = make_exception(module, module_name, "MpiError", "Raised for MPI errors reported by cuDecomp.",
                                        exceptions.cudecomp_error.ptr());
  exceptions.nccl_error =
      make_exception(module, module_name, "NcclError", "Raised for NCCL errors reported by cuDecomp.",
                     exceptions.cudecomp_error.ptr());
  exceptions.nvshmem_error =
      make_exception(module, module_name, "NvshmemError", "Raised for NVSHMEM errors reported by cuDecomp.",
                     exceptions.cudecomp_error.ptr());
  exceptions.nvml_error =
      make_exception(module, module_name, "NvmlError", "Raised for NVML errors reported by cuDecomp.",
                     exceptions.cudecomp_error.ptr());
  return exceptions;
}

void check_result(cudecompResult_t result, const ExceptionTypes& exceptions) {
  const char* name = "CUDECOMP_RESULT_UNKNOWN";
  PyObject* exception = exceptions.cudecomp_error.ptr();
  switch (result) {
  case CUDECOMP_RESULT_SUCCESS: return;
  case CUDECOMP_RESULT_INVALID_USAGE:
    name = "CUDECOMP_RESULT_INVALID_USAGE";
    exception = exceptions.invalid_usage_error.ptr();
    break;
  case CUDECOMP_RESULT_NOT_SUPPORTED:
    name = "CUDECOMP_RESULT_NOT_SUPPORTED";
    exception = exceptions.not_supported_error.ptr();
    break;
  case CUDECOMP_RESULT_INTERNAL_ERROR:
    name = "CUDECOMP_RESULT_INTERNAL_ERROR";
    exception = exceptions.internal_error.ptr();
    break;
  case CUDECOMP_RESULT_CUDA_ERROR:
    name = "CUDECOMP_RESULT_CUDA_ERROR";
    exception = exceptions.cuda_error.ptr();
    break;
  case CUDECOMP_RESULT_CUTENSOR_ERROR:
    name = "CUDECOMP_RESULT_CUTENSOR_ERROR";
    exception = exceptions.cutensor_error.ptr();
    break;
  case CUDECOMP_RESULT_MPI_ERROR:
    name = "CUDECOMP_RESULT_MPI_ERROR";
    exception = exceptions.mpi_error.ptr();
    break;
  case CUDECOMP_RESULT_NCCL_ERROR:
    name = "CUDECOMP_RESULT_NCCL_ERROR";
    exception = exceptions.nccl_error.ptr();
    break;
  case CUDECOMP_RESULT_NVSHMEM_ERROR:
    name = "CUDECOMP_RESULT_NVSHMEM_ERROR";
    exception = exceptions.nvshmem_error.ptr();
    break;
  case CUDECOMP_RESULT_NVML_ERROR:
    name = "CUDECOMP_RESULT_NVML_ERROR";
    exception = exceptions.nvml_error.ptr();
    break;
  }

  const std::string message = std::string(name) + " (" + std::to_string(static_cast<int>(result)) + ")";
  PyErr_SetString(exception, message.c_str());
  throw py::error_already_set();
}

[[noreturn]] void throw_invalid_usage(const ExceptionTypes& exceptions, const std::string& message) {
  PyErr_SetString(exceptions.invalid_usage_error.ptr(), ("CUDECOMP_RESULT_INVALID_USAGE: " + message).c_str());
  throw py::error_already_set();
}

bool is_integer(py::handle value) { return PyIndex_Check(value.ptr()) && !PyBool_Check(value.ptr()); }

// pybind11 reports failed conversions as cast_error, which surfaces as a bare RuntimeError. Report them as ValueError.
template <typename T> T cast_or_value_error(py::handle value, const char* message) {
  try {
    return value.cast<T>();
  } catch (const py::cast_error&) { throw py::value_error(message); }
}

constexpr const char* kStreamRangeMessage = "stream handle must be a non-negative integer that fits in a pointer";

// Follow nvmath-python: without an explicit stream, use the current stream of the input array's package. Only CuPy
// is recognized; other providers default to stream 0. CuPy is checked in sys.modules so it is never imported here.
std::uintptr_t default_stream(py::handle input) {
#if !CUDECOMP_BUILD_CPU_ONLY
  PyObject* cupy = PyDict_GetItemString(PyImport_GetModuleDict(), "cupy");
  if (cupy != nullptr && py::isinstance(input, py::handle(cupy).attr("ndarray"))) {
    return py::handle(cupy).attr("cuda").attr("get_current_stream")().attr("ptr").cast<std::uintptr_t>();
  }
#endif
  return 0;
}

std::uintptr_t normalize_stream(py::object stream, py::handle input) {
  if (stream.is_none()) { return default_stream(input); }
  if (is_integer(stream)) { return cast_or_value_error<std::uintptr_t>(stream, kStreamRangeMessage); }
  // CUDA stream protocol (cuda.core and recent CuPy): __cuda_stream__() returns (version, handle), version 0.
  if (py::hasattr(stream, "__cuda_stream__")) {
    const py::object protocol = stream.attr("__cuda_stream__")();
    if (!PyTuple_Check(protocol.ptr()) || py::len(protocol) != 2) {
      throw py::type_error("stream.__cuda_stream__() must return a (version, handle) tuple");
    }
    const py::tuple values = py::reinterpret_borrow<py::tuple>(protocol);
    if (!is_integer(values[0]) ||
        cast_or_value_error<int64_t>(values[0], "stream.__cuda_stream__() version is out of range") != 0) {
      throw py::type_error("stream.__cuda_stream__() must report protocol version 0");
    }
    if (!is_integer(values[1])) { throw py::type_error("stream.__cuda_stream__() handle must be an integer"); }
    return cast_or_value_error<std::uintptr_t>(values[1], kStreamRangeMessage);
  }
  py::object pointer;
  try {
    pointer = stream.attr("ptr");
  } catch (py::error_already_set& exc) {
    if (!exc.matches(PyExc_AttributeError)) { throw; }
    throw py::type_error("stream must be None, an integer CUDA stream handle, an object implementing "
                         "__cuda_stream__, or an object with an integer ptr");
  }
  if (!is_integer(pointer)) { throw py::type_error("stream.ptr must be an integer CUDA stream handle"); }
  return cast_or_value_error<std::uintptr_t>(pointer, kStreamRangeMessage);
}

template <typename T, std::size_t N> void copy_array(const std::array<T, N>& src, T (&dst)[N]) {
  for (std::size_t i = 0; i < N; ++i) {
    dst[i] = src[i];
  }
}

template <typename T, std::size_t Rows, std::size_t Cols>
void copy_matrix(const std::array<std::array<T, Cols>, Rows>& src, T (&dst)[Rows][Cols]) {
  for (std::size_t row = 0; row < Rows; ++row) {
    for (std::size_t col = 0; col < Cols; ++col) {
      dst[row][col] = src[row][col];
    }
  }
}

// Keep the native struct alive for as long as any NumPy view refers to its fixed-size storage.
template <typename T, std::size_t N> py::array_t<T> make_array_view(T (&values)[N], py::handle owner) {
  return py::array_t<T>({N}, values, owner);
}

template <typename T, std::size_t Rows, std::size_t Cols>
py::array_t<T> make_array_view(T (&values)[Rows][Cols], py::handle owner) {
  return py::array_t<T>({Rows, Cols}, &values[0][0], owner);
}

std::optional<cudecompDataType_t> data_type_from_kind(char kind, int64_t itemsize) {
  if (kind == 'f' && itemsize == 4) { return CUDECOMP_FLOAT; }
  if (kind == 'f' && itemsize == 8) { return CUDECOMP_DOUBLE; }
  if (kind == 'c' && itemsize == 8) { return CUDECOMP_FLOAT_COMPLEX; }
  if (kind == 'c' && itemsize == 16) { return CUDECOMP_DOUBLE_COMPLEX; }
  return std::nullopt;
}

void set_array_dtype(ArrayInfo& info, char kind, int64_t itemsize) {
  info.itemsize = itemsize;
  info.dtype = data_type_from_kind(kind, itemsize);
  // Unsupported numeric types cannot be reinterpreted by supplying an explicit dtype.
  info.has_numeric_dtype = kind == 'f' || kind == 'c' || kind == 'i' || kind == 'u' || kind == 'b';
}

#if !CUDECOMP_BUILD_CPU_ONLY
void parse_typestr(const std::string& typestr, ArrayInfo& info, const ExceptionTypes& exceptions) {
  if (!typestr.empty() && typestr.front() == '>') {
    throw_invalid_usage(exceptions, "__cuda_array_interface__['typestr'] must not use big-endian byte order");
  }

  std::size_t digits_begin = typestr.size();
  while (digits_begin > 0 && std::isdigit(static_cast<unsigned char>(typestr[digits_begin - 1]))) {
    --digits_begin;
  }

  int64_t itemsize = 0;
  const auto result = std::from_chars(typestr.data() + digits_begin, typestr.data() + typestr.size(), itemsize);
  if (result.ec != std::errc{} || itemsize <= 0) {
    throw py::value_error("__cuda_array_interface__['typestr'] has invalid item size");
  }
  const char kind = digits_begin == 0 ? '\0' : typestr[digits_begin - 1];
  set_array_dtype(info, kind, itemsize);
}

bool strides_are_c_contiguous(const std::vector<int64_t>& shape, const std::vector<int64_t>& strides,
                              int64_t itemsize) {
  int64_t expected_stride = itemsize;
  for (std::size_t reverse_i = shape.size(); reverse_i > 0; --reverse_i) {
    const std::size_t i = reverse_i - 1;
    if (shape[i] <= 1) { continue; }
    if (strides[i] != expected_stride) { return false; }
    expected_stride *= shape[i];
  }
  return true;
}

ArrayInfo array_info(py::object array, const char* name, const ExceptionTypes& exceptions) {
  py::object interface_object;
  try {
    // Providers may construct the metadata in a property getter. Read it once.
    interface_object = array.attr("__cuda_array_interface__");
  } catch (py::error_already_set& exc) {
    if (!exc.matches(PyExc_AttributeError)) { throw; }
    throw py::type_error(std::string("CUDA build requires an array exposing __cuda_array_interface__ for '") + name +
                         "'; NumPy host arrays are not supported.");
  }

  if (!PyDict_Check(interface_object.ptr())) { throw py::type_error("__cuda_array_interface__ must be a dict"); }
  py::dict interface = py::reinterpret_borrow<py::dict>(interface_object);

  const py::object data_object = interface["data"];
  const py::object shape_object = interface["shape"];
  const py::object typestr_object = interface["typestr"];
  const py::object version_object = interface["version"];
  const int64_t version =
      cast_or_value_error<int64_t>(version_object, "__cuda_array_interface__['version'] must be an integer");
  if (version < 0 || version > 3) {
    throw py::value_error("__cuda_array_interface__['version'] must be between 0 and 3");
  }
  if (interface.contains("mask") && !interface["mask"].is_none()) {
    throw_invalid_usage(exceptions, "__cuda_array_interface__['mask'] must be None; masked arrays are not supported");
  }
  const auto data = cast_or_value_error<std::pair<std::uintptr_t, bool>>(
      data_object, "__cuda_array_interface__['data'] must be a (pointer, read_only) tuple");

  ArrayInfo info;
  info.ptr = data.first;
  info.read_only = data.second;
  info.shape =
      cast_or_value_error<std::vector<int64_t>>(shape_object,
                                                "__cuda_array_interface__['shape'] must be a tuple of integers");
  parse_typestr(cast_or_value_error<std::string>(typestr_object,
                                                 "__cuda_array_interface__['typestr'] must be a string"),
                info, exceptions);

  const py::object strides_object = interface.contains("strides") ? py::object(interface["strides"]) : py::none();
  if (!strides_object.is_none()) {
    const auto strides = cast_or_value_error<std::vector<int64_t>>(
        strides_object, "__cuda_array_interface__['strides'] must be None or a tuple of integers");
    if (strides.size() != info.shape.size()) {
      throw py::value_error("__cuda_array_interface__['strides'] must match shape length");
    }
    info.c_contiguous = strides_are_c_contiguous(info.shape, strides, info.itemsize);
  }
  return info;
}

#else
ArrayInfo array_info(py::object array, const char* name, const ExceptionTypes& exceptions) {
  if (!py::isinstance<py::array>(array)) {
    throw py::type_error(std::string("CPU build requires a NumPy host array for '") + name +
                         "'; CUDA arrays are not supported.");
  }

  py::array numpy_array = py::reinterpret_borrow<py::array>(array);
  const py::dtype dtype = numpy_array.dtype();
  if (!py::cast<bool>(dtype.attr("isnative"))) {
    throw_invalid_usage(exceptions, "NumPy array dtype must use native byte order");
  }

  ArrayInfo info;
  info.ptr = reinterpret_cast<std::uintptr_t>(numpy_array.data());
  info.read_only = !numpy_array.writeable();
  set_array_dtype(info, dtype.kind(), dtype.itemsize());
  info.c_contiguous = (numpy_array.flags() & py::array::c_style) != 0;
  info.shape.reserve(static_cast<std::size_t>(numpy_array.ndim()));
  for (py::ssize_t i = 0; i < numpy_array.ndim(); ++i) {
    info.shape.push_back(numpy_array.shape(i));
  }

  return info;
}
#endif

cudecompDataType_t infer_data_type(std::optional<cudecompDataType_t> dtype, const ArrayInfo& input) {
  if (dtype) { return *dtype; }
  if (!input.dtype) { throw py::type_error("cannot infer a supported dtype from the input array"); }
  return *input.dtype;
}

py::tuple make_int32_tuple(const std::array<int32_t, 3>& values) {
  return py::make_tuple(values[0], values[1], values[2]);
}

PyPencilInfo normalize_pencil_info(const cudecompPencilInfo_t& raw_info) {
  PyPencilInfo info;
  info.size = raw_info.size;

  for (std::size_t i = 0; i < 3; ++i) {
    const std::size_t raw_i = 2 - i;
    info.shape[i] = raw_info.shape[raw_i];
    info.lo[i] = raw_info.lo[raw_i];
    info.hi[i] = raw_info.hi[raw_i];
    info.order[i] = raw_info.order[raw_i];
    info.halo_extents[i] = raw_info.halo_extents[i];
    info.padding[i] = raw_info.padding[i];
  }
  return info;
}

py::object array_module_dtype_for_data_type(py::object array_module, cudecompDataType_t dtype) {
  switch (dtype) {
  case CUDECOMP_FLOAT: return array_module.attr("float32");
  case CUDECOMP_DOUBLE: return array_module.attr("float64");
  case CUDECOMP_FLOAT_COMPLEX: return array_module.attr("complex64");
  case CUDECOMP_DOUBLE_COMPLEX: return array_module.attr("complex128");
  }
  throw py::type_error("dtype must be a supported cuDecomp DataType");
}

using TransposeFunction = cudecompResult_t (*)(cudecompHandle_t, cudecompGridDesc_t, void*, void*, void*,
                                               cudecompDataType_t, const int32_t*, const int32_t*, const int32_t*,
                                               const int32_t*, cudaStream_t);

struct TransposeSpec {
  const char* name;
  int32_t input_axis;
  int32_t output_axis;
  TransposeFunction function;
};

const TransposeSpec kTransposeXToY{"transpose_x_to_y", 0, 1, cudecompTransposeXToY};
const TransposeSpec kTransposeYToZ{"transpose_y_to_z", 1, 2, cudecompTransposeYToZ};
const TransposeSpec kTransposeZToY{"transpose_z_to_y", 2, 1, cudecompTransposeZToY};
const TransposeSpec kTransposeYToX{"transpose_y_to_x", 1, 0, cudecompTransposeYToX};

using HaloFunction = cudecompResult_t (*)(cudecompHandle_t, cudecompGridDesc_t, void*, void*, cudecompDataType_t,
                                          const int32_t*, const bool*, int32_t, const int32_t*, cudaStream_t);

struct HaloSpec {
  int32_t axis;
  HaloFunction function;
};

const HaloSpec kUpdateHalosX{0, cudecompUpdateHalosX};
const HaloSpec kUpdateHalosY{1, cudecompUpdateHalosY};
const HaloSpec kUpdateHalosZ{2, cudecompUpdateHalosZ};

std::string shape_string(const std::vector<int64_t>& shape) { return py::str(py::tuple(py::cast(shape))); }

void validate_pencil_array(const ArrayInfo& info, const cudecompPencilInfo_t& pencil_info, cudecompDataType_t dtype,
                           int64_t dtype_size, const char* name, bool output, const ExceptionTypes& exceptions) {
  if ((info.dtype && *info.dtype != dtype) || (!info.dtype && info.has_numeric_dtype)) {
    throw_invalid_usage(exceptions, "dtype does not match array dtype");
  }

  const std::vector<int64_t> expected_shape = {pencil_info.shape[2], pencil_info.shape[1], pencil_info.shape[0]};
  if (info.shape != expected_shape) {
    throw_invalid_usage(exceptions, std::string(name) + " shape " + shape_string(info.shape) +
                                        " is not compatible with expected C-order shape " +
                                        shape_string(expected_shape));
  }
  if (!info.c_contiguous) { throw_invalid_usage(exceptions, std::string(name) + " must be C-contiguous"); }
  if (output && info.read_only) { throw_invalid_usage(exceptions, std::string(name) + " must be writable"); }
  if (!info.dtype && info.itemsize != dtype_size) {
    throw_invalid_usage(exceptions, std::string(name) + " item size does not match dtype size");
  }
}

py::object allocate_default_transpose_output(py::object input, const cudecompPencilInfo_t& output_info,
                                             cudecompDataType_t dtype, size_t output_nbytes, std::uintptr_t stream,
                                             const TransposeSpec& spec) {
#if CUDECOMP_BUILD_CPU_ONLY
  py::object numpy = py::module_::import("numpy");
  return numpy.attr("empty")(py::make_tuple(output_info.shape[2], output_info.shape[1], output_info.shape[0]),
                             py::arg("dtype") = array_module_dtype_for_data_type(numpy, dtype), py::arg("order") = "C");
#else
  py::object cupy = py::module_::import("cupy");

  if (!py::isinstance(input, cupy.attr("ndarray"))) {
    throw py::type_error(std::string(spec.name) + " with out=None requires a CuPy input");
  }

  py::object cuda = cupy.attr("cuda");
  // Use the default pool rather than cupy.cuda.alloc: stream-ordered allocators (malloc_async, MemoryAsyncPool)
  // keep only a weakref to the allocation stream and fall back to the current stream at free time. The
  // ExternalStream wrapper below is temporary, so frees would lose ordering with the operation stream. The
  // default pool keys its arenas by raw stream pointer and holds no stream object.
  py::object allocator = cupy.attr("get_default_memory_pool")().attr("malloc");
  py::object previous_stream = cuda.attr("get_current_stream")();
  py::object allocation_stream;
  py::object stream_type = cuda.attr("Stream");
  if (py::hasattr(stream_type, "from_external")) {
    allocation_stream = stream_type.attr("from_external")(py::cast(CudaStreamPointer{stream}));
  } else {
    allocation_stream = cuda.attr("ExternalStream")(py::int_(stream));
  }

  // CuPy context managers rewind their stack, which can differ from the current stream after .use().
  // Save and restore the actual current stream so allocation is ordered with the native operation.
  py::object allocation;
  try {
    allocation_stream.attr("use")();
    allocation = allocator(output_nbytes);
  } catch (...) {
    previous_stream.attr("use")();
    throw;
  }
  previous_stream.attr("use")();

  return cupy.attr("ndarray")(py::make_tuple(output_info.shape[2], output_info.shape[1], output_info.shape[0]),
                              py::arg("dtype") = array_module_dtype_for_data_type(cupy, dtype),
                              py::arg("memptr") = allocation, py::arg("order") = "C");
#endif
}

py::object transpose_operation(PyGridDesc& grid_desc, py::object input, py::object output,
                               std::optional<cudecompDataType_t> dtype_object,
                               const std::optional<std::array<int32_t, 3>>& input_halo_extents,
                               const std::optional<std::array<int32_t, 3>>& output_halo_extents,
                               const std::optional<std::array<int32_t, 3>>& input_padding,
                               const std::optional<std::array<int32_t, 3>>& output_padding, py::object stream,
                               const TransposeSpec& spec, const ExceptionTypes& exceptions) {
  cudecompPencilInfo_t input_info, output_info;
  // Metadata queries are host-only; keep the GIL until the actual operation.
  cudecompResult_t result =
      cudecompGetPencilInfo(grid_desc.handle->handle, grid_desc.grid_desc, &input_info, spec.input_axis,
                            input_halo_extents ? input_halo_extents->data() : nullptr,
                            input_padding ? input_padding->data() : nullptr);
  check_result(result, exceptions);
  result = cudecompGetPencilInfo(grid_desc.handle->handle, grid_desc.grid_desc, &output_info, spec.output_axis,
                                 output_halo_extents ? output_halo_extents->data() : nullptr,
                                 output_padding ? output_padding->data() : nullptr);
  check_result(result, exceptions);

  const ArrayInfo input_array_info = array_info(input, "input", exceptions);
  const cudecompDataType_t dtype = infer_data_type(dtype_object, input_array_info);
  int64_t dtype_size = 0;
  result = cudecompGetDataTypeSize(dtype, &dtype_size);
  check_result(result, exceptions);
  validate_pencil_array(input_array_info, input_info, dtype, dtype_size, "input", false, exceptions);

  const std::uintptr_t stream_ptr = normalize_stream(stream, input);
  void* output_ptr = nullptr;
  if (output.is_none()) {
    output = allocate_default_transpose_output(input, output_info, dtype,
                                               static_cast<size_t>(output_info.size) * static_cast<size_t>(dtype_size),
                                               stream_ptr, spec);
    // The allocator establishes the output dtype, layout, and writability; only its pointer is needed.
#if CUDECOMP_BUILD_CPU_ONLY
    output_ptr = py::cast<py::array>(output).mutable_data();
#else
    output_ptr = reinterpret_cast<void*>(py::cast<std::uintptr_t>(output.attr("data").attr("ptr")));
#endif
  } else {
    const ArrayInfo output_array_info = array_info(output, "out", exceptions);
    validate_pencil_array(output_array_info, output_info, dtype, dtype_size, "out", true, exceptions);
    output_ptr = reinterpret_cast<void*>(output_array_info.ptr);
  }

  void* input_ptr = reinterpret_cast<void*>(input_array_info.ptr);
  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream_ptr);

  {
    py::gil_scoped_release release;
    result =
        spec.function(grid_desc.handle->handle, grid_desc.grid_desc, input_ptr, output_ptr, CUDECOMP_WORKSPACE_AUTO,
                      dtype, input_halo_extents.has_value() ? input_halo_extents->data() : nullptr,
                      output_halo_extents.has_value() ? output_halo_extents->data() : nullptr,
                      input_padding.has_value() ? input_padding->data() : nullptr,
                      output_padding.has_value() ? output_padding->data() : nullptr, cuda_stream);
  }
  check_result(result, exceptions);
  return output;
}

py::object halo_operation(PyGridDesc& grid_desc, py::object input, const std::array<int32_t, 3>& halo_extents,
                          int32_t dim, std::optional<cudecompDataType_t> dtype_object,
                          const std::optional<std::array<bool, 3>>& halo_periods,
                          const std::optional<std::array<int32_t, 3>>& padding, py::object stream, const HaloSpec& spec,
                          const ExceptionTypes& exceptions) {
  cudecompPencilInfo_t input_info;
  cudecompResult_t result = cudecompGetPencilInfo(grid_desc.handle->handle, grid_desc.grid_desc, &input_info, spec.axis,
                                                  halo_extents.data(), padding ? padding->data() : nullptr);
  check_result(result, exceptions);

  const ArrayInfo input_array_info = array_info(input, "input", exceptions);
  const cudecompDataType_t dtype = infer_data_type(dtype_object, input_array_info);
  int64_t dtype_size = 0;
  result = cudecompGetDataTypeSize(dtype, &dtype_size);
  check_result(result, exceptions);

  validate_pencil_array(input_array_info, input_info, dtype, dtype_size, "input", true, exceptions);

  void* input_ptr = reinterpret_cast<void*>(input_array_info.ptr);
  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(normalize_stream(stream, input));

  {
    py::gil_scoped_release release;
    result = spec.function(grid_desc.handle->handle, grid_desc.grid_desc, input_ptr, CUDECOMP_WORKSPACE_AUTO, dtype,
                           halo_extents.data(), halo_periods.has_value() ? halo_periods->data() : nullptr, dim,
                           padding.has_value() ? padding->data() : nullptr, cuda_stream);
  }
  check_result(result, exceptions);
  return input;
}

} // namespace

PYBIND11_MODULE(_cudecomp, module) {
  py::class_<CudaStreamPointer>(module, "_CudaStreamPointer")
      .def("__cuda_stream__", [](const CudaStreamPointer& stream) { return py::make_tuple(0, stream.ptr); });
  module.doc() = "Python bindings for cuDecomp.";
  module.attr("__version__") = std::to_string(CUDECOMP_MAJOR) + "." + std::to_string(CUDECOMP_MINOR) + "." +
                               std::to_string(CUDECOMP_PATCH) + (CUDECOMP_BUILD_CPU_ONLY ? "+cpu" : "");

  py::enum_<cudecompTransposeCommBackend_t>(module, "TransposeCommBackend", py::arithmetic())
      .value("MPI_P2P", CUDECOMP_TRANSPOSE_COMM_MPI_P2P)
      .value("MPI_P2P_PL", CUDECOMP_TRANSPOSE_COMM_MPI_P2P_PL)
      .value("MPI_A2A", CUDECOMP_TRANSPOSE_COMM_MPI_A2A)
      .value("NCCL", CUDECOMP_TRANSPOSE_COMM_NCCL)
      .value("NCCL_PL", CUDECOMP_TRANSPOSE_COMM_NCCL_PL)
      .value("NVSHMEM", CUDECOMP_TRANSPOSE_COMM_NVSHMEM)
      .value("NVSHMEM_PL", CUDECOMP_TRANSPOSE_COMM_NVSHMEM_PL)
      .value("NVSHMEM_SM", CUDECOMP_TRANSPOSE_COMM_NVSHMEM_SM);

  py::enum_<cudecompHaloCommBackend_t>(module, "HaloCommBackend", py::arithmetic())
      .value("MPI", CUDECOMP_HALO_COMM_MPI)
      .value("MPI_BLOCKING", CUDECOMP_HALO_COMM_MPI_BLOCKING)
      .value("NCCL", CUDECOMP_HALO_COMM_NCCL)
      .value("NVSHMEM", CUDECOMP_HALO_COMM_NVSHMEM)
      .value("NVSHMEM_BLOCKING", CUDECOMP_HALO_COMM_NVSHMEM_BLOCKING);

  py::enum_<cudecompDataType_t>(module, "DataType", py::arithmetic())
      .value("FLOAT", CUDECOMP_FLOAT)
      .value("DOUBLE", CUDECOMP_DOUBLE)
      .value("FLOAT_COMPLEX", CUDECOMP_FLOAT_COMPLEX)
      .value("DOUBLE_COMPLEX", CUDECOMP_DOUBLE_COMPLEX);

  py::enum_<cudecompAutotuneGridMode_t>(module, "AutotuneGridMode", py::arithmetic())
      .value("TRANSPOSE", CUDECOMP_AUTOTUNE_GRID_TRANSPOSE)
      .value("HALO", CUDECOMP_AUTOTUNE_GRID_HALO);

  py::enum_<cudecompRankOrder_t>(module, "RankOrder", py::arithmetic())
      .value("DEFAULT", CUDECOMP_RANK_ORDER_DEFAULT)
      .value("ROW_MAJOR", CUDECOMP_RANK_ORDER_ROW_MAJOR)
      .value("COL_MAJOR", CUDECOMP_RANK_ORDER_COL_MAJOR);

  py::enum_<cudecompResult_t>(module, "Result", py::arithmetic())
      .value("SUCCESS", CUDECOMP_RESULT_SUCCESS)
      .value("INVALID_USAGE", CUDECOMP_RESULT_INVALID_USAGE)
      .value("NOT_SUPPORTED", CUDECOMP_RESULT_NOT_SUPPORTED)
      .value("INTERNAL_ERROR", CUDECOMP_RESULT_INTERNAL_ERROR)
      .value("CUDA_ERROR", CUDECOMP_RESULT_CUDA_ERROR)
      .value("CUTENSOR_ERROR", CUDECOMP_RESULT_CUTENSOR_ERROR)
      .value("MPI_ERROR", CUDECOMP_RESULT_MPI_ERROR)
      .value("NCCL_ERROR", CUDECOMP_RESULT_NCCL_ERROR)
      .value("NVSHMEM_ERROR", CUDECOMP_RESULT_NVSHMEM_ERROR)
      .value("NVML_ERROR", CUDECOMP_RESULT_NVML_ERROR);

  const ExceptionTypes exceptions = register_exceptions(module);
  module.def(
      "get_data_type_size",
      [exceptions](cudecompDataType_t dtype) {
        int64_t dtype_size = 0;
        const cudecompResult_t result = cudecompGetDataTypeSize(dtype, &dtype_size);
        check_result(result, exceptions);
        return dtype_size;
      },
      py::arg("dtype"), "Return the size in bytes of a cuDecomp data type.");

  module.def("transpose_comm_backend_to_string", &cudecompTransposeCommBackendToString, py::arg("backend"),
             "Return the C API string for a transpose communication backend.");

  module.def("halo_comm_backend_to_string", &cudecompHaloCommBackendToString, py::arg("backend"),
             "Return the C API string for a halo communication backend.");

  py::class_<cudecompGridDescConfig_t>(
      module, "GridDescConfig",
      "Mutable grid configuration. Array fields are writable NumPy views retaining this object's storage. "
      "Whole-field assignment accepts fixed-size tuples, lists, or NumPy arrays; view edits use NumPy assignment "
      "rules.")
      .def(py::init([exceptions]() {
             cudecompGridDescConfig_t config;
             check_result(cudecompGridDescConfigSetDefaults(&config), exceptions);
             return config;
           }),
           "Initialize a grid descriptor config with the C API defaults.")
      .def_property(
          "gdims",
          [](py::object self) {
            auto& config = self.cast<cudecompGridDescConfig_t&>();
            return make_array_view(config.gdims, self);
          },
          py::cpp_function([](cudecompGridDescConfig_t& config,
                              const std::array<int32_t, 3>& value) { copy_array(value, config.gdims); },
                           py::arg("self"), py::arg("value").noconvert()))
      .def_property(
          "gdims_dist",
          [](py::object self) {
            auto& config = self.cast<cudecompGridDescConfig_t&>();
            return make_array_view(config.gdims_dist, self);
          },
          py::cpp_function([](cudecompGridDescConfig_t& config,
                              const std::array<int32_t, 3>& value) { copy_array(value, config.gdims_dist); },
                           py::arg("self"), py::arg("value").noconvert()))
      .def_property(
          "pdims",
          [](py::object self) {
            auto& config = self.cast<cudecompGridDescConfig_t&>();
            return make_array_view(config.pdims, self);
          },
          py::cpp_function([](cudecompGridDescConfig_t& config,
                              const std::array<int32_t, 2>& value) { copy_array(value, config.pdims); },
                           py::arg("self"), py::arg("value").noconvert()))
      .def_readwrite("rank_order", &cudecompGridDescConfig_t::rank_order)
      .def_readwrite("transpose_comm_backend", &cudecompGridDescConfig_t::transpose_comm_backend)
      .def_property(
          "transpose_axis_contiguous",
          [](py::object self) {
            auto& config = self.cast<cudecompGridDescConfig_t&>();
            return make_array_view(config.transpose_axis_contiguous, self);
          },
          [](cudecompGridDescConfig_t& config, const std::array<bool, 3>& value) {
            copy_array(value, config.transpose_axis_contiguous);
          })
      .def_property(
          "transpose_mem_order",
          [](py::object self) {
            auto& config = self.cast<cudecompGridDescConfig_t&>();
            return make_array_view(config.transpose_mem_order, self);
          },
          py::cpp_function(
              [](cudecompGridDescConfig_t& config, const std::array<std::array<int32_t, 3>, 3>& value) {
                copy_matrix(value, config.transpose_mem_order);
              },
              py::arg("self"), py::arg("value").noconvert()))
      .def_readwrite("halo_comm_backend", &cudecompGridDescConfig_t::halo_comm_backend);

  py::class_<cudecompGridDescAutotuneOptions_t>(
      module, "GridDescAutotuneOptions",
      "Mutable autotune options. Array fields are writable NumPy views retaining this object's storage. "
      "Whole-field assignment accepts fixed-size tuples, lists, or NumPy arrays; view edits use NumPy assignment "
      "rules.")
      .def(py::init([exceptions]() {
             cudecompGridDescAutotuneOptions_t options;
             check_result(cudecompGridDescAutotuneOptionsSetDefaults(&options), exceptions);
             return options;
           }),
           "Initialize grid descriptor autotune options with the C API defaults.")
      .def_property(
          "n_warmup_trials", [](const cudecompGridDescAutotuneOptions_t& options) { return options.n_warmup_trials; },
          py::cpp_function([](cudecompGridDescAutotuneOptions_t& options,
                              int32_t value) { options.n_warmup_trials = value; },
                           py::arg("self"), py::arg("value").noconvert()))
      .def_property(
          "n_trials", [](const cudecompGridDescAutotuneOptions_t& options) { return options.n_trials; },
          py::cpp_function([](cudecompGridDescAutotuneOptions_t& options, int32_t value) { options.n_trials = value; },
                           py::arg("self"), py::arg("value").noconvert()))
      .def_readwrite("grid_mode", &cudecompGridDescAutotuneOptions_t::grid_mode)
      .def_readwrite("dtype", &cudecompGridDescAutotuneOptions_t::dtype)
      .def_readwrite("allow_uneven_decompositions", &cudecompGridDescAutotuneOptions_t::allow_uneven_decompositions)
      .def_readwrite("disable_mpi_backends", &cudecompGridDescAutotuneOptions_t::disable_mpi_backends)
      .def_readwrite("disable_nccl_backends", &cudecompGridDescAutotuneOptions_t::disable_nccl_backends)
      .def_readwrite("disable_nvshmem_backends", &cudecompGridDescAutotuneOptions_t::disable_nvshmem_backends)
      .def_readwrite("skip_threshold", &cudecompGridDescAutotuneOptions_t::skip_threshold)
      .def_readwrite("autotune_transpose_backend", &cudecompGridDescAutotuneOptions_t::autotune_transpose_backend)
      .def_property(
          "transpose_use_inplace_buffers",
          [](py::object self) {
            auto& options = self.cast<cudecompGridDescAutotuneOptions_t&>();
            return make_array_view(options.transpose_use_inplace_buffers, self);
          },
          [](cudecompGridDescAutotuneOptions_t& options, const std::array<bool, 4>& value) {
            copy_array(value, options.transpose_use_inplace_buffers);
          })
      .def_property(
          "transpose_op_weights",
          [](py::object self) {
            auto& options = self.cast<cudecompGridDescAutotuneOptions_t&>();
            return make_array_view(options.transpose_op_weights, self);
          },
          [](cudecompGridDescAutotuneOptions_t& options, const std::array<double, 4>& value) {
            copy_array(value, options.transpose_op_weights);
          })
      .def_property(
          "transpose_input_halo_extents",
          [](py::object self) {
            auto& options = self.cast<cudecompGridDescAutotuneOptions_t&>();
            return make_array_view(options.transpose_input_halo_extents, self);
          },
          py::cpp_function(
              [](cudecompGridDescAutotuneOptions_t& options, const std::array<std::array<int32_t, 3>, 4>& value) {
                copy_matrix(value, options.transpose_input_halo_extents);
              },
              py::arg("self"), py::arg("value").noconvert()))
      .def_property(
          "transpose_output_halo_extents",
          [](py::object self) {
            auto& options = self.cast<cudecompGridDescAutotuneOptions_t&>();
            return make_array_view(options.transpose_output_halo_extents, self);
          },
          py::cpp_function(
              [](cudecompGridDescAutotuneOptions_t& options, const std::array<std::array<int32_t, 3>, 4>& value) {
                copy_matrix(value, options.transpose_output_halo_extents);
              },
              py::arg("self"), py::arg("value").noconvert()))
      .def_property(
          "transpose_input_padding",
          [](py::object self) {
            auto& options = self.cast<cudecompGridDescAutotuneOptions_t&>();
            return make_array_view(options.transpose_input_padding, self);
          },
          py::cpp_function(
              [](cudecompGridDescAutotuneOptions_t& options, const std::array<std::array<int32_t, 3>, 4>& value) {
                copy_matrix(value, options.transpose_input_padding);
              },
              py::arg("self"), py::arg("value").noconvert()))
      .def_property(
          "transpose_output_padding",
          [](py::object self) {
            auto& options = self.cast<cudecompGridDescAutotuneOptions_t&>();
            return make_array_view(options.transpose_output_padding, self);
          },
          py::cpp_function(
              [](cudecompGridDescAutotuneOptions_t& options, const std::array<std::array<int32_t, 3>, 4>& value) {
                copy_matrix(value, options.transpose_output_padding);
              },
              py::arg("self"), py::arg("value").noconvert()))
      .def_readwrite("autotune_halo_backend", &cudecompGridDescAutotuneOptions_t::autotune_halo_backend)
      .def_property(
          "halo_extents",
          [](py::object self) {
            auto& options = self.cast<cudecompGridDescAutotuneOptions_t&>();
            return make_array_view(options.halo_extents, self);
          },
          py::cpp_function([](cudecompGridDescAutotuneOptions_t& options,
                              const std::array<int32_t, 3>& value) { copy_array(value, options.halo_extents); },
                           py::arg("self"), py::arg("value").noconvert()))
      .def_property(
          "halo_periods",
          [](py::object self) {
            auto& options = self.cast<cudecompGridDescAutotuneOptions_t&>();
            return make_array_view(options.halo_periods, self);
          },
          [](cudecompGridDescAutotuneOptions_t& options, const std::array<bool, 3>& value) {
            copy_array(value, options.halo_periods);
          })
      .def_property(
          "halo_axis", [](const cudecompGridDescAutotuneOptions_t& options) { return options.halo_axis; },
          py::cpp_function([](cudecompGridDescAutotuneOptions_t& options, int32_t value) { options.halo_axis = value; },
                           py::arg("self"), py::arg("value").noconvert()))
      .def_property(
          "halo_padding",
          [](py::object self) {
            auto& options = self.cast<cudecompGridDescAutotuneOptions_t&>();
            return make_array_view(options.halo_padding, self);
          },
          py::cpp_function([](cudecompGridDescAutotuneOptions_t& options,
                              const std::array<int32_t, 3>& value) { copy_array(value, options.halo_padding); },
                           py::arg("self"), py::arg("value").noconvert()));

  py::class_<PyPencilInfo>(module, "PencilInfo", "Immutable C-order Python-normalized cuDecomp pencil metadata.")
      .def_property_readonly("shape", [](const PyPencilInfo& info) { return make_int32_tuple(info.shape); })
      .def_property_readonly("lo", [](const PyPencilInfo& info) { return make_int32_tuple(info.lo); })
      .def_property_readonly("hi", [](const PyPencilInfo& info) { return make_int32_tuple(info.hi); })
      .def_property_readonly("order", [](const PyPencilInfo& info) { return make_int32_tuple(info.order); })
      .def_property_readonly("halo_extents",
                             [](const PyPencilInfo& info) { return make_int32_tuple(info.halo_extents); })
      .def_property_readonly("padding", [](const PyPencilInfo& info) { return make_int32_tuple(info.padding); })
      .def_property_readonly("size", [](const PyPencilInfo& info) { return info.size; });

  py::class_<PyHandle, std::shared_ptr<PyHandle>>(
      module, "Handle",
      "Long-lived cuDecomp handle. Initialization and finalize() are collective over the provided MPI communicator.")
      .def(py::init([exceptions](py::object mpi_comm) {
             const MPI_Fint mpi_comm_f = py::cast<MPI_Fint>(mpi_comm.attr("py2f")());
             auto handle = std::make_shared<PyHandle>();
             cudecompResult_t result = CUDECOMP_RESULT_SUCCESS;
             {
               py::gil_scoped_release release;
               result = cudecompInit_F(&handle->handle, mpi_comm_f);
             }
             check_result(result, exceptions);
             return handle;
           }),
           py::arg("mpi_comm"),
           "Collectively initialize cuDecomp from an mpi4py communicator. Call finalize() explicitly on all ranks.")
      .def(
          "finalize",
          [exceptions](PyHandle& handle) {
            if (handle.handle == nullptr) { return; }

            cudecompResult_t result = CUDECOMP_RESULT_SUCCESS;
            {
              py::gil_scoped_release release;
              result = cudecompFinalize(handle.handle);
            }
            check_result(result, exceptions);
            handle.handle = nullptr;
          },
          "Collectively finalize the cuDecomp handle exactly once. Repeated Python calls are no-ops.")
      .def(
          "create_grid",
          [exceptions](std::shared_ptr<PyHandle> handle, cudecompGridDescConfig_t& config,
                       const std::optional<cudecompGridDescAutotuneOptions_t>& autotune_options) {
            const auto* options = autotune_options ? &*autotune_options : nullptr;

            auto grid_desc = std::make_shared<PyGridDesc>(handle);
            cudecompResult_t result = CUDECOMP_RESULT_SUCCESS;
            {
              py::gil_scoped_release release;
              result = cudecompGridDescCreate(handle->handle, &grid_desc->grid_desc, &config, options);
            }
            check_result(result, exceptions);
            return grid_desc;
          },
          py::arg("config"), py::arg("autotune_options") = py::none(),
          "Collectively create a grid descriptor from a GridDescConfig and optional explicit autotune options.");

  py::class_<PyGridDesc, std::shared_ptr<PyGridDesc>>(
      module, "GridDesc",
      "cuDecomp grid descriptor. Destruction is collective and must be requested explicitly with destroy().")
      .def(
          "destroy",
          [exceptions](PyGridDesc& grid_desc) {
            if (grid_desc.grid_desc == nullptr) { return; }

            cudecompResult_t result = CUDECOMP_RESULT_SUCCESS;
            {
              py::gil_scoped_release release;
              result = cudecompGridDescDestroy(grid_desc.handle->handle, grid_desc.grid_desc);
            }
            check_result(result, exceptions);
            grid_desc.grid_desc = nullptr;
          },
          "Collectively destroy the grid descriptor exactly once. Repeated Python calls are no-ops.")
      .def(
          "get_config",
          [exceptions](const PyGridDesc& grid_desc) {
            cudecompGridDescConfig_t config;
            const cudecompResult_t result =
                cudecompGetGridDescConfig(grid_desc.handle->handle, grid_desc.grid_desc, &config);
            check_result(result, exceptions);
            return config;
          },
          "Return the resolved grid descriptor configuration.")
      .def(
          "get_pencil_info",
          [exceptions](const PyGridDesc& grid_desc, int32_t axis,
                       const std::optional<std::array<int32_t, 3>>& halo_extents,
                       const std::optional<std::array<int32_t, 3>>& padding) {
            cudecompPencilInfo_t pencil_info;
            const cudecompResult_t result =
                cudecompGetPencilInfo(grid_desc.handle->handle, grid_desc.grid_desc, &pencil_info, axis,
                                      halo_extents ? halo_extents->data() : nullptr,
                                      padding ? padding->data() : nullptr);
            check_result(result, exceptions);
            return normalize_pencil_info(pencil_info);
          },
          py::arg("axis").noconvert(), py::arg("halo_extents").noconvert() = py::none(),
          py::arg("padding").noconvert() = py::none(),
          "Return immutable C-order Python-normalized pencil metadata for a zero-based axis.")
      .def(
          "transpose_x_to_y",
          [exceptions](PyGridDesc& grid_desc, py::object input, py::object out, std::optional<cudecompDataType_t> dtype,
                       const std::optional<std::array<int32_t, 3>>& input_halo_extents,
                       const std::optional<std::array<int32_t, 3>>& output_halo_extents,
                       const std::optional<std::array<int32_t, 3>>& input_padding,
                       const std::optional<std::array<int32_t, 3>>& output_padding, py::object stream) {
            return transpose_operation(grid_desc, std::move(input), std::move(out), dtype, input_halo_extents,
                                       output_halo_extents, input_padding, output_padding, std::move(stream),
                                       kTransposeXToY, exceptions);
          },
          py::arg("input"), py::arg("out") = py::none(), py::arg("dtype") = py::none(),
          py::arg("input_halo_extents").noconvert() = py::none(),
          py::arg("output_halo_extents").noconvert() = py::none(), py::arg("input_padding").noconvert() = py::none(),
          py::arg("output_padding").noconvert() = py::none(), py::arg("stream") = py::none(),
          "Transpose an X-pencil CUDA array into a Y-pencil CUDA array.")
      .def(
          "transpose_y_to_z",
          [exceptions](PyGridDesc& grid_desc, py::object input, py::object out, std::optional<cudecompDataType_t> dtype,
                       const std::optional<std::array<int32_t, 3>>& input_halo_extents,
                       const std::optional<std::array<int32_t, 3>>& output_halo_extents,
                       const std::optional<std::array<int32_t, 3>>& input_padding,
                       const std::optional<std::array<int32_t, 3>>& output_padding, py::object stream) {
            return transpose_operation(grid_desc, std::move(input), std::move(out), dtype, input_halo_extents,
                                       output_halo_extents, input_padding, output_padding, std::move(stream),
                                       kTransposeYToZ, exceptions);
          },
          py::arg("input"), py::arg("out") = py::none(), py::arg("dtype") = py::none(),
          py::arg("input_halo_extents").noconvert() = py::none(),
          py::arg("output_halo_extents").noconvert() = py::none(), py::arg("input_padding").noconvert() = py::none(),
          py::arg("output_padding").noconvert() = py::none(), py::arg("stream") = py::none(),
          "Transpose a Y-pencil CUDA array into a Z-pencil CUDA array.")
      .def(
          "transpose_z_to_y",
          [exceptions](PyGridDesc& grid_desc, py::object input, py::object out, std::optional<cudecompDataType_t> dtype,
                       const std::optional<std::array<int32_t, 3>>& input_halo_extents,
                       const std::optional<std::array<int32_t, 3>>& output_halo_extents,
                       const std::optional<std::array<int32_t, 3>>& input_padding,
                       const std::optional<std::array<int32_t, 3>>& output_padding, py::object stream) {
            return transpose_operation(grid_desc, std::move(input), std::move(out), dtype, input_halo_extents,
                                       output_halo_extents, input_padding, output_padding, std::move(stream),
                                       kTransposeZToY, exceptions);
          },
          py::arg("input"), py::arg("out") = py::none(), py::arg("dtype") = py::none(),
          py::arg("input_halo_extents").noconvert() = py::none(),
          py::arg("output_halo_extents").noconvert() = py::none(), py::arg("input_padding").noconvert() = py::none(),
          py::arg("output_padding").noconvert() = py::none(), py::arg("stream") = py::none(),
          "Transpose a Z-pencil CUDA array into a Y-pencil CUDA array.")
      .def(
          "transpose_y_to_x",
          [exceptions](PyGridDesc& grid_desc, py::object input, py::object out, std::optional<cudecompDataType_t> dtype,
                       const std::optional<std::array<int32_t, 3>>& input_halo_extents,
                       const std::optional<std::array<int32_t, 3>>& output_halo_extents,
                       const std::optional<std::array<int32_t, 3>>& input_padding,
                       const std::optional<std::array<int32_t, 3>>& output_padding, py::object stream) {
            return transpose_operation(grid_desc, std::move(input), std::move(out), dtype, input_halo_extents,
                                       output_halo_extents, input_padding, output_padding, std::move(stream),
                                       kTransposeYToX, exceptions);
          },
          py::arg("input"), py::arg("out") = py::none(), py::arg("dtype") = py::none(),
          py::arg("input_halo_extents").noconvert() = py::none(),
          py::arg("output_halo_extents").noconvert() = py::none(), py::arg("input_padding").noconvert() = py::none(),
          py::arg("output_padding").noconvert() = py::none(), py::arg("stream") = py::none(),
          "Transpose a Y-pencil CUDA array into an X-pencil CUDA array.")
      .def(
          "update_halos_x",
          [exceptions](PyGridDesc& grid_desc, py::object input, const std::array<int32_t, 3>& halo_extents, int32_t dim,
                       std::optional<cudecompDataType_t> dtype, const std::optional<std::array<bool, 3>>& halo_periods,
                       const std::optional<std::array<int32_t, 3>>& padding, py::object stream) {
            return halo_operation(grid_desc, std::move(input), halo_extents, dim, dtype, halo_periods, padding,
                                  std::move(stream), kUpdateHalosX, exceptions);
          },
          py::arg("input"), py::arg("halo_extents").noconvert(), py::arg("dim").noconvert(),
          py::arg("dtype") = py::none(), py::arg("halo_periods") = py::none(),
          py::arg("padding").noconvert() = py::none(), py::arg("stream") = py::none(),
          "Update halo regions of an X-pencil CUDA array in place.")
      .def(
          "update_halos_y",
          [exceptions](PyGridDesc& grid_desc, py::object input, const std::array<int32_t, 3>& halo_extents, int32_t dim,
                       std::optional<cudecompDataType_t> dtype, const std::optional<std::array<bool, 3>>& halo_periods,
                       const std::optional<std::array<int32_t, 3>>& padding, py::object stream) {
            return halo_operation(grid_desc, std::move(input), halo_extents, dim, dtype, halo_periods, padding,
                                  std::move(stream), kUpdateHalosY, exceptions);
          },
          py::arg("input"), py::arg("halo_extents").noconvert(), py::arg("dim").noconvert(),
          py::arg("dtype") = py::none(), py::arg("halo_periods") = py::none(),
          py::arg("padding").noconvert() = py::none(), py::arg("stream") = py::none(),
          "Update halo regions of a Y-pencil CUDA array in place.")
      .def(
          "update_halos_z",
          [exceptions](PyGridDesc& grid_desc, py::object input, const std::array<int32_t, 3>& halo_extents, int32_t dim,
                       std::optional<cudecompDataType_t> dtype, const std::optional<std::array<bool, 3>>& halo_periods,
                       const std::optional<std::array<int32_t, 3>>& padding, py::object stream) {
            return halo_operation(grid_desc, std::move(input), halo_extents, dim, dtype, halo_periods, padding,
                                  std::move(stream), kUpdateHalosZ, exceptions);
          },
          py::arg("input"), py::arg("halo_extents").noconvert(), py::arg("dim").noconvert(),
          py::arg("dtype") = py::none(), py::arg("halo_periods") = py::none(),
          py::arg("padding").noconvert() = py::none(), py::arg("stream") = py::none(),
          "Update halo regions of a Z-pencil CUDA array in place.")
      .def(
          "get_shifted_rank",
          [exceptions](const PyGridDesc& grid_desc, int32_t axis, int32_t dim, int32_t displacement, bool periodic) {
            int32_t shifted_rank = -1;
            const cudecompResult_t result = cudecompGetShiftedRank(grid_desc.handle->handle, grid_desc.grid_desc, axis,
                                                                   dim, displacement, periodic, &shifted_rank);
            check_result(result, exceptions);
            return shifted_rank;
          },
          py::arg("axis").noconvert(), py::arg("dim").noconvert(), py::arg("displacement").noconvert(),
          py::arg("periodic"), "Return the global rank shifted from this rank along a zero-based pencil dimension.");
}
