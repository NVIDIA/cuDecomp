find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 2.12 CONFIG REQUIRED)

set(CUDECOMP_PYTHON_BUILD_DIR ${CMAKE_BINARY_DIR}/python/cudecomp)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cudecomp/__init__.py
  ${CUDECOMP_PYTHON_BUILD_DIR}/__init__.py
  COPYONLY
)

pybind11_add_module(cudecomp_python MODULE
  cudecomp/cudecomp_py.cc
)
target_compile_features(cudecomp_python PRIVATE cxx_std_17)
# NVHPC does not define NDEBUG in optimized configurations; keep pybind11's per-reference GIL assertions in Debug only.
target_compile_definitions(cudecomp_python PRIVATE
  "$<$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>,$<CONFIG:MinSizeRel>>:NDEBUG>"
)
target_include_directories(cudecomp_python
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../include
)
target_link_libraries(cudecomp_python PRIVATE cudecomp)
if (APPLE)
  set(CUDECOMP_PYTHON_RPATH_ORIGIN "@loader_path")
else()
  set(CUDECOMP_PYTHON_RPATH_ORIGIN "$ORIGIN")
endif()
set_target_properties(cudecomp_python PROPERTIES
  OUTPUT_NAME _cudecomp
  LIBRARY_OUTPUT_DIRECTORY ${CUDECOMP_PYTHON_BUILD_DIR}
  BUILD_RPATH "${CUDECOMP_PYTHON_RPATH_ORIGIN}/../../lib"
)

# Python packages are installed only through scikit-build-core (pip). Plain CMake builds produce the extension in
# the build tree for CTest but do not install it.
if (SKBUILD)
  set_target_properties(cudecomp_python PROPERTIES INSTALL_RPATH "${CUDECOMP_PYTHON_RPATH_ORIGIN}/.libs")
  install(TARGETS cudecomp_python LIBRARY DESTINATION cudecomp)
endif()
