find_package(hipfft REQUIRED)

add_executable(tg)

target_sources(tg
  PRIVATE
  tg.cu
)

set_target_properties(tg PROPERTIES HIP_ARCHITECTURES "${CUDECOMP_HIP_ARCH_LIST}")

target_include_directories(tg
  PRIVATE
  ${CMAKE_BINARY_DIR}/include
  ${MPI_CXX_INCLUDE_DIRS}
  ${NVHPC_CUFFT_INCLUDE_DIR}
)
target_link_libraries(tg PRIVATE MPI::MPI_CXX)
target_link_libraries(tg PRIVATE hip::hipfft)
target_link_libraries(tg PRIVATE hipdecomp)
set_target_properties(tg PROPERTIES LINKER_LANGUAGE CXX)

install(
  TARGETS tg
  RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/examples/cc/taylor_green
)
