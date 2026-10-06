add_executable(tg)

target_sources(tg
  PRIVATE
  tg.cu
)

set_target_properties(tg PROPERTIES CUDA_ARCHITECTURES "${CUDECOMP_CUDA_CC_LIST}")
target_compile_options(tg PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets >)
target_compile_options(tg PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress 177>) # Silence unused function warning

target_include_directories(tg
  PRIVATE
  ${CMAKE_BINARY_DIR}/include
  ${MPI_CXX_INCLUDE_DIRS}
  ${CUDECOMP_CUFFT_INCLUDE_DIR}
)
target_link_libraries(tg PRIVATE MPI::MPI_CXX)
target_link_libraries(tg PRIVATE ${CUDECOMP_CUFFT_TARGET})
target_link_libraries(tg PRIVATE cudecomp)
set_target_properties(tg PROPERTIES LINKER_LANGUAGE CXX)

install(
  TARGETS tg
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/examples/cc/taylor_green
)
