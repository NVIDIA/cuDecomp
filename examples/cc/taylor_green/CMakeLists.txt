add_executable(tg)

target_sources(tg
  PRIVATE
  tg.cu
)

if (CMAKE_VERSION VERSION_LESS 3.18)
  target_compile_options(tg PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: ${CUDA_CC_FLAGS}>)
else()
  set_target_properties(tg PROPERTIES CUDA_ARCHITECTURES "${CUDECOMP_CUDA_CC_LIST}")
endif()

target_include_directories(tg
  PRIVATE
  ${CMAKE_BINARY_DIR}/include
  ${MPI_CXX_INCLUDE_DIRS}
  ${NVHPC_CUFFT_INCLUDE_DIR}
)
target_link_libraries(tg PRIVATE MPI::MPI_CXX)
target_link_libraries(tg PRIVATE NVHPC::CUFFT)
target_link_libraries(tg PRIVATE cudecomp)
set_target_properties(tg PROPERTIES LINKER_LANGUAGE CXX)

install(
  TARGETS tg
  RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/examples/cc/taylor_green
)
