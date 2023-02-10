set(benchmark_targets
  benchmark_r2c
  benchmark_c2c
  benchmark_r2c_f
  benchmark_c2c_f
)

foreach(tgt ${benchmark_targets})
  add_executable(${tgt})
  if (CMAKE_VERSION VERSION_LESS 3.18)
    target_compile_options(${tgt} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: ${CUDA_CC_FLAGS}>)
  else()
    set_target_properties(${tgt} PROPERTIES CUDA_ARCHITECTURES "${CUDA_CC_LIST}")
  endif()
  target_sources(${tgt}
    PRIVATE
    benchmark.cu
  )
  target_include_directories(${tgt}
	                     PRIVATE
			     ${CMAKE_CURRENT_SOURCE_DIR}/../include
			     ${NVHPC_CUFFT_INCLUDE_DIE})
  target_link_libraries(${tgt} PRIVATE cudecomp)
  target_link_libraries(${tgt} PRIVATE NVHPC::CUFFT)
endforeach()

target_compile_definitions(benchmark_r2c PRIVATE R2C)
target_compile_definitions(benchmark_c2c PRIVATE C2C)
target_compile_definitions(benchmark_r2c_f PRIVATE R2C USE_FLOAT)
target_compile_definitions(benchmark_c2c_f PRIVATE R2C USE_FLOAT)

install(
  TARGETS ${benchmark_targets}
  RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/benchmark
)
