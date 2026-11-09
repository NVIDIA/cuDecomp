find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(python_test_labels "python;mpi")
if (CUDECOMP_BUILD_CPU_ONLY)
  list(APPEND python_test_labels cpu)
endif()

foreach(python_test IN ITEMS api transpose halo)
  # Run through mpi4py so an uncaught exception on any rank aborts the job instead of leaving the other ranks
  # blocked in a collective until the test timeout.
  add_test(
    NAME cudecomp_python_${python_test}
    COMMAND ${MPIEXEC_EXECUTABLE} ${CUDECOMP_TEST_MPIEXEC_PREFLAGS_LIST}
            ${MPIEXEC_NUMPROC_FLAG} 4
            ${Python3_EXECUTABLE} -m mpi4py ${CMAKE_CURRENT_SOURCE_DIR}/test_${python_test}.py
  )
  set_tests_properties(cudecomp_python_${python_test} PROPERTIES
    LABELS "${python_test_labels};${python_test}"
    TIMEOUT 120
  )
  # Prepend so dependencies provided through the caller's PYTHONPATH (e.g. environment modules) stay importable.
  if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.22)
    set_tests_properties(cudecomp_python_${python_test} PROPERTIES
      ENVIRONMENT_MODIFICATION "PYTHONPATH=path_list_prepend:${CMAKE_BINARY_DIR}/python"
    )
  else()
    set_tests_properties(cudecomp_python_${python_test} PROPERTIES
      ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/python"
    )
  endif()
endforeach()
