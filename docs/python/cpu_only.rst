CPU-only build
==============

The Python bindings support the toolkit-free :doc:`CPU-only build <../cpu_only>`
for API testing and debugging. Install from the repository root with:

.. code-block:: shell

   CUDECOMP_BUILD_CPU_ONLY=ON python -m pip install .

Use the same setting when creating or installing CPU source archives. CPU
package versions have a ``+cpu`` suffix. Use separate environments for CPU and
GPU variants, and load only one variant per process.

The CPU binding accepts C-contiguous NumPy arrays with native byte order and
allocates NumPy outputs when ``out`` is omitted. CuPy is not required. Operations
complete synchronously and stream arguments are ignored. Only MPI backends are
supported; set both process dimensions explicitly. Autotuning is unsupported.
