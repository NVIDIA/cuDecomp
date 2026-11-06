.. _python-basic-usage-ref:

Basic Usage Guide
=================
We can now walk through how to set up and run a basic program with cuDecomp. The code snippets in this section use CuPy, NumPy, and ``mpi4py``;
see also the `Python basic usage example
<https://github.com/NVIDIA/cuDecomp/blob/main/examples/python/basic_usage/basic_usage.py>`_.
This example assumes we are using 4 GPUs.

Starting up cuDecomp
--------------------
First, initialize MPI and assign CUDA devices. Importing ``mpi4py.MPI`` normally initializes MPI. In this example, we assign CUDA devices based on the local rank.

.. code-block:: python

   import cupy as cp
   from mpi4py import MPI
   import numpy as np
   import cudecomp

   comm = MPI.COMM_WORLD
   rank = comm.Get_rank()
   nranks = comm.Get_size()

   local_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
   local_rank = local_comm.Get_rank()
   cp.cuda.Device(local_rank).use()

Next, we can initialize cuDecomp with :py:class:`cudecomp.Handle` using the MPI global communicator and obtain a handle.

.. code-block:: python

   handle = cudecomp.Handle(comm)

Creating a grid descriptor
--------------------------
Next, we need to create a grid descriptor. To do this, we first need to create and populate a grid descriptor
configuration object, which provides basic information to the library required to set up the grid descriptor.

We create a configuration object using :py:class:`cudecomp.GridDescConfig`, which initializes its fields to defaults.
Array fields are writable NumPy views; individual elements, rows, or slices can be modified, or an entire field can be
assigned a tuple, list, or NumPy array. Views retain the configuration object; use ``.copy()`` for an independent copy.

.. code-block:: python

   config = cudecomp.GridDescConfig()

First, we can set the :code:`pdims` (process grid) entries in the configuration object. :code:`pdims[0]` corresponds to :math:`P_{\text{rows}}`
and :code:`pdims[1]` corresponds to :math:`P_{\text{cols}}`. In this example, we use a :math:`2 \times 2` process grid.

.. code-block:: python

   config.pdims = (2, 2)  # P_rows, P_cols

By default, ranks are assigned to the process grid in row-major order. To use column-major rank assignment, set
:code:`rank_order` in the configuration object.

.. code-block:: python

   config.rank_order = cudecomp.RankOrder.COL_MAJOR

Next, we set the :code:`gdims` (global grid) entries in the configuration object. These values correspond to the :math:`X`, :math:`Y`, and :math:`Z`
dimensions of the global grid. In this example, we use a global grid with dimensions :math:`64 \times 64 \times 64`.

.. code-block:: python

   config.gdims = (64, 64, 64)  # X, Y, Z

For additional flexibility, the configuration object contains an optional entry :code:`gdims_dist` that indicates to the library
that the global domain of dimension :code:`gdims` should be distributed across processes with elements divided among processes
as though the global domain was of dimension :code:`gdims_dist`. This can be useful when dealing with padded domain dimensions.
The entries in :code:`gdims_dist` must be less than or equal to the entries in :code:`gdims` and any extra elements are associated with the last rank in any row or column communicator.

Next, we set the desired communication backends for transpose (:code:`transpose_comm_backend`) and/or
halo communication (:code:`halo_comm_backend`). See documentation of :ref:`cudecompTransposeCommBackend_t-ref` and
:ref:`cudecompHaloCommBackend_t-ref` for the available communication backends options.

.. code-block:: python

   config.transpose_comm_backend = cudecomp.TransposeCommBackend.MPI_P2P
   config.halo_comm_backend = cudecomp.HaloCommBackend.MPI

We can next set the values of :code:`transpose_axis_contiguous`, which are boolean flags indicating to the library
the memory layout of the pencil buffers to use, by axis. See the shared
:ref:`pencil memory layout description <pencil-memory-layouts-ref>` for the available layouts and their use.
Those orders are listed from fastest to slowest; Python C-order array dimensions are listed in reverse order.
In this example, we set this flag to true for all directions.

.. code-block:: python

   config.transpose_axis_contiguous[:] = True

Advanced users who require more flexibility in the memory layout of the pencil buffers can override the layouts available via
:code:`transpose_axis_contiguous` by setting the :code:`transpose_mem_order` array in the configuration object. This array enables
users to set arbitrary memory layout orders for the pencil buffers by axis. For example, a user can set this object as follows to
have pencil memory in :math:`[X, Y, Z]` order for the :math:`X`-pencil and :math:`[Z, Y, X]` order for the :math:`Y`- and :math:`Z`-pencils:

.. code-block:: python

   config.transpose_mem_order[0] = [0, 1, 2]
   config.transpose_mem_order[1] = [2, 1, 0]
   config.transpose_mem_order[2] = [2, 1, 0]

With the grid descriptor configuration object created and populated, we can now create the grid descriptor.
The optional ``autotune_options`` argument to :py:meth:`cudecomp.Handle.create_grid` sets autotuning options.
See :doc:`autotuning` for a detailed overview of this feature. In this example, we will not autotune and leave
this argument unspecified.

.. code-block:: python

   grid = handle.create_grid(config)

Allocate pencil memory
-----------------------------------
Once the grid descriptor is created, we can now query information about the decomposition and allocate device memory
to use for the pencil data. See :doc:`../overview` for the shared decomposition concepts.

First, we can query basic information (i.e. metadata) about the pencil configurations that the library
assigned to this process using the :py:meth:`cudecomp.GridDesc.get_pencil_info` function. This function returns a
pencil info object (:py:class:`cudecomp.PencilInfo`) that contains the shape, global lower and upper
index bounds (:code:`lo` and :code:`hi`), size of the pencil, and an :code:`order` array to indicate the memory layout
that will be used (to handle permuted layouts). Additionally, there are :code:`halo_extents`  and :code:`padding` data
members that indicates halo and padding configurations for the pencil, by axis. This data member is a copy of the
argument provided to the function and is stored for convenience.

Pencil information describes the storage required for the halo and padding arguments supplied to the query.
Use the same halo and padding arguments in subsequent operations. Python arrays must be C-contiguous and match
the queried shape; queries do not allocate pencil storage.

In this example, we apply halo elements to the :math:`X`-pencils only. For the other pencils,
we leave ``halo_extents`` unspecified, which defaults to no halo regions. Similarly, we leave ``padding``
unspecified to disable padding for all pencils. Axes are zero-based: 0 is X, 1 is Y, and 2 is Z.

.. code-block:: python

   # Get X-pencil information (with halo elements).
   halo_extents_x = (1, 1, 1)
   pinfo_x = grid.get_pencil_info(0, halo_extents=halo_extents_x)

   # Get Y-pencil information.
   pinfo_y = grid.get_pencil_info(1)

   # Get Z-pencil information.
   pinfo_z = grid.get_pencil_info(2)

With the information from the pencil info objects, we can now allocate device memory to use with cuDecomp.
In this example, we allocate a single device buffer :code:`data_d` that is large enough to hold the largest
pencil assigned to this process, across the three axes. We also allocate an equivalently sized buffer on the
host, :code:`data`, for convenience.

.. code-block:: python

   data_num_elements = max(pinfo_x.size, pinfo_y.size, pinfo_z.size)

   # Allocate device buffer.
   data_d = cp.empty(data_num_elements, dtype=cp.float64)

   # Allocate host buffer.
   data = np.empty(data_num_elements, dtype=np.float64)

.. _python-pencil-layout-ref:

Working with pencil data
------------------------
The pencil info objects are also used to access and manipulate data within the allocated pencil buffers.
For illustrative purposes, we will use the :math:`X`-pencil info object here, but this
will work for any of the axis pencils.

Here is an example of accessing the :math:`X`-pencil buffer data on the host using a triple loop.
Python pencil shapes and bounds are in local C-order; ``order`` maps local dimensions to global X/Y/Z axes.
The ``lo`` and ``hi`` bounds are zero-based and inclusive, excluding halos and padding.

.. code-block:: python

   data_x = data[:pinfo_x.size].reshape(pinfo_x.shape)
   for i in range(pinfo_x.shape[0]):
       for j in range(pinfo_x.shape[1]):
           for k in range(pinfo_x.shape[2]):
               # Compute global grid coordinates from local coordinates and lower bounds.
               gx = [0, 0, 0]
               for local_axis, local_index in enumerate((i, j, k)):
                   global_axis = pinfo_x.order[local_axis]
                   gx[global_axis] = local_index + pinfo_x.lo[local_axis]

                   # Account for the X-pencil's halo elements in this direction.
                   gx[global_axis] -= pinfo_x.halo_extents[global_axis]

               # Set the buffer element using a function of the global coordinates.
               data_x[i, j, k] = gx[0] + gx[1] + gx[2]

After assigning values on the host, we can copy the initialized host data to the GPU:

.. code-block:: python

   data_d[:pinfo_x.size].set(data[:pinfo_x.size])

Allocating workspace
-----------------------------
Besides device memory to store pencil data, cuDecomp also requires workspace buffers on the device. For transposes, the workspace
is used to facilitate local packing/unpacking and transposition operations (which are currently performed
out-of-place). As a result, this workspace buffer will be approximately 2x the size of the largest pencil
assigned to this process. For halo communication, the workspace is used to facilitate local packing of non-contiguous
halo elements.

Python uses automatically managed workspaces. No workspace allocation or workspace argument is required;
cuDecomp allocates and grows the workspace as needed, preserves ordering with the operation stream,
and releases the workspace when the handle is finalized.

Transposing the data
--------------------
Now, we can use cuDecomp's transposition routines to transpose our data. In these calls, we are using
the ``data_d`` array as both input and output (in-place), but you can also use distinct input and output buffers for
out-of-place operations. Python requires input and output views with the corresponding pencil shapes.
For transposes between :math:`Y`- and :math:`Z`-pencils, we leave halo extents unspecified.
For all transposes, we leave padding unspecified to disable padding. The data type is inferred from the arrays.

.. code-block:: python

   # Create C-contiguous views of the shared pencil buffer.
   data_x_d = data_d[:pinfo_x.size].reshape(pinfo_x.shape)
   data_y_d = data_d[:pinfo_y.size].reshape(pinfo_y.shape)
   data_z_d = data_d[:pinfo_z.size].reshape(pinfo_z.shape)

   # Transpose from X-pencils to Y-pencils.
   grid.transpose_x_to_y(data_x_d, out=data_y_d, input_halo_extents=halo_extents_x)

   # Transpose from Y-pencils to Z-pencils.
   grid.transpose_y_to_z(data_y_d, out=data_z_d)

   # Transpose from Z-pencils to Y-pencils.
   grid.transpose_z_to_y(data_z_d, out=data_y_d)

   # Transpose from Y-pencils to X-pencils.
   grid.transpose_y_to_x(data_y_d, out=data_x_d, output_halo_extents=halo_extents_x)

Updating halo regions
---------------------
In this example, we have halos for the :math:`X`-pencils only. We can use cuDecomp's halo update
routines to update the halo regions of this pencil in the three domain directions. In this example,
we set the :code:`halo_periods` argument to enable periodic halos along all directions.
We leave the padding arguments unspecified to disable padding.

.. code-block:: python

   # Setting for periodic halos in all directions.
   halo_periods = (True, True, True)

   # Update X-pencil halos in X direction.
   grid.update_halos_x(data_x_d, halo_extents_x, dim=0, halo_periods=halo_periods)

   # Update X-pencil halos in Y direction.
   grid.update_halos_x(data_x_d, halo_extents_x, dim=1, halo_periods=halo_periods)

   # Update X-pencil halos in Z direction.
   grid.update_halos_x(data_x_d, halo_extents_x, dim=2, halo_periods=halo_periods)

Cleaning up and finalizing the library
--------------------------------------
Finally, we can clean up resources. Complete GPU work before releasing its arrays, then destroy the grid
and finalize the handle on all ranks. Automatically managed workspaces are released by ``handle.finalize()``.
MPI is finalized by ``mpi4py`` when the program exits.

.. code-block:: python

   cp.cuda.Stream.null.synchronize()
   del data_x_d, data_y_d, data_z_d, data_d, data_x, data
   grid.destroy()
   handle.finalize()
   local_comm.Free()

Building and running the example
--------------------------------
Refer to the `Python build instructions in the README <https://github.com/NVIDIA/cuDecomp#python-experimental>`_
to install the Python package. No separate compilation of the Python example is required.

Once installed, the program can be executed using :code:`mpirun` or an equivalent parallel launcher:

.. code-block:: shell

   mpirun -n 4 python basic_usage.py

We highly suggest making usage of the :code:`bind.sh` shell script in the :code:`utils` directory to assist in process/NUMA binding,
to ensure processes are bound to node resources optimally (e.g. that processes are launched on CPU cores with close affinity to GPUs.) This is an example usage of the :code:`bind.sh` script for Perlmutter system:

.. code::

  srun -N1 --tasks-per-node 4 --bind=none bind.sh --cpu=pm_map.sh --mem=pm_map.sh -- python basic_usage.py

The :code:`pm_map.sh` is a file (which can be found in the :code:`utils` directory) containing the following:

.. code::

  bind_cpu_cores=([0]="48-63,112-127" [1]="32-47,96-111" [2]="16-31,80-95" [3]="0-15,64-79")

  bind_mem=([0]="3" [1]="2" [2]="1" [3]="0")

These bash arrays list CPU core ranges (:code:`bind_cpu_cores`) and NUMA domains (:code:`bind_mem`)  to pin each process to, by local rank. The :code:`bind.sh` script will use these arrays to pin processes using :code:`numactl`.
