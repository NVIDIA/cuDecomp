# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cupy as cp
from mpi4py import MPI
import numpy as np

import cudecomp


def main():
    # Initialize MPI and start up cuDecomp.
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    local_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
    local_rank = local_comm.Get_rank()
    cp.cuda.Device(local_rank).use()

    handle = cudecomp.Handle(comm)

    # Create cuDecomp grid descriptor with autotuning enabled.
    config = cudecomp.GridDescConfig()
    config.gdims = (64, 64, 64)
    config.transpose_axis_contiguous[:] = True

    config.pdims = (0, 0)  # P_rows, P_cols

    options = cudecomp.GridDescAutotuneOptions()

    # General options.
    options.n_warmup_trials = 3
    options.n_trials = 5
    options.dtype = cudecomp.DataType.DOUBLE
    options.disable_mpi_backends = False
    options.disable_nccl_backends = False
    options.disable_nvshmem_backends = False
    options.skip_threshold = 0.0

    # Process grid autotuning options.
    options.grid_mode = cudecomp.AutotuneGridMode.TRANSPOSE
    options.allow_uneven_decompositions = True

    # Transpose communication backend autotuning options.
    options.autotune_transpose_backend = True
    options.transpose_use_inplace_buffers[:] = True
    options.transpose_op_weights[:] = 1.0
    options.transpose_input_halo_extents[0] = [1, 1, 1]   # X-to-Y input.
    options.transpose_output_halo_extents[3] = [1, 1, 1]  # Y-to-X output.

    # Halo communication backend autotuning options.
    options.autotune_halo_backend = True
    options.halo_axis = 0
    options.halo_extents = (1, 1, 1)
    options.halo_periods = (True, True, True)

    grid = handle.create_grid(config, autotune_options=options)

    if rank == 0:
        print(f"running on {config.pdims[0]} x {config.pdims[1]} process grid...")
        print(f"running using {cudecomp.transpose_comm_backend_to_string(config.transpose_comm_backend)} transpose backend...")
        print(f"running using {cudecomp.halo_comm_backend_to_string(config.halo_comm_backend)} halo backend...")

    # Allocating pencil memory.
    # Get X-pencil information (with halo elements).
    halo_extents_x = (1, 1, 1)
    pinfo_x = grid.get_pencil_info(0, halo_extents=halo_extents_x)

    # Get Y-pencil information.
    pinfo_y = grid.get_pencil_info(1)

    # Get Z-pencil information.
    pinfo_z = grid.get_pencil_info(2)

    # Allocating pencil buffers.
    data_num_elements = max(pinfo_x.size, pinfo_y.size, pinfo_z.size)

    # Allocate device buffer.
    data_d = cp.empty(data_num_elements, dtype=cp.float64)

    # Allocate host buffer.
    data = np.empty(data_num_elements, dtype=np.float64)

    # Initializing pencil data on the host.
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

    # Copy initialized data to the GPU.
    data_d[:pinfo_x.size].set(data[:pinfo_x.size])

    # Transposing data.
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

    # Updating halos.
    # Setting for periodic halos in all directions.
    halo_periods = (True, True, True)

    # Update X-pencil halos in X direction.
    grid.update_halos_x(data_x_d, halo_extents_x, dim=0, halo_periods=halo_periods)

    # Update X-pencil halos in Y direction.
    grid.update_halos_x(data_x_d, halo_extents_x, dim=1, halo_periods=halo_periods)

    # Update X-pencil halos in Z direction.
    grid.update_halos_x(data_x_d, halo_extents_x, dim=2, halo_periods=halo_periods)

    # Cleanup resources.
    cp.cuda.Stream.null.synchronize()
    del data_x_d, data_y_d, data_z_d, data_d, data_x, data
    grid.destroy()
    handle.finalize()
    local_comm.Free()


if __name__ == "__main__":
    main()
