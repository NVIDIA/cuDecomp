# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys

import cupy as cp
from mpi4py import MPI
import numpy as np

import cudecomp


def main():
    # Initialize MPI and start up cuDecomp
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nranks = comm.Get_size()

    if nranks != 4:
        print("This example requires 4 ranks to run. Exiting...\n")
        sys.exit(1)

    local_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
    local_rank = local_comm.Get_rank()
    cp.cuda.Device(local_rank).use()

    handle = cudecomp.Handle(comm)

    # Create cuDecomp grid
    config = cudecomp.GridDescConfig()

    config.pdims = (2, 2)  # P_rows, P_cols
    config.gdims = (64, 64, 64)  # X, Y, Z

    config.transpose_comm_backend = cudecomp.TransposeCommBackend.MPI_P2P
    config.halo_comm_backend = cudecomp.HaloCommBackend.MPI

    config.transpose_axis_contiguous = (True, True, True)

    grid = handle.create_grid(config)

    # Creating pencil arrays

    # Get X-pencil information (with halo elements)
    pinfo_x = grid.get_pencil_info(0, halo_extents=(1, 1, 1))

    # Get Y-pencil information
    pinfo_y = grid.get_pencil_info(1)

    # Get Z-pencil information
    pinfo_z = grid.get_pencil_info(2)

    # Create initial X-pencil array (on host)
    x_data = np.empty(pinfo_x.shape, dtype=np.float64)

    # Initializing pencil array (on host)
    for i in range(pinfo_x.shape[0]):
        for j in range(pinfo_x.shape[1]):
            for k in range(pinfo_x.shape[2]):
                # Compute global grid coordinates. To compute these, we offset the local coordinates using the lower
                # bound, lo, and use the order array to map the local coordinate order to the global coordinate order.
                gx = [0, 0, 0]
                gx[pinfo_x.order[0]] = i + pinfo_x.lo[0]
                gx[pinfo_x.order[1]] = j + pinfo_x.lo[1]
                gx[pinfo_x.order[2]] = k + pinfo_x.lo[2]

                # Since the X-pencil also has halo elements, we apply an additional offset for the halo elements in
                # each direction, again using the order array to apply the extent to the appropriate global coordinate.
                gx[pinfo_x.order[0]] -= pinfo_x.halo_extents[pinfo_x.order[0]]
                gx[pinfo_x.order[1]] -= pinfo_x.halo_extents[pinfo_x.order[1]]
                gx[pinfo_x.order[2]] -= pinfo_x.halo_extents[pinfo_x.order[2]]

                # Finally, we can set the buffer element, for example using a function based on the global coordinates.
                x, y, z = gx
                x_data[i, j, k] = x + y + z

    # Copy host pencil to device
    x_data = cp.asarray(x_data)

    # Transposing data

    # Transpose from X-pencils to Y-pencils
    y_data = grid.transpose_x_to_y(x_data, input_halo_extents=pinfo_x.halo_extents)

    # Transpose from Y-pencils to Z-pencils
    z_data = grid.transpose_y_to_z(y_data)

    # Transpose from Z-pencils to Y-pencils
    y_data = grid.transpose_z_to_y(z_data)

    # Transpose from Y-pencils to X-pencils
    x_data = grid.transpose_y_to_x(y_data, output_halo_extents=pinfo_x.halo_extents)

    # Updating halos

    # Setting periodic halos for all directions
    halo_periods = (True, True, True)

    # Update X-pencil halos in X direction
    grid.update_halos_x(x_data, halo_extents=pinfo_x.halo_extents, halo_periods=halo_periods, dim=0)

    # Update X-pencil halos in Y direction
    grid.update_halos_x(x_data, halo_extents=pinfo_x.halo_extents, halo_periods=halo_periods, dim=1)

    # Update X-pencil halos in Z direction
    grid.update_halos_x(x_data, halo_extents=pinfo_x.halo_extents, halo_periods=halo_periods, dim=2)

    # Cleanup resources
    grid.destroy()
    handle.finalize()


if __name__ == "__main__":
    main()
