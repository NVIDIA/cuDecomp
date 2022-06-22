# FFT benchmarking example

In this sub-directory, there is a FFT benchmarking example to illustrate cuDecomp usage in the
context of performing a distributed 3D C2C or R2C FFT. The benchmarking executable accepts
a number of flags to control the configuration of the run (run `./benchmark_c2c -h` or similar
for a listing of options.)

To enable running sweeps of this program across a number of different configurations, there is
a [`benchmark_runner.py`](benchmark_runner.py) script you can use. This script runs the benchmark through a sequence of flag options,
defined by the different configurations in [`benchmark_config.yaml`](benchmark_config.yaml). The `*_test` configurations will run through
the full set of possible configurations and also perform error checking (defined by the `benchmark_base_test` entry).
Otherwise, the configurations are set to run a more limited set of configurations defined by the `benchmark_base` entry.

An example of how to launch an 8 GPU benchmark run on a DGXA100 using this script is:
```
python benchmark_runner.py \
  --launcher_cmd "mpirun -np 8 --mca pml ucx --bind-to none ../utils/bind.sh --cpu=../utils/dgxa100_map.sh --mem=../utils/dgxa100_map.sh --" \
  --ngpu 8 \
  --gx 1024 \
  --gy 1024 \
  --gz 1024 \
  --csvfile benchmark_c2c.dgxa100.8gpu.n1024.csv \
  benchmark_c2c
```

The `launcher_cmd` argument defines how the runner script should launch the benchmark executable using `mpirun`, `srun` or similar.
In this example, we also use the `bind.sh` script to handle core/NUMA domain affinity of the tasks. The `--ngpu`, `--gx`, `--gy`,
and `--gz` options define the number of GPUs to run on (should be equivalent to number of tasks in most instances), and the X, Y,
and Z dimensions of the domain to perform the FFT. `--csvfile` is the name of the file for the benchmark runner script to record results.
The final positional option is the configuration name (from `benchmark_config.yaml`) to use for the run.

To visualize the benchmark results, a [`plot_heatmaps.py`](heatmap_scripts/plot_heatmaps.py) script to plot heatmaps from the data captured in the csv files. Running the script on the a csv file like the following:
```
python plot_heatmaps.py --csvfile benchmark_c2c.dgxa100.8gpu.n1024.csv --output_prefix benchmark_c2c.dgxa100.8gpu.n1024
```
will generate one or more image files `benchmark_c2c.dgxa100.8gpu.n1024_*.png` that contain heatmap plots, with each file corresponding to a distinct set of options (e.g. precision, axis-contiguous settings, in-place or out-of-place, etc.), which are listed in the plot title.
Several sample csv files and generated heatmap plots for 2048^3 C2C FFTs on a DGX A100 (80GB) system using NVHPC SDK 22.5, can be found in the [samples](heatmap_scripts/samples) directory.

We can examine one of the sample plots ([`benchmark_c2c.dgxa100.8gpu.n2048_1.png`](heatmap_scripts/sample/benchmark_c2c.dgxa100.8gpu.n2048_1.png)), shown below, to explain the content.
![heatmap_example](heatmap_scripts/sample/benchmark_c2c.dgxa100.8gpu.n2048_1.png?raw=true)
The title of the plot lists the options used to generate these results. Specifically, this plot corresponds to a double-precision C2C FFT on a 2048 x 2048 x 2048 grid. Additionally, it is using `axis-contiguous` pencils in all dimensions, using in-place buffers, and not using managed memory.

The subplot on the left (in green) records the FFT performance, in GFLOP/s, measured for each possible pairing of process grid dimension and transpose communication backend (listed on the horizontal and vertical axes respectively). In this subplot, higher is better.

The subplot on the right (in purple) records the average transpose trial time, in milliseconds, measured by the autotuner during a full autotuning run (i.e. process grid and transpose backend autotuning). In this subplot, lower is better. The right-arrow symbols mark the minimum value within rows, indicating the process grid achieving minimum transpose time for a given communication backend. The down-arrow symbols mark the minimum value within columns, indicating the communication backend achieving minimum transpose time for a given process grid. The star symbol marks the minimum value across the rows and columns, indicating the process grid and communication backend pair the autotuner would select that minimized transpose time.

The results in this plot indicate an important consideration when using cuDecomp autotuning. The autotuner seeks to minimize communication
time, which is a good proxy for end-to-end performance in many codes, but not always. In this FFT benchmark code, some optimizations are
made based on the process grid dimensions, meaning some decompositions might have more efficient compute than others. Additionally, some grid
configurations might yield better memory layouts for the FFTs than others, introducing more variation in compute efficiency across process grid
decompositions. For this reason, the minimum communication time might not always yield the highest performance. We see this is the case
in the plot above, where the autotuner would select the `1 x 8 + NVSHMEM (pipelined)` configuration, while the `8 x 1 + NVSHMEM (pipelined)`
configuration achieves the peak FFT performance. With that said, the selected configuration would still achieve high performance relative to
many of the other options.

For situations where some known optimizations can greatly benefit specific process grid dimensions, consider limiting the cuDecomp autotuner to
only backend autotuning with a fixed process grid.
