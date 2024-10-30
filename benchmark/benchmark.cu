#include <algorithm>
#include <array>
#include <complex>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

#include <getopt.h>

#include <mpi.h>

#include <cuda_runtime.h>
#include <cufftXt.h>

#include <cuda/std/complex>

#include <cudecomp.h>
#include <internal/checks.h>

#ifdef USE_FLOAT
using real_t = float;
#define TOL (5e-4)
#else
using real_t = double;
#define TOL (1e-10)
#endif
using complex_t = cuda::std::complex<real_t>;

#ifdef USE_FLOAT
#ifdef R2C
static cufftType get_cufft_type_r2c(float) { return CUFFT_R2C; }
static cufftType get_cufft_type_c2r(float) { return CUFFT_C2R; }
#endif
static cufftType get_cufft_type_c2c(float) { return CUFFT_C2C; }
static cudecompDataType_t get_cudecomp_datatype(cuda::std::complex<float>) { return CUDECOMP_FLOAT_COMPLEX; }
#else
#ifdef R2C
static cufftType get_cufft_type_r2c(double) { return CUFFT_D2Z; }
static cufftType get_cufft_type_c2r(double) { return CUFFT_Z2D; }
#endif
static cufftType get_cufft_type_c2c(double) { return CUFFT_Z2Z; }
static cudecompDataType_t get_cudecomp_datatype(cuda::std::complex<double>) { return CUDECOMP_DOUBLE_COMPLEX; }

#endif

template <typename T> static std::vector<T> process_timings(std::vector<T> times, T scale = 1) {
  std::sort(times.begin(), times.end());
  double t_min = times[0];
  CHECK_MPI_EXIT(MPI_Allreduce(MPI_IN_PLACE, &t_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD));
  double t_max = times[times.size() - 1];
  CHECK_MPI_EXIT(MPI_Allreduce(MPI_IN_PLACE, &t_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));

  double t_avg = std::accumulate(times.begin(), times.end(), T(0)) / times.size();
  CHECK_MPI_EXIT(MPI_Allreduce(MPI_IN_PLACE, &t_avg, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));
  int nranks;
  CHECK_MPI_EXIT(MPI_Comm_size(MPI_COMM_WORLD, &nranks));
  t_avg /= nranks;

  for (auto& t : times) { t = (t - t_avg) * (t - t_avg); }
  double t_var = std::accumulate(times.begin(), times.end(), T(0)) / times.size();
  CHECK_MPI_EXIT(MPI_Allreduce(MPI_IN_PLACE, &t_var, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));
  t_var /= nranks;
  double t_std = std::sqrt(t_var);

  return {t_min * scale, t_max * scale, t_avg * scale, t_std * scale};
}

template <typename T, typename TS> __global__ static void scale(T* U, TS scale_factor, cudecompPencilInfo_t info) {

  const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= info.size) return;

  U[i] *= scale_factor;
}

static void usage(const char* pname) {

  const char* bname = rindex(pname, '/');
  if (!bname) {
    bname = pname;
  } else {
    bname++;
  }

  fprintf(stdout,
          "Usage: %s [options]\n"
          "options:\n"
          "\t-x|--gx <NX>\n"
          "\t\tX-dimension of grid. (default: 256) \n"
          "\t-y|--gy <NY>\n"
          "\t\tY-dimension of grid. (default: 256) \n"
          "\t-z|--gz <NZ>\n"
          "\t\tZ-dimension of grid. (default: 256) \n"
          "\t-r|--pr <PROCROWS>\n"
          "\t\tRow dimension of process grid. (default: 0, autotune) \n"
          "\t-c|--pc <PROCCOLS>\n"
          "\t--acx\n"
          "\t\tX-dimension axis-contiguous pencils setting. (default: 0) \n"
          "\t--acy\n"
          "\t\tY-dimension axis-contiguous pencils setting. (default: 0) \n"
          "\t--acz\n"
          "\t\tZ-dimension axis-contiguous pencils setting. (default: 0) \n"
          "\t\tColumn dimension of process grid. (default: 0, autotune) \n"
          "\t-w|--nwarmup <NWARMUP>\n"
          "\t\tNumber of warmup iterations. (default: 3)\n"
          "\n"
          "\t-t|--ntrials <NTRIALS>\n"
          "\t\tNumber of trial iterations. (default: 5) \n"
          "\n"
          "\t-k|--skip-threshold <THRESHOLD>\n"
          "\t\tAutotuner skip threshold setting. (default: 0.0) \n"
          "\n"
          "\t-b|--backend <INTEGER>\n"
          "\t\tCommunication backend to use. Choices: 0:AUTOTUNE, 1:MPI_P2P, 2:MPI_P2P_PL, 3:MPI_A2A, 4:NCCL, "
          "5:NCCL_PL, 6:NVSHMEM, 7:NVSHMEM_PL. (default: 0) \n"
          "\n"
          "\t-o|--out-of-place\n"
          "\t\tFlag to test out of place operation. \n"
          "\t-m|--use-managed-memory\n"
          "\t\tFlag to test operation with managed memory.\n"
          "\t-s|--skip-correctness-tests\n"
          "\t\tFlag to skip checking results for correctness.\n"
          "\t-n|--no-slab-opt\n"
          "\t\tFlag to disable slab optimizations (e.g. 2D local FFTs, transpose skipping).\n\n",
          bname);
  exit(EXIT_SUCCESS);
}

int main(int argc, char** argv) {
  CHECK_MPI_EXIT(MPI_Init(nullptr, nullptr));
  int rank, nranks;
  CHECK_MPI_EXIT(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  CHECK_MPI_EXIT(MPI_Comm_size(MPI_COMM_WORLD, &nranks));

  MPI_Comm local_comm;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local_comm);
  int local_rank;
  MPI_Comm_rank(local_comm, &local_rank);
  CHECK_CUDA_EXIT(cudaSetDevice(local_rank));

  // Parse command-line arguments
  int gx = 256;
  int gy = 256;
  int gz = 256;
  int pr = 0;
  int pc = 0;
  cudecompTransposeCommBackend_t comm_backend = static_cast<cudecompTransposeCommBackend_t>(0);
  std::array<bool, 3> axis_contiguous{};
  bool out_of_place = false;
  bool use_managed_memory = false;
  int nwarmup = 3;
  int ntrials = 5;
  bool skip_correctness_tests = false;
  double skip_threshold = 0.0;
  bool no_slab_opt = false;

  while (1) {
    static struct option long_options[] = {{"gx", required_argument, 0, 'x'},
                                           {"gy", required_argument, 0, 'y'},
                                           {"gz", required_argument, 0, 'z'},
                                           {"backend", required_argument, 0, 'b'},
                                           {"pr", required_argument, 0, 'r'},
                                           {"pc", required_argument, 0, 'c'},
                                           {"acx", required_argument, 0, '1'},
                                           {"acy", required_argument, 0, '2'},
                                           {"acz", required_argument, 0, '3'},
                                           {"nwarmup", required_argument, 0, 'w'},
                                           {"ntrials", required_argument, 0, 't'},
                                           {"skip-threshold", required_argument, 0, 'k'},
                                           {"out-of-place", no_argument, 0, 'o'},
                                           {"use-managed-memory", no_argument, 0, 'm'},
                                           {"skip-correctness-tests", no_argument, 0, 's'},
                                           {"no-slab-opt", no_argument, 0, 'n'},
                                           {"help", no_argument, 0, 'h'},
                                           {0, 0, 0, 0}};

    int option_index = 0;
    int ch = getopt_long(argc, argv, "x:y:z:b:r:c:1:2:3:w:t:k:b:omsnh", long_options, &option_index);
    if (ch == -1) break;

    switch (ch) {
    case 0: break;
    case 'x': gx = static_cast<uint64_t>(atoll(optarg)); break;
    case 'y': gy = static_cast<uint64_t>(atoll(optarg)); break;
    case 'z': gz = static_cast<uint64_t>(atoll(optarg)); break;
    case 'r': pr = atoi(optarg); break;
    case 'c': pc = atoi(optarg); break;
    case '1': axis_contiguous[0] = atoi(optarg); break;
    case '2': axis_contiguous[1] = atoi(optarg); break;
    case '3': axis_contiguous[2] = atoi(optarg); break;
    case 'w': nwarmup = atoi(optarg); break;
    case 't': ntrials = atoi(optarg); break;
    case 'k': skip_threshold = atof(optarg); break;
    case 'b': comm_backend = static_cast<cudecompTransposeCommBackend_t>(atoi(optarg)); break;
    case 'o': out_of_place = true; break;
    case 'm': use_managed_memory = true; break;
    case 's': skip_correctness_tests = true; break;
    case 'n': no_slab_opt = true; break;
    case 'h':
      if (rank == 0) { usage(argv[0]); }
      exit(EXIT_SUCCESS);
    case '?': exit(EXIT_FAILURE);
    default: fprintf(stderr, "unknown option: %c\n", ch); exit(EXIT_FAILURE);
    }
  }

  std::array<int, 2> pdims;
  pdims[0] = pr;
  pdims[1] = pc;

  // Initialize cuDecomp
  cudecompHandle_t handle;
  CHECK_CUDECOMP_EXIT(cudecompInit(&handle, MPI_COMM_WORLD));

  // Setup grid descriptors
  cudecompGridDescConfig_t config;
  CHECK_CUDECOMP_EXIT(cudecompGridDescConfigSetDefaults(&config));
  config.pdims[0] = pdims[0];
  config.pdims[1] = pdims[1];
  config.transpose_axis_contiguous[0] = axis_contiguous[0];
  config.transpose_axis_contiguous[1] = axis_contiguous[1];
  config.transpose_axis_contiguous[2] = axis_contiguous[2];

  cudecompGridDescAutotuneOptions_t options;
  CHECK_CUDECOMP_EXIT(cudecompGridDescAutotuneOptionsSetDefaults(&options));
  options.dtype = get_cudecomp_datatype(complex_t(0));
  for (int i = 0; i < 4; ++i) { options.transpose_use_inplace_buffers[i] = !out_of_place; }

  if (comm_backend != 0) {
    config.transpose_comm_backend = comm_backend;
  } else {
    options.autotune_transpose_backend = true;
  }
  options.skip_threshold = skip_threshold;

#ifdef R2C
  cudecompGridDesc_t grid_desc_c; // complex grid
  std::array<int32_t, 3> gdim_c{gx / 2 + 1, gy, gz};
  config.gdims[0] = gdim_c[0];
  config.gdims[1] = gdim_c[1];
  config.gdims[2] = gdim_c[2];
  CHECK_CUDECOMP_EXIT(cudecompGridDescCreate(handle, &grid_desc_c, &config, &options));

  cudecompGridDesc_t grid_desc_r;                          // real grid
  std::array<int32_t, 3> gdim_r{(gx / 2 + 1) * 2, gy, gz}; // with padding
  config.gdims[0] = gdim_r[0];
  config.gdims[1] = gdim_r[1];
  config.gdims[2] = gdim_r[2];
  CHECK_CUDECOMP_EXIT(cudecompGridDescCreate(handle, &grid_desc_r, &config, nullptr));

#else
  cudecompGridDesc_t grid_desc_c; // complex grid
  std::array<int32_t, 3> gdim_c{gx, gy, gz};
  config.gdims[0] = gdim_c[0];
  config.gdims[1] = gdim_c[1];
  config.gdims[2] = gdim_c[2];
  CHECK_CUDECOMP_EXIT(cudecompGridDescCreate(handle, &grid_desc_c, &config, &options));
#endif

  if (rank == 0) printf("Running on %d x %d process grid...\n", config.pdims[0], config.pdims[1]);
#ifdef R2C
  if (rank == 0) printf("Running on %d x %d x %d (real) spatial grid...\n", gdim_r[0], gdim_r[1], gdim_r[2]);
#else
  if (rank == 0) printf("Running on %d x %d x %d (complex) spatial grid...\n", gdim_c[0], gdim_c[1], gdim_c[2]);
#endif
  if (rank == 0)
    printf("Running with %s backend...\n", cudecompTransposeCommBackendToString(config.transpose_comm_backend));

#ifdef R2C
  // Get x-pencil information (real)
  cudecompPencilInfo_t pinfo_x_r;
  CHECK_CUDECOMP_EXIT(cudecompGetPencilInfo(handle, grid_desc_r, &pinfo_x_r, 0, nullptr));
#endif

  // Get x-pencil information (complex)
  cudecompPencilInfo_t pinfo_x_c;
  CHECK_CUDECOMP_EXIT(cudecompGetPencilInfo(handle, grid_desc_c, &pinfo_x_c, 0, nullptr));

  // Get y-pencil information (complex)
  cudecompPencilInfo_t pinfo_y_c;
  CHECK_CUDECOMP_EXIT(cudecompGetPencilInfo(handle, grid_desc_c, &pinfo_y_c, 1, nullptr));

  // Get z-pencil information (complex)
  cudecompPencilInfo_t pinfo_z_c;
  CHECK_CUDECOMP_EXIT(cudecompGetPencilInfo(handle, grid_desc_c, &pinfo_z_c, 2, nullptr));

  // Get workspace size
  int64_t num_elements_work_c;
  CHECK_CUDECOMP_EXIT(cudecompGetTransposeWorkspaceSize(handle, grid_desc_c, &num_elements_work_c));

  // Setup cuFFT
  bool slab_xy = false;
  bool slab_yz = false;
  bool slab_xyz = false;
#ifdef R2C
  // x-axis real-to-complex
  cufftHandle cufft_plan_r2c_x;
  CHECK_CUFFT_EXIT(cufftCreate(&cufft_plan_r2c_x));
  CHECK_CUFFT_EXIT(cufftSetAutoAllocation(cufft_plan_r2c_x, 0));
  size_t work_sz_r2c_x = 0;

  // x-axis complex-to-real
  cufftHandle cufft_plan_c2r_x;
  CHECK_CUFFT_EXIT(cufftCreate(&cufft_plan_c2r_x));
  CHECK_CUFFT_EXIT(cufftSetAutoAllocation(cufft_plan_c2r_x, 0));
  size_t work_sz_c2r_x = 0;

  if (!no_slab_opt && config.pdims[0] == 1 && config.pdims[1] == 1) {
    // single rank, x-y-z slab: use 3D FFT
    slab_xyz = true;
    CHECK_CUFFT_EXIT(cufftMakePlan3d(cufft_plan_r2c_x, gx, gy, gz, get_cufft_type_r2c(real_t(0)),
                                     &work_sz_r2c_x));
    CHECK_CUFFT_EXIT(cufftMakePlan3d(cufft_plan_c2r_x, gx, gy, gz, get_cufft_type_c2r(real_t(0)),
                                     &work_sz_c2r_x));
  } else if (!no_slab_opt && config.pdims[0] == 1) {
    // x-y slab: use 2D FFT
    slab_xy = true;
    std::array<int, 2> n{gx, gy};
    CHECK_CUFFT_EXIT(cufftMakePlanMany(
        cufft_plan_r2c_x, 2, n.data(), nullptr, 1, pinfo_x_r.shape[0] * pinfo_x_r.shape[1], nullptr, 1,
        pinfo_x_c.shape[0] * pinfo_x_c.shape[1], get_cufft_type_r2c(real_t(0)), pinfo_x_r.shape[2], &work_sz_r2c_x));
    CHECK_CUFFT_EXIT(cufftMakePlanMany(
        cufft_plan_c2r_x, 2, n.data(), nullptr, 1, pinfo_x_c.shape[0] * pinfo_x_c.shape[1], nullptr, 1,
        pinfo_x_r.shape[0] * pinfo_x_r.shape[1], get_cufft_type_c2r(real_t(0)), pinfo_x_c.shape[2], &work_sz_c2r_x));
  } else {
    CHECK_CUFFT_EXIT(cufftMakePlan1d(cufft_plan_r2c_x, gx, get_cufft_type_r2c(real_t(0)),
                                     pinfo_x_r.shape[1] * pinfo_x_r.shape[2], &work_sz_r2c_x));
    CHECK_CUFFT_EXIT(cufftMakePlan1d(cufft_plan_c2r_x, gx, get_cufft_type_c2r(real_t(0)),
                                     pinfo_x_c.shape[1] * pinfo_x_c.shape[2], &work_sz_c2r_x));
  }
#else
  // x-axis complex-to-complex
  cufftHandle cufft_plan_c2c_x;
  CHECK_CUFFT_EXIT(cufftCreate(&cufft_plan_c2c_x));
  CHECK_CUFFT_EXIT(cufftSetAutoAllocation(cufft_plan_c2c_x, 0));
  size_t work_sz_c2c_x = 0;

  if (!no_slab_opt && config.pdims[0] == 1 && config.pdims[1] == 1) {
    // single rank, x-y-z slab: use 3D FFT
    slab_xyz = true;
    CHECK_CUFFT_EXIT(cufftMakePlan3d(cufft_plan_c2c_x, gx, gy, gz, get_cufft_type_c2c(real_t(0)),
                                     &work_sz_c2c_x));
  } else if (!no_slab_opt && config.pdims[0] == 1) {
    // x-y slab: use 2D FFT
    slab_xy = true;
    std::array<int, 2> n{gy, gx};
    CHECK_CUFFT_EXIT(cufftMakePlanMany(
        cufft_plan_c2c_x, 2, n.data(), nullptr, 1, pinfo_x_c.shape[0] * pinfo_x_c.shape[1], nullptr, 1,
        pinfo_x_c.shape[0] * pinfo_x_c.shape[1], get_cufft_type_c2c(real_t(0)), pinfo_x_c.shape[2], &work_sz_c2c_x));
  } else {
    CHECK_CUFFT_EXIT(cufftMakePlan1d(cufft_plan_c2c_x, gx, get_cufft_type_c2c(real_t(0)),
                                     pinfo_x_c.shape[1] * pinfo_x_c.shape[2], &work_sz_c2c_x));
  }
#endif

  size_t fftsize = static_cast<size_t>(gx) * static_cast<size_t>(gy) * static_cast<size_t>(gz);

  // y-axis complex-to-complex
  cufftHandle cufft_plan_c2c_y;
  CHECK_CUFFT_EXIT(cufftCreate(&cufft_plan_c2c_y));
  CHECK_CUFFT_EXIT(cufftSetAutoAllocation(cufft_plan_c2c_y, 0));
  size_t work_sz_c2c_y = 0;
  if (!no_slab_opt && !slab_xyz && config.pdims[1] == 1) {
    // y-z slab: use 2D FFT
    slab_yz = true;
    if (axis_contiguous[1]) {
      std::array<int, 2> n{gz, gy};
      CHECK_CUFFT_EXIT(cufftMakePlanMany(
          cufft_plan_c2c_y, 2, n.data(), nullptr, 1, pinfo_y_c.shape[0] * pinfo_y_c.shape[1], nullptr, 1,
          pinfo_y_c.shape[0] * pinfo_y_c.shape[1], get_cufft_type_c2c(real_t(0)), pinfo_y_c.shape[2], &work_sz_c2c_y));
    } else {
      // Note: In this case, both slab dimensions are strided, leading to slower performance using
      // 2D FFT. Run 1D + 1D instead.
      slab_yz = false;
      CHECK_CUFFT_EXIT(cufftMakePlanMany(cufft_plan_c2c_y, 1, &gy /* unused */, &gy, pinfo_y_c.shape[0], 1, &gy,
                                         pinfo_y_c.shape[0], 1, get_cufft_type_c2c(real_t(0)), pinfo_y_c.shape[0],
                                         &work_sz_c2c_y));
      // std::array<int, 2> n{gz, gy};
      // std::array<int, 2> embed{pinfo_y_c.shape[2], pinfo_y_c.shape[1]};
      // CHECK_CUFFT_EXIT(cufftMakePlanMany(cufft_plan_c2c_y, 2, n.data(),
      //                                   embed.data(), pinfo_y_c.shape[0], 1,
      //                                   embed.data(), pinfo_y_c.shape[0], 1,
      //                                   get_cufft_type_c2c(real_t(0)), pinfo_y_c.shape[0], &work_sz_c2c_y));
    }
  } else {
    if (axis_contiguous[1]) {
      CHECK_CUFFT_EXIT(cufftMakePlan1d(cufft_plan_c2c_y, gy, get_cufft_type_c2c(real_t(0)),
                                       pinfo_y_c.shape[1] * pinfo_y_c.shape[2], &work_sz_c2c_y));
    } else {
      CHECK_CUFFT_EXIT(cufftMakePlanMany(cufft_plan_c2c_y, 1, &gy /* unused */, &gy, pinfo_y_c.shape[0], 1, &gy,
                                         pinfo_y_c.shape[0], 1, get_cufft_type_c2c(real_t(0)), pinfo_y_c.shape[0],
                                         &work_sz_c2c_y));
    }
  }

  // z-axis complex-to-complex
  cufftHandle cufft_plan_c2c_z;
  CHECK_CUFFT_EXIT(cufftCreate(&cufft_plan_c2c_z));
  CHECK_CUFFT_EXIT(cufftSetAutoAllocation(cufft_plan_c2c_z, 0));
  size_t work_sz_c2c_z = 0;
  if (axis_contiguous[2]) {
    CHECK_CUFFT_EXIT(cufftMakePlan1d(cufft_plan_c2c_z, gz, get_cufft_type_c2c(real_t(0)),
                                     pinfo_z_c.shape[1] * pinfo_z_c.shape[2], &work_sz_c2c_z));
  } else {
    CHECK_CUFFT_EXIT(cufftMakePlanMany(cufft_plan_c2c_z, 1, &gz /* unused */, &gz,
                                       pinfo_z_c.shape[0] * pinfo_z_c.shape[1], 1, &gz,
                                       pinfo_z_c.shape[0] * pinfo_z_c.shape[1], 1, get_cufft_type_c2c(real_t(0)),
                                       pinfo_z_c.shape[0] * pinfo_z_c.shape[1], &work_sz_c2c_z));
  }

  // Allocate data arrays
#ifdef R2C
  int64_t data_sz =
      std::max(std::max(std::max(2 * pinfo_x_c.size, 2 * pinfo_y_c.size), 2 * pinfo_z_c.size), pinfo_x_r.size) *
      sizeof(real_t);
  int64_t work_sz_decomp = 2 * num_elements_work_c * sizeof(real_t);
  int64_t work_sz_cufft = std::max(std::max(work_sz_r2c_x, std::max(work_sz_c2c_y, work_sz_c2c_z)), work_sz_c2r_x);
  int64_t work_sz = std::max(work_sz_decomp, work_sz_cufft);
#else
  int64_t data_sz = std::max(std::max(2 * pinfo_x_c.size, 2 * pinfo_y_c.size), 2 * pinfo_z_c.size) * sizeof(real_t);
  int64_t work_sz_decomp = 2 * num_elements_work_c * sizeof(real_t);
  int64_t work_sz_cufft = std::max(work_sz_c2c_x, std::max(work_sz_c2c_y, work_sz_c2c_z));
  int64_t work_sz = std::max(work_sz_decomp, work_sz_cufft);
#endif

  void* ref = malloc(data_sz);
  void* data = malloc(data_sz);

  void* work = malloc(work_sz);
  void *data_d, *data2_d, *work_d;

  if (use_managed_memory) {
    CHECK_CUDA_EXIT(cudaMallocManaged(&data_d, data_sz));
  } else {
    CHECK_CUDA_EXIT(cudaMalloc(&data_d, data_sz));
  }
  CHECK_CUDECOMP_EXIT(cudecompMalloc(handle, grid_desc_c, reinterpret_cast<void**>(&work_d), work_sz));
  if (out_of_place) {
    if (use_managed_memory) {
      CHECK_CUDA_EXIT(cudaMallocManaged(&data2_d, data_sz));
    } else {
      CHECK_CUDA_EXIT(cudaMalloc(&data2_d, data_sz));
    }
  }

  real_t* ref_r = static_cast<real_t*>(ref);
  real_t* data_r = static_cast<real_t*>(data);

  real_t* data_r_d = static_cast<real_t*>(data_d);
#ifdef R2C
  real_t* data2_r_d = static_cast<real_t*>(data2_d);
#endif
  complex_t* data_c_d = static_cast<complex_t*>(data_d);
  complex_t* data2_c_d = static_cast<complex_t*>(data2_d);
  complex_t* work_c_d = static_cast<complex_t*>(work_d);

  // Assign cuFFT work area
#ifdef R2C
  CHECK_CUFFT_EXIT(cufftSetWorkArea(cufft_plan_r2c_x, work_d));
  CHECK_CUFFT_EXIT(cufftSetWorkArea(cufft_plan_c2r_x, work_d));
#else
  CHECK_CUFFT_EXIT(cufftSetWorkArea(cufft_plan_c2c_x, work_d));
#endif
  CHECK_CUFFT_EXIT(cufftSetWorkArea(cufft_plan_c2c_y, work_d));
  CHECK_CUFFT_EXIT(cufftSetWorkArea(cufft_plan_c2c_z, work_d));

  if (!skip_correctness_tests) {
    // Initialize data to random values
    std::default_random_engine rng;
    std::uniform_real_distribution<real_t> dist(0, 1);
#ifdef R2C
    for (int64_t i = 0; i < pinfo_x_r.size; ++i) {
      ref_r[i] = (i % pinfo_x_r.shape[0] < gx) ? dist(rng) : 0;
      data_r[i] = ref_r[i];
    }
    CHECK_CUDA_EXIT(cudaMemcpy(data_r_d, data_r, pinfo_x_r.size * sizeof(real_t), cudaMemcpyHostToDevice));
#else
    for (int64_t i = 0; i < 2 * pinfo_x_c.size; ++i) {
      ref_r[i] = dist(rng);
      data_r[i] = ref_r[i];
    }
    CHECK_CUDA_EXIT(cudaMemcpy(data_r_d, data_r, 2 * pinfo_x_c.size * sizeof(real_t), cudaMemcpyHostToDevice));
#endif
  }

  // Run 3D FFT sequence
  complex_t* input = data_c_d;
  complex_t* output = data_c_d;
  if (out_of_place) output = data2_c_d;
#ifdef R2C
  real_t* input_r = data_r_d;
  real_t* output_r = data_r_d;
  if (out_of_place) output_r = data2_r_d;
#endif

  std::vector<double> trial_times(ntrials);
  double ts = 0;
  for (int trial = 0; trial < nwarmup + ntrials; ++trial) {
    if (trial == nwarmup) {
      CHECK_CUDA_EXIT(cudaDeviceSynchronize());
      CHECK_MPI_EXIT(MPI_Barrier(MPI_COMM_WORLD));
      ts = MPI_Wtime();
    }

#ifdef R2C
    CHECK_CUFFT_EXIT(cufftXtExec(cufft_plan_r2c_x, input_r, input, CUFFT_FORWARD));
#else
    CHECK_CUFFT_EXIT(cufftXtExec(cufft_plan_c2c_x, input, input, CUFFT_FORWARD));
#endif

    if (!slab_xyz) {
      CHECK_CUDECOMP_EXIT(cudecompTransposeXToY(handle, grid_desc_c, input, output, work_c_d,
                                                get_cudecomp_datatype(complex_t(0)), nullptr, nullptr, 0));
    }

    if (!slab_xy && !slab_xyz) {
      if (axis_contiguous[1] || slab_yz) {
        CHECK_CUFFT_EXIT(cufftXtExec(cufft_plan_c2c_y, output, output, CUFFT_FORWARD));
      } else {
        for (int i = 0; i < pinfo_y_c.shape[2]; ++i) {
          CHECK_CUFFT_EXIT(cufftXtExec(cufft_plan_c2c_y, output + i * (pinfo_y_c.shape[0] * pinfo_y_c.shape[1]),
                                       output + i * (pinfo_y_c.shape[0] * pinfo_y_c.shape[1]), CUFFT_FORWARD));
        }
      }
    }

    if (out_of_place) std::swap(input, output);
#ifdef R2C
    if (out_of_place) std::swap(input_r, output_r);
#endif

    // For y-z slab case, no need to perform yz transposes or z-axis FFT
    if (!slab_yz && !slab_xyz) {
      CHECK_CUDECOMP_EXIT(cudecompTransposeYToZ(handle, grid_desc_c, input, output, work_c_d,
                                                get_cudecomp_datatype(complex_t(0)), nullptr, nullptr, 0));
    }

    if (!slab_yz && !slab_xyz) {
      CHECK_CUFFT_EXIT(cufftXtExec(cufft_plan_c2c_z, output, output, CUFFT_FORWARD));
      CHECK_CUFFT_EXIT(cufftXtExec(cufft_plan_c2c_z, output, output, CUFFT_INVERSE));
    }

    if (out_of_place) std::swap(input, output);
#ifdef R2C
    if (out_of_place) std::swap(input_r, output_r);
#endif

    if (!slab_yz && !slab_xyz) {
      CHECK_CUDECOMP_EXIT(cudecompTransposeZToY(handle, grid_desc_c, input, output, work_c_d,
                                                get_cudecomp_datatype(complex_t(0)), nullptr, nullptr, 0));
    }

    if (!slab_xy && !slab_xyz) {
      if (axis_contiguous[1] || slab_yz) {
        CHECK_CUFFT_EXIT(cufftXtExec(cufft_plan_c2c_y, output, output, CUFFT_INVERSE));
      } else {
        for (int i = 0; i < pinfo_y_c.shape[2]; ++i) {
          CHECK_CUFFT_EXIT(cufftXtExec(cufft_plan_c2c_y, output + i * (pinfo_y_c.shape[0] * pinfo_y_c.shape[1]),
                                       output + i * (pinfo_y_c.shape[0] * pinfo_y_c.shape[1]), CUFFT_INVERSE));
        }
      }
    }

    if (out_of_place) std::swap(input, output);
#ifdef R2C
    if (out_of_place) std::swap(input_r, output_r);
#endif

    if (!slab_xyz) {
      CHECK_CUDECOMP_EXIT(cudecompTransposeYToX(handle, grid_desc_c, input, output, work_c_d,
                                                get_cudecomp_datatype(complex_t(0)), nullptr, nullptr, 0));
    }
#ifdef R2C
    CHECK_CUFFT_EXIT(cufftXtExec(cufft_plan_c2r_x, output, output_r, CUFFT_INVERSE));
#else
    CHECK_CUFFT_EXIT(cufftXtExec(cufft_plan_c2c_x, output, output, CUFFT_INVERSE));
#endif

    if (trial >= nwarmup) {
      CHECK_CUDA_EXIT(cudaDeviceSynchronize());
      CHECK_MPI_EXIT(MPI_Barrier(MPI_COMM_WORLD));
      double te = MPI_Wtime();
      trial_times[trial - nwarmup] = (te - ts) / 2; // division by two for fwd or bwd only time
    }

    // Note: excluding scaling from timing
#ifdef R2C
    scale<<<(pinfo_x_r.size + 1024 - 1) / 1024, 1024>>>(output_r, 1.0 / fftsize, pinfo_x_r);
    if (out_of_place) std::swap(input, output);
    if (out_of_place) std::swap(input_r, output_r);
#else
    scale<<<(pinfo_x_c.size + 1024 - 1) / 1024, 1024>>>(output, 1.0 / fftsize, pinfo_x_c);
    if (out_of_place) std::swap(input, output);
#endif

    if (trial >= nwarmup) {
      CHECK_CUDA_EXIT(cudaDeviceSynchronize());
      CHECK_MPI_EXIT(MPI_Barrier(MPI_COMM_WORLD));
      double te = MPI_Wtime();
      ts = te;
    }
  }

  // Check for errors
  double err_max_local = 0.0;
  if (!skip_correctness_tests) {
#ifdef R2C
    CHECK_CUDA_EXIT(cudaMemcpy(data_r, input_r, pinfo_x_r.size * sizeof(real_t), cudaMemcpyDeviceToHost));
    for (int64_t i = 0; i < pinfo_x_r.size; ++i) {
      if (i % pinfo_x_r.shape[0] < gx) {
        real_t ref_val = ref_r[i];
        real_t result_val = data_r[i];
        double err = std::abs((ref_val - result_val));
        err_max_local = std::max(err, err_max_local);
        if (!(err <= TOL)) {
          printf("FAILURE:  expected %f, got %f\n", ref_r[i], data_r[i]);
          exit(EXIT_FAILURE);
        }
      }
    }
#else
    CHECK_CUDA_EXIT(cudaMemcpy(data_r, input, 2 * pinfo_x_c.size * sizeof(real_t), cudaMemcpyDeviceToHost));
    for (int64_t i = 0; i < 2 * pinfo_x_c.size; ++i) {
      real_t ref_val = ref_r[i];
      real_t result_val = data_r[i];
      double err = std::abs((ref_val - result_val));
      err_max_local = std::max(err, err_max_local);
      if (!(err <= TOL)) {
        printf("FAILURE:  expected %f, got %f\n", ref_r[i], data_r[i]);
        exit(EXIT_FAILURE);
      }
    }
#endif
  }
  double err_max = 0.0;
  CHECK_MPI_EXIT(MPI_Reduce(&err_max_local, &err_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));

  // Gather memory usage info
  size_t free_mem, total_mem;
  CHECK_CUDA_EXIT(cudaMemGetInfo(&free_mem, &total_mem));
  double used_mem = (total_mem - free_mem) / (1024 * 1024);

  double used_mem_max = 0, used_mem_min = 0;
  CHECK_MPI_EXIT(MPI_Reduce(&used_mem, &used_mem_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
  CHECK_MPI_EXIT(MPI_Reduce(&used_mem, &used_mem_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD));

  // Compute performance statistics
  std::sort(trial_times.begin(), trial_times.end());
  double flopcount = 5.0 * fftsize * std::log(static_cast<double>(fftsize)) * 1e-9 / std::log(2.0);
  std::vector<double> trial_flops(ntrials);
  for (int i = 0; i < ntrials; ++i) { trial_flops[i] = flopcount / trial_times[i]; }

  auto times = process_timings(trial_times, 1000.);
  auto flops = process_timings(trial_flops);

  if (rank == 0) {
    printf("Result Summary:\n");
    printf("\t FFT size: %d x %d x %d  \n", gx, gy, gz);
#ifdef R2C
    printf("\t FFT mode: %s \n", "R2C");
#else
    printf("\t FFT mode: %s \n", "C2C");
#endif
#ifdef USE_FLOAT
    printf("\t Precision: %s \n", "single");
#else
    printf("\t Precision: %s \n", "double");
#endif
    printf("\t Process grid: %d x %d \n", config.pdims[0], config.pdims[1]);
    printf("\t Comm backend: %s \n", cudecompTransposeCommBackendToString(config.transpose_comm_backend));
    printf("\t Axis contiguous: %d %d %d \n", static_cast<int>(axis_contiguous[0]),
           static_cast<int>(axis_contiguous[1]), static_cast<int>(axis_contiguous[2]));
    printf("\t Out of place: %s \n", (out_of_place) ? "true" : "false");
    printf("\t Managed memory: %s \n", (use_managed_memory) ? "true" : "false");
    printf("\t Slab optimizations: %s \n", (slab_xy || slab_yz || slab_xyz) ? "true" : "false");
    printf("\t Time min/max/avg/std [ms]: %f/%f/%f/%f \n", times[0], times[1], times[2], times[3]);
    printf("\t Throughput min/max/avg/std [GFLOPS/s]: %f/%f/%f/%f \n", flops[0], flops[1], flops[2], flops[3]);
    if (skip_correctness_tests) {
      printf("\t Max error: <skipped>\n");
    } else {
      printf("\t Max error: %.10e \n", err_max);
    }
    printf("\t Memory usage range (per GPU) [MB]: %f - %f \n", used_mem_min, used_mem_max);
  }

  CHECK_CUDA_EXIT(cudaFree(data_d));
  CHECK_CUDECOMP_EXIT(cudecompFree(handle, grid_desc_c, work_d));
#ifdef R2C
  CHECK_CUDECOMP_EXIT(cudecompGridDescDestroy(handle, grid_desc_r));
#endif
  CHECK_CUDECOMP_EXIT(cudecompGridDescDestroy(handle, grid_desc_c));
  CHECK_CUDECOMP_EXIT(cudecompFinalize(handle));
  CHECK_MPI_EXIT(MPI_Finalize());
}
