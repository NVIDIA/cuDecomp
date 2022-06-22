import argparse
import itertools
import math
import os
import subprocess
import sys
import time

import yaml

def get_factors(n):
  factors = []
  for i in range(1, n + 1):
    if i > math.sqrt(n):
      break
    if n % i == 0:
      factors.append(i)
      if (n // i != i):
        factors.append(n // i)
  return sorted(factors)

def load_yaml_config(yaml_file, config_name):
  with open(yaml_file, 'r') as f:
    config = yaml.safe_load(f)[config_name]
  return config

def generate_command_lines(config, args):
  generic_cmd = f"{args.launcher_cmd}"
  generic_cmd += " {0}"

  generic_cmd += " --pr {1} --pc {2}"
  if config["skip_correctness_testing"]:
    generic_cmd += " -s"

  generic_cmd += f" --gx {args.gx} --gy {args.gy} --gz {args.gz}"

  cmds = []
  prs = get_factors(args.ngpu)
  pcs = [args.ngpu // x for x in prs]
  if config['run_autotuning']:
    prs = [0] + prs
    pcs = [0] + pcs
    if (not 0 in config['backend']):
      config['backend'].append(0)

  for pr, pc in zip(prs, pcs):
    for vals in itertools.product(*[config[x] for x in config['args']]):
      arg_dict = dict(zip(config['args'], vals))
      cmd = generic_cmd.format(f"{config['executable_prefix']}", pr, pc)
      cmd += " "  + " ".join([f"--{x} {y}" for x, y in arg_dict.items()])

      # Only run full (grid and backend) autotuning cases
      if (config['run_autotuning']):
        if ((pr == 0 and pc == 0) and arg_dict["backend"] != 0):
          continue
        elif ((pr != 0 and pc != 0) and arg_dict["backend"] == 0):
          continue

      extra_flags = []
      extra_flags.append(['-m' if x else '' for x in config['managed_memory']])
      extra_flags.append(['-o' if x else '' for x in config['out_of_place']])
      extra_flags.append(['--acx 1 --acy 1 --acz 1' if x else '' for x in config['axis_contiguous']])
      for extras in itertools.product(*extra_flags):
        cmds.append(f"{cmd} {' '.join(filter(lambda x: x != '', extras))}")

  return cmds

def setup_env(config, args):
  print("Setting environment variables...")
  # Environment variables from config
  for var, val in config['env_vars'].items():
    os.environ[var] = f"{val}"
    print(f"Set {var} = {val}")

  # Dynamically setting NVSHMEM_SYMMETRIC_SIZE based on expected workspace size
  # for input grid dimensions (with 5% margin)
  symmetric_size = 0
  wordsize = 8 if "_f" in config["executable_prefix"] else 16
  if "c2c" in config["executable_prefix"]:
    symmetric_size = 2 * (args.gx * args.gy * args.gz * wordsize) // args.ngpu
  else:
    symmetric_size = 2* ((args.gx // 2 + 1) * args.gy * args.gz * wordsize) // args.ngpu

  symmetric_size = int(1.05 * symmetric_size)
  os.environ["NVSHMEM_SYMMETRIC_SIZE"] = f"{symmetric_size}"
  print(f"Set NVSHMEM_SYMMETRIC_SIZE = {symmetric_size}")

def add_csv_entry(csvfile, cmd, stdout_str):
  if not os.path.exists(csvfile):
    # Create file with header
    with open(csvfile, 'w') as f:
      f.write("nx,ny,nz,fft_mode,precision,pr,pc,backend,"
              "acx,acy,acz,out_of_place,managed,"
              "tmin,tmax,tavg,tstd,gfmin,gfmax,gfavg,gfstd,"
              "at_grid,at_backend,at_results\n")

  at_grid = "--pr 0" in cmd and "--pc 0" in cmd
  at_backend = "--backend 0" in cmd

  lines = stdout_str.split('\n')

  at_lines = [x for x in lines if "CUDECOMP:" in x]
  at_results = []
  if at_lines:
    for i, line in enumerate(at_lines):
      if "grid:" in line and not "SELECTED" in line:
        grid = line.split(', ')[0].split(': ')[-1].strip()
        backend = line.split(', ')[1].split(': ')[-1].strip()
        tmin, tmax, tavg, tstd = [float(x) for x in at_lines[i+1].split(': ')[-1].split('/')]
        at_results.append(f"{grid},{backend},{tmin},{tmax},{tavg},{tstd}")

  if at_results:
    at_results_str = ";".join(at_results)
  else:
    at_results_str = ""

  results_lines = stdout_str.split("Result Summary:")[-1].split('\n')

  for line in results_lines:
    line = line.strip()
    if "FFT size" in line:
      nx, ny, nz = [int(x) for x in line.split(': ')[1].split('x')]
    elif "FFT mode" in line:
      fft_mode = line.split(': ')[1]
    elif "Precision" in line:
      precision = line.split(': ')[1]
    elif "Process grid" in line:
      pr, pc = [int(x) for x in line.split(': ')[1].split('x')]
    elif "Comm backend" in line:
      backend = line.split(': ')[1]
    elif "Axis contiguous" in line:
      acx, acy, acz = [int(x) for x in line.split(': ')[1].split(' ')]
    elif "Out of place" in line:
      out_of_place = True if "true" in line.split(': ')[1] else False
    elif "Managed memory" in line:
      managed = True if "true" in line.split(': ')[1] else False
    elif "Time" in line:
      tmin, tmax, tavg, tstd = [float(x) for x in line.split(': ')[1].split('/')]
    elif "Throughput" in line:
      gfmin, gfmax, gfavg, gfstd = [float(x) for x in line.split(': ')[1].split('/')]

  with open(csvfile, 'a') as f:
    f.write(f"{nx},{ny},{nz},{fft_mode},{precision},{pr},{pc},{backend},"
            f"{acx},{acy},{acz},{out_of_place},{managed},"
            f"{tmin},{tmax},{tavg},{tstd},{gfmin},{gfmax},{gfavg},{gfstd},"
            f"{at_grid},{at_backend},\"{at_results_str}\"\n")

def run_test(cmd, args):
  print(f"command: {cmd}", sep="")
  cmd_fields = cmd.split()

  failed = False
  try:
    status = subprocess.run(cmd_fields, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            timeout=300, check=True)
  except subprocess.TimeoutExpired as ex:
    print(f" FAILED (timeout)")
    print(f"Failing output:\n{ex.stdout.decode('utf-8')}")
    failed = True
  except subprocess.CalledProcessError as ex:
    print(f" FAILED")
    print(f"Failing output:\n{ex.stdout.decode('utf-8')}")
    failed = True
  else:
    print(" PASSED")
    if args.csvfile:
      add_csv_entry(args.csvfile, cmd, status.stdout.decode('utf-8'))

  if failed:
    if args.exit_on_failure:
      print("Stopping tests...")
      sys.exit(1)
    return False

  if args.verbose:
    print(f"Passing output:\n{status.stdout.decode('utf-8')}")

  return True

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--launcher_cmd', type=str, required=True, help='parallel launch command')
  parser.add_argument('--ngpu', type=int, required=True, help='number of gpus')
  parser.add_argument('--gx', type=int, required=True, help='fft dimension in X')
  parser.add_argument('--gy', type=int, required=True, help='fft dimension in Y')
  parser.add_argument('--gz', type=int, required=True, help='fft dimension in Z')
  parser.add_argument('--csvfile', type=str, default='', required=False, help='csv file to dump results (will append if file exists)')
  parser.add_argument('--verbose', action='store_true', required=False, help='flag to enable full run output')
  parser.add_argument('--exit_on_failure', action='store_true', required=False, help='flag to control whether script exits on case failure')
  parser.add_argument('config_name', type=str, help='configuration name from benchmark_configs.yaml')
  args = parser.parse_args()

  config = load_yaml_config("benchmark_config.yaml", args.config_name)

  cmds = generate_command_lines(config, args)
  setup_env(config, args)

  print(f"Running {len(cmds)} tests...")
  t0 = time.time()
  failed_cmds = []
  for i,c in enumerate(cmds):
    status = run_test(c, args)

    if not status:
      failed_cmds.append(c)

    if (i+1) % 10 == 0:
      t1 = time.time()
      print(f"Completed {i+1}/{len(cmds)} tests, running time {t1-t0} s")
  print(f"Completed all tests, running time {time.time() - t0} s")

  if len(failed_cmds) == 0:
    print("Passed all tests.")
    return 0
  else:
    print(f"Failed {len(failed_cmds)} / {len(cmds)} tests. Failing commands:")
    for c in failed_cmds:
      print(f"\t{c}")
    return -1

if __name__ == "__main__":
  main()
