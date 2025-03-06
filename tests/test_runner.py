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

def should_skip_case(arg_dict):
  skip = False

  try:
    # No need to test periodic flags if halo extent is zero
    if arg_dict["hex"] == 0 and arg_dict["hpx"] == 1:
      skip = True
    if arg_dict["hey"] == 0 and arg_dict["hpy"] == 1:
      skip = True
    if arg_dict["hez"] == 0 and arg_dict["hpz"] == 1:
      skip = True

    # Skip cases with all halo extents as zero
    if arg_dict["hex"] == 0 and arg_dict["hey"] == 0 and arg_dict["hez"] == 0:
      skip = True
  except:
    pass

  return skip

def generate_mem_order_args(zero_indexed):
   if zero_indexed:
     orders_ax = [" ".join(x) for x in itertools.permutations(["0", "1", "2"])]
   else:
     orders_ax = [" ".join(x) for x in itertools.permutations(["1", "2", "3"])]
   args = [" ".join([x, y, x]) for x, y in itertools.product(orders_ax, orders_ax)]

   return args

def generate_command_lines(config, args):
  if (config["test_mem_order"]):
    config['args'].append("mem_order")
    config['mem_order'] = generate_mem_order_args(not config["fortran_indexing"])

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
      #cmd = generic_cmd.format(f"{config['executable_prefix']}_{dtype}", pr, pc)
      cmd = f"--pr {pr} --pc {pc} "
      cmd += " ".join([f"--{x} {y}" for x, y in arg_dict.items()])

      # Only run full (grid and backend) autotuning cases
      if (config['run_autotuning']):
        if ((pr == 0 and pc == 0) and arg_dict["backend"] != 0):
          continue
        elif ((pr != 0 and pc != 0) and arg_dict["backend"] == 0):
          continue

      # Check additional skip conditions
      if should_skip_case(arg_dict): continue

      extra_flags = []
      extra_flags.append(['-m' if x else '' for x in config['managed_memory']])
      extra_flags.append(['-o' if x else '' for x in config['out_of_place']])
      for extras in itertools.product(*extra_flags):
        cmds.append(f"{cmd} {' '.join(filter(lambda x: x != '', extras))}")


  return cmds

def setup_env(config, args):
  print("Setting environment variables...")
  # Environment variables from config
  for var, val in config['env_vars'].items():
    os.environ[var] = f"{val}"
    print(f"Set {var} = {val}")

def run_test(cmd, args):
  cmd_fields = cmd.split()
  try:
    sp = subprocess.Popen(cmd_fields, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    while sp.poll() is None:
      line = sp.stdout.readline()
      print(line.decode('utf-8'), end='')

    if sp.poll() != 0:
      return False

  except:
    return False

  return True

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--launcher_cmd', type=str, required=True, help='parallel launch command')
  parser.add_argument('--ngpu', type=int, required=True, help='number of gpus')
  parser.add_argument('--exit_on_failure', action='store_true', required=False, help='flag to control whether script exits on case failure')
  parser.add_argument('config_name', type=str, help='configuration name from test_configs.yaml')
  args = parser.parse_args()

  config = load_yaml_config("test_config.yaml", args.config_name)

  cmds = generate_command_lines(config, args)
  with open(f"{args.config_name}_cases.txt", 'w') as f:
    for c in cmds:
      f.write(c)
      f.write("\n")
  setup_env(config, args)

  t0 = time.time()
  failed_dtypes = []
  print(f"Running tests for dtypes ({', '.join(config['dtypes'])})...")
  for dtype in config['dtypes']:
    print(f"Running {dtype} tests...")
    cmd = f"{args.launcher_cmd} {config['executable_prefix']}_{dtype} -f {args.config_name}_cases.txt"
    status = run_test(cmd, args)

    if not status:
      failed_dtypes.append(dtype)
      print(f"Failed {dtype} tests.")

      if (args.exit_on_failure):
        print("Stopping tests...")
        return 1

  if len(failed_dtypes) == 0:
    print(f"Passed all tests for all dtypes, running time {time.time() - t0} s")
    return 0
  else:
    print(f"Failed tests for dtypes ({', '.join(failed_dtypes)})., running time {time.time() - t0} s")
    return 1


if __name__ == "__main__":
  main()
