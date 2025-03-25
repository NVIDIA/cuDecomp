import argparse
import ast
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

def should_skip_case(arg_dict, key):
  skip = False

  if key == 'transpose':
    # Skip cases with all halo extents, padding, and gdimdist as zero
    if ((arg_dict["hex"] == "0 0 0" and arg_dict["hey"] == "0 0 0" and arg_dict["hez"] == "0 0 0") and
        (arg_dict["pdx"] == "0 0 0" and arg_dict["pdy"] == "0 0 0" and arg_dict["pdz"] == "0 0 0") and
        (arg_dict["gd"] == "0 0 0")):
      skip = True

    # Skip cases where halo extents in X and Z are unequal as these cases are redundant
    if (arg_dict["hex"] != arg_dict["hez"]):
      skip = True

    # Skip cases where padding in X and Z are unequal as these cases are redundant
    if (arg_dict["pdx"] != arg_dict["pdz"]):
      skip = True

  elif key == 'transpose_mix':
    skip = should_skip_case(arg_dict, 'transpose')

    # Skip cases where all halo extents are zero
    if (arg_dict["hex"] == "0 0 0" and arg_dict["hey"] == "0 0 0" and arg_dict["hez"] == "0 0 0"):
      skip = True

    # Skip cases where all padding is zero
    if (arg_dict["pdx"] == "0 0 0" and arg_dict["pdy"] == "0 0 0" and arg_dict["pdz"] == "0 0 0"):
      skip = True

  elif key == 'halo':
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

  elif key == 'halo_padding':
    skip = should_skip_case(arg_dict, 'halo')

    # Skip cases where all padding is zero
    if (arg_dict["pdx"] == 0 and arg_dict["pdy"] == 0 and arg_dict["pdz"] == 0):
      skip = True


  return skip

def generate_mem_order_args(zero_indexed, is_halo_test=False):
   if zero_indexed:
     orders_ax = [" ".join(x) for x in itertools.permutations(["0", "1", "2"])]
   else:
     orders_ax = [" ".join(x) for x in itertools.permutations(["1", "2", "3"])]
   if is_halo_test:
     args = orders_ax
   else:
     args = [" ".join([x, y, x]) for x, y in itertools.product(orders_ax, orders_ax)]

   return args

def generate_command_lines(config, args):
  if (config["test_mem_order"]):
    config['args'].append("mem_order")
    config['mem_order'] = generate_mem_order_args(not config["fortran_indexing"], config["is_halo_test"])

  if (config["test_mem_order_override"]):
    config['args'].append("mem_order_override")
    config['mem_order_override'] = generate_mem_order_args(not config["fortran_indexing"], config["is_halo_test"])

  cmds = []
  prs = get_factors(args.ngpu)
  if len(prs) > 3:
    prs = [prs[0], prs[len(prs) // 2], prs[-1]]
  if config['use_single_pdim']:
    prs = [prs[min(len(prs) + 1, 1)]]

  pcs = [args.ngpu // x for x in prs]
  if config['run_autotuning']:
    prs = [0] + prs
    pcs = [0] + pcs
    if (not 0 in config['backend']):
      config['backend'].append(0)
  for pr, pc in zip(prs, pcs):
    for vals in itertools.product(*[config[x] for x in config['args']]):
      arg_dict = dict(zip(config['args'], vals))
      cmd = f"--pr {pr} --pc {pc} "
      cmd += " ".join([f"--{x} {y}" for x, y in arg_dict.items()])

      # Only run full (grid and backend) autotuning cases
      if (config['run_autotuning']):
        if ((pr == 0 and pc == 0) and arg_dict["backend"] != 0):
          continue
        elif ((pr != 0 and pc != 0) and arg_dict["backend"] == 0):
          continue

      # Check additional skip conditions
      if (config['apply_skips']):
        if should_skip_case(arg_dict, config['apply_skips']): continue

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
  parser.add_argument('--config_overrides', type=str, required=False, help="string list of semi-colon separated key:value pairs, e.g. \"backend:[1,2];dtype:['C64']\"")
  parser.add_argument('config_name', type=str, help='configuration name from test_configs.yaml')
  args = parser.parse_args()

  config = load_yaml_config("test_config.yaml", args.config_name)

  if (args.config_overrides):
    entries = args.config_overrides.split(';')
    for e in entries:
      fields = e.split(':')
      key = fields[0].strip()
      value = ast.literal_eval(fields[1].strip())
      config[key] = value

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
    cmd = f"{args.launcher_cmd} {config['executable_prefix']}_{dtype} --testfile {args.config_name}_cases.txt"
    status = run_test(cmd, args)

    if not status:
      failed_dtypes.append(dtype)
      print(f"Failed {dtype} tests.")

      if (args.exit_on_failure):
        print("Stopping tests...")
        os.remove(f"{args.config_name}_cases.txt")
        return 1

  if len(failed_dtypes) == 0:
    print(f"Passed all tests for all dtypes, running time {time.time() - t0} s")
    retcode = 0
  else:
    print(f"Failed tests for dtypes ({', '.join(failed_dtypes)})., running time {time.time() - t0} s")
    retcode = 1

  os.remove(f"{args.config_name}_cases.txt")
  return retcode

if __name__ == "__main__":
  main()
