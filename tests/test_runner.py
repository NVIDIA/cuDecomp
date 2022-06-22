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

def generate_command_lines(config, args):
  generic_cmd = f"{args.launcher_cmd}"
  generic_cmd += " {0}"

  generic_cmd += " --pr {1} --pc {2}"

  cmds = []
  for dtype in config['dtypes']:
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
        cmd = generic_cmd.format(f"{config['executable_prefix']}_{dtype}", pr, pc)
        cmd += " "  + " ".join([f"--{x} {y}" for x, y in arg_dict.items()])

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
  parser.add_argument('--verbose', action='store_true', required=False, help='flag to enable full run output')
  parser.add_argument('--exit_on_failure', action='store_true', required=False, help='flag to control whether script exits on case failure')
  parser.add_argument('config_name', type=str, help='configuration name from test_configs.yaml')
  args = parser.parse_args()

  config = load_yaml_config("test_config.yaml", args.config_name)

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
