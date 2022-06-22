import argparse
import collections
import itertools
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas

def get_cases(df):
  cases = collections.OrderedDict()
  cases["grid"] = list(set((x, y, z) for x, y, z in zip(df["nx"], df["ny"], df["nz"])))
  cases["precision"] = df["precision"].unique()
  cases["axis_contiguous"] = list(set((x, y, z) for x, y, z in zip(df["acx"], df["acy"], df["acz"])))
  cases["out_of_place"] = df["out_of_place"].unique()
  cases["managed"] = df["managed"].unique()
  cases["fft_mode"] = df["fft_mode"].unique()

  return cases

def get_case_df(df, grid, precision, axis_contiguous, out_of_place, managed, fft_mode):
    cond = (df["nx"] == grid[0]) & (df["ny"] == grid[1]) & (df["nz"] == grid[2])
    cond = (cond) & (df["precision"] == precision)
    cond = (cond) & (df["acx"] == axis_contiguous[0]) & (df["acy"] == axis_contiguous[1]) & (df["acz"] == axis_contiguous[2])
    cond = (cond) & (df["out_of_place"] == out_of_place)
    cond = (cond) & (df["managed"] == managed)
    cond = (cond) & (df["fft_mode"] == fft_mode)

    return df[cond], f"{fft_mode}, {precision}, grid: {grid}, axis_contiguous: {axis_contiguous}, out_of_place: {out_of_place}, managed: {managed}"

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

def plot_heatmap(ax, data, backends, factors, cmap_name='Greens', x_label=True, y_label=True,
                 mark_min_col=False, mark_min_row=False, mark_min_all=False, precision=0,
                 title=""):

  ax.matshow(data, vmin=np.min(data), vmax=np.max(data), interpolation='none', cmap=plt.get_cmap(cmap_name))

  ax.xaxis.set_ticks(np.arange(0,data.shape[1]))
  ax.yaxis.set_ticks(np.arange(0,data.shape[0]))
  ax.xaxis.set_ticks(np.arange(0,data.shape[1])-0.5, minor=True)
  ax.yaxis.set_ticks(np.arange(0,data.shape[0])-0.5, minor=True)
  ax.grid(which='minor', color='w', linestyle='-', linewidth=2)

  if x_label:
    ax.xaxis.set_ticklabels([f"{x} x {factors[-1] // x}" for x in factors], fontsize=12)
  else:
    ax.xaxis.set_ticklabels([])

  if y_label:
    ax.yaxis.set_ticklabels(backends, fontsize=12)
  else:
    ax.yaxis.set_ticklabels([])

  ax.tick_params(axis=u'both', which=u'both',length=0)

  cm = plt.get_cmap('Greys_r')
  for i in range(data.shape[0]):
    for j in range(data.shape[1]):
      #color = cm((data[i,j]-np.min(data))/(np.max(data) - np.min(data)))
      color = cm(float((data[i,j]-np.min(data))/(np.max(data) - np.min(data)) > 0.5))
      ndigits = int(np.log10(data[i,j])) + precision
      ax.text(j-0.165*ndigits/2, i, f"{{:.{precision}f}}".format(data[i,j]), color=color, fontsize=12, fontweight='demi')

      if (mark_min_col and data[i,j] == np.min(data[:,j])):
        ax.plot(j-0.39, i+0.39, color=color, marker='v', markersize=10)
      if (mark_min_row and data[i,j] == np.min(data[i,:])):
        ax.plot(j+0.39, i-0.39, color=color, marker='>', markersize=10)
      if (mark_min_all and data[i,j] == np.min(data[:,:])):
        ax.plot(j-0.39, i-0.39, color=color, marker='*', markersize=10)

  if title:
    ax.set_title(title, y=-0.05)

def plot_case_heatmaps(df_in, case_label, outputfile):
  backends = sorted(df_in["backend"].unique().tolist())
  ngpu = df_in["pr"].iloc[0] * df_in["pc"].iloc[0]
  factors = get_factors(ngpu)
  heatmap_data = np.zeros([len(backends), len(factors)])

  fig, axes = plt.subplots(1,2,figsize=(3*len(factors),8))
  fig.suptitle(case_label, fontsize=14)

  # Plot FFT GFLOPS heatmap
  df = df_in[(~df_in["at_grid"]) & (~df_in["at_backend"])]
  for b in backends:
    df_backend = df[df["backend"] == b]

    heatmap_i = backends.index(b)
    for i, row in df_backend.iterrows():
      pr = row["pr"]
      pc = row["pc"]
      gfavg = row["gfavg"]
      heatmap_j = factors.index(pr)

      if (heatmap_data[heatmap_i, heatmap_j] != 0):
        print("Duplicate entry detected!")
      heatmap_data[heatmap_i, heatmap_j] = gfavg
  plot_heatmap(axes[0], heatmap_data, backends, factors,
               title="FFT Performanace [GFLOP/s]")

   # Plot full autotuned heatmap
  heatmap_data[:] = 0
  df = df_in[(df_in["at_grid"]) & (df_in["at_backend"])]
  for i, row in df.iterrows():
    at_results = row["at_results"].split(';')
    for entry in at_results:
      fields = entry.split(',')
      pr, pc = [int(x) for x in fields[0].split('x')]
      backend = fields[1]
      tmin, tmax, tavg, tstd = [float(x) for x in fields[2:]]
      heatmap_i = backends.index(backend)
      heatmap_j = factors.index(pr)
      if (heatmap_data[heatmap_i, heatmap_j] != 0):
        print("Duplicate entry detected!")
      heatmap_data[heatmap_i, heatmap_j] = tavg
  plot_heatmap(axes[1], heatmap_data, backends, factors, cmap_name='Purples',
               x_label=True, y_label=False, precision=2, mark_min_all=True,
               mark_min_col=True, mark_min_row=True, title="Avg. Transpose Trial Time [ms]")

  plt.tight_layout()
  plt.savefig(outputfile)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--csvfile', type=str, required=True, help='csvfile containing benchmark run data')
  parser.add_argument('--output_prefix', type=str, required=True, help='Output prefix for heatmap PNG files. Output is written to <output_prefix>_*.png, with one file per case found in csv.')
  args = parser.parse_args()

  df = pandas.read_csv(args.csvfile)
  cases = get_cases(df)
  for i, (grid, precision, axis_contiguous, out_of_place, managed, fft_mode) in enumerate(itertools.product(*cases.values())):
    case_df, case_label = get_case_df(df, grid, precision, axis_contiguous, out_of_place, managed, fft_mode)
    plot_case_heatmaps(case_df, case_label, f"{args.output_prefix}_{i}.png")

if __name__ == "__main__":
  main()
