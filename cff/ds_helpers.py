from typing import List
import numpy as np
import itertools
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec
from scipy.stats import ttest_ind

def normalize(x):
  return (x - np.mean(x)) / np.std(x)

def minmax_normalizer(df):
  return (df - df.min()) / (df.max() - df.min())

# def counterfactual_fairness(P, *protected_attrs: List[np.ndarray]):
#   unique_vals = (np.unique(attr) for attr in protected_attrs)
#   interventions = itertools.product(*unique_vals)
#   for intervention in interventions:


def build_plot(P, A, R, S, G, L, F, colors=None, pc_samps=1000, figscale=8, fontsize=20):
  if colors is None:
    colors = {
      (0, 0): [(0.882, 0.529, 0.000, 1.000), (1.000, 0.647, 0.000, 0.500)],
      (1, 0): [(0.882, 0.000, 0.000, 1.000), (1.000, 0.000, 0.000, 0.500)],
      (0, 1): [(0.000, 0.882, 0.000, 1.000), (0.000, 1.000, 0.000, 0.500)],
      (1, 1): [(0.000, 0.000, 0.882, 1.000), (0.000, 0.000, 1.000, 0.500)]
    }
  gs = GridSpec(3, 4)
  gs.update(wspace=0, hspace=0)
  kwargs_hist = dict(bins=25, histtype='stepfilled', stacked=True)
  kwargs_text = dict(horizontalalignment='left', verticalalignment='top', fontsize=fontsize)
  fig = plt.figure(figsize=(4 * figscale, 3 * figscale))
  ax_dict = dict()
  for i, tup in enumerate(itertools.product([0, 1], [0, 1])):
    j, k = tup
    ind = (R == j) & (S == k)
    ax = fig.add_subplot(gs[j, k])
    ax.hist([A[ind & P], A[ind & ~P]], color=colors[tup], **kwargs_hist)
    ax.axvline(x=0, ls='dotted', color='black')
    ax.text(0.02, 0.98, 'R={0:}, S={1:}'.format(j, k), transform=ax.transAxes, **kwargs_text)
    ax.set_yticks([])
    ax.set_xlim([-5, 5])
    ax.set_xticks([])
    ax_dict[i] = ax
  ylim = [0, 1.05 * max([ax.get_ylim()[1] for ax in ax_dict.values()])]
  for ax in ax_dict.values(): ax.set_ylim(ylim)
  ax = fig.add_subplot(gs[0:2, 2:])
  # ax.hist([A[P], A[~P]], color=['darkgray', 'lightgray'], **kwargs_hist)
  ax.hist([F[P], F[~P]], color=['darkgray', 'lightgray'], **kwargs_hist)
  ax.axvline(x=0, ls='dotted', color='black')
  ax.text(0.01, 0.99, 'All', transform=ax.transAxes, **kwargs_text)
  ax.set_yticks([])
  ax.set_xlim([-5, 5])
  ax.set_xticks([])
  ax = fig.add_subplot(gs[2:, 0:])
  z = ['A', 'G', 'L', 'F']
  x = range(len(z))
  df = pd.DataFrame({'A': A.flat, 'G': G.flat, 'L': L.flat, 'F': F.flat}, columns=z)
  df = minmax_normalizer(df)
  idx = np.random.choice(range(len(df)), pc_samps)
  colors = pd.DataFrame({'R': R.flat, 'S': S.flat}, columns=['R', 'S'])\
    .apply(tuple, axis=1).apply(lambda i: colors[i])
  for i in df.index[idx]:
    color = colors[i][0] if P[i] else colors[i][1]
    alpha = 0.100 if P[i] else 0.008
    ax.plot(x,
            df.loc[i],
            color=color,
            alpha=alpha)
  ax.set_ylim([0, 1])
  ax.set_xlim([x[0], x[-1]])
  ax.set_xticks(x)
  ax.set_xticklabels(z)
  for _x in x: ax.axvline(x=_x, lw=1, ls='dotted', color='black')
  for tick in ax.yaxis.get_major_ticks():
    tick.label1On = False
    tick.tick1On = False
  for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(fontsize)
  return fig
