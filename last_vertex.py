#!/usr/bin/env python3.11
"""
Author: David Brewster (dbrewster@g.harvard.edu)
Summary: The goal of this simulation code is to determine
which vertex is likely to be the final vertex to be infected
given that fixation occurs.
"""
from dataclasses import dataclass
from functools import lru_cache, partial
from multiprocessing import Pool
from pathlib import Path
from typing import Callable, Iterable, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import seaborn as sns

from utils import sample, trial_cond_fix_last_vertex

plt.rc('text.latex', preamble=r'\usepackage{amssymb}')

INITIAL_NODE_PLACEMENT = 0
NUM_WORKERS = 8

@dataclass
class GraphGenerator:
  name: str
  generate: Callable[[int], nx.DiGraph]
  layout: Optional[Callable[[nx.DiGraph], dict]] = None

def work(G, N, samples, r):
  print(f"{r=}")
  results = []
  for vertex in sample(lambda: trial_cond_fix_last_vertex(G, {INITIAL_NODE_PLACEMENT}, r), samples):
    results.append((N, r, vertex))
  return results

def last_vertex(
  Ns: List[int],
  graph_generator: GraphGenerator,
  Rs: Iterable[float],
  samples: int = 1000,
  overwrite: bool = True,
  use_existing_file: bool = False,
) -> pd.DataFrame:
  file_name = f"data/{graph_generator.name}-estimated-N-vs-ft-{max(Ns)}.pkl"
  # Ns = list(range(2, N+1))
  if use_existing_file and Path(file_name).exists():
    df = pd.read_pickle(file_name)
  else:
    data = []
    for N in Ns:
      print(f"{N=}")
      G = graph_generator.generate(N)

      with Pool(NUM_WORKERS) as p:
        for results in p.map(partial(work, G, N, samples), Rs):
          data.extend(results)

    df = pd.DataFrame(data, columns=["Population size", "r", "Last vertex"])
    if overwrite:
      df.to_pickle(file_name)

  return df

def is_undirected(G: nx.DiGraph):
  return all((v, u) in G.edges() for (u, v) in G.edges())

def plot_last_vertices_graph(df: pd.DataFrame, N, graph_generator: GraphGenerator, samples, R, relative=True):
  freqs = df[(df["Population size"] == N) & (df["r"] == R)]["Last vertex"].value_counts(normalize=True).to_dict()
  G = graph_generator.generate(N)
  if is_undirected(G):
    G = nx.to_undirected(G)
  cmap = plt.get_cmap('Blues')
  pos = graph_generator.layout(G) if graph_generator.layout else nx.kamada_kawai_layout(G)
  values = [freqs.get(node, 0) for node in G.nodes()]
  norm = plt.Normalize(vmin=min(values) if relative else 0, vmax=max(values) if relative else 1)
  node_colors = [cmap(norm(value)) for value in values]
  print(node_colors)
  
  nx.draw_networkx_nodes(
    G,
    pos,
    cmap=cmap,
    node_color=node_colors,
    edgecolors='black',
  )
  # Create a color bar
  sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
  sm.set_array([])
  cbar = plt.colorbar(sm, orientation='vertical')
  cbar.set_label('Probability')
  cbar.set_ticks([min(values), max(values)])
  cbar.set_ticklabels([f'min={min(values)}', f'max={max(values)}'])

  nx.draw_networkx_edges(G, pos) # , connectionstyle="arc3,rad=0.1", arrows=True)

  # for color, node in zip(font_colors, G.nodes()):
  nx.draw_networkx_labels(G, pos, {INITIAL_NODE_PLACEMENT: r'$\bigstar$'}, font_color='red')

  plt.savefig(f'figs/{graph_generator.name}-N-{N}-R-{R}-samples-{samples}.png', dpi=300, bbox_inches="tight")
  plt.show()

def plot_last_vertices(df: pd.DataFrame, N, samples, Rs):
  PAD = 0.5
  _, axes = plt.subplots(1, len(Rs))
  for r_idx, r in enumerate(Rs):
    show_legend = (r_idx+1) == len(Rs)
    ax = sns.histplot(
      df[df.r == r],
      ax=axes[r_idx],
      hue="Last vertex",
      x="Population size",
      multiple="stack",
      discrete=True,
      legend=show_legend,
    )
    if show_legend:
      sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    axes[r_idx].set_title(f"{r=}")
    for n in range(2, N+1):
      axes[r_idx].plot([n-PAD, n+1-PAD], [samples/n, samples/n], '-', lw=1, c='red')
  plt.show()

def plot_prob_initial_is_last(df: pd.DataFrame, N, graph_generator: GraphGenerator, samples):
  df = df[df["Population size"] == N]
  df = df.groupby(['Population size', 'r']).apply(lambda g: sum(g['Last vertex'] == INITIAL_NODE_PLACEMENT) / len(g)).reset_index(name='p')
  plot = sns.lineplot(
    df,
    x="r",
    y="p",
    marker='o',
    linestyle='--',
  )
  ax = plt.gca()
  ax.set(xlabel='Relative fitness, $r$', ylabel='Probability last is initial, $p$')
  plt.savefig(f'figs/p-vs-r-{graph_generator.name}-N-{N}-samples-{samples}.png', dpi=300, bbox_inches="tight")
  plt.show()

def star_central_graph(N: int) -> nx.DiGraph:
  G = nx.DiGraph()
  for i in range(1, N):
    G.add_edge(0, i)
    G.add_edge(i, 0)
  return G

def star_outskirt_graph(N: int) -> nx.DiGraph:
  G = nx.DiGraph()
  for i in range(N-1):
    G.add_edge(N-1, i)
    G.add_edge(i, N-1)
  return G

def complete_graph(N: int) -> nx.DiGraph:
  G = nx.DiGraph()
  for i in range(N):
    for j in range(i+1, N):
      G.add_edge(i, j)
      G.add_edge(j, i)
  return G

def undirected_cycle(N: int) -> nx.DiGraph:
  G = nx.DiGraph()
  for idx in range(N):
    G.add_edge(idx, (idx+1) % N)
    G.add_edge((idx+1) % N, idx)
  return G

def directed_cycle(N: int) -> nx.DiGraph:
  G = nx.DiGraph()
  for idx in range(N):
    G.add_edge(idx, (idx+1) % N)
  return G

def line_graph(N: int) -> nx.DiGraph:
  G = nx.DiGraph()
  for idx in range(N-1):
    G.add_edge(idx, idx+1)
    G.add_edge(idx+1, idx)
  return G

def grid_graph(N: int, periodic=False) -> nx.DiGraph:
  n, m = 0, 0
  for mm in range(int(np.sqrt(N)), 0, -1):
    n, r = divmod(N, mm)
    if r == 0:
      m = mm
      break
  print(m, n)
  def loc(a, b):
    if not periodic and not (0 <= a < m and 0 <= b < n): return None
    if a < 0: a = m-1
    if b < 0: b = n-1
    if a >= m: a = 0
    if b >= n: b = 0
    return n * a + b

  G = nx.DiGraph()
  for i in range(m):
    for j in range(n):
      node = n*i + j
      up = loc(i-1, j)
      down = loc(i+1, j) 
      left = loc(i, j-1) 
      right = loc(i, j+1) 

      if up is not None:
        G.add_edge(node, up)
        G.add_edge(up, node)
      if down is not None:
        G.add_edge(node, down)
        G.add_edge(down, node)
      if left is not None:
        G.add_edge(node, left)
        G.add_edge(left, node)
      if right is not None:
        G.add_edge(node, right)
        G.add_edge(right, node)

  return G


def setup():
  sns.set_theme(font_scale=2, rc={'text.usetex' : True})
  sns.set_style("whitegrid", {
    'axes.grid' : False,
    'axes.spines.left': False,
    'axes.spines.right': False,
    'axes.spines.top': False,
    'axes.spines.bottom': False,
  })

def main2():
  setup()
  Ns = list(range(2, 30+1))
  SAMPLES = 10000
  Rs = (1,)
  df = last_vertex(
    Ns,
    GraphGenerator(name="directed-cycle", generate=directed_cycle),
    samples=SAMPLES,
    Rs=Rs,
  )
  # plot_last_vertices(df, max(Ns), SAMPLES, Rs)

def main():
  setup()
  N = 10
  delta = .05
  epsilon = .01
  SAMPLES = 1000 # int(np.ceil(4/epsilon**2 * np.log(2*N/delta))) # P[exists st. |X/s-EX/s|>eps] <= 1-delta # 10000
  print(f"{SAMPLES=}")
  gen=GraphGenerator(name="directed-cycle", generate=directed_cycle)
  Rs = np.linspace(1, 40, 500)
  df = last_vertex(
    [N],
    gen,
    samples=SAMPLES,
    Rs=Rs,
  )
  plot_prob_initial_is_last(df, N, gen, SAMPLES)

def main3():
  setup()
  N = 3
  R = 1
  SAMPLES = 100000
  # gen = GraphGenerator(name="line", generate=line_graph, layout=nx.circular_layout)
  # gen = GraphGenerator(name="grid-periodic", generate=lambda n: grid_graph(n, periodic=True), layout=lambda G: {node: [node//int(np.sqrt(N)), node % int(np.sqrt(N))] for node in G.nodes()})
  # gen = GraphGenerator(name="grid", generate=grid_graph, layout=lambda G: {node: [node//int(np.sqrt(N)), node % int(np.sqrt(N))] for node in G.nodes()})
  # gen = GraphGenerator(name="star-outskirt", generate=star_outskirt_graph, layout=nx.kamada_kawai_layout)
  # gen = GraphGenerator(name="star-central", generate=star_central_graph, layout=nx.kamada_kawai_layout)
  gen = GraphGenerator(name="directed-cycle", generate=directed_cycle, layout=nx.circular_layout)
  # gen = GraphGenerator(name='undirected-cycle', generate=undirected_cycle, layout=nx.circular_layout)
  # gen = GraphGenerator(name='complete', generate=complete_graph, layout=nx.circular_layout)
  df = last_vertex(
    [N],
    gen,
    samples=SAMPLES,
    Rs=(R,),
  )
  plot_last_vertices_graph(df, N, gen, SAMPLES, R)



@lru_cache(maxsize=None)
def W(a, b, r, N):
  k = (b-a+1) % N
  return N + (r-1) * k

import numpy as np

@lru_cache(maxsize=None)
def LV(i, r, N):
  X = np.zeros(shape=(N**2+1, N**2+1))
  y = np.zeros(N**2+1)
  idx = lambda a, b: a*N + b
  rev_idx = lambda index: (index // N, index % N)
  IMPOSSIBLE = N**2
  for a in range(N):
    for b in range(N):
      w = W(a, b, r, N)
      row = idx(a, b)
      if (b+1) % N == a:
        X[row, idx(a, b)] = 1
        y[row] = int(b == i)
        # return int(b == 1)
      elif a == b:
        X[row, idx(a, b)] = 1
        X[row, idx(a, (b+1)%N)] = -(r/w) * w/(r+1)
        X[row, IMPOSSIBLE] = -(1/w) * w/(r+1)
        y[row] = 0
        # return (r/w) / (1-1/w) * LV(a, (b+1)%N, i, r, N, level+1) * w/(r+1)
      else:
        X[row, idx(a, b)] = 1
        X[row, idx(a, (b+1)%N)] = -(r/w) * w/(r+1)
        X[row, idx((a+1)%N, b)] = -(1/w) * w/(r+1)
        y[row] = 0
        # return (r/w*LV(a, (b+1)%N, i, r, N, level+1) + 1/w*LV((a+1)%N, b, i, r, N, level+1)) * w/(r+1)
  X[N**2, IMPOSSIBLE] = 1
  y[N**2] = 0
  print(X, y)
  vsolution = np.linalg.solve(X, y)

  solution = {}
  for index, p in enumerate(vsolution):
    solution[rev_idx(index)] = p

  return solution


if __name__ == '__main__':
  main()

"""
  N = 3
  R = 1.1
  l = [LV(i, R, N) for i in range(N)]
  print(l)

  for a in range(N):
    for i in range(N):
      print(i, a, l[i][(a, a)])
    print('---')
  # main()
"""