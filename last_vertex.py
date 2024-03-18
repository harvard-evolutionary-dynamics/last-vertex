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
ROUND = lambda steps, precision: (steps // precision) * precision

@dataclass
class GraphGenerator:
  name: str
  generate: Callable[[int], nx.DiGraph]
  layout: Optional[Callable[[nx.DiGraph], dict]] = None

def work(G, N, samples, r):
  print(f"{r=}")
  results = []
  for vertex, steps in sample(lambda: trial_cond_fix_last_vertex(G, {INITIAL_NODE_PLACEMENT}, r, count_steps=True), samples):
    results.append((N, r, vertex, ROUND(steps, 500)))
  return results

def last_vertex(
  Ns: List[int],
  graph_generator: GraphGenerator,
  Rs: Iterable[float],
  samples: int = 1000,
  overwrite: bool = True,
  use_existing_file: bool = False,
  thread: bool = True,
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

    if thread:
      with Pool(NUM_WORKERS) as p:
        for results in p.map(partial(work, G, N, samples), Rs):
          data.extend(results)
    else:
      for results in map(partial(work, G, N, samples), Rs):
        data.extend(results)

    df = pd.DataFrame(data, columns=["Population size", "r", "Last vertex", "Steps"])
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
  cbar.set_ticklabels([f'min={min(values):.3f}', f'max={max(values):.3f}'])

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

def plot_prob_last(df: pd.DataFrame, N, graph_generator: GraphGenerator, samples):
  df = df[df["Population size"] == N]
  assert len(df["r"].unique()) == 1
  df = df.groupby(['Population size', 'r']).value_counts(normalize=True).reset_index(name='p')
  df["Last vertex"] += 1
  missing_vertices = set(range(1, N+1)) - set(df["Last vertex"].values)
  for vertex in missing_vertices:
    df.loc[len(df.index)] = [N, df["r"][0], vertex, 0]  
  # print(df)
  ax = sns.barplot(
    df,
    x="Last vertex",
    y="p",
    width=1,
    palette='Greens_d',
  )
  # ax.set_xticks(range(1, N+1))
  # ax.set_xticklabels(range(1, N+1))
  ax.set(xlabel='Location, $i$', ylabel='Probability last is $i$, $p$')
  fig = ax.get_figure()
  fig.savefig(f'figs/p-vs-i-R-1-{graph_generator.name}-N-{N}-samples-{samples}.png', dpi=300, bbox_inches="tight")
  # plt.show()

def plot_time_vs_prob_last(df: pd.DataFrame, N, graph_generator: GraphGenerator, samples):
  df = df[df["Population size"] == N]
  assert len(df["r"].unique()) == 1
  df = df.groupby(['Population size', 'r', 'Steps']).value_counts(normalize=True).reset_index(name='p')
  df["Last vertex"] += 1
  missing_vertices = set(range(1, N+1)) - set(df["Last vertex"].values)
  for vertex in missing_vertices:
    df.loc[len(df.index)] = [N, df["r"][0], vertex, 0]  
  # print(df)
  results = df.pivot(
    columns="Last vertex",
    values="p",
    index="Steps",
  )
  results.sort_index(level=0, ascending=False, inplace=True)
  ax = sns.heatmap(
    results,
    # width=1,
    # palette='Greens_d',
  )
  # ax.set_xticks(range(1, N+1))
  # ax.set_xticklabels(range(1, N+1))
  # ax.set(xlabel='Location, $i$', ylabel='Probability last is $i$, $p$')
  fig = ax.get_figure()
  # fig.savefig(f'figs/p-vs-i-R-1-{graph_generator.name}-N-{N}-samples-{samples}.png', dpi=300, bbox_inches="tight")
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
   #  'axes.spines.left': False,
   #  'axes.spines.right': False,
   #  'axes.spines.top': False,
   #  'axes.spines.bottom': False,
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

def main1():
  setup()
  with Pool(NUM_WORKERS) as p:
    for _ in p.map(do, range(10, 11)): ...

def do(N):
  print(f'starting {N}')
  # N = 10
  delta = .05
  epsilon = .01
  SAMPLES = int(np.ceil(4/epsilon**2 * np.log(2*N/delta))) # P[exists st. |X/s-EX/s|>eps] <= 1-delta # 10000
  print(f"{SAMPLES=}")
  gen=GraphGenerator(name="directed-cycle", generate=directed_cycle)
  Rs = (1,) # np.linspace(1, 40, 200)
  df = last_vertex(
    [N],
    gen,
    samples=SAMPLES,
    Rs=Rs,
    thread=False,
  )
  plot_prob_last(df, N, gen, SAMPLES)
  print(f'done with {N}')
# plot_prob_initial_is_last(df, N, gen, SAMPLES)

def mainhh():
  setup()
  N = 25
  R = 10
  SAMPLES = 9999 # 100000
  # gen = GraphGenerator(name="line", generate=line_graph, layout=nx.circular_layout)
  gen = GraphGenerator(name="grid-periodic", generate=lambda n: grid_graph(n, periodic=True), layout=lambda G: {node: [node//int(np.sqrt(N)), node % int(np.sqrt(N))] for node in G.nodes()})
  # gen = GraphGenerator(name="grid", generate=grid_graph, layout=lambda G: {node: [node//int(np.sqrt(N)), node % int(np.sqrt(N))] for node in G.nodes()})
  # gen = GraphGenerator(name="star-outskirt", generate=star_outskirt_graph, layout=nx.kamada_kawai_layout)
  # gen = GraphGenerator(name="star-central", generate=star_central_graph, layout=nx.kamada_kawai_layout)
  # gen = GraphGenerator(name="directed-cycle", generate=directed_cycle, layout=nx.circular_layout)
  # gen = GraphGenerator(name='undirected-cycle', generate=undirected_cycle, layout=nx.circular_layout)
  # gen = GraphGenerator(name='complete', generate=complete_graph, layout=nx.circular_layout)
  df = last_vertex(
    [N],
    gen,
    samples=SAMPLES,
    Rs=(R,),
  )
  plot_last_vertices_graph(df, N, gen, SAMPLES, R)


SLACK = 1e-6
def sd(N, rblock):
  print('start', rblock)
  bestp, bestpr = 0, 0
  for r in rblock:
    fix = solve_directed(N, r=r, slack=SLACK)
    if fix[0] > bestp:
      bestp = fix[0]
      bestpr = r
  print('done', rblock, bestp, bestpr)
  return bestp, bestpr

def mainhi():
  setup()
  SLACK=1e-6
  data = []
  for N in range(2, 20+1):
    print(N)
    best_r = 0
    best_p = 0
    for R in np.linspace(1, 2*N, (100)*2*N):
      fix, ext = solve_directed(N, r=R, slack=SLACK)
      if fix[0] > best_p:
        best_r = R
        best_p = fix[0]
    data.append((N, best_r, best_p))

  df = pd.DataFrame(columns=["N", "best_r", "best_p"], data=data)
  print(df)
  ax = sns.lineplot(
    df,
    x="N",
    y="best_p",
    # width=1,
    # palette='Greens_d',
    marker='o',
    linestyle='--',
  )
  ax.set(xlabel='Population size, $N$', ylabel='Corresponding probability, $p$')
  fig = ax.get_figure()
  fig.savefig(f'figs/best-corresponding-p-directed-slack-{1e-6}.png', dpi=300, bbox_inches="tight")


def main12():
  setup()
  ax = sns.lineplot(
    pd.DataFrame(
      columns=["N", "best r", "p"],
      data=[
        (3,1.0,0.1904761903431444),
        (4,1.0,0.26666666672875483),
        (5,2.4979999999999998,0.27512895726956244),
        (6,4.137444444444445,0.28353233603037115),
        (7,5.446,0.2929224173005908),
        (8,6.6033333333333335,0.301245511494557),
        (9,7.688555555555555,0.3081990363230064),
        (10,8.73722222222222,0.31394055417283956),
    ]),
    marker='o',
    linestyle='--',
    x="N",
    y="best r",
  )
  ax.set(xlabel='Population size, $N$', ylabel='Best fitness, $r$')
  # ax.set_xticks(range(len(df)))
  # ax.set_xticklabels(['1'] + ([''] * (len(df)-2)) + [f'{N}'])
  fig = ax.get_figure()
  fig.savefig(f'figs/best-r-directed-slack-{1e-6}.png', dpi=300, bbox_inches="tight")

def main111():
  setup()
  BLOCKS = 10
  INTERVALS = 10
  ITERATIONS = 3
  n = 9
  for N in range(n, n+1):
    print(N)
    rlo, rhi = 1, 2*N
    for _ in range(ITERATIONS):
      print(rlo, rhi)
      blocksize = (rhi - rlo) / BLOCKS
      rblocks = [
        np.linspace(rlo + blocksize*i, rlo + blocksize*(i+1), INTERVALS)
        for i in range(BLOCKS)
      ]
      with Pool(NUM_WORKERS) as p:
        bests, rs = zip(*p.map(partial(sd, N), rblocks))

      i = np.argmax(bests, keepdims=1)[0]
      rlo, rhi = rlo + blocksize*i, rlo + blocksize*(i+1)
      bestr = rs[i]
      bestp = bests[i]
      print('best r and p:', bestr, bestp)


def main111():
  N = 3
  fix = solve_directed(N, r=1)

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

def mainlol():
  setup()
  N = 40
  # delta = .05
  # epsilon = .01
  # SAMPLES = int(np.ceil(4/epsilon**2 * np.log(2*N/delta))) # P[exists st. |X/s-EX/s|>eps] <= 1-delta # 10000
  SAMPLES = 100000
  print(f"{SAMPLES=}")
  gen=GraphGenerator(name="directed-cycle", generate=directed_cycle)
  Rs = (2,) # np.linspace(1, 40, 200)
  df = last_vertex(
    [N],
    gen,
    samples=SAMPLES,
    Rs=Rs,
    thread=False,
  )
  plot_time_vs_prob_last(df, N, gen, SAMPLES)

def mainnope():
  setup()
  SLACK = 1e-6
  R = 1
  for N in range(2, 40+1):
    print(N)
    fix, ext = solve_directed(N, r=R, slack=SLACK)
    print(fix)
    data=[(idx+1, p) for idx, p in fix.items()]
    df = pd.DataFrame(columns=["Last vertex", "p"], data=data)
    ax = sns.barplot(
      df,
      x="Last vertex",
      y="p",
      width=1,
      palette='Greens_d',
    )
    # ax.set(xlabel='Location, $i$', ylabel='Probability last is $i$, $p$')
    ax.set(xlabel='', ylabel='')
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(['1'] + ([''] * (len(df)-2)) + [f'{N}'])
    ys = np.linspace(0, 1, 11, endpoint=True) 
    ax.set_yticks(ys)
    ax.set_yticklabels([''] * len(ys))
    fig = ax.get_figure()
    fig.savefig(f'figs-present/undirected-fixation-1/p-vs-i-R-{R}-fixation-N-{N}-slack-{SLACK}.png', dpi=300, bbox_inches="tight")
    plt.clf()
 
import sympy 

from itertools import product
import scipy.sparse.linalg

def solve_directed(N: int, r: float = 1, slack=1e-6):
  # (i, l) -> (ip, lp)
  A = {}
  idx = lambda a, b: a * (N+1) + b
  ridx = lambda id: (id // (N+1), id % (N+1))

  A = np.zeros((N*(N+1), N*(N+1)))
  for i, ip in product(range(N), repeat=2):
    for l, lp in product(range(N+1), repeat=2):
      row, col = idx(i, l), idx(ip, lp)
      A[row, col] = 0
      # directed
      # if l in (0, N): A[row, col] = (int(lp == l and ip == i))
      # elif ip == i:   A[row, col] = (1/ (r+1)) * int(lp == l - 1)
      # elif ip == (i+1) % N: A[row, col] = r/(r+1) * int(lp == l + 1)

      # undirected
      if l in (0, N): A[row, col] = (int(lp == l and ip == i))
      elif ip == i:   A[row, col] = (r/(2*(r+1))) * int(lp == l + 1) + (1/(2*(r+1))) * int(lp == l - 1)
      elif ip == (i-1) % N: A[row, col] = (1/(2*(r+1))) * int(lp == l - 1)
      elif ip == (i+1) % N: A[row, col] = r/(2*(r+1)) * int(lp == l + 1)

  # S, U = scipy.linalg.eig(A.T)
  #stationary = np.array(U[:, np.where(np.abs(S - 1.) < 1e-8)[0][0]].flat)
  #stationary = stationary / np.sum(stationary)

  # print(S)
  # print(U)
  x = np.zeros((N*(N+1),))
  x[idx(0, 1)] = 1

  # print(A)
  # P, D = sympy.Matrix(A).diagonalize()
  # print(D)

  # jv = sympy.symbols('t')
  # print(D)
  # input()
  # print(D**jv)
  # input()
  # aa = x.dot(P).dot(D**jv).dot(P.inv())
  # print(len(aa))
  # sympy.print_latex(aa[0].simplify())
  # for j in range(1, 100):
  #   print(aa[0].subs({jv: j}).evalf())
  # input()

  k = 0
  s = +np.inf
  while s >= slack:
    k = k*2 if k > 0 else 1
    b = x@np.linalg.matrix_power(A, k)
    ans = {}
    for idx, p in enumerate(b):
      ans[ridx(idx)] = p
    totalf = 0
    totale = 0
    fix = {}
    ext = {}
    s = 0
    for (i, l), p in ans.items():
      if 0 < l < N:
        s += p
        continue
      elif l == N:
        fix[i] = p
        totalf += p
      elif l == 0:
        ext[i] = p
        totale += p

    if totalf == 0 or totale == 0:
      s = +np.inf
      continue

    for i in range(N):
      fix[i] /= totalf
      ext[i] /= totale

  # print(k)
  return fix, ext


def solve_directedd(N: int, slack=1e-6):
  # (i, l) -> (ip, lp)
  A = {}
  idx = lambda a, b: a * (N+1) + b
  ridx = lambda id: (id // (N+1), id % (N+1))

  r = sympy.symbols('r')
  A = np.zeros((N*(N+1), N*(N+1)), dtype=sympy.Rational)
  for i, ip in product(range(N), repeat=2):
    for l, lp in product(range(N+1), repeat=2):
      row, col = idx(i, l), idx(ip, lp)
      A[row, col] = 0
      if l in (0, N): A[row, col] = sympy.Rational(int(lp == l and ip == i), 1)
      elif ip == i:   A[row, col] = (1/ (r+1)) * int(lp == l - 1)
      elif ip == (i+1) % N: A[row, col] = (r/ (r+1)) * int(lp == l + 1)

  # S, U = scipy.linalg.eig(A.T)
  #stationary = np.array(U[:, np.where(np.abs(S - 1.) < 1e-8)[0][0]].flat)
  #stationary = stationary / np.sum(stationary)

  # print(S)
  # print(U)
  x = np.zeros((N*(N+1),), dtype=sympy.Rational)
  x[idx(0, 1)] = 1

  print(A)
  input()
  P, D = sympy.Matrix(A).diagonalize()
  print(D)

  jv = sympy.symbols('t')
  print(D)
  input()
  print(D**jv)
  input()
  aa = x.dot(P).dot(D**jv).dot(P.inv())
  print(len(aa))
  sympy.print_latex(aa[0].simplify())
  for j in range(1, 100):
    print(aa[0].subs({jv: j}).evalf())
  input()
  k = 0
  s = +np.inf
  while s >= slack:
    k = k*2 if k > 0 else 1
    b = x@np.linalg.matrix_power(A, k)
    ans = {}
    for idx, p in enumerate(b):
      ans[ridx(idx)] = p
    totalf = 0
    totale = 0
    fix = {}
    ext = {}
    s = 0
    for (i, l), p in ans.items():
      if 0 < l < N:
        s += p
        continue
      elif l == N:
        fix[i] = p
        totalf += p
      elif l == 0:
        ext[i] = p
        totale += p

    if totalf == 0 or totale == 0:
      s = +np.inf
      continue

    for i in range(N):
      fix[i] /= totalf
      ext[i] /= totale

  return fix, ext

  


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