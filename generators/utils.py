import networkx as nx
import random
import seaborn as sns

from typing import Optional, Set


def setup():
  sns.set_theme(font_scale=2, rc={'text.usetex' : True})
  sns.set_style("whitegrid", {
    'axes.grid' : False,
   #  'axes.spines.left': False,
   #  'axes.spines.right': False,
   #  'axes.spines.top': False,
   #  'axes.spines.bottom': False,
  })


def sample(fn, times):
  count = 0
  while count < times:
    if (ans := fn()) is not None:
      yield ans
      count += 1

def trial_cond_fix_last_vertex(G: nx.DiGraph, S: Optional[Set], r: float, count_steps: bool = False):
  if S is None:
    S = {random.choice(list(G.nodes()))}

  N = len(G)
  V = G.nodes()
  mutants = set()
  mutants |= S
  steps = 0

  dier = None
  while V - mutants:
    if not mutants: return None if not count_steps else (None, steps)
    k = len(mutants)
    if random.random() < r*k/(N + (r-1)*k):
      birther = random.choice(list(mutants))
    else:
      birther = random.choice(list(V - mutants))

    dier = random.choice([w for (_, w) in G.out_edges(birther)])
    assert birther != dier
    if birther in mutants:
      mutants.add(dier)
    elif dier in mutants:
      mutants.remove(dier)
    
    steps += 1
  return dier if not count_steps else (dier, steps)