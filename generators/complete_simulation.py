import networkx as nx
import numpy as np
import pandas as pd

from simulation import GraphGenerator, last_vertex

from graphs import complete_graph

SAMPLES = 100000
N = 10

def main():
  print(f"{SAMPLES=}")
  data = []
  gen = GraphGenerator(name='complete', generate=complete_graph, layout=nx.circular_layout)
  Rs = np.linspace(1, 3, 10)
  for k in range(1, N):
    df = last_vertex(
      [N],
      gen,
      initial_node_placements=set(range(k)),
      samples=SAMPLES,
      Rs=Rs,
      thread=True,
    )
    print(df)
    ddf = (
      df
      .drop(columns=['Population size', 'Steps'])
      .groupby(by=['r'])
      .value_counts(normalize=True)
      .reset_index(name='p')
      .sort_values(by=['r', 'Last vertex'])
    )
    ddf['num_initial_mutants'] = k
    for r in Rs:
      ddfr = ddf[ddf['r'] == r]
      print(f'{k=}, {r=}')
      left = ddfr[ddfr['Last vertex'] < k]['p'].sum()
      # right = ddfr[ddfr['Last vertex'] >= k]['p'].sum()
      data.append({
        'graph_family': 'complete',
        'N': 10,
        'r': r,
        'num_initial_mutants': k,
        'p_last_resident_on_initial_mutant_location': left,
      })

  processed_df = pd.DataFrame(data)
  processed_df.to_csv('./data/complete-simulation.csv')

if __name__ == '__main__':
  main()