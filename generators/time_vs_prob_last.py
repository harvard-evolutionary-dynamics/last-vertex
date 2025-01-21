import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from simulation import GraphGenerator

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


if __name__ == '__main__':
  ...