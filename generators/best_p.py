import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utils import setup
from cycle_numerical import solve_cycle

def main():
  setup()
  SLACK=1e-6
  data = []
  for N in range(80, 80+1):
    print(N)
    best_r = 0
    best_p = 0
    for R in np.linspace(N-5, N, 200):
      fix, ext = solve_cycle(N, r=R, slack=SLACK)
      print(N, R, fix[0])
      if fix[0] > best_p:
        best_r = R
        best_p = fix[0]
    data.append((N, best_r, best_p))
    print((N, best_r, best_p))

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
  plt.plot()
  # fig.savefig(f'figs/best-corresponding-p-directed-slack-{1e-6}.png', dpi=300, bbox_inches="tight")

if __name__ == '__main__':
  main()