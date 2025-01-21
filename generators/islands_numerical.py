import numpy as np
from itertools import product

def fix_and_ext_islands(A, x, N1, N2, slack, idx, ridx):
  k = 0
  # slack = 1e-12
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
    for (n1, n2), p in ans.items():
      if 0 < n1+n2 < N1+N2:
        continue
      elif n1+n2 >= N1+N2:
        last = 0
        if n1+n2 > N1+N2:
          # assert n1+n2 == N1+N2+1, (n1, n2, N1, N2, p)
          # n2 -= 1
          last = 1
        fix[last] = p
        totalf += p
      elif n1+n2 == 0:
        ext[-1] = p
        totale += p

    s = 1-totalf-totale
    # print(totalf + totale, s)
    if totalf == 0 or totale == 0:
      s = +np.inf
      continue

    # s = 1-totalf-totale
    # print(totalf + totale, s)
    for last in range(2):
      fix[last] /= totalf

  # print(k)
  print(fix, ext)
  return fix, ext


def solve_islands(N1: int, N2: int, mu12: float, mu21: float, r: float = 1, rho1: float = 1, rho2: float = 1, slack=1e-6):
  # (i, l) -> (ip, lp)
  A = {}
  idx = lambda a, b: a * (N2+1) + b
  ridx = lambda id: (id // (N2+1), (id % (N2+1)) + (N2+1)*int(id == (N1+1)*(N2+1)))

  A = np.zeros(((N1+1)*(N2+1)+1, (N1+1)*(N2+1)+1))
  A[(N1+1)*(N2+1), (N1+1)*(N2+1)] = 1

  for n1, n1p in product(range(N1+1), repeat=2):
    for n2, n2p in product(range(N2+1), repeat=2):
      row, col = idx(n1, n2), idx(n1p, n2p)
      A[row, col] = 0
      w = ((r-1)*n1 + N1) + ((r-1)*n2 + N2)
      done = False
      if n1+n2 not in (0, N1+N2) and n1p+n2p in (0, N1+N2):
        # potentially absorbing state.
        done = True
      if n1+n2 in (0, N1+N2):
        A[row, col] = int(n1p == n1 and n2p == n2)
      elif n1p == n1 and n2p == n2+1:
        # picked element in left to reproduce to right
        # picked elemnt in right to reproduce in right
        A[row, col + done] = (n1*r/w) * ((N2-n2)*mu12/(N1*rho1+mu12*N2)) + (n2*r/w) * ((N2-n2)*rho2/(N2*rho2+mu21*N1))
      elif n1p == n1+1 and n2p == n2:
        # picked element in left to reproduce to left
        # picked elemnt in right to reproduce in left
        A[row, col] = (n1*r/w) * ((N1-n1)*rho1/(N1*rho1+mu12*N2)) + (n2*r/w) * ((N1-n1)*mu21/(N2*rho2+mu21*N1))
      elif n1p == n1 and n2p == n2-1:
        # picked element in left to reproduce to right
        # picked elemnt in right to reproduce in right
        A[row, col] = ((N1-n1)/w) * (n2*mu12/(N1*rho1+mu12*N2)) + ((N2-n2)/w) * (n2*rho2/(N2*rho2+mu21*N1))
      elif n1p == n1-1 and n2p == n2:
        # picked element in left to reproduce to left
        # picked elemnt in right to reproduce in left
        A[row, col] = ((N1-n1)/w) * (n1*rho1/(N1*rho1+mu12*N2)) + ((N2-n2)/w) * (n1*mu21/(N2*rho2+mu21*N1))
      elif n1p == n1 and n2p == n2 and (0 < n1+n2 < N1+N2):
        # picked element in left to reproduce to left
        # picked element in left to reproduce to right
        # picked elemnt in right to reproduce in right
        # picked elemnt in right to reproduce in left
        A[row, col] = (
            ((N1-n1)/w) * (((N1-n1)*rho1 + (N2-n2)*mu12)/(N1*rho1+mu12*N2))
          + ((N2-n2)/w) * (((N2-n2)*rho2 + (N1-n1)*mu21)/(N2*rho2+mu21*N1))
          + (n1*r/w) * ((n1*rho1 + n2*mu12)/(N1*rho1+mu12*N2))
          + (n2*r/w) * ((n2*rho2 + n1*mu21)/(N2*rho2+mu21*N1))
        )



  x = np.zeros((N1+1)*(N2+1)+1,)
  x[idx(1, 0)] = 1
  return fix_and_ext_islands(A, x, N1, N2, slack, idx, ridx)