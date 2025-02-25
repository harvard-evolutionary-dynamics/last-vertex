import numpy as np
import scipy as sp 
import scipy.linalg
from collections import defaultdict

STEPS = 1_000_000

np.set_printoptions(suppress=True)

def solve(a: int, b: int, r: float):
  transitions = defaultdict(dict)

  for i in range(b+1):
    w = (a-1+i)*r + (b-i+1)
    if i+1 <= b:
      transitions[i][i+1] = ((a-1+i)*r)/w * (b-i)/b
    if i-1 >= 0:
      transitions[i][i-1] = (b-i+1)/w * i/b

  for i, t in transitions.items():
    t[i] = 1-sum(t.values())

  A = np.array([[
    transitions[i].get(j, 0)
    for j in range(b+1)
  ] for i in range(b+1)])

  x = np.zeros(shape=(b+1,))
  x[0] = 1
  b = x@np.linalg.matrix_power(A, STEPS)
  print(scipy.linalg.eig(A, left=True, right=False))
  # idx, *_ = np.where(np.isclose(eigvals, 1))
  # bs = eigvecs[idx]/np.sum(eigvecs[idx])
  # print(b, bs)
  return b

def main():
  a = 10
  b = 2
  r = 2
  """r>1/2"""
  print(f"{a=}, {b=}, {r=}")
  print(solve(a=a, b=b, r=r)[-1], f"conjecture={1-1/(2*r)}")

if __name__ == '__main__':
  main()