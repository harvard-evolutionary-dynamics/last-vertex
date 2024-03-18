a = [0, 0, 0]
for i in range(0, 3*10+1):
  a[i % 3] += 1/(2**(2*(i+1)))

for k in range(3):
  print(a[k]/(sum(a)))