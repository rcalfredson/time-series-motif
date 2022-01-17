import argparse
import numpy as np

p = argparse.ArgumentParser(
    description="parse two CSV files as NumPy arrays and compare via isclose()"
)
p.add_argument('f1', help='path to the first file')
p.add_argument('f2', help='path to the second file')
opts = p.parse_args()

a1 = np.genfromtxt(opts.f1, delimiter=',')
a2 = np.genfromtxt(opts.f2, delimiter=',')

print('a1:', a1)
print('a2:', a2)
print('are they close?', np.isclose(a1, a2))
