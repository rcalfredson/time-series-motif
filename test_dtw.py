import numpy as np
from lib.dtw import dtw

print('result:', dtw(np.arange(1, 11), np.arange(30, 76), 36)[0])