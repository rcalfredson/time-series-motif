import numpy as np

signal_freq = 360
s_min = 150
s_max = 450
maxdist = 250
pred_len = 1000
verbosity = 1

ecg_data = np.loadtxt(open("data/ecg_example.csv", "rb"), delimiter=",")
x = ecg_data[:-pred_len]
