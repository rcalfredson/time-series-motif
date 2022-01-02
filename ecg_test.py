import matplotlib.pyplot as plt
import numpy as np

from lib.motif import forecast_motifs

signal_freq = 360
s_min = 150
s_max = 450
maxdist = 250
pred_len = 1000

ecg_data = np.loadtxt(open("data/ecg_example.csv", "rb"), delimiter=",")
ecg_data = np.expand_dims(ecg_data, 0)
x = ecg_data[:, :-pred_len]
x_p, idx, starts, ends, p_idx, p_starts, p_ends, models = forecast_motifs(
    x, pred_len, s_min, s_max, maxdist
)
plt.figure(1, (700, 300))
times = np.arange(x.shape[0]) / signal_freq
plt.scatter(times, x, linewidths=3, c='black')
plt.show()