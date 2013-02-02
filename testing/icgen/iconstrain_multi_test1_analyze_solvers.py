import numpy as np
from matplotlib import pyplot as pl
import csv
import glob
import os

def plot_peak_progression(data_file, figure, rows_subfigs, columns_subfigs, subfig, goal = 0.1):
    with open(data_file, 'r') as f:
        reader = csv.reader(f)
        rows = []
        for row in reader:
            rows.append(row)
    N_peaks = len(rows[0])
    for i, row in enumerate(rows):
        for peak_nr in range(N_peaks):
            row[peak_nr] = float(row[peak_nr])
        rows[i] = row
    rows = np.array(rows)
    
    iterations = range(len(rows))
    
    method = os.path.basename(data_file)[:-4]
    
    axes = figure.add_subplot(rows_subfigs, columns_subfigs, subfig)

    for i in range(N_peaks):
        axes.plot(iterations, rows[:,i], label='peak %i' % i, lw = 0.25)
        axes.plot((0,len(rows)), (goal, goal), 'r')

    axes.set_yscale('log')
    axes.set_title('%s' % method)
    axes.legend()
    
    all_peaks_below_goal = np.prod(rows < goal, axis=1)
    print method, "stap met alles onder goal: ", ("%i (totaal %i)" % (np.argwhere(all_peaks_below_goal)[0], all_peaks_below_goal.sum()) if np.any(all_peaks_below_goal) else "nee")

#~ directory = '/net/schmidt/data/users/pbos/sw/code/egp/testing/icgen/iconstrain_multi_test1_solvers'
directory = '/net/schmidt/data/users/pbos/sw/code/egp/testing/icgen/iconstrain_multi_test1b_solvers'
filenames = glob.glob(directory+'/*.csv')

fig = pl.figure()
N_subfigs = len(filenames)
rows_subfigs = 3 #np.floor(np.sqrt(N_subfigs))-1
columns_subfigs = 2 #np.ceil(N_subfigs/rows_subfigs)

goal = 0.2

for subfig, filename in enumerate(filenames):
    plot_peak_progression(filename, fig, rows_subfigs, columns_subfigs, subfig, goal = goal)

fig.show()
