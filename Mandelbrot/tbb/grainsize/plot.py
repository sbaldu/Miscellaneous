#!/usr/bin/python3

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sys

# Read in the data
data = pd.read_csv('data.csv')

# get x and y data
grainsize = data['grainsize']
exec_times = data.iloc[:, 1:].mean(axis='columns')

# save plot as png
if sys.argv[-1] == "save":
    sns.lineplot(x=grainsize, y=exec_times)
    plt.xlabel('Grainsize')
    plt.ylabel('Execution time (μs)')
    plt.grid(linestyle='--', linewidth=0.2)
    plt.savefig('./images/grainsize.png', dpi=200, format='png', bbox_inches='tight')
# show plot
elif sys.argv[-1] == "show":
    sns.lineplot(x=grainsize, y=exec_times)
    plt.xlabel('Grainsize')
    plt.ylabel('Execution time (μs)')
    plt.grid(linestyle='--', linewidth=0.2)
    plt.show()
else:
    print("Usage: python3 plot.py [save|show]")
