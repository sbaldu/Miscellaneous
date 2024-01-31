#!/usr/bin/python3

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Read in the data
data = pd.read_csv('data.csv')

# get x and y data
grainsize = data['grainsize']
exec_times = data.iloc[:, 1:].mean(axis='columns')

# show plot
sns.lineplot(x=grainsize, y=exec_times)
plt.xlabel('Grainsize')
plt.ylabel('Execution time (μs)')
plt.grid(linestyle='--', linewidth=0.2)
plt.show()
# save plot as png
sns.lineplot(x=grainsize, y=exec_times)
plt.xlabel('Grainsize')
plt.ylabel('Execution time (μs)')
plt.grid(linestyle='--', linewidth=0.2)
plt.savefig('./images/grainsize.png', dpi=200, format='png', bbox_inches='tight')
