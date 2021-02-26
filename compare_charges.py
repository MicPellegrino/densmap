import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

def array_from_file( filename ):
    my_list = []
    with open(filename, 'r') as f:
        for line in f:
            my_list.append(float(line.split()[0]))
    return np.array(my_list)

folder_names = ['SpreadingData/FlatQ1', \
                'SpreadingData/FlatQ2', \
                'SpreadingData/FlatQ3', \
                'SpreadingData/FlatQ4', \
                'SpreadingData/FlatQ5']

labels = ['126.8 deg', '95.0 deg', '69.1 deg', '38.8 deg', '14.7 deg']

t_max = 0.0

plt.title('Spreading branches', fontsize=30.0)

for k in range(5) :
    time = array_from_file(folder_names[k]+'/time.txt')
    t_max = max( t_max, max(time) )
    foot_l = array_from_file(folder_names[k]+'/foot_l.txt')
    foot_r = array_from_file(folder_names[k]+'/foot_r.txt')
    center = 0.5*(foot_r+foot_l)
    branch_right = foot_r - center
    plt.plot(time, branch_right, linewidth=2.5, label=labels[k])

plt.legend(fontsize=20.0)
plt.xticks(fontsize=20.0)
plt.yticks(fontsize=20.0)
plt.xlim([0.0, t_max])
plt.show()
