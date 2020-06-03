import densmap as dm

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

def array_from_file( filename ):
    my_list = []
    with open(filename, 'r') as f:
        for line in f:
            my_list.append(float(line.split()[0]))
    return np.array(my_list)

time = array_from_file('substrate_loc/1nm/time.txt')

# radius_r_1 = array_from_file('substrate_loc/1nm/radius_r.txt')
# radius_c_1 = array_from_file('substrate_loc/1nm/radius_c.txt')
angle_c_1 = array_from_file('substrate_loc/1nm/angle_c.txt')
angle_r_1 = array_from_file('substrate_loc/1nm/angle_r.txt')
# difference_1 = array_from_file('substrate_loc/1nm/difference.txt')

# radius_r_2 = array_from_file('substrate_loc/2nm/radius_r.txt')
# radius_c_2 = array_from_file('substrate_loc/2nm/radius_c.txt')
angle_c_2 = array_from_file('substrate_loc/2nm/angle_c.txt')
angle_r_2 = array_from_file('substrate_loc/2nm/angle_r.txt')
# difference_2 = array_from_file('substrate_loc/2nm/difference.txt')

# radius_r_3 = array_from_file('substrate_loc/3nm/radius_r.txt')
# radius_c_3 = array_from_file('substrate_loc/3nm/radius_c.txt')
angle_c_3 = array_from_file('substrate_loc/3nm/angle_c.txt')
angle_r_3 = array_from_file('substrate_loc/3nm/angle_r.txt')
# difference_3 = array_from_file('substrate_loc/3nm/difference.txt')

fs = 25.0
fs_legend = 20.0

plt.plot(time, angle_c_1, 'b-', linewidth=2.0, label='1nm')
plt.plot(time, angle_r_1, 'b--', linewidth=2.5)
plt.plot(time, angle_c_2, 'r-', linewidth=2.0, label='2nm')
plt.plot(time ,angle_r_2, 'r--', linewidth=2.5)
plt.plot(time, angle_c_3, 'g-', linewidth=2.0, label='3nm')
plt.plot(time ,angle_r_3, 'g--', linewidth=2.5)
plt.plot([0.0,0.0],[0.0,0.0], 'k--', linewidth=2.5, label='cap')
plt.ylabel(r'$\theta(t)$ [nm]', fontsize=fs)
plt.xlabel('t [ps]', fontsize=fs)
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
plt.xlim([time[0],time[-1]])
plt.legend(fontsize=fs_legend, loc='best')
plt.title('Sensitivity of measurement location', fontsize=fs)
plt.show()
