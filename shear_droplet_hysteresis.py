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

time = array_from_file('/home/michele/densmap/ShearHyst/Q1Q2/time.txt')

q1_tl = array_from_file('/home/michele/densmap/ShearHyst/Q1Q2/angle_tl.txt')
q1_tr = array_from_file('/home/michele/densmap/ShearHyst/Q1Q2/angle_tr.txt')
q1_bl = array_from_file('/home/michele/densmap/ShearHyst/Q1Q2/angle_bl.txt')
q1_br = array_from_file('/home/michele/densmap/ShearHyst/Q1Q2/angle_br.txt')
q1_angle = 0.25 * ( q1_tl + q1_tr + q1_bl + q1_br )

q3_tl = array_from_file('/home/michele/densmap/ShearHyst/Q3Q2/angle_tl.txt')
q3_tr = array_from_file('/home/michele/densmap/ShearHyst/Q3Q2/angle_tr.txt')
q3_bl = array_from_file('/home/michele/densmap/ShearHyst/Q3Q2/angle_bl.txt')
q3_br = array_from_file('/home/michele/densmap/ShearHyst/Q3Q2/angle_br.txt')
q3_angle = 0.25 * ( q3_tl + q3_tr + q3_bl + q3_br )

target = 99.61
tr_std = 1.68

plt.plot(time, q1_angle, 'r-', linewidth=1.5, label='from ~123deg')
plt.plot(time, q3_angle, 'b-', linewidth=1.5, label='from ~80deg')
plt.plot(time, target*np.ones(len(time)), 'k-', linewidth=1.5, label='99.61+/-1.68deg')
plt.plot(time, (target+tr_std)*np.ones(len(time)), 'k--')
plt.plot(time, (target-tr_std)*np.ones(len(time)), 'k--')
plt.xlim([time[0], time[-1]])
plt.legend()
plt.show()
