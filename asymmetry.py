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

time = array_from_file('poster/advancing/time.txt')

radius_r_adv = array_from_file('poster/advancing/radius_r.txt')
radius_c_adv = array_from_file('poster/advancing/radius_c.txt')
angle_c_adv = array_from_file('poster/advancing/angle_c.txt')
angle_r_adv = array_from_file('poster/advancing/angle_r.txt')
difference_adv = array_from_file('poster/advancing/difference.txt')

radius_r_rec = array_from_file('poster/receding/radius_r.txt')
radius_c_rec = array_from_file('poster/receding/radius_c.txt')
angle_c_rec = array_from_file('poster/receding/angle_c.txt')
angle_r_rec = array_from_file('poster/receding/angle_r.txt')
difference_rec = array_from_file('poster/receding/difference.txt')

fs = 25.0
fs_legend = 20.0

fig, axs = plt.subplots(1, 2)

axs[0].plot(time, radius_c_adv, 'b-', linewidth=2.0, label='advancing')
axs[0].plot(time, radius_r_adv, 'b--', linewidth=2.5)
axs[0].plot(time, radius_c_rec, 'r-', linewidth=2.0, label='receding')
axs[0].plot(time ,radius_r_rec, 'r--', linewidth=2.5)
axs[0].plot([0.0,0.0],[0.0,0.0], 'k--', linewidth=2.5, label='cap')
# axs[0].set_xlabel('t [ps]', fontsize=fs)
axs[0].set_ylabel('$R(t)$ [nm]', fontsize=fs)
axs[0].tick_params(axis='x', labelsize=fs)
axs[0].tick_params(axis='y', labelsize=fs)
axs[0].legend(fontsize=fs_legend, loc='lower right')
axs[0].set_title('Spreading radius', fontsize=fs)
axs[0].set_xlabel('t [ps]', fontsize=fs)
axs[0].set_xlim([time[0],time[-1]])

axs[1].plot(time, angle_c_adv, 'b-', linewidth=2.0, label='advancing')
axs[1].plot(time, angle_r_adv, 'b--', linewidth=2.5)
axs[1].plot(time, angle_c_rec, 'r-', linewidth=2.0, label='receding')
axs[1].plot(time, angle_r_rec, 'r--', linewidth=2.5)
axs[1].plot([0.0,0.0],[0.0,0.0], 'k--', linewidth=2.5, label='cap')
# axs[1].set_xlabel('t [ps]', fontsize=fs)
axs[1].set_ylabel(r'$\theta(t)$ [deg]', fontsize=fs)
axs[1].tick_params(axis='x', labelsize=fs)
axs[1].tick_params(axis='y', labelsize=fs)
axs[1].legend(fontsize=fs_legend, loc='best')
axs[1].set_title('Contact angle', fontsize=fs)
axs[1].set_xlabel('t [ps]', fontsize=fs)
axs[1].set_xlim([time[0],time[-1]])

"""
axs[1,0].plot(time, radius_c_rec, 'k-', linewidth=2.5, label='contour')
axs[1,0].plot(time,radius_r_rec, 'g-', linewidth=2.5, label='cap')
axs[1,0].set_xlabel('t [ps]', fontsize=fs)
axs[1,0].set_ylabel('Receding\n$R(t)$ [nm]', fontsize=fs)
axs[1,0].tick_params(axis='x', labelsize=fs)
axs[1,0].tick_params(axis='y', labelsize=fs)
axs[1,0].legend(fontsize=fs_legend)
# axs[1,0].set_title('Spreading radius', fontsize=fs)

axs[1,1].plot(time, angle_c_rec, 'b-', linewidth=2.0, label='average')
axs[1,1].plot(time, difference_rec, 'r-', linewidth=2.0, label='difference')
axs[1,1].plot(time, angle_r_rec, 'g-', linewidth=2.5, label='cap')
axs[1,1].set_xlabel('t [ps]', fontsize=fs)
axs[1,1].set_ylabel(r'$\theta(t)$ [deg]', fontsize=fs)
axs[1,1].tick_params(axis='x', labelsize=fs)
axs[1,1].tick_params(axis='y', labelsize=fs)
axs[1,1].legend(fontsize=fs_legend)
# axs[1,1].set_title('Contact angle', fontsize=fs)
"""

plt.show()
