import densmap as dm

import numpy as np

import scipy.ndimage as smg

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

def array_from_file( filename ):
    my_list = []
    with open(filename, 'r') as f:
        for line in f:
            my_list.append(float(line.split()[0]))
    return np.array(my_list)

def compute_vel (r, dt) :
    velocity = np.zeros(r.size)
    velocity[1:-1] = (r[2:]-r[:-2])/(2.0*dt)
    velocity[0] = (r[1]-r[0])/dt
    velocity[-1] = (r[-1]-r[-2])/dt
    return velocity

time = array_from_file('Adv/Flat/time.txt')
dt = time[1]-time[0]

radius_values = []
velocity_values = []

radius_values.append(array_from_file('Rec/Flat/radius_c.txt'))

for k in range(1,6) :
    radius_values.append( array_from_file('Rec/Wave'+str(k)+'/radius_c.txt') )

cap_speed_m1 = (1.0/72.0)*1e3

sigma_r = 10.0
for k in range(6) :
    radius_filtered = smg.gaussian_filter1d(radius_values[k], sigma=0.25*sigma_r)
    velocity_values.append( cap_speed_m1*compute_vel(radius_filtered, dt) )

a = np.array([0.00, 0.75, 1.00, 1.25, 1.50, 1.75])
a_max = max(a)

plt.plot(time, velocity_values[0], label='a=0.00', linewidth=2.00, c=cm.hot(a[0]/(a_max+1)))
plt.plot(time, velocity_values[1], label='a=0.75', linewidth=2.00, c=cm.hot(a[1]/(a_max+1)))
plt.plot(time, velocity_values[2], label='a=1.00', linewidth=2.00, c=cm.hot(a[2]/(a_max+1)))
plt.plot(time, velocity_values[3], label='a=1.25', linewidth=2.00, c=cm.hot(a[3]/(a_max+1)))
plt.plot(time, velocity_values[4], label='a=1.50', linewidth=2.00, c=cm.hot(a[4]/(a_max+1)))
plt.plot(time, velocity_values[5], label='a=1.75', linewidth=2.00, c=cm.hot(a[5]/(a_max+1)))
plt.legend(fontsize=20.0)
plt.xticks(fontsize=20.0)
plt.yticks(fontsize=20.0)
plt.xlabel('t [ps]', fontsize=20.0)
plt.ylabel(r'$Ca(t)$ [nondim.]', fontsize=20.0)
plt.xlim([0,max(time)])
plt.title('Comparison velocities', fontsize=20.0)
plt.show()
