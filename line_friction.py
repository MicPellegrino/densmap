import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

def array_from_file( filename ):
    my_list = []
    with open(filename, 'r') as f:
        for line in f:
            my_list.append(float(line.split()[0]))
    return np.array(my_list)

def cos( theta ) :
    return np.cos(np.deg2rad(theta))

folder_name = 'SpreadingData/H2Q4'

time = array_from_file(folder_name+'/time.txt')

foot_l = array_from_file(folder_name+'/foot_l.txt')
foot_r = array_from_file(folder_name+'/foot_r.txt')

angle_l = array_from_file(folder_name+'/angle_l.txt')
angle_r = array_from_file(folder_name+'/angle_r.txt')

radius = array_from_file(folder_name+'/radius_fit.txt')

# Cutoff inertial phase
dt = time[1]-time[0]
T_cut = 1200.0      # [ps]
N = int( T_cut / dt )
time = time[N:]
foot_l = foot_l[N:]
foot_r = foot_r[N:]
angle_l = angle_l[N:]
angle_r = angle_r[N:]

# Obtain velocity
velocity_l = np.zeros(len(foot_l))
velocity_r = np.zeros(len(foot_r))
velocity_l[1:-1] = -0.5 * np.subtract(foot_l[2:],foot_l[0:-2]) / dt
velocity_r[1:-1] = 0.5 * np.subtract(foot_r[2:],foot_r[0:-2]) / dt
velocity_l[0] = -( foot_l[1]-foot_l[0] ) / dt
velocity_r[0] = ( foot_r[1]-foot_r[0] ) / dt
velocity_l[-1] = -( foot_l[-1]-foot_l[-2] ) / dt
velocity_r[-1] = ( foot_r[-1]-foot_r[-2] ) / dt
# Also end values ...

"""
velocity = np.zeros(len(radius))
velocity[1:-1] = 0.5 * np.subtract(radius[2:],radius[0:-2]) / dt
velocity[0] = ( radius[1]-radius[0] ) / dt
velocity[-1] = (radius[-1]-radius[-2]) / dt
"""

plt.plot(time, velocity_l, 'b-')
plt.plot(time, velocity_r, 'r-')
plt.show()

plt.plot(cos(angle_l), velocity_l, 'bo')
plt.plot(cos(angle_r), velocity_r, 'rx')
plt.show()
