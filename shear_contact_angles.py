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

time = array_from_file('/home/michele/densmap/ShearDropModes/Q1/time.txt')

mean_angle = np.zeros(5)
std_angle = np.zeros(5)

t_0 = 3000
idx_0 = np.abs( time-t_0 ).argmin()

for q in range(5) :
    tl = array_from_file('/home/michele/densmap/ShearDropModes/Q'+str(q+1)+'/angle_bl.txt')
    tr = array_from_file('/home/michele/densmap/ShearDropModes/Q'+str(q+1)+'/angle_br.txt')
    bl = array_from_file('/home/michele/densmap/ShearDropModes/Q'+str(q+1)+'/angle_tl.txt')
    br = array_from_file('/home/michele/densmap/ShearDropModes/Q'+str(q+1)+'/angle_tr.txt')
    theta = 0.25 * ( tl + tr + bl + br )
    mean_angle[q] = theta[idx_0:].mean()
    std_angle[q] = theta[idx_0:].std()

print("Charge vs contact angle:")
for q in range(5) :
    print("q:"+str(q+1)+"\tmean="+str(mean_angle[q])+"\tstd="+str(std_angle[q]))


# Lennard-Jones

time = array_from_file('/home/michele/densmap/ShearChar/LJ/time.txt')
tl = array_from_file('/home/michele/densmap/ShearChar/LJ/angle_bl.txt')
tr = array_from_file('/home/michele/densmap/ShearChar/LJ/angle_br.txt')
bl = array_from_file('/home/michele/densmap/ShearChar/LJ/angle_tl.txt')
br = array_from_file('/home/michele/densmap/ShearChar/LJ/angle_tr.txt')
theta_lj = 0.25 * ( tl + tr + bl + br )
print("LJ\tmean="+str(theta_lj.mean())+"\tstd="+str(theta_lj.std()))
