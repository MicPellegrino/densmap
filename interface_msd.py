import densmap as dm

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

"""
    Computing the mean-square displacement from the reference average interface position 
"""

FP = dm.fitting_parameters( par_file='ShearChar/parameters_shear.txt' )

folder_name = FP.folder_name
file_root = 'flow_'

Lx = FP.lenght_x
Lz = FP.lenght_z

vel_x, vel_z = dm.read_velocity_file(folder_name+'/'+file_root+'00001.dat')
Nx = vel_x.shape[0]
Nz = vel_x.shape[1]
hx = Lx/Nx
hz = Lz/Nz
z = hz*np.arange(0.0,Nz,1.0, dtype=float)

n_init = FP.first_stamp
n_fin = FP.last_stamp
# n_fin = 1100
dt = FP.time_step

# To tune
delta_th = 2.0
z0 = 0.85
n_transient = int(max(0, n_fin-800))
n_dump = 10

# Init data structures
print("Initialization")
file_name = file_root+'{:05d}'.format(n_transient)+'.dat'
density_array = dm.read_density_file(folder_name+'/'+file_name, bin='y')
bulk_density = dm.detect_bulk_density(density_array, delta_th)
left_intf_mean, right_intf_mean = dm.detect_interface_int(density_array, 0.5*bulk_density, hx, hz, z0)

print("Computing mean interface")
n = 0
for i in range(n_transient+1, n_fin+1 ) :
    n += 1
    if i % n_dump == 0 :
        print("Obtainig frame "+str(i))
    file_name = file_root+'{:05d}'.format(i)+'.dat'
    density_array = dm.read_density_file(folder_name+'/'+file_name, bin='y')
    bulk_density = dm.detect_bulk_density(density_array, delta_th)
    left_intf, right_intf = dm.detect_interface_int(density_array, 0.5*bulk_density, hx, hz, z0)
    left_intf_mean += left_intf
    right_intf_mean += right_intf

left_intf_mean /= (n+1)
right_intf_mean /= (n+1)

print("Computing mean square displacement")
n = 0
left_intf_msd = np.zeros( left_intf_mean.shape, dtype=float )
left_intf_msd[1,:] = left_intf_mean[1,:]
right_intf_msd = np.zeros( right_intf_mean.shape, dtype=float )
right_intf_msd[1,:] = left_intf_mean[1,:]
for i in range(n_transient+1, n_fin+1 ) :
    n += 1
    if i % n_dump == 0 :
        print("Obtainig frame "+str(i))
    file_name = file_root+'{:05d}'.format(i)+'.dat'
    density_array = dm.read_density_file(folder_name+'/'+file_name, bin='y')
    bulk_density = dm.detect_bulk_density(density_array, delta_th)
    left_intf, right_intf = dm.detect_interface_int(density_array, 0.5*bulk_density, hx, hz, z0)
    diff_left = left_intf[0,:]-left_intf_mean[0,:]
    diff_right = right_intf[0,:]-right_intf_mean[0,:]
    left_intf_msd[0,:] += diff_left*diff_left
    right_intf_msd[0,:] += diff_right*diff_right

left_intf_msd[0,:] /= n
right_intf_msd[0,:] /= n

plt.plot(left_intf_mean[0,:], left_intf_mean[1,:], 'k-', linewidth=2.0, label='mean interface position')
plt.plot(right_intf_mean[0,:], left_intf_mean[1,:], 'k-', linewidth=2.0)
plt.plot(left_intf_mean[0,:]+np.sqrt(left_intf_msd[0,:]), left_intf_mean[1,:], 'k--', linewidth=1.5, label='sqrt(mean-square-displacement)')
plt.plot(right_intf_mean[0,:]+np.sqrt(right_intf_msd[0,:]), left_intf_mean[1,:], 'k--', linewidth=1.5)
plt.plot(left_intf_mean[0,:]-np.sqrt(left_intf_msd[0,:]), left_intf_mean[1,:], 'k--', linewidth=1.5)
plt.plot(right_intf_mean[0,:]-np.sqrt(right_intf_msd[0,:]), left_intf_mean[1,:], 'k--', linewidth=1.5)
plt.title('Interface; time window='+str((n_fin-n_transient)*dt*1e-3)+'ns', fontsize=20.0)
plt.legend(fontsize=20.0)
plt.xticks(fontsize=20.0)
plt.yticks(fontsize=20.0)
plt.show()

