import densmap as dm

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

FP = dm.fitting_parameters( par_file='SlipLenght/parameters_slip.txt' )

folder_name = FP.folder_name
file_root = 'flow_'

Lx = FP.lenght_x
Lz = FP.lenght_z

vel_x, vel_z = dm.read_velocity_file(folder_name+'/'+file_root+'00001.dat')
Nx = vel_x.shape[0]
Nz = vel_x.shape[1]
hx = Lx/Nx
hz = Lz/Nz
x = hx*np.arange(0.0,Nx,1.0, dtype=float)
z = hz*np.arange(0.0,Nz,1.0, dtype=float)
X, Z = np.meshgrid(x, z, sparse=False, indexing='ij')

profile_velocity_x = np.zeros( len(z), dtype=float )
profile_velocity_z = np.zeros( len(z), dtype=float )
profile_kinetic_energy = np.zeros( len(z), dtype=float )

spin_up_steps = 49
n_init = FP.first_stamp + spin_up_steps
n_fin = FP.last_stamp
dt = FP.time_step

n_dump = 10
print("Producing averaged profile ")
for idx in range( n_init, n_fin ):
    if idx%n_dump==0 :
        print("Obtainig frame "+str(idx))
        t_label = str(dt*idx)+' ps'
    # Time-averaging window
    rho = dm.read_density_file(folder_name+'/'+file_root+'{:05d}'.format(idx)+'.dat', bin='y')
    U, V = dm.read_velocity_file(folder_name+'/'+file_root+'{:05d}'.format(idx)+'.dat')
    K = 0.5*np.multiply( rho, np.multiply(U, U)+np.multiply(V, V) )
    profile_velocity_x = np.add( np.mean(U, axis=0), profile_velocity_x )
    profile_velocity_z = np.add( np.mean(V, axis=0), profile_velocity_z )
    profile_kinetic_energy = np.add( np.mean(K, axis=0), profile_kinetic_energy )

profile_velocity_x /= n_fin-n_init
profile_velocity_z /= n_fin-n_init
profile_kinetic_energy /= n_fin-n_init

# Linear regression on velocity profile
"""
offset_z = 0.85     # [nm]
idx_low = (np.abs(z - offset_z)).argmin()
idx_high = (np.abs( np.flip(z)-offset_z )).argmin()
z_linear = z[idx_low: idx_high]
profile_linear = profile_velocity_x[idx_low: idx_high]
p = np.polyfit(z_linear, profile_linear, deg=1)
U = 0.01            # [nm/ps]
L = 29.0919983      # [nm]
slip_lenght = L/2 - U/p[0] 
print("Estimate slip lenght: lambda = "+str(slip_lenght)+" nm")
lin_fit = np.polyval(p, z)
"""

# Parabolic fitting (for Poiseuille flows)
offset_z = 0.85     # [nm]
idx_low = (np.abs(z - offset_z)).argmin()
idx_high = (np.abs( np.flip(z)-offset_z )).argmin()
z_parab = z[idx_low: idx_high]
profile_parab = profile_velocity_x[idx_low: idx_high]
p = np.polyfit(z_parab, profile_parab, deg=2)
par_fit = np.polyval(p, z)

plt.plot(z, profile_velocity_x, 'b-', linewidth=1.5)
plt.plot(z, par_fit, 'r--', linewidth=1.25, label='fit')
plt.legend(fontsize=20.0)
plt.xlabel('z [nm]', fontsize=30.0)
plt.ylabel('u [nm/ps]', fontsize=30.0)
plt.xticks(fontsize=30.0)
plt.yticks(fontsize=30.0)
plt.xlim([z[0], z[-1]])
plt.title('Poiseuille flow', fontsize=30.0)
plt.show()

"""
plt.plot(z, profile_velocity_x, 'b-', linewidth=1.25, label='U')
plt.plot(z, profile_velocity_z, 'r-', linewidth=1.25, label='V')
plt.plot(z, profile_kinetic_energy, 'k-', linewidth=1.25, label='K')
plt.legend(fontsize=20.0)
plt.xlabel('z [nm]', fontsize=30.0)
plt.ylabel('values [nondim.]', fontsize=30.0)
plt.xticks(fontsize=30.0)
plt.yticks(fontsize=30.0)
plt.title('Channel flow under double shear', fontsize=30.0)
plt.show()
"""

"""
plt.plot(z, profile_velocity_x, 'k.', markersize=5.0, label='U')
plt.plot(z, lin_fit, 'r-', linewidth=1.25, label='lin. fit')
plt.legend(fontsize=20.0)
plt.xlabel('z [nm]', fontsize=30.0)
plt.ylabel('U [nm/ps]', fontsize=30.0)
plt.xticks(fontsize=30.0)
plt.yticks(fontsize=30.0)
plt.title('Channel flow under double shear', fontsize=30.0)
plt.show()
"""
