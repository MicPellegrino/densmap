import densmap as dm

import numpy as np
import numpy.random as rng

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

import scipy.optimize as opt

file_root = 'flow_'

# Linear flow profile fit
print("[densmap] Fitting LINEAR flow profile")

# FP = dm.fitting_parameters( par_file='parameters_shear.txt' )
# FP = dm.fitting_parameters( par_file='parameters_viscosity.txt' )
# folder_name = "/home/michele/python_for_md/PureWater/DeformEm3"
folder_name = "/home/michele/python_for_md/TestDeform/DeformNVT-10nsm1/Flow"

# Lx = 4.97532
# Lz = 9.95063
Lx = 9.93710
Lz = 9.93710

vel_x, vel_z = dm.read_velocity_file(folder_name+'/'+file_root+'00001.dat')
U_avg = vel_x
Nx = vel_x.shape[0]
Nz = vel_x.shape[1]
hx = Lx/Nx
hz = Lz/Nz
x = hx*np.arange(0.0,Nx,1.0, dtype=float)+0.5*hx
z = hz*np.arange(0.0,Nz,1.0, dtype=float)+0.5*hz
X, Z = np.meshgrid(x, z, sparse=False, indexing='ij')

profile_velocity_x = np.zeros( len(z), dtype=float )
profile_velocity_z = np.zeros( len(z), dtype=float )
profile_kinetic_energy = np.zeros( len(z), dtype=float )

spin_up_steps = 0
n_init = 23
n_fin = 273
dt = 1

n_dump = 10
tot_amu = 0
print("Producing averaged profile ")
for idx in range( n_init, n_fin ):
    if idx%n_dump==0 :
        print("Obtainig frame "+str(idx))
        t_label = str(dt*idx)+' ps'
        print("tot amu = "+str(tot_amu))
    # Time-averaging window
    rho = dm.read_density_file(folder_name+'/'+file_root+'{:05d}'.format(idx)+'.dat', bin='y')
    U, V = dm.read_velocity_file(folder_name+'/'+file_root+'{:05d}'.format(idx)+'.dat')
    K = 0.5*np.multiply( rho, np.multiply(U, U)+np.multiply(V, V) )
    profile_velocity_x = np.add( np.mean(U, axis=0), profile_velocity_x )
    profile_velocity_z = np.add( np.mean(V, axis=0), profile_velocity_z )
    profile_kinetic_energy = np.add( np.mean(K, axis=0), profile_kinetic_energy )
    tot_amu = np.sum( np.sum( rho ) )
    if idx>1 :
        U_avg += U

U_avg /= (n_fin-n_init)

profile_velocity_x /= n_fin-n_init
profile_velocity_z /= n_fin-n_init
profile_kinetic_energy /= n_fin-n_init

print(np.min(profile_velocity_x))
print(np.max(profile_velocity_x))

# v_def = 1.98742e-3    # [nm/ps]
v_def = 9.9371e-3       

plt.plot(z, 1e3*profile_velocity_x, 'o', markersize=27.5, label='MD measurements', markeredgecolor='k', markeredgewidth=6, markerfacecolor="None")
plt.plot(z, 1e3*v_def*(z-0.5*Lz)/Lz, 'b--', linewidth=5, label=r'$\dot{\gamma}$=0.2nm$^{-1}$')
plt.legend(fontsize=50)
plt.xlabel('z [nm]', fontsize=50)
plt.ylabel(r'$u_x$ $\times10^3$ [nm/ps]', fontsize=50)
plt.tick_params(axis='both', labelsize=50)
plt.xlim([0.0, Lz])
plt.show()
