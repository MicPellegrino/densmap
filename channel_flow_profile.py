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
FP = dm.fitting_parameters( par_file='parameters_viscosity.txt' )
folder_name = FP.folder_name

Lx = FP.lenght_x
Lz = FP.lenght_z

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
n_init = FP.first_stamp + spin_up_steps
n_fin = FP.last_stamp
dt = FP.time_step

n_dump = 10
tot_amu = 0
print("Producing averaged profile ")
for idx in range( n_init, n_fin ):
    if idx%n_dump==0 :
        # print("Obtainig frame "+str(idx))
        t_label = str(dt*idx)+' ps'
        # print("tot amu = "+str(tot_amu))
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

Ly = Lx
volume = Lx*Ly*Lz
# 2020
# h2o_density = 1.66053904020*tot_amu/(volume)
# 2021
h2o_density = 1.66053904020*tot_amu / (Nx*Nz)
print("density = "+str(h2o_density))

# Cosine fit
k = 2*np.pi/Lz
def u_forced(z, V) :
    return V*np.cos(k*z)
# Separate z and profile_velocity_x in training and test set

n_kmean = 25
cos_acceleration = 1e-4    # nm/ps**2

viscosity_estimate = []

for kmean in range(n_kmean) :

    print("k = ", kmean)

    perm = rng.permutation(len(z))
    idx_train = perm[0:int(0.8*len(z))]
    idx_test = perm[int(0.8*len(z)):]

    popt, _ = opt.curve_fit(u_forced, z[idx_train], profile_velocity_x[idx_train], p0=0.1)

    V_hat = popt[0]
    print("V=",V_hat)
    print("k=",k)
    viscosity_estimate.append( (1e-3)*(h2o_density*cos_acceleration)/(V_hat*k**2) )

viscosity_estimate = np.array(viscosity_estimate)
mean_viscosity_estimate = np.mean(viscosity_estimate)
delta_viscosity_estimate = np.max(viscosity_estimate)-np.min(viscosity_estimate)

print("viscosity_estimate = "+str(mean_viscosity_estimate)+" +/- "+str(delta_viscosity_estimate)+" mPa*s (centipoise)")

u_fit = np.vectorize(lambda z : u_forced(z, V_hat))

a_input = cos_acceleration*np.cos(k*z)

"""
fig, ax1 = plt.subplots(1,1)

ax1.plot(1e3*a_input, z, 'r-', linewidth=8)
ax1.set_ylabel('z [nm]', fontsize=75)
ax1.set_xlabel(r'$a_x$ $\times10^3$ [nm/ps$^2$]', fontsize=75)
ax1.tick_params(axis='both', labelsize=65)
ax1.set_ylim([0.0, Lz])
ax1.set_box_aspect(1)

plt.show()
"""

fig, ax2 = plt.subplots(1,1)

ax2.plot(1e3*profile_velocity_x, z, 'o', markersize=27.5, label='MD measurements', markeredgecolor='k', markeredgewidth=6, markerfacecolor="None")
ax2.plot(1e3*u_fit(z), z, 'b--', linewidth=8, label='best fit')
ax2.legend(fontsize=50, loc="upper left")
ax2.set_ylabel('z [nm]', fontsize=50)
ax2.set_xlabel(r'$u_x$ $\times10^3$ [nm/ps]', fontsize=50)
ax2.tick_params(axis='both', labelsize=50)
ax2.set_ylim([0.0, Lz])
ax2.set_box_aspect(1)

plt.show()
