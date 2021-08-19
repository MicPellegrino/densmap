import densmap as dm

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

import scipy.optimize as opt

file_root = 'flow_'

# Linear flow profile fit
print("[densmap] Fitting LINEAR flow profile")

FP = dm.fitting_parameters( par_file='parameters_shear.txt' )
# FP = dm.fitting_parameters( par_file='parameters_viscosity.txt' )
folder_name = FP.folder_name

Lx = FP.lenght_x
Lz = FP.lenght_z

vel_x, vel_z = dm.read_velocity_file(folder_name+'/'+file_root+'00001.dat')
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

profile_velocity_x /= n_fin-n_init
profile_velocity_z /= n_fin-n_init
profile_kinetic_energy /= n_fin-n_init

Ly = 4.65840
volume = Lx*Ly*Lz
# 2020
# h2o_density = 1.66053904020*tot_amu/(volume)
# 2021
h2o_density = 1.66053904020*tot_amu / (Nx*Nz)
print("density = "+str(h2o_density))

# Linear regression on velocity profile
"""
offset_z = 0.75     # [nm]
idx_low = (np.abs(z - offset_z)).argmin()
idx_high = (np.abs( np.flip(z)-offset_z )).argmin()
z_linear = z[idx_low: idx_high]
profile_linear = profile_velocity_x[idx_low: idx_high]
p_lin = np.polyfit(z_linear, profile_linear, deg=1)
print(p_lin)
U_lin = 0.01            # [nm/ps]
"""

# L = 29.0919983      # [nm]
# slip_lenght = L/2 - U/p[0]
# print("Estimate slip lenght: lambda = "+str(slip_lenght)+" nm")
# lin_fit = np.polyval(p_lin, z)

# Parabolib flow profile fit
"""
print("[densmap] Fitting PARABOLIC flow profile")

FP = dm.fitting_parameters( par_file='SlipLenght/parameters_slip_parab.txt' )
folder_name = FP.folder_name

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

# Parabolic fitting (for Poiseuille flows)
offset_z = 0.85     # [nm]
idx_low = (np.abs(z - offset_z)).argmin()
idx_high = (np.abs( np.flip(z)-offset_z )).argmin()
z_parab = z[idx_low: idx_high]
profile_parab = profile_velocity_x[idx_low: idx_high]
p_par = np.polyfit(z_parab, profile_parab, deg=2)
print(p_par)
# par_fit = np.polyval(p, z)
"""

"""
print("[densmap] Estimate of the slip lenght")
# slip_lenght = U_lin/p_lin[0]-0.5*channel_width
# slip_lenght = np.sqrt( (U_lin/p_lin[0])**2 + p_par[2]/p_par[0] - 0.25*Lz**2 )
slip_lenght = 0.5 * ( -(3.0/4.0)*(U_lin/p_lin[0]) + np.sqrt( (9.0/16.0)*(U_lin/p_lin[0])**2 - 0.5*((U_lin/p_lin[0])**2-0.25*Lz**2+p_par[2]/p_par[0]) ) )
print("slip_lenght = "+str(slip_lenght)+" nm")

print("[densmap] Estimate of channel width")
# channel_width = -p_par[1]/p_par[0]
channel_width = np.sqrt( Lz**2 + 4.0*slip_lenght**2 -4.0*p_par[2]/p_par[0] ) - 2.0*slip_lenght
print("channel_width = "+str(channel_width)+" nm")
"""

"""
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

# Cosine fit
k = 2*np.pi/Lz
def u_forced(z, V) :
    return V*np.cos(k*z)

popt, _ = opt.curve_fit(u_forced, z, profile_velocity_x, p0=0.1)

V_hat = popt[0]
cos_acceleration = 39.44e-3     # nm/ps**2
viscosity_estimate = (1e-3)*(h2o_density*cos_acceleration)/(V_hat*k**2)

print("viscosity_estimate = "+str(viscosity_estimate)+" mPa*s (centipoise)")

u_fit = np.vectorize(lambda z : u_forced(z, V_hat))

plt.plot(z, profile_velocity_x, 'k.', markersize=7.5, label='U')
plt.plot(z, u_fit(z), 'b--')
# plt.plot(z, lin_fit, 'r-', linewidth=1.25, label='lin. fit')
plt.legend(fontsize=20.0)
plt.xlabel('z [nm]', fontsize=30.0)
plt.ylabel('U [nm/ps]', fontsize=30.0)
plt.xticks(fontsize=30.0)
plt.yticks(fontsize=30.0)
plt.title('Channel flow under cosine acceleration', fontsize=30.0)
plt.xlim([0.0, Lz])
# plt.title('Channel flow under double shear', fontsize=30.0)
plt.show()
