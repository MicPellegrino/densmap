import densmap as dm
import numpy as np
import matplotlib.pyplot as plt

import scipy.optimize as opt

file_root = 'flow_'

FP = dm.fitting_parameters( par_file='parameters_test.txt' )

folder_poiseuille = FP.folder_name+'ConfinedPoiseuille_Q1_match/Flow'
folder_couette    = FP.folder_name+'ConfinedCouette_Q1_match/Flow'
Lx = FP.lenght_x
Lz = FP.lenght_z

vel_x, vel_z = dm.read_velocity_file(folder_poiseuille+'/'+file_root+'00001.dat')
Nx = vel_x.shape[0]
Nz = vel_x.shape[1]
hx = Lx/Nx
hz = Lz/Nz
x = hx*np.arange(0.0,Nx,1.0, dtype=float)+0.5*hx
print(x)
z = (hz*np.arange(0.0,Nz,1.0, dtype=float)+0.5*hz)-0.5*Lz
X, Z = np.meshgrid(x, z, sparse=False, indexing='ij')

n_init = FP.first_stamp
n_fin = FP.last_stamp
dt = FP.time_step

velocity_profile_poiseuille = np.zeros( len(z), dtype=float )
velocity_profile_couette = np.zeros( len(z), dtype=float )

print("Producing averaged profile ")
for idx in range( n_init, n_fin ):
    U, V = dm.read_velocity_file(folder_poiseuille+'/'+file_root+'{:05d}'.format(idx)+'.dat')
    velocity_profile_poiseuille = np.add( np.mean(U, axis=0), velocity_profile_poiseuille )
    U, V = dm.read_velocity_file(folder_couette+'/'+file_root+'{:05d}'.format(idx)+'.dat')
    velocity_profile_couette = np.add( np.mean(U, axis=0), velocity_profile_couette )

velocity_profile_poiseuille /= n_fin-n_init
velocity_profile_couette /= n_fin-n_init

n_exclude = 10
fun_poiseuille = lambda zz, p2, fp : fp * ( zz**2 - p2 )
fun_couette = lambda zz, p1 : p1 * zz
p_poiseuille, _  = opt.curve_fit(fun_poiseuille, z[n_exclude:-n_exclude], \
        velocity_profile_poiseuille[n_exclude:-n_exclude])
p_couette, _     = opt.curve_fit(fun_couette, z[n_exclude:-n_exclude], \
        velocity_profile_couette[n_exclude:-n_exclude])
# p_poiseuille = np.polyfit(z[n_exclude:-n_exclude], velocity_profile_poiseuille[n_exclude:-n_exclude], 2)
# p_couette = np.polyfit(z[n_exclude:-n_excludefp], velocity_profile_couette[n_exclude:-n_exclude], 1)
U = 1.760695*0.066
print("p_1 = "+str(p_couette[0]/U))
print("p_2 = "+str(p_poiseuille[0]))

# Estimates
p_1 = p_couette[0]
p_2 = p_poiseuille[0]
delta = max(0.0, (U/p_1)**2 - p_2)
print(p_couette[0])
print(p_poiseuille[0])
print((U/p_1)**2 - p_2)
print('delta = '+str(delta))
channel_height = 2.0 * (U/p_1 - np.sqrt( delta ))
slip_lenght    =  +np.sqrt( delta )
print("channel height:   L = "+str(channel_height)+" nm")
print("slip lenght: lambda = "+str(slip_lenght)+ " nm")

print(velocity_profile_couette)

fig, (ax1, ax2) = plt.subplots(1,2)

vN = 10
# range_v_c = np.linspace(min(velocity_profile_couette), max(velocity_profile_couette), vN)
# range_v_p = np.linspace(min(velocity_profile_poiseuille), max(velocity_profile_poiseuille), vN)
range_v_p = np.linspace(-0.03, 0.15, vN)
range_v_c = np.linspace(-0.09, 0.09, vN)

# Poiseuille
ax1.plot(velocity_profile_poiseuille, z, 'k-', linewidth=2.0)
ax1.plot(velocity_profile_poiseuille[n_exclude:-n_exclude], z[n_exclude:-n_exclude], 'r.', markersize=15.0)
ax1.plot( range_v_p, 0.5*channel_height*np.ones(vN), 'g--', linewidth=2.0)
ax1.plot( range_v_p, -0.5*channel_height*np.ones(vN), 'g--', linewidth=2.0)
# ax1.plot(np.polyval(p_poiseuille, z), z, 'b--', linewidth=3.0)
ax1.plot(fun_poiseuille(z, *p_poiseuille), z, 'b--', linewidth=3.0)
ax1.set_title(r'Poiseuille flow, $q_1$', fontsize=25.0)
ax1.set_ylabel(r'$z$ [nm]', fontsize=25.0)
ax1.set_xlabel(r'$u$ [nm/ps]', fontsize=25.0)
ax1.tick_params(axis='both', which='major', labelsize=20)
ax1.set_ylim([z[0], z[-1]])
# ax1.set_xlim([range_v_p[0], range_v_p[-1]])

# Couette
ax2.plot(velocity_profile_couette, z, 'k-', linewidth=2.0)
ax2.plot(velocity_profile_couette[n_exclude:-n_exclude], z[n_exclude:-n_exclude], 'r.', markersize=15.0)
ax2.plot( range_v_c, 0.5*channel_height*np.ones(vN), 'g--', linewidth=2.0)
ax2.plot( range_v_c, -0.5*channel_height*np.ones(vN), 'g--', linewidth=2.0)
# ax2.plot(np.polyval(p_couette, z), z, 'b--', linewidth=3.0)
ax2.plot(fun_couette(z, *p_couette), z, 'b--', linewidth=3.0)
ax2.set_title(r'Couette flow, $q_1$', fontsize=25.0)
ax2.set_ylabel(r'$z$ [nm]', fontsize=25.0)
ax2.set_xlabel(r'$u$ [nm/ps]', fontsize=25.0)
ax2.tick_params(axis='both', which='major', labelsize=20)
ax2.set_ylim([z[0], z[-1]])
# ax2.set_xlim([range_v_c[0], range_v_c[-1]])

plt.show()
