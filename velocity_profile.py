import densmap as dm

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

# FP = dm.fitting_parameters( par_file='parameters_shear.txt' )
FP = dm.fitting_parameters( par_file='parameters_test.txt' )
folder_name = FP.folder_name
file_root = 'flow_'
Lx = FP.lenght_x
Lz = FP.lenght_z
dz = FP.dz

print("Creating meshgrid")
vel_x, vel_z = dm.read_velocity_file(folder_name+'/'+file_root+'00001.dat')
Nx = vel_x.shape[0]
Nz = vel_x.shape[1]
hx = Lx/Nx
hz = Lz/Nz
x = hx*np.arange(0.0,Nx,1.0, dtype=float)+0.5*hx
z = hz*np.arange(0.0,Nz,1.0, dtype=float)+0.5*hz

idx_bin_upp = np.argmin(np.abs(z-dz))
idx_bin_low = np.argmin(np.abs(z-Lz+dz))

n_init = FP.first_stamp
n_fin = FP.last_stamp
dt = FP.time_step

# Tunable parameters
n_dump = 10
a = 0.1
x_max = 0.5*Lx + a*0.5*Lz
x_min = 0.5*Lx - a*0.5*Lz
i_max = np.argmin(np.abs(x-x_max))
i_min = np.argmin(np.abs(x-x_min))
# U = 0.00167
U = 0.00334
# U = 0.00668
# U = 0.00835

velocity_profile = np.zeros( Nz, dtype=float )
### Vertical velocity profile ###
vervel_profile = np.zeros( Nz, dtype=float )
density_profile = np.zeros( Nz, dtype=float )
###
vx_bin_upp = np.zeros( Nx, dtype=float )
vx_bin_low = np.zeros( Nx, dtype=float )
vz_bin_upp = np.zeros( Nx, dtype=float )
vz_bin_low = np.zeros( Nx, dtype=float )
for idx in range(n_init, n_fin+1) :
    if idx%n_dump==0 :
        print("Obtainig frame "+str(idx))
    v_x, v_z = dm.read_velocity_file(folder_name+'/'+file_root+'{:05d}'.format(idx)+'.dat')
    rho = dm.read_density_file(folder_name+'/'+file_root+'{:05d}'.format(idx)+'.dat')
    velocity_profile += np.mean(v_x[i_min:i_max+1,], axis=0)
    ### Vertical velocity profile ###
    vervel_profile += np.mean(v_z[i_min:i_max+1,], axis=0) 
    density_profile += np.mean(rho[i_min:i_max+1,], axis=0)
    ###
    vx_bin_upp += v_x[:,idx_bin_upp]
    vx_bin_low += v_x[:,idx_bin_low]
    vz_bin_upp += v_z[:,idx_bin_upp]
    vz_bin_low += v_z[:,idx_bin_low]
velocity_profile /= (n_fin-n_init+1)
vx_bin_upp /= (n_fin-n_init+1)
vx_bin_low /= (n_fin-n_init+1)
vz_bin_upp /= (n_fin-n_init+1)
vz_bin_low /= (n_fin-n_init+1)
### Vertical velocity profile ###
vervel_profile /= (n_fin-n_init+1)
density_profile /= (n_fin-n_init+1)
rho0 = np.mean(density_profile[int(0.25*Nz):int(0.75*Nz)])
###

# Polyfit velocity profile (escluding a few bins close to the wall)
p = np.polyfit(z[5:-6], velocity_profile[5:-6], deg=1)
lin_fit = z*p[0] + p[1]
z_p = (U-p[1])/p[0]
z_m = -(U+p[1])/p[0]
print("z_+ = "+str(z_p))
print("z_- = "+str(z_m))

plt.plot(z/Lz, vervel_profile/U, 'r-', linewidth=2.5)
plt.plot(z/Lz, density_profile/rho0, 'k-', linewidth=2.5)
plt.ylabel("v, rho [-1]", fontsize=25)
plt.xlabel("z [-1]", fontsize=25)
plt.title(FP.folder_name, fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlim([0,1])
plt.show()

print("Near wall vertical velocity = "+str(max(np.abs(vervel_profile))))
print("Wall velocity = "+str(U))

plt.plot(z, velocity_profile, 'k-', linewidth=2.5)
# plt.plot(z, lin_fit, 'b--', linewidth=1.75)
plt.plot(z, np.ones(Nz)*U, 'g--', linewidth=2.5)
plt.plot(x, np.zeros(Nx), 'g-.', linewidth=1.5)
plt.plot(z, -np.ones(Nz)*U, 'g--', linewidth=2.5)
plt.ylabel("v [nm/ps]", fontsize=25)
plt.xlabel("z [nm]", fontsize=25)
plt.title(FP.folder_name, fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylim([-1.5*U, 1.5*U])
plt.xlim([0,Lz])
plt.show()

plt.plot(x, vx_bin_upp, 'b-', linewidth=2.0, label=r'$v_x$')
plt.plot(x, vx_bin_low, 'b-', linewidth=2.0)
plt.plot(x, vz_bin_upp, 'r-', linewidth=2.0, label=r'$v_z$')
plt.plot(x, vz_bin_low, 'r-', linewidth=2.0)
plt.plot(x, np.ones(Nx)*U, 'g--', linewidth=2.5)
plt.plot(x, np.zeros(Nx), 'g-.', linewidth=1.5)
plt.plot(x, -np.ones(Nx)*U, 'g--', linewidth=2.5)
plt.ylabel("v [nm/ps]", fontsize=25)
plt.xlabel("x [nm]", fontsize=25)
plt.title(FP.folder_name, fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=20)
plt.xlim([50,110])
plt.ylim([- 2.0*U, 2.0*U])
plt.show()
