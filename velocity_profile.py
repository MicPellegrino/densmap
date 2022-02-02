import densmap as dm

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

FP = dm.fitting_parameters( par_file='parameters_shear.txt' )
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
X, Z = np.meshgrid(x, z, sparse=False, indexing='ij')

idx_bin_upp = np.argmin(np.abs(z-dz))
idx_bin_low = np.argmin(np.abs(z-Lz+dz))
idx_half_plane = np.argmin(np.abs(z-0.5*Lz))

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
# U = 0.0
# U = 0.00167
# U = 0.00334
U = 0.006590649942987458
# U = 0.00835

rho = dm.read_density_file(folder_name+'/'+file_root+'00001.dat')
rho_avg = np.zeros( (Nx,Nz) )
x_com_0 = np.sum(np.multiply(rho,X))/np.sum(rho)

xcom_profile = np.zeros( Nz, dtype=float )
velocity_profile = np.zeros( Nz, dtype=float )
velocity_profile2 = np.zeros( Nz, dtype=float )
### Vertical velocity profile ###
vervel_profile = np.zeros( Nz, dtype=float )
density_profile_z = np.zeros( Nz, dtype=float )
density_profile_z2 = np.zeros( Nz, dtype=float )
###
vx_bin_upp = np.zeros( Nx, dtype=float )
vx_bin_low = np.zeros( Nx, dtype=float )
vx_bin_half = np.zeros( Nx, dtype=float )
vz_bin_upp = np.zeros( Nx, dtype=float )
vz_bin_low = np.zeros( Nx, dtype=float )
density_profile_x = np.zeros( Nx, dtype=float )
for idx in range(n_init, n_fin+1) :
    if idx%n_dump==0 :
        print("Obtainig frame "+str(idx))
    v_x, v_z = dm.read_velocity_file(folder_name+'/'+file_root+'{:05d}'.format(idx)+'.dat')
    if idx%n_dump==0 :
        print("Sample velocity = "+str(np.mean(np.mean(v_z)))+" nm/ps")
    # Compute the center of mass of the system
    rho = dm.read_density_file(folder_name+'/'+file_root+'{:05d}'.format(idx)+'.dat')
    rho_avg += rho
    intertial_matrix = np.multiply(rho,X)
    x_com = np.sum(intertial_matrix)/np.sum(rho)
    bins_com = int((x_com-x_com_0)/hx)
    # Horizontal velocity profile 
    # velocity_profile += np.mean(v_x[i_min+bins_com:i_max+1+bins_com,], axis=0)
    # Center on the center of mass locally along y
    for j in range(Nz) :
        rho_sum = np.sum(rho[:,j])
        if rho_sum > 0 : 
            x_com_loc = np.sum(intertial_matrix[:,j])/rho_sum
        else :
            x_com_loc = 0.5*Lx
        xcom_profile[j] += x_com_loc
        bins_com_loc = int(x_com_loc/hx)
        d = int(0.5*(i_max-i_min))
        loc_mean_vel = np.mean(v_x[bins_com_loc-d:d+1+bins_com_loc,j], axis=0)
        velocity_profile[j] += loc_mean_vel
        velocity_profile2[j] += loc_mean_vel*loc_mean_vel
    # Vertical velocity profile 
    vervel_profile += np.mean(v_z[i_min+bins_com:i_max+1+bins_com,], axis=0) 
    # Density profile
    loc_mean_den = np.mean(rho[i_min+bins_com:i_max+1+bins_com,], axis=0)
    density_profile_z += loc_mean_den
    density_profile_z2 += loc_mean_den*loc_mean_den
    ###
    vx_bin_upp += np.roll( v_x[:,idx_bin_upp], bins_com )
    vx_bin_low += np.roll( v_x[:,idx_bin_low], bins_com )
    vz_bin_upp += np.roll( v_z[:,idx_bin_upp], bins_com )
    vz_bin_low += np.roll( v_z[:,idx_bin_low], bins_com )
    vx_bin_half += v_x[:,idx_half_plane]
    density_profile_x += rho[:,idx_half_plane]
velocity_profile /= (n_fin-n_init+1)
vx_bin_upp /= (n_fin-n_init+1)
vx_bin_low /= (n_fin-n_init+1)
vz_bin_upp /= (n_fin-n_init+1)
vz_bin_low /= (n_fin-n_init+1)
vx_bin_half /= (n_fin-n_init+1)
rho_avg /= (n_fin-n_init+1)
xcom_profile /= (n_fin-n_init+1)
density_profile_x /= (n_fin-n_init+1)
### Vertical velocity profile ###
vervel_profile /= (n_fin-n_init+1)
density_profile_z /= (n_fin-n_init+1)
rho0 = np.mean(density_profile_z[int(0.25*Nz):int(0.75*Nz)])
###

# Polyfit velocity profile (escluding a few bins close to the wall)
p = np.polyfit(z[5:-6], velocity_profile[5:-6], deg=1)
lin_fit = z*p[0] + p[1]
z_p = (U-p[1])/p[0]
z_m = -(U+p[1])/p[0]
print("z_+ = "+str(z_p))
print("z_- = "+str(z_m))

# Linear regression of near wall velocity values
import statsmodels.api as sm

v_bot = velocity_profile[3:15]/U + 1
v_top = np.flip(np.abs(velocity_profile[-15:-3]))/U - 1
z_lm = (z[3:15]-z[3])/Lz

mod_bot = sm.OLS(v_bot,z_lm)
mod_top = sm.OLS(v_top,z_lm)
fii_bot = mod_bot.fit()
fii_top = mod_top.fit()
p_values_top = fii_top.summary2().tables[1]['P>|t|']
p_values_bot = fii_bot.summary2().tables[1]['P>|t|']
print(fii_bot.summary())
print(p_values_bot)
print(fii_top.summary())
print(p_values_top)

"""

plt.plot(z, vervel_profile, 'r-', linewidth=2.5)
plt.ylabel("v [nm/ps]", fontsize=25)
plt.xlabel("z [nm]", fontsize=25)
plt.title(FP.folder_name, fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlim([0,Lz])
plt.show()

print("Near wall vertical velocity = "+str(max(np.abs(vervel_profile))))
print("Wall velocity = "+str(U))

"""
"""

plt.plot(z, density_profile_z/density_profile_z[int(0.5*Nz)])
plt.plot(z, velocity_profile/U, 'k-', linewidth=2.5)
# plt.plot(z, lin_fit, 'b--', linewidth=1.75)
plt.plot(z, np.ones(Nz), 'g--', linewidth=2.5)
plt.plot(x, np.zeros(Nx), 'g-.', linewidth=1.5)
plt.plot(z, -np.ones(Nz), 'g--', linewidth=2.5)
plt.ylabel("v [nm/ps]", fontsize=25)
plt.xlabel("z [nm]", fontsize=25)
plt.title(FP.folder_name, fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
# plt.ylim([-1.5*U, 1.5*U])
plt.xlim([0,Lz])
plt.show()

"""

z_lf_bot = z[3:15]
z_lf_top = z[-15:-3]
v_lf_bot = -velocity_profile[3:15]/U
v_lf_top = velocity_profile[-15:-3]/U
pbot = np.polyfit(z_lf_bot, v_lf_bot, 1)
ptop = np.polyfit(z_lf_top, v_lf_top, 1)

fig, ax = plt.subplots(1, 2)
bulk_density = density_profile_z[int(0.5*Nz)]
ax[0].plot(density_profile_z[0:15]/bulk_density, z[0:15], 'k-', linewidth=2.75, markersize=15, label=r'$\rho/\rho_l$')
ax[0].set_ylabel(r'$y$ [nm]', fontsize=30.0)
ax[0].tick_params(axis='both', labelsize=25)
ax[0].plot(-velocity_profile[0:15]/U, z[0:15], 'rx', linewidth=2.5, markersize=15, mew=2.5, label=r'$-v_x/U_w$')
ax[0].tick_params(axis='both', labelsize=25)
ax[0].legend(fontsize=30)
ax[0].plot(np.polyval(pbot, z_lf_bot), z_lf_bot, 'b--', linewidth=2.0)
ax[0].set_title('(a)', x=-0.075, fontsize=30)
# ax[1].set_ylabel(r'$y$ [nm]', fontsize=30.0)
ax[1].plot(density_profile_z[-15:]/bulk_density, z[-15:], 'k-', linewidth=2.75, markersize=15)
ax[1].tick_params(axis='both', labelsize=25)
ax[1].plot(velocity_profile[-15:]/U, z[-15:], 'rx', linewidth=2.5, markersize=15, mew=2.5, label=r'$v_x/U_w$')
ax[1].tick_params(axis='both', labelsize=25)
ax[1].plot(np.polyval(ptop, z_lf_top), z_lf_top, 'b--', linewidth=2.0)
ax[1].legend(fontsize=30)
ax[1].set_title('(b)', x=-0.075, fontsize=30)
plt.show()

"""

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
# plt.xlim([50,110])
# plt.ylim([- 2.0*U, 2.0*U])
plt.show()

"""
"""

plt.plot(x, vx_bin_half, 'b-', linewidth=2.0)
plt.ylabel("v_x [nm/ps]", fontsize=25)
plt.xlabel("x [nm]", fontsize=25)
plt.title("Horizontal velocity, z=0.5L_z", fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=20)
plt.xlim([0,Lx])
# plt.ylim([- 2.0*U, 2.0*U])
plt.show()

"""
"""

plt.plot(z, density_profile_z, 'b-', linewidth=2.0)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=20)
plt.xlim([0,Lz])
plt.show()

"""
"""

plt.plot(x, density_profile_x, 'b-', linewidth=2.0)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=20)
plt.xlim([0,Lx])
plt.show()

"""