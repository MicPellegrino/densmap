import densmap as dm
import numpy as np
import matplotlib.pyplot as plt

import scipy.optimize as opt

file_root = 'flow_'

FP = dm.fitting_parameters( par_file='parameters_test.txt' )

# folder_poiseuille = FP.folder_name+'LJPoiseuille/Flow_epsilon03_f6'
# folder_couette    = FP.folder_name+'LJCouette/Flow_epsilon03'
folder_poiseuille = FP.folder_name+'ConfinedPoiseuille_Q1_match/Flow'
folder_couette    = FP.folder_name+'ConfinedCouette_Q1/Flow'
Lx = FP.lenght_x
Lz = FP.lenght_z

vel_x, vel_z = dm.read_velocity_file(folder_poiseuille+'/'+file_root+'00001.dat')
Nx = vel_x.shape[0]
Nz = vel_x.shape[1]
hx = Lx/Nx
hz = Lz/Nz
x = hx*np.arange(0.0,Nx,1.0, dtype=float)+0.5*hx
z = (hz*np.arange(0.0,Nz,1.0, dtype=float)+0.5*hz)
z_s = 1.2
z_f = 1.8
n_exclude = np.argmin(np.abs(z-z_f))
n_data = len(z)-n_exclude
print("# exclude = "+str(n_exclude))
z = (hz*np.arange(0.0,Nz,1.0, dtype=float)+0.5*hz)-0.5*Lz
X, Z = np.meshgrid(x, z, sparse=False, indexing='ij')

n_init = FP.first_stamp
n_fin = FP.last_stamp
dt = FP.time_step

velocity_profile_poiseuille = np.zeros( len(z), dtype=float )
velocity_profile_couette = np.zeros( len(z), dtype=float )

# print("Producing averaged profile ")
for idx in range( n_init, n_fin ):
    U, V = dm.read_velocity_file(folder_poiseuille+'/'+file_root+'{:05d}'.format(idx)+'.dat')
    velocity_profile_poiseuille = np.add( np.mean(U, axis=0), velocity_profile_poiseuille )
    U, V = dm.read_velocity_file(folder_couette+'/'+file_root+'{:05d}'.format(idx)+'.dat')
    velocity_profile_couette = np.add( np.mean(U, axis=0), velocity_profile_couette )

velocity_profile_poiseuille /= n_fin-n_init
velocity_profile_couette /= n_fin-n_init

# Double-checking the parabolic profile
p_dc = np.polyfit(z[n_exclude:-n_exclude], velocity_profile_poiseuille[n_exclude:-n_exclude], 2)
dc_parabola = np.polyval(p_dc, z)
print("Raw parameter parabola")
print(p_dc)

fun_poiseuille = lambda zz, p2, fp : fp * ( zz**2 - p2 )
fun_couette = lambda zz, p1 : p1 * zz
p_poiseuille, p_poiseuille_cov  = opt.curve_fit(fun_poiseuille, z[n_exclude:-n_exclude], \
        velocity_profile_poiseuille[n_exclude:-n_exclude], maxfev=10000, p0=[p_dc[2]/p_dc[0], p_dc[0]])
p_couette, p_couette_cov     = opt.curve_fit(fun_couette, z[n_exclude:-n_exclude], \
        velocity_profile_couette[n_exclude:-n_exclude])
# p_poiseuille = np.polyfit(z[n_exclude:-n_exclude], velocity_profile_poiseuille[n_exclude:-n_exclude], 2)
# p_couette = np.polyfit(z[n_exclude:-n_excludefp], velocity_profile_couette[n_exclude:-n_exclude], 1)

# LJ
# U = 0.05

# U = 2.3241174*0.05
U = 0.066

# print("Coeff. value")
# print("p_1 = "+str(p_couette[0]))
# print("p_2 = "+str(p_poiseuille[0]))
# print("Coeff. uncertainty")
std_p_1 = np.sqrt(p_couette_cov[0,0])/np.sqrt(n_data)
std_p_2 = np.sqrt(p_poiseuille_cov[0,0])/np.sqrt(n_data)
# print("std(p_1) = "+str(std_p_1))
# print("std(p_2) = "+str(std_p_2))

# Estimates
p_1 = p_couette[0]
p_2 = p_poiseuille[0]
pressure_factor = p_poiseuille[1]
# print("press. fac. = "+str(p_poiseuille[0]))
delta = max(0.0, (U/p_1)**2 - p_2)
# print(p_couette[0])
# print(p_poiseuille[0])
# print('comp. = '+str( np.sign((U/p_1)**2 - p_2)*np.sqrt(np.abs((U/p_1)**2 - p_2)) ))
# print('delta = '+str( delta ))
channel_height = 2.0 * (U/p_1 - np.sqrt( delta ))
slip_lenght    =  +np.sqrt( delta )

# Perturbation
delta_p = max(0.0, (U/(p_1-std_p_1))**2 - (p_2-std_p_2))
delta_m = max(0.0, (U/(p_1+std_p_1))**2 - (p_2+std_p_2))
slip_lenght_p = np.sqrt( delta_p )
slip_lenght_m = np.sqrt( delta_m )
channel_height_p = 2.0 * (U/(p_1-std_p_1) - np.sqrt( delta_m ))
channel_height_m = 2.0 * (U/(p_1+std_p_1) - np.sqrt( delta_p ))
print("channel height:   L = "+str(channel_height)+" nm")
print(" +/- 1 sigma : ["+str(channel_height_m)+","+str(channel_height_p)+"]")
print("slip lenght: lambda = "+str(slip_lenght)+ " nm")
print(" +/- 1 sigma : ["+str(slip_lenght_m)+","+str(slip_lenght_p)+"]")

# Slip length estimate from Couette alone
lambda_star = ( U - np.max(np.abs(velocity_profile_couette)) ) / p_1
# lambda_star_ub = lambda_star+0.5*hz
# lambda_star_lb = lambda_star-0.5*hz
lambda_star_ub = 2.0*lambda_star
lambda_star_lb = 0.0

# Define new function to fit the Poiseuille results
fun_poiseuille_posterior = lambda zz, L_star :  pressure_factor * ( zz**2 - lambda_star*L_star - 0.25*L_star*L_star )
fun_poiseuille_posterior_ub = lambda zz, L_star :  pressure_factor * ( zz**2 - lambda_star_ub*L_star - 0.25*L_star*L_star )
fun_poiseuille_posterior_lb = lambda zz, L_star :  pressure_factor * ( zz**2 - lambda_star_lb*L_star - 0.25*L_star*L_star )
p_poiseuille_posterior, _  = opt.curve_fit(fun_poiseuille_posterior, z[n_exclude:-n_exclude], \
        velocity_profile_poiseuille[n_exclude:-n_exclude])
p_poiseuille_posterior_ub, _  = opt.curve_fit(fun_poiseuille_posterior_ub, z[n_exclude:-n_exclude], \
        velocity_profile_poiseuille[n_exclude:-n_exclude])
p_poiseuille_posterior_lb, _  = opt.curve_fit(fun_poiseuille_posterior_lb, z[n_exclude:-n_exclude], \
        velocity_profile_poiseuille[n_exclude:-n_exclude])

L_star = p_poiseuille_posterior[0]
L_star_ub = p_poiseuille_posterior_ub[0]
L_star_lb = p_poiseuille_posterior_lb[0]

print("DOUBLE-CHECK")
print("lambda_star = "+str(lambda_star)+ " nm")
print("L_star = "+str(L_star)+ " nm")

# Maximum shear rate estimate
gammadot_max_couette = 2.0*U/(L_star+2.0*lambda_star)
gammadot_max_poiseuille = -pressure_factor*L_star
print("max estimate shear rate Couette    = "+str(gammadot_max_couette))
print("max estimate shear rate Poiseuille = "+str(gammadot_max_poiseuille))

fig, (ax1, ax2) = plt.subplots(1,2)

vN = 10
range_v_c = np.linspace(min(velocity_profile_couette) - 0.15*(np.abs(min(velocity_profile_couette))), \
        max(velocity_profile_couette) + 0.15*(np.abs(max(velocity_profile_couette))), vN)
range_v_p = np.linspace(min(velocity_profile_poiseuille) - 0.15*(np.abs(min(velocity_profile_poiseuille))), \
        max(velocity_profile_poiseuille) + 0.15*(np.abs(max(velocity_profile_poiseuille))), vN)

# Poiseuille
ax1.plot(velocity_profile_poiseuille, z, 'k-', linewidth=2.0)
ax1.plot(velocity_profile_poiseuille[n_exclude:-n_exclude], z[n_exclude:-n_exclude], 'r.', markersize=15.0)
ax1.plot( range_v_p, 0.5*channel_height*np.ones(vN), 'g-', linewidth=2.0)
ax1.plot( range_v_p, 0.5*(channel_height+slip_lenght)*np.ones(vN), 'g--', linewidth=1.75)
ax1.plot( range_v_p, -0.5*channel_height*np.ones(vN), 'g-', linewidth=2.0)
ax1.plot( range_v_p, -0.5*(channel_height+slip_lenght)*np.ones(vN), 'g--', linewidth=1.75)
# ax1.fill_between( range_v_p, 0.5*L_star_ub*np.ones(vN), 0.5*L_star_lb*np.ones(vN), color='g', alpha=0.5)
# ax1.fill_between( range_v_p, -0.5*L_star_ub*np.ones(vN), -0.5*L_star_lb*np.ones(vN), color='g', alpha=0.5)
ax1.plot(fun_poiseuille(z, *p_poiseuille), z, 'b--', linewidth=3.0)
#
ax1.plot(dc_parabola, z, 'b-')
#
ax1.fill_betweenx(z, fun_poiseuille_posterior_lb(z, *p_poiseuille_posterior_lb), \
        fun_poiseuille_posterior_ub(z, *p_poiseuille_posterior_ub), color='m', alpha=0.5)
ax1.set_title(r'Poiseuille flow profile', fontsize=25.0)
ax1.set_ylabel(r'$z$ [nm]', fontsize=25.0)
ax1.set_xlabel(r'$u$ [nm/ps]', fontsize=25.0)
ax1.tick_params(axis='both', which='major', labelsize=20)
ax1.set_ylim([z[0], z[-1]])
ax1.set_xlim([range_v_p[0], range_v_p[-1]])

# Couette
ax2.plot(velocity_profile_couette, z, 'k-', linewidth=2.0)
ax2.plot(velocity_profile_couette[n_exclude:-n_exclude], z[n_exclude:-n_exclude], 'r.', markersize=15.0, label='MD data')
ax2.plot( range_v_c, 0.5*channel_height*np.ones(vN), 'g-', linewidth=2.0, label='wall')
ax2.plot( range_v_c, 0.5*(channel_height+slip_lenght)*np.ones(vN), 'g--', linewidth=1.75, label='wall +/- slip len.')
ax2.plot( range_v_c, -0.5*channel_height*np.ones(vN), 'g-', linewidth=2.0)
ax2.plot( range_v_c, -0.5*(channel_height+slip_lenght)*np.ones(vN), 'g--', linewidth=1.75)
# ax2.fill_between( range_v_c, 0.5*L_star_ub*np.ones(vN), 0.5*L_star_lb*np.ones(vN), color='g', alpha=0.5, label='wall')
# ax2.fill_between( range_v_c, -0.5*L_star_ub*np.ones(vN), -0.5*L_star_lb*np.ones(vN), color='g', alpha=0.5)
ax2.plot(fun_couette(z, *p_couette), z, 'b--', linewidth=3.0, label='best fit')
ax2.set_title(r'Couette flow profile', fontsize=25.0)
# ax2.set_ylabel(r'$z$ [nm]', fontsize=25.0)
ax2.set_xlabel(r'$u$ [nm/ps]', fontsize=25.0)
ax2.tick_params(axis='both', which='major', labelsize=20)
ax2.set_ylim([z[0], z[-1]])
ax2.set_xlim([range_v_c[0], range_v_c[-1]])
ax2.legend(fontsize=20.0)

plt.show()
