import densmap as dm
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.optimize as opt

FP = dm.fitting_parameters( par_file='parameters_shear_waterol.txt' )

z_si = 0.65
z_star = 1.3
rho_sol_ref = 0.8
rho_gol_ref = 1-rho_sol_ref

folder_name = FP.folder_name
file_root = 'flow_'

Lx = FP.lenght_x
Lz = FP.lenght_z
print("Lz=", Lz)

n_init = FP.first_stamp
n_fin = FP.last_stamp

# CREATING MESHGRID
print("Creating meshgrid")
rho = dm.read_density_file(folder_name+'/'+file_root+'{:05d}'.format(n_init)+'.dat')
rho_sol = dm.read_density_file(folder_name+'/'+file_root+'SOL_'+'{:05d}'.format(n_init)+'.dat')
rho_gol = dm.read_density_file(folder_name+'/'+file_root+'GOL_'+'{:05d}'.format(n_init)+'.dat')
Nx = rho.shape[0]
Nz = rho.shape[1]
hx = Lx/Nx
hz = Lz/Nz
x = hx*(0.5+np.arange(0.0,Nx,1.0, dtype=float))
z = hz*(0.5+np.arange(0.0,Nz,1.0, dtype=float))
X, Z = np.meshgrid(x, z, sparse=False, indexing='ij')

j_hslice = int(0.5*Nz)

xcom = np.sum(rho*X)/np.sum(rho)
icom_old = np.abs(x-xcom).argmin()

dx_left =   []
dx_right =  []

xcom_avg = 0

for n in range(n_init+1, n_fin+1) :
    if n%10 == 0 :
        print("n = ", n)
    # TODO: Should be translated with the center of mass ...
    rmap = dm.read_density_file(folder_name+'/'+file_root+'{:05d}'.format(n)+'.dat')
    rmap_sol = dm.read_density_file(folder_name+'/'+file_root+'SOL_'+'{:05d}'.format(n)+'.dat')
    rmap_gol = dm.read_density_file(folder_name+'/'+file_root+'GOL_'+'{:05d}'.format(n)+'.dat')

    # Contact line position
    bulk_density = dm.detect_bulk_density(rmap, density_th=FP.max_vapour_density)
    b_bl, b_br = dm.detect_interface_loc(rmap, hx, hz, FP.substrate_location, FP.bulk_location, wall='l')
    b_tl, b_tr = dm.detect_interface_loc(rmap, hx, hz, FP.substrate_location, FP.bulk_location, wall='t')
    
    dx_left.append(b_tl[0][-1]-b_bl[0][0])
    dx_right.append(b_tr[0][-1]-b_br[0][0])  

    xcom = np.sum(rmap*X)/np.sum(rmap)
    icom_new = np.abs(x-xcom).argmin()
    rho += np.roll(rmap, icom_new-icom_old, axis=0)
    rho_sol += np.roll(rmap_sol, icom_new-icom_old, axis=0)
    rho_gol += np.roll(rmap_gol, icom_new-icom_old, axis=0)
    icom_old = icom_new

    xcom_avg += xcom

dt = 12.5   # [ps]
cl_displacement = 0.5*(np.array(dx_left)+np.array(dx_right))
time = np.linspace(0, dt*len(dx_left), len(dx_left))

rho /= (n_fin-n_init+1)
xcom_avg /= (n_fin-n_init+1)
rho_gol /= (n_fin-n_init+1)
rho_sol /= (n_fin-n_init+1)

rho_hslice = rho[:, j_hslice-2:j_hslice+3]
rho_hslice = np.mean(rho_hslice, axis=1)
rho_sol_hslice = rho_sol[:, j_hslice-2:j_hslice+3]
rho_sol_hslice = np.mean(rho_sol_hslice, axis=1)
rho_gol_hslice = rho_gol[:, j_hslice-2:j_hslice+3]
rho_gol_hslice = np.mean(rho_gol_hslice, axis=1)

bulk_density = dm.detect_bulk_density(rho, density_th=FP.max_vapour_density)

# Testing the circle fit (compare to local fit at equilibrium)
intf_contour = dm.detect_contour(rho, 0.5*bulk_density, hx, hz)
xc_l, zc_l, R_l, residue_l = dm.circle_fit_meniscus(intf_contour, z_th=FP.substrate_location, Lz=FP.lenght_z, Midx=0.5*FP.lenght_x, wing='l')
xc_r, zc_r, R_r, residue_r = dm.circle_fit_meniscus(intf_contour, z_th=FP.substrate_location, Lz=FP.lenght_z, Midx=0.5*FP.lenght_x, wing='r')
h = FP.substrate_location
cot_circle_bl = (h-zc_l)/np.sqrt(R_l*R_l-(h-zc_l)**2)
cot_circle_tl = (zc_l+h-Lz)/np.sqrt(R_l*R_l-(zc_l+h-Lz)**2)
cot_circle_br = (h-zc_r)/np.sqrt(R_r*R_r-(h-zc_r)**2)
cot_circle_tr = (zc_r+h-Lz)/np.sqrt(R_r*R_r-(zc_r+h-Lz)**2)
cot_circle = 0.25*(cot_circle_bl+cot_circle_tl+cot_circle_br+cot_circle_tr)
theta_circle = np.rad2deg( 0.5*math.pi+np.arctan( cot_circle ) )
theta_circle = (180-theta_circle)*(cot_circle>-1) + theta_circle*(cot_circle<=-1)

equilibrium = False

branches = dict()

branches['bl'], branches['br'] = dm.detect_interface_loc(rho, hx, hz, FP.substrate_location, FP.bulk_location, wall='l')
branches['tl'], branches['tr'] = dm.detect_interface_loc(rho, hx, hz, FP.substrate_location, FP.bulk_location, wall='t')

left_branch, right_branch = dm.detect_interface_int(rho, 0.5*bulk_density, hx, hz, FP.substrate_location)
i_com_avg = np.argmin(np.abs(x-xcom_avg))
nz_slice = int(0.5*Nz)
rho_slice = rho[:,nz_slice]
nx_win = 25
x_low = x[i_com_avg-nx_win]
x_upp = x[i_com_avg+nx_win]
rho_bulk_loc = dm.mean_density_loc(rho_slice, x, nx_win)

nx_slice = 75

# Slice of water
profile_sol = np.mean( rho_sol[i_com_avg-nx_slice:i_com_avg+nx_slice,:], axis=0 )

# Slice of glycerol
profile_gol = np.mean( rho_gol[i_com_avg-nx_slice:i_com_avg+nx_slice,:], axis=0 )

# Computing the effective mass fraction
i_ll = np.argmin(np.abs(z-z_si))
i_lu = np.argmin(np.abs(z-z_star))
i_ul = np.argmin(np.abs(z-(Lz-z_star)))
i_uu = np.argmin(np.abs(z-(Lz-z_si)))
alpha_w = 0.5*( np.mean(profile_sol[i_ll:i_lu+1]/bulk_density) + np.mean(profile_sol[i_ul:i_uu+1]/bulk_density) )
alpha_g = 0.5*( np.mean(profile_gol[i_ll:i_lu+1]/bulk_density) + np.mean(profile_gol[i_ul:i_uu+1]/bulk_density) )
alpha_w_eff = alpha_w/(alpha_w+alpha_g)
alpha_g_eff = alpha_g/(alpha_w+alpha_g)

near_wall_sol_upp = rho_sol[i_com_avg-nx_slice:i_com_avg+nx_slice,i_ul:i_uu+1]
near_wall_sol_low = rho_sol[i_com_avg-nx_slice:i_com_avg+nx_slice,i_ll:i_lu+1]
near_wall_gol_upp = rho_gol[i_com_avg-nx_slice:i_com_avg+nx_slice,i_ul:i_uu+1]
near_wall_gol_low = rho_gol[i_com_avg-nx_slice:i_com_avg+nx_slice,i_ll:i_lu+1]
alpha_w_std = 0.5*( np.std( np.mean(near_wall_sol_upp,axis=1)/bulk_density ) + 
    np.std( np.mean(near_wall_sol_low,axis=1)/bulk_density ) )
alpha_g_std = 0.5*( np.std( np.mean(near_wall_gol_upp,axis=1)/bulk_density ) + 
    np.std( np.mean(near_wall_gol_low,axis=1)/bulk_density ) )
"""
alpha_w_std = min(alpha_w_std,alpha_g_std)
alpha_g_std = min(alpha_w_std,alpha_g_std)
"""

print("########################################################")
print("alpha_w_eff = "+str(alpha_w_eff)+" +/- "+str(alpha_w_std))
print("alpha_g_eff = "+str(alpha_g_eff)+" +/- "+str(alpha_g_std))
print("########################################################")

# Average between sides
hNz = int(Nz/2)
profile_sol = 0.5*(profile_sol+np.flip(profile_sol))
profile_sol = profile_sol[0:hNz]

profile_sol_cs = np.cumsum(profile_sol)
profile_sol_cs /= profile_sol_cs[-1]

# Average between sides
profile_gol = 0.5*(profile_gol+np.flip(profile_gol))
profile_gol = profile_gol[0:hNz]

profile_gol_cs = np.cumsum(profile_gol)
profile_gol_cs /= profile_gol_cs[-1]

z_half = z[1:hNz]
i_bulk = 10
rho_sol_ref = np.mean(profile_sol[i_bulk:])/bulk_density
rho_gol_ref = np.mean(profile_gol[i_bulk:])/bulk_density

fig2, (ax11,ax33) = plt.subplots(1,2)
"""
ax11.semilogy(profile_sol[1:]/bulk_density, z_half, 'mo-', linewidth=3, label='water',
    markersize=8, markeredgewidth=2.5, mfc='w')
ax11.semilogy(profile_gol[1:]/bulk_density, z_half, 'gD-', linewidth=3, label='glycerol',
    markersize=8, markeredgewidth=2.5, mfc='w')
ax11.semilogy([rho_sol_ref, rho_sol_ref], [z_half[0], z_half[-1]], 'm--', linewidth=2)
ax11.semilogy([rho_gol_ref, rho_gol_ref], [z_half[0], z_half[-1]], 'g--', linewidth=2)
ax11.semilogy([0, 1], [z_si, z_si], 'k--', linewidth=2.5)
# ax11.plot([0, 1], [Lz-z_si, Lz-z_si], 'k--', linewidth=2.5)
ax11.semilogy([0, 1], [z_star, z_star], 'k:', linewidth=2.5)
# ax11.plot([0, 1], [Lz-z_star, Lz-z_star], 'k:', linewidth=2.5)
ax11.set_xlim([0.0, 1.0])
ax11.set_ylabel(r'$z$ [nm]', fontsize=35)
ax11.set_xlabel(r'$\rho(z)/\rho_{tot}$', fontsize=35)
ax11.tick_params(axis="both", labelsize=25)
ax11.legend(fontsize=25, loc='upper center')
# plt.show()
"""
ax11.semilogx(z_half, profile_sol[1:]/bulk_density, 'mo-', linewidth=3, label='water',
    markersize=8, markeredgewidth=2.5, mfc='w')
ax11.semilogx(z_half, profile_gol[1:]/bulk_density, 'gD-', linewidth=3, label='glycerol',
    markersize=8, markeredgewidth=2.5, mfc='w')
ax11.semilogx([z_half[0], z_half[-1]], [rho_sol_ref, rho_sol_ref], 'm--', linewidth=2)
ax11.semilogx([z_half[0], z_half[-1]], [rho_gol_ref, rho_gol_ref], 'g--', linewidth=2)
ax11.semilogx([z_si, z_si], [0, 1], 'k--', linewidth=2.5)
ax11.semilogx([z_star, z_star], [0, 1], 'k:', linewidth=2.5)
ax11.set_ylim([0.0, 1.0])
ax11.set_xlabel(r'$z$ [nm]', fontsize=35)
ax11.set_ylabel(r'$\rho(z)/\rho_{tot}$', fontsize=35)
ax11.tick_params(axis="both", labelsize=25)
ax11.legend(fontsize=25, loc='upper center')
# plt.show()

lb = 8
# fig3, ax33 = plt.subplots()
ax33.plot(z[0:lb], profile_sol_cs[0:lb], 'mo-', linewidth=3.0, markersize=8, markeredgewidth=2.5, mfc='w')
ax33.plot(z[0:lb], profile_gol_cs[0:lb], 'gD-', linewidth=3.0, markersize=8, markeredgewidth=2.5, mfc='w')
ax33.plot([z_si, z_si], [0.0, max(profile_sol_cs[lb-1],profile_gol_cs[lb-1])], 'k--', linewidth=2.5, label='silica')
ax33.plot([z_star, z_star], [0.0, max(profile_sol_cs[lb-1],profile_gol_cs[lb-1])], 'k:', linewidth=2.5)
ax33.legend(fontsize=25)
ax33.set_xlabel(r'$z$ [nm]', fontsize=35)
# ax33.set_ylabel(r'$\int\rho(z)dz/\rho_{sum}$', fontsize=30)
ax33.set_ylabel(r'$\overline{\rho}(z)$', fontsize=30)
ax33.tick_params(axis="both", labelsize=25)

plt.subplots_adjust(left=0.075, bottom=0.1, right=0.975, top=0.9, wspace=0.25, hspace=0.1)
plt.show()

"""
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(x, rho_slice/rho_bulk_loc, 'b-', linewidth=2.5)
ax1.plot(left_branch[0][nz_slice], 0.5*bulk_density/rho_bulk_loc, 'rx', markersize=15, markeredgewidth=2.25, label='interface')
ax1.plot(right_branch[0][nz_slice], 0.5*bulk_density/rho_bulk_loc, 'rx', markersize=15, markeredgewidth=2.25)
ax1.plot(xcom_avg, 0.5*bulk_density/rho_bulk_loc, 'ko', markersize=8, label='center of mass')
ax1.plot([x_low, x_low], [0, 1.2], 'k--', linewidth=2.5, label='window')
ax1.plot([x_upp, x_upp], [0, 1.2], 'k--', linewidth=2.5)
ax1.set_xlim([x[0],x[-1]])
ax1.set_ylim([0, 1.2])
ax1.legend(fontsize=20)
ax1.set_xlabel(r'$x$ [nm]', fontsize=25)
ax1.set_ylabel(r'$\rho_j(x)/\rho_{j,bulk}$ [1]', fontsize=25)
ax1.tick_params(axis="both", labelsize=25)
"""
# plt.show()

xfit = right_branch[0]-right_branch[0][0]
zfit = right_branch[1]-FP.substrate_location

contact_angles = dict()
contact_error = dict()

for k in branches.keys() :

    if k == 'bl' or k == 'br' :
        xfit = branches[k][0]-branches[k][0][0]
        zfit = branches[k][1]-FP.substrate_location
    else :
        xfit = branches[k][0]-branches[k][0][-1]
        zfit = (Lz-FP.substrate_location)-branches[k][1]

    fun2bfit = lambda x, a : a*np.abs(x)

    param, pcov = opt.curve_fit(fun2bfit, xfit, zfit)

    contact_angles[k] = np.rad2deg(np.arctan( param[0] ))
    theta_p = np.rad2deg(np.arctan( param[0]+pcov ))
    theta_m = np.rad2deg(np.arctan( param[0]-pcov ))
    contact_error[k] = np.abs(0.5*(theta_p-theta_m))[0][0]
    print("theta = "+str(contact_angles[k])+" +/- "+str(contact_error[k]))

if equilibrium :

    theta0 = 0.25*(contact_angles['bl']+contact_angles['tl']+contact_angles['br']+contact_angles['tr'])
    theta0_err = contact_error['bl']+contact_error['tl']+contact_error['br']+contact_error['tr']

    print("theta_0 = "+str(np.round(theta0,2))+" +/- "+str(np.round(theta0_err,2)))

else :

    theta_adv = 0.5*(contact_angles['tl']+contact_angles['br'])
    theta_rec = 0.5*(contact_angles['bl']+contact_angles['tr'])
    err_adv = contact_error['tl']+contact_error['br']
    err_rec = contact_error['bl']+contact_error['tr']

    print("theta_adv = "+str(np.round(theta_adv,2))+" +/- "+str(np.round(err_adv,2)))
    print("theta_rec = "+str(np.round(theta_rec,2))+" +/- "+str(np.round(err_rec,2)))

# rangle = np.linspace(0,2*np.pi,200)

left, bottom, width, height = [0.535, 0.38, 0.22, 0.22]

"""
ax3 = fig.add_axes([left, bottom, width, height])
ax3.pcolormesh(X, Z, rho/bulk_density, cmap=cm.Blues)
ax3.plot(branches['bl'][0], branches['bl'][1], 'rx', markersize=11, markeredgewidth=2.2)
ax3.plot(branches['bl'][0], branches['bl'][1], 'rx', markersize=11, markeredgewidth=2.2)
"""

plin = np.polyfit(branches['bl'][0], branches['bl'][1], 1)
xlin = np.linspace(branches['bl'][0][0], 58.0)

"""
plt.plot(xlin, np.polyval(plin, xlin), 'r-.', linewidth=2.25)
ax3.axis('scaled')
ax3.set_ylim([0, 2.5])
ax3.set_xlim([55.5, 58.0])
ax3.set_xlabel('$x$ [nm]', fontsize=17.5)
ax3.set_ylabel('$z$ [nm]', fontsize=17.5)
ax3.tick_params(axis="both", labelsize=12.5)
"""

"""
vis_map = ax2.pcolormesh(X, Z, rho/bulk_density, cmap=cm.Blues)
cbar = fig.colorbar(vis_map, ax=ax2)
cbar.set_label(r'$\rho/\rho_{bulk}$ [1]', fontsize=25)
cbar.ax.tick_params(labelsize=20)
ax2.plot(left_branch[0], left_branch[1]+0.5*hz, 'r-', linewidth=3.5, label='interface')
ax2.plot(right_branch[0], right_branch[1]+0.5*hz, 'r-', linewidth=3.5)
"""

tp = np.linspace(0, 2*np.pi, 500)

"""
ax2.plot(xc_l+R_l*np.cos(tp), zc_l+R_l*np.sin(tp), 'k--', linewidth=2, label='circle fit')
ax2.axis('scaled')
ax2.set_xlabel('$x$ [nm]', fontsize=25)
ax2.set_ylabel('$z$ [nm]', fontsize=25)
ax2.tick_params(axis="both", labelsize=25)
ax2.set_ylim([0, Lz])
ax2.set_xlim([42.5, 67.5])
ax2.legend(fontsize=20, loc='upper left')
"""

# plt.show()

# i_com_avg-nx_slice:i_com_avg+nx_slice

plt.pcolormesh(X, Z, rho_sol/bulk_density, cmap=cm.Blues)
plt.plot(left_branch[0], left_branch[1]+0.5*hz, 'r-', linewidth=3.0)
plt.plot(right_branch[0], right_branch[1]+0.5*hz, 'r-', linewidth=3.0)
plt.plot([x[i_com_avg-nx_slice],x[i_com_avg-nx_slice]], [z[0],z[-1]], 'k-', linewidth=2)
plt.plot([x[i_com_avg+nx_slice],x[i_com_avg+nx_slice]], [z[0],z[-1]], 'k-', linewidth=2)
plt.axis('scaled')
plt.xlabel('$x$ [nm]', fontsize=35)
plt.ylabel('$z$ [nm]', fontsize=35)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()