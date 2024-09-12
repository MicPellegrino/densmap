import densmap as dm
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.optimize as opt

FP = dm.fitting_parameters( par_file='parameters_shear.txt' )

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

dx_left = []
dx_right = []

xcom_avg = 0

for n in range(n_init+1, n_fin+1) :
    if n%10 == 0 :
        print("n = ", n)
    # TODO: Should be translated with the center of mass ...
    rmap = dm.read_density_file(folder_name+'/'+file_root+'{:05d}'.format(n)+'.dat')

    bulk_density = dm.detect_bulk_density(rmap, density_th=FP.max_vapour_density)
    b_bl, b_br = dm.detect_interface_loc(rmap, hx, hz, FP.substrate_location, FP.bulk_location, wall='l')
    b_tl, b_tr = dm.detect_interface_loc(rmap, hx, hz, FP.substrate_location, FP.bulk_location, wall='t')
    
    dx_left.append(b_tl[0][-1]-b_bl[0][0])
    dx_right.append(b_tr[0][-1]-b_br[0][0])  

    xcom = np.sum(rmap*X)/np.sum(rmap)
    icom_new = np.abs(x-xcom).argmin()
    rho += np.roll(rmap, icom_new-icom_old, axis=0)
    icom_old = icom_new

    xcom_avg += xcom

dt = 12.5   # [ps]
cl_displacement = 0.5*(np.array(dx_left)+np.array(dx_right))
time = np.linspace(0, dt*len(dx_left), len(dx_left))

rho /= (n_fin-n_init+1)
xcom_avg /= (n_fin-n_init+1)

rho_hslice = rho[:, j_hslice-2:j_hslice+3]
rho_hslice = np.mean(rho_hslice, axis=1)

bulk_density = dm.detect_bulk_density(rho, density_th=FP.max_vapour_density)

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

nx_slice = 35
z_si = 0.65

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(x, rho_slice/rho_bulk_loc, 'b-', linewidth=2.5)
ax1.plot(left_branch[0][nz_slice], 0.5*bulk_density/rho_bulk_loc, 'rx', markersize=15, markeredgewidth=2.25, label='interface')
ax1.plot(right_branch[0][nz_slice], 0.5*bulk_density/rho_bulk_loc, 'rx', markersize=15, markeredgewidth=2.25)
ax1.plot(xcom_avg, 0.5*bulk_density/rho_bulk_loc, 'ko', markersize=8, label='center of mass')
ax1.plot([x_low, x_low], [0, 1.3], 'k--', linewidth=2.5, label='window')
ax1.plot([x_upp, x_upp], [0, 1.3], 'k--', linewidth=2.5)
ax1.set_xlim([x[0],x[-1]])
ax1.set_ylim([0, 1.3])
ax1.legend(fontsize=25)
ax1.set_xlabel(r'$x$ [nm]', fontsize=30)
ax1.set_ylabel(r'$\rho_j(x)/\rho_{j,bulk}$ [1]', fontsize=30)
ax1.tick_params(axis="both", labelsize=25)

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

left, bottom, width, height = [0.545, 0.38, 0.22, 0.22]
ax3 = fig.add_axes([left, bottom, width, height])
ax3.pcolormesh(X, Z, rho/bulk_density, cmap=cm.Blues)
ax3.plot(branches['bl'][0], branches['bl'][1], 'rx', markersize=11, markeredgewidth=2.2)
ax3.plot(branches['bl'][0], branches['bl'][1], 'rx', markersize=11, markeredgewidth=2.2)

plin = np.polyfit(branches['bl'][0], branches['bl'][1], 1)
xlin = np.linspace(branches['bl'][0][0], 58.0)
plt.plot(xlin, np.polyval(plin, xlin), 'r-.', linewidth=2.25)

ax3.axis('scaled')
ax3.set_ylim([0, 2.5])
ax3.set_xlim([56.5, 59.0])
ax3.set_xlabel('$x$ [nm]', fontsize=22.5)
ax3.set_ylabel('$z$ [nm]', fontsize=22.5)
ax3.tick_params(axis="both", labelsize=22.5)

vis_map = ax2.pcolormesh(X, Z, rho/bulk_density, cmap=cm.Blues)
cbar = fig.colorbar(vis_map, ax=ax2)
cbar.set_label(r'$\rho/\rho_{bulk}$ [1]', fontsize=30)
cbar.ax.tick_params(labelsize=25)
ax2.plot(left_branch[0], left_branch[1]+0.5*hz, 'r-', linewidth=3.5, label='interface')
ax2.plot(right_branch[0], right_branch[1]+0.5*hz, 'r-', linewidth=3.5)

tp = np.linspace(0, 2*np.pi, 500)
ax2.plot(xc_l+R_l*np.cos(tp), zc_l+R_l*np.sin(tp), 'k--', linewidth=2, label='circle fit')

ax2.axis('scaled')
ax2.set_xlabel('$x$ [nm]', fontsize=30)
ax2.set_ylabel('$z$ [nm]', fontsize=30)
ax2.tick_params(axis="both", labelsize=25)

ax2.set_ylim([0, Lz])
ax2.set_xlim([42.5, 67.5])

ax2.legend(fontsize=25, loc='upper left')

plt.show()