import densmap as dm
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

import itertools

from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

folder = 'VelocityProfileHexane'
filenames = ['HexCa005q60','HexCa006q60','HexCa007q60','HexCa008q60','HexCa009q60','HexCa010q60','HexCa011q60','HexCa012q60']

"""
delta_lambda = 8.939130434782607e-06 # Ca=0.12
delta_lambda = 8.194202898550723e-06 # Ca=0.11
delta_lambda = 7.4492753623188395e-6 # Ca=0.10
delta_lambda = 6.7043478260869555dle-06 # Ca=0.09
delta_lambda = 5.959420289855072e-06 # Ca=0.08
delta_lambda = 5.214492753623188e-06 # Ca=0.07
delta_lambda = 4.469565217391304e-06 # Ca=0.06
delta_lambda = 3.7246376811594197e-06 # Ca=0.05
"""
dl = np.array([3.7246376811594197e-06,4.469565217391304e-06,5.214492753623188e-06,5.959420289855072e-06,
               6.7043478260869555e-06,7.4492753623188395e-06,8.194202898550723e-06,8.939130434782607e-06])
dt = 0.002
dx = 1.0
uw = dx*dl/dt

Lz = 21.15280
H = 18.35

# Fitting function for the velocity profile
v_smooth_fit = lambda z, a, b : a*(z-0.5)**3 + b*(z-0.5)

# Silicon
# zw_si = 0.9599999

# Oxygen
# zw_si = 1.1110000

# Between oxygen and water
zw_si = 1.2110000

# Phase field
# zw_si = 0.5*(Lz-H)

i_exc = 7

npzfile = np.load(folder+'/'+filenames[0]+'_vxz_wat.npz')
z_scaled = npzfile['arr_0']/Lz

vxz_wat_avg = np.zeros_like(z_scaled)
vxz_hex_avg = np.zeros_like(z_scaled)

n_data = len(dl)
n_batch = int(0.8*len(dl))

fac = 1.0/n_data
for i in range(n_data) :

    npzfile = np.load(folder+'/'+filenames[i]+'_vxz_wat.npz')
    vxz_wat_scaled = npzfile['arr_1']/uw[i]
    vxz_wat_avg += fac*vxz_wat_scaled

    plt.plot(z_scaled[i_exc:-i_exc],vxz_wat_scaled[i_exc:-i_exc],'bo')

    npzfile = np.load(folder+'/'+filenames[i]+'_vxz_hex.npz')
    vxz_hex_scaled = npzfile['arr_1']/uw[i]
    vxz_hex_avg += fac*vxz_hex_scaled

    plt.plot(z_scaled[i_exc:-i_exc],vxz_hex_scaled[i_exc:-i_exc],'kx')

plt.ylabel(r'$u_x$/$u_w$',fontsize=35)
plt.xlabel(r'$z$/$L_z$',fontsize=35)
plt.tick_params(labelsize=25)
plt.show()

plt.plot(z_scaled[i_exc:-i_exc],vxz_wat_avg[i_exc:-i_exc],'bo',ms=7.5)
plt.plot(z_scaled[i_exc:-i_exc],vxz_hex_avg[i_exc:-i_exc],'kx',ms=10)

p1 = np.polyfit(z_scaled[i_exc:-i_exc], vxz_wat_avg[i_exc:-i_exc], deg=3)
p2 = np.polyfit(z_scaled[i_exc:-i_exc], vxz_hex_avg[i_exc:-i_exc], deg=3)

popt1, pcov1 = opt.curve_fit(v_smooth_fit, z_scaled[i_exc:-i_exc], vxz_wat_avg[i_exc:-i_exc], p0=(p1[0],p1[2]))
popt2, pcov2 = opt.curve_fit(v_smooth_fit, z_scaled[i_exc:-i_exc], vxz_hex_avg[i_exc:-i_exc], p0=(p2[0],p2[2]))

v_smooth_fit1 = lambda z : popt1[0]*(z-0.5)**3 + popt1[1]*(z-0.5)
v_smooth_fit2 = lambda z : popt2[0]*(z-0.5)**3 + popt2[1]*(z-0.5)
dv_smooth_fit1 = lambda z : 3*popt1[0]*(z-0.5)**2 + popt1[1]
dv_smooth_fit2 = lambda z : 3*popt2[0]*(z-0.5)**2 + popt2[1]

ls_wat = (1-v_smooth_fit1((Lz-zw_si)/Lz))/dv_smooth_fit1((Lz-zw_si)/Lz)
ls_hex = (1-v_smooth_fit2((Lz-zw_si)/Lz))/dv_smooth_fit2((Lz-zw_si)/Lz)

plt.plot(z_scaled, v_smooth_fit(z_scaled,*popt1), 'b-', linewidth=4)
plt.plot(z_scaled, v_smooth_fit(z_scaled,*popt2), 'k-', linewidth=4)

plt.plot(z_scaled,  np.ones(len(z_scaled)), 'k-', linewidth=5)
plt.plot(z_scaled, -np.ones(len(z_scaled)), 'k-', linewidth=5)
plt.plot([zw_si/Lz,zw_si/Lz], [-1,1], 'r:', linewidth=5)
plt.plot([1-zw_si/Lz,1-zw_si/Lz], [-1,1], 'r:', linewidth=5)
plt.xlim([0,1])
plt.ylim([-1,1])
plt.ylabel(r'$u_x$/$u_w$',fontsize=35)
plt.xlabel(r'$z$/$L_z$',fontsize=35)
plt.tick_params(labelsize=25)

plt.show()

# Doing cross-validation to estimate uncertainty
fac = 1/n_batch
ls_wat_list = []
ls_hex_list = []
for comb in itertools.combinations(range(n_data), n_batch):
    print(comb)
    vxz_wat_avg = np.zeros_like(z_scaled)
    vxz_hex_avg = np.zeros_like(z_scaled)
    for i in comb :
        npzfile = np.load(folder+'/'+filenames[i]+'_vxz_wat.npz')
        vxz_wat_scaled = npzfile['arr_1']/uw[i]
        vxz_wat_avg += fac*vxz_wat_scaled
        npzfile = np.load(folder+'/'+filenames[i]+'_vxz_hex.npz')
        vxz_hex_scaled = npzfile['arr_1']/uw[i]
        vxz_hex_avg += fac*vxz_hex_scaled
    popt1, pcov1 = opt.curve_fit(v_smooth_fit, z_scaled[i_exc:-i_exc], vxz_wat_avg[i_exc:-i_exc], p0=(p1[0],p1[2]))
    popt2, pcov2 = opt.curve_fit(v_smooth_fit, z_scaled[i_exc:-i_exc], vxz_hex_avg[i_exc:-i_exc], p0=(p2[0],p2[2]))
    v_smooth_fit1 = lambda z : popt1[0]*(z-0.5)**3 + popt1[1]*(z-0.5)
    v_smooth_fit2 = lambda z : popt2[0]*(z-0.5)**3 + popt2[1]*(z-0.5)
    dv_smooth_fit1 = lambda z : 3*popt1[0]*(z-0.5)**2 + popt1[1]
    dv_smooth_fit2 = lambda z : 3*popt2[0]*(z-0.5)**2 + popt2[1]
    ls_wat_list.append((1-v_smooth_fit1((Lz-zw_si)/Lz))/dv_smooth_fit1((Lz-zw_si)/Lz))
    ls_hex_list.append((1-v_smooth_fit2((Lz-zw_si)/Lz))/dv_smooth_fit2((Lz-zw_si)/Lz))
ls_wat_list = np.array(ls_wat_list)
ls_hex_list = np.array(ls_hex_list)
sigma_ls_wat = np.std(ls_wat_list)
sigma_ls_hex = np.std(ls_hex_list)

print('Water   : l_s = ', ls_wat*Lz, "+/-", sigma_ls_wat*Lz)
print('Hexane  : l_s = ', ls_hex*Lz, "+/-", sigma_ls_hex*Lz)