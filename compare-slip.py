import densmap as dm
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

folder1 = 'SlipLJ'
folder2 = 'SlipSiO'
file_root = 'flow_'

dl = 7.4492753623188395e-6
dt = 0.002
dx = 1.0
uw = dx*dl/dt
print(uw)

n_init = 100
n_fin = 400

Lx1 = 44.82
Lx2 = 45.0
Lz = 17.0

zw_lj = 1.09
zw_si = 1.15

temp1 = dm.read_density_file(folder1+'/'+file_root+str(n_init).zfill(5)+'.dat')
temp2 = dm.read_density_file(folder2+'/'+file_root+str(n_init).zfill(5)+'.dat')

Nx = temp1.shape[0]
Nz = temp1.shape[1]

hx1 = Lx1/Nx
hx2 = Lx2/Nx
hz = Lz/Nz

x1 = hx1*np.arange(0.0,Nx,1.0, dtype=float)+0.5*hx1
x2 = hx2*np.arange(0.0,Nx,1.0, dtype=float)+0.5*hx2

z = hz*np.arange(0.0,Nz,1.0, dtype=float)+0.5*hz

ihalf = (Nx//2)+(Nx%2)

vx1_avg = np.zeros((Nx,Nz))
vx2_avg = np.zeros((Nx,Nz))

for idx in range(n_init, n_fin+1):
    
    rho1 = dm.read_density_file(folder1+'/'+file_root+'{:05d}'.format(idx)+'.dat', bin='y')
    rho2 = dm.read_density_file(folder2+'/'+file_root+'{:05d}'.format(idx)+'.dat', bin='y')

    rhox1 = np.mean(rho1,axis=1)
    xcom1 = np.sum(rhox1*x1)/np.sum(rhox1)
    icom1 = int(np.round(xcom1/hx1))
    ishift1 = ihalf-icom1

    rhox2 = np.mean(rho2,axis=1)
    xcom2 = np.sum(rhox2*x2)/np.sum(rhox2)
    icom2 = int(np.round(xcom2/hx2))
    ishift2 = ihalf-icom2

    vx1, vz1 = dm.read_velocity_file(folder1+'/'+file_root+'{:05d}'.format(idx)+'.dat')
    vx1 = np.roll(vx1, ishift1, axis=0)
    vz1 = np.roll(vz1, ishift1, axis=0)

    vx2, vz2 = dm.read_velocity_file(folder2+'/'+file_root+'{:05d}'.format(idx)+'.dat')
    vx2 = np.roll(vx2, ishift2, axis=0)
    vz2 = np.roll(vz2, ishift2, axis=0)
    
    vx1_avg += vx1
    vx2_avg += vx2

vx1_avg /= (n_fin+1-n_init)
vx2_avg /= (n_fin+1-n_init)

n_avg = 3
vx1_profile = np.mean(vx1_avg[ihalf-n_avg:ihalf+n_avg+1,:],axis=0)
vx2_profile = np.mean(vx2_avg[ihalf-n_avg:ihalf+n_avg+1,:],axis=0)

n_skip = 5
z_clean = z[n_skip:-n_skip]
vx1_profile_clean = vx1_profile[n_skip:-n_skip]
vx1_profile_clean[0] = vx1_profile_clean[1]
vx1_profile_clean[-1] = vx1_profile_clean[-2]
vx2_profile_clean = vx2_profile[n_skip:-n_skip]
vx2_profile_clean[0] = vx2_profile_clean[1]
vx2_profile_clean[-1] = vx2_profile_clean[-2]

vx1_profile_clean = vx1_profile_clean.reshape(len(vx1_profile_clean)//5,5)
vx1_std = np.std(vx1_profile_clean,axis=1)
vx1_profile_clean = np.mean(vx1_profile_clean,axis=1)
vx2_profile_clean = vx2_profile_clean.reshape(len(vx2_profile_clean)//5,5)
vx2_std = np.std(vx2_profile_clean,axis=1)
vx2_profile_clean = np.mean(vx2_profile_clean,axis=1)
z_clean = z_clean.reshape(len(z_clean)//5,5)
z_clean = np.mean(z_clean,axis=1)

p1 = np.polyfit(z_clean, vx1_profile_clean, deg=3)
p2 = np.polyfit(z_clean, vx2_profile_clean, deg=3)

vx1_profile_fit = np.polyval(p1, z_clean)
vx2_profile_fit = np.polyval(p2, z_clean)

v_smooth_fit = lambda z, a, b : a*(z-0.5*Lz)**3 + b*(z-0.5*Lz)
popt1, pcov1 = opt.curve_fit(v_smooth_fit, z_clean, vx1_profile_clean, p0=(p1[0],p1[2]))
popt2, pcov2 = opt.curve_fit(v_smooth_fit, z_clean, vx2_profile_clean, p0=(p2[0],p2[2]))

v_smooth_fit1 = lambda z : popt1[0]*(z-0.5*Lz)**3 + popt1[1]*(z-0.5*Lz)
v_smooth_fit2 = lambda z : popt2[0]*(z-0.5*Lz)**3 + popt2[1]*(z-0.5*Lz)
dv_smooth_fit1 = lambda z : 3*popt1[0]*(z-0.5*Lz)**2 + popt1[1]
dv_smooth_fit2 = lambda z : 3*popt2[0]*(z-0.5*Lz)**2 + popt2[1]

ls_lj = (uw-v_smooth_fit1(Lz-zw_lj))/dv_smooth_fit1(Lz-zw_lj)
ls_si = (uw-v_smooth_fit2(Lz-zw_si))/dv_smooth_fit2(Lz-zw_si)

print('LJ   : l_s = ', ls_lj)
print('SiO2 : l_s = ', ls_si)

plt.errorbar(vx1_profile_clean/uw, z_clean, xerr=vx1_std/uw, yerr=hz*np.ones(len(vx1_std)), 
    marker='o', ms=12.5, elinewidth=4 ,ls='none', color='b', label='L-J')
plt.errorbar(vx2_profile_clean/uw, z_clean, xerr=vx2_std/uw, yerr=hz*np.ones(len(vx1_std)), 
    marker='o', ms=12.5, elinewidth=4 ,ls='none', color='r', label=r'SiO$_2$')

plt.plot(v_smooth_fit(z,*popt1)/uw, z, 'b-', linewidth=4)
plt.plot(v_smooth_fit(z,*popt2)/uw, z, 'r-', linewidth=4)

plt.plot( np.ones(len(z)), z, 'k-', linewidth=5)
plt.plot(-np.ones(len(z)), z, 'k-', linewidth=5)
plt.plot([-1,1], [zw_lj,zw_lj], 'b--', linewidth=5)
plt.plot([-1,1], [zw_si,zw_si], 'r:', linewidth=5)
plt.plot([-1,1], [Lz-zw_lj,Lz-zw_lj], 'b--', linewidth=5)
plt.plot([-1,1], [Lz-zw_si,Lz-zw_si], 'r:', linewidth=5)
plt.ylim([0,Lz])
plt.xlim([-1.1,1.1])

plt.legend(fontsize=45, loc='center right', bbox_to_anchor=[0.9,0.2])
plt.xlabel(r'$u_x$/$u_w$',fontsize=50)
plt.ylabel(r'$z$',fontsize=50)
plt.tick_params(labelsize=40)

plt.show()