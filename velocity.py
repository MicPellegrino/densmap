import densmap as dm

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

folder_name = 'RawFlowData'
file_name = 'combined_00100.dat'

# PARAMETERS TO TUNE
Lx = 75.60000
Lz = 28.00000

vel_x, vel_z = dm.read_velocity_file(folder_name+'/'+file_name)
kin_ener = 0.5*(np.multiply(vel_x, vel_x)+np.multiply(vel_z, vel_z))

Nx = vel_x.shape[0]
Nz = vel_x.shape[1]
hx = Lx/Nx          # [nm]
hz = Lz/Nz          # [nm]
x = hx*np.arange(0.0,Nx,1.0, dtype=float)
z = hz*np.arange(0.0,Nz,1.0, dtype=float)
X, Z = np.meshgrid(x, z, sparse=False, indexing='ij')

# Try to average with values at previous steps
n_hist = 10
n_step = 100
n_hist = min(n_hist, n_step)
w = np.exp(-np.linspace(0.0,5.0,n_hist))
w = w / np.sum(w)
print("Performing time average")
vel_x *= w[0]
vel_z *= w[0]
kin_ener *= w[0]
for k in range(n_hist-1) :
    tmp_x, tmp_z = dm.read_velocity_file(folder_name+'/'+'combined_'+'{:05d}'.format(n_step-k)+'.dat')
    vel_x += w[k+1]*tmp_x
    vel_z += w[k+1]*tmp_z
    tmp = 0.5*(np.multiply(tmp_x, tmp_x)+np.multiply(tmp_z, tmp_z))
    kin_ener += w[k+1]*tmp

r_mol = 0.39876
smoother = dm.smooth_kernel(r_mol, hx, hz)
smooth_vel_x = dm.convolute(vel_x, smoother)
smooth_vel_z = dm.convolute(vel_z, smoother)
smooth_kin_ener = dm.convolute(kin_ener, smoother)

plt.streamplot(x, z, smooth_vel_x.transpose(), smooth_vel_z.transpose(), \
    density=2.0, color='w')
plt.pcolormesh(X, Z, smooth_kin_ener, cmap=cm.jet)
plt.axis('scaled')
plt.xlim([0,Lx])
plt.ylim([0,Lz/2])
plt.show()
