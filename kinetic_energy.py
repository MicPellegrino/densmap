import densmap as dm

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

folder_name = '20nm/flow_adv_w1'
file_name = 'flow_00150.dat'
base_name = 'flow_'
# folder_name = '100nm/third_run'
# file_name = 'flow_00900.dat'
# folder_name = 'RawFlowData'
# file_name = 'flow_SOL_00100.dat'

# PARAMETERS TO TUNE
Lx = 60.00000
Lz = 35.37240
# Lx = 300.00000
# Lz = 200.44360
# Lx = 75.60000
# Lz = 28.00000

vel_x, vel_z = dm.read_velocity_file(folder_name+'/'+file_name)
rho = dm.read_density_file(folder_name+'/'+file_name, bin='y')
# p_x = np.multiply(rho, vel_x)
# p_z = np.multiply(rho, vel_z)
Nx = vel_x.shape[0]
Nz = vel_x.shape[1]
hx = Lx/Nx          # [nm]
hz = Lz/Nz          # [nm]
x = hx*np.arange(0.0,Nx,1.0, dtype=float)
z = hz*np.arange(0.0,Nz,1.0, dtype=float)
X, Z = np.meshgrid(x, z, sparse=False, indexing='ij')

kin_ener = 0.5*np.multiply(rho, np.multiply(vel_x, vel_x)+np.multiply(vel_z, vel_z))

# Try to average with values at previous steps
n_hist = 40
n_step = 150
n_hist = min(n_hist, n_step)
w = np.exp(-np.linspace(0.0,5.0,n_hist))
w = w / np.sum(w)
print("Performing time average")
# smooth_kin_ener *= w[0]
kin_ener *= w[0]
for k in range(n_hist-1) :
    vel_x, vel_z = dm.read_velocity_file(folder_name+'/'+base_name+'{:05d}'.format(n_step-k)+'.dat')
    tmp = 0.5*np.multiply(rho, np.multiply(vel_x, vel_x)+np.multiply(vel_z, vel_z))
    # tmp = dm.convolute(kin_ener, smoother)
    kin_ener += w[k+1]*tmp

p = 2.0
# r_mol = 0.39876
r_mol = p*0.09584
smoother = dm.smooth_kernel(r_mol, hx, hz)
smooth_kin_ener = dm.convolute(kin_ener, smoother)

plt.pcolormesh(X, Z, smooth_kin_ener, cmap=cm.jet)
plt.colorbar()
plt.axis('scaled')
plt.xlim([0,Lx])
plt.ylim([0,Lz])
plt.show()
