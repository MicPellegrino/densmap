import densmap as dm

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

folder_name = 'ShearChar/Q2/'
file_name = 'flow_00300.dat'
Lx = 159.75000
Lz = 30.36600

density_array = dm.read_density_file(folder_name+'/'+file_name, bin='y')

nx = density_array.shape[0]
nz = density_array.shape[1]

hx = Lx/nx          # [nm]
hz = Lz/nz          # [nm]

x = hx*np.arange(0.0, nx, 1.0, dtype=float)+0.5*hx
z = hz*np.arange(0.0, nz, 1.0, dtype=float)+0.5*hz
X, Z = np.meshgrid(x, z, sparse=False, indexing='ij')

r_h2o = 0.09584
alpha = 2.0
print("Conv. kernel support = "+str(alpha*r_h2o)+" nm")

smoother = dm.smooth_kernel(alpha*r_h2o, hx, hz)
smooth_density_array = dm.convolute(density_array, smoother)

# Window for profile tomography (half+/-win)
idx_half = int(nz/2)
win = 2

profile_no_smooth = np.zeros(nx, dtype=float)
profile_smooth    = np.zeros(nx, dtype=float)

for i in range(-win,win+1) :
    profile_no_smooth += density_array[:,idx_half+i]
    profile_smooth += smooth_density_array[:,idx_half+i]

profile_no_smooth /= (2*win+1)
profile_smooth /= (2*win+1)

plt.plot(x, profile_no_smooth, 'b-', linewidth=1.5, label='no filtering')
plt.plot(x, profile_smooth, 'r-', linewidth=1.5, label='with filtering')
plt.title("Conv. radius = "+str(alpha*r_h2o)+" nm; #bins win. = "+str(2*win+1), fontsize=15.0)
plt.xlim([x[0],x[-1]])
plt.legend(fontsize=15.0)
plt.xlabel("x [nm]", fontsize=15.0)
plt.ylabel("density [amu]", fontsize=15.0)
plt.show()
