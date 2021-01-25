import densmap as dm

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as manimation

mpl.use("Agg")

# Output file name
output_file_name = "t94_ca01_diff_ds.mp4"

folder_ens1 = 'Theta94Ca010_10nm'
folder_ens2 = 'Theta94Ca01_double'
file_root = 'flow_'

# PARAMETERS TO TUNE
Lx = 159.57001
Lz = 29.04330

# CREATING MESHGRID
print("Creating meshgrid")
density_array = dm.read_density_file(folder_ens1+'/'+file_root+'00001.dat', bin='y')
Nx = density_array.shape[0]
Nz = density_array.shape[1]
hx = Lx/Nx
hz = Lz/Nz
x = hx*np.arange(0.0,Nx,1.0, dtype=float)
z = hz*np.arange(0.0,Nz,1.0, dtype=float)
X, Z = np.meshgrid(x, z, sparse=False, indexing='ij')

# Section for density computation
N_low = int(np.floor(70.0/hx))
N_upp = int(np.ceil(90.0/hx))

# INITIALIZING SMOOTHING KERNEL
r_mol = 0.39876
smoother = dm.smooth_kernel(r_mol, hx, hz)

n_init = 1
n_fin = 592
dt = 12.5
delta_th = 2.0

n_dump = 10
print("Producing movie of the difference in number density")
FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='Meniscus density profile', artist='Michele Pellegrino',
    comment='Just the tracked contour of a shear droplet')
writer = FFMpegWriter(fps=30, metadata=metadata)
fig = plt.figure()
area = []

# Density profile
density_profile = np.zeros( Nz, dtype=float )

with writer.saving(fig, output_file_name, 250):
    t_label = '0.0'
    for idx in range(1, n_fin-n_init+1 ):
        plt.xlabel('x [nm]')
        plt.ylabel('z [nm]')
        if idx%n_dump==0 :
            print("Obtainig frame "+str(idx))
            t_label = str(dt*idx)+' ps'
        density_array = dm.read_density_file(folder_ens1+'/'+file_root+ \
            '{:05d}'.format(idx)+'.dat', bin='y') - \
            dm.read_density_file(folder_ens2+'/'+file_root+ \
            '{:05d}'.format(idx)+'.dat', bin='y')
        # PLOT ORIGINAL DENSITY
        plt.pcolormesh(X, Z, density_array, cmap=cm.bone)
        # PLOT SMOOTHED DENSITY
        # smooth_density_array = dm.convolute(density_array, smoother)
        # plt.pcolormesh(X, Z, smooth_density_array, cmap=cm.bone)
        
        """
        bulk_density = dm.detect_bulk_density(smooth_density_array, delta_th)
        indicator = dm.density_indicator(smooth_density_array,0.5*bulk_density)
        if idx%n_dump==0 :
            area.append(hx*hz*np.sum(indicator))
            print("Droplet area = "+str(area[-1])+" nm^2")
        density_profile += np.sum( density_array[N_low:N_upp,:], axis=0 )
        plt.pcolormesh(X, Z, indicator, cmap=cm.bone)
        intf_contour = dm.detect_contour(smooth_density_array, 0.5*bulk_density, hx, hz)
        plt.plot(intf_contour[0,:], intf_contour[1,:], 'r-', linewidth=1.5)
        """

        plt.axis('scaled')
        plt.title('Density profile @'+t_label)
        writer.grab_frame()
        plt.cla()
        plt.clf()

mpl.use("TkAgg")

# POST-PROCESSING ...
"""
density_profile /= (n_fin-n_init)
area = np.array(area)
avg_area = np.mean(area)
print("Average droplet area = "+str(avg_area)+" nm^2")
plt.plot(z, density_profile)
plt.show()
Nz_th = int(5.0/hz)
bulk_density = np.mean(density_profile[Nz_th:Nz-Nz_th])
L_hat = (1.0/bulk_density) * sum(density_profile) * hz
print("L_hat = "+str(L_hat)+" nm")
"""
