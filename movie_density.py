import densmap as dm

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as manimation

mpl.use("Agg")

# Output file name
output_file_name = "meniscus_ca_10.mp4"

folder_name = 'Ca10Phobic'
file_root = 'flow_'

# PARAMETERS TO TUNE
Lx = 159.75000
Lz = 30.36600

# CREATING MESHGRID
print("Creating meshgrid")
density_array = dm.read_density_file(folder_name+'/'+file_root+'00001.dat', bin='y')
Nx = density_array.shape[0]
Nz = density_array.shape[1]
hx = Lx/Nx
hz = Lz/Nz
x = hx*np.arange(0.0,Nx,1.0, dtype=float)
z = hz*np.arange(0.0,Nz,1.0, dtype=float)
X, Z = np.meshgrid(x, z, sparse=False, indexing='ij')

# INITIALIZING SMOOTHING KERNEL
r_mol = 0.39876
smoother = dm.smooth_kernel(r_mol, hx, hz)

n_init = 1
n_fin = 480
dt = 25.0
delta_th = 2.0

n_dump = 10
print("Producing movie of the number density")
FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='Meniscus density profile', artist='Michele Pellegrino',
    comment='Just the tracked contour of a shear droplet')
writer = FFMpegWriter(fps=30, metadata=metadata)
fig = plt.figure()
with writer.saving(fig, output_file_name, 250):
    t_label = '0.0'
    for idx in range(1, n_fin-n_init+1 ):
        plt.xlabel('x [nm]')
        plt.ylabel('z [nm]')
        if idx%n_dump==0 :
            print("Obtainig frame "+str(idx))
            t_label = str(dt*idx)+' ps'
        density_array = dm.read_density_file(folder_name+'/'+file_root+ \
            '{:05d}'.format(idx)+'.dat', bin='y')
        smooth_density_array = dm.convolute(density_array, smoother)
        plt.pcolormesh(X, Z, smooth_density_array, cmap=cm.bone)
        bulk_density = dm.detect_bulk_density(smooth_density_array, delta_th)
        intf_contour = dm.detect_contour(smooth_density_array, 0.5*bulk_density, hx, hz)
        plt.plot(intf_contour[0,:], intf_contour[1,:], 'r-', linewidth=1.5)
        plt.axis('scaled')
        plt.title('Density profile @'+t_label)
        writer.grab_frame()
        plt.cla()
        plt.clf()

mpl.use("TkAgg")
