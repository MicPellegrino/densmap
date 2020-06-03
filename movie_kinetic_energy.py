import densmap as dm

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as manimation

mpl.use("Agg")

folder_name = 'RawFlowData'
file_root = 'flow_SOL_'

# PARAMETERS TO TUNE
Lx = 75.60000
Lz = 28.00000

# CREATING MESHGRID
print("Creating meshgrid")
vel_x, vel_z = dm.read_velocity_file(folder_name+'/'+file_root+'00001.dat')
kin_ener = 0.5*(np.multiply(vel_x, vel_x)+np.multiply(vel_z, vel_z))
Nx = vel_x.shape[0]
Nz = vel_x.shape[1]
hx = Lx/Nx
hz = Lz/Nz
x = hx*np.arange(0.0,Nx,1.0, dtype=float)
z = hz*np.arange(0.0,Nz,1.0, dtype=float)
X, Z = np.meshgrid(x, z, sparse=False, indexing='ij')

# INITIALIZING SMOOTHING KERNEL
r_mol = 0.39876
smoother = dm.smooth_kernel(r_mol, hx, hz)
# TIME AVERAGING
n_aver = 10

n_init = 1
n_fin = 200
dt = 10.0

n_dump = 10
print("Producing movie of the kinetic energy")
FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='Shear droplet kinetic energy', artist='Michele Pellegrino',
    comment='Just the tracked contour of a shear droplet')
writer = FFMpegWriter(fps=30, metadata=metadata)
fig = plt.figure()
# fig_ener = plt.pcolormesh(X, Z, kin_ener, cmap=cm.jet)
plt.xlabel('x [nm]')
plt.ylabel('z [nm]')
# smooth_kin_ener = np.NaN
with writer.saving(fig, "kinetic_energy.mp4", 250):
    for idx in range(1, n_fin-n_init+1 ):
        if idx%n_dump==0 :
            print("Obtainig frame "+str(idx))
            t_label = str(dt*idx)+' ps'
            plt.title('Kinetic energy @'+t_label)
        # Time-averaging window
        n_hist = min(n_aver, idx)
        w = np.exp(-np.linspace(0.0,5.0,n_hist))
        w = w / np.sum(w)
        for k in range(n_hist) :
            vel_x, vel_z = dm.read_velocity_file(folder_name+'/'+'flow_SOL_'+'{:05d}'.format(idx-k)+'.dat')
            kin_ener = 0.5*(np.multiply(vel_x, vel_x)+np.multiply(vel_z, vel_z))
            tmp = dm.convolute(kin_ener, smoother)
            if k == 0 :
                smooth_kin_ener = w[k]*tmp
            else :
                smooth_kin_ener += w[k]*tmp
        plt.pcolormesh(X, Z, smooth_kin_ener, cmap=cm.jet)
        plt.axis('scaled')
        writer.grab_frame()
        # textvar.remove()

mpl.use("TkAgg")
