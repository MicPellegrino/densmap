import densmap as dm

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as manimation

mpl.use("Agg")

# folder_name = 'Regen'
# file_root = 'flow_'
folder_name = '20nm/flow_adv_w1'
file_root = 'flow_'

# PARAMETERS TO TUNE
# Lx = 159.75000
# Lz = 30.36600
Lx = 60.00000
Lz = 35.37240

# CREATING MESHGRID
print("Creating meshgrid")
vel_x, vel_z = dm.read_velocity_file(folder_name+'/'+file_root+'00001.dat')
Nx = vel_x.shape[0]
Nz = vel_x.shape[1]
hx = Lx/Nx
hz = Lz/Nz
x = hx*np.arange(0.0,Nx,1.0, dtype=float)
z = hz*np.arange(0.0,Nz,1.0, dtype=float)
X, Z = np.meshgrid(x, z, sparse=False, indexing='ij')

# INITIALIZING SMOOTHING KERNEL
p = 2.0
r_mol = p*0.09584
smoother = dm.smooth_kernel(r_mol, hx, hz)
# Tecnically unnecessary ...
smoother_den = dm.smooth_kernel(r_mol/p, hx, hz)
# TIME AVERAGING
n_aver = 25

n_init = 1
n_fin = 1000
dt = 8.0
# dt = 25.0
# dt = 0.5*25.0

n_dump = 10
print("Producing movie of the kinetic energy")
FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict( title='Meniscus kinetic energy Ca=1.0', \
    artist='Michele Pellegrino', \
    comment='Just the tracked contour of a spreading droplet' )
writer = FFMpegWriter(fps=30, metadata=metadata)
fig = plt.figure(figsize=(14.0,7.0))
v_x_list = []
with writer.saving(fig, "droplet_dissipation.mp4", 250):
    t_label = '0.0'
    for idx in range(1, n_fin-n_init+1 ):
        if idx%n_dump==0 :
            print("Obtainig frame "+str(idx))
            t_label = str(dt*idx)+' ps'
        # Time-averaging window
        n_hist = min(n_aver, idx)
        w = np.exp(-np.linspace(0.0,5.0,n_hist))
        w = w / np.sum(w)
        tmp_x, tmp_z = dm.read_velocity_file(folder_name+'/'+file_root+'{:05d}'.format(idx)+'.dat')
        if idx > n_aver :
            v_x_list.append(tmp_x)
            v_x_list.pop(0)
        else :
            v_x_list.append(tmp_x)
        for k in range(n_hist) :
            if k == 0 :
                smooth_v_x = w[0]*v_x_list[-1]
            else :
                smooth_v_x += w[k]*v_x_list[-k-1]
        smooth_v_x = dm.convolute(smooth_v_x, smoother)
        dv_dx = ( np.roll(smooth_v_x, 1, axis=0) + np.roll(smooth_v_x, -1, axis=0) ) / (2.0*hz)
        dissipation = np.multiply( dv_dx, dv_dx )
        # Tecnically unnecessary...
        rho = dm.read_density_file(folder_name+'/'+file_root+'{:05d}'.format(idx)+'.dat', bin='y')
        rho = dm.convolute(rho, smoother_den)
        plt.pcolormesh(X, Z, np.multiply(rho, dissipation), cmap=cm.jet, vmin=0.0, vmax=3.5)
        plt.colorbar()
        plt.xlabel('x [nm]')
        plt.ylabel('z [nm]')
        plt.title('Viscous energy @'+t_label)
        plt.axis('scaled')
        writer.grab_frame()
        plt.cla()
        plt.clf()

mpl.use("TkAgg")
