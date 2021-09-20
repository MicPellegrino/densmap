import densmap as dm

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as manimation

mpl.use("Agg")

FP = dm.fitting_parameters( par_file='parameters_shear.txt' )

folder_name = FP.folder_name
file_root = 'flow_'

Lx = FP.lenght_x
Lz = FP.lenght_z

# CREATING MESHGRID
print("Creating meshgrid")
vel_x, vel_z = dm.read_velocity_file(folder_name+'/'+file_root+'00100.dat')
Nx = vel_x.shape[0]
Nz = vel_x.shape[1]
hx = Lx/Nx
hz = Lz/Nz
x = hx*np.arange(0.0,Nx,1.0, dtype=float)
z = hz*np.arange(0.0,Nz,1.0, dtype=float)
X, Z = np.meshgrid(x, z, sparse=False, indexing='ij')

# Manually tune, crop window [nm]
x0_crop = 0.0
x1_crop = Lx
z0_crop = 0.0
z1_crop = Lz
idx_x0 = np.argmin(np.abs(x-x0_crop))
idx_x1 = np.argmin(np.abs(x-x1_crop))
idx_z0 = np.argmin(np.abs(z-z0_crop))
idx_z1 = np.argmin(np.abs(z-z1_crop))
x0_crop = x[idx_x0]
x1_crop = x[idx_x1]
z0_crop = z[idx_z0]
z1_crop = z[idx_z1]
x_crop = x[idx_x0:idx_x1]
z_crop = z[idx_z0:idx_z1]
X_crop, Z_crop = np.meshgrid(x_crop, z_crop, sparse=False, indexing='ij') 

print("Zoom-in window: ["+str(x0_crop)+","+str(x1_crop)+"]x["+str(z0_crop)+","+str(z1_crop)+"], (dX x dZ)")

# Testing .vtk output function
# vtk_folder = "/home/michele/densmap/BreakageVtk"
# dm.export_vector_vtk(x_crop, z_crop, hx, hz, 2.5, vel_x[idx_x0:idx_x1,idx_z0:idx_z1], vel_z[idx_x0:idx_x1,idx_z0:idx_z1])

# INITIALIZING SMOOTHING KERNEL
p = 2.0
r_mol = p*FP.r_mol
smoother = dm.smooth_kernel(r_mol, hx, hz)

# TIME AVERAGING
n_aver = 20

n_init = FP.first_stamp
n_fin = FP.last_stamp
dt = FP.time_step

n_dump = 10
print("Producing movie of the kinetic energy")
FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict( title='Shear flow Ca=0.3, Q1', \
    artist='Michele Pellegrino', \
    comment='Results from flow data binning' )
writer = FFMpegWriter(fps=30, metadata=metadata)
fig = plt.figure(figsize=(14.0,7.0))
p_x_list = []
p_z_list = []
kin_ener_list = []
with writer.saving(fig, "shear_q1_ca15.mp4", 250):
    t_label = '0.0'
    for idx in range(n_init, n_fin+1):
        if idx%n_dump==0 :
            print("Obtainig frame "+str(idx))
            t_label = str(dt*idx)+' ps'
        # Time-averaging window
        n_hist = min(n_aver, idx-n_init+1)
        w = np.exp(-np.linspace(0.0,5.0,n_hist))
        w = w / np.sum(w)
        rho = dm.read_density_file(folder_name+'/'+file_root+'{:05d}'.format(idx)+'.dat', bin='y')
        rho = rho[idx_x0:idx_x1,idx_z0:idx_z1]
        tmp_x, tmp_z = dm.read_velocity_file(folder_name+'/'+file_root+'{:05d}'.format(idx)+'.dat')
        tmp_x = tmp_x[idx_x0:idx_x1,idx_z0:idx_z1]
        tmp_z = tmp_z[idx_x0:idx_x1,idx_z0:idx_z1]
        tmp_e = 0.5*np.multiply( rho, np.multiply(tmp_x, tmp_x)+np.multiply(tmp_z, tmp_z) )
        tmp_x = np.multiply(rho, tmp_x)
        tmp_z = np.multiply(rho, tmp_z)
        if idx-n_init+1 > n_aver :
            p_x_list.append(tmp_x)
            p_x_list.pop(0)
            p_z_list.append(tmp_z)
            p_z_list.pop(0)
            kin_ener_list.append(tmp_e)
            kin_ener_list.pop(0)
        else :
            p_x_list.append(tmp_x)
            p_z_list.append(tmp_z)
            kin_ener_list.append(tmp_e)
        for k in range(n_hist) :
            if k == 0 :
                smooth_p_x = w[0]*p_x_list[-1]
                smooth_p_z = w[0]*p_z_list[-1]
                smooth_kin_ener = w[0]*kin_ener_list[-1]
            else :
                smooth_p_x += w[k]*p_x_list[-k-1]
                smooth_p_z += w[k]*p_z_list[-k-1]
                smooth_kin_ener += w[k]*kin_ener_list[-k-1]
        smooth_kin_ener = dm.convolute(smooth_kin_ener, smoother)
        smooth_p_x = dm.convolute(smooth_p_x, smoother)
        smooth_p_z = dm.convolute(smooth_p_z, smoother)
        plt.streamplot(x_crop, z_crop, smooth_p_x.transpose(), smooth_p_z.transpose(), \
            density=2.0, arrowsize=0.5, color='w', linewidth=0.5)
        plt.pcolormesh(X_crop, Z_crop, smooth_kin_ener, cmap=cm.inferno, vmin=0.0, vmax=1.5)
        plt.colorbar()
        plt.xlabel('x [nm]')
        plt.ylabel('z [nm]')
        plt.title('Kinetic energy @'+t_label)
        plt.axis('scaled')
        writer.grab_frame()
        plt.cla()
        plt.clf()

        # Testing .vtk output function
        """
        if n_hist==n_aver :
            # print(str(idx).zfill(5))
            dm.export_vector_vtk(x_crop, z_crop, hx, hz, 2.5, smooth_p_x, smooth_p_z,file_name=vtk_folder+"/momentum_"+str(idx).zfill(5)+".vtk")
        """

mpl.use("TkAgg")
