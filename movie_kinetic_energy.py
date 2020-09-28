import densmap as dm

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as manimation

mpl.use("Agg")

# FP = dm.fitting_parameters( par_file='SlipLenght/parameters_slip.txt' )
FP = dm.fitting_parameters( par_file='parameters_shear.txt' )

# folder_name = 'Theta90Ca30Init1000ps'
folder_name = FP.folder_name
file_root = 'flow_'
# folder_name = '20nm/flow_adv_w1'
# file_root = 'flow_'

# PARAMETERS TO TUNE
# Lx = 159.75000
# Lz = 30.36600
Lx = FP.lenght_x
Lz = FP.lenght_z
# Lx = 60.00000
# Lz = 35.37240

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
p = 3.0
# r_mol = p*0.09584
r_mol = p*FP.r_mol
smoother = dm.smooth_kernel(r_mol, hx, hz)
# TIME AVERAGING
n_aver = 50

n_init = FP.first_stamp
# n_fin = 100
n_fin = FP.last_stamp
# dt = 8.0
# dt = 25.0
dt = FP.time_step

n_dump = 10
print("Producing movie of the kinetic energy")
FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict( title='Shear flow Ca=0.2, Q2', \
    artist='Michele Pellegrino', \
    comment='Results from flow data binning' )
writer = FFMpegWriter(fps=30, metadata=metadata)
fig = plt.figure(figsize=(14.0,7.0))
p_x_list = []
p_z_list = []
kin_ener_list = []
with writer.saving(fig, "shear_q2_ca02.mp4", 250):
    t_label = '0.0'
    for idx in range(1, n_fin-n_init+1 ):
        if idx%n_dump==0 :
            print("Obtainig frame "+str(idx))
            t_label = str(dt*idx)+' ps'
        # Time-averaging window
        n_hist = min(n_aver, idx)
        w = np.exp(-np.linspace(0.0,5.0,n_hist))
        w = w / np.sum(w)
        rho = dm.read_density_file(folder_name+'/'+file_root+'{:05d}'.format(idx)+'.dat', bin='y')
        tmp_x, tmp_z = dm.read_velocity_file(folder_name+'/'+file_root+'{:05d}'.format(idx)+'.dat')
        tmp_e = 0.5*np.multiply( rho, np.multiply(tmp_x, tmp_x)+np.multiply(tmp_z, tmp_z) )
        tmp_x = np.multiply(rho, tmp_x)
        tmp_z = np.multiply(rho, tmp_z)
        if idx > n_aver :
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
        plt.streamplot(x, z, smooth_p_x.transpose(), smooth_p_z.transpose(), \
            density=2.0, arrowsize=0.5, color='w', linewidth=0.5)
        plt.pcolormesh(X, Z, smooth_kin_ener, cmap=cm.jet)
        plt.colorbar()
        plt.xlabel('x [nm]')
        plt.ylabel('z [nm]')
        plt.title('Kinetic energy @'+t_label)
        plt.axis('scaled')
        writer.grab_frame()
        plt.cla()
        plt.clf()

mpl.use("TkAgg")
