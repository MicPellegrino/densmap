import densmap as dm

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm


## ## ### # ## ###
# INITIALIZATION #
# #### ## # # ## #

FP = dm.fitting_parameters( par_file='parameters_shear.txt' )

folder_name = FP.folder_name
file_root = 'flow_'
Lx = FP.lenght_x
Lz = FP.lenght_z
vel_x, vel_z = dm.read_velocity_file(folder_name+'/'+file_root+'00001.dat')
Nx = vel_x.shape[0]
Nz = vel_x.shape[1]
hx = Lx/Nx
hz = Lz/Nz
x = hx*np.arange(0.0,Nx,1.0, dtype=float)+0.5*hx
z = hz*np.arange(0.0,Nz,1.0, dtype=float)+0.5*hz
X, Z = np.meshgrid(x, z, sparse=False, indexing='ij')
n_init = FP.first_stamp
n_fin = FP.last_stamp
dt = FP.time_step
delta_th = 2.0
z0 = 0.80
i0 = np.abs(z-z0).argmin()
n_transient = int(max(1, n_fin-1200))
n_dump = 10

def fin_dif_2ord( interface ) :
    fd = np.zeros(interface.shape)
    fd[1:-1] = 0.5*(interface[2:]-interface[0:-2])/hz
    fd[0]    = 0.5*(-3*interface[0]+4*interface[1]-interface[2])/hz
    fd[-1]   = 0.5*(3*interface[-1]-4*interface[-2]+interface[-3])/hz
    return fd

arctan = lambda xp : np.rad2deg(np.arctan(xp))
arctan = np.vectorize(arctan)


## # ## # ## #
# STATISTICS #
# ## ## # # ##

# Let's have a broader picture
X_int = dict()
X_int['ml'] = [] 
X_int['mr'] = []
X_int['br'] = []
X_int['bl'] = []
X_int['tr'] = []
X_int['tl'] = []

# Center of mass
X_com = []

# Init data structures
print("Initialization")
file_name = file_root+'{:05d}'.format(n_transient)+'.dat'
density_array = dm.read_density_file(folder_name+'/'+file_name, bin='y')
# Center of mass (instant.)
xc = np.sum(np.multiply(density_array, X))/np.sum(density_array)
X_com.append( xc )
# COM for every slice along z
"""
x_com_slices = np.sum(np.multiply(density_array, X),axis=0)/np.sum(density_array,axis=0)
x_com_slices -= xc
p_linfit = np.polyfit(z[i0+10:Nz-i0-10], x_com_slices[i0+10:Nz-i0-10], deg=1)
"""
# Interface fitting
bulk_density = dm.detect_bulk_density(density_array, delta_th)
left_intf_mean, right_intf_mean = dm.detect_interface_int(density_array, 0.5*bulk_density, hx, hz, z0)
left_intf_mean[0,:] -= xc
right_intf_mean[0,:] -= xc
left_ca_mean  = arctan(fin_dif_2ord(left_intf_mean[0,:]))
right_ca_mean = arctan(fin_dif_2ord(right_intf_mean[0,:]))
N_mid = int(0.5*len(left_intf_mean[0,:]))
X_int['ml'].append(left_intf_mean[0,N_mid])
X_int['bl'].append(left_intf_mean[0,1])
X_int['mr'].append(right_intf_mean[0,N_mid])
X_int['br'].append(right_intf_mean[0,1])
X_int['tl'].append(left_intf_mean[0,-2])
X_int['tr'].append(right_intf_mean[0,-2]) 

print("Computing mean interface")
n = 1
for i in range(n_transient+1, n_fin+1 ) :
    n += 1
    if i % n_dump == 0 :
        print("Obtainig frame "+str(i))
    file_name = file_root+'{:05d}'.format(i)+'.dat'
    density_array = dm.read_density_file(folder_name+'/'+file_name, bin='y')
    xc = np.sum(np.multiply(density_array, X))/np.sum(density_array)
    X_com.append( xc )
    """
    x_com_slices = np.sum(np.multiply(density_array, X),axis=0)/np.sum(density_array,axis=0)
    x_com_slices -= xc
    p_linfit = np.polyfit(z[i0+10:Nz-i0-10], x_com_slices[i0+10:Nz-i0-10], deg=1)
    """
    bulk_density = dm.detect_bulk_density(density_array, delta_th)
    left_intf, right_intf = dm.detect_interface_int(density_array, 0.5*bulk_density, hx, hz, z0)
    left_intf[0,:] -= xc
    right_intf[0,:] -= xc
    left_ca_mean  += arctan(fin_dif_2ord(left_intf[0,:]))
    right_ca_mean += arctan(fin_dif_2ord(right_intf[0,:]))
    X_int['ml'].append(left_intf[0,N_mid])
    X_int['bl'].append(left_intf[0,1])
    X_int['mr'].append(right_intf[0,N_mid])
    X_int['br'].append(right_intf[0,1])
    X_int['tl'].append(left_intf[0,-2])
    X_int['tr'].append(right_intf[0,-2])
    left_intf_mean += left_intf
    right_intf_mean += right_intf

left_intf_mean /= n
right_intf_mean /= n
left_ca_mean /= n
right_ca_mean /= n

print("Computing mean square displacement")
n = 0
left_intf_msd = np.zeros( left_intf_mean.shape, dtype=float )
left_intf_msd[1,:] = left_intf_mean[1,:]
right_intf_msd = np.zeros( right_intf_mean.shape, dtype=float )
right_intf_msd[1,:] = left_intf_mean[1,:]
for i in range(n_transient, n_fin+1 ) :
    n += 1
    if i % n_dump == 0 :
        print("Obtainig frame "+str(i))
    file_name = file_root+'{:05d}'.format(i)+'.dat'
    density_array = dm.read_density_file(folder_name+'/'+file_name, bin='y')
    xc = np.sum(np.multiply(density_array, X))/np.sum(density_array)
    """
    x_com_slices = np.sum(np.multiply(density_array, X),axis=0)/np.sum(density_array,axis=0)
    x_com_slices -= xc
    p_linfit = np.polyfit(z[i0+10:Nz-i0-10], x_com_slices[i0+10:Nz-i0-10], deg=1)
    """
    bulk_density = dm.detect_bulk_density(density_array, delta_th)
    left_intf, right_intf = dm.detect_interface_int(density_array, 0.5*bulk_density, hx, hz, z0)
    left_intf[0,:] -= xc 
    right_intf[0,:] -= xc 
    diff_left = left_intf[0,:]-left_intf_mean[0,:]
    diff_right = right_intf[0,:]-right_intf_mean[0,:]
    left_intf_msd[0,:] += diff_left*diff_left
    right_intf_msd[0,:] += diff_right*diff_right

left_intf_msd[0,:] /= (n-1)
right_intf_msd[0,:] /= (n-1)


### #### ## ### # ## #
# CROSS-CORRELATIONS #
# ### ### ## #### # ##

# Fluctuations (signal - average)
for l in X_int.keys() :
    X_int[l] = np.array(X_int[l])
    X_int[l] -= np.mean(X_int[l])

X_com = np.array(X_com)
X_com -= np.mean(X_com)

ACF = dict()
for l in X_int.keys() :
    ACF[l] = np.zeros(X_int[l].shape)
    N = len(ACF[l])
    for k in range(N) :
        ACF[l][k] = np.sum(X_int[l]*np.roll(X_int[l], k))/N
    ACF[l] = ACF[l][:len(ACF[l])//2] 

# Correlation between top and bottom wetted area center
X_top = 0.5 * ( X_int['tr'] - X_int['tl'] )
X_bot = 0.5 * ( X_int['br'] - X_int['bl'] )
C_tb = np.zeros(X_top.shape)
N = len(C_tb)
for k in range(N) :
    C_tb[k] = np.sum(X_top*np.roll(X_bot, k))/N
C_tb = C_tb[:len(C_tb)//2] 


### ### #
# PLOTS #
# ### ###

time = np.linspace(0.0, dt*(n_fin-n_transient+1), len(X_int['ml']))

# INTERFACE POSITION SIGNAL #
plt.title('Time series (middle and m.c.l.)', fontsize=20.0)
plt.plot(time, X_com, 'k-', linewidth=1.5, label='com')
plt.plot(time, X_int['ml'], 'm-', linewidth=1.5, label='middle')
plt.plot(time, X_int['bl'], 'g-', linewidth=1.5, label='contact line')
plt.plot(time, np.zeros(time.shape), 'k--', linewidth=1.5)
plt.legend(fontsize=20.0)
plt.xticks(fontsize=20.0)
plt.yticks(fontsize=20.0)
plt.xlim([time[0], time[-1]])
plt.xlabel('t [ps]', fontsize=20.0)
plt.ylabel('x [nm]', fontsize=20.0)
plt.show()

# WETTED AREA MEAN POSITION # 
plt.title('Time series (wetted area)', fontsize=20.0)
plt.plot(time, X_top, 'b-', linewidth=1.5, label='top')
plt.plot(time, X_bot, 'r-', linewidth=1.5, label='bottom')
plt.plot(time, np.zeros(time.shape), 'k--', linewidth=1.5)
plt.legend(fontsize=20.0)
plt.xticks(fontsize=20.0)
plt.yticks(fontsize=20.0)
plt.xlim([time[0], time[-1]])
plt.xlabel('t [ps]', fontsize=20.0)
plt.ylabel('x [nm]', fontsize=20.0)
plt.show()

# INTERFACE POSITION ACF #
time = np.linspace(0.0, 0.5*dt*(n_fin-n_transient+1), len(ACF['ml']))
plt.title('ACF (middle and m.c.l.)', fontsize=20.0)
plt.plot(time, ACF['ml'], 'm-', linewidth=1.5, label='middle')
plt.plot(time, ACF['bl'], 'g-', linewidth=1.5, label='contact line')
plt.plot(time, np.zeros(time.shape), 'k--', linewidth=1.5)
plt.legend(fontsize=20.0)
plt.xticks(fontsize=20.0)
plt.yticks(fontsize=20.0)
plt.xlim([time[0], time[-1]])
plt.xlabel('t [ps]', fontsize=20.0)
plt.ylabel('x [nm]', fontsize=20.0)
plt.show()

# WETTED AREA CORRELATION # 
plt.title('Correlation (top-bottom)', fontsize=20.0)
plt.plot(time, C_tb, 'k-', linewidth=1.5, label='C_tb')
plt.plot(time, np.zeros(time.shape), 'k--', linewidth=1.5)
plt.legend(fontsize=20.0)
plt.xticks(fontsize=20.0)
plt.yticks(fontsize=20.0)
plt.xlim([time[0], time[-1]])
plt.xlabel('t [ps]', fontsize=20.0)
plt.ylabel('x [nm]', fontsize=20.0)
plt.show()

# MEAN INTERFACE +/- STD #
plt.plot(left_intf_mean[0,:], left_intf_mean[1,:], 'k-', linewidth=2.0, label='mean interface position')
plt.plot(right_intf_mean[0,:], left_intf_mean[1,:], 'k-', linewidth=2.0)
plt.plot(left_intf_mean[0,:]+np.sqrt(left_intf_msd[0,:]), left_intf_mean[1,:], 'k--', linewidth=1.5, label='sqrt(mean-square-displacement)')
plt.plot(right_intf_mean[0,:]+np.sqrt(right_intf_msd[0,:]), left_intf_mean[1,:], 'k--', linewidth=1.5)
plt.plot(left_intf_mean[0,:]-np.sqrt(left_intf_msd[0,:]), left_intf_mean[1,:], 'k--', linewidth=1.5)
plt.plot(right_intf_mean[0,:]-np.sqrt(right_intf_msd[0,:]), left_intf_mean[1,:], 'k--', linewidth=1.5)
plt.title('Interface; time window='+str((n_fin-n_transient)*dt*1e-3)+'ns', fontsize=20.0)
plt.legend(fontsize=20.0)
plt.xticks(fontsize=20.0)
plt.yticks(fontsize=20.0)
plt.xlabel('x [nm]', fontsize=20.0)
plt.ylabel('z [nm]', fontsize=20.0)
plt.show()

# RMS DISPLACEMENT #
plt.plot( np.sqrt(left_intf_msd[0,:]), left_intf_mean[1,:] , 'b--', linewidth=1.5, label='left')
plt.plot( np.sqrt(right_intf_msd[0,:]), left_intf_mean[1,:], 'r--', linewidth=1.5, label='right')
plt.title('Interface; time window='+str((n_fin-n_transient)*dt*1e-3)+'ns', fontsize=20.0)
plt.legend(fontsize=20.0)
plt.xticks(fontsize=20.0)
plt.yticks(fontsize=20.0)
plt.xlim([0.0, 1.0])
plt.xlabel('x [nm]', fontsize=20.0)
plt.ylabel('z [nm]', fontsize=20.0)
plt.show()

# MEAN CONTACT ANGLES #
plt.plot( left_intf_mean[1,:], left_ca_mean, 'b--', linewidth=1.5, label='left')
plt.plot( left_intf_mean[1,:], right_ca_mean, 'r--', linewidth=1.5, label='right')
plt.title('Interface; time window='+str((n_fin-n_transient)*dt*1e-3)+'ns', fontsize=20.0)
plt.legend(fontsize=20.0)
plt.xticks(fontsize=20.0)
plt.yticks(fontsize=20.0)
plt.xlim([left_intf_mean[1,0], left_intf_mean[1,-1]])
plt.xlabel('z [nm]', fontsize=20.0)
plt.ylabel('theta [deg]', fontsize=20.0)
plt.show()
