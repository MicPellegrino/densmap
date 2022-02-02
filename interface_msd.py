import densmap as dm

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

import pandas as pd


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
n_transient = FP.first_stamp
n_dump = 10

def fin_dif_2ord( interface ) :
    fd = np.zeros(interface.shape)
    fd[1:-1] = 0.5*(interface[2:]-interface[0:-2])/hz
    fd[0]    = 0.5*(-3*interface[0]+4*interface[1]-interface[2])/hz
    fd[-1]   = 0.5*(3*interface[-1]-4*interface[-2]+interface[-3])/hz
    return fd

arctan = lambda xp : np.rad2deg(np.arctan(xp))
arccot = lambda xp : 90.0 - np.rad2deg(np.arctan(xp))
arctan = np.vectorize(arctan)
arccot = np.vectorize(arccot)

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
# Interface fitting
bulk_density = dm.detect_bulk_density(density_array, delta_th)
left_intf_mean, right_intf_mean = dm.detect_interface_int(density_array, 0.5*bulk_density, hx, hz, z0)
left_intf_mean[0,:] -= xc
right_intf_mean[0,:] -= xc
left_intf2_mean = left_intf_mean[0,:]*left_intf_mean[0,:]
right_intf2_mean = right_intf_mean[0,:]*right_intf_mean[0,:]
# left_intf4_mean = left_intf_mean[0,:]*left_intf_mean[0,:]*left_intf_mean[0,:]*left_intf_mean[0,:]
# right_intf4_mean = right_intf_mean[0,:]*right_intf_mean[0,:]*right_intf_mean[0,:]*right_intf_mean[0,:]
d1_left = fin_dif_2ord(left_intf_mean[0,:])
d1_right = fin_dif_2ord(right_intf_mean[0,:])
left_ca_mean  = arccot(fin_dif_2ord(left_intf_mean[0,:]))
right_ca_mean = arccot(fin_dif_2ord(right_intf_mean[0,:]))
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
    bulk_density = dm.detect_bulk_density(density_array, delta_th)
    left_intf, right_intf = dm.detect_interface_int(density_array, 0.5*bulk_density, hx, hz, z0)
    left_intf[0,:] -= xc
    right_intf[0,:] -= xc
    dl = fin_dif_2ord(left_intf[0,:])
    dr = fin_dif_2ord(right_intf[0,:])
    d1_left += dl
    d1_right += dr
    left_ca_mean  += arccot(dl)
    right_ca_mean += arccot(dr)
    X_int['ml'].append(left_intf[0,N_mid])
    X_int['bl'].append(left_intf[0,1])
    X_int['mr'].append(right_intf[0,N_mid])
    X_int['br'].append(right_intf[0,1])
    X_int['tl'].append(left_intf[0,-2])
    X_int['tr'].append(right_intf[0,-2])
    left_intf_mean += left_intf
    right_intf_mean += right_intf
    left_intf2_mean += left_intf[0,:]*left_intf[0,:]
    right_intf2_mean += right_intf[0,:]*right_intf[0,:]
    # left_intf4_mean += left_intf[0,:]*left_intf[0,:]*left_intf[0,:]*left_intf[0,:]
    # right_intf4_mean += right_intf[0,:]*right_intf[0,:]*right_intf[0,:]*right_intf[0,:]

left_intf_mean /= n
right_intf_mean /= n
left_intf2_mean /= n
right_intf2_mean /= n
# left_intf4_mean /= n
# right_intf4_mean /= n
left_ca_mean /= n
right_ca_mean /= n
d1_left /= n
d1_right /= n
d2_left = fin_dif_2ord(d1_left)
d2_right = fin_dif_2ord(d1_right)
k_left = d2_left/((1+d1_left**2)**(1.5))
k_right = d2_right/((1+d1_right**2)**(1.5))

left_intf_msd = left_intf2_mean - left_intf_mean[0,:]*left_intf_mean[0,:]
right_intf_msd = right_intf2_mean - right_intf_mean[0,:]*right_intf_mean[0,:]

# left_intf_msd_var = \
#     left_intf4_mean - left_intf2_mean*left_intf2_mean # - 4 * left_intf_mean[0,:]**4 + 4 * (left_intf_mean[0,:]**2)*left_intf2_mean
# right_intf_msd_var = \
#     right_intf4_mean - right_intf2_mean*right_intf2_mean - 4 * right_intf_mean[0,:]**4 + 4 * (right_intf_mean[0,:]**2)*right_intf2_mean


### #### ## ## #### #
# AUTO-CORRELATIONS #
# ### ### ## ### # ##

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

### #### ## ## ### # #
# CROSS-CORRELATIONS #
# ### ### ## ### ## ##

data = {'TL': X_int['bl'],
        'BL': X_int['br'],
        'TR': X_int['tl'],
        'BR': X_int['tr'] }
df = pd.DataFrame(data,columns=['TL','BL','TR','BR'])
corrMatrix = pd.DataFrame.corr(df)
print(corrMatrix)

print(r"std(X_ml) = ", np.sqrt(np.var(X_int['ml'])))
print(r"std(X_bl) = ", np.sqrt(np.var(X_int['bl'])))

rmsf = dict()
std_rmsf = dict()
max_rmsf = 0 
max_k = ''
err_mean_top = 0
err_mean_bot = 0
err_mean_mid = 0
for k in X_int.keys() :
    rmsf[k] = np.sqrt(np.var(X_int[k]))
    std_rmsf[k] = (np.var(X_int[k]*X_int[k]))**0.25
    if rmsf[k] > max_rmsf :
        max_rmsf = rmsf[k]
        max_k = k
    if k =='tl' or k=='br' :
        err_mean_top += std_rmsf[k]
    if k =='tr' or k=='bl' :
        err_mean_bot += std_rmsf[k]
    if k =='ml' or k=='mr' :
        err_mean_mid += std_rmsf[k]

dec_time = 500
Neff = len(X_int[max_k])/(500/dt)

# err_rmsf_right = ((right_intf_msd_var)**0.25)/np.sqrt(Neff)
# err_rmsf_left = ((left_intf_msd_var)**0.25)/np.sqrt(Neff)
# err_mean = 0.5*(err_rmsf_right+err_rmsf_left)

err_max = std_rmsf[max_k]/np.sqrt(Neff)

err_mean_top = 0.5*err_mean_top/np.sqrt(Neff)
err_mean_bot = 0.5*err_mean_bot/np.sqrt(Neff)
err_mean_mid = 0.5*err_mean_mid/np.sqrt(Neff)

err_mean = np.zeros(left_intf_mean[1,:].size)
midpoint = int(0.5*len(err_mean))
otherhalf = len(err_mean)-midpoint
for ii in range(0,midpoint) :
    err_mean[ii] = ((midpoint-ii)*err_mean_bot + ii*err_mean_mid)/midpoint
for ii in range(midpoint, len(err_mean)) :
    err_mean[ii] = ((len(err_mean)-ii)*err_mean_mid + (ii-midpoint)*err_mean_top)/otherhalf

print("rmsf["+max_k+"] = ", rmsf[max_k] )
print("err(rmsf["+max_k+"]) = ", err_max )

### ### #
# PLOTS #
# ### ###

time = np.linspace(0.0, dt*(n_fin-n_transient+1), len(X_int['ml']))

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)

# MEAN INTERFACE +/- STD #
ax1.plot(left_intf_mean[0,:], left_intf_mean[1,:], 'k-', linewidth=2.0, label=r'$<x>$')
ax2.plot(right_intf_mean[0,:], right_intf_mean[1,:], 'k-', linewidth=2.0)
ax1.fill_betweenx(left_intf_mean[1,:], left_intf_mean[0,:]-np.sqrt(left_intf_msd), left_intf_mean[0,:]+np.sqrt(left_intf_msd), \
    facecolor='lightblue', label=r'$\pm$RMSF')
ax2.fill_betweenx(right_intf_mean[1,:], right_intf_mean[0,:]-np.sqrt(right_intf_msd), right_intf_mean[0,:]+np.sqrt(right_intf_msd), facecolor='lightblue')
print("Time window: "+str((n_fin-n_transient)*dt*1e-3)+"ns")
ax1.set_title('Left interface', fontsize=30.0)
ax2.set_title('Right interface', fontsize=30.0)
ax1.legend(fontsize=20.0)
ax1.tick_params(labelsize=25)
ax1.set_xlabel('x [nm]', fontsize=30.0)
ax1.set_ylabel('y [nm]', fontsize=30.0)
ax1.set_xlim([-26, -14])
ax1.set_ylim([0, Lz])
ax2.tick_params(labelsize=25)
ax2.set_xlabel('x [nm]', fontsize=30.0)
ax2.set_xlim([14, 26])
ax2.set_ylim([0, Lz])
ax3.tick_params(labelsize=25)
ax3.set_xlabel('x [nm]', fontsize=30.0)

# RMS DISPLACEMENT #
rmsf_profile = 0.5*(np.sqrt(left_intf_msd)+np.flip(np.sqrt(right_intf_msd)))
ax3.plot( rmsf_profile, left_intf_mean[1,:] , 'k-', linewidth=3.0)
ax3.fill_betweenx(left_intf_mean[1,:], rmsf_profile-err_mean, rmsf_profile+err_mean , color='lightgrey', label=r'$\pm$ std err')
ax3.plot( [0.26763089802534887, 0.26763089802534887], [0,Lz], 'g:', linewidth=2.0, label=r'$l_{th}$')
ax3.plot( [0.3166, 0.3166], [0,Lz], 'r-.', linewidth=2.0, label=r'$\sigma_{SPC/E}$')
ax3.plot( [0.45, 0.45], [0,Lz], 'b--', linewidth=2.0, label=r'$d_{hex}$')
ax3.set_title('RMSF (avg. left&right)', fontsize=30.0)
ax3.tick_params(labelsize=25)
ax3.set_xlim([0.0, 0.75])
ax3.set_ylim([0, Lz])
ax3.set_xlabel('x [nm]', fontsize=30.0)
ax3.legend(fontsize=20.0)
plt.show()

fig, (ax3, ax4) = plt.subplots(2, 1)
# INTERFACE POSITION SIGNAL #
ax3.set_title('Time series', fontsize=30.0)
ax3.plot(time*(1e-3), X_int['ml'], 'm-', linewidth=1.5, label='middle')
ax3.plot(time*(1e-3), X_int['bl'], 'g-', linewidth=1.5, label='contact line')
ax3.plot(time*(1e-3), np.zeros(time.shape), 'k--', linewidth=1.5)
ax3.legend(fontsize=20.0)
ax3.tick_params(labelsize=25.0)
ax3.set_xlim([time[0]*(1e-3), time[-1]*(1e-3)])
# ax3.set_xlabel(r'$t$ [ps]', fontsize=20.0)
ax3.set_ylabel(r'$\Delta x$ [nm]', fontsize=30.0)
# INTERFACE POSITION ACF #
time = np.linspace(0.0, 0.5*dt*(n_fin-n_transient+1), len(ACF['ml']))
ax4.set_title('ACF', fontsize=30.0)
ax4.plot(time*(1e-3), ACF['ml']/np.var(X_int['ml']), 'm-', linewidth=2.0, label='middle')
ax4.plot(time*(1e-3), ACF['bl']/np.var(X_int['bl']), 'g-', linewidth=2.0, label='contact line')
ax4.plot(time*(1e-3), np.zeros(time.shape), 'k--', linewidth=1.5)
ax4.legend(fontsize=20.0)
ax4.tick_params(labelsize=25.0)
ax4.set_xlim([time[0]*(1e-3), time[-1]*(1e-3)])
ax4.set_xlabel(r'$t$ [ns]', fontsize=30.0)
# ax4.set_ylabel(r'$<x(0)x(t)>$ [nm^2]', fontsize=20.0)
ax4.set_ylabel(r'$\frac{<\Delta x(0)\Delta x(t)>}{<\Delta x(0)^2>}$ [1]', fontsize=30.0)
plt.show()

# Saving interface and c.l. series over time
save_folder = 'ContactLinesSignals/Q2Ca020'
FP.folder_name
for l in X_int.keys() :
    np.savetxt(save_folder+'/'+FP.folder_name+'_'+l+'.txt', X_int[l])