import densmap as dm
import numpy as np
import math

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

# def contact_angle_from_density ( filename, smoother, density_th=2.0, h=1.85 ) :
#     density_array = dm.read_density_file(filename, bin='y')
#     smooth_density_array = dm.convolute(density_array, smoother)
#     bulk_density = dm.detect_bulk_density(smooth_density_array, density_th)
#     intf_contour = dm.detect_contour(smooth_density_array, 0.5*bulk_density, hx, hz)
#     xc, zc, R, residue = dm.circle_fit(intf_contour, h)
#     cot_circle = (h-zc)/np.sqrt(R*R-(h-zc)**2)
#     theta_circle = np.rad2deg( -np.arctan( cot_circle )+0.5*math.pi )
#     theta_circle = theta_circle + 180*(theta_circle<=0)
#     return theta_circle

def contact_angle_from_density ( filename, smoother, density_th=2.0, h=1.85 ) :
    _, theta = dm.equilibrium_from_density( filename, smoother, density_th, hx, hz, h )
    return theta

# a = np.array([0.00, 0.75, 1.00, 1.25, 1.50, 1.75])
a = np.array([0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75])

# PARAMETERS TO TUNE
Lx = 60.00000       # [nm]
Lz = 35.37240       # [nm]
r_h2o = 0.09584

# Detecting dimensions
dummy_file = '20nm/flow_sit_f/flow_00125.dat'
density_array = dm.read_density_file(dummy_file, bin='y')
Nx = density_array.shape[0]
Nz = density_array.shape[1]
hx = Lx/Nx
hz = Lz/Nz
nx = Nx
nz = Nz
x = hx*np.arange(0.0, nx, 1.0, dtype=float)
z = hz*np.arange(0.0, nz, 1.0, dtype=float)
X, Z = np.meshgrid(x, z, sparse=False, indexing='ij')

# Smoothing kernel once for all
smoother = dm.smooth_kernel(r_h2o, hx, hz)

master_folder = '20nm/'
folders_sit = 'flow_sit'
folders_rec = 'flow_rec'
files_sit = []
files_rec = []

# files_sit.append(master_folder+'flow_sit_f/flow_00125.dat')
# files_rec.append(master_folder+'flow_rec_f/flow_00375.dat')
files_sit.append(master_folder+'flow_sit_f/')
files_rec.append(master_folder+'flow_rec_f/')

for k in [2,1] :
    files_sit.append(master_folder+folders_sit+'_h'+str(k)+'/')
    files_rec.append(master_folder+folders_rec+'_h'+str(k)+'/')

for k in range(1,6) :
    # files_sit.append(master_folder+folders_sit+'_w'+str(k)+'/flow_00125.dat')
    # files_rec.append(master_folder+folders_rec+'_w'+str(k)+'/flow_00375.dat')
    files_sit.append(master_folder+folders_sit+'_w'+str(k)+'/')
    files_rec.append(master_folder+folders_rec+'_w'+str(k)+'/')

files_sit[0] = '20nm/flow_sit_f_thermo/'

theta_adv = []
theta_rec = []

"""
n_max_adv = 125
n_max_rec = 375
n_mean = 25
"""

n_max_adv = 750
n_max_adv_flat = 500
n_max_rec = 1000
n_mean = 30

for k in range(8):
    print('Obtaining contact angles '+str(k+1)+'/8')
    # Advancing angle
    theta = 0
    for i in range(n_mean) :
        if k == 0:
            theta += contact_angle_from_density( \
                files_sit[k]+'flow_'+'{:05d}'.format(n_max_adv_flat-i)+'.dat', smoother )
        else :
            theta += contact_angle_from_density( \
                files_sit[k]+'flow_'+'{:05d}'.format(n_max_adv-i)+'.dat', smoother )
    theta /= n_mean
    theta_adv.append( theta )
    # Receiding angle
    theta = 0
    for i in range(n_mean) :
        theta += contact_angle_from_density( \
            files_rec[k]+'flow_'+'{:05d}'.format(n_max_rec-i)+'.dat', smoother )
    theta /= n_mean
    theta_rec.append( theta )

theta_adv = np.array(theta_adv)
theta_rec = np.array(theta_rec)

# EXTRA DATA FROM LONGER RUNS

"""

a_extra = np.array([0.00, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75])
theta_adv_extra = np.zeros(len(a_extra))
theta_rec_extra = np.zeros(len(a_extra))

n_max_adv_extra = 750
n_max_rec_extra = 1000
n_mean = 50

print('Obtaining extra contact angles: W1')
# Advancing angle EXT W1
theta = 0
for i in range(n_mean) :
    theta += contact_angle_from_density( \
        '20nm/flow_sit_w1/'+'flow_'+'{:05d}'.format(n_max_adv_extra-i)+'.dat', smoother )
theta /= n_mean
theta_adv_extra[-5] = theta
# Receiding angle EXT W1
theta = 0
for i in range(n_mean) :
    theta += contact_angle_from_density( \
        '20nm/flow_rec_w1/'+'flow_'+'{:05d}'.format(n_max_rec_extra-i)+'.dat', smoother )
theta /= n_mean
theta_rec_extra[-5] = theta

print('Obtaining extra contact angles: W2')
# Advancing angle EXT W2
theta = 0
for i in range(n_mean) :
    theta += contact_angle_from_density( \
        '20nm/flow_sit_w2/'+'flow_'+'{:05d}'.format(n_max_adv_extra-i)+'.dat', smoother )
theta /= n_mean
theta_adv_extra[-4] = theta
# Receiding angle EXT W2
theta = 0
for i in range(n_mean) :
    theta += contact_angle_from_density( \
        '20nm/flow_rec_w2/'+'flow_'+'{:05d}'.format(n_max_rec_extra-i)+'.dat', smoother )
theta /= n_mean
theta_rec_extra[-4] = theta

print('Obtaining extra contact angles: W3')
# Advancing angle EXT W3
theta = 0
for i in range(n_mean) :
    theta += contact_angle_from_density( \
        '20nm/flow_sit_w3/'+'flow_'+'{:05d}'.format(n_max_adv_extra-i)+'.dat', smoother )
theta /= n_mean
theta_adv_extra[-3] = theta
# Receiding angle EXT W3
# theta = np.NaN
theta = 0
for i in range(n_mean) :
    theta += contact_angle_from_density( \
        '20nm/flow_rec_w3/'+'flow_'+'{:05d}'.format(n_max_rec_extra-i)+'.dat', smoother )
theta /= n_mean
theta_rec_extra[-3] = theta

print('Obtaining extra contact angles: W4')
# Advancing angle EXT W4
theta = 0
for i in range(n_mean) :
    theta += contact_angle_from_density( \
        '20nm/flow_sit_w4/'+'flow_'+'{:05d}'.format(n_max_adv_extra-i)+'.dat', smoother )
theta /= n_mean
theta_adv_extra[-2] = theta
# Receiding angle EXT W4
theta = 0
for i in range(n_mean) :
    theta += contact_angle_from_density( \
        '20nm/flow_rec_w4/'+'flow_'+'{:05d}'.format(n_max_rec_extra-i)+'.dat', smoother )
theta /= n_mean
theta_rec_extra[-2] = theta

print('Obtaining extra contact angles: W5')
# Advancing angle EXT W5
theta = 0
for i in range(n_mean) :
    theta += contact_angle_from_density( \
        '20nm/flow_sit_w5/'+'flow_'+'{:05d}'.format(n_max_adv_extra-i)+'.dat', smoother )
theta /= n_mean
theta_adv_extra[-1] = theta
# Receiding angle EXT W5
theta = 0
for i in range(n_mean) :
    theta += contact_angle_from_density( \
        '20nm/flow_rec_w5/'+'flow_'+'{:05d}'.format(n_max_rec_extra-i)+'.dat', smoother )
theta /= n_mean
theta_rec_extra[-1] = theta

# Different heights
print('Obtaining extra contact angles: H1')
# Advancing angle EXT H1
theta = 0
for i in range(n_mean) :
    theta += contact_angle_from_density( \
        '20nm/flow_sit_h1/'+'flow_'+'{:05d}'.format(n_max_adv_extra-i)+'.dat', smoother )
theta /= n_mean
theta_adv_extra[1] = theta
# Receiding angle EXT H1
theta = 0
for i in range(n_mean) :
    theta += contact_angle_from_density( \
        '20nm/flow_rec_h1/'+'flow_'+'{:05d}'.format(n_max_rec_extra-i)+'.dat', smoother )
theta /= n_mean
theta_rec_extra[1] = theta

n_max_adv_extra = 500
n_max_rec_extra = 1000

print('Obtaining extra contact angles: FLAT')
# Advancing angle EXT FLAT
theta = 0
for i in range(n_mean) :
    theta += contact_angle_from_density( \
        '20nm/flow_sit_f_thermo/'+'flow_'+'{:05d}'.format(n_max_adv_extra-i)+'.dat', smoother )
theta /= n_mean
theta_adv_extra[0] = theta
# Receiding angle EXT FLAT
theta = 0
for i in range(n_mean) :
    theta += contact_angle_from_density( \
        '20nm/flow_rec_f/'+'flow_'+'{:05d}'.format(n_max_rec_extra-i)+'.dat', smoother )
theta /= n_mean
theta_rec_extra[0] = theta

"""

plt.plot(a, theta_adv, 'b^--', linewidth=1.25, markersize=17.5, label='advancing')
plt.plot(a, theta_rec, 'rv--', linewidth=1.25, markersize=17.5, label='receding')
plt.plot(a, theta_adv-theta_rec, 'ks--', linewidth=1.25, markersize=17.5, label='difference')
"""
plt.plot(a_extra, theta_adv_extra, 'bP', linewidth=2, markersize=15)
plt.plot(a_extra, theta_rec_extra, 'rP', linewidth=2, markersize=15)
plt.plot(a_extra, theta_adv_extra-theta_rec_extra, 'kP', linewidth=2, markersize=15)
"""
plt.legend(fontsize=20.0)
plt.xlabel('a [nondim.]', fontsize=30.0)
plt.ylabel(r'$\theta$ [deg]', fontsize=30.0)
plt.xticks(fontsize=30.0)
plt.yticks(fontsize=30.0)
plt.title(r'Contact angle difference (@8.0ns, $\theta_0=110$deg)', fontsize=30.0)
plt.show()
