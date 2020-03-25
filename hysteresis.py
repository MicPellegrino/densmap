import densmap as dm
import numpy as np
import math

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

def contact_angle_from_density ( filename, smoother, density_th=2.0, h=1.85 ) :
    density_array = dm.read_density_file(filename, bin='y')
    smooth_density_array = dm.convolute(density_array, smoother)
    bulk_density = dm.detect_bulk_density(smooth_density_array, density_th)
    intf_contour = dm.detect_contour(smooth_density_array, 0.5*bulk_density, hx, hz)
    xc, zc, R, residue = dm.circle_fit(intf_contour, h)
    cot_circle = (h-zc)/np.sqrt(R*R-(h-zc)**2)
    theta_circle = np.rad2deg( -np.arctan( cot_circle )+0.5*math.pi )
    theta_circle = theta_circle + 180*(theta_circle<=0)
    return theta_circle

a = np.array([0.00, 0.75, 1.00, 1.25, 1.50, 1.75])

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
x = hx*np.arange(0.0,nx,1.0, dtype=float)
z = hz*np.arange(0.0,nz,1.0, dtype=float)
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

for k in range(1,6) :
    # files_sit.append(master_folder+folders_sit+'_w'+str(k)+'/flow_00125.dat')
    # files_rec.append(master_folder+folders_rec+'_w'+str(k)+'/flow_00375.dat')
    files_sit.append(master_folder+folders_sit+'_w'+str(k)+'/')
    files_rec.append(master_folder+folders_rec+'_w'+str(k)+'/')

theta_adv = []
theta_rec = []

n_max_adv = 125
n_max_rec = 375
n_mean = 10

for k in range(6):
    print('Obtaining contact angles '+str(k+1)+'/6')
    # Advancing angle
    # left_branch, right_branch, points_l, points_r = \
    #     dm.detect_contact_line(intf_contour, z_min=1.85, z_max=10.0, x_half=30.0)
    # foot_l, foot_r, theta_l, theta_r, cot_l, cot_r = \
    #     dm.detect_contact_angle(points_l, points_r, order=2)
    # theta_adv.append( 0.5*(theta_l+theta_r) )
    theta = 0
    for i in range(n_mean) :
        theta += contact_angle_from_density( \
            files_sit[k]+'flow_'+'{:05d}'.format(n_max_adv-i)+'.dat', smoother )
    theta /= n_mean
    theta_adv.append( theta )
    # Receiding angle
    # left_branch, right_branch, points_l, points_r = \
    #     dm.detect_contact_line(intf_contour, z_min=1.85, z_max=10.0, x_half=30.0)
    # foot_l, foot_r, theta_l, theta_r, cot_l, cot_r = \
    #     dm.detect_contact_angle(points_l, points_r, order=2)
    # theta_rec.append( 0.5*(theta_l+theta_r) )
    theta = 0
    for i in range(n_mean) :
        theta += contact_angle_from_density( \
            files_rec[k]+'flow_'+'{:05d}'.format(n_max_rec-i)+'.dat', smoother )
    theta /= n_mean
    theta_rec.append( theta )

theta_adv = np.array(theta_adv)
theta_rec = np.array(theta_rec)

plt.plot(a, theta_adv, 'bx--', linewidth=2, markersize=15, label='advancing')
plt.plot(a, theta_rec, 'rx--', linewidth=2, markersize=15, label='receding')
plt.plot(a, theta_adv-theta_rec, 'ko--', linewidth=2, markersize=10, label='difference')
plt.legend(fontsize=20.0)
plt.xlabel('a [nondim.]', fontsize=20.0)
plt.ylabel(r'$\theta_{eq}$ [deg]', fontsize=20.0)
plt.xticks(fontsize=20.0)
plt.yticks(fontsize=20.0)
plt.title('Contact angle hysterisis', fontsize=20.0)
plt.show()
