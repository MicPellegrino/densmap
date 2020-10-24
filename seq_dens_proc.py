import densmap as dm
import matplotlib.pyplot as plt
import numpy as np

FP = dm.fitting_parameters( par_file='/home/michele/densmap/ShearChar/parameters_shear.txt' )

# NB: contour tracking should check whether there are actually kfin-kinit files!!!
CD = dm.shear_tracking(FP.folder_name, FP.first_stamp, FP.last_stamp, FP, \
    file_root = '/flow_', contact_line = True)

# Testing xmgrace output
# CD.save_xvg('InterfaceTest')

# Testing cl distribution binning
"""
signal = np.array(CD.foot['tr'])[:,0]
N = len(signal)
N_in = 300
sign_mean, sign_std, bin_vector, distribution = \
        dm.position_distribution( signal[N_in:], int(np.sqrt(N-N_in)) )
CD.plot_contact_line_pdf(N_in=300)
"""
# plt.step(bin_vector, distribution)
# plt.show()

# Testing plot
CD.plot_radius()
CD.plot_angles()

# Movie
dz = FP.dz
CD.movie_contour(FP.lenght_x, FP.lenght_z, dz,  circle=False, contact_line = True)

# SAVING WHAT NEEDED
# Droplet
"""
spreading_radius = np.array(CD.foot_right)-np.array(CD.foot_left)
mean_contact_angle = 0.5*(np.array(CD.angle_right)+np.array(CD.angle_left))
hysteresis = np.array(CD.angle_right)-np.array(CD.angle_left)
"""
# Shear
CD.save_to_file('/home/michele/densmap/ShearChar/LJ')

"""
t = np.array(CD.time)
spreading_radius = np.array(CD.spreading_radius)[:,0]
mean_contact_angle = np.array(CD.mean_contact_angle)
hysteresis = np.array(CD.hysteresis)

save_dir = 'Adv/Wave1/'
np.savetxt(save_dir+'time.txt', t)
np.savetxt(save_dir+'radius_c.txt', spreading_radius)
np.savetxt(save_dir+'angle_c.txt', mean_contact_angle)
np.savetxt(save_dir+'difference.txt', hysteresis)
np.savetxt(save_dir+'radius_r.txt', CD.radius_circle)
np.savetxt(save_dir+'angle_r.txt', CD.angle_circle)
"""

 ############
#### MISC ####
 ############

# Prototype parameters medium droplet
"""
FP.time_step = 8.0
FP.lenght_x = 60.00000
FP.lenght_z = 35.37240
FP.r_mol = 0.09584
FP.max_vapour_density = 2.0
FP.substrate_location = 1.85
FP.bulk_location = 10.0
FP.simmetry_plane = 30.0
FP.interpolation_order = 2
"""
# Prototype parameters large droplet
"""
FP.time_step = 20.0
FP.lenght_x = 300.00000
FP.lenght_z = 200.44360
FP.r_mol = 0.09584
FP.max_vapour_density = 2.0
FP.substrate_location = 5.5
FP.bulk_location = 25.0
FP.simmetry_plane = 150.0
FP.interpolation_order = 1
"""

# Inline plotting on subplots
"""
fig, axs = plt.subplots(1, 2)
axs[0].plot(CD.time, CD.spreading_radius[:,0], 'k-', linewidth=2.5, label='contour')
axs[0].plot(CD.time, CD.radius_circle, 'g-', linewidth=2.5, label='cap')
axs[0].set_xlabel('t [ps]', fontsize=20.0)
axs[0].set_ylabel(r'$R(t)$ [nm]', fontsize=20.0)
axs[0].tick_params(axis='x', labelsize=20.0)
axs[0].tick_params(axis='y', labelsize=20.0)
axs[0].legend(fontsize=20.0)
axs[0].set_title('Spreading radius', fontsize=20.0)
axs[1].plot(CD.time, CD.mean_contact_angle, 'b-', linewidth=2.0, label='average')
axs[1].plot(CD.time, CD.hysteresis, 'r-', linewidth=2.0, label='difference')
axs[1].plot(CD.time, CD.angle_circle, 'g-', linewidth=2.5, label='cap')
axs[1].set_xlabel('t [ps]', fontsize=20.0)
axs[1].set_ylabel(r'$\theta(t)$ [deg]', fontsize=20.0)
axs[1].tick_params(axis='x', labelsize=20.0)
axs[1].tick_params(axis='y', labelsize=20.0)
axs[1].legend(fontsize=20.0)
axs[1].set_title('Contact angle', fontsize=20.0)
plt.show()
"""
