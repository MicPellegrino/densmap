import densmap as dm
import matplotlib.pyplot as plt
import numpy as np

FP = dm.fitting_parameters( par_file='parameters_droplet.txt' )

# NB: contour tracking should check whether there are actually kfin-kinit files!!!
CD = dm.droplet_tracking(FP.folder_name, FP.first_stamp, FP.last_stamp, FP, \
    file_root = '/flow_', contact_line = True)

# Testing plot
CD.plot_radius()
CD.plot_angles()

# Movie
dz = FP.dz
CD.movie_contour(FP.lenght_x, FP.lenght_z, dz,  circle=True, contact_line = True)

# SAVING WHAT NEEDED
# Droplet
"""
spreading_radius = np.array(CD.foot_right)-np.array(CD.foot_left)
mean_contact_angle = 0.5*(np.array(CD.angle_right)+np.array(CD.angle_left))
hysteresis = np.array(CD.angle_right)-np.array(CD.angle_left)
"""
# Shear
CD.save_to_file('SpreadingData/FlatQ3')

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
