import densmap as dm
import matplotlib.pyplot as plt
import numpy as np

FP = dm.fitting_parameters( par_file='parameters_medium.txt' )
"""
# Medium
FP.time_step = 8.0
FP.lenght_x = 60.00000
FP.lenght_z = 35.37240
FP.r_mol = 0.09584
FP.max_vapour_density = 2.0
FP.substrate_location = 1.85
FP.bulk_location = 10.0
FP.simmetry_plane = 30.0
FP.interpolation_order = 2
# Large
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

# NB: conutour tracking should check whether there are actually kfin-kinit files!!!
# save_dir = 'Rec/Wave5/'
# CD = dm.contour_tracking('100nm/spreading', 1, 200, FP)
CD = dm.contour_tracking(FP.folder_name, FP.first_stamp, FP.last_stamp, FP)

CD.plot_radius()
CD.plot_angles()

dz = FP.dz
CD.movie_contour(FP.lenght_x, FP.lenght_z, dz)

# SAVING WHAT NEEDED
# spreading_radius = np.array(CD.foot_right)-np.array(CD.foot_left)
# mean_contact_angle = 0.5*(np.array(CD.angle_right)+np.array(CD.angle_left))
# hysteresis = np.array(CD.angle_right)-np.array(CD.angle_left)
t = np.array(CD.time)
spreading_radius = np.array(CD.spreading_radius)
mean_contact_angle = np.array(CD.mean_contact_angle)
hysteresis = np.array(CD.hysteresis)

# np.savetxt(save_dir+'time.txt', t)
# np.savetxt(save_dir+'radius_c.txt', spreading_radius)
# np.savetxt(save_dir+'angle_c.txt', mean_contact_angle)
# np.savetxt(save_dir+'difference.txt', hysteresis)
# np.savetxt(save_dir+'radius_r.txt', CD.radius_circle)
# np.savetxt(save_dir+'angle_r.txt', CD.angle_circle)
