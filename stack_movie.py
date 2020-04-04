import densmap as dm
import matplotlib.pyplot as plt
import numpy as np

FP_1 = dm.fitting_parameters( par_file='parameters1.txt' )
FP_2 = dm.fitting_parameters( par_file='parameters2.txt' )

FP_3 = dm.fitting_parameters( par_file='parameters.txt' )

CD = dm.contour_tracking(FP_1.folder_name, FP_1.first_stamp, \
    FP_1.last_stamp, FP_1)
CD_2 = dm.contour_tracking(FP_2.folder_name, FP_2.first_stamp, \
    FP_2.last_stamp, FP_2)
CD_3 = dm.contour_tracking(FP_3.folder_name, FP_3.first_stamp, \
    FP_3.last_stamp, FP_3)

CD.merge(CD_2)
CD.merge(CD_3)

CD.plot_radius()
CD.plot_angles()
dz = FP_1.dz
CD.movie_contour(FP_1.lenght_x, FP_1.lenght_z, dz)

# SAVING WHAT NEEDED
spreading_radius = np.array(CD.foot_right)-np.array(CD.foot_left)
mean_contact_angle = 0.5*(np.array(CD.angle_right)+np.array(CD.angle_left))
hysteresis = np.array(CD.angle_right)-np.array(CD.angle_left)
t = np.array(CD.time)
spreading_radius = np.array(CD.spreading_radius)
mean_contact_angle = np.array(CD.mean_contact_angle)
hysteresis = np.array(CD.hysteresis)
save_dir = 'Large/'
np.savetxt(save_dir+'time.txt', t)
np.savetxt(save_dir+'radius_c.txt', spreading_radius)
np.savetxt(save_dir+'angle_c.txt', mean_contact_angle)
np.savetxt(save_dir+'difference.txt', hysteresis)
np.savetxt(save_dir+'radius_r.txt', CD.radius_circle)
np.savetxt(save_dir+'angle_r.txt', CD.angle_circle)
