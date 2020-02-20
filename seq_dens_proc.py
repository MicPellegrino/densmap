import densmap as dm
import matplotlib.pyplot as plt
import numpy as np

FP = dm.fitting_parameters()
FP.time_step = 2.0
FP.lenght_x = 23.86485
FP.lenght_z = 17.68620
FP.r_mol = 0.09584
FP.max_vapour_density = 2.0
FP.substrate_location = 2.0
FP.bulk_location = 5.0
FP.simmetry_plane = 12.0
FP.interpolation_order = 1

# NB: conutour tracking should check whether there are actually kfin-kinit files!!!
CD = dm.contour_tracking('flow_data4', 50, 400, FP)

# CD.plot_radius()
# CD.plot_angles()

dz = 3.0
rad = 1.0
CD.movie_contour(FP.lenght_x, FP.lenght_z, dz, rad)

# spreading_radius = np.array(CD.foot_right)-np.array(CD.foot_left)
# mean_contact_angle = 0.5*(np.array(CD.angle_right)+np.array(CD.angle_left))
# hysteresis = np.absolute(np.array(CD.angle_right)-np.array(CD.angle_left))
# t = np.array(CD.time)

# plt.plot(t, spreading_radius, 'k-')
# plt.title('Spreading radius')
# plt.xlabel('time [ps]')
# plt.ylabel('r(t) [nm]')
# plt.show()

# plt.plot(t, mean_contact_angle, 'b-', label='average')
# plt.plot(t, hysteresis, 'r-', label='hysterisis')
# plt.title('Contact angle')
# plt.xlabel('t [ps]')
# plt.ylabel('theta(t) [deg]')
# plt.legend()
# plt.show()
