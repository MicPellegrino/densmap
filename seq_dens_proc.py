import densmap as dm
import matplotlib.pyplot as plt
import numpy as np

FP = dm.fitting_parameters()
FP.time_step = 8.0
FP.lenght_x = 60.00000
FP.lenght_z = 35.37240
FP.r_mol = 0.09584
FP.max_vapour_density = 2.0
FP.substrate_location = 1.85
FP.bulk_location = 10.0
FP.simmetry_plane = 30.0
FP.interpolation_order = 2

# NB: conutour tracking should check whether there are actually kfin-kinit files!!!
CD = dm.contour_tracking('flow_20nm_rec', 1, 375, FP)

CD.plot_radius()
CD.plot_angles()

dz = 2.5
rad = 1.0
CD.movie_contour(FP.lenght_x, FP.lenght_z, dz, rad)

# Saving what we need
# spreading_radius = np.array(CD.foot_right)-np.array(CD.foot_left)
# mean_contact_angle = 0.5*(np.array(CD.angle_right)+np.array(CD.angle_left))
# hysteresis = np.array(CD.angle_right)-np.array(CD.angle_left)
t = np.array(CD.time)
spreading_radius = np.array(CD.spreading_radius)
mean_contact_angle = np.array(CD.mean_contact_angle)
hysteresis = np.array(CD.hysteresis)
np.savetxt('time.txt', t)
np.savetxt('radius.txt', spreading_radius)
np.savetxt('angle.txt', mean_contact_angle)
np.savetxt('hysteresis.txt', hysteresis)

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
