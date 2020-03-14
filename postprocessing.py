import densmap as dm
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as smg

time = []
with open('time.txt', 'r') as f:
    for line in f:
        time.append(float(line.split()[0]))
radius = []
with open('radius.txt', 'r') as f:
    for line in f:
        radius.append(float(line.split()[0]))
angle = []
with open('angle.txt', 'r') as f:
    for line in f:
        angle.append(float(line.split()[0]))
hysteresis = []
with open('hysteresis.txt', 'r') as f:
    for line in f:
        hysteresis.append(float(line.split()[0]))

time = np.array(time)
radius = np.array(radius)
angle = np.array(angle)
hysteresis = np.array(hysteresis)

radius_filtered = smg.gaussian_filter1d(radius, sigma=5.0)
angle_filtered = smg.gaussian_filter1d(angle, sigma=5.0)
hysteresis_filtered = smg.gaussian_filter1d(hysteresis, sigma=5.0)

# plt.plot(time, radius_filtered)
# plt.plot(time, angle_filtered)
# plt.plot(time, hysteresis_filtered)
# plt.show()

dt = time[1]-time[0]
velocity = np.zeros(radius_filtered.size)
velocity[1:-1] = (radius_filtered[2:]-radius_filtered[:-2])/(2.0*dt)
velocity[0] = (radius_filtered[1]-radius_filtered[0])/dt
velocity[-1] = (radius_filtered[-1]-radius_filtered[-2])/dt

# plt.plot(time, radius_filtered)
# plt.plot(time, angle_filtered)
# plt.plot(time, hysteresis_filtered)
# plt.plot(time, velocity)
x = velocity
# y = angle
y = hysteresis
plt.plot(x, y, 'k.', markersize=10.0)
# plt.quiver(x[:-1], y[:-1], x[1:]-x[:-1], y[1:]-y[:-1], scale_units='xy', angles='xy', scale=1, width=0.0015)
# plt.title('Contact line speed vs contact angle', fontsize=20.0)
plt.title('Contact line speed vs hysteresis', fontsize=20.0)
plt.xlabel('U [nm/ps]', fontsize=20.0)
# plt.ylabel('theta [deg]', fontsize=20.0)
plt.ylabel('$|\Delta\Theta|$ [deg]', fontsize=20.0)
plt.show()
