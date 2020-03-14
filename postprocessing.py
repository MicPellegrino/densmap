import densmap as dm
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as smg
import scipy.signal as sgn

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

# POWER SPECTRA
f_r, P_r = sgn.periodogram(radius, detrend='linear')
f_a, P_a = sgn.periodogram(angle, detrend='linear')

# f = f_a[1:-1]
# Pxx_den = P_a[1:-1]
# plt.semilogy(f, Pxx_den, 'k-')
# plt.xlabel('frequency [Hz]')
# plt.ylabel('PSD [V**2/Hz]')
# plt.title('Radius (P.S.)')
# plt.title('Angle (P.S.)')
# plt.xlim([min(f), max(f)])
# plt.show()

# FILTERING
sigma_r = 10.0
sigma_a = 5.66
radius_filtered = smg.gaussian_filter1d(radius, sigma=0.25*sigma_r)
angle_filtered = smg.gaussian_filter1d(angle, sigma=0.25*sigma_a)
hysteresis_filtered = smg.gaussian_filter1d(hysteresis, sigma=0.25*sigma_a)

time_cut = time[5:-5]
radius_cut = radius_filtered[5:-5]
time_rescale = time_cut/max(time_cut)
radius_rescale = radius_cut/max(radius_cut)
# r_10 = 1.25*radius_rescale[1]*time_rescale[10:-250]**(1.0/10.0)
# r_3 = 3.5*radius_rescale[1]*time_rescale[10:-250]**(1.0/3.0)
# r_2 = 3.5*radius_rescale[1]*time_rescale[10:-250]**(1.0/2.0)
r_10 = 0.6*radius_rescale[1]*time_rescale[10:-250]**(-1.0/10.0)
r_3 = 0.5*radius_rescale[1]*time_rescale[10:-250]**(-1.0/3.0)
r_2 = 0.25*radius_rescale[1]*time_rescale[10:-250]**(-1.0/2.0)
# plt.loglog(time_rescale[10:-250], r_10, 'r--', label=r'$\alpha=1/10$')
# plt.loglog(time_rescale[10:-250], r_3, 'b--', label=r'$\alpha=1/3$')
# plt.loglog(time_rescale[10:-250], r_2, 'g--', label=r'$\alpha=1/2$')
plt.loglog(time_rescale[10:-250], r_10, 'r--', label=r'$\alpha=-1/10$')
plt.loglog(time_rescale[10:-250], r_3, 'b--', label=r'$\alpha=-1/3$')
plt.loglog(time_rescale[10:-250], r_2, 'g--', label=r'$\alpha=-1/2$')
plt.loglog(time_rescale, radius_rescale, 'k-', linewidth=2.0, label='MD')
plt.legend(fontsize=20.0)
plt.title('Spreading radius (log-log)', fontsize=20.0)
plt.xlabel('t [nondim.]', fontsize=20.0)
plt.ylabel('r(t) [nondim.]', fontsize=20.0)
plt.show()

# COMPUTING VELOCITY
dt = time[1]-time[0]
velocity = np.zeros(radius_filtered.size)
velocity[1:-1] = (radius_filtered[2:]-radius_filtered[:-2])/(2.0*dt)
velocity[0] = (radius_filtered[1]-radius_filtered[0])/dt
velocity[-1] = (radius_filtered[-1]-radius_filtered[-2])/dt

plt.plot(time, velocity, 'k-')
plt.title('Spreading speed dr/dt', fontsize=20.0)
plt.xlabel('t [ps]', fontsize=20.0)
plt.ylabel('v(t) [nm/ps]', fontsize=20.0)
# plt.show()

x = velocity
y = angle
# y = hysteresis
plt.plot(velocity, angle, 'k.', markersize=10.0)
plt.plot([0.0, 0.0], [min(angle)-5, max(angle)+5], 'r--', linewidth=2.0)
# plt.quiver(x[:-1], y[:-1], x[1:]-x[:-1], y[1:]-y[:-1], scale_units='xy', angles='xy', scale=1, width=0.0015)
plt.title('Contact line speed vs contact angle', fontsize=20.0)
plt.xlabel('U [nm/ps]', fontsize=20.0)
plt.ylabel('theta [deg]', fontsize=20.0)
plt.ylim([min(angle)-5, max(angle)+5])
# plt.show()

plt.plot(velocity, hysteresis, 'k.', markersize=10.0)
plt.plot([0.0, 0.0], [min(hysteresis)-5, max(hysteresis)+5], 'r--', linewidth=2.0)
plt.title('Contact line speed vs hysteresis', fontsize=20.0)
plt.xlabel('U [nm/ps]', fontsize=20.0)
plt.ylabel('$\Delta\Theta$ [deg]', fontsize=20.0)
plt.ylim([min(hysteresis)-5, max(hysteresis)+5])
# plt.show()
