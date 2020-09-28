import densmap as dm
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as smg
import scipy.signal as sgn

def array_from_file( filename ):
    my_list = []
    with open(filename, 'r') as f:
        for line in f:
            my_list.append(float(line.split()[0]))
    return np.array(my_list)

folder_name = 'SpreadingData/H2Q4'
t_in = 400.0

time = array_from_file(folder_name+'/time.txt')
n_ini = np.argmin(np.abs(time-t_in))
time = time[n_ini:-1]
radius = array_from_file(folder_name+'/radius_fit.txt')[n_ini:-1]
foot_l = array_from_file(folder_name+'/foot_l.txt')[n_ini:-1]
foot_r = array_from_file(folder_name+'/foot_r.txt')[n_ini:-1]
angle_l = array_from_file(folder_name+'/angle_l.txt')[n_ini:-1]
angle_r = array_from_file(folder_name+'/angle_r.txt')[n_ini:-1]
angle = 0.5*(angle_l+angle_r)
hysteresis = np.abs(angle_l-angle_r)

plt.plot(time-t_in, foot_l, 'b-', linewidth=3.0, label='left')
plt.plot(time-t_in, foot_r, 'r-', linewidth=3.0, label='right')
plt.xlim([time[0]-t_in,time[-1]-t_in])
plt.xlabel('time [ps]', fontsize=30.0)
plt.ylabel('distance [nm]',fontsize=30.0)
plt.legend(fontsize=30.0)
plt.xticks(fontsize=30.0)
plt.yticks(fontsize=30.0)
plt.title('CL position', fontsize=30.0)
plt.show()
plt.show()

"""
plt.semilogx(time-t_in, foot_r-foot_l, 'k-', linewidth=3.5, label='Spreading radius')
plt.semilogx(time-t_in, radius, 'k--', linewidth=3.5, label='LS fit')
plt.xlim([time[1]-t_in,time[-1]-t_in])
plt.xlabel('time [ps]', fontsize=30.0)
plt.ylabel('distance [nm]',fontsize=30.0)
plt.legend(fontsize=30.0)
plt.xticks(fontsize=30.0)
plt.yticks(fontsize=30.0)
plt.title('Spreading radius (semilog)', fontsize=30.0)
plt.show()
"""

"""
    TRY TO ADD IN THE SAME PLOT BOTH THE ADVANCING AND THE RECEDING CURVES
    SIMILAR PLOT AS IN ROLLEY GUTHMANN, 2007
"""

"""
    FILTERING
"""
"""
# POWER SPECTRA
f_r, P_r = sgn.periodogram(radius, detrend='linear')
f_a, P_a = sgn.periodogram(angle, detrend='linear')

f = f_a[1:-1]
Pxx_den = P_a[1:-1]
plt.semilogy(f, Pxx_den, 'k-')
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.title('Radius (P.S.)')
plt.title('Angle (P.S.)')
plt.xlim([min(f), max(f)])
plt.show()

# FILTERING
# Manually tuned, but this should actually be dictated by thermal fluctuations
sigma_r = 5.0
sigma_a = 5.0
radius_filtered = smg.gaussian_filter1d(radius, sigma=sigma_r)
angle_filtered = smg.gaussian_filter1d(angle, sigma=sigma_a)
hysteresis_filtered = smg.gaussian_filter1d(hysteresis, sigma=sigma_a)

plt.plot(time, angle_filtered, 'k-')
plt.title('Spreading angle', fontsize=20.0)
plt.xlabel('t [ps]', fontsize=20.0)
plt.ylabel('theta [deg]', fontsize=20.0)
plt.show()

# COMPUTING VELOCITY
dt = time[1]-time[0]
velocity = np.zeros(radius_filtered.size)
velocity[1:-1] = (radius_filtered[2:]-radius_filtered[:-2])/(2.0*dt)
velocity[0] = (radius_filtered[1]-radius_filtered[0])/dt
velocity[-1] = (radius_filtered[-1]-radius_filtered[-2])/dt

plt.plot(time, velocity, 'k-')
plt.plot(time, np.zeros(len(time)), 'r--')
plt.title('Spreading speed dr/dt', fontsize=20.0)
plt.xlabel('t [ps]', fontsize=20.0)
plt.ylabel('v(t) [nm/ps]', fontsize=20.0)
plt.show()

N = len(time)
# x = np.cos(np.deg2rad(angle))
x = np.cos(np.deg2rad(angle_filtered))
# y = np.log(np.abs(velocity))
y = velocity

plt.plot(x[int(N/2):], y[int(N/2):], 'k.', markersize=10.0)
# plt.plot(x[:int(N/2)], y[:int(N/2)], 'b.', markersize=10.0)
plt.plot([-1.0, 1.0], [0.0, 0.0], 'r--', linewidth=2.0)
plt.xlim([-1.0,1.0])
plt.title('Contact line speed vs contact angle', fontsize=20.0)
plt.ylabel('U [nm/ps]', fontsize=20.0)
plt.xlabel(r'$-\cos(\theta_d)$ [nondim.]', fontsize=20.0)
plt.show()

plt.plot(velocity, hysteresis, 'k.', markersize=10.0)
plt.plot([0.0, 0.0], [min(hysteresis)-5, max(hysteresis)+5], 'r--', linewidth=2.0)
plt.title('Contact line speed vs hysteresis', fontsize=20.0)
plt.xlabel('U [nm/ps]', fontsize=20.0)
plt.ylabel('$\Delta\Theta$ [deg]', fontsize=20.0)
plt.ylim([min(hysteresis)-5, max(hysteresis)+5])
plt.show()
"""

# SPREADING LAW
"""
time_red = time/np.max(time)
radius_red = radius/np.max(radius)
r_10 = time**(1.0/10.0)
r_3 = time**(1.0/3.0)
r_2 = time**(1.0/2.0)
plt.semilogx(time, r_10, 'r--', label=r'$\alpha=1/10$')
plt.semilogx(time, r_3, 'b--', label=r'$\alpha=1/3$')
plt.semilogx(time, r_2, 'g--', label=r'$\alpha=1/2$')
plt.semilogx(time, radius, 'k-', linewidth=2.0, label='MD')
plt.legend(fontsize=20.0)
plt.title('Spreading radius (log-log)', fontsize=20.0)
plt.xlabel('t [nondim.]', fontsize=20.0)
plt.ylabel('r(t) [nondim.]', fontsize=20.0)
plt.show()
"""
