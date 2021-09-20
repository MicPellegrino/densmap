import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

import scipy.ndimage as smg
import scipy.signal as sgn
import scipy.optimize as opt

def array_from_file( filename ):
    my_list = []
    with open(filename, 'r') as f:
        for line in f:
            my_list.append(float(line.split()[0]))
    return np.array(my_list)

def cos( theta ) :
    return np.cos(np.deg2rad(theta))
cos_vec = np.vectorize(cos)

def sin( theta ) :
    return np.sin(np.deg2rad(theta))
sin_vec = np.vectorize(sin)

# Plotting params
plot_sampling = 15
plot_tcksize = 17.5

# Reference units
mu = 0.877                  # [mPa*s]
gamma = 57.8                # [mPa*m]
U_ref = (gamma/mu)*1e-3     # [nm/ps]

# Time window
T_ini = 100.0               # [ps]
T_fin = 24310.0             # [ps]

# Minimum CL advancement velocity threshold (rough estimate, a bit arbitrary)
vmin = 0.0001                # [nm/ps]

# Rolling average
delta_t_avg = 600           # [ps]

# Obtaining the signal from saved .txt files
folder_name = 'SpreadingData/FlatQ3ADV'
time = array_from_file(folder_name+'/time.txt')
foot_l = array_from_file(folder_name+'/foot_l.txt')
foot_r = array_from_file(folder_name+'/foot_r.txt')
angle_l = array_from_file(folder_name+'/angle_l.txt')
angle_r = array_from_file(folder_name+'/angle_r.txt')
radius = array_from_file(folder_name+'/radius_fit.txt')
angle_circle = array_from_file(folder_name+'/angle_fit.txt')

# Cutoff inertial phase
dt = time[1]-time[0]
time_window = T_fin-T_ini
print("Time window = "+str(time_window))
print("Time step   = "+str(dt))
N_ini = int( T_ini / dt )
N_fin = int( T_fin / dt )
time = time[N_ini:N_fin]
foot_l = foot_l[N_ini:N_fin]
foot_r = foot_r[N_ini:N_fin]
angle_l = angle_l[N_ini:N_fin]
angle_r = angle_r[N_ini:N_fin]
radius = radius[N_ini:N_fin]
angle_circle = angle_circle[N_ini:N_fin]

# Averaging between laft and right c.l.
contact_line_pos = 0.5*(foot_r-foot_l)
# contact_line_pos = 0.5*radius
contact_angle = 0.5*(angle_r+angle_l)

# Rational polynomial fit
def rational(x, p, q):
    return np.polyval(p, x) / np.polyval(q + [1.0], x)

def rational_3_3(x, p0, p1, p2, q1, q2):
    return rational(x, [p0, p1, p2], [q1, q2])

def rational_4_2(x, p0, p1, p2, p4, q1):
    return rational(x, [p0, p1, p2, p4], [q1,])

# popt, pcov = opt.curve_fit(rational_3_3, time, contact_line_pos)
popt, pcov = opt.curve_fit(rational_4_2, time, contact_line_pos)

# Velocity from raw data (very noisy)
velocity_l = np.zeros(len(foot_l))
velocity_r = np.zeros(len(foot_r))
velocity_l[1:-1] = -0.5 * np.subtract(foot_l[2:],foot_l[0:-2]) / dt
velocity_r[1:-1] = 0.5 * np.subtract(foot_r[2:],foot_r[0:-2]) / dt
velocity_l[0] = -( foot_l[1]-foot_l[0] ) / dt
velocity_r[0] = ( foot_r[1]-foot_r[0] ) / dt
velocity_l[-1] = -( foot_l[-1]-foot_l[-2] ) / dt
velocity_r[-1] = ( foot_r[-1]-foot_r[-2] ) / dt

# Velocity from rational approximation (3,3)
# p_0 = np.array(popt[0:3])
# p_1 = np.polyder(p_0, m=1)
# q_0 = np.concatenate((popt[3:5],[1.0]))
# q_1 = np.polyder(q_0, m=1)
# def velocity_fit(t) :
#     num = ( np.polyval(p_1,t)*np.polyval(q_0,t) - np.polyval(p_0,t)*np.polyval(q_1,t) )
#     den = ( np.polyval(q_0,t) )**2
#     return num/den
# velocity_fit = velocity_fit(time)

# Velocity from rational approximation (4,2)
p_0 = np.array(popt[0:4])
p_1 = np.polyder(p_0, m=1)
q_0 = np.concatenate((popt[4:5],[1.0]))
q_1 = np.polyder(q_0, m=1)
def velocity_fit(t) :
    num = ( np.polyval(p_1,t)*np.polyval(q_0,t) - np.polyval(p_0,t)*np.polyval(q_1,t) )
    den = ( np.polyval(q_0,t) )**2
    return num/den
velocity_fit = velocity_fit(time)

time_ns = time*1e-3

fig1, (ax1, ax2) = plt.subplots(1, 2)

ax1.set_title('Spreading branches', fontsize=25.0)
ax1.plot(time_ns, foot_l, 'b-', linewidth=2.0, label='left (raw)')
ax1.plot(time_ns, foot_r, 'r-', linewidth=2.0, label='right (raw)')
ax1.plot(time_ns, contact_line_pos, 'k-', linewidth=2.0, label=r'1/2$\cdot$(right-left)')
# ax1.plot(time_ns, rational_3_3(time, *popt), 'k--', linewidth=3.5, label='r. p. fit')
ax1.plot(time_ns, rational_4_2(time, *popt), 'k--', linewidth=3.5, label='r. p. fit')
ax1.set_xlabel('t [ns]', fontsize=30.0)
ax1.set_ylabel('x [nm]', fontsize=30.0)
ax1.set_ylim([0.0, 175.0])
ax1.set_xlim([time_ns[0], time_ns[-1]])
ax1.legend(fontsize=20.0)
ax1.tick_params(axis='x', labelsize=plot_tcksize)
ax1.tick_params(axis='y', labelsize=plot_tcksize)

# Spreading curves
ax2.loglog(time_ns, contact_line_pos, 'k-', linewidth=3.0, label='position')
ax2.loglog(time_ns[200:], 50*time_ns[200:]**(1/10), 'm--', linewidth=3.0, label=r'Tanner$\sim t^{1/10}$')
ax2.loglog(time_ns[0:75], 37*time_ns[0:75]**(1/2), 'g-.', linewidth=3.0, label=r'inertial$\sim t^{1/2}$')
ax2.set_xlabel('t [ns]', fontsize=30.0)
ax2.set_ylabel('x [nm]', fontsize=30.0)
ax2.legend(fontsize=20.0)

plt.show()

N_avg = int(delta_t_avg/dt)
contact_angle = np.convolve(contact_angle, np.ones(N_avg)/N_avg, mode='same')

fig3, (ax5, ax6) = plt.subplots(1, 2)

# Plot contact angles and wetting speed

idx_steady = np.argmin(np.abs(velocity_fit-vmin))
t_steady = dt*idx_steady
idx_zero = np.argmin(np.abs(velocity_fit))
t_zero = dt*idx_zero

ax5.set_title('Contact line speed', fontsize=25.0)
ax5.plot(time_ns[0::plot_sampling], 1e3*velocity_l[0::plot_sampling], 'bs', markersize=8,  markerfacecolor="None", label='left (raw)')
ax5.plot(time_ns[0::plot_sampling], 1e3*velocity_r[0::plot_sampling], 'rd', markersize=10, markerfacecolor="None", label='right (raw)')
ax5.plot(time_ns[0::plot_sampling], 1e3*vmin*np.ones(len(time_ns[0::plot_sampling])), 'k-', linewidth=1.75)
ax5.plot(time_ns[0::plot_sampling], np.zeros(len(time_ns[0::plot_sampling])), 'k-', linewidth=1.75)
ax5.plot([1e-3*t_steady, 1e-3*t_steady], [1e3*min(velocity_l), 1e3*max(velocity_l)] , 'k-', linewidth=1.75)
ax5.plot([1e-3*t_zero, 1e-3*t_zero], [1e3*min(velocity_l), 1e3*max(velocity_l)] , 'k-', linewidth=1.75)
ax5.plot(time_ns, 1e3*velocity_fit, 'k--',  linewidth=3.5, label='r. p. derivative')
ax5.set_xlabel('t [ns]', fontsize=30.0)
ax5.set_ylabel('dx/dt [nm/ns]', fontsize=30.0)
ax5.set_xlim([time_ns[0], time_ns[-1]])
ax5.legend(fontsize=20.0)
ax5.tick_params(axis='x', labelsize=plot_tcksize)
ax5.tick_params(axis='y', labelsize=plot_tcksize)

ax6.set_title('Dynamic contact angle', fontsize=25.0)
# plt.plot(time, angle_l, 'b-', linewidth=1.0, label='left (raw)')
# plt.plot(time, angle_r, 'r-', linewidth=1.0, label='right (raw)')
ax6.plot(time_ns[0::plot_sampling], angle_l[0::plot_sampling], 'bs', markersize=8, markeredgewidth=1.0, markerfacecolor="None", label='left (raw)')
ax6.plot(time_ns[0::plot_sampling], angle_r[0::plot_sampling], 'rd', markersize=10, markeredgewidth=1.0, markerfacecolor="None", label='right (raw)')
ax6.plot(time_ns, angle_circle, 'g-', linewidth=3.5, label='circle')
ax6.plot(time_ns[N_avg:-N_avg], contact_angle[N_avg:-N_avg], 'k--', linewidth=3.5, label='rolling average')
ax6.set_xlabel('t [ns]', fontsize=30.0)
ax6.set_ylabel(r'$\theta$ [deg]', fontsize=30.0)
ax6.set_xlim([time_ns[0], time_ns[-1]])
ax6.legend(fontsize=20.0)
ax6.tick_params(axis='x', labelsize=plot_tcksize)
ax6.tick_params(axis='y', labelsize=plot_tcksize)

plt.show()

# Rescale speed and remove part that is steady
# velocity_fit_red = velocity_fit[N_avg:-N_avg]/U_ref
velocity_fit_red = velocity_fit[N_avg:idx_steady]/U_ref

# Assessing Cox law
# diff3 = np.concatenate( (np.deg2rad(angle_circle[N_avg:-N_avg])**3 - np.deg2rad(angle_l[N_avg:-N_avg])**3, \
#     np.deg2rad(angle_circle[N_avg:-N_avg])**3 - np.deg2rad(angle_r[N_avg:-N_avg])**3), axis=None )
diff3 = np.concatenate( (np.deg2rad(angle_circle[N_avg:idx_steady])**3 - np.deg2rad(angle_l[N_avg:idx_steady])**3, \
    np.deg2rad(angle_circle[N_avg:idx_steady])**3 - np.deg2rad(angle_r[N_avg:idx_steady])**3), axis=None )
# vec_cox = velocity_fit_red
vec_cox = np.concatenate( (velocity_fit_red,velocity_fit_red), axis=None)
cox = lambda t, p : p*t
p_cox, _ = opt.curve_fit(cox, vec_cox, diff3)
length_ratio = np.exp(p_cox[0]/9.0)
print("L/L_m (Cox) = "+str(length_ratio))

plt.title('Viscous bending effect', fontsize=30.0)
# plt.plot(vec_cox[0::plot_sampling], diff3[0::plot_sampling], 'k.', markerfacecolor="None", markersize=22.5, markeredgewidth=2.0, label='MD')
plt.semilogx(vec_cox[0::2*plot_sampling], diff3[0::2*plot_sampling], 'k.', markerfacecolor="None", markersize=22.5, markeredgewidth=2.0, label='MD')
Ca = np.linspace(0.004, max(vec_cox+0.05), 250)
plt.semilogx(Ca, p_cox[0]*Ca, 'm--', linewidth=3.0, label=r'fit: $(\theta_c^3-\theta_m^3)\sim p_0\cdot U$')
plt.legend(fontsize=20.0)
plt.ylabel(r'$\theta_c^3-\theta_m^3$ [-1]', fontsize=30.0)
plt.xlabel(r'$U/U_{ref}$ [-1]', fontsize=30.0)
plt.yticks(fontsize=plot_tcksize)
plt.xticks(fontsize=plot_tcksize)
plt.xlim([Ca[0], Ca[-1]])
plt.show()

theta_0_micro = 0.5*(np.mean(angle_l[idx_zero:])+np.mean(angle_r[idx_zero:]))
err_theta_0_micro = np.std(np.concatenate((angle_l[idx_zero:], angle_r[idx_zero:]))) \
        / np.sqrt(len(np.concatenate((angle_l[idx_zero:], angle_r[idx_zero:]))))
theta_0_macro = np.mean(angle_circle[idx_zero:])
err_theta_0_macro = np.std(angle_circle[idx_zero:])/np.sqrt(len(angle_circle[idx_zero:]))

print("theta_0 (micro) = "+str(theta_0_micro)+"+/-"+str(err_theta_0_micro))
print("theta_0 (macro) = "+str(theta_0_macro)+"+/-"+str(err_theta_0_macro))

theta_0 = 0.5*(theta_0_micro+theta_0_macro)
print("theta_0 (reference) = "+str(theta_0))

# Transform into cos and plot againts velocity
# cos_ca = cos(theta_0)-cos_vec(contact_angle[N_avg:-N_avg])
cos_ca = cos(theta_0)-cos_vec(contact_angle[N_avg:idx_steady])
# cos_ca = cos(theta_0)-cos_vec(angle_circle[N_avg:-N_avg])
cos_range = np.linspace(0.02, max(cos_ca), 250)

# Fit polynomial of desired order
# coef_ca = np.polyfit(cos_ca, velocity_fit_red, 3, w=weights)

# Fit y = a1*x + a3*x^3
mkt_3 = lambda x, a1, a3 : a1*x + a3*(x**3)
therm_fun = lambda t, b0, b1 : b0 * np.exp(-b1*(0.5*sin(t)+cos(t))**2) * (cos(theta_0)-cos(t))
popt, _ = opt.curve_fit(mkt_3, cos_ca, velocity_fit_red)
# popt_therm, _ = opt.curve_fit(therm_fun, contact_angle[N_avg:-N_avg], velocity_fit_red)
popt_therm, _ = opt.curve_fit(therm_fun, contact_angle[N_avg:idx_steady], velocity_fit_red)

# Contact line friction estimate from nonlinear MKT
mu_st = 1.0/popt[0]
mu_th = 1.0/popt_therm[0]
print("mu* (mkt)   = "+str( mu_st ))
print("mu* (therm) = "+str( mu_th ))

beta = popt[1] * mu_st
mu_st_fun = lambda xi : mu_st / (1.0 + beta*xi**2 )
mu_th_fun = lambda t : mu_th * np.exp(popt_therm[1]*(0.5*sin(t)+cos(t))**2)

print('beta = '+str(beta))
print('a    = '+str(popt_therm[1]))

# Example: Ca=0.1
# theta_d = 105.5
# theta_d = 84.0
# xi_d = cos(theta_0) - cos(theta_d)
xi_range = np.linspace(-2.0, 2.0, 500)
t_range  = np.linspace(0.0, 180.0, 500)

fig2, ((ax3, ax4), (ax33, ax44)) = plt.subplots(2, 2)
### MKT FORMULA ###
ax3.set_title('MKT expression fit', fontsize=25.0)
ax3.plot(cos_ca[0::plot_sampling], velocity_fit_red[0::plot_sampling], 'k.', markerfacecolor="None", markersize=22.5, markeredgewidth=2.0, label='MD')
ax3.plot(cos_range, mkt_3(cos_range, *popt), 'm-.', linewidth=3.0, label=r'fit: $U\sim a_1 x + a_3 x^3$')
ax3.plot(cos_range, popt[0]*cos_range, 'g--', linewidth=3.0, label=r'linear limit for $\delta\cos\rightarrow 0$')
ax3.set_xlabel(r'$\cos\theta_0-\cos\theta$ [1]', fontsize=25.0)
ax3.set_ylabel(r'$U/U_{ref}$ [1]', fontsize=25.0)
ax3.legend(fontsize=20.0)
ax3.tick_params(axis='x', labelsize=plot_tcksize)
ax3.tick_params(axis='y', labelsize=plot_tcksize)

ax4.set_title('Angle-dependent contact line friction', fontsize=30.0)
ax4.plot(xi_range, mu_st_fun(xi_range), 'k-', linewidth=2.75, label=r'$\hat{\mu}_f^*\;/\;[1+\beta(\delta\cos)^2]$')
ax4.set_xlabel(r'$\delta\cos$ [1]', fontsize=25.0)
ax4.set_ylabel(r'$\mu_f^*$ [1]', fontsize=25.0)
# ax4.legend(fontsize=20.0)
ax4.tick_params(axis='x', labelsize=plot_tcksize)
ax4.tick_params(axis='y', labelsize=plot_tcksize)
ax4.set_xlim([-2.0, 2.0])
ax4.set_ylim([0.0, 1.25*mu_st])
###################

# plt.show()
# fig3, (ax33, ax44) = plt.subplots(1, 2)

### P&B FORMULA ###
# angle_abs = contact_angle[N_avg:-N_avg]
angle_abs = contact_angle[N_avg:idx_steady]
ax33.plot(angle_abs[0::plot_sampling], velocity_fit_red[0::plot_sampling], 'k.', markerfacecolor="None", markersize=22.5, markeredgewidth=2.0, label='MD')
theta_thermo = np.linspace(min(angle_abs), max(angle_abs), 250)
ax33.plot(theta_thermo, therm_fun(theta_thermo, *popt_therm), 'c-', linewidth=2.5, label='Johansson & Hess 2018')
ax33.set_xlabel(r'$\theta$ [deg]', fontsize=25.0)
ax33.set_ylabel(r'$U/U_{ref}$ [1]', fontsize=25.0)
ax33.legend(fontsize=20.0)
ax33.tick_params(axis='x', labelsize=plot_tcksize)
ax33.tick_params(axis='y', labelsize=plot_tcksize)

# therm_fun = lambda t, b0, b1 : b0 * np.exp(-b1*(0.5*sin(t)+cos(t))**2) * (cos(theta_0)-cos(t))
ax44.plot(theta_thermo, mu_th_fun(theta_thermo), 'k-', linewidth=2.75)
ax44.set_xlabel(r'$\theta$ [deg]', fontsize=25.0)
ax44.set_ylabel(r'$\mu_f^*$ [1]', fontsize=25.0)
ax44.tick_params(axis='x', labelsize=plot_tcksize)
ax44.tick_params(axis='y', labelsize=plot_tcksize)
###################
plt.show()
