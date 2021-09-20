import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

import scipy.ndimage as smg
import scipy.signal as sgn
import scipy.optimize as opt

from matplotlib import cm

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

# Reference units
mu = 0.877                  # [mPa*s]
gamma = 57.8                # [mPa*m]
U_ref = (gamma/mu)*1e-3     # [nm/ps]

# Time window
T_ini = 100.0               # [ps]
T_fin = 24310.0             # [ps]

theta_0 = 72.0196332498935

folder_name = 'SpreadingData/FlatQ3ADV'
time = array_from_file(folder_name+'/time.txt')
foot_l = array_from_file(folder_name+'/foot_l.txt')
foot_r = array_from_file(folder_name+'/foot_r.txt')
angle_l = array_from_file(folder_name+'/angle_l.txt')
angle_r = array_from_file(folder_name+'/angle_r.txt')
radius = array_from_file(folder_name+'/radius_fit.txt')
angle_circle = array_from_file(folder_name+'/angle_fit.txt')
dt = time[1]-time[0]
time_window = T_fin-T_ini
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
p_0 = np.array(popt[0:4])
p_1 = np.polyder(p_0, m=1)
q_0 = np.concatenate((popt[4:5],[1.0]))
q_1 = np.polyder(q_0, m=1)
def velocity_fit(t) :
    num = ( np.polyval(p_1,t)*np.polyval(q_0,t) - np.polyval(p_0,t)*np.polyval(q_1,t) )
    den = ( np.polyval(q_0,t) )**2
    return num/den
velocity_fit = velocity_fit(time)

# Minimum CL advancement velocity threshold (rough estimate, a bit arbitrary)
vmin_values = np.linspace(0.0, 1.875e-3, 20)        # [nm/ps]
# Rolling average
delta_t_avg_values = np.linspace(100, 1100, 20)     # [ps]

err_mkt = np.zeros((20,20))
err_th = np.zeros((20,20))

i = 0

for vmin in vmin_values :

    j = 0

    for delta_t_avg in delta_t_avg_values :
        
        print("(i,j) = ("+str(i)+","+str(j)+")")

        print("vmin = "+str(1e3*vmin)+" [nm/ns]")
        print("delta_t_avg = "+str(delta_t_avg)+" [ps]")
        
        N_avg = int(delta_t_avg/dt)
        contact_angle = np.convolve(contact_angle, np.ones(N_avg)/N_avg, mode='same')
        idx_steady = np.argmin(np.abs(velocity_fit-vmin))
        t_steady = dt*idx_steady
        velocity_fit_red = velocity_fit[N_avg:idx_steady]/U_ref
        cos_ca = cos(theta_0)-cos_vec(contact_angle[N_avg:idx_steady])
        mkt_3 = lambda x, a1, a3 : a1*x + a3*(x**3)
        therm_fun = lambda t, b0, b1 : b0 * np.exp(-b1*(0.5*sin(t)+cos(t))**2) * (cos(theta_0)-cos(t))
        popt, _ = opt.curve_fit(mkt_3, cos_ca, velocity_fit_red)
        popt_therm, _ = opt.curve_fit(therm_fun, contact_angle[N_avg:idx_steady], velocity_fit_red)
        mu_st = 1.0/popt[0]
        mu_th = 1.0/popt_therm[0]
        beta = popt[1] * mu_st
        mu_st_fun = lambda xi : mu_st / (1.0 + beta*xi**2 )
        mu_th_fun = lambda t : mu_th * np.exp(popt_therm[1]*(0.5*sin(t)+cos(t))**2)

        err_mkt[i,j] = np.sqrt( np.sum((mu_st_fun(cos_ca)-velocity_fit_red)*(mu_st_fun(cos_ca)-velocity_fit_red)) \
            / len(velocity_fit_red) )
        err_th[i,j] = np.sqrt( np.sum((mu_th_fun(contact_angle[N_avg:idx_steady])-velocity_fit_red) \
            * (mu_th_fun(contact_angle[N_avg:idx_steady])-velocity_fit_red)) / len(velocity_fit_red) )

        print("err_mkt = "+str(err_mkt[i,j]))
        print("err_th = "+str(err_th[i,j]))

        j += 1

    i += 1

X, Y = np.meshgrid(vmin_values, delta_t_avg_values, sparse=False, indexing='ij')

# fig1, (ax1, ax2) = plt.subplots(1, 2)
# ax1.pcolormesh(X, Y, err_mkt, cmap=cm.bwr, vmin=0, vmax=1)
# ax2.pcolormesh(X, Y, err_th, cmap=cm.bwr, vmin=0, vmax=1)
plt.pcolormesh(X, Y, err_th+err_mkt, cmap=cm.bwr, vmin=0, vmax=50.0)
plt.show()
