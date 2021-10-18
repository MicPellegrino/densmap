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

tan = lambda t : sin(t)/cos(t)

# Plotting params
plot_sampling = 30
plot_tcksize = 17.5

# Reference units
mu = 0.877                  # mPa*s
gamma = 57.8                # mPa*m
U_ref = (gamma/mu)*1e-3     # nm/ps
theta_0 = 68.8              # deg

# Obtaining the signal from saved .txt files
folder_name_adv = 'SpreadingData/FlatQ3ADV'
folder_name_rec = 'SpreadingData/FlatQ3REC'
folder_name_cap = 'SpreadingData/FlatQ3CAP'
folder_name_rec2 = 'SpreadingData/FlatQ3REC2'

# Double-checking theta0

theta0_meniscus = array_from_file('ShearDropModes/NeoQ3/angle_circle.txt')
Nmen = int(0.5*len(theta0_meniscus))
theta0_meniscus_average = np.mean(theta0_meniscus[Nmen:-1])
print("equuil. meniscus -> theta0 = "+str(theta0_meniscus_average))

# Rational polynomial fit
def rational(x, p, q):
    return np.polyval(p, x) / np.polyval(q + [1.0], x)
def rational_3_3(x, p0, p1, p2, q1, q2):
    return rational(x, [p0, p1, p2], [q1, q2])
def rational_4_2(x, p0, p1, p2, p4, q1):
    return rational(x, [p0, p1, p2, p4], [q1,])

class ContactLine :

    def __init__(self, t, fl, fr, al, ar, r, ac) :
        self.time = t
        self.foot_l = fl
        self.foot_r = fr
        self.angle_l = al
        self.angle_r = ar
        self.radius = r
        self.angle_circle = ac
        self.contact_line_pos = 0.5*(fr-fl)
        self.idx_steady=-1

    def fit_cl(self, fit_fun) :
        popt, pcov = opt.curve_fit(fit_fun, self.time, self.contact_line_pos)
        self.contact_line_fit = fit_fun(self.time, *popt)
        self.p_fit = popt

    def compute_velocity(self, deg_num, deg_tot) :
        p_0 = np.array(self.p_fit[0:deg_num])
        p_1 = np.polyder(p_0, m=1)
        q_0 = np.concatenate((self.p_fit[deg_num:deg_tot],[1.0]))
        q_1 = np.polyder(q_0, m=1)
        def v_fit(t) :
            num = ( np.polyval(p_1,t)*np.polyval(q_0,t) - np.polyval(p_0,t)*np.polyval(q_1,t) )
            den = ( np.polyval(q_0,t) )**2
            return num/den
        self.velocity_fit = v_fit(self.time)
        
    def average_angles(self, N_avg) :
        ca = 0.5*(self.angle_r+self.angle_l)
        self.angle_l_avg = np.convolve(self.angle_l, np.ones(N_avg)/N_avg, mode='same')
        self.angle_r_avg = np.convolve(self.angle_r, np.ones(N_avg)/N_avg, mode='same')
        self.contact_angle = np.convolve(ca, np.ones(N_avg)/N_avg, mode='same')

# Minimum CL advancement velocity threshold (rough estimate, a bit arbitrary)
vmin = 1e-4                 # [nm/ps]

# Time window
time = array_from_file(folder_name_adv+'/time.txt')
dt = time[1]-time[0]
T_off = 500
N = int( T_off / dt )

# Reading advancing c.l.
foot_l = array_from_file(folder_name_adv+'/foot_l.txt')
foot_r = array_from_file(folder_name_adv+'/foot_r.txt')
angle_l = array_from_file(folder_name_adv+'/angle_l.txt')
angle_r = array_from_file(folder_name_adv+'/angle_r.txt')
radius = array_from_file(folder_name_adv+'/radius_fit.txt')
angle_circle = array_from_file(folder_name_adv+'/angle_fit.txt')
adv = ContactLine(time[N:], foot_l[N:] ,foot_r[N:], angle_l[N:], angle_r[N:], radius[N:], angle_circle[N:])

# Reading receding c.l.
time = array_from_file(folder_name_rec+'/time.txt')
foot_l = array_from_file(folder_name_rec+'/foot_l.txt')
foot_r = array_from_file(folder_name_rec+'/foot_r.txt')
angle_l = array_from_file(folder_name_rec+'/angle_l.txt')
angle_r = array_from_file(folder_name_rec+'/angle_r.txt')
radius = array_from_file(folder_name_rec+'/radius_fit.txt')
angle_circle = array_from_file(folder_name_rec+'/angle_fit.txt')
rec = ContactLine(time[N:], foot_l[N:] ,foot_r[N:], angle_l[N:], angle_r[N:], radius[N:], angle_circle[N:])

# Advancing replica (capillary)
time = array_from_file(folder_name_cap+'/time.txt')
foot_l = array_from_file(folder_name_cap+'/foot_l.txt')
foot_r = array_from_file(folder_name_cap+'/foot_r.txt')
angle_l = array_from_file(folder_name_cap+'/angle_l.txt')
angle_r = array_from_file(folder_name_cap+'/angle_r.txt')
radius = array_from_file(folder_name_cap+'/radius_fit.txt')
angle_circle = array_from_file(folder_name_cap+'/angle_fit.txt')
cap = ContactLine(time[N:], foot_l[N:] ,foot_r[N:], angle_l[N:], angle_r[N:], radius[N:], angle_circle[N:])

# Reading receding c.l.
time = array_from_file(folder_name_rec2+'/time.txt')
foot_l = array_from_file(folder_name_rec2+'/foot_l.txt')
foot_r = array_from_file(folder_name_rec2+'/foot_r.txt')
angle_l = array_from_file(folder_name_rec2+'/angle_l.txt')
angle_r = array_from_file(folder_name_rec2+'/angle_r.txt')
radius = array_from_file(folder_name_rec2+'/radius_fit.txt')
angle_circle = array_from_file(folder_name_rec2+'/angle_fit.txt')
rec2 = ContactLine(time[N:], foot_l[N:] ,foot_r[N:], angle_l[N:], angle_r[N:], radius[N:], angle_circle[N:])

fig1, (ax1, ax2) = plt.subplots(1, 2)
ax1.set_title('Spreading branches', fontsize=25.0)
ax1.plot(adv.time, adv.foot_l, 'r-', linewidth=2.5, label='inertial')
ax1.plot(adv.time, adv.foot_r, 'r-', linewidth=2.5)
ax1.plot(cap.time, cap.foot_l, 'g-', linewidth=2.5, label='capillary')
ax1.plot(cap.time, cap.foot_r, 'g-', linewidth=2.5)
ax1.set_xlabel('t [ps]', fontsize=25.0)
ax1.set_ylabel('x [nm]', fontsize=25.0)
ax1.tick_params(axis='x', labelsize=plot_tcksize)
ax1.tick_params(axis='y', labelsize=plot_tcksize)
ax1.set_ylim([0, 160])
ax2.plot(rec.time, rec.foot_l, 'b-', linewidth=2.5)
ax2.plot(rec.time, rec.foot_r, 'b-', linewidth=2.5)
ax2.plot(rec2.time, rec2.foot_l, 'm-', linewidth=2.5)
ax2.plot(rec2.time, rec2.foot_r, 'm-', linewidth=2.5)
ax2.set_xlabel('t [ps]', fontsize=25.0)
ax2.set_ylabel('x [nm]', fontsize=25.0)
ax2.tick_params(axis='x', labelsize=plot_tcksize)
ax2.tick_params(axis='y', labelsize=plot_tcksize)
ax2.set_ylim([0, 160])
plt.show()

adv.fit_cl(rational_4_2)
adv.compute_velocity(4, 5)
rec.fit_cl(rational_3_3)
rec.compute_velocity(3, 5)
cap.fit_cl(rational_4_2)
cap.compute_velocity(4, 5)
rec2.fit_cl(rational_3_3)
rec2.compute_velocity(3, 5)

plt.title('Average left+right', fontsize=25.0)
plt.plot(adv.time, adv.contact_line_pos, 'r-', linewidth=2.5, label='adv. (direct)')
plt.plot(cap.time, cap.contact_line_pos, 'g-', linewidth=2.5, label='adv. (from Q1)')
plt.plot(rec.time, rec.contact_line_pos, 'b-', linewidth=2.5, label='rec. (from Q5)')
plt.plot(rec2.time, rec2.contact_line_pos, 'm-', linewidth=2.5, label='rec. (from Q4)')
plt.plot(adv.time, adv.contact_line_fit, 'r--', linewidth=3.0)
plt.plot(rec.time, rec.contact_line_fit, 'b--', linewidth=3.0)
plt.plot(rec2.time, rec2.contact_line_fit, 'm--', linewidth=3.0)
plt.plot(cap.time, cap.contact_line_fit, 'g--', linewidth=3.0)
plt.xticks(fontsize=plot_tcksize)
plt.yticks(fontsize=plot_tcksize)
plt.legend(fontsize=20)
plt.xlabel('t [ps]', fontsize=25.0)
plt.ylabel('x [nm]', fontsize=25.0)
plt.show()

plt.title('Spreading speed', fontsize=25.0)
plt.plot(adv.time, adv.velocity_fit, 'r-', linewidth=2.5, label='adv. (inertial)')
plt.plot(cap.time, cap.velocity_fit, 'g-', linewidth=2.5, label='adv. (capillary)')
plt.plot(rec.time, rec.velocity_fit, 'b-', linewidth=2.5, label='receding')
plt.plot(rec2.time, rec2.velocity_fit, 'm-', linewidth=2.5, label='receding')
plt.xticks(fontsize=plot_tcksize)
plt.yticks(fontsize=plot_tcksize)
plt.legend(fontsize=20)
plt.xlabel('t [ps]', fontsize=25.0)
plt.ylabel('u_cl [nm/ps]', fontsize=25.0)
plt.show()

adv.idx_steady = np.argmin(np.abs(adv.velocity_fit-vmin))
cap.idx_steady = np.argmin(np.abs(cap.velocity_fit-vmin))
rec.idx_steady = np.argmin(np.abs(rec.velocity_fit-vmin))
rec2.idx_steady = np.argmin(np.abs(rec2.velocity_fit-vmin))

# Rolling average
delta_t_avg = 880               # [ps]
N_avg = int(delta_t_avg/dt)
adv.average_angles(N_avg)
rec.average_angles(N_avg)
cap.average_angles(N_avg)
rec2.average_angles(N_avg)

N_eq = int(0.667*len(adv.time[N_avg:-N_avg]))
# print("theta_adv = "+str(np.mean(adv.contact_angle[N_eq:-N_avg]))+" deg")
# print("theta_rec = "+str(np.mean(rec.contact_angle[N_eq:-N_avg]))+" deg")
print("theta_adv = "+str(np.mean(adv.angle_circle[N_eq:-N_avg]))+" deg")
print("theta_rec = "+str(np.mean(rec.angle_circle[N_eq:-N_avg]))+" deg")

theta_0_adv = 73.13203235002977         # [deg]
theta_0_rec = 71.45227316787968         # [deg]
# theta_0 = 0.5*(theta_0_adv+theta_0_rec)
theta_0 = 72.29215275895473

plt.title('Contact angles (rolling average)', fontsize=25.0)
plt.plot(adv.time[N_avg:-N_avg], adv.contact_angle[N_avg:-N_avg], 'r-', linewidth=2.5, label='adv. (direct)')
plt.plot(cap.time[N_avg:-N_avg], cap.contact_angle[N_avg:-N_avg], 'g-', linewidth=2.5, label='adv. (from Q1)')
plt.plot(rec.time[N_avg:-N_avg], rec.contact_angle[N_avg:-N_avg], 'b-', linewidth=2.5, label='rec. (from Q5)')
plt.plot(rec2.time[N_avg:-N_avg], rec2.contact_angle[N_avg:-N_avg], 'm-', linewidth=2.5, label='rec. (from Q4)')
plt.plot(adv.time[N_eq:-N_avg], theta_0_adv*np.ones(len(adv.time[N_eq:-N_avg])), 'r--', linewidth=3.0)
plt.plot(adv.time[N_eq:-N_avg], theta_0_rec*np.ones(len(adv.time[N_eq:-N_avg])), 'b--', linewidth=3.0)
# plt.plot(cap.time[N_eq:-N_avg], theta_0*np.ones(len(cap.time[N_eq:-N_avg])), 'k--', linewidth=3.0)
plt.xticks(fontsize=plot_tcksize)
plt.yticks(fontsize=plot_tcksize)
plt.legend(fontsize=20)
plt.xlabel('t [ps]', fontsize=25.0)
plt.ylabel('theta [deg]', fontsize=25.0)
plt.show()

"""
c = cos(theta_0)-cos(adv.contact_angle[N_avg:-N_avg])
v = adv.velocity_fit[N_avg:-N_avg]/U_ref
c = cos(theta_0)-cos(cap.contact_angle[N_avg:-N_avg])
v = cap.velocity_fit[N_avg:-N_avg]/U_ref
c = cos(theta_0)-cos(rec.contact_angle[N_avg:-N_avg])
v = rec.velocity_fit[N_avg:-N_avg]/U_ref
c = cos(theta_0)-cos(rec2.contact_angle[N_avg:-N_avg])
v = rec2.velocity_fit[N_avg:-N_avg]/U_ref
"""

reduced_velocity_micro = np.concatenate( 
        (adv.velocity_fit[N_avg:min(len(adv.velocity_fit)-N_avg,adv.idx_steady)]/U_ref, \
        cap.velocity_fit[N_avg:min(len(cap.velocity_fit)-N_avg,cap.idx_steady)]/U_ref, \
        rec.velocity_fit[N_avg:min(len(rec.velocity_fit)-N_avg,rec.idx_steady)]/U_ref, \
        rec2.velocity_fit[N_avg:min(len(rec2.velocity_fit)-N_avg,rec2.idx_steady)]/U_ref), \
        axis=None )
reduced_cosine_micro = np.concatenate( 
        (cos(theta_0)-cos(adv.contact_angle[N_avg:min(len(adv.contact_angle)-N_avg,adv.idx_steady)]), \
        cos(theta_0)-cos(cap.contact_angle[N_avg:min(len(cap.contact_angle)-N_avg,cap.idx_steady)]),  \
        cos(theta_0)-cos(rec.contact_angle[N_avg:min(len(rec.contact_angle)-N_avg,rec.idx_steady)]), \
        cos(theta_0)-cos(rec2.contact_angle[N_avg:min(len(rec2.contact_angle)-N_avg,rec2.idx_steady)])), \
        axis=None )
angle_micro = np.concatenate( 
        (adv.contact_angle[N_avg:min(len(adv.contact_angle)-N_avg,adv.idx_steady)], \
        cap.contact_angle[N_avg:min(len(cap.contact_angle)-N_avg,cap.idx_steady)],  \
        rec.contact_angle[N_avg:min(len(rec.contact_angle)-N_avg,rec.idx_steady)], \
        rec2.contact_angle[N_avg:min(len(rec2.contact_angle)-N_avg,rec2.idx_steady)]), \
        axis=None )


reduced_velocity_advance = np.concatenate(
        (adv.velocity_fit[N_avg:min(len(adv.velocity_fit)-N_avg,adv.idx_steady)]/U_ref, \
        cap.velocity_fit[N_avg:min(len(cap.velocity_fit)-N_avg,cap.idx_steady)]/U_ref), \
        axis=None)
reduced_cosine_advance = np.concatenate(
        (cos(theta_0)-cos(adv.contact_angle[N_avg:min(len(adv.contact_angle)-N_avg,adv.idx_steady)]), \
        cos(theta_0)-cos(cap.contact_angle[N_avg:min(len(cap.contact_angle)-N_avg,cap.idx_steady)])), \
        axis=None)
angle_advance = np.concatenate(
        (adv.contact_angle[N_avg:min(len(adv.contact_angle)-N_avg,adv.idx_steady)], \
        cap.contact_angle[N_avg:min(len(cap.contact_angle)-N_avg,cap.idx_steady)]), \
        axis=None)

reduced_velocity_recede = np.concatenate( 
        (rec.velocity_fit[N_avg:min(len(rec.velocity_fit)-N_avg,rec.idx_steady)]/U_ref, \
        rec2.velocity_fit[N_avg:min(len(rec2.velocity_fit)-N_avg,rec2.idx_steady)]/U_ref), \
        axis=None )
reduced_cosine_recede = np.concatenate( 
        (cos(theta_0)-cos(rec.contact_angle[N_avg:min(len(rec.contact_angle)-N_avg,rec.idx_steady)]), \
        cos(theta_0)-cos(rec2.contact_angle[N_avg:min(len(rec2.contact_angle)-N_avg,rec2.idx_steady)])), \
        axis=None )
angle_recede = np.concatenate( 
        (rec.contact_angle[N_avg:min(len(rec.contact_angle)-N_avg,rec.idx_steady)], \
        rec2.contact_angle[N_avg:min(len(rec2.contact_angle)-N_avg,rec2.idx_steady)]), \
        axis=None )

cos_lim = 0.2   # +/-
idx_micro = np.where( np.abs(reduced_cosine_micro) <= cos_lim )
# Linear CL friction model
p_micro = np.polyfit( reduced_cosine_micro[idx_micro], reduced_velocity_micro[idx_micro], 1 )

print("mu_f = "+str(1/p_micro[0]))

theta_0 = 0.5*( theta_0_adv + theta_0_rec )

# Petter
therm_fun = lambda t, b0, b1 : b0 * np.exp(-b1*(0.5*sin(t)+cos(t))**2) * (cos(theta_0)-cos(t))

# MKT
mkt_3 = lambda x, a1, a3 : a1*x + a3*(x**3)

# Prior
print("PRIOR")
popt_therm, pcov_therm = opt.curve_fit(therm_fun, angle_advance, reduced_velocity_advance)
print("cov_thermo: ")
print(pcov_therm)
error_therm_receding = \
        np.sqrt(np.sum((reduced_velocity_recede-therm_fun(angle_recede, *popt_therm))**2))/len(angle_recede)
print("err_thermo = "+str(error_therm_receding))
popt, pcov_mkt = opt.curve_fit(mkt_3, reduced_cosine_advance, reduced_velocity_advance)
print("cov_mkt: ")
print(pcov_mkt)
error_mkt_receding = \
        np.sqrt(np.sum((reduced_velocity_recede-mkt_3(reduced_cosine_recede, *popt))**2))/len(reduced_cosine_recede)
print("err_mkt = "+str(error_mkt_receding))

# Posterior
print("POSTERIOR")
popt_therm, pcov_therm = opt.curve_fit(therm_fun, angle_micro, reduced_velocity_micro)
print("cov_thermo: ")
print(pcov_therm)
error_therm_receding = \
        np.sqrt(np.sum((reduced_velocity_recede-therm_fun(angle_recede, *popt_therm))**2))/len(angle_recede)
print("err_thermo = "+str(error_therm_receding))
popt, pcov_mkt = opt.curve_fit(mkt_3, reduced_cosine_micro, reduced_velocity_micro)
print("cov_mkt: ")
print(pcov_mkt)
error_mkt_receding = \
        np.sqrt(np.sum((reduced_velocity_recede-mkt_3(reduced_cosine_recede, *popt))**2))/len(reduced_cosine_recede)
print("err_mkt = "+str(error_mkt_receding))

mu_st_fun = lambda xi : mu_st / (1.0 + beta*xi**2 )
mu_th_fun = lambda t : mu_th * np.exp(popt_therm[1]*(0.5*sin(t)+cos(t))**2)

mu_st = 1.0/popt[0]
mu_th = 1.0/popt_therm[0]
print("mu* (mkt)   = "+str( mu_st ))
print("mu* (therm) = "+str( mu_th ))
beta = popt[1] * mu_st
print("beta = "+str(beta))
print("expa = "+str(popt_therm[1]))
cos_range = np.linspace(min(reduced_cosine_micro), max(reduced_cosine_micro), 250)
# cos_range = np.linspace(-1.0, 1.0, 250)
xi_range = np.linspace(-2.0, 2.0, 500)

fig2, ((ax3, ax4), (ax33, ax44)) = plt.subplots(2, 2)
### MKT FORMULA ###
# ax3.set_title('MKT expression fit', fontsize=25.0)
ax3.plot(reduced_cosine_micro[0::plot_sampling], reduced_velocity_micro[0::plot_sampling], 'k.', markerfacecolor="None", markersize=20.0, markeredgewidth=1.75, label='MD')
ax3.plot(cos_range, mkt_3(cos_range, *popt), 'm-.', linewidth=3.0, label=r'fit: $U\sim a_1 x + a_3 x^3$')
ax3.plot(cos_range, popt[0]*cos_range, 'g--', linewidth=3.0, label=r'linear limit for $\delta\cos\rightarrow 0$')
ax3.set_xlabel(r'$\cos\theta_0-\cos\theta$ [1]', fontsize=25.0)
ax3.set_ylabel(r'$U/U_{ref}$ [1]', fontsize=25.0)
ax3.legend(fontsize=20.0)
ax3.tick_params(axis='x', labelsize=plot_tcksize)
ax3.tick_params(axis='y', labelsize=plot_tcksize)

# ax4.set_title('Angle-dependent contact line friction', fontsize=30.0)
ax4.plot(xi_range, mu_st_fun(xi_range), 'k-', linewidth=2.75, label=r'$\hat{\mu}_f^*\;/\;[1+\beta(\delta\cos)^2]$')
ax4.set_xlabel(r'$\delta\cos$ [1]', fontsize=25.0)
ax4.set_ylabel(r'$\mu_f^*$ [1]', fontsize=25.0)
# ax4.legend(fontsize=20.0)
ax4.tick_params(axis='x', labelsize=plot_tcksize)
ax4.tick_params(axis='y', labelsize=plot_tcksize)
ax4.set_xlim([-2.0, 2.0])
ax4.set_ylim([0.0, 1.25*mu_st])
###################

### P&B FORMULA ###
ax33.plot(angle_micro[0::plot_sampling], reduced_velocity_micro[0::plot_sampling], 'k.', markerfacecolor="None", markersize=20.0, markeredgewidth=1.75)
theta_thermo = np.linspace(min(angle_micro), max(angle_micro), 250)
# theta_thermo = np.linspace(0.0, 180.0, 250)
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
