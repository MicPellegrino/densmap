from math import isnan

import numpy as np
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

avg_theta_0 = 95.67050716578869
std_theta_0 = 4.299262399062776

folders = [ 'ShearDynamic/Q2_Ca005', 
            'ShearDynamic/Q2_Ca010', 
            'ShearDynamic/Q2_Ca015',
            'ShearDynamic/Q2_Ca020', 
            'ShearDynamic/Q2_Ca025' ]

capillary_number = np.array([ 0.05, 0.10, 0.15, 0.20, 0.25])

# Init averaging
t_0 = 4500

adv_collect = []
rec_collect = []
avg_angle_adv = []
std_angle_adv = []
avg_angle_rec = []
std_angle_rec = []
for fn in folders :
    time = array_from_file(fn+'/time.txt')
    idx_0 = np.abs( time-t_0 ).argmin()
    tl = array_from_file(fn+'/angle_tl.txt')[idx_0:]
    br = array_from_file(fn+'/angle_br.txt')[idx_0:]
    tr = array_from_file(fn+'/angle_tr.txt')[idx_0:]
    bl = array_from_file(fn+'/angle_bl.txt')[idx_0:]
    adv = 0.5 * ( tl + br )
    adv_collect.append( adv )
    rec = 0.5 * ( tr + bl )
    rec_collect.append( rec )
    avg_angle_adv.append( np.mean( adv ) )
    std_angle_adv.append( np.std( adv ) )
    avg_angle_rec.append( np.mean( rec ) )
    std_angle_rec.append( np.std( rec ) )

avg_angle_adv = np.array( avg_angle_adv )
avg_angle_rec = np.array( avg_angle_rec )
std_angle_adv = np.array( std_angle_adv )
std_angle_rec = np.array( std_angle_rec )

def mkt_formula(cap, a_mkt, t0) :
    din_t = np.rad2deg( np.arccos( np.cos(np.deg2rad(t0)) - cap*a_mkt )  )
    return din_t

def lin_pf_formula(cap, a_pf, t0) :
    din_t = np.rad2deg( np.deg2rad(t0) + 2.0*np.sqrt(2.0)*a_pf*cap/3.0 )
    return din_t

mkt_fit = lambda cap, a_mkt : mkt_formula(cap, a_mkt, avg_theta_0)
mkt_fit_ps = lambda cap, a_mkt : mkt_formula(cap, a_mkt, avg_theta_0+std_theta_0)
mkt_fit_ms = lambda cap, a_mkt : mkt_formula(cap, a_mkt, avg_theta_0-std_theta_0)

# MKT #
friction_ratio_0 = 3.0
popt_adv, pcov_adv = \
        opt.curve_fit(mkt_fit, capillary_number, avg_angle_adv, p0=friction_ratio_0)
popt_rec, pcov_rec = \
        opt.curve_fit(mkt_fit, -capillary_number, avg_angle_rec, p0=friction_ratio_0)

mu_ratio_adv = popt_adv[0]
mu_ratio_rec = popt_rec[0]

capillary_adv = np.linspace(0, 0.3, 100)
capillary_rec = np.linspace(-0.30, 0.0, 100)
mkt_adv = np.vectorize(lambda u : mkt_fit(u, mu_ratio_adv))
mkt_rec = np.vectorize(lambda u : mkt_fit(u, mu_ratio_rec))

# Fitting for +/- standard deviation #
friction_ratio_0 = 0.5*(mu_ratio_adv+mu_ratio_rec)
popt_adv_p = opt.curve_fit(mkt_fit_ps, capillary_number, avg_angle_adv, p0=friction_ratio_0)
popt_rec_p = opt.curve_fit(mkt_fit_ps, -capillary_number, avg_angle_rec, p0=friction_ratio_0)
popt_adv_m = opt.curve_fit(mkt_fit_ms, capillary_number, avg_angle_adv, p0=friction_ratio_0)
popt_rec_m = opt.curve_fit(mkt_fit_ms, -capillary_number, avg_angle_rec, p0=friction_ratio_0)
mkt_adv_p = np.vectorize(lambda u : mkt_fit_ps(u, popt_adv_p[0]))
mkt_rec_p = np.vectorize(lambda u : mkt_fit_ps(u, popt_rec_p[0]))
mkt_adv_m = np.vectorize(lambda u : mkt_fit_ms(u, popt_adv_m[0]))
mkt_rec_m = np.vectorize(lambda u : mkt_fit_ms(u, popt_rec_m[0]))
std_adv = (0.5*np.abs(popt_adv_p[0]-popt_adv_m[0]))[0]
std_rec = (0.5*np.abs(popt_rec_p[0]-popt_rec_m[0]))[0]
######################################

eb1 = plt.errorbar( capillary_number, avg_angle_adv, yerr=std_angle_adv, \
        ecolor='r', fmt='ro', elinewidth=2, capsize=7.5, capthick=2.5, ms=8.0, \
        label='MD adv (+/-std)' )
eb2 = plt.errorbar( -capillary_number, avg_angle_rec, yerr=std_angle_rec, \
        ecolor='b', fmt='bs', elinewidth=2, capsize=7.5, capthick=2.5, ms=8.0, \
        label='MD rec (+/-std)' )
"""
eb3 = plt.violinplot(adv_collect, positions=capillary_number, \
        widths=0.035, showmeans=False, showmedians=False, showextrema=False)
for pc in eb3['bodies']:
    pc.set_facecolor('red')
eb4 = plt.violinplot(rec_collect, positions=-capillary_number, \
    widths=0.03, showmeans=False, showmedians=False, showextrema=False)
for pc in eb4['bodies']:
    pc.set_facecolor('blue')
"""
plt.plot(capillary_adv, mkt_adv(capillary_adv), 'r--', linewidth=2.0, \
        label=r'MKT, $\mu_f/\mu=$'+'{:.2f}'.format(mu_ratio_adv)+ \
        '+/-'+'{:.3f}'.format( std_adv ) )
plt.plot(capillary_rec, mkt_rec(capillary_rec), 'b--', linewidth=2.0, \
        label=r'MKT, $\mu_f/\mu=$'+'{:.2f}'.format(mu_ratio_rec)+ \
        '+/-'+'{:.3f}'.format( std_rec ) )
plt.plot([0.275, 0.275], [50.0, 140.0], 'k--', label='stab. threshold')
plt.plot([-0.275, -0.275], [50.0, 140.0], 'k--')

# Plotting +/- standard deviation #
"""
plt.plot(capillary_adv, mkt_adv_p(capillary_adv), 'r-.', linewidth=1.0 )
plt.plot(capillary_rec, mkt_rec_p(capillary_rec), 'b-.', linewidth=1.0 )
plt.plot(capillary_adv, mkt_adv_m(capillary_adv), 'r-.', linewidth=1.0 )
plt.plot(capillary_rec, mkt_rec_m(capillary_rec), 'b-.', linewidth=1.0 )
"""
###################################

plt.title("Fit of MKT and estimate of c.l. friction", fontsize=30.0)
plt.legend(fontsize=20.0, loc='lower right')
plt.xticks(fontsize=20.0)
plt.yticks(fontsize=20.0)
plt.ylabel(r'$\theta_d$ [deg]', fontsize=25.0)
plt.xlabel(r'$Ca$ [-1]', fontsize=25.0)
plt.xlim([-0.3, 0.3])
plt.ylim([50.0, 140.0])
plt.show()

# Fitting PF #
pf_fit = lambda cap, a_pf : lin_pf_formula(cap, a_pf, avg_theta_0)
pf_fit_ps = lambda cap, a_pf : lin_pf_formula(cap, a_pf, avg_theta_0+std_theta_0)
pf_fit_ms = lambda cap, a_pf : lin_pf_formula(cap, a_pf, avg_theta_0-std_theta_0)

friction_ratio_0 = 1.5
popt_adv, pcov_adv = \
        opt.curve_fit(pf_fit, capillary_number, avg_angle_adv, p0=friction_ratio_0)
popt_rec, pcov_rec = \
        opt.curve_fit(pf_fit, -capillary_number, avg_angle_rec, p0=friction_ratio_0)

mu_ratio_adv = popt_adv[0]
mu_ratio_rec = popt_rec[0]

capillary_adv = np.linspace(0, 0.3, 100)
capillary_rec = np.linspace(-0.30, 0.0, 100)
lin_pf_adv = np.vectorize(lambda u : pf_fit(u, mu_ratio_adv))
lin_pf_rec = np.vectorize(lambda u : pf_fit(u, mu_ratio_rec))

# Fitting for +/- standard deviation #
friction_ratio_0 = 0.5*(mu_ratio_adv+mu_ratio_rec)
popt_adv_p = opt.curve_fit(pf_fit_ps, capillary_number, avg_angle_adv, p0=friction_ratio_0)
popt_rec_p = opt.curve_fit(pf_fit_ps, -capillary_number, avg_angle_rec, p0=friction_ratio_0)
popt_adv_m = opt.curve_fit(pf_fit_ms, capillary_number, avg_angle_adv, p0=friction_ratio_0)
popt_rec_m = opt.curve_fit(pf_fit_ms, -capillary_number, avg_angle_rec, p0=friction_ratio_0)
mkt_adv_p = np.vectorize(lambda u : pf_fit_ps(u, popt_adv_p[0]))
mkt_rec_p = np.vectorize(lambda u : pf_fit_ps(u, popt_rec_p[0]))
mkt_adv_m = np.vectorize(lambda u : pf_fit_ms(u, popt_adv_m[0]))
mkt_rec_m = np.vectorize(lambda u : pf_fit_ms(u, popt_rec_m[0]))
std_adv = (0.5*np.abs(popt_adv_p[0]-popt_adv_m[0]))[0]
std_rec = (0.5*np.abs(popt_rec_p[0]-popt_rec_m[0]))[0]
######################################

eb1 = plt.errorbar( capillary_number, avg_angle_adv, yerr=std_angle_adv, \
        ecolor='r', fmt='ro', elinewidth=2, capsize=7.5, capthick=2.5, ms=8.0, \
        label='MD adv (+/-std)' )
eb2 = plt.errorbar( -capillary_number, avg_angle_rec, yerr=std_angle_rec, \
        ecolor='b', fmt='bs', elinewidth=2, capsize=7.5, capthick=2.5, ms=8.0, \
        label='MD rec (+/-std)' )

plt.plot(capillary_adv, lin_pf_adv(capillary_adv), 'r--', linewidth=2.0, \
        label=r'PF, $\mu_f/\mu=$'+'{:.2f}'.format(mu_ratio_adv)+ \
        '+/-'+'{:.3f}'.format( std_adv ) )
plt.plot(capillary_rec, lin_pf_rec(capillary_rec), 'b--', linewidth=2.0, \
        label=r'PF, $\mu_f/\mu=$'+'{:.2f}'.format(mu_ratio_rec)+ \
        '+/-'+'{:.3f}'.format( std_rec ) )
plt.plot([0.275, 0.275], [50.0, 140.0], 'k--', label='stab. threshold')
plt.plot([-0.275, -0.275], [50.0, 140.0], 'k--')

plt.title("Fit of PF (Yue&Feng) and estimate of c.l. friction", fontsize=30.0)
plt.legend(fontsize=20.0, loc='lower right')
plt.xticks(fontsize=20.0)
plt.yticks(fontsize=20.0)
plt.ylabel(r'$\theta_d$ [deg]', fontsize=25.0)
plt.xlabel(r'$Ca$ [-1]', fontsize=25.0)
plt.xlim([-0.3, 0.3])
plt.ylim([50.0, 140.0])
plt.show()
