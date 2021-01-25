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

g_dir = \
    lambda x : (1.0/9.0) * (x**3) - 0.00183985 * (x**4.5) + (1.845823*1e-6) * (x**12.258487) 
g_inv = \
    lambda x : (9.0*x)**(1.0/3.0) + 0.0727387 * x - 0.0515388 * (x**2) + 0.00341336 * (x**3)

def cox_formula(cap, a_cox, t0) :
    din_t= np.rad2deg( g_inv( g_dir( np.deg2rad(t0) ) + cap*np.log10(a_cox) ) )
    return din_t

cox_fit = lambda cap, a_cox : cox_formula(cap, a_cox, avg_theta_0)

lenght_ratio_0 = 10.0
popt_adv, pcov_adv = \
        opt.curve_fit(cox_fit, capillary_number, avg_angle_adv, p0=lenght_ratio_0)
popt_rec, pcov_rec = \
        opt.curve_fit(cox_fit, -capillary_number, avg_angle_rec, p0=lenght_ratio_0)

len_ratio_adv = popt_adv[0]
len_ratio_rec = popt_rec[0]

capillary_adv = np.linspace(0, 0.3, 100)
capillary_rec = np.linspace(-0.30, 0.0, 100)
cox_adv = np.vectorize(lambda u : cox_fit(u, len_ratio_adv))
cox_rec = np.vectorize(lambda u : cox_fit(u, len_ratio_rec))

eb1 = plt.errorbar( capillary_number, avg_angle_adv, yerr=std_angle_adv, \
        ecolor='r', fmt='ro', elinewidth=2, capsize=7.5, capthick=2.5, ms=8.0, \
        label='MD adv' )
eb2 = plt.errorbar( -capillary_number, avg_angle_rec, yerr=std_angle_rec, \
        ecolor='b', fmt='bs', elinewidth=2, capsize=7.5, capthick=2.5, ms=8.0, \
        label='MD rec' )
eb3 = plt.violinplot(adv_collect, positions=capillary_number, \
        widths=0.035, showmeans=False, showmedians=False, showextrema=False)
for pc in eb3['bodies']:
    pc.set_facecolor('red')
eb4 = plt.violinplot(rec_collect, positions=-capillary_number, \
    widths=0.03, showmeans=False, showmedians=False, showextrema=False)
for pc in eb4['bodies']:
    pc.set_facecolor('blue')
plt.plot(capillary_adv, cox_adv(capillary_adv), 'r--', linewidth=2.0, \
        label=r'Cox, $L/\lambda=$'+'{:.2f}'.format(len_ratio_adv) )
plt.plot(capillary_rec, cox_rec(capillary_rec), 'b--', linewidth=2.0, \
        label=r'Cox, $L/\lambda=$'+'{:.2f}'.format(len_ratio_rec) )
plt.plot([0.275, 0.275], [50.0, 140.0], 'k--', label='stab. threshold')
plt.plot([-0.275, -0.275], [50.0, 140.0], 'k--')

plt.title("Fit of Cox law and estimate of c.l. friction", fontsize=30.0)
plt.legend(fontsize=20.0, loc='lower right')
plt.xticks(fontsize=20.0)
plt.yticks(fontsize=20.0)
plt.ylabel(r'$\theta_d$ [deg]', fontsize=25.0)
plt.xlabel(r'$Ca$ [-1]', fontsize=25.0)
plt.xlim([-0.3, 0.3])
plt.ylim([50.0, 140.0])
plt.show()
