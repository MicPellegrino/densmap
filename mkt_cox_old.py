import numpy as np
from math import isnan
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def array_from_file( filename ):
    my_list = []
    with open(filename, 'r') as f:
        for line in f:
            my_list.append(float(line.split()[0]))
    return np.array(my_list)

avg_theta_0 = [124.85899804168973, 95.67050716578869, 70.57848168277057, \
        42.75912300977666, 23.68220826178995];
std_theta_0 = [3.7292178915609497, 4.299262399062776, 4.165214790945542, \
        2.906634093761972, 2.161987679094327];

Ca = 0.1
len_ratio = 12.5
friction_ratio = 2.5

# direction == 1 : advancing
# direction == -1 : receding
direction = 1
ar = (1+direction)//2

# Mathieu 2003 (valid for small theta only?)
g_dir = lambda x : (1.0/9.0) * (x**3) - 0.00183985 * (x**4.5) + (1.845823*1e-6) * (x**12.258487) 
g_inv = lambda x : (9.0*x)**(1.0/3.0) + 0.0727387 * x - 0.0515388 * (x**2) + 0.00341336 * (x**3)

def cox_relation(eq_theta) :
    din_theta = np.rad2deg( g_inv( g_dir( np.deg2rad(eq_theta) ) + direction*Ca*np.log10(len_ratio) ) )
    if isnan(din_theta) :
        din_theta = 180.0*ar
    return din_theta

def mkt_relation(eq_theta) :
    din_theta = np.rad2deg( np.sqrt(2.0*direction*Ca*friction_ratio + np.deg2rad(eq_theta)**2) )
    if isnan(din_theta) :
        din_theta = 180.0*ar
    return din_theta

# Reading MD simulation results
time = array_from_file('/home/michele/densmap/ShearDynamic/Q2_Ca010/time.txt')
dt = 12.5
t_0 = 2000
idx_0 = np.abs( time-t_0 ).argmin()
avg_angle_md = []
std_angle_md = []
if direction == 1 :
    tl = []
    br = []
    tl.append( array_from_file('/home/michele/densmap/ShearDynamic/Q1_Ca010/angle_tl.txt')[idx_0:] )
    br.append( array_from_file('/home/michele/densmap/ShearDynamic/Q1_Ca010/angle_br.txt')[idx_0:] )
    tl.append( array_from_file('/home/michele/densmap/ShearDynamic/Q2_Ca010/angle_tl.txt')[idx_0:] )
    br.append( array_from_file('/home/michele/densmap/ShearDynamic/Q2_Ca010/angle_br.txt')[idx_0:] )
    tl.append( array_from_file('/home/michele/densmap/ShearDynamic/Q3_Ca010/angle_tl.txt')[idx_0:] )
    br.append( array_from_file('/home/michele/densmap/ShearDynamic/Q3_Ca010/angle_br.txt')[idx_0:] )
    avg_angle_md.append( np.mean(0.5*(tl[0]+br[0])) )
    avg_angle_md.append( np.mean(0.5*(tl[1]+br[1])) )
    avg_angle_md.append( np.mean(0.5*(tl[2]+br[2])) )
    std_angle_md.append( np.std(0.5*(tl[0]+br[0])) )
    std_angle_md.append( np.std(0.5*(tl[1]+br[1])) )
    std_angle_md.append( np.std(0.5*(tl[2]+br[2])) )
elif direction == -1 :
    tr = []
    bl = []
    tr.append( array_from_file('/home/michele/densmap/ShearDynamic/Q1_Ca010/angle_tr.txt')[idx_0:] )
    bl.append( array_from_file('/home/michele/densmap/ShearDynamic/Q1_Ca010/angle_bl.txt')[idx_0:] )
    tr.append( array_from_file('/home/michele/densmap/ShearDynamic/Q2_Ca010/angle_tr.txt')[idx_0:] )
    bl.append( array_from_file('/home/michele/densmap/ShearDynamic/Q2_Ca010/angle_bl.txt')[idx_0:] )
    tr.append( array_from_file('/home/michele/densmap/ShearDynamic/Q3_Ca010/angle_tr.txt')[idx_0:] )
    bl.append( array_from_file('/home/michele/densmap/ShearDynamic/Q3_Ca010/angle_bl.txt')[idx_0:] )
    avg_angle_md.append( np.mean(0.5*(tr[0]+bl[0])) )
    avg_angle_md.append( np.mean(0.5*(tr[1]+bl[1])) )
    avg_angle_md.append( np.mean(0.5*(tr[2]+bl[2])) )
    std_angle_md.append( np.std(0.5*(tr[0]+bl[0])) )
    std_angle_md.append( np.std(0.5*(tr[1]+bl[1])) )
    std_angle_md.append( np.std(0.5*(tr[2]+bl[2])) )

# Cast into np
avg_angle_md = np.array(avg_angle_md)
avg_theta_0 = np.array(avg_theta_0) 

# Optimize Cox and MKT parameters
def cox_fit(t, a_cox) :
    din_theta = np.rad2deg( g_inv( g_dir( np.deg2rad(t) ) + direction*Ca*np.log10(a_cox) ) )
    return din_theta
def mkt_fit(t, a_mkt) :
    din_theta = np.rad2deg( np.sqrt(2.0*direction*Ca*a_mkt + np.deg2rad(t)**2) )
    return din_theta
popt_cox, pcov_cox = curve_fit(cox_fit, avg_theta_0[0:3], avg_angle_md, p0=len_ratio)
popt_mkt, pcov_mkt = curve_fit(mkt_fit, avg_theta_0[0:3], avg_angle_md, p0=friction_ratio)
print("Fit length ratio: L/lambda = "+str(popt_cox[0]))
print("Fit friction ratio: mu_f/mu = "+str(popt_mkt[0]))

# Evaluate Cox / MKT
len_ratio = popt_cox[0]
friction_ratio = popt_mkt[0]
n_val = 300
theta_0_vec = np.linspace(0.0, 150.0, n_val)
theta_d_cox = np.zeros(n_val)
theta_d_mkt = np.zeros(n_val)
for k in range(n_val) :
    theta_d_cox[k] = cox_relation(theta_0_vec[k])
    theta_d_mkt[k] = mkt_relation(theta_0_vec[k])
theta_d_0 = []
theta_d_psd = []
theta_d_msd = []
for k in range(5) :
    theta_d_0.append( cox_relation(avg_theta_0[k]) )
    theta_d_psd.append( cox_relation(avg_theta_0[k]+std_theta_0[k]) )
    theta_d_msd.append( cox_relation(avg_theta_0[k]-std_theta_0[k]) )
print("Equilibrium CA:")
print(avg_theta_0)
print("Estimates for theta_d_0:")
print(theta_d_0)

# Plot Cox relation
plt.plot(theta_0_vec, theta_d_cox, 'b-', linewidth=2.0, label='Cox relation')

# Plot MKT relation
plt.plot(theta_0_vec, theta_d_mkt, 'm-.', linewidth=2.0, label='MKT')

# This plot is a bit messy: I should utilize error bars to encode info about standard deviation

# Plot analytical estimate (theta=94)
y_err_asym = [np.array(theta_d_0)-np.array(theta_d_msd), np.array(theta_d_psd)-np.array(theta_d_0)]
eb1 = plt.errorbar( avg_theta_0, theta_d_0, xerr=std_theta_0, yerr=y_err_asym, \
        ecolor='k', fmt='ks', elinewidth=1.5, capsize=5.0, capthick=1.75, ms=6.0, label='Analytical (Cox)' )
# eb1[-1][0].set_linestyle('--')
# eb1[-1][1].set_linestyle('--')

eb2 = plt.errorbar( avg_theta_0[0:3], avg_angle_md, yerr=std_angle_md, \
        ecolor='r', fmt='ro', elinewidth=2, capsize=7.5, capthick=2.5, ms=8.0, label='MD' )
eb2[-1][0].set_linestyle('--')

plt.legend(fontsize=17.5)

plt.title(r"Cox law and MKT, Ca="+str(Ca)+", L/$\lambda$="+"{:.3f}".format(len_ratio)+", $\mu_f/\mu$="+"{:.3f}".format(friction_ratio), fontsize=20.0)
plt.xlabel(r"$\theta_0$ [deg]", fontsize=15.0)
plt.ylabel(r"$\theta_d$ [deg]", fontsize=15.0)
plt.xticks(fontsize=15.0)
plt.yticks(fontsize=15.0)
plt.show()
