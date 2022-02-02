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

theta0    = dict()
theta     = dict()
ca        = dict()
theta_fit = dict()
err       = dict()

theta0['q1'] = 126.01
theta['q1'] = np.array([130.88, 131.49, 137.93, 142.13, 126.65, 126.06, 124.32, 125.42])
err['q1'] =  np.array([1.027, 0.3170, 0.3123, 0.2346, 1.745, 1.0524, 1.3838, 3.128])
ca['q1'] = np.array([0.15, 0.30, 0.60, 0.90, -0.15, -0.30, -0.60, -0.90])

theta0['q2'] = 94.9
theta['q2'] = np.array([100.29, 104.47, 106.63, 110.03, 116.12, 91.92, 89.29, 84.71, 80.10, 70.97])
err['q2'] =  np.array([3.4160, 0.9342, 0.7004, 0.7892, 4.4365, 5.1389, 1.5293,  2.194,  1.553, 6.9790])
ca['q2'] = np.array([0.05, 0.10, 0.15, 0.20, 0.25, -0.05, -0.10, -0.15, -0.20, -0.25])

theta0['q3'] = 70.5
theta['q3'] = np.array([78.32, 81.25, 84.77, 86.10, 87.19, 67.04, 64.84, 62.39, 55.35, 50.16])
err['q3'] =  np.array([0.6232, 0.9263, 1.578,  1.773, 2.925, 0.5128, 0.4506, 0.9719, 2.350, 4.213])
ca['q3'] = np.array([0.03, 0.05, 0.06, 0.08, 0.10, -0.03, -0.05, -0.06, -0.08, -0.10])

theta0['q4'] = 39.2
theta['q4'] = np.array([40.09, 43.36, 42.43, 36.54, 31.2, 29.34])
err['q4'] =  np.array([0.2937,  0.8878,  0.2764, 0.3322, 0.2743, 0.1396])
ca['q4'] = np.array([0.010, 0.015, 0.020, -0.010, -0.015, -0.020])

for l in theta0.keys():
   
    low_bound = min(theta[l])-max(err[l])
    upp_bound = max(theta[l])+max(err[l])
    theta_fit[l] = np.linspace(low_bound, upp_bound, 50)

cos = lambda t : np.cos(np.deg2rad(t))
sin = lambda t : np.sin(np.deg2rad(t))

"""
def lin_pf_formula(t, a_pf, t0) :
    return (3.0/np.sqrt(2.0))*((cos(t0)-cos(t))/sin(t))/a_pf
"""

"""
def lin_pf_formula(t, a_pf, beta, t0) :
    return 2.0*(cos(t0)-cos(t))*((1+beta*(cos(t0)-cos(t))**2)/a_pf)
"""

def lin_pf_formula(t, a_pf, beta, t0) :
    return 2.0*(cos(t0)-cos(t))/(a_pf*np.exp(beta*(0.5*sin(t)+cos(t))**2))

pf_fit  = dict()
muf_pf = dict()
beta = dict()
err_pf = dict()
err_beta = dict()
friction_ratio_0 = 0.1
beta_0 = 1

for l in theta0.keys() :

    # pf_fit[l]  = lambda t, a_pf : lin_pf_formula(t, a_pf, theta0[l])
    pf_fit[l]  = lambda t, a_pf, beta : lin_pf_formula(t, a_pf, beta, theta0[l])
    popt, pcov = opt.curve_fit(pf_fit[l], theta[l], ca[l], p0=(friction_ratio_0, beta_0))
    muf_pf[l] = popt[0]
    beta[l] = popt[1]
    err_pf[l] = np.sqrt( pcov[0,0]/len(ca[l]) )
    err_beta[l] = np.sqrt( pcov[1,1]/len(ca[l]) )

print("muf_pf")
print(muf_pf)

print("beta")
print(beta)

print("err_pf")
print(err_pf)

print("err_beta")
print(err_beta)

size_markers = 7.5
size_edges = 2.0
size_lines = 2.0

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

ax1.tick_params(axis='both', labelsize=22.5)
"""
ax1.plot( theta_fit['q1'], lin_pf_formula(theta_fit['q1'], muf_pf['q1'], theta0['q1']), \
        'k-.', linewidth=size_lines, label='fit' )
"""
ax1.plot( theta_fit['q1'], lin_pf_formula(theta_fit['q1'], muf_pf['q1'], beta['q1'], theta0['q1']), \
        'k-.', linewidth=size_lines, label='fit' )
ax1.plot( [min(theta_fit['q1']), max(theta_fit['q1'])], [0.0, 0.0], 'k:', label='equilibrium' )
ax1.plot( [theta0['q1'], theta0['q1']], [ca['q1'][-1], -ca['q1'][-1]], 'k:'  )
ax1.errorbar( theta['q1'], ca['q1'], xerr=err['q1'], \
        fmt='ko', elinewidth=2, markersize=size_markers, \
        markerfacecolor='none', markeredgewidth=size_edges, label='MD' )
ax1.text(min(theta['q1'])-2, -ca['q1'][-2], r'(a)    $\theta_0=127$°', fontsize=30.0, \
        bbox={'facecolor': 'white', 'alpha': 1.0, 'edgecolor':'none', 'pad': 10})

ax1.legend(fontsize=22.5, loc='lower right')
ax1.tick_params(axis='both', labelsize=22.5)
ax1.set_ylabel(r'$Ca$', fontsize=30.0)

"""
ax2.plot( theta_fit['q2'], lin_pf_formula(theta_fit['q2'], muf_pf['q2'], theta0['q2']), \
        'k-.', linewidth=size_lines, label='fit (adv.)' )
"""
ax2.plot( theta_fit['q2'], lin_pf_formula(theta_fit['q2'], muf_pf['q2'], beta['q2'], theta0['q2']), \
        'k-.', linewidth=size_lines, label='fit (adv.)' )
ax2.errorbar( theta['q2'], ca['q2'], xerr=err['q2'], \
        fmt='ko', elinewidth=2, markersize=size_markers, \
        markerfacecolor='none', markeredgewidth=size_edges, label='MD (adv.)' )
ax2.plot( [min(theta_fit['q2']), max(theta_fit['q2'])], [0.0, 0.0], 'k:'  )
ax2.plot( [theta0['q2'], theta0['q2']], [ca['q2'][-1], -ca['q2'][-1]], 'k:'  )
ax2.text(min(theta['q2']), -ca['q2'][-2], r'(b)    $\theta_0=95$°', fontsize=30.0, \
        bbox={'facecolor': 'white', 'alpha': 1.0, 'edgecolor':'none', 'pad': 10})

ax2.tick_params(axis='both', labelsize=22.5)

"""
ax3.plot( theta_fit['q3'], lin_pf_formula(theta_fit['q3'], muf_pf['q3'], theta0['q3']), \
        'k-.', linewidth=size_lines )
"""
ax3.plot( theta_fit['q3'], lin_pf_formula(theta_fit['q3'], muf_pf['q3'], beta['q3'], theta0['q3']), \
        'k-.', linewidth=size_lines )
ax3.errorbar( theta['q3'], ca['q3'], xerr=err['q3'], \
        fmt='ko', elinewidth=2, markersize=size_markers, \
        markerfacecolor='none', markeredgewidth=size_edges, label='MD (adv.)' )
ax3.plot( [min(theta_fit['q3']), max(theta_fit['q3'])], [0.0, 0.0], 'k:'  )
ax3.plot( [theta0['q3'], theta0['q3']], [ca['q3'][-1], -ca['q3'][-1]], 'k:'  )
ax3.text(min(theta['q3'])-3, -ca['q3'][-2], r'(c)    $\theta_0=69$°', fontsize=30.0, \
        bbox={'facecolor': 'white', 'alpha': 1.0, 'edgecolor':'none', 'pad': 10})

ax3.set_xlabel(r'$\theta$', fontsize=30.0)
ax3.set_ylabel(r'$Ca$', fontsize=30.0)
ax3.tick_params(axis='both', labelsize=22.5)

ax4.set_title(r'$q_4$', fontsize=30.0)
"""
ax4.plot( theta_fit['q4'], lin_pf_formula(theta_fit['q4'], muf_pf['q4'], theta0['q4']), \
        'k-.', linewidth=size_lines )
"""
ax4.plot( theta_fit['q4'], lin_pf_formula(theta_fit['q4'], muf_pf['q4'], beta['q4'],theta0['q4']), \
        'k-.', linewidth=size_lines )
ax4.errorbar( theta['q4'], ca['q4'], xerr=err['q4'], \
        fmt='ko', elinewidth=2, markersize=size_markers, \
        markerfacecolor='none', markeredgewidth=size_edges, label='MD (adv.)' )
ax4.plot( [min(theta_fit['q4']), max(theta_fit['q4'])], [0.0, 0.0], 'k:'  )
ax4.plot( [theta0['q4'], theta0['q4']], [ca['q4'][-1], -ca['q4'][-1]], 'k:'  )
ax4.text(min(theta['q4']), -ca['q4'][-2], r'(d)    $\theta_0=38$°', fontsize=30.0, \
        bbox={'facecolor': 'white', 'alpha': 1.0, 'edgecolor':'none', 'pad': 10})

ax4.set_xlabel(r'$\theta$', fontsize=30.0)
ax4.tick_params(axis='both', labelsize=22.5)

plt.show()
