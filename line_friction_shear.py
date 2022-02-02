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
theta_adv = dict()
theta_rec = dict()
ca        = dict()
theta_rec_fit = dict()
theta_adv_fit = dict()
err_adv = dict()
err_rec = dict()

theta0['q1'] = 126.01
theta_rec['q1'] = np.array([126.65, 126.06, 124.32, 125.42])
theta_adv['q1'] = np.array([130.88, 131.49, 137.93, 142.13])
# err_adv['q1'] =  np.array([1.027, 0.8453, 1.430, 1.371])
# err_rec['q1'] =  np.array([1.745, 2.414,  3.128, 3.128])
err_adv['q1'] =  np.array([1.027, 0.3170, 0.3123, 0.2346])
err_rec['q1'] =  np.array([1.745, 1.0524, 1.3838, 3.128])
ca['q1'] = np.array([0.15, 0.30, 0.60, 0.90])

theta0['q2'] = 94.9
theta_rec['q2'] = np.array([91.92, 89.29, 84.71, 80.10, 70.97])
theta_adv['q2'] = np.array([100.29, 104.47, 106.63, 110.03, 116.12])
# err_adv['q2'] =  np.array([0.8453, 0.9123, 0.7004, 0.7892, 1.248])
# err_rec['q2'] =  np.array([0.7467, 1.302,  2.194,  1.553,  2.381])
err_adv['q2'] =  np.array([3.4160, 0.9342, 0.7004, 0.7892, 4.4365])
err_rec['q2'] =  np.array([5.1389, 1.5293,  2.194,  1.553, 6.9790])
ca['q2'] = np.array([0.05, 0.10, 0.15, 0.20, 0.25])

theta0['q3'] = 70.5
theta_rec['q3'] = np.array([67.04, 64.84, 62.39, 55.35, 50.16])
theta_adv['q3'] = np.array([78.32, 81.25, 84.77, 86.10, 87.19])
# err_adv['q3'] =  np.array([0.5283, 0.9263, 1.578,  1.773, 2.925])
# err_rec['q3'] =  np.array([0.7760, 0.4506, 0.9719, 2.350, 4.213])
err_adv['q3'] =  np.array([0.6232, 0.9263, 1.578,  1.773, 2.925])
err_rec['q3'] =  np.array([0.5128, 0.4506, 0.9719, 2.350, 4.213])
ca['q3'] = np.array([0.03, 0.05, 0.06, 0.08, 0.10])

theta0['q4'] = 39.2
theta_rec['q4'] = np.array([36.54, 31.2, 29.34])
theta_adv['q4'] = np.array([40.09, 43.36, 42.43])
# err_adv['q4'] =  np.array([1.634,  1.151,  2.880])
# err_rec['q4'] =  np.array([0.7367, 0.4542, 0.8947])
err_adv['q4'] =  np.array([0.2937,  0.8878,  0.2764])
err_rec['q4'] =  np.array([0.3322, 0.2743, 0.1396])
ca['q4'] = np.array([0.010, 0.015, 0.020])

for l in theta0.keys():
   
    low_bound = min(theta_rec[l])
    upp_bound = max(theta_adv[l])
    theta_rec_fit[l] = np.linspace(low_bound, theta0[l], 25)
    theta_adv_fit[l] = np.linspace(theta0[l], upp_bound, 25)


cos = lambda t : np.cos(np.deg2rad(t))
sin = lambda t : np.sin(np.deg2rad(t))

def lin_pf_formula(t, a_pf, t0) :
    return (3.0/np.sqrt(2.0))*((cos(t0)-cos(t))/sin(t))/a_pf

"""
def lin_pf_formula(t, a_pf, t0) :
    return 2.0*(cos(t0)-cos(t))/a_pf
"""

mkt_fit = dict()
pf_fit  = dict()
muf_mkt_adv = dict()
muf_mkt_rec = dict()
muf_pf_adv = dict()
muf_pf_rec = dict()
err_mkt_adv = dict()
err_mkt_rec = dict()
err_pf_adv = dict()
err_pf_rec = dict()
friction_ratio_0 = 0.1

for l in theta0.keys() :

    pf_fit[l]  = lambda t, a_pf : lin_pf_formula(t, a_pf, theta0[l])
    popt_adv, pcov_adv = opt.curve_fit(pf_fit[l], theta_adv[l], ca[l], p0=friction_ratio_0)
    muf_pf_adv[l] = popt_adv[0]
    err_pf_adv[l] = np.sqrt( pcov_adv[0,0]/len(ca[l]) )
    popt_rec, pcov_rec = opt.curve_fit(pf_fit[l], theta_rec[l], -ca[l], p0=friction_ratio_0)
    muf_pf_rec[l] = popt_rec[0]
    err_pf_rec[l] = np.sqrt( pcov_rec[0,0]/len(ca[l]) )

print("muf_pf_adv")
print(muf_pf_adv)

print("err_pf_adv")
print(err_pf_adv)

print("muf_pf_rec")
print(muf_pf_rec)

print("err_pf_rec")
print(err_pf_rec)

size_markers = 7.5
size_edges = 2.0
size_lines = 2.0

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

ax1.set_ylabel(r'$Ca$', fontsize=30.0)
ax1.tick_params(axis='both', labelsize=22.5)
ax1.plot( theta_adv_fit['q1'], lin_pf_formula(theta_adv_fit['q1'], muf_pf_adv['q1'], theta0['q1']), \
        'r-.', linewidth=size_lines, label='fit (adv.)' )
ax1.plot( theta_rec_fit['q1'], lin_pf_formula(theta_rec_fit['q1'], muf_pf_rec['q1'], theta0['q1']), \
        'b--', linewidth=size_lines, label='fit (rec.)' )
ax1.plot( [min(theta_rec_fit['q1']), max(theta_adv_fit['q1'])], [0.0, 0.0], 'k:', label='equilibrium' )
ax1.plot( [theta0['q1'], theta0['q1']], [-ca['q1'][-1], ca['q1'][-1]], 'k:'  )
# ax1.plot( theta_adv['q1'], ca['q1'], 'rs', markersize=size_markers, label='MD (adv.)' )
# ax1.plot( theta_rec['q1'], -ca['q1'], 'bD', markersize=size_markers, label='MD (rec.)' )
ax1.errorbar( theta_adv['q1'], ca['q1'], xerr=err_adv['q1'], \
        fmt='rs', elinewidth=2, markersize=size_markers, \
        markerfacecolor='none', markeredgewidth=size_edges, label='MD (adv.)' )
ax1.errorbar( theta_rec['q1'], -ca['q1'], xerr=err_rec['q1'], \
        fmt='bD', elinewidth=2, markersize=size_markers, \
        markerfacecolor='none', markeredgewidth=size_edges, label='MD (rec.)' )
ax1.tick_params(axis='both', labelsize=22.5)
ax1.legend(fontsize=22.5, loc='lower right')
# ax1.text(min(theta_rec_fit['q1']), ca['q1'][-2], r'$q_1$', fontsize=22.5)
ax1.text(min(theta_rec_fit['q1'])-2, ca['q1'][-2], r'(a)    $\theta_0=127$°', fontsize=30.0, \
        bbox={'facecolor': 'white', 'alpha': 1.0, 'edgecolor':'none', 'pad': 10})
# ax1.set_aspect(1 / ax1.get_data_ratio())

ax2.plot( theta_adv_fit['q2'], lin_pf_formula(theta_adv_fit['q2'], muf_pf_adv['q2'], theta0['q2']), \
        'r-.', linewidth=size_lines, label='fit (adv.)' )
ax2.plot( theta_rec_fit['q2'], lin_pf_formula(theta_rec_fit['q2'], muf_pf_rec['q2'], theta0['q2']), \
        'b--', linewidth=size_lines, label='fit (rec.)' )
# ax2.plot( theta_adv['q2'], ca['q2'], 'rs', markersize=size_markers, label='MD (adv.)' )
# ax2.plot( theta_rec['q2'], -ca['q2'], 'bD', markersize=size_markers, label='MD (rec.)' )
ax2.errorbar( theta_adv['q2'], ca['q2'], xerr=err_adv['q2'], \
        fmt='rs', elinewidth=2, markersize=size_markers, \
        markerfacecolor='none', markeredgewidth=size_edges, label='MD (adv.)' )
ax2.errorbar( theta_rec['q2'], -ca['q2'], xerr=err_rec['q2'], \
        fmt='bD', elinewidth=2, markersize=size_markers, \
        markerfacecolor='none', markeredgewidth=size_edges, label='MD (rec.)' )
ax2.plot( [min(theta_rec_fit['q2']), max(theta_adv_fit['q2'])], [0.0, 0.0], 'k:'  )
ax2.plot( [theta0['q2'], theta0['q2']], [-ca['q2'][-1], ca['q2'][-1]], 'k:'  )
# ax2.set_xlabel(r'$\theta$', fontsize=20.0)
# ax2.set_ylabel(r'$Ca$', fontsize=20.0)
ax2.tick_params(axis='both', labelsize=22.5)
# ax2.legend(fontsize=15.0)
# ax2.text(min(theta_rec_fit['q2']), ca['q2'][-2], r'$q_2$', fontsize=22.5)
ax2.text(min(theta_rec_fit['q2']), ca['q2'][-2], r'(b)    $\theta_0=95$°', fontsize=30.0, \
        bbox={'facecolor': 'white', 'alpha': 1.0, 'edgecolor':'none', 'pad': 10})
# ax2.set_aspect(1 / ax2.get_data_ratio())

ax3.plot( theta_adv_fit['q3'], lin_pf_formula(theta_adv_fit['q3'], muf_pf_adv['q3'], theta0['q3']), \
        'r-.', linewidth=size_lines )
ax3.plot( theta_rec_fit['q3'], lin_pf_formula(theta_rec_fit['q3'], muf_pf_rec['q3'], theta0['q3']), \
        'b--', linewidth=size_lines )
# ax3.plot( theta_adv['q3'], ca['q3'], 'rs', markersize=size_markers )
# ax3.plot( theta_rec['q3'], -ca['q3'], 'bD', markersize=size_markers )
ax3.errorbar( theta_adv['q3'], ca['q3'], xerr=err_adv['q3'], \
        fmt='rs', elinewidth=2, markersize=size_markers, \
        markerfacecolor='none', markeredgewidth=size_edges, label='MD (adv.)' )
ax3.errorbar( theta_rec['q3'], -ca['q3'], xerr=err_rec['q3'], \
        fmt='bD', elinewidth=2, markersize=size_markers, \
        markerfacecolor='none', markeredgewidth=size_edges, label='MD (rec.)' )
ax3.plot( [min(theta_rec_fit['q3']), max(theta_adv_fit['q3'])], [0.0, 0.0], 'k:'  )
ax3.plot( [theta0['q3'], theta0['q3']], [-ca['q3'][-1], ca['q3'][-1]], 'k:'  )
ax3.set_xlabel(r'$\theta$', fontsize=30.0)
ax3.set_ylabel(r'$Ca$', fontsize=30.0)
ax3.tick_params(axis='both', labelsize=22.5)
# ax3.text(min(theta_rec_fit['q3']), ca['q3'][-2], r'$q_3$', fontsize=22.5)
ax3.text(min(theta_rec_fit['q3'])-3, ca['q3'][-2], r'(c)    $\theta_0=69$°', fontsize=30.0, \
        bbox={'facecolor': 'white', 'alpha': 1.0, 'edgecolor':'none', 'pad': 10})
# ax3.set_aspect(1 / ax3.get_data_ratio())

ax4.set_title(r'$q_4$', fontsize=30.0)
ax4.plot( theta_adv_fit['q4'], lin_pf_formula(theta_adv_fit['q4'], muf_pf_adv['q4'], theta0['q4']), \
        'r-.', linewidth=size_lines )
ax4.plot( theta_rec_fit['q4'], lin_pf_formula(theta_rec_fit['q4'], muf_pf_rec['q4'], theta0['q4']), \
        'b--', linewidth=size_lines )
# ax4.plot( theta_adv['q4'], ca['q4'], 'rs' , markersize=size_markers)
# ax4.plot( theta_rec['q4'], -ca['q4'], 'bD' , markersize=size_markers)
ax4.errorbar( theta_adv['q4'], ca['q4'], xerr=err_adv['q4'], \
        fmt='rs', elinewidth=2, markersize=size_markers, \
        markerfacecolor='none', markeredgewidth=size_edges, label='MD (adv.)' )
ax4.errorbar( theta_rec['q4'], -ca['q4'], xerr=err_rec['q4'], \
        fmt='bD', elinewidth=2, markersize=size_markers, \
        markerfacecolor='none', markeredgewidth=size_edges, label='MD (rec.)' )
ax4.plot( [min(theta_rec_fit['q4']), max(theta_adv_fit['q4'])], [0.0, 0.0], 'k:'  )
ax4.plot( [theta0['q4'], theta0['q4']], [-ca['q4'][-1], ca['q4'][-1]], 'k:'  )
ax4.set_xlabel(r'$\theta$', fontsize=30.0)
# ax4.set_ylabel(r'$Ca$', fontsize=20.0)
ax4.tick_params(axis='both', labelsize=22.5)
# ax4.text(min(theta_rec_fit['q4']), ca['q4'][-2], r'$q_4$', fontsize=22.5)
ax4.text(min(theta_rec_fit['q4']), ca['q4'][-2], r'(d)    $\theta_0=38$°', fontsize=30.0, \
        bbox={'facecolor': 'white', 'alpha': 1.0, 'edgecolor':'none', 'pad': 10})
# ax4.set_aspect(1 / ax4.get_data_ratio())

plt.show()
