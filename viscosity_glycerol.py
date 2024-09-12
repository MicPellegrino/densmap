import numpy as np
import matplotlib.pyplot as plt

ms = 15
cs = 10
elw = 3.0
ct = 2.75

# Viscosity out of equilibrium MD, thremo=1ps
"""
vis_est = np.array([0.66, 1.20, 2.33, 7.10, 44.8, 899])
vis_err = np.array([0.01, 0.02, 0.06, 0.20, 1.10, 27.5])
mass_fr = np.array([0.0,  0.2,  0.4,  0.6,  0.8,  1.0])
"""
# Thermo=2ps, apart alpha_g=0.6 and alpha_g=1.0
vis_est = np.array([0.69, 1.33, 2.51, 7.10, 45.7, 899])
vis_err = np.array([0.01, 0.02, 0.04, 0.20, 1.11, 27.5])
mass_fr = np.array([0.0,  0.2,  0.4,  0.6,  0.8,  1.0])

forcing = dict()
viscosity_avg = dict()
viscosity_std = dict()

experimental = dict()
einstein = dict()
ein_err = dict()

fmt = dict()
fmte = dict()

tag = dict()

# Have a function that does dictionary-dictionary operations ...

eta_h2o = vis_est[0]

forcing['0p']           = [ 1e-2,       1e-3,   1e-4    ]
viscosity_avg['0p']     = [ 0.68,       0.68,   0.71    ]
viscosity_std['0p']     = [ 0.002,      0.01,   0.05    ]
fmt['0p']               = 'cv'
fmte['0p']              = 'c--'
tag['0p']               = r'$\alpha_g = 0.0$'
einstein['0p']          = vis_est[0]
ein_err['0p']           = vis_err[0]

forcing['20p']          = [ 1e-2,       1e-3,   1e-4    ]
viscosity_avg['20p']    = [ 1.11,       1.10,   1.05    ]
viscosity_std['20p']    = [ 0.01,       0.02,   0.05    ]
fmt['20p']              = 'mH'
fmte['20p']             = 'm--'
tag['20p']              = r'$\alpha_g = 0.2$'
einstein['20p']         = vis_est[1]
ein_err['20p']          = vis_err[1]

forcing['40p']          = [ 1e-2,       1e-3,   1e-4    ]
viscosity_avg['40p']    = [ 2.28,       2.52,   1.75    ]
viscosity_std['40p']    = [ 0.02,       0.05,   0.11    ]
fmt['40p']              = 'bo'
# experimental['40p']   = 4.0
fmte['40p']             = 'b--'
tag['40p']              = r'$\alpha_g = 0.4$'
einstein['40p']         = vis_est[2]
ein_err['40p']          = vis_err[2]

forcing['60p']          = [ 1e-2,       1e-3,   1e-4    ]
viscosity_avg['60p']    = [ 6.62,       6.23,   9.76    ]
viscosity_std['60p']    = [ 0.05,       0.18,   1.86    ]
fmt['60p']              = 'rD'
# experimental['60p']   = 11.0
fmte['60p']             = 'r--'
tag['60p']              = r'$\alpha_g = 0.6$'
einstein['60p']         = vis_est[3]
ein_err['60p']          = vis_err[3]

forcing['80p']          = [ 1e-2,       1e-3    ]
viscosity_avg['80p']    = [ 36.2,       43.4    ]
viscosity_std['80p']    = [ 0.57,       2.19    ]
fmt['80p']              = 'gs'
# experimental['80p']   = 60.0
fmte['80p']             = 'g--'
tag['80p']              = r'$\alpha_g = 0.8$'
einstein['80p']         = vis_est[4]
ein_err['80p']          = vis_err[4]

forcing['100p']         = [ 1e-2,       ]
viscosity_avg['100p']   = [ 894,        ]
viscosity_std['100p']   = [ 38.9,       ]
fmt['100p']             = 'kx'
fmte['100p']            = 'k--'
tag['100p']             = r'$\alpha_g = 1.0$'
einstein['100p']        = vis_est[5]
ein_err['100p']         = vis_err[5]

fig, (ax1,ax2) = plt.subplots(1,2)

for k in forcing.keys() :

    viscosity_avg[k] = np.array(viscosity_avg[k])
    viscosity_std[k] = np.array(viscosity_std[k])

    if k == '100p' :
        ax1.errorbar(forcing[k], viscosity_avg[k], yerr=viscosity_std[k], fmt=fmt[k], markeredgewidth=elw,
            markersize=ms, capsize=cs, elinewidth=elw, capthick=ct, label=tag[k])
        ax1.text(0.15*((1e-4)+(1e-3)), 0.75*einstein[k], tag[k], fontsize=22.5)

    else :
        ax1.errorbar(forcing[k], viscosity_avg[k], yerr=viscosity_std[k], fmt=fmt[k], markeredgewidth=elw,
            markersize=ms, capsize=cs, elinewidth=elw, capthick=ct, markerfacecolor='w', label=tag[k])
        ax1.text(0.15*((1e-4)+(1e-3)), 1.125*einstein[k], tag[k], fontsize=22.5)

    ax1.plot([1e-4, 1e-2], [einstein[k], einstein[k]], fmte[k], linewidth=2.75)
    ax1.fill_between([1e-4, 1e-2], [einstein[k]+ein_err[k], einstein[k]+ein_err[k]], 
        [einstein[k]-ein_err[k], einstein[k]-ein_err[k]], color=fmte[k][0], alpha=0.5)

print(viscosity_avg)

# ax.plot([1e-4, 1e-4], [10, 10], 'k--', linewidth=2.75, label="experimental")
ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.tick_params(axis='both', labelsize=25.0)
ax1.set_ylabel(r'$\eta$ [cP]', fontsize=30.0)
ax1.set_xlabel(r'$\xi$ [nm/ps$^2$]', fontsize=30.0)
# ax1.legend(fontsize=20)
# ax1.set_box_aspect(1)
# plt.show()

# Experimental data (Segur and Oberstar 1951)
mass_fr_exp = [0.0, 0.09971, 0.19885, 0.298, 0.39946, 0.49865, 0.5973, 0.64778, 0.66729, 0.69828, 
        0.74707, 0.7976, 0.84872, 0.897, 0.90678, 0.91714, 0.92921, 0.93842, 0.9482, 0.95798, 0.96777, 0.97756, 0.98734,  1.0 ]
beta_exp = [0.0, 0.064235, 0.123732, 0.174936, 0.219921, 0.253653, 0.272577, 0.276116, 0.274926, 
        0.271067, 0.258614, 0.236092, 0.2035, 0.157286, 0.145141, 0.130034, 0.116408, 0.102486, 0.087676, 0.072569, 0.05539, 0.036137, 0.018957, 0.0 ]

# Experimental data (Shankar et al 1994)


beta = np.log(vis_est/vis_est[-1])/np.log(vis_est[0]/vis_est[-1])-(1-mass_fr)
beta_p = np.log((vis_est+vis_err)/(vis_est[-1]+vis_err[-1])) / np.log((vis_est[0]+vis_err[0])/(vis_est[-1]+vis_err[-1])) - (1-mass_fr)
beta_m = np.log((vis_est-vis_err)/(vis_est[-1]-vis_err[-1])) / np.log((vis_est[0]-vis_err[0])/(vis_est[-1]-vis_err[-1])) - (1-mass_fr)

import scipy.optimize as opt

f_beta_fit = lambda x, a, b : (a*b*x*(1-x))/(a*x+b*(1-x))
popt, pcov = opt.curve_fit(f_beta_fit, mass_fr, beta)

mass_vec = np.linspace(0,1,200)
beta_fit = f_beta_fit(mass_vec, *popt)

a = popt[0]
b = popt[1]
f_visco = lambda x : (a*b*x*(1-x))/(a*x+b*(1-x)) + (1-x)
print("#######################")
alpha_g_eff = 0.720+0.037
eta_eff = vis_est[-1] * (vis_est[0]/vis_est[-1])**(f_visco(alpha_g_eff))
print("eta_eff = "+str(eta_eff))
print("#######################")

# ax2.errorbar(mass_fr, beta, yerr=beta_p-beta_m, fmt='rs', markersize=0.4*ms, capsize=cs, elinewidth=elw, capthick=ct, label="eq. MD")
ax2.plot(mass_vec, beta_fit, 'b-', linewidth=3.5, label='Cheng (2008)')
ax2.plot(mass_fr_exp, beta_exp, 'ko', markersize=0.75*ms, linewidth=2.0, markeredgewidth=2.5, mfc='none', label='Segur and Oberstar (1951)')
ax2.plot(mass_fr, beta, 'rs', markersize=ms, markeredgewidth=3.0, markerfacecolor='w', label="equilibrium MD")
# ax2.plot([-0.025,1.025], [0,0], 'k--', linewidth=2.0)
ax2.set_xlim([-0.025,1.025])
ax2.set_ylim([0.0, 0.3])
ax2.legend(fontsize=20, loc='lower center')
ax2.set_xlabel(r"$\alpha_g$", fontsize=30)
ax2.set_ylabel(r"$\beta$", fontsize=30)
ax2.tick_params(axis='both', labelsize=25.0)
plt.subplots_adjust(left=0.075, bottom=0.1, right=0.975, top=0.9, wspace=0.25, hspace=0.1)
plt.show()