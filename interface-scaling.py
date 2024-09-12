import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

mark_size = 25
mark_width = 5
line_width = 4

axis_font = 45
legend_font = 40
ticks_font = 35

### ??? ###

L_par = [2.82929, 
        4.80196, 
        6.74132, 
        8.65852]
"""
w_sol = np.array([0.3948822864858782, 
        0.448761717073666,
        0.4815323651164674,
        0.6026108704093641])
w_but = np.array([0.4155157129351387, 
        0.4683611067720915,
        0.49996284899250343,
        0.6159208891138912  ])
w_tot = np.array([0.3119806465932117, 
        0.36839642127258965,
        0.4091056315619314,
        0.5495929345471884])
"""
w_sol = np.array([0.3705438975826468, 
                0.4549488344140541,
                0.48248389119679425,
                0.5860751821568618])
w_but = np.array([0.3902221774019854, 
                0.47377679557584457,
                0.5001258106608445,
                0.6004121239262584  ])
w_tot = np.array([0.290940460983936, 
                0.37787757314749504,
                0.4143472816599415,
                0.5294278138457117])

logL_par = np.log(L_par)
w2_sol = w_sol*w_sol
w2_but = w_but*w_but
w2_tot = w_tot*w_tot

gamma = 0.5*63.4969     # bar nm
T = 300                 # K
kb = 1.38e-1            # bar nm^3 K^-1
twopi = 2*np.pi

prefac = (kb*T)/(twopi*gamma)
f = lambda logL, a : prefac*(logL-a)

# f = lambda logL, a, gamma : (kb*T)/(twopi*gamma)*(logL-a)

popt_s, _ = opt.curve_fit(f, logL_par, w2_sol)
popt_b, _ = opt.curve_fit(f, logL_par, w2_but)
popt_t, _ = opt.curve_fit(f, logL_par, w2_tot)

lb_s = np.exp(popt_s[0])
print('lb_s = '+str(lb_s)+' nm')
lb_b = np.exp(popt_b[0])
print('lb_b = '+str(lb_b)+' nm')
lb_t = np.exp(popt_t[0])
print('lb_t = '+str(lb_t)+' nm')

"""
gamma_s = np.exp(popt_s[1])
print('gamma_s = '+str(gamma_s)+' bar*nm')
gamma_b = np.exp(popt_b[1])
print('gamma_b = '+str(gamma_b)+' bar*nm')
gamma_t = np.exp(popt_t[1])
print('gamma_t = '+str(gamma_t)+' bar*nm')
"""

fig, ax = plt.subplots()
ax.plot(logL_par, w2_sol, 'bo', ms=mark_size, markerfacecolor="None",
         markeredgecolor='blue', markeredgewidth=mark_width, label='water')
ax.plot(logL_par, f(logL_par, *popt_s), 'b--', linewidth=line_width)
ax.plot(logL_par, w2_but, 'rx', ms=mark_size, markerfacecolor="None",
         markeredgecolor='red', markeredgewidth=mark_width, label='butanol')
ax.plot(logL_par, f(logL_par, *popt_b), 'r--', linewidth=line_width)
ax.plot(logL_par, w2_tot, 'ks', ms=mark_size, markerfacecolor="None",
         markeredgecolor='black', markeredgewidth=mark_width, label='total')
ax.plot(logL_par, f(logL_par, *popt_t), 'k--', linewidth=line_width)
ax.legend(fontsize=legend_font)
ax.set_xlabel(r'log($L_\parallel$/$l_b$) []', fontsize=axis_font)
ax.set_ylabel(r'$\varepsilon_{th}^2$ [nm$^2$]', fontsize=axis_font)
ax.set_box_aspect(0.75)
plt.tick_params(labelsize=ticks_font)
plt.show()

"""
plt.loglog(L_par, w_sol, 'bo', ms=15, markerfacecolor="None",
         markeredgecolor='blue', markeredgewidth=4)
plt.loglog(L_par, w_but, 'rx', ms=17.5, markerfacecolor="None",
         markeredgecolor='red', markeredgewidth=4)
plt.loglog(L_par, w_tot, 'ks', ms=15, markerfacecolor="None",
         markeredgecolor='black', markeredgewidth=4)
plt.tick_params(labelsize=30)
plt.xlabel(r'$L_\parallel$ [nm]', fontsize=40)
plt.ylabel(r'$\varepsilon_{th}$ [nm]', fontsize=40)
plt.show()
"""