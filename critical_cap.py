import numpy as np
import matplotlib.pyplot as plt

import scipy.optimize as opt
import scipy.special as spc

# Estimate of b0

g0 = 1
print("g0="+str(g0))

def func_pow3(t, a) :
    return a*t**3

def func_impl(d, t, g) :
    return d*np.log(((d**(1.0/3.0))*(t**2))/g)-1.0

equil_contact_angles = np.deg2rad(np.array([127.4, 95.0, 69.1, 38.8, 14.7]))

last_steady = np.array([0.6, 0.25, 0.10, 0.03, 0.01])
first_unsteady = np.array([0.9, 0.3, 0.15, 0.05, 0.05])

critical_estimate = 0.5*(last_steady + first_unsteady)

popt, pcov = opt.curve_fit(func_pow3, equil_contact_angles, critical_estimate, p0=1.0)
a = popt[0]
print("a="+str(a))
theta = np.deg2rad(np.linspace(10.0, 130.0, 100))
ca_cr_est = a*theta**3
plt.plot(np.rad2deg(theta), ca_cr_est, 'k--', linewidth=2.5, label=r'$Ca_{cr}\sim a_0\cdot\theta_0^3$')

d_cr_egg = np.zeros(theta.shape)
for i in range(len(theta)):
    d_cr_egg[i] = opt.fsolve(lambda x : func_impl(x, theta[i], g0), x0=a*theta[i]**3)
    # d_cr_egg[i] = opt.bisect(lambda x : func_impl(x, theta[i], g0), a=0.1, b=1.0) 
ca_cr_egg = d_cr_egg*(theta**3)/9.0
plt.plot(np.rad2deg(theta), ca_cr_egg, 'k--', label='Eggers (2004)')

"""
plt.plot( equil_contact_angles, first_unsteady, \
        'o', color='orange', markersize=12.0, label='first unsteady' )
plt.plot( equil_contact_angles, last_steady, \
        'o', color='green', markersize=12.0, label='last steady' )
plt.plot( equil_contact_angles, critical_estimate, \
        'x', color='black', markersize=17.5, markeredgewidth=2.5 )
"""
plt.plot( np.rad2deg(equil_contact_angles), first_unsteady, \
        'x', color='red', markersize=17.5, markeredgewidth=3.5, label='first unsteady' )
plt.plot( np.rad2deg(equil_contact_angles), last_steady, \
        'o', color='green', markersize=13.5, label='last steady' )

plt.title(r'Critical capillary number ($L_z\sim30$nm, $\mu_l/\mu_v=\infty$)', fontsize=40.0)
plt.legend(fontsize=30.0)
plt.xticks(fontsize=27.5)
plt.yticks(fontsize=27.5)
plt.xlabel('Contact angle [deg]', fontsize=37.5)
plt.ylabel('Capillary number [-1]', fontsize=37.5)
plt.xlim([0.0, 130.0])
plt.show()
