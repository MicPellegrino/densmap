import numpy as np
import matplotlib.pyplot as plt

import scipy.optimize as opt
import scipy.special as spc

# Estimate of b0

fact, _, _, _ = spc.airy(-1.0188)
b0 = -np.log(2.0*fact*(18**(1.0/3.0))*np.pi*1e-6)
print("b0="+str(b0))

def func_pow3(t, a) :
    return a*t**3

def func_impl(c, t, b) :
    return ( c - ((t**3)/9.0) / ( np.log(t*(c**(1.0/3.0))) + b ) )

def func_impl_p(c, t, b) :
    return ( 1 + ((t**3)/9.0) / ( c*( ( np.log(t*(c**(1.0/3.0))) + b )**2 ) ) )

"""
def func_egg(t, b, a) :
    return opt.fsolve( lambda x : func_impl(x, t, b), x0=func_pow3(t,a), \
            fprime = lambda x : func_impl_p(x, t, b) )
"""
def func_egg(t, b) :
    x0, _ = opt.bisect( lambda x : func_impl(x, t, b), a=0.0, b=0.3 )
    return x0

# Exclude the mosy hydrophobic case (thermostat issues)

# equil_contact_angles = np.array([124.86, 95.67, 70.58, 42.76, 23.68])
# equil_contact_angles = np.array([95.67, 70.58, 42.76, 23.68])
equil_contact_angles = np.deg2rad(np.array([95.0, 69.1, 38.8, 14.7]))

# last_steady = np.array([0.6, 0.25, 0.10, 0.05, 0.01])
# first_unsteady = np.array([0.9, 0.3, 0.20, 0.10, 0.05])
last_steady = np.array([0.25, 0.10, 0.05, 0.01])
first_unsteady = np.array([0.3, 0.20, 0.10, 0.05])

critical_estimate = 0.5*(last_steady + first_unsteady)

popt, pcov = opt.curve_fit(func_pow3, equil_contact_angles, critical_estimate, p0=1.0)
a = popt[0]
print("a="+str(a))
theta = np.deg2rad(np.linspace(0.0, 120.0, 150))
ca_cr_est = a*theta**3
plt.plot(np.rad2deg(theta), ca_cr_est, 'k-', label=r'$Ca_{cr}\sim a\cdot\theta_0^3$')

"""
popt, pcov = opt.curve_fit(lambda t, b : func_egg(t, b, a), \
        equil_contact_angles, critical_estimate, p0=b0)
"""
"""
popt, pcov = opt.curve_fit(func_egg, equil_contact_angles, critical_estimate, p0=b0)
b = popt[0]
print("b="+str(b))
ca_cr_egg = np.zeros(theta.shape)
for i in range(len(theta)):
    ca_cr_egg[i] = opt.fsolve(lambda x : func_impl(x, theta[i], b), x0=a*theta[i]**3)
plt.plot(np.rad2deg(theta), ca_cr_egg, 'k--', label='Eggers (2004)')
"""

"""
plt.plot( equil_contact_angles, first_unsteady, \
        'o', color='orange', markersize=12.0, label='first unsteady' )
plt.plot( equil_contact_angles, last_steady, \
        'o', color='green', markersize=12.0, label='last steady' )
plt.plot( equil_contact_angles, critical_estimate, \
        'x', color='black', markersize=17.5, markeredgewidth=2.5 )
"""
plt.plot( np.rad2deg(equil_contact_angles), first_unsteady, \
        'x', color='red', markersize=17.5, markeredgewidth=2.5, label='first unsteady' )
plt.plot( np.rad2deg(equil_contact_angles), last_steady, \
        'o', color='green', markersize=12.0, label='last steady' )

plt.title(r'Critical capillary number ($L_z\sim30$nm)', fontsize=40.0)
plt.legend(fontsize=30.0)
plt.xticks(fontsize=25.0)
plt.yticks(fontsize=25.0)
plt.xlabel('Contact angle [deg]', fontsize=37.5)
plt.ylabel('Capillary number [-1]', fontsize=37.5)
plt.xlim([0.0, 120.0])
plt.show()
