import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.special as spc

def func_impl(d, t, g) :
    return d*np.log(((d**(1.0/3.0))*(t**2))/g)-1.0
g0 = 10
a0 = 0.06719734261355896 

theta = np.deg2rad(np.linspace(10.0, 130.0, 100))
d_cr_egg = np.zeros(theta.shape)
for i in range(len(theta)):
    d_cr_egg[i] = opt.fsolve(lambda x : func_impl(x, theta[i], g0), x0=a0*theta[i]**3)

c_cr_egg = d_cr_egg*(theta**3)/9.0

plt.plot(np.rad2deg(theta), c_cr_egg, 'k--', label='Eggers (2004)')
plt.title(r'Critical capillary number ($L_z\sim30$nm, $\mu_l/\mu_v=\infty$)', fontsize=40.0)
plt.legend(fontsize=30.0)
plt.xticks(fontsize=27.5)
plt.yticks(fontsize=27.5)
plt.xlabel('Contact angle [deg]', fontsize=37.5)
plt.ylabel(r'$Ca$ [1]', fontsize=37.5)
plt.xlim([0.0, 130.0])
plt.show()
