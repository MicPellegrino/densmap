import numpy as np
import matplotlib.pyplot as plt

forcing = np.array([0.0012444, 39.44e-4, 0.0125, 39.44e-3]) # [nm/ps^2]
mean_m1 = np.array([1404.41, 1425.25, 1450.9, 1625.25])     # [1/(Pa*s)]
err_m1 = np.array([43, 13, 5.7, 5.4])                       # [1/(Pa*s)]

mean_visc = (1e3)/mean_m1
mean_m = (1e3)/(mean_m1+err_m1)
mean_p = (1e3)/(mean_m1-err_m1)

# fig, ax = plt.plot( )
err_visc = [(mean_visc-mean_m), (mean_p-mean_visc)]

plt.errorbar(forcing, mean_visc, yerr=err_visc, fmt='ko', ecolor='k', markersize=7.5)
plt.xscale("log")
plt.title("Viscosity estimate (300K, 1bar) (semilogx)", fontsize=25.0)
plt.xlabel("Forcing [nm/ps^2]", fontsize=20.0)
plt.ylabel("Viscosity [mPa*s]", fontsize=20.0)
plt.xticks(fontsize=15.0)
plt.yticks(fontsize=15.0)
plt.ylim([0.5, 0.8])

plt.show()
