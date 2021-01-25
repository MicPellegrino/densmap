import numpy as np
import matplotlib.pyplot as plt

equil_contact_angles = np.array([124.86, 95.67, 70.58, 42.76, 23.68])

last_steady = np.array([0.6, 0.25, 0.10, 0.05, 0.01])
first_unsteady = np.array([0.9, 0.3, 0.20, 0.10, 0.05])

critical_estimate = 0.5*(last_steady + first_unsteady)

plt.plot( equil_contact_angles, first_unsteady, \
        'o', color='orange', markersize=12.0, label='first unsteady' )
plt.plot( equil_contact_angles, last_steady, \
        'o', color='green', markersize=12.0, label='last steady' )
plt.plot( equil_contact_angles, critical_estimate, \
        'x', color='black', markersize=17.5, markeredgewidth=2.5 )
plt.title(r'Critical capillary number ($L_z\sim30$nm)', fontsize=30.0)
plt.legend(fontsize=20.0)
plt.xticks(fontsize=20.0)
plt.yticks(fontsize=20.0)
plt.xlabel('Contact angle [deg]', fontsize=30.0)
plt.ylabel('Capillary number [-1]', fontsize=30.0)
plt.show()
