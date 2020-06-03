import densmap as dm

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

def array_from_file( filename ):
    my_list = []
    with open(filename, 'r') as f:
        for line in f:
            my_list.append(float(line.split()[0]))
    return np.array(my_list)

time = array_from_file('Adv/Flat/time.txt')
spreading_radius = dict()
spreading_angle = dict()
charges = ['Q2','Q3','Q4','Q5']
q_vals = [0.60, 0.67, 0.74, 0.79]
q_max = max(q_vals)
q_diff = q_vals[-1]-q_vals[0]
for q in charges :
    if q=='Q5' :
        spreading_radius[q] = array_from_file('Adv/Flat/radius_c.txt')
        spreading_angle[q] = array_from_file('Adv/Flat/angle_r.txt')
    else :
        spreading_radius[q] = array_from_file('FlatSpreading/'+q+'/radius_c.txt')
        spreading_angle[q] = array_from_file('FlatSpreading/'+q+'/angle_r.txt')

for k in range(len(charges)):
    """
    plt.plot(time, spreading_radius[charges[k]], label='q='+str(q_vals[k]), \
    linewidth=2.00, c=cm.hot((q_max-q_vals[k])/(1.5*q_diff)))
    """
    plt.plot(time, spreading_angle[charges[k]], label='q='+str(q_vals[k]), \
    linewidth=2.00, c=cm.hot((q_max-q_vals[k])/(1.5*q_diff)))
plt.legend(fontsize=20.0)
plt.xticks(fontsize=20.0)
plt.yticks(fontsize=20.0)
plt.xlabel('t [ps]', fontsize=20.0)
# plt.ylabel(r'$R$ [nm]', fontsize=20.0)
plt.ylabel(r'$\theta$ [deg]', fontsize=20.0)
plt.xlim([0,max(time)])
plt.title('Comparison spreading curves', fontsize=20.0)
plt.show()
