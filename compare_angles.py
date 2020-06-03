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

mode = 'REC'

if mode == 'ADV' :
    time = array_from_file('Adv/Flat/time.txt')
    angle_f = array_from_file('Adv/Flat/angle_r.txt')
    angle_h2 = array_from_file('Adv/Height2/angle_r.txt')
    angle_h1 = array_from_file('Adv/Height1/angle_r.txt')
    angle_w1 = array_from_file('Adv/Wave1/angle_r.txt')
    angle_w2 = array_from_file('Adv/Wave2/angle_r.txt')
    angle_w3 = array_from_file('Adv/Wave3/angle_r.txt')
    angle_w4 = array_from_file('Adv/Wave4/angle_r.txt')
    angle_w5 = array_from_file('Adv/Wave5/angle_r.txt')

if mode == 'REC' :
    time = array_from_file('Rec/Flat/time.txt')
    angle_h2 = array_from_file('Rec/Height2/angle_r.txt')
    angle_h1 = array_from_file('Rec/Height1/angle_r.txt')
    angle_f = array_from_file('Rec/Flat/angle_r.txt')
    angle_w1 = array_from_file('Rec/Wave1/angle_r.txt')
    angle_w2 = array_from_file('Rec/Wave2/angle_r.txt')
    angle_w3 = array_from_file('Rec/Wave3/angle_r.txt')
    angle_w4 = array_from_file('Rec/Wave4/angle_r.txt')
    angle_w5 = array_from_file('Rec/Wave5/angle_r.txt')

if mode == 'ADV' :
    angle_h2 = angle_h2-angle_h2[0]
    angle_h1 = angle_h1-angle_h1[0]
    angle_w1 = angle_w1-angle_w1[0]
    angle_w2 = angle_w2-angle_w2[0]
    angle_w3 = angle_w3-angle_w3[0]
    angle_w4 = angle_w4-angle_w4[0]
    angle_w5 = angle_w5-angle_w5[0]
    angle_f = angle_f-angle_f[0]

if mode == 'REC' :
    angle_h2 = -(angle_h2-angle_h2[-1])
    angle_h1 = -(angle_h1-angle_h1[-1])
    angle_w1 = -(angle_w1-angle_w1[-1])
    angle_w2 = -(angle_w2-angle_w2[-1])
    angle_w3 = -(angle_w3-angle_w3[-1])
    angle_w4 = -(angle_w4-angle_w4[-1])
    angle_w5 = -(angle_w5-angle_w5[-1])
    angle_f = -(angle_f-angle_f[-1])
    """
    angle_w1 = angle_w1 - ( angle_w1[-1] - 1 )
    angle_w2 = angle_w2 - ( angle_w2[-1] - 1 )
    angle_w3 = angle_w3 - ( angle_w3[-1] - 1 )
    angle_w4 = angle_w4 - ( angle_w4[-1] - 1 )
    angle_w5 = angle_w5 - ( angle_w5[-1] - 1 )
    """

a = np.array([0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75])
a_max = max(a)

plt.plot(time, angle_f, label='a=0.00', linewidth=3.00, c=cm.hot(a[0]/(a_max+1)))
plt.plot(time, angle_h2, label='a=0.25', linewidth=1.50, c=cm.hot(a[1]/(a_max+1)))
plt.plot(time, angle_h1, label='a=0.50', linewidth=1.50, c=cm.hot(a[2]/(a_max+1)))
plt.plot(time, angle_w1, label='a=0.75', linewidth=1.50, c=cm.hot(a[3]/(a_max+1)))
plt.plot(time, angle_w2, label='a=1.00', linewidth=1.50, c=cm.hot(a[4]/(a_max+1)))
plt.plot(time, angle_w3, label='a=1.25', linewidth=1.50, c=cm.hot(a[5]/(a_max+1)))
plt.plot(time, angle_w4, label='a=1.50', linewidth=1.50, c=cm.hot(a[6]/(a_max+1)))
plt.plot(time, angle_w5, label='a=1.75', linewidth=1.50, c=cm.hot(a[7]/(a_max+1)))
plt.legend(fontsize=20.0)
plt.xticks(fontsize=20.0)
plt.yticks(fontsize=20.0)
plt.xlabel('t [ps]', fontsize=20.0)
plt.ylabel(r'$\theta(t)-\hat{\theta}$ [nondim.]', fontsize=20.0)
# plt.ylabel(r'$\theta(t)/\theta_{fin}$ [nondim.]', fontsize=20.0)
plt.xlim([0,max(time)])
plt.title('Comparison spreading angles', fontsize=20.0)
plt.show()
