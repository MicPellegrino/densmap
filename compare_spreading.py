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

# time = array_from_file('Adv/Flat/time.txt')
# radius_f = array_from_file('Adv/Flat/radius_c.txt')
# radius_w1 = array_from_file('Adv/Wave1/radius_c.txt')
# radius_w2 = array_from_file('Adv/Wave2/radius_c.txt')
# radius_w3 = array_from_file('Adv/Wave3/radius_c.txt')
# radius_w4 = array_from_file('Adv/Wave4/radius_c.txt')
# radius_w5 = array_from_file('Adv/Wave5/radius_c.txt')

# time = array_from_file('Rec/Flat/time.txt')
# radius_f = array_from_file('Rec/Flat/radius_c.txt')
# radius_w1 = array_from_file('Rec/Wave1/radius_c.txt')
# radius_w2 = array_from_file('Rec/Wave2/radius_c.txt')
# radius_w3 = array_from_file('Rec/Wave3/radius_c.txt')
# radius_w4 = array_from_file('Rec/Wave4/radius_c.txt')
# radius_w5 = array_from_file('Rec/Wave5/radius_c.txt')

# time = array_from_file('Adv/Flat/time.txt')
# angle_f = array_from_file('Adv/Flat/angle_r.txt')
# angle_w1 = array_from_file('Adv/Wave1/angle_r.txt')
# angle_w2 = array_from_file('Adv/Wave2/angle_r.txt')
# angle_w3 = array_from_file('Adv/Wave3/angle_r.txt')
# angle_w4 = array_from_file('Adv/Wave4/angle_r.txt')
# angle_w5 = array_from_file('Adv/Wave5/angle_r.txt')

time = array_from_file('Rec/Flat/time.txt')
angle_f = array_from_file('Rec/Flat/angle_r.txt')
angle_w1 = array_from_file('Rec/Wave1/angle_r.txt')
angle_w2 = array_from_file('Rec/Wave2/angle_r.txt')
angle_w3 = array_from_file('Rec/Wave3/angle_r.txt')
angle_w4 = array_from_file('Rec/Wave4/angle_r.txt')
angle_w5 = array_from_file('Rec/Wave5/angle_r.txt')

# radius_f = radius_f/radius_f[-1]
# radius_w1 = radius_w1/radius_w1[-1]
# radius_w2 = radius_w2/radius_w2[-1]
# radius_w3 = radius_w3/radius_w3[-1]
# radius_w4 = radius_w4/radius_w4[-1]
# radius_w5 = radius_w5/radius_w5[-1]

angle_f = angle_f/angle_f[-1]
angle_w1 = angle_w1/angle_w1[-1]
angle_w2 = angle_w2/angle_w2[-1]
angle_w3 = angle_w3/angle_w3[-1]
angle_w4 = angle_w4/angle_w4[-1]
angle_w5 = angle_w5/angle_w5[-1]

a = np.array([0.00, 0.75, 1.00, 1.25, 1.50, 1.75])
a_max = max(a)

# plt.plot(time, radius_f, label='a=0.00', linewidth=2.00, c=cm.hot(a[0]/(a_max+1)))
# plt.plot(time, radius_w1, label='a=0.75', linewidth=2.00, c=cm.hot(a[1]/(a_max+1)))
# plt.plot(time, radius_w2, label='a=1.00', linewidth=2.00, c=cm.hot(a[2]/(a_max+1)))
# plt.plot(time, radius_w3, label='a=1.25', linewidth=2.00, c=cm.hot(a[3]/(a_max+1)))
# plt.plot(time, radius_w4, label='a=1.50', linewidth=2.00, c=cm.hot(a[4]/(a_max+1)))
# plt.plot(time, radius_w5, label='a=1.75', linewidth=2.00, c=cm.hot(a[5]/(a_max+1)))
plt.plot(time, angle_f, label='a=0.00', linewidth=2.00, c=cm.hot(a[0]/(a_max+1)))
plt.plot(time, angle_w1, label='a=0.75', linewidth=2.00, c=cm.hot(a[1]/(a_max+1)))
plt.plot(time, angle_w2, label='a=1.00', linewidth=2.00, c=cm.hot(a[2]/(a_max+1)))
plt.plot(time, angle_w3, label='a=1.25', linewidth=2.00, c=cm.hot(a[3]/(a_max+1)))
plt.plot(time, angle_w4, label='a=1.50', linewidth=2.00, c=cm.hot(a[4]/(a_max+1)))
plt.plot(time, angle_w5, label='a=1.75', linewidth=2.00, c=cm.hot(a[5]/(a_max+1)))
plt.legend(fontsize=20.0)
plt.xticks(fontsize=20.0)
plt.yticks(fontsize=20.0)
plt.xlabel('t [ps]', fontsize=20.0)
# plt.ylabel(r'$R(t)/R_0$ [nondim.]', fontsize=20.0)
# plt.ylabel(r'$R(t)/R_{fin}$ [nondim.]', fontsize=20.0)
# plt.ylabel(r'$\theta(t)/\theta_0$ [nondim.]', fontsize=20.0)
plt.ylabel(r'$\theta(t)/\theta_{fin}$ [nondim.]', fontsize=20.0)
plt.xlim([0,max(time)])
plt.title('Comparison spreading curves', fontsize=20.0)
plt.show()
