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

# Initial droplet shape
D_0 = 20.0

mode = 'REC'

# ADVANCING
if mode == 'ADV' :
    time = array_from_file('Adv/Flat/time.txt')
    radius_f = array_from_file('Adv/Flat/radius_c.txt')
    radius_h2 = array_from_file('Adv/Height2/radius_c.txt')
    radius_h1 = array_from_file('Adv/Height1/radius_c.txt')
    radius_w1 = array_from_file('Adv/Wave1/radius_c.txt')
    radius_w2 = array_from_file('Adv/Wave2/radius_c.txt')
    radius_w3 = array_from_file('Adv/Wave3/radius_c.txt')
    radius_w4 = array_from_file('Adv/Wave4/radius_c.txt')
    radius_w5 = array_from_file('Adv/Wave5/radius_c.txt')

# RECEIDING
if mode == 'REC' :
    time = array_from_file('Rec/Flat/time.txt')
    radius_f = array_from_file('Rec/Flat/radius_c.txt')
    radius_h2 = array_from_file('Rec/Height2/radius_c.txt')
    radius_h1 = array_from_file('Rec/Height1/radius_c.txt')
    radius_w1 = array_from_file('Rec/Wave1/radius_c.txt')
    radius_w2 = array_from_file('Rec/Wave2/radius_c.txt')
    radius_w3 = array_from_file('Rec/Wave3/radius_c.txt')
    radius_w4 = array_from_file('Rec/Wave4/radius_c.txt')
    radius_w5 = array_from_file('Rec/Wave5/radius_c.txt')

# ADVANCING
if mode == 'ADV' :
    """
    radius_h2 = radius_h2/radius_f[0]
    radius_h1 = radius_h1/radius_f[0]
    radius_w1 = radius_w1/radius_f[0]
    radius_w2 = radius_w2/radius_f[0]
    radius_w3 = radius_w3/radius_f[0]
    radius_w4 = radius_w4/radius_f[0]
    radius_w5 = radius_w5/radius_f[0]
    radius_f = radius_f/radius_f[0]
    radius_h2 = radius_h2 - ( radius_h2[0] - 1 )
    radius_h1 = radius_h1 - ( radius_h1[0] - 1 )
    radius_w1 = radius_w1 - ( radius_w1[0] - 1 )
    radius_w2 = radius_w2 - ( radius_w2[0] - 1 )
    radius_w3 = radius_w3 - ( radius_w3[0] - 1 )
    radius_w4 = radius_w4 - ( radius_w4[0] - 1 )
    radius_w5 = radius_w5 - ( radius_w5[0] - 1 )
    """
    radius_h2 = radius_h2/D_0
    radius_h1 = radius_h1/D_0
    radius_w1 = radius_w1/D_0
    radius_w2 = radius_w2/D_0
    radius_w3 = radius_w3/D_0
    radius_w4 = radius_w4/D_0
    radius_w5 = radius_w5/D_0
    radius_f = radius_f/D_0
    radius_h2 = radius_h2 - ( radius_h2[0] - 1 )
    radius_h1 = radius_h1 - ( radius_h1[0] - 1 )
    radius_w1 = radius_w1 - ( radius_w1[0] - 1 )
    radius_w2 = radius_w2 - ( radius_w2[0] - 1 )
    radius_w3 = radius_w3 - ( radius_w3[0] - 1 )
    radius_w4 = radius_w4 - ( radius_w4[0] - 1 )
    radius_w5 = radius_w5 - ( radius_w5[0] - 1 )
    radius_f = radius_f - ( radius_f[0] - 1 )

# RECEIDING
if mode == 'REC' :
    """
    radius_h2 = radius_h2/radius_f[-1]
    radius_h1 = radius_h1/radius_f[-1]
    radius_w1 = radius_w1/radius_f[-1]
    radius_w2 = radius_w2/radius_f[-1]
    radius_w3 = radius_w3/radius_f[-1]
    radius_w4 = radius_w4/radius_f[-1]
    radius_w5 = radius_w5/radius_f[-1]
    radius_f = radius_f/radius_f[-1]
    radius_h2 = radius_h2 - ( radius_h2[-1] - 1 )
    radius_h1 = radius_h1 - ( radius_h1[-1] - 1 )
    radius_w1 = radius_w1 - ( radius_w1[-1] - 1 )
    radius_w2 = radius_w2 - ( radius_w2[-1] - 1 )
    radius_w3 = radius_w3 - ( radius_w3[-1] - 1 )
    radius_w4 = radius_w4 - ( radius_w4[-1] - 1 )
    radius_w5 = radius_w5 - ( radius_w5[-1] - 1 )
    """
    radius_h2 = radius_h2/D_0
    radius_h1 = radius_h1/D_0
    radius_w1 = radius_w1/D_0
    radius_w2 = radius_w2/D_0
    radius_w3 = radius_w3/D_0
    radius_w4 = radius_w4/D_0
    radius_w5 = radius_w5/D_0
    radius_f = radius_f/D_0
    radius_h2 = radius_h2 - ( radius_h2[-1] - 1 )
    radius_h1 = radius_h1 - ( radius_h1[-1] - 1 )
    radius_w1 = radius_w1 - ( radius_w1[-1] - 1 )
    radius_w2 = radius_w2 - ( radius_w2[-1] - 1 )
    radius_w3 = radius_w3 - ( radius_w3[-1] - 1 )
    radius_w4 = radius_w4 - ( radius_w4[-1] - 1 )
    radius_w5 = radius_w5 - ( radius_w5[-1] - 1 )
    radius_f = radius_f - ( radius_f[-1] - 1 )

# RELATIVE SPREADING
radius_h2_rel = radius_h2 - radius_f
radius_h1_rel = radius_h1 - radius_f
radius_w1_rel = radius_w1 - radius_f
radius_w2_rel = radius_w2 - radius_f
radius_w3_rel = radius_w3 - radius_f
radius_w4_rel = radius_w4 - radius_f
radius_w5_rel = radius_w5 - radius_f
radius_f_rel = radius_f - radius_f

a = np.array([0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75])
a_max = max(a)

plt.plot(time, radius_f, label='a=0.00', linewidth=3.00, c=cm.hot(a[0]/(a_max+1)))
plt.plot(time, radius_h2, label='a=0.25', linewidth=1.50, c=cm.hot(a[1]/(a_max+1)))
plt.plot(time, radius_h1, label='a=0.50', linewidth=1.50, c=cm.hot(a[2]/(a_max+1)))
plt.plot(time, radius_w1, label='a=0.75', linewidth=1.50, c=cm.hot(a[3]/(a_max+1)))
plt.plot(time, radius_w2, label='a=1.00', linewidth=1.50, c=cm.hot(a[4]/(a_max+1)))
plt.plot(time, radius_w3, label='a=1.25', linewidth=1.50, c=cm.hot(a[5]/(a_max+1)))
plt.plot(time, radius_w4, label='a=1.50', linewidth=1.50, c=cm.hot(a[6]/(a_max+1)))
plt.plot(time, radius_w5, label='a=1.75', linewidth=1.50, c=cm.hot(a[7]/(a_max+1)))
plt.legend(fontsize=20.0)
plt.xticks(fontsize=20.0)
plt.yticks(fontsize=20.0)
plt.xlabel('t [ps]', fontsize=20.0)
plt.ylabel(r'$R^*$ [nondim.]', fontsize=20.0)
plt.xlim([0,max(time)])
plt.title('Comparison spreading curves', fontsize=20.0)
plt.show()

plt.plot(time, radius_f_rel, label='a=0.00', linewidth=3.00, c=cm.hot(a[0]/(a_max+1)))
plt.plot(time, radius_h2_rel, label='a=0.25', linewidth=1.50, c=cm.hot(a[1]/(a_max+1)))
plt.plot(time, radius_h1_rel, label='a=0.50', linewidth=1.50, c=cm.hot(a[2]/(a_max+1)))
plt.plot(time, radius_w1_rel, label='a=0.75', linewidth=1.50, c=cm.hot(a[3]/(a_max+1)))
plt.plot(time, radius_w2_rel, label='a=1.00', linewidth=1.50, c=cm.hot(a[4]/(a_max+1)))
plt.plot(time, radius_w3_rel, label='a=1.25', linewidth=1.50, c=cm.hot(a[5]/(a_max+1)))
plt.plot(time, radius_w4_rel, label='a=1.50', linewidth=1.50, c=cm.hot(a[6]/(a_max+1)))
plt.plot(time, radius_w5_rel, label='a=1.75', linewidth=1.50, c=cm.hot(a[7]/(a_max+1)))
plt.legend(fontsize=20.0)
plt.xticks(fontsize=20.0)
plt.yticks(fontsize=20.0)
plt.xlabel('t [ps]', fontsize=20.0)
plt.ylabel(r'$R^*-R^*_{flat}$ [nondim.]', fontsize=20.0)
plt.xlim([0,max(time)])
plt.title('Comparison relative displacement', fontsize=20.0)
plt.show()
