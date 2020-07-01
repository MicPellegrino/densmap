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

time = array_from_file('ShearDropModes/time.txt')
radius_upper = array_from_file('ShearDropModes/radius_upper.txt')
radius_lower = array_from_file('ShearDropModes/radius_lower.txt')
position_upper = array_from_file('ShearDropModes/position_upper.txt')
position_lower = array_from_file('ShearDropModes/position_lower.txt')

difference = radius_upper-radius_lower
diff_avg = np.mean(difference)
diff_std = np.std(difference)
print('Difference')
print('mean = '+str(diff_avg))
print('std = '+str(diff_std))

plt.plot(time, radius_lower, 'b-', linewidth=2.0, label='lower')
plt.plot(time, radius_upper, 'r-', linewidth=2.0, label='upper')
plt.plot(time, difference, 'k-', linewidth=2.0, label='difference')
plt.plot(time, diff_avg*np.ones(len(time)), 'g-', linewidth=1.5, label='mean')
plt.plot(time, diff_avg*np.ones(len(time))+5.0*diff_std*np.ones(len(time)), 'g--', linewidth=1.5, label='+/- 5std')
plt.plot(time, diff_avg*np.ones(len(time))-5.0*diff_std*np.ones(len(time)), 'g--', linewidth=1.5)
plt.legend(fontsize=20.0)
plt.xticks(fontsize=20.0)
plt.yticks(fontsize=20.0)
plt.xlabel('t [ps]', fontsize=20.0)
plt.ylabel('surface/width [nm]', fontsize=20.0)
plt.xlim([0,max(time)])
plt.title('Comparison relative wetted area', fontsize=20.0)
plt.show()

offset = position_upper-position_lower
off_avg = np.mean(offset)
off_std = np.std(offset)
print('Offset')
print('mean = '+str(off_avg))
print('std = '+str(off_std))

plt.plot(time, position_lower, 'b-', linewidth=2.0, label='lower')
plt.plot(time, position_upper, 'r-', linewidth=2.0, label='upper')
plt.plot(time, offset, 'k-', linewidth=2.0, label='difference')
plt.plot(time, off_avg*np.ones(len(time)), 'g-', linewidth=1.5, label='mean')
plt.plot(time, off_avg*np.ones(len(time))+5.0*off_std*np.ones(len(time)), 'g--', linewidth=1.5, label='+/- 5std')
plt.plot(time, off_avg*np.ones(len(time))-5.0*off_std*np.ones(len(time)), 'g--', linewidth=1.5)
plt.legend(fontsize=20.0)
plt.xticks(fontsize=20.0)
plt.yticks(fontsize=20.0)
plt.xlabel('t [ps]', fontsize=20.0)
plt.ylabel('position [nm]', fontsize=20.0)
plt.xlim([0,max(time)])
plt.title('Comparison relative foot position', fontsize=20.0)
plt.show()
