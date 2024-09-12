import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

def array_from_file( filename ):
    my_list = []
    with open(filename, 'r') as f:
        for line in f:
            my_list.append(float(line.split()[0]))
    return np.array(my_list)

folder_names = ['SignalsContactAngles/0p',  \
                'SignalsContactAngles/20p', \
                'SignalsContactAngles/40p', \
                'SignalsContactAngles/60p', \
                'SignalsContactAngles/80p-res', \
                'SignalsContactAngles/100p-res']

labels = ['Water', r'20% glycerol', r'40% glycerol', r'60% glycerol', r'80% glycerol', r'100% glycerol']

colors = ['c','m','b','r','g','k']

t_aver = 25000.0

plt.title('Contact angles', fontsize=30.0)

time = dict()
angle = dict()
angle_mean = dict()
angle_std = dict()

for k in range(6) :
    l = labels[k]
    time[l] = array_from_file(folder_names[k]+'/time.txt')
    time[l] = time[l]-time[l][0]
    t0 = time[l][-1]-t_aver
    i0 = np.argmin(np.abs(t0-time[l]))
    angle[l] = array_from_file(folder_names[k]+'/angle_fit.txt')
    angle_mean[l] = np.mean(angle[l][i0:])
    angle_std[l] = np.std(angle[l][i0:])
    plt.plot(time[l], angle[l], color=colors[k], linewidth=2.5)
    plt.plot(time[l][i0:], angle_mean[l]*np.ones_like(time[l][i0:]), 
        color=colors[k], linewidth=5, label=labels[k])

plt.legend(fontsize=20.0)
plt.xticks(fontsize=20.0)
plt.yticks(fontsize=20.0)
plt.show()

print(angle_mean)
print(angle_std)