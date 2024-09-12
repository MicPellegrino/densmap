import numpy as np
import matplotlib.pyplot as plt

cos  = lambda t : np.cos(np.deg2rad(t))
acos = lambda c : np.rad2deg(np.arccos(c))

def array_from_file( filename ):
    my_list = []
    with open(filename, 'r') as f:
        for line in f:
            my_list.append(float(line.split()[0]))
    return np.array(my_list)

# input_file_folder = '/home/michele/densmap/ShearDropModes/NeoQ2'
input_file_folder = '/home/michele/densmap/ShearDynamic/Q2_Ca025'

angles = dict()
cosine = dict()
cl_pos = dict()
avg_angles = dict()

time = array_from_file(input_file_folder+'/time.txt')
xcom = array_from_file(input_file_folder+'/xcom.txt')
angle_circle = array_from_file(input_file_folder+'/angle_circle.txt')

# If hydrophobic:
# angle_circle = 180-angle_circle

dt = time[1]-time[0]
n_transient = int(2000/dt)

loactions = ['tl', 'tr', 'bl', 'br']

for k in loactions :
    angles[k] = array_from_file(input_file_folder+'/angle_'+k+'.txt')
    cosine[k] = cos(angles[k])
    avg_angles[k] = np.mean(angles[k][n_transient:])
    # avg_angles[k] = acos(np.mean(cosine[k][n_transient:]))
    plt.plot(time, cosine[k], label=k)
    print(k+" : <theta0> = "+str(avg_angles[k]))

mean_contact_angle_adv = 0.5*(avg_angles['br']+avg_angles['tl'])
mean_contact_angle_rec = 0.5*(avg_angles['bl']+avg_angles['tr'])

equilibrium_contact_angle = 0.5*(mean_contact_angle_adv+mean_contact_angle_rec)

print("Mean c.a. advancing  = "+str(mean_contact_angle_adv))
print("Mean c.a. receding   = "+str(mean_contact_angle_rec))
print("Equilibrium c.a.         = "+str(equilibrium_contact_angle))
print("Equilibrium c.a. (c.f.)  = "+str(np.mean(angle_circle[n_transient:])))

plt.plot(time, cos(angle_circle), 'k--', linewidth=1.5)
plt.ylabel(r'$\theta$ [deg]')
plt.xlabel('t [ps]')
plt.legend()
plt.show()