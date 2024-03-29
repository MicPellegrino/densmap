import numpy as np
import matplotlib.pyplot as plt

def array_from_file( filename ):
    my_list = []
    with open(filename, 'r') as f:
        for line in f:
            my_list.append(float(line.split()[0]))
    return np.array(my_list)

folder_name = 'ShearChar/Q4Ext'

time = array_from_file(folder_name+'/time.txt')

centers = dict()
width = dict()

centers['top'] = array_from_file(folder_name+'/position_upper.txt')
centers['bot'] = array_from_file(folder_name+'/position_lower.txt')
width['top'] = array_from_file(folder_name+'/radius_upper.txt')
width['bot'] = array_from_file(folder_name+'/radius_lower.txt')

contact_line = dict()

contact_line['tl'] = centers['top'] - 0.5*width['top']
contact_line['tr'] = centers['top'] + 0.5*width['top']
contact_line['bl'] = centers['bot'] - 0.5*width['bot']
contact_line['br'] = centers['bot'] + 0.5*width['bot']

# Advancing lines are TR and BL
# Receding lines are TL and BR

plt.plot(time, contact_line['tl'], 'b--', label='TL (rec)', linewidth=2.5)
plt.plot(time, contact_line['tr'], 'r--', label='TR (adv)', linewidth=2.5)
plt.plot(time, contact_line['bl'], 'g-.', label='BL (adv)', linewidth=2.5)
plt.plot(time, contact_line['br'], 'k-.', label='BR (rec)', linewidth=2.5)
plt.legend(fontsize=20.0)
plt.title(r'Contact line position (relaxation, $\theta_0$=37.8$\deg$)', fontsize=30.0)
plt.xlabel('time [ps]', fontsize=25.0)
plt.ylabel('position [nm]', fontsize=25.0)
plt.xticks(fontsize=25.0)
plt.yticks(fontsize=25.0)
plt.xlim([0,time[-1]])
plt.show()
