import numpy as np
import matplotlib.pyplot as plt

# Avoid: use ensemble averaging instead...
import scipy.ndimage as smg
import scipy.signal as sgn

def array_from_file( filename ):
    my_list = []
    with open(filename, 'r') as f:
        for line in f:
            my_list.append(float(line.split()[0]))
    return np.array(my_list)

folder_name = 'ShearCL/t47ca011'

time = array_from_file(folder_name+'/time.txt')
dt = time[1]-time[0]

centers = dict()
width = dict()

centers['top'] = array_from_file(folder_name+'/position_upper.txt')
centers['bot'] = array_from_file(folder_name+'/position_lower.txt')
width['top'] = array_from_file(folder_name+'/radius_upper.txt')
width['bot'] = array_from_file(folder_name+'/radius_lower.txt')

contact_angle = dict()
contact_angle['tl'] = array_from_file(folder_name+'/angle_tl.txt')
contact_angle['tr'] = array_from_file(folder_name+'/angle_tr.txt')
contact_angle['bl'] = array_from_file(folder_name+'/angle_bl.txt')
contact_angle['br'] = array_from_file(folder_name+'/angle_br.txt')

contact_line = dict()
contact_line['tl'] = centers['top'] - 0.5*width['top']
contact_line['tr'] = centers['top'] + 0.5*width['top']
contact_line['bl'] = centers['bot'] - 0.5*width['bot']
contact_line['br'] = centers['bot'] + 0.5*width['bot']

labels = ['tl', 'tr', 'bl', 'br']
sigma_pos = 5.0     # MANUALLY TUNED
contact_line_fil = dict()
for l in labels :
    contact_line_fil[l] = smg.gaussian_filter1d(contact_line[l], sigma=sigma_pos)

contact_speed = dict()
for l in labels :
    # Unfiltered
    contact_speed[l] = np.zeros( contact_line[l].shape )
    contact_speed[l][1:-2] = 0.5*(contact_line[l][2:-1]-contact_line[l][0:-3])/dt
    contact_speed[l][0] = (contact_line[l][1]-contact_line[l][0])/dt
    contact_speed[l][-1] = (contact_line[l][-1]-contact_line[l][-2])/dt
    """
    contact_speed[l] = np.zeros( contact_line_fil[l].shape )
    contact_speed[l][1:-2] = 0.5*(contact_line_fil[l][2:-1]-contact_line_fil[l][0:-3])/dt
    contact_speed[l][0] = (contact_line_fil[l][1]-contact_line_fil[l][0])/dt
    contact_speed[l][-1] = (contact_line_fil[l][-1]-contact_line_fil[l][-2])/dt
    """

# Plot of CL position over time
plt.plot(time, contact_line['tl'], 'b--', label='TL (rec)')
plt.plot(time, contact_line['tr'], 'r--', label='TR (adv)')
plt.plot(time, contact_line['bl'], 'r-.', label='BL (adv)')
plt.plot(time, contact_line['br'], 'b-.', label='BR (rec)')
plt.plot(time, contact_line_fil['tl'], 'b-')
plt.plot(time, contact_line_fil['tr'], 'r-')
plt.plot(time, contact_line_fil['bl'], 'r-')
plt.plot(time, contact_line_fil['br'], 'b-')
plt.legend(fontsize=20.0)
plt.title('Contact line (Ca=0.11, theta=47deg)', fontsize=30.0)
plt.ylabel('contact speed [nm]', fontsize=25.0)
plt.xlabel('time [ps]', fontsize=25.0)
plt.xticks(fontsize=25.0)
plt.yticks(fontsize=25.0)
plt.xlim([0,time[-1]])
plt.show()

# Plot of CA over time
plt.plot(time, contact_angle['tl'], 'b--', label='TL (rec)')
plt.plot(time, contact_angle['tr'], 'r--', label='TR (adv)')
plt.plot(time, contact_angle['bl'], 'r-.', label='BL (adv)')
plt.plot(time, contact_angle['br'], 'b-.', label='BR (rec)')
plt.legend(fontsize=20.0)
plt.title('Contact angle (Ca=0.11, theta=47deg)', fontsize=30.0)
plt.ylabel('contact angle [deg]', fontsize=25.0)
plt.xlabel('time [ps]', fontsize=25.0)
plt.xticks(fontsize=25.0)
plt.yticks(fontsize=25.0)
plt.xlim([0,time[-1]])
plt.show()

# Plot of CL speed over time
plt.plot(time, contact_speed['tl'], 'bx', label='TL (rec)')
plt.plot(time, contact_speed['tr'], 'rx', label='TR (adv)')
plt.plot(time, contact_speed['bl'], 'ro', label='BL (adv)')
plt.plot(time, contact_speed['br'], 'bo', label='BR (rec)')
plt.legend(fontsize=20.0)
plt.title('Contact line speed (Ca=0.11, theta=47deg)', fontsize=30.0)
plt.ylabel('contact speed [nm/ps]', fontsize=25.0)
plt.xlabel('time [ps]', fontsize=25.0)
plt.xticks(fontsize=25.0)
plt.yticks(fontsize=25.0)
plt.xlim([0,time[-1]])
plt.show()

# Plot CA vs CL position
plt.plot(contact_angle['tl'], contact_line['tl'], 'bx', label='TL (rec)')
plt.plot(contact_angle['tr'], contact_line['tr'], 'rx', label='TR (adv)')
plt.plot(contact_angle['bl'], contact_line['bl'], 'ro', label='BL (adv)')
plt.plot(contact_angle['br'], contact_line['br'], 'bo', label='BR (rec)')
plt.legend(fontsize=20.0)
plt.title('Contact line position vs contact angle (Ca=0.11, theta=47deg)', fontsize=30.0)
plt.xlabel('contact angle [deg]', fontsize=25.0)
plt.ylabel('position [nm]', fontsize=25.0)
plt.xticks(fontsize=25.0)
plt.yticks(fontsize=25.0)
# plt.xlim([0,time[-1]])
plt.show()

# Plot CA vs CL position
plt.plot(contact_angle['tl'], contact_speed['tl'], 'bx', label='TL (rec)')
plt.plot(contact_angle['tr'], contact_speed['tr'], 'rx', label='TR (adv)')
plt.plot(contact_angle['bl'], contact_speed['bl'], 'ro', label='BL (adv)')
plt.plot(contact_angle['br'], contact_speed['br'], 'bo', label='BR (rec)')
plt.legend(fontsize=20.0)
plt.title('Contact line position vs contact angle (Ca=0.11, theta=47deg)', fontsize=30.0)
plt.xlabel('contact angle [deg]', fontsize=25.0)
plt.ylabel('speed [nm/ps]', fontsize=25.0)
plt.xticks(fontsize=25.0)
plt.yticks(fontsize=25.0)
# plt.xlim([0,time[-1]])
plt.show()
