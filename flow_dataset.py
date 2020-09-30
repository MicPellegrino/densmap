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

flow_speed = array_from_file( 'FLOW_SPEED.txt' )
flow_position = array_from_file( 'FLOW_POSITION.txt' )

z0 = flow_position[ int(0.5*len(flow_position)) ]

z_data_tot = flow_position[ int(0.5*len(flow_position)): ] - z0
u_data_tot = flow_speed[ int(0.5*len(flow_speed)): ]

N = 7

z_data = z_data_tot[0::N]
u_data = 10*u_data_tot[0::N]

print(z_data)
print(u_data)

plt.plot(z_data_tot, u_data_tot, 'b-')
plt.plot(z_data, u_data, 'rx')
plt.show()
