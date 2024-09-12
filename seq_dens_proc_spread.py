import densmap as dm
import matplotlib.pyplot as plt
import numpy as np

FP = dm.fitting_parameters( par_file='parameters_amorphous.txt' )
# FP = dm.fitting_parameters( par_file='parameters_droplet.txt' )
# FP = dm.fitting_parameters( par_file='parameters_small.txt' )
# FP = dm.fitting_parameters( par_file='parameters_ca_gly.txt' )

# Rough substrate parameters
# R15
"""
height = 0.4
waven  = 1.675135382692315
phi_0  = 1.5525217216960845
h_0    = 0.9129142857142857
fs = lambda x : height * np.sin(waven*x+phi_0) + h_0
dfs = lambda x : height * waven * np.cos(waven*x+phi_0)
CD = dm.droplet_tracking(FP.folder_name, FP.first_stamp, FP.last_stamp, FP, \
    file_root='/flow_', contact_line=True, f_sub=fs, df_sub=dfs)
"""

# R05
"""
height = 0.1333333333333333
waven  = 5.025696662822574
phi_0  = 4.638399152217284
h_0    = 0.6416666666666667
fs = lambda x : height * np.sin(waven*x+phi_0) + h_0
dfs = lambda x : height * waven * np.cos(waven*x+phi_0)
CD = dm.droplet_tracking(FP.folder_name, FP.first_stamp, FP.last_stamp, FP, \
    file_root='/flow_', contact_line=True, f_sub=fs, df_sub=dfs)
"""

# n06a10
"""
a = 1.0
n = 6
Lx = 82.80000/4
waven  = 2*np.pi*n/Lx
height = a/waven
phi_0  = 0
h_0    = 3.0
fs = lambda x : height * np.sin(waven*x+phi_0) + h_0
dfs = lambda x : height * waven * np.cos(waven*x+phi_0)
CD = dm.droplet_tracking(FP.folder_name, FP.first_stamp, FP.last_stamp, FP, \
    file_root='/flow_', contact_line=True, f_sub=fs, df_sub=dfs)
"""

### Flat substrate ###
CD = dm.droplet_tracking(FP.folder_name, FP.first_stamp, FP.last_stamp, FP, \
    file_root='/flow_', contact_line=True, mode='int')

### Testing plot ###
CD.plot_radius()
CD.plot_angles()

### Movie ###
dz = FP.dz
# CD.movie_contour([0, FP.lenght_x], [0, FP.lenght_z], dz, circle=True, contact_line=True, fun_sub=fs, dfun_sub=dfs)
# CD.movie_contour([0, FP.lenght_x], [0, FP.lenght_z], dz, circle=True, contact_line=True)

### Saving data ###
# CD.save_to_file('/home/michele/densmap/Amorphous/Ep40QSi4')
# CD.save_to_file('/home/michele/densmap/SpreadingDataGlycerol/T4R5')
# CD.save_to_file('/home/michele/densmap/SignalsContactAngles/20p')
# CD.save_to_file('/home/michele/densmap/SpreadingDataWaterButanol/Bwqm08/')
# CD.save_to_file('/home/michele/densmap/SignalsContactAngles/100p-res/')
# CD.save_to_file('/home/michele/densmap/AmorphousSpreading/n12a02/R4/')