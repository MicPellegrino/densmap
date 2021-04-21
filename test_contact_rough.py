import densmap as dm
import matplotlib.pyplot as plt
import numpy as np

FP = dm.fitting_parameters( par_file='parameters_droplet.txt' )

height = 0.4
waven  = 1.675135382692315
phi_0  = 1.5525217216960845
h_0    = 0.9129142857142857
fs = lambda x : height * np.sin(waven*x+phi_0) + h_0
dfs = lambda x : height * waven * np.cos(waven*x+phi_0)

# NB: contour tracking should check whether there are actually kfin-kinit files!!!
CD = dm.droplet_tracking(FP.folder_name, FP.first_stamp, FP.last_stamp, FP, \
    file_root = '/flow_', contact_line = True, f_sub = fs, df_sub = dfs )

# Testing plot
CD.plot_radius()
CD.plot_angles()

# Movie
dz = FP.dz
CD.movie_contour([40.0, 70.0], [0.0, FP.lenght_z/10], dz,  circle=True, contact_line = True, fun_sub=fs, dfun_sub=dfs)
