import densmap as dm
import matplotlib.pyplot as plt
import numpy as np

FP = dm.fitting_parameters( par_file='parameters_shear.txt' )

# NB: conutour tracking should check whether there are actually kfin-kinit files!!!

CD = dm.shear_tracking(FP.folder_name, FP.first_stamp, FP.last_stamp, FP, \
    file_root = '/flow_', contact_line = True)

CD.plot_radius()
CD.plot_angles()

dz = FP.dz
CD.movie_contour(FP.lenght_x, FP.lenght_z, dz, contact_line = True)

CD.save_to_file('ShearDropModes')
