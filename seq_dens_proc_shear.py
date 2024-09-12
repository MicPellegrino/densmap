import densmap as dm
import matplotlib.pyplot as plt
import numpy as np

# FP = dm.fitting_parameters( par_file='parameters_shear_hex.txt' )
FP = dm.fitting_parameters( par_file='parameters_amorphous_friction.txt' )

# NB: contour tracking should check whether there are actually kfin-kinit files!!!
"""
CD = dm.shear_tracking(FP.folder_name, FP.first_stamp, FP.last_stamp, FP, \
    file_root = '/flow_SOL_', contact_line=True, fit_ca=False, mode='sk', ens=0)
"""

"""
CD = dm.shear_tracking(FP.folder_name, FP.first_stamp, FP.last_stamp, FP, \
    file_root = '/flow_SOL_', contact_line=True, fit_ca=False, mode='loc', ens=0)
"""

CD = dm.shear_tracking(FP.folder_name, FP.first_stamp, FP.last_stamp, FP, \
    file_root = '/flow_', contact_line=True, fit_ca=True, mode='loc', ens=0)

# Saving plots
# CD.plot_radius()
# CD.plot_angles()

# Saving to .txt files
# Non-equilibrium (shear)
# CD.save_to_file('ShearDynamic/Q3_Ca010')
# CD.save_to_file('ShearDropModes/NeoQ2')
# CD.save_to_file('ShearWatBut/C005Q66')
# CD.save_to_file('ShearWatHex/C008Q65')
# CD.save_to_file('ShearWatPen/C002Q65')
CD.save_to_file('AmorphousFriction/Ep40Q4Ca00')

# Saving movie
CD.movie_contour(FP.lenght_x, FP.lenght_z, FP.dz,  circle=False, contact_line=True)