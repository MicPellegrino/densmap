#!/usr/bin/python3

import numpy as np
import matplotlib as mpl
mpl.use('Agg')                    # For plotting without X server
import matplotlib.pyplot as plt
import sys
import os.path

# [TO-DO FOR USER, will not work without correct path]
# Add path to densmap code (be vary to use the correct version)
# sys.path.append('/..')

import densmap as dm

plt.rcParams['text.usetex']=True
plt.rcParams.update({'figure.autolayout': True})  # to make sure that labels are inside printing area

font = {'family' : 'DejaVu Sans',
        'weight' : 'bold',
        'size'   : 18}

plt.rc('font', **font)

# Set the folder name
MDname = "Flow"
# Set the start index for the averaging (outputted from previous script + 1)
iStart = 2200
# iStart = 1

# Get the total number of files
Nmd = len([f for f in os.listdir(MDname) if os.path.isfile(os.path.join(MDname, f))])

# Nmd = Nmd-2 # If there are extra files
# # Override if needed
# Nmd = 634
# Nmd = iStart+207 # Take first 2.5 ns of the data
# Nmd = iStart+65

# fraction of bulk/mean density to define the interface point
dItf = 0.5

# Read the first file, display some outputs
MDflow, info = dm.read_data(MDname+"/flow_%05d.dat"%iStart)
Nx  = info['shape'][0]
Ny  = info['shape'][1]
print("Read file with Nx = %d, Ny = %d bins! Measurement units: nm"%(Nx,Ny))
Xar = np.array( MDflow['X'] )
Yar = np.array( MDflow['Y'] )
Xar = Xar.reshape((Nx,Ny))
Yar = Yar.reshape((Nx,Ny))
# Display the minimum and maximum binning coordinates
bin_dx = Xar[1,0]-Xar[0,0]
bin_dy = Yar[0,1]-Yar[0,0]
print("x in [%f, %f], dx = %f"%(Xar[0,0],Xar[Nx-1,0],bin_dx))
print("y in [%f, %f], dy = %f"%(Yar[0,0],Yar[0,Ny-1],bin_dy))

# For GROMACS version 2020
# bin_dz = 4.67650
# densScale = 1.6605402/(bin_dx*bin_dy*bin_dz)

# For GROMACS version 2021
densScale = 1.6605402

np.savetxt("02_Xmatrix.txt", Xar,header="X coordinate data")
np.savetxt("02_Ymatrix.txt", Yar,header="Y coordinate data")

# Compute the mass centrum coordinate to work with. For vertical direction, the mass centrum is not corrected,
# because it can
# be different due to different length of wetted areas.
XcTrg = 0.5*(Xar[0,0]+Xar[-1,0])

# Initiate with zero arrays
MarS = np.zeros((Nx,Ny))
UarS = np.zeros((Nx,Ny))
VarS = np.zeros((Nx,Ny))

# Loop over all files in steady regime and and compute the density matrix sum
for k in range(iStart,Nmd+1):
    # Read the MD data
    MDflow, info = dm.read_data(MDname+"/flow_%05d.dat"%k)
    # Reshape and re-normalise the mass array into density array [kg/m3]
    Mar = np.array( MDflow['M'] )
    Mar = Mar.reshape((Nx,Ny))*densScale
    # Compute the actual mass centrum and shift the field to target mass centrum
    Xc   = np.sum(Xar*Mar)/np.sum(Mar)
    intS = int((XcTrg-Xc)/bin_dx) - 1
    remS =     (XcTrg-Xc)         - intS*bin_dx
    MarL = np.roll(Mar,intS  ,axis=0)
    MarR = np.roll(Mar,intS+1,axis=0)
    Mar  = MarL + (MarR - MarL)*remS/bin_dx
    XcR  = np.sum(Xar*Mar)/np.sum(Mar)
    print("(file %05d) Target Xc = %f, original Xc = %f, recomputed Xc = %f!"%(k,XcTrg,Xc,XcR))
    # Shift the velocity arrays and interpolate
    Uar = np.array( MDflow['U'] ); Uar = Uar.reshape((Nx,Ny)); Uorg = np.sum(Uar);
    Var = np.array( MDflow['V'] ); Var = Var.reshape((Nx,Ny)); Vorg = np.sum(Var);
    UarL = np.roll(Uar,intS  ,axis=0)
    UarR = np.roll(Uar,intS+1,axis=0)
    Uar  = UarL + (UarR - UarL)*remS/bin_dx
    VarL = np.roll(Var,intS  ,axis=0)
    VarR = np.roll(Var,intS+1,axis=0)
    Var  = VarL + (VarR - VarL)*remS/bin_dx
    # Add the result to the sum
    MarS= MarS + Mar
    # UarS= UarS + Uar
    # VarS= VarS + Var
    UarS= UarS + Uar/(Nmd+1-iStart)
    VarS= VarS + Var/(Nmd+1-iStart)

# Get the average matrices
Mar = MarS/(Nmd+1-iStart)
# Uar = UarS/(Nmd+1-iStart)
# Var = VarS/(Nmd+1-iStart)
Uar = UarS
Var = VarS

# Save the averaged data
np.savetxt("02_average_wMcent_Mmatrix.txt", Mar,header="Averaged M data over %d files, %f ns"%((Nmd+1-iStart),(Nmd+1-iStart)*12.5/1000))
np.savetxt("02_average_wMcent_Umatrix.txt", Uar,header="Averaged U data over %d files, %f ns"%((Nmd+1-iStart),(Nmd+1-iStart)*12.5/1000))
np.savetxt("02_average_wMcent_Vmatrix.txt", Var,header="Averaged V data over %d files, %f ns"%((Nmd+1-iStart),(Nmd+1-iStart)*12.5/1000))
