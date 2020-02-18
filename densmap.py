"""
    A little library for processing density maps from molecular dynamics simulations
"""

import math

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

import scipy as sc
from scipy import signal

import skimage as sk
from skimage import measure

################################################################################
################################################################################
################################################################################

"""
    #########################################################
    ### From strata program (courtesy of Petter Johanson) ###
    #########################################################
"""

def read_data(filename, decimals=5):
    """Read field data from a file name.

    Determines which of the simple formats in this module to use and
    returns data read using the proper function.

    Coordinates are rounded to input number of decimals.

    Args:
        filename (str): A file to read data from.

    Keyword Args:
        decimals (int): Number of decimals for coordinates.

    Returns:
        (dict, dict): 2-tuple of dict's with data and information. See
            strata.dataformats.read.read_data_file for more information.

    """

    def guess_read_function(filename):
        """Return handle to binary or plaintext function."""

        def is_binary(filename, checksize=512):
            with open(filename, 'r') as fp:
                try:
                    fp.read(checksize)
                    return False
                except UnicodeDecodeError:
                    return True

        if is_binary(filename):
            return read_binsimple
        else:
            return read_plainsimple

    read_function = guess_read_function(filename)
    data = read_function(filename)
    for coord in ['X', 'Y']:
        data[coord] = data[coord].round(decimals)

    info = calc_information(data['X'], data['Y'])

    x0, y0 = info['origin']
    dx, dy = info['spacing']
    nx, ny = info['shape']

    x = x0 + dx * np.arange(nx, dtype=np.float64)
    y = y0 + dy * np.arange(ny, dtype=np.float64)

    xs, ys = np.meshgrid(x, y, indexing='ij')

    data['X'] = xs.ravel()
    data['Y'] = ys.ravel()

    return data, info


def calc_information(X, Y):
    """Return a dict of system information calculated from input cell positions.

    Calculates system origin ('origin'), bin spacing ('spacing'), number of cells
    ('num_bins') and shape ('shape').

    Args:
        X, Y (array_like): Arrays with cell positions.

    Returns:
        dict: Information about system in dictionary.

    """

    def calc_shape(X, Y):
        data = np.zeros((len(X), ), dtype=[('X', np.float), ('Y', np.float)])
        data['X'] = X
        data['Y'] = Y

        data.sort(order=['Y', 'X'])

        y0 = data['Y'][0]
        nx = 1

        try:
            while np.abs(data['Y'][nx] - y0) < 1e-4:
                nx += 1
        except IndexError:
           pass

        ny = len(X) // nx

        return nx, ny

    def calc_spacing(X, Y, nx, ny):
        def calc_1d(xs, n):
            x0 = np.min(xs)
            x1 = np.max(xs)

            try:
                return (x1 - x0) / (n - 1)
            except:
                return 0.0

        dx = calc_1d(X, nx)
        dy = calc_1d(Y, ny)

        return dx, dy

    def calc_origin(X, Y):
        return X.min(), Y.min()

    if len(X) != len(Y):
        raise ValueError("Lengths of X and Y arrays not equal")

    nx, ny = calc_shape(X, Y)
    dx, dy = calc_spacing(X, Y, nx, ny)
    x0, y0 = calc_origin(X, Y)

    info = {
        'shape': (nx, ny),
        'spacing': (dx, dy),
        'num_bins': nx * ny,
        'origin': (x0, y0),
    }

    return info


def read_binsimple(filename):
    """Return data and information read from a simple binary format.

    Args:
        filename (str): A file to read data from.

    Returns:
        dict: Data with field labels as keys.

    """

    def read_file(filename):
        # Fixed field order of format
        fields = ['X', 'Y', 'N', 'T', 'M', 'U', 'V']
        raw_data = np.fromfile(filename, dtype='float32')

        # Unpack into dictionary
        data = {}
        stride = len(fields)
        for i, field in enumerate(fields):
            data[field] = raw_data[i::stride]

        return data

    data = read_file(filename)

    return data


def read_plainsimple(filename):
    """Return field data from a simple plaintext format.

    Args:
        filename (str): A file to read data from.

    Returns:
        dict: Data with field labels as keys.
    """

    def read_file(filename):
        raw_data = np.genfromtxt(filename, names=True)

        # Unpack into dictionary
        data = {}
        for field in raw_data.dtype.names:
            data[field] = raw_data[field]

        return data

    data = read_file(filename)

    return data

################################################################################
################################################################################
################################################################################

"""
    Class for storing information regarding droplet spreding
"""
class contour_data :

    def __init__(contour_data):
        print("[densmap] Initializing contour data structure")

    time = []
    contour = []
    branch_left = []
    branch_right = []
    foot_left = []
    foot_right = []
    angle_left = []
    angle_right = []
    cotangent_left = []
    cotangent_right = []

class fitting_parameters :

    def __init__(fitting_parameters):
        print("[densmap] Initializing fitting parameters data structure")

    time_step = 0.0             # [ps]
    lenght_x = 0.0              # [nm]
    lenght_z = 0.0              # [nm]
    r_mol = 0.0                 # [nm]
    max_vapour_density = 0.0    # [MASS] -> give a look at flow program
    substrate_location = 0.0    # [nm]
    bulk_location = 0.0         # [nm]
    simmetry_plane = 0.0        # [nm]
    interpolation_order = 1     # [nondim.]

"""
    Read input file containing density map
"""
def read_density_file (
    filename,
    bin,
    n_bin_x=0,
    n_bin_z=0
    ) :

    assert bin=='y' or bin=='n', \
        "Specify whether the input file is binary (bin='y') or not (bin='n')"

    # Read file format as produced from by the '-flow' oprion of mdrun
    if bin=='y' :

        # print('[densmap] Reading binary file in -flow format')

        data, info = read_data(filename)
        Nx = info['shape'][0]
        Nz = info['shape'][1]
        density_array = np.array( data['M'] )
        density_array = density_array.reshape((Nx,Nz))

    # Read file format as produced by densmap programme
    else :

        # print('[densmap] Reading text file in gmx densmap format')

        assert n_bin_x > 0 and n_bin_z > 0, \
            "Specify a positive number of bins when reading from gmx densmap output"

        Nx = n_bin_x
        Nz = n_bin_z
        idx = 0
        density_array = np.zeros( (Nx+1) * (Nz+1), dtype=float )
        for line in open(filename, 'r'):
            vals = np.array( [ float(i) for i in line.split() ] )
            density_array[idx:idx+len(vals)] = vals
            idx += len(vals)
        density_array = density_array.reshape((Nx+1,Nz+1))
        density_array = density_array[1:-1,1:-1]

    return density_array

"""
    Crops the array containg density values to the desired values
"""
def crop_density_map (
    density_array,
    new_x_s=0,
    new_x_f=-1,
    new_z_s=0,
    new_z_f=-1
    ):

    density_array_crop = density_array[new_x_s:new_x_f,new_z_s:new_z_f]
    nx = density_array_crop.shape[0]
    nz = density_array_crop.shape[1]

    return density_array_crop, nx, nz

"""
    Compute the gradient in (x,z) using central finite different scheme
"""
def gradient_density (
    density_array,
    hx,
    hz
    ) :

    assert hx>0 and hz>0, \
        "Provide a finite positive value for the bin size in x and z direction"

    nx = density_array.shape[0]
    nz = density_array.shape[1]
    grad_x = np.zeros((nx, nz), dtype=float)
    grad_z = np.zeros((nx, nz), dtype=float)
    for i in range(1, nx-1) :
        for j in range (1, nz-1) :
            grad_x[i,j] = 0.5*( density_array[i+1,j] - density_array[i-1,j] ) / hx
            grad_z[i,j] = 0.5*( density_array[i,j+1] - density_array[i,j-1] ) / hz

    return grad_x, grad_z

"""
    Computes smoothing kernel
"""
def smooth_kernel (
    radius,
    hx,
    hz
    ) :

    assert hx>0 and hz>0, \
        "Provide a finite positive value for the bin size in x and z direction"

    sigma = 2*radius
    Nker_x = int(sigma/hx)
    Nker_z = int(sigma/hz)

    smoother = np.zeros((2*Nker_x+1, 2*Nker_z+1))
    for i in range(-Nker_x,Nker_x+1):
        for j in range(-Nker_z,Nker_z+1):
            dist2 = sigma**2-(i*hx)**2-(j*hz)**2
            if dist2 >= 0:
                smoother[i+Nker_x][j+Nker_z] = np.sqrt(dist2)
            else :
                smoother[i+Nker_x][j+Nker_z] = 0.0
    smoother = smoother / np.sum(np.sum(smoother))

    return smoother

"""
    Convolute (periodically) the density map with the smoothing kernel
"""
def convolute (
    density_array,
    smoother
    ) :

    smooth_density_array = sc.signal.convolve2d(density_array, smoother,
        mode='same', boundary='wrap')

    return smooth_density_array

"""
    Identify the bulk density value by inspecting the density value distribution
"""
def detect_bulk_density (
    density_array,
    density_th,
    n_hist_bins=100
    ) :

    hist, bin_edges = np.histogram(density_array[density_array >= density_th].flatten(),
        bins=n_hist_bins)
    bulk_density = bin_edges[np.argmax(hist)]

    return bulk_density

"""
    Identify the contour line corresponding to the desired density value
"""
def detect_contour (
    density_array,
    density_target,
    hx,
    hz
    ) :

    assert hx>0 and hz>0, \
        "Provide a finite positive value for the bin size in x and z direction"

    contour = sk.measure.find_contours(density_array, density_target)

    assert len(contour)>=1, \
        "No contour line found for the target density value"

    if len(contour)>1 :
        print( "[densmap] More than one contour found for the target density value (returning the one with most points)" )
        contour = sorted(contour, key=lambda x : len(x))

    h = np.array([[hx, 0.0],[0.0, hz]])
    contour = np.matmul(h,(contour[-1].transpose()))
    return contour

"""
    Identify the points near the contact line
"""
def detect_contact_line (
    contour,
    z_min,
    z_max,
    x_half,
    m = 5,
    d = 10
    ) :

    left_branch_x = []
    left_branch_z = []
    right_branch_x = []
    right_branch_z = []

    for i in range(contour.shape[1]) :
        if contour[1,i] > z_min and contour[1,i] <= z_max :
            if contour[0,i] <= x_half :
                left_branch_x.append(contour[0,i])
                left_branch_z.append(contour[1,i])
            else :
                right_branch_x.append(contour[0,i])
                right_branch_z.append(contour[1,i])

    left_branch = np.empty((2,len(left_branch_x)))
    right_branch = np.empty((2,len(right_branch_x)))
    left_branch[0,:] = left_branch_x
    left_branch[1,:] = left_branch_z
    right_branch[0,:] = right_branch_x
    right_branch[1,:] = right_branch_z

    idx_l = np.argsort(left_branch[1,:])
    idx_r = np.argsort(right_branch[1,:])
    left_branch = left_branch[:,idx_l]
    right_branch = right_branch[:,idx_r]

    points_l = left_branch[:,0:d*m:d]
    points_r = right_branch[:,0:d*m:d]

    return left_branch, right_branch, points_l, points_r

"""
    Fit a polynomial to the contact line points in order to obtain the c.a.
"""
def detect_contact_angle (
    points_l,
    points_r,
    order
    ) :

    assert order<len(points_l[1,:]), \
        "Interpolation order is larger than the number of points"

    p_l = np.polyfit( points_l[1,:] , points_l[0,:] , order )
    p_r = np.polyfit( points_r[1,:] , points_r[0,:] , order )

    foot_l = np.polyval( p_l, points_l[1,0] )
    foot_r = np.polyval( p_r, points_r[1,0] )

    cot_l = (p_l[1]+2.0*p_l[0]*points_l[1,0])
    cot_r = (p_r[1]+2.0*p_r[0]*points_r[1,0])
    cot_l = 0.0
    cot_r = 0.0
    for n in range(1,order+1) :
        cot_l += n*p_l[order-n]*points_l[1,0]**(n-1)
        cot_r += n*p_r[order-n]*points_r[1,0]**(n-1)

    theta_l = np.rad2deg( -np.arctan( cot_l )+0.5*math.pi )
    theta_l = theta_l + 180*(theta_l<=0)
    theta_r = - np.rad2deg( -np.arctan( cot_r )+0.5*math.pi )
    theta_r = theta_r + 180*(theta_r<=0)

    return foot_l, foot_r, theta_l, theta_r, cot_l, cot_r

"""
    Build up contour data starting from flow_%%%%%.dat files
"""
# Print info each ... steps
K_INFO = 50
def contour_tracking (
    folder_name,
    k_init,
    k_end,
    fit_param
    ) :

    # Data structure that will be outputted
    CD = contour_data()

    # Read the first density snapshot, in order to get the values needed to contruct the smoothing kernel
    file_name = folder_name+'/flow_'+str(k_init).zfill(5)+'.dat'
    print("[densmap] Reading "+file_name)
    density_array = read_density_file(file_name, bin='y')
    Nx = density_array.shape[0]
    Nz = density_array.shape[1]
    hx = fit_param.lenght_x/Nx
    hz = fit_param.lenght_z/Nz
    print("[densmap] Initialize smoothing kernel")
    smoother = smooth_kernel(fit_param.r_mol, hx, hz)
    # Append the values for the first time-step
    smooth_density_array = convolute(density_array, smoother)
    bulk_density = detect_bulk_density(smooth_density_array, density_th=fit_param.max_vapour_density)
    intf_contour = detect_contour(smooth_density_array, 0.5*bulk_density, hx, hz)
    left_branch, right_branch, points_l, points_r = \
        detect_contact_line(intf_contour, z_min=fit_param.substrate_location,
        z_max=fit_param.bulk_location, x_half=fit_param.simmetry_plane)
    foot_l, foot_r, theta_l, theta_r, cot_l, cot_r = \
        detect_contact_angle(points_l, points_r, order=fit_param.interpolation_order)
    CD.time.append( fit_param.time_step*k_init )
    CD.contour.append( intf_contour )
    CD.branch_left.append( left_branch )
    CD.branch_right.append( right_branch )
    CD.foot_left.append( foot_l )
    CD.foot_right.append( foot_r )
    CD.angle_left.append( theta_l )
    CD.angle_right.append( theta_r )
    CD.cotangent_left.append( cot_l )
    CD.cotangent_right.append( cot_r )

    for k in range(k_init+1, k_end+1) :
        file_name = folder_name+'/flow_'+str(k).zfill(5)+'.dat'
        if k % K_INFO == 0 :
            print("[densmap] Reading "+file_name)
        # Loop
        density_array = read_density_file(file_name, bin='y')
        smooth_density_array = convolute(density_array, smoother)
        bulk_density = detect_bulk_density(smooth_density_array, density_th=fit_param.max_vapour_density)
        intf_contour = detect_contour(smooth_density_array, 0.5*bulk_density, hx, hz)
        left_branch, right_branch, points_l, points_r = \
            detect_contact_line(intf_contour, z_min=fit_param.substrate_location,
            z_max=fit_param.bulk_location, x_half=fit_param.simmetry_plane)
        foot_l, foot_r, theta_l, theta_r, cot_l, cot_r = \
            detect_contact_angle(points_l, points_r, order=fit_param.interpolation_order)
        CD.time.append( fit_param.time_step*k )
        CD.contour.append( intf_contour )
        CD.branch_left.append( left_branch )
        CD.branch_right.append( right_branch )
        CD.foot_left.append( foot_l )
        CD.foot_right.append( foot_r )
        CD.angle_left.append( theta_l )
        CD.angle_right.append( theta_r )
        CD.cotangent_left.append( cot_l )
        CD.cotangent_right.append( cot_r )

    return CD
