"""
    A little library for processing density maps from molecular dynamics simulations
"""

import math

import numpy as np

import matplotlib as mpl
mpl.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as manimation

import scipy as sc
from scipy import signal

import skimage as sk
from skimage import measure

"""
    Circle fitting library (thanks to marian42: https://github.com/marian42/circle-fit.git)
"""
import circle_fit as cf

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
    Roughness parameter calculation
"""
rough_parameter = lambda a : (2.0/np.pi) * np.sqrt(a+1.0) * sc.special.ellipe(a/(a+1.0))

"""
    Class for storing information regarding droplet spreding
"""
class droplet_data :

    def __init__(droplet_data):
        print("[densmap] Initializing contour data structure")
        droplet_data.time = []
        droplet_data.contour = []
        droplet_data.branch_left = []
        droplet_data.branch_right = []
        droplet_data.foot_left = []
        droplet_data.foot_right = []
        droplet_data.angle_left = []
        droplet_data.angle_right = []
        droplet_data.cotangent_left = []
        droplet_data.cotangent_right = []
        droplet_data.circle_rad = []
        droplet_data.circle_xc = []
        droplet_data.circle_zc = []
        droplet_data.circle_res = []
        droplet_data.angle_circle = []
        droplet_data.radius_circle = []

    def merge(droplet_data, new_data):
        new_time = [  droplet_data.time[-1] + t for t in new_data.time]
        droplet_data.time = droplet_data.time + new_time
        droplet_data.contour = droplet_data.contour + new_data.contour
        droplet_data.branch_left = droplet_data.branch_left + new_data.branch_left
        droplet_data.branch_right = droplet_data.branch_right + new_data.branch_right
        droplet_data.foot_left = droplet_data.foot_left + new_data.foot_left
        droplet_data.foot_right = droplet_data.foot_right + new_data.foot_right
        droplet_data.angle_left = droplet_data.angle_left + new_data.angle_left
        droplet_data.angle_right = droplet_data.angle_right + new_data.angle_right
        droplet_data.cotangent_left = droplet_data.cotangent_left + new_data.cotangent_left
        droplet_data.cotangent_right = droplet_data.cotangent_right + new_data.cotangent_right
        droplet_data.circle_rad = droplet_data.circle_rad + new_data.circle_rad
        droplet_data.circle_xc = droplet_data.circle_xc + new_data.circle_xc
        droplet_data.circle_zc = droplet_data.circle_zc + new_data.circle_zc
        droplet_data.circle_res = droplet_data.circle_res + new_data.circle_res
        droplet_data.angle_circle = droplet_data.angle_circle + new_data.angle_circle
        droplet_data.radius_circle = droplet_data.radius_circle + new_data.radius_circle

    def save_to_file(droplet_data):
        # Those should be saved BEFORE poltting, not after...
        droplet_data.file_angles = 'contact_angles.dat'
        droplet_data.file_feet = 'contact_points.dat'
        droplet_data.file_contout = 'contour.dat'
        droplet_data.file_branches = 'branches.dat'

    def plot_radius(droplet_data):
        mpl.use("Agg")
        droplet_data.spreading_radius = \
            np.array(droplet_data.foot_right)-np.array(droplet_data.foot_left)
        plt.figure()
        plt.plot(droplet_data.time, droplet_data.spreading_radius[:,0], 'k-', label='contour')
        plt.plot(droplet_data.time, droplet_data.radius_circle, 'g-', label='cap')
        plt.title('Spreading radius', fontsize=20.0)
        plt.xlabel('t [ps]', fontsize=20.0)
        plt.ylabel('R(t) [nm]', fontsize=20.0)
        plt.legend()
        plt.show()
        plt.savefig('spreading_radius.eps')
        mpl.use("TkAgg")

    def plot_angles(droplet_data):
        mpl.use("Agg")
        droplet_data.mean_contact_angle = \
            0.5*(np.array(droplet_data.angle_right)+np.array(droplet_data.angle_left))
        droplet_data.hysteresis = \
            np.array(droplet_data.angle_right)-np.array(droplet_data.angle_left)
        plt.figure()
        plt.plot(droplet_data.time, droplet_data.mean_contact_angle, 'b-', label='average')
        plt.plot(droplet_data.time, droplet_data.hysteresis, 'r-', label='difference')
        plt.plot(droplet_data.time, droplet_data.angle_circle, 'g-', label='cap')
        plt.title('Contact angle', fontsize=20.0)
        plt.xlabel('t [ps]', fontsize=20.0)
        plt.ylabel('theta(t) [deg]', fontsize=20.0)
        plt.legend()
        plt.show()
        plt.savefig('contact_angles.eps')
        mpl.use("TkAgg")

    def movie_contour(droplet_data, crop_x, crop_z, dz, circle=True, contact_line=True):
        mpl.use("Agg")
        print("[densmap] Producing movie of the interface dynamics")
        FFMpegWriter = manimation.writers['ffmpeg']
        metadata = dict(title='Spreading Droplet Contour', artist='Michele Pellegrino',
            comment='Just the tracked contour of a spreding droplet')
        writer = FFMpegWriter(fps=30, metadata=metadata)
        fig = plt.figure()
        fig_cont, = plt.plot([], [], 'k-', linewidth=1.5)
        fig_left, = plt.plot([], [], 'r-', linewidth=1.0)
        fig_right, = plt.plot([], [], 'b-', linewidth=1.0)
        fig_pl, = plt.plot([], [], 'r.', linewidth=1.5)
        fig_pr, = plt.plot([], [], 'b.', linewidth=1.5)
        fig_cir, = plt.plot([], [], 'g-', linewidth=0.75)
        plt.title('Droplet Spreading')
        plt.xlabel('x [nm]')
        plt.ylabel('z [nm]')
        s = np.linspace(0,2*np.pi,250)
        with writer.saving(fig, "contour_movie.mp4", 250):
            for i in range( len(droplet_data.contour) ):
                dx_l = dz * droplet_data.cotangent_left[i]
                dx_r = dz * droplet_data.cotangent_right[i]
                if circle :
                    circle_x = droplet_data.circle_xc[i] + droplet_data.circle_rad[i]*np.cos(s)
                    circle_z = droplet_data.circle_zc[i] + droplet_data.circle_rad[i]*np.sin(s)
                fig_cont.set_data(droplet_data.contour[i][0,:], droplet_data.contour[i][1,:])
                t_label = str(droplet_data.time[i])+' ps'
                textvar = plt.text(1.5, 14.0, t_label)
                if contact_line :
                    fig_left.set_data([droplet_data.foot_left[i][0], droplet_data.foot_left[i][0]+dx_l],
                        [droplet_data.foot_left[i][1], droplet_data.foot_left[i][1]+dz])
                    fig_right.set_data([droplet_data.foot_right[i][0], droplet_data.foot_right[i][0]+dx_r],
                        [droplet_data.foot_right[i][1], droplet_data.foot_right[i][1]+dz])
                    fig_pl.set_data(droplet_data.foot_left[i][0], droplet_data.foot_left[i][1])
                    fig_pr.set_data(droplet_data.foot_right[i][0], droplet_data.foot_right[i][1])
                if circle :
                    fig_cir.set_data(circle_x, circle_z)
                plt.axis('scaled')
                plt.xlim(0, crop_x)
                plt.ylim(0, crop_z)
                writer.grab_frame()
                textvar.remove()
        mpl.use("TkAgg")

class shear_data :

    """
        Legend:
            'bl' = bottom left
            'br' = bottom right
            'tl' = top left
            'tr' = top right
    """

    def __init__(shear_data):
        print("[densmap] Initializing contour data structure")
        shear_data.time = []
        shear_data.contour = []
        shear_data.branch = dict()
        shear_data.branch['bl'] = []
        shear_data.branch['br'] = []
        shear_data.branch['tl'] = []
        shear_data.branch['tr'] = []
        shear_data.foot = dict()
        shear_data.foot['bl'] = []
        shear_data.foot['br'] = []
        shear_data.foot['tl'] = []
        shear_data.foot['tr'] = []
        shear_data.angle = dict()
        shear_data.angle['bl'] = []
        shear_data.angle['br'] = []
        shear_data.angle['tl'] = []
        shear_data.angle['tr'] = []
        shear_data.cotangent = dict()
        shear_data.cotangent['bl'] = []
        shear_data.cotangent['br'] = []
        shear_data.cotangent['tl'] = []
        shear_data.cotangent['tr'] = []

    def save_to_file(shear_data, save_dir):
        print("[densmap] Saving to .txt files")
        np.savetxt(save_dir+'/time.txt', np.array(shear_data.time))
        radius_upper = np.abs( np.array(shear_data.foot['tr']) - np.array(shear_data.foot['tl']) )
        radius_lower = np.abs( np.array(shear_data.foot['br']) - np.array(shear_data.foot['bl']) )
        position_upper = 0.5*np.abs( np.array(shear_data.foot['tr']) + np.array(shear_data.foot['tl']) )
        position_lower = 0.5*np.abs( np.array(shear_data.foot['br']) + np.array(shear_data.foot['bl']) )
        np.savetxt(save_dir+'/radius_upper.txt', radius_upper[:,0])
        np.savetxt(save_dir+'/radius_lower.txt', radius_lower[:,0])
        np.savetxt(save_dir+'/position_upper.txt', position_upper[:,0])
        np.savetxt(save_dir+'/position_lower.txt', position_lower[:,0])
        np.savetxt(save_dir+'/angle_bl.txt', shear_data.angle['bl'])
        np.savetxt(save_dir+'/angle_br.txt', shear_data.angle['br'])
        np.savetxt(save_dir+'/angle_tl.txt', shear_data.angle['tl'])
        np.savetxt(save_dir+'/angle_tr.txt', shear_data.angle['tr'])

    def plot_radius(shear_data, fig_name='spreading_radius.eps'):
        mpl.use("Agg")
        print("[densmap] Producing plot for spreading radius")
        radius_upper = np.abs( np.array(shear_data.foot['tr']) - np.array(shear_data.foot['tl']) )
        radius_lower = np.abs( np.array(shear_data.foot['br']) - np.array(shear_data.foot['bl']) )
        plt.figure()
        """
            Change once the upper half tracking is added
        """
        plt.plot(shear_data.time, radius_upper[:,0], 'k-', label='upper')
        plt.plot(shear_data.time, radius_lower[:,0], 'g-', label='lower')
        plt.title('Spreading radius', fontsize=20.0)
        plt.xlabel('t [ps]', fontsize=20.0)
        plt.ylabel('R(t) [nm]', fontsize=20.0)
        plt.legend()
        plt.show()
        plt.savefig(fig_name)
        mpl.use("TkAgg")

    def plot_angles(shear_data, fig_name='contact_angles.eps'):
        mpl.use("Agg")
        print("[densmap] Producing plot for contact angles")
        plt.figure()
        plt.plot(shear_data.time, shear_data.angle['bl'], 'b-', label='bottom left')
        plt.plot(shear_data.time, shear_data.angle['br'], 'r-', label='bottom right')
        plt.plot(shear_data.time, shear_data.angle['tl'], 'c-', label='top left')
        plt.plot(shear_data.time, shear_data.angle['tr'], 'm-', label='top right')
        plt.title('Contact angle', fontsize=20.0)
        plt.xlabel('t [ps]', fontsize=20.0)
        plt.ylabel('theta(t) [deg]', fontsize=20.0)
        plt.legend()
        plt.show()
        plt.savefig(fig_name)
        mpl.use("TkAgg")

    def movie_contour(shear_data, crop_x, crop_z, dz, contact_line=True):
        mpl.use("Agg")
        print("[densmap] Producing movie of the interface dynamics")
        FFMpegWriter = manimation.writers['ffmpeg']
        metadata = dict(title='Spreading Droplet Contour', artist='Michele Pellegrino',
            comment='Just the tracked contour of a spreding droplet')
        writer = FFMpegWriter(fps=30, metadata=metadata)
        fig = plt.figure()
        fig_cont, = plt.plot([], [], '-', color='tab:gray', linewidth=1.5)
        fig_fbl, = plt.plot([], [], 'b-', linewidth=1.0)
        fig_fbr, = plt.plot([], [], 'r-', linewidth=1.0)
        fig_ftl, = plt.plot([], [], 'c-', linewidth=1.0)
        fig_ftr, = plt.plot([], [], 'm-', linewidth=1.0)
        fig_pbl, = plt.plot([], [], 'b.', linewidth=1.5)
        fig_pbr, = plt.plot([], [], 'r.', linewidth=1.5)
        fig_ptl, = plt.plot([], [], 'c.', linewidth=1.5)
        fig_ptr, = plt.plot([], [], 'm.', linewidth=1.5)
        fig_pos_up, = plt.plot([], [], 'kx', linewidth=1.5)
        fig_pos_lw, = plt.plot([], [], 'gx', linewidth=1.5)
        plt.title('Droplet Shear')
        plt.xlabel('x [nm]')
        plt.ylabel('z [nm]')
        s = np.linspace(0,2*np.pi,250)
        with writer.saving(fig, "contour_movie.mp4", 250):
            for i in range( len(shear_data.contour) ):
                dx_l = dz * shear_data.cotangent['bl'][i]
                dx_r = dz * shear_data.cotangent['br'][i]
                fig_cont.set_data(shear_data.contour[i][0,:], shear_data.contour[i][1,:])
                t_label = str(shear_data.time[i])+' ps'
                """
                    The position of the time label should be an input (or at leat a macro)
                """
                textvar = plt.text(1.5, 14.0, t_label)
                if contact_line :
                    fig_fbl.set_data([shear_data.foot['bl'][i][0], shear_data.foot['bl'][i][0]+dx_l],
                        [shear_data.foot['bl'][i][1], shear_data.foot['bl'][i][1]+dz])
                    fig_fbr.set_data([shear_data.foot['br'][i][0], shear_data.foot['br'][i][0]+dx_r],
                        [shear_data.foot['br'][i][1], shear_data.foot['br'][i][1]+dz])
                    fig_pbl.set_data(shear_data.foot['bl'][i][0], shear_data.foot['bl'][i][1])
                    fig_pbr.set_data(shear_data.foot['br'][i][0], shear_data.foot['br'][i][1])
                    # Flipping again to the upper wall
                    fig_ftl.set_data([shear_data.foot['tl'][i][0], shear_data.foot['tl'][i][0]+dx_l],
                        [crop_z-shear_data.foot['tl'][i][1], crop_z-shear_data.foot['tl'][i][1]-dz])
                    fig_ftr.set_data([shear_data.foot['tr'][i][0], shear_data.foot['tr'][i][0]+dx_r],
                        [crop_z-shear_data.foot['tr'][i][1], crop_z-shear_data.foot['tr'][i][1]-dz])
                    fig_ptl.set_data(shear_data.foot['tl'][i][0], crop_z-shear_data.foot['tl'][i][1])
                    fig_ptr.set_data(shear_data.foot['tr'][i][0], crop_z-shear_data.foot['tr'][i][1])
                    fig_pos_up.set_data( 0.5*(shear_data.foot['tl'][i][0]+shear_data.foot['tr'][i][0]), \
                        0.5*(crop_z-shear_data.foot['tl'][i][1]+crop_z-shear_data.foot['tr'][i][1]) )
                    fig_pos_lw.set_data( 0.5*(shear_data.foot['bl'][i][0]+shear_data.foot['br'][i][0]), \
                        0.5*(shear_data.foot['bl'][i][1]+shear_data.foot['br'][i][1]) )
                plt.axis('scaled')
                plt.xlim(0, crop_x)
                plt.ylim(0, crop_z)
                writer.grab_frame()
                textvar.remove()
        mpl.use("TkAgg")


def dictionify( file_name, sp = '=' ) :
    par_dict = dict()
    par_file = open( file_name, 'r')
    for line in par_file :
        line = line.strip().replace(' ','')
        cols = line.split(sp)
        par_dict[cols[0]] = cols[1]
    par_file.close()
    return par_dict

class fitting_parameters :

    time_step = 0.0             # [ps]
    lenght_x = 0.0              # [nm]
    lenght_z = 0.0              # [nm]
    r_mol = 0.0                 # [nm]
    max_vapour_density = 0.0    # [MASS] -> give a look at flow program
    substrate_location = 0.0    # [nm]
    bulk_location = 0.0         # [nm]
    simmetry_plane = 0.0        # [nm]
    interpolation_order = 1     # [nondim.]
    folder_name = ''            # [string]
    first_stamp = 0             # [int]
    last_stamp = 0              # [int]
    dz = 0.0                    # [nm]

    def __init__(fitting_parameters, par_file=None):
        print("[densmap] Initializing fitting parameters data structure")
        if par_file != None :
            par_dict = dictionify(par_file)
            fitting_parameters.time_step = \
                float(par_dict['time_step'])
            fitting_parameters.lenght_x = \
                float(par_dict['lenght_x'])
            fitting_parameters.lenght_z = \
                float(par_dict['lenght_z'])
            fitting_parameters.r_mol = \
                float(par_dict['r_mol'])
            fitting_parameters.max_vapour_density = \
                float(par_dict['max_vapour_density'])
            fitting_parameters.substrate_location = \
                float(par_dict['substrate_location'])
            fitting_parameters.bulk_location = \
                float(par_dict['bulk_location'])
            fitting_parameters.simmetry_plane = \
                float(par_dict['simmetry_plane'])
            fitting_parameters.interpolation_order = \
                int(par_dict['interpolation_order'])
            fitting_parameters.folder_name = \
                par_dict['folder_name']
            fitting_parameters.first_stamp = \
                int(par_dict['first_stamp'])
            fitting_parameters.last_stamp = \
                int(par_dict['last_stamp'])
            fitting_parameters.dz = \
                float(par_dict['dz'])

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
    Read velocity field from .dat data
"""
def read_velocity_file (
    filename
    ) :

    data, info = read_data(filename)
    # print(data)
    Nx = info['shape'][0]
    Nz = info['shape'][1]
    vel_x = np.array( data['U'] )
    vel_z = np.array( data['V'] )
    vel_x = vel_x.reshape((Nx,Nz))
    vel_z = vel_z.reshape((Nx,Nz))

    return vel_x, vel_z

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
    Returns an array d[i][j] s.t.
"""
def density_indicator (
    density_array,
    target_density
    ) :

    idx = np.zeros( density_array.shape, dtype=int )
    idx[density_array >= target_density] = 1

    return idx

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

    foot_l = ( np.polyval( p_l, points_l[1,0] ), points_l[1,0] )
    foot_r = ( np.polyval( p_r, points_r[1,0] ), points_r[1,0] )

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
# Print info each K_INFO steps
K_INFO = 50
def droplet_tracking (
    folder_name,
    k_init,
    k_end,
    fit_param,
    file_root = '/flow_',
    contact_line = True
    ) :

    # Data structure that will be outputted
    CD = droplet_data()

    # Read the first density snapshot, in order to get the values needed to contruct the smoothing kernel
    file_name = folder_name+file_root+str(k_init).zfill(5)+'.dat'
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
    if contact_line :
        left_branch, right_branch, points_l, points_r = \
            detect_contact_line(intf_contour, z_min=fit_param.substrate_location,
            z_max=fit_param.bulk_location, x_half=fit_param.simmetry_plane)
        foot_l, foot_r, theta_l, theta_r, cot_l, cot_r = \
            detect_contact_angle(points_l, points_r, order=fit_param.interpolation_order)
    else :
        left_branch = np.NaN
        right_branch = np.NaN
        points_l = np.NaN
        points_r = np.NaN
        foot_l = np.NaN
        foot_r = np.NaN
        theta_l = np.NaN
        theta_r = np.NaN
        cot_l = np.NaN
        cot_r = np.NaN
    xc, zc, R, residue = circle_fit(intf_contour, z_th=fit_param.substrate_location)
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
    CD.circle_rad.append(R)
    CD.circle_xc.append(xc)
    CD.circle_zc.append(zc)
    CD.circle_res.append(residue)

    # ANGLE FROM CAP FITTING
    h = fit_param.substrate_location
    cot_circle = (h-zc)/np.sqrt(R*R-(h-zc)**2)
    theta_circle = np.rad2deg( -np.arctan( cot_circle )+0.5*math.pi )
    theta_circle = theta_circle + 180*(theta_circle<=0)
    CD.angle_circle.append(theta_circle)
    # RADIUS FROM CAP FITTING
    CD.radius_circle.append(2*np.sqrt(R*R-(h-zc)**2))

    for k in range(k_init+1, k_end+1) :
        file_name = folder_name+file_root+str(k).zfill(5)+'.dat'
        if k % K_INFO == 0 :
            print("[densmap] Reading "+file_name)
        # Loop
        density_array = read_density_file(file_name, bin='y')
        smooth_density_array = convolute(density_array, smoother)
        bulk_density = detect_bulk_density(smooth_density_array, density_th=fit_param.max_vapour_density)
        intf_contour = detect_contour(smooth_density_array, 0.5*bulk_density, hx, hz)
        if contact_line :
            left_branch, right_branch, points_l, points_r = \
                detect_contact_line(intf_contour, z_min=fit_param.substrate_location,
                z_max=fit_param.bulk_location, x_half=fit_param.simmetry_plane)
            foot_l, foot_r, theta_l, theta_r, cot_l, cot_r = \
                detect_contact_angle(points_l, points_r, order=fit_param.interpolation_order)
        else :
            left_branch = np.NaN
            right_branch = np.NaN
            points_l = np.NaN
            points_r = np.NaN
            foot_l = np.NaN
            foot_r = np.NaN
            theta_l = np.NaN
            theta_r = np.NaN
            cot_l = np.NaN
            cot_r = np.NaN
        xc, zc, R, residue = circle_fit(intf_contour, z_th=fit_param.substrate_location)
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
        CD.circle_rad.append(R)
        CD.circle_xc.append(xc)
        CD.circle_zc.append(zc)
        CD.circle_res.append(residue)

        # ANGLE FROM CAP FITTING
        cot_circle = (h-zc)/np.sqrt(R*R-(h-zc)**2)
        theta_circle = np.rad2deg( -np.arctan( cot_circle )+0.5*math.pi )
        theta_circle = theta_circle + 180*(theta_circle<=0)
        CD.angle_circle.append(theta_circle)
        # RADIUS FROM CAP FITTING
        CD.radius_circle.append(2*np.sqrt(R*R-(h-zc)**2))

    return CD

"""
    Same as above, but for a shear droplet
"""
def shear_tracking (
    folder_name,
    k_init,
    k_end,
    fit_param,
    file_root = '/flow_',
    contact_line = True
    ) :

    # Data structure that will be outputted
    CD = shear_data()

    # Read the first density snapshot, in order to get the values needed to contruct the smoothing kernel
    file_name = folder_name+file_root+str(k_init).zfill(5)+'.dat'
    print("[densmap] Reading "+file_name)
    density_array = read_density_file(file_name, bin='y')
    Nx = density_array.shape[0]
    Nz = density_array.shape[1]
    hx = fit_param.lenght_x/Nx
    hz = fit_param.lenght_z/Nz
    print("[densmap] Initialize smoothing kernel")
    """
        Should actually check if r_mol is less than the bin size; in that case
        performing density averaging is useless
    """
    smoother = smooth_kernel(fit_param.r_mol, hx, hz)

    # Append the values for the first time-step
    smooth_density_array = convolute(density_array, smoother)
    bulk_density = detect_bulk_density(smooth_density_array, density_th=fit_param.max_vapour_density)
    intf_contour = detect_contour(smooth_density_array, 0.5*bulk_density, hx, hz)
    """
        Stuff below has to be modified
    """
    if contact_line :
        b_left_branch, b_right_branch, b_points_l, b_points_r = \
            detect_contact_line(intf_contour, z_min=fit_param.substrate_location,
            z_max=fit_param.bulk_location, x_half=fit_param.simmetry_plane)
        b_foot_l, b_foot_r, b_theta_l, b_theta_r, b_cot_l, b_cot_r = \
            detect_contact_angle(b_points_l, b_points_r, order=fit_param.interpolation_order)
        # Flip interface contour
        intf_contour_flip = np.stack((intf_contour[0,:], fit_param.lenght_z-intf_contour[1,:]))
        t_left_branch, t_right_branch, t_points_l, t_points_r = \
            detect_contact_line(intf_contour_flip, z_min=fit_param.substrate_location,
            z_max=fit_param.bulk_location, x_half=fit_param.simmetry_plane)
        t_foot_l, t_foot_r, t_theta_l, t_theta_r, t_cot_l, t_cot_r = \
            detect_contact_angle(t_points_l, t_points_r, order=fit_param.interpolation_order)
    else :
        b_left_branch = np.NaN
        b_right_branch = np.NaN
        b_points_l = np.NaN
        b_points_r = np.NaN
        b_foot_l = np.NaN
        b_foot_r = np.NaN
        b_theta_l = np.NaN
        b_theta_r = np.NaN
        b_cot_l = np.NaN
        b_cot_r = np.NaN
        t_left_branch = np.NaN
        t_right_branch = np.NaN
        t_points_l = np.NaN
        t_points_r = np.NaN
        t_foot_l = np.NaN
        t_foot_r = np.NaN
        t_theta_l = np.NaN
        t_theta_r = np.NaN
        t_cot_l = np.NaN
        t_cot_r = np.NaN
    CD.time.append( fit_param.time_step*k_init )
    CD.contour.append( intf_contour )
    CD.branch['bl'].append( b_left_branch )
    CD.branch['br'].append( b_right_branch )
    CD.foot['bl'].append( b_foot_l )
    CD.foot['br'].append( b_foot_r )
    CD.angle['bl'].append( b_theta_l )
    CD.angle['br'].append( b_theta_r )
    CD.cotangent['bl'].append( b_cot_l )
    CD.cotangent['br'].append( b_cot_r )
    CD.branch['tl'].append( t_left_branch )
    CD.branch['tr'].append( t_right_branch )
    CD.foot['tl'].append( t_foot_l )
    CD.foot['tr'].append( t_foot_r )
    CD.angle['tl'].append( t_theta_l )
    CD.angle['tr'].append( t_theta_r )
    CD.cotangent['tl'].append( t_cot_l )
    CD.cotangent['tr'].append( t_cot_r )

    for k in range(k_init+1, k_end+1) :
        file_name = folder_name+file_root+str(k).zfill(5)+'.dat'
        if k % K_INFO == 0 :
            print("[densmap] Reading "+file_name)
        # Loop
        density_array = read_density_file(file_name, bin='y')
        smooth_density_array = convolute(density_array, smoother)
        bulk_density = detect_bulk_density(smooth_density_array, density_th=fit_param.max_vapour_density)
        intf_contour = detect_contour(smooth_density_array, 0.5*bulk_density, hx, hz)
        if contact_line :
            b_left_branch, b_right_branch, b_points_l, b_points_r = \
                detect_contact_line(intf_contour, z_min=fit_param.substrate_location,
                z_max=fit_param.bulk_location, x_half=fit_param.simmetry_plane)
            b_foot_l, b_foot_r, b_theta_l, b_theta_r, b_cot_l, b_cot_r = \
                detect_contact_angle(b_points_l, b_points_r, order=fit_param.interpolation_order)
            # Flip interface contour
            intf_contour_flip = np.stack((intf_contour[0,:], fit_param.lenght_z-intf_contour[1,:]))
            t_left_branch, t_right_branch, t_points_l, t_points_r = \
                detect_contact_line(intf_contour_flip, z_min=fit_param.substrate_location,
                z_max=fit_param.bulk_location, x_half=fit_param.simmetry_plane)
            t_foot_l, t_foot_r, t_theta_l, t_theta_r, t_cot_l, t_cot_r = \
                detect_contact_angle(t_points_l, t_points_r, order=fit_param.interpolation_order)
        else :
            b_left_branch = np.NaN
            b_right_branch = np.NaN
            b_points_l = np.NaN
            b_points_r = np.NaN
            b_foot_l = np.NaN
            b_foot_r = np.NaN
            b_theta_l = np.NaN
            b_theta_r = np.NaN
            b_cot_l = np.NaN
            b_cot_r = np.NaN
            t_left_branch = np.NaN
            t_right_branch = np.NaN
            t_points_l = np.NaN
            t_points_r = np.NaN
            t_foot_l = np.NaN
            t_foot_r = np.NaN
            t_theta_l = np.NaN
            t_theta_r = np.NaN
            t_cot_l = np.NaN
            t_cot_r = np.NaN
        xc, zc, R, residue = circle_fit(intf_contour, z_th=fit_param.substrate_location)
        CD.time.append( fit_param.time_step*k )
        CD.contour.append( intf_contour )
        CD.branch['bl'].append( b_left_branch )
        CD.branch['br'].append( b_right_branch )
        CD.foot['bl'].append( b_foot_l )
        CD.foot['br'].append( b_foot_r )
        CD.angle['bl'].append( b_theta_l )
        CD.angle['br'].append( b_theta_r )
        CD.cotangent['bl'].append( b_cot_l )
        CD.cotangent['br'].append( b_cot_r )
        CD.branch['tl'].append( t_left_branch )
        CD.branch['tr'].append( t_right_branch )
        CD.foot['tl'].append( t_foot_l )
        CD.foot['tr'].append( t_foot_r )
        CD.angle['tl'].append( t_theta_l )
        CD.angle['tr'].append( t_theta_r )
        CD.cotangent['tl'].append( t_cot_l )
        CD.cotangent['tr'].append( t_cot_r )

    return CD

"""
    Finds the better circle fitting the droplet contour
"""
def circle_fit(intf_contour, z_th=0.0) :
    M = len(intf_contour[0,:])
    data_circle_x = []
    data_circle_z = []
    for k in range(M) :
        if intf_contour[1,k] > z_th :
            data_circle_x.append(intf_contour[0,k])
            data_circle_z.append(intf_contour[1,k])
    data_circle_x = np.array(data_circle_x)
    data_circle_z = np.array(data_circle_z)
    # t = np.linspace(0,2*np.pi,250)
    # circle_x = xc + R*np.cos(t)
    # circle_z = zc + R*np.sin(t)
    return cf.least_squares_circle(np.stack((data_circle_x, data_circle_z), axis=1))

"""
    Function to obtain equilibrium radius and c.a.
"""
def equilibrium_from_density ( filename, smoother, density_th, hx, hz, h ) :
    density_array = read_density_file(filename, bin='y')
    smooth_density_array = convolute(density_array, smoother)
    bulk_density = detect_bulk_density(smooth_density_array, density_th)
    intf_contour = detect_contour(smooth_density_array, 0.5*bulk_density, hx, hz)
    xc, zc, R, residue = circle_fit(intf_contour, h)
    radius_circle = 2*np.sqrt(R*R-(h-zc)**2)
    cot_circle = (h-zc)/np.sqrt(R*R-(h-zc)**2)
    theta_circle = np.rad2deg( -np.arctan( cot_circle )+0.5*math.pi )
    theta_circle = theta_circle + 180*(theta_circle<=0)
    return radius_circle, theta_circle
