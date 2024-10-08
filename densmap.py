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

# A bit sloppy, but it works:
# True:  2021 (with header)
# False: 2020 (without header)
READ_HEADER = True

"""
    #########################################################
    ### From strata program (courtesy of Petter Johanson) ###
    #########################################################
"""

# 2021
FIELDS = ['X', 'Y', 'N', 'T', 'M', 'U', 'V']

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
    
    # 2021
    if READ_HEADER :
    
        with open(filename, 'rb') as fp:
            fields, num_values, info = read_header(fp)
            data = read_values(fp, num_values, fields)

        x0, y0 = info['origin']
        nx, ny = info['shape']
        dx, dy = info['spacing']

        x = x0 + dx * (np.arange(nx) + 0.5)
        y = y0 + dy * (np.arange(ny) + 0.5)
        xs, ys = np.meshgrid(x, y, indexing='ij')

        grid = np.zeros((nx, ny), dtype=[(l, np.float) for l in FIELDS])
        grid['X'] = xs
        grid['Y'] = ys

        for l in ['N', 'T', 'M', 'U', 'V']:
            grid[l][data['IX'], data['IY']] = data[l]

        grid = grid.ravel()

        return {l: grid[l] for l in FIELDS}, info
    
    # 2020
    else:

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

#2020
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

# 2020
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

# 2020
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

# 2021
def read_values(fp, num_values, fields):
    dtypes = {
        'IX': np.uint64,
        'IY': np.uint64,
        'N': np.float32,
        'T': np.float32,
        'M': np.float32,
        'U': np.float32,
        'V': np.float32,
    }

    return {
        l: np.fromfile(fp, dtype=dtypes[l], count=num_values)
        for l in fields
    }

#2021
def read_header(fp):
    """Read header information and forward the pointer to the data."""

    def read_shape(line):
        return tuple(int(v) for v in line.split()[1:3])

    def read_spacing(line):
        return tuple(float(v) for v in line.split()[1:3])

    def read_num_values(line):
        return int(line.split()[1].strip())

    def parse_field_labels(line):
        return line.split()[1:]

    def read_header_string(fp):
        buf_size = 1024
        header_str = ""

        while True:
            buf = fp.read(buf_size)

            pos = buf.find(b'\0')

            if pos != -1:
                header_str += buf[:pos].decode("ascii")
                offset = buf_size - pos - 1
                fp.seek(-offset, 1)
                break
            else:
                header_str += buf.decode("ascii")

        return header_str

    info = {}
    header_str = read_header_string(fp)

    for line in header_str.splitlines():
        line_type = line.split(maxsplit=1)[0].upper()

        if line_type == "SHAPE":
            info['shape'] = read_shape(line)
        elif line_type == "SPACING":
            info['spacing'] = read_spacing(line)
        elif line_type == "ORIGIN":
            info['origin'] = read_spacing(line)
        elif line_type == "FIELDS":
            fields = parse_field_labels(line)
        elif line_type == "NUMDATA":
            num_values = read_num_values(line)

    info['num_bins'] = info['shape'][0] * info['shape'][1]

    return fields, num_values, info

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

# Base class for storing flow data
"""
class flow_data :
    # ...
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
        droplet_data.sub_angle_left = []
        droplet_data.sub_angle_right = []
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

    def save_to_file(droplet_data, save_dir):
        print("[densmap] Saving to .txt files")
        np.savetxt(save_dir+'/time.txt', np.array(droplet_data.time))
        np.savetxt(save_dir+'/foot_l.txt', np.array(droplet_data.foot_left)[:,0])
        np.savetxt(save_dir+'/foot_r.txt', np.array(droplet_data.foot_right)[:,0])
        np.savetxt(save_dir+'/angle_l.txt', np.array(droplet_data.angle_left))
        np.savetxt(save_dir+'/angle_r.txt', np.array(droplet_data.angle_right))
        np.savetxt(save_dir+'/sub_angle_r.txt', np.array(droplet_data.sub_angle_right))
        np.savetxt(save_dir+'/sub_angle_l.txt', np.array(droplet_data.sub_angle_left))
        np.savetxt(save_dir+'/radius_fit.txt', droplet_data.radius_circle)
        np.savetxt(save_dir+'/angle_fit.txt', droplet_data.angle_circle)

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

    def movie_contour(droplet_data, crop_x, crop_z, dz, circle=True, contact_line=True, fun_sub=None, dfun_sub=None):
        mpl.use("Agg")
        print("[densmap] Producing movie of the interface dynamics")
        FFMpegWriter = manimation.writers['ffmpeg']
        metadata = dict(title='Spreading Droplet Contour', artist='Michele Pellegrino',
            comment='Just the tracked contour of a spreding droplet')
        writer = FFMpegWriter(fps=30, metadata=metadata)
        fig = plt.figure()
        fig_cont, = plt.plot([], [], 'k-', linewidth=1.5)
        # CURRENTLY TESTING: CONTACT LINE OVER ROUGH SUBSTRATES #
        fig_left, = plt.plot([], [], 'r-', linewidth=1.0)
        fig_right, = plt.plot([], [], 'b-', linewidth=1.0)
        fig_pl, = plt.plot([], [], 'r.', linewidth=1.5)
        fig_pr, = plt.plot([], [], 'b.', linewidth=1.5)
        if not(fun_sub==None) :
            x_eval = np.linspace(crop_x[0], crop_x[1], 500)
            fig_substrate, = plt.plot([], [], 'm-', linewidth=1.0)
            fig_substrate.set_data(x_eval, fun_sub(x_eval))
        #########################################################
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
                    # CURRENTLY TESTING: CONTACT LINE OVER ROUGH SUBSTRATES #
                    fig_left.set_data([droplet_data.foot_left[i][0], droplet_data.foot_left[i][0]+dx_l],
                        [droplet_data.foot_left[i][1], droplet_data.foot_left[i][1]+dz])
                    fig_right.set_data([droplet_data.foot_right[i][0], droplet_data.foot_right[i][0]+dx_r],
                        [droplet_data.foot_right[i][1], droplet_data.foot_right[i][1]+dz])
                    fig_pl.set_data(droplet_data.foot_left[i][0], droplet_data.foot_left[i][1])
                    fig_pr.set_data(droplet_data.foot_right[i][0], droplet_data.foot_right[i][1])
                    #########################################################
                if circle :
                    fig_cir.set_data(circle_x, circle_z)
                plt.axis('scaled')
                plt.xlim(crop_x[0], crop_x[1])      
                plt.ylim(crop_z[0], crop_z[1])
                writer.grab_frame()
                textvar.remove()
        mpl.use("TkAgg")

    def save_xvg(droplet_data, folder) :
        for k in range(len(droplet_data.contour)) : 
            output_interface_xmgrace(droplet_data.contour[k], folder+"/interface_"+str(k+1).zfill(5)+".xvg")

class shear_data :

    """
        Legend:
            'bl' = bottom left
            'br' = bottom right
            'tl' = top left
            'tr' = top right
    """

    def __init__(self):
        print("[densmap] Initializing contour data structure")
        self.time = []
        self.contour = []
        self.branch = dict()
        self.branch['bl'] = []
        self.branch['br'] = []
        self.branch['tl'] = []
        self.branch['tr'] = []
        self.foot = dict()
        self.foot['bl'] = []
        self.foot['br'] = []
        self.foot['tl'] = []
        self.foot['tr'] = []
        self.angle = dict()
        self.angle['bl'] = []
        self.angle['br'] = []
        self.angle['tl'] = []
        self.angle['tr'] = []
        self.cotangent = dict()
        self.cotangent['bl'] = []
        self.cotangent['br'] = []
        self.cotangent['tl'] = []
        self.cotangent['tr'] = []
        self.circle_rad_l = []
        self.circle_xc_l = []
        self.circle_zc_l = []
        self.circle_res_l = []
        self.circle_rad_r = []
        self.circle_xc_r = []
        self.circle_zc_r = []
        self.circle_res_r = []
        self.xcom = []
        self.zcom = []
        self.angle_circle_mean = []

    def save_to_file(self, save_dir):
        print("[densmap] Saving to .txt files")
        np.savetxt(save_dir+'/time.txt', np.array(self.time))
        radius_upper = np.abs( np.array(self.foot['tr']) - np.array(self.foot['tl']) )
        radius_lower = np.abs( np.array(self.foot['br']) - np.array(self.foot['bl']) )
        position_upper = 0.5*np.abs( np.array(self.foot['tr']) + np.array(self.foot['tl']) )
        position_lower = 0.5*np.abs( np.array(self.foot['br']) + np.array(self.foot['bl']) )
        np.savetxt(save_dir+'/radius_upper.txt', radius_upper[:,0])
        np.savetxt(save_dir+'/radius_lower.txt', radius_lower[:,0])
        np.savetxt(save_dir+'/position_upper.txt', position_upper[:,0])
        np.savetxt(save_dir+'/position_lower.txt', position_lower[:,0])
        np.savetxt(save_dir+'/angle_bl.txt', self.angle['bl'])
        np.savetxt(save_dir+'/angle_br.txt', self.angle['br'])
        np.savetxt(save_dir+'/angle_tl.txt', self.angle['tl'])
        np.savetxt(save_dir+'/angle_tr.txt', self.angle['tr'])
        np.savetxt(save_dir+'/xcom.txt', self.xcom)
        np.savetxt(save_dir+'/zcom.txt', self.zcom)
        np.savetxt(save_dir+'/angle_circle.txt', self.angle_circle_mean)

    def plot_radius(self, fig_name='spreading_radius.eps'):
        mpl.use("Agg")
        print("[densmap] Producing plot for spreading radius")
        radius_upper = np.abs( np.array(self.foot['tr']) - np.array(self.foot['tl']) )
        radius_lower = np.abs( np.array(self.foot['br']) - np.array(self.foot['bl']) )
        plt.figure()
        """
            Change once the upper half tracking is added
        """
        plt.plot(self.time, radius_upper[:,0], 'k-', label='upper')
        plt.plot(self.time, radius_lower[:,0], 'g-', label='lower')
        plt.title('Spreading radius', fontsize=20.0)
        plt.xlabel('t [ps]', fontsize=20.0)
        plt.ylabel('R(t) [nm]', fontsize=20.0)
        plt.legend()
        plt.show()
        plt.savefig(fig_name)
        mpl.use("TkAgg")

    def plot_angles(self, fig_name='contact_angles.eps'):
        mpl.use("Agg")
        print("[densmap] Producing plot for contact angles")
        plt.figure()
        plt.plot(self.time, self.angle['bl'], 'b-', label='bottom left')
        plt.plot(self.time, self.angle['br'], 'r-', label='bottom right')
        plt.plot(self.time, self.angle['tl'], 'c-', label='top left')
        plt.plot(self.time, self.angle['tr'], 'm-', label='top right')
        plt.title('Contact angle', fontsize=20.0)
        plt.xlabel('t [ps]', fontsize=20.0)
        plt.ylabel('theta(t) [deg]', fontsize=20.0)
        plt.legend()
        plt.show()
        plt.savefig(fig_name)
        mpl.use("TkAgg")

    def movie_contour(self, crop_x, crop_z, dz, circle=True, contact_line=True):
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
        fig_cir_right, = plt.plot([], [], 'k--', linewidth=1.00)
        fig_cir_left, = plt.plot([], [], 'k--', linewidth=1.00)
        fig_com, = plt.plot([], [], 'ks', linewidth=1.5)
        plt.title('Droplet Shear')
        plt.xlabel('x [nm]')
        plt.ylabel('z [nm]')
        s = np.linspace(0,2*np.pi,250)
        with writer.saving(fig, "contour_movie.mp4", 250):
            for i in range( len(self.contour) ):
                dx_bl = dz * self.cotangent['bl'][i]
                dx_br = dz * self.cotangent['br'][i]
                dx_tl = dz * self.cotangent['tl'][i]
                dx_tr = dz * self.cotangent['tr'][i]
                if circle :
                    circle_x_l = self.circle_xc_l[i] + self.circle_rad_l[i]*np.cos(s)
                    circle_z_l = self.circle_zc_l[i] + self.circle_rad_l[i]*np.sin(s)
                    circle_x_r = self.circle_xc_r[i] + self.circle_rad_r[i]*np.cos(s)
                    circle_z_r = self.circle_zc_r[i] + self.circle_rad_r[i]*np.sin(s)
                fig_cont.set_data(self.contour[i][0,:], self.contour[i][1,:])
                fig_com.set_data(self.xcom[i], self.zcom[i])
                t_label = str(self.time[i])+' ps'
                """
                    The position of the time label should be an input (or at leat a macro)
                """
                textvar = plt.text(1.5, 14.0, t_label)
                if contact_line :
                    fig_fbl.set_data([self.foot['bl'][i][0], self.foot['bl'][i][0]+dx_bl],
                        [self.foot['bl'][i][1], self.foot['bl'][i][1]+dz])
                    fig_fbr.set_data([self.foot['br'][i][0], self.foot['br'][i][0]+dx_br],
                        [self.foot['br'][i][1], self.foot['br'][i][1]+dz])
                    fig_pbl.set_data(self.foot['bl'][i][0], self.foot['bl'][i][1])
                    fig_pbr.set_data(self.foot['br'][i][0], self.foot['br'][i][1])
                    # Flipping again to the upper wall
                    fig_ftl.set_data([self.foot['tl'][i][0], self.foot['tl'][i][0]+dx_tl],
                        [crop_z-self.foot['tl'][i][1], crop_z-self.foot['tl'][i][1]-dz])
                    fig_ftr.set_data([self.foot['tr'][i][0], self.foot['tr'][i][0]+dx_tr],
                        [crop_z-self.foot['tr'][i][1], crop_z-self.foot['tr'][i][1]-dz])
                    fig_ptl.set_data(self.foot['tl'][i][0], crop_z-self.foot['tl'][i][1])
                    fig_ptr.set_data(self.foot['tr'][i][0], crop_z-self.foot['tr'][i][1])
                    fig_pos_up.set_data( 0.5*(self.foot['tl'][i][0]+self.foot['tr'][i][0]), \
                        0.5*(crop_z-self.foot['tl'][i][1]+crop_z-self.foot['tr'][i][1]) )
                    fig_pos_lw.set_data( 0.5*(self.foot['bl'][i][0]+self.foot['br'][i][0]), \
                        0.5*(self.foot['bl'][i][1]+self.foot['br'][i][1]) )
                if circle :
                    fig_cir_left.set_data(circle_x_l, circle_z_l)
                    fig_cir_right.set_data(circle_x_r, circle_z_r)
                plt.axis('scaled')
                plt.xlim(0, crop_x)
                plt.ylim(0, crop_z)
                writer.grab_frame()
                textvar.remove()
        mpl.use("TkAgg")

    def save_xvg(self, folder, mode='contour') :
        if mode=='contour' :
            for k in range(len(self.contour)) : 
                output_interface_xmgrace(self.contour[k], folder+"/interface_"+str(k+1).zfill(5)+".xvg")
        elif mode=='interface' :
            for k in range(len(self.branch['bl'])) :
                output_interface_xmgrace(self.branch['bl'][k], folder+"/int_l_"+str(k+1).zfill(5)+".xvg")
            for k in range(len(self.branch['br'])) :
                output_interface_xmgrace(self.branch['br'][k], folder+"/int_r_"+str(k+1).zfill(5)+".xvg")

    def plot_contact_line_pdf(self, N_in=0,  fig_name='cl_pdf.eps') :
        signal_tr = np.array(self.foot['tr'])[:,0]
        N = len(signal_tr)
        sign_mean_tr, sign_std_tr, bin_vector_tr, distribution_tr = position_distribution( signal_tr[N_in:], int(np.sqrt(N-N_in)) ) 
        signal_tl = np.array(self.foot['tl'])[:,0]
        sign_mean_tl, sign_std_tl, bin_vector_tl, distribution_tl = position_distribution( signal_tl[N_in:], int(np.sqrt(N-N_in)) ) 
        signal_br = np.array(self.foot['br'])[:,0]
        sign_mean_br, sign_std_br, bin_vector_br, distribution_br = position_distribution( signal_br[N_in:], int(np.sqrt(N-N_in)) ) 
        signal_bl = np.array(self.foot['bl'])[:,0]
        sign_mean_bl, sign_std_bl, bin_vector_bl, distribution_bl = position_distribution( signal_bl[N_in:], int(np.sqrt(N-N_in)) )
        mpl.use("Agg")
        print("[densmap] Producing plot for contact angles")
        plt.figure()
        plt.step(bin_vector_bl, distribution_bl, 'b-', label='BL, mean='+"{:.3f}".format(sign_mean_bl)+"nm")
        plt.step(bin_vector_br, distribution_br, 'r-', label='BR, mean='+"{:.3f}".format(sign_mean_br)+"nm")
        plt.step(bin_vector_tl, distribution_tl, 'c-', label='TL, mean='+"{:.3f}".format(sign_mean_tl)+"nm")
        plt.step(bin_vector_tr, distribution_tr, 'm-', label='TR, mean='+"{:.3f}".format(sign_mean_tr)+"nm")
        plt.title('CL PDF', fontsize=20.0)
        plt.xlabel('position [nm]', fontsize=20.0)
        plt.ylabel('PDF', fontsize=20.0)
        plt.legend()
        plt.show()
        plt.savefig(fig_name)
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
    bin = 'y',
    n_bin_x = 0,
    n_bin_z = 0
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
    Read temperature field from .dat data
"""
def read_temperature_file (
    filename
    ) :

    data, info = read_data(filename)
    # print(data)
    Nx = info['shape'][0]
    Nz = info['shape'][1]
    temp = np.array( data['T'] )
    temp = temp.reshape((Nx,Nz))

    return temp

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

    contour = sk.measure.find_contours(density_array, density_target, fully_connected='high')

    assert len(contour)>=1, \
        "No contour line found for the target density value"

    if len(contour)>1 :
        # print( "[densmap] More than one contour found for the target density value (returning the one with most points)" )
        contour = sorted(contour, key=lambda x : len(x))

    # From indices space to physical space
    h = np.array([[hx, 0.0],[0.0, hz]])
    contour = np.matmul(h,(contour[-1].transpose()))

    # Evaluate at the center of the bin
    contour = contour + np.array([[0.5*hx],[0.5*hz]])

    return contour

"""
    Pin-point the contact line position over a corrugated substrate
"""
def detect_contact_rough(
    contour,
    delta_th,
    fun_sub
    ) :

    N = len(contour[0,:])
    delta = np.abs( contour[1,:] - fun_sub(contour[0,:]) )
    for i in range(N) :
        if delta[i] > delta_th :
            delta[i] = np.nan
        else :
            delta[i] = 0
    x = contour[0,:] + delta
    idx_l = np.nanargmin(x)
    idx_r = np.nanargmax(x)

    return idx_l, idx_r

"""
    Left and right branch
"""
def retrace_local_profile (
    contour,
    z_th,
    idx_l,
    idx_r
    ) :

    fl = True
    fr = True
    
    # Left branch
    x_l = []
    z_l = []
    i = idx_l
    while (fl) :
        if contour[1,i] < z_th + contour[1,idx_l] :
            x_l.append(contour[0,i])
            z_l.append(contour[1,i])
            i += 1
        else :
            fl = False

    # Right branch
    x_r = []
    z_r = []
    i = idx_r
    while (fr) :
        if contour[1,i] < z_th + contour[1,idx_r] :
            x_r.append(contour[0,i])
            z_r.append(contour[1,i])
            i -= 1
        else :
            fr = False

    return np.stack((np.array(x_l), np.array(z_l))), np.stack((np.array(x_r), np.array(z_r)))

"""
    Detect x and z coordinates of the c.o.m.
"""
def detect_com (
    density_array,
    X,
    Z
    ) :

    x_com = np.sum(np.multiply(density_array,X))/np.sum(density_array)
    z_com = np.sum(np.multiply(density_array,Z))/np.sum(density_array)

    return x_com, z_com

def detect_interface_int (
    density_array,
    density_target,
    hx,
    hz,
    z0,
    offset = 0.5
    ) :

    Nx = density_array.shape[0]
    Nz = density_array.shape[1]
    x = hx*np.arange(0.0, Nx, 1.0, dtype=float)+0.5*hx
    z = hz*np.arange(0.0, Nz, 1.0, dtype=float)+0.5*hz

    M = int(np.ceil(offset*Nx))
    i0 = np.abs(z-z0).argmin()
    
    left_branch = np.zeros( (2,Nz-2*i0), dtype=float )
    right_branch = np.zeros( (2,Nz-2*i0), dtype=float )

    for j in range(i0,Nz-i0) :
        
        left_branch[1,j-i0] = z[j]
        right_branch[1,j-i0] = z[j]

        for i in range(0,M) :
            if density_array[i,j] > density_target :
                left_branch[0,j-i0] = \
                        ((density_array[i,j]-density_target)*x[i-1]+(density_target-density_array[i-1,j])*x[i])/(density_array[i,j]-density_array[i-1,j])
                break
        
        for i in range(Nx-1, Nx-M+1, -1) :
            if density_array[i,j] > density_target :
                right_branch[0,j-i0] = \
                        ((density_array[i,j]-density_target)*x[i+1]+(density_target-density_array[i+1,j])*x[i])/(density_array[i,j]-density_array[i+1,j])
                break

    return left_branch, right_branch

"""
    Aux. function to find the mean local bulk density for a horizontal 'slice' of water
"""

def mean_density_loc(density_slice, x, nwin) :
    xcom = np.sum(density_slice*x)/np.sum(density_slice)
    icom = np.abs(x-xcom).argmin()
    return np.mean(density_slice[icom-nwin:icom+nwin+1])

def detect_interface_loc (
    density_array,
    hx,
    hz,
    z0,
    zmax,
    offset = 0.5,
    wall = 'l',
    nwin = 25
    ) :
    
    Nx = density_array.shape[0]
    Nz = density_array.shape[1]
    x = hx*np.arange(0.0, Nx, 1.0, dtype=float)+0.5*hx
    z = hz*np.arange(0.0, Nz, 1.0, dtype=float)+0.5*hz
    Lz = Nz*hz

    M = int(np.ceil(offset*Nx))

    if wall=='l' :
        i0 = np.abs(z-z0).argmin()
        imax = np.abs(z-zmax).argmin()
        left_branch = np.zeros( (2,imax-i0+1), dtype=float )
        right_branch = np.zeros( (2,imax-i0+1), dtype=float )
        for j in range(i0, imax+1) :
            left_branch[1,j-i0] = z[j]
            right_branch[1,j-i0] = z[j]
            density_target = 0.5*mean_density_loc(density_array[:,j], x, nwin)
            for i in range(1,M) :
                if density_array[i,j] > density_target :
                    left_branch[0,j-i0] = \
                            ((density_array[i,j]-density_target)*x[i-1]+(density_target-density_array[i-1,j])*x[i])/(density_array[i,j]-density_array[i-1,j])
                    break
            for i in range(Nx-2, Nx-M+1, -1) :
                if density_array[i,j] > density_target :
                    right_branch[0,j-i0] = \
                            ((density_array[i,j]-density_target)*x[i+1]+(density_target-density_array[i+1,j])*x[i])/(density_array[i,j]-density_array[i+1,j])
                    break
    else :
        imax = np.abs(z-Lz+z0).argmin()+1
        i0 = np.abs(z-Lz+zmax).argmin()+1
        left_branch = np.zeros( (2,imax-i0), dtype=float )
        right_branch = np.zeros( (2,imax-i0), dtype=float )
        for j in range(i0, imax) :
            left_branch[1,j-i0] = z[j]
            right_branch[1,j-i0] = z[j]
            density_target = 0.5*mean_density_loc(density_array[:,j], x, nwin)
            for i in range(1,M) :
                if density_array[i,j] > density_target :
                    left_branch[0,j-i0] = \
                            ((density_array[i,j]-density_target)*x[i-1]+(density_target-density_array[i-1,j])*x[i])/(density_array[i,j]-density_array[i-1,j])
                    break
            for i in range(Nx-2, Nx-M+1, -1) :
                if density_array[i,j] > density_target :
                    right_branch[0,j-i0] = \
                            ((density_array[i,j]-density_target)*x[i+1]+(density_target-density_array[i+1,j])*x[i])/(density_array[i,j]-density_array[i+1,j])
                    break

    return left_branch, right_branch

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
    contact_line = True,
    f_sub = None,
    df_sub = None,
    mode = 'sk',
    comshift=False
    ) :

    # Modes for obtaining the L/R interfaces
    """
        'sk'  : marching squares, smoothed density ("soft" way)
        'int' : bin-wise interpolation, non-smoothed density ("hard" way)
    """
    assert mode=='sk' or mode =='int', "Unrecognized interface traking mode"

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
    x = hx*(np.arange(0.0,Nx,1.0, dtype=float)+0.5)
    z = hz*(np.arange(0.0,Nz,1.0, dtype=float)+0.5)
    X, Z = np.meshgrid(x, z, sparse=False, indexing='ij')
    x_com, z_com = detect_com(density_array, X, Z)
    x_ref = hx*(Nx//2+0.5)
    offset_ix = Nx//2-int(x_com/hx)
    density_array = np.roll(density_array,offset_ix*comshift,axis=0)

    print("[densmap] Initialize smoothing kernel")
    smoother = smooth_kernel(fit_param.r_mol, hx, hz)

    # Append the values for the first time-step
    if mode == 'sk' :
        smooth_density_array = convolute(density_array, smoother)
        bulk_density = detect_bulk_density(smooth_density_array, density_th=fit_param.max_vapour_density)
        intf_contour = detect_contour(smooth_density_array, 0.5*bulk_density, hx, hz)
    else :
        bulk_density = detect_bulk_density(density_array, density_th=fit_param.max_vapour_density)
        intf_contour = detect_contour(density_array, 0.5*bulk_density, hx, hz)
    if contact_line :
        # Flat substrate
        if f_sub == None :
            if mode == 'sk' :
                left_branch, right_branch, points_l, points_r = \
                    detect_contact_line(intf_contour, z_min=fit_param.substrate_location,
                    z_max=fit_param.bulk_location, x_half=fit_param.simmetry_plane)
                foot_l, foot_r, theta_l, theta_r, cot_l, cot_r = \
                    detect_contact_angle(points_l, points_r, order=fit_param.interpolation_order)
                sub_theta_l = 0.0
                sub_theta_r = 0.0
            else :
                z0 = fit_param.substrate_location
                zmax = fit_param.bulk_location
                left_branch, right_branch = detect_interface_loc(density_array, hx, hz, z0, zmax)
                # Take the first 10 points (change -> make it generic!)
                points_l = left_branch[:,:]
                points_r = right_branch[:,:]
                foot_l, foot_r, theta_l, theta_r, cot_l, cot_r = \
                    detect_contact_angle(points_l, points_r, order=fit_param.interpolation_order)
                sub_theta_l = 0.0
                sub_theta_r = 0.0
        # Rough substrate
        # Leave as it is for the time being...
        else :
            ### THIS SHOULD BE USER-DEFINED! ###
            hbins = 0.6
            hdz = 2.0
            ### ############################ ###
            idx_cll, idx_clr = detect_contact_rough( intf_contour, hbins, f_sub )
            points_l, points_r = retrace_local_profile(intf_contour, hdz, idx_cll, idx_clr)
            foot_l, foot_r, theta_l, theta_r, cot_l, cot_r = \
                detect_contact_angle(points_l, points_r, order=fit_param.interpolation_order)
            left_branch = np.NaN
            right_branch = np.NaN
            # SUB THETA 
            sub_theta_l = - np.rad2deg( np.arctan( df_sub(foot_l[0]) ) )
            sub_theta_r = - np.rad2deg( np.arctan( df_sub(foot_r[0]) ) )
    else :
        left_branch = np.NaN
        right_branch = np.NaN
        points_l = np.NaN
        points_r = np.NaN
        foot_l = np.NaN
        foot_r = np.NaN
        theta_l = np.NaN
        theta_r = np.NaN
        sub_theta_l = np.NaN
        sub_theta_r = np.NaN
        cot_l = np.NaN
        cot_r = np.NaN
    xc, zc, R, residue = circle_fit_droplet(intf_contour, z_th=fit_param.substrate_location)

    foot_l = (foot_l[0]+comshift*(x_com-x_ref),foot_l[1])
    foot_r = (foot_r[0]+comshift*(x_com-x_ref),foot_r[1])

    CD.time.append( fit_param.time_step*k_init )
    CD.contour.append( intf_contour )
    CD.branch_left.append( left_branch )
    CD.branch_right.append( right_branch )
    CD.foot_left.append( foot_l )
    CD.foot_right.append( foot_r )
    CD.angle_left.append( theta_l )
    CD.angle_right.append( theta_r )
    CD.sub_angle_left.append( sub_theta_l )
    CD.sub_angle_right.append( sub_theta_r )
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

        ### COM RE-CENTERING
        x = hx*(np.arange(0.0,Nx,1.0, dtype=float)+0.5)
        z = hz*(np.arange(0.0,Nz,1.0, dtype=float)+0.5)
        X, Z = np.meshgrid(x, z, sparse=False, indexing='ij')
        x_com, z_com = detect_com(density_array, X, Z)
        x_ref = hx*(Nx//2+0.5)
        offset_ix = Nx//2-int(x_com/hx)
        density_array = np.roll(density_array,offset_ix*comshift,axis=0)

        smooth_density_array = convolute(density_array, smoother)
        bulk_density = detect_bulk_density(smooth_density_array, density_th=fit_param.max_vapour_density)
        intf_contour = detect_contour(smooth_density_array, 0.5*bulk_density, hx, hz)
        if contact_line :
            # Flat substrate
            if f_sub == None :
                if mode == 'sk' :
                    left_branch, right_branch, points_l, points_r = \
                        detect_contact_line(intf_contour, z_min=fit_param.substrate_location,
                        z_max=fit_param.bulk_location, x_half=fit_param.simmetry_plane)
                    foot_l, foot_r, theta_l, theta_r, cot_l, cot_r = \
                        detect_contact_angle(points_l, points_r, order=fit_param.interpolation_order)
                    sub_theta_l = 0.0
                    sub_theta_r = 0.0
                else :
                    z0 = fit_param.substrate_location
                    zmax = fit_param.bulk_location
                    left_branch, right_branch = detect_interface_loc(density_array, hx, hz, z0, zmax)
                    # Take the first 10 points (change -> make it generic!)
                    points_l = left_branch[:,:]
                    points_r = right_branch[:,:]
                    foot_l, foot_r, theta_l, theta_r, cot_l, cot_r = \
                        detect_contact_angle(points_l, points_r, order=fit_param.interpolation_order)
                    sub_theta_l = 0.0
                    sub_theta_r = 0.0
            # Rough substate
            # Leave as it is for the time being...
            else :
                ### THIS SHOULD BE USER-DEFINED! ###
                hbins = 0.6
                hdz = 2.0
                ### ############################ ###
                idx_cll, idx_clr = detect_contact_rough(intf_contour, hbins, f_sub )
                points_l, points_r = retrace_local_profile(intf_contour, hdz, idx_cll, idx_clr)
                foot_l, foot_r, theta_l, theta_r, cot_l, cot_r = \
                    detect_contact_angle(points_l, points_r, order=fit_param.interpolation_order)
                left_branch = np.NaN
                right_branch = np.NaN
                # SUB THETA 
                sub_theta_l = - np.rad2deg( np.arctan( df_sub(foot_l[0]) ) )
                sub_theta_r = - np.rad2deg( np.arctan( df_sub(foot_r[0]) ) )
        else :
            left_branch = np.NaN
            right_branch = np.NaN
            points_l = np.NaN
            points_r = np.NaN
            foot_l = np.NaN
            foot_r = np.NaN
            theta_l = np.NaN
            theta_r = np.NaN
            sub_theta_l = np.NaN
            sub_theta_r = np.NaN
            cot_l = np.NaN
            cot_r = np.NaN
        xc, zc, R, residue = circle_fit_droplet(intf_contour, z_th=fit_param.substrate_location)

        foot_l = (foot_l[0]+comshift*(x_com-x_ref),foot_l[1])
        foot_r = (foot_r[0]+comshift*(x_com-x_ref),foot_r[1])

        CD.time.append( fit_param.time_step*k )
        CD.contour.append( intf_contour )
        CD.branch_left.append( left_branch )
        CD.branch_right.append( right_branch )
        CD.foot_left.append( foot_l )
        CD.foot_right.append( foot_r )
        CD.angle_left.append( theta_l )
        CD.angle_right.append( theta_r )
        CD.sub_angle_left.append( sub_theta_l )
        CD.sub_angle_right.append( sub_theta_r )
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
    Modification: add more than one ensemble
    ens = 0: no ensemble averaging
"""
def shear_tracking (
    folder_name,
    k_init,
    k_end,
    fit_param,
    file_root = '/flow_',
    contact_line = True,
    fit_ca = True,
    mode = 'sk',
    ens = 0
    ) :

    # Modes for obtaining the L/R interfaces
    """
        'sk'  : marching squares, smoothed density ("soft" way)
        'int' : bin-wise interpolation, non-smoothed density ("hard" way)
        'loc' : bin-wise interpolation local to the substrate (efficient to extract microscopic c.a.)
    """
    assert mode=='sk' or mode=='int' or mode=='loc', "Unrecognized interface traking mode"

    # Data structure that will be outputted
    CD = shear_data()

    # Read the first density snapshot, in order to get the values needed to contruct the smoothing kernel
    file_name = folder_name+file_root+str(k_init).zfill(5)+'.dat'
    print("[densmap] Reading "+file_name)
    density_array = np.NaN
    if ens>0 :
        file_name = folder_name+'_ens1'+file_root+str(k_init).zfill(5)+'.dat'
        density_array = read_density_file(file_name, bin='y')
        for k in range(1,ens) :
            file_name = folder_name+'_ens'+str(k+1)+file_root+str(k_init).zfill(5)+'.dat'
            density_array += read_density_file(file_name, bin='y')
        density_array /= ens
    else :
        density_array = read_density_file(file_name, bin='y')
    Nx = density_array.shape[0]
    Nz = density_array.shape[1]
    hx = fit_param.lenght_x/Nx
    hz = fit_param.lenght_z/Nz
    # For c.o.m. computation #
    # Cell-centered
    x = hx*(np.arange(0.0,Nx,1.0, dtype=float)+0.5)
    z = hz*(np.arange(0.0,Nz,1.0, dtype=float)+0.5)
    X, Z = np.meshgrid(x, z, sparse=False, indexing='ij')
    x_com, z_com = detect_com(density_array, X, Z)
    # ###################### #
    print("[densmap] Initialize smoothing kernel")
    """
        Should actually check if r_mol is less than the bin size; in that case
        performing density averaging is useless
    """
    smoother = smooth_kernel(fit_param.r_mol, hx, hz)

    # Append the values for the first time-step
    if mode == 'sk' :
        smooth_density_array = convolute(density_array, smoother)
        bulk_density = detect_bulk_density(smooth_density_array, density_th=fit_param.max_vapour_density)
        intf_contour = detect_contour(smooth_density_array, 0.5*bulk_density, hx, hz)
    else :
        bulk_density = detect_bulk_density(density_array, density_th=fit_param.max_vapour_density)
        intf_contour = detect_contour(density_array, 0.5*bulk_density, hx, hz)

    if contact_line :
        if mode == 'sk' :
            """
                Obtain contour using scikit image processing
                More roboust, less precise
            """
            b_left_branch, b_right_branch, b_points_l, b_points_r = \
                detect_contact_line(intf_contour, z_min=fit_param.substrate_location,
                z_max=fit_param.bulk_location, x_half=fit_param.simmetry_plane)
            # Flip interface contour
            intf_contour_flip = np.stack((intf_contour[0,:], fit_param.lenght_z-intf_contour[1,:]))
            t_left_branch, t_right_branch, t_points_l, t_points_r = \
                detect_contact_line(intf_contour_flip, z_min=fit_param.substrate_location,
                z_max=fit_param.bulk_location, x_half=fit_param.simmetry_plane)
        elif mode == 'int' :
            """
                Span-wise interpolation on the global bulk density value
                Less robust, more precise
            """
            z0 = fit_param.substrate_location
            b_left_branch, b_right_branch = detect_interface_int(density_array, 0.5*bulk_density, hx, hz, z0)
            t_left_branch = np.stack( (np.flip(b_left_branch[0,:]), \
                fit_param.lenght_z-b_left_branch[1,:]) )
            t_right_branch = np.stack( (np.flip(b_right_branch[0,:]), \
                fit_param.lenght_z-b_right_branch[1,:]) )
            # Take the first 10 points (change -> make it generic!)
            b_points_l = b_left_branch[:,0:15:3]
            b_points_r = b_right_branch[:,0:15:3]
            t_points_l = np.stack( (t_left_branch[0,0:15:3], \
                fit_param.lenght_z-t_left_branch[1,0:15:3] ) )
            t_points_r = np.stack( (t_right_branch[0,0:15:3], \
                fit_param.lenght_z-t_right_branch[1,0:15:3] ) )
        else :
            """
                Same as above, but local to the water slice along x
                Captures the layering region better
            """
            z0 = fit_param.substrate_location
            zmax = fit_param.bulk_location
            b_left_branch, b_right_branch = detect_interface_loc(density_array, hx, hz, z0, zmax, wall='l')
            b_points_l = b_left_branch[:,:]
            b_points_r = b_right_branch[:,:]
            # Add top wall measurement
            t_left_branch, t_right_branch = detect_interface_loc(density_array, hx, hz, z0, zmax, wall='u')
            t_left_branch = np.stack( (np.flip(t_left_branch[0,:]), np.flip(t_left_branch[1,:])) )
            t_right_branch = np.stack( (np.flip(t_right_branch[0,:]), np.flip(t_right_branch[1,:])) )
            t_points_l = np.stack( (t_left_branch[0,:], \
                fit_param.lenght_z-t_left_branch[1,:] ) )
            t_points_r = np.stack( (t_right_branch[0,:], \
                fit_param.lenght_z-t_right_branch[1,:] ) )
    else :
        b_foot_l = np.NaN
        b_foot_r = np.NaN
        t_foot_l = np.NaN
        t_foot_r = np.NaN
        b_left_branch = np.NaN
        b_right_branch = np.NaN
        b_points_l = np.NaN
        b_points_r = np.NaN
        t_left_branch = np.NaN
        t_right_branch = np.NaN
        t_points_l = np.NaN
        t_points_r = np.NaN
        b_theta_l = np.NaN
        b_theta_r = np.NaN
        b_cot_l = np.NaN
        b_cot_r = np.NaN
        t_theta_l = np.NaN
        t_theta_r = np.NaN
        t_cot_l = np.NaN
        t_cot_r = np.NaN

    if fit_ca :
        b_foot_l, b_foot_r, b_theta_l, b_theta_r, b_cot_l, b_cot_r = \
            detect_contact_angle(b_points_l, b_points_r, order=fit_param.interpolation_order)
        t_foot_l, t_foot_r, t_theta_l, t_theta_r, t_cot_l, t_cot_r = \
            detect_contact_angle(t_points_l, t_points_r, order=fit_param.interpolation_order)
    elif contact_line :
        b_foot_l = ( b_points_l[0,0], b_points_l[1,0] )
        b_foot_r = ( b_points_r[0,0], b_points_r[1,0] )
        t_foot_l = ( t_points_l[0,0], t_points_l[1,0] )
        t_foot_r = ( t_points_r[0,0], t_points_r[1,0] )
        b_theta_l = np.NaN
        b_theta_r = np.NaN
        b_cot_l = np.NaN
        b_cot_r = np.NaN
        t_theta_l = np.NaN
        t_theta_r = np.NaN
        t_cot_l = np.NaN
        t_cot_r = np.NaN

    xc_l, zc_l, R_l, residue_l = circle_fit_meniscus(intf_contour, z_th=fit_param.substrate_location, 
        Lz=fit_param.lenght_z, Midx=0.5*fit_param.lenght_x, wing='l')
    xc_r, zc_r, R_r, residue_r = circle_fit_meniscus(intf_contour, z_th=fit_param.substrate_location, 
        Lz=fit_param.lenght_z, Midx=0.5*fit_param.lenght_x, wing='r')
    CD.circle_rad_l.append(R_l)
    CD.circle_xc_l.append(xc_l)
    CD.circle_zc_l.append(zc_l)
    CD.circle_res_l.append(residue_l)
    CD.circle_rad_r.append(R_r)
    CD.circle_xc_r.append(xc_r)
    CD.circle_zc_r.append(zc_r)
    CD.circle_res_r.append(residue_r)

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
    CD.xcom.append(x_com)
    CD.zcom.append(z_com)

    # ANGLE FROM CAP FITTING
    h = fit_param.substrate_location
    Lz = fit_param.lenght_z
    cot_circle_bl = (h-zc_l)/np.sqrt(R_l*R_l-(h-zc_l)**2)
    cot_circle_tl = (zc_l+h-Lz)/np.sqrt(R_l*R_l-(zc_l+h-Lz)**2)
    cot_circle_br = (h-zc_r)/np.sqrt(R_r*R_r-(h-zc_r)**2)
    cot_circle_tr = (zc_r+h-Lz)/np.sqrt(R_r*R_r-(zc_r+h-Lz)**2)
    cot_circle = 0.25*(cot_circle_bl+cot_circle_tl+cot_circle_br+cot_circle_tr)
    theta_circle = np.rad2deg( 0.5*math.pi+np.arctan( cot_circle ) )
    theta_circle = (180-theta_circle)*(cot_circle>-1) + theta_circle*(cot_circle<=-1)
    CD.angle_circle_mean.append(theta_circle)

    for k in range(k_init+1, k_end+1) :
        file_name = folder_name+file_root+str(k).zfill(5)+'.dat'
        if k % K_INFO == 0 :
            print("[densmap] Reading "+file_name)
        # Loop

        density_array = np.NaN
        if ens>0 :
            file_name = folder_name+'_ens1'+file_root+str(k).zfill(5)+'.dat'
            density_array = read_density_file(file_name, bin='y')
            for kk in range(1,ens) :
                file_name = folder_name+'_ens'+str(kk+1)+file_root+str(k).zfill(5)+'.dat'
                density_array += read_density_file(file_name, bin='y')
            density_array /= ens
        else :
            density_array = read_density_file(file_name, bin='y')

        x_com, z_com = detect_com(density_array, X, Z)

        # density_array = read_density_file(file_name, bin='y')
        if mode == 'sk' :
            smooth_density_array = convolute(density_array, smoother)
            bulk_density = detect_bulk_density(smooth_density_array, density_th=fit_param.max_vapour_density)
            intf_contour = detect_contour(smooth_density_array, 0.5*bulk_density, hx, hz)
        else :
            bulk_density = detect_bulk_density(density_array, density_th=fit_param.max_vapour_density)
            intf_contour = detect_contour(density_array, 0.5*bulk_density, hx, hz)

        if contact_line :
            if mode == 'sk' :
                b_left_branch, b_right_branch, b_points_l, b_points_r = \
                    detect_contact_line(intf_contour, z_min=fit_param.substrate_location,
                    z_max=fit_param.bulk_location, x_half=fit_param.simmetry_plane)
                # Flip interface contour
                intf_contour_flip = np.stack((intf_contour[0,:], fit_param.lenght_z-intf_contour[1,:]))
                t_left_branch, t_right_branch, t_points_l, t_points_r = \
                    detect_contact_line(intf_contour_flip, z_min=fit_param.substrate_location,
                    z_max=fit_param.bulk_location, x_half=fit_param.simmetry_plane)
            elif mode == 'int' :
                z0 = fit_param.substrate_location
                b_left_branch, b_right_branch = detect_interface_int(density_array, \
                        0.5*bulk_density, hx, hz, z0)
                t_left_branch = np.stack( (np.flip(b_left_branch[0,:]), \
                        fit_param.lenght_z-b_left_branch[1,:]) )
                t_right_branch = np.stack( (np.flip(b_right_branch[0,:]), \
                        fit_param.lenght_z-b_right_branch[1,:]) )
                # Take the first 10 points (change -> make it generic!)
                b_points_l = b_left_branch[:,0:15:3]
                b_points_r = b_right_branch[:,0:15:3]
                t_points_l = np.stack( (t_left_branch[0,0:15:3], \
                    fit_param.lenght_z-t_left_branch[1,0:15:3] ) )
                t_points_r = np.stack( (t_right_branch[0,0:15:3], \
                    fit_param.lenght_z-t_right_branch[1,0:15:3] ) )
            else :
                z0 = fit_param.substrate_location
                zmax = fit_param.bulk_location
                b_left_branch, b_right_branch = detect_interface_loc(density_array, hx, hz, z0, zmax, wall='l')
                b_points_l = b_left_branch[:,:]
                b_points_r = b_right_branch[:,:]
                # Add top wall measurement
                t_left_branch, t_right_branch = detect_interface_loc(density_array, hx, hz, z0, zmax, wall='u')
                t_left_branch = np.stack( (np.flip(t_left_branch[0,:]), np.flip(t_left_branch[1,:])) )
                t_right_branch = np.stack( (np.flip(t_right_branch[0,:]), np.flip(t_right_branch[1,:])) )
                t_points_l = np.stack( (t_left_branch[0,:], \
                    fit_param.lenght_z-t_left_branch[1,:] ) )
                t_points_r = np.stack( (t_right_branch[0,:], \
                    fit_param.lenght_z-t_right_branch[1,:] ) )
        else :
            b_foot_l = np.NaN
            b_foot_r = np.NaN
            t_foot_l = np.NaN
            t_foot_r = np.NaN
            b_left_branch = np.NaN
            b_right_branch = np.NaN
            b_points_l = np.NaN
            b_points_r = np.NaN
            t_left_branch = np.NaN
            t_right_branch = np.NaN
            t_points_l = np.NaN
            t_points_r = np.NaN
            b_theta_l = np.NaN
            b_theta_r = np.NaN
            b_cot_l = np.NaN
            b_cot_r = np.NaN
            t_theta_l = np.NaN
            t_theta_r = np.NaN
            t_cot_l = np.NaN
            t_cot_r = np.NaN

        if fit_ca :
            b_foot_l, b_foot_r, b_theta_l, b_theta_r, b_cot_l, b_cot_r = \
                detect_contact_angle(b_points_l, b_points_r, order=fit_param.interpolation_order)
            t_foot_l, t_foot_r, t_theta_l, t_theta_r, t_cot_l, t_cot_r = \
                detect_contact_angle(t_points_l, t_points_r, order=fit_param.interpolation_order)
        elif contact_line :
            b_foot_l = ( b_points_l[0,0], b_points_l[1,0] )
            b_foot_r = ( b_points_r[0,0], b_points_r[1,0] )
            t_foot_l = ( t_points_l[0,0], t_points_l[1,0] )
            t_foot_r = ( t_points_r[0,0], t_points_r[1,0] )
            b_theta_l = np.NaN
            b_theta_r = np.NaN
            b_cot_l = np.NaN
            b_cot_r = np.NaN
            t_theta_l = np.NaN
            t_theta_r = np.NaN
            t_cot_l = np.NaN
            t_cot_r = np.NaN

        xc_l, zc_l, R_l, residue_l = circle_fit_meniscus(intf_contour, z_th=fit_param.substrate_location, 
            Lz=fit_param.lenght_z, Midx=0.5*fit_param.lenght_x, wing='l')
        xc_r, zc_r, R_r, residue_r = circle_fit_meniscus(intf_contour, z_th=fit_param.substrate_location, 
            Lz=fit_param.lenght_z, Midx=0.5*fit_param.lenght_x, wing='r')
        CD.circle_rad_l.append(R_l)
        CD.circle_xc_l.append(xc_l)
        CD.circle_zc_l.append(zc_l)
        CD.circle_res_l.append(residue_l)
        CD.circle_rad_r.append(R_r)
        CD.circle_xc_r.append(xc_r)
        CD.circle_zc_r.append(zc_r)
        CD.circle_res_r.append(residue_r)

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
        CD.xcom.append(x_com)
        CD.zcom.append(z_com)

        # ANGLE FROM CAP FITTING
        h = fit_param.substrate_location
        Lz = fit_param.lenght_z
        cot_circle_bl = (h-zc_l)/np.sqrt(R_l*R_l-(h-zc_l)**2)
        cot_circle_tl = (zc_l+h-Lz)/np.sqrt(R_l*R_l-(zc_l+h-Lz)**2)
        cot_circle_br = (h-zc_r)/np.sqrt(R_r*R_r-(h-zc_r)**2)
        cot_circle_tr = (zc_r+h-Lz)/np.sqrt(R_r*R_r-(zc_r+h-Lz)**2)
        cot_circle = 0.25*(cot_circle_bl+cot_circle_tl+cot_circle_br+cot_circle_tr)
        theta_circle = np.rad2deg( 0.5*math.pi+np.arctan( cot_circle ) )
        theta_circle = (180-theta_circle)*(cot_circle>-1) + theta_circle*(cot_circle<=-1)
        CD.angle_circle_mean.append(theta_circle)

    return CD

"""
    Finds the better circle fitting the droplet contour
"""
def circle_fit_droplet(intf_contour, z_th=0.0) :
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

def circle_fit_meniscus(intf_contour, z_th, Lz, Midx, wing) :
    assert wing=='l' or wing=='r', \
            "Specify if the wing to fit with LS circle is left (wing='l') or right (wing='r')"
    M = len(intf_contour[0,:])
    data_circle_x = []
    data_circle_z = []
    if wing=='l' :
        for k in range(M) :
            if intf_contour[1,k] > z_th and intf_contour[1,k] < Lz-z_th and intf_contour[0,k]<Midx:
                data_circle_x.append(intf_contour[0,k])
                data_circle_z.append(intf_contour[1,k])
    else :
        for k in range(M) :
            if intf_contour[1,k] > z_th and intf_contour[1,k] < Lz-z_th and intf_contour[0,k]>Midx:
                data_circle_x.append(intf_contour[0,k])
                data_circle_z.append(intf_contour[1,k])
    return cf.least_squares_circle(np.stack((data_circle_x, data_circle_z), axis=1))

"""
    Function to obtain equilibrium radius and c.a.
"""
def equilibrium_from_density ( filename, smoother, density_th, hx, hz, h ) :
    density_array = read_density_file(filename, bin='y')
    smooth_density_array = convolute(density_array, smoother)
    bulk_density = detect_bulk_density(smooth_density_array, density_th)
    intf_contour = detect_contour(smooth_density_array, 0.5*bulk_density, hx, hz)
    xc, zc, R, residue = circle_fit_droplet(intf_contour, h)
    radius_circle = 2*np.sqrt(R*R-(h-zc)**2)
    cot_circle = (h-zc)/np.sqrt(R*R-(h-zc)**2)
    theta_circle = np.rad2deg( -np.arctan( cot_circle )+0.5*math.pi )
    theta_circle = theta_circle + 180*(theta_circle<=0)
    return radius_circle, theta_circle

"""
    Function to output interface points as .xvg file
"""
def output_interface_xmgrace ( intf_contour, file_name='interface.xvg' ) :
    
    f_out = open(file_name, 'w') 
    
    # HEADER
    f_out.write("# Interface coordinates of a droplet\n")
    f_out.write("# \n")
    f_out.write("# Created by: densmap\n")
    f_out.write("# Creation date: [...]\n")
    f_out.write("# Using module version: 0.0.0\n")
    f_out.write("# \n")
    f_out.write("# Working directory: [...]\n")
    f_out.write("# Input files:\n")
    f_out.write("# \t[...]\n")
    f_out.write("# \n")
    f_out.write("# Input options:\n")
    f_out.write("# \tRecenter: [...]\n")
    f_out.write("# \tMass cut-off: [...]\n")
    f_out.write("# \tRadius cut-off: [...]\n")
    f_out.write("# \tRequired # of bins: [...]\n")
    f_out.write("# \n")
    f_out.write("# x (nm) y (nm)\n")
    
    # WRITING INTERFACE
    M = len(intf_contour[0,:])
    for k in range(M) :
        f_out.write("%.3f %.3f\n" % (intf_contour[0,k], intf_contour[1,k]) )

    f_out.close()

"""
    Function to bin the distribution of single c.l. position
"""
def position_distribution ( signal, N, std_mult=6.0 ) :
   
    sign_mean = np.mean(signal) 
    sign_std = np.std(signal)

    x_low = -std_mult*sign_std
    x_upp =  std_mult*sign_std
    
    dx = (x_upp-x_low)/N

    bin_vector = np.linspace(x_low+dx, x_upp-dx, N-1)
    distribution = np.zeros( len(bin_vector), dtype=float )

    for k in range(len(signal)) :
       idx = ( np.abs( bin_vector-(signal[k]-sign_mean) ) ).argmin()
       distribution[idx] += 1.0

    distribution = distribution/(dx*sum(distribution))

    return sign_mean, sign_std, bin_vector, distribution

# Functions to export flow data in ASCII .vtk format (suitable for Paraview)

"""
    Scalars (e.g. density)
"""
def export_scalar_vtk(x, z, hx, hz, Ly, array, file_name="flow_output.vtk"):
    
    fvtk = open(file_name, 'w')

    fvtk.write("# vtk DataFile Version 3.0\n")
    fvtk.write("Vtk output for binned flow configurations (metric in Ångström)\n")
    fvtk.write("ASCII\n")
    fvtk.write("DATASET STRUCTURED_POINTS\n")
    
    fvtk.write("DIMENSIONS "+str(len(x))+" 2 "+str(len(z))+"\n")
    fvtk.write("ORIGIN "+str(10.0*x[0])+" 0.0 "+str(10.0*z[0])+"\n")
    fvtk.write("SPACING "+str(10.0*hx)+" "+str(5.0*Ly)+" "+str(10.0*hz)+"\n")
    
    fvtk.write("CELL_DATA "+str((len(x)-1)*(len(z)-1))+"\n")
    fvtk.write("POINT_DATA "+str(2*len(x)*len(z))+"\n")
    fvtk.write("SCALARS density float 1\n")
    fvtk.write("LOOKUP_TABLE default\n")

    for k in range(len(z)) :
        for j in range(2) :
            for i in range(len(x)) :
                fvtk.write( str( array[i,k] )+"\n" )

    fvtk.close()

"""
    To compare with Phase Field
"""
def export_scalarxy_vtk(x, y, hx, hy, Lz, array, file_name="flow_output.vtk"):
    
    fvtk = open(file_name, 'w')

    fvtk.write("# vtk DataFile Version 3.0\n")
    fvtk.write("Vtk output for binned flow configurations (metric in Ångström)\n")
    fvtk.write("ASCII\n")
    fvtk.write("DATASET STRUCTURED_POINTS\n")
    
    fvtk.write("DIMENSIONS "+str(len(x))+" "+str(len(y))+" 2\n")
    fvtk.write("ORIGIN "+str(x[0])+" "+str(y[0])+" 0.0\n")
    fvtk.write("SPACING "+str(hx)+" "+str(hy)+" "+str(Lz)+"\n")
    
    fvtk.write("CELL_DATA "+str((len(x)-1)*(len(y)-1))+"\n")
    fvtk.write("POINT_DATA "+str(2*len(x)*len(y))+"\n")
    fvtk.write("SCALARS density float 1\n")
    fvtk.write("LOOKUP_TABLE default\n")

    for j in range(2) :
        for k in range(len(y)) :
            for i in range(len(x)) :
                fvtk.write( str( array[i,k] )+"\n" )

    fvtk.close()

"""
    Vectors (e.g. velocity)
"""
def export_vector_vtk(x, z, hx, hz, Ly, array_x, array_z, file_name="flow_output.vtk"):
    
    fvtk = open(file_name, 'w')

    fvtk.write("# vtk DataFile Version 3.0\n")
    fvtk.write("Vtk output for binned flow configurations (metric in Ångström)\n")
    fvtk.write("ASCII\n")
    fvtk.write("DATASET STRUCTURED_POINTS\n")
    
    fvtk.write("DIMENSIONS "+str(len(x))+" 2 "+str(len(z))+"\n")
    fvtk.write("ORIGIN "+str(10.0*x[0])+" 0.0 "+str(10.0*z[0])+"\n")
    fvtk.write("SPACING "+str(10.0*hx)+" "+str(5.0*Ly)+" "+str(10.0*hz)+"\n")
    
    fvtk.write("CELL_DATA "+str((len(x)-1)*(len(z)-1))+"\n")
    fvtk.write("POINT_DATA "+str(2*len(x)*len(z))+"\n")
    fvtk.write("VECTORS velocity float\n")

    for k in range(len(z)) :
        for j in range(2) :
            for i in range(len(x)) :
                vx_str = str(array_x[i,k])
                vz_str = str(array_z[i,k])
                fvtk.write(vx_str+" 0.00000 "+vz_str+"\n")

    fvtk.close()

"""
    To compare with Phase Field
"""
def export_vectorxy_vtk(x, y, hx, hy, Lz, array_x, array_y, file_name="flow_output.vtk"):
    
    fvtk = open(file_name, 'w')

    fvtk.write("# vtk DataFile Version 3.0\n")
    fvtk.write("Vtk output for binned flow configurations (metric in Ångström)\n")
    fvtk.write("ASCII\n")
    fvtk.write("DATASET STRUCTURED_POINTS\n")
    
    fvtk.write("DIMENSIONS "+str(len(x))+" "+str(len(y))+" 2\n")
    fvtk.write("ORIGIN "+str(x[0])+" "+str(y[0])+" 0.0\n")
    fvtk.write("SPACING "+str(hx)+" "+str(hy)+" "+str(Lz)+"\n")
    
    fvtk.write("CELL_DATA "+str((len(x)-1)*(len(y)-1))+"\n")
    fvtk.write("POINT_DATA "+str(2*len(x)*len(y))+"\n")
    fvtk.write("VECTORS velocity float\n")

    for j in range(2) :
        for k in range(len(y)) :
            for i in range(len(x)) :
                vx_str = str(array_x[i,k])
                vy_str = str(array_y[i,k])
                fvtk.write(vx_str+" "+vy_str+" 0.00000\n")

    fvtk.close()