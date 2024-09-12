import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
from sklearn.cluster import DBSCAN
from matplotlib import cm
import alphashape
import time as cpu_time
from collections import Counter
import shapely

# Global variables
N_SAMPLES_Y = 100
N_WRITER = 250

# Utilities to deal with .gro files

def count_line( file_name ) :
    n_lines = 0
    f_in = open( file_name, 'r')
    for line in f_in :
        n_lines += 1
    f_in.close()
    return n_lines

def read_gro_line( line ) :
    line_data = [None]*10
    line_data[0] = int(line[0:5].strip())                   # Residue serial
    line_data[1] = str(line[5:10].strip())                  # Residue name
    line_data[2] = str(line[10:15].strip())                 # Atom name
    line_data[3] = int(line[15:20].strip())                 # Atom serial
    line_data[4] = float(line[20:28].strip())               # Pos X
    line_data[5] = float(line[28:36].strip())               # Pos Y
    line_data[6] = float(line[36:44].strip())               # Pos Z
    return line_data

def most_frequent(List):
    occurence_count = Counter(List)
    return occurence_count.most_common(1)[0][0]


class ClusterHullProfile :

    def __init__(self, Ly, xmax, xmin, frame_init, frame_fin, read_folder, save_folder,
                left_int=1, root='cv', movie=False, movie_name="contact_line_alpha.mp4", frame_plot=None) :
        
        # Geometric parameters
        self.Ly = Ly
        self.xmax = xmax
        self.xmin = xmin
        self.left_int = left_int

        # Frames and folders
        self.frm_ini = frame_init
        self.frm_fin = frame_fin
        self.read_fld = read_folder
        self.save_fld = save_folder
        self.fr = root

        # Movie
        self.make_movie = movie
        self.movie_name = movie_name

        # Plot
        if frame_plot==None :
            self.frame_plot = self.frm_ini
        else :
            self.frame_plot = frame_plot

    def cluster_hull(self, alpha0=2.25, minimum_lattice_distance=0.33) :

        timeout_error = False
        concave_error = False

        y_fun = np.linspace(0.0, self.Ly, N_SAMPLES_Y)
        np.savetxt(self.save_fld+'/'+self.fr+'y.txt', y_fun)

        if self.make_movie :
            mpl.use("Agg")
        print("[ -- Producing movie of the contact line in (x,y) -- ]")
        output_file_name = self.movie_name
        FFMpegWriter = manimation.writers['ffmpeg']
        metadata = dict(title='Contact Line', artist='Michele Pellegrino',
                        comment='Contact line x(y) over time')
        writer = FFMpegWriter(fps=12, metadata=metadata)
        fig = plt.figure()

        with writer.saving(fig, output_file_name, N_WRITER) :

            for frame in range(self.frm_ini, self.frm_fin+1) :

                print("[ ClusterHull: reading frame "+str(frame).zfill(4)+" ]")

                # Read from .gro and add periodic images
                file_name = self.read_fld+'/'+self.fr+str(frame).zfill(4)+'.gro'
                x = []
                y = []
                n_lines = count_line(file_name)
                n = 0   
                in_file = open(file_name, 'r')
                for line in in_file :
                    n += 1
                    if n > 2 and n < n_lines :
                        line_data = read_gro_line(line)
                        if line_data[2] == "OW" :
                            x.append(line_data[4])
                            y.append(line_data[5])
                in_file.close()
                for k in range(len(y)) :
                    if y[k] > 0.75*self.Ly :
                        y.append(y[k]-self.Ly)
                        x.append(x[k])
                    elif y[k] < 0.25*self.Ly :
                        y.append(y[k]+self.Ly)
                        x.append(x[k])

                # Perform 2D clustering and find the label with the most points
                x = np.array(x)
                y = np.array(y)
                xcom = np.mean(x)
                points = np.column_stack((x, y))
                clustering = DBSCAN(eps=minimum_lattice_distance, min_samples=1).fit(points)
                cl_label = most_frequent(clustering.labels_)

                # Obtain a concave hull
                x_cl = np.take(x, np.where((clustering.labels_==cl_label) * \
                     ((x<xcom)*self.left_int+(x>xcom)*(1-self.left_int)) )[0])
                y_cl = np.take(y, np.where((clustering.labels_==cl_label) * \
                     ((x<xcom)*self.left_int+(x>xcom)*(1-self.left_int)) )[0])
                points_cl = np.column_stack((np.array(x_cl), np.array(y_cl)))
                if frame == self.frm_ini and alpha0 == 0:
                    print("[ Optimizing alpha, please wait... ]")
                    alpha = alphashape.optimizealpha(points_cl)
                    print("[ alpha = "+'{:05.3f}'.format(alpha)+" ]")
                else :
                    alpha = alpha0
                hull = alphashape.alphashape(points_cl, alpha)
                
                # Checking hull object instance
                if isinstance(hull, shapely.geometry.multipolygon.MultiPolygon) :
                    print("[ Warning: alphashape is returning more than one polygon as concave hull ]")
                    print("[    -> re-optimizing alpha...                                           ]")
                    alpha = alphashape.optimizealpha(points_cl)
                    hull = alphashape.alphashape(points_cl, alpha)
                    print("[ alpha = "+'{:05.3f}'.format(alpha)+" ]")
                    if isinstance(hull, shapely.geometry.multipolygon.MultiPolygon) :
                        print("[ Error: cannot obtain an optimal value of alpha for a single polygon ]")
                        print("[    -> try changing clustering epsilon instead                       ]")
                        concave_error = True
                        break
                    print("[ Re-setting alpha to its original value... ]")
                    alpha = alpha0
                    print("[ alpha = "+'{:05.3f}'.format(alpha)+" ]")
                hull_pts = hull.exterior.coords.xy

                # Obtain the cl function x(y) at this frame
                hull_cl_x = []
                hull_cl_y = []
                s1 = False
                s2 = True
                l = len(hull_pts[0])
                hull_pts_x = hull_pts[0][0:l-1]
                hull_pts_y = hull_pts[1][0:l-1]
                if not(self.left_int) :
                    hull_pts_x = np.flip(hull_pts_x)
                    hull_pts_y = np.flip(hull_pts_y)
                l = len(hull_pts_x)
                n = 0
                # Checking timeout
                timeout = cpu_time.time() + 5.0         # Five seconds from now
                while(s2) :
                    n_new = n
                    n_old = (n-1)%l
                    y_old = hull_pts_y[n_old]
                    y_new = hull_pts_y[n_new]
                    if y_old <= self.Ly and y_new >= self.Ly and s1 :
                        s2 = False
                    if y_old <= 0 and y_new >= 0 :
                        s1 = True
                        hull_cl_x.append(hull_pts_x[n_old])
                        hull_cl_y.append(hull_pts_y[n_old])
                    if s1 :
                        hull_cl_x.append(hull_pts_x[n_new])
                        hull_cl_y.append(hull_pts_y[n_new])
                    n = (n+1)%l
                    if cpu_time.time() > timeout :
                        print("[ Error: selection of contact line countour points is taking too long ]")
                        print("[    -> plotting concave hull, you may want to change alpha or eps    ]")
                        s2 = False
                        timeout_error = True
                if timeout_error :
                    break
                x_fun = np.interp(y_fun, hull_cl_y, hull_cl_x)

                # Saving cl x(y) on a .txt file
                np.savetxt(self.save_fld+'/'+self.fr+str(frame).zfill(4)+'.txt', x_fun)

                if frame==self.frame_plot :
                    plt.axis('scaled')
                    plt.ylim([self.xmin, self.xmax])
                    plt.xlim([0.0, self.Ly])
                    plt.title('CL @t='+str(frame).zfill(4)+'ps', fontsize=17.5)
                    plt.fill_between(y_fun, x_fun, y2=self.xmax*self.left_int+self.xmin*(1-self.left_int), \
                        color='tab:cyan')
                    plt.plot(y, x, 'k.')
                    plt.xlabel("y [nm]", fontsize=15.0)
                    plt.ylabel("x [nm]", fontsize=15.0)
                    plt.xticks(fontsize=12.5)
                    plt.yticks(fontsize=12.5)
                    plt.show()

                if self.make_movie :
                    plt.axis('scaled')
                    plt.ylim([self.xmin, self.xmax])
                    plt.xlim([0.0, self.Ly])
                    plt.title('CL @t='+str(frame).zfill(4)+'ps', fontsize=17.5)
                    plt.fill_between(y_fun, x_fun, y2=self.xmax*self.left_int+self.xmin*(1-self.left_int), \
                        color='tab:cyan')
                    plt.plot(y, x, 'k.')
                    plt.xlabel("y [nm]", fontsize=15.0)
                    plt.ylabel("x [nm]", fontsize=15.0)
                    plt.xticks(fontsize=12.5)
                    plt.yticks(fontsize=12.5)
                    writer.grab_frame()
                    plt.cla()
                    plt.clf()
                    
        if self.make_movie :
            mpl.use("TkAgg")

        if timeout_error:
            fig, ax = plt.subplots()
            ax.scatter(x, y, c=1+clustering.labels_, s=150, cmap=cm.Paired)
            ax.plot(hull_pts[0], hull_pts[1], 'b--')
            ax.plot(hull_pts[0][0], hull_pts[1][0], 'bo')
            ax.plot(hull_pts[0][1], hull_pts[1][1], 'bx')
            ax.plot([self.xmin, self.xmax], [0.0, 0.0], 'r-')
            ax.plot([self.xmin, self.xmax], [self.Ly, self.Ly], 'r-')
            plt.xlabel("x [nm]", fontsize=25)
            plt.ylabel("y [nm]", fontsize=25)
            plt.xticks(fontsize=20.0)
            plt.yticks(fontsize=20.0)
            plt.show()

        if concave_error:
            fig, ax = plt.subplots()
            ax.scatter(x, y, c=1+clustering.labels_, s=150, cmap=cm.Paired)
            ax.plot(xcom, 0.5*self.Ly, 'rx', markersize=25)
            ax.plot([self.xmin, self.xmax], [0.0, 0.0], 'r-')
            ax.plot([self.xmin, self.xmax], [self.Ly, self.Ly], 'r-')
            plt.xlabel("x [nm]", fontsize=25)
            plt.ylabel("y [nm]", fontsize=25)
            plt.xticks(fontsize=20.0)
            plt.yticks(fontsize=20.0)
            plt.show()


def main() :

    Ly = 4.67650
    xmin = 87.61667500000001, - 6.0*0.2956466545980184
    xmax = 87.61667500000001, + 6.0*0.2956466545980184
    frame_init = 1
    frame_fin = 3999
    read_folder = '/home/michele/BeskowDiag/Select_Q3_C0100/TrjGroBR'
    save_folder = '/home/michele/densmap/ContactLineProfiles/Ca0100BR'

    test_cl = ClusterHullProfile(Ly, xmax, xmin, frame_init, frame_fin, read_folder, save_folder, movie=False, movie_name="q3_c0100_alpha_shape.mp4")
    test_cl.cluster_hull(alpha0=3.0, minimum_lattice_distance=0.375)

if __name__ == "__main__":

    main()
