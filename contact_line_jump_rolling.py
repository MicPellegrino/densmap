import numpy as np
import scipy.interpolate as sci_int
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
from matplotlib import cm

# Global variables
N_WRITER = 250

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

def array_from_file( filename ):
    my_list = []
    with open(filename, 'r') as f:
        for line in f:
            my_list.append(float(line.split()[0]))
    return np.array(my_list)

def find_local_maxima( x ) :
    n = len(x)
    xmax = []
    imax = []
    for i in range(n) :
        if x[i]>x[(i+1)%n] and x[i]>x[(i-1)%n] :
            xmax.append(x[i])
            imax.append(i)
    return xmax, imax

infty_norm = lambda x1, x2 : np.max(np.abs(x1-x2))


class ContactLineClassifier :

    def __init__(self, Ly, xmax, xmin, frame_init, frame_fin, read_folder, gro_folder,
                left_int=1, pbcz=0, root='cv', movie=False, movie_name="contact_line_disp.mp4") :

        # Geometric parameters
        self.Ly = Ly
        self.xmax = xmax
        self.xmin = xmin
        self.left_int = left_int
        self.pbcz = pbcz

        # Frames and folders
        self.frm_ini = frame_init
        self.frm_fin = frame_fin
        self.read_fld = read_folder
        self.grom_fld = gro_folder
        self.fr = root

        # Movie
        self.make_movie = movie
        self.movie_name = movie_name

    def positions_from_file(self, file_name) :
        x_atom = []
        y_atom = []
        z_atom = []
        resid = []
        n_lines = count_line(file_name)
        n = 0
        n_temp = 0
        in_file = open(file_name, 'r')
        for line in in_file :
            n += 1
            if n > 2 and n < n_lines :
                line_data = read_gro_line(line)
                n_temp += 1
                if n_temp == 3 :
                    n_temp = 0
                if line_data[2] == "OW" :
                    x_atom.append(line_data[4])
                    y_atom.append(line_data[5])
                    z_atom.append(line_data[6])
                    resid.append(line_data[0])
        in_file.close()
        return np.array(x_atom), np.array(y_atom), np.array(z_atom), np.array(resid) 

    def classify_jump_roll(self, 
                disp_thresh=0.2, r_thresh=0.15, z_thresh=0.15, tlag=1):

        y = array_from_file(self.read_fld+'/'+self.fr+'y.txt')
        xold = array_from_file(self.read_fld+'/'+self.fr+str(self.frm_ini).zfill(4)+'.txt')

        mpl.use("Agg")
        print("[ -- Producing movie of the contact line in (x,y) -- ]")
        output_file_name = self.movie_name
        FFMpegWriter = manimation.writers['ffmpeg']
        metadata = dict(title='Contact Line', artist='Michele Pellegrino',
                        comment='Contact line x(y) over time')
        writer = FFMpegWriter(fps=12, metadata=metadata)
        fig = plt.figure()

        # Statistics
        n_rolls_back = 0
        n_jumps_back = 0
        n_rolls_forth = 0
        n_jumps_forth = 0
        arrow_base_x = []
        arrow_base_y = []
        arrow_dx = []
        arrow_dy = []

        # Assessing false positives
        n_false_p = 0
        n_cond1 = 0
        n_cond2 = 0 
        n_cond3 = 0
        n_cond4 = 0

        with writer.saving(fig, output_file_name, N_WRITER) :

            for frame in range(self.frm_ini+tlag, self.frm_fin+1) :

                print("[ JumpRolling: reading frame "+str(frame).zfill(4)+" ]")
                true_pos = False
                xnew = array_from_file(self.read_fld+'/'+self.fr+str(frame).zfill(4)+'.txt')
                max_disp = infty_norm(xnew, xold)

                if max_disp >= disp_thresh :

                    displacement = xnew-xold

                    # THIS MAY BE NOT NEEDED #
                    """
                    _, idx_peaks = find_local_maxima(np.abs(displacement)*(np.abs(displacement)>=disp_thresh))
                    y_peaks = np.array(idx_peaks)*(y[1]-y[0])
                    x_peaks = np.array(xnew[idx_peaks])
                    print("-- Retracing locations --")
                    print("x = "+str(x_peaks))
                    print("y = "+str(y_peaks))
                    """
                    ##########################

                    file_new = self.grom_fld+'/cv'+str(frame).zfill(4)+'.gro'
                    file_old = self.grom_fld+'/cv'+str(frame-tlag).zfill(4)+'.gro'
                    x_atom_new, y_atom_new, z_atom_new, resid_new = self.positions_from_file(file_new)
                    x_atom_old, y_atom_old, z_atom_old, resid_old = self.positions_from_file(file_old)

                    # Making a vector of displacement matching residue id
                    resid_dx = []
                    dx_atom = []
                    dy_atom = []
                    dz_atom = []
                    x_atom = []
                    y_atom = []
                    tabu_id = set()
                    for i_old in range(len(resid_old)) :
                        for i_new in range(len(resid_new)) :
                            if resid_old[i_old] == resid_new[i_new] and not(resid_new[i_new] in tabu_id):
                                resid_dx.append(resid_new[i_new])
                                dx_atom.append(x_atom_new[i_new]-x_atom_old[i_old])
                                dy_atom.append(y_atom_new[i_new]-y_atom_old[i_old])
                                dz_atom.append(z_atom_new[i_new]-z_atom_old[i_old])
                                # At the new position (arbitrary)
                                x_atom.append(x_atom_new[i_new])
                                # Define the displacement location along y (arbitrary)
                                # y_atom.append(0.5*(y_atom_new[i_new]+y_atom_old[i_old]))
                                y_atom.append(y_atom_new[i_new])
                                tabu_id.add(resid_new[i_new])
                    resid_dx = np.array(resid_dx)
                    dx_atom = np.array(dx_atom)
                    dy_atom = np.array(dy_atom)
                    dz_atom = np.array(dz_atom)
                    x_atom = np.array(x_atom)
                    y_atom = np.array(y_atom)

                    # Interpolate the dispalcement dx to find its values at positions y
                    disp_interp = sci_int.interp1d(y, displacement, kind='nearest', fill_value='extrapolate')
                    posx_interp = sci_int.interp1d(y, xnew, kind='nearest', fill_value='extrapolate')
                    dx_eval = np.array(disp_interp(y_atom))
                    x_eval = np.array(posx_interp(y_atom))

                    # Find the index that maximize agreement
                    _, idx_atom_list = find_local_maxima( (dx_eval**2>=disp_thresh**2)/((dx_eval-dx_atom)**2) )

                    n_true_p_loc = 0
                    for idx_atom in idx_atom_list :
                        
                        idx_atom = np.argmin( ((dx_eval-dx_atom)**2)/(dx_eval**2) )
                        r = np.sqrt( dx_atom[idx_atom]**2 + dy_atom[idx_atom]**2 + dz_atom[idx_atom]**2 )

                        # Checking conditions
                        # Molecule is not too far from the contact line
                        cond1 = np.abs(x_eval[idx_atom]-x_atom[idx_atom])<0.5*r_thresh
                        if cond1 :
                            # Molecule has moved enough
                            cond2 = r > r_thresh
                            if cond2 :
                                # Molecule hasn't moved across b.c.
                                cond3 = r < 0.5*self.Ly
                                if cond3 :
                                    # Molecule has same displacement as c.l.
                                    cond4 = ( dx_atom[idx_atom]*dx_eval[idx_atom]>0 )
                                    if cond4 :
                                        # We have a true positive!
                                        true_pos = True
                                        n_true_p_loc += 1
                                        # Jump
                                        if ( np.abs(dz_atom[idx_atom]) > z_thresh ) :
                                            if dx_eval[idx_atom]<0 :
                                                n_rolls_back += 1
                                            else :
                                                n_rolls_forth += 1
                                        # Roll
                                        else :
                                            if dx_eval[idx_atom]<0 :
                                                n_jumps_back += 1
                                            else :
                                                n_jumps_forth += 1
                                    else :
                                        n_false_p += 1
                                        n_cond4 += 1
                                else :
                                    n_false_p += 1
                                    n_cond3 += 1
                            else :
                                n_false_p += 1
                                n_cond2 += 1
                        else :
                            n_false_p += 1
                            n_cond1 += 1
                    if n_true_p_loc > 0 :
                        print("-- true positives found: "+str(n_true_p_loc))

                    # Find the closest atoms
                    """
                    res_peaks = []
                    for k in range(len(x_peaks)) :
                        idx_atom = np.argmin( (x_peaks[k]-x_atom_new)**2 + (y_peaks[k]-y_atom_new)**2 + \
                                                (self.pbcz-z_atom_new)**2 )
                        res_peaks.append( resid_new[idx_atom] )
                        idx_old = -1
                        idx_dx = -1
                        for i in range(len(resid_old)) :
                            if resid_new[idx_atom] == resid_old[i] :
                                idx_old = i
                                break
                        for i in range(len(resid_dx)) :
                            if resid_new[idx_atom] == resid_dx[i] :
                                idx_dx = i
                                break
                        if idx_old >= 0 and idx_dx >= 0 :
                            cond1 = ( np.abs(z_atom_old[idx_old]-z_atom_new[idx_atom]) > z_thresh )
                            r = np.sqrt( (x_atom_new[idx_atom]-x_atom_old[idx_old])**2 + 
                                (y_atom_new[idx_atom]-y_atom_old[idx_old])**2 +
                                (z_atom_new[idx_atom]-z_atom_old[idx_old])**2 )
                            epsilon = np.abs(dx_atom[idx_dx]-displacement[idx_peaks[k]])
                            # Change condition check
                            cond2 = r > r_thresh
                            # cond2 = epsilon < epsilon_0
                            cond3 = ( np.sign(x_atom_new[idx_atom]-x_atom_old[idx_old]) == np.sign(displacement[idx_peaks[k]]) )
                            cond4 = r < 0.5*self.Ly
                            arrow_base_x.append(x_atom_old[idx_old])
                            arrow_base_y.append(y_atom_old[idx_old])
                            arrow_dx.append(x_atom_new[idx_atom]-x_atom_old[idx_old])
                            arrow_dy.append(y_atom_new[idx_atom]-y_atom_old[idx_old])
                            if ( ( cond1 and cond3 ) or ( cond2 and cond3 ) ) and cond4 :
                                print("-- A TRUE POSITIVE!")
                                true_pos = True
                                if cond1 :
                                    if displacement[idx_peaks[k]]<0 :
                                        n_rolls_back += 1
                                    else :
                                        n_rolls_forth += 1
                                else :
                                    if displacement[idx_peaks[k]]<0 :
                                        n_jumps_back += 1
                                    else :
                                        n_jumps_forth += 1   
                            else :
                                n_false_p += 1
                                if not(cond1) :
                                    n_cond1 += 1
                                if not(cond2) :
                                    n_cond2 += 1
                                if not(cond3) :
                                    n_cond3 += 1
                                if not(cond4) :
                                    n_cond4 += 1

                    print("res = "+str(res_peaks)) 
                    print("-- ------------------- --")

                    """

                if self.make_movie :
                    plt.axis('scaled')
                    plt.ylim([self.xmin,self.xmax])
                    plt.xlim([0.0, self.Ly])
                    plt.title('CL @t='+str(frame).zfill(4)+'ps', fontsize=17.5)
                    plt.plot(y, xold, 'k--', linewidth=1.5)
                    plt.plot(y, xnew, 'k-', linewidth=2.0)
                    plt.fill_between(y, xnew,  y2=self.xmax*self.left_int+self.xmin*(1-self.left_int), color='tab:cyan')
                    if max_disp < disp_thresh :
                        plt.fill_between(y, xold, xnew, color='tab:green')
                    else :
                        if true_pos :
                            plt.fill_between(y, xold, xnew, color='tab:red')
                        else :
                            plt.fill_between(y, xold, xnew, color='tab:orange')
                    if len(arrow_base_x) > 0 :
                        for j in range(len(arrow_base_x)) :
                            plt.arrow(arrow_base_y[j], arrow_base_x[j], arrow_dy[j], arrow_dx[j], 
                                head_width=0.05, head_length=0.075, fc='k', ec='k')
                    plt.xlabel("y [nm]", fontsize=15.0)
                    plt.ylabel("x [nm]", fontsize=15.0)
                    plt.xticks(fontsize=12.5)
                    plt.yticks(fontsize=12.5)
                    writer.grab_frame()
                    plt.cla()
                    plt.clf()

                xold = xnew
                arrow_base_x = []
                arrow_base_y = []
                arrow_dx = []
                arrow_dy = []

        n_true_p = n_jumps_back+n_rolls_back+n_jumps_forth+n_rolls_forth

        print("# TRUE POSITIVES  = "+str(n_true_p))
        print("-- ------------------- --")
        print("# FALSE POSITIVES   = "+str(n_false_p))
        print("too far from c.l.   = "+str(n_cond1))
        print("thresh dispacement  = "+str(n_cond2))
        print("periodic b.c.       = "+str(n_cond3))
        print("jump/roll direction = "+str(n_cond4))
        print("-- ------------------- --")
        print("     JUMPS ROLLS")
        print("BACK "+str(n_jumps_back).ljust(5)+" "+str(n_rolls_back).ljust(5))
        print("FORT "+str(n_jumps_forth).ljust(5)+" "+str(n_rolls_forth).ljust(5))
        print("-- ------------------- --")

        return n_true_p, n_false_p

def main() :

    Ly = 4.67650
    xmin = 57.884825 - 6.0*0.2352970874766621
    xmax = 57.884825 + 6.0*0.2352970874766621
    frame_init = 1
    frame_fin = 3500
    gro_folder = '/home/michele/BeskowDiag/Select_Q3_Equil/TrjGroBL'
    input_folder = '/home/michele/densmap/ContactLineProfiles/EquilQ3BL'


    clc = ContactLineClassifier(Ly, xmax, xmin, frame_init, frame_fin, input_folder, gro_folder, 
                                pbcz=0, left_int=1, movie=False, movie_name="q3_eq_displacement.mp4")
    ntp, nfp = clc.classify_jump_roll(tlag=1)

    print("False positive fraction = "+str(nfp/(nfp+ntp)))
    print("Total true positives = "+str(ntp))

if __name__ == "__main__":
    main()
