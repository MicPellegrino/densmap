import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
from matplotlib import cm

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

input_folder = '/home/michele/densmap/ContactLineProfiles/Ca002BL'
gro_folder = '/home/michele/BeskowDiag/Select_Q4_C0020/TrjGroBL'

y = array_from_file(input_folder+'/cly.txt')

Ly = 4.67650
pbcz = 0.0
# pbcz = 30.63400
xmin = 48.618700000000004-5.0*0.4229010640800045-0.4989684091837756
xmax = 48.618700000000004+5.0*0.4229010640800045+0.4989684091837756
left_int = 1

hx = 0.01
hy = 0.01

adsorb_density = np.zeros((int((xmax-xmin)/hx)+1, int(Ly/hy)+1), dtype=float)

# Thresholds
disp_thresh = 0.2
r_thresh = 0.15
z_thresh = 0.15
epsilon_0 = 0.05

frame_init = 1
frame_fin = 100

xold = array_from_file(input_folder+'/cl'+str(frame_init).zfill(4)+'.txt')

# Prepare for movie making
mpl.use("Agg")
print("[ -- Producing movie of the contact line in (x,y) -- ]")
output_file_name = "contact_line_disp.mp4"
FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='Contact Line', artist='Michele Pellegrino',
    comment='Contact line x(y) over time')
writer = FFMpegWriter(fps=12, metadata=metadata)
fig = plt.figure()
movie = True

n_rolls_back = 0
n_jumps_back = 0
n_rolls_forth = 0
n_jumps_forth = 0
arrow_base_x = []
arrow_base_y = []
arrow_dx = []
arrow_dy = []

# Computing the dipole distribution of adsorbed molecules
z_adsorp = 1.0
temp_x = []
temp_z = []
dipole_orient = []
n_h2o_up = 0
n_h2o_down = 0

# Assessing false positives
n_false_p = 0
n_cond1 = 0
n_cond2 = 0 
n_cond3 = 0
n_cond4 = 0

# Time lag (ps)
tlag = 1

with writer.saving(fig, output_file_name, 250) :

    for frame in range(frame_init+tlag, frame_fin+1) :

        print("[ reading frame "+str(frame).zfill(4)+" ]")

        true_pos = False

        xnew = array_from_file(input_folder+'/cl'+str(frame).zfill(4)+'.txt')

        max_disp = infty_norm(xnew, xold)

        if max_disp >= disp_thresh :
            
            displacement = xnew-xold
            _, idx_peaks = find_local_maxima(np.abs(displacement)*(np.abs(displacement)>=disp_thresh))
            y_peaks = np.array(idx_peaks)*(y[1]-y[0])
            x_peaks = np.array(xnew[idx_peaks])

            print("-- Retracing locations --")
            print("x = "+str(x_peaks))
            print("y = "+str(y_peaks)) 

            file_new = gro_folder+'/cv'+str(frame).zfill(4)+'.gro'
            file_old = gro_folder+'/cv'+str(frame-tlag).zfill(4)+'.gro'
            x_atom_new = []
            y_atom_new = []
            z_atom_new = []
            resid_new = []
            x_atom_old = []
            y_atom_old = []
            z_atom_old = []
            resid_old = []
            n_lines_new = count_line(file_new)
            n_line_old = count_line(file_old)
            n = 0
            n_temp = 0
            in_file = open(file_new, 'r')
            for line in in_file :
                n += 1
                if n > 2 and n < n_lines_new :
                    line_data = read_gro_line(line)
                    temp_x.append(line_data[4])
                    temp_z.append(line_data[6])
                    n_temp += 1
                    if n_temp == 3 :
                        n_temp = 0
                        if temp_z[0] < z_adsorp or temp_z[1] < z_adsorp or temp_z[2] < z_adsorp :
                            xd = 0.5*(temp_x[2]+temp_x[1])-temp_x[0]
                            zd = 0.5*(temp_z[2]+temp_z[1])-temp_z[0]
                            # mu1 = xd/np.sqrt(xd**2+zd**2)
                            # mu2 = zd/np.sqrt(xd**2+zd**2)
                            mu2 = zd
                            # dipole_orient.append(np.sign(mu1)*np.arctan(mu2/mu1))
                            if not(np.isnan(mu2)) :
                                dipole_orient.append(mu2)
                                n_h2o_up += (mu2>0)
                                n_h2o_down += (mu2<=0)
                        temp_x = []
                        temp_z = []
                    if line_data[2] == "OW" :
                        x_atom_new.append(line_data[4])
                        y_atom_new.append(line_data[5])
                        z_atom_new.append(line_data[6])
                        resid_new.append(line_data[0])
                        if line_data[4]>0 and line_data[4]<xmax and line_data[5]>0 and line_data[5]<Ly :
                            adsorb_density[ int((line_data[4]-xmin)/hx), int(line_data[5]/hy) ] += 1
            in_file.close()
            x_atom_new = np.array(x_atom_new)
            y_atom_new = np.array(y_atom_new)
            z_atom_new = np.array(z_atom_new)
            resid_new = np.array(resid_new)
            n = 0   
            in_file = open(file_old, 'r')
            for line in in_file :
                n += 1
                if n > 2 and n < n_line_old :
                    line_data = read_gro_line(line)
                    if line_data[2] == "OW" :
                        x_atom_old.append(line_data[4])
                        y_atom_old.append(line_data[5])
                        z_atom_old.append(line_data[6])
                        resid_old.append(line_data[0])
            in_file.close()
            x_atom_old = np.array(x_atom_old)
            y_atom_old = np.array(y_atom_old)
            z_atom_old = np.array(z_atom_old)
            resid_old = np.array(resid_old)

            # Making a vector of displacement matching residue id
            resid_dx = []
            dx_atom = []
            tabu_id = set()
            for i_old in range(len(resid_old)) :
                for i_new in range(len(resid_new)) :
                    if resid_old[i_old] == resid_new[i_new] and not(resid_new[i_new] in tabu_id):
                        resid_dx.append(resid_new[i_new])
                        dx_atom.append(x_atom_new[i_new]-x_atom_old[i_old])
                        tabu_id.add(resid_new[i_new])    

            # Find the closest atoms
            res_peaks = []
            for k in range(len(x_peaks)) :
                idx_atom = np.argmin( (x_peaks[k]-x_atom_new)**2 + (y_peaks[k]-y_atom_new)**2 + (pbcz-z_atom_new)**2 )
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
                    cond4 = r < 0.5*Ly
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

        if movie :
            plt.axis('scaled')
            plt.ylim([xmin,xmax])
            plt.xlim([0.0, Ly])
            plt.title('CL @t='+str(frame).zfill(4)+'ps', fontsize=17.5)
            plt.plot(y, xold, 'k--', linewidth=1.5)
            plt.plot(y, xnew, 'k-', linewidth=2.0)
            plt.fill_between(y, xnew,  y2=xmax*left_int+xmin*(1-left_int), color='tab:cyan')
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


print("# TRUE POSITIVES  = "+str(n_jumps_back+n_rolls_back+n_jumps_forth+n_rolls_forth))
print("-- ------------------- --")
print("# FALSE POSITIVES   = "+str(n_false_p))
print("failed rolls        = "+str(n_cond1))
print("thresh dispacement  = "+str(n_cond2))
print("jump/roll direction = "+str(n_cond3))
print("periodic b.c.       = "+str(n_cond4))
print("-- ------------------- --")
print("     JUMPS ROLLS")
print("BACK "+str(n_jumps_back).ljust(5)+" "+str(n_rolls_back).ljust(5))
print("FORT "+str(n_jumps_forth).ljust(5)+" "+str(n_rolls_forth).ljust(5))
print("-- ------------------- --")

mpl.use("TkAgg")
print("# adsorbed molecules = "+str(len(dipole_orient)))
dipole_orient = 2*(np.array(dipole_orient)-0.5*(max(dipole_orient)+min(dipole_orient))) \
        /(max(dipole_orient)-min(dipole_orient))
n_bin=40
print("f_up   = "+str(n_h2o_up/len(dipole_orient)) )
print("f_down = "+str(n_h2o_down/len(dipole_orient)) )
print("-- ------------------- --")
# print(np.sum(np.isnan(dipole_orient)))
p_dipole, bin_edges = np.histogram(dipole_orient, bins=n_bin, range=(-1.0, 1.0))
p_dipole = p_dipole/np.sum(p_dipole)
u_dipole = -np.log(p_dipole)
# plt.plot(np.linspace(-1.0, 1.0, len(u_dipole)), u_dipole)
# plt.show()

# np.savetxt('p_dipole.txt', p_dipole)
# np.savetxt('bin_edges.txt', bin_edges)

"""
fig2, (ax1, ax2) = plt.subplots(1,2)

p_replica = array_from_file('p_dipole.txt')
u_replica = -np.log(0.5*(p_dipole+p_replica))

vec = np.linspace(-1.0, 1.0, len(u_replica))
pord = np.polyfit(vec, u_replica, deg=6)

ax1.plot(vec, u_replica, 'k.', markersize=7.5)
ax1.plot(vec, np.polyval(pord, vec), 'k-', linewidth=2.75)
# plt.title("Free energy landascape of absorbed water molecules", fontsize=25)
ax1.set_xlabel("dipole orientation", fontsize=20)
ax1.set_ylabel("$U$ [$k_BT$]", fontsize=20)
ax1.tick_params(axis='x', labelsize=15.0)
ax1.tick_params(axis='y', labelsize=15.0)

x = np.linspace(xmin, xmax, int((xmax-xmin)/hx)+1)
y = np.linspace(0.0, Ly, int(Ly/hy)+1)
X, Y = np.meshgrid(x,y, indexing='ij')
adsorb_density /= np.sum(np.sum(adsorb_density))
plt_colormap = ax2.pcolormesh(X, Y, adsorb_density, cmap=cm.gist_yarg)
fig.colorbar(plt_colormap, ax=ax2)
# plt.title("Overview of adsorption sites", fontsize=25)
ax2.set_xlabel("x [nm]", fontsize=20)
ax2.set_ylabel("y [nm]", fontsize=20)
ax2.tick_params(axis='x', labelsize=15.0)
ax2.tick_params(axis='y', labelsize=15.0)
ax2.axis("scaled")
plt.show()
"""
