import numpy as np
import scipy.interpolate as sci_int
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
from matplotlib import cm
import matplotlib.tri as tri

# Global variables
N_WRITER = 250
SURFTENS = 5.78e-2      # [Pa*m]
VISCOSITY = 8.77e-4     # [Pa*s]
TIME_STEP = 1e-3        # [ns] 

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
                capillary_number=0.0, left_int=1, pbcz=0, dsio2 = 0.45,
                root='cv', movie=False, movie_name="contact_line_disp.mp4") :

        # The capillary number is signed
        # TR: Ca>0
        # BR: Ca<0
        # TL: Ca>0
        # BL: Ca<0
        self.wall_velocity = 0.5*SURFTENS*capillary_number/VISCOSITY    # [nm/ns]

        # Geometric parameters
        self.Ly = Ly
        self.xmax = xmax
        self.xmin = xmin
        self.left_int = left_int
        self.pbcz = pbcz
        self.dsio2 = dsio2

        # Frames and folders
        self.frm_ini = frame_init
        self.frm_fin = frame_fin
        self.read_fld = read_folder
        self.grom_fld = gro_folder
        self.fr = root

        # Movie
        self.make_movie = movie
        self.movie_name = movie_name

    def initialize_vh(self, hx_vh = 0.02, hz_vh = 0.02, xlow=None, xupp=None, zlow=0.6, zupp=1.6) :
        if xlow == None :
            xl = self.xmin
        else :
            xl = xlow
        if xupp == None :
            xu = self.xmax
        else :
            xu = xupp
        xlim = xu-xl
        zlim = zupp-zlow
        """
        self.vh_nx = int(2*xlim/hx_vh)
        self.vh_nz = int(2*zlim/hz_vh)
        self.vh_lowx = -xlim
        self.vh_lowz = -zlim
        self.hx_vh = 2*xlim/self.vh_nx
        self.hz_vh = 2*zlim/self.vh_nz
        """
        self.vh_nx = int(xlim/hx_vh)
        self.vh_nz = int(zlim/hz_vh)
        self.vh_lowx = 0
        self.vh_lowz = 0
        self.hx_vh = xlim/self.vh_nx
        self.hz_vh = zlim/self.vh_nz
        self.vanhove_map = np.zeros((self.vh_nx, self.vh_nz), dtype=float)
        self.vanhove_x = np.zeros(self.vh_nx, dtype=float)
        self.vanhove_z = np.zeros(self.vh_nz, dtype=float)

    def bin_vh(self, disp_x, disp_z) :
        i = int((disp_x-self.vh_lowx)/self.hx_vh)
        j = int((disp_z-self.vh_lowz)/self.hz_vh)
        if i>=0 and i<self.vh_nx and j>=0 and j<self.vh_nz :
            self.vanhove_map[i,j] += 1
            self.vanhove_x[i] += 1
            self.vanhove_z[j] += 1

    def normalize_vh(self) :
        self.vanhove_map /= sum(sum(self.vanhove_map))*self.hx_vh*self.hz_vh
        self.vanhove_x /= sum(self.vanhove_x)*self.hx_vh
        self.vanhove_z /= sum(self.vanhove_z)*self.hz_vh

    def save_vh(self, folder_name, tag) :
        x = np.linspace( self.vh_lowx, self.vh_lowx+self.vh_nx*self.hx_vh, self.vh_nx ) + 0.5*self.hx_vh
        z = np.linspace( self.vh_lowz, self.vh_lowz+self.vh_nz*self.hz_vh, self.vh_nz ) + 0.5*self.hz_vh
        X, Z = np.meshgrid(x, z, indexing='ij')
        np.savetxt(folder_name+'/vanhove_xz_'+tag+'.txt', np.c_[X.flatten(),Z.flatten(),self.vanhove_map.flatten()])
        np.savetxt(folder_name+'/vanhove_x_'+tag+'.txt', np.c_[x,self.vanhove_x])
        np.savetxt(folder_name+'/vanhove_z_'+tag+'.txt', np.c_[z,self.vanhove_z])

    def plot_vh(self, drefx, drefz, tlag) :

        x = np.linspace( self.vh_lowx, self.vh_lowx+self.vh_nx*self.hx_vh, self.vh_nx ) # + 0.5*self.hx_vh
        z = np.linspace( self.vh_lowz, self.vh_lowz+self.vh_nz*self.hz_vh, self.vh_nz ) # + 0.5*self.hz_vh
        X, Z = np.meshgrid(x, z, indexing='ij')
        # np.savetxt('van-hove.txt', np.c_[X.flatten(),Z.flatten(),self.vanhove_map.flatten()])

        # plt.pcolormesh(X, Z, -np.log(self.vanhove_map), cmap=cm.hot)
        plt.pcolormesh(X, Z, self.vanhove_map, cmap=cm.hot)
        plt.grid()
        # plt.contourf(X, Z, -np.log(self.vanhove_map), cmap=cm.hot)
        plt.title(r"$t_{lag}=$"+str(tlag)+"ps", fontsize=80)
        plt.ylabel(r'$\Delta z$ [nm]', fontsize=70)
        plt.xlabel(r'$\Delta x$ [nm]', fontsize=70)
        plt.xticks(fontsize=50)
        plt.yticks(fontsize=50)
        plt.show()

        plt.title(r"$t_{lag}=$"+str(tlag)+"ps", fontsize=80)
        plt.step(x/drefx, drefx*self.vanhove_x, 'r-', linewidth=8.5, label=r'$G_x$')
        plt.step(z/drefz, drefz*self.vanhove_z, 'b-', linewidth=8.5, label=r'$G_z$')
        plt.legend(fontsize=75)
        plt.xlabel(r'$r/d_{ref}$ [1]', fontsize=70)
        plt.xticks(fontsize=70)
        plt.yticks(fontsize=70)
        plt.xlim([0,5])
        plt.show()

    def positions_from_file(self, file_name, atom_name="OW") :
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
                if line_data[2] == atom_name :
                    x_atom.append(line_data[4])
                    y_atom.append(line_data[5])
                    z_atom.append(line_data[6])
                    resid.append(line_data[0])
        in_file.close()
        return np.array(x_atom), np.array(y_atom), np.array(z_atom), np.array(resid) 

    def order_parameter(self, nbinphi=60 , nbinz=100, zlow=0.6, zupp=1.6, levels=None, vmax=None):

        """
            Do the analysis of the order parameter for the adsorption into the substrate
        """

        z_phi_map = np.zeros( (nbinz, nbinphi), dtype=float )
        hphi = 2/nbinphi
        dz = zupp-zlow
        hz = dz/nbinz

        x = np.linspace(zlow,zupp,nbinz)+0.5*hz
        y = np.linspace(-1,1,nbinphi)+0.5*hphi
        X, Y = np.meshgrid(x, y, indexing='ij')

        for frame in range(self.frm_ini, self.frm_fin+1) :

            file_input = self.grom_fld+'/cv'+str(frame).zfill(4)+'.gro'
            x_o, y_o, z_o, _ = self.positions_from_file(file_input)
            x_h1, y_h1, z_h1, _ = self.positions_from_file(file_input, atom_name="HW1")
            x_h2, y_h2, z_h2, _ = self.positions_from_file(file_input, atom_name="HW2")

            x_dip = 0.5*(x_h1+x_h2)-x_o
            y_dip = 0.5*(y_h1+y_h2)-y_o
            z_dip = 0.5*(z_h1+z_h2)-z_o
            r_dip = np.sqrt( x_dip*x_dip + y_dip*y_dip +z_dip*z_dip )

            cos_mu = z_dip/r_dip
            # phi_mu = np.rad2deg(np.arccos(z_dip/r_dip))

            """
            for i in range(len(phi_mu)) :
                
                i_phi = int(phi_mu[i]/hphi)
                i_phi = i_phi*(i_phi<nbinphi)*(i_phi>=0) + (nbinphi-1)*(i_phi>=nbinphi)
                i_z = int((z_o[i]-zlow)/hz)
                i_z = i_z*(i_z<nbinz)*(i_z>=0) + (nbinz-1)*(i_z>=nbinz)
                z_phi_map[i_z, i_phi] += (r_dip[i]<0.5)
            """

            for i in range(len(cos_mu)) :
                
                i_phi = int((1+cos_mu[i])/hphi)
                i_phi = i_phi*(i_phi<nbinphi)*(i_phi>=0) + (nbinphi-1)*(i_phi>=nbinphi)
                i_z = int((z_o[i]-zlow)/hz)
                i_z = i_z*(i_z<nbinz)*(i_z>=0) + (nbinz-1)*(i_z>=nbinz)
                z_phi_map[i_z, i_phi] += (r_dip[i]<0.5)

        z_phi_map = z_phi_map/np.sum(z_phi_map*hphi*hz)

        z_phi_map = -np.log(z_phi_map)

        """
            SAVE TO DO SOME TESTING
        """
        z_phi_map[np.where(z_phi_map==np.inf)] = np.max(z_phi_map[np.where(z_phi_map!=np.inf)])
        np.savetxt('free-energy.txt', np.c_[X.flatten(),Y.flatten(),z_phi_map.flatten()])
        # np.savetxt('free-energy.txt', z_phi_map)
        # First minimum
        ilow=int((0.85-zlow)/hz)
        iupp=int((0.95-zlow)/hz)
        jupp=int(1/hphi)
        jlow=int(0.1/hphi)
        min1 = np.where(z_phi_map[ilow:iupp,jlow:jupp]==np.min(z_phi_map[ilow:iupp,jlow:jupp]))
        min1[0][0] += ilow
        min1[1][0] += jlow
        # Second minimum
        ilow=int((0.65-zlow)/hz)
        iupp=int((0.75-zlow)/hz)
        jlow=int(1.9/hphi)
        jupp=int(2/hphi)
        min2 = np.where(z_phi_map[ilow:iupp,jlow:jupp]==np.min(z_phi_map[ilow:iupp,jlow:jupp]))
        min2[0][0] += ilow
        min2[1][0] += jlow
        ###########################################
        xminima = [x[min1[0][0]],x[min2[0][0]]]
        yminima = [y[min1[1][0]],y[min2[1][0]]]
        minvals = [z_phi_map[min1[0][0],min1[1][0]], z_phi_map[min2[0][0],min2[1][0]]]
        np.savetxt('minima.txt', np.c_[np.array(xminima),np.array(yminima),np.array(minvals)] )

        vmin = np.min(z_phi_map)

        # CS1 = plt.contourf(X, Y, z_phi_map, cmap=cm.hot, levels=levels, vmin=vmin, vmax=vmax)
        CS1 = plt.pcolormesh(X, Y, z_phi_map, cmap=cm.hot, vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(CS1)
        cbar.ax.get_yaxis().labelpad = 45
        cbar.ax.set_ylabel('$k_B T$', rotation=270, fontsize=25)
        plt.plot(xminima[0], yminima[0], 'wo', markersize=10.0)
        # plt.text(xminima[0]+0.05, yminima[0]+10,str(minvals[0]), c='w')
        plt.plot(xminima[1], yminima[1], 'wo', markersize=10.0)
        # plt.text(xminima[1]+0.05, yminima[1]+10,str(minvals[1]), c='w')
        plt.ylabel(r'$\cos\varphi$ [1]', fontsize=25)
        plt.xlabel(r'$z_O$ [nm]', fontsize=25)
        # plt.axis('equal')
        plt.show()

    def order_parameter_local(self, nbinphi=60, nbinz=100, zlow=0.6, zupp=1.6, levels=None, vmax=None, rh2o=0.3, zsi=0.0) :

        """
            Do the analysis of the order parameter for the adsorption into the substrate
        """

        z_phi_map = np.zeros( (nbinz, nbinphi), dtype=float )
        hphi = 2/nbinphi
        dz = zupp-zlow
        hz = dz/nbinz

        x = np.linspace(zlow,zupp,nbinz)+0.5*hz-zsi
        y = np.linspace(-1,1,nbinphi)+0.5*hphi
        X, Y = np.meshgrid(x, y, indexing='ij')

        ycl = array_from_file(self.read_fld+'/'+self.fr+'y.txt')

        for frame in range(self.frm_ini, self.frm_fin+1) :

            xcl = array_from_file(self.read_fld+'/'+self.fr+str(frame).zfill(4)+'.txt')
            posx_interp = sci_int.interp1d(ycl, xcl, kind='nearest', fill_value='extrapolate')

            file_input = self.grom_fld+'/cv'+str(frame).zfill(4)+'.gro'
            x_o, y_o, z_o, _ = self.positions_from_file(file_input)
            x_h1, y_h1, z_h1, _ = self.positions_from_file(file_input, atom_name="HW1")
            x_h2, y_h2, z_h2, _ = self.positions_from_file(file_input, atom_name="HW2")

            x_dip = 0.5*(x_h1+x_h2)-x_o
            y_dip = 0.5*(y_h1+y_h2)-y_o
            z_dip = 0.5*(z_h1+z_h2)-z_o
            r_dip = np.sqrt( x_dip*x_dip + y_dip*y_dip +z_dip*z_dip )

            cos_mu = z_dip/r_dip
            idx_loc = np.where( (x_o<posx_interp(y_o)+rh2o) & (x_o>posx_interp(y_o)-rh2o) )[0]

            # for i in range(len(cos_mu)) :
            for i in idx_loc :
                
                i_phi = int((1+cos_mu[i])/hphi)
                i_phi = i_phi*(i_phi<nbinphi)*(i_phi>=0) + (nbinphi-1)*(i_phi>=nbinphi)
                i_z = int((z_o[i]-zlow)/hz)
                i_z = i_z*(i_z<nbinz)*(i_z>=0) + (nbinz-1)*(i_z>=nbinz)
                z_phi_map[i_z, i_phi] += (r_dip[i]<0.5)

        z_phi_map = z_phi_map/np.sum(z_phi_map*hphi*hz)

        z_phi_map = -np.log(z_phi_map)

        """
            SAVE TO DO SOME TESTING
        """
        # z_phi_map[np.where(z_phi_map==np.inf)] = np.max(z_phi_map[np.where(z_phi_map!=np.inf)])
        z_phi_map[np.where(z_phi_map==np.inf)] = vmax
        np.savetxt('free-energy.txt', np.c_[X.flatten(),Y.flatten(),z_phi_map.flatten()])
        # First minimum
        ilow=int((0.85-zlow)/hz)
        iupp=int((0.95-zlow)/hz)
        jupp=int(1/hphi)
        jlow=int(0.1/hphi)
        min1 = np.where(z_phi_map[ilow:iupp,jlow:jupp]==np.min(z_phi_map[ilow:iupp,jlow:jupp]))
        min1[0][0] += ilow
        min1[1][0] += jlow
        # Second minimum
        ilow=int((0.65-zlow)/hz)
        iupp=int((0.75-zlow)/hz)
        jlow=int(1.9/hphi)
        jupp=int(2/hphi)
        min2 = np.where(z_phi_map[ilow:iupp,jlow:jupp]==np.min(z_phi_map[ilow:iupp,jlow:jupp]))
        min2[0][0] += ilow
        min2[1][0] += jlow
        ###########################################
        xminima = [x[min1[0][0]],x[min2[0][0]]]
        yminima = [y[min1[1][0]],y[min2[1][0]]]
        minvals = [z_phi_map[min1[0][0],min1[1][0]], z_phi_map[min2[0][0],min2[1][0]]]
        np.savetxt('minima.txt', np.c_[np.array(xminima),np.array(yminima),np.array(minvals)] )

        vmin = np.min(z_phi_map)

        CS1 = plt.contourf(X, Y, z_phi_map, cmap=cm.hot, levels=levels, vmin=vmin, vmax=vmax)
        # CS1 = plt.pcolormesh(X, Y, z_phi_map, cmap=cm.hot, vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(CS1)
        cbar.ax.get_yaxis().labelpad = 45
        cbar.ax.set_ylabel('$k_B T$', rotation=270, fontsize=50)
        cbar.ax.tick_params(labelsize=25)
        plt.plot(xminima[0], yminima[0], 'wo', markersize=10.0)
        plt.plot(xminima[1], yminima[1], 'wo', markersize=10.0)
        plt.ylabel(r'$\cos\varphi$ [1]', fontsize=40)
        plt.xlabel(r'$z_O$ [nm]', fontsize=40)
        plt.tick_params(labelsize=30)
        plt.show()

    def van_hove(self, tlag=1, outfreq=10) :

        print("[densmap] Estimating the Van Hove function in two linear dimensions")

        for frame in range(self.frm_ini+tlag, self.frm_fin+1) :

            if (frame-self.frm_ini+tlag) % outfreq == 0 :
                print("[densmap] frame", frame)

            # Reading new and old atom position
            file_new = self.grom_fld+'/cv'+str(frame).zfill(4)+'.gro'
            file_old = self.grom_fld+'/cv'+str(frame-tlag).zfill(4)+'.gro'
            x_atom_new, y_atom_new, z_atom_new, resid_new = self.positions_from_file(file_new)
            x_atom_old, y_atom_old, z_atom_old, resid_old = self.positions_from_file(file_old)

            for i in range(len(x_atom_old)) :

                j_idx = np.argwhere(resid_new==resid_old[i])

                # Do not count atoms that have travelled across period dimensions
                if len(j_idx)==0 or j_idx[0][0]>=len(y_atom_old):
                    dyi = self.Ly
                else :
                    dyi = np.abs(y_atom_new[j_idx[0][0]]-y_atom_old[i])
                
                if dyi < 0.5*self.Ly :
                    
                    j = j_idx[0][0]

                    # disp_x = x_atom_new[j]-x_atom_old[i]
                    # disp_z = z_atom_new[j]-z_atom_old[i]

                    # Let's look at the absolute unsigned displacement
                    disp_x = np.abs(x_atom_new[j]-x_atom_old[i])
                    disp_z = np.abs(z_atom_new[j]-z_atom_old[i])

                    self.bin_vh(disp_x, disp_z)

        self.normalize_vh()

    def van_hove_local(self, tlag=1, outfreq=10, rh2o=0.3) :

        print("[densmap] Estimating the Van Hove function in two linear dimensions")

        y = array_from_file(self.read_fld+'/'+self.fr+'y.txt')

        for frame in range(self.frm_ini+tlag, self.frm_fin+1) :

            if (frame-self.frm_ini+tlag) % outfreq == 0 :
                print("[densmap] frame", frame)

            xcl = array_from_file(self.read_fld+'/'+self.fr+str(frame).zfill(4)+'.txt')
            posx_interp = sci_int.interp1d(y, xcl, kind='nearest', fill_value='extrapolate')

            file_new = self.grom_fld+'/cv'+str(frame).zfill(4)+'.gro'
            file_old = self.grom_fld+'/cv'+str(frame-tlag).zfill(4)+'.gro'
            x_atom_new, y_atom_new, z_atom_new, resid_new = self.positions_from_file(file_new)
            x_atom_old, y_atom_old, z_atom_old, resid_old = self.positions_from_file(file_old)

            # Find set of 'new' atoms that are close enough to the contact line
            idx_loc = np.where( (x_atom_new<posx_interp(y_atom_new)+rh2o) & (x_atom_new>posx_interp(y_atom_new)-rh2o) )[0]

            for i in idx_loc :

                if i<len(resid_old) :

                    j_idx = np.argwhere(resid_new==resid_old[i])

                    # Do not count atoms that have travelled across period dimensions
                    if len(j_idx)==0 or j_idx[0][0]>=len(y_atom_old):
                        dyi = self.Ly
                    else :
                        dyi = np.abs(y_atom_new[j_idx[0][0]]-y_atom_old[i])
                
                    if dyi < 0.5*self.Ly :

                        j = j_idx[0][0]

                        disp_x = x_atom_new[j]-x_atom_old[i]
                        disp_z = z_atom_new[j]-z_atom_old[i]

                        self.bin_vh(disp_x, disp_z)

        self.normalize_vh()


    def markov_chain(self, n_block=10, zl1=0.9, zl2=1.1, sepc=0.5, rh2o=0.3, tlag=1, outfreq=10) :

        print("[densmap] Estimating the transition matrix of the Markov process")

        def markov_classifier(z, c) :

            """
                Given (z_O, cos(phi)) returns the state associated to the instantaneous conformation
            """

            if z<=zl2 and z>zl1 and c<=sepc :
                return 1
            elif z<=zl1 and c>sepc :
                return 0
            else :
                return 2

        transition_matrix_output = np.zeros((4,4))
        error_matrix_output = np.zeros((4,4))

        states = dict()

        ycl = array_from_file(self.read_fld+'/'+self.fr+'y.txt')

        n_frames = self.frm_fin+1-self.frm_ini+tlag

        n_avg = 0

        for frame_block in range(self.frm_ini+tlag, self.frm_fin+1, n_block) :

            transition_matrix = np.zeros((4,4))
            error_matrix = np.zeros((4,4))

            for frame in range(frame_block,min(frame_block+n_block,self.frm_fin+1)) :

                if (frame-self.frm_ini+tlag) % outfreq == 0 :
                    print("[densmap] frame", frame)

                # New
                xcl = array_from_file(self.read_fld+'/'+self.fr+str(frame).zfill(4)+'.txt')
                posx_interp = sci_int.interp1d(ycl, xcl, kind='nearest', fill_value='extrapolate')
                file_input = self.grom_fld+'/cv'+str(frame).zfill(4)+'.gro'
                x_o, y_o, z_o, resid = self.positions_from_file(file_input)
                x_h1, y_h1, z_h1, _ = self.positions_from_file(file_input, atom_name="HW1")
                x_h2, y_h2, z_h2, _ = self.positions_from_file(file_input, atom_name="HW2")
                x_dip = 0.5*(x_h1+x_h2)-x_o
                y_dip = 0.5*(y_h1+y_h2)-y_o
                z_dip = 0.5*(z_h1+z_h2)-z_o
                r_dip = np.sqrt( x_dip*x_dip + y_dip*y_dip +z_dip*z_dip )
                cos_mu = z_dip/r_dip
                idx_loc = np.where( (x_o<posx_interp(y_o)+rh2o) & (x_o>posx_interp(y_o)-rh2o) )[0]
                x_new = x_o
                y_new = y_o
                resid_new = resid

                class_new = dict()
                for i in range(len(resid)) :
                    if i in idx_loc :
                        class_new[resid[i]] = markov_classifier(z_o[i], cos_mu[i])
                    else :
                        class_new[resid[i]] = 2

                # Old
                xcl = array_from_file(self.read_fld+'/'+self.fr+str(frame-tlag).zfill(4)+'.txt')
                posx_interp = sci_int.interp1d(ycl, xcl, kind='nearest', fill_value='extrapolate')
                file_input = self.grom_fld+'/cv'+str(frame-tlag).zfill(4)+'.gro'
                x_o, y_o, z_o, resid = self.positions_from_file(file_input)
                x_h1, y_h1, z_h1, _ = self.positions_from_file(file_input, atom_name="HW1")
                x_h2, y_h2, z_h2, _ = self.positions_from_file(file_input, atom_name="HW2")
                x_dip = 0.5*(x_h1+x_h2)-x_o
                y_dip = 0.5*(y_h1+y_h2)-y_o
                z_dip = 0.5*(z_h1+z_h2)-z_o
                r_dip = np.sqrt( x_dip*x_dip + y_dip*y_dip +z_dip*z_dip )
                cos_mu = z_dip/r_dip
                idx_loc = np.where( (x_o<posx_interp(y_o)+rh2o) & (x_o>posx_interp(y_o)-rh2o) )[0]
                x_old = x_o
                y_old = y_o
                resid_old = resid

                class_old = dict()
                for i in range(len(resid)) :
                    if i in idx_loc :
                        class_old[resid[i]] = markov_classifier(z_o[i], cos_mu[i])
                    else :
                        class_old[resid[i]] = 2

                # From the outside
                inlet = set(class_new) - set(class_old)
                for rj in inlet :
                    transition_matrix[class_new[rj],2] += 1

                # To the outside
                outlet = set(class_old) - set(class_new)
                for ri in outlet :
                    transition_matrix[2,class_old[ri]] += 1

                # Inside c.l. region
                cl_region = set(class_old).intersection(set(class_new))
                for r in cl_region :
                    if class_new[r] == 0 and class_old[r] == 0 :
                        dx = x_new[np.where(resid_new==r)[0][0]]-x_old[np.where(resid_old==r)[0][0]]
                        dy = y_new[np.where(resid_new==r)[0][0]]-y_old[np.where(resid_old==r)[0][0]]
                        d = np.sqrt(dx*dx+dy*dy)
                        if d > 0.5*self.dsio2 :
                            class_new[r] = 3
                    transition_matrix[class_new[r],class_old[r]] += 1
                    transition_matrix[class_old[r],class_new[r]] += 1

            # Normalizing transition matrix
            transition_matrix_output += transition_matrix
            transition_matrix /= np.sum(transition_matrix, axis=1)
            error_matrix = transition_matrix*transition_matrix

            n_avg += 1
            error_matrix_output += error_matrix

        # Normalizing transition matrix
        """
        norm_factor = np.sum(transition_matrix, axis=1)
        error_matrix = error_matrix-transition_matrix*transition_matrix
        error_matrix = np.sqrt(error_matrix/(self.frm_fin+1-self.frm_ini+tlag))
        error_matrix /= norm_factor
        transition_matrix /= norm_factor
        """

        transition_matrix_output /= np.sum(transition_matrix_output, axis=1)
        error_matrix_output /= n_avg
        error_matrix_output = error_matrix_output - transition_matrix_output*transition_matrix_output
        error_matrix_output = np.sqrt(error_matrix_output/n_avg)

        # Testing
        print(transition_matrix_output)
        print(np.sum(transition_matrix_output, axis=0))

        return transition_matrix_output, error_matrix_output


    def classify_jump_roll(self, 
                disp_thresh=0.2, r_thresh=0.15, z_thresh=0.15, tlag=1, silica_folder=None, fps=12, frame_plot=5):

        stick_dx = tlag*TIME_STEP*self.wall_velocity

        y = array_from_file(self.read_fld+'/'+self.fr+'y.txt')
        xold = array_from_file(self.read_fld+'/'+self.fr+str(self.frm_ini).zfill(4)+'.txt')

        if self.make_movie :
            mpl.use("Agg")
        print("[ -- Producing movie of the contact line in (x,y) -- ]")
        output_file_name = self.movie_name
        FFMpegWriter = manimation.writers['ffmpeg']
        metadata = dict(title='Contact Line', artist='Michele Pellegrino',
                        comment='Contact line x(y) over time')
        writer = FFMpegWriter(fps=fps, metadata=metadata)
        fig, _ = plt.subplots()

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
                
                # The displacement needs to account for the steady wall velocity
                # I.e: if the wall is displaced by dX over tframes and xnew-xold = dX, 
                # then the molecule has not moved (significantly)
                max_disp = infty_norm(xnew, xold+stick_dx)

                file_new = self.grom_fld+'/cv'+str(frame).zfill(4)+'.gro'
                file_old = self.grom_fld+'/cv'+str(frame-tlag).zfill(4)+'.gro'
                x_atom_new, y_atom_new, z_atom_new, resid_new = self.positions_from_file(file_new)
                x_atom_old, y_atom_old, z_atom_old, resid_old = self.positions_from_file(file_old)

                if silica_folder != None :
                    file_lattice = silica_folder+'/cv'+str(frame).zfill(4)+'.gro'
                    x_lattice, y_lattice, _, _ = self.positions_from_file(file_lattice, atom_name="SI")
                    y_lattice = np.concatenate((y_lattice-self.Ly, y_lattice, y_lattice+self.Ly), axis=0)
                    x_lattice = np.concatenate((x_lattice, x_lattice, x_lattice), axis=0)
                    i_patch = np.where((y_lattice<=self.Ly+0.5)*(y_lattice>-0.5)*(x_lattice<=self.xmax+0.5)*(x_lattice>self.xmin-0.5))
                    y_lattice = y_lattice[i_patch]
                    x_lattice = x_lattice[i_patch]
                    triang = tri.Triangulation(y_lattice, x_lattice)

                if max_disp >= disp_thresh :

                    displacement = xnew-xold-stick_dx

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
                                dx_atom.append(x_atom_new[i_new]-x_atom_old[i_old]-stick_dx)
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

                if frame==frame_plot and not(self.make_movie) :

                    plt.axis('scaled')
                    plt.ylim([self.xmin,self.xmax])
                    plt.xlim([0.0, self.Ly])
                    plt.plot(y, xnew, 'k-', linewidth=7.5, zorder=4)
                    plt.fill_between(y, xnew,  y2=self.xmax*self.left_int+self.xmin*(1-self.left_int), color='tab:cyan', zorder=1)

                    plt.fill_between(y, xnew,  y2=xnew+0.3, color='tab:green', zorder=1)
                    plt.fill_between(y, xnew,  y2=xnew-0.3, color='tab:green', zorder=1)

                    # Plot the water molecules
                    rspce = 300*0.15
                    # rspce = ax.transData.transform([rspce,0])[0] - ax.transData.transform([0,0])[0]
                    plt.scatter(y_atom_new, x_atom_new, alpha=0.6, color='tab:grey', s=np.pi*rspce**2, zorder=3, linewidths=1.0, edgecolors='k')

                    # Plot lattice triangulation
                    if silica_folder != None :
                        plt.triplot(triang, color='tab:grey', lw=1.0, zorder=2)

                    plt.xlabel("y [nm]", fontsize=85)
                    plt.ylabel("x [nm]", fontsize=85)
                    plt.xticks(fontsize=65)
                    plt.yticks(fontsize=65)

                    plt.show()

                if self.make_movie :

                    plt.axis('scaled')
                    plt.ylim([self.xmin,self.xmax])
                    plt.xlim([0.0, self.Ly])
                    plt.title('CL @t='+str(frame).zfill(4)+'ps', fontsize=17.5)
                    plt.plot(y, xold, 'k--', linewidth=1.5, zorder=2)
                    plt.plot(y, xnew, 'k-', linewidth=2.0, zorder=2)
                    plt.fill_between(y, xnew,  y2=self.xmax*self.left_int+self.xmin*(1-self.left_int), color='tab:cyan', zorder=1)
                    plt.fill_between(y, xnew,  y2=xnew+0.3, color='tab:green', zorder=1)
                    plt.fill_between(y, xnew,  y2=xnew-0.3, color='tab:green', zorder=1)

                    """
                    if max_disp < disp_thresh :
                        plt.fill_between(y, xold, xnew, color='tab:green', zorder=1)
                    else :
                        if true_pos :
                            plt.fill_between(y, xold, xnew, color='tab:red', zorder=1)
                        else :
                            plt.fill_between(y, xold, xnew, color='tab:orange', zorder=1)
                    if len(arrow_base_x) > 0 :
                        for j in range(len(arrow_base_x)) :
                            plt.arrow(arrow_base_y[j], arrow_base_x[j], arrow_dy[j], arrow_dx[j], 
                                head_width=0.05, head_length=0.075, fc='k', ec='k')
                    """

                    # Plot the water molecules
                    rspce = 50*0.15
                    # rspce = ax.transData.transform([rspce,0])[0] - ax.transData.transform([0,0])[0]
                    plt.scatter(y_atom_new, x_atom_new, alpha=0.5, color='tab:grey', s=np.pi*rspce**2, zorder=2, linewidths=0.5, edgecolors='k')

                    # Plot lattice triangulation
                    if silica_folder != None :
                        plt.triplot(triang, color='tab:grey', lw=0.5, zorder=2)

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
    frame_init = 1
    frame_fin = 3999
    
    """
    xmin = 52.2545 - 6.0*0.18429527937524576
    xmax = 52.2545 + 6.0*0.18429527937524576
    gro_folder = '/home/michele/BeskowDiag/Select_Q4_Equil/TrjGroBL'
    silica_folder = '/home/michele/BeskowDiag/Select_Q4_Equil/SubBottom'
    input_folder = '/home/michele/densmap/ContactLineProfiles/EquilQ4BL/'
    """

    xmin = 57.884825 - 6.0*0.2352970874766621
    xmax = 57.884825 + 6.0*0.2352970874766621
    gro_folder = '/home/michele/BeskowDiag/Select_Q3_Equil/TrjGroBL'
    silica_folder = '/home/michele/BeskowDiag/Select_Q3_Equil/SubBottom'
    input_folder = '/home/michele/densmap/ContactLineProfiles/EquilQ3BL/'

    """
    xmin = 48.618700000000004 - 6.0*0.4229010640800045
    xmax = 48.618700000000004 + 6.0*0.4229010640800045
    gro_folder = '/home/michele/BeskowDiag/Select_Q4_C0020/TrjGroBL'
    silica_folder = '/home/michele/BeskowDiag/Select_Q4_C0020/SubBottom'
    input_folder = '/home/michele/densmap/ContactLineProfiles/Ca002BL'
    """

    """
    xmin = 97.13137500000002 - 6.0*0.3454136786159459
    xmax = 97.13137500000002 + 6.0*0.3454136786159459
    gro_folder = '/home/michele/BeskowDiag/Select_Q4_C0020/TrjGroBR'
    silica_folder = '/home/michele/BeskowDiag/Select_Q4_C0020/SubBottom'
    input_folder = '/home/michele/densmap/ContactLineProfiles/Ca002BR'
    """

    clc = ContactLineClassifier(Ly, xmax, xmin, frame_init, frame_fin, input_folder, gro_folder, 
                                capillary_number=-0.02, pbcz=0, left_int=1, movie=True, movie_name="q3.mp4", root='cv')

    # ntp, nfp = clc.classify_jump_roll(tlag=1, silica_folder=silica_folder, fps=6)

    clc.order_parameter_local(nbinphi=100 , nbinz=100, levels=30, vmax=5)
    # clc.order_parameter(nbinphi=100 , nbinz=100, levels=30, vmax=5)

    """

    tlag_vec = np.linspace(0,150,31, dtype=int)

    p_jump = []
    p_roll = []
    p_jump_err = []
    p_roll_err = []
    for tlag in tlag_vec :
        print("### tlag = "+str(tlag)+"ps ###")
        T, sigmaT = clc.markov_chain(tlag=tlag)
        p_jump.append(T[3,0])
        p_roll.append(T[0,1])
        p_jump_err.append(sigmaT[3,0])
        p_roll_err.append(sigmaT[0,1])

    np.savetxt('./p_jump_38rec.txt', np.array(p_jump))
    np.savetxt('./p_roll_38rec.txt', np.array(p_roll))
    np.savetxt('./p_jump_38rec_err.txt', np.array(p_jump_err))
    np.savetxt('./p_roll_38rec_err.txt', np.array(p_roll_err))
    np.savetxt('./tlag_vec_38rec.txt', np.array(tlag_vec))

    plt.step(tlag_vec, p_jump, 'r-', linewidth=7.5, label='jump')
    plt.step(tlag_vec, p_roll, 'b-', linewidth=7.5, label='roll')
    plt.legend(fontsize=60)
    plt.xlabel(r'$t_{lag}$ [ps]', fontsize=50)
    plt.ylabel(r'$p_{ij}$ [1]', fontsize=50)
    plt.xticks(fontsize=50)
    plt.yticks(fontsize=50)
    plt.show()

    """

    """

    lags = np.logspace(0,3,10,dtype=int)

    for lk in lags :

        clc.initialize_vh(xlow=0, xupp=2)
        clc.van_hove(tlag=lk)
        clc.save_vh('VanHove', str(lk).zfill(4))
        # clc.plot_vh()

    """
    
    """
    lag_vec = np.unique(np.logspace(np.log(1),np.log(500),base=np.e,dtype=int,num=11))
    drefx = np.sqrt(3)*0.45/2
    drefz = 0.94-0.69
    """

    """
    for tl in lag_vec :
        print("t_lag = "+str(tl))
        clc.initialize_vh(xlow=0, xupp=2)
        clc.van_hove(tlag=tl)
        # clc.save_vh('VanHove', str(lk).zfill(4))
        clc.plot_vh(drefx, drefz, tl)
    """

    # print("False positive fraction = "+str(nfp/(nfp+ntp)))
    # print("Total true positives = "+str(ntp))

if __name__ == "__main__":
    main()
