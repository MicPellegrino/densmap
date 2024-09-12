import numpy as np
import matplotlib.pyplot as plt


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

def positions_from_file(file_name, natoms=3, atom_name="OW") :
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
                if n_temp == natoms :
                    n_temp = 0
                if line_data[2] == atom_name :
                    x_atom.append(line_data[4])
                    y_atom.append(line_data[5])
                    z_atom.append(line_data[6])
                    resid.append(line_data[0])
        in_file.close()
        return np.array(x_atom), np.array(y_atom), np.array(z_atom), np.array(resid) 


class Tracker :

    def __init__(self, folder_name, natoms=3, atom_name="OW") :
        self.fn = folder_name
        self.resdict = dict()
        self.natoms=natoms
        self.atom_name=atom_name

    def scan_resid(self, n, n0=0, root='cv') :
        for k in range(n0,n) :
            x, y, z, r = positions_from_file(self.fn+'/'+root+str(k+1).zfill(4)+'.gro',natoms=self.natoms,atom_name=self.atom_name)
            for resid in r :
                if resid in self.resdict :
                    self.resdict[resid] += 1
                else :
                    self.resdict[resid] = 1
        self.resdict=dict(sorted(self.resdict.items(), key=lambda item: item[1], reverse=True))

    def track_res(self, resid, n, n0=0, root='cv') :
        walk_x = []
        walk_y = []
        for k in range(n0,n) :
            x, y, z, r = positions_from_file(self.fn+'/'+root+str(k+1).zfill(4)+'.gro',natoms=self.natoms,atom_name=self.atom_name)
            idx = np.where(r==resid)[0]
            if idx.size > 0 :
                walk_x.append(x[idx[0]])
                walk_y.append(y[idx[0]])
            else :
                walk_x.append(None)
                walk_y.append(None)
        return np.array(walk_x,dtype=np.float64), np.array(walk_y,dtype=np.float64)


def main() :

    Lx = 29.70000
    Ly = 4.67654
    Lz = 10.00000

    hdsi = 0.5*0.45
    Ns = int(Lx/hdsi)

    sites = np.linspace(0,hdsi*Ns,Ns)

    rf = '/home/michele/python_for_md/Glycerol40p/SmallMeniscus'
    
    """
    tracker_sol = Tracker(rf+'/SOL')
    tracker_sol.scan_resid(599)
    walk_x, walk_y = tracker_sol.track_res(7081, 599)
    t = np.linspace(0,len(walk_x),len(walk_x))    
    for sp in sites :
        plt.plot([t[0],t[-1]],[sp,sp],'r-')
    plt.plot(t, walk_x,'k.')
    plt.ylim([np.nanmin(walk_x),np.nanmax(walk_x)])
    plt.show()
    """

    tracker_gol = Tracker(rf+'/GOL',natoms=14,atom_name="O")
    tracker_gol.scan_resid(599)
    walk_x, walk_y = tracker_gol.track_res(1602, 599)
    t = np.linspace(0,len(walk_x),len(walk_x))    
    for sp in sites :
        plt.plot([t[0],t[-1]],[sp,sp],'r-')
    plt.plot(t, walk_x,'k.')
    plt.ylim([np.nanmin(walk_x),np.nanmax(walk_x)])
    plt.show()
    plt.hist(walk_x)
    plt.show()

if __name__ == "__main__":
    main()
