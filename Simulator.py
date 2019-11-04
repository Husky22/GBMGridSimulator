import warnings
import random
warnings.simplefilter('ignore')
import numpy as np
import astromodels
from threeML import * 
import astropy.units as u
import astropy.coordinates as coord
from gbmgeometry import PositionInterpolator, GBM, GBMFrame, gbm_frame
from astropy.coordinates import BaseCoordinateFrame, Attribute, RepresentationMapping
from astropy.coordinates import frame_transform_graph


class Simulator():
    """
    Fermi GBM Simulator

    """

    def __init__(self,source_number, spectrum_matrix_dimensions ):
        self.N=source_number
        self.spectrum_dimension=spectrum_matrix_dimensions


    def fibonacci_sphere(self,randomize=True):
        """
        The standard algorithm for isotropic point distribution on a sphere based on the fibonacci-series
        """
        samples=self.N
        rnd=1.
        if randomize:
            rnd=random.random()*samples
            points=[]
            offset=2./samples
            increment=math.pi*(3.-math.sqrt(5.))

            for i in range(samples):
                y=((i*offset)-1)+(offset/2)
                r=math.sqrt(1-y**2)
                phi=((i+rnd)%samples)*increment
                x=math.cos(phi)*r
                z=math.sin(phi)*r
                points.append(GridPoint([x,y,z],self.spectrum_dimension))
            return np.array(points)

    def voronoi_sphere(self):

        """
        A modified better version of the fibonacci algorithm
        """
        samples=self.N
        points=[]
        gr = (1+np.sqrt(5))/2
        e=11/2
        for i in range(samples-2):
            if i==0:
                t1=0
                t2=0
            elif i==samples-2:
                t1=1
                t2=0

            else:
                t1=((i+e+1/2) / (samples+2*e))%1
                t2=(i/gr)%1

            p1=np.arccos(2 * t1-1)-np.pi/2
            p2=2*np.pi * t2
            x=np.cos(p1)*np.cos(p2)
            y=np.cos(p1)*np.sin(p2)
            z=np.sin(p1)

            points.append(GridPoint([x,y,z],self.spectrum_dimension))

        return np.array(points)



    def setup(self,algorithm='Voronoi'):

        if algorithm=='Fibonacci':
            self.grid = self.fibonacci_sphere()

        elif algorithm=='Voronoi':
            self.grid = self.voronoi_sphere()

        for point in self.grid:
            point.generate_spectrum()

    def generate_j2000(self,trigdat):
        for gp in self.grid:
            gp.add_j2000(trigdat)



class GridPoint():
    def __init__(self,coord,dim):
        self.coord = coord
        self.dim=dim
        self.j2000 = None


    def generate_spectrum(self):
        """ Compute sample cutoff powerlaw spectra
        """
        n=self.dim[0]
        m=self.dim[1]
        self.spectrum_matrix=np.empty(self.dim,dtype=classmethod)
        index=np.linspace(-1.5,-1,n)
        epmax=np.linspace(100,400,m)
        i=0
        # source=PointSource()
        for index_i in index:
            j=0
            for epmax_i in epmax:
                self.spectrum_matrix[i,j]= astromodels.Cutoff_powerlaw(K=epmax_i,index=index_i)
                j+=1
            i+=1
    def add_j2000(self,trigdat,time=0.,final_frame=coord.FK5):
        """ Calculate the corresponding Ra and Dec coordinates
            for the already given coordinate in the Fermi-Frame
        """

        position_interpolator= PositionInterpolator(trigdat=trigdat)
        fermi=GBM(position_interpolator.quaternion(time),
                  position_interpolator.sc_pos(time) * u.km)

        x,y,z=position_interpolator.sc_pos(time)
        q1,q2,q3,q4=position_interpolator.quaternion(time)
        frame=GBMFrame(sc_pos_X=x,sc_pos_Y=y,sc_pos_Z=z,quaternion_1=q1,quaternion_2=q2,quaternion_3=q3,quaternion_4=q4,SCX=self.coord[0]*u.km,SCY=self.coord[1]*u.km,SCZ=self.coord[2]*u.km,representation='cartesian')
        # Muessen die Punkte ins unendliche projiziert werden?
        icrsdata=gbm_frame.gbm_to_j2000(frame,final_frame)
        self.j2000=icrsdata

    





simulation=Simulator(20,[4,4])
simulation.setup('Voronoi')

trigdat="/home/niklas/Dokumente/Bachelor/rawdata/191017391/glg_trigdat_all_bn191017391_v01.fit"

simulation.generate_j2000(trigdat)

print(simulation.grid[1])
