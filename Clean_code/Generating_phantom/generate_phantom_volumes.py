import sys
import numpy as np
from init_obj import *
from att_coef_object import *
from math import sin, cos, pi
import h5py

TYPE = 'float64'   

if __name__ == '__main__':

    # file name to save phantom data
    phantomFileName = 'phantom.h5'
    phantomUseFile = False
    
    # object parameters
    obj = {}
    obj = init_obj(obj, plot=False)
   

    #-------------------------------------------------------------------------#
    ################### GENERATION OF PHANTOM DATA ############################
    
    # phantom grid - voxel centers for phantom
    r = 1.0          # radial extent of phantom in [cm]
    dpc = False      # dpc = True -> generation of phase phantom
                     # dpc = False -> generation of absorption phantom
    
    Nxp = 64 # number of voxels in x-direction
    Nyp = 64 # number of voxels in y-direction
    Nzp = 64 # number of voxels in z-direction

    dxp = 2.0*r/Nxp  # voxel size in x-direction
    dyp = 2.0*r/Nyp  # voxel size in y-direction
    dzp = 2.0*r/Nzp  # voxel size in z-direction
    bxp = -dxp*Nxp/2.0 
    byp = -dyp*Nyp/2.0
    bzp = -dzp*Nzp/2.0 
    # positions of voxel centres
    xcp = np.linspace(-dxp/2*(Nxp-1), dxp/2*(Nxp-1), Nxp)
    ycp = np.linspace(-dyp/2*(Nyp-1), dyp/2*(Nyp-1), Nyp)  
    zcp = np.linspace(-dzp/2*(Nzp-1), dzp/2*(Nzp-1), Nzp) 
    

    if phantomUseFile:
        #load phantom from a file
        phantomf = h5py.File(phantomPhaseFileName, 'r') 
        phantomp = phantomf['phantom'][:]
        Nxp = phantomf.attrs['Nxp'] 
        Nyp = phantomf.attrs['Nyp'] 
        Nzp = phantomf.attrs['Nzp'] 
        dxp = phantomf.attrs['dxp'] 
        dyp = phantomf.attrs['dyp'] 
        dzp = phantomf.attrs['dzp'] 
        bxp = phantomf.attrs['bxp'] 
        byp = phantomf.attrs['byp'] 
        bzp = phantomf.attrs['bzp'] 
        phantomf.close()
    else:
	#compute phantom
        phantom = np.zeros((Nxp,Nyp,Nzp), dtype=TYPE)
        for sl in np.arange(Nzp):
            print "Slice", sl
            for row in np.arange(Nxp):
                for col in np.arange(Nyp):
                    x = [xcp[row], ycp[col], zcp[sl]]
                    if (xcp[row]**2 + ycp[col]**2) <= r**2:
                        phantom[row,col,sl] = reconstruct_phantom(obj, x, dpc)
        phantomf = h5py.File(phantomFileName, 'w')   
        phantomf.create_dataset('phantom', data=phantom, compression='gzip', \
                                 compression_opts=9) 
        phantomf.attrs['Nxp'] = Nxp 
        phantomf.attrs['Nyp'] = Nyp
        phantomf.attrs['Nzp'] = Nzp
        phantomf.attrs['dxp'] = dxp 
        phantomf.attrs['dyp'] = dyp
        phantomf.attrs['dzp'] = dzp
        phantomf.attrs['bxp'] = bxp 
        phantomf.attrs['byp'] = byp
        phantomf.attrs['bzp'] = bzp         
        phantomf.close()   
    
   
