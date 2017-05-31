import sys
import numpy as np
from init_obj import *
from att_coef_object import *
from base import *
from ray_tracing_Siddon_v4 import *
from forward_projection import *
import h5py
from time import time
import cProfile, pstats, StringIO
import argparse


TYPE = 'float64'   

def getArgs():
    parser = argparse.ArgumentParser(description = 
    '''Script performs reconstruction of object slice from spiral CT 
    projections. Projection collection is simulated using cone beam  
    geometry with curved detector. The 3D Shepp-Logan phantom  
    is used as simulated object.''')
    
    # detector related parameters
    parser.add_argument('-D', '--D', dest='D', type=np.float64, default=6.0,
                        help = '''Curved detector radius in [cm] or ''' 
                        + '''shortest distance from source to detector in ''' 
                        + ''' flat detector geometry ''')
    parser.add_argument('-H', '--H', dest='H', type=np.float64, default=0.5,
                        help = 'Detector height in [cm]')
    parser.add_argument('-Nr', '--Nr', dest='Nr', type=np.int16, default=16,
                        help = 'Number of detector rows')   
    parser.add_argument('-Nc', '--Nc', dest='Nc', type=np.int16, default=138,
                        help = 'Number of detector columns')
    parser.add_argument('-dew', '--dew', dest='dew', 
                        type=np.float64, default=0.5/16,
                        help = '''Detector element width [cm]''')
    parser.add_argument('-deh', '--deh', dest='deh', 
                        type=np.float64, default=0.5/16,
                        help = '''Detector element height [cm]''')
    parser.add_argument('-shiftd', '--shift_detector',
                        nargs='+', type=np.float,
                        default = [0, 0.0],
                        help=''' Position of the source point projection '''
                           + ''' onto the detector''')    
    parser.add_argument("-d", "--detector", type=str, default = 'curved',
                        help = 'Detector type: curved or flat')


    # source trajectory parameters
    parser.add_argument('-R', '--R', dest='R', type=np.float64, default=3.0,
                        help = 'Source trajectory radius in [cm]') 
    parser.add_argument('-Pfactor', '--Pfactor', dest='Pfactor', 
                        type=np.float64, default=1.0,
                        help = 'Proportion of maximal pitch, P = Pfactor*Pmax')
    parser.add_argument('-ds', '--ds', dest='ds', 
                        type=np.float64, default=0.5/16,
                        help = 'Stepsize of source point parameter in [rad]')


    # reconstruction parameters
    parser.add_argument('-r', '--r', dest='r', type=np.float64, default=1.0,
                        help = '''ROI radius in [cm]''')
    parser.add_argument('-Nx', '--Nx', dest='Nx', type=np.int16, default=256,
                        help = 'Number of rows of reconstructed slice')   
    parser.add_argument('-Ny', '--Ny', dest='Ny', type=np.int16, default=256,
                        help = 'Number of columns of reconstructed slice')    
    parser.add_argument("-img", "--imaging", type=str, default = 'phase',
                        help = 'imaging contrast type: absorption or phase')
    

    # precomputed files 
    parser.add_argument("-PIUsef", "--PIUseFile", type=bool, 
                        default = False,
                        help = 'Use precomputed PI line interval range')
    parser.add_argument("-PIf", "--PIFileName", type=str, 
                        default = 'PIfile.mat',
                        help = 'Name of the file with stored PI lines')

    parser.add_argument("-phaUsef", "--phantomUseFile", type=bool, 
                        default = True,
                        help = 'Use phantom data stored in a file')
    parser.add_argument("-phaf", "--phantomFileName", type=str, 
                        default = 'phantom_dummy.h5',
                        help = 'File name containing phantom data')
    parser.add_argument("-projUsef", "--projUseFile", type=bool, 
                        default = True,
                        help = 'Use projection data stored in a file')
    parser.add_argument("-projf", "--projFileName", type=str, 
                        default = 'projection_dummy.h5',
                        help = 'File name for projection data')
    args = parser.parse_args()
    return args 



if __name__ == '__main__':


    #pr = cProfile.Profile()
    #pr.enable()

    # Get arguments
    args = getArgs()

    phantomFileName = 'phantomAbsorb4.h5'
    phantomPhaseFileName = 'phantomPhase4.h5'
    phantomUseFile = True
    projPhaseFileName = 'projectionPhase4_v4.h5'
    projFileName = 'projectionAbsorb4.h5'
    projUseFile = True

    dpc = True
 
    # initialize geometry parameters
    conf = init_scan_geometry(args)
    # object parameters
    obj = {}
    obj = init_obj(obj, plot=False)
   
    # reconstruction slice position
    x3 = -0.25

    # geometry parameters
    R = args.R                   # source trajectory radius
    D = args.D                   # curved detector radius
    H = args.H                   # curved detector height 
    Nr = args.Nr                 # curved detector number rows 
    Nc = args.Nc                 # curved detector number columns  
    r = args.r                   # radius of a circle containing object
    Nx = args.Nx                 # number of columns in rec. object image
    Ny = args.Ny                 # number of rows in rec. object image
    delta_w = args.deh           # detector element height
    arc_alpha = args.dew         # detector element width (arc lenght)
    delta_alpha = arc_alpha/D    # angular size of detector element
    shift_detector = args.shift_detector   # shift between source point
                                 # projection and center of detector, 
                                 # [alpha_shift, w_shift]
    halfAngle = delta_alpha*Nc/2 # cone beam half angle
    rm = R * sin(halfAngle)      # max. radius of reconstructable FOV 
    halfAngleFOV = np.arcsin(r/R)# half angle of FOV     

    if (delta_alpha/2.0 - shift_detector[0])>1e-2:
        sys.exit('''\nError: shift_detector not the same as in phantom''')

    if D != 2*R:
        sys.exit('''\nError: Curved detector radius should be '''\
                +'''initializedto two times source trajectory '''\
                +'''radius!\n''')
    if np.mod(Nr,2):
        sys.exit('''\nError: Number of detector rows was taken'''\
                +''' to be even''')
    if np.mod(Nc,2):
        sys.exit('''\nError: Number of detector columns was taken'''\
                +''' to be even''')
    if r>rm or r<=0:
        sys.exit('''\nError: Radius of ROI should be smaller''')
      
    # parametrized source trajectory
    Pmax = maxPitch(D, Nr, R, r, delta_w) # maximal pitch
    P =  Pmax
    h = P/(2*pi)    
    smin = x3/h - pi                   # minimal sb 
    smax = x3/h + 2*pi                 # maximal st
    ds = arc_alpha                     # source point stepsize 
    s = np.arange(smin,smax,ds) 
    if np.mod(len(s),2)!=0:
        s = np.arange(smin,smax+ds,ds)
    Np = len(s)                        # number of projections
    # source point positions
    ys = np.array([[R*cos(s[i]) for i in range(Np)], \
                   [R*sin(s[i]) for i in range(Np)], \
                   [P/(2*pi)*s[i] for i in range(Np)]], dtype=TYPE)

    # detector grid
    # alpha - angular sampling on detector grid   
    alpha = np.linspace(-delta_alpha/2*(Nc-1), delta_alpha/2*(Nc-1), Nc)\
            - shift_detector[0]
    # w - z-axis sampling on detector grid
    w = np.linspace(-delta_w/2*(Nr-1), delta_w/2*(Nr-1), Nr)\
            - shift_detector[1]

    #-------------------------------------------------------------------------#
    # phase phantom    
    # phantom grid - voxel centers for phantom

    factor = 4
    Nxp = 256*factor
    Nyp = 256*factor
    Nzp = 256*factor

    dxp = 2.0*r/Nxp
    dyp = 2.0*r/Nyp
    dzp = 2.0*r/Nzp
    bxp = -dxp*Nxp/2.0
    byp = -dyp*Nyp/2.0
    bzp = -dzp*Nzp/2.0 
    xcp = np.linspace(-dxp/2*(Nxp-1), dxp/2*(Nxp-1), Nxp) 
    ycp = np.linspace(-dyp/2*(Nyp-1), dyp/2*(Nyp-1), Nyp)  
    zcp = np.linspace(-dzp/2*(Nzp-1), dzp/2*(Nzp-1), Nzp) 
    
    if not phantomUseFile:
        phantom = np.zeros((Nxp,Nyp,Nzp), dtype=TYPE)
        for sl in np.arange(Nzp):
            print 'Slice', sl
            for row in np.arange(Nxp):
                for col in np.arange(Nyp):
                    x = [xcp[row], ycp[col], zcp[sl]]
                    if (xcp[row]**2 + ycp[col]**2) <= r**2:
                        phantom[row,col,sl] = reconstruct_phantom(obj, x, dpc)
        phantomf = h5py.File(phantomPhaseFileName, 'w')   
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
    
    # compute gradient of refraction index decrement 
    # centered derivatives
    #gradPhantom = np.array(np.gradient(phantom, dxp, dyp, dzp))
    '''plt.figure()
    extent = [yc[0], yc[-1], xc[0], xc[-1]]
    plt.imshow((gradPhantom[1][:,:,95]), cmap=plt.cm.gray, extent=extent, \
                     interpolation='none')
    plt.title('Gradient of refraction index decrement phantom')
    plt.colorbar()
    plt.show() ''' 

    #yp = ys[:, range(0,60,10)]
    #p =[s[0], s[10], s[20], s[30], s[40], s[50]]
    #p = [s[0]]
    # compute projection measurements for phase contrast imaging
    if projUseFile:
        projf = h5py.File(projPhaseFileName, 'r')   
        Dfpc = projf['Dfpc'][:] 
        projf.close()
         
    else:
        Dfpc = siddon_cone_beam_projection(phantomPhaseFileName, ys, s, shift_detector, 
                                            alpha, w, Nr, Nc, r, R, D, H, 
                                            Nxp, Nyp, Nzp, dxp, dyp, 
                                            dzp, bxp, byp, bzp, dpc)

        #Dfpc = siddon_cone_beam_projection(phantom, ys, p, shift_detector, 
        #                                   alpha, w, Nr, Nc, r, D, 
        #                                   Nxp, Nyp, Nzp, dxp, dyp, 
        #                                   dzp, bxp, byp, bzp, dpc)

        projf = h5py.File(projPhaseFileName, 'w')   
        projf.create_dataset('Dfpc', data=Dfpc, compression='gzip', \
                                 compression_opts=9) 
        projf.close()

    
         
    #proj_abs_analytic = cone_beam_projection(obj, conf, ys, s, shift_detector)


    #rec_slice1 = absorption_reconstruction(Df, x3, args)
    #rec_slice2 = absorption_reconstruction(proj_abs_analytic, x3, args)
    
    #pr.disable()
    #sm = StringIO.StringIO()
    #sortby = 'cumulative'
    #ps = pstats.Stats(pr, stream=sm).sort_stats(sortby)
    #ps.print_stats()
    #print sm.getvalue()
    N = 100
    plt.figure()
    plt.imshow(g1[N,:,:], cmap=plt.cm.gray, \
                     interpolation='none')
    plt.title('Phase Siddon')
    plt.colorbar()
    plt.show()

    plt.figure()
    plt.plot(g1[N,13,:])
    #plt.hold(True)
    #plt.plot(proj_abs_analytic[0,13,:],'r')
    plt.show()

    '''
    for i in range(9):
        plt.figure()
        plt.imshow(phantom[:,:,100*i], cmap=plt.cm.gray, \
                        interpolation='none')
        plt.title('Phantom')
        plt.colorbar()
        plt.show()'''
