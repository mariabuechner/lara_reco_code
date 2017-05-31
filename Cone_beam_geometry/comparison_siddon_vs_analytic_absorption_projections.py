import sys
import numpy as np
from init_obj import *
from att_coef_object import *
from base import *
from ray_tracing_Siddon import *
from forward_projection import *
from time import time
import cProfile, pstats, StringIO
import h5py

TYPE = 'float64'    

if __name__ == '__main__':


    pr = cProfile.Profile()
    pr.enable()

    # Get arguments
    args = getArgs()
    phantomFileName = 'phantomAbsorb4.h5'
    phantomPhaseFileName = 'phantomPhase4.h5'
    phantomUseFile = False
    projPhaseFileName = 'projectionPhase4.h5'
    projUseFile = False

    dpc = True
 
    # initialize geometry parameters
    conf = init_scan_geometry(args)
    # object parameters
    obj = {}
    obj = init_obj(obj, plot=False)
   
    # reconstruction slice position
    x3 = -0.25

    # geometry parameters
    R = conf['source_trajectory']['radius']
    D = conf['curved_detector']['radius']
    H = conf['curved_detector']['height']
    Nr = conf['curved_detector']['number_rows']
    Nc = conf['curved_detector']['number_columns'] 
    r = conf['ROI']['radius']    # radius of a circle containing object
    Nx = conf['ROI']['NX']       # number of columns in object image
    Ny = conf['ROI']['NY']       # number of rows in object image
    delta_w = H/Nr               # detector element height
    arc_alpha = delta_w          # detector element width (arc lenght)
    delta_alpha = arc_alpha/D    # angular size of detector element
    shift_detector = [delta_alpha/2, 0] # shift between source point
                                 # projection and center of detector, 
                                 # [alpha_shift, w_shift]
    halfAngle = delta_alpha*Nc/2 # cone beam half angle
    rm = R * sin(halfAngle)      # max. radius of reconstructable FOV 
    halfAngleFOV = np.arcsin(r/R)# half angle of FOV    

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
    
    if phantomUseFile:
        phantomf = h5py.File(phantomPhaseFileName, 'r') 
        phantom = phantomf['phantom'][:]
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
        phantom = np.zeros((Nxp,Nyp,Nzp), dtype=TYPE)
        for sl in np.arange(Nzp*0.75):
            print 'Slice', sl
            for row in np.arange(Nxp):
                for col in np.arange(Nyp):
                    x = [xcp[row], ycp[col], zcp[sl]]
                    if (xcp[row]**2 + ycp[col]**2) <= r**2:
                        phantom[row,col,sl] = reconstruct_phantom(obj, x, True)
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
    gradPhantom = np.array(np.gradient(phantom, dxp, dyp, dzp))
    '''plt.figure()
    extent = [ycp[0], ycp[-1], xcp[0], xcp[-1]]
    plt.imshow((gradPhantom[1][:,:,95]), cmap=plt.cm.gray, extent=extent, \
                     interpolation='none')
    plt.title('Phantom')
    plt.colorbar()
    plt.show()   ''' 
    
    # absorption phantom made for comparison siddon vs analytical solution
    if phantomUseFile:
        phantomf = h5py.File(phantomFileName, 'r') 
        phantom = phantomf['phantom'][:]
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
        phantom = np.zeros((Nxp,Nyp,Nzp), dtype=TYPE)
        for sl in np.arange(Nzp*0.75):
            print 'Slice', sl
            for row in np.arange(Nxp):
                for col in np.arange(Nyp):
                    x = [xcp[row], ycp[col], zcp[sl]]
                    if (xcp[row]**2 + ycp[col]**2) <= r**2:
                        phantom[row,col,sl] = reconstruct_phantom(obj, x, False)
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

    
    p = [s[0]]
    '''
    projection_phase = siddon_cone_beam_projection(gradPhantom, ys, p, shift_detector, 
                                                alpha, w, Nr, Nc, r, D, Nx, Ny, Nz, dx, dy, 
                                                dz, bx, by, bz, True)
    '''
    projection_absorb = siddon_cone_beam_projection(phantom, ys, p, shift_detector, 
                                                alpha, w, Nr, Nc, r, D, Nxp, Nyp, Nzp, dxp, dyp, 
                                                dzp, bxp, byp, bzp, False)  

    proj_abs_analytic = cone_beam_projection(obj, conf, ys, p, shift_detector)

    
    pr.disable()
    s = StringIO.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print s.getvalue()
    savemat('single_absorbtion_projections', dict(analytic = proj_abs_analytic, siddon = projection_absorb))  


    '''
    plt.figure()
    res = plt.imshow(projection_phase[0,:,:], cmap=plt.cm.gray, \
                     interpolation='none')
    plt.title('Phase Siddon')
    plt.colorbar()
    plt.show()
    '''
    '''
    plt.figure()
    res = plt.imshow(projection_absorb[0,:,:], cmap=plt.cm.gray, \
                     interpolation='none')
    plt.title('Absorption Siddon')
    plt.colorbar()
    plt.show()   

    plt.figure()
    res = plt.imshow(proj_abs_analytic[0,:,:], cmap=plt.cm.gray, \
                     interpolation='none')
    plt.title('Absorption analytic')
    plt.colorbar()
    plt.show() 

    plt.figure()
    res = plt.imshow(np.abs(proj_abs_analytic[0,:,:] - projection_absorb[0,:,:]), cmap=plt.cm.gray, \
                     interpolation='none')
    plt.title('Absorbtion analytic vs Siddon')
    plt.colorbar()
    plt.show() 


    plt.figure()
    plt.plot(proj_abs_analytic[0,4,:])
    plt.hold(True)
    plt.plot(projection_absorb[0,4,:], 'r')
    plt.title('9th detector row analytic vs siddon absorbtion projection')
    plt.show() 
    '''        
