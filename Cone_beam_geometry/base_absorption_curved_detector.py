from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import sys
import numpy as np
from numpy import linalg as la
from scipy.io import savemat, loadmat
from init_obj import *
from att_coef_object import * 
from forward_projection import *
import numpy.fft as ft
from scipy.optimize import brentq
from functools import partial
from math import cos, sin, tan, atan, atan2, sqrt, floor, ceil, pi
from time import time
import argparse
#import cProfile, pstats, StringIO

TYPE = 'float64'


def getArgs():
    parser = argparse.ArgumentParser(description = 
    '''Script performs reconstruction of object slice from spiral CT 
    projections. Projection collection is simulated using cone beam  
    geometry with curved detector. The 3D Shepp-Logan phantom  
    is used as simulated object.''')
 
    parser.add_argument('-D', '--D', dest='D', type=np.float64, default=6.0,
                        help = 'Curved detector radius in [cm]')
    parser.add_argument('-H', '--H', dest='H', type=np.float64, default=0.5,
                        help = 'Curved detector height in [cm]')
    parser.add_argument('-Nr', '--Nr', dest='Nr', type=np.int16, default=16,
                        help = 'Number of rows of the curved detector')   
    parser.add_argument('-Nc', '--Nc', dest='Nc', type=np.int16, default=138,
                        help = 'Number of columns of the curved detector')
    parser.add_argument('-r', '--r', dest='r', type=np.float64, default=1.0,
                        help = '''ROI radius in [cm]''')
    parser.add_argument('-Nx', '--Nx', dest='Nx', type=np.int16, default=256,
                        help = 'Number of rows of the curved detector')   
    parser.add_argument('-Ny', '--Ny', dest='Ny', type=np.int16, default=256,
                        help = 'Number of columns of the curved detector')
    parser.add_argument('-R', '--R', dest='R', type=np.float64, default=3.0,
                        help = 'Source trajectory radius in [cm]') 
    parser.add_argument('-P', '--P', dest='P', type=np.float64, default=0.25,
                        help = 'Pitch of source trajectory in [cm]')
    parser.add_argument('-ds', '--ds', dest='ds', type=np.float64, default=1.0,
                        help = 'Stepsize of source point parameter in [rad]')

    args = parser.parse_args()
    return args

def init_scan_geometry(args):
    settings = {}
    # source trajectory (helix) parameters 
    settings.update({'source_trajectory':  
                        {'radius' : args.R}})        

    # curved detector parameters
    settings.update({'curved_detector': 
                        {'radius': args.D,
                         'height':args.H,
                         'number_rows':args.Nr,
                         'number_columns':args.Nc}}) 

    # ROI parameters
    settings.update({'ROI': 
                        {'radius': args.r,
                         'NX': args.Nx,
                         'NY': args.Ny}})     
    return settings

def nextpow2(i):
    n = 1
    while n < i: n *= 2
    return n

def k_line_point_w(D, P, R, psi, alpha):
    '''
    Computes w coordinate of a point with angular coordinate alpha
    on the k-line
    Input: D - radius of detector
           P - pitch of spiral source trajectory 
           R - radius of spiral source trajectory
           psi - angle defining k-plane
           alpha - angle of the point on the detector
    '''
    const = D*P / (2*pi*R)
    if np.abs(psi) < 1e-5:
        # psi = 0, psi/tan(psi) = 1
        return const * sin(alpha)
    else:
        return const * (psi*cos(alpha) + psi/tan(psi)*sin(alpha))

def number_k_lines(alpha_m, Nr, P, Pmax):
    '''
    Computes minimal number of k-lines for filtering step.
    Selected number of k-lines assures that the k-lines at the half fan angle 
    have maximal spacing equal to the detector row thickness divided by 2.
    Input: halfAngle - beam half angle 
           Nr - number of rows
           P - used pitch value
           Pmax - maximal value for pitch
    Output: number of k-lines (even number) 
    '''
    N = ceil((Nr-1) * P/Pmax * (1+(pi/2+alpha_m)*tan(alpha_m)))
    if np.mod(N,2) == 0:
        return N
    else:
        return N+1
    
def maxPitch(D, Nr, R, r, delta_w):
    '''
    Computes maximal pitch value for spiral source trajectory
    Input: D - radius of detector 
           Nr - number of rows
           R - radius of source trajectory
           r - radius of ROI
           delta_w - detector element height
    Output: maximal pitch value  
    '''
    alpha = np.arcsin(r/R)
    return pi*R*delta_w*(Nr-1)*cos(alpha)/(D*(pi/2+alpha))  


def cos_lambda_scaling(g1, w, D, Np, Nr, Nc):
    '''
    Scales each row of the projections with cos(lambda) = D/sqrt(D^2 + w^2). 
    w is coordinate of curved detector in the direction of rotation-axis, 
    therefore, it is constant for each row.
    Input: g1 - projections
           w -  coordinate of curved detector pixel along rotation axis
           Np - number of projections
           Nr - number of detector rows
           Nc - number of detector columns
    Output: g1 - cos(lambda) scaled projection data   
    '''
    g2 = np.zeros((Np-1, Nr, Nc-1), dtype=TYPE)
    for i in range(Nr):  
        g2[:,i,:] = g1[:,i,:]*D/sqrt((D**2+w[Nr-1-i]**2))
    return g2


def cos_alpha_scaling(g5, alpha, Np, Nr, Nc):
    '''
    Scales each column of the projections with cos(alpha). 
    Alpha is angular coordinate of curved detector, therefore, 
    it is constant for each column.
    Input: g5 - projections
           alpha - angular coordinate of curved detector pixel
           Np - number of projections
           Nr - number of detector rows
           Nc - number of detector columns
    Output: g6 - cos(alpha) scaled projection data   
    '''
    g6 = np.zeros((Np-1, Nr, Nc), dtype=TYPE)
    for p in range(Np-1):
        for i in range(Nr):
            g6[p,i,:] = np.cos(alpha)*g5[p,i,:]
    return g6    


def Hilbert_filter(Ns):
    '''
    Returns samples of the Hilbert kernel hH(sin(gamma))
    Input: Ns - number of samples
    '''
    if np.mod(Ns,2)==0:
        sys.exit('''\nError: Ns should be odd ''')
    
    h1 = np.zeros((1,Ns), dtype = TYPE)
    for i in np.arange(Ns):
        if np.mod(i-(Ns-1)/2.0-0.5,2) != 0:
            temp = pi*(i-(Ns-1)/2.0-0.5)
            h1[0,i] = (1-cos(temp))/temp
    return h1


def Hanning_sin(Ns):
    '''
    Returns samples of the Hanning window
    Input: Ns - number of samples
    '''  
    h1 = np.zeros((1,Ns), dtype = TYPE)
    for i in np.arange(Ns):
        h1[0,i] = sin(pi*(i-(Ns-1)/2.0-0.5)/(Ns-1))**2
    return h1


def filter_klines(klines, Np, Nk, Nc):
    '''
    Returns filtered k-lines
    Input: klines - k-lines for one source point position
           Np - number of projections
           Nk - number of k-lines
           Nc - number of detector columns
    Each k-line is filtered with Hilbert filter h(sin(alpha)). Filtering has
    been performed in Fourier domain.Processing in Fourier domain 
    requires zeropadding.  
    '''
    
    g4 = np.zeros((Np-1, Nk, Nc-1), dtype=TYPE)
    # convolution parameters
    L = 2*Nc-1+Nc-1-1 # size of convolution result
    L = nextpow2(L)   # zeropadding to length 2^(n)>L
    Nzs = L-(Nc-1)    # number of zeros to add into signal
    Nzf = L-(2*Nc-1)  # number of zeros to add into filter

    # compute Hilbert filter samples, same number of samples as a k-line  
    h = Hilbert_filter(2*Nc-1) 
    
    # zero padding filter
    hpad = np.concatenate((np.zeros((1,Nzf/2.0+1)), h, \
                           np.zeros((1,Nzf/2.0))), axis=1) 
    
    # compute Fourier transform of the filter
    # FT of Hilbert filter is windowed with Hanning window
    fth = ft.fftshift(ft.fft(ft.ifftshift(hpad)) * Hanning_sin(L))
    
    # zero padding klines
    klpad = np.concatenate((np.zeros((Np-1,Nk,Nzs/2.0+1)), klines, \
                            np.zeros((Np-1,Nk,Nzs/2.0))), axis=2)

    # compute Fourier transform for the k-lines
    ftkl = ft.fftshift(ft.fft(ft.ifftshift(klpad, axes=2), axis=2), axes=2)

    # filtering in Fourier domain (multiplication of FFTs)
    ftfiltkl = np.repeat(np.repeat(fth, Nk, axis=0),Np-1, axis=0).\
                         reshape(Np-1, Nk, L) * ftkl
    
    # compute inverse Fourier transform 
    filtkl = np.real(ft.fftshift(ft.ifft(ft.ifftshift(ftfiltkl, axes=2), \
                     axis=2), axes=2))

    #return Nc-1 samples    
    g4 = filtkl[:,:,(floor(L/2.0)-Nc/2.0+1):(floor(L/2.0)+Nc/2.0+1-1)]
    return g4


def chain_rule_derivation(Df, alpha, s, delta_alpha, ds, Np, Nr, Nc):
    '''
    Computes partial derivative of projection data Df(y(s), theta(s,x, gama))
    with respect to angular parameter s of the source trajectory, while keeping
    fixed direction theta.       
    Input: Df - projection data
           alpha - array of angular coordinates for curved detector
           s - array of source trajectory parameter values 
           delta_alpha - stepsize of angular parameter of curved detector
           ds - stepsize of angular parameter of source trajectory  
           Np - number of projections
           Nr - number of detector rows
           Nc - number of detector columns
    Output: g1 - derivative of projection data 
    '''
    alpha = alpha + delta_alpha/2.0 # shift by half a sample in alpha
    s = s + ds/2.0                  # shift by half a sample in s 
    g1 = np.zeros((Np-1,Nr,Nc-1), dtype=TYPE)
    g1 = (Df[1:,:,:-1] - Df[:-1,:,:-1])/(2.0*ds) \
       + (Df[1:,:,1:] - Df[:-1,:,1:])/(2.0*ds) \
       + (Df[:-1,:,1:] - Df[:-1,:,:-1])/(2.0*delta_alpha) \
       + (Df[1:,:,1:] - Df[1:,:,:-1])/(2.0*delta_alpha) 
    return g1, alpha, s


def proj2klines(g2, psi, alpha, w, D, P, R, Np, Nc, Nr, Nk):
    '''
    Projection samples are placed on the cylindrical grid with spacing delta_w  
    and arc_alpha. Filtering is performed along lines where k-plane 
    intersects detector - k-lines. Proj2klines computes values for samples on 
    k-lines at the center of detector columns. Simple linear intepolation in w 
    direction.
    Input: g2 - projections (s, w, alpha)
           psi - angles which definine k-planes(s,psi)
           alpha - angles which define the center of projection columns
           w - z-axis distance for the center of projection rows
           D - distance source-detector
           P - pitch of source trajectory
           R - radius of source trajectory
           Np - number of projections
           Nc - number of detector columns
           Nr - number of detector rows
           Nk - number of k-lines
    Output: k-lines (s, psi, alpha)              
    '''
    g3 = np.zeros((Np-1, Nk, Nc-1), dtype=TYPE)
    wk = np.zeros((Nk, Nc-1), dtype=TYPE)
    for kl in range(Nk):
        for j in range(Nc-1):     
            wk[Nk-1-kl,j] = k_line_point_w(D, P, R, psi[kl], alpha[j])
            w1i = floor((wk[Nk-1-kl,j]-w[0])/delta_w)
            w2i = w1i+1
            t = (wk[Nk-1-kl,j]-w[w1i])/delta_w
            if w2i<0 or w2i>=Nr:
                g3[:,Nk-1-kl,j] = (1-t) * g2[:,Nr-1-w1i,j]
            else:
                g3[:,Nk-1-kl,j] = (1-t) * g2[:,Nr-1-w1i,j] + \
                                      t * g2[:,Nr-1-w2i,j] 
    return g3,wk


def klines2proj(g4, wk, w, delta_w, Np, Nc, Nr, Nk):
    '''
    Samples are placed on the k-lines.It is needed to set values to pixels
    on cylindrical grid with spacings delta_w and delta_alpha. Inverse problem 
    of proj2klines. 
    Input: g4 - filtered klines (s, psi, alpha)
           wk - z-axis distance for the k-line samples
           w - z-axis distance for the center of projection rows
           Np - number of projections
           Nc - number of detector columns
           Nr - number of detector rows
           Nk - number of k-lines
    '''
    g5 = np.zeros((Np-1, Nr, Nc), dtype=TYPE)
    mask = np.zeros((Nr, Nc-1), dtype=TYPE)
    for kl in range(Nk-1):
        for j in range(Nc-1):
            w1i = np.max([0, floor((wk[Nk-1-kl,j]-w[0])/delta_w)])
            w2i = w1i + 1 
            if w2i <= Nr-1:
                c = (w[w1i]-wk[Nk-1-kl,j])/(wk[Nk-1-kl-1,j]-wk[Nk-1-kl,j])   
                d = (w[w2i]-wk[Nk-1-kl,j])/(wk[Nk-1-kl-1,j]-wk[Nk-1-kl,j]) 
                if mask[Nr-1-w1i,j] == 0 and mask[Nr-1-w2i,j] == 0:
                    g5[:,Nr-1-w1i,j] = (c)*g4[:,Nk-1-kl-1,j] + \
                                       (1-c)*g4[:,Nk-1-kl,j]                   
                    g5[:,Nr-1-w2i,j] = (d)*g4[:,Nk-1-kl-1,j] + \
                                       (1-d)*g4[:,Nk-1-kl,j]
                    mask[Nr-1-w1i,j] = 1
                    mask[Nr-1-w2i,j] = 1
                elif mask[Nr-1-w2i,j] == 0:                                 
                    g5[:,Nr-1-w2i,j] = (d)*g4[:,Nk-1-kl-1,j] + \
                                       (1-d)*g4[:,Nk-1-kl,j]
                    mask[Nr-1-w2i,j] = 1
            else:
                c = (w[w1i]-wk[Nk-1-kl,j])/(wk[Nk-1-kl-1,j]-wk[Nk-1-kl,j])
                if mask[Nr-1-w1i,j] == 0:
                    g5[:,Nr-1-w1i,j] = (c)*g4[:,Nk-1-kl-1,j] + \
                                       (1-c)*g4[:,Nk-1-kl,j]
                    mask[Nr-1-w1i,j] = 1
    return g5

def PIpoint(x3, r, gamma, R, h, sb):
    '''
    Solving nonlinear function f=0 gives solution for sb
    Input: x3 - z-direction coordinate of point x in 3D space
           r - radial cylindrical coordinate of point 
           gamma - angular cylindrical coordinate of point x  
                   x = [r*cos(gamma), r*sin(gamma),x3]
           h - pitch of the spiral source trajectory 
           sb - bottom source point point parameter of PI interval
    '''
    ang = gamma-sb
    f = h *((pi - 2.0*atan(r*sin(ang) / (R - r*cos(ang)))) * \
           (1 + (r**2 - R**2) / (2*R*(R-r*cos(ang)))) + sb) - x3
    return f

def PIinterval(x, R, h):
    '''
    Computation of PI line interval points for a single point in ROI space
    Input: x - point in 3D space for which PI interval is to be determined
           R - radius of the spiral source trajectory
           h - pitch of the spiral source trajectory
    Output: sb - bottom source point parameter of PI interval
            st - top source point parameter of PI interval 
    '''
    gamma = atan2(x[1],x[0])
    r = sqrt(x[0]**2+x[1]**2)
    
    f = partial(PIpoint, x[2], r, gamma, R, h)
    sb = brentq(f, x[2]/h-pi, x[2]/h)
    
    alphax = atan(r*sin(gamma-sb) / (R - r*cos(gamma-sb)))
    st = sb+pi-2*alphax
    return sb, st


def PIinterval_range(xc, yc, x3, Nx, Ny, R, h, PIUseFile, PIFileName):
    '''Computation of the PI-interval start and end point for each point in the 
       slice. Voxels at the same x,y position, but in diferent slices will have
       diferent PI interval ranges.Thus, you can reuse computed values 
       (use values stored in the file) only if you are reconstructing 
       same slice.   
       Input: xc, yc - x and y coordinates of voxels in the reconstructed slice
              Nx, Ny - number of voxels in reconstructed slice
              R - radius of source trajectory
              h - pitch divided with 2pi
              PIUseFile - are pre-computed values for PI interval of 
                          reconstructed slice in a file
              PIFileName - file name 
       Output: PI - array of start and end point of PI interval for each point 
                    in the slice
    '''
    if PIUseFile:
        temp = loadmat(PIFileName)
        PI = temp['PI'];
    else:       
        PI = np.zeros((Nx,Ny,2)) 
        for row in range(Nx):
            for col in range(Ny):
                x = [xc[row], yc[col], x3]
                if (x[0]**2 + x[1]**2) < r**2:
                    PI[row,col,:] = PIinterval(x, R, h)
        savemat(PIFileName, dict(PI=PI)) 
    return PI      
    

def backprojection(g6, s, xc, yc, x3, Nx, Ny, ds, r, D, shift_detector, 
                    delta_alpha, delta_w, alpha, w, PI):
    '''Computation of reconstructed slice voxel value by backprojecting 
       filtered and weighted projection values.  
       Input: g6 - filtered projection data
              s - array of source trajectory parameter values 
              xc, yc - x and y coordinates of voxels in the reconstructed slice
              x3  - z coordinate of voxels in the reconstructed slice
              Nx, Ny - number of voxels in reconstructed slice
              ds - stepsize of angular parameter of source trajectory
              r - radius of ROI in the slice
              D - distance source-detector
              shift_detector - shift between source point projection and center
                               of detector [alpha_shift, w_shift]
              delta_alpha - stepsize of angular parameter of curved detector
              delta_w -
              alpha - array of angular coordinates for curved detector
              w - z-axis distance for the center of projection rows
              PI - Pi interval range for each point in the slice
       Output: rec_slice - reconstructed slice
    '''

    cossp = np.cos(s)
    sinsp = np.sin(s)

    #backprojection
    rec_slice = np.zeros((Nx,Ny), dtype=TYPE)
    for row in range(Nx):
        if (row % 50)==0:     
            print 'Reconstructing slice row', row
        for col in range(Ny):
            if (xc[row]**2 + yc[col]**2) < r**2:
               # PI interval for reconstruction voxel x
               [sb, st] = PI[row,col,:]
               if sb < s[0] or st >s[-1] or sb == st:
                  continue
               # PI interval projection indices
               sbi = int(floor((sb-s[0])/ds))
               sti = int(ceil((st-s[0])/ds))
               # weighting 1/abs(x-y(s))
               vstar = R-xc[row]*cossp[sbi:sti+1]-yc[col]*sinsp[sbi:sti+1]
                
               # x point projection coordinates (alpha_star, w_star)
               alphastar = np.arctan((-xc[row]*sinsp[sbi:sti+1] + \
                                       yc[col]*cossp[sbi:sti+1])/vstar)
               wstar = D * np.cos(alphastar)/vstar * (x3-ys[2,sbi:sti+1]) 
                
               # nearest neighbour coordinates
               ai = ((alphastar-alpha[0]+shift_detector[0])/delta_alpha)
               ai = (np.round(ai)).astype(np.int) 
               wi = (wstar-w[0]+shift_detector[1])/delta_w               
               wi = Nr-1-np.round(wi).astype(np.int)
       
               #scaling projection values
               Nrho = sti-sbi+1
               rho = np.zeros((Nrho,))
               din = (s[sbi:sti+1]-sb)/ds
               dout = (st-s[sbi:sti+1])/ds
               if any((s[sbi:sti+1] > (sb-ds)) * \
                      (s[sbi:sti+1] < sb)):
                   ind = np.where((s[sbi:sti+1] > (sb-ds)) * \
                                  (s[sbi:sti+1] < sb))
                   rho[ind] = (1+din[ind])**2/2.0
               if any((s[sbi:sti+1] > sb) * (s[sbi:sti+1] < (sb+ds))):
                   ind = np.where((s[sbi:sti+1] > sb) * \
                                  (s[sbi:sti+1] < (sb+ds)))
                   rho[ind] = 0.5+din[ind]-din[ind]**2/2.0
               if any((s[sbi:sti+1] > (sb+ds)) * (s[sbi:sti+1] < (st-ds))):
                   ind = np.where((s[sbi:sti+1] > (sb+ds)) * \
                                  (s[sbi:sti+1] < (st-ds)))
                   rho[ind] = 1.0
               if any((s[sbi:sti+1] > (st-ds)) * (s[sbi:sti+1] < st)):
                   ind = np.where((s[sbi:sti+1] > (st-ds)) * \
                                  (s[sbi:sti+1] < st))
                   rho[ind] = 0.5+dout[ind]-dout[ind]**2/2.0
               if any((s[sbi:sti+1] > st) * (s[sbi:sti+1] < (st+ds))):
                   ind = np.where((s[sbi:sti+1] > st) * \
                                  (s[sbi:sti+1] < (st+ds)))
                   rho[ind] = (1+dout[ind])**2/2.0
               # contribution of point x to reconstruction voxel x
               f =   np.sum(ds/(2*pi) * rho * \
                           np.diag(g6[sbi:sti+1, wi, ai])/vstar)                                    
               rec_slice[row, col] += f
    return rec_slice


def absorption_reconstruction(Df, x3, args):
    '''
    Computes reconstruction of single slice at z = x3 from projections Df 
    collected using scanning geometry/setup parameters given in args.
    Input: Df - projection measurements data
           x3 - z-coordinate of reconstructed slice voxels
           args - scanning geometry parametes 
    Output: rec_slice - reconstructed slice  
    '''
    PIUseFile = True
    PIFileName = 'PIfile.mat' 
   
    # geometry parameters
    R = args.R                   # source trajectory radius
    D = args.D                   # curved detector radius
    H = args.H                   # curved detector height 
    Nr = args.Nr                 # curved detector number rows 
    Nc = args.Nc                 # curved detector number columns  
    r = args.r                   # radius of a circle containing object
    Nx = args.Nx                 # number of columns in rec. object image
    Ny = args.Ny                 # number of rows in rec. object image
    delta_w = H/Nr               # detector element height
    arc_alpha = delta_w          # detector element width (arc lenght)
    delta_alpha = arc_alpha/D    # angular size of detector element
    shift_detector = [delta_alpha/2.0, 0] # shift between source point
                                 # projection and center of detector, 
                                 # [alpha_shift, w_shift]
    halfAngle = delta_alpha*Nc/2 # cone beam half angle
    rm = R * sin(halfAngle)      # max. radius of reconstructable FOV 
    halfAngleFOV = np.arcsin(r/R)# half angle of FOV    

    if D != 2*R:
        sys.exit('''\nError: Curved detector radius should be '''\
                +'''initialized to two times source trajectory '''\
                +'''radius!\n''')
    if np.mod(Nr,2):
        sys.exit('''\nError: Number of detector rows was taken'''\
                +''' to be even''')
    if np.mod(Nc,2):
        sys.exit('''\nError: Number of detector columns was taken'''\
                +''' to be even''')
    if r>rm or r<=0:
        sys.exit('''\nError: Radius of ROI is too big or''' \
               + ''' angular extent of detector is too small.''')
      
    # source trajectory parameters
    Pmax = maxPitch(D, Nr, R, r, delta_w) # maximal pitch
    P =  Pmax                             # args.P
    h = P/(2*pi)    
    smin = x3/h - pi                      # minimal sb 
    smax = x3/h + 2*pi                    # maximal st
    ds = arc_alpha                        # source point stepsize 
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
    # w - z-axis (roation axis) sampling on detector grid
    w = np.linspace(-delta_w/2*(Nr-1), delta_w/2*(Nr-1), Nr)\
            - shift_detector[1]
   
    # reconstruction slice grid - voxel centers for the object in ROI
    dx = 2.0*r/Nx
    dy = 2.0*r/Ny
    # reconstructed slice voxel positions
    xc = np.linspace(-dx/2*(Nx-1), dx/2*(Nx-1), Nx) 
    yc = np.linspace(-dy/2*(Ny-1), dy/2*(Ny-1), Ny) 

    # psi is angle used to define k-planes(s,psi)
    # k-plane contains points ys(s), ys(s+psi) and ys(s+2*psi)
    Nkmin = number_k_lines(halfAngleFOV, Nr, P, Pmax)
    Nk = np.round(Nkmin).astype(np.int) #4*Nr
    delta_psi = (pi + 2*halfAngleFOV) / Nk
    min_psi = -pi/2-halfAngleFOV
    max_psi = pi/2+halfAngleFOV
    psi = np.arange(min_psi, max_psi, delta_psi)

    #------------------------------------------------------------------------# 
    # absorbtion reconstruction pipeline 
    g1, alpha, s = chain_rule_derivation(Df, alpha, s, delta_alpha, ds, 
                                         Np, Nr, Nc)              
    g2 = cos_lambda_scaling(g1, w, D, Np, Nr, Nc)
    g3, wk = proj2klines(g2, psi, alpha, w, D, P, R, Np, Nc, Nr, Nk)
    g4 = filter_klines(g3, Np, Nk, Nc)          
    g5 = klines2proj(g4, wk, w, delta_w, Np, Nc, Nr, Nk)
    g6 = cos_alpha_scaling(g5, alpha, Np, Nr, Nc)
    PI = PIinterval_range(xc, yc, Nx, x3, Ny, R, h, PIUseFile, PIFileName)
    rec_slice = backprojection(g6, s, xc, yc, x3, Nx, Ny, ds, r, D, 
                                shift_detector, delta_alpha, delta_w, alpha,
                                w, PI) 
    #------------------------------------------------------------------------# 
              
    return rec_slice  


def PIinterval_Katsevich(x, s, ys, cossp, sinsp, psi, D, P, R, ds, \
                          alphamax, wmax, Np):
    '''Not used'''
    rin = np.Inf
    rout = np.Inf
    flagst = True
    flagsb = True
    sb = np.Inf
    st = np.Inf
    
    for p in range(1,Np):
        # detector coordinate system
        v = R-x[0]*cossp[p]-x[1]*sinsp[p]
        alphax = atan((-x[0]*sinsp[p]+x[1]*cossp[p])/v)
        wx = D * cos(alphax)/v * (x[2]-ys[2,p]) 
        if np.abs(alphax) > np.abs(alphamax) or np.abs(wx) > np.abs(wmax):
            continue
       
        #alphai_near = np.argmin(np.abs(alpha-alphax))
        wtop_near = k_line_point_w(D, P, R, psi[Nk-1], alphax)
        wbottom_near = k_line_point_w(D, P, R, psi[0], alphax)

        if np.abs(rin) < np.abs(wtop_near - wx) and flagsb:          
            rin_d = wtop_near - wx;
            sb = s[p-1]-rin*ds/(rin_d-rin)
            flagsb = False
            continue
    
        if np.abs(rout) < np.abs(wbottom_near - wx) and flagst:       
            rout_d = wbottom_near-wx; 
            st = s[p-1] - rout*ds/(rout_d-rout)         
            flagst = False     
            continue
        
        if flagsb == False and flagst == False:
           break
    
        rin = wtop_near - wx;
        rout = wbottom_near - wx;
    return sb, st


if __name__ == '__main__':

    # Get arguments
    args = getArgs()

    t = time()
    projUseFile = True
    projFileName = 'projection_data.mat'
    PIUseFile = True
    PIFileName = 'PIfile.mat'
    phantomUseFile = True
    phantomFileName = 'phantom.mat'
   
    # initialize geometry parameters
    conf = init_scan_geometry(args)
    # object parameters
    obj = {}
    obj = init_obj(obj, plot=False)
    
    # reconstructed slice position
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
    shift_detector = [delta_alpha/2.0, 0] # shift between source point
                                 # projection and center of detector, 
                                 # [alpha_shift, w_shift]
    halfAngle = delta_alpha*Nc/2 # cone beam half angle
    rm = R * sin(halfAngle)      # max. radius of reconstructable FOV 
    halfAngleFOV = np.arcsin(r/R)# half angle of FOV    

    if D != 2*R:
        sys.exit('''\nError: Curved detector radius should be '''\
                +'''initialized to two times source trajectory '''\
                +'''radius!\n''')
    if np.mod(Nr,2):
        sys.exit('''\nError: Number of detector rows was taken'''\
                +''' to be even''')
    if np.mod(Nc,2):
        sys.exit('''\nError: Number of detector columns was taken'''\
                +''' to be even''')
    if r>rm or r<=0:
        sys.exit('''\nError: Radius of ROI is too big.''')
      
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

    #########################################################################
    # 2D forward projection
    tcp = time()
    if projUseFile == True:
        temp = loadmat(projFileName)
        Df = temp['Df']
    else:
        Df = cone_beam_projection(obj, conf, ys, s, shift_detector, False)
        savemat(projFileName, dict(Df=Df, M=Nr,\
            N=Nc, r=r,R=R, D=D, height=H, delta_w= delta_w, \
            delta_alpha=delta_alpha, P=P, h=h, MX=Nx, MY=Ny, delta_s=ds,\
            Z=x3, K=Np))
    tcp = time()-tcp
    print 'Computing projections time: ', tcp

    ########################### Reconstruction ###############################

    sys.exit()
    
    #Computation of derivatives
    tcp = time()
    g1, alpha, s = chain_rule_derivation(Df, alpha, s, delta_alpha, ds,
                                         Np, Nr, Nc) 
    tcp = time()-tcp 
    print 'Computing derivatives time: ', tcp

    '''plt.figure()
    plt.imshow(g1[20,:,:],  cmap=plt.cm.gray, interpolation='none')
    plt.title('Derivative')
    plt.colorbar()
    plt.show()'''
              
    savemat('derivative.mat', dict(der=g1, alpha=alpha, s=s))
    #------------------------------------------------------------------------#   
    
    #cos(lambda) scaling
    tcp = time()
    g2 = cos_lambda_scaling(g1, w, D, Np, Nr, Nc)
    tcp = time()-tcp
    print 'cos(lambda scaling) time: ', tcp

    '''plt.figure()
    plt.imshow(g2[20,:,:],  cmap=plt.cm.gray, interpolation='none')
    plt.title('Cos(lambda scaling)')
    plt.colorbar()
    plt.show()'''
 
    savemat('g2.mat', dict(scg2=g2))
    #------------------------------------------------------------------------#

    #forward height rebinning
    '''
    Samples are placed on the rectangular grid with spacing delta_w  
    and arc_alpha. Filtering is performed along lines where k-plane 
    intersects detector. Forward height rebinning does simple linear 
    interpolation in w direction. Interpolated samples are not equidistantly 
    spaced on k-line.
     
    '''
    tcp = time()
    # psi is angle used to define k-planes(s,psi)
    # k-plane contains points ys(s), ys(s+psi) and ys(s+2*psi)
    Nkmin = number_k_lines(halfAngleFOV, Nr, P, Pmax)
    Nk = np.round(Nkmin).astype(np.int)
    delta_psi = (pi + 2*halfAngleFOV) / Nk
    min_psi = -pi/2-halfAngleFOV
    max_psi = pi/2+halfAngleFOV
    psi = np.arange(min_psi, max_psi, delta_psi)

    # g2(s,w,alpha) -> g3(s,psi,alpha)
    g3, wk = proj2klines(g2, psi, alpha, w, D, P, R, Np, Nc, Nr, Nk)
    tcp = time()-tcp
    print 'Forward height rebinning time: ', tcp

    '''plt.figure()
    plt.imshow(g3[20,:,:],  cmap=plt.cm.gray, interpolation='none')
    plt.title('Forward height rebinning')
    plt.colorbar()
    plt.show()'''

    savemat('g3.mat', dict(fhr=g3, wk=wk))
    #------------------------------------------------------------------------#
    
    #Hilbert transform kernel filtering 
    tcp = time()
    g4 = filter_klines(g3, Np, Nk, Nc)    
    tcp = time()-tcp
    print 'Filtering time: ', tcp
        
    '''plt.figure()
    plt.imshow(g4[20,:,:],  cmap=plt.cm.gray, interpolation='none')
    plt.title('Hilbert filtered projection')
    plt.colorbar()
    plt.show()'''

    savemat('g4.mat', dict(fg4=g4))
    #------------------------------------------------------------------------#

    #backward height rebinning
    
    '''
    Samples are placed on the k-lines and we need to set values to pixels
    on rectangular grid with spacings delta_w and delta_alpha. Inverse problem 
    of forward height rebinning   
    '''
    tcp = time()
    #g4(s,psi,alpha) -> g5(s,w,alpha)
    g5 = klines2proj(g4, wk, w, delta_w, Np, Nc, Nr, Nk)
    tcp = time()-tcp
    print 'Backward height rebinning time: ', tcp

    '''plt.figure()
    plt.imshow(g5[20,:,:],  cmap=plt.cm.gray, interpolation='none')
    plt.title('Backward height rebinning')
    plt.colorbar()
    plt.show()'''

    savemat('g5.mat', dict(bhr=g5))
    #------------------------------------------------------------------------#
    
    #cos(alpha) scaling
    tcp = time()
    g6 = cos_alpha_scaling(g5, alpha, Np, Nr, Nc)
    tcp = time()-tcp
    print 'Cosine alpha scaling time: ', tcp

    '''plt.figure()
    plt.imshow(g6[20,:,:],  cmap=plt.cm.gray, interpolation='none')
    plt.title('Cosine (alpha) scaling')
    plt.colorbar()
    plt.show()'''

    savemat('g6.mat', dict(scg6=g6))
    #------------------------------------------------------------------------#
    
    # reconstruction grid - voxel centers for the object in ROI
    dx = 2.0*r/Nx
    dy = 2.0*r/Ny
    xc = np.linspace(-dx/2*(Nx-1), dx/2*(Nx-1), Nx) 
    yc = np.linspace(-dy/2*(Ny-1), dy/2*(Ny-1), Ny) 

    tcp = time()
    PI = PIinterval_range(xc, yc, Nx, x3, Ny, R, h, PIUseFile, PIFileName)
    tcp = time()-tcp
    print 'PI-interval computation time: ', tcp

    tcp = time()
    rec_slice = backprojection(g6, s, xc, yc, x3, Nx, Ny, ds, r, D, 
                                shift_detector, delta_alpha, delta_w, alpha,
                                w, PI)
    tcp = time()-tcp
    print 'Backprojection time: ', tcp
    t = time()-t
    print 'Total time: ', t

    
    plt.figure()
    extent = [yc[0], yc[-1], xc[0], xc[-1]]
    res = plt.imshow(rec_slice, cmap=plt.cm.gray, extent=extent, \
                     interpolation='none', origin='lower')
    res.set_clim(0.0, 0.07)
    plt.title('Reconstructed slice')
    plt.xlabel('y')
    plt.ylabel('x')
    plt.colorbar()
    plt.show()
    
    savemat('reconstructed_slice_rho.mat', dict(res_rho = rec_slice))    
              
    
    #------------------------------------------------------------------------# 
    ################# Single command reconstruction ##########################
    

    #pr = cProfile.Profile()
    #pr.enable()
    rec_slice1 = absorption_reconstruction(Df, x3, args)
    #pr.disable()
    #sm = StringIO.StringIO()
    #sortby = 'cumulative'
    #ps = pstats.Stats(pr, stream=sm).sort_stats(sortby)
    #ps.print_stats()
    #print sm.getvalue()

    plt.figure()
    extent = [yc[0], yc[-1], xc[0], xc[-1]]
    res = plt.imshow(rec_slice, cmap=plt.cm.gray, extent=extent, \
                     interpolation='none', origin='lower')
    res.set_clim(0.0, 0.07)
    plt.title('Reconstructed slice')
    plt.xlabel('y')
    plt.ylabel('x')
    plt.colorbar()
    plt.show()

    #------------------------------------------------------------------------#
    # phantom slice
    if phantomUseFile:
        temp = loadmat(phantomFileName)
        phantom = temp['phantom']
    else:
        phantom = np.zeros((Nx,Ny), dtype=TYPE)
        for row in np.arange(Nx):
            if np.mod(row,50)==0:     
                print 'Row', row
            for col in np.arange(Ny):
                x = [xc[row], yc[col], x3]
                if (xc[row]**2 + yc[col]**2) < r**2:
                    phantom[row,col] = reconstruction_phantom(obj, x, False)

    plt.figure()
    extent = [yc[0], yc[-1], xc[0], xc[-1]]
    res = plt.imshow(phantom, cmap=plt.cm.gray, extent=extent, \
                     interpolation='none', origin='lower')
    res.set_clim(0.0, 0.07)
    plt.title('Phantom')
    plt.xlabel('y')
    plt.ylabel('x')
    plt.colorbar()
    plt.show()
                
    savemat('phantom.mat', dict(phantom = phantom))  

    plt.figure()
    plt. plot(rec_slice1[:,138])
    plt.hold(True)
    plt.plot(phantom[:,138])
    plt.title('Slice cross section in y direction')
    plt.ylabel('Rec. Att. Coef. Value')
    plt.xlabel('y')
    plt.show()                
    

    #------------------------------------------------------------------------#
    # 'PI Line interval computation'
    '''temp = loadmat('PIfile.mat');
    PI = temp['PI']
    temp = loadmat('PIfile_k.mat')
    PIk = temp['PI']
    mask_dif = np.zeros((Nx,Ny))
    for row in np.arange(Nx):
        if np.mod(row,50)==0:     
            print 'Row', row
        for col in np.arange(Ny):
            sbd = PI[row,col,0]-PIk[row,col,0]
            std = PI[row,col,1]-PIk[row,col,1]
            if np.abs(sbd)>ds or np.abs(std)>ds:
                mask_dif[row,col] = np.max(np.abs([sbd,std]))/ds
            if PI[row,col,0]<s[0] or PI[row,col,1]>s[-1]:
                print 'a',(row,col, PI[row,col,0], PI[row,col,1])
            if PIk[row,col,0]<s[0] or PIk[row,col,1]>s[-1]:
                print 'k', (row,col, PIk[row,col,0], PIk[row,col,1])
            
    plt.figure()
    res = plt.imshow(mask_dif, cmap=plt.cm.gray, interpolation='none', \
                     origin='lower')
    plt.title('PI interval computation diff.')
    plt.xlabel('y')
    plt.ylabel('x')
    plt.colorbar()
    plt.show()


    mask_difPI = np.zeros((Nx,Ny))
    mask_difPIk = np.zeros((Nx,Ny))
    for row in np.arange(Nx):
        if np.mod(row,50)==0:     
            print 'Row', row
        for col in np.arange(Ny):
            PId = PI[row,col,1]-PI[row,col,0]
            PIkd = PIk[row,col,1]-PIk[row,col,0]
            mask_difPI[row,col] = PId/ds
            mask_difPIk[row,col] = PIkd/ds
            
    plt.figure()
    res = plt.imshow(mask_difPI, cmap=plt.cm.gray, interpolation='none', \
                     origin='lower')
    plt.title('Number of projections PI.')
    plt.xlabel('y')
    plt.ylabel('x')
    plt.colorbar()
    plt.show()

    plt.figure()
    res = plt.imshow(mask_difPIk, cmap=plt.cm.gray, interpolation='none', \
                     origin='lower')
    plt.title('Number of projections PIk.')
    plt.xlabel('y')
    plt.ylabel('x')
    plt.colorbar()
    plt.show()'''



              
    

