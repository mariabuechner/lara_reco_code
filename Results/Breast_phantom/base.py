import matplotlib.pyplot as plt
import sys
import numpy as np
from numpy import linalg as la
from att_coef_object import * 
from forward_projection import *
from ray_tracing_Siddon_v4 import *
import numpy.fft as ft
from scipy.optimize import brentq
from functools import partial
from math import cos, sin, tan, atan, atan2, sqrt, floor, ceil, pi
from numpy import dot
import argparse
import os
import h5py
#import cProfile, pstats, StringIO

TYPE = 'float64'

def getArgs():
    parser = argparse.ArgumentParser(description = 
    '''Script performs reconstruction of object slice from spiral CT 
    projections. Projection collection is simulated using cone beam  
    geometry with curved/flat detector. ''')
    
    # detector related parameters
    parser.add_argument('-D', '--D', dest='D', type=np.float64, default=50.0,
                        help = '''Curved detector radius in [cm] or ''' 
                        + '''shortest distance from source to detector in ''' 
                        + ''' flat detector geometry ''')
    parser.add_argument('-H', '--H', dest='H', type=np.float64, default=200e-4,
                        help = 'Detector height in [cm]')
    parser.add_argument('-Nr', '--Nr', dest='Nr', type=np.int16, default=4,
                        help = 'Number of detector rows')   
    parser.add_argument('-Nc', '--Nc', dest='Nc', type=np.int16, default=3400,
                        help = 'Number of detector columns')
    parser.add_argument('-dew', '--dew', dest='dew', 
                        type=np.float64, default=50e-4,
                        help = '''Detector element width [cm]''')
    parser.add_argument('-deh', '--deh', dest='deh', 
                        type=np.float64, default=50e-4,
                        help = '''Detector element height [cm]''')
    parser.add_argument('-shiftd', '--shift_detector',
                        nargs='+', type=np.float,
                        default = [0.0, 0.0],
                        help=''' Position of the source point projection '''
                           + ''' onto the detector''')    
    parser.add_argument("-d", "--detector", type=str, default = 'curved',
                        help = 'Detector type: curved or flat')

    # source trajectory parameters
    parser.add_argument('-R', '--R', dest='R', type=np.float64, default=25.0,
                        help = 'Source trajectory radius in [cm]') 
    parser.add_argument('-Pfactor', '--Pfactor', dest='Pfactor', 
                        type=np.float64, default=0.5,
                        help = 'Proportion of maximal pitch, P = Pfactor*Pmax')
    parser.add_argument('-ds', '--ds', dest='ds', 
                        type=np.float64, default=50e-4/4.0,
                        help = 'Stepsize of source point parameter in [rad]')

    # reconstruction parameters
    parser.add_argument('-r', '--r', dest='r', type=np.float64, default=4.0,
                        help = '''ROI radius in [cm]''')
    parser.add_argument('-Nx', '--Nx', dest='Nx', type=np.int16, default=256,
                        help = 'Number of rows of reconstructed slice')   
    parser.add_argument('-Ny', '--Ny', dest='Ny', type=np.int16, default=256,
                        help = 'Number of columns of reconstructed slice')    
    parser.add_argument("-img", "--imaging", type=str, default = 'absorption',
                        help = 'imaging contrast type: absorption or phase')
    
    # precomputed files 
    parser.add_argument("-PIUsef", "--PIUseFile", type=bool, 
                        default = False,
                        help = 'Use precomputed PI line interval range')
    parser.add_argument("-PIf", "--PIFileName", type=str, 
                        default = 'PIfile.h5',
                        help = 'Name of the file with stored PI lines')
    parser.add_argument("-phaUsef", "--phantomUseFile", type=bool, 
                        default = False,
                        help = 'Use phantom data stored in a file')
    parser.add_argument("-phaf", "--phantomFileName", type=str, 
                        default = 'phantom_data/Phasephantom4.h5',
                        help = '''File name containing phantom data, ''' 
                           +'''needed for phase contrast forward projection''')
    parser.add_argument("-projUsef", "--projUseFile", type=bool, 
                        default = True,
                        help = 'Use projection data stored in a file')
    parser.add_argument("-projf", "--projFileName", type=str, 
                        default = 'analytical_absorption_4_3400_5284.h5',
                        help = 'File name for projection data')
    args = parser.parse_args()
    return args

def init_scan_geometry(args):
    settings = {}
    # source trajectory (helix) parameters 
    settings.update({'source_trajectory':  
                        {'radius' : args.R}})        

    # curved detector parameters
    settings.update({'detector': 
                        {'DSD': args.D,            # detector-source distance
                         'height':args.H,
                         'number_rows':args.Nr,
                         'number_columns':args.Nc,
                         'pixel_width': args.dew,
                         'pixel_height': args.deh}}) 

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


def k_line_point_w_flat_detector(D, P, R, psi, u):
    '''
    Computes w coordinate of a point with detector coordinate u on the k-line
    Input: D - radius of detector
           P - pitch of spiral source trajectory 
           R - radius of spiral source trajectory
           psi - angle defining k-plane
           u - flat detector coordinate in direction perpendicular to rotation
               axis 
    '''
    const = D*P / (2*pi*R)
    if np.abs(psi) < 1e-5:
        # psi = 0, psi/tan(psi) = 1
        return const * u/D
    else:
        return const * (psi + psi/tan(psi) * u/D) 

def number_k_lines(alpha_m, D, P, R, delta_w):
    '''
    Computes minimal number of k-lines for filtering step.(Curved detector)
    Selected number of k-lines assures that the k-lines at the half fan angle 
    have maximal spacing equal to the detector row thickness divided by 2.
    Input: alpha_m - beam half angle 
           D - distance from source to detector  
           P - used pitch value
           R - radius of source trajectory
           delta_w - detector element height
    Output: number of k-lines (even number) 
    '''
    N = ceil(D*P*(pi+2*alpha_m) / (2*pi*R*cos(alpha_m)*delta_w) * \
             (1+(pi/2+alpha_m)*tan(alpha_m)))
    if np.mod(N,2) == 0:
        return N
    else:
        return N+1

def number_k_lines_flat_detector(alpha_m, D, P, R, delta_w):
    '''
    Computes minimal number of k-lines for filtering step.(Flat detector)
    Selected number of k-lines assures that the k-lines at the half fan angle 
    have maximal spacing equal to the detector row thickness divided by 2.
    Input: alpha_m - beam half angle 
           D - distance from source to detector  
           P - used pitch value
           R - radius of source trajectory
           delta_w - detector element height
    Output: number of k-lines (even number) 
    '''
    N =ceil(D*P*(pi+2*alpha_m) / (pi*R*delta_w) * \
            (1+tan(alpha_m)**2 + (pi/2+alpha_m)*tan(alpha_m)/cos(alpha_m)**2))
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

def maxPitch_flat_detector(D, Nr, R, r, dw):
    '''
    Computes maximal pitch value for spiral source trajectory
    Input: D - radius of detector 
           Nr - number of rows
           R - radius of source trajectory
           r - radius of ROI
           dw - detector element height
    Output: maximal pitch value  
    '''
    u = D * tan(np.arcsin(r/R))
    return pi*R*D*dw*(Nr-1)/((u**2+D**2)*(pi/2+atan(u/D)))  


def cos_lambda_scaling(g1, w, D, Np, Nr, Nc):
    '''
    Scales each row of the projections with cos(lambda) = D/sqrt(D^2 + w^2). 
    w is coordinate of curved detector in the direction of rotation-axis, 
    therefore, it is constant for each row.
    Input: g1 - projections
           w -  coordinates of curved detector pixel along rotation axis
           Np - number of projections
           Nr - number of detector rows
           Nc - number of detector columns
    Output: g1 - cos(lambda) scaled projection data   
    '''
    g2 = np.zeros((Np-1, Nr, Nc-1), dtype=TYPE)
    for i in range(Nr):  
        g2[:,i,:] = g1[:,i,:]*D/sqrt((D**2+w[Nr-1-i]**2))
    return g2


def cos_lambda_scaling_flat_detector(g1, w, u, D, Np, Nr, Nc):
    '''
    Scales each row of the projections with 
    cos(lambda) = D/sqrt(D^2 + u^2 + w^2). 
    Input: g1 - projections
           w -  coordinates of curved detector pixel along rotation axis
           Np - number of projections
           Nr - number of detector rows
           Nc - number of detector columns
    Output: g1 - cos(lambda) scaled projection data   
    '''
    g2 = np.zeros((Np-1, Nr, Nc-1), dtype=TYPE)
    for i in range(1, Nr):  
        for j in range(Nc-1):
            g2[:,i,j] = g1[:,i,j]*D/sqrt((D**2 + u[j]**2 + w[Nr-1-i]**2))
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

def zeta_scaling(Df, s, alpha, delta_alpha, delta_w, R, D, P, Nr, Nc, \
                                                        shift_detector):
    '''
    2D refraction-angle projections get scaled with zeta (defined 
    in paper from Li et al. (equation 14.))
    Input:Df - 2D refraction-angle projections
          s - angular parameter of source trajectory
          delta_alpha - curved detector element angular size
          delta_w - curved detector element height
          R - source trajectory radius
          D - source-to-detector shortest distance
          Nr - number of columns
          Nc - number of rows
          shift_detector - coordinates of source point projection onto 
                           the detector             
    Output: g1 - scaled 2D refraction-angle projections  
    '''
    Np = len(s) # number of projections
    derys = np.array([[-R*sin(s[i]) for i in range(Np)], \
                      [R*cos(s[i]) for i in range(Np)],  \
                      [P/(2*pi) for i in range(Np)]], dtype=TYPE)

    g1 = np.zeros((Np,Nr,Nc), dtype=TYPE)
    for p in range(Np):
        print p
        # source point coordinate system
        eu = np.array([-sin(s[p]), cos(s[p]), 0], dtype=TYPE)
        ev = np.array([-cos(s[p]), -sin(s[p]), 0], dtype=TYPE)
        ez = np.array([0,0,1.0], dtype=TYPE)
        
        # theta - direction of all rays in the cone beam that hit detector   
        theta = cone_beam(delta_alpha, delta_w, D, Nr, Nc, shift_detector,\
                          eu, ev, ez)
        for i in range(Nr):
            for j in range(Nc): 
                rgl1 = np.array([theta[i,j,1], -theta[i,j,0], 0]) / \
                                (sqrt(theta[i,j,1]**2+theta[i,j,0]**2))
                gl2 = np.array([-sin(s[p]+alpha[j]), cos(s[p]+alpha[j]), 0])
                rgl2 = np.cross(theta[i,j,:], gl2)
                rgl1gl2 = dot(rgl1, rgl2)
                zeta = dot(derys[:,p],rgl1) - \
                       dot(derys[:,p],rgl2) * rgl1gl2
                zeta = zeta/(1 - rgl1gl2**2)
                g1[p,i,j] = zeta * Df[p,i,j]
    return g1


def zeta_scaling_flat_detector(Df, s, du, dw, R, D, P, Nr, Nc, shift_detector):
    '''
    2D refraction-angle projections get scaled with zeta (defined 
    in paper from Li et al. (equation 14.))
    Input:Df - 2D refraction-angle projections
          s - angular parameter of source trajectory
          du, dw - flat detector element width/height
          R - source trajectory radius
          D - source-to-detector shortest distance
          Nr - number of columns
          Nc - number of rows
          shift_detector - coordinates of source point projection onto 
                           the detector             
    Output: g1 - scaled 2D refraction-angle projections  
    '''
    Np = len(s) # number of projections
    derys = np.array([[-R*sin(s[i]) for i in range(Np)], \
                      [R*cos(s[i]) for i in range(Np)],  \
                      [P/(2*pi) for i in range(Np)]], dtype=TYPE)

    g1 = np.zeros((Np,Nr,Nc), dtype=TYPE)
    for p in range(Np):
        print p
        # source point coordinate system
        eu = np.array([-sin(s[p]), cos(s[p]), 0], dtype=TYPE)
        ev = np.array([-cos(s[p]), -sin(s[p]), 0], dtype=TYPE)
        ez = np.array([0,0,1.0], dtype=TYPE)
        
        # theta - direction of all rays in the cone beam that hit detector   
        theta = cone_beam_flat_detector(du, dw, D, Nr, Nc, shift_detector, \
                                        eu, ev, ez)

        for i in range(Nr):
            for j in range(Nc): 
                rgl1 = np.array([theta[i,j,1], -theta[i,j,0], 0]) / \
                                (sqrt(theta[i,j,1]**2+theta[i,j,0]**2))
                gl2 = np.array([-sin(s[p]), cos(s[p]), 0])
                rgl2 = np.cross(theta[i,j,:], gl2)
                rgl1gl2 = dot(rgl1, rgl2)
                zeta = dot(derys[:,p],rgl1) - \
                       dot(derys[:,p],rgl2) * rgl1gl2
                zeta = zeta/(1 - rgl1gl2**2)
                g1[p,i,j] = zeta * Df[p,i,j]
    return g1


def Hilbert_filter(Ns, shift_detector, sample_step):
    '''
    Returns samples of the Hilbert kernel.
    hH(sin(gamma)) same as hH(gamma) if halfAngleFOV/sin(halAngleFOV)-> 1. 
    Input: Ns - number of samples
    '''
    if np.mod(Ns,2)==0:
        sys.exit('''\nError: Ns should be odd ''')
    # +0.5 sample shift is coming from derivative 
    shift = - shift_detector[0]/sample_step + 0.5
    h1 = np.zeros((1,Ns), dtype = TYPE)
    for i in np.arange(Ns):
        if np.mod(i- Ns/2.0 + shift,2) != 0:
            temp = pi*(i- Ns/2.0 + shift)
            h1[0,i] = (1-cos(temp))/temp
    return h1


def Hanning_sin(Ns, shift_detector, sample_step):
    '''
    Returns samples of the Hanning window
    Input: Ns - number of samples
           shift_detector - shift between source point projection and center
                            of detector [alpha_shift/u_shift, w_shift]
           sample_step - distance between sample in k-line 
                     (delta_alpha for curved detector or du for flat detector)
    Output: Hanning window samples
    '''  
    shift = - shift_detector[0]/sample_step + 0.5
    h1 = np.zeros((1,Ns), dtype = TYPE)
    for i in np.arange(Ns):
        h1[0,i] = sin(pi*(i-Ns/2.0+shift)/(Ns-1))**2
    return h1

def zeropad(Ns, Nzs, shift_detector, sample_step):
    '''
       Computed the number of zeros which should be added on left and right 
       side of the signal.
       Input: Ns - signal size
              Nzs - number of zeros to  be added
              shift_detector - shift between source point projection and center
                               of detector [alpha_shift/u_shift, w_shift]
              sample_step - distance between sample in k-line 
                      (delta_alpha for curved detector or du for flat detector)
       Output: Nleft - number of zeros to add on the left side
               Nright - number of zeros to add on the right side  
    '''
    shift = - shift_detector[0]/sample_step + 0.5
    sample_grid = np.linspace(0,Ns-1,Ns)- Ns/2.0 + shift
    Npoz = sum(sample_grid>=0)
    Nneg = Ns-Npoz
    Ndiff = Npoz-Nneg
    Nleft = 0
    Nright = 0
    if Ndiff > 0:
      #add more zeros to left
      Nleft += Ndiff 
    else:
      #add more zeros to right side
      Ndiff = -Ndiff
      Nright += Ndiff 
    Nleft +=  (Nzs-Ndiff)/2.0
    Nright += + (Nzs-Ndiff)/2.0 
    return  Nleft, Nright


def filter_klines(klines, Np, Nk, Nc, shift_detector, sample_step):
    '''
    Returns filtered k-lines
    Input: klines - k-lines for one source point position
           Np - number of projections
           Nk - number of k-lines
           Nc - number of detector columns
           shift_detector - shift between source point projection and center
                            of detector [alpha_shift/u_shift, w_shift]
           sample_step - distance between sample in k-line 
                     (delta_alpha for curved detector or du for flat detector)
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
    h = Hilbert_filter(2*Nc-1, shift_detector, sample_step) 
    
    # zero padding filter
    Nleft, Nright = zeropad(2*Nc-1, Nzf, shift_detector, sample_step)
    hpad = np.concatenate((np.zeros((1, Nleft)), h, np.zeros((1, Nright))), \
                          axis=1) 
    
    # compute Fourier transform of the filter
    # FT of Hilbert filter is windowed with Hanning window
    fth = ft.fftshift(ft.fft(ft.ifftshift(hpad)) * \
                             Hanning_sin(L, shift_detector, sample_step))
    
    # zero padding klines
    Nleft, Nright = zeropad(Nc-1, Nzs, shift_detector, sample_step)
    klpad = np.concatenate((np.zeros((Np-1, Nk, Nleft)), klines, \
                            np.zeros((Np-1, Nk, Nright))), axis=2)

    # compute Fourier transform for the k-lines
    ftkl = ft.fftshift(ft.fft(ft.ifftshift(klpad, axes=2), axis=2), axes=2)

    # filtering in Fourier domain (multiplication of FFTs)
    ftfiltkl = np.repeat(np.repeat(fth, Nk, axis=0),Np-1, axis=0).\
                         reshape(Np-1, Nk, L) * ftkl
    
    # compute inverse Fourier transform 
    filtkl = np.real(ft.fftshift(ft.ifft(ft.ifftshift(ftfiltkl, axes=2), \
                     axis=2), axes=2))

    #return Nc-1 samples   
    g4 = filtkl[:, :, Nleft:(L-Nright)]
    return g4


def filter_klines_slow(klines, Np, Nk, Nc, shift_detector, sample_step):
    '''
    Returns filtered k-lines
    Input: klines - k-lines for one source point position
           Np - number of projections
           Nk - number of k-lines
           Nc - number of detector columns
           shift_detector - shift between source point projection and center
                            of detector [alpha_shift/u_shift, w_shift]
           sample_step - distance between sample in k-line 
                     (delta_alpha for curved detector or du for flat detector)
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
    h = Hilbert_filter(2*Nc-1, shift_detector, sample_step) 
    
    # zero padding filter
    Nleft, Nright = zeropad(2*Nc-1, Nzf, shift_detector, sample_step)
    hpad = np.concatenate((np.zeros((1, Nleft)), h, np.zeros((1, Nright))), \
                          axis=1) 
    
    # compute Fourier transform of the filter
    # FT of Hilbert filter is windowed with Hanning window
    fth = ft.fftshift(ft.fft(ft.ifftshift(hpad)) * \
                             Hanning_sin(L, shift_detector, sample_step))

    Nleft, Nright = zeropad(Nc-1, Nzs, shift_detector, sample_step)
    for p in range(Np-1):
        #print p
        # zero padding kline
        klpad = np.concatenate((np.zeros((Nk, Nleft)), klines[p,:,:], \
                                np.zeros((Nk, Nright))), axis=1)
    
        # compute Fourier transform for the k-line
        ftkl = ft.fftshift(ft.fft(ft.ifftshift(klpad, axes=1), axis=1), axes=1)
    
        # filtering in Fourier domain (multiplication of FFTs)
        ftfiltkl = np.repeat(fth, Nk, axis=0).reshape(Nk, L) * ftkl
        
        # compute inverse Fourier transform 
        filtkl = np.real(ft.fftshift(ft.ifft(ft.ifftshift(ftfiltkl, axes=1), \
                        axis=1), axes=1))
        #return Nc-1 samples   
        g4[p,:,:] = filtkl[:, Nleft:(L-Nright)]
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

def chain_rule_derivation_flat_detector(Df, u, w, s, du, dw, ds, Np, Nr, Nc, D):
    '''
    Computes partial derivative of projection data Df(y(s), theta(s,x, gama))
    with respect to angular parameter s of the source trajectory at
    fixed direction theta.       
    Input: Df - projection data
           u, w - pixel coordinates for flat detector 
           s - array of source trajectory parameter values 
           du - flat detector pixel width
           dw - flat detector pixel height
           ds - stepsize of angular parameter of source trajectory  
           Np - number of projections
           Nr - number of detector rows
           Nc - number of detector columns
           D - distance source-detector
    Output: g1 - derivative of projection data 
    '''
    u = u + du/2.0                  # shift by half a sample in u
    w = w + dw/2.0                  # shift by half a sample in w
    s = s + ds/2.0                  # shift by half a sample in s 
    g1 = np.zeros((Np-1, Nr, Nc-1), dtype=TYPE)
    pDfds = (Df[1:,:-1,:-1] - Df[:-1,:-1,:-1])/(4.0*ds) \
          + (Df[1:,:-1,1:] - Df[:-1,:-1,1:])/(4.0*ds)   \
          + (Df[1:,1:,1:] - Df[:-1,1:,1:])/(4.0*ds)     \
          + (Df[1:,1:,:-1] - Df[:-1,1:,:-1])/(4.0*ds) 

    pDfdw = (Df[1:,1:,:-1] - Df[1:,:-1,:-1])/(4.0*dw)   \
          + (Df[1:,1:,1:] - Df[1:,:-1,1:])/(4.0*dw)     \
          + (Df[:-1,1:,1:] - Df[:-1,:-1,1:])/(4.0*dw)   \
          + (Df[:-1,1:,:-1] - Df[:-1,:-1,:-1])/(4.0*dw) 

    pDfdu = (Df[1:,1:,1:] - Df[1:,1:,:-1])/(4.0*du)     \
          + (Df[1:,:-1,1:] - Df[1:,:-1,:-1])/(4.0*du)   \
          + (Df[:-1,1:,1:] - Df[:-1,1:,:-1])/(4.0*du)   \
          + (Df[:-1,:-1,1:] - Df[:-1,:-1,:-1])/(4.0*du) 

    for i in range(1,Nr):  
        for j in range(Nc-1):
            g1[:, i, j] = pDfds[:, i-1, j]  \
                          + (u[j]**2 + D**2)/D * pDfdu[:, i-1, j] \
                          + u[j] * w[Nr-1-i] / D * pDfdw[:,i-1,j]

    return g1, u, w, s


def proj2klines(g2, psi, alpha, w, delta_w, D, P, R, Np, Nc, Nr, Nk):
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
           delta_w - height of detector's pixel
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
            if wk[Nk-1-kl,j] < w[0] or wk[Nk-1-kl,j]>w[-1]:
                continue
            w2i = w1i+1
            t = (wk[Nk-1-kl,j]-w[w1i])/delta_w
            if w2i<=0 or w2i>=Nr:
                g3[:,Nk-1-kl,j] = (1-t) * g2[:,Nr-1-w1i,j]
            else:
                g3[:,Nk-1-kl,j] = (1-t) * g2[:,Nr-1-w1i,j] + \
                                      t * g2[:,Nr-1-w2i,j] 
    return g3,wk

def proj2klines_flat_detector(g2, psi, u, w, delta_w, D, P, R, 
                              Np, Nc, Nr, Nk):
    '''
    Projection samples are placed on the rectangular grid with spacing dw  
    and du. Filtering is performed along lines where k-plane 
    intersects detector - k-lines. Proj2klines computes values for samples on 
    k-lines at the center of detector columns. Simple linear intepolation in w 
    direction.
    Input: g2 - projections (s, w, u)
           psi - angles which definine k-planes(s,psi)
           u, w - pixel coordinates for flat detector 
           delta_w - height of detector's pixel
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
            wk[Nk-1-kl,j] = k_line_point_w_flat_detector(D, P, R, psi[kl],u[j])
            if wk[Nk-1-kl,j] < w[0] or wk[Nk-1-kl,j]>w[-1]:
                continue
            w1i = floor((wk[Nk-1-kl,j]-w[0])/delta_w)
            w2i = w1i+1
            t = (wk[Nk-1-kl,j]-w[w1i])/delta_w
            if w2i<=1 or w2i>=Nr-1:
                g3[:,Nk-1-kl,j] = (1-t) * g2[:,Nr-1-w1i,j]
            else:
                g3[:,Nk-1-kl,j] = (1-t) * g2[:,Nr-1-w1i,j] + \
                                      t * g2[:,Nr-1-w2i,j] 
    return g3,wk


def klines2proj(g4, wk, w, delta_w, Np, Nc, Nr, Nk):
    '''
    Samples are placed on the k-lines.It is needed to set values to pixels
    on grid with spacings delta_w/dw and delta_alpha/du. Inverse problem 
    of proj2klines. 
    Input: g4 - filtered klines (s, psi, alpha/u)
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
            if wk[Nk-1-kl,j] < w[0] or wk[Nk-1-kl,j]>w[-1]:
                continue
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


def PIinterval_range(xc, yc, x3, Nx, Ny, r, R, h, PIUseFile, PIFileName):
    '''Computation of the PI-interval start and end point for each point in the 
       slice. Voxels at the same x,y position, but in diferent slices will have
       diferent PI interval ranges.Thus, you can reuse computed values 
       (use values stored in the file) only if you are reconstructing 
       same slice.   
       Input: xc, yc - x and y coordinates of voxels in the reconstructed slice
              Nx, Ny - number of voxels in reconstructed slice
              r - radius of ROI 
              R - radius of source trajectory
              h - pitch divided with 2pi
              PIUseFile - are pre-computed values for PI interval of 
                          reconstructed slice in a file
              PIFileName - file name 
       Output: PI - array of start and end point of PI interval for each point 
                    in the slice
    '''
    if PIUseFile:
        PIf = h5py.File(PIFileName, 'r')  
        PI = PIf['PI'][:]
        PIf.close()
    else:       
        PI = np.zeros((Nx,Ny,2)) 
        for row in range(Nx):
            for col in range(Ny):
                x = [xc[row], yc[col], x3]
                if (x[0]**2 + x[1]**2) < r**2:
                    PI[row,col,:] = PIinterval(x, R, h)
        PIf = h5py.File(PIFileName, 'w')  
        PIf.create_dataset('PI', data=PI, compression='gzip', \
                                    compression_opts=9)
        PIf.close()  
    return PI      
    

def backprojection(g6, ys, s, xc, yc, x3, Nx, Ny, ds, r, R, D, shift_detector, 
                    delta_alpha, delta_w, alpha, w, Nr, PI):
    '''Computation of reconstructed slice voxel value by backprojecting 
       filtered and weighted projection values.  
       Input: g6 - filtered projection data
              ys - source trajectory points
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
              delta_w - curved detector element height
              alpha - array of angular coordinates for curved detector
              w - z-axis distance for the center of projection rows
              Nr - number of detector rows
              PI - Pi interval range for each point in the slice
       Output: rec_slice - reconstructed slice
    '''

    cossp = np.cos(s)
    sinsp = np.sin(s)

    correct_rounding = - np.mod([abs(shift_detector[0])/delta_alpha,
                                 abs(shift_detector[1])/delta_w],1.0)
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
               # weighting abs(x-y(s))
               vstar = R-xc[row]*cossp[sbi:sti+1]-yc[col]*sinsp[sbi:sti+1]
                
               # x point projection coordinates (alpha_star, w_star)
               alphastar = np.arctan((-xc[row]*sinsp[sbi:sti+1] + \
                                       yc[col]*cossp[sbi:sti+1])/vstar)
               wstar = D * np.cos(alphastar)/vstar * (x3-ys[2,sbi:sti+1]) 
                
               # nearest neighbour coordinates
               ai = ((alphastar-alpha[0])/delta_alpha) + correct_rounding[0]
               ai = (np.round(ai)).astype(np.int) 
               wi = (wstar-w[0])/delta_w + correct_rounding[1]               
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


def backprojection_flat_detector(g6, ys, s, xc, yc, x3, Nx, Ny, ds, r, R, D, 
                    shift_detector, du, dw, u, w, Nr, PI):
    '''Computation of reconstructed slice voxel value by backprojecting 
       filtered and weighted projection values.  
       Input: g6 - filtered projection data
              ys - source trajectory points
              s - array of source trajectory parameter values 
              xc, yc - x and y coordinates of voxels in the reconstructed slice
              x3  - z coordinate of voxels in the reconstructed slice
              Nx, Ny - number of voxels in reconstructed slice
              ds - stepsize of angular parameter of source trajectory
              r - radius of ROI in the slice
              R - radius of source trajectory
              D - distance source-detector
              shift_detector - shift between source point projection and center
                               of detector [alpha_shift, w_shift]
              du, dw - width/height of flat detector pixel 
              u, w - pixel coordinates for flat detector
              Nr - number of detector rows
              PI - Pi interval range for each point in the slice
       Output: rec_slice - reconstructed slice
    '''

    cossp = np.cos(s)
    sinsp = np.sin(s)

    # rounding correction is needed in case when shift detector has 
    # a multiple of du/2 and/or multiple of dw/2 because samples are 
    # positioned at non-intiger indices (e.g. -1.5,-0.5, 0.5, 1.5, 
    # ...) and rounding always overestimates index position
    correct_rounding = - np.mod([abs(shift_detector[0])/du, 
                                 abs(shift_detector[1])/dw],1.0)
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
               # weighting abs(x-y(s))
               vstar = R-xc[row]*cossp[sbi:sti+1]-yc[col]*sinsp[sbi:sti+1]
                
               # x point projection coordinates (alpha_star, w_star)
               ustar = D / vstar * (-xc[row]*sinsp[sbi:sti+1] + \
                                     yc[col]*cossp[sbi:sti+1]) 
               wstar = D / vstar * (x3-ys[2, sbi:sti+1]) 
                

               # nearest neighbour coordinates
               ui = ((ustar-u[0])/du) + correct_rounding[0]
               ui = (np.round(ui)).astype(np.int) 
               wi = (wstar-w[0])/dw + correct_rounding[1]            
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
                           np.diag(g6[sbi:sti+1, wi, ui])/vstar)                                    
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
    PIUseFile = args.PIUseFile
    PIFileName = args.PIFileName
   
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
    P =  args.Pfactor*Pmax                # args.P
    h = P/(2*pi)    
    smin = x3/h - pi                      # minimal sb 
    smax = x3/h + pi + 2*halfAngleFOV     # maximal st
    ds = args.ds                          # source point stepsize 
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
    Nkmin = number_k_lines(halfAngleFOV, D, P, R, delta_w)
    Nk = 4*np.round(Nkmin).astype(np.int)
    delta_psi = (pi + 2*halfAngleFOV) / Nk
    min_psi = -pi/2-halfAngleFOV
    max_psi = pi/2+halfAngleFOV
    psi = np.arange(min_psi, max_psi, delta_psi)


    #------------------------------------------------------------------------# 
    # absorbtion reconstruction pipeline 
    g1, alpha, s = chain_rule_derivation(Df, alpha, s, delta_alpha, ds, 
                                         Np, Nr, Nc)      
    g2 = cos_lambda_scaling(g1, w, D, Np, Nr, Nc)
    g3, wk = proj2klines(g2, psi, alpha, w, delta_w, D, P, R, Np, Nc, Nr, Nk)
    g4 = filter_klines_slow(g3, Np, Nk, Nc, shift_detector, delta_alpha)          
    g5 = klines2proj(g4, wk, w, delta_w, Np, Nc, Nr, Nk)
    g6 = cos_alpha_scaling(g5, alpha, Np, Nr, Nc)
    PI = PIinterval_range(xc, yc, x3, Nx, Ny, r, R, h, PIUseFile, PIFileName)
    rec_slice = backprojection(g6, ys, s, xc, yc, x3, Nx, Ny, ds, r, R, D, 
                                shift_detector, delta_alpha, delta_w, alpha,
                                w, Nr, PI) 
    #------------------------------------------------------------------------# 
              
    return rec_slice  


def absorption_reconstruction_flat_detector(Df, x3, args):
    '''
    Computes reconstruction of single slice at z = x3 from projections Df 
    collected using scanning geometry/setup parameters given in args 
    (flat detector).
    Input: Df - projection measurements data
           x3 - z-coordinate of reconstructed slice voxels
           args - scanning geometry parametes 
    Output: rec_slice - reconstructed slice  
    '''
    PIUseFile = args.PIUseFile
    PIFileName = args.PIFileName
   
    # geometry parameters
    R = args.R                   # source trajectory radius
    D = args.D                   # shortest distance flat detector-source 
    H = args.H                   # flat detector height 
    Nr = args.Nr                 # flat detector number rows 
    Nc = args.Nc                 # flat detector number columns  
    r = args.r                   # radius of a circle containing object
    Nx = args.Nx                 # number of columns in rec. object image
    Ny = args.Ny                 # number of rows in rec. object image
    dw = args.deh                # detector element height
    du = args.dew                # detector element width
    shift_detector = args.shift_detector  # shift between source point
                                 # projection and center of detector, 
                                 # [alpha_shift, w_shift]
    halfAngle = atan((du*Nc/2+du/2.0)/D) # cone beam half angle
    rm = R * sin(halfAngle)      # max. radius of reconstructable FOV 
    halfAngleFOV = np.arcsin(r/R)# half angle of FOV      

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
    Pmax = maxPitch_flat_detector(D, Nr, R, r, dw) # maximal pitch
    P =  args.Pfactor * Pmax                       # args.P
    h = P/(2*pi)    
    smin = x3/h - pi                      # minimal sb 
    smax = x3/h + pi + 2*halfAngleFOV     # maximal st
    ds = args.ds                          # source point stepsize 
    s = np.arange(smin,smax,ds) 
    if np.mod(len(s),2)!=0:
        s = np.arange(smin,smax+ds,ds)
    Np = len(s)                        # number of projections
    # source point positions
    ys = np.array([[R*cos(s[i]) for i in range(Np)], \
                   [R*sin(s[i]) for i in range(Np)], \
                   [P/(2*pi)*s[i] for i in range(Np)]], dtype=TYPE)

    # detector grid
    # u, w - detector grid coordinates    
    u = np.linspace(-du/2*(Nc-1), du/2*(Nc-1), Nc) - shift_detector[0]
    w = np.linspace(-dw/2*(Nr-1), dw/2*(Nr-1), Nr) - shift_detector[1]
   
    # reconstruction slice grid - voxel centers for the object in ROI
    dx = 2.0*r/Nx
    dy = 2.0*r/Ny
    # reconstructed slice voxel positions
    xc = np.linspace(-dx/2*(Nx-1), dx/2*(Nx-1), Nx) 
    yc = np.linspace(-dy/2*(Ny-1), dy/2*(Ny-1), Ny) 

    # psi is angle used to define k-planes(s,psi)
    # k-plane contains points ys(s), ys(s+psi) and ys(s+2*psi)
    Nkmin = number_k_lines_flat_detector(halfAngleFOV, D, P, R, delta_w)
    Nk = 4*np.round(Nkmin).astype(np.int)
    delta_psi = (pi + 2*halfAngleFOV) / Nk
    min_psi = -pi/2-halfAngleFOV
    max_psi = pi/2+halfAngleFOV
    psi = np.arange(min_psi, max_psi, delta_psi)

    #------------------------------------------------------------------------# 
    # absorbtion reconstruction pipeline 
    g1, u, w, s = chain_rule_derivation_flat_detector(Df, u, w, s, du, dw, ds, 
                                                      Np, Nr, Nc, D)            
    g2 = cos_lambda_scaling_flat_detector(g1, w, u, D, Np, Nr, Nc)
    g3, wk = proj2klines_flat_detector(g2, psi, u, w, dw, D, P, 
                                       R, Np, Nc, Nr, Nk)  
    g4 = filter_klines(g3, Np, Nk, Nc, shift_detector, du)   
    g5 = klines2proj(g4, wk, w, dw, Np, Nc, Nr, Nk)
    PI = PIinterval_range(xc, yc, x3, Nx, Ny, r, R, h, PIUseFile, PIFileName)
    rec_slice = backprojection_flat_detector(g5, ys, s, xc, yc, x3, Nx, Ny, ds, 
                                             r, R, D, shift_detector, du, dw, 
                                             u, w, Nr, PI) 

    #------------------------------------------------------------------------# 
    return rec_slice 


def phase_reconstruction(Df, x3, args):
    '''
    Computes reconstruction of single slice at z = x3 from projections Df 
    collected using scanning geometry/setup parameters given in args.
    Input: Df - projection measurements data (refraction angle data)
           x3 - z-coordinate of reconstructed slice voxels
           args - scanning geometry parametes 
    Output: rec_slice - reconstructed slice  
    '''
    PIUseFile = args.PIUseFile
    PIFileName = args.PIFileName
   
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
    P =  args.Pfactor*Pmax                # args.P 
    #P = 0.29066822283489996
    h = P/(2*pi)    
    smin = x3/h - pi                      # minimal sb 
    smax = x3/h + pi + 2*halfAngleFOV     # maximal st
    ds = args.ds                          # source point stepsize 
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
    Nkmin = number_k_lines(halfAngleFOV, D, P, R, delta_w)
    Nk = 4*np.round(Nkmin).astype(np.int)
    delta_psi = (pi + 2*halfAngleFOV) / Nk
    min_psi = -pi/2-halfAngleFOV
    max_psi = pi/2+halfAngleFOV
    psi = np.arange(min_psi, max_psi, delta_psi)


    #------------------------------------------------------------------------# 
    # phase reconstruction pipeline
    print 'Zeta_scaling' 
    g1 = zeta_scaling(Df, s, alpha, delta_alpha, delta_w, R, D, P, Nr, Nc, \
                                                               shift_detector)
    g1prime = g1[:-1,:,:-1]
    g2 = cos_lambda_scaling(g1prime, w, D, Np, Nr, Nc)
    g3, wk = proj2klines(g2, psi, alpha, w, delta_w, D, P, R, Np, Nc, Nr, Nk)
    g4 = filter_klines_slow(g3, Np, Nk, Nc, shift_detector, delta_alpha)          
    g5 = klines2proj(g4, wk, w, delta_w, Np, Nc, Nr, Nk)
    g6 = cos_alpha_scaling(g5, alpha, Np, Nr, Nc)
    PI = PIinterval_range(xc, yc, x3, Nx, Ny, r, R, h, PIUseFile, PIFileName)
    rec_slice = backprojection(g6, ys, s, xc, yc, x3, Nx, Ny, ds, r, R, D, 
                                shift_detector, delta_alpha, delta_w, alpha,
                                w, Nr, PI) 
    #------------------------------------------------------------------------# 
              
    return rec_slice 


def phase_reconstruction_flat_detector(Df, x3, args):
    '''
    Computes reconstruction of single slice at z = x3 from projections Df 
    collected using scanning geometry/setup parameters given in args.
    (flat detector)
    Input: Df - projection measurements data (refraction angle data)
           x3 - z-coordinate of reconstructed slice voxels
           args - scanning geometry parametes 
    Output: rec_slice - reconstructed slice  
    '''
    PIUseFile = args.PIUseFile
    PIFileName = args.PIFileName
   
    # geometry parameters
    R = args.R                   # source trajectory radius
    D = args.D                   # shortest distance flat detector-source 
    H = args.H                   # flat detector height 
    Nr = args.Nr                 # flat detector number rows 
    Nc = args.Nc                 # flat detector number columns  
    r = args.r                   # radius of a circle containing object
    Nx = args.Nx                 # number of columns in rec. object image
    Ny = args.Ny                 # number of rows in rec. object image
    dw = args.deh                # detector element height
    du = args.dew                # detector element width
    shift_detector = args.shift_detector  # shift between source point
                                 # projection and center of detector, 
                                 # [alpha_shift, w_shift]
    halfAngle = atan((du*Nc/2+du/2.0)/D) # cone beam half angle
    rm = R * sin(halfAngle)      # max. radius of reconstructable FOV 
    halfAngleFOV = np.arcsin(r/R)# half angle of FOV      

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
    Pmax = maxPitch_flat_detector(D, Nr, R, r, dw) # maximal pitch
    P =  args.Pfactor * Pmax                       # args.P
    h = P/(2*pi)    
    smin = x3/h - pi                      # minimal sb 
    smax = x3/h + pi + 2*halfAngleFOV     # maximal st
    ds = args.ds                          # source point stepsize 
    s = np.arange(smin,smax,ds) 
    if np.mod(len(s),2)!=0:
        s = np.arange(smin,smax+ds,ds)
    Np = len(s)                        # number of projections
    # source point positions
    ys = np.array([[R*cos(s[i]) for i in range(Np)], \
                   [R*sin(s[i]) for i in range(Np)], \
                   [P/(2*pi)*s[i] for i in range(Np)]], dtype=TYPE)

    # detector grid
    # u, w - detector grid coordinates    
    u = np.linspace(-du/2*(Nc-1), du/2*(Nc-1), Nc) - shift_detector[0]
    w = np.linspace(-dw/2*(Nr-1), dw/2*(Nr-1), Nr) - shift_detector[1]
   
    # reconstruction slice grid - voxel centers for the object in ROI
    dx = 2.0*r/Nx
    dy = 2.0*r/Ny
    # reconstructed slice voxel positions
    xc = np.linspace(-dx/2*(Nx-1), dx/2*(Nx-1), Nx) 
    yc = np.linspace(-dy/2*(Ny-1), dy/2*(Ny-1), Ny) 

    # psi is angle used to define k-planes(s,psi)
    # k-plane contains points ys(s), ys(s+psi) and ys(s+2*psi)
    Nkmin = number_k_lines_flat_detector(halfAngleFOV, D, P, R, dw)
    Nk = 4*np.round(Nkmin).astype(np.int)
    delta_psi = (pi + 2*halfAngleFOV) / Nk
    min_psi = -pi/2-halfAngleFOV
    max_psi = pi/2+halfAngleFOV
    psi = np.arange(min_psi, max_psi, delta_psi)


    #------------------------------------------------------------------------# 
    # phase reconstruction pipeline 
    g1 = zeta_scaling_flat_detector(Df, s, du, dw, R, D, P, Nr, Nc, \
                                    shift_detector)
    g1prime = g1[:-1,:,:-1]
    g2 = cos_lambda_scaling_flat_detector(g1prime, w, u, D, Np, Nr, Nc)
    g3, wk = proj2klines_flat_detector(g2, psi, u, w, dw, D, P, 
                                       R, Np, Nc, Nr, Nk)
    g4 = filter_klines_slow(g3, Np, Nk, Nc, shift_detector, du)          
    g5 = klines2proj(g4, wk, w, dw, Np, Nc, Nr, Nk)
    PI = PIinterval_range(xc, yc, x3, Nx, Ny, r, R, h, PIUseFile, PIFileName)
    rec_slice = backprojection_flat_detector(g5, ys, s, xc, yc, x3, Nx, Ny, ds, 
                                             r, R, D, shift_detector, du, dw, 
                                             u, w, Nr, PI) 
    #------------------------------------------------------------------------# 
    return rec_slice  


def forward_projections(x3, args, dpc):
    '''
    Analytical computation of forward projection data -> absorption 
    Ray tracing using Siddon algorithm -> phase constrast
    '''
    projUseFile = args.projUseFile
    projFileName = args.projFileName
    phantomFileName = args.phantomFileName
   
    if projUseFile: 
        print 'Loading projection data from', projFileName
        projf = h5py.File(projFileName, 'r')  
        Df = projf['Df'][:]
        projf.close()
    else:
        # initialize geometry parameters
        conf = init_scan_geometry(args)
        # object parameters
        obj = {}
        obj = init_obj(obj, plot=False)
        
        # geometry parameters
        R = args.R                   # source trajectory radius
        D = args.D                   # curved detector radius
        H = args.H                   # curved detector height 
        Nr = args.Nr                 # curved detector number rows 
        Nc = args.Nc                 # curved detector number columns  
        r = args.r                   # radius of a circle containing object
        Nx = args.Nx                 # number of columns in rec. object image
        Ny = args.Ny                 # number of rows in rec. object image
        dw = args.deh                # detector element height
        arc_alpha = args.dew         # detector element width (curved detector)
        du = args.dew                # detector element width (flat detector)
        da = arc_alpha/D             # angular size of detector element
        shift_detector = args.shift_detector    # shift between source point
                                     # projection and center of detector, 
                                     # [alpha_shift, w_shift]
        halfAngle = da*Nc/2 # cone beam half angle
        rm = R * sin(halfAngle)      # max. radius of reconstructable FOV 
        halfAngleFOV = np.arcsin(r/R)# half angle of FOV    
    
        if D != 2*R and args.detector == 'curved':
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
        if args.detector == 'curved':
            Pmax = maxPitch(D, Nr, R, r, dw)               # maximal pitch
        elif args.detector == 'flat':
            Pmax = maxPitch_flat_detector(D, Nr, R, r, dw) # maximal pitch
        else:
            sys.exit('\nError: Unknown detector type')
   
        P =  args.Pfactor*Pmax                # args.P
        #P = 0.29066822283489996
        h = P/(2.0*pi)    
        smin = x3/h - pi                      # minimal sb 
        smax = x3/h + pi + 2*halfAngleFOV     # maximal st
        ds = args.ds                          # source point stepsize 
        s = np.arange(smin,smax,ds) 
        if np.mod(len(s),2)!=0:
            s = np.arange(smin,smax+ds,ds)
        Np = len(s)                           # number of projections
        # source point positions
        ys = np.array([[R*cos(s[i]) for i in range(Np)], \
                       [R*sin(s[i]) for i in range(Np)], \
                       [P/(2*pi)*s[i] for i in range(Np)]], dtype=TYPE)
   
        # 2D forward projections 
        if args.imaging == 'absorption':
            if args.detector == 'curved':
                Df = cone_beam_projection(obj, conf, ys, s, 
                                          shift_detector, False)
            elif args.detector == 'flat':
                Df = cone_beam_projection_flat_detector(obj, conf, ys, s,
                                                        shift_detector, False)
            else:
                sys.exit('\nError: Unknown detector type') 
        elif args.imaging == 'phase':
            print "Slow generation of phase projections.Use c code."
            phantomf = h5py.File(phantomFileName, 'r') 
            #phantomp = phantomf['phantom'][:]
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
            if args.detector == 'curved':
                alpha = np.linspace(-da/2*(Nc-1), da/2*(Nc-1), Nc) \
                                    - shift_detector[0]
                w = np.linspace(-dw/2*(Nr-1), dw/2*(Nr-1), Nr) \
                                    - shift_detector[1]
                print Nr, Nc, r, R, D, H, dxp, dyp, dzp, smin, smax, Np, ds
                sys.exit()
                Df = siddon_cone_beam_projection(phantomFileName, ys, s, 
                           alpha, w, Nr, Nc, r, R, D, H, Nxp, Nyp, Nzp,
                           dxp, dyp, dzp, bxp, byp, bzp, True)
            elif args.detector == 'flat':
                u = np.linspace(-du/2*(Nc-1), du/2*(Nc-1), Nc) \
                                    - shift_detector[0]
                w = np.linspace(-dw/2*(Nr-1), dw/2*(Nr-1), Nr) \
                                    - shift_detector[1]
                Df = siddon_cone_beam_projection_flat_detector(phantomFileName,
                           ys, s, u, w, Nr, Nc, r, R, D, H, Nxp, Nyp, Nzp, 
                           dxp, dyp, dzp, bxp, byp, bzp, True)
            else:
                sys.exit('\nError: Unknown detector type')            

        projf = h5py.File(projFileName, 'w')  
        projf.create_dataset('Df', data=Df, compression='gzip', \
                                            compression_opts=9)
        projf.close()      
    return Df


if __name__ == '__main__':

    # Get arguments
    args = getArgs()

    # reconstructed slice position
    x3 = 0

    '''#create directory to store results
    wpath = os.getcwd()
    res_dir = 'projection_curved_00'
    res_dir_path = wpath + "\\" + res_dir
    if not os.path.isdir(res_dir_path):
        try:
            os.makedirs(res_dir_path)
        except OSError:
            pass
    args.rdp = res_dir_path '''

    if args.imaging == 'absorption':
        # 2D forward projection for absorption 
        Df = forward_projections(x3, args, False)
        '''plt.figure()
        res = plt.imshow(Df[0,:,:], cmap=plt.cm.gray, \
                            interpolation='none')
        plt.title('Projection')
        plt.xlabel('y')
        plt.ylabel('x')
        plt.colorbar()
        plt.show()'''

        if args.detector == 'curved': 
            # reconstructed slice z = x3
            print 'Reconstruction curved detector absorbtion'
            rec_slice = absorption_reconstruction(Df, x3, args)
        elif args.detector == 'flat':
            # reconstructed slice z = x3
            rec_slice = absorption_reconstruction_flat_detector(Df, x3, args)
        else:
            sys.exit('''\nError: Unknown detector type.''')

        plt.figure()
        extent = [-1, 1, -1, 1]
        res = plt.imshow(rec_slice, cmap=plt.cm.gray, extent=extent,\
                        interpolation='none')
        res.set_clim(0.0, 20e-12)
        plt.title('Reconstructed slice')
        plt.xlabel('y')
        plt.ylabel('x')
        plt.colorbar()
        plt.show()
        recFileName = 'rec_' + args.projFileName 
        recf = h5py.File(recFileName, 'w') 
        recf.create_dataset('rec_slice', data=rec_slice, compression='gzip', \
                                compression_opts=9)
        recf.close() 

    elif args.imaging == 'phase':
        # 2D forward projection for phase
        Df = forward_projections(x3, args, True)
        '''plt.figure()
        res = plt.imshow(Df[0,:,:], cmap=plt.cm.gray, \
                            interpolation='none')
        plt.title('Projection')
        plt.xlabel('y')
        plt.ylabel('x')
        plt.colorbar()
        plt.show()'''
        if args.detector == 'curved': 
            # reconstructed slice z = x3
            print 'Reconstruction curved detector phase'
            rec_slice = phase_reconstruction(Df, x3, args)
        elif args.detector == 'flat':
            print 'TO DO'
            sys.exit()
        else:
            sys.exit('''\nError: Unknown detector type.''')

        plt.figure()
        extent = [-args.r , args.r , -args.r , args.r]
        res = plt.imshow(rec_slice, cmap=plt.cm.gray, extent=extent,\
                        interpolation='none')
        plt.title('Reconstructed phase slice')
        plt.xlabel('y')
        plt.ylabel('x')
        plt.colorbar()
        plt.show()
        recFileName = 'rec_' + args.projFileName 
        recf = h5py.File(recFileName, 'w') 
        recf.create_dataset('rec_slice', data=rec_slice, compression='gzip', \
                                compression_opts=9)
        recf.close() 
    else:
        sys.exit('''\nError: Unknown imaging contrast.''')


    #------------------------------------------------------------------------#
    # phantom slice z = x3
    phantomUseFileSlice = False
    phantomFileNameSlice = 'phantom_data/'+ args.imaging + 'phantom_slice_256.h5'

    if phantomUseFileSlice:
        phantomf = h5py.File(phantomFileNameSlice, 'r')  
        phantom = phantomf['phantom'][:]
        phantomf.close() 
    else:
        # object parameters
        obj = {}
        obj = init_obj(obj, plot=False)

        r = args.r                   # radius of a circle containing object
        Nx = args.Nx                 # number of columns in rec. object image
        Ny = args.Ny                 # number of rows in rec. object image
        dpc = args.imaging == 'phase'# True - phase object 
                                     # False - absorption object
                                       
        # reconstruction slice grid - voxel centers for the object in ROI
        dx = 2.0*r/Nx
        dy = 2.0*r/Ny
        # voxel positions in reconstructed slice
        xc = np.linspace(-dx/2*(Nx-1), dx/2*(Nx-1), Nx) 
        yc = np.linspace(-dy/2*(Ny-1), dy/2*(Ny-1), Ny) 
       
        phantom = np.zeros((Nx,Ny), dtype=TYPE)
        for row in np.arange(Nx):
            if np.mod(row,50)==0:     
                print 'Row phantom', row
            for col in np.arange(Ny):
                x = [xc[row], yc[col], x3]
                if (xc[row]**2 + yc[col]**2) < r**2:
                    phantom[row,col] = reconstruct_phantom(obj, x, dpc)
        phantomf = h5py.File(phantomFileNameSlice, 'w')  
        phantomf.create_dataset('phantom', data=phantom, compression='gzip', \
                                    compression_opts=9)
        phantomf.close()  



    plt.figure()
    extent = [-args.r , args.r , -args.r , args.r ]
    res = plt.imshow(phantom, cmap=plt.cm.gray, extent=extent, \
                     interpolation='none')
    res.set_clim(0.0, 20e-12)
    plt.title('Phantom')
    plt.xlabel('y')
    plt.ylabel('x')
    plt.colorbar()
    plt.show()
                    

    plt.figure()
    plt. plot(rec_slice[:,139])
    plt.hold(True)
    plt.plot(phantom[:,139])
    plt.title('Slice cross section in y direction')
    plt.ylabel('Rec. Att. Coef. Value')
    plt.xlabel('y')
    plt.show()                
    



              
    

