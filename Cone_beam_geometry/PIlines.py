'''
Comparison of PI line start and end point computation using method proposed by 
Noo et all. and the Kayle-Champley method described in Wunderlich's paper. 
'''

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import sys
import numpy as np
from numpy import linalg as la
from scipy.io import savemat, loadmat
from init_v2 import *
from init_obj import *
from forward_projection import *
import numpy.fft as ft
from base import *
from scipy.optimize import *
from functools import partial
TYPE = 'float64'

def PIfun(x3, r, gamma, R, h, sb):
    ang = gamma-sb
    f = h *((np.pi - 2.0*np.arctan(r*np.sin(ang) / (R - r*np.cos(ang)))) * \
           (1 + (r**2 - R**2) / (2*R*(R-r*np.cos(ang)))) + sb) - x3
    return f

# reference coordiante system
i = np.array([1.0,0.0,0.0], dtype=TYPE)
j = np.array([0.0,1.0,0.0], dtype=TYPE)
k = np.array([0.0,0.0,1.0], dtype=TYPE)

# initialize geometry parameters
conf = {}
conf = init_v2(conf)
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
rm = R * np.sin(halfAngle)   # max. radius of reconstructable FOV 
halfAngleFOV = np.arcsin(r/R)# FOV defined half angle    

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
h = P/(2*np.pi)    
smin = x3/h - np.pi                   # minimal sb 
smax = x3/h + 2*np.pi                 # maximal st
ds = arc_alpha                        # source point stepsize 
s = np.arange(smin,smax,ds) 
if np.mod(len(s),2)!=0:
    s = np.arange(smin,smax+ds,ds)
ys = np.array([R*np.cos(s), R*np.sin(s), P/(2*np.pi)*s], dtype=TYPE)

# detector grid
# alpha - angular sampling on detector grid   
alpha = np.linspace(-delta_alpha/2*(Nc-1), delta_alpha/2*(Nc-1), Nc)\
        - shift_detector[0]
# w - z-axis sampling on detector grid
w = np.linspace(-delta_w/2*(Nr-1), delta_w/2*(Nr-1), Nr)\
        - shift_detector[1]
Np = len(s)

# klines
#alpha = alpha+delta_alpha/2
Nk = 4*Nr
delta_psi = (np.pi + 2*halfAngleFOV) / Nk
min_psi = -np.pi/2-halfAngleFOV
max_psi = np.pi/2+halfAngleFOV
psi = np.arange(min_psi, max_psi, delta_psi)

x = np.array([0.23046875, 0.93359375, -0.25])
#----------------------------------------------------------------------------#

# PI Interval computation based on Noo. et all 

rin = np.Inf
rout = np.Inf
flagst = True
flagsb = True
sb = np.Inf
st = np.Inf

for si in range(1,Np):
    # detector coordinate system
    eu = np.array([-np.sin(s[si]), np.cos(s[si]), 0], dtype=TYPE)
    ev = np.array([-np.cos(s[si]), -np.sin(s[si]), 0], dtype=TYPE)
    ez = np.array([0,0,1.0], dtype=TYPE)
    
    alphax = np.arctan(np.dot(x,eu)/(R+np.dot(x,ev))) 
    wx = D*np.cos(alphax)*np.dot((x-ys[:,si]),ez)/(R+np.dot(x,ev))
    if np.abs(alphax) > np.abs(alpha[0]-2*delta_alpha) or np.abs(wx) > np.abs(w[0]-2*delta_w):
       #print(alphax,wx)
       continue
    
    #alphai_near = np.argmin(np.abs(alpha-alphax))
    wtop_near = k_line_point_w(D, P, R, psi[Nk-1], alphax)
    wbottom_near = k_line_point_w(D, P, R, psi[0], alphax)
    
    if np.abs(rin) < np.abs(wtop_near - wx) and flagsb:          
        rin_d = wtop_near - wx;
        sb = s[si-1]-rin*ds/(rin_d-rin)
        
        flagsb = False
        #print (flagst, rin, rin_d, sb)
        continue

    if np.abs(rout) < np.abs(wbottom_near - wx) and flagst: 
        print (si, rout, wbottom_near - wx)          
        rout_d= wbottom_near-wx; 
        st = s[si-1]-rout*ds/(rout_d-rout)  
        flagst = False     
        #print (rout, rout_d, sb,st)
        continue
    
    if flagsb == False and flagst == False:
        break

    rin = wtop_near - wx;
    rout = wbottom_near - wx;

print (sb, st)
#----------------------------------------------------------------------------#
 

# PI Line computation based on Kyle Champley

h = P/(2*np.pi)
gamma = np.arctan2(x[1],x[0])
r = np.sqrt(x[0]**2+x[1]**2)

f = partial(PIfun, x[2], r, gamma, R, h)
sb2 = brentq(f, x[2]/h-np.pi, x[2]/h)

alphax = np.arctan(r*np.sin(gamma-sb2) / (R - r*np.cos(gamma-sb2)))
st2 = sb2+np.pi-2*alphax

print (sb2,st2)



