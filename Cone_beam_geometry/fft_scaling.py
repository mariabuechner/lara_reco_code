'''
Small example to find out which samples to take out after filtering in freq. 
domain to correspond to convolution done with 'same' flag.
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

TYPE = 'float64'

N = 6
klines = np.array([1,2,3,6,4,2]).reshape((1,N))
h = np.array([3,2,1,1,2,3]).reshape((1,N))
klines = klines[0,0:N-1]
h = h[0,0:N-1]
Nc = N-1
L = Nc+Nc-1
Nn = np.ceil(N/2.0)
Nz=4*N

kpad = np.concatenate((np.zeros((1,np.floor((Nz+1)/2.0)))[0], klines, np.zeros((1,np.ceil((Nz+1)/2.0)))[0]), axis=1)
hpad = np.concatenate((np.zeros((1,np.floor((Nz+1)/2.0)))[0], h, np.zeros((1,np.ceil((Nz+1)/2.0)))[0]), axis=1)

K = ft.fftshift(ft.fft(ft.ifftshift(kpad)))
H = ft.fftshift(ft.fft(ft.ifftshift(hpad)))

Y = K*H
y = np.real(ft.fftshift(ft.ifft(ft.ifftshift(Y))))
print y[np.floor((Nz+1)/2.0)-Nn:np.floor((Nz+1)/2.0)-2+L-1]
print y[(Nz+Nc)/2.0-Nn:(Nz+Nc)/2.0-Nn+Nc]

yc=np.convolve(klines,h)
print yc
ycv=np.convolve(klines,h,'same')
print ycv