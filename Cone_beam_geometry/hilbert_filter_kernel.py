'''
Comparison of filtering and Hanning windowing in time domain and freq. domain
'''
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import sys
import numpy as np
from numpy import linalg as la
from scipy.io import savemat, loadmat
from init_v2 import *
from init_obj import *
import numpy.fft as ft

TYPE = 'float64'

def Hilbert_filter(alpha, delta_alpha):
    '''
    Returns samples of the Hilbert kernel
    Input: alpha - positions of samples
           dela_alpha - sampling step
    '''
    Ns = len(alpha)
    if np.mod(Ns,2):
        sys.exit('''\nError: Ns should be even and equal to Nc''')
    
    # number of samples in the kernel = Ns-1 
    # choosen to match number of samples in k-lines
    h1 = np.zeros((1,Ns-1), dtype = TYPE)
    h2 = np.zeros((1,Ns-1), dtype = TYPE)
    for i in np.arange(Ns-1):
        if np.mod(i-(Ns-2)/2.0-0.5,2) != 0:
            temp = np.pi*(i-(Ns-2)/2.0-0.5)
            h2[0,i] = (1-np.cos(temp))/temp
            h1[0,i] = (1-np.cos(temp))/temp*alpha[i]/(np.sin(alpha[i]))
    return h1, h2

def Hilbert_filter_index(Nc, alpha):
    '''
    Returns samples of the Hilbert kernel
    Input: Nc - number of detector columns 
    '''
    if np.mod(Nc,2):
        sys.exit('''\nError: Nc should be even''')
    
    # number of samples in the kernel = Nc-1 
    # choosen to match number of samples in k-lines
    h1 = np.zeros((1,Nc-1), dtype = TYPE)
    for i in np.arange(Nc-1):
        if np.mod(i-(Nc-2)/2.0-0.5,2) != 0:
            temp = np.pi*(i-(Nc-2)/2.0-0.5)
            h1[0,i] = (1-np.cos(temp))/temp
    return h1


def my_hanning_cos(Ns):
    '''
    Returns samples of the Hanning window
    Input: Ns - number of samples
    '''  
    h1 = np.zeros((1,Ns), dtype = TYPE)
    for i in np.arange(Ns):
        h1[0,i] = np.cos(np.pi*(i-(Ns-1)/2.0-0.5)/(Ns-1))**2
    return h1

def my_hanning_sin(Ns):
    '''
    Returns samples of the Hanning window
    Input: Ns - number of samples
    '''  
    h1 = np.zeros((1,Ns), dtype = TYPE)
    for i in np.arange(Ns):
        h1[0,i] = np.sin(np.pi*(i-(Ns-1)/2.0-0.5)/(Ns-1))**2
    return h1

def nextpow2(i):
    n = 1
    while n < i: n *= 2
    return n


if __name__ == '__main__':
    H = 0.5
    Nr = 16    
    Nc = 138
    D = 6.0
    delta_alpha = H/Nr/D
    shift_detector = [0, 0.0]
    alpha = np.linspace(-delta_alpha/2*(Nc-1), delta_alpha/2*(Nc-1), Nc)\
            - shift_detector[0]
    alpha = alpha + delta_alpha*0.5

    
    h1,h2 = Hilbert_filter(alpha, delta_alpha)
    plt.figure()
    plt.plot(np.transpose(h2))
    plt.hold(True)
    plt.plot(np.transpose(h1), '.r')
    plt.legend(('Ideal Hilbert filter', 'Hilbert sin(alpha)'))
    plt.grid()
    plt.show()
    plt.title('Nc-1 samples')


    Ncd = 2*Nc-2
    alphad = np.linspace(-delta_alpha/2*(Ncd-1), delta_alpha/2*(Ncd-1), Ncd)\
            - shift_detector[0]
    alphad = alphad + delta_alpha*0.5
    h4,h5 = Hilbert_filter(alphad,delta_alpha)
    h3 = Hilbert_filter_index(Ncd, alphad)
    plt.figure()
    plt.plot(np.transpose(h3))
    plt.hold(True)
    plt.plot(np.transpose(h4), '.r')
    plt.legend(('Hf_index', 'Hf_alpha'))
    plt.grid()
    plt.show()
    plt.title('2Nc-1 samples')


    # windowing in Fourier domain
    fth3 = ft.fft(ft.ifftshift(h3))
    win = my_hanning_sin(Ncd-1)
    fth3win = fth3 * win
    plt.figure()
    plt.plot(np.transpose(np.abs(fth3)))
    plt.hold(True)
    plt.plot(np.transpose(np.abs(fth3win)), '.r')
    plt.legend(('original', 'windowed'))
    plt.grid()
    plt.show()
    plt.title('2Nc-1 samples Fourier transform')

    # FT-> time domain
    th3 = np.real(ft.fftshift(ft.ifft(fth3, axis=1), axes=1))
    th3win = np.real(ft.fftshift(ft.ifft(fth3win, axis=1), axes=1))
    plt.figure()
    plt.plot(np.transpose(th3))
    plt.hold(True)
    plt.plot(np.transpose(th3win), '.r')
    plt.legend(('original', 'windowed'))
    plt.grid()
    plt.show()
    plt.title('2Nc-1 samples time domain')

    # windowing in time domain
    win = np.real(ft.fftshift(ft.ifft(ft.ifftshift(my_hanning_cos(Ncd-1), \
                  axes=1), axis=1), axes=1))
    twinh3 = np.convolve(win[0], h3[0],'same')
    plt.figure()
    plt.plot(np.transpose(th3win))
    plt.hold(True)
    plt.plot(np.transpose(twinh3), 'r')
    plt.legend(('windowed in FT domain', 'Windowed in time domain'))
    plt.grid()
    plt.show()
    plt.title('2Nc-1 samples time domain')

    '''
    ####################Filtering in Fourier and time domain###################
    #signal
    s = np.sin(2*np.pi*alpha)*np.cos(40*np.pi*alpha)
    s = s[:-1].reshape((1,Nc-1))
    # size of convolution result    
    L = Nc-1+Nc-1-1
    Nz = Nc # even number 
    Nn = np.ceil((Nc-1)/2.0)
    spad = np.concatenate((np.zeros((1,np.floor((Nz+1)/2.0))), s, \
                           np.zeros((1,np.ceil((Nz+1)/2.0)))), axis=1)
    fts = ft.fftshift(ft.fft(ft.ifftshift(spad))) 

    #filter 
    h = h1[0].reshape(1,Nc-1) 
    hpad = np.concatenate((np.zeros((1,np.floor((Nz+1)/2.0))), h, \
                           np.zeros((1,np.ceil((Nz+1)/2.0)))), axis=1)
    fth = ft.fftshift(ft.fft(ft.ifftshift(hpad) * my_hanning_sin(Nz+Nc)))
    
    #filtering in Fourier domain
    filtfts = fth * fts
    ftfs = np.real(ft.fftshift(ft.ifft(ft.ifftshift(filtfts))))
    res = ftfs[0,np.floor((Nz+1)/2.0)-Nn:np.floor((Nz+1)/2.0)-2+L-1]
    res_valid = ftfs[0,(Nz+Nc)/2.0-Nn-1:(Nz+Nc)/2.0-Nn+Nc-1]

    #filtering in time domain
    tfs = np.convolve(twinh3,s[0],'same')
    tfs_valid = np.convolve(twinh3,s[0],'valid')
    plt.figure()
    plt.hold(True)
    plt.plot(np.transpose(res),'b')
    plt.plot(np.transpose(tfs), '.r')
    plt.legend(('filtered in Fourier domain', 'filtered in time domain'))
    plt.grid()
    plt.show()
    plt.title('Filtering comparison Nc-1+Nc-1-1 samples')

    plt.figure()
    plt.hold(True)
    plt.plot(np.transpose(res_valid),'b')
    plt.plot(np.transpose(tfs_valid), '.r')
    plt.legend(('filtered in Fourier domain', 'filtered in time domain'))
    plt.grid()
    plt.show()
    plt.title('Filtering comparison Nc-1 samples')
    '''
    #------------------------------------------------------------------------#    
    
    #filter is not zero padded (true samples instead)
    #signal
    s = np.sin(2*np.pi*alpha)*np.cos(40*np.pi*alpha)
    s = s[:-1].reshape((1,Nc-1))
     
    L = 2*Nc-1+Nc-1-1 # size of convolution result   
    L = nextpow2(L)   # make size be 2^(n) number
    Nzs = L-(Nc-1)    # number of zeros to add into signal
    Nzf = L-(2*Nc-1)  # number of zeros to add into filter
    
    #Zero padding signal
    spad = np.concatenate((np.zeros((1,Nzs/2.0+1)), s, np.zeros((1,Nzs/2.0))), axis=1)
    
    # Filter
    h = Hilbert_filter_index(2*Nc, alpha)
    # Zero padding filter    
    hpad = np.concatenate((np.zeros((1,Nzf/2.0+1)), h, np.zeros((1,Nzf/2.0))), axis=1)
    
    #Compute Fourier transform     
    fts = ft.fftshift(ft.fft(ft.ifftshift(spad))) 
    fth = ft.fftshift(ft.fft(ft.ifftshift(hpad))*my_hanning_sin(L)) 
    
    #filtering in Fourier domain
    filtfts = fth * fts
    ftfs = np.real(ft.fftshift(ft.ifft(ft.ifftshift(filtfts))))
    res2 = ftfs[0, (np.floor(L/2.0)-(Nc-1)):(np.floor(L/2.0)+(Nc-1)+1)]
    res2_valid = ftfs[0, (np.floor(L/2.0)-Nc/2.0):(np.floor(L/2.0)+Nc/2.0+1)]

    win = np.real(ft.fftshift(ft.ifft(ft.ifftshift(my_hanning_cos(2*Nc-1)))))
    twinh = np.convolve(win[0], h[0],'same')
    
    #filtering in time domain
    tfs = np.convolve(twinh,s[0],'same')
    tfs_valid = np.convolve(twinh,s[0],'valid')

    plt.figure()
    plt.hold(True)
    plt.plot(np.transpose(res2),'b')
    plt.plot(np.transpose(tfs), '.r')
    plt.legend(('filtered in Fourier domain', 'filtered in time domain'))
    plt.grid()
    plt.show()
    plt.title('Filtering comparison Nc-1+Nc-1-1 samples')

    plt.figure()
    plt.hold(True)
    plt.plot(np.transpose(res2_valid),'b')
    plt.plot(np.transpose(tfs_valid), '.r')
    plt.legend(('filtered in Fourier domain', 'filtered in time domain'))
    plt.grid()
    plt.show()
    plt.title('Filtering comparison Nc-1 samples')

    