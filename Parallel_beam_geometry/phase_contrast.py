import numpy as np
import numpy.fft as ft
import matplotlib.pyplot as plt
from skimage.transform import radon
from skimage.transform import iradon


TYPE = 'float64'


def zeropadding(data, dim, factor):
    '''
    Adding zeros along the dimension dim
    Input: data - array where zeros are added
           dim - can be 'x' (adding zeros in rows) or 'y' 
                 (adding zeros in columns)
           factor - multiplied with the dimension of array in which zeros will 
                    be added = number of zeros.     
    '''
    nx = np.size(data, 1)
    ny = np.size(data, 0)
    if dim == 'x':
        zpad = np.ceil(nx*factor/2)
        temp = np.zeros((ny, zpad), dtype=TYPE)
        data = np.concatenate((temp, data, temp), axis=1)
    elif dim == 'y':
        zpad = np.ceil(ny*factor/2)
        temp = np.zeros((zpad, nx), dtype=TYPE)
        data = np.concatenate((temp, data, temp), axis=0)
    else:
        print 'Wrong dimension setting: dim <- x or y'
    return data


def remove_zeropadding(data , dim, factor, Nt, Nr):
    '''
    Removing zeros along the dimension dim (used to revert zeropadding)
    Input: data - array where zeros are removed
           dim - can be 'x' (adding zeros in rows) or 'y' 
                 (adding zeros in columns)
           factor - should have same value as it was used for zeropadding     
    '''
    nx = np.size(data, 1)
    ny = np.size(data, 0)
    if dim == 'x':
        zpad = np.ceil((nx - np.float32(nx)/(factor+1))/2)
        data = data[:,zpad:zpad+Nr]
    elif dim == 'y':
        zpad = np.ceil((ny - np.float32(ny)/(factor+1))/2)
        data = data[zpad:zpad+Nt,:]
    else:
        print 'Wrong dimension setting: dim <- x or y'
    return data


def ramp_filter(Nr, dr):
    '''
    Returns samples of ramp filter
        h[0] = 1.0/(4*DR**2)
        h[n is even] = -1/(n*np.pi*DR)**2
        h[n is odd] = 0
    '''
    n = np.arange(-Nr/2, Nr/2, dtype=TYPE)
    h = np.zeros((1, Nr))
    for i in range(len(n)):
        if n[i] == 0.0:
            h[0,i] = 1.0/(4*dr**2)      
        elif np.mod(n[i], 2):
            h[0,i] = -1/(n[i]*np.pi*dr)**2
    return h


def dpc_filter(Nr, dr):
    '''
    Returns samples of dpc filter
    '''
    n = np.arange(-Nr/2, Nr/2, dtype=TYPE)
    h = np.zeros((1, Nr), dtype=TYPE)
    for k in range(len(n)):
        h[0,k] = n[k]/8.0/dr * np.sinc(np.pi/2*n[k])**2
    return h 


def dpc_filter_sinogram(sino, Nt, Nr, dr):
    '''
    Returns filtered sinogram
    Filter is defined in Fourier space.
    Input: sino - sinogram 
    '''
    # filter transfer function
    fth = -1j/2.0/np.pi * np.sign(np.arange(-Nr/2, Nr/2, dtype=TYPE))
    fth = fth.reshape((1,Nr))

    # compute Fourier transform
    ftsinopad = dr * ft.fft(ft.ifftshift(sino, axes=1), axis=1)
    ftsinopad = ft.fftshift(ftsinopad, axes=1)

    # compute Fourier transform of filtered sinogram
    ftfiltsino = np.repeat(fth, Nt, axis=0) * ftsinopad
    
    # compute Inverse Fourier transform of filteref sinogram
    ftfiltsino = ft.ifftshift(ftfiltsino, axes=1)
    filtsino = ft.fftshift(ft.ifft(ftfiltsino, axis=1),axes=1)/dr
    
    return np.real(filtsino)


def filter_sinogram(sino, h, Nt, Nr, dr):
    '''
    Returns filtered sinogram
    Input: sino - sinogram
           h - filter samples
    Filtering has been performed in Fourier domain.
    Processing in Fourier domain requires zeropadding.  
    '''
    # zeropadding
    hpad = zeropadding(h, 'x', 1)
    sinopad = zeropadding(sino, 'x', 1)

    # compute Fourier transforms
    fthpad = dr * ft.fft(ft.ifftshift(hpad))
    ftsinopad = dr * ft.fft(ft.ifftshift(sinopad, axes=1), axis=1)
    
    # compute Fourier transform of filtered sinogram
    ftfiltsino = np.repeat(fthpad, Nt, axis=0) * ftsinopad
    
    # compute nverse Fourier transform of filteref sinogram
    filtsino = np.real(ft.fftshift(ft.ifft(ftfiltsino, axis=1), axes=1))/dr
    
    #remove added zeros     
    filtsino = remove_zeropadding(filtsino, 'x', 1, Nt, Nr)
    return filtsino


def generate_circle(Nx, Ny, xc, yc, rad, att):
    ''' 
    Generetas simple phantom image (circular objet)
    Input: Nx - number of columns in the image, 
           Ny - number of rows in the image
           xc,yc - coordinates for the center of circular object, px
           rad - radius of the circular object, px 
           att - attenuation coefficient of the object
    '''
    obj_img = np.zeros((Ny,Nx), dtype=TYPE)
    i = np.arange(0,Nx)
    j = np.arange(0,Ny)
    ii, jj = np.meshgrid(i,j)
    pi, pj = np.where((ii-xc)**2+(jj-yc)**2 < rad**2)
    obj_img[pj,pi] = att
    return obj_img



if __name__ == '__main__':

    Nx = 64
    Ny = 64
    Nr = 64
    Nt = 64
    dx = 2
    dy = 2
    dr = 2

    '''obj_img = generate_circle(Nx, Ny, Nx/2, Ny/2, Nx/4, 1) + \
              generate_circle(Nx, Ny, 3*Nx/4, 3*Ny/4, Nx/8, 0.2) + \
              generate_circle(Nx, Ny, Nx/4, Ny/4, Nx/4, 0.5) + \
              generate_circle(Nx, Ny, 3*Nx/4, Ny/4, Nx/4, 0.75)'''
    obj_img = generate_circle(Nx, Ny, Nx/2, Ny/2, Nx/4, 1)

    # generate differential phase sinogram
    # 1. compute normal Radontransform
    # 2. compute derivative along the projection line 
    prj_ds = np.zeros((Nt,Nr), dtype=TYPE)
    thetas = np.arange(Nt) * 180.0/Nt;
    prj = np.transpose(radon(obj_img, theta=thetas, circle=True))
    prj_ds[:, 0:-1] = np.diff(prj)
    '''plt.figure()
    plt.imshow(prj, cmap=plt.cm.gray, interpolation='none')
    plt.title('Sinogram')
    plt.show()
    plt.figure()
    plt.imshow(prj_ds, cmap=plt.cm.gray, interpolation='none')
    plt.title('Differential phase sinogram')'''


    # filter dpc sinogram
    h = dpc_filter(Nr, dr)
    fprj_ds = dpc_filter_sinogram(prj_ds, Nt, Nr, dr)
    plt.figure()
    plt.imshow(fprj_ds, cmap=plt.cm.gray, interpolation='none')
    plt.title('Filtered differential phase sinogram (hkapa)')
    plt.show()

    # reconstruction using built-in inverse Radon transform
    thetas = np.arange(Nt) * 180.0/(Nt);
    irs = iradon(np.transpose(fprj_ds), theta=thetas, circle=True, filter=None)
    plt.figure()
    plt.imshow(irs, cmap=plt.cm.gray, interpolation='none')
    plt.title('Inverse Radon transform reconstruction (hkappa)')
    plt.show()
    
    fprj_ds = filter_sinogram(prj_ds, h, Nt, Nr, dr)
    plt.figure()
    plt.imshow(fprj_ds, cmap=plt.cm.gray, interpolation='none')
    plt.title('Filtered differential phase sinogram (h)')
    plt.show()

    # reconstruction using built-in inverse Radon transform
    thetas = np.arange(Nt) * 180.0/(Nt);
    irs = iradon(np.transpose(fprj_ds), theta=thetas, circle=True, filter=None)
    plt.figure()
    plt.imshow(irs, cmap=plt.cm.gray, interpolation='none')
    plt.title('Inverse Radon transform reconstruction (h)')
    plt.show()