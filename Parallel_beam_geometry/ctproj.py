import sys
import numpy as np
import numpy.fft as ft
import matplotlib.pyplot as plt
import scipy.ndimage as scpndi 
from scipy.interpolate import griddata


# Object parameters: 
CX = [0, 40, -75]   # x coordinate of center
CY = [0, 40, 0]     # y coordinate of center
R = [100, 25, 12]   # radius: R
ATT = [1, 2, 4]     # attenuation coeff. 

# Imaging parameters:
DR = 2              # distance between radial samples in cm
FOV = 256           # field of view in cm
NR = 128            # number of radial samples in a projection
NT = 100            # number of angles 
NX = 128            # number of columns in reconstructed image
NY = 128            # number of rows in reconstructed image

TYPE = 'float64'    # array's type


def projection(theta, obj_num, att):
    '''
    Returns projection values for a given theta. For a radial position 
    t, projection value is equal to the distance ray travels through the object
    times the attenuation coeff. of the object.
    Distance  = sqrt((x1-x2)**2 + (y1-y2)**2) where A(x1,y1) and B(x2,y2) 
                correspond to points on the object where the ray enters and 
                exits the object   
    The coordinates of this points are solution of equations:
    (1) t = x*sin(theta) + y*cos(theta) - ray 
    (2) (x-CX)**2 + (y-CY)**2 = R**2    - circle with center at (CX,CY)
    There are 2 special cases:
    (a) for rays almost parallel with y-axis (theta ~ pi/2)
        To compute the distance, find solution of:
        (1) t = y*sin(theta)
        (2) (x-CX)**2 + (y-CY)**2 = R**2
    (b) for rays almost parallel with x-axis (theta ~ pi or zero) 
        To compute the distance, find solution of:
        (1) t = y*sin(theta)
        (2) (x-CX)**2 + (y-CY)**2 = R**2
    '''
    prj = np.asarray([0]*NR, dtype=TYPE)
    temp = np.asarray([0]*NR, dtype=TYPE)
    t = np.linspace(-FOV/2+1, FOV/2, FOV/2) * DR/2
    
    if theta < 1e-3:
        #for rays almost parallel with y-axis 
        temp = R[obj_num]**2 - (t/np.cos(theta)-CX[obj_num])**2
    elif np.abs(theta-np.pi/2) < 1e-3:
        #for rays almost parallel with x-axis
        temp = R[obj_num]**2 - (t/np.sin(theta)-CY[obj_num])**2
    else:
        temp = (((t/np.cos(theta)-CX[obj_num])*np.tan(theta) + CY[obj_num])**2\
               - (1+np.tan(theta)**2) * ((t/np.cos(theta)-CX[obj_num])**2 + \
               CY[obj_num]**2 - R[obj_num]**2)) / (1+np.tan(theta)**2)

    for i in range(NR):
        if (temp[i] >= 0):
            prj[i] = 2*np.sqrt(temp[i])*att
        else:
            prj[i] = 0
    return prj


def sinogram():
    '''
    Computes sinogram of 3 circular shape objects. 
    '''    
    sin = np.zeros((NT, NR), dtype=TYPE)
    thetas = np.arange(NT) * np.pi/NT;
    for i in np.arange(NT):
        sin[i] = projection(thetas[i], 0, ATT[0]) + \
                 projection(thetas[i], 1, ATT[1]-ATT[0]) + \
                 projection(thetas[i], 2, ATT[2]-ATT[0]) 
    return sin


def backprojection(sino):
    '''
    Smearing of projections over the image space
    Input: sino - sinogram
    '''
    temp =  np.zeros((NX, NY), dtype=TYPE)
    thetas = np.arange(NT) * np.pi/NT;
    for i in np.arange(NT):
        temp[:,:] = np.pi/NT*sino[i]
        if i == 0:
            bp = temp        
        else:
            bp = bp + scpndi.rotate(temp, thetas[i]*180/np.pi, reshape=False) 
    bp[bp < 0] = 0 
    return bp
  
  
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


def remove_zeropadding(data , dim, factor):
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
        data = data[:,zpad:zpad+NR]
    elif dim == 'y':
        zpad = np.ceil((ny - np.float32(ny)/(factor+1))/2)
        data = data[zpad:zpad+NT,:]
    else:
        print 'Wrong dimension setting: dim <- x or y'
    return data


def ramp_filter():
    '''
    Returns samples of ramp filter
        h[0] = 1.0/(4*DR**2)
        h[n is even] = -1/(n*np.pi*DR)**2
        h[n is odd] = 0
    '''
    n = np.arange(-NR/2, NR/2, dtype=TYPE)
    h = np.zeros((1, NR))
    for i in range(len(n)):
        if n[i] == 0.0:
            h[0,i] = 1.0/(4*DR**2)      
        elif np.mod(n[i], 2):
            h[0,i] = -1/(n[i]*np.pi*DR)**2
    return h


def filter_sinogram(sino, h):
    '''
    Returns filtered sinogram
    Input: sino - sinogram
           h - ramp filter samples
    Each projection is filtered with ramp filter. Filtering has been performed
    in Fourier domain.Processing in Fourier domain requires zeropadding.  
    '''
    # zeropadding
    hpad = zeropadding(h, 'x', 1)
    sinopad = zeropadding(sino, 'x', 1)

    # compute Fourier transforms
    fthpad = DR * ft.fft(ft.ifftshift(hpad))
    ftsinopad = DR * ft.fft(ft.ifftshift(sinopad, axes=1), axis=1)
    
    # compute Fourier transform of filtered sinogram
    ftfiltsino = np.repeat(fthpad, NT, axis=0) * ftsinopad
    
    # compute nverse Fourier transform of filteref sinogram
    filtsino = np.real(ft.fftshift(ft.ifft(ftfiltsino, axis=1), axes=1))/DR
    
    #remove added zeros     
    filtsino = remove_zeropadding(filtsino, 'x', 1)
    return filtsino


def direct_fourier_rec(sino):
    '''
    Returns reconstructed 2D slice using Fourier interpolation  
    Input: sino - sinogram
    Direct implementation of Fourier slice theorem:
        - aply 1D Fourier transform on projections
        - fit/map radial samples into cartesian grid (requires intepolation)
        - do 2D inverse Fourier transform to obtain reconstructed image 
    '''
    # zeropadding
    sinopad = zeropadding(sino, 'x', 1)
    nr = np.size(sinopad, 1)

    # compute 1D Fourier transforms of projections 
    temp = ft.ifftshift(sinopad, axes=1)
    ftsinopad = DR * ft.fftshift(ft.fft(temp, axis=1), axes=1)
    
    # locations of interpolated Fourier samples 
    kx = 1.0/FOV * np.arange(-NX, NX, dtype=TYPE).reshape((1, 2*NX))
    ky = 1.0/FOV * np.arange(NY, -NY, -1, dtype=TYPE).reshape((2*NY, 1))    
    kxx = np.repeat(kx, 2*NY, axis=0)
    kyy = np.repeat(ky, 2*NX, axis=1)
    
    # locations of radial Fourier samples 
    thetas = np.pi/NT * np.arange(NT, dtype=TYPE).reshape((NT,1))
    kxin = 1.0/FOV * np.arange(-nr/2, nr/2, dtype=TYPE).reshape((1, nr))
    kyin = 1.0/FOV * np.arange(-nr/2, nr/2, dtype=TYPE).reshape((1, nr))
    kxxin = np.repeat(kxin, NT, axis=0) * np.repeat(np.cos(thetas), nr, axis=1) 
    kyyin = np.repeat(kyin, NT, axis=0) * np.repeat(np.sin(thetas), nr, axis=1) 
    
    points = np.empty((NT*nr, 2), dtype=TYPE)    
    points[:,0] = kxxin.reshape(NT*nr)
    points[:,1] = kyyin.reshape(NT*nr)
  
    ftdata = griddata(points, ftsinopad.reshape(NT*nr), (kxx, kyy), \
                      method='linear')
    ftdata[np.isnan(ftdata)] = 0
    
    # compute 2D inverse Fourier transform
    data = np.abs(ft.fftshift(ft.ifft2(ft.ifftshift(ftdata)))) / DR**2
    
    #return NY x NX data
    data = data[64:192,64:192]
    return data   

       
      
if __name__ == '__main__':
    # sinogram generated using analytical formulae for projections
    s = sinogram()
    # impulse response of ramp filter
    h = ramp_filter()
    # filtered sinogram
    fs = filter_sinogram(s,h)

    # comparison of original and filtered sinogram
    t = np.linspace(-FOV/2+1,FOV/2-1,FOV/2)*DR/2
    plt.figure()
    plt.plot(t, s[0,:]/np.max(s[0,:]), t, fs[0,:]/np.max(fs[0,:]), '--g')
    plt.grid(True)
    plt.title('Comparison of projection and filtered projection at theta=0')
    plt.xlabel('Radial position [mm]')
    plt.ylabel('Projection value')
    plt.show()

    # backprojection of original sinogram
    bp = backprojection(s)
    plt.figure()
    plt.imshow(bp, cmap=plt.cm.gray)
    plt.title('Simple backprojection image')
    plt.show()
    
    # backprojection of filterd sinogram
    fbp = backprojection(fs)
    plt.figure()
    plt.imshow(fbp, cmap=plt.cm.gray)
    plt.title('Filtered backprojection reconstruction')
    plt.show()

    # reconstruction using direct Fouerier method    
    dfm = direct_fourier_rec(s)
    plt.figure()
    plt.imshow(dfm, cmap=plt.cm.gray)
    plt.title('Fourier interpolation reconstruction')
    plt.show()
    
    # comparison of profiles from filtered backprojection and direct Fourier 
    # reconstruction
    t = np.linspace(-FOV/2+1,FOV/2-1,FOV/2)*DR/2
    plt.figure()
    plt.plot(t, fbp[64,:], t, dfm[64,:], 'g')
    plt.grid(True)
    plt.title('Profiles through reconstructed images')
    plt.xlabel('Radial position [mm]')
    plt.ylabel('Projection value')
    plt.legend(['FBP profile', 'FI profile'])
    plt.show()

 


    