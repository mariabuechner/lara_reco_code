import numpy as np
import numpy.fft as ft
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid.axislines import SubplotZero
import time


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


def filter_sinogram(sino, h, Nt, Nr, dr):
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
    fthpad = dr * ft.fft(ft.ifftshift(hpad))
    ftsinopad = dr * ft.fft(ft.ifftshift(sinopad, axes=1), axis=1)
    
    # compute Fourier transform of filtered sinogram
    ftfiltsino = np.repeat(fthpad, Nt, axis=0) * ftsinopad
    
    # compute nverse Fourier transform of filteref sinogram
    filtsino = np.real(ft.fftshift(ft.ifft(ftfiltsino, axis=1), axes=1))/dr
    
    #remove added zeros     
    filtsino = remove_zeropadding(filtsino, 'x', 1, Nt, Nr)

    return filtsino


def toft_radon(obj_img, Nx, Ny, dx, dy, Nt, Nr, dr):
    '''
    Computes Radon transform of 2D object image.
    Based on Peter Toft algorithm
    Input:  obj_img - phantom image
            Nx, Ny - number of pixels
            dx, dy - dimensions of pixels
            Nt - number of angles 
            Nr - number of radial samples
            dr - distance between centers of pixels in cm
    Output: prj - sinogram of the object image
                - rows -> projection angle
                - columns -> radial sample    
    '''
    xmin = -(Nx-1)/2.0*dx
    ymin = -(Ny-1)/2.0*dy

    prj = np.zeros((Nt,Nr), dtype=TYPE)
    
    for angle in np.arange(Nt):
        sina = np.sin(angle*np.pi/Nt)
        cosa = np.cos(angle*np.pi/Nt)
        rmin = xmin*cosa + ymin*sina
        for t in np.arange(Nr):
            if sina > 1.0/np.sqrt(2):
                for n in np.arange(Nx): 
                    a = -dx*cosa/(dy*sina)
                    b = (rho(t, Nr, dr) - rmin)/(dy*sina)
                    m = np.round(a*n + b)
                    if m > Nx -1 or m < 0:
                        continue
                    prj[angle, t] += dx/np.abs(sina) * obj_img[Ny-1-m,n]                       
            else :
                for m in np.arange(Ny):                
                    a = -dy*sina/(dx*cosa)
                    b = (rho(t, Nr, dr) - rmin)/(dx*cosa)
                    n = np.round(a*m + b)
                    if n > Ny -1 or n < 0:
                        continue
                    prj[angle, t] += dy/np.abs(cosa) * obj_img[Ny-1-m,n]  
    return prj

def toft_iradon(prj, Nt, Nr, dr, Nx, Ny, dx, dy):
    '''
    Computes Inverse Radon Transform from sinogram.
    Based on Peter Toft algorithm 
    Input:  prj - sinogram, row -> projection angle, column -> radial sample
            Nt - number of angles 
            Nr - number of radial samples
            dr - distance between centers of pixels in cm
            Nx, Ny - number of pixels
            dx, dy - dimensions of pixels
    Output: obj_img - object image  
    '''
    xmin = -(Nx-1)/2.0*dx
    ymin = -(Ny-1)/2.0*dy

    obj_img = np.zeros((Ny,Nx), dtype=TYPE)
    
    for angle in np.arange(Nt):
        sina = np.sin(angle*np.pi/Nt)
        cosa = np.cos(angle*np.pi/Nt)
        rmin = xmin*cosa + ymin*sina
        for t in np.arange(Nr):
            if sina > 1.0/np.sqrt(2):
                for n in np.arange(Nx): 
                    a = -dx*cosa/(dy*sina)
                    b = (rho(t, Nr, dr) - rmin)/(dy*sina)
                    m = np.round(a*n + b)
                    if m > Nx -1 or m < 0:
                        continue
                    obj_img[Ny-1-m,n] += dx/np.abs(sina) * prj[angle, t]                       
            else :
                for m in np.arange(Ny):                
                    a = -dy*sina/(dx*cosa)
                    b = (rho(t, Nr, dr) - rmin)/(dx*cosa)
                    n = np.round(a*m + b)
                    if n > Ny -1 or n < 0:
                        continue
                    obj_img[Ny-1-m,n] += dy/np.abs(cosa) * prj[angle, t]  
    return  np.pi/Nt * obj_img


def rho(r, Nr, dr):
    return (-(Nr-1)/2.0 + r)*dr

 
def test_n_computation(theta, t, Nx, Ny, dx, dy, Nt, Nr, dr):
    xmin = -(Nx-1)/2.0*dx
    ymin = -(Ny-1)/2.0*dy
    sina = np.sin(theta)
    cosa = np.cos(theta)
    rmin = xmin*cosa + ymin*sina  
    a = -dy*sina/(dx*cosa)
    b = (rho(t, Nr, dr) - rmin)/(dx*cosa)
    data = np.zeros((2,Ny))  
    for m in np.arange(Ny):
        data[0,m] = a*m+b
        data[1,m] = m
    return data


def test_m_computation(theta, t, Nx, Ny, dx, dy, Nt, Nr, dr):
    xmin = -(Nx-1)/2.0*dx
    ymin = -(Ny-1)/2.0*dy
    sina = np.sin(theta)
    cosa = np.cos(theta)  
    rmin = xmin*cosa + ymin*sina
    a = -dx*cosa/(dy*sina)
    b = (rho(t, Nr, dr) - rmin)/(dy*sina)
    data = np.zeros((2, Nx)) 
    for n in np.arange(Nx):
        data[0, n] = n
        data[1, n] = a*n+b
    return data


def test_m_n_computation(obj_img, theta, Nx, Ny, dx, dy, Nt, Nr, dr):
    
    bx = -Nx/2 * dx 
    by = -Ny/2 * dy 
    xmin = -(Nx-1)/2.0*dx
    ymin = -(Ny-1)/2.0*dy
    
    legend_list = []
    fig = plt.figure(figsize=(8,8))
    ax = SubplotZero(fig, 111)
    fig.add_subplot(ax)
    plt.imshow(obj_img, cmap=plt.cm.gray, interpolation='none', \
                        extent=[bx,-bx, by, -by], alpha=0.5)
    for t in np.arange(Nr):   
        if np.sin(theta) < 1.0/np.sqrt(2):
            data = test_n_computation(theta, t, Nx, Ny, dx, dy, Nt, Nr, dr)
        else:
            data = test_m_computation(theta, t, Nx, Ny, dx, dy, Nt, Nr, dr)
        #rdata = np.round(data)
        plt.plot(data[0,:]*dx+xmin, data[1,:]*dy+ymin,'*-', linewidth = 3 )  
        legend_list.append('(t=' + str(t) + ')')
    for direction in ["xzero", "yzero"]:
        ax.axis[direction].set_axisline_style("-|>")
        ax.axis[direction].set_visible(True)
    ax.xaxis.set_ticks(np.arange(-Nx,Nx,2))
    ax.yaxis.set_ticks(np.arange(-Ny,Ny,2))
    plt.axis('equal')
    plt.grid(True)
    plt.legend(legend_list)
    plt.title('Angle: ' + str(theta*180/np.pi))
 
    
def generate_circle(Nx, Ny, xc, yc, rad, att):
    ''' 
    Generetas simple phantom image (circular objet)
    Input: Nx - number of columns in the image, 
           Ny - number of rows in the image
           xc, yc - coordinates for the center of circular object, px
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

    Nx = 8
    Ny = 8
    Nr = 2*Nx-1
    Nt = 2*np.ceil(np.pi*(Nx-1))
    dx = 2
    dy = 2
    dr = 2

    obj_img = generate_circle(Nx, Ny, Nx/2, Ny/2, Nx/4, 1)

    plt.figure()
    plt.imshow(obj_img, cmap=plt.cm.gray, interpolation='none')
    plt.title('Object')
    plt.show()

    prj = toft_radon(obj_img, Nx, Ny, dx, dy, Nt, Nr, dr)

    plt.figure()
    plt.imshow(prj, cmap=plt.cm.gray, interpolation='none')
    plt.title('Radon transform')
    plt.show()

    rec = toft_iradon(prj, Nt, Nr, dr, Nx, Ny, dx, dy)

    plt.figure()
    plt.imshow(rec, cmap=plt.cm.gray, interpolation='none')
    plt.title('Inverse Radon transform')
    plt.show()

    h = ramp_filter(Nr, dr)
    fprj = filter_sinogram(prj, h, Nt, Nr, dr)
    f_rec = toft_iradon(fprj, Nt, Nr, dr, Nx, Ny, dx, dy)

    plt.figure()
    plt.imshow(f_rec, cmap=plt.cm.gray, interpolation='none')
    plt.title('Inverse Radon transform (filtered sinogram)')
    plt.show()