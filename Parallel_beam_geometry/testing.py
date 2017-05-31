import numpy as np
import numpy.fft as ft
import matplotlib.pyplot as plt
from skimage.transform import radon
from skimage.transform import iradon
from scipy.interpolate import griddata
import time

TYPE = 'float64'

###############################################################################

def phantom(Nx, Ny, xc, yc, a, b, ang, refInd, plot):
    phantom = np.zeros((Ny,Nx), dtype = TYPE)
    i = np.arange(-1.0+1.0/Nx,1.0+1.0/Nx, 2.0/Nx)
    j = np.transpose(np.arange(1.0-1.0/Ny,-1.0-1.0/Ny, -2.0/Ny))
    ii, jj = np.meshgrid(i,j)
    
    for n in range(len(xc)):
        iirot = (ii-xc[n])*np.cos(ang[n]) + (jj-yc[n])*np.sin(ang[n]) 
        jjrot = -(ii-xc[n])*np.sin(ang[n]) + (jj-yc[n])*np.cos(ang[n])
        pj, pi = np.where((b[n]*iirot)**2 + (a[n]*jjrot)**2 < (a[n]*b[n])**2)
        phantom[pj,pi] = phantom[pj,pi] + refInd[n]

    if plot == True:
        plt.figure()
        plt.imshow(phantom,  cmap=plt.cm.gray, interpolation='none')
        plt.title('Phantom')
        plt.colorbar()
        plt.show()
    
    return phantom


###############################################################################

def projection(theta, xc, yc, a, b, fi, rho, Nr, dr):
    xcrot = dx/2
    ycrot = -dy/2 
    prj = np.zeros((Nr,), dtype=TYPE)    
    angle = theta - fi
    c = (a * np.cos(angle))**2 + (b * np.sin(angle))**2 
    for r in np.arange(-Nr/2, Nr/2):
        t = r*dr - ((xc - xcrot)*np.cos(theta) + (yc - ycrot)*np.sin(theta))
        if c - t**2 >= 0:
            prj[r+Nr/2] = 2.0*rho*a*b/c * np.sqrt(c - t**2)
    return prj            


def analytical_radon(thetas, Nt, Nr, dr, xc, yc, a, b, fi, rho, plot):   
    sino = np.zeros((Nt,Nr), dtype=TYPE)    
    for n in range(len(xc)):
        for j in range(Nt):
            sino[j,:] += projection(thetas[j], xc[n], yc[n], a[n], b[n], \
                                    fi[n], rho[n], Nr, dr)

    if plot == True:
        plt.figure()
        plt.imshow(sino,  cmap=plt.cm.gray, interpolation='none')
        plt.title('Sinogram')
        plt.colorbar()
        plt.show()

    return sino


###############################################################################

def get_p1(ang, t, r, xc, yc, dr):
    '''
    Return x and y coordinate for the intersection point of ray and detector
    Input: ang - projection angle
           t - index of radial sample
           xc, yc - coordinates of the center of rotation
           dr - distance between radial samples
    '''
    x0 = -r*np.sin(ang) + xc 
    y0 =  r*np.cos(ang) + yc
    return x0 + t*dr*np.cos(ang), y0 + t*dr*np.sin(ang)


def get_p2(ang, sdd, p1x, p1y):
    '''
    Return x and y coordinate for the source point
    Input: ang - projection angle
           sdd - source to detector distance
           p1x, py1 - coordinates of the intersection point of ray and detector
    '''
    return p1x + sdd*np.cos(ang-np.pi/2), p1y + sdd*np.sin(ang-np.pi/2)


def index2alpha(i, p1, p2, b, d):
    return ((b + np.float64(i)*d) - p2)/(p1 - p2)


def alpha2index(alpha, p1, p2, b, d):
    return (p2 + alpha*(p1-p2)- b)/d


def plane_index(alpha, index, p1, p2, b, d, N):
    if np.abs(alpha - index2alpha(index, p1, p2, b, d)) < 1e-6:
        return index
    else:
        return alpha2index(alpha, p1, p2, b, d)

def min_max_plane_indices(Nx, Ny, p1x, p1y, p2x, p2y, bx, by, dx, dy):
    '''
    Returns indices of entering and exiting x and y-planes for ray connecting
    p1 and p2 points.
    Input: Nx - number of columns in the image
           Ny - number of rowsin the image
           p1x, p1y - coordinates of p1 point
           p2x, p2y - coordinates of p2 point
           bx - x coordinate of bottom left corner of the image 
           by - y coordinate of bottom right corner of the image
           dx - pixel size in column/x direction
           dy - pixel size in row/y direction
    Output: imin - index of the first x-plane crossed by the ray
            imax - index of the last x-plane crossed by the ray
            jmin - index of the first y-plane crossed by the ray
            jmax - index of the last y-plane crossed by the ray
    '''
    #cases : (1) p1x = p2x (angle theta = 0 degree)
    #        (2) p1y = p2y (angle theta = 90 degree)
    #        (3) p1y > p2y (angle theta = [0,90> degree)
    #        (4) p1y < p2y (angle theta = <90,180> degree)   
    if np.abs(p1x-p2x) < 1e-3:
        alpha_min = index2alpha(0, p1y, p2y, by, dy)
        alpha_max = index2alpha(Ny, p1y, p2y, by, dy) 
        imin = -1
        imax = -1
        jmin = 0
        jmax = Ny
    elif np.abs(p1y-p2y) < 1e-3:
        alpha_min = index2alpha(Nx, p1x, p2x, bx, dx)
        alpha_max = index2alpha(0, p1x, p2x, bx, dx) 
        jmin = -1 
        jmax = -1
        imin = Nx
        imax = 0
    elif p1y > p2y:      
        alpha_min = np.max([index2alpha(Nx, p1x, p2x, bx, dx),\
                            index2alpha(0, p1y, p2y, by, dy)])
        alpha_max = np.min([index2alpha(Ny, p1y, p2y, by, dy),\
                            index2alpha(0, p1x, p2x, bx, dx)])  
        imin = np.floor(plane_index(alpha_min, Nx, p1x, p2x, bx, dx, Nx))
        imax = np.ceil(plane_index(alpha_max, 0, p1x, p2x, bx, dx, Nx))   
        jmin = np.ceil(plane_index(alpha_min, 0, p1y, p2y, by, dy, Ny))
        jmax = np.floor(plane_index(alpha_max, Ny, p1y, p2y, by, dy, Ny))  
    else:
        alpha_min = np.max([index2alpha(Nx, p1x, p2x, bx, dx),\
                            index2alpha(Ny, p1y, p2y, by, dy)])
        alpha_max = np.min([index2alpha(0, p1y, p2y, by, dy),\
                            index2alpha(0, p1x, p2x, bx, dx)])
        imin = np.floor(plane_index(alpha_min, Nx, p1x, p2x, bx, dx, Nx))
        imax = np.ceil(plane_index(alpha_max, 0, p1x, p2x, bx, dx, Nx))   
        jmin = np.floor(plane_index(alpha_min, Ny, p1y, p2y, by, dy, Ny))
        jmax = np.ceil(plane_index(alpha_max, 0, p1y, p2y, by, dy, Ny))
    return imin, imax, jmin, jmax


def siddon_radon(obj_img, Nx, Ny, dx, dy, Nt, Nr, dr):
    '''
    Computes Radon transform of 2D object image.
    Based on Siddon algorithm for ray tracing.
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
    # define reference system
    # center of the image will be at the center of coordinate system
    bx = -Nx/2 * dx
    by = -Ny/2 * dy
    # field of view
    fovx = Nx * dx
    fovy = Ny * dy
    # radius of rotation for source/detector
    r = np.ceil(np.sqrt(fovx**2 + fovy**2))
    # center of rotation    
    xc = dx/2
    yc = -dy/2
    # source - detector distance
    sdd = 2*r
   
    prj = np.zeros((Nt,Nr), dtype=TYPE) 

    for angle in np.arange(Nt):
        l = np.zeros((Ny,Nx), dtype=TYPE) 
        for t in np.arange(-Nr/2, Nr/2):
            
            # find p1-detector and p2-source coordinates    
            p1x, p1y = get_p1(angle*np.pi/Nt, t, r, xc, yc, dr)                         
            p2x, p2y = get_p2(angle*np.pi/Nt, sdd, p1x, p1y)
            
            # find indices of entering and exiting planes 
            imin, imax, jmin, jmax = min_max_plane_indices(Nx, Ny, p1x, p1y, \
                                                           p2x, p2y, bx, by, \
                                                           dx, dy)

            # find alpha for indices in interval [ilo, ihi] and [jlo,jhi]
            alpha_x = []
            alpha_y = []
            if imin != -1 and imax != -1:
                ilo = min([imin,imax])
                ihi = max([imin, imax])
                alpha_x = [index2alpha(i, p1x, p2x, bx, dx) \
                           for i in np.arange(ilo,ihi+1)] 
            if jmin != -1 and jmax != -1:
                jlo = min([jmin,jmax])
                jhi = max([jmin, jmax])
                alpha_y = [index2alpha(j, p1y, p2y, by, dy) \
                           for j in np.arange(jlo,jhi+1)]
            # merge parameters without duplicates
            alpha_xy = list(set(alpha_x)|set(alpha_y))
            # sort parameters (ascending values) 
            alpha_xy.sort()
          
            # compute i,j for alphas
            for m in range(1,len(alpha_xy)):            
                a = (alpha_xy[m] + alpha_xy[m-1])/2
                i = np.floor(alpha2index(a, p1x, p2x, bx, dx))
                j = np.floor(alpha2index(a, p1y, p2y, by, dy))
                # compute intersection length for m-th pixel
                # print (j, i)
                if (i >= Nx) or (j >= Ny) or (j < 0) or (i < 0):
                    continue
                else:   
                    l[Ny-1-j,i] = (alpha_xy[m] - alpha_xy[m-1])*sdd
                    prj[angle,t+Nr/2] += l[Ny-1-j,i]*obj_img[Ny-1-j,i]
 
    return prj


def siddon_iradon(sino, Nt, Nr, dr, Nx, Ny, dx, dy):
    '''
    Computes Inverse Radon Transform from sinogram.
    Based on Siddon algorithm for ray tracing.
    Input:  sino - sinogram, row -> projection angle, column -> radial sample
            Nx, Ny - number of pixels
            dx, dy - dimensions of pixels
            Nt - number of angles 
            Nr - number of radial samples
            dr - distance between centers of pixels in cm
    Output: obj_img - object image  
    '''
    # define reference system
    bx = -Nx/2 * dx
    by = -Ny/2 * dy
    fovx = Nx * dx
    fovy = Ny * dy
    r = np.ceil(np.sqrt(fovx**2 + fovy**2))
    xc = dx/2
    yc = -dy/2
    sdd = 2*r
   
    obj_img = np.zeros((Ny,Nx), dtype=TYPE) 

    for angle in np.arange(Nt):
        l = np.zeros((Ny,Nx), dtype=TYPE)
        for t in np.arange(-Nr/2, Nr/2):

            # find p1-detector and p2-source coordinates    
            p1x, p1y = get_p1(angle*np.pi/Nt, t, r, xc, yc, dr)                         
            p2x, p2y = get_p2(angle*np.pi/Nt, sdd, p1x, p1y)
            
            # find indices of entering and exiting planes 
            imin, imax, jmin, jmax = min_max_plane_indices(Nx, Ny, p1x, p1y, \
                                                           p2x, p2y, bx, by, \
                                                           dx, dy)

            # find alpha for indices in interval [ilo, ihi] and [jlo,jhi]
            alpha_x = []
            alpha_y = []
            if imin != -1 and imax != -1:
                ilo = min([imin,imax])
                ihi = max([imin, imax])
                alpha_x = [index2alpha(i, p1x, p2x, bx, dx) \
                           for i in np.arange(ilo,ihi+1)] 
            if jmin != -1 and jmax != -1:
                jlo = min([jmin,jmax])
                jhi = max([jmin, jmax])
                alpha_y = [index2alpha(j, p1y, p2y, by, dy) \
                           for j in np.arange(jlo,jhi+1)]
            # merge parameters without duplicates
            alpha_xy = list(set(alpha_x)|set(alpha_y))
            # sort parameters (ascending values) 
            alpha_xy.sort()

            # compute i,j for alphas
            for m in range(1,len(alpha_xy)):            
                a = (alpha_xy[m] + alpha_xy[m-1])/2
                i = np.floor(alpha2index(a, p1x, p2x, bx, dx))
                j = np.floor(alpha2index(a, p1y, p2y, by, dy))
                # compute intersection length for m-th pixel
                #print (j, i)
                if (i >= Nx) or (j >= Ny) or (j < 0) or (i < 0):
                    continue
                else:  
                    l[Ny-1-j,i] = (alpha_xy[m] - alpha_xy[m-1])*sdd 
                    obj_img[Ny-1-j,i] += l[Ny-1-j,i]*sino[angle,t+Nr/2]
                    #print (angle, t+Nr/2, sino[angle,t+Nr/2])
    return obj_img


###############################################################################

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


def backprojection_iradon(sino, Nt, Nr, dr, Nx, Ny, dx, dy):
    '''
    Computes Inverse Radon Transform from sinogram.
    Input:  sinogram - row -> projection angle, column -> radial sample
            Nx, Ny - number of pixels
            dx, dy - dimensions of pixels
            Nt - number of angles 
            Nr - number of radial samples
            dr - distance between centers of pixels in cm
    Output: obj_img - object image  
    '''
    # define reference system
    bx = -Nx/2 * dx
    by = -Ny/2 * dy
    xc = dx/2
    yc = -dy/2
    
    obj_img = np.zeros((Ny,Nx), dtype=TYPE) 
    x = np.arange(bx+dx/2,-bx+dx/2, dx)
    y = np.arange(by+dy/2,-by+dy/2, dy)
    prj_points = np.zeros(2);
    for i in np.arange(len(x)):
        for j in np.arange(len(y)):
            for angle in np.arange(Nt):
                a = np.pi*angle/Nt
                t = ((x[i]-xc)*np.cos(a) + (y[j]-yc)*np.sin(a))/dr 
                if np.floor(t+Nr/2) < 0:
                    prj_points = [-Nr**2, -Nr/2-1, -Nr/2]
                    prj_val = [0, 0, sino[angle, 0]]
                    obj_img[Ny-1-j,i] += np.interp(t+Nr/2, prj_points, prj_val)
                elif np.ceil(t+Nr/2) > Nr-1:  
                    prj_points = [Nr-1, Nr, Nr**2]
                    prj_val = [sino[angle, Nr-1], 0, 0]
                    obj_img[Ny-1-j,i] += np.interp(t+Nr/2, prj_points, prj_val)              
                else:
                    prj_points = [np.floor(t+Nr/2), np.ceil(t+Nr/2)]
                    prj_val = sino[angle, prj_points]
                    obj_img[Ny-1-j,i] += np.interp(t+Nr/2, prj_points, prj_val)
    return np.pi/Nt*obj_img


def direct_fourier_rec(sino, Nt, Nr, dr, Nx, Ny, dx, dy):
    '''
    Returns reconstructed 2D slice using Fourier interpolation  
    Input: sino - sinogram
    Direct implementation of Fourier slice theorem:
        - aply 1D Fourier transform on projections
        - fit/map radial samples into cartesian grid (requires intepolation)
        - do 2D inverse Fourier transform to obtain reconstructed image 
    '''
    fovx = Nx * dx
    fovy = Ny * dy

    # zeropadding
    sinopad = zeropadding(sino, 'x', 1)
    nr = np.size(sinopad, 1)

    # compute 1D Fourier transforms of projections 
    temp = ft.ifftshift(sinopad, axes=1)
    ftsinopad = dr * ft.fftshift(ft.fft(temp, axis=1), axes=1)
    
    # locations of interpolated Fourier samples 
    kx = 1.0/fovx * np.arange(-Nx, Nx, dtype=TYPE).reshape((1, 2*Nx))
    ky = 1.0/fovy * np.arange(Ny, -Ny, -1, dtype=TYPE).reshape((2*Ny, 1))    
    kxx = np.repeat(kx, 2*Ny, axis=0)
    kyy = np.repeat(ky, 2*Nx, axis=1)
    
    # locations of radial Fourier samples 
    thetas = np.pi/Nt * np.arange(Nt, dtype=TYPE).reshape((Nt,1))
    kxin = 1.0/fovx * np.arange(-nr/2, nr/2, dtype=TYPE).reshape((1, nr))
    kyin = 1.0/fovy * np.arange(-nr/2, nr/2, dtype=TYPE).reshape((1, nr))
    kxxin = np.repeat(kxin, Nt, axis=0) * np.repeat(np.cos(thetas), nr, axis=1) 
    kyyin = np.repeat(kyin, Nt, axis=0) * np.repeat(np.sin(thetas), nr, axis=1) 
    
    points = np.empty((Nt*nr, 2), dtype=TYPE)    
    points[:,0] = kxxin.reshape(Nt*nr)
    points[:,1] = kyyin.reshape(Nt*nr)
  
    ftdata = griddata(points, ftsinopad.reshape(Nt*nr), (kxx, kyy), \
                         method='linear')
    ftdata[np.isnan(ftdata)] = 0
    
    # compute 2D inverse Fourier transform
    data = np.abs(ft.fftshift(ft.ifft2(ft.ifftshift(ftdata)))) / dr**2
    
    #return Nx x Ny data
    data = data[64:192,64:192]
    return data  

###############################################################################

def rho(r, Nr, dr):
    return (-(Nr-1)/2.0 + r)*dr


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
    return obj_img


###############################################################################

if __name__ == '__main__':
    
    Nx = 128
    Ny = 128
    thetas = np.arange(0.0, np.pi, np.pi/100)
    Nr = 128
    dr = 2.0/Nx
    dx = 2.0/Nx
    dy = 2.0/Ny
    Nt = len(thetas)
    h = ramp_filter(Nr, dr)

    # Shepp Logan phantom   
    # xc, yc - center coordinates for ellipses
    xc = [0.0, 0.0, 0.22, -0.22, 0.0, 0.0, 0.0, -0.08, 0.0, 0.06] 
    yc = [0.0, -0.0184, 0.0, 0.0, 0.35, 0.1, -0.1, -0.605, -0.605, -0.605]
    # a - major axis for ellipses
    a = [0.92, 0.874, 0.31, 0.41, 0.25, 0.046, 0.046, 0.046, 0.023, 0.046]
    # b - minor axis for ellipses
    b = [0.69, 0.6624, 0.11, 0.16, 0.21, 0.046, 0.046, 0.023, 0.023, 0.023]
    # angdeg - rotation angle for ellipses in degrees
    angdeg = [90.0, 90.0, 72.0, 108.0, 90.0, 0.0, 0.0, 0.0, 0.0, 90.0]
    # ang - rotation angle for ellipses in radians
    ang = map(lambda x: x/180.0*np.pi, angdeg)
    # relative refraction indexfor the ellipses
    relRefInd = [2.0, -0.98, -1.02, -1.02, 0.4, 0.4, 0.4, 0.6, 0.6, 0.6]
    '''
    
    #Testing unit:
    xc = [0.2]
    yc = [0.2]
    a = [0.2]
    b = [0.2]
    ang = [0]
    relRefInd = [1]
    '''

    phantom = phantom(Nx, Ny, xc, yc, a, b, ang, relRefInd, plot=True)
    asino = analytical_radon(thetas, Nt, Nr, dr, xc, yc, a, b, ang, relRefInd, plot=True)  # rotation about (dx/2, -dy/2)
    fasino = filter_sinogram(asino, h, Nt, Nr, dr)
    '''
    blt_asino_rec = iradon(np.transpose(fasino), theta=thetas*180.0/np.pi, filter=None, circle=True)     # rotation about (dx/2, -dy/2)
    plt.figure()
    plt.imshow(blt_asino_rec,  cmap=plt.cm.gray, interpolation='none')
    plt.title('Built-in Rec. - Analytical Sin. (dx/2, -dy/2)')
    plt.colorbar()
    plt.show()
    plt.savefig('blt_rec_ana_sin.tif', format = "tif")
    
    fbp_asino_rec = backprojection_iradon(fasino, Nt, Nr, dr, Nx, Ny, dx, dy)  # rotation about (dx/2, -dx/2)
    plt.figure()
    plt.imshow(fbp_asino_rec,  cmap=plt.cm.gray, interpolation='none')
    plt.title('Filtered backprojection Rec. (dx/2, -dy/2) - Analytical Sin. (dx/2, -dy/2)')
    plt.colorbar()
    plt.show()
    plt.savefig('fbp_rec_ana_sin.tif', format = "tif")

    t_asino_rec = toft_iradon(fasino, Nt, Nr, dr, Nx, Ny, dx, dy)              # rotation about (0,0)
    plt.figure()
    plt.imshow(t_asino_rec,  cmap=plt.cm.gray, interpolation='none')
    plt.title('Peter Toft Iradon Rec. - Analytical Sin. (0, 0)')
    plt.colorbar()
    plt.show()
    plt.savefig('toft_rec_ana_sin.tif', format = "tif")


    s_asino_rec = siddon_iradon(fasino, Nt, Nr, dr, Nx, Ny, dx, dy)            # rotation about (dx/2, -dy/2)
    plt.figure()
    plt.imshow(s_asino_rec,  cmap=plt.cm.gray, interpolation='none')
    plt.title('Siddon Iradon Rec. - Analytical Sin. (dx/2, -dy/2)')
    plt.colorbar()
    plt.show()
    plt.savefig('siddon_rec_ana_sin.tif', format = "tif")
    
    #dfr_asino_rec = direct_fourier_rec(asino, Nt, Nr, dr, Nx, Ny, dx, dy)     # rotation about (dx/2, -dx/2)
    '''
    
    tsino = toft_radon(phantom, Nx, Ny, dx, dy, Nt, Nr, dr)                    # rotation about (0,0)
    ftsino = filter_sinogram(tsino, h, Nt, Nr, dr)
    t_tsino_rec = backprojection_iradon(ftsino, Nt, Nr, dr, Nx, Ny, dx, dy)
    plt.figure()
    plt.imshow(t_tsino_rec,  cmap=plt.cm.gray, interpolation='none')
    plt.title('Filter backprojection Rec. - Toft Sin. (dx/2, -dy/2)')
    plt.colorbar()
    plt.show()
    plt.savefig('fbp_rec_toft_sin.tif', format = "tif")

    '''
    ssino = siddon_radon(phantom, Nx, Ny, dx, dy, Nt, Nr, dr)                  # rotation about (dx/2, -dy/2)
    fssino = filter_sinogram(ssino, h, Nt, Nr, dr)
    s_ssino_rec = siddon_iradon(fssino, Nt, Nr, dr, Nx, Ny, dx, dy)
    plt.figure()
    plt.imshow(s_ssino_rec,  cmap=plt.cm.gray, interpolation='none')
    plt.title('Sidon Iradon Rec. - Siddon Sin. (dx/2, -dy/2)')
    plt.colorbar()
    plt.show()
    plt.savefig('siddon_rec_siddon_sin.tif', format = "tif")

    bltsino = radon(phantom, theta=thetas*180.0/np.pi, circle=True)            # rotation about (dx/2, -dy/2)
    blt_bltsino_rec = iradon(bltsino, theta=thetas*180.0/np.pi, circle=True)   
    plt.figure()
    plt.imshow(blt_bltsino_rec,  cmap=plt.cm.gray, interpolation='none')
    plt.title('Built-in Iradon Rec. - Built-in Sin. (dx/2, -dy/2)')
    plt.colorbar()
    plt.show()
    plt.savefig('blt_rec_blt_sin.tif', format = "tif")
    '''