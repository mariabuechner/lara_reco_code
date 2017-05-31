import numpy as np
import numpy.fft as ft
import matplotlib.pyplot as plt
#from skimage.transform import radon
#from skimage.transform import iradon
from mpl_toolkits.axes_grid.axislines import SubplotZero
import time

TYPE = 'float64'

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


def filter_sinogram_windowing_book(sino, h, Nt, Nr, dr):
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
    
    #multiply with Hamming window
    Hwindow = np.repeat(np.reshape(np.hamming(2*Nr), (1,2*Nr)), Nt, axis=0) 
    ftfiltsino = ftfiltsino * Hwindow

    # compute nverse Fourier transform of filteref sinogram
    filtsino = np.real(ft.fftshift(ft.ifft(ftfiltsino, axis=1), axes=1))/dr
    
    #remove added zeros     
    filtsino = remove_zeropadding(filtsino, 'x', 1, Nt, Nr)

    return filtsino


def filter_sinogram_windowing(sino, h, Nt, Nr, dr):
    '''
    Returns filtered sinogram
    Input: sino - sinogram
           h - ramp filter samples
    Each projection is filtered with ramp filter. Filtering has been performed
    in Fourier domain.Processing in Fourier domain requires zeropadding.  
    '''

    #multiply with Hamming window
    Hwindow = np.repeat(np.reshape(np.hamming(Nr), (1,Nr)), Nt, axis=0)
    sino = sino *Hwindow 
    h = h * np.reshape(np.hamming(Nr), (1,Nr))

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



def test_min_max_plane_indices(obj_img, Nx, Ny, dx, dy, Nt, Nr, dr):
    
    bx = -Nx/2 * dx
    by = -Ny/2 * dy
    fovx = Nx * dx
    fovy = Ny * dy
    r = np.ceil(np.sqrt((fovx)**2 + (fovy)**2))
    xc = dx/2
    yc = -dy/2
    sdd = 2*r
    
    p1x = np.zeros(Nt, dtype=TYPE)
    p1y = np.zeros(Nt, dtype=TYPE)
    p2x = np.zeros(Nt, dtype=TYPE)
    p2y = np.zeros(Nt, dtype=TYPE)
    for angle in np.arange(Nt):
        text = '' 
        legend_list = []
        fig = plt.figure(figsize=(8,8))
        ax = SubplotZero(fig, 111)
        fig.add_subplot(ax)
        plt.imshow(obj_img, cmap=plt.cm.gray, interpolation='none', \
                            extent=[bx,-bx, by, -by], alpha=0.5)
        for t in np.arange(-Nr/2, Nr/2):
            # find p1-detector and p2-source coordinates  
            p1x[angle], p1y[angle] = get_p1(angle*np.pi/Nt, t, r, xc, yc, dr)                         
            p2x[angle], p2y[angle] = get_p2(angle*np.pi/Nt, sdd, p1x[angle], \
                                            p1y[angle])
            plt.plot( [p1x[angle], p2x[angle]] ,[p1y[angle], p2y[angle]], ':',\
                      linewidth = 3 )  
            imin, imax, jmin, jmax = min_max_plane_indices(Nx, Ny, \
                p1x[angle], p1y[angle], p2x[angle], p2y[angle], bx, by, dx, dy)
            text += '(t=' + str(t) + ') imin:' + str(imin) + ', imax:' + \
                    str(imax) + ' ,jmin:' + str(jmin) + ' ,jmax:' + \
                    str(jmax) + '\n'
            legend_list.append('(t=' + str(t) + ')')
        for direction in ["xzero", "yzero"]:
            ax.axis[direction].set_axisline_style("-|>")
            ax.axis[direction].set_visible(True)
        ax.xaxis.set_ticks(np.arange(-Nx,Nx,2))
        ax.yaxis.set_ticks(np.arange(-Ny,Ny,2))
        plt.axis('equal')
        plt.grid(True)
        plt.legend(legend_list)
        plt.title(text, fontsize=10) 
        
        
   
def test_get_p1(Nr, Nt, sdd, r, xc, yc, dr):
    for a in np.arange(Nt):
        y = np.zeros(Nr, dtype=TYPE)
        x = np.zeros(Nr, dtype=TYPE)  
        for t in np.arange(-Nr/2, Nr/2): 
            x[t+Nr/2], y[t+Nr/2] = get_p1(a*np.pi/Nt, t, r, xc, yc, dr)
        plt.plot(x,y,'*')
        plt.show()
    return 1          


def test_get_p2(Nr, Nt, sdd, r, xc, yc, dr):
    for a in np.arange(Nt):
        y1 = np.zeros(Nr, dtype=TYPE)
        x1 = np.zeros(Nr, dtype=TYPE)
        y2 = np.zeros(Nr, dtype=TYPE)
        x2 = np.zeros(Nr, dtype=TYPE)  
        for t in np.arange(-Nr/2, Nr/2): 
            x1[t+Nr/2], y1[t+Nr/2] = get_p1(a*np.pi/Nt,t,r,xc,yc,dr) 
            x2[t+Nr/2], y2[t+Nr/2] = get_p2(a*np.pi/Nt, sdd, x1[t+Nr/2], \
                                            y1[t+Nr/2]) 
        plt.plot(x2, y2, 'p')
        plt.show()
    return 1 


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


def backprojection_iradon(sino, Nt, Nr, dr, Nx, Ny, dx, dy):
    '''
    Computes Inverse Radon Transform from sinogram.
    Steps: 1D FTT of projections -> gridding -> 2D IFFT
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


def test_t_values(angles, i, j, Nt, Nr, dr, Nx, Ny, dx, dy):
    bx = -Nx/2 * dx
    by = -Ny/2 * dy
    x = np.arange(bx+dx/2,-bx+dx/2, dx)
    y = np.arange(by+dy/2,-by+dy/2, dy)
    for angle in angles:
        t = ((x[i]-dr/2.0)*np.cos(angle) + (y[j]+dr/2.0)*np.sin(angle))/dr
        print 'Point [', str(i), ',', str(j), '], angle: ', str(angle), 't:', \
              str(t), '\n'
    return 0

            
     
if __name__ == '__main__':

    Nx = 8
    Ny = 8
    Nr = 8
    Nt = 8
    dx = 2
    dy = 2
    dr = 2

    '''obj_img = generate_circle(Nx, Ny, Nx/2, Ny/2, Nx/4, 1) + \
              generate_circle(Nx, Ny, 3*Nx/4, 3*Ny/4, Nx/8, 0.2) + \
              generate_circle(Nx, Ny, Nx/4, Ny/4, Nx/4, 0.5) + \
              generate_circle(Nx, Ny, 3*Nx/4, Ny/4, Nx/4, 0.75)'''
    obj_img = generate_circle(Nx, Ny, Nx/2, Ny/2, Nx/4, 1)

    plt.figure()
    plt.imshow(obj_img, cmap=plt.cm.gray, interpolation='none')
    plt.title('Object')
    plt.show()
    
    start = time.clock()
    p = siddon_radon(obj_img, Nx, Ny, dx, dy, Nt, Nr, dr)        
    end = time.clock()
    print 'Siddon_radon', end - start
    # theta=np.arange(Nt) * 180.0/Nt
    #r = np.transpose(radon(obj_img, theta, circle=True))
    # introducing shift for angles > 90 degrees
    #ap = np.zeros((Nt,Nr), dtype=TYPE) 
    #ap[np.arange(Nt/2), :] = p[np.arange(Nt/2), :]
    #temp = p[::-1]
    #ap[np.arange(Nt/2,Nt), :] = temp[np.arange(Nt/2,Nt), :] 

    plt.figure()
    plt.imshow(p, cmap=plt.cm.gray, interpolation='none')
    plt.title('Siddon_radon.')
    plt.show()

    '''plt.figure()
    plt.imshow(r, cmap=plt.cm.gray, interpolation='none')
    plt.title('Built-in Radon.')
    plt.show()'''
 
    #print 'Max diff.', np.max(np.abs(r-p))

    # reconstruction using sidon_iradon
    start = time.clock()
    ip = backprojection_iradon(p, Nt, Nr, dr, Nx, Ny, dx, dy)   
    end = time.clock()
    print 'Backprojection_iradon', end - start
    plt.figure()
    plt.imshow(ip, cmap=plt.cm.gray, interpolation='none')
    plt.title('Backprojection_iradon')
    plt.show()

    '''# reconstruction using built-in iradon
    ir = iradon(np.transpose(r), theta=np.arange(Nt) * 180.0/Nt, circle=True)
    plt.figure()
    plt.imshow(ir, cmap=plt.cm.gray, interpolation='none')
    plt.title('Built-in iradon (built-in radon)')
    plt.show()


    # reconstruction using built-in iradon
    ipr = iradon(np.transpose(p), theta=np.arange(Nt) * 180.0/Nt, circle=True)
    plt.figure()
    plt.imshow(ipr, cmap=plt.cm.gray, interpolation='none')
    plt.title('Built-in iradon (siddon_radon)')
    plt.show()'''

    # reconstruction using siddon_iradon and filtered sinogram
    h = ramp_filter(Nr, dr)
    fp = filter_sinogram(p, h, Nt, Nr, dr)
    start = time.clock()
    isfp = siddon_iradon(fp, Nt, Nr, dr, Nx, Ny, dx, dy) 
    end = time.clock()
    print 'Siddon_iradon', end - start
    isfp[isfp<0] = 0
    plt.figure()
    plt.imshow(isfp, cmap=plt.cm.gray, interpolation='none')
    plt.title('siddon_iradon(filtered sinogram)')
    plt.show()
    
    # reconstruction using siddon_iradon and filtered sinogram
    bpfp = backprojection_iradon(fp, Nt, Nr, dr, Nx, Ny, dx, dy)   
    bpfp[bpfp<0] = 0
    plt.figure()
    plt.imshow(bpfp, cmap=plt.cm.gray, interpolation='none')
    plt.title('backprojection_iradon(filtered sinogram)')
    plt.show()
