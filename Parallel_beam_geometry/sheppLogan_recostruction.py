import sys
import numpy as np
import numpy.fft as ft
import matplotlib.pyplot as plt
import scipy.ndimage as scpndi 
from scipy.interpolate import griddata
from skimage.transform import radon
from skimage.transform import iradon


# Imaging parameters:
DR = 2              # distance between radial samples in cm
FOV = 256           # field of view in cm
NR = 128            # number of radial samples in a projection
NT = 100            # number of angles 
NX = 128            # number of columns in reconstructed image
NY = 128            # number of rows in reconstructed image

TYPE = 'float64'    # array's type

################Filippo's code to generate Shepp-Logan phantom#################
class Ellipse:
    ##  DEFINE THE CLASS FIELDS
    def __init__ ( self , a , b , orig , rotaz , value , npix ):
        ##  Horizontal ( when "rotaz" = 0 ) semi-axis
        self.a = a * 0.5 * npix

        ##  Vertical ( when "rotaz" = 0 ) semi-axis
        self.b = b * 0.5 * npix

        ##  Origin of the ellipse; if no translation, 
        ##  it corresponds to the origin of the image
        if orig[0] >= -1 and orig[0] <= 1 and orig[1] >= -1 and orig[1] <= 1: 
            self.ctr = np.zeros( 2 , dtype=TYPE )
            self.ctr[0] = ( orig[0] + 0.5 ) * npix
            self.ctr[1] = ( orig[1] + 0.5 ) * npix
        else:
            sys.exit('''\nERROR: the ellipse origin should be placed in the square 
                         ( [ -1 , 1 ] , [ -1 , 1 ]  ) and then it gets rescaled
                         with the image size!\n''')

        ##  Rotation angle in radiants
        self.alpha = rotaz * np.pi / 180.0

        ##  Constant gray level inside the ellipse
        self.value = value

        ##  Size of the npix
        self.npix = npix


    ##  EQUATION OF THE ELLIPSE         
    def equation( self , x , y ):
        x0 = self.ctr[0]
        y0 = self.ctr[1]
        alpha = self.alpha
        a = self.a
        b = self.b
        
        return ( \
                 ( ( x - x0 ) * np.cos( alpha ) + ( y - y0 ) * np.sin( alpha ) ) * \
                     ( ( x - x0 ) * np.cos( alpha ) + ( y - y0 ) * np.sin( alpha ) ) / ( a * a ) + \
                 ( ( x - x0 ) * np.sin( alpha ) - ( y - y0 ) * np.cos( alpha )  ) * \
                     ( ( x - x0 ) * np.sin( alpha ) - ( y - y0 ) * np.cos( alpha ) ) / ( b * b ) \
               )           
        

    ##  FUNCTION TO COLOR THE ELLIPSE WITH ITS VALUE
    def color_ellipse( self , image ):
        if self.npix != image.shape[0]:
            sys.exit('''\nERROR: the ellipse does not belong to the selected image!\n''')   

        x = np.arange( self.npix )
        y = np.arange( self.npix )
        x , y = np.meshgrid( x , y )

        ind_ellipse_zero = np.argwhere( ( self.equation( x[:,:] , y[:,:] ) <= 1 ) )
        image[ind_ellipse_zero[:,0],ind_ellipse_zero[:,1]] += self.value


def lutSheppLogan( npix , nang ):
    ##  LUT is a 10 X 7 list, since the Shepp-Logan phantom consists of
    ##  10 ellipses and each of those ellipses is characterized by 6 numbers:
    ##  horizontal semi-axis , vertical semi-axis , origin abscissa , origin
    ##  ordinate , rotation angle , constant gray level of the ellipse ,
    ##  size of the entire image
    LUT = []

    ##  In order:
    ##  1st number  --->  horizontal ( if rotaz. is zero ) semi-axis
    ##  2nd number  --->  vertical ( if rotaz. is zero ) semi-axis
    ##  3rd number  --->  abscissa of the ellipse origin with respect to the image centre
    ##                    values can go from ( -1 , +1 )
    ##  4th number  --->  ordinate of the ellipse origin with respect to the image centre
    ##                    values can go from ( -1 , +1 ) 
    ##  5th number  --->  rotation angle
    ##  6th number  --->  constant gray level
    ##  7th number  --->  size of the entire image

    ##  Test ellipse   
    #ellipseTest = Ellipse( 0.6 , 0.3 , [ 0.1 , 0.1 ] , 0.0 , 1  , npix )
    #LUT.append( ellipseTest )

    ##  Big white ellipse in the background    
    ellipse1 = Ellipse( 0.92 , 0.69 , [ 0.0 , 0.0 ] , 90.0 , 2  , npix )
    LUT.append( ellipse1 )

    ##  Big gray ellipse in the background
    ellipse2 = Ellipse( 0.874 , 0.6624 , [ 0.0 , -0.0100 ] , 90.0 , -0.98 , npix )
    LUT.append( ellipse2 )     

    ##  Right black eye 
    ellipse3 = Ellipse( 0.31 , 0.11 , [ 0.11 , 0.0 ] , 72.0 , -1.02 , npix )
    LUT.append( ellipse3 )          

    ##  Left black eye 
    ellipse4 = Ellipse( 0.41 , 0.16 , [ -0.11 , 0.0 ] , 108.0 , -1.02 , npix ) 
    LUT.append( ellipse4 ) 
    
    ##  Big quasi-circle on the top
    ellipse5 = Ellipse( 0.25 , 0.21 , [ 0.0 , 0.17 ] , 90.0 , 0.4 , npix )
    LUT.append( ellipse5 )       

    ##   
    ellipse6 = Ellipse( 0.046 , 0.046 , [ 0.0 , 0.05 ] , 0.0 , 0.4 , npix ) 
    LUT.append( ellipse6 )

    ##   
    ellipse7 = Ellipse( 0.046 , 0.046 , [ 0.0 , -0.1 ] , 0.0 , 0.4 , npix ) 
    LUT.append( ellipse7 ) 
    
    ##  
    ellipse8 = Ellipse( 0.046 , 0.023 , [ -0.04 , -0.305 ] , 0.0 , 0.6 , npix )
    LUT.append( ellipse8 )       

    ##   
    ellipse9 = Ellipse( 0.023 , 0.023 , [ 0.0 , -0.305 ] , 0.0 , 0.6 , npix ) 
    LUT.append( ellipse9 )

    ##   
    ellipse10 = Ellipse( 0.046 , 0.023 , [ 0.03 , -0.305 ] , 90.0 , 0.6 , npix ) 
    LUT.append( ellipse10 )      

    return LUT         

def createSheppLogan( LUT , npix ):
    ##  Allocate memory for the phantom
    phantom = np.zeros( ( npix , npix ) , dtype=TYPE )

    ##  Draw Shepp-Logan
    num_ellipse = len( LUT )

    for i in range( num_ellipse ):
        LUT[i].color_ellipse( phantom )

    return phantom
###############################################################################

def approx_projection(prj, t):
    hw = (NR-1)*DR/2.0 
    if np.abs(np.round(t)-t) < 1e-10:
        t = np.round(t)
    if t >= hw or t <= -hw:
        return 0
    it1 = np.floor((t+hw)/DR)
    it2 = it1+1
    t1 = it1*DR - hw
    t2 = it2*DR - hw
    f = (t - t1)/(t2 - t1)
    return (1 - f) * prj[it1] + f * prj[it2]
    

def backprojection(sino):
    '''
    Smearing of projections over the image space
    Input: sino - sinogram
    '''
    bp =  np.zeros((NY, NX), dtype=TYPE)
    for i in np.arange(0,NX):
        for j in np.arange(0,NY):
            x = DR*i - (NX-1)*DR/2
            y = -DR*j + (NY+1)*DR/2
            for k in np.arange(NT):
                theta = k*np.pi/NT
                t = x*np.cos(theta) + y*np.sin(theta) 
                bp[j,i] += approx_projection(sino[k,:], t)   
    bp = bp*np.pi/NT
    bp[bp < 0] = 0
    return bp    

def backprojection_rot(sino):
    '''
    Smearing of projections over the image space based on usage of imrotate
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
    
    # impulse response of ramp filter
    h = ramp_filter()

    # create look-up-table of ellipses
    LUT = lutSheppLogan(NX , NT)

    # create Shepp-Logan phantom
    phantom = createSheppLogan(LUT, NX)
    plt.figure()
    plt.imshow(phantom, cmap=plt.cm.gray)
    plt.title('Shepp-Logan phantom')
    plt.show()
    
    # Radon transform -> sinogram
    thetas = np.arange(NT) * 180.0/NT;
    radphantom = np.transpose(radon(phantom, theta=thetas, circle=True))
    plt.figure()
    plt.imshow(radphantom, cmap=plt.cm.gray)
    plt.title('Radon transform of Shepp-Logan phantom')
    plt.show()
    
    '''
    # backprojection of original sinogram
    bp = backprojection(radphantom)
    plt.figure()
    plt.imshow(bp, cmap=plt.cm.gray)
    plt.title('Simple backprojection image')
    plt.show()
    '''
    # filtered sinogram
    fs = filter_sinogram(radphantom,h)
   
    # backprojection of filterd sinogram
    fbp = backprojection(fs)
    plt.figure()
    plt.imshow(fbp, cmap=plt.cm.gray)
    plt.title('Filtered backprojection reconstruction')
    plt.show()

    # backprojection of filterd sinogram
    fbp_rot = backprojection_rot(fs)
    plt.figure()
    plt.imshow(fbp_rot, cmap=plt.cm.gray)
    plt.title('Filtered backprojection (using imrotate) reconstruction')
    plt.show()

    # reconstruction using direct Fouerier method    
    dfm = direct_fourier_rec(radphantom)
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


    # reconstruction using built-in inverse Radon transform
    irs = iradon(np.transpose(phantom), theta=thetas, circle=True)
    plt.figure()
    plt.imshow(irs, cmap=plt.cm.gray)
    plt.title('Inverse Radon transform reconstruction')
    plt.show()