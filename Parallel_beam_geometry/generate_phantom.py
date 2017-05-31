import numpy as np
import matplotlib.pyplot as plt

TYPE = 'float64'

def phantom(Nx, Ny, xc, yc, a, b, ang, refInd, plot):
    phantom = np.zeros((Ny,Nx), dtype = TYPE)
    i = np.arange(-1.0,1.0, 2.0/Nx)
    j = np.transpose(np.arange(1.0,-1.0, -2.0/Ny))
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
        plt.show()
    
    return phantom


def projection(theta, xc, yc, a, b, fi, rho, Nr, dr): 
    prj = np.zeros((Nr,), dtype=TYPE)    
    angle = theta - fi
    c = (a * np.cos(angle))**2 + (b * np.sin(angle))**2 
    for r in np.arange(-(Nr-1)/2, (Nr+1)/2):
        t = r*dr - (xc*np.cos(theta) + yc*np.sin(theta))
        if c - t**2 >= 0:
            prj[r+(Nr-1)/2] = 2.0*rho*a*b/c * np.sqrt(c - t**2)
    return prj            


def analytical_radon(thetas, xc, yc, a, b, fi, rho, plot):   
    N = np.max([Nx, Ny])
    Nr = np.round(N * np.sqrt(2.0))
    # make Nr odd
    if np.mod(Nr,2) == 0:
        Nr = Nr+1
    dr = np.min(2.0/Nx,2.0/Ny)
    Nt = len(thetas)

    sino = np.zeros((Nt,Nr), dtype=TYPE)    
    for n in range(len(xc)):
        for j in range(Nt):
            sino[j,:] += projection(thetas[j], xc[n], yc[n], a[n], b[n], \
                                    fi[n], rho[n], Nr, dr)

    if plot == True:
        plt.figure()
        plt.imshow(sino,  cmap=plt.cm.gray, interpolation='none')
        plt.title('Sinogram')
        plt.show()

    return sino



if __name__ == '__main__':
    
    Nx = 128
    Ny = 128
    
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
    Testing unit:
    xc = [0.5]
    yc = [0.5]
    a = [0.5]
    b = [0.1]
    ang = [np.pi/4]
    relRefInd = [1]
    '''

    thetas = np.arange(0.0, np.pi, np.pi/100)
    phantom = phantom(Nx, Ny, xc, yc, a, b, ang, relRefInd, plot=True)
    sino = analytical_radon(thetas, xc, yc, a, b, ang, relRefInd, plot=True)


 
    