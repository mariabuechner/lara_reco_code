'''
Displays centres of the detector pixels as blue dots,
displays center of the detector as red dot,
displays source position with green dot. 
Useful to figure out what delta (or in base.py shift_detector variable) means.
'''
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import matplotlib.pyplot as plt
from math import pi

def ray(D,alpha,w, eu, ev, ez):
    return D*np.cos(alpha)*ev + D*np.sin(alpha)*eu + w*ez

def cone_beam(D, H, Nr, Nc, delta, eu, ev, ez):
    dw = H/Nr
    da = dw/D
    alpha = np.linspace(-da/2*(Nc-1), da/2*(Nc-1), Nc) - delta[0]
    w = np.linspace(-dw/2*(Nr-1), dw/2*(Nr-1), Nr) - delta[1]
    
    theta = np.zeros((Nr,Nc,3)) # direction to the pixel's center
    for i in range(Nr):
        for j in range(Nc):
            theta[Nr-1-i,j,:] = ray(D, alpha[j], w[i], eu, ev, ez)
    return theta

if __name__ == '__main__':
    R = 1.0
    D = 2*R
    
    H = 0.5
    Nr = 4
    Nc = 4
    
    s = np.pi/4
    P = H
    ys = np.array([R*np.cos(s), R*np.sin(s), P/2/np.pi*s])
    
    eu = np.array([-np.sin(s), np.cos(s), 0]) 
    ev = np.array([-np.cos(s), -np.sin(s), 0])
    ez = np.array([0, 0, 1])
    
    dw = H/Nr
    da = dw/D
    delta = [0, 0]; # which point on detector will be hit by the ray defined 
                        # with alpha = 0 and w = 0
                        # delta = [0, 0] corresponds to heigh/2, width/2

    theta = cone_beam(D,H,Nr,Nc,delta,eu,ev,ez);     
    cd = np.zeros((Nr,Nc,3)) # coordinates of the pixel's center
    for i in range(Nr):
        for j in range(Nc):
            cd[Nr-1-i,j,:] = ys + theta[Nr-1-i,j,:]
    
    detector_center = (ys + np.array([D*np.cos(0.0)*ev + D*np.sin(0.0)*eu + 0*ez])).reshape(3,)
    source = ys.reshape(3,)
    x = cd[:,:,0].reshape(Nr*Nc,)
    y = cd[:,:,1].reshape(Nr*Nc,)
    z = cd[:,:,2].reshape(Nr*Nc,)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(x,y,z)
    ax.scatter(source[0], source[1], source[2], color='g')
    ax.scatter(detector_center[0], detector_center[1], detector_center[2], color='r')

    '''
    alphaep = da/2*Nc - delta[0]
    alphaen = -da/2*Nc - delta[0]
    wep = dw/2*Nr - delta[1] 
    wen = -dw/2*Nr - delta[1]

    plane1 = [[tuple(ys), \
               tuple(ys+ray(D,alphaep,wep,eu,ev,ez)),\
               tuple(ys+ray(D,alphaen,wep,eu,ev,ez)),\
               ]] 
    collection = Poly3DCollection(plane1, linewidths=0.5, alpha=0.5)
    face_color = [0.5, 0.5, 1] 
    collection.set_facecolor(face_color)
    ax.add_collection3d(collection)

    plane2 = [[tuple(ys+ray(D,alphaep,wen,eu,ev,ez)),\
               tuple(ys+ray(D,alphaen,wen,eu,ev,ez)),\
               tuple(ys), \
               ]] 
    collection = Poly3DCollection(plane2, linewidths=0.5, alpha=0.5)
    face_color = [0.5, 0.5, 1] 
    collection.set_facecolor(face_color)
    ax.add_collection3d(collection)

    plane3 = [[tuple(ys), \
               tuple(ys+ray(D,alphaep,wep,eu,ev,ez)),\
               tuple(ys+ray(D,alphaep,wen,eu,ev,ez)),\
               ]] 
    collection = Poly3DCollection(plane3, linewidths=0.5, alpha=0.5)
    face_color = [0.5, 0.5, 1] 
    collection.set_facecolor(face_color)
    ax.add_collection3d(collection)

    
    plane4 = [[tuple(ys), \
               tuple(ys+ray(D,alphaen,wep,eu,ev,ez)),\
               tuple(ys+ray(D,alphaen,wen,eu,ev,ez)),\
               ]] 
    collection = Poly3DCollection(plane4, linewidths=0.5, alpha=0.5)
    face_color = [0.5, 0.5, 1] 
    collection.set_facecolor(face_color)
    ax.add_collection3d(collection)
    '''
    plt.show()

    
