from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from numpy import linalg as la
import numpy as np
from numpy import array
from math import sqrt

N = 4
r = 4
NTOT = N*5+5+1
SQRT2 = sqrt(2.0)
SQ = 0.25

def init_obj(obj, plot):
    '''
    Creates dictionary for breast phantom with parameters:
    x_axis - first half axes of ellipsoids [cm]
    y_axis - second half axes of ellipsoids[cm]
    z_axis - third half axes of ellipsoids [cm]
    attenuation_coeff - x-ray attenuation coefficient 
    refra_ind_decrement - refraction index decrement 
    center_x - x-coordinates for the centers of ellipsoids 
    center_y - y-coordinates for the centers of ellipsoids
    center_z - z-coordinates for the centers of ellipsoids
    alpha - rotation angles (in radians) around z-axis (normal axis, yaw)
    beta - rotation angles (in radians) around lateral axis
    transformation_matrix - transformation from unit sphere to ellipsoid 
    inv_transformation_matrix - tranformaton from ellipsoid to unit sphere
    '''
    obj.update({'ellipse':\
                {'x_axis':[4.0]+N*[0.026]+N*[0.020]+N*[0.016]+N*[0.25]+N*[0.125] +
                          [1.0, 0.5, 0.375, 0.25, 0.125],
                 'y_axis':[4.0]+N*[0.026]+N*[0.020]+N*[0.016]+N*[0.25]+N*[0.125] +
                          [1.0, 0.5, 0.375, 0.25, 0.125],                             
                 'z_axis':[4.0]+N*[0.026]+N*[0.020]+N*[0.016]+N*[0.25]+N*[0.125] +
                          [1.0, 0.5, 0.375, 0.25, 0.125],
                 'attenuation_coeff':
                          [4.81100e-12] + 3*N*[3.36003e-10-4.81100e-12] + 
                           (5+2*N)*[5.44598e-12-4.81100e-12],
                 'refra_ind_decrement': 
                          [1.0741e-7] + 3*N*[3.13070e-7-1.0741e-7] +
                           (5+2*N)*[1.12412e-7-1.0741e-7],
                 'center_x': [0.0]+list(-np.linspace(0.5,r/SQRT2-SQ,N))
                            +list(np.linspace(0.5,r/SQRT2-SQ,N))
                            +list(np.linspace(0.5,r/SQRT2-SQ,N))
                            +list(-np.linspace(0.5,r/SQRT2-SQ,N))
                            +list(-np.linspace(0.5,r-SQ,N))
                            + [0.0, 2.25, 0.0, 0.0, 0.0],
                 'center_y': [0.0]+list(np.linspace(0.5,r/SQRT2-SQ,N))
                            +list(np.linspace(0.5,r/SQRT2-SQ,N))
                            +list(-np.linspace(0.5,r/SQRT2-SQ,N))
                            +list(-np.linspace(0.5,r/SQRT2-SQ,N))
                            + N*[0.0]
                            + [2.25, 0, -2.5, -1.5, -1],
                 'center_z': [0.0]+N*[0.0]+N*[0.0]+N*[0.0]+N*[0.0]+N*[0.0]+5*[0.0],
                 'alpha': [0.0]+N*[0.0]+N*[0.0]+N*[0.0]+N*[0.0]+N*[0.0]+5*[0.0],
                 'beta': [0]*NTOT}})
    
    A = np.zeros((len(obj['ellipse']['x_axis']),3,3))
    invA = np.zeros((len(obj['ellipse']['x_axis']),3,3))
    for i in range(len(obj['ellipse']['x_axis'])):    
        angle_xy = obj['ellipse']['alpha'][i]
        angle_zxy = obj['ellipse']['beta'][i]
        radii = [obj['ellipse']['x_axis'][i], 
                 obj['ellipse']['y_axis'][i], 
                 obj['ellipse']['z_axis'][i]]
        Ascale = np.zeros((3,3))
        Ascale[:,0] = [radii[0], 0, 0]
        Ascale[:,1] = [0, radii[1], 0]
        Ascale[:,2] = [0, 0, radii[2]]        
        Arot_z = np.zeros((3,3))
        Arot_z[:,0] = [np.cos(angle_xy), -np.sin(angle_xy), 0]
        Arot_z[:,1] = [np.sin(angle_xy), np.cos(angle_xy),0]
        Arot_z[:,2] = [0,0,1]
        Arot_x = np.zeros((3,3))
        Arot_x[:,0] = [1,0,0]
        Arot_x[:,1] = [0, np.cos(angle_zxy), -np.sin(angle_zxy)]
        Arot_x[:,2] = [0, np.sin(angle_zxy), np.cos(angle_zxy)]
        Arot_y = np.zeros((3,3))
        Arot_y[:,1] = [0,1,0]
        Arot_y[:,0] = [np.cos(angle_zxy), 0, np.sin(angle_zxy)]
        Arot_y[:,2] = [-np.sin(angle_zxy), 0, np.cos(angle_zxy)]
        A[i,:,:] = np.dot(Ascale, np.dot(np.dot(Arot_x,Arot_y),Arot_z))
        invA[i,:,:] = la.inv(A[i,:,:])
    obj['ellipse'].update({'transformation_matrix': A})   
    obj['ellipse'].update({'inv_transformation_matrix': invA})

    if plot==True:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i in range(len(obj['ellipse']['x_axis'])):
            opacity = i*0.01+0.5
            center = [obj['ellipse']['center_x'][i], 
                      obj['ellipse']['center_y'][i], 
                      obj['ellipse']['center_z'][i]]            
            radii = [obj['ellipse']['x_axis'][i], 
                     obj['ellipse']['y_axis'][i], 
                     obj['ellipse']['z_axis'][i]]
            A = obj['ellipse']['transformation_matrix'][i]           
            alpha = np.linspace(0,2*np.pi,100)
            theta = np.linspace(0,np.pi,100)
            x= np.outer(np.cos(alpha), np.sin(theta))
            y =np.outer(np.sin(alpha), np.sin(theta))
            z =np.outer(np.ones_like(alpha), np.cos(theta))           

            for i in range(len(alpha)):
                for j in range(len(theta)):
                        temp = np.dot([x[i,j], y[i,j], z[i,j]], A)
                        x[i,j] = temp[0]+center[0]
                        y[i,j] = temp[1]+center[1]
                        z[i,j] = temp[2]+center[2]
            '''axes = np.array([[1,0.0,0.0],\
                            [0.0,1,0.0],\
                            [0.0,0.0,1]])
            #plot coordinate system axis
            for p in axes:
                X3 = np.linspace(-p[0], p[0], 100) + center[0]
                Y3 = np.linspace(-p[1], p[1], 100) + center[1]
                Z3 = np.linspace(-p[2], p[2], 100) + center[2]
                ax.plot(X3, Y3, Z3, color='r')
            # scale and rotate accordingly
            for i in range(len(axes)):
                axes[i] = np.dot(axes[i],A)
            # plot ellipsoid axes
            for p in axes:
                X3 = np.linspace(-p[0], p[0], 100) + center[0]
                Y3 = np.linspace(-p[1], p[1], 100) + center[1]
                Z3 = np.linspace(-p[2], p[2], 100) + center[2]
                ax.plot(X3, Y3, Z3, color='b')'''
            ax.plot_surface(x, y, z, color='k', linewidth=0, alpha=opacity)
            plt.show()                 
    return obj

if __name__ == '__main__':
    obj = {}
    obj = init_obj(obj, plot=True)
