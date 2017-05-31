'''
    Simple test for intersection_unit_sphere function
    A vector from center of unit sphere to source point is rotated about z-axis
    Intersection with unit sphere for all rotated vector directions are 
    measured and plotted.         
'''
import sys
import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt

TYPE = 'float64'

def intersection_unit_sphere(z,theta):
    '''
    Input: 
        - z - vector from center of the sphere towards the source point
        - theta - direction of the x-ray, unit vector
    Output:
        - intersection length of x-ray passing thorough sphere 
    '''
    if abs(la.norm(theta)-1) > 0.01:
        sys.exit('''\nError: Theta should be unit vector.\n''')
    d = np.dot(z,theta)         # d - projection of z onto unit vector theta
    c = la.norm(z-d*theta)**2   # c - squared shortest distance of ray to unit 
                                # sphere center, c**2 =  norm(z)**2 - d**2 
    if c < 1:
        return 2*np.sqrt(1-c)
    else:
        return 0

if __name__ == '__main__':
    '''
        Simple test for intersection_unit_sphere function
    A vector from center of unit sphere to source point is rotated about z-axis
    Intersection with unit sphere for all rotated vector directions are 
    measured and plotted.         
    '''
    N = 500
    center = np.array([2,2,2], dtype=TYPE) # unit sphere center
    source = np.array([0,0,0], dtype=TYPE) # source position
    theta_c = center-source                # intiial direction of vector   
    theta_c = theta_c/la.norm(theta_c)     # conversion to unit vector

    alpha = np.linspace(0,2*np.pi,N)       # rotation angles about z-axis
    il = np.zeros((N,), dtype=TYPE)        # intersection length array
    
    for i in np.arange(N):
        # Arot_z - matrix for rotation for angle value alpha[i] about z-axis  
        Arot_z = np.array([[np.cos(alpha[i]), np.sin(alpha[i]), 0], \
                           [-np.sin(alpha[i]), np.cos(alpha[i]), 0],\
                           [0,0,1]], dtype=TYPE)
        theta = np.dot(theta_c, Arot_z)    # rotation of initial vector  
        il[i] = intersection_unit_sphere(source-center,theta)
    
    plt.figure()
    plt.plot(alpha,il)
    plt.xlabel('Angle')
    plt.ylabel('Intersection length')
    plt.show()
        
