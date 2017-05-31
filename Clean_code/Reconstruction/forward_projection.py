from mpl_toolkits.mplot3d import Axes3D
import sys
import numpy as np
from numpy import linalg as la
from init_obj import *
from math import cos, sin

TYPE = 'float64'

def intersection_unit_sphere(z,theta):
    '''
    Input: 
        - z - vector from source point towards center of ellipse
        - theta - direction of the x-ray, unit vector
    Output:
        - intersection length of x-ray passing through unit sphere 
    '''
    if abs(la.norm(theta)-1) > 0.01:
        sys.exit('''\nError: Theta should be unit vector.\n''')
    # d - projection of z onto unit vector theta
    d = np.dot(z,theta)    
    # c - squared shortest distance of a ray to unit sphere center
    # c**2 =  norm(z)**2 - d**2 
    c = la.norm(z-d*theta)**2              
    if c < 1:
        return 2*np.sqrt(1-c)
    else:
        return 0        
    
def xray_transform(obj, ys, theta, phase):
    '''
    Computes line integral for a ray passing through an object.
    First, the intersection lenght of a ray emanating from source point 
    ys at direction theta and object(ellipsoids) is computed. Then, the 
    intersection lengths are multiplied with relative attenuation 
    coefficients and summed up. 
    Input: obj - dictionary with parameters for ellipsoids
           ys - source point position
           theta - x-ray direction
           phase - use attenuation coefficient (if False) or 
                   use refraction index decrement (if True) to compute x-ray 
                   transform
    Output: res - returns line integral for a ray   
    '''
    # number of ellipsoids in the object
    Nellipsis = len(obj['ellipse']['x_axis'])
    res = 0
    for i in range(Nellipsis):
        # attenuation coefficient or refraction index decrement
        if phase == False:
            mu = obj['ellipse']['attenuation_coeff'][i]
        else:
            mu = obj['ellipse']['refra_ind_decrement'][i]
        # invA - matrix operator to transform ellipsoid into sphere 
        invA = obj['ellipse']['inv_transformation_matrix'][i]
        # ellipsoid's center  
        center = np.array([obj['ellipse']['center_x'][i], \
                           obj['ellipse']['center_y'][i], \
                           obj['ellipse']['center_z'][i]], dtype=TYPE) 
        # vector from source point towards center of ellipse  
        z = center-ys                      
        # transformed z
        Az = np.dot(z, invA)  
        # transformed theta     
        Atheta = np.dot(theta, invA)
        normAtheta = la.norm(Atheta)
        res += mu/normAtheta * \
               intersection_unit_sphere(Az,Atheta/normAtheta)  
    return res

def ray(D, alpha, w, eu, ev, ez):
    '''
    Curved detector function.
    Ray is defined with the pixel on the detector through which is passing 
    Input: D - radius of detector   
           alpha - angular coordinate of pixel's center
           w - coordinate of pixel's center along rotation-axis
           eu, ev, ez - rotated (source point) coordinate system
    Output: direction vector for a ray, not unit vector
    '''
    return D*cos(alpha)*ev + D*sin(alpha)*eu + w*ez

def ray_flat_detector(D, u, w, eu, ev, ez):
    '''
    Ray is defined with the pixel on the detector through which is passing 
    Input: D - radius of detector   
           u, w - coordinates of pixel's center for flat detector
           eu, ev, ez - rotated (source point) coordinate system
    Output: direction vector for a ray, not unit vector
    '''
    return D*ev + u*eu + w*ez

def cone_beam(da, dw, D, Nr, Nc, delta, eu, ev, ez):
    '''
    Cone beam is defined with directions of all rays going from the 
    source point and hitting the centers of detector elements.
    Input: da - detector element angular size
           dw - detector element height
           D - detector radius
           Nr - number of detector rows
           Nc - number of detector columns
           delta - shift between point of impinging x-ray with alpha=0, 
                   w=0 and the detector center
           eu, ev, ez - rotated coordinate system
           return_alpha_w - if True, alpha and w arrays will be returned 
                            together with theta array
    Output: theta - direction vectors for rays, unit vector
    '''
    # alpha - angles describing detector element position  
    alpha = np.linspace(-da/2*(Nc-1), da/2*(Nc-1), Nc) - delta[0]
    # w - describing detector element position in z-axis direction
    w = np.linspace(-dw/2*(Nr-1), dw/2*(Nr-1), Nr) - delta[1]
    # alpha=0 and w=0 corresponds to point where detector is touching 
    # the helix
    
    # direction vectors (from source point to the center of pixels)
    theta = np.zeros((Nr,Nc,3)) 
    for i in range(Nr):
        for j in range(Nc):
            theta[Nr-1-i,j,:] = ray(D, alpha[j], w[i], eu, ev, ez)
            theta[Nr-1-i,j,:] = theta[Nr-1-i,j,:]/la.norm(theta[Nr-1-i,j,:])
    return theta

def cone_beam_flat_detector(du, dw, D, Nr, Nc, delta, eu, ev, ez):
    '''
    Cone beam is defined with directions of all rays going from the 
    source point and hitting the centers of detector elements.
    Input: du, dw - detector element size (width, height)
           D - detector radius
           Nr - number of detector rows
           Nc - number of detector columns
           delta - shift between point of impinging x-ray with alpha=0, 
                   w=0 and the detector center
           eu, ev, ez - rotated coordinate system
           return_alpha_w - if True, alpha and w arrays will be returned 
                            together with theta array
    Output: theta - direction vectors for rays, unit vector
            alpha - fan-angles for detector columns
            w - positions for detector rows      
    '''
    # alpha - angles describing detector element position  
    # detector grid
    # u, w - detector grid coordinates    
    u = np.linspace(-du/2*(Nc-1), du/2*(Nc-1), Nc) - delta[0]
    w = np.linspace(-dw/2*(Nr-1), dw/2*(Nr-1), Nr) - delta[1]
    # u=0 and w=0 corresponds to point where detector is touching 
    # the helix
    
    # direction vectors (from source point to the center of pixels)
    theta = np.zeros((Nr,Nc,3)) 
    for i in range(Nr):
        for j in range(Nc):
            theta[Nr-1-i,j,:] = ray_flat_detector(D, u[j], w[i], eu, ev, ez)
            theta[Nr-1-i,j,:] = theta[Nr-1-i,j,:]/la.norm(theta[Nr-1-i,j,:])
    return theta

def cone_beam_projection(obj, conf, ys, s, delta, phase=False):
    '''
    Computes 2D projection for cone beam x-ray source
    Input: obj - dictionary with parameters for ellipsoids
           conf - dictionary with parameters for detector,...
           ys - all posible source point position
           s - all possible values for source point trajectory
           sp - s for source point position the 2D projection will be 
                computed
           delta - shift between source point projection and center of 
                   detector 
           phase - use attenuation coefficient (if False) or 
                   use refraction index decrement (if True) to compute x-ray 
                   transform
    Output: Df - 2d projection collected at particular source point
                 position
    '''
    
    # geometry parameters: detector parameters
    D = conf['detector']['DSD']
    H = conf['detector']['height']
    Nr = conf['detector']['number_rows']
    Nc = conf['detector']['number_columns'] 
    delta_w = conf['detector']['pixel_height']   # detector element height
    arc_alpha = conf['detector']['pixel_width']  # detector element width (arc) 
    delta_alpha = arc_alpha/D  # angular size of detector element
    Np = len(s)                # number of projections

    Df = np.zeros((Np,Nr,Nc), dtype=TYPE)
    for p in range(Np):
        print p
        # detector coordinate system
        eu = np.array([-sin(s[p]), cos(s[p]), 0], dtype=TYPE)
        ev = np.array([-cos(s[p]), -sin(s[p]), 0], dtype=TYPE)
        ez = np.array([0,0,1.0], dtype=TYPE)
        
        # theta - direction of all rays in the cone beam that hit detector   
        theta = cone_beam(delta_alpha, delta_w, D, Nr, Nc, delta,\
                                    eu, ev, ez)
        for i in range(Nr):
            for j in range(Nc): 
                Df[p,Nr-1-i,j] = xray_transform(obj, ys[:,p], theta[Nr-1-i,j],
                                                phase)
    return Df

def cone_beam_projection_flat_detector(obj, conf, ys, s, delta, phase=False):
    '''
    Computes 2D projection for cone beam x-ray source
    Input: obj - dictionary with parameters for ellipsoids
           conf - dictionary with parameters for detector,...
           ys - all posible source point position
           s - all possible values for source point trajectory
           sp - s for source point position the 2D projection will be 
                computed
           delta - shift between source point projection and center of 
                   detector 
           phase - use attenuation coefficient (if False) or 
                   use refraction index decrement (if True) to compute x-ray 
                   transform
    Output: Df - 2d projection collected at particular source point
                 position
    '''
    
    # geometry parameters: detector parameters
    D = conf['detector']['DSD']
    H = conf['detector']['height']
    Nr = conf['detector']['number_rows']
    Nc = conf['detector']['number_columns'] 
    dw = conf['detector']['pixel_height']   # detector element height
    du = conf['detector']['pixel_width']    # detector element width 
    Np = len(s)                # number of projections

    Df = np.zeros((Np,Nr,Nc), dtype=TYPE)
    for p in range(Np):
        print p
        # detector coordinate system
        eu = np.array([-sin(s[p]), cos(s[p]), 0], dtype=TYPE)
        ev = np.array([-cos(s[p]), -sin(s[p]), 0], dtype=TYPE)
        ez = np.array([0,0,1.0], dtype=TYPE)
        
        # theta - direction of all rays in the cone beam that hit detector   
        theta = cone_beam_flat_detector(du, dw, D, Nr, Nc, delta, eu, ev, ez)

        for i in range(Nr):
            for j in range(Nc): 
                Df[p,Nr-1-i,j] = xray_transform(obj, ys[:,p], theta[Nr-1-i,j], 
                                                phase)
    return Df

    
