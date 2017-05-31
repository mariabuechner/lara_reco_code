from mpl_toolkits.mplot3d import Axes3D
import sys
import numpy as np
from numpy import linalg as la
from init import *
from init_obj import *

TYPE = 'float64'

def object_surface_coordinates(obj, i):
    ''' 
    Used for plotting objects
    Input: obj - dictionary with parameters for ellipsoids
           i - index for ellipsoid
    '''    
    center = [obj['ellipse']['center_x'][i], \
              obj['ellipse']['center_y'][i], \
              obj['ellipse']['center_z'][i]]            
    A = obj['ellipse']['transformation_matrix'][i] 
    # 2 angular parameters           
    alpha = np.linspace(0,2*np.pi,100)
    theta = np.linspace(0,np.pi,100)
    # coordinates on the surface of unit sphere with center (0,0,0)
    x = np.outer(np.cos(alpha), np.sin(theta))
    y = np.outer(np.sin(alpha), np.sin(theta))
    z = np.outer(np.ones_like(alpha), np.cos(theta))           
    
    for i in range(len(alpha)):
        for j in range(len(theta)):
                # scaling and rotation operation
                temp = np.dot([x[i,j], y[i,j], z[i,j]], A) 
                # translation
                x[i,j] = temp[0]+center[0]
                y[i,j] = temp[1]+center[1]
                z[i,j] = temp[2]+center[2]
    return x,y,z

def intersection_unit_sphere(z,theta):
    '''
    Input: 
        - z - vector from source point towards center of ellipse
        - theta - direction of the x-ray, unit vector
    Output:
        - intersection length of x-ray passing thorough sphere 
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
def xray_transform(obj, ys, theta):
    '''
    Computes line integral for a ray passing through an object.
    First, the intersection lenght of a ray emanating from source point 
    ys at direction theta and object(ellipsoids) is computed. Then, the 
    intersection lengths are multiplied with relative attenuation 
    coefficients and summed up. 
    Input: obj - dictionary with parameters for ellipsoids
           ys - source point position
           theta - x-ray direction
    Output: res - returns line integral for a ray   
    '''
    # number of ellipsoids in the object
    Nellipsis = len(obj['ellipse']['x_axis'])
    res = 0
    for i in range(Nellipsis):
        # attenuation coefficient
        mu = obj['ellipse']['attenuation_coeff'][i]

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
    Input: D - radius of detector   
           alpha - fan-angle coordinate of pixel's center
           w - coordiante of pixel's center along z-axis
           eu, ev, ez - rotated (source point) coordinate system
    Output: direction vector for a ray, not unit vector
    '''
    return D*np.cos(alpha)*ev + D*np.sin(alpha)*eu + w*ez

def cone_beam(da, dw, Nr, Nc, delta, eu, ev, ez, return_alpha_w):
    '''
    Cone beam is defined with directions of all rays going from the 
    source point and hitting the centers of detector elements.
    Input: da - detector element angular size
           dw - detector element height
           Nr - number of rows
           Nc - number of columns
           delta - shift between point of impinging x-ray with alpha=0, 
                   w=0 and the detector center
           eu, ev, ez - rotated coordinate system
           return_alpha_w - if True, alpha and w arrays will be returned 
                            together with theta array
    Output: theta - direction vectors for rays, not unit vectors
            alpha - fan-angles for detector columns
            w - positions for detector rows      
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
    if return_alpha_w:
        return theta, alpha, w
    else:
        return theta

def forward_projection(obj, conf, sp, delta):
    '''
    Computes 2D projection for cone beam x-ray source
    Input: obj - dictionary with parameters for ellipsoids
           conf - dictionary with parameters for detector,...
           ys - all posible source point position
           s - all possible values for source point trajectory
           sp - s for source point position the 2D projection will be 
                computed
           delta - shift between source point projectin and center of 
                   detector 
    Output: Df - 2d projection collected at ys[:,sp] source point position
    '''
    # geometry parameters: detector parameters
    D = conf['curved_detector']['radius']
    H = conf['curved_detector']['height']
    Nr = conf['curved_detector']['number_rows']
    Nc = conf['curved_detector']['number_columns'] 
    delta_w = H/Nr             # detector element height
    arc_alpha = delta_w        # detector element width (arc lenght) !!!!!!!!!!!!!Check this with Maria
    delta_alpha = arc_alpha/D  # angular size of detector element

    # detector coordinate system
    eu = np.array([-np.sin(s[sp]), np.cos(s[sp]), 0], dtype=TYPE)
    ev = np.array([-np.cos(s[sp]), -np.sin(s[sp]), 0], dtype=TYPE)
    ez = np.array([0,0,1.0], dtype=TYPE)
    
    # theta - direction of all rays in the cone beam that hit detector   
    theta, alpha, w = cone_beam(delta_alpha, delta_w, Nr, Nc, delta,\
                                eu, ev, ez, True)
    
    # make 2D forward projection
    Df = np.zeros((Nr,Nc), dtype=TYPE)
    # coordinates of center of pixels
    cd = np.zeros((Nr,Nc,3)) 
    for i in range(Nr):
        for j in range(Nc): 
            cd[Nr-1-i,j,:] = ys[:,sp] + theta[Nr-1-i,j]
            Df[Nr-1-i,j] = xray_transform(obj, ys[:,sp], theta[Nr-1-i,j])
    return cd, Df
    
if __name__ == '__main__':

    # reference coordiante system
    i = np.array([1.0,0.0,0.0], dtype=TYPE)
    j = np.array([0.0,1.0,0.0], dtype=TYPE)
    k = np.array([0.0,0.0,1.0], dtype=TYPE)
    
    # geometry parameters: source trajectory and detector parameters
    conf = {}
    conf = init(conf)
    # object parameters
    obj = {}
    obj = init_obj(obj, plot=False)
    
    # geometry parameters: source trajectory
    R = conf['source_trajectory']['radius']
    P = conf['source_trajectory']['pitch']
    N = conf['source_trajectory']['number_turns']
    ds = conf['source_trajectory']['delta_s']
    D = conf['curved_detector']['radius']
    H = conf['curved_detector']['height']
    Nr = conf['curved_detector']['number_rows']
    Nc = conf['curved_detector']['number_columns'] 

    if D != 2*R:
        sys.exit('''\nError: Curved detector radius should be '''\
                +'''initializedto two times source trajectory '''\
                +'''radius!\n''')
    if np.mod(Nr,2):
        sys.exit('''\nError: Number of detector rows was taken'''\
                +''' to be even''')
    if np.mod(Nc,2):
        sys.exit('''\nError: Number of detector columns was taken'''\
                +''' to be even''')
    
    # parametrized source trajectory
    smin = 0
    smax = N*2*np.pi
    s = np.arange(smin,smax,ds)
    zmin = -1 
    ys = np.array([R*np.cos(s), R*np.sin(s), P/(2*np.pi)*(s)+zmin], dtype=TYPE)

    #2D forward projection
    # number of 2D projection
    Np = len(s)     
    Df = np.zeros((Np,Nr,Nc), dtype=TYPE)
    # coordinates of center of pixels
    cd = np.zeros((Nr,Nc,3))   
    for sp in [20]: # range(len(s))               
        # shift between center of the detector and point where cylinder of 
        # source trajectory is tangential to detector
        # delta - [fan-angle, w distance]
        #       - point where cylinder containing source trajectory is 
        #         tangential to detector
        delta = [0, 0] # delta = [0, 0] corresponds to heigh/2, width/2

        cd, Df[sp,:,:] = forward_projection(obj, conf, sp, delta)

    # detector coordinate system
    eu = np.array([-np.sin(s[sp]), np.cos(s[sp]), 0], dtype=TYPE)
    ev = np.array([-np.cos(s[sp]), -np.sin(s[sp]), 0], dtype=TYPE)
    ez = np.array([0,0,1.0], dtype=TYPE)
              
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # detector points
    detector_center = (ys[:,sp] + np.array([D*np.cos(0.0)*ev + D*np.sin(0.0)*eu + 0*ez])).reshape(3,)
    source = ys[:,sp].reshape(3,)
    xd = cd[:,:,0].reshape(Nr*Nc,)
    yd = cd[:,:,1].reshape(Nr*Nc,)
    zd = cd[:,:,2].reshape(Nr*Nc,)
    ax.scatter(xd,yd,zd)
    ax.scatter(source[0], source[1], source[2], color='g')
    ax.scatter(detector_center[0], detector_center[1], detector_center[2], color='r')  

    # object surface 
    for n in range(len(obj['ellipse']['x_axis'])):
        xo, yo, zo = object_surface_coordinates(obj, n) 
        ax.plot_surface(xo, yo, zo, linewidth=0, color='k', alpha=0.2)
    plt.show()

    plt.figure()
    plt.imshow(Df[sp,:,:],  cmap=plt.cm.gray, interpolation='none')
    plt.title('2D Projection')
    plt.colorbar()
    plt.show()
    
    sp = 47
    ys = ys[:,sp]
    print ys
    theta = np.array([D*np.cos(0.0)*ev + D*np.sin(0.0)*eu + 0*ez]).reshape(3,)
    print theta
    print xray_transform(obj, ys, theta)