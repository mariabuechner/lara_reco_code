import sys
from numpy import array, dot
from math import sqrt
#from init import *
from init_obj import *

TYPE = 'float64'    
    
def reconstruct_phantom(obj, x, phase=False):
    '''
    Computes attenuation coefficient value or refraction index decrement
    for point x inside the phantom.
    Input: obj - dictionary with parameters of phantom
           x - point in the FOV
           phase - use attenuation coefficient (if False) or 
                   use refraction index decrement (if True) 
    Output: res - attenuation coefficient value at point x  
    '''
    Nellipsis = len(obj['ellipse']['x_axis'])
    res = 0
    for i in range(Nellipsis):
        # attenuation coefficient or refraction index decrement
        if phase == False:
            mu = obj['ellipse']['attenuation_coeff'][i]
        else:
            mu = obj['ellipse']['refra_ind_decrement'][i]
        # invA - matrix operator for ellipse to unit sphere transform 
        invA = obj['ellipse']['inv_transformation_matrix'][i]
        # ellipse's center  
        center = array([obj['ellipse']['center_x'][i], \
                        obj['ellipse']['center_y'][i], \
                        obj['ellipse']['center_z'][i]], dtype=TYPE) 
        # vector from object point towards center of ellipse  
        z = center-x                 
        # transformed z
        Az = dot(z, invA)  
        if sqrt(sum(Az**2))<=1:
            res += mu 
    return res

if __name__ == '__main__':


    # object parameters
    obj = {}
    obj = init_obj(obj, plot=True)

    x = [0.06, -0.6, -0.25]
    fx = reconstruct_phantom(obj, x, True)
    print fx
