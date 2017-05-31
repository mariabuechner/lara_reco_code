import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid.axislines import SubplotZero
from init_obj_unit_sphere import *
from att_coef_object import *
from math import cos, sin, tan, ceil, floor, sqrt, pi
from itertools import product, combinations
from numpy import where, delete, dot

TYPE = 'float64'

def index2alpha(i, p1, p2, b, d):
    return ((b + 1.0*i*d) - p2)/(p1 - p2)


def alpha2index(alpha, p1, p2, b, d):
    return (p2 + alpha*(p1-p2) - b)/d


def point(alpha, dp, sp):
    [p1x, p1y, p1z] = dp
    [p2x, p2y, p2z] = sp
    return [p2x + alpha*(p1x-p2x), 
            p2y + alpha*(p1y-p2y),
            p2z + alpha*(p1z-p2z)]

def entry_exit_point(Nx, Ny, Nz, dp, sp, bx, by, bz, dx, dy, dz):
        [p1x, p1y, p1z] = dp
        [p2x, p2y, p2z] = sp
        entry = []
        exitp = []
        alphax_min = min([index2alpha(Nx, p1x, p2x, bx, dx),\
                            index2alpha(0, p1x, p2x, bx, dx)])
        alphax_max = max([index2alpha(Nx, p1x, p2x, bx, dx),\
                            index2alpha(0, p1x, p2x, bx, dx)]) 
        alphay_min = min([index2alpha(Ny, p1y, p2y, by, dy),\
                            index2alpha(0, p1y, p2y, by, dy)])
        alphay_max = max([index2alpha(Ny, p1y, p2y, by, dy),\
                            index2alpha(0, p1y, p2y, by, dy)]) 
        alphaz_max = max([index2alpha(Nz, p1z, p2z, bz, dz),\
                            index2alpha(0, p1z, p2z, bz, dz)]) 
        alpha_min = max([alphax_min, alphay_min])
        alpha_max = min([alphax_max, alphay_max, alphaz_max])
        if alpha_max > alpha_min:
            entry = point(alpha_min, dp, sp)
            exitp = point(alpha_max, dp, sp)
        return entry, exitp

def min_max_plane_indices(Nx, Ny, Nz, dp, sp, bx, by, bz, dx, dy, dz):
    '''
    Returns indices of entering and exiting x and y-planes for ray connecting
    p1 and p2 points.
    Input: dp - detector point coordinates
           sp - source point coordinates
           p1x, p1y, p1z - coordinates of detector point
           p2x, p2y, p2z - coordinates of source point
           Nx, Ny, Nz - number of voxels
           dx, dy, dz - dimensions of voxels
           bx, by, bz - intersection of first x, y and z planes
    Output: imin - index of the first x-plane crossed by the ray
            imax - index of the last x-plane crossed by the ray
            jmin - index of the first y-plane crossed by the ray
            jmax - index of the last y-plane crossed by the ray
            kmin - index of the first z-plane crossed by the ray
            kmax - index of the last z-plane crossed by the ray
            alpha_min - alpha for entrance point into ROI
            alpha_max - alpha for exit point
    '''
    [p1x, p1y, p1z] = dp
    [p2x, p2y, p2z] = sp
    alphax_min = min([index2alpha(Nx, p1x, p2x, bx, dx),\
                        index2alpha(0, p1x, p2x, bx, dx)])
    alphax_max = max([index2alpha(Nx, p1x, p2x, bx, dx),\
                        index2alpha(0, p1x, p2x, bx, dx)]) 
    alphay_min = min([index2alpha(Ny, p1y, p2y, by, dy),\
                        index2alpha(0, p1y, p2y, by, dy)])
    alphay_max = max([index2alpha(Ny, p1y, p2y, by, dy),\
                        index2alpha(0, p1y, p2y, by, dy)]) 
    alphaz_min = min([index2alpha(Nz, p1z, p2z, bz, dz),\
                        index2alpha(0, p1z, p2z, bz, dz)])
    alphaz_max = max([index2alpha(Nz, p1z, p2z, bz, dz),\
                        index2alpha(0, p1z, p2z, bz, dz)]) 
    alpha_min = max([alphax_min, alphay_min])
    alpha_max = min([alphax_max, alphay_max, alphaz_max])
    imin = -1
    jmin = -1
    kmin = -1
    imax = -1
    jmax = -1
    kmax = -1
    if alpha_max > alpha_min:  
        if p1x < p2x :
            if alpha_min == alphax_min:
                imin = Nx
            else:
                imin = floor(alpha2index(alpha_min, p1x, p2x, bx, dx))

            if alpha_max == alphax_max:
                imax = 0
            else:
                imax = ceil(alpha2index(alpha_max, p1x, p2x, bx, dx))
        elif p1x > p2x:
            if alpha_max == alphax_max:
                imax = Nx
            else:
                imax = floor(alpha2index(alpha_max, p1x, p2x, bx, dx))

            if alpha_min == alphax_min:
                imin = 0
            else:
                imin = ceil(alpha2index(alpha_min, p1x, p2x, bx, dx))


        if p1y < p2y :
            if alpha_min == alphay_min:
                jmin = Nx
            else:
                jmin = floor(alpha2index(alpha_min, p1y, p2y, by, dy))

            if alpha_max == alphax_max:
                jmax = 0
            else:
                jmax = ceil(alpha2index(alpha_max, p1y, p2y, by, dy))
        elif p1y > p2y:
            if alpha_max == alphay_max:
                jmax = Nx
            else:
                jmax = floor(alpha2index(alpha_max, p1y, p2y, by, dy))

            if alpha_min == alphay_min:
                jmin = 0
            else:
                jmin = ceil(alpha2index(alpha_min, p1y, p2y, by, dy))

        if p1z < p2z :
            if alpha_min == alphaz_min:
                kmin = Nx
            else:
                kmin = floor(alpha2index(alpha_min, p1z, p2z, bz, dz))

            if alpha_max == alphaz_max:
                kmax = 0
            else:
                kmax = ceil(alpha2index(alpha_max, p1z, p2z, bz, dz))
        elif p1z > p2z:
            if alpha_max == alphaz_max:
                kmax = Nx
            else:
                kmax = floor(alpha2index(alpha_max, p1z, p2z, bz, dz))

            if alpha_min == alphaz_min:
                kmin = 0
            else:
                kmin = ceil(alpha2index(alpha_min, p1z, p2z, bz, dz))    
 
    return imin, imax, jmin, jmax, kmin, kmax, alpha_min, alpha_max

def find_intersection_alphas(dp, sp, bx, by, bz, dx, dy, dz, 
                             imin, imax, jmin, jmax, kmin, kmax, 
                             alpha_min, alpha_max):
    '''
    This function is finding alphas for points on voxel sides(edges, 
    planes), and returns sorted array of those values.
    Alpha parametrizes the source-detector line.
    Alpha = 0 => Source point
    Alpha = 1 => Detector point      
    Input: dp - detector point coordinates
           sp - source point coordinates
           dx, dy, dz - dimensions of voxels
           bx, by, bz - intersection of first x, y and z planes
           imin - index of the first x-plane crossed by the ray
           imax - index of the last x-plane crossed by the ray
           jmin - index of the first y-plane crossed by the ray
           jmax - index of the last y-plane crossed by the ray
           kmin - index of the first z-plane crossed by the ray
           kmax - index of the last z-plane crossed by the ray
           alpha_min - alpha for entrance point into ROI
           alpha_max - alpha for exit point
    Output: Sorted array of alpha values for points on the voxel's boarder
    '''
    [p1x, p1y, p1z] = dp
    [p2x, p2y, p2z] = sp
    # find alpha for indices in interval [ilo, ihi], [jlo,jhi] and [klo,khi]
    alpha_x = []
    alpha_y = []
    alpha_z = []
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
    if kmin != -1 and kmax != -1:
        klo = min([kmin, kmax])
        khi = max([kmin, kmax])
        alpha_z = [index2alpha(k, p1z, p2z, bz, dz) \
                    for k in np.arange(klo,khi+1)]

    # merge parameters without duplicates
    alpha_xyz = list(set(alpha_x)|set(alpha_y)|set(alpha_z))
    # sort parameters (ascending values) 
    alpha_xyz.sort()

    # remove alphas < alpha_min (alpha for the entrance of ROI) and 
    #        alphas > alpha_max (alpha for the exit of ROI)
    mask = where(np.array(alpha_xyz) < alpha_min)
    alpha_xyz = delete(alpha_xyz, mask)
    mask = where(np.array(alpha_xyz) > alpha_max)
    alpha_xyz = delete(alpha_xyz, mask)
    return alpha_xyz

def find_voxel_indices(alpha, dp, sp, dx, dy, dz, bx, by, bz):
    '''
    Returns closest voxel indices for a specified alpha on source-detectr line.
    Input: dp - detector point coordinates
           sp - source point coordinates
           dx, dy, dz - dimensions of voxels
           bx, by, bz - intersection of first x, y and z planes
    Output: voxel indices
    '''
    [p1x, p1y, p1z] = dp
    [p2x, p2y, p2z] = sp
    i = floor(alpha2index(alpha, p1x, p2x, bx, dx))
    j = floor(alpha2index(alpha, p1y, p2y, by, dy))
    k = floor(alpha2index(alpha, p1z, p2z, bz, dz))
    return i,j,k
 

def test_min_max_plane_indices(Nx, Ny, Nz, dx, dy, dz, R, r, dp, sp):

    ''' Used for testing developed functions on 4x4 object and 2x2 detector'''
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_aspect("auto")
    ax.set_autoscale_on(True)
    ax.set_xlim3d([-10, 10]) 
    ax.set_ylim3d([-10, 10]) 
    ax.set_zlim3d([-10, 10]) 
    ax.set_xlabel('x') 
    ax.set_ylabel('y') 
    ax.set_zlabel('z') 

    xc = np.linspace(-dx/2*(Nx-1), dx/2*(Nx-1), Nx) 
    yc = np.linspace(-dy/2*(Ny-1), dy/2*(Ny-1), Ny)  
    zc = np.linspace(-dz/2*(Nz-1), dz/2*(Nz-1), Nz) 
    px = [-dx/2, dx/2]
    for sl in np.arange(Nz):
        for row in np.arange(Nx):
            for col in np.arange(Ny):
                # object voxels
                for s, e in combinations(np.array(list(product(px,px,px))), 2):
                    if np.sum(np.abs(s-e)) == px[1]-px[0]:
                        s = s + [xc[row], yc[col], zc[sl]]
                        e = e + [xc[row], yc[col], zc[sl]]
                        ax.plot3D(*zip(s,e), color="b", linewidth=0.05)   
    
    # source point 
    #sp = [R, 0, 0] 
    # detector point
    #intersz = 3.0/4.0*dz
    #intersy = 5.0/4.0*dy
    #alpha =  tan(intersy/(R-r))
    #w = 2*R*intersz/(R-r)
    #dp = [-R*cos(2*alpha), R*sin(2*alpha), w]
    ax.plot3D(*zip(sp,dp), color="g")

    bx = -dx*Nx/2.0
    by = -dy*Ny/2.0
    bz = -dz*Nz/2.0 
    ab,ba = entry_exit_point(Nx, Ny, Nz, dp, sp, bx, by, bz, dx, dy, dz)
    if ab != [] or ba != []:
        ax.scatter(*zip(ab,ba), color="g", s=50)

    # ROI
    d = [-r, r]
    for s, e in combinations(np.array(list(product(d,d,d))), 2):
        if np.sum(np.abs(s-e)) == d[1]-d[0]:
            ax.plot3D(*zip(s,e), color="b")


    imin, imax, jmin, jmax, kmin, kmax, alpha_min, alpha_max = \
            min_max_plane_indices(Nx, Ny, Nz, dp, sp, bx, by, bz, dx, dy, dz)

    alphas = find_intersection_alphas(dp, sp, bx, by, bz, dx, dy, dz, 
                                      imin, imax, jmin, jmax, kmin, kmax,
                                      alpha_min, alpha_max)

    
    for ang in range(1, np.size(alphas)):
        ax.scatter(*zip(point(alphas[ang],dp,sp)), color="g", s=10)
        a = (alphas[ang]+alphas[ang-1])/2.0
        i, j, k = find_voxel_indices(a, dp, sp, dx, dy, dz, bx, by, bz)
        print 
        for s, e in combinations(np.array(list(product(px,px,px))), 2):
            if np.sum(np.abs(s-e)) == px[1]-px[0]:
                s = s + [xc[i], yc[j], zc[k]]
                e = e + [xc[i], yc[j], zc[k]]
                ax.plot3D(*zip(s,e), color="r", linewidth=0.5)   
    
    plt.show()

   
def siddon_xray_transform(obj, dp, sp, r, D, Nx, Ny, Nz, dx, dy, dz, 
                          bx, by, bz, refv, dpc=False):
    '''
    Computes x-ray transform for 3D object.
    Based on original Siddon algorithm for ray tracing.
    Input:  obj - phantom (voxel values = attenuation coef., absorbtion) or
                 gradient of phantom (voxel values = gradient value of
                     refraction ind. decrement in x, y, z direction, DPC) 
            dp - detector point coordinates
            sp - source point coordinates
            r - radius of the ROI
            D - curved detector radius 
            Nx, Ny, Nz - number of voxels
            dx, dy, dz - dimensions of voxels
            bx, by, bz - intersection of first x, y and z planes 
            refv - vector perpendicular to grating lines
            dpc - use attenuation coefficient (if False) or 
                    use refraction index decrement (if True) to compute x-ray 
                    transform            
    Output: prj - projection value for a single x-ray passing through object  
    '''
    if dpc:
       if len(obj.shape)!=4:
            sys.exit('''\nError: Object has to have 3 components: 
                                 - partial derivative in x-direction
                                 - partial derivative in y-direction
                                 - partial derivative in z-direction
                                 to compute cone beam projections for 
                                 phase contrast imaging. ''')

    # source - detector distance
    sdd = sqrt(sum((dp-sp)**2))
    prj = 0
    
    # find entrace and exit plane indices for the x-ray 
    imin, imax, jmin, jmax, kmin, kmax, alpha_min, alpha_max = \
            min_max_plane_indices(Nx, Ny, Nz, dp, sp, bx, by, bz, dx, dy, dz)

   
    # find alpha values for all planes intersected by x-ray
    alphas = find_intersection_alphas(dp, sp, bx, by, bz, dx, dy, dz, 
                                      imin, imax, jmin, jmax, kmin, kmax,
                                      alpha_min, alpha_max)      

    # compute voxel indices i,j, k for two consecutive alpha
    for m in range(1,len(alphas)):            
        a = (alphas[m] + alphas[m-1])/2.0
        i, j, k = find_voxel_indices(a, dp, sp, dx, dy, dz, bx, by, bz)
        # compute intersection length through voxel
        l = (alphas[m] - alphas[m-1])*sdd
        if not dpc:
            # for absorbtion, multiply intersection length with att. coef.
            prj += l*obj[i,j,k] 
        else:
            # for dpc, project gradient of ref.ind. dec onto refv, and multiply
            # with intersection length
            prj += l*(dot(obj[:,i,j,k],refv))
    return prj

def ray(D,alpha,w, eu, ev, ez):
    '''
    Ray is defined with impinging pixel on the detector 
    Input: D - radius of detector   
           alpha - angular coordinate of pixel's center
           w - coordinate of pixel's center along rotation-axis
           eu, ev, ez - rotated (source point) coordinate system
    Output: ray vector - connecting source point and detector point,
                         not a unit vector
    '''
    return D*np.cos(alpha)*ev + D*np.sin(alpha)*eu + w*ez

def siddon_cone_beam_projection(obj, sp, s, delta, alpha, w, Nr, Nc, r, D, 
                                Nx, Ny, Nz, dx, dy, dz, bx, by, bz, dpc=False):
    '''
    Computes 2D projection for cone beam x-ray source
    Input: obj - phantom (voxel values = attenuation coef., absorbtion) or
                 gradient of phantom (voxel values = gradient value of
                     refraction ind. decrement in x, y, z direction, DPC) 
           sp - array of source point position(s) 
           s - array of angular parameters for source point position(s)
           alpha - angular coordinate of pixel's center
           w - coordinate of pixel's center in direction 
                    parallel to rotation-axis
           Nr/Nc - number of detector rows/ columns  
           r - radius of the ROI
           D - curved detector's radius 
           Nx, Ny, Nz - number of voxels
           dx, dy, dz - dimensions of voxels
           bx, by, bz - intersection of first x, y and z planes  
           delta - shift between source point projection and center of 
                   detector 
           dpc - use attenuation coefficient (if False) or 
                   use refraction index decrement (if True) to compute x-ray 
                   transform
    Output: Df - 2d projection measurments collected at particular source point
                 position
    '''  
    if dpc:
       if len(obj.shape)!=4:
            sys.exit('''\nError: Object has to have 3 components: 
                            - partial derivative in x-direction
                            - partial derivative in y-direction
                            - partial derivative in z-direction
                            to compute cone beam projections for 
                            phase contrast imaging. ''')

    Np = len(s) # number of projections

    Df = np.zeros((Np,Nr,Nc), dtype=TYPE)
    for p in range(Np):
        print p
        # detector coordinate system
        eu = np.array([-sin(s[p]), cos(s[p]), 0], dtype=TYPE)
        ev = np.array([-cos(s[p]), -sin(s[p]), 0], dtype=TYPE)
        ez = np.array([0,0,1.0], dtype=TYPE)
        refv = [0, 0, 0]
        for i in range(Nr):
            for j in range(Nc): 
                theta = ray(D, alpha[j], w[i], eu, ev, ez)
                dp = theta + sp[:,p]
                if dpc:
                    # gradient is projected onto the vector which is 
                    # perpendicular to ray direction (theta) and gradient line
                    # direction (ez)
                    # refv - vector pointing towards x-ray refraction
                    #      - equal to np.cross(theta, ez)
                    theta = theta/sqrt(sum(theta**2))
                    refv = np.zeros_like(theta)
                    refv[0] = theta[1]
                    refv[1] = -theta[0] 
                    #print refv
                #print sqrt(sum((dp-sp[:,p])**2)), D
                #test_min_max_plane_indices(Nx, Ny, Nz, dx, dy, dz, R, 
                #                           r, dp, sp[:,p])
                Df[p,Nr-1-i,j] = siddon_xray_transform(obj, dp, sp[:,p], r, D, 
                                                       Nx, Ny, Nz, dx, dy, dz, 
                                                       bx, by, bz, refv, dpc)
    return Df
          
     
if __name__ == '__main__':

    # number of voxels for phantom 
    dpc = True
    Nx = 4
    Ny = 4
    Nz = 4
    # voxel size in cm
    dx = 2
    dy = 2
    dz = 2
    # intersection of first x-plane, first y-plane and z-plane
    bx = -dx*Nx/2.0
    by = -dy*Ny/2.0
    bz = -dz*Nz/2.0 
    r = dx*Nx/2.0
    # number of rows and columns for detector
    Nr = 2
    Nc = 2
    R = 10.00
    D = 2.0*R

    # test for rays passing through one column
    H = 10.00
    delta_w = H/Nr               # detector element height
    arc_alpha = delta_w*0.5       # detector element width (arc lenght)

    '''# test for rays exiting through z planes
    H = 25.00
    delta_w = H/Nr               # detector element height
    arc_alpha = delta_w*0.5       # detector element width (arc lenght)

    # testing for rays exiting through y planes
    H = 10.00
    delta_w = H/Nr               # detector element height
    arc_alpha = delta_w*3.0      # detector element width (arc lenght)'''

    delta_alpha = arc_alpha/D    # angular size of detector element
    shift_detector = [0, 0]      # shift between source point
                                 # projection and center of detector, 
                                 # [alpha_shift, w_shift]



    # detector grid
    # alpha - angular sampling on detector grid   
    alpha = np.linspace(-delta_alpha/2*(Nc-1), delta_alpha/2*(Nc-1), Nc)\
            - shift_detector[0]
    # w - z-axis sampling on detector grid
    w = np.linspace(-delta_w/2*(Nr-1), delta_w/2*(Nr-1), Nr)\
            - shift_detector[1]

    # phantom - unit sphere 
    obj = {}
    obj = init_obj_unit_sphere(obj,False) # object parameters
    xc = np.linspace(-dx/2*(Nx-1), dx/2*(Nx-1), Nx) 
    yc = np.linspace(-dy/2*(Ny-1), dy/2*(Ny-1), Ny)  
    zc = np.linspace(-dz/2*(Nz-1), dz/2*(Nz-1), Nz) 
    phantom = np.zeros((Nx,Ny,Nz), dtype=TYPE)
    for sl in np.arange(Nz):
        for row in np.arange(Nx):
            for col in np.arange(Ny):
                x = [xc[row], yc[col], zc[sl]]
                if (xc[row]**2 + yc[col]**2 + zc[sl]**2) <= r**2:
                    phantom[row,col,sl] = reconstruct_phantom(obj, x, dpc)


    '''plt.figure()
    extent = [-r/2.0, r/2.0, -r/2.0, r/2.0]
    res = plt.imshow((phantom[:,:,0]), cmap=plt.cm.gray, extent=extent, \
                     interpolation='none')
    plt.title('Phantom')
    plt.colorbar()
    plt.show()   '''


    # compute gradient of refraction index decrement 
    # centered derivatives
    gradPhantom = np.array(np.gradient(phantom, dx, dy, dz))

    # source point 
    P = 0.1
    s = np.array([0])
    sp = np.array([R*np.cos(s), R*np.sin(s), P/(2.0*np.pi)*s], dtype=TYPE)
    projection = siddon_cone_beam_projection(gradPhantom, sp, s, shift_detector, 
                                             alpha, w, Nr, Nc, r, D, Nx, Ny, Nz, dx, dy, 
                                             dz, bx, by, bz, dpc)
            
    




    
