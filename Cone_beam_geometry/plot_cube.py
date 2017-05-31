from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from itertools import product, combinations
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_aspect("auto")
ax.set_autoscale_on(True)
ax.set_xlim3d([-10, 10]) 
ax.set_ylim3d([-10, 10]) 
ax.set_zlim3d([-10, 10]) 

Nx = 4
Ny = 4
Nz = 4
dx = 2
dy = 2
dz = 2
xc = np.linspace(-dx/2*(Nx-1), dx/2*(Nx-1), Nx) 
yc = np.linspace(-dy/2*(Ny-1), dy/2*(Ny-1), Ny)  
zc = np.linspace(-dz/2*(Nz-1), dz/2*(Nz-1), Nz) 
for sl in np.arange(Nz):
    for row in np.arange(Nx):
        for col in np.arange(Ny):
            # object voxels
            r = [-1, 1]
            for s, e in combinations(np.array(list(product(r,r,r))), 2):
                if np.sum(np.abs(s-e)) == r[1]-r[0]:
                    s = s + [xc[row], yc[col], zc[sl]]
                    e = e + [xc[row], yc[col], zc[sl]]
                    ax.plot3D(*zip(s,e), color="b", linewidth=0.2)


#dibujar punto
#ax.scatter([0],[0],[0],color="g",s=100)

# ROI
d = [-4, 4]
for s, e in combinations(np.array(list(product(d,d,d))), 2):
    if np.sum(np.abs(s-e)) == d[1]-d[0]:
        ax.plot3D(*zip(s,e), color="g")

plt.show()