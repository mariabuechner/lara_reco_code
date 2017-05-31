To generate 3D phantom data folowing files have to be located in the same folder:
- generate_phantom_volumes.py
- att_coef_object.py
- init_obj.py - choose one file from folder "Objects" and rename it to init_obj.py

In generate_phantom_volumes.py change:
- line 13  - specify file name 
- line 25  - 2*r is the width of one side of the cube containing phantom volume
- line 26  - specify phantom type (phase or absorption)
- lines 29-31 - specify number of voxels in the cube 


 