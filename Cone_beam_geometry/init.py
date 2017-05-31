'''
Old version of function init_scan_geometry(args) in base.py
'''
def init(settings):
    # source trajectory (helix) parameters 
    settings.update({'source_trajectory':  
                        {'radius' : 3.0,\
                         'pitch' : 0.5,\
                         'number_turns' : 2.0,\
                         'delta_s': 0.1}})        

    # curved detector parameters
    settings.update({'detector': \
                        {'DSD': 6.0,\
                         'height':0.5,\
                         'number_rows':32,\
                         'number_columns':64,
                         'pixel_width': 0.5/16,
                         'pixel_height': 0.5/16}}) 

    # ROI parameters
    settings.update({'ROI': \
                       {'extent_x': [-1, 1],\
                        'extent_y': [-1, 1],\
                        'extent_z': [-1, 1],\
                        'Nx_Ny_Nz': [256, 256, 256],\
                        'circle': True,\
                        'radius': 1}})     
    return settings