'''
Old version of function init_scan_geometry(args) in base.py
'''
def init_v2(settings):
    # source trajectory (helix) parameters 
    settings.update({'source_trajectory':  
                        {'radius' : 3.0}})        

    # curved detector parameters
    settings.update({'curved_detector': \
                        {'radius': 6.0,\
                         'height':0.5,\
                         'number_rows':16,\
                         'number_columns':138}}) 

    # ROI parameters
    settings.update({'ROI': {'circle': True, \
                             'radius': 1.0,\
                             'NX': 256,\
                             'NY': 256}})     
    return settings