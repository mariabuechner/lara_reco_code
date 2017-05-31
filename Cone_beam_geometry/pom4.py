'''
Some plot commands.
'''
import h5py
import numpy as np
import Image


projFileName = 'absorption_384_385_Temp.h5'
projf = h5py.File(projFileName, 'r')  
Dfc = projf['phantom'][:]
projf.close() 

projFileName = 'analytical_projection4_2x.h5'
projf = h5py.File(projFileName, 'r')  
Dfa = projf['Df'][:]
projf.close() 

projFileName = 'projection4_v4_2x.h5'
projf = h5py.File(projFileName, 'r')  
Dfp = projf['Df'][:]
projf.close() 

im = Image.fromarray(Dfc)
im.save('raw_analytic_abs_rec4_2x.tif')


plt.figure()
res = plt.imshow(Dfc[:,:,0], cmap=plt.cm.gray, \
                    interpolation='none')
plt.colorbar()
plt.show()

plt.figure()
res = plt.imshow(Dfa[0,:,:], cmap=plt.cm.gray, \
                    interpolation='none')
plt.colorbar()
plt.show()

plt.figure()
res = plt.imshow(Dfp[0,:,:]-Dfc[0,:,:], cmap=plt.cm.gray, \
                    interpolation='none')
plt.colorbar()
plt.show()
  
Nr = 2*16
Nrp= 8*Nr
Nc = 2*138
Ncp = 8*Nc
Np = size(Dfc,0)
dy = 0.5/16
dyp = 0.5/16/2

def resample_detector(Df2x, Np, Nrp, Ncp, Nr, Nc):
    factor =int(Nrp/Nr)
    Df = np.zeros((Np, Nr, Nc))
    for p in range(Np):
        print p
        for i in range(Nr):
            for j in range(Nc):
                Df[p,i,j] = sum(Df2x[p, factor*i:factor*i+factor,factor*j:factor*j+factor])/factor**2
    return Df


def differentiation_detector_row(Df2x, Np, Nrp, Ncp, dyp):
    diffDf = np.zeros((Np, Nrp-1, Ncp))
    diffDf = (Df2x[:,:,1:] - Df2x[:,:,:-1]) / dyp
    return diffDf

def average_detector_column(diffDf, Np, Nr, Nc):
    Df = np.zeros((Np, Nr, Nc))
    temp = (diffDf[:,1:,:]+diffDf[:,:-1,:])/2.0
    Df = temp[:,::2,::2]
    return Df
    
projPhaseFileName = 'df_1024_2x_compressed.h5'
projf = h5py.File(projPhaseFileName, 'w')  
projf.create_dataset('Df', data=Dfc, compression='gzip', compression_opts=9)
projf.close()  

projPhaseFileName = 'df_256_2x_from_16x.h5'
projf = h5py.File(projPhaseFileName, 'w')  
projf.create_dataset('Df', data=Df, compression='gzip', compression_opts=9)
projf.close()  

projFileName = 'rec_df_256_2x_from_4x.h5'
projf = h5py.File(projFileName, 'r') 
rec_slice = projf['rec_slice'][:] 
#projf.create_dataset('rec_slice', data=rec_slice1, compression='gzip', compression_opts=9)
projf.close()  

projFileName = 'BreastPhantom_slice.h5'
projf = h5py.File(projFileName, 'r')  
phantom = projf['phantom'][:]
projf.close() 

plt.figure()
res = plt.imshow(rec_slice, cmap=plt.cm.gray, \
                    interpolation='none')
res.set_clim(0.0, 0.07)
plt.colorbar()
plt.show()

plt.figure()
plt. plot(rec_slice[:,138])
plt.hold(True)
plt.plot(phantom[:,138])
plt.title('Slice cross section in y direction')
plt.ylabel('Rec. Att. Coef. Value')
plt.xlabel('y')
plt.show()    