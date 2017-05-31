    '''
    Some plotting commands. 
    '''    

    temp = loadmat('reconstructed_slice_rho1.mat')
    rec_slice= temp['res_rho']   
 
    plt.figure()
    extent = [yc[0], yc[-1], xc[0], xc[-1]]
    res = plt.imshow(rec_slice, cmap=plt.cm.gray, extent=extent, interpolation='none', origin='lower')
    res.set_clim(0.0, 0.07)
    plt.title('Reconstructed slice')
    plt.xlabel('y')
    plt.ylabel('x')
    plt.colorbar()
    plt.show()

    temp = loadmat('reconstructed_slice_rho2.mat')
    rec_rho= temp['res_rho']   
 
    plt.figure()
    extent = [yc[0], yc[-1], xc[0], xc[-1]]
    res = plt.imshow(rec_rho, cmap=plt.cm.gray, extent=extent, interpolation='none', origin='lower')
    res.set_clim(0.0, 0.07)
    plt.title('Reconstructed slice')
    plt.xlabel('y')
    plt.ylabel('x')
    plt.colorbar()
    plt.show()


    plt.figure()
    res = plt.imshow(np.log(np.abs(rec_rho-rec_slice)), cmap=plt.cm.gray, extent=extent, interpolation='none', origin='lower')
    #res.set_clim(0.0, 0.07)
    plt.title('Reconstructed slices diff.')
    plt.xlabel('y')
    plt.ylabel('x')
    plt.colorbar()
    plt.show()


    plt.figure()
    extent = [yc[0], yc[-1], xc[0], xc[-1]]
    res = plt.imshow(phantom, cmap=plt.cm.gray, extent=extent, interpolation='none', origin='lower')
    res.set_clim(0.0, 0.07)
    plt.title('Phantom')
    plt.xlabel('y')
    plt.ylabel('x')
    plt.colorbar()
    plt.show()


    plt.figure()
    res = plt.imshow(np.log(np.abs(rec_rho-phantom)), cmap=plt.cm.gray, extent=extent, interpolation='none', origin='lower')
    #res.set_clim(0.0, 0.07)
    plt.title('Reconstructed slice - phantom diff.')
    plt.xlabel('y')
    plt.ylabel('x')
    plt.colorbar()
    plt.show()

    plt.figure()
    res = plt.imshow(np.log(np.abs(rec_slice-phantom)), cmap=plt.cm.gray, extent=extent, interpolation='none', origin='lower')
    #res.set_clim(0.0, 0.07)
    plt.title('Reconstructed slice-phantom diff')
    plt.xlabel('y')
    plt.ylabel('x')
    plt.colorbar()
    plt.show()