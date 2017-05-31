clear all
load('projection_data_z_0_25_g.mat')
mg = g;
load('data_zp25c.mat')
max(mg(:)-g(:))
min(mg(:)-g(:))
imshow(squeeze(mg(2,:,:)-g(2,:,:)),[])

clear all
load('projection_data_z_0_25_der.mat')
mg = g1;
load('data_zp25c_der.mat')
g = g1;
max(mg(:)-g(:))
min(mg(:)-g(:))
imshow(squeeze(mg(2,:,:)-g(2,:,:)),[])
clear g1

clear all
load('projection_data_z_0_25_g2.mat')
mg = g2;
load('data_zp25c_g2.mat')
g = g2;
max(mg(:)-g(:))
min(mg(:)-g(:))
imshow(squeeze(mg(2,:,:)-g(2,:,:)),[])
clear g2

clear all
load('projection_data_z_0_25_fhr.mat')
mg = g3;
load('data_zp25c_fhr.mat')
g = g3;
max(mg(:)-g(:))
min(mg(:)-g(:))
imshow(squeeze(mg(2,:,:)-g(2,:,:)),[])
clear g3

clear all
load('projection_data_z_0_25_ht.mat')
mg = g4;
load('data_zp25c_ht.mat')
g = g4;
max(mg(:)-g(:))
min(mg(:)-g(:))
imshow(squeeze(mg(2,:,:)-g(2,:,:)),[])
clear g4