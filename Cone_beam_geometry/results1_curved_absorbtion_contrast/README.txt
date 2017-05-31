projection4_v4_2x.h5 - absorption projections computed using ray tracing Siddon version 4 from phantom(1024,1024,1024), with curved detector (each pixel was splitted on 4 smaller elements)
abs_rec4_v4_2x.h5 - slice at x3=-0.25 reconstructed from projections in projection4_v4_2x.h5 (2x Nr,Nc, /2 dew,deh, maxPitch(Nr/2,2*delta_w) <- ensure same pitch as for Nr,Nc, dew, deh settings)

projection4_v4.h5 - generated from projection4_v4_2x.h5 by averaging 4 pixel values. Projections have 4 times less pixels.
abs_rec4_v4.h5 -slice at x3=-0.25 reconstructed from projections in projection4_v4.h5

analytical_projection4_2x.h5 - absorption projections computed analytically with curved detector with 4 times bigger the number of elements
analytic_abs_rec4_2x.h5 - slice at x3=-0.25 reconstructed from projections in analytical_projection4_2x.h5 (2x Nr,Nc, /2 dew,deh, maxPitch(Nr/2,2*delta_w) <- ensure same pitch as for Nr,Nc, dew, deh settings) 
 
analytical_projection4.h5 - absorption projections from analytical_projection4_2x.h5 by averaging
analytic_abs_rec4.h5 - -slice at x3=-0.25 reconstructed from projections in analytical_projection4.h5

Note: Prominent aftifacts in abs_rec4_v4_2x, not observed in phase testing. Reason: coming from the difference in forward projection phantom values and forward projection.
