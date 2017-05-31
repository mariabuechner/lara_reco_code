Phaseprojection4_v4_2x.h5 - projections computed using ray tracing Siddon version 4 (partial gradient computation) from phantom(1024,1024,1024), with curved detector (each pixel was splitted on 4 smaller pixels)
Rec4_v4_2x.h5 - slice at x3=-0.25 reconstructed from projections in Phaseprojection4_v4_2x.h5 (2x Nr,Nc, /2 dew,deh, maxPitch(Nr/2,2*delta_w) <- ensure same pitch as for Nr,Nc, dew, deh settings)
Phaseprojection4_v4.h5 - generated from Phaseprojection4_v4_2x.h5 by averaging 4 pixel values. projections have 4 times less pixels.
Rec4_v4.h5 -slice at x3=-0.25 reconstructed from projections in Phaseprojection4_v4.h5 

