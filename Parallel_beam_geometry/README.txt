ctproj.py - implementation of tasks 1-7 of ctproj.pdf

SheppLogan_reconstruction.py - uses Filippo's code to generate phantom
		             - uses built-in Radon trnasform to generate sinogram
			     - 2 types of backprojection functions: 
					1) basedon imrotate - same implementation as in ctproj.							
					2) for a given (x,y) point and projection angle theta, t = x*cos(theta) + y*sin(theta) is computed. For t values for which projections value is not known, linear interpolation was used to compute projection value. 
			     - reconstruction by direct implementation of Fouerier Slice Theorem
			     - reconstruction based on built-in inverse Radon transform  																
