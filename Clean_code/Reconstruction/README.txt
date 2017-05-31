----------------------------- base.py and base_phase.py-----------------------------------------------------

Use base_phase.py to compute reconstructions of the object from its projections.
The difference between base.py and base_phase.[y comes from how they treat phase 
projection data.
 All the function were written to handle projections after differenctiation step. 
After that step, we have one projection less and one row/ column less. In base.py, 
we just delete single row/column and projection from the phase projection data, in 
base_phase.py we use whole phase projection data 
(no deletions). So, better use base_phase.py. 
For absoprtion recnstruction, there is no difference between the two files.



------------------------------------ Files needed -----------------------------------------------------------

base_phase.py or base.py 
	- contains all the reconstruction related function
init_obj.py
	- contains phanotm parameters (att. coef. valus, ref.ind. dec. values, center, radii)
	- can be found in "Objects" folder
att_coef_object.py
	- contains functions that compute phantom object value at point x
ray_tracing_Siddon_v4.py
	- contains functions for the numerical computation of forward projection 
forward_projection.py
	- contains functions for the analytical computation of forawrd projection
Phasephantom.h5 	
	- file generated using functions in "Generating_phantom" 
	- contains generated 3D phantom object, need only for phase contrast


Note: The absorption imaging will compute always the projections using analytical formula.		 



-------------------------------- Input parameters to change--------------------------------------------------

Familiarize with the parameters in functions getArgs()
Change only those parameters.

Recommendation: Generate forward projections for the phase contrast using the C code
		 in folder "Forward_projection_C".

If you run the phase forward projection generation in Python check that you have enough 
memory for storing phantom and the gradient of the phantom. Keep track of memory usage!
       

Note: To sample source trajectory, I was using formula H/Nr/r. 
	H - detector height
	Nr - number of detector rows
	r - radius of object extent
      

------------------------------------------- Output ----------------------------------------------------------

3 images: reconstructed slice, slice of original phantom, a cross section line

files: PIfile.h5 - file with PI interval start and stop values for each point in the slice
       xxxxphantom_slice.h5 - file with the computed slice of the phantom, xxxx is either phase or absorption.
       rec_XXXX_projection_file_name.h5 - reconstructed slice data  
