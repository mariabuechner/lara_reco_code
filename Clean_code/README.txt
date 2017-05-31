------------------------------ Test reconstruction---------------------------------
-----------------------------------------------------------------------------------


-------------------------Curved detector phase reconstruction----------------------

Make a copy of "Reconstruction" folder.
Copy projection data files from "Forward_projection_C/Forward_projection_results"
 folder into "Reconstruction" folder copy. Run the base_phase.py script with 
default parameters.



-------------------------Flat detector phase reconstruction------------------------

Make a copy of "Reconstruction" folder.
Copy projection data files from "Forward_projection_C/Forward_projection_results"
 folder into "Reconstruction" folder copy. In the base_phase.py script change the
 projection data file name and detector type paramater, and run it again.



-------------------------Curved detector absorption reconstruction-----------------

Make a copy of "Reconstruction" folder.
In the base_phase.py script, change the projection data file name, set the flag to
 use projection data stored in a file to false, set the detector type to curved, 
and imaging contrast type to absorption. Run the script.

Note: This will take more time because the forward projections are first computed.




-------------------------Flat detector absorption reconstruction-------------------

Make a copy of "Reconstruction" folder.
In the base_phase.py script, change the projection data file name, set the flag to 
use file to false, set the detector type to flat, and imaging contrast type to 
absorption. Run the script.

Note: This will take more time because the forward porjections are first computed.







 
