-----------------------------ray_tracing_Siddon_v4.cc 
- computes forward projections based on ray tracing algortihm


-----------------------------Files needed
ray_tracing_Siddon_v4.cc
3D_phantom_data.h5 - e.g PhasePhantom4.h5 - containig Shepp-Logan phantom

-----------------------------Input paramters to change:
- lines 16:22:
#define NX 1024           // number of phantom voxels in x-direction
#define NY 1024           // number of phantom voxels in y-direction
#define NZ 1024           // number of phantom voxels in z-direction
#define MAXZ_SLICE 180    // number of slices of phantom to load while computing 
			     a single projection
#define NP 1076           // number of projections
#define NC 276            // number of detector elements in a column
#define NR 32             // numebr of detector elements in a row


- all lines in the main function which have a comment in the line
	- for flat detector projections: use function siddon_cone_beam_projection_flat_detector 
	- for curved detector projections: use siddon_cone_beam_projection

Note: MAXZ_SLICE parameter can be found by using slice_range function in ray_tracing_Siddon_v4.py which returns kmin and kmax. 
MAXZ_SLICE=kmax-kmin+1, but round it to some slightly larger number to be sure you have all the slices that are needed to compute the projection.
E.g. if theoretically computed MAXZ_SLICE = 8, round it to 10.


-----------------------------Compile the script:
g++ ray_tracing_Siddon_v4.cc -o ray_tracing_Siddon.exe -mcmodel=large -O2 -std=c++0x -lm -lhdf5_cpp -lhdfls5

-----------------------------Run created executable:
./ray_tracing_Siddon.exe



-----------------------------Output files
siddon_projections_...file_name.h5 - file containig projection data




---------------------------------Note on memory usage

Note: if the phantom is 1024x1024x1024 with double values -> memory usage is ~ 8 Gb
      for the phase projections, from these phantom are created its derivatives in 3 directions -> 3*8Gb.
      Think that for compiling and running the c code, you'll need a computer with more than 8 Gb RAM.
      
      The implementation is not loading the whole phantom, but the 1024x1024xMAXZ_SLICE part, so it reduces 
      the memory usage. 