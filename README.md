--------
##Proton Computed Tomography (pCT) Image Reconstruction (release v1.0)
--------
####This program expects the coordinates of individual protons as they pass through the pair of silicon strip detectors upstream and downstream from the scanned object along with the calibrated Water Equivalent Path Length (WEPL) calculated from the ADC measurements acquired from the 5-stage scintillator.  The angle of the scanned object's rotating stage at the time each proton passes through the system is expected to be **(1)** an integer in the case of discrete rotations or **(2)** a floating point value calculated from the stage's rotational rate and each proton's time stamp.          
-------------------------------------------------------------------------------------
###__Instructions:__
-------------------------------------------------------------------------------------
_________________
__Accessing Tardis:__  
_________________
**(1)** login to Kodiak as usual.    
**(2)** ssh to one of Tardis' GPU compute nodes, n=3, 4, or 5, using `ssh ecsn00n` (__Kodiak is not a GPU cluster__).    
______________
__Data and Code:__   
______________
**(1)** Copy the input data to the compute node's local solid state drive, either to `/local/pCT_data/organized_data` if it is organized according to the standardized naming/organizational scheme or `/local/pCT_data/temp_input_data/<username>` if  it is unorganized.  Similarly, if the output data isn't going to be organized, it should be written to the user's `/local/pCT_data/reconstruction_data/temp_output_data/<username>` subdirectory. 
**(2)** Each user should have a directory `/local/pCT_code/Reconstruction/<username>` created for them to store their code, but new user directories can be created by first executing the `load_pct_function.sh` script (available as part of the `pCT_Tools` repository of the `pCT-collaboration` GitHub account) and then executing the `create_recon_user` function loaded by this script (`create_recon_user -h` for function help) to add a new user directory.  
**(3)** This repository can then be cloned to the user's code directory.  Make sure to enter `git checkout release_development` to switch to this branch before continuing.  There is also an `add_rcode_repo` function that makes it very easy to clone this and other reconstruction program repositories without much user input.
**(4)** To specify the required C++/ and CUDA compilers, their corresponding modules must be loaded by entering the commands `module load gcc/4.9.2` and `module load cuda70/toolkit`.  Additional CUDA modules which may optionally be loaded are `cuda70/blas`, `cuda70/fft`, `cuda70/gdk`, `cuda70/nsight`, and `cuda70/profiler`.  TO view the currently loaded modules, enter the `module list` command.

__________________
__Compiling/Executing Program:__   
__________________
**(1)** The input/output data parent directories and reconstruction specific subdirectories must be set by the user by modifying the `INPUT_DIRECTORY`, `OUTPUT_DIRECTORY`, `INPUT_FOLDER`, and `OUTPUT_FOLDER` variables in the header `pCT_Reconstruction_Data_Segments_Blake.h`.  The other program parameters are also specified by setting the values of the corresponding variables in the header as well.
**(2)** The Makefile in this repository is already setup to compile/run the program on Tardis and can be executed by entering the command `make run` (or simply `make` to compile only).  However, users can manually compile and run the code separately by entering the commands:

#####`nvcc -std=c++11 -gencode arch=compute_35,code=sm_35 -O3 pCT_Reconstruction_Data_Segments_Blake.cu -o recon.out`
#####`./recon.out`

-------------------------------------------------------------------------------------
###__Data and Execution Details:__
-------------------------------------------------------------------------------------
__________________
__Input Data:__    
__________________
It is much faster to read/write data to a compute node's local solid-state drive than across the network to/from the network attached storage (NAS) device, so input data is copied to the compute node and the program's output data/images are written to this local drive.  The data in `/ion` on the NAS drive is synchronized daily and copied to Tardis' master node (WHartnell:*/ro-sync/ion*) so the `rsync` file copying tool can quickly copy input data to the compute nodes and fix/update the local copy of the data when necessary using the Infiniband network link between WHartnell (172.30.10.1) and each compute node (172.30.10.x).  If the input data was recently uploaded and has not yet been synchronized, this will revert to a standard network data copy.  
__________________
__Data Management:__   
__________________
In addition to the standard reconstruction output data/images, the program also creates a text file with a list of important program parameters and their values for the current execution.  This same information is also appended to a global execution log (.csv file), providing users with a list of every reconstruction previously performed, the location and properties of the input data used and output data generated, the GitHub account/repository/branch/commit corresponding to the executed code, and the parameter values used in each case.  A copy of the actual code executed is also included with the output data/images for debugging/reproduction purposes.  

If organized input data is used, all of the output files are automatically named/organized and copied to the NAS drive so they are immediately available for analysis and transfer to an external compute system.  Otherwise, these data and code file management/copying tasks must be performed manually.  By default, the input data parent directory is `/local/pCT_data/organized_data` and the output data parent directory is `/local/pCT_data/reconstruction_data` and when organized input/output data is used, their corresponding parameters do not need to be changed (`input_directory` and `output_directory`).  The input/output folders are defined according to the phantom name, scan type/date/properties, and preprocessing/reconstruction dates and for organized data, the 2 are identical. 

----------
####__Reconstruction Program Details:__
-------------------------------------------------------------------------
######__Reconstruction Process, Phase 1: Import, process, and prepare input projection data for reconstruction__
___________________________
**(1)** Check to see if the required input projection data exists on the compute node's local drive and if it does not, copy the data from the permanent storage directory.  		   
**(2)** Initialize arrays/vectors for storage of input data and intermediate data generated during execution and initiate timer used to record the program's computation time.  		 	 	 
**(3)** Create a uniquely named output data directory on the compute node's local drive and in the permanent data storage directory. 						              
**(4)** Import portion of input projection data from disk, potentially from multiple files (gantry angles) simultaneously, and determine if **BOTH** the entry **AND** the exit paths of each proton intersect the reconstruction volume and, hence, pass through it.  
**(5)** For protons that pass through the reconstruction volume, accumulate entry/exit coordinates/angles of this intersection and their WEPL, gantry angle, and bin number.  
**(6)** Perform statistical analysis of each bin (mean/std dev) and remove histories with WEPL or xy/xz angle greater than 3 std devs from mean.  
**(7)** Perform hull-detection  (SC, MSC, SM, and/or FBP).  
**(8)** Perform any image filtering (2D/3D median/average) or data processing (WEPL distributions, radiographs, etc.) specified in *config* file.  
**(9)** Write the data/images specified in *config* file but not required for reconstruction to disk.  
**(10)** Choose the hull and perform the specified method for constructing the initial iterate and write these to `hull.txt` and `x_0.txt`, respectively.  
**(11)** Using the corresponding hull and the specified data transfer method, determine if **BOTH** the entry **AND** the exit paths of each proton intersect the hull and, thus, pass through it.  
**(12)** For protons that pass through the hull, determine their entry/exit coordinates/angles and update the values found in `(5)` to these, record their entry *x*/*y*/*z* voxel, and remove all data found in `(5)` for protons that do not pass through the hull.   
___________________________
######**Reconstruction Process, Phase 2: Perform image reconstruction**
___________________________
**(1)** Allocate GPU arrays for the MLP and WEPL data used in reconstruction as well as images such as the hull and the intermediate data and output images generated during reconstruction.  		   	
**(2)** Using the specified data transfer method, use the hull entry/ecxit information to calculate the most likely path (MLP) for each proton and use the specified feasibility seeking algorithm to calculate the update value for each voxel.  	 	  	
**(3)** Optionally attenuate updates applied to voxels using specified s-curve function.	 	 	 	 	 	 	 
**(4)** Before or after each iteration, optionally perform total variation superiorization (TVS) before/after each iteration of feasibility seeking algorithm
**(5)** After each iteration, write the image (*x*) to disk as `x_k.txt`, where *k* denotes the iteration number.  
**(6)** After each iteration, optionally apply a 2D/3D median/average filter with specified radius *r*, writing the resulting image to disk as `x_k_xxx_xx_rx`, where `xxx` is *med* or *avg*, `xx` is *2D* or *3D*, and the last `x` specifies the filter width *w = 2r+1* corresponding to the filter radius *r* specified in the *configurations* file.  
**(7)** After reconstruction is complete, the output data/image file permissions are changed to provide everyone with read access and then the output data written to the compute node's local drive is copied to permanent storage.	 	 
**(8)** Write the value of the option/parameter variables used to control program behavior and effect reconstructed images along with the execution timing information on various portions of the program 

-------------------------------------------------------------------------
####__Reconstruction approaches/techniques, algorithms, and implementations:__
-------------------------------------------------------------------------
Data/task parallelism is inherent in nearly every aspect of preprocessing and reconstruction and has been exploited (except for MLP and image reconstruction) using GPGPU programming (CUDA).  The host is primarily used for disk/user IO, manage host/GPU data allocations/transfers, and to configure/launch GPU kernels. 

The program also expects the file/folder naming and organizational scheme to adhere to the pCT data format defined in the pCT documentation.  The organizational scheme encodes the properties of the data acquisition process (dates and scanned object) and the various data dependencies, information that is extracted and used by the program.  Each time the program is executed, this information is used to locate the desired input `projection_xxx.bin' data, automatically create/name the output data directories where the output data/images are to be written, and an entry in an execution log is added which  records the input/output data/image information and program options/settings associated with the reconstruction.  

If the input data is maintained in the proper pCT Data format, the user need only modify the header file to specify the desired program behavior and execute the program; the reading of the input projection data and all output data/images and execution information (execution times, option/parameter values, etc.) are performed automatically.  While testing new algorithms and other changes/additions, the output folder may be named to encode the relevant test settings/options/parameters for easier analysis, but such data is not intended for shared data directories, so this does not cause any issues.   

The pCT data directory's hierarchy of subdirectories not only provides a consistent organizational scheme for the collaboration and encodes the details of the data acquisition, it also establishes data dependencies for at each stage of reconstruction.  The results at each stage are in a subdirectory of a folder named by date, providing the ability to recreate results by acquiring the associated version of each program from GitHub.  This also makes it possible to isolate and assess the effects of a particular development and compare to previous results by ensuring that the other programs used to generate the previous data are the same.  With multiple programs in the reconstruction process, it is important to be able to perform such isolated comparisons, particularly when assessing the way changes in early stages of reconstruction propagate and ultimately effect the reconstructed images.  Such propagation effects are extremely important and have been largely absent in the past and although it is infeasible to perform such analyses of previous results, imposing the pCT data format and using GitHub in program development will now make this possible.    

Overwriting existing output directories is prevented by searching the permanent data storage directory in /ion and folder naming proceeds by appending `_i` (beginning with *i = 1*) until a unique directory name has been achieved.  If overwriting is permitted, then directory names are not changed and any existing data in these directories may be overwritten since file names do not change.  Once directories and file name are all defined, these directories are created on the local compute node and in the permanent data storage location so other compute nodes find that these directories already exist. 

However, there are a few images whose file names can change, namely those that result from filtering as the size of the filter neighbourhood is reflected in their file name.  For example, a 2D median filter with filter width 7 (radius 3) applied to the FBP image is named `FBP_med_2D_r7.txt`.  For `FBP`, `hull`, `x_0`, `x_k`, and `x`, the *config* file has `key/value` pairs specifying the radii to use when applying median filter and average filters and it is permissible to generate 1 or more of these during preprocessing/reconstruction.  However, although there is the option to generate and save each of these during preprocessing, the *config* file is used to specify which is to be used as the hull (`hull_h`) and the initial iterate (`x_h`), with `x_h` being updated after each iteration of reconstruction; i.e., each iteration of reconstruction is stored in the same `x_h` global variable and the only way each iteration is known after reconstruction is because the initial iterate is written to disk as `x_0.txt` and after the *k*th iteration it is saved as `x_k.txt`.

Hull endpoint and most likely path (MLP) calculations can now proceed by transferring the requisite data to the GPU all at once or one chunk at a time as needed with the associated target GPU arrays either allocated once and reused or allocated/freed each time.  The MLP calculations make use of precalculated tables for the sine/cosine terms, the coefficients of the correlation matrices, and the polynomials involved in the calculations.  Protons that skim the object and therefore have a low signal to noise  (SNR) ratio and these can largely be avoided by optionally ignoring MLP paths with fewer than some user definable number of voxels.  

The iterative projection algorithms that have been implemented and are currently available for use are ART, DROP, and 2 variations of a Robust approach to DROP (still in development and being improved).  There are several variations of Total Variation Superiorization (TVS) that were recently added and are now available for use, providing the ability to (1) perform multiple iterations of TVS per feasibility seeking iteration, (2) change the order of the superiorization and feasibility seeking portions of TVS+DROP, (3) skip the total variation (TV) check between each perturbation applied, and/or (4) choose the parallel implementation of TVS. 

