=========================================================================
Proton Computed Tomography (pCT) Preprocessing/Image Reconstruction Program
=========================================================================
This program expects proton tracker coordinate and Water Equivalent Path Length (WEPL) measurements acquired from various gantry angles and:

Phase 1:
(1) Extracts execution settings, options, and parameters from configuration file "config.cfg" and its location (if in non-default location) passed as command line argument  
(2) Removes statistical outliers and irrelevant histories from data set  
(3) Performs hull-detection  
(4) Performs MLP  
(5) Write preprocessing data to disk: system matrix A (MLP path info), vector x0 (initial iterate/guess), vector b (WEPL measurements), and hull detected  

Phase 2:
(1) Reconstructs a pCT image via iterative projection methods  

Currently implemented iterative projection method algorithms are ART, DROP, and 2 variations of a robust approach to DROP in development (total variation/superiorization has not been implemented yet).  The data/task parallelism is inherent in nearly every aspect of preprocessing and reconstruction and has been exploited (except for MLP and image reconstruction) using GPGPU programming (CUDA).  The host is primarily used only for allocating/freeing GPU memory,  transferring data to/from the GPU, and configuring/launching GPU kernels.  SAll options/parameters affecting program behavior are specified via key/value pairs config file and the location of the desired config file is specified as a command line arguments added as flags when executing the program.  The relaxation parameter LAMBDA and scale factors to apply to the update of the first N voxels are also specified as flags upon program execution.  Running the program and specifying these options is accomplished via the command line structure:

./pct_reconstruction [.cfg file address] [LAMBDA] [C1, C2, C3, ..., CN]
