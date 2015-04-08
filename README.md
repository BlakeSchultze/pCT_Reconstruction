=========================================================================
Proton Computed Tomography (pCT) Preprocessing/Image Reconstruction Program
=========================================================================
This program expects proton tracker coordinate and Water Equivalent Path Length (WEPL) measurements acquired from various gantry angles and:

Phase 1:

(1) Extracts execution settings, options, and parameters from configuration file "settings.cfg" and its location (if in non-default location) passed as command line argument  
(2) Removes statistical outliers and irrelevant histories from data set  
(3) Performs hull-detection  
(4) Performs MLP  
(5) Write preprocessing data to disk: system matrix A (MLP path info), vector x0 (initial iterate/guess), vector b (WEPL measurements), and hull detected  

Phase 2:

(1) Reconstructs a pCT image via iterative projection methods  

Currently implemented iterative projection method algorithms are ART, DROP, and 2 variations of a robust approach to DROP in development (total variation/superiorization has not been implemented yet).  The data/task parallelism is inherent in nearly every aspect of preprocessing and reconstruction and has been exploited (except for MLP and image reconstruction) using GPGPU programming (CUDA).  The host is primarily used only for allocating/freeing GPU memory,  transferring data to/from the GPU, and configuring/launching GPU kernels.  All options/parameters affecting program behavior are specified via key/value pairs in configuration file "settings.cfg" and if this is not located in the expected default location, its path can be passed as a command line argument.  With this approach, since changing parameters does not modify the program's source code, it does not need to be recompiled each time an option/parameter is changed and the program can be ran by simply launching its .exe file unless the .cfg file is not in its default location.  To simplify the implementation of this approach, the program adheres to and enforces the pCT data file/folder naming and organizational scheme for expected location of input data and creation of folders where output data is written to disk (includes some flexibility in path specifications if this cannot be followed exactly but file naming is strictly followed).

If the configuration file is not in its default location, the program must be launch from the command line via the following:

./pct_reconstruction [.cfg file address]
