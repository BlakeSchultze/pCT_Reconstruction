=========================================================================
pCT_Reconstruction
=========================================================================
This program reads the proton track and energy measurements acquired from various gantry angles and:

(1) Removes irrelevant and statistical outliers from data set
(2) Performs hull-detection
(3) Performs MLP
(4) Reconstructs a pCT image via iterative projection methods

At the moment, (4) has not been implemented yet and (3) only works on a single manually defined history and is performed on the host, as this was a necessary step in development and verification.  Having now passed verification, the next step is to modify the routine to read the necessary data directly from the data arrays/vectors and repeat for each history.  Upon verification of this process, the routine will be converted to a CUDA implementation and execution moved to the GPU.  Once this has been completed, development will continue to (4), beginning with the simplest of the iterative projection methods (ART) and will eventually include additional algorithms (e.g. DROP, SART, BIP, SAP, etc.).
