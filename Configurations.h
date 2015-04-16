#pragma once

//#include <C:\Users\Blake\Documents\GitHub\pCT_Reconstruction\pCT_Reconstruction.h>
#include <C:\Users\Blake\Documents\GitHub\pCT_Reconstruction\Constants.h>
#include <C:\Users\Blake\Documents\GitHub\pCT_Reconstruction\Globals.h>

typedef unsigned long long ULL;
typedef unsigned int uint;
struct configurations;
enum LOG_ENTRIES;

// 32 uint (128), 1 int (4), 38 double (304), 22 bool (22) = 458 + padding = 464
struct configurations
{
	//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------//
	//----------------------------------------------------------------------- Output option parameters ------------------------------------------------------------------------//
	//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------//
	// 14 uint, 23 double,
	//char* PROJECTION_DATA_DIR, * PREPROCESSING_DIR_D, * RECONSTRUCTION_DIR_D;
	//char* OBJECT_D, * RUN_DATE_D, * RUN_NUMBER_D, * PROJECTION_DATA_DATE_D, * PREPROCESS_DATE_D, * RECONSTRUCTION_DATE_D;
	double RECON_CYL_RADIUS_D, RECON_CYL_DIAMETER_D, RECON_CYL_HEIGHT_D, IMAGE_WIDTH_D, IMAGE_HEIGHT_D, IMAGE_THICKNESS_D, VOXEL_WIDTH_D, VOXEL_HEIGHT_D, VOXEL_THICKNESS_D, SLICE_THICKNESS_D;
	double X_ZERO_COORDINATE_D, Y_ZERO_COORDINATE_D, Z_ZERO_COORDINATE_D, RAM_LAK_TAU_D;
	double GANTRY_ANGLE_INTERVAL_D, ANGULAR_BIN_SIZE_D, SSD_T_SIZE_D, SSD_V_SIZE_D, T_SHIFT_D, U_SHIFT_D, V_SHIFT_D, T_BIN_SIZE_D, V_BIN_SIZE_D;
	double LAMBDA_D, LAMBDA, ETA_D, HULL_FILTER_THRESHOLD_D, FBP_AVG_THRESHOLD_D, X_0_FILTER_THRESHOLD_D;
	double SC_THRESHOLD_D, MSC_THRESHOLD_D, SM_LOWER_THRESHOLD_D, SM_UPPER_THRESHOLD_D, SM_SCALE_THRESHOLD_D;
	double VOXEL_STEP_SIZE_D, MLP_U_STEP_D, CONSTANT_CHORD_NORM_D, CONSTANT_LAMBDA_SCALE_D;

	uint NUM_SCANS_D, MAX_GPU_HISTORIES_D, MAX_CUTS_HISTORIES_D, T_BINS_D, V_BINS_D, COLUMNS_D, ROWS_D, SLICES_D, SIGMAS_2_KEEP_D;
	uint GANTRY_ANGLES_D, NUM_FILES_D, ANGULAR_BINS_D, NUM_BINS_D, NUM_VOXELS_D;
	uint SIZE_BINS_CHAR_D, SIZE_BINS_BOOL_D, SIZE_BINS_INT_D, SIZE_BINS_UINT_D, SIZE_BINS_FLOAT_D, SIZE_IMAGE_CHAR_D;
	uint SIZE_IMAGE_BOOL_D, SIZE_IMAGE_INT_D, SIZE_IMAGE_UINT_D, SIZE_IMAGE_FLOAT_D, SIZE_IMAGE_DOUBLE_D;
	uint ITERATIONS_D, BLOCK_SIZE_D, HULL_FILTER_RADIUS_D, X_0_FILTER_RADIUS_D, FBP_AVG_RADIUS_D, FBP_MEDIAN_RADIUS_D;	
	uint MSC_DIFF_THRESH_D;	

	int PSI_SIGN_D;
	SCAN_TYPES DATA_TYPE;									// Specify the type of input data: EXPERIMENTAL, SIMULATED_G, SIMULATED_T
	HULL_TYPES HULL_TYPE;									// Specify which of the HULL_TYPES to use in this run's MLP calculations
	FILTER_TYPES FBP_FILTER_TYPE;		  					// Specifies which of the defined filters will be used in FBP
	X_0_TYPES	X_0_TYPE;									// Specify which of the HULL_TYPES to use in this run's MLP calculations
	RECON_ALGORITHMS RECONSTRUCTION_METHOD; 				// Specify which of the projection algorithms to use for image reconstruction
	
	bool ADD_DATA_LOG_ENTRY_D, CONSOLE_OUTPUT_2_DISK_D;
	bool IMPORT_PREPROCESSING_D, PERFORM_RECONSTRUCTION_D, PREPROCESS_OVERWRITE_OK_D, RECON_OVERWRITE_OK_D;
	bool FBP_ON_D, AVG_FILTER_FBP_D, MEDIAN_FILTER_FBP_D, IMPORT_FILTERED_FBP_D, SC_ON_D, MSC_ON_D, SM_ON_D;
	bool AVG_FILTER_HULL_D, AVG_FILTER_ITERATE_D;//, MLP_FILE_EXISTS_D, HISTORIES_FILE_EXISTS_D;
	bool WRITE_MSC_COUNTS_D, WRITE_SM_COUNTS_D, WRITE_X_FBP_D, WRITE_FBP_HULL_D, WRITE_AVG_FBP_D, WRITE_MEDIAN_FBP_D, WRITE_BIN_WEPLS_D, WRITE_WEPL_DISTS_D, WRITE_SSD_ANGLES_D;	
	//*************************************************************************************************************************************************************************//
	//*********************************************************************** Output option parameters ************************************************************************//
	//*************************************************************************************************************************************************************************//
	configurations
	(
		//char* projection_data_dir		= "D:\\pCT_Data\\Output",							
		//char* preprocessing_dir_p 		= "CTP404\\input_CTP404_4M",
		//char* reconstruction_dir_p 		= "CTP404\\input_CTP404_4M\\Robust2\\ETA0001",
		//char* object_p					= "Object",
		//char* run_date_p				= "MMDDYYYY",
		//char* run_number_p				= "Run",
		//char* projection_data_date_p	= "MMDDYYYY",
		//char* preprocess_date_p			= "MMDDYYYY",
		//char* reconstruction_date_p		= "MMDDYYYY",
		uint num_scans_p 				= 1,								// [#] Total number of scans of same object
		uint max_gpu_histories_p		= 1500000,							// [#] Number of histories to process on the GPU at a time, based on GPU capacity
		uint max_cuts_histories_p 		= 1500000,	
		uint columns_p 					= 200,
		uint rows_p 					= 200.4,
		//uint slices_p 					= 32,
		uint sigmas_2_keep_p 			= 3,
		double gantry_angle_interval_p 	= 4,
		double angular_bin_size_p 		= 4.0,
		double ssd_t_size_p 			= 35.0,
		double ssd_v_size_p 			= 9.0,
		double t_shift_p 				= 0.0,
		double u_shift_p 				= 0.0,
		double v_shift_p 				= 0.0,
		double t_bin_size_p 			= 0.1,
		double v_bin_size_p 			= 0.25,
		double recon_cyl_radius_p 		= 10.0,
		double slice_thickness_p 		= 0.25,
		//---------------------------------------------------------------------------------------------------------------------------------------------------------------------//
		//------------------------------------------------------------------- Enumerated type parameters/options --------------------------------------------------------------//
		//---------------------------------------------------------------------------------------------------------------------------------------------------------------------//	
		SCAN_TYPES data_type_p						= SIMULATED_G,			// Specifies the source of the input data (EXPERIMENTAL = 0, GEANT4 = 1, TOPAS = 2)
		HULL_TYPES hull_type_p						= MSC_HULL,				// Specify which hull detection method to use for MLP calculations (IMPORT = 0, SC = 1, MSC = 2, SM = 3, FBP = 4)
		FILTER_TYPES fbp_filter_type_p				= SHEPP_LOGAN,		  	// Specifies which of the defined filters to use in FBP (RAM_LAK = 0, SHEPP_LOGAN = 1, NONE = 2)
		X_0_TYPES x_0_type_p						= HYBRID,				// Specify which initial iterate to use for reconstruction (IMPORT = 0, HULL = 1, FBP = 2, HYBRID = 3, ZEROS = 4)
		RECON_ALGORITHMS reconstruction_method_p	= DROP,					// Specify algorithm to use for image reconstruction (ART = 0, DROP = 1, BIP = 2, SAP = 3, ROBUST1 = 4, ROBUST2 = 5)
		//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------//
		//---------------------------------------------------------- Reconstruction and image filtering parameters/options --------------------------------------------------------//
		//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------//
		uint iterations_p				= 12,								// [#] of iterations through the entire set of histories to perform in iterative image reconstruction
		uint block_size_p				= 60,								// [#] of paths to use for each update: ART = 1, 
		uint hull_filter_radius_p 		= 1,								// [#] Averaging filter neighborhood radius in: [voxel - AVG_FILTER_SIZE, voxel + AVG_FILTER_RADIUS]
		uint x_0_filter_radius_p		= 3,
		uint fbp_avg_radius_p			= 1,
		uint fbp_median_radius_p		= 3,	
		int psi_sign_p					= 1,
		double lambda_p 				= 0.0001,
		double eta_p                    = 0.0001,
		double hull_filter_threshold_p	= 0.1,								// [#] Threshold ([0.0, 1.0]) used by averaging filter to identify voxels belonging to the hull
		double fbp_avg_threshold_p		= 0.1,
		double x_0_filter_threshold_p	= 0.1,
		//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------//
		//----------------------------------------------------------------------- Hull-Detection Parameters -----------------------------------------------------------------------//
		//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------//
		uint msc_diff_thresh_p			= 50,								// [#] Threshold on difference in counts between adjacent voxels used by MSC for edge detection
		double sc_threshold_p			= 0.0,								// [cm] If WEPL < SC_THRESHOLD, SC assumes the proton missed the object
		double msc_threshold_p			= 0.0,								// [cm] If WEPL < MSC_THRESHOLD, MSC assumes the proton missed the object
		double sm_lower_threshold_p		= 6.0,								// [cm] If WEPL >= SM_THRESHOLD, SM assumes the proton passed through the object
		double sm_upper_threshold_p		= 21.0,								// [cm] If WEPL > SM_UPPER_THRESHOLD, SM ignores this history
		double sm_scale_threshold_p		= 1.0,								// [cm] Threshold scaling factor used by SM to adjust edge detection sensitivity
		//---------------------------------------------------------------------------------------------------------------------------------------------------------------------//
		//------------------------------------------------------------ Program execution behavior options/parameters ----------------------------------------------------------//
		//---------------------------------------------------------------------------------------------------------------------------------------------------------------------//
		bool add_data_log_entry_p		= true,								// [T/F] Add log entry for data generated during execution (T) or not (F)
		bool console_output_2_disk_p	= false,							// [T/F] Redirect console window output to text file (T) or leave it as stdout (F)
		bool import_preprocessing_p		= true,								// [T/F] Import preprocessed data previously generated, i.e. A/x0/b/hull/MLP), (T) or generate it (F) 
		bool perform_reconstruction_p	= true,								// [T/F] Perform reconstruction (T) or not (F)
		bool preprocess_overwrite_ok_p	= false,							// [T/F] Allow preprocessing data to be overwritten (T) or not (F)
		bool recon_overwrite_ok_p 		= false,							// [T/F] Allow reconstruction data to be overwritten (T) or not (F)
		bool fbp_on_p					= true,								// [T/F] Turn FBP on (T) or off (F)
		bool avg_filter_fbp_p			= false,							// [T/F] Apply averaging filter to initial iterate (T) or not (F)
		bool median_filter_fbp_p		= false,
		bool import_filtered_fbp_p		= false,
		bool sc_on_p					= false,							// [T/F] Turn Space Carving on (T) or off (F)
		bool msc_on_p					= true,								// [T/F] Turn Modified Space Carving on (T) or off (F)
		bool sm_on_p					= false,							// [T/F] Turn Space Modeling on (T) or off (F)
		bool avg_filter_hull_p			= true,								// [T/F] Apply averaging filter to hull (T) or not (F)
		bool avg_filter_iterate_p		= false,							// [T/F] Apply averaging filter to initial iterate (T) or not (F)
		//bool mlp_file_exists_p			= false,
		//bool histories_file_exists_p	= false,	
		//---------------------------------------------------------------------------------------------------------------------------------------------------------------------//
		//----------------------------------------------------------------------- Output option parameters --------------------------------------------------------------------//
		//---------------------------------------------------------------------------------------------------------------------------------------------------------------------//
		bool write_msc_counts_p			= true,								// [T/F] Write MSC counts array to disk (T) or not (F) before performing edge detection 
		bool write_sm_counts_p			= true,								// [T/F] Write SM counts array to disk (T) or not (F) before performing edge detection 
		bool write_x_fbp_p				= true,								// [T/F] Write FBP image before thresholding to disk (T) or not (F)
		bool write_fbp_hull_p			= true,								// [T/F] Write FBP hull to disk (T) or not (F)
		bool write_avg_fbp_p			= true,								// [T/F] Write average filtered FBP image before thresholding to disk (T) or not (F)
		bool write_median_fbp_p			= false,							// [T/F] Write median filtered FBP image to disk (T) or not (F)
		bool write_bin_wepls_p			= false,							// [T/F] Write WEPLs for each bin to disk (T) for WEPL distribution analysis, or do not (F)
		bool write_wepl_dists_p			= false,							// [T/F] Write mean WEPL values to disk (T) or not (F): t bin = columns, v bin = rows, 1 angle per file
		bool write_ssd_angles_p			= false								// [T/F] Write angles for each proton through entry/exit tracker planes to disk (T), or do not (F)
	):
	//*************************************************************************************************************************************************************************//
	//*********************************************************************** Parameter Instantiations ************************************************************************//
	//*************************************************************************************************************************************************************************//
	//PROJECTION_DATA_DIR(projection_data_dir),
	//PREPROCESSING_DIR_D(preprocessing_dir_p),
	//RECONSTRUCTION_DIR_D(reconstruction_dir_p),
	//OBJECT_D(object_p),
	//RUN_DATE_D(run_date_p),
	//RUN_NUMBER_D(run_number_p),
	//PROJECTION_DATA_DATE_D(projection_data_date_p),
	//PREPROCESS_DATE_D(preprocess_date_p),
	//RECONSTRUCTION_DATE_D(reconstruction_date_p),
	NUM_SCANS_D(num_scans_p),												// *[#] Total number of scans of same object	
	MAX_GPU_HISTORIES_D(max_gpu_histories_p),								// *[#] Number of histories to process on the GPU at a time, based on GPU capacity
	MAX_CUTS_HISTORIES_D(max_cuts_histories_p),								// *[#] Number of histories to process on the GPU at a time, based on GPU capacity
	GANTRY_ANGLE_INTERVAL_D(gantry_angle_interval_p),						// *[degrees] Angle between successive projection angles	
	ANGULAR_BIN_SIZE_D(angular_bin_size_p),									// *[degrees] Angle between adjacent bins in angular (rotation) direction
	SSD_T_SIZE_D(ssd_t_size_p),												// *[cm] Length of SSD in t (lateral) direction
	SSD_V_SIZE_D(ssd_v_size_p),												// *[cm] Length of SSD in v (vertical) direction
	T_BIN_SIZE_D(t_bin_size_p),												// *[cm] Distance between adjacent bins in t (lateral) direction
	V_BIN_SIZE_D(v_bin_size_p),												// *[cm] Distance between adjacent bins in v (vertical) direction
	T_SHIFT_D(t_shift_p),													// *[cm] Amount by which to shift all t coordinates on input
	U_SHIFT_D(u_shift_p),													// *[cm] Amount by which to shift all u coordinates on input
	V_SHIFT_D(v_shift_p),													// *[cm] Amount by which to shift all v coordinates on input
	SIGMAS_2_KEEP_D(sigmas_2_keep_p),										// *[#] Number of standard deviations from mean to allow before cutting the history 
	RECON_CYL_RADIUS_D(recon_cyl_radius_p),									// *[cm] Radius of reconstruction cylinder
	COLUMNS_D(columns_p),													// *[#] Number of voxels in the x direction (i.e., number of columns) of image
	ROWS_D(rows_p),															// *[#] Number of voxels in the y direction (i.e., number of rows) of image
	VOXEL_THICKNESS_D(slice_thickness_p),									// *[cm] distance between top and bottom of each slice in image
	SLICE_THICKNESS_D(slice_thickness_p),									// *[cm] distance between top and bottom of each slice in image
	//---------------------------------------------------------------------------------------------------------------------------------------------------------------------//
	//------------------------------------------------------------------- Enumerated type parameters/options --------------------------------------------------------------//
	//---------------------------------------------------------------------------------------------------------------------------------------------------------------------//	
	DATA_TYPE(data_type_p),													// Specifies the source of the input data (EXPERIMENTAL = 0, GEANT4 = SIMULATED_G = 1, TOPAS = SIMULATED_T = 2)
	HULL_TYPE(hull_type_p),													// Specify which hull detection method to use for MLP calculations (IMPORT = 0, SC = 1, MSC = 2, SM = 3, FBP = 4)
	FBP_FILTER_TYPE(fbp_filter_type_p),		  								// Specifies which of the defined filters to use in FBP (RAM_LAK = 0, SHEPP_LOGAN = 1, NONE = 2)
	X_0_TYPE(x_0_type_p),													// Specify which initial iterate to use for reconstruction (IMPORT = 0, HULL = 1, FBP = 2, HYBRID = 3, ZEROS = 4)
	RECONSTRUCTION_METHOD(reconstruction_method_p),							// Specify algorithm to use for image reconstruction (ART = 0, DROP = 1, BIP = 2, SAP = 3, ROBUST1 = 4, ROBUST2 = 5)
	//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------//
	//--------------------------------------------------------- Options/parameters dependent on others read from config file --------------------------------------------------//
	//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------//
	NUM_FILES_D( NUM_SCANS_D * GANTRY_ANGLES_D ),							// *[#] 1 file per gantry angle per translation
	GANTRY_ANGLES_D(uint( 360 / GANTRY_ANGLE_INTERVAL_D )),					// *[#] Total number of projection angles
	T_BINS_D(uint( ssd_t_size_p / t_bin_size_p + 0.5 )),					// *[#] Number of bins (i.e. quantization levels) for t (lateral) direction 
	V_BINS_D(uint( ssd_v_size_p / v_bin_size_p + 0.5 )),					// *[#] Number of bins (i.e. quantization levels) for v (vertical) direction
	ANGULAR_BINS_D(uint( 360 / ANGULAR_BIN_SIZE_D + 0.5 )),					// *[#] Number of bins (i.e. quantization levels) for path angle 
	NUM_BINS_D( ANGULAR_BINS_D * T_BINS_D * V_BINS_D ),						// *[#] Total number of bins corresponding to possible 3-tuples [ANGULAR_BIN, T_BIN, V_BIN],
	SLICES_D(uint( RECON_CYL_HEIGHT_D / slice_thickness_p)),				// *[#] Number of voxels in the z direction (i.e., number of slices) of image
	NUM_VOXELS_D( COLUMNS_D * ROWS_D * SLICES_D ),							// *[#] Total number of voxels (i.e. 3-tuples [column, row, slice]) in image	
	RECON_CYL_DIAMETER_D( 2 * RECON_CYL_RADIUS_D ),							// *[cm] Diameter of reconstruction cylinder
	RECON_CYL_HEIGHT_D(SSD_V_SIZE_D - 1.0),									// *[cm] Height of reconstruction cylinder
	IMAGE_WIDTH_D(RECON_CYL_DIAMETER_D),									// *[cm] Distance between left and right edges of each slice in image
	IMAGE_HEIGHT_D(RECON_CYL_DIAMETER_D),									// *[cm] Distance between top and bottom edges of each slice in image
	IMAGE_THICKNESS_D(RECON_CYL_HEIGHT_D),									// *[cm] Distance between bottom of bottom slice and top of the top slice of image
	VOXEL_WIDTH_D(RECON_CYL_DIAMETER_D / COLUMNS_D),						// *[cm] distance between left and right edges of each voxel in image
	VOXEL_HEIGHT_D(RECON_CYL_DIAMETER_D / ROWS_D),							// *[cm] distance between top and bottom edges of each voxel in image
	X_ZERO_COORDINATE_D(-RECON_CYL_RADIUS_D),								// *[cm] x-coordinate corresponding to left edge of 1st voxel (i.e. column) in image space
	Y_ZERO_COORDINATE_D(RECON_CYL_RADIUS_D),								// *[cm] y-coordinate corresponding to top edge of 1st voxel (i.e. row) in image space
	Z_ZERO_COORDINATE_D(RECON_CYL_HEIGHT_D/2),								// *[cm] z-coordinate corresponding to top edge of 1st voxel (i.e. slice) in image space
	RAM_LAK_TAU_D(2/sqrt(2.0) * T_BIN_SIZE_D),								// *[#] Defines tau in Ram-Lak filter calculation, estimated from largest frequency in slice 
	//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------//
	//---------------------------------------------------------- Memory allocation size for arrays (binning, image) -----------------------------------------------------------//
	//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------//
	SIZE_BINS_CHAR_D(	NUM_BINS_D		* sizeof(char)	),					// *[bytes] Amount of memory required for a character array used for binning
	SIZE_BINS_BOOL_D(	NUM_BINS_D		* sizeof(bool)	),					// *[bytes] Amount of memory required for a boolean array used for binning
	SIZE_BINS_INT_D(	NUM_BINS_D		* sizeof(int)	),					// *[bytes] Amount of memory required for an integer array used for binning
	SIZE_BINS_UINT_D(	NUM_BINS_D		* sizeof(uint)	),					// *[bytes] Amount of memory required for an unsigned integer array used for binning
	SIZE_BINS_FLOAT_D(	NUM_BINS_D		* sizeof(float)	),					// *[bytes] Amount of memory required for a floating point array used for binning
	SIZE_IMAGE_CHAR_D(	NUM_VOXELS_D	* sizeof(char)	),					// *[bytes] Amount of memory required for a character array used for binning
	SIZE_IMAGE_BOOL_D(	NUM_VOXELS_D	* sizeof(bool)	),					// *[bytes] Amount of memory required for a boolean array used for binning
	SIZE_IMAGE_INT_D(	NUM_VOXELS_D	* sizeof(int)	),					// *[bytes] Amount of memory required for an integer array used for binning
	SIZE_IMAGE_UINT_D(	NUM_VOXELS_D	* sizeof(uint)	),					// *[bytes] Amount of memory required for an unsigned integer array used for binning
	SIZE_IMAGE_FLOAT_D(	NUM_VOXELS_D	* sizeof(float)	),					// *[bytes] Amount of memory required for a floating point array used for binning
	SIZE_IMAGE_DOUBLE_D(NUM_VOXELS_D	* sizeof(double)),					// *[bytes] Amount of memory required for a floating point array used for binning
	//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------//
	//----------------------------------------------------------------------------- MLP Parameters ----------------------------------------------------------------------------//
	//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------//
	VOXEL_STEP_SIZE_D( VOXEL_WIDTH_D / 2 ),									// *[cm] Length of the step taken along the path, i.e. change in depth per step for
	MLP_U_STEP_D( VOXEL_WIDTH_D / 2),										// *[cm] Size of the step taken along u direction during MLP; depth difference between successive MLP points
	CONSTANT_CHORD_NORM_D(pow(VOXEL_WIDTH_D, 2.0)),							// *[cm^2] Precalculated value of ||a_i||^2 for use of constant chord length
	CONSTANT_LAMBDA_SCALE_D(VOXEL_WIDTH_D * LAMBDA_D),						// *[cm] Precalculated value of |a_i| * LAMBDA for use of constant chord length
	//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------//
	//---------------------------------------------------------- Reconstruction and image filtering parameters/options --------------------------------------------------------//
	//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------//
	ITERATIONS_D(iterations_p),												// *[#] of iterations through the entire set of histories to perform in iterative image reconstruction
	BLOCK_SIZE_D(block_size_p),												// *[#] of paths to use for each update: e.g., ART = 1, 
	HULL_FILTER_RADIUS_D(hull_filter_radius_p),								// *[#] Radius of average filter neighborhood applied to hull: [voxel - r, voxel + r]	
	X_0_FILTER_RADIUS_D(x_0_filter_radius_p),								// *[#] Radius of average filter neighborhood applied to x_0: [voxel - r, voxel + r]
	FBP_AVG_RADIUS_D(fbp_avg_radius_p),										// *[#] Radius of average filter neighborhood applied to FBP: [voxel - r, voxel + r]
	FBP_MEDIAN_RADIUS_D(fbp_median_radius_p),								// *[#] Radius of median filter neighborhood applied to FBP: [voxel - r, voxel + r]
	PSI_SIGN_D(psi_sign_p),													// *[+1/-1] Sign specifying the sign to use for Psi in scaling residual for updates in robust technique to reconstruction	
	LAMBDA_D(lambda_p),														// *[#] Relaxation parameter used in update calculations in reconstruction algorithms
	LAMBDA(lambda_p),														// *[#] Relaxation parameter used in update calculations in reconstruction algorithms
	ETA_D(eta_p),															// *[#] Value used in calculation of Psi = (1-x_i) * ETA used in robust technique to reconstruction
	HULL_FILTER_THRESHOLD_D(hull_filter_threshold_p),						// *[#] Threshold ([0.0, 1.0]) used by averaging filter to identify voxels belonging to the hull
	FBP_AVG_THRESHOLD_D(fbp_avg_threshold_p),								// *[#] Threshold ([0.0, 1.0]) used by averaging filter to identify voxels belonging to the FBP image
	X_0_FILTER_THRESHOLD_D(x_0_filter_threshold_p),							// *[#] Threshold ([0.0, 1.0]) used by averaging filter to identify voxels belonging to the initial iterate x_0
	//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------//
	//----------------------------------------------------------------------- Hull-detection parameters -----------------------------------------------------------------------//
	//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------//
	MSC_DIFF_THRESH_D(msc_diff_thresh_p),									// *[#] Threshold on difference in counts between adjacent voxels used by MSC for edge detection
	SC_THRESHOLD_D(sc_threshold_p),											// *[cm] If WEPL < SC_THRESHOLD, SC assumes the proton missed the object
	MSC_THRESHOLD_D(msc_threshold_p),										// *[cm] If WEPL < MSC_THRESHOLD, MSC assumes the proton missed the object
	SM_LOWER_THRESHOLD_D(sm_lower_threshold_p),								// *[cm] If WEPL >= SM_THRESHOLD, SM assumes the proton passed through the object
	SM_UPPER_THRESHOLD_D(sm_upper_threshold_p),								// *[cm] If WEPL > SM_UPPER_THRESHOLD, SM ignores this history
	SM_SCALE_THRESHOLD_D(sm_scale_threshold_p),								// *[cm] Threshold scaling factor used by SM to adjust edge detection sensitivity
	//*************************************************************************************************************************************************************************//
	//*********************************************************************** Preprocessing control options *******************************************************************//
	//*************************************************************************************************************************************************************************//
	CONSOLE_OUTPUT_2_DISK_D(console_output_2_disk_p	),						// [T/F] Redirect console window output to text file (T) or leave it as stdout (F)	
	ADD_DATA_LOG_ENTRY_D(add_data_log_entry_p),								// *[T/F] Add log entry for data generated during execution (T) or not (F)
	IMPORT_PREPROCESSING_D(import_preprocessing_p),							// *[T/F] Import preprocessed data previously generated, i.e. A/x0/b/hull/MLP), (T) or generate it (F) 
	PERFORM_RECONSTRUCTION_D(perform_reconstruction_p),						// *[T/F] Perform reconstruction (T) or not (F)
	PREPROCESS_OVERWRITE_OK_D(preprocess_overwrite_ok_p),					// *[T/F] Allow preprocessing data to be overwritten (T) or not (F)
	RECON_OVERWRITE_OK_D(recon_overwrite_ok_p),								// *[T/F] Allow reconstruction data to be overwritten (T) or not (F)
	FBP_ON_D(fbp_on_p),														// *[T/F] Turn FBP on (T) or off (F)
	AVG_FILTER_FBP_D(avg_filter_fbp_p),										// *[T/F] Apply averaging filter to initial iterate (T) or not (F)
	MEDIAN_FILTER_FBP_D(median_filter_fbp_p),								// *[T/F] Apply median filtering to FBP (T) or not (F)
	IMPORT_FILTERED_FBP_D(import_filtered_fbp_p),							// *[T/F] Import filtered FBP from disk (T) or not (F)
	SC_ON_D(sc_on_p),														// *[T/F] Turn Space Carving on (T) or off (F)
	MSC_ON_D(msc_on_p),														// *[T/F] Turn Modified Space Carving on (T) or off (F)
	SM_ON_D(sm_on_p),														// *[T/F] Turn Space Modeling on (T) or off (F)
	AVG_FILTER_HULL_D(avg_filter_hull_p),									// *[T/F] Apply averaging filter to hull (T) or not (F)	
	AVG_FILTER_ITERATE_D(avg_filter_iterate_p),								// *[T/F] Apply averaging filter to initial iterate x_0 (T) or not (F)	
	//MLP_FILE_EXISTS_D(mlp_file_exists_p),									// *[T/F] MLP.bin preprocessing data exists (T) or not (F)
	//HISTORIES_FILE_EXISTS_D(histories_file_exists_p),						// *[T/F] Histories.bin preprocessing data exists (T) or not (F)
	//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------//
	//--------------------------------------------------------- Control of writing optional intermediate data to disk  --------------------------------------------------------//
	//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------//
	WRITE_MSC_COUNTS_D(write_msc_counts_p),									// *[T/F] Write MSC counts array to disk (T) or not (F) before performing edge detection 
	WRITE_SM_COUNTS_D(write_sm_counts_p),									// *[T/F] Write SM counts array to disk (T) or not (F) before performing edge detection 
	WRITE_X_FBP_D(write_x_fbp_p),											// *[T/F] Write FBP image before thresholding to disk (T) or not (F)
	WRITE_FBP_HULL_D(write_fbp_hull_p),										// *[T/F] Write FBP hull to disk (T) or not (F)
	WRITE_AVG_FBP_D(write_avg_fbp_p),										// *[T/F] Write average filtered FBP image before thresholding to disk (T) or not (F)
	WRITE_MEDIAN_FBP_D(write_median_fbp_p),									// *[T/F] Write median filtered FBP image to disk (T) or not (F)
	WRITE_BIN_WEPLS_D(write_bin_wepls_p),									// *[T/F] Write WEPLs for each bin to disk (T) for WEPL distribution analysis, or do not (F)
	WRITE_WEPL_DISTS_D(write_wepl_dists_p),									// *[T/F] Write mean WEPL values to disk (T) or not (F): t bin = columns, v bin = rows, 1 angle per file
	WRITE_SSD_ANGLES_D(write_ssd_angles_p)									// *[T/F] Write angles for each proton through entry/exit tracker planes to disk (T), or do not (F)
	{};
};
//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------//
//--------------------------------------------------------- Instantiation of host/GPU global variables for configuration settings --------------------------------------------------------//
//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------//
configurations parameters;
configurations *parameters_h = &parameters;
configurations *parameters_d;

/***********************************************************************************************************************************************************************************************************************/
/******************************************************************************************* Host function forward declarations ****************************************************************************************/
/***********************************************************************************************************************************************************************************************************************/

// Execution Control Functions
bool is_bad_angle( const int );	// Just for use with Micah's simultated data
void timer( bool, clock_t, clock_t);
void pause_execution();
void exit_program_if( bool );

// Memory transfers and allocations/deallocations
void initial_processing_memory_clean();
void resize_vectors( unsigned int );
void shrink_vectors( unsigned int );
void allocations( const unsigned int );
void reallocations( const unsigned int );
void post_cut_memory_clean(); 

// Image Initialization/Construction Functions
template<typename T> void initialize_host_image( T*& );
template<typename T> void add_ellipse( T*&, int, double, double, double, double, T );
template<typename T> void add_circle( T*&, int, double, double, double, T );
template<typename O> void import_image( O*&, char*, char*, DISK_WRITE_MODE );
template<typename T> void binary_2_txt_images( char*, char*, T*& );

// Preprocessing setup and initializations 
//void apply_execution_arguments();
void apply_execution_arguments(unsigned int, char**);
void assign_SSD_positions();
void initializations();
void count_histories();	
void count_histories_v0();
void reserve_vector_capacity(); 

// Preprocessing functions
void read_energy_responses( const int, const int, const int );
void read_data_chunk( const uint, const uint, const uint );
void read_data_chunk_v0( const uint, const uint, const uint );
void read_data_chunk_v02( const uint, const uint, const uint );
void apply_tuv_shifts( unsigned int );
void convert_mm_2_cm( unsigned int );
void recon_volume_intersections( const uint );
void binning( const uint );
void calculate_means();
void initialize_stddev();
void sum_squared_deviations( const uint, const uint );
void calculate_standard_deviations();
void statistical_cuts( const uint, const uint );
void initialize_sinogram();
void construct_sinogram();
void FBP();
void x_FBP_2_hull();
void filter();
void backprojection();

// Hull-Detection
template<typename T> void initialize_hull( T*&, T*& );
void hull_initializations();
void hull_detection( const uint );
void hull_detection_finish();
void SC( const uint );
void MSC( const uint );
void MSC_edge_detection();
void SM( const uint );
void SM_edge_detection();
void SM_edge_detection_2();
void hull_selection();

// Image filtering functions
template<typename H, typename D> void averaging_filter( H*&, D*&, int, bool, double );
template<typename D> __global__ void averaging_filter_GPU( configurations*, D*, D*, int, bool, double );
template<typename T> void median_filter_2D( T*&, unsigned int );
template<typename T> void median_filter_2D( T*&, T*&, unsigned int );
template<typename D> __global__ void median_filter_GPU( configurations*, D*, D*, int, bool, double );
template<typename T> void median_filter_3D( T*&, T*&, unsigned int );
template<typename T, typename T2> __global__ void apply_averaging_filter_GPU( configurations*, T*, T2* );
void median_filter_FBP_2D( float*&, uint );
void median_filter_FBP_3D( float*&, uint );

// MLP
void MLP();
template<typename O> bool find_MLP_endpoints( O*&, double, double, double, double, double, double&, double&, double&, int&, int&, int&, bool);
void collect_MLP_endpoints();
unsigned int find_MLP_path( unsigned int*&, double*&, double, double, double, double, double, double, double, double, double, double, int, int, int );
double mean_chord_length( double, double, double, double, double, double );
double mean_chord_length2( double, double, double, double, double, double, double, double );
double EffectiveChordLength(double, double);

// Preprocessing data IO
void export_MLP_path( FILE*, unsigned int*&, unsigned int);
void export_histories();
unsigned int import_MLP_path(FILE*, unsigned int*&);
unsigned int import_histories();
void export_hull();
void import_hull();

// New routine test functions
void import_X_0_TYPES();
void generate_history_sequence(ULL, ULL, ULL* );
void verify_history_sequence(ULL, ULL, ULL* );

// Image Reconstruction
void define_X_0_TYPES();
void create_hull_image_hybrid();
void image_reconstruction();
template< typename T, typename L, typename R> T discrete_dot_product( L*&, R*&, unsigned int*&, unsigned int );
template< typename A, typename X> double update_vector_multiplier( double, A*&, X*&, unsigned int*&, unsigned int );
template< typename A, typename X> void update_iterate( double, A*&, X*&, unsigned int*&, unsigned int );
// uses mean chord length for each element of ai instead of individual chord lengths
template< typename T, typename RHS> T scalar_dot_product( double, RHS*&, unsigned int*&, unsigned int );
double scalar_dot_product2( double, float*&, unsigned int*&, unsigned int );
template< typename X> double update_vector_multiplier2( double, double, X*&, unsigned int*&, unsigned int );
double update_vector_multiplier22( double, double, float*&, unsigned int*&, unsigned int );
template< typename X> void update_iterate2( double, double, X*&, unsigned int*&, unsigned int );
void update_iterate22( double, double, float*&, unsigned int*&, unsigned int );
template<typename X, typename U> void calculate_update( double, double, X*&, U*&, unsigned int*&, unsigned int );
template<typename X, typename U> void update_iterate3( X*&, U*& );
void DROP_blocks( unsigned int*&, float*&, double, unsigned int, double, double*&, unsigned int*& );
void DROP_update( float*&, double*&, unsigned int*& );
void DROP_blocks2( unsigned int*&, float*&, double, unsigned int, double, double*&, unsigned int*&, unsigned int&, unsigned int*& );
void DROP_update2( float*&, double*&, unsigned int*&, unsigned int&, unsigned int*& );
void DROP_blocks3( unsigned int*&, float*&, double, unsigned int, double, double*&, unsigned int*&, unsigned int& );
void DROP_update3( float*&, double*&, unsigned int*&, unsigned int&);
void DROP_blocks_robust2( unsigned int*&, float*&, double, unsigned int, double, double*&, unsigned int*&, double*& );
void DROP_update_robust2( float*&, double*&, unsigned int*&, double*& );
void DROP_blocks_robust1( unsigned int*&, float*&, double, unsigned int, double, double*&, unsigned int*& );
void update_x( float*&, double*&, unsigned int*& );

// Input/output functions
void binary_2_ASCII();
template<typename T> void array_2_disk( char*, char*, DISK_WRITE_MODE, T*, const int, const int, const int, const int, const bool );
template<typename T> void vector_2_disk( char*, char*, DISK_WRITE_MODE, std::vector<T>, const int, const int, const int, const bool );
template<typename T> void t_bins_2_disk( FILE*, const std::vector<int>&, const std::vector<T>&, const BIN_ANALYSIS_TYPE, const int );
template<typename T> void bins_2_disk( char*, char*, DISK_WRITE_MODE, const std::vector<int>&, const std::vector<T>&, const BIN_ANALYSIS_TYPE, const BIN_ANALYSIS_FOR, const BIN_ORGANIZATION, ... );
template<typename T> void t_bins_2_disk( FILE*, int*&, T*&, const unsigned int, const BIN_ANALYSIS_TYPE, const BIN_ORGANIZATION, int );
template<typename T> void bins_2_disk( char*, char*, DISK_WRITE_MODE, int*&, T*&, const int, const BIN_ANALYSIS_TYPE, const BIN_ANALYSIS_FOR, const BIN_ORGANIZATION, ... );
void combine_data_sets();

// Image position/voxel calculation functions
int calculate_voxel( double, double, double );
int position_2_voxel( double, double, double );
int positions_2_voxels(const double, const double, const double, int&, int&, int& );
void voxel_2_3D_voxels( int, int&, int&, int& );
double voxel_2_position( int, double, int, int );
void voxel_2_positions( int, double&, double&, double& );
double voxel_2_radius_squared( int );

// Voxel walk algorithm functions
double distance_remaining( double, double, int, int, double, int );
double edge_coordinate( double, int, double, int, int );
double path_projection( double, double, double, int, double, int, int );
double corresponding_coordinate( double, double, double, double );
void take_2D_step( const int, const int, const int, const double, const double, const double, const double, const double, const double, const double, const double, const double, double&, double&, double&, int&, int&, int&, int&, double&, double&, double& );
void take_3D_step( const int, const int, const int, const double, const double, const double, const double, const double, const double, const double, const double, const double, double&, double&, double&, int&, int&, int&, int&, double&, double&, double& );

// Host helper functions		
template< typename T, typename T2> T max_n( int, T2, ...);
template< typename T, typename T2> T min_n( int, T2, ...);
template<typename T> T* sequential_numbers( int, int );
void bin_2_indexes( int, int&, int&, int& );
void print_section_separator(char );
void print_section_header( char*, char );
void print_section_exit( char*, char* );
const char * bool_2_string( bool b ){ return b ? "true" : "false"; }

// Generic IO helper functions
template<typename T> T cin_until_valid( T*, int, char* );
char((&current_MMDD( char(&)[5]))[5]);
char((&current_MMDDYYYY( char(&)[9]))[9]);
template<typename T> char((&minimize_trailing_zeros( T, char(&)[64]) )[64]);
std::string terminal_response(char*);
char((&terminal_response( char*, char(&)[256]))[256]);
bool directory_exists(char* );
unsigned int create_unique_dir( char* );
bool file_exists (const char* file_location) { return (bool)std::ifstream(file_location); };
bool file_exists2 (const char* file_location) { return std::ifstream(file_location).good(); };
bool file_exists3 (const char*);
bool blank_line( char c ) { return (c != '\n') && (c!= '\t') && (c != ' '); };
void fgets_validated(char *line, int buf_size, FILE*);
struct generic_IO_container read_key_value_pair( FILE* );
void print_copyright_notice();

// Configuration option/parameter handling
void set_configs_2_defines();
void set_defines_2_configs();
void export_D_configuration_parameters();
void export_configuration_parameters();
void set_data_info( char *);
bool preprocessing_data_exists();
void add_object_directory(char*, char*);
int add_run_directory(char*, char*, char*, char*, SCAN_TYPES );
int add_pCT_Images_dir(char*, char*, char*, char*, SCAN_TYPES );

void write_reconstruction_settings();
void read_config_file();
bool key_is_string_parameter( char* );
bool key_is_floating_point_parameter( char* );
bool key_is_integer_parameter( char* );
bool key_is_boolean_parameter( char* );
void set_string_parameter( generic_IO_container & );
void set_floating_point_parameter( generic_IO_container & );
void set_integer_parameter( generic_IO_container & );
void set_boolean_parameter( generic_IO_container & );
void set_parameter( generic_IO_container & );
void set_execution_date();
void set_IO_paths();
void view_config_file();
void set_dependent_parameters();
void parameters_2_GPU();

// Log file functions
LOG_OBJECT read_log();
std::vector<int> scan_log_4_matches( LOG_OBJECT );
std::string format_log_entry(const char*, uint  );
LOG_LINE construct_log_entry();
void new_log_entry( LOG_OBJECT );
void add_log_entry( LOG_ENTRIES );
void log_add_entry( LOG_ENTRIES );
void print_log( LOG_OBJECT );
void write_log( LOG_OBJECT);
void log_write_test();
void log_write_test2();
void log_write_test3();

// Test functions
void test_func();
void test_func2( std::vector<int>&, std::vector<double>&);
void test_transfer();
void test_transfer_GPU(double*, double*, double*);
/***********************************************************************************************************************************************************************************************************************/
/************************************************************************************** Device (GPU) function forward declarations *************************************************************************************/
/***********************************************************************************************************************************************************************************************************************/

// Preprocessing routines
__device__ bool calculate_intercepts( configurations*, double, double, double, double&, double& );
__global__ void recon_volume_intersections_GPU( configurations*, uint, int*, bool*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float* );
__global__ void binning_GPU( configurations*, uint, int*, int*, bool*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float* );
__global__ void calculate_means_GPU( configurations*, int*, float*, float*, float* );
__global__ void sum_squared_deviations_GPU( configurations*, uint, int*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*  );
__global__ void calculate_standard_deviations_GPU( configurations*, int*, float*, float*, float* );
__global__ void statistical_cuts_GPU( configurations*, uint, int*, int*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, bool* );
__global__ void construct_sinogram_GPU( configurations*, int*, float* );
__global__ void filter_GPU( configurations*, float*, float* );
__global__ void backprojection_GPU( configurations*, float*, float* );
__global__ void x_FBP_2_hull_GPU( configurations*, float*, bool* );

// Hull-Detection 
template<typename T> __global__ void initialize_hull_GPU( configurations*, T* );
__global__ void SC_GPU( configurations*, const uint, bool*, int*, bool*, float*, float*, float*, float*, float*, float*, float* );
__global__ void MSC_GPU( configurations*, const uint, int*, int*, bool*, float*, float*, float*, float*, float*, float*, float* );
__global__ void SM_GPU( configurations*, const uint, int*, int*, bool*, float*, float*, float*, float*, float*, float*, float* );
__global__ void MSC_edge_detection_GPU( configurations*, int* );
__global__ void SM_edge_detection_GPU( configurations*, int*, int* );
__global__ void SM_edge_detection_GPU_2( configurations*, int*, int* );
__global__ void carve_differences( configurations*, int*, int* );
template<typename H, typename D> __global__ void averaging_filter_GPU( configurations*, H*, D*, int, bool, double );
template<typename D> __global__ void apply_averaging_filter_GPU( configurations*, D*, D* );

// MLP: IN DEVELOPMENT
template<typename O> __device__ bool find_MLP_endpoints_GPU( configurations*, O*&, double, double, double, double, double, double&, double&, double&, int&, int&, int&, bool);
__device__ int find_MLP_path_GPU( configurations*, int*&, double*&, double, double, double, double, double, double, double, double, double, double, int, int, int );
__device__ void MLP_GPU(configurations*);

// Image Reconstruction
__global__ void create_hull_image_hybrid_GPU( configurations*, bool*&, float*& );
//template< typename X> __device__ double update_vector_multiplier2( configurations*, double, double, X*&, int*, int );
__device__ double scalar_dot_product_GPU_2( configurations*, double, float*&, int*, int );
__device__ double update_vector_multiplier_GPU_22( configurations*, double, double, float*&, int*, int );
//template< typename X> __device__ void update_iterate2( configurations*, double, double, X*&, int*, int );
__device__ void update_iterate_GPU_22( configurations*, double, double, float*&, int*, int );
__global__ void update_x_GPU( configurations*, float*&, double*&, unsigned int*& );

// Image position/voxel calculation functions
__device__ int calculate_voxel_GPU( configurations*, double, double, double );
__device__ int positions_2_voxels_GPU(configurations*, const double, const double, const double, int&, int&, int& );
__device__ int position_2_voxel_GPU( configurations*, double, double, double );
__device__ void voxel_2_3D_voxels_GPU( configurations*, int, int&, int&, int& );
__device__ double voxel_2_position_GPU( configurations*, int, double, int, int );
__device__ void voxel_2_positions_GPU( configurations*, int, double&, double&, double& );
__device__ double voxel_2_radius_squared_GPU( configurations*, int );

// Voxel walk algorithm functions
__device__ double distance_remaining_GPU( configurations*, double, double, int, int, double, int );
__device__ double edge_coordinate_GPU( configurations*, double, int, double, int, int );
__device__ double path_projection_GPU( configurations*, double, double, double, int, double, int, int );
__device__ double corresponding_coordinate_GPU( configurations*, double, double, double, double );
__device__ void take_2D_step_GPU( configurations*, const int, const int, const int, const double, const double, const double, const double, const double, const double, const double, const double, const double, double&, double&, double&, int&, int&, int&, int&, double&, double&, double& );
__device__ void take_3D_step_GPU( configurations*, const int, const int, const int, const double, const double, const double, const double, const double, const double, const double, const double, const double, double&, double&, double&, int&, int&, int&, int&, double&, double&, double& );

// Device helper functions

// New routine test functions
__global__ void test_func_GPU( configurations*, int* );
__global__ void test_func_device( configurations*, double*, double*, double* );

/***********************************************************************************************************************************************************************************************************************/
/**************** Functions for reading configuration file, setting associated option/parameter variables, and enforcement of data storage naming/organization scheme for all data/folders generated  ******************/
/***********************************************************************************************************************************************************************************************************************/
void set_configs_2_defines()
{
	/*parameters.NUM_SCANS_D = NUM_SCANS;
	parameters.GANTRY_ANGLE_INTERVAL_D = GANTRY_ANGLE_INTERVAL;
	parameters.SSD_T_SIZE_D = SSD_T_SIZE;
	parameters.SSD_V_SIZE_D = SSD_V_SIZE;
	parameters.T_SHIFT_D = T_SHIFT;
	parameters.U_SHIFT_D = U_SHIFT;
	parameters.T_BIN_SIZE_D = T_BIN_SIZE;
	parameters.T_BINS_D = T_BINS;
	parameters.V_BIN_SIZE_D = V_BIN_SIZE;
	parameters.V_BINS_D = V_BINS;
	parameters.ANGULAR_BIN_SIZE_D = ANGULAR_BIN_SIZE;
	parameters.SIGMAS_2_KEEP_D = SIGMAS_2_KEEP;
	parameters.RECON_CYL_RADIUS_D = RECON_CYL_RADIUS;
	parameters.RECON_CYL_HEIGHT_D = RECON_CYL_HEIGHT;
	parameters.IMAGE_WIDTH_D = IMAGE_WIDTH;
	parameters.IMAGE_HEIGHT_D = IMAGE_HEIGHT;
	parameters.IMAGE_THICKNESS_D = IMAGE_THICKNESS;
	parameters.COLUMNS_D = COLUMNS;
	parameters.ROWS_D = ROWS;
	parameters.SLICES_D = SLICES;
	parameters.VOXEL_WIDTH_D = VOXEL_WIDTH;
	parameters.VOXEL_HEIGHT_D = VOXEL_HEIGHT;
	parameters.VOXEL_THICKNESS_D = VOXEL_THICKNESS;
	parameters.LAMBDA_D = LAMBDA;
	parameters.LAMBDA = LAMBDA;*/
}
void set_defines_2_configs()
{
	//NUM_SCANS = parameters.NUM_SCANS_D;
	//GANTRY_ANGLE_INTERVAL = parameters.GANTRY_ANGLE_INTERVAL_D;
	//SSD_T_SIZE = parameters.SSD_T_SIZE_D;
	//SSD_V_SIZE = parameters.SSD_V_SIZE_D;
	//T_SHIFT = parameters.T_SHIFT_D;
	//U_SHIFT = parameters.U_SHIFT_D;
	//T_BIN_SIZE = parameters.T_BIN_SIZE_D;
	//T_BINS = parameters.T_BINS_D;
	//V_BIN_SIZE = parameters.V_BIN_SIZE_D;
	//V_BINS = parameters.V_BINS_D;
	//ANGULAR_BIN_SIZE = parameters.ANGULAR_BIN_SIZE_D;
	//SIGMAS_2_KEEP = parameters.SIGMAS_2_KEEP_D;
	//RECON_CYL_RADIUS = parameters.RECON_CYL_RADIUS_D;
	//RECON_CYL_HEIGHT = parameters.RECON_CYL_HEIGHT_D;
	//IMAGE_WIDTH = parameters.IMAGE_WIDTH_D;
	//IMAGE_HEIGHT = parameters.IMAGE_HEIGHT_D;
	//IMAGE_THICKNESS = parameters.IMAGE_THICKNESS_D;
	//COLUMNS = parameters.COLUMNS_D;
	//ROWS = parameters.ROWS_D;
	//SLICES = parameters.SLICES_D;
	//VOXEL_WIDTH = parameters.VOXEL_WIDTH_D;
	//VOXEL_HEIGHT = parameters.VOXEL_HEIGHT_D;
	//VOXEL_THICKNESS = parameters.VOXEL_THICKNESS_D;
	//LAMBDA = parameters.LAMBDA_D;
}
void export_D_configuration_parameters()
{
	char run_settings_filename[512];
	puts("Writing configuration_parameters struct elements to disk...");

	sprintf(run_settings_filename, "%s\\%s", PREPROCESSING_DIR, CONFIG_FILENAME);

	std::ofstream run_settings_file(run_settings_filename);		
	if( !run_settings_file.is_open() ) {
		printf("ERROR: run settings file not created properly %s!\n", run_settings_filename);	
		exit_program_if(true);
	}
	char buf[64];
	run_settings_file.setf (std::ios_base::showpoint);
	/*run_settings_file << "PROJECTION_DATA_DIR = "		<<  "\""	<< parameters.PROJECTION_DATA_DIR						<< "\"" << std::endl;
	run_settings_file << "PREPROCESSING_DIR_D = "		<<  "\""	<< parameters.PREPROCESSING_DIR_D						<< "\"" << std::endl;
	run_settings_file << "RECONSTRUCTION_DIR_D = "		<<  "\""	<< parameters.RECONSTRUCTION_DIR_D						<< "\"" << std::endl;
	run_settings_file << "OBJECT_D = "					<<  "\""	<< parameters.OBJECT_D									<< "\"" << std::endl;
	run_settings_file << "RUN_DATE_D = "				<<  "\""	<< parameters.RUN_DATE_D								<< "\"" << std::endl;
	run_settings_file << "RUN_NUMBER_D = "				<<  "\""	<< parameters.RUN_NUMBER_D								<< "\"" << std::endl;
	run_settings_file << "PROJECTION_DATA_DATE_D = "	<<  "\""	<< parameters.PROJECTION_DATA_DATE_D					<< "\"" << std::endl;
	run_settings_file << "RUN_NUMBER_D = "				<<  "\""	<< parameters.RUN_NUMBER_D								<< "\"" << std::endl;
	run_settings_file << "PREPROCESS_DATE_D = "			<<  "\""	<< parameters.PREPROCESS_DATE_D							<< "\"" << std::endl;
	run_settings_file << "RECONSTRUCTION_DATE_D = "		<<  "\""	<< parameters.RECONSTRUCTION_DATE_D						<< "\"" << std::endl;*/

	run_settings_file << "PROJECTION_DATA_DIR = "	<<  "\""	<< PROJECTION_DATA_DIR								<< "\"" << std::endl;
	run_settings_file << "PREPROCESSING_DIR = "		<<  "\""	<< PREPROCESSING_DIR								<< "\"" << std::endl;
	run_settings_file << "RECONSTRUCTION_DIR = "	<<  "\""	<< RECONSTRUCTION_DIR								<< "\"" << std::endl;
	run_settings_file << "OBJECT = "				<<  "\""	<< OBJECT											<< "\"" << std::endl;
	run_settings_file << "RUN_DATE = "				<<  "\""	<< RUN_DATE											<< "\"" << std::endl;
	run_settings_file << "RUN_NUMBER = "			<<  "\""	<< RUN_NUMBER										<< "\"" << std::endl;
	run_settings_file << "PROJECTION_DATA_DATE = "	<<  "\""	<< PROJECTION_DATA_DATE								<< "\"" << std::endl;
	run_settings_file << "RUN_NUMBER = "			<<  "\""	<< RUN_NUMBER										<< "\"" << std::endl;
	run_settings_file << "PREPROCESS_DATE = "		<<  "\""	<< PREPROCESS_DATE									<< "\"" << std::endl;
	run_settings_file << "RECONSTRUCTION_DATE = "	<<  "\""	<< RECONSTRUCTION_DATE								<< "\"" << std::endl;

	run_settings_file << "NUM_SCANS = "				<< parameters.NUM_SCANS_D												<< std::endl;
	run_settings_file << "T_BINS = "				<< parameters.T_BINS_D													<< std::endl;
	run_settings_file << "V_BINS = "				<< parameters.V_BINS_D													<< std::endl;
	run_settings_file << "COLUMNS = "				<< parameters.COLUMNS_D													<< std::endl;
	run_settings_file << "ROWS = "					<< parameters.ROWS_D													<< std::endl;
	run_settings_file << "SLICES = "				<< parameters.SLICES_D													<< std::endl;
	run_settings_file << "SIGMAS_2_KEEP = "			<< parameters.SIGMAS_2_KEEP_D											<< std::endl;
	run_settings_file << "GANTRY_ANGLE_INTERVAL = "	<< minimize_trailing_zeros( parameters.GANTRY_ANGLE_INTERVAL_D, buf	)	<< std::endl;
	run_settings_file << "ANGULAR_BIN_SIZE = "		<< minimize_trailing_zeros( parameters.ANGULAR_BIN_SIZE_D, buf )		<< std::endl;
	run_settings_file << "SSD_T_SIZE = "			<< minimize_trailing_zeros( parameters.SSD_T_SIZE_D, buf )				<< std::endl;
	run_settings_file << "SSD_V_SIZE = "			<< minimize_trailing_zeros( parameters.SSD_V_SIZE_D, buf )				<< std::endl;
	run_settings_file << "T_SHIFT = "				<< minimize_trailing_zeros( parameters.T_SHIFT_D, buf )					<< std::endl;
	run_settings_file << "U_SHIFT = "				<< minimize_trailing_zeros( parameters.U_SHIFT_D, buf )					<< std::endl;
	run_settings_file << "T_BIN_SIZE = "			<< minimize_trailing_zeros( parameters.T_BIN_SIZE_D, buf )				<< std::endl;	
	run_settings_file << "V_BIN_SIZE = "			<< minimize_trailing_zeros( parameters.V_BIN_SIZE_D, buf )				<< std::endl;		
	run_settings_file << "RECON_CYL_RADIUS = "		<< minimize_trailing_zeros( parameters.RECON_CYL_RADIUS_D, buf )		<< std::endl;
	run_settings_file << "RECON_CYL_HEIGHT = "		<< minimize_trailing_zeros( parameters.RECON_CYL_HEIGHT_D, buf )		<< std::endl;
	run_settings_file << "IMAGE_WIDTH = "			<< minimize_trailing_zeros( parameters.IMAGE_WIDTH_D, buf )				<< std::endl;
	run_settings_file << "IMAGE_HEIGHT = "			<< minimize_trailing_zeros( parameters.IMAGE_HEIGHT_D, buf )			<< std::endl;
	run_settings_file << "IMAGE_THICKNESS = "		<< minimize_trailing_zeros( parameters.IMAGE_THICKNESS_D, buf )			<< std::endl;
	run_settings_file << "VOXEL_WIDTH = "			<< minimize_trailing_zeros( parameters.VOXEL_WIDTH_D, buf )				<< std::endl;
	run_settings_file << "VOXEL_HEIGHT = "			<< minimize_trailing_zeros( parameters.VOXEL_HEIGHT_D, buf )			<< std::endl;
	run_settings_file << "VOXEL_THICKNESS = "		<< minimize_trailing_zeros( parameters.VOXEL_THICKNESS_D, buf )			<< std::endl;
	run_settings_file << "LAMBDA = "				<< minimize_trailing_zeros( parameters.LAMBDA_D, buf )					<< std::endl;
	run_settings_file << "LAMBDA = "				<< minimize_trailing_zeros( parameters.LAMBDA, buf )					<< std::endl;
	run_settings_file.close();	
}
void export_configuration_parameters()
{
	char run_settings_filename[512];
	puts("Writing configuration_parameters struct elements to disk...");

	sprintf(run_settings_filename, "%s\\%s", PREPROCESSING_DIR, CONFIG_FILENAME);

	std::ofstream run_settings_file(run_settings_filename);		
	if( !run_settings_file.is_open() ) {
		printf("ERROR: run settings file not created properly %s!\n", run_settings_filename);	
		exit_program_if(true);
	}
	char buf[64];
	run_settings_file.setf (std::ios_base::showpoint);
	/*run_settings_file << "PROJECTION_DATA_DIR = "	<<  "\""	<< PROJECTION_DATA_DIR								<< "\"" << std::endl;
	run_settings_file << "PREPROCESSING_DIR = "		<<  "\""	<< PREPROCESSING_DIR								<< "\"" << std::endl;
	run_settings_file << "RECONSTRUCTION_DIR = "	<<  "\""	<< RECONSTRUCTION_DIR								<< "\"" << std::endl;
	run_settings_file << "OBJECT = "				<<  "\""	<< OBJECT											<< "\"" << std::endl;
	run_settings_file << "RUN_DATE = "				<<  "\""	<< RUN_DATE											<< "\"" << std::endl;
	run_settings_file << "RUN_NUMBER = "			<<  "\""	<< RUN_NUMBER										<< "\"" << std::endl;
	run_settings_file << "PROJECTION_DATA_DATE = "	<<  "\""	<< PROJECTION_DATA_DATE								<< "\"" << std::endl;
	run_settings_file << "RUN_NUMBER = "			<<  "\""	<< RUN_NUMBER										<< "\"" << std::endl;
	run_settings_file << "PREPROCESS_DATE = "		<<  "\""	<< PREPROCESS_DATE									<< "\"" << std::endl;
	run_settings_file << "RECONSTRUCTION_DATE = "	<<  "\""	<< RECONSTRUCTION_DATE								<< "\"" << std::endl;
	run_settings_file << "NUM_SCANS = "							<<  NUM_SCANS												<< std::endl;
	run_settings_file << "T_BINS = "							<<  T_BINS													<< std::endl;
	run_settings_file << "V_BINS = "							<<  V_BINS													<< std::endl;
	run_settings_file << "COLUMNS = "							<<  COLUMNS													<< std::endl;
	run_settings_file << "ROWS = "								<<  ROWS													<< std::endl;
	run_settings_file << "SLICES = "							<<  SLICES													<< std::endl;
	run_settings_file << "SIGMAS_2_KEEP = "						<<  SIGMAS_2_KEEP											<< std::endl;
	run_settings_file << "GANTRY_ANGLE_INTERVAL = "				<< minimize_trailing_zeros( GANTRY_ANGLE_INTERVAL, buf )	<< std::endl;
	run_settings_file << "ANGULAR_BIN_SIZE = "					<< minimize_trailing_zeros( ANGULAR_BIN_SIZE, buf )			<< std::endl;
	run_settings_file << "SSD_T_SIZE = "						<< minimize_trailing_zeros( SSD_T_SIZE, buf )				<< std::endl;
	run_settings_file << "SSD_V_SIZE = "						<< minimize_trailing_zeros( SSD_V_SIZE, buf )				<< std::endl;
	run_settings_file << "T_SHIFT = "							<< minimize_trailing_zeros( T_SHIFT, buf )					<< std::endl;
	run_settings_file << "U_SHIFT = "							<< minimize_trailing_zeros( U_SHIFT, buf )					<< std::endl;
	run_settings_file << "T_BIN_SIZE = "						<< minimize_trailing_zeros( T_BIN_SIZE, buf )				<< std::endl;	
	run_settings_file << "V_BIN_SIZE = "						<< minimize_trailing_zeros( V_BIN_SIZE, buf )				<< std::endl;		
	run_settings_file << "RECON_CYL_RADIUS = "					<< minimize_trailing_zeros( RECON_CYL_RADIUS, buf )			<< std::endl;
	run_settings_file << "RECON_CYL_HEIGHT = "					<< minimize_trailing_zeros( RECON_CYL_HEIGHT, buf )			<< std::endl;
	run_settings_file << "IMAGE_WIDTH = "						<< minimize_trailing_zeros( IMAGE_WIDTH, buf )				<< std::endl;
	run_settings_file << "IMAGE_HEIGHT = "						<< minimize_trailing_zeros( IMAGE_HEIGHT, buf )				<< std::endl;
	run_settings_file << "IMAGE_THICKNESS = "					<< minimize_trailing_zeros( IMAGE_THICKNESS, buf )			<< std::endl;
	run_settings_file << "VOXEL_WIDTH = "						<< minimize_trailing_zeros( VOXEL_WIDTH, buf )				<< std::endl;
	run_settings_file << "VOXEL_HEIGHT = "						<< minimize_trailing_zeros( VOXEL_HEIGHT, buf )				<< std::endl;
	run_settings_file << "VOXEL_THICKNESS = "					<< minimize_trailing_zeros( VOXEL_THICKNESS, buf )			<< std::endl;
	run_settings_file << "LAMBDA = "							<< minimize_trailing_zeros( LAMBDA, buf )					<< std::endl;
	run_settings_file.close();*/
}
void set_data_info( char * path)
{
	//char OBJECT[]					= "Object";
	//char* DATA_TYPE_DIR;
	//char RUN_DATE[]					= "MMDDYYYY";
	//char RUN_NUMBER[]				= "Run";
	//char PREPROCESS_DATE[]			= "MMDDYYYY";
	//char RECONSTRUCTION_DIR[]		= "Reconstruction";
	//char RECONSTRUCTION_DATE[]				= "MMDDYYYY";
	//char PCT_IMAGES[]				= "Images";
	//char REFERENCE_IMAGES[]			= "Reference_Images";
	//char TEST_OUTPUT_FILE[]			= "export_testing.cfg";
	char * pch = strtok (path,  "\\ : \n");
	  while (pch != NULL)
	  {
		printf ("%s\n",pch);
		pch = strtok (NULL, "\\ : \n");
		if( strcmp(pch, "pCT_Data") == 0) 
			break;
	  }
	pch = strtok (NULL, "\\ : \n");
	if(pch != NULL)
		if( strcmp(pch, "object_name") == 0)
			cout << "object_name found" << endl;
	pch = strtok (NULL, "\\ : \n");
	if(pch != NULL)
		if( strcmp(pch, "Experimental") == 0)
			cout << "Experimental found" << endl;
	pch = strtok (NULL, "\\ : \n");
	if(pch != NULL)
		if( strcmp(pch, "DDMMYYYY") == 0)
			cout << "DDMMYYYY found" << endl;
}
bool preprocessing_data_exists()
{
	return true;
}
//void add_log_entry()
//{
//	// PATH_2_PCT_DATA_DIR/LOG_FILENAME
//	char buf[64];	
//	//char* open_quote_pos;
//	const uint buf_size = 1024;
//	char log_path[256];
//	sprintf( log_path, "%s/%s", PATH_2_PCT_DATA_DIR, LOG_FILENAME );
//	FILE* log_file = fopen(log_path, "r+" );
//	char line[buf_size];
//	//int string_leng5th;
//	//generic_IO_container input_value;
//	// Remove leading spaces/tabs and return first line which does not begin with comment command "//" and is not blank
//	char item[256], item_name[256], remainder[256];
//	char object[256], scan_type[256], run_date[256], run_num[256], proj_date[256], pre_date[256], recon_date[256];
//	
//	fgets_validated(line, buf_size, log_file);
//	sscanf (line, "%s: %s", &item, &object );
//	//while( strcmp( object, OBJECT ) != 0 )
//	printf("%s = %s\n", line, object );
//	while( strcmp( item, "Object" ) != 0 )
//	{
//		fgets_validated(line, buf_size, log_file);
//		sscanf (line, "%s: %s", &item, &object );
//		printf("%s = %s\n", line, object );
//		if( feof(log_file) )
//		{
//			return;
//			puts("Return");
//		}
//	}
//	//if( strcmp( object, "object_name" ) < 0)
//
//	//else
//
//	fgets_validated(line, buf_size, log_file);
//	sscanf (line, "Object: %s", &object );
//	fprintf(log_file, "Hello");
//	//sscanf (line, " %s = %s %s", &item, &item_name, &remainder );idated(log_file, line, buf_size);
//	//if( strcmp( line, OBJECT ) == 0 )
//	//{
//	//	fgets_validated(log_file, line, buf_size);
//	//	fprintf (log_file, "\n" );
//	//}
//	//else
//	//	fprintf (log_file, "Object = %s\n", OBJECT);
//	//puts("Hello");
//	//fprintf( log_file, "\tScan Type = %s : %s %s %s %s %s ", SCAN_TYPE, RUN_DATE, RUN_NUMBER, PROJECTION_DATA_DATE, PREPROCESS_DATE, RECONSTRUCTION_DATE);
//	////fwrite( &reconstruction_histories, sizeof(unsigned int), 1, log_file );
//	fclose(log_file);
//}
//bool find_log_item( FILE* log_file, char* log_item, char* log_item_name )
//{
//	char buf[64];	
//	//char* open_quote_pos;
//	const uint buf_size = 1024;
//	char key_value_pair[256];
//	char item[256], item_name[256], remainder[256];
//	sprintf( key_value_pair, "%s = %s", log_item, log_item_name );
//	char line[buf_size];
//
//	fgets_validated(log_file, line, buf_size);
//	sscanf (line, " %s = %s %s", &item, &item_name, &remainder );
//	while( strcmp( line, item ) < 0 )
//	{
//		fgets_validated(log_file, line, buf_size);
//		sscanf (line, " %s = %s %s", &item, &item_name, &remainder );
//	}
//	if( strcmp( line, log_item ) == 0 )
//		fprintf (log_file, "\nObject: %s\n", log_item );
//	else
//		fprintf (log_file, "\n" );
//	return true;
//}
void add_object_directory(char* pct_data_dir, char* object_name)
{
	char mkdir_command[128];
	sprintf(mkdir_command, "mkdir %s\\%s", pct_data_dir, object_name );
	system(mkdir_command);
}
int add_run_directory(char* pct_data_dir, char* object_name, char* run_date, char* run_number, SCAN_TYPES data_type )
{

	char mkdir_command[256];
	char data_directory[256];
	char images_directory[256];
	char data_type_dir[15];
	char run_date_dir[256];

	if( data_type == EXPERIMENTAL )
	{
		strcpy(data_type_dir, "Experimental");
	}
	else if( data_type == SIMULATED_G || data_type == SIMULATED_T )
	{
		strcpy(data_type_dir, "Simulated");
	}
	else
	{
		puts("ERROR: Invalid data type; must be EXPERIMENTAL or SIMULATED ");
		exit(1);
	}
	sprintf(run_date_dir, "\"%s\\pCT_Data\\%s\\%s\\%s\\%s\"", pct_data_dir, object_name, data_type_dir, run_date, run_number );

	char options[3] = {'q','c','d'};
	char* cin_message = "Enter 'o' to overwrite any existing data, 'd' to create numbered duplicate of directory, or 'q' to quit program";

	int i = 0;
	if( directory_exists(run_date_dir))
	{
		//puts(run_date_dir);
		puts("ERROR: Directory (and possibly data) already exists for this run date/number");
		switch(cin_until_valid( options, 3, cin_message ) )
		{
		case 'd':	i = create_unique_dir( run_date_dir );
					printf("duplicating directory with _%d added to date directory name", i);		break;
		case 'q':	puts("c selected"); exit(1);												break;
		case 'o':	puts("overwriting existing data"); 
		}
	}
	//if( i > 0 )
		//sprintf(run_number, "%s_%d",run_number, i );
	puts(run_number);
	sprintf(data_directory, "%s\\Data", run_date_dir );
	sprintf(images_directory, "%s\\Images", run_date_dir );

	sprintf(mkdir_command, "mkdir %s\\Input", data_directory );
	//puts(mkdir_command);
	system(mkdir_command);
	sprintf(mkdir_command, "mkdir %s\\Output", data_directory );
	//puts(mkdir_command);
	system(mkdir_command);

	sprintf(mkdir_command, "mkdir %s\\pCT_Images", images_directory );
	//puts(mkdir_command);
	system(mkdir_command);
	sprintf(mkdir_command, "mkdir %s\\pCT_Images\\DDMMYYYY", images_directory );
	//puts(mkdir_command);
	system(mkdir_command);
	sprintf(mkdir_command, "mkdir %s\\Reference_Images", images_directory );
	//puts(mkdir_command);
	system(mkdir_command);
	return i;
}
int add_pCT_Images_dir(char* pct_data_dir, char* object_name, char* run_date, char* run_number, SCAN_TYPES data_type )
{
	char pCT_Images_directory[256];
	char data_type_dir[15];
	char pCT_Images_date[9];

	if( data_type == EXPERIMENTAL )
	{
		strcpy(data_type_dir, "Experimental");
	}
	else if( data_type == SIMULATED_G || data_type == SIMULATED_T)
	{
		strcpy(data_type_dir, "Simulated");
	}
	else
	{
		puts("ERROR: Invalid data type; must be EXPERIMENTAL or SIMULATED ");
		exit(1);
	}
	current_MMDDYYYY(pCT_Images_date);
	sprintf(pCT_Images_directory, "\"%s\\pCT_Data\\%s\\%s\\%s\\%s\\Images\\pCT_Images\\%s\"", pct_data_dir, object_name, data_type_dir, run_date, run_number, pCT_Images_date );	
	return create_unique_dir( pCT_Images_directory );
}

void write_reconstruction_settings() 
{
	FILE* settings_file = fopen("reconstruction_settings.txt", "w");
	time_t rawtime;
	struct tm * timeinfo;

	time (&rawtime);
	timeinfo = localtime (&rawtime);
	fprintf (settings_file, "Current local time and date: %s", asctime(timeinfo));
	fprintf(settings_file, "PRIME_OFFSET = %d \n",  PRIME_OFFSET);
	fprintf(settings_file, "AVG_FILTER_HULL = %s \n",  bool_2_string(parameters.AVG_FILTER_HULL_D));
	
	fprintf(settings_file, "HULL_FILTER_RADIUS = %d \n",  parameters.HULL_FILTER_RADIUS_D);
	fprintf(settings_file, "HULL_FILTER_THRESHOLD = %d \n",  parameters.HULL_FILTER_THRESHOLD_D);
	fprintf(settings_file, "LAMBDA = %d \n",  parameters.LAMBDA_D);

	// fwrite( &reconstruction_histories, sizeof(unsigned int), 1, export_histories );
	switch( parameters.X_0_TYPE )
	{
		case X_HULL:		fprintf(settings_file, "x_0 = X_HULL\n");		break;
		case X_FBP:		fprintf(settings_file, "x_0 = x_FBP\n");	break;
		case HYBRID:		fprintf(settings_file, "x_0 = HYBRID\n");		break;
		case ZEROS:			fprintf(settings_file, "x_0 = ZEROS\n");		break;
		case IMPORT_X_0:		fprintf(settings_file, "x_0 = IMPORT\n");		break;
	}
	fprintf(settings_file, "IMPORT_FILTERED_FBP = %d \n", bool_2_string(parameters.IMPORT_FILTERED_FBP_D) );
	if( parameters.IMPORT_FILTERED_FBP_D )
	{
		fprintf(settings_file, "FILTERED_FBP_PATH = %d \n",  FBP_PATH);
	}
	switch( parameters.RECONSTRUCTION_METHOD )
	{
		case ART:		fprintf(settings_file, "RECON_ALGORITHM = ART\n");		break;
		case BIP:		fprintf(settings_file, "RECON_ALGORITHM = BIP\n");	break;
		case DROP:		fprintf(settings_file, "RECON_ALGORITHM = DROP\n");	break;
		case SAP:		fprintf(settings_file, "RECON_ALGORITHM = SAP\n");	break;
		case ROBUST1:			fprintf(settings_file, "RECON_ALGORITHM = ROBUST1\n");		break;
		case ROBUST2:		fprintf(settings_file, "RECON_ALGORITHM = ROBUST2\n");		break;
	}
	fclose(settings_file);
}
void read_config_file()
{		
	// Extract current directory (executable path) terminal response from system command "chdir" 
	//cout <<  terminal_response("echo %cd%") << endl;
	if( !CONFIG_PATH_PASSED )
	{
		std::string str =  terminal_response("chdir");
		const char* cstr = str.c_str();
		PROJECTION_DATA_DIR = (char*) calloc( strlen(cstr), sizeof(char));
		std::copy( cstr, &cstr[strlen(cstr)-1], PROJECTION_DATA_DIR );
		print_section_header( "Config file location set to current execution directory :", '*' );	
		print_section_separator('-');
		printf("%s\n", PROJECTION_DATA_DIR );
		print_section_separator('-');
	}
	CONFIG_FILE_PATH  = (char*) calloc( strlen(PROJECTION_DATA_DIR) + strlen(CONFIG_FILENAME) + 1, sizeof(char) );
	sprintf(CONFIG_FILE_PATH, "%s/%s", PROJECTION_DATA_DIR, CONFIG_FILENAME );
	FILE* input_file = fopen(CONFIG_FILE_PATH, "r" );
	print_section_header( "Reading key/value pairs from configuration file and setting corresponding execution parameters", '*' );
	while( !feof(input_file) )
	{		
		generic_IO_container input_value = read_key_value_pair(input_file);
		set_parameter( input_value );
		if( input_value.input_type_ID > STRING )
			puts("invalid type_ID");
	}
	fclose(input_file);
	print_section_exit( "Finished reading configuration file and setting execution parameters", "====>" );
}
bool key_is_string_parameter( char* key )
{
	if
	( 
			strcmp (key, "PROJECTION_DATA_DIR") == 0 
		||	strcmp (key, "PREPROCESSING_DIR") == 0 
		||	strcmp (key, "RECONSTRUCTION_DIR") == 0 
		||	strcmp (key, "PATH_2_PCT_DATA_DIR") == 0 
		||	strcmp (key, "OBJECT") == 0 
		||	strcmp (key, "RUN_DATE") == 0 
		||	strcmp (key, "RUN_NUMBER") == 0 
		||	strcmp (key, "PROJECTION_DATA_DATE") == 0 
		||	strcmp (key, "PREPROCESS_DATE") == 0 
		||	strcmp (key, "RECONSTRUCTION_DATE") == 0 
		||	strcmp (key, "USER_NAME") == 0
	)
		return true;
	else
		return false;
}
bool key_is_floating_point_parameter( char* key )
{
	if
	( 
			strcmp (key, "GANTRY_ANGLE_INTERVAL") == 0 
		||	strcmp (key, "SSD_T_SIZE") == 0 
		||	strcmp (key, "SSD_V_SIZE") == 0 
		||	strcmp (key, "T_SHIFT") == 0 
		||	strcmp (key, "U_SHIFT") == 0 
		||	strcmp (key, "V_SHIFT") == 0 
		||	strcmp (key, "T_BIN_SIZE") == 0 
		||	strcmp (key, "V_BIN_SIZE") == 0 
		||	strcmp (key, "ANGULAR_BIN_SIZE") == 0 
		||	strcmp (key, "RECON_CYL_RADIUS") == 0 
		||	strcmp (key, "RECON_CYL_HEIGHT") == 0 
		||	strcmp (key, "IMAGE_WIDTH") == 0 
		||	strcmp (key, "IMAGE_HEIGHT") == 0 
		||	strcmp (key, "IMAGE_THICKNESS") == 0 
		||	strcmp (key, "VOXEL_WIDTH") == 0 
		||	strcmp (key, "VOXEL_HEIGHT") == 0 
		||	strcmp (key, "VOXEL_THICKNESS") == 0 
		||	strcmp (key, "SLICE_THICKNESS") == 0 
		||	strcmp (key, "LAMBDA") == 0 
		||	strcmp (key, "ETA") == 0 
		||	strcmp (key, "HULL_FILTER_THRESHOLD") == 0 
		||	strcmp (key, "FBP_AVG_THRESHOLD") == 0 
		||	strcmp (key, "X_0_FILTER_THRESHOLD") == 0 
		||	strcmp (key, "SC_THRESHOLD") == 0 
		||	strcmp (key, "MSC_THRESHOLD") == 0 
		||	strcmp (key, "SM_LOWER_THRESHOLD") == 0 
		||	strcmp (key, "SM_UPPER_THRESHOLD") == 0 
		||	strcmp (key, "SM_SCALE_THRESHOLD") == 0  
	)
		return true;
	else
		return false;
}
bool key_is_integer_parameter( char* key )
{
	if
	( 
			strcmp (key, "DATA_TYPE") == 0
		|| 	strcmp (key, "HULL_TYPE") == 0
		||	strcmp (key, "FBP_FILTER_TYPE") == 0
		||	strcmp (key, "X_0_TYPE") == 0
		||	strcmp (key, "RECONSTRUCTION_METHOD") == 0
		||	strcmp (key, "NUM_SCANS") == 0
		||	strcmp (key, "MAX_GPU_HISTORIES") == 0
		||	strcmp (key, "MAX_CUTS_HISTORIES") == 0
		||	strcmp (key, "T_BINS") == 0
		||	strcmp (key, "V_BINS") == 0
		||	strcmp (key, "SIGMAS_2_KEEP") == 0
		||	strcmp (key, "COLUMNS") == 0
		||	strcmp (key, "ROWS") == 0
		||	strcmp (key, "SLICES") == 0
		||	strcmp (key, "ITERATIONS") == 0
		||	strcmp (key, "BLOCK_SIZE") == 0
		||	strcmp (key, "HULL_FILTER_RADIUS") == 0
		||	strcmp (key, "X_0_FILTER_RADIUS") == 0
		||	strcmp (key, "FBP_AVG_RADIUS") == 0
		||	strcmp (key, "FBP_MEDIAN_RADIUS") == 0
		||	strcmp (key, "PSI_SIGN") == 0
		||	strcmp (key, "MSC_DIFF_THRESH") == 0
	)
		return true;
	else
		return false;
}
bool key_is_boolean_parameter( char* key )
{
	if
	( 
			strcmp (key, "ADD_DATA_LOG_ENTRY") == 0
		||	strcmp (key, "CONSOLE_OUTPUT_2_DISK") == 0
		||	strcmp (key, "IMPORT_PREPROCESSING") == 0
		||	strcmp (key, "PERFORM_RECONSTRUCTION") == 0
		||	strcmp (key, "PREPROCESS_OVERWRITE_OK") == 0
		||	strcmp (key, "RECON_OVERWRITE_OK") == 0
		||	strcmp (key, "FBP_ON") == 0
		||	strcmp (key, "AVG_FILTER_FBP") == 0
		||	strcmp (key, "MEDIAN_FILTER_FBP") == 0
		||	strcmp (key, "IMPORT_FILTERED_FBP") == 0
		||	strcmp (key, "SC_ON") == 0
		||	strcmp (key, "MSC_ON") == 0
		||	strcmp (key, "SM_ON") == 0
		||	strcmp (key, "AVG_FILTER_HULL") == 0
		||	strcmp (key, "AVG_FILTER_ITERATE") == 0
		||	strcmp (key, "WRITE_MSC_COUNTS") == 0
		||	strcmp (key, "WRITE_SM_COUNTS") == 0
		||	strcmp (key, "WRITE_X_FBP") == 0
		||	strcmp (key, "WRITE_FBP_HULL") == 0
		||	strcmp (key, "WRITE_AVG_FBP") == 0
		||	strcmp (key, "WRITE_MEDIAN_FBP") == 0
		||	strcmp (key, "WRITE_BIN_WEPLS") == 0
		||	strcmp (key, "WRITE_WEPL_DISTS") == 0
		||	strcmp (key, "WRITE_SSD_ANGLES") == 0 
	)
		return true;
	else
		return false;
}
void set_string_parameter( generic_IO_container &value )
{
	printf("set to \"%s\"\n", value.string_input);
	if( strcmp (value.key, "PROJECTION_DATA_DIR") == 0 )
	{		
		//print_section_separator('-');
		puts("");
		PROJECTION_DATA_DIR = (char*) calloc( strlen(value.string_input) + 1, sizeof(char));
		std::copy( value.string_input, &value.string_input[strlen(value.string_input)], PROJECTION_DATA_DIR );
		PROJECTION_DATA_DIR_SET = true;
	}
	else if( strcmp (value.key, "PREPROCESSING_DIR") == 0 )
	{
		puts("");
		//print_section_separator('-');
		PREPROCESSING_DIR = (char*) calloc( strlen(value.string_input) + 1, sizeof(char));
		std::copy( value.string_input, &value.string_input[strlen(value.string_input)], PREPROCESSING_DIR );
		PREPROCESSING_DIR_SET = true;
	}
	else if( strcmp (value.key, "RECONSTRUCTION_DIR") == 0 )
	{
		puts("");
		//print_section_separator('-');
		RECONSTRUCTION_DIR = (char*) calloc( strlen(value.string_input) + 1, sizeof(char));
		std::copy( value.string_input, &value.string_input[strlen(value.string_input)], RECONSTRUCTION_DIR );
		RECONSTRUCTION_DIR_SET = true;
	}
	else if( strcmp (value.key, "PATH_2_PCT_DATA_DIR") == 0 )
	{
		PATH_2_PCT_DATA_DIR = (char*) calloc( strlen(value.string_input) + 1, sizeof(char));
		std::copy( value.string_input, &value.string_input[strlen(value.string_input)], PATH_2_PCT_DATA_DIR );
		PATH_2_PCT_DATA_DIR_SET = true;
	}
	else if( strcmp (value.key, "OBJECT") == 0 )
	{
		OBJECT = (char*) calloc( strlen(value.string_input) + 1, sizeof(char));
		std::copy( value.string_input, &value.string_input[strlen(value.string_input)], OBJECT );
		OBJECT_SET = true;
	}
	else if( strcmp (value.key, "RUN_DATE") == 0 )
	{
		RUN_DATE = (char*) calloc( strlen(value.string_input) + 1, sizeof(char));
		std::copy( value.string_input, &value.string_input[strlen(value.string_input)], RUN_DATE );
		RUN_DATE_SET = true;
	}
	else if( strcmp (value.key, "RUN_NUMBER") == 0 )
	{
		RUN_NUMBER = (char*) calloc( strlen(value.string_input) + 1, sizeof(char));
		std::copy( value.string_input, &value.string_input[strlen(value.string_input)], RUN_NUMBER );
		RUN_NUMBER_SET = true;
	}
	else if( strcmp (value.key, "PROJECTION_DATA_DATE") == 0 )
	{
		PROJECTION_DATA_DATE = (char*) calloc( strlen(value.string_input) + 1, sizeof(char));
		std::copy( value.string_input, &value.string_input[strlen(value.string_input)], PROJECTION_DATA_DATE );
		PROJECTION_DATA_DATE_SET = true;
	}
	else if( strcmp (value.key, "PREPROCESS_DATE") == 0 )
	{
		PREPROCESS_DATE = (char*) calloc( strlen(value.string_input) + 1, sizeof(char));
		std::copy( value.string_input, &value.string_input[strlen(value.string_input)], PREPROCESS_DATE );
		PREPROCESS_DATE_SET = true;
	}
	else if( strcmp (value.key, "RECONSTRUCTION_DATE") == 0 )
	{
		RECONSTRUCTION_DATE = (char*) calloc( strlen(value.string_input) + 1, sizeof(char));
		std::copy( value.string_input, &value.string_input[strlen(value.string_input)], RECONSTRUCTION_DATE );
		RECONSTRUCTION_DATE_SET = true;
	}
	else if( strcmp (value.key, "USER_NAME") == 0 )
	{
		USER_NAME = (char*) calloc( strlen(value.string_input) + 1, sizeof(char));
		std::copy( value.string_input, &value.string_input[strlen(value.string_input)], USER_NAME );
		USER_NAME_SET = true;
	}
	else
	{
		puts("ERROR: Procedure for setting this key is undefined");
		exit_program_if(true);
	}
}
void set_floating_point_parameter( generic_IO_container &value )
{
	char buf[64];
	if( value.input_type_ID == INTEGER )
			printf("converted to a double and ");
	//printf("set to %s\n", minimize_trailing_zeros(value.double_input, buf));
	printf("set to %s\n", minimize_trailing_zeros(value.double_input, buf));
	if( strcmp (value.key, "GANTRY_ANGLE_INTERVAL") == 0 )
	{
		if( value.double_input < 0 )
		{
			puts("ERROR: Negative value give to parameter that should be positive.\n  Correct the configuration file and rerun program");
			exit_program_if(true);
		}
		//GANTRY_ANGLE_INTERVAL = value.double_input;
		parameters.GANTRY_ANGLE_INTERVAL_D = value.double_input;
	}
	else if( strcmp (value.key, "SSD_T_SIZE") == 0 )
	{
		if( value.double_input < 0 )
		{
			puts("ERROR: Negative value give to parameter that should be positive.\n  Correct the configuration file and rerun program");
			exit_program_if(true);
		}
		//SSD_T_SIZE = value.double_input;
		parameters.SSD_T_SIZE_D = value.double_input;
	}
	else if( strcmp (value.key, "SSD_V_SIZE") == 0 )
	{
		if( value.double_input < 0 )
		{
			puts("ERROR: Negative value give to parameter that should be positive.\n  Correct the configuration file and rerun program");
			exit_program_if(true);
		}
		//SSD_V_SIZE = value.double_input;
		parameters.SSD_V_SIZE_D = value.double_input;
	}
	else if( strcmp (value.key, "T_SHIFT") == 0 )
	{
		//T_SHIFT = value.double_input;
		parameters.T_SHIFT_D = value.double_input;
	}
	else if( strcmp (value.key, "U_SHIFT") == 0 )
	{
		//U_SHIFT = value.double_input;
		parameters.U_SHIFT_D = value.double_input;
	}
	else if( strcmp (value.key, "V_SHIFT") == 0 )
	{
		//V_SHIFT = value.double_input;
		parameters.V_SHIFT_D = value.double_input;
	}
	else if( strcmp (value.key, "T_BIN_SIZE") == 0 )
	{
		if( value.double_input < 0 )
		{
			puts("ERROR: Negative value give to parameter that should be positive.\n  Correct the configuration file and rerun program");
			exit_program_if(true);
		}
		//T_BIN_SIZE = value.double_input;
		parameters.T_BIN_SIZE_D = value.double_input;
	}
	else if( strcmp (value.key, "V_BIN_SIZE") == 0 )
	{
		if( value.double_input < 0 )
		{
			puts("ERROR: Negative value give to parameter that should be positive.\n  Correct the configuration file and rerun program");
			exit_program_if(true);
		}
		//V_BIN_SIZE = value.double_input;
		parameters.V_BIN_SIZE_D = value.double_input;
	}
	else if( strcmp (value.key, "ANGULAR_BIN_SIZE") == 0 )
	{
		if( value.double_input < 0 )
		{
			puts("ERROR: Negative value give to parameter that should be positive.\n  Correct the configuration file and rerun program");
			exit_program_if(true);
		}
		//ANGULAR_BIN_SIZE = value.double_input;
		parameters.ANGULAR_BIN_SIZE_D = value.double_input;
	}
	else if( strcmp (value.key, "RECON_CYL_RADIUS") == 0 )
	{
		if( value.double_input < 0 )
		{
			puts("ERROR: Negative value give to parameter that should be positive.\n  Correct the configuration file and rerun program");
			exit_program_if(true);
		}
		//RECON_CYL_RADIUS = value.double_input;
		parameters.RECON_CYL_RADIUS_D = value.double_input;
	}
	else if( strcmp (value.key, "RECON_CYL_HEIGHT") == 0 )
	{
		if( value.double_input < 0 )
		{
			puts("ERROR: Negative value give to parameter that should be positive.\n  Correct the configuration file and rerun program");
			exit_program_if(true);
		}
		//RECON_CYL_HEIGHT = value.double_input;
		parameters.RECON_CYL_HEIGHT_D = value.double_input;
	}
	else if( strcmp (value.key, "IMAGE_WIDTH") == 0 )
	{
		if( value.double_input < 0 )
		{
			puts("ERROR: Negative value give to parameter that should be positive.\n  Correct the configuration file and rerun program");
			exit_program_if(true);
		}
		//IMAGE_WIDTH = value.double_input;
		parameters.IMAGE_WIDTH_D = value.double_input;
	}
	else if( strcmp (value.key, "IMAGE_HEIGHT") == 0 )
	{
		if( value.double_input < 0 )
		{
			puts("ERROR: Negative value give to parameter that should be positive.\n  Correct the configuration file and rerun program");
			exit_program_if(true);
		}
		//IMAGE_HEIGHT = value.double_input;
		parameters.IMAGE_HEIGHT_D = value.double_input;
	}
	else if( strcmp (value.key, "IMAGE_THICKNESS") == 0 )
	{
		if( value.double_input < 0 )
		{
			puts("ERROR: Negative value give to parameter that should be positive.\n  Correct the configuration file and rerun program");
			exit_program_if(true);
		}
		//IMAGE_THICKNESS = value.double_input;
		parameters.IMAGE_THICKNESS_D = value.double_input;
	}
	else if( strcmp (value.key, "VOXEL_WIDTH") == 0 )
	{
		//VOXEL_WIDTH = value.double_input;
		parameters.VOXEL_WIDTH_D = value.double_input;
	}
	else if( strcmp (value.key, "VOXEL_HEIGHT") == 0 )
	{
		if( value.double_input < 0 )
		{
			puts("ERROR: Negative value give to parameter that should be positive.\n  Correct the configuration file and rerun program");
			exit_program_if(true);
		}
		//VOXEL_HEIGHT = value.double_input;
		parameters.VOXEL_HEIGHT_D = value.double_input;
	}
	else if( strcmp (value.key, "VOXEL_THICKNESS") == 0 )
	{
		if( value.double_input < 0 )
		{
			puts("ERROR: Negative value give to parameter that should be positive.\n  Correct the configuration file and rerun program");
			exit_program_if(true);
		}
		//VOXEL_THICKNESS = value.double_input;
		parameters.VOXEL_THICKNESS_D =  value.double_input;
		//SLICE_THICKNESS = value.double_input;
		//parameters.SLICE_THICKNESS_D =  value.double_input;
	}
	else if( strcmp (value.key, "SLICE_THICKNESS") == 0 )
	{
		if( value.double_input < 0 )
		{
			puts("ERROR: Negative value give to parameter that should be positive.\n  Correct the configuration file and rerun program");
			exit_program_if(true);
		}
		//SLICE_THICKNESS = value.double_input;
		parameters.SLICE_THICKNESS_D =  value.double_input;
	}
	
	else if( strcmp (value.key, "LAMBDA") == 0 )
	{
		if( value.double_input < 0 )
		{
			puts("ERROR: Negative value give to parameter that should be positive.\n  Correct the configuration file and rerun program");
			exit_program_if(true);
		}
		//LAMBDA = value.double_input;
		parameters.LAMBDA = value.double_input;
		parameters.LAMBDA_D = value.double_input;
	}
	else if( strcmp (value.key, "ETA") == 0 )
	{
		if( value.double_input < 0 )
		{
			puts("ERROR: Negative value give to parameter that should be positive.\n  Correct the configuration file and rerun program");
			exit_program_if(true);
		}
		//ETA = value.double_input;
		parameters.ETA_D = value.double_input;
	}
	else if( strcmp (value.key, "HULL_FILTER_THRESHOLD") == 0 )
	{
		if( value.double_input < 0 )
		{
			puts("ERROR: Negative value give to parameter that should be positive.\n  Correct the configuration file and rerun program");
			exit_program_if(true);
		}
		//HULL_FILTER_THRESHOLD = value.double_input;
		parameters.HULL_FILTER_THRESHOLD_D = value.double_input;
	}
	else if( strcmp (value.key, "FBP_AVG_THRESHOLD") == 0 )
	{
		//FBP_AVG_THRESHOLD = value.double_input;
		parameters.FBP_AVG_THRESHOLD_D = value.double_input;
	}
	else if( strcmp (value.key, "X_0_FILTER_THRESHOLD") == 0 )
	{
		if( value.double_input < 0 )
		{
			puts("ERROR: Negative value give to parameter that should be positive.\n  Correct the configuration file and rerun program");
			exit_program_if(true);
		}
		//X_0_FILTER_THRESHOLD = value.double_input;
		parameters.X_0_FILTER_THRESHOLD_D = value.double_input;
	}
	//------------------------------------------------------------------------------//
	//------------------------------------------------------------------------------//
	//------------------------------------------------------------------------------//
	else if( strcmp (value.key, "SC_THRESHOLD") == 0 )
	{
		//SC_THRESHOLD = value.double_input;
		parameters.SC_THRESHOLD_D = value.double_input;
	}
	else if( strcmp (value.key, "MSC_THRESHOLD") == 0 )
	{
		//MSC_THRESHOLD = value.double_input;
		parameters.MSC_THRESHOLD_D = value.double_input;
	}
	else if( strcmp (value.key, "SM_LOWER_THRESHOLD") == 0 )
	{
		//SM_LOWER_THRESHOLD = value.double_input;
		parameters.SM_LOWER_THRESHOLD_D = value.double_input;
	}
	else if( strcmp (value.key, "SM_UPPER_THRESHOLD") == 0 )
	{
		//SM_UPPER_THRESHOLD = value.double_input;
		parameters.SM_UPPER_THRESHOLD_D = value.double_input;
	}
	else if( strcmp (value.key, "SM_SCALE_THRESHOLD") == 0 )
	{
		if( value.double_input < 0 )
		{
			puts("ERROR: Negative value give to parameter that should be positive.\n  Correct the configuration file and rerun program");
			exit_program_if(true);
		}
		//SM_SCALE_THRESHOLD = value.double_input;
		parameters.SM_SCALE_THRESHOLD_D = value.double_input;
	}
	else
	{
		puts("ERROR: Procedure for setting this key is undefined");
		exit_program_if(true);
	}
}
void set_integer_parameter( generic_IO_container &value )
{
	if( value.input_type_ID == DOUBLE )
		printf("converted to an integer and ");
	if( strcmp (value.key, "DATA_TYPE") == 0 )
	{	
		if( value.integer_input < 0 )
		{
			puts("given a negative value for an unsigned integer variable.\n  Correct the configuration file and rerun program");
			exit_program_if(true);
		}
		exit_program_if(print_scan_type(value.integer_input));
		// EXPERIMENTAL = 0, GEANT4 = 1, TOPAS = 2
		parameters.DATA_TYPE = SCAN_TYPES(value.integer_input);
		//DATA_TYPE = SCAN_TYPES(value.integer_input);
	}
	else if( strcmp (value.key, "HULL_TYPE") == 0 )
	{
		if( value.integer_input < 0 )
		{
			puts("given a negative value for an unsigned integer variable.\n  Correct the configuration file and rerun program");
			exit_program_if(true);
		}
		exit_program_if(print_hull_type(value.integer_input));
		// IMPORT = 0, SC = 1, MSC = 2, SM = 3, FBP = 4
		parameters.HULL_TYPE = HULL_TYPES(value.integer_input);
		//HULL = HULL_TYPES(value.integer_input);
	}
	else if( strcmp (value.key, "FBP_FILTER_TYPE") == 0 )
	{
		if( value.integer_input < 0 )
		{
			puts("given a negative value for an unsigned integer variable.\n  Correct the configuration file and rerun program");
			exit_program_if(true);
		}
		exit_program_if(print_filter_type(value.integer_input));
		// RAM_LAK = 0, SHEPP_LOGAN = 1, NONE = 2
		parameters.FBP_FILTER_TYPE = FILTER_TYPES(value.integer_input);
		//FBP_FILTER = FILTER_TYPES(value.integer_input);
	}
	else if( strcmp (value.key, "X_0_TYPE") == 0 )
	{
		if( value.integer_input < 0 )
		{
			puts("given a negative value for an unsigned integer variable.\n  Correct the configuration file and rerun program");
			exit_program_if(true);
		}
		exit_program_if(print_x_0_type(value.integer_input));
		// IMPORT = 0, HULL = 1, FBP = 2, HYBRID = 3, ZEROS = 4
		parameters.X_0_TYPE = X_0_TYPES(value.integer_input);
		//X_0 = X_0_TYPES(value.integer_input);
	}
	else if( strcmp (value.key, "RECONSTRUCTION_METHOD") == 0 )
	{
		if( value.integer_input < 0 )
		{
			puts("given a negative value for an unsigned integer variable.\n  Correct the configuration file and rerun program");
			exit_program_if(true);
		}	
		exit_program_if(print_recon_algorithm(value.integer_input));
		// ART = 0, DROP = 1, BIP = 2, SAP = 3, ROBUST1 = 4, ROBUST2 = 5 
		parameters.RECONSTRUCTION_METHOD = RECON_ALGORITHMS(value.integer_input);
		//RECONSTRUCTION = RECON_ALGORITHMS(value.integer_input);
	}
	//------------------------------------------------------------------------------//
	//------------------------------------------------------------------------------//
	//------------------------------------------------------------------------------//
	else if( strcmp (value.key, "NUM_SCANS") == 0 )
	{
		if( value.integer_input < 0 )
		{
			puts("given a negative value for an unsigned integer variable.\n  Correct the configuration file and rerun program");
			exit_program_if(true);
		}
		printf("set to %d\n", value.integer_input);
		//NUM_SCANS = value.integer_input;
		parameters.NUM_SCANS_D = value.integer_input;
	}
	else if( strcmp (value.key, "MAX_GPU_HISTORIES") == 0 )
	{
		if( value.integer_input < 0 )
		{
			puts("given a negative value for an unsigned integer variable.\n  Correct the configuration file and rerun program");
			exit_program_if(true);
		}
		printf("set to %d\n", value.integer_input);
		//MAX_GPU_HISTORIES = value.integer_input;
		parameters.MAX_GPU_HISTORIES_D = value.integer_input;
	}
	else if( strcmp (value.key, "MAX_CUTS_HISTORIES") == 0 )
	{
		if( value.integer_input < 0 )
		{
			puts("given a negative value for an unsigned integer variable.\n  Correct the configuration file and rerun program");
			exit_program_if(true);
		}
		printf("set to %d\n", value.integer_input);
		//MAX_CUTS_HISTORIES = value.integer_input;
		parameters.MAX_CUTS_HISTORIES_D = value.integer_input;
	}
	else if( strcmp (value.key, "T_BINS") == 0 )
	{
		if( value.integer_input < 0 )
		{
			puts("given a negative value for an unsigned integer variable.\n  Correct the configuration file and rerun program");
			exit_program_if(true);
		}
		printf("set to %d\n", value.integer_input);
		//T_BINS = value.integer_input;
		parameters.T_BINS_D = value.integer_input;
	}
	else if( strcmp (value.key, "V_BINS") == 0 )
	{
		if( value.integer_input < 0 )
		{
			puts("given a negative value for an unsigned integer variable.\n  Correct the configuration file and rerun program");
			exit_program_if(true);
		}
		printf("set to %d\n", value.integer_input);
		//V_BINS = value.integer_input;
		parameters.V_BINS_D = value.integer_input;
	}
	else if( strcmp (value.key, "SIGMAS_2_KEEP") == 0 )
	{
		if( value.integer_input < 0 )
		{
			puts("given a negative value for an unsigned integer variable.\n  Correct the configuration file and rerun program");
			exit_program_if(true);
		}
		printf("set to %d\n", value.integer_input);
		//SIGMAS_2_KEEP = value.integer_input;
		parameters.SIGMAS_2_KEEP_D = value.integer_input;
	}
	else if( strcmp (value.key, "COLUMNS") == 0 )
	{
		if( value.integer_input < 0 )
		{
			puts("given a negative value for an unsigned integer variable.\n  Correct the configuration file and rerun program");
			exit_program_if(true);
		}
		printf("set to %d\n", value.integer_input);
		//COLUMNS = value.integer_input;
		parameters.COLUMNS_D = value.integer_input;
	}
	else if( strcmp (value.key, "ROWS") == 0 )
	{
		if( value.integer_input < 0 )
		{
			puts("given a negative value for an unsigned integer variable.\n  Correct the configuration file and rerun program");
			exit_program_if(true);
		}
		printf("set to %d\n", value.integer_input);
		//ROWS = value.integer_input;
		parameters.ROWS_D = value.integer_input;
	}
	else if( strcmp (value.key, "SLICES") == 0 )
	{
		if( value.integer_input < 0 )
		{
			puts("given a negative value for an unsigned integer variable.\n  Correct the configuration file and rerun program");
			exit_program_if(true);
		}
		printf("set to %d\n", value.integer_input);
		//SLICES = value.integer_input;
		parameters.SLICES_D = value.integer_input;
	}
	//------------------------------------------------------------------------------//
	//------------------------------------------------------------------------------//
	//------------------------------------------------------------------------------//
	else if( strcmp (value.key, "ITERATIONS") == 0 )
	{
		if( value.integer_input < 0 )
		{
			puts("given a negative value for an unsigned integer variable.\n  Correct the configuration file and rerun program");
			exit_program_if(true);
		}
		printf("set to %d\n", value.integer_input);
		//ITERATIONS = value.double_input;
		parameters.ITERATIONS_D = value.integer_input;
	}
	else if( strcmp (value.key, "BLOCK_SIZE") == 0 )
	{
		if( value.integer_input < 0 )
		{
			puts("given a negative value for an unsigned integer variable.\n  Correct the configuration file and rerun program");
			exit_program_if(true);
		}
		printf("set to %d\n", value.integer_input);
		//BLOCK_SIZE = value.double_input;
		parameters.BLOCK_SIZE_D =  value.integer_input;
	}
	else if( strcmp (value.key, "HULL_FILTER_RADIUS") == 0 )
	{
		printf("set to %d\n", value.integer_input);
		//HULL_FILTER_RADIUS = value.double_input;
		parameters.HULL_FILTER_RADIUS_D =  value.integer_input;
	}
	else if( strcmp (value.key, "X_0_FILTER_RADIUS") == 0 )
	{
		if( value.integer_input < 0 )
		{
			puts("given a negative value for an unsigned integer variable.\n  Correct the configuration file and rerun program");
			exit_program_if(true);
		}
		printf("set to %d\n", value.integer_input);
		//X_0_FILTER_RADIUS = value.double_input;
		parameters.X_0_FILTER_RADIUS_D =  value.integer_input;
	}
	else if( strcmp (value.key, "FBP_AVG_RADIUS") == 0 )
	{
		if( value.integer_input < 0 )
		{
			puts("given a negative value for an unsigned integer variable.\n  Correct the configuration file and rerun program");
			exit_program_if(true);
		}
		printf("set to %d\n", value.integer_input);
		//FBP_AVG_RADIUS = value.double_input;
		parameters.FBP_AVG_RADIUS_D = value.integer_input;
	}
	else if( strcmp (value.key, "FBP_MEDIAN_RADIUS") == 0 )
	{
		if( value.integer_input < 0 )
		{
			puts("given a negative value for an unsigned integer variable.\n  Correct the configuration file and rerun program");
			exit_program_if(true);
		}
		printf("set to %d\n", value.integer_input);
		//FBP_MEDIAN_RADIUS = value.double_input;
		parameters.FBP_MEDIAN_RADIUS_D = value.integer_input;
	}
	else if( strcmp (value.key, "PSI_SIGN") == 0 )
	{
		printf("set to %d\n", value.integer_input);
		//PSI_SIGN = value.double_input;
		parameters.PSI_SIGN_D = value.integer_input;
	}
	//------------------------------------------------------------------------------//
	//------------------------------------------------------------------------------//
	//------------------------------------------------------------------------------//
	else if( strcmp (value.key, "MSC_DIFF_THRESH") == 0 )
	{
		if( value.integer_input < 0 )
		{
			puts("given a negative value for an unsigned integer variable.\n  Correct the configuration file and rerun program");
			exit_program_if(true);
		}
		printf("set to %d\n", value.integer_input);
		//MSC_DIFF_THRESH = value.double_input;
		parameters.MSC_DIFF_THRESH_D = value.integer_input;
	}
	else
	{
		puts("ERROR: Procedure for setting this key is undefined");
		exit_program_if(true);
	}
}
void set_boolean_parameter( generic_IO_container &value )
{
	if( value.input_type_ID == INTEGER || value.input_type_ID == DOUBLE )
		printf("converted to a boolean and ");
	printf("set to %s\n", value.string_input );

	if( strcmp (value.key, "ADD_DATA_LOG_ENTRY") == 0 )
	{
		//ADD_DATA_LOG_ENTRY = value.boolean_input;
		parameters.ADD_DATA_LOG_ENTRY_D = value.boolean_input;
	}
	else if( strcmp (value.key, "CONSOLE_OUTPUT_2_DISK") == 0 )
	{
		//CONSOLE_OUTPUT_2_DISK = value.boolean_input;
		parameters.CONSOLE_OUTPUT_2_DISK_D = value.boolean_input;
	}
	else if( strcmp (value.key, "IMPORT_PREPROCESSING") == 0 )
	{
		//IMPORT_PREPROCESSING = value.boolean_input;
		parameters.IMPORT_PREPROCESSING_D = value.boolean_input;
	}
	else if( strcmp (value.key, "PERFORM_RECONSTRUCTION") == 0 )
	{
		//PERFORM_RECONSTRUCTION = value.boolean_input;
		parameters.PERFORM_RECONSTRUCTION_D = value.boolean_input;
	}
	else if( strcmp (value.key, "PREPROCESS_OVERWRITE_OK") == 0 )
	{
		//PREPROCESS_OVERWRITE_OK = value.boolean_input;
		parameters.PREPROCESS_OVERWRITE_OK_D = value.boolean_input;
	}
	else if( strcmp (value.key, "RECON_OVERWRITE_OK") == 0 )
	{
		//RECON_OVERWRITE_OK = value.boolean_input;
		parameters.RECON_OVERWRITE_OK_D = value.boolean_input;
	}
	else if( strcmp (value.key, "FBP_ON") == 0 )
	{
		//FBP_ON = value.boolean_input;
		parameters.FBP_ON_D = value.boolean_input;
	}
	else if( strcmp (value.key, "AVG_FILTER_FBP") == 0 )
	{
		//AVG_FILTER_FBP = value.boolean_input;
		parameters.AVG_FILTER_FBP_D = value.boolean_input;
	}
	else if( strcmp (value.key, "MEDIAN_FILTER_FBP") == 0 )
	{
		//MEDIAN_FILTER_FBP = value.boolean_input;
		parameters.MEDIAN_FILTER_FBP_D = value.boolean_input;
	}
	else if( strcmp (value.key, "IMPORT_FILTERED_FBP") == 0 )
	{
		//IMPORT_FILTERED_FBP = value.boolean_input;
		parameters.IMPORT_FILTERED_FBP_D = value.boolean_input;
	}
	else if( strcmp (value.key, "SC_ON") == 0 )
	{
		//SC_ON = value.boolean_input;
		parameters.SC_ON_D = value.boolean_input;
	}
	else if( strcmp (value.key, "MSC_ON") == 0 )
	{
		//MSC_ON = value.boolean_input;
		parameters.MSC_ON_D = value.boolean_input;
	}
	else if( strcmp (value.key, "SM_ON") == 0 )
	{
		//SM_ON = value.boolean_input;
		parameters.SM_ON_D = value.boolean_input;
	}
	else if( strcmp (value.key, "AVG_FILTER_HULL") == 0 )
	{
		//AVG_FILTER_HULL = value.boolean_input;
		parameters.AVG_FILTER_HULL_D = value.boolean_input;
	}
	else if( strcmp (value.key, "AVG_FILTER_ITERATE") == 0 )
	{
		//AVG_FILTER_ITERATE = value.boolean_input;
		parameters.AVG_FILTER_ITERATE_D = value.boolean_input;
	}
	//------------------------------------------------------------------------------//
	//------------------------------------------------------------------------------//
	//------------------------------------------------------------------------------//
	else if( strcmp (value.key, "WRITE_MSC_COUNTS") == 0 )
	{
		//WRITE_MSC_COUNTS = value.boolean_input;
		parameters.WRITE_MSC_COUNTS_D = value.boolean_input;
	}
	else if( strcmp (value.key, "WRITE_SM_COUNTS") == 0 )
	{
		//WRITE_SM_COUNTS = value.boolean_input;
		parameters.WRITE_SM_COUNTS_D = value.boolean_input;
	}
	else if( strcmp (value.key, "WRITE_X_FBP") == 0 )
	{
		//WRITE_X_FBP = value.boolean_input;
		parameters.WRITE_X_FBP_D = value.boolean_input;
	}
	else if( strcmp (value.key, "WRITE_FBP_HULL") == 0 )
	{
		//WRITE_FBP_HULL = value.boolean_input;
		parameters.WRITE_FBP_HULL_D = value.boolean_input;
	}
	else if( strcmp (value.key, "WRITE_AVG_FBP") == 0 )
	{
		//WRITE_AVG_FBP = value.boolean_input;
		parameters.WRITE_AVG_FBP_D = value.boolean_input;
	}
	else if( strcmp (value.key, "WRITE_MEDIAN_FBP") == 0 )
	{
		//WRITE_MEDIAN_FBP = value.boolean_input;
		parameters.WRITE_MEDIAN_FBP_D = value.boolean_input;
	}
	else if( strcmp (value.key, "WRITE_BIN_WEPLS") == 0 )
	{
		//WRITE_BIN_WEPLS = value.boolean_input;
		parameters.WRITE_BIN_WEPLS_D = value.boolean_input;
	}
	else if( strcmp (value.key, "WRITE_WEPL_DISTS") == 0 )
	{
		//WRITE_WEPL_DISTS = value.boolean_input;
		parameters.WRITE_WEPL_DISTS_D = value.boolean_input;
	}
	else if( strcmp (value.key, "WRITE_SSD_ANGLES") == 0 )
	{
		//WRITE_SSD_ANGLES = value.boolean_input;
		parameters.WRITE_SSD_ANGLES_D = value.boolean_input;
	}
	else
	{
		puts("ERROR: Procedure for setting this key is undefined");
		exit_program_if(true);
	}
}
void set_parameter( generic_IO_container &value )
{
	printf("----> %s was ", value.key);
	if( key_is_string_parameter(value.key) )
		set_string_parameter(value);
	else if( key_is_floating_point_parameter(value.key) )
		set_floating_point_parameter(value);
	else if( key_is_integer_parameter(value.key) )
		set_integer_parameter(value);
	else if( key_is_boolean_parameter(value.key) )
		set_boolean_parameter(value);
	else
		puts("\nNo match for this key");
}
void set_execution_date()
{
	current_MMDDYYYY( EXECUTION_DATE);

	char* preprocess_date = EXECUTION_DATE;
	PREPROCESS_DATE = (char*) calloc( strlen(preprocess_date) + 1, sizeof(char) ); 
	std::copy( preprocess_date, preprocess_date + strlen(preprocess_date), PREPROCESS_DATE );	

	char* reconstruction_date = EXECUTION_DATE;
	RECONSTRUCTION_DATE = (char*) calloc( strlen(reconstruction_date) + 1, sizeof(char) ); 
	std::copy( reconstruction_date, reconstruction_date + strlen(reconstruction_date), RECONSTRUCTION_DATE );
}
void set_IO_paths()
{
	char mkdir_command[256];
	print_section_header( "Setting I/O path parameters and creating directories", '*' );

	//----------------------------------------------------------------------------------------------------------------------------------------------------------//
	//------------------ Set the data type directory to "Experimental" or "Simulated" depending on value of DATA_TYPE specified in config file -----------------//
	//----------------------------------------------------------------------------------------------------------------------------------------------------------//	
	switch( parameters.DATA_TYPE )
	{
		case EXPERIMENTAL	: 	DATA_TYPE_DIR = EXPERIMENTAL_DIR_NAME;	break;
		case SIMULATED_G	: 	DATA_TYPE_DIR = SIMULATIONS_DIR_NAME;	break;
		case SIMULATED_T	: 	DATA_TYPE_DIR = SIMULATIONS_DIR_NAME;	break;
		default				:	puts("ERROR: Invalid DATA_TYPE selected.  Must be EXPERIMENTAL, SIMULATED_G, or SIMULATED_T.");
								exit(1);

	};
	//----------------------------------------------------------------------------------------------------------------------------------------------------------//
	//-------------------- Determine if the individual key/value pairs associated with data directories can be combined to set their paths ---------------------//
	//----------------------------------------------------------------------------------------------------------------------------------------------------------//
	PROJECTION_DATA_DIR_CONSTRUCTABLE = PATH_2_PCT_DATA_DIR_SET && OBJECT_SET && RUN_DATE_SET  && RUN_NUMBER_SET  && PROJECTION_DATA_DATE_SET;
	PREPROCESSING_DIR_CONSTRUCTABLE = ( PROJECTION_DATA_DIR_CONSTRUCTABLE || PROJECTION_DATA_DIR_SET ) && PREPROCESS_DATE_SET;
	RECONSTRUCTION_DIR_CONSTRUCTABLE = ( PREPROCESSING_DIR_CONSTRUCTABLE || PREPROCESSING_DIR_SET ) && RECONSTRUCTION_DATE_SET;
	//----------------------------------------------------------------------------------------------------------------------------------------------------------//
	//---------------- Set projection data directory based on option (1) explicit path in config file, (2) individual object/run properties --------------------//
	//--------------------- specified in config file, or (3) automatically based on current execution directory or command line argument -----------------------//
	//----------------------------------------------------------------------------------------------------------------------------------------------------------//
	if( PROJECTION_DATA_DIR_SET )
	{
		char* h1 = strstr( PROJECTION_DATA_DIR, PCT_DATA_DIR_NAME );
		char* h2 = strstr( h1, "/" )     + 1;
		char* h3 = strstr( h2 + 1, "/" ) + 1;	
		char* h4 = strstr( h3 + 1, "/" ) + 1;
		char* h5 = strstr( h4 + 1, "/" ) + 1;
		char* h6 = strstr( h5 + 1, "/" ) + 1;
		char* h7 = strstr( h6 + 1, "/" ) + 1;

		PATH_2_PCT_DATA_DIR		= (char*) calloc( (int)(h1 - PROJECTION_DATA_DIR),	sizeof(char) ); 
		OBJECT					= (char*) calloc( (int)(h3 - h2 + 1),				sizeof(char) );
		SCAN_TYPE				= (char*) calloc( (int)(h4 - h3 + 1),				sizeof(char) ); 
		RUN_DATE				= (char*) calloc( (int)(h5 - h4 + 1),				sizeof(char) ); 
		RUN_NUMBER				= (char*) calloc( (int)(h6 - h5 + 1),				sizeof(char) ); 
		PROJECTION_DATA_DATE	= (char*) calloc( (int)(strlen(h7) + 1),			sizeof(char) ); 
		
		std::copy( PROJECTION_DATA_DIR, h1 - 1, PATH_2_PCT_DATA_DIR );		 
		std::copy( h2, h3 - 1, OBJECT );
		std::copy( h3, h4 - 1, SCAN_TYPE );
		std::copy( h4, h5 - 1, RUN_DATE );
		std::copy( h5, h6 - 1, RUN_NUMBER );
		std::copy( h7, h7 + strlen(h7), PROJECTION_DATA_DATE );
	}
	else if( PROJECTION_DATA_DIR_CONSTRUCTABLE )
	{		
		uint length = strlen(PATH_2_PCT_DATA_DIR) + strlen(PCT_DATA_DIR_NAME) + strlen(OBJECT) + strlen(SCAN_TYPE) + strlen(RUN_DATE) + strlen(RUN_NUMBER) + strlen(PROJECTION_DATA_DIR_NAME) + strlen(PROJECTION_DATA_DATE);
		PROJECTION_DATA_DIR = (char*) calloc( length + 1, sizeof(char) ); 
		sprintf(PROJECTION_DATA_DIR,"%s/%s/%s/%s/%s/%s/%s/%s", PATH_2_PCT_DATA_DIR, PCT_DATA_DIR_NAME, OBJECT, SCAN_TYPE, RUN_DATE, RUN_NUMBER, PROJECTION_DATA_DIR_NAME, PROJECTION_DATA_DATE );		
	}
	else
		puts("Projection data directory was not (properly) specified in settings.cfg and is being set based on current execution directory and date to\n");		
	print_section_separator('~');
	printf("PROJECTION_DATA_DIR = %s\n", PROJECTION_DATA_DIR );
	print_section_separator('~');
	
	//----------------------------------------------------------------------------------------------------------------------------------------------------------//
	//---- Set path parameter for preprocessing data using explicit name from config file or based on projection data directory and current execution date. ----// 
	//---------- Create this directory if it doesn't exist, otherwise overwrite any existing data or create new directory with _i appended to its name  --------//
	//----------------------------------------------------------------------------------------------------------------------------------------------------------//
	if( !PREPROCESSING_DIR_SET )
	{
		PREPROCESSING_DIR = (char*) calloc( strlen(PROJECTION_DATA_DIR) + strlen(RECONSTRUCTION_DIR_NAME) + strlen(PREPROCESS_DATE) + 1, sizeof(char) ); 
		sprintf(PREPROCESSING_DIR,"%s/%s/%s", PROJECTION_DATA_DIR, RECONSTRUCTION_DIR_NAME, PREPROCESS_DATE);		
	}		
	if( parameters.PREPROCESS_OVERWRITE_OK_D )
	{
		sprintf(mkdir_command, "mkdir \"%s\"", PREPROCESSING_DIR );
		if( system( mkdir_command ) )
			puts("\nNOTE: Any existing data in this directory will be overwritten");
		//std::string text = buffer.str();
		////std::cout << text << endl;
		//printf( "Hello %s\n", text );
	}
	else
		create_unique_dir( PREPROCESSING_DIR );
	print_section_separator('~');
	printf("PREPROCESSING_DIR = %s\n", PREPROCESSING_DIR );
	print_section_separator('~');
	//----------------------------------------------------------------------------------------------------------------------------------------------------------//
	//---- Set path parameter for reconstruction data using explicit name from config file or based on projection data directory and current execution date. ---// 
	//---------- Create this directory if it doesn't exist, otherwise overwrite any existing data or create new directory with _i appended to its name  --------//
	//----------------------------------------------------------------------------------------------------------------------------------------------------------//
	if( !RECONSTRUCTION_DIR_SET )
	{
		RECONSTRUCTION_DIR = (char*) calloc( strlen(PREPROCESSING_DIR) + strlen(PCT_IMAGES_DIR_NAME) + strlen(RECONSTRUCTION_DATE) + 1, sizeof(char) ); 
		sprintf(RECONSTRUCTION_DIR,"%s/%s/%s", PREPROCESSING_DIR, PCT_IMAGES_DIR_NAME, RECONSTRUCTION_DATE);		
	}
	if( parameters.RECON_OVERWRITE_OK_D )
	{
		sprintf(mkdir_command, "mkdir \"%s\"", RECONSTRUCTION_DIR );
		if( system( mkdir_command ) )
			puts("\nNOTE: Any existing data in this directory will be overwritten");
		//std::string text = buffer.str();
		//printf( "Hello %s\n", text );
	}
	else
		create_unique_dir( RECONSTRUCTION_DIR );
	puts("");
	print_section_separator('~');
	printf("RECONSTRUCTION_DIR = %s\n", RECONSTRUCTION_DIR );
	print_section_separator('~');

	print_section_exit("Finished setting paths to I/O data directories and creating associated folders", "====>" );
	//----------------------------------------------------------------------------------------------------------------------------------------------------------//
	//----------------------------------------------- Set file names for preprocessing data generated as output ------------------------------------------------//
	//----------------------------------------------------------------------------------------------------------------------------------------------------------//
	print_section_header( "Preprocessing and reconstruction data/images generated will be written to and/or read from the following paths", '*' );
	
	HULL_FILENAME = (char*) calloc( strlen(HULL_BASENAME) + strlen(HULL_FILE_EXTENSION) + 1, sizeof(char) );
	FBP_FILENAME = (char*) calloc( strlen(FBP_BASENAME) + strlen(FBP_FILE_EXTENSION) + 1, sizeof(char) );
	FBP_MEDIAN_2D_FILENAME = (char*) calloc( strlen(FBP_MEDIAN_2D_BASENAME) + 3 + strlen(FBP_MEDIANS_FILE_EXTENSION) + 1, sizeof(char) );
	FBP_MEDIAN_3D_FILENAME = (char*) calloc( strlen(FBP_MEDIAN_3D_BASENAME) + 3 + strlen(FBP_MEDIANS_FILE_EXTENSION) + 1, sizeof(char) );
	X_0_FILENAME = (char*) calloc( strlen(X_0_BASENAME) + strlen(X_0_FILE_EXTENSION) + 1, sizeof(char) );
	MLP_FILENAME = (char*) calloc( strlen(MLP_BASENAME) + strlen(MLP_FILE_EXTENSION) + 1, sizeof(char) );
	RECON_HISTORIES_FILENAME = (char*) calloc( strlen(RECON_HISTORIES_BASENAME) + strlen(HISTORIES_FILE_EXTENSION) + 1, sizeof(char) );
	X_FILENAME = (char*) calloc( strlen(X_BASENAME) + strlen(X_FILE_EXTENSION) + 1, sizeof(char) );

	sprintf( HULL_FILENAME,"%s%s", HULL_BASENAME, HULL_FILE_EXTENSION );
	sprintf( FBP_FILENAME,"%s%s", FBP_BASENAME, FBP_FILE_EXTENSION );
	sprintf( FBP_MEDIAN_2D_FILENAME,"%s_2D%d%s", FBP_MEDIAN_2D_BASENAME, 2 * parameters.FBP_MEDIAN_RADIUS_D + 1, FBP_MEDIANS_FILE_EXTENSION);
	sprintf( FBP_MEDIAN_3D_FILENAME,"%s_3D%d%s", FBP_MEDIAN_3D_BASENAME, 2 * parameters.FBP_MEDIAN_RADIUS_D + 1, FBP_MEDIANS_FILE_EXTENSION);
	sprintf( X_0_FILENAME, "%s%s", X_0_BASENAME, X_0_FILE_EXTENSION );
	sprintf( MLP_FILENAME,"%s%s", MLP_BASENAME, MLP_FILE_EXTENSION );
	sprintf( RECON_HISTORIES_FILENAME,"%s%s", RECON_HISTORIES_BASENAME, HISTORIES_FILE_EXTENSION);
	sprintf( X_FILENAME,"%s%s", X_BASENAME, X_FILE_EXTENSION);

	printf("HULL_FILENAME = %s\n\n", HULL_FILENAME );	
	printf("FBP_FILENAME = %s\n\n", FBP_FILENAME );
	printf("FBP_MEDIAN_2D_FILENAME = %s\n\n", FBP_MEDIAN_2D_FILENAME );
	printf("FBP_MEDIAN_3D_FILENAME = %s\n\n", FBP_MEDIAN_3D_FILENAME );
	printf("X_0_FILENAME = %s\n\n", X_0_FILENAME );	
	printf("MLP_FILENAME = %s\n\n", MLP_FILENAME );	
	printf("RECON_HISTORIES_FILENAME = %s\n\n", RECON_HISTORIES_FILENAME );	
	printf("X_FILENAME = %s\n", X_FILENAME );
	//----------------------------------------------------------------------------------------------------------------------------------------------------------//
	//----------------------------- Set paths to preprocessing and reconstruction data using associated directory and file names -------------------------------//
	//----------------------------------------------------------------------------------------------------------------------------------------------------------//
	
	print_section_header( "File names of preprocessing data generated as output", '*' );	

	HULL_PATH = (char*) calloc( strlen(PREPROCESSING_DIR) + strlen(HULL_FILENAME) + 1, sizeof(char) );
	FBP_PATH = (char*) calloc( strlen(PREPROCESSING_DIR) + strlen(FBP_FILENAME) + 1, sizeof(char) );
	FBP_MEDIAN_2D_PATH = (char*) calloc( strlen(PREPROCESSING_DIR) + strlen(FBP_MEDIAN_2D_FILENAME) + 1, sizeof(char) );
	FBP_MEDIAN_3D_PATH = (char*) calloc( strlen(PREPROCESSING_DIR) + strlen(FBP_MEDIAN_3D_FILENAME) + 1, sizeof(char) );
	X_0_PATH = (char*) calloc( strlen(PREPROCESSING_DIR) + strlen(X_0_FILENAME) + 1, sizeof(char) );
	MLP_PATH = (char*) calloc( strlen(PREPROCESSING_DIR) + strlen(MLP_FILENAME) + 1, sizeof(char) );
	RECON_HISTORIES_PATH = (char*) calloc( strlen(PREPROCESSING_DIR) + strlen(RECON_HISTORIES_FILENAME) + 1, sizeof(char) );
	X_PATH = (char*) calloc( strlen(RECONSTRUCTION_DIR) + strlen(X_FILENAME) + 1, sizeof(char) );
	 
	sprintf( HULL_PATH,"%s/%s", PREPROCESSING_DIR, HULL_FILENAME );
	sprintf( FBP_PATH,"%s/%s", PREPROCESSING_DIR, FBP_FILENAME );
	sprintf(FBP_MEDIAN_2D_PATH,"%s/%s", PREPROCESSING_DIR, FBP_MEDIAN_2D_FILENAME);
	sprintf(FBP_MEDIAN_3D_PATH,"%s/%s", PREPROCESSING_DIR, FBP_MEDIAN_3D_FILENAME);
	sprintf( X_0_PATH, "%s/%s", PREPROCESSING_DIR, X_0_FILENAME );
	sprintf(MLP_PATH,"%s/%s", PREPROCESSING_DIR, MLP_FILENAME );
	sprintf(RECON_HISTORIES_PATH,"%s/%s", PREPROCESSING_DIR, RECON_HISTORIES_FILENAME );
	sprintf(X_PATH,"%s/%s", RECONSTRUCTION_DIR, X_FILENAME);
	
	printf("HULL_PATH = %s\n\n", HULL_PATH );	
	printf("FBP_PATH = %s\n\n", FBP_PATH );
	printf("FBP_MEDIAN_2D_PATH = %s\n\n", FBP_MEDIAN_2D_PATH );
	printf("FBP_MEDIAN_3D_PATH = %s\n\n", FBP_MEDIAN_3D_PATH );
	printf("X_0_PATH = %s\n\n", X_0_PATH );	
	printf("MLP_PATH = %s\n\n", MLP_PATH );	
	printf("RECON_HISTORIES_PATH = %s\n\n", RECON_HISTORIES_PATH );	
	printf("X_PATH = %s\n", X_PATH );

	//----------------------------------------------------------------------------------------------------------------------------------------------------------//
	//------------------------------------- Extract information about scan data used for preprocessig and reconstruction ---------------------------------------//
	//----------------------------------------------------------------------------------------------------------------------------------------------------------//
	print_section_header( "Directory names extracted from projection data path which characterize the data used as input to preprocessing and reconstruction", '*' );	
	
	printf("PATH_2_PCT_DATA_DIR = %s\n", PATH_2_PCT_DATA_DIR);
	printf("OBJECT = %s\n", OBJECT);
	printf("SCAN_TYPE =  %s\n", SCAN_TYPE);
	printf("RUN_DATE = %s\n", RUN_DATE);
	printf("RUN_NUMBER = %s\n", RUN_NUMBER);
	printf("PROJECTION_DATA_DATE = %s\n", PROJECTION_DATA_DATE);

	print_section_exit("Finished setting file names of input/output data files and the paths to where they are to be written", "====>" );
}
void view_config_file()
{
	char filename[256]; 

	#if defined(_WIN32) || defined(_WIN64)
		sprintf(filename, "%s %s %s", "start", "wordpad", CONFIG_FILENAME);
		terminal_response(filename);
    #else
		sprintf(filename, "%s %s", "touch", CONFIG_FILENAME);
		terminal_response(filename);
    #endif
	
}
void set_dependent_parameters()
{
	parameters.GANTRY_ANGLES_D		= uint( 360 / parameters.GANTRY_ANGLE_INTERVAL_D );								// [#] Total number of projection angles
	parameters.NUM_FILES_D			= parameters.NUM_SCANS_D * parameters.GANTRY_ANGLES_D;							// [#] 1 file per gantry angle per translation
	parameters.T_BINS_D				= uint( parameters.SSD_T_SIZE_D / parameters.T_BIN_SIZE_D + 0.5 );				// [#] Number of bins (i.e. quantization levels) for t (lateral) direction 
	parameters.V_BINS_D				= uint( parameters.SSD_V_SIZE_D/ parameters.V_BIN_SIZE_D + 0.5 );				// [#] Number of bins (i.e. quantization levels) for v (vertical) direction 
	parameters.ANGULAR_BINS_D		= uint( 360 / parameters.ANGULAR_BIN_SIZE_D + 0.5 );							// [#] Number of bins (i.e. quantization levels) for path angle 
	parameters.NUM_BINS_D			= parameters.ANGULAR_BINS_D * parameters.T_BINS_D * parameters.V_BINS_D;		// [#] Total number of bins corresponding to possible 3-tuples [ANGULAR_BIN, T_BIN, V_BIN]
	parameters.RECON_CYL_HEIGHT_D	= parameters.SSD_V_SIZE_D - 1.0;												// [cm] Height of reconstruction cylinder
	parameters.RECON_CYL_DIAMETER_D	= 2 * parameters.RECON_CYL_RADIUS_D;											// [cm] Diameter of reconstruction cylinder
	parameters.SLICES_D				= uint( parameters.RECON_CYL_HEIGHT_D / parameters.SLICE_THICKNESS_D);			// [#] Number of voxels in the z direction (i.e., number of slices) of image
	parameters.NUM_VOXELS_D			= parameters.COLUMNS_D * parameters.ROWS_D * parameters.SLICES_D;				// [#] Total number of voxels (i.e. 3-tuples [column, row, slice]) in image
	parameters.IMAGE_WIDTH_D		= parameters.RECON_CYL_DIAMETER_D;												// [cm] Distance between left and right edges of each slice in image
	parameters.IMAGE_HEIGHT_D		= parameters.RECON_CYL_DIAMETER_D;						// [cm] Distance between top and bottom edges of each slice in image
	parameters.IMAGE_THICKNESS_D	= parameters.RECON_CYL_HEIGHT_D;						// [cm] Distance between bottom of bottom slice and top of the top slice of image
	parameters.VOXEL_WIDTH_D		= parameters.IMAGE_WIDTH_D / parameters.COLUMNS_D;		// [cm] Distance between left and right edges of each voxel in image
	parameters.VOXEL_HEIGHT_D		= parameters.IMAGE_HEIGHT_D / parameters.ROWS_D;		// [cm] Distance between top and bottom edges of each voxel in image
	parameters.VOXEL_THICKNESS_D	= parameters.IMAGE_THICKNESS_D / parameters.SLICES_D;	// [cm] Distance between top and bottom of each slice in image
	parameters.X_ZERO_COORDINATE_D	= -parameters.RECON_CYL_RADIUS_D;						// [cm] x-coordinate corresponding to left edge of 1st voxel (i.e. column) in image space
	parameters.Y_ZERO_COORDINATE_D	= parameters.RECON_CYL_RADIUS_D;						// [cm] y-coordinate corresponding to top edge of 1st voxel (i.e. row) in image space
	parameters.Z_ZERO_COORDINATE_D	= parameters.RECON_CYL_HEIGHT_D/2;						// [cm] z-coordinate corresponding to top edge of 1st voxel (i.e. slice) in image space
	parameters.RAM_LAK_TAU_D		= 2/ROOT_TWO * parameters.T_BIN_SIZE_D;					// Defines tau in Ram-Lak filter calculation, estimated from largest frequency in slice 
	/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
	/*---------------------------------------------------------- Memory allocation size for arrays (binning, image) -----------------------------------------------------------*/
	/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
	parameters.SIZE_BINS_CHAR_D		= ( parameters.NUM_BINS_D   * sizeof(char)	);			// Amount of memory required for a character array used for binning
	parameters.SIZE_BINS_BOOL_D		= ( parameters.NUM_BINS_D  * sizeof(bool)	);			// Amount of memory required for a boolean array used for binning
	parameters.SIZE_BINS_INT_D		= ( parameters.NUM_BINS_D   * sizeof(int)	);			// Amount of memory required for a integer array used for binning
	parameters.SIZE_BINS_UINT_D		= ( parameters.NUM_BINS_D   * sizeof(uint)	);			// Amount of memory required for a integer array used for binning
	parameters.SIZE_BINS_FLOAT_D	= ( parameters.NUM_BINS_D	 * sizeof(float));			// Amount of memory required for a floating point array used for binning
	parameters.SIZE_IMAGE_CHAR_D	= ( parameters.NUM_VOXELS_D * sizeof(char)	);			// Amount of memory required for a character array used for binning
	parameters.SIZE_IMAGE_BOOL_D	= ( parameters.NUM_VOXELS_D * sizeof(bool)	);			// Amount of memory required for a boolean array used for binning
	parameters.SIZE_IMAGE_INT_D		= ( parameters.NUM_VOXELS_D * sizeof(int)	);			// Amount of memory required for a integer array used for binning
	parameters.SIZE_IMAGE_UINT_D	= ( parameters.NUM_VOXELS_D * sizeof(uint)	);			// Amount of memory required for a integer array used for binning
	parameters.SIZE_IMAGE_FLOAT_D	= ( parameters.NUM_VOXELS_D * sizeof(float) );			// Amount of memory required for a floating point array used for binning
	parameters.SIZE_IMAGE_DOUBLE_D	= ( parameters.NUM_VOXELS_D * sizeof(double));			// Amount of memory required for a floating point array used for binning
	/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
	/*-------------------------------------------------------------- Iterative Image Reconstruction Parameters ----------------------------------------------------------------*/
	/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
	parameters.VOXEL_STEP_SIZE_D		= ( parameters.VOXEL_WIDTH_D / 2 );						// [cm] Length of the step taken along the path, i.e. change in depth per step for
	parameters.MLP_U_STEP_D				= ( parameters.VOXEL_WIDTH_D / 2);						// Size of the step taken along u direction during MLP; depth difference between successive MLP points
	parameters.CONSTANT_CHORD_NORM_D	= pow(parameters.VOXEL_WIDTH_D, 2.0);
	parameters.CONSTANT_LAMBDA_SCALE_D	= parameters.VOXEL_WIDTH_D * parameters.LAMBDA_D;
}
void parameters_2_GPU()
{
	
	cudaMalloc((void**) &parameters_d, sizeof(parameters) );
	cudaMemcpy( parameters_d, parameters_h,	sizeof(parameters),	cudaMemcpyHostToDevice );
	//cudaMalloc((void**) &parameters_GPU, sizeof(parameters) );
	//cudaMemcpy( parameters_GPU, parameters,	sizeof(parameters),	cudaMemcpyHostToDevice );
}
/***********************************************************************************************************************************************************************************************************************/
/************************************************************************************************ END OF SOURCE CODE ***************************************************************************************************/
/***********************************************************************************************************************************************************************************************************************/