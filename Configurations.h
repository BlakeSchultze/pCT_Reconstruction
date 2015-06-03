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
	//char* PROJECTION_DATA_DIR, * PREPROCESSING_DIR, * RECONSTRUCTION_DIR;
	//char* OBJECT, * RUN_DATE, * RUN_NUMBER, * PROJECTION_DATA_DATE, * PREPROCESS_DATE, * RECONSTRUCTION_DATE;
	double RECON_CYL_RADIUS, RECON_CYLIAMETER, RECON_CYL_HEIGHT, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_THICKNESS, VOXEL_WIDTH, VOXEL_HEIGHT, VOXEL_THICKNESS, SLICE_THICKNESS;
	double X_ZERO_COORDINATE, Y_ZERO_COORDINATE, Z_ZERO_COORDINATE, RAM_LAK_TAU;
	double GANTRY_ANGLE_INTERVAL, ANGULAR_BIN_SIZE, SSD_T_SIZE, SSD_V_SIZE, T_SHIFT, U_SHIFT, V_SHIFT, T_BIN_SIZE, V_BIN_SIZE;
	double LAMBDA, ETA;
	double HULL_RSP_THRESHOLD, SC_THRESHOLD, MSC_THRESHOLD, SM_LOWER_THRESHOLD, SM_UPPER_THRESHOLD, SM_SCALE_THRESHOLD;
	double VOXEL_STEP_SIZE, MLP_U_STEP, CONSTANT_CHORD_NORM, CONSTANT_LAMBDA_SCALE;

	uint NUM_SCANS, MAX_GPU_HISTORIES, MAX_CUTS_HISTORIES, T_BINS, V_BINS, COLUMNS, ROWS, SLICES, SIGMAS_2_KEEP;
	uint GANTRY_ANGLES, NUM_FILES, ANGULAR_BINS, NUM_BINS, NUM_VOXELS;
	//uint SIZE_BINS_CHAR, SIZE_BINS_BOOL, SIZE_BINS_INT, SIZE_BINS_UINT, SIZE_BINS_FLOAT, SIZE_IMAGE_CHAR;
	//uint SIZE_IMAGE_BOOL, SIZE_IMAGE_INT, SIZE_IMAGE_UINT, SIZE_IMAGE_FLOAT, SIZE_IMAGE_DOUBLE;
	uint ITERATIONS, BLOCK_SIZE;
	uint HULL_MED_FILTER_RADIUS, FBP_MED_FILTER_RADIUS, X_0_MED_FILTER_RADIUS, X_K_MED_FILTER_RADIUS, X_MED_FILTER_RADIUS;
	uint HULL_AVG_FILTER_RADIUS, FBP_AVG_FILTER_RADIUS, X_0_AVG_FILTER_RADIUS, X_K_AVG_FILTER_RADIUS, X_AVG_FILTER_RADIUS;
	uint MSC_DIFF_THRESH;	

	int PSI_SIGN;
	SCAN_TYPES DATA_TYPE;									// Specify the type of input data: EXPERIMENTAL, SIMULATED_G, SIMULATED_T
	HULL_TYPES HULL_TYPE;									// Specify which of the HULL_TYPES to use in this run's MLP calculations
	FILTER_TYPES FBP_FILTER_TYPE;		  					// Specifies which of the defined filters will be used in FBP
	X_0_TYPES	X_0_TYPE;									// Specify which of the HULL_TYPES to use in this run's MLP calculations
	RECON_ALGORITHMS RECONSTRUCTION_METHOD; 				// Specify which of the projection algorithms to use for image reconstruction
	
	bool FBP_ON, SC_ON, MSC_ON, SM_ON;
	bool IMPORT_PREPROCESSING, PERFORM_RECONSTRUCTION, PREPROCESS_OVERWRITE_OK, RECON_OVERWRITE_OK, MLP_IN_LOOP, IMPORT_DATA_ITERATIVELY;
	bool MEDIAN_FILTER_HULL, MEDIAN_FILTER_FBP, MEDIAN_FILTER_X_0, MEDIAN_FILTER_X_K, MEDIAN_FILTER_X;
	bool AVG_FILTER_HULL, AVG_FILTER_FBP, AVG_FILTER_X_0, AVG_FILTER_X_K, AVG_FILTER_X;
	bool WRITE_MSC_COUNTS, WRITE_SM_COUNTS, WRITE_X_FBP, WRITE_FBP_HULL, WRITE_AVG_FBP, WRITE_MEDIAN_FBP, WRITE_BIN_WEPLS, WRITE_WEPL_DISTS, WRITE_SSD_ANGLES;	
	bool ADD_DATA_LOG_ENTRY, STDOUT_2_DISK, USER_INPUT_REQUESTS_OFF, DEBUG_TEXT_ON;
	bool EXIT_AFTER_BINNING, EXIT_AFTER_HULLS, EXIT_AFTER_CUTS, EXIT_AFTER_SINOGRAM, EXIT_AFTER_FBP;
	//*************************************************************************************************************************************************************************//
	//*********************************************************************** Output option parameters ************************************************************************//
	//*************************************************************************************************************************************************************************//
	configurations
	(
		uint num_scans_p 				= 1,								// [#] Total number of scans of same object
		uint max_gpu_histories_p		= 1500000,							// [#] Number of histories to process on the GPU at a time, based on GPU capacity
		uint max_cuts_histories_p 		= 1500000,	
		uint columns_p 					= 200,
		uint rows_p 					= 200,
		uint slices_p 					= 32,
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
		SCAN_TYPES data_type_p						= SIMULATED_G,		// Specifies the source of the input data (EXPERIMENTAL = 0, GEANT4 = 1, TOPAS = 2)
		HULL_TYPES hull_type_p						= MSC_HULL,			// Specify which hull detection method to use for MLP calculations (IMPORT = 0, SC = 1, MSC = 2, SM = 3, FBP = 4)
		FILTER_TYPES fbp_filter_type_p				= SHEPP_LOGAN,		// Specifies which of the defined filters to use in FBP (RAM_LAK = 0, SHEPP_LOGAN = 1, NONE = 2)
		X_0_TYPES x_0_type_p						= HYBRID,			// Specify which initial iterate to use for reconstruction (IMPORT = 0, HULL = 1, FBP = 2, HYBRID = 3, ZEROS = 4)
		RECON_ALGORITHMS reconstruction_method_p	= DROP,				// Specify algorithm to use for image reconstruction (ART = 0, DROP = 1, BIP = 2, SAP = 3, ROBUST1 = 4, ROBUST2 = 5)
		//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------//
		//---------------------------------------------------------- Reconstruction and image filtering parameters/options --------------------------------------------------------//
		//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------//
		uint iterations_p				= 12,							// [#] of iterations through the entire set of histories to perform in iterative image reconstruction
		uint block_size_p				= 60,							// [#] of paths to use for each update: ART = 1, 
		int psi_sign_p					= 1,
		double lambda_p 				= 0.0001,
		double eta_p                    = 0.0001,
		//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------//
		//----------------------------------------------------------------------- Hull-Detection Parameters -----------------------------------------------------------------------//
		//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------//
		double hull_rsp_threshold_p		= 0.1,							// [#] Maximum RSP for voxels assumed to belong to hull
		uint msc_diff_thresh_p			= 50,							// [#] Threshold on difference in counts between adjacent voxels used by MSC for edge detection
		double sc_threshold_p			= 0.0,							// [cm] If WEPL < SC_THRESHOLD, SC assumes the proton missed the object
		double msc_threshold_p			= 0.0,							// [cm] If WEPL < MSC_THRESHOLD, MSC assumes the proton missed the object
		double sm_lower_threshold_p		= 6.0,							// [cm] If WEPL >= SM_THRESHOLD, SM assumes the proton passed through the object
		double sm_upper_threshold_p		= 21.0,							// [cm] If WEPL > SM_UPPER_THRESHOLD, SM ignores this history
		double sm_scale_threshold_p		= 1.0,							// [cm] Threshold scaling factor used by SM to adjust edge detection sensitivity
		//---------------------------------------------------------------------------------------------------------------------------------------------------------------------//
		//------------------------------------------------------------ Program execution behavior options/parameters ----------------------------------------------------------//
		//---------------------------------------------------------------------------------------------------------------------------------------------------------------------//
		bool fbp_on_p					= true,							// [T/F] Turn FBP on (T) or off (F)
		bool sc_on_p					= false,						// [T/F] Turn Space Carving on (T) or off (F)
		bool msc_on_p					= true,							// [T/F] Turn Modified Space Carving on (T) or off (F)
		bool sm_on_p					= false,						// [T/F] Turn Space Modeling on (T) or off (F)
		bool import_preprocessing_p		= true,							// [T/F] Import preprocessed data previously generated, i.e. A/x0/b/hull/MLP), (T) or generate it (F) 
		bool perform_reconstruction_p	= true,							// [T/F] Perform reconstruction (T) or not (F)
		bool preprocess_overwrite_ok_p	= false,						// [T/F] Allow preprocessing data to be overwritten (T) or not (F)
		bool recon_overwrite_ok_p 		= false,						// [T/F] Allow reconstruction data to be overwritten (T) or not (F)
		bool mlp_in_loop_p				= false,						// [T/F] Perform MLP calculations for each GPU block/iteration (T) or not (F)
		bool import_data_iteratively_p	= true,							// [T/F] Import preprocessing data directly into block arrays as needed (T) or in entirety (F)
		//bool mlp_file_exists_p			= false,
		//bool histories_file_exists_p	= false,	
		//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------//
		//----------------------------------------------------------------------- Filtering options/parameters ------------------------------------------------------------------------//
		//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------//
		bool median_filter_hull_p			= false,					// [T/F] Apply median filter to hull (T) or not (F)
		bool median_filter_fbp_p			= false, 					// [T/F] Apply median filter to FBP (T) or not (F)
		bool median_filter_x_0_p			= false,					// [T/F] Apply median filter to initial iterate (T) or not (F)
		bool median_filter_x_k_p			= false,					// [T/F] Apply median filter to reconstructed image after each iteration (T) or not (F)
		bool median_filter_x_p				= false,					// [T/F] Apply median filter to final reconstructed image (T) or not (F)
		bool avg_filter_hull_p				= false,					// [T/F] Apply averaging filter to hull (T) or not (F)
		bool avg_filter_fbp_p				= false,					// [T/F] Apply averaging filter to initial iterate (T) or not (F)
		bool avg_filter_x_0_p				= false,					// [T/F] Apply averaging filter to initial iterate (T) or not (F)
		bool avg_filter_x_k_p				= false,					// [T/F] Apply averaging filter to initial iterate (T) or not (F)
		bool avg_filter_x_p					= false,					// [T/F] Apply averaging filter to initial iterate (T) or not (F)
		uint hull_med_filter_radius_p		= false,					// [#] Radius of median filter neighborhood applied to hull: [voxel - r, voxel + r]
		uint fbp_med_filter_radius_p		= false,					// [#] Radius of median filter neighborhood applied to FBP: [voxel - r, voxel + r]							
		uint x_0_med_filter_radius_p		= false,					// [#] Radius of median filter neighborhood applied to x_0: [voxel - r, voxel + r]'
		uint x_k_med_filter_radius_p		= false,					// [#] Radius of median filter neighborhood applied to x_k: [voxel - r, voxel + r]
		uint x_med_filter_radius_p			= false,					// [#] Radius of median filter neighborhood applied to x: [voxel - r, voxel + r]
		uint hull_avg_filter_radius_p		= false, 					// [#] Radius of average filter neighborhood applied to hull: [voxel - r, voxel + r]
		uint fbp_avg_filter_radius_p		= false,					// [#] Radius of average filter neighborhood applied to FBP: [voxel - r, voxel + r]
		uint x_0_avg_filter_radius_p		= false,					// [#] Radius of average filter neighborhood applied to x_0: [voxel - r, voxel + r]
		uint x_k_avg_filter_radius_p		= false,					// [#] Radius of average filter neighborhood applied to x_k: [voxel - r, voxel + r]
		uint x_avg_filter_radius_p			= false,					// [#] Radius of average filter neighborhood applied to x: [voxel - r, voxel + r]
		//---------------------------------------------------------------------------------------------------------------------------------------------------------------------//
		//----------------------------------------------------------------------- Output option parameters --------------------------------------------------------------------//
		//---------------------------------------------------------------------------------------------------------------------------------------------------------------------//
		bool write_msc_counts_p			= true,							// [T/F] Write MSC counts array to disk (T) or not (F) before performing edge detection 
		bool write_sm_counts_p			= true,							// [T/F] Write SM counts array to disk (T) or not (F) before performing edge detection 
		bool write_x_fbp_p				= true,							// [T/F] Write FBP image before thresholding to disk (T) or not (F)
		bool write_fbp_hull_p			= true,							// [T/F] Write FBP hull to disk (T) or not (F)
		bool write_avg_fbp_p			= true,							// [T/F] Write average filtered FBP image before thresholding to disk (T) or not (F)
		bool write_median_fbp_p			= false,						// [T/F] Write median filtered FBP image to disk (T) or not (F)
		bool write_bin_wepls_p			= false,						// [T/F] Write WEPLs for each bin to disk (T) for WEPL distribution analysis, or do not (F)
		bool write_wepl_dists_p			= false,						// [T/F] Write mean WEPL values to disk (T) or not (F): t bin = columns, v bin = rows, 1 angle per file
		bool write_ssd_angles_p			= false,						// [T/F] Write angles for each proton through entry/exit tracker planes to disk (T), or do not (F)
		//---------------------------------------------------------------------------------------------------------------------------------------------------------------------//
		//----------------------------------------------------------------------- Program Execution Control -------------------------------------------------------------------//
		//---------------------------------------------------------------------------------------------------------------------------------------------------------------------//
		bool stdout_2_disk_p			= false,						// [T/F] Redirect console window output to text file (T) or leave it as stdout (F)
		bool user_input_requests_off_p	= false,						// [T/F] Skip all functions that pause execution while waiting for user input (T) or allow user input requests (F)
		bool add_data_log_entry_p		= true,							// [T/F] Add log entry for data generated during execution (T) or not (F)
		bool debug_text_on_p			= true,							// Provide (T) or suppress (F) print statements to console during execution
		bool exit_after_binning_p		= false,						// Exit program early after completing data read and initial processing
		bool exit_after_hulls_p			= true,							// Exit program early after completing hull-detection
		bool exit_after_cuts_p			= false,						// Exit program early after completing statistical cuts
		bool exit_after_sinogram_p		= false,						// Exit program early after completing the ruction of the sinogram
		bool exit_after_fbp_p			= false							// Exit program early after completing FBP
	):
	//*************************************************************************************************************************************************************************//
	//*********************************************************************** Parameter Instantiations ************************************************************************//
	//*************************************************************************************************************************************************************************//
	NUM_SCANS(num_scans_p),												// *[#] Total number of scans of same object	
	MAX_GPU_HISTORIES(max_gpu_histories_p),								// *[#] Number of histories to process on the GPU at a time, based on GPU capacity
	MAX_CUTS_HISTORIES(max_cuts_histories_p),							// *[#] Number of histories to process on the GPU at a time, based on GPU capacity
	GANTRY_ANGLE_INTERVAL(gantry_angle_interval_p),						// *[degrees] Angle between successive projection angles	
	ANGULAR_BIN_SIZE(angular_bin_size_p),								// *[degrees] Angle between adjacent bins in angular (rotation) direction
	SSD_T_SIZE(ssd_t_size_p),											// *[cm] Length of SSD in t (lateral) direction
	SSD_V_SIZE(ssd_v_size_p),											// *[cm] Length of SSD in v (vertical) direction
	T_BIN_SIZE(t_bin_size_p),											// *[cm] Distance between adjacent bins in t (lateral) direction
	V_BIN_SIZE(v_bin_size_p),											// *[cm] Distance between adjacent bins in v (vertical) direction
	T_SHIFT(t_shift_p),													// *[cm] Amount by which to shift all t coordinates on input
	U_SHIFT(u_shift_p),													// *[cm] Amount by which to shift all u coordinates on input
	V_SHIFT(v_shift_p),													// *[cm] Amount by which to shift all v coordinates on input
	SIGMAS_2_KEEP(sigmas_2_keep_p),										// *[#] Number of standard deviations from mean to allow before cutting the history 
	RECON_CYL_RADIUS(recon_cyl_radius_p),								// *[cm] Radius of reconstruction cylinder
	COLUMNS(columns_p),													// *[#] Number of voxels in the x direction (i.e., number of columns) of image
	ROWS(rows_p),														// *[#] Number of voxels in the y direction (i.e., number of rows) of image
	VOXEL_THICKNESS(slice_thickness_p),									// *[cm] distance between top and bottom of each slice in image
	SLICE_THICKNESS(slice_thickness_p),									// *[cm] distance between top and bottom of each slice in image
	//---------------------------------------------------------------------------------------------------------------------------------------------------------------------//
	//------------------------------------------------------------------- Enumerated type parameters/options --------------------------------------------------------------//
	//---------------------------------------------------------------------------------------------------------------------------------------------------------------------//	
	DATA_TYPE(data_type_p),												// Specifies the source of the input data (EXPERIMENTAL = 0, GEANT4 = SIMULATED_G = 1, TOPAS = SIMULATED_T = 2)
	HULL_TYPE(hull_type_p),												// Specify which hull detection method to use for MLP calculations (IMPORT = 0, SC = 1, MSC = 2, SM = 3, FBP = 4)
	FBP_FILTER_TYPE(fbp_filter_type_p),		  							// Specifies which of the defined filters to use in FBP (RAM_LAK = 0, SHEPP_LOGAN = 1, NONE = 2)
	X_0_TYPE(x_0_type_p),												// Specify which initial iterate to use for reconstruction (IMPORT = 0, HULL = 1, FBP = 2, HYBRID = 3, ZEROS = 4)
	RECONSTRUCTION_METHOD(reconstruction_method_p),						// Specify algorithm to use for image reconstruction (ART = 0, DROP = 1, BIP = 2, SAP = 3, ROBUST1 = 4, ROBUST2 = 5)
	//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------//
	//--------------------------------------------------------- Options/parameters dependent on others read from config file --------------------------------------------------//
	//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------//
	NUM_FILES( num_scans_p * GANTRY_ANGLES ),							// *[#] 1 file per gantry angle per translation
	GANTRY_ANGLES(static_cast<uint>( 360 / gantry_angle_interval_p )),	// *[#] Total number of projection angles
	T_BINS(static_cast<uint>( ssd_t_size_p / t_bin_size_p + 0.5 )),		// *[#] Number of bins (i.e. quantization levels) for t (lateral) direction 
	V_BINS(static_cast<uint>( ssd_v_size_p / v_bin_size_p + 0.5 )),		// *[#] Number of bins (i.e. quantization levels) for v (vertical) direction
	ANGULAR_BINS(static_cast<uint>( 360 / angular_bin_size_p + 0.5 )),	// *[#] Number of bins (i.e. quantization levels) for path angle 
	NUM_BINS( ANGULAR_BINS * T_BINS * V_BINS ),							// *[#] Total number of bins corresponding to possible 3-tuples [ANGULAR_BIN, T_BIN, V_BIN],
	SLICES(static_cast<uint>( RECON_CYL_HEIGHT / slice_thickness_p)),	// *[#] Number of voxels in the z direction (i.e., number of slices) of image
	NUM_VOXELS( columns_p * rows_p * SLICES ),							// *[#] Total number of voxels (i.e. 3-tuples [column, row, slice]) in image	
	RECON_CYLIAMETER( 2 * recon_cyl_radius_p ),							// *[cm] Diameter of reconstruction cylinder
	RECON_CYL_HEIGHT(ssd_v_size_p - 1.0),								// *[cm] Height of reconstruction cylinder
	IMAGE_WIDTH(recon_cyl_radius_p * 2),								// *[cm] Distance between left and right edges of each slice in image
	IMAGE_HEIGHT(recon_cyl_radius_p * 2),								// *[cm] Distance between top and bottom edges of each slice in image
	IMAGE_THICKNESS(ssd_v_size_p - 1.0),								// *[cm] Distance between bottom of bottom slice and top of the top slice of image
	VOXEL_WIDTH(recon_cyl_radius_p * 2 / columns_p),					// *[cm] distance between left and right edges of each voxel in image
	VOXEL_HEIGHT(recon_cyl_radius_p * 2 / rows_p),						// *[cm] distance between top and bottom edges of each voxel in image
	X_ZERO_COORDINATE(-recon_cyl_radius_p),								// *[cm] x-coordinate corresponding to left edge of 1st voxel (i.e. column) in image space
	Y_ZERO_COORDINATE(recon_cyl_radius_p),								// *[cm] y-coordinate corresponding to top edge of 1st voxel (i.e. row) in image space
	Z_ZERO_COORDINATE((ssd_v_size_p - 1.0)/2),							// *[cm] z-coordinate corresponding to top edge of 1st voxel (i.e. slice) in image space
	RAM_LAK_TAU(2/sqrt(2.0) * t_bin_size_p),							// *[#] Defines tau in Ram-Lak filter calculation, estimated from largest frequency in slice 
	//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------//
	//----------------------------------------------------------------------------- MLP Parameters ----------------------------------------------------------------------------//
	//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------//
	VOXEL_STEP_SIZE( VOXEL_WIDTH / 2 ),									// *[cm] Length of the step taken along the path, i.e. change in depth per step for
	MLP_U_STEP( VOXEL_WIDTH / 2),										// *[cm] Size of the step taken along u direction during MLP; depth difference between successive MLP points
	CONSTANT_CHORD_NORM(pow(VOXEL_WIDTH, 2.0)),							// *[cm^2] Precalculated value of ||a_i||^2 for use of constant chord length
	CONSTANT_LAMBDA_SCALE(VOXEL_WIDTH * LAMBDA),						// *[cm] Precalculated value of |a_i| * LAMBDA for use of constant chord length
	//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------//
	//---------------------------------------------------------- Reconstruction and image filtering parameters/options --------------------------------------------------------//
	//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------//
	ITERATIONS(iterations_p),											// *[#] of iterations through the entire set of histories to perform in iterative image reconstruction
	BLOCK_SIZE(block_size_p),											// *[#] of paths to use for each update: e.g., ART = 1, 
	PSI_SIGN(psi_sign_p),												// *[+1/-1] Sign specifying the sign to use for Psi in scaling residual for updates in robust technique to reconstruction	
	LAMBDA(lambda_p),													// *[#] Relaxation parameter used in update calculations in reconstruction algorithms
	//LAMBDA(lambda_p),													// *[#] Relaxation parameter used in update calculations in reconstruction algorithms
	ETA(eta_p),															// *[#] Value used in calculation of Psi = (1-x_i) * ETA used in robust technique to reconstruction
	//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------//
	//----------------------------------------------------------------------- Hull-detection parameters -----------------------------------------------------------------------//
	//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------//
	HULL_RSP_THRESHOLD(hull_rsp_threshold_p),							// [#] Maximum RSP for voxels assumed to belong to hull
	MSC_DIFF_THRESH(msc_diff_thresh_p),									// *[#] Threshold on difference in counts between adjacent voxels used by MSC for edge detection
	SC_THRESHOLD(sc_threshold_p),										// *[cm] If WEPL < SC_THRESHOLD, SC assumes the proton missed the object
	MSC_THRESHOLD(msc_threshold_p),										// *[cm] If WEPL < MSC_THRESHOLD, MSC assumes the proton missed the object
	SM_LOWER_THRESHOLD(sm_lower_threshold_p),							// *[cm] If WEPL >= SM_THRESHOLD, SM assumes the proton passed through the object
	SM_UPPER_THRESHOLD(sm_upper_threshold_p),							// *[cm] If WEPL > SM_UPPER_THRESHOLD, SM ignores this history
	SM_SCALE_THRESHOLD(sm_scale_threshold_p),							// *[cm] Threshold scaling factor used by SM to adjust edge detection sensitivity
	//*************************************************************************************************************************************************************************//
	//*********************************************************************** Preprocessing control options *******************************************************************//
	//*************************************************************************************************************************************************************************//
	FBP_ON(fbp_on_p),													// *[T/F] Turn FBP on (T) or off (F)
	SC_ON(sc_on_p),														// *[T/F] Turn Space Carving on (T) or off (F)
	MSC_ON(msc_on_p),													// *[T/F] Turn Modified Space Carving on (T) or off (F)
	SM_ON(sm_on_p),														// *[T/F] Turn Space Modeling on (T) or off (F)
	IMPORT_PREPROCESSING(import_preprocessing_p),						// *[T/F] Import preprocessed data previously generated, i.e. A/x0/b/hull/MLP), (T) or generate it (F) 
	PERFORM_RECONSTRUCTION(perform_reconstruction_p),					// *[T/F] Perform reconstruction (T) or not (F)
	PREPROCESS_OVERWRITE_OK(preprocess_overwrite_ok_p),					// *[T/F] Allow preprocessing data to be overwritten (T) or not (F)
	RECON_OVERWRITE_OK(recon_overwrite_ok_p),							// *[T/F] Allow reconstruction data to be overwritten (T) or not (F)
	MLP_IN_LOOP(mlp_in_loop_p),											// [T/F] Perform MLP calculations for each GPU block/iteration (T) or not (F)
	IMPORT_DATA_ITERATIVELY(import_data_iteratively_p),					// [T/F] Import preprocessing data directly into block arrays as needed (T) or in entirety (F)
	//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------//
	//----------------------------------------------------------------------- Filtering options/parameters ------------------------------------------------------------------------//
	//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------//
	MEDIAN_FILTER_HULL(median_filter_hull_p),							// [T/F] Apply median filter to hull (T) or not (F)
	MEDIAN_FILTER_FBP(median_filter_fbp_p), 							// [T/F] Apply median filter to FBP (T) or not (F)
	MEDIAN_FILTER_X_0(median_filter_x_0_p),								// [T/F] Apply median filter to initial iterate (T) or not (F)
	MEDIAN_FILTER_X_K(median_filter_x_k_p),								// [T/F] Apply median filter to reconstructed image after each iteration (T) or not (F)
	MEDIAN_FILTER_X(median_filter_x_p),									// [T/F] Apply median filter to final reconstructed image (T) or not (F)
	AVG_FILTER_HULL(avg_filter_hull_p),									// [T/F] Apply averaging filter to hull (T) or not (F)
	AVG_FILTER_FBP(avg_filter_fbp_p),									// [T/F] Apply averaging filter to initial iterate (T) or not (F)
	AVG_FILTER_X_0(avg_filter_x_0_p),									// [T/F] Apply averaging filter to initial iterate (T) or not (F)
	AVG_FILTER_X_K(avg_filter_x_k_p),									// [T/F] Apply averaging filter to initial iterate (T) or not (F)
	AVG_FILTER_X(avg_filter_x_p),										// [T/F] Apply averaging filter to initial iterate (T) or not (F)
	HULL_MED_FILTER_RADIUS(hull_med_filter_radius_p),					// [#] Radius of median filter neighborhood applied to hull: [voxel - r, voxel + r]
	FBP_MED_FILTER_RADIUS(fbp_med_filter_radius_p),						// [#] Radius of median filter neighborhood applied to FBP: [voxel - r, voxel + r]							
	X_0_MED_FILTER_RADIUS(x_0_med_filter_radius_p),						// [#] Radius of median filter neighborhood applied to x_0: [voxel - r, voxel + r]'
	X_K_MED_FILTER_RADIUS(x_k_med_filter_radius_p),						// [#] Radius of median filter neighborhood applied to x_k: [voxel - r, voxel + r]
	X_MED_FILTER_RADIUS(x_med_filter_radius_p),							// [#] Radius of median filter neighborhood applied to x: [voxel - r, voxel + r]
	HULL_AVG_FILTER_RADIUS(hull_avg_filter_radius_p), 					// [#] Radius of average filter neighborhood applied to hull: [voxel - r, voxel + r]
	FBP_AVG_FILTER_RADIUS(fbp_avg_filter_radius_p),						// [#] Radius of average filter neighborhood applied to FBP: [voxel - r, voxel + r]
	X_0_AVG_FILTER_RADIUS(x_0_avg_filter_radius_p),						// [#] Radius of average filter neighborhood applied to x_0: [voxel - r, voxel + r]
	X_K_AVG_FILTER_RADIUS(x_k_avg_filter_radius_p),						// [#] Radius of average filter neighborhood applied to x_k: [voxel - r, voxel + r]
	X_AVG_FILTER_RADIUS(x_avg_filter_radius_p),							// [#] Radius of average filter neighborhood applied to x: [voxel - r, voxel + r]
	//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------//
	//--------------------------------------------------------- Control of writing optional intermediate data to disk  --------------------------------------------------------//
	//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------//
	WRITE_MSC_COUNTS(write_msc_counts_p),								// *[T/F] Write MSC counts array to disk (T) or not (F) before performing edge detection 
	WRITE_SM_COUNTS(write_sm_counts_p),									// *[T/F] Write SM counts array to disk (T) or not (F) before performing edge detection 
	WRITE_X_FBP(write_x_fbp_p),											// *[T/F] Write FBP image before thresholding to disk (T) or not (F)
	WRITE_FBP_HULL(write_fbp_hull_p),									// *[T/F] Write FBP hull to disk (T) or not (F)
	WRITE_AVG_FBP(write_avg_fbp_p),										// *[T/F] Write average filtered FBP image before thresholding to disk (T) or not (F)
	WRITE_MEDIAN_FBP(write_median_fbp_p),								// *[T/F] Write median filtered FBP image to disk (T) or not (F)
	WRITE_BIN_WEPLS(write_bin_wepls_p),									// *[T/F] Write WEPLs for each bin to disk (T) for WEPL distribution analysis, or do not (F)
	WRITE_WEPL_DISTS(write_wepl_dists_p),									// *[T/F] Write mean WEPL values to disk (T) or not (F): t bin = columns, v bin = rows, 1 angle per file
	WRITE_SSD_ANGLES(write_ssd_angles_p),								// *[T/F] Write angles for each proton through entry/exit tracker planes to disk (T), or do not (F)
	//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------//
	//---------------------------------------------------------------------- Program Execution Control  -----------------------------------------------------------------------//
	//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------//
	STDOUT_2_DISK(stdout_2_disk_p	),										// [T/F] Redirect console window output to text file (T) or leave it as stdout (F)	
	USER_INPUT_REQUESTS_OFF(user_input_requests_off_p),					// [T/F] Skip all functions that pause execution while waiting for user input (T) or allow user input requests (F)
	ADD_DATA_LOG_ENTRY(add_data_log_entry_p),								// *[T/F] Add log entry for data generated during execution (T) or not (F)
	DEBUG_TEXT_ON(debug_text_on_p),										// Provide (T) or suppress (F) print statements to console during execution
	EXIT_AFTER_BINNING(exit_after_binning_p),							// Exit program early after completing data read and initial processing
	EXIT_AFTER_HULLS(exit_after_hulls_p),								// Exit program early after completing hull-detection
	EXIT_AFTER_CUTS(exit_after_cuts_p),									// Exit program early after completing statistical cuts
	EXIT_AFTER_SINOGRAM(exit_after_sinogram_p),							// Exit program early after completing the ruction of the sinogram
	EXIT_AFTER_FBP(exit_after_fbp_p)									// Exit program early after completing FBP
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
void FBP_2_hull();
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
void define_hull();
void define_x_0();

// Image filtering functions
template<typename H, typename D> void averaging_filter( H*&, D*&, int, bool, double );
template<typename D> __global__ void averaging_filter_GPU( configurations*, D*, D*, int, bool, double );
template<typename T> void median_filter_2D( T*&, unsigned int );
template<typename T> void median_filter_2D( T*&, T*&, unsigned int );
template<typename D> __global__ void median_filter_GPU( configurations*, D*, D*, int, bool, double );
template<typename T> void median_filter_3D( T*&, T*&, unsigned int );
template<typename T, typename T2> __global__ void apply_averaging_filter_GPU( configurations*, T*, T2* );
template<typename T> void test_median_filter_radii(T*&, char* );

// History Ordering
void generate_history_sequence(ULL, ULL );
void verify_history_sequence(ULL, ULL );
void print_history_sequence(ULL, ULL );

// Average chord length
double mean_chord_length2( double, double, double, double, double, double, double, double );
double mean_chord_length( double, double, double, double, double, double );
double EffectiveChordLength(double, double);
float EffectiveChordLength(float, float);

// MLP
template<typename O> bool find_MLP_endpoints( O*&, double, double, double, double, double, double&, double&, double&, int&, int&, int&, bool );
void collect_MLP_endpoints();
unsigned int calculate_MLP( unsigned int*&, float*&, double, double, double, double, double, double, double, double, double, double, int, int, int );

// Preprocessing Data I/O
void export_hull();
void export_x_0();
void export_WEPL();
void export_histories();
void export_voxels_per_path();
void export_avg_chord_lengths();
void accumulate_and_export_preprocessed_data();
void import_hull();
void import_x_0();
uint import_WEPL();
uint import_histories();
uint import_voxels_per_path();
uint import_avg_chord_lengths();
bool prepare_2_import_preprocessed_data( bool );
void allocate_reconstruction_arrays(uint );
uint vector_copy_block_arrays( uint, uint);
uint import_block_arrays( uint);

// Image Reconstruction (host)
void image_reconstruction();
void DROP_import_data();
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
void print_copyright_notice();
void std_IO_2_disk();

// Generic IO helper functions
template<typename T> T cin_until_valid( T*, int, char* );
char((&current_MMDD( char(&)[5]))[5]);
char((&current_MMDDYYYY( char(&)[9]))[9]);
template<typename T> char((&minimize_trailing_zeros( T, char(&)[64]) )[64]);
std::string terminal_response(char*);
char((&terminal_response( char*, char(&)[256]))[256]);
bool directory_exists(char* );
unsigned int create_unique_dir( char* );
void get_dir_filenames(std::vector<std::string> &, const std::string &);
void get_dir_filenames_matching(const std::string &, const std::string &, std::vector<std::string> &, std::vector<std::string> &);
bool file_exists (const char* file_location) { return static_cast<bool>( std::ifstream(file_location) ); };
bool file_exists2 (const char* file_location) { return std::ifstream(file_location).good(); };
bool file_exists3 (const char*);
bool blank_line( char c ) { return (c != '\n') && (c!= '\t') && (c != ' '); };
const char * bool_2_string( bool b ){ return b ? "true" : "false"; }
void fgets_validated(char *line, int buf_size, FILE*);
struct generic_IO_container read_key_value_pair( FILE* );

// Configuration option/parameter handling
bool preprocessing_data_exists();
void add_object_directory(char*, char*);
int add_run_directory(char*, char*, char*, char*, SCAN_TYPES );
int add_pCT_Images_dir(char*, char*, char*, char*, SCAN_TYPES );

CONFIG_LINE split_config_comments(char*);
void write_config( CONFIG_OBJECT);
void fgets_config(char*, int, FILE*, CONFIG_OBJECT&);
uint parse_config_file_line( FILE*, CONFIG_OBJECT& );
CONFIG_OBJECT config_file_2_object();
bool key_is_string_parameter( char* );
bool key_is_floating_point_parameter( char* );
bool key_is_unsigned_integer_parameter( char* );
bool key_is_integer_parameter( char* );
bool key_is_boolean_parameter( char* );
void set_string_parameter( generic_IO_container & );
void set_floating_point_parameter( generic_IO_container & );
void set_unsigned_integer_parameter( generic_IO_container & );
void set_integer_parameter( generic_IO_container & );
void set_boolean_parameter( generic_IO_container & );
void set_parameter( generic_IO_container & );
void set_file_extension( char*, DISK_WRITE_MODE );
void set_execution_date();
void set_IO_directories();
void view_config_file();
void set_dependent_parameters();
void set_IO_file_extensions();
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
__global__ void test_transfer_GPU(float*, float*, float*);
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
__global__ void FBP_2_hull_GPU( configurations*, float*, bool* );

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
__device__ int calculate_MLP_GPU( configurations*, int*&, double*&, double, double, double, double, double, double, double, double, double, double, int, int, int );
__device__ void MLP_GPU(configurations*);

// Image Reconstruction
__global__ void DROP_calculate_update_GPU(configurations*, int*, int*, float*, float*, int*, float*, float* );
__global__ void update_x_1D_GPU( configurations* parameters, int*&, float*&, float*& );
__global__ void update_x_3D_GPU( configurations* parameters, int*&, float*&, float*&  );
__global__ void DROP_apply_update_GPU(configurations*, int*, float*, float* );
//template< typename X> __device__ double update_vector_multiplier2( configurations*, double, double, X*&, int*, int );
__device__ double scalar_dot_product_GPU_2( configurations*, double, float*&, int*, int );
__device__ double update_vector_multiplier_GPU_22( configurations*, double, double, float*&, int*, int );
//template< typename X> __device__ void update_iterate2( configurations*, double, double, X*&, int*, int );
__device__ void update_iterate_GPU_22( configurations*, double, double, float*&, int*, int );

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
	CONFIG_PATH  = (char*) calloc( strlen(PROJECTION_DATA_DIR) + strlen(CONFIG_FILENAME) + 1, sizeof(char) );
	sprintf(CONFIG_PATH, "%s/%s", PROJECTION_DATA_DIR, CONFIG_FILENAME );
	FILE* input_file = fopen(CONFIG_PATH, "r" );
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
void view_config_file()
{
	char filename[256]; 

	#if defined(_WIN32) || defined(_WIN64)
		sprintf(filename, "%s %s %s", "start", "wordpad", CONFIG_PATH);
		terminal_response(filename);
    #else
		sprintf(filename, "%s %s", "touch", CONFIG_FILENAME);
		terminal_response(filename);
    #endif
	
}
CONFIG_LINE split_config_comments(char* comment_line)
{
	std::string entry_string(comment_line);
	entry_string.pop_back(); // Pop off endline character
	for( int i = 0; i < (int)entry_string.length(); i++)
	{
		if(entry_string[i] == '\t' )
		{
			entry_string[i] = ' ';
			entry_string.insert(i, 3, ' ');
		}
		
	}
	entry_string.resize(CONFIG_LINE_WIDTH, ' ');
	CONFIG_LINE parsed_comment;
	std::string temp;
	size_t comment_position = 0, temp_length = 0;
	for( int i = 0; i < NUM_CONFIG_FIELDS; i++ )
	{
		temp_length = CONFIG_FIELD_WIDTHS[i];
		temp = entry_string.substr(comment_position, temp_length );
		parsed_comment.push_back(temp);
		comment_position += temp_length;
	}
	return parsed_comment;
}
void write_config( CONFIG_OBJECT config_object)
{
	std::ofstream config_file(CONFIG_PATH);
	int i;

	if( !config_file.is_open() )
		config_file.open(CONFIG_PATH);
	else
	{
		for( i = 0; i < config_object.size() - 1; i++ )
		{
			config_file << std::noskipws;		
			for( int j = 0; j < NUM_CONFIG_FIELDS; j++ )
				config_file <<  config_object[i][j];
			config_file << endl;
		}
		for( int j = 0; j < NUM_CONFIG_FIELDS; j++ )
			config_file <<  config_object[i][j];
	}
	config_file.close();
}
void fgets_config(char *line, int buf_size, FILE* input_file, CONFIG_OBJECT& config_object)
{
    bool done = false;
	char* line_copy = line;
    while(!done)
    {
        if( fgets(line, buf_size, input_file ) == NULL )										// Read a line from the file
            return;		
		line_copy = line;
		while( *line == ' ' || *line == '\t' )
			line++;
        if( std::find_if(line, &line[strlen(line)], blank_line ) == ( &line[strlen(line)] ) )	// Skip lines with only "\n", "\t", and/or " "
		{
			config_object.push_back(split_config_comments(line_copy));
			continue;
		}
        else if( strncmp( line, "//", 2 ) == 0 )												// Skip any comment lines
        {
			config_object.push_back(split_config_comments(line_copy));
			continue;
		}
        else																					// Got a valid data line so return with this line
			done = true;    
    }
}
uint parse_config_file_line( FILE* input_file, CONFIG_OBJECT& config_object )
{
	char key[128], equal_sign[128], value[256], comments[512];	
	const uint buf_size = 1024;
	char line[buf_size];
	std::size_t found;
	CONFIG_LINE config_line;
	uint parameters_changed = 0;
	char* parameter;
	
	// Remove leading spaces/tabs and return first line which does not begin with comment command "//" and is not blank
	fgets_config(line, buf_size, input_file, config_object);	
	
	// Having now read a non comment/blank line and removed leading spaces/tabs, parse it into {key}{=}{value}//{comments} format
	int filled = sscanf (line, "%s %s %s // %s", &key, &equal_sign, &value, &comments);

	std::string line_string(line), key_string(key), equal_string(equal_sign), value_string(value), comment_string(comments);

	// n =				  1			   1		   2			2		   3		   3	 ...	 n			 n       
	// i =	 0			  1			   2		   3			4		   5		   6	 ...   2n - 1  		 2n		  2n + 1
	// [program name][parameter 1][new val 1][parameter 2][new val 2][parameter 3][new val 3]...[parameter n][new val n][cfg path]
	for( uint n = 1; n <= NUM_PARAMETERS_2_CHANGE; n++)
	{
		parameter =  RUN_ARGUMENTS[2 * n - 1];
		if( strcmp(key, parameter) == 0 )
		{
			if(		key_is_string_parameter( parameter )
				||	key_is_floating_point_parameter( parameter )
				||	key_is_integer_parameter( parameter )
				||	key_is_boolean_parameter( parameter ) 
			)
			{
				value_string = std::string(RUN_ARGUMENTS[2 * n]);	
				parameters_changed++;
			}
		}
	}
	//
	if( filled <= 3 )
		comment_string = "";
	else
	{
		found = line_string.find("//");
		comment_string = line_string.substr(found, std::string::npos);
		comment_string.pop_back();
	}
	if( (int)value_string.length() > VALUE_FIELD_WIDTH )
		comment_string.insert(0, value_string.substr(VALUE_FIELD_WIDTH, std::string::npos ) );	

	key_string.resize(KEY_FIELD_WIDTH, ' ');
	equal_string.resize(EQUALS_FIELD_WIDTH, ' ');
	value_string.resize(VALUE_FIELD_WIDTH, ' ');		
	comment_string.resize(COMMENT_FIELD_WIDTH, ' ');

	config_line.push_back(key_string);
	config_line.push_back(equal_string);
	config_line.push_back(value_string);
	config_line.push_back(comment_string);
	config_object.push_back(config_line);

	return parameters_changed;
}
CONFIG_OBJECT config_file_2_object()
{		
	// Extract current directory (executable path) terminal response from system command "chdir" 
	CONFIG_OBJECT config_object;
	uint parameters_changed = 0;
	if( !CONFIG_PATH_PASSED )
	{
		std::string str =  terminal_response("chdir");
		const char* cstr = str.c_str();
		PROJECTION_DATA_DIR = (char*) calloc( strlen(cstr), sizeof(char));
		std::copy( cstr, &cstr[strlen(cstr) - 1], PROJECTION_DATA_DIR );
		print_section_header( "Config file location set to current execution directory :", '*' );	
		print_section_separator('-');
		printf("%s\n", PROJECTION_DATA_DIR );
		print_section_separator('-');
	}
	CONFIG_PATH  = (char*) calloc( strlen(PROJECTION_DATA_DIR) + strlen(CONFIG_FILENAME) + 1, sizeof(char) );
	//sprintf(CONFIG_PATH, "C:/Users/Blake/Documents/pCT_Data/object_name/Experimental/MMDDYYYY/run_number/Output/MMDDYYYY/settings.cfg" );
	sprintf(CONFIG_PATH, "%s\\%s", PROJECTION_DATA_DIR, CONFIG_FILENAME );
	FILE* input_file = fopen(CONFIG_PATH, "r" );
	print_section_header( "Reading key/value pairs from configuration file and setting corresponding execution parameters", '*' );
	while( !feof(input_file) )	
		parameters_changed += parse_config_file_line(input_file, config_object);
	fclose(input_file);
	print_section_exit( "Finished reading configuration file and setting execution parameters", "====>" );	
	if( parameters_changed < NUM_PARAMETERS_2_CHANGE )
	{
		puts("ERROR: Parameter specified for value change does not have a valid key."); 
		exit_program_if(true);
	}
	write_config( config_object);
	return config_object;
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
		||	strcmp (key, "HULL_RSP_THRESHOLD") == 0 
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
bool key_is_unsigned_integer_parameter( char* key )
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
		||	strcmp (key, "HULL_MED_FILTER_RADIUS") == 0
		||	strcmp (key, "FBP_MED_FILTER_RADIUS") == 0
		||	strcmp (key, "X_0_MED_FILTER_RADIUS") == 0
		||	strcmp (key, "X_K_MED_FILTER_RADIUS") == 0
		||	strcmp (key, "X_MED_FILTER_RADIUS") == 0
		||	strcmp (key, "HULL_AVG_FILTER_RADIUS") == 0
		||	strcmp (key, "FBP_AVG_FILTER_RADIUS") == 0
		||	strcmp (key, "X_0_AVG_FILTER_RADIUS") == 0
		||	strcmp (key, "X_K_AVG_FILTER_RADIUS") == 0
		||	strcmp (key, "X_AVG_FILTER_RADIUS") == 0							
		||	strcmp (key, "MSC_DIFF_THRESH") == 0
	)
		return true;
	else
		return false;
}
bool key_is_integer_parameter( char* key )
{
	if
	(	
		strcmp (key, "PSI_SIGN") == 0
	)
		return true;
	else
		return false;
}
bool key_is_boolean_parameter( char* key )
{
	if
	( 
			strcmp (key, "FBP_ON") == 0
		||	strcmp (key, "SC_ON") == 0
		||	strcmp (key, "MSC_ON") == 0
		||	strcmp (key, "SM_ON") == 0
		||	strcmp (key, "IMPORT_PREPROCESSING") == 0
		||	strcmp (key, "PERFORM_RECONSTRUCTION") == 0
		||	strcmp (key, "PREPROCESS_OVERWRITE_OK") == 0
		||	strcmp (key, "RECON_OVERWRITE_OK") == 0
		||	strcmp (key, "MLP_IN_LOOP") == 0
		||	strcmp (key, "IMPORT_DATA_ITERATIVELY") == 0
		||	strcmp (key, "MEDIAN_FILTER_HULL") == 0
		||	strcmp (key, "MEDIAN_FILTER_FBP") == 0
		||	strcmp (key, "MEDIAN_FILTER_X_0") == 0
		||	strcmp (key, "MEDIAN_FILTER_X_K") == 0
		||	strcmp (key, "MEDIAN_FILTER_X") == 0
		||	strcmp (key, "AVG_FILTER_HULL") == 0
		||	strcmp (key, "AVG_FILTER_FBP") == 0
		||	strcmp (key, "AVG_FILTER_X_0") == 0
		||	strcmp (key, "AVG_FILTER_X_K") == 0
		||	strcmp (key, "AVG_FILTER_X") == 0
		||	strcmp (key, "WRITE_MSC_COUNTS") == 0
		||	strcmp (key, "WRITE_SM_COUNTS") == 0
		||	strcmp (key, "WRITE_X_FBP") == 0
		||	strcmp (key, "WRITE_FBP_HULL") == 0
		||	strcmp (key, "WRITE_AVG_FBP") == 0
		||	strcmp (key, "WRITE_MEDIAN_FBP") == 0
		||	strcmp (key, "WRITE_BIN_WEPLS") == 0
		||	strcmp (key, "WRITE_WEPL_DISTS") == 0
		||	strcmp (key, "WRITE_SSD_ANGLES") == 0 
		||	strcmp (key, "ADD_DATA_LOG_ENTRY") == 0
		||	strcmp (key, "STDOUT_2_DISK") == 0
		||	strcmp (key, "USER_INPUT_REQUESTS_OFF") == 0
		||	strcmp (key, "DEBUG_TEXT_ON") == 0
		||	strcmp (key, "EXIT_AFTER_BINNING") == 0
		||	strcmp (key, "EXIT_AFTER_HULLS") == 0
		||	strcmp (key, "EXIT_AFTER_CUTS") == 0
		||	strcmp (key, "EXIT_AFTER_SINOGRAM") == 0
		||	strcmp (key, "EXIT_AFTER_FBP") == 0	
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
		puts("PROJECTION_DATA_DIR_SET");
	}
	else if( strcmp (value.key, "PREPROCESSING_DIR") == 0 )
	{
		puts("");
		//print_section_separator('-');
		PREPROCESSING_DIR = (char*) calloc( strlen(value.string_input) + 1, sizeof(char));
		std::copy( value.string_input, &value.string_input[strlen(value.string_input)], PREPROCESSING_DIR );
		PREPROCESSING_DIR_SET = true;
		puts("PREPROCESSING_DIR_SET");
	}
	else if( strcmp (value.key, "RECONSTRUCTION_DIR") == 0 )
	{
		puts("");
		//print_section_separator('-');
		RECONSTRUCTION_DIR = (char*) calloc( strlen(value.string_input) + 1, sizeof(char));
		std::copy( value.string_input, &value.string_input[strlen(value.string_input)], RECONSTRUCTION_DIR );
		RECONSTRUCTION_DIR_SET = true;
		puts("RECONSTRUCTION_DIR_SET");
	}
	else if( strcmp (value.key, "PATH_2_PCT_DATA_DIR") == 0 )
	{
		PATH_2_PCT_DATA_DIR = (char*) calloc( strlen(value.string_input) + 1, sizeof(char));
		std::copy( value.string_input, &value.string_input[strlen(value.string_input)], PATH_2_PCT_DATA_DIR );
		PATH_2_PCT_DATA_DIR_SET = true;
		puts("PATH_2_PCT_DATA_DIR_SET");
	}
	else if( strcmp (value.key, "OBJECT") == 0 )
	{
		OBJECT = (char*) calloc( strlen(value.string_input) + 1, sizeof(char));
		std::copy( value.string_input, &value.string_input[strlen(value.string_input)], OBJECT );
		OBJECT_SET = true;
		puts("OBJECT_SET");
	}
	else if( strcmp (value.key, "RUN_DATE") == 0 )
	{
		RUN_DATE = (char*) calloc( strlen(value.string_input) + 1, sizeof(char));
		std::copy( value.string_input, &value.string_input[strlen(value.string_input)], RUN_DATE );
		RUN_DATE_SET = true;
		puts("RUN_DATE_SET");
	}
	else if( strcmp (value.key, "RUN_NUMBER") == 0 )
	{
		RUN_NUMBER = (char*) calloc( strlen(value.string_input) + 1, sizeof(char));
		std::copy( value.string_input, &value.string_input[strlen(value.string_input)], RUN_NUMBER );
		RUN_NUMBER_SET = true;
		puts("RUN_NUMBER_SET");
	}
	else if( strcmp (value.key, "PROJECTION_DATA_DATE") == 0 )
	{
		PROJECTION_DATA_DATE = (char*) calloc( strlen(value.string_input) + 1, sizeof(char));
		std::copy( value.string_input, &value.string_input[strlen(value.string_input)], PROJECTION_DATA_DATE );
		PROJECTION_DATA_DATE_SET = true;
		puts("PROJECTION_DATA_DATE_SET");
	}
	else if( strcmp (value.key, "PREPROCESS_DATE") == 0 )
	{
		PREPROCESS_DATE = (char*) calloc( strlen(value.string_input) + 1, sizeof(char));
		std::copy( value.string_input, &value.string_input[strlen(value.string_input)], PREPROCESS_DATE );
		PREPROCESS_DATE_SET = true;
		puts("PREPROCESS_DATE_SET");
	}
	else if( strcmp (value.key, "RECONSTRUCTION_DATE") == 0 )
	{
		RECONSTRUCTION_DATE = (char*) calloc( strlen(value.string_input) + 1, sizeof(char));
		std::copy( value.string_input, &value.string_input[strlen(value.string_input)], RECONSTRUCTION_DATE );
		RECONSTRUCTION_DATE_SET = true;
		puts("RECONSTRUCTION_DATE_SET");
	}
	else if( strcmp (value.key, "USER_NAME") == 0 )
	{
		USER_NAME = (char*) calloc( strlen(value.string_input) + 1, sizeof(char));
		std::copy( value.string_input, &value.string_input[strlen(value.string_input)], USER_NAME );
		USER_NAME_SET = true;
		puts("USER_NAME_SET");
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
		parameters.GANTRY_ANGLE_INTERVAL = value.double_input;
	}
	else if( strcmp (value.key, "SSD_T_SIZE") == 0 )
	{
		if( value.double_input < 0 )
		{
			puts("ERROR: Negative value give to parameter that should be positive.\n  Correct the configuration file and rerun program");
			exit_program_if(true);
		}
		//SSD_T_SIZE = value.double_input;
		parameters.SSD_T_SIZE = value.double_input;
	}
	else if( strcmp (value.key, "SSD_V_SIZE") == 0 )
	{
		if( value.double_input < 0 )
		{
			puts("ERROR: Negative value give to parameter that should be positive.\n  Correct the configuration file and rerun program");
			exit_program_if(true);
		}
		//SSD_V_SIZE = value.double_input;
		parameters.SSD_V_SIZE = value.double_input;
	}
	else if( strcmp (value.key, "T_SHIFT") == 0 )
	{
		//T_SHIFT = value.double_input;
		parameters.T_SHIFT = value.double_input;
	}
	else if( strcmp (value.key, "U_SHIFT") == 0 )
	{
		//U_SHIFT = value.double_input;
		parameters.U_SHIFT = value.double_input;
	}
	else if( strcmp (value.key, "V_SHIFT") == 0 )
	{
		//V_SHIFT = value.double_input;
		parameters.V_SHIFT = value.double_input;
	}
	else if( strcmp (value.key, "T_BIN_SIZE") == 0 )
	{
		if( value.double_input < 0 )
		{
			puts("ERROR: Negative value give to parameter that should be positive.\n  Correct the configuration file and rerun program");
			exit_program_if(true);
		}
		//T_BIN_SIZE = value.double_input;
		parameters.T_BIN_SIZE = value.double_input;
	}
	else if( strcmp (value.key, "V_BIN_SIZE") == 0 )
	{
		if( value.double_input < 0 )
		{
			puts("ERROR: Negative value give to parameter that should be positive.\n  Correct the configuration file and rerun program");
			exit_program_if(true);
		}
		//V_BIN_SIZE = value.double_input;
		parameters.V_BIN_SIZE = value.double_input;
	}
	else if( strcmp (value.key, "ANGULAR_BIN_SIZE") == 0 )
	{
		if( value.double_input < 0 )
		{
			puts("ERROR: Negative value give to parameter that should be positive.\n  Correct the configuration file and rerun program");
			exit_program_if(true);
		}
		//ANGULAR_BIN_SIZE = value.double_input;
		parameters.ANGULAR_BIN_SIZE = value.double_input;
	}
	else if( strcmp (value.key, "RECON_CYL_RADIUS") == 0 )
	{
		if( value.double_input < 0 )
		{
			puts("ERROR: Negative value give to parameter that should be positive.\n  Correct the configuration file and rerun program");
			exit_program_if(true);
		}
		//RECON_CYL_RADIUS = value.double_input;
		parameters.RECON_CYL_RADIUS = value.double_input;
	}
	else if( strcmp (value.key, "RECON_CYL_HEIGHT") == 0 )
	{
		if( value.double_input < 0 )
		{
			puts("ERROR: Negative value give to parameter that should be positive.\n  Correct the configuration file and rerun program");
			exit_program_if(true);
		}
		//RECON_CYL_HEIGHT = value.double_input;
		parameters.RECON_CYL_HEIGHT = value.double_input;
	}
	else if( strcmp (value.key, "IMAGE_WIDTH") == 0 )
	{
		if( value.double_input < 0 )
		{
			puts("ERROR: Negative value give to parameter that should be positive.\n  Correct the configuration file and rerun program");
			exit_program_if(true);
		}
		//IMAGE_WIDTH = value.double_input;
		parameters.IMAGE_WIDTH = value.double_input;
	}
	else if( strcmp (value.key, "IMAGE_HEIGHT") == 0 )
	{
		if( value.double_input < 0 )
		{
			puts("ERROR: Negative value give to parameter that should be positive.\n  Correct the configuration file and rerun program");
			exit_program_if(true);
		}
		//IMAGE_HEIGHT = value.double_input;
		parameters.IMAGE_HEIGHT = value.double_input;
	}
	else if( strcmp (value.key, "IMAGE_THICKNESS") == 0 )
	{
		if( value.double_input < 0 )
		{
			puts("ERROR: Negative value give to parameter that should be positive.\n  Correct the configuration file and rerun program");
			exit_program_if(true);
		}
		//IMAGE_THICKNESS = value.double_input;
		parameters.IMAGE_THICKNESS = value.double_input;
	}
	else if( strcmp (value.key, "VOXEL_WIDTH") == 0 )
	{
		//VOXEL_WIDTH = value.double_input;
		parameters.VOXEL_WIDTH = value.double_input;
	}
	else if( strcmp (value.key, "VOXEL_HEIGHT") == 0 )
	{
		if( value.double_input < 0 )
		{
			puts("ERROR: Negative value give to parameter that should be positive.\n  Correct the configuration file and rerun program");
			exit_program_if(true);
		}
		//VOXEL_HEIGHT = value.double_input;
		parameters.VOXEL_HEIGHT = value.double_input;
	}
	else if( strcmp (value.key, "VOXEL_THICKNESS") == 0 )
	{
		if( value.double_input < 0 )
		{
			puts("ERROR: Negative value give to parameter that should be positive.\n  Correct the configuration file and rerun program");
			exit_program_if(true);
		}
		//VOXEL_THICKNESS = value.double_input;
		parameters.VOXEL_THICKNESS =  value.double_input;
		//SLICE_THICKNESS = value.double_input;
		//parameters.SLICE_THICKNESS =  value.double_input;
	}
	else if( strcmp (value.key, "SLICE_THICKNESS") == 0 )
	{
		if( value.double_input < 0 )
		{
			puts("ERROR: Negative value give to parameter that should be positive.\n  Correct the configuration file and rerun program");
			exit_program_if(true);
		}
		//SLICE_THICKNESS = value.double_input;
		parameters.SLICE_THICKNESS =  value.double_input;
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
		parameters.LAMBDA = value.double_input;
	}
	else if( strcmp (value.key, "ETA") == 0 )
	{
		if( value.double_input < 0 )
		{
			puts("ERROR: Negative value give to parameter that should be positive.\n  Correct the configuration file and rerun program");
			exit_program_if(true);
		}
		//ETA = value.double_input;
		parameters.ETA = value.double_input;
	}
	//------------------------------------------------------------------------------//
	//------------------------------------------------------------------------------//
	//------------------------------------------------------------------------------//
	else if( strcmp (value.key, "HULL_RSP_THRESHOLD") == 0 )
	{
		if( value.double_input < 0 )
		{
			puts("ERROR: Negative value give to parameter that should be positive.\n  Correct the configuration file and rerun program");
			exit_program_if(true);
		}
		//HULL_RSP_THRESHOLD = value.double_input;
		parameters.HULL_RSP_THRESHOLD = value.double_input;
	}
	else if( strcmp (value.key, "SC_THRESHOLD") == 0 )
	{
		//SC_THRESHOLD = value.double_input;
		parameters.SC_THRESHOLD = value.double_input;
	}
	else if( strcmp (value.key, "MSC_THRESHOLD") == 0 )
	{
		//MSC_THRESHOLD = value.double_input;
		parameters.MSC_THRESHOLD = value.double_input;
	}
	else if( strcmp (value.key, "SM_LOWER_THRESHOLD") == 0 )
	{
		//SM_LOWER_THRESHOLD = value.double_input;
		parameters.SM_LOWER_THRESHOLD = value.double_input;
	}
	else if( strcmp (value.key, "SM_UPPER_THRESHOLD") == 0 )
	{
		//SM_UPPER_THRESHOLD = value.double_input;
		parameters.SM_UPPER_THRESHOLD = value.double_input;
	}
	else if( strcmp (value.key, "SM_SCALE_THRESHOLD") == 0 )
	{
		if( value.double_input < 0 )
		{
			puts("ERROR: Negative value give to parameter that should be positive.\n  Correct the configuration file and rerun program");
			exit_program_if(true);
		}
		//SM_SCALE_THRESHOLD = value.double_input;
		parameters.SM_SCALE_THRESHOLD = value.double_input;
	}
	else
	{
		puts("ERROR: Procedure for setting this key is undefined");
		exit_program_if(true);
	}
}
void set_unsigned_integer_parameter( generic_IO_container &value )
{
	if( value.input_type_ID == DOUBLE )
		printf("converted to an integer and ");

	if( value.integer_input < 0 )
	{
		puts("ERROR: Negative value specified for an unsigned integer variable.\n  Correct the configuration file and rerun program");
		exit_program_if(true);
	}

	if( strcmp (value.key, "DATA_TYPE") == 0 )
	{			
		exit_program_if(print_scan_type(value.integer_input));
		// EXPERIMENTAL = 0, GEANT4 = 1, TOPAS = 2
		parameters.DATA_TYPE = static_cast<SCAN_TYPES>(value.integer_input);
	}
	else if( strcmp (value.key, "HULL_TYPE") == 0 )
	{
		exit_program_if(print_hull_type(value.integer_input));
		// IMPORT = 0, SC = 1, MSC = 2, SM = 3, FBP = 4
		parameters.HULL_TYPE = static_cast<HULL_TYPES>(value.integer_input);
	}
	else if( strcmp (value.key, "FBP_FILTER_TYPE") == 0 )
	{
		exit_program_if(print_filter_type(value.integer_input));
		// RAM_LAK = 0, SHEPP_LOGAN = 1, NONE = 2
		parameters.FBP_FILTER_TYPE = static_cast<FILTER_TYPES>(value.integer_input);
	}
	else if( strcmp (value.key, "X_0_TYPE") == 0 )
	{
		exit_program_if(print_x_0_type(value.integer_input));
		// IMPORT = 0, HULL = 1, FBP = 2, HYBRID = 3, ZEROS = 4
		parameters.X_0_TYPE = static_cast<X_0_TYPES>(value.integer_input);
	}
	else if( strcmp (value.key, "RECONSTRUCTION_METHOD") == 0 )
	{
		exit_program_if(print_recon_algorithm(value.integer_input));
		// ART = 0, DROP = 1, BIP = 2, SAP = 3, ROBUST1 = 4, ROBUST2 = 5 
		parameters.RECONSTRUCTION_METHOD = static_cast<RECON_ALGORITHMS>(value.integer_input);
	}
	//------------------------------------------------------------------------------//
	//------------------------------------------------------------------------------//
	//------------------------------------------------------------------------------//
	else if( strcmp (value.key, "NUM_SCANS") == 0 )
		parameters.NUM_SCANS = value.integer_input;
	else if( strcmp (value.key, "MAX_GPU_HISTORIES") == 0 )
		parameters.MAX_GPU_HISTORIES = value.integer_input;
	else if( strcmp (value.key, "MAX_CUTS_HISTORIES") == 0 )
		parameters.MAX_CUTS_HISTORIES = value.integer_input;
	else if( strcmp (value.key, "T_BINS") == 0 )
		parameters.T_BINS = value.integer_input;
	else if( strcmp (value.key, "V_BINS") == 0 )
		parameters.V_BINS = value.integer_input;
	else if( strcmp (value.key, "SIGMAS_2_KEEP") == 0 )
		parameters.SIGMAS_2_KEEP = value.integer_input;
	else if( strcmp (value.key, "COLUMNS") == 0 )
		parameters.COLUMNS = value.integer_input;
	else if( strcmp (value.key, "ROWS") == 0 )
		parameters.ROWS = value.integer_input;
	else if( strcmp (value.key, "SLICES") == 0 )
		parameters.SLICES = value.integer_input;
	//------------------------------------------------------------------------------//
	//------------------------------------------------------------------------------//
	//------------------------------------------------------------------------------//
	else if( strcmp (value.key, "ITERATIONS") == 0 )
		parameters.ITERATIONS = value.integer_input;
	else if( strcmp (value.key, "BLOCK_SIZE") == 0 )
		parameters.BLOCK_SIZE =  value.integer_input;
	else if( strcmp (value.key, "HULL_MED_FILTER_RADIUS") == 0 )
		parameters.HULL_MED_FILTER_RADIUS = value.integer_input;
	else if( strcmp (value.key, "FBP_MED_FILTER_RADIUS") == 0 )
		parameters.FBP_MED_FILTER_RADIUS = value.integer_input;
	else if( strcmp (value.key, "X_0_MED_FILTER_RADIUS") == 0 )
		parameters.X_0_MED_FILTER_RADIUS = value.integer_input;
	else if( strcmp (value.key, "X_K_MED_FILTER_RADIUS") == 0 )
		parameters.X_K_MED_FILTER_RADIUS = value.integer_input;
	else if( strcmp (value.key, "X_MED_FILTER_RADIUS") == 0 )
		parameters.X_MED_FILTER_RADIUS = value.integer_input;
	else if( strcmp (value.key, "HULL_AVG_FILTER_RADIUS") == 0 )
		parameters.HULL_AVG_FILTER_RADIUS = value.integer_input;
	else if( strcmp (value.key, "FBP_AVG_FILTER_RADIUS") == 0 )
		parameters.FBP_AVG_FILTER_RADIUS = value.integer_input;
	else if( strcmp (value.key, "X_0_AVG_FILTER_RADIUS") == 0 )
		parameters.X_0_AVG_FILTER_RADIUS = value.integer_input;
	else if( strcmp (value.key, "X_K_AVG_FILTER_RADIUS") == 0 )
		parameters.X_K_AVG_FILTER_RADIUS = value.integer_input;
	else if( strcmp (value.key, "X_AVG_FILTER_RADIUS") == 0 )
		parameters.X_AVG_FILTER_RADIUS = value.integer_input;
	//------------------------------------------------------------------------------//
	//------------------------------------------------------------------------------//
	//------------------------------------------------------------------------------//
	else if( strcmp (value.key, "MSC_DIFF_THRESH") == 0 )
		parameters.MSC_DIFF_THRESH = value.integer_input;
	else
	{
		puts("ERROR: Procedure for setting this key is undefined");
		exit_program_if(true);
	}
	printf("set to %d\n", value.integer_input);
}
void set_integer_parameter( generic_IO_container &value )
{
	if( value.input_type_ID == DOUBLE )
		printf("converted to an integer and ");
	
	if( strcmp (value.key, "PSI_SIGN") == 0 )
		parameters.PSI_SIGN = value.integer_input;
	else
	{
		puts("ERROR: Procedure for setting this key is undefined");
		exit_program_if(true);
	}
	printf("set to %d\n", value.integer_input);
}
void set_boolean_parameter( generic_IO_container &value )
{
	if( value.input_type_ID == INTEGER || value.input_type_ID == DOUBLE )
		printf("converted to a boolean and ");
	printf("set to %s\n", value.string_input );

	if( strcmp (value.key, "FBP_ON") == 0 )
		parameters.FBP_ON = value.boolean_input;
	else if( strcmp (value.key, "SC_ON") == 0 )
		parameters.SC_ON = value.boolean_input;
	else if( strcmp (value.key, "MSC_ON") == 0 )
		parameters.MSC_ON = value.boolean_input;
	else if( strcmp (value.key, "SM_ON") == 0 )
		parameters.SM_ON = value.boolean_input;
	else if( strcmp (value.key, "IMPORT_PREPROCESSING") == 0 )
		parameters.IMPORT_PREPROCESSING = value.boolean_input;
	else if( strcmp (value.key, "PERFORM_RECONSTRUCTION") == 0 )
		parameters.PERFORM_RECONSTRUCTION = value.boolean_input;
	else if( strcmp (value.key, "PREPROCESS_OVERWRITE_OK") == 0 )
		parameters.PREPROCESS_OVERWRITE_OK = value.boolean_input;
	else if( strcmp (value.key, "RECON_OVERWRITE_OK") == 0 )
		parameters.RECON_OVERWRITE_OK = value.boolean_input;
	else if( strcmp (value.key, "MLP_IN_LOOP") == 0 )
		parameters.MLP_IN_LOOP = value.boolean_input;
	else if( strcmp (value.key, "IMPORT_DATA_ITERATIVELY") == 0 )
		parameters.IMPORT_DATA_ITERATIVELY = value.boolean_input;
	//------------------------------------------------------------------------------//
	//------------------------------------------------------------------------------//
	//------------------------------------------------------------------------------//
	else if( strcmp (value.key, "AVG_FILTER_HULL") == 0 )
		parameters.AVG_FILTER_HULL = value.boolean_input;
	else if( strcmp (value.key, "AVG_FILTER_FBP") == 0 )
		parameters.AVG_FILTER_FBP = value.boolean_input;
	else if( strcmp (value.key, "AVG_FILTER_X_0") == 0 )
		parameters.AVG_FILTER_X_0 = value.boolean_input;
	else if( strcmp (value.key, "AVG_FILTER_X_K") == 0 )
		parameters.AVG_FILTER_X_K = value.boolean_input;
	else if( strcmp (value.key, "AVG_FILTER_X") == 0 )
		parameters.AVG_FILTER_X = value.boolean_input;
	else if( strcmp (value.key, "MEDIAN_FILTER_HULL") == 0 )
		parameters.MEDIAN_FILTER_HULL = value.boolean_input;
	else if( strcmp (value.key, "MEDIAN_FILTER_FBP") == 0 )
		parameters.MEDIAN_FILTER_FBP = value.boolean_input;
	else if( strcmp (value.key, "MEDIAN_FILTER_X_0") == 0 )
		parameters.MEDIAN_FILTER_X_0 = value.boolean_input;
	else if( strcmp (value.key, "MEDIAN_FILTER_X_K") == 0 )
		parameters.MEDIAN_FILTER_X_K = value.boolean_input;
	else if( strcmp (value.key, "MEDIAN_FILTER_X") == 0 )
		parameters.MEDIAN_FILTER_X = value.boolean_input;
	//------------------------------------------------------------------------------//
	//------------------------------------------------------------------------------//
	//------------------------------------------------------------------------------//
	else if( strcmp (value.key, "WRITE_MSC_COUNTS") == 0 )
		parameters.WRITE_MSC_COUNTS = value.boolean_input;
	else if( strcmp (value.key, "WRITE_SM_COUNTS") == 0 )
		parameters.WRITE_SM_COUNTS = value.boolean_input;
	else if( strcmp (value.key, "WRITE_X_FBP") == 0 )
		parameters.WRITE_X_FBP = value.boolean_input;
	else if( strcmp (value.key, "WRITE_FBP_HULL") == 0 )
		parameters.WRITE_FBP_HULL = value.boolean_input;
	else if( strcmp (value.key, "WRITE_AVG_FBP") == 0 )
		parameters.WRITE_AVG_FBP = value.boolean_input;
	else if( strcmp (value.key, "WRITE_MEDIAN_FBP") == 0 )
		parameters.WRITE_MEDIAN_FBP = value.boolean_input;
	else if( strcmp (value.key, "WRITE_BIN_WEPLS") == 0 )
		parameters.WRITE_BIN_WEPLS = value.boolean_input;
	else if( strcmp (value.key, "WRITE_WEPL_DISTS") == 0 )
		parameters.WRITE_WEPL_DISTS = value.boolean_input;
	else if( strcmp (value.key, "WRITE_SSD_ANGLES") == 0 )
		parameters.WRITE_SSD_ANGLES = value.boolean_input;
	//------------------------------------------------------------------------------//
	//------------------------------------------------------------------------------//
	//------------------------------------------------------------------------------//
	else if( strcmp (value.key, "STDOUT_2_DISK") == 0 )
		parameters.STDOUT_2_DISK = value.boolean_input;
	else if( strcmp (value.key, "USER_INPUT_REQUESTS_OFF") == 0 )
		parameters.USER_INPUT_REQUESTS_OFF = value.boolean_input;
	else if( strcmp (value.key, "ADD_DATA_LOG_ENTRY") == 0 )
		parameters.ADD_DATA_LOG_ENTRY = value.boolean_input;
	else if( strcmp (value.key, "DEBUG_TEXT_ON") == 0 )
		parameters.DEBUG_TEXT_ON = value.boolean_input;
	else if( strcmp (value.key, "EXIT_AFTER_BINNING") == 0 )
		parameters.EXIT_AFTER_BINNING = value.boolean_input;
	else if( strcmp (value.key, "EXIT_AFTER_HULLS") == 0 )
		parameters.EXIT_AFTER_HULLS = value.boolean_input;
	else if( strcmp (value.key, "EXIT_AFTER_CUTS") == 0 )
		parameters.EXIT_AFTER_CUTS = value.boolean_input;
	else if( strcmp (value.key, "EXIT_AFTER_SINOGRAM") == 0 )
		parameters.EXIT_AFTER_SINOGRAM = value.boolean_input;
	else if( strcmp (value.key, "EXIT_AFTER_FBP") == 0 )
		parameters.EXIT_AFTER_FBP = value.boolean_input;
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
	else if( key_is_unsigned_integer_parameter(value.key) )
		set_unsigned_integer_parameter(value);
	else if( key_is_integer_parameter(value.key) )
		set_integer_parameter(value);
	else if( key_is_boolean_parameter(value.key) )
		set_boolean_parameter(value);
	else
		puts("\nNo match for this key");
}
void set_file_extension( char file_extension[5], DISK_WRITE_MODE format )
{
	if( format == TEXT )
		sprintf( file_extension, ".txt" );
	else if( format == BINARY )
		sprintf( file_extension, ".bin" );
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
void set_dependent_parameters()
{
	parameters.GANTRY_ANGLES		= static_cast<uint>( 360 / parameters.GANTRY_ANGLE_INTERVAL );								// [#] Total number of projection angles
	parameters.NUM_FILES			= parameters.NUM_SCANS * parameters.GANTRY_ANGLES;							// [#] 1 file per gantry angle per translation
	parameters.T_BINS				= static_cast<uint>( parameters.SSD_T_SIZE / parameters.T_BIN_SIZE + 0.5 );				// [#] Number of bins (i.e. quantization levels) for t (lateral) direction 
	parameters.V_BINS				= static_cast<uint>( parameters.SSD_V_SIZE/ parameters.V_BIN_SIZE + 0.5 );				// [#] Number of bins (i.e. quantization levels) for v (vertical) direction 
	parameters.ANGULAR_BINS		= static_cast<uint>( 360 / parameters.ANGULAR_BIN_SIZE + 0.5 );							// [#] Number of bins (i.e. quantization levels) for path angle 
	parameters.NUM_BINS			= parameters.ANGULAR_BINS * parameters.T_BINS * parameters.V_BINS;		// [#] Total number of bins corresponding to possible 3-tuples [ANGULAR_BIN, T_BIN, V_BIN]
	parameters.RECON_CYL_HEIGHT	= parameters.SSD_V_SIZE - 1.0;												// [cm] Height of reconstruction cylinder
	parameters.RECON_CYLIAMETER	= 2 * parameters.RECON_CYL_RADIUS;											// [cm] Diameter of reconstruction cylinder
	parameters.SLICES				= static_cast<uint>( parameters.RECON_CYL_HEIGHT / parameters.SLICE_THICKNESS);			// [#] Number of voxels in the z direction (i.e., number of slices) of image
	parameters.NUM_VOXELS			= parameters.COLUMNS * parameters.ROWS * parameters.SLICES;				// [#] Total number of voxels (i.e. 3-tuples [column, row, slice]) in image
	parameters.IMAGE_WIDTH		= parameters.RECON_CYLIAMETER;												// [cm] Distance between left and right edges of each slice in image
	parameters.IMAGE_HEIGHT		= parameters.RECON_CYLIAMETER;						// [cm] Distance between top and bottom edges of each slice in image
	parameters.IMAGE_THICKNESS	= parameters.RECON_CYL_HEIGHT;						// [cm] Distance between bottom of bottom slice and top of the top slice of image
	parameters.VOXEL_WIDTH		= parameters.IMAGE_WIDTH / parameters.COLUMNS;		// [cm] Distance between left and right edges of each voxel in image
	parameters.VOXEL_HEIGHT		= parameters.IMAGE_HEIGHT / parameters.ROWS;		// [cm] Distance between top and bottom edges of each voxel in image
	parameters.VOXEL_THICKNESS	= parameters.IMAGE_THICKNESS / parameters.SLICES;	// [cm] Distance between top and bottom of each slice in image
	parameters.X_ZERO_COORDINATE	= -parameters.RECON_CYL_RADIUS;						// [cm] x-coordinate corresponding to left edge of 1st voxel (i.e. column) in image space
	parameters.Y_ZERO_COORDINATE	= parameters.RECON_CYL_RADIUS;						// [cm] y-coordinate corresponding to top edge of 1st voxel (i.e. row) in image space
	parameters.Z_ZERO_COORDINATE	= parameters.RECON_CYL_HEIGHT/2;						// [cm] z-coordinate corresponding to top edge of 1st voxel (i.e. slice) in image space
	parameters.RAM_LAK_TAU		= 2/ROOT_TWO * parameters.T_BIN_SIZE;					// Defines tau in Ram-Lak filter calculation, estimated from largest frequency in slice 
	/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
	/*---------------------------------------------------------- Memory allocation size for arrays (binning, image) -----------------------------------------------------------*/
	/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
	SIZE_BINS_CHAR		= ( parameters.NUM_BINS   * sizeof(char)	);		// Amount of memory required for a character array used for binning
	SIZE_BINS_BOOL		= ( parameters.NUM_BINS   * sizeof(bool)	);		// Amount of memory required for a boolean array used for binning
	SIZE_BINS_INT		= ( parameters.NUM_BINS   * sizeof(int)		);		// Amount of memory required for a integer array used for binning
	SIZE_BINS_UINT		= ( parameters.NUM_BINS   * sizeof(uint)	);		// Amount of memory required for a integer array used for binning
	SIZE_BINS_FLOAT		= ( parameters.NUM_BINS	  * sizeof(float)	);		// Amount of memory required for a single precision floating point array used for binning
	SIZE_BINS_DOUBLE	= ( parameters.NUM_BINS	  * sizeof(double)	);		// Amount of memory required for a double precision floating point array used for binning
	SIZE_IMAGE_CHAR		= ( parameters.NUM_VOXELS * sizeof(char)	);		// Amount of memory required for a character array used for binning
	SIZE_IMAGE_BOOL		= ( parameters.NUM_VOXELS * sizeof(bool)	);		// Amount of memory required for a boolean array used for binning
	SIZE_IMAGE_INT		= ( parameters.NUM_VOXELS * sizeof(int)		);		// Amount of memory required for a integer array used for binning
	SIZE_IMAGE_UINT		= ( parameters.NUM_VOXELS * sizeof(uint)	);		// Amount of memory required for a integer array used for binning
	SIZE_IMAGE_FLOAT	= ( parameters.NUM_VOXELS * sizeof(float)	);		// Amount of memory required for a single precision floating point array used for binning
	SIZE_IMAGE_DOUBLE	= ( parameters.NUM_VOXELS * sizeof(double)	);		// Amount of memory required for a double precision floating point array used for binning
	/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
	/*-------------------------------------------------------------- Iterative Image Reconstruction Parameters ----------------------------------------------------------------*/
	/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
	parameters.VOXEL_STEP_SIZE		= ( parameters.VOXEL_WIDTH / 2 );						// [cm] Length of the step taken along the path, i.e. change in depth per step for
	parameters.MLP_U_STEP				= ( parameters.VOXEL_WIDTH / 2);						// Size of the step taken along u direction during MLP; depth difference between successive MLP points
	parameters.CONSTANT_CHORD_NORM	= pow(parameters.VOXEL_WIDTH, 2.0);
	parameters.CONSTANT_LAMBDA_SCALE	= parameters.VOXEL_WIDTH * parameters.LAMBDA;
}
void set_IO_file_extensions()
{
	set_file_extension( PROJECTION_DATA_FILE_EXTENSION, PROJECTION_DATA_WRITE_MODE );	// File extension of the files containing the projection data (tracker/WEPL/gantry angle) used as input to preprocessing
	set_file_extension( RADIOGRAPHS_FILE_EXTENSION, RADIOGRAPHS_WRITE_MODE );			// File extension of the files containing the radiograph images from each projection angle before/after performing cuts
	set_file_extension( WEPL_DISTS_FILE_EXTENSION, WEPL_DISTS_WRITE_MODE );				// File extension of the files containing the WEPL distribution images from each projection angle before/after performing cuts
	set_file_extension( HULL_FILE_EXTENSION, HULL_WRITE_MODE );							// File extension of the file containing the SC, MSC, SM, or FBP hull image as specified by the settings.cfg file 
	set_file_extension( FBP_FILE_EXTENSION, FBP_WRITE_MODE );							// File extension of the file containing the FBP image
	set_file_extension( FBP_MEDIANS_FILE_EXTENSION, FBP_MEDIANS_WRITE_MODE );			// File extension of the file containing the median filtered FBP images
	set_file_extension( X_0_FILE_EXTENSION, X_0_WRITE_MODE );							// File extension of the file containing the FBP, hull, or FBP/hull hybrid initial iterate image as specified by the settings.cfg file
	set_file_extension( MLP_FILE_EXTENSION, MLP_WRITE_MODE );							// File extension of the file containing the MLP path data
	set_file_extension( WEPL_FILE_EXTENSION, WEPL_WRITE_MODE );							// File extension of the file containing the WEPL measurements for each MLP path
	set_file_extension( HISTORIES_FILE_EXTENSION, HISTORIES_WRITE_MODE );				// File extension of the file containing the x/y/z hull entry/exit coordinates/angles, x/y/z hull entry voxels, gantry angle, and bin # for each reconstruction history
	set_file_extension( VOXELS_PER_PATH_FILE_EXTENSION, VOXELS_PER_PATH_WRITE_MODE );	// File extension of the file containing the # of intersected voxels per MLP path
	set_file_extension( AVG_CHORDS_FILE_EXTENSION, AVG_CHORDS_WRITE_MODE );				// File extension of the file containing the effective (average) chord length for each MLP path
	set_file_extension( X_FILE_EXTENSION, X_WRITE_MODE );								// File extension of the file containing the reconstructed images after each of the N iterations (e.g., x_1, x_2, x_3, ..., x_N)

}
void set_IO_directories()
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

		puts(PROJECTION_DATA_DIR);
		puts(PATH_2_PCT_DATA_DIR);
	}
	else if( PROJECTION_DATA_DIR_CONSTRUCTABLE )
	{		
		size_t length = strlen(PATH_2_PCT_DATA_DIR) + strlen(PCT_DATA_DIR_NAME) + strlen(OBJECT) + strlen(SCAN_TYPE) + strlen(RUN_DATE) + strlen(RUN_NUMBER) + strlen(PROJECTION_DATA_DIR_NAME) + strlen(PROJECTION_DATA_DATE);
		PROJECTION_DATA_DIR = (char*) calloc( length + 1, sizeof(char) ); 
		sprintf(PROJECTION_DATA_DIR,"%s\\%s\\%s\\%s\\%s\\%s\\%s\\%s", PATH_2_PCT_DATA_DIR, PCT_DATA_DIR_NAME, OBJECT, SCAN_TYPE, RUN_DATE, RUN_NUMBER, PROJECTION_DATA_DIR_NAME, PROJECTION_DATA_DATE );		
		puts("Construct\n");
		puts(PROJECTION_DATA_DIR);
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
		sprintf(PREPROCESSING_DIR,"%s\\%s\\%s", PROJECTION_DATA_DIR, RECONSTRUCTION_DIR_NAME, PREPROCESS_DATE);		
	}		
	if( parameters.PREPROCESS_OVERWRITE_OK )
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
		sprintf(RECONSTRUCTION_DIR,"%s\\%s\\%s", PREPROCESSING_DIR, PCT_IMAGES_DIR_NAME, RECONSTRUCTION_DATE);		
	}
	if( parameters.RECON_OVERWRITE_OK )
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
	print_section_exit("Finished setting paths to I/O data directories and creating associated folders", "====>" );
}
void set_IO_filenames()
{
	//----------------------------------------------------------------------------------------------------------------------------------------------------------//
	//----------------------------------------------- Set file names for preprocessing data generated as output ------------------------------------------------//
	//----------------------------------------------------------------------------------------------------------------------------------------------------------//
	print_section_header( "Preprocessing and reconstruction data/images generated will be written to and/or read from the following paths", '*' );

	HULL_FILENAME = (char*) calloc( strlen(HULL_BASENAME) + strlen(HULL_FILE_EXTENSION), sizeof(char) );
	HULL_MEDIAN_2D_FILENAME = (char*) calloc( strlen(HULL_BASENAME) + strlen(MEDIAN_FILTER_2D_POSTFIX) + 1 + strlen(HULL_FILE_EXTENSION) + 1, sizeof(char) );
	HULL_MEDIAN_3D_FILENAME = (char*) calloc( strlen(HULL_BASENAME) + strlen(MEDIAN_FILTER_3D_POSTFIX) + 1 + strlen(HULL_FILE_EXTENSION) + 1, sizeof(char) );
	HULL_AVG_2D_FILENAME  = (char*) calloc( strlen(HULL_BASENAME) + strlen(AVERAGE_FILTER_2D_POSTFIX) + 1 + strlen(HULL_FILE_EXTENSION) + 1, sizeof(char) );
	HULL_AVG_3D_FILENAME  = (char*) calloc( strlen(HULL_BASENAME) + strlen(AVERAGE_FILTER_3D_POSTFIX) + 1 + strlen(HULL_FILE_EXTENSION) + 1, sizeof(char) );
	HULL_COMBO_2D_FILENAME  = (char*) calloc( strlen(HULL_BASENAME) + 1 + strlen(MEDIAN_FILTER_2D_POSTFIX) + 1 + 1 + strlen(AVERAGE_FILTER_2D_POSTFIX) + 1 + strlen(HULL_FILE_EXTENSION), sizeof(char) );
	HULL_COMBO_3D_FILENAME  = (char*) calloc( strlen(HULL_BASENAME) + 1 + strlen(MEDIAN_FILTER_3D_POSTFIX) + 1 + 1 + strlen(AVERAGE_FILTER_3D_POSTFIX) + strlen(HULL_FILE_EXTENSION), sizeof(char) );
	
	FBP_FILENAME = (char*) calloc( strlen(FBP_BASENAME) + strlen(FBP_FILE_EXTENSION), sizeof(char) );
	FBP_MEDIAN_2D_FILENAME = (char*) calloc( strlen(FBP_BASENAME) + strlen(MEDIAN_FILTER_2D_POSTFIX) + 1 + strlen(FBP_FILE_EXTENSION) + 1, sizeof(char) );
	FBP_MEDIAN_3D_FILENAME = (char*) calloc( strlen(FBP_BASENAME) + strlen(MEDIAN_FILTER_3D_POSTFIX) + 1 + strlen(FBP_FILE_EXTENSION) + 1, sizeof(char) );
	FBP_AVG_2D_FILENAME  = (char*) calloc( strlen(FBP_BASENAME) + strlen(AVERAGE_FILTER_2D_POSTFIX) + 1 + strlen(FBP_FILE_EXTENSION) + 1, sizeof(char) );
	FBP_AVG_3D_FILENAME  = (char*) calloc( strlen(FBP_BASENAME) + strlen(AVERAGE_FILTER_3D_POSTFIX) + 1 + strlen(FBP_FILE_EXTENSION) + 1, sizeof(char) );
	FBP_COMBO_2D_FILENAME  = (char*) calloc( strlen(FBP_BASENAME) + 1 + strlen(MEDIAN_FILTER_2D_POSTFIX) + 1 + 1 + strlen(AVERAGE_FILTER_2D_POSTFIX) + 1 + strlen(FBP_FILE_EXTENSION), sizeof(char) );
	FBP_COMBO_3D_FILENAME  = (char*) calloc( strlen(FBP_BASENAME) + 1 + strlen(MEDIAN_FILTER_3D_POSTFIX) + 1 + 1 + strlen(AVERAGE_FILTER_3D_POSTFIX) + 1 + strlen(FBP_FILE_EXTENSION), sizeof(char) );
	
	X_0_FILENAME = (char*) calloc( strlen(X_0_BASENAME) + strlen(X_0_FILE_EXTENSION), sizeof(char) );
	X_0_MEDIAN_2D_FILENAME = (char*) calloc( strlen(X_0_BASENAME) + strlen(MEDIAN_FILTER_2D_POSTFIX) + 1 + strlen(X_0_FILE_EXTENSION) + 1, sizeof(char) ); 
	X_0_MEDIAN_3D_FILENAME = (char*) calloc( strlen(X_0_BASENAME) + strlen(MEDIAN_FILTER_3D_POSTFIX) + 1 + strlen(X_0_FILE_EXTENSION) + 1, sizeof(char) ); 
	X_0_AVG_2D_FILENAME   = (char*) calloc( strlen(X_0_BASENAME) + strlen(AVERAGE_FILTER_2D_POSTFIX) + 1 + strlen(X_0_FILE_EXTENSION) + 1, sizeof(char) );
	X_0_AVG_3D_FILENAME   = (char*) calloc( strlen(X_0_BASENAME) + strlen(AVERAGE_FILTER_3D_POSTFIX) + 1 + strlen(X_0_FILE_EXTENSION) + 1, sizeof(char) );
	X_0_COMBO_2D_FILENAME   = (char*) calloc( strlen(X_0_BASENAME) + 1 + strlen(MEDIAN_FILTER_2D_POSTFIX) + 1 + 1 + strlen(AVERAGE_FILTER_2D_POSTFIX) + 1 + strlen(X_0_FILE_EXTENSION), sizeof(char) );
	X_0_COMBO_3D_FILENAME   = (char*) calloc( strlen(X_0_BASENAME) + 1 + strlen(MEDIAN_FILTER_3D_POSTFIX) + 1 + 1 + strlen(AVERAGE_FILTER_3D_POSTFIX) + 1 + strlen(X_0_FILE_EXTENSION), sizeof(char) );
	
	X_FILENAME_BASE = (char*) calloc( strlen(X_BASENAME), sizeof(char) );
	X_MEDIAN_2D_FILENAME_BASE = (char*) calloc( strlen(X_BASENAME) + strlen(MEDIAN_FILTER_2D_POSTFIX) + 1, sizeof(char) );
	X_MEDIAN_3D_FILENAME_BASE = (char*) calloc( strlen(X_BASENAME) + strlen(MEDIAN_FILTER_3D_POSTFIX) + 1, sizeof(char) );	
	X_AVG_2D_FILENAME_BASE  = (char*) calloc( strlen(X_BASENAME) + strlen(AVERAGE_FILTER_2D_POSTFIX) + 1, sizeof(char) );
	X_AVG_3D_FILENAME_BASE  = (char*) calloc( strlen(X_BASENAME) + strlen(AVERAGE_FILTER_3D_POSTFIX) + 1, sizeof(char) );
	X_COMBO_2D_FILENAME_BASE  = (char*) calloc( strlen(X_BASENAME) + 1 + strlen(MEDIAN_FILTER_2D_POSTFIX) + 1 + 1 + strlen(AVERAGE_FILTER_2D_POSTFIX) + 1, sizeof(char) );
	X_COMBO_3D_FILENAME_BASE  = (char*) calloc( strlen(X_BASENAME) + 1 + strlen(MEDIAN_FILTER_3D_POSTFIX) + 1 + 1 + strlen(AVERAGE_FILTER_3D_POSTFIX) + 1, sizeof(char) );

	MLP_FILENAME = (char*) calloc( strlen(MLP_BASENAME) + strlen(MLP_FILE_EXTENSION), sizeof(char) );
	WEPL_FILENAME = (char*) calloc( strlen(WEPL_BASENAME) + strlen(WEPL_FILE_EXTENSION), sizeof(char) );
	HISTORIES_FILENAME = (char*) calloc( strlen(HISTORIES_BASENAME) + strlen(HISTORIES_FILE_EXTENSION), sizeof(char) );
	VOXELS_PER_PATH_FILENAME = (char*) calloc( strlen(VOXELS_PER_PATH_BASENAME) + strlen(VOXELS_PER_PATH_FILE_EXTENSION), sizeof(char) );
	AVG_CHORDS_FILENAME = (char*) calloc( strlen(VOXELS_PER_PATH_BASENAME) + strlen(AVG_CHORDS_FILE_EXTENSION), sizeof(char) );
	
	sprintf( HULL_FILENAME,"%s%s", HULL_BASENAME, HULL_FILE_EXTENSION );
	sprintf( HULL_AVG_2D_FILENAME, "%s_%s%d%s", HULL_BASENAME, AVERAGE_FILTER_2D_POSTFIX, 2 * parameters.HULL_AVG_FILTER_RADIUS + 1, HULL_FILE_EXTENSION);
	sprintf( HULL_MEDIAN_2D_FILENAME, "%s_%s%d%s", HULL_BASENAME, MEDIAN_FILTER_2D_POSTFIX, 2 * parameters.HULL_MED_FILTER_RADIUS + 1, HULL_FILE_EXTENSION);
	sprintf( HULL_MEDIAN_3D_FILENAME, "%s_%s%d%s", HULL_BASENAME, MEDIAN_FILTER_3D_POSTFIX, 2 * parameters.HULL_MED_FILTER_RADIUS + 1, HULL_FILE_EXTENSION);
	sprintf( HULL_AVG_3D_FILENAME, "%s_%s%d%s", HULL_BASENAME, AVERAGE_FILTER_3D_POSTFIX, 2 * parameters.HULL_AVG_FILTER_RADIUS + 1, HULL_FILE_EXTENSION);
	sprintf( HULL_COMBO_2D_FILENAME, "%s_%s%d_%s%d%s", HULL_BASENAME, MEDIAN_FILTER_2D_POSTFIX, 2 * parameters.HULL_MED_FILTER_RADIUS + 1, AVERAGE_FILTER_2D_POSTFIX, 2 * parameters.HULL_AVG_FILTER_RADIUS + 1, HULL_FILE_EXTENSION);
	sprintf( HULL_COMBO_3D_FILENAME, "%s_%s%d_%s%d%s", HULL_BASENAME, MEDIAN_FILTER_3D_POSTFIX, 2 * parameters.HULL_MED_FILTER_RADIUS + 1, AVERAGE_FILTER_3D_POSTFIX, 2 * parameters.HULL_AVG_FILTER_RADIUS + 1, HULL_FILE_EXTENSION);
	
	sprintf( FBP_FILENAME,"%s%s", FBP_BASENAME, FBP_FILE_EXTENSION );
	sprintf( FBP_AVG_2D_FILENAME, "%s_%s%d%s", FBP_BASENAME, AVERAGE_FILTER_2D_POSTFIX, 2 * parameters.FBP_AVG_FILTER_RADIUS + 1, FBP_FILE_EXTENSION);
	sprintf( FBP_MEDIAN_2D_FILENAME, "%s_%s%d%s", FBP_BASENAME, MEDIAN_FILTER_2D_POSTFIX, 2 * parameters.FBP_MED_FILTER_RADIUS + 1, FBP_FILE_EXTENSION);
	sprintf( FBP_MEDIAN_3D_FILENAME, "%s_%s%d%s", FBP_BASENAME, MEDIAN_FILTER_3D_POSTFIX, 2 * parameters.FBP_MED_FILTER_RADIUS + 1, FBP_FILE_EXTENSION);
	sprintf( FBP_AVG_3D_FILENAME, "%s_%s%d%s", FBP_BASENAME, AVERAGE_FILTER_3D_POSTFIX, 2 * parameters.FBP_AVG_FILTER_RADIUS + 1, FBP_FILE_EXTENSION);
	sprintf( FBP_COMBO_2D_FILENAME, "%s_%s%d_%s%d%s", FBP_BASENAME, MEDIAN_FILTER_2D_POSTFIX, 2 * parameters.FBP_MED_FILTER_RADIUS + 1, AVERAGE_FILTER_2D_POSTFIX, 2 * parameters.FBP_AVG_FILTER_RADIUS + 1, FBP_FILE_EXTENSION);
	sprintf( FBP_COMBO_3D_FILENAME, "%s_%s%d_%s%d%s", FBP_BASENAME, MEDIAN_FILTER_3D_POSTFIX, 2 * parameters.FBP_MED_FILTER_RADIUS + 1, AVERAGE_FILTER_3D_POSTFIX, 2 * parameters.FBP_AVG_FILTER_RADIUS + 1, FBP_FILE_EXTENSION);
	
	sprintf( X_0_FILENAME, "%s%s", X_0_BASENAME, X_0_FILE_EXTENSION );
	sprintf( X_0_AVG_2D_FILENAME, "%s_%s%d%s", X_0_BASENAME, AVERAGE_FILTER_2D_POSTFIX, 2 * parameters.X_0_AVG_FILTER_RADIUS + 1, X_0_FILE_EXTENSION);
	sprintf( X_0_MEDIAN_2D_FILENAME, "%s_%s%d%s", X_0_BASENAME, MEDIAN_FILTER_2D_POSTFIX, 2 * parameters.X_0_MED_FILTER_RADIUS + 1, X_0_FILE_EXTENSION);
	sprintf( X_0_MEDIAN_3D_FILENAME, "%s_%s%d%s", X_0_BASENAME, MEDIAN_FILTER_3D_POSTFIX, 2 * parameters.X_0_MED_FILTER_RADIUS + 1, X_0_FILE_EXTENSION);
	sprintf( X_0_AVG_3D_FILENAME, "%s_%s%d%s", X_0_BASENAME, AVERAGE_FILTER_3D_POSTFIX, 2 * parameters.X_0_AVG_FILTER_RADIUS + 1, X_0_FILE_EXTENSION);
	sprintf( X_0_COMBO_2D_FILENAME, "%s_%s%d_%s%d%s", X_0_BASENAME, MEDIAN_FILTER_2D_POSTFIX, 2 * parameters.X_0_MED_FILTER_RADIUS + 1, AVERAGE_FILTER_2D_POSTFIX, 2 * parameters.X_0_AVG_FILTER_RADIUS + 1, X_0_FILE_EXTENSION);
	sprintf( X_0_COMBO_3D_FILENAME, "%s_%s%d_%s%d%s", X_0_BASENAME, MEDIAN_FILTER_3D_POSTFIX, 2 * parameters.X_0_MED_FILTER_RADIUS + 1, AVERAGE_FILTER_3D_POSTFIX, 2 * parameters.X_0_AVG_FILTER_RADIUS + 1, X_0_FILE_EXTENSION);
	
	sprintf( X_FILENAME_BASE,"%s", X_BASENAME);
	sprintf( X_AVG_2D_FILENAME_BASE, "%s_%s%d", X_BASENAME, AVERAGE_FILTER_2D_POSTFIX, 2 * parameters.X_AVG_FILTER_RADIUS + 1);
	sprintf( X_MEDIAN_2D_FILENAME_BASE, "%s_%s%d", X_BASENAME, MEDIAN_FILTER_2D_POSTFIX, 2 * parameters.X_MED_FILTER_RADIUS + 1);
	sprintf( X_MEDIAN_3D_FILENAME_BASE, "%s_%s%d", X_BASENAME, MEDIAN_FILTER_3D_POSTFIX, 2 * parameters.X_MED_FILTER_RADIUS + 1);
	sprintf( X_AVG_3D_FILENAME_BASE, "%s_%s%d", X_BASENAME, AVERAGE_FILTER_3D_POSTFIX, 2 * parameters.X_AVG_FILTER_RADIUS + 1);
	sprintf( X_COMBO_2D_FILENAME_BASE, "%s_%s%d_%s%d", X_BASENAME, MEDIAN_FILTER_2D_POSTFIX, 2 * parameters.X_MED_FILTER_RADIUS + 1, AVERAGE_FILTER_2D_POSTFIX, 2 * parameters.X_AVG_FILTER_RADIUS + 1);
	sprintf( X_COMBO_3D_FILENAME_BASE, "%s_%s%d_%s%d", X_BASENAME, MEDIAN_FILTER_3D_POSTFIX, 2 * parameters.X_MED_FILTER_RADIUS + 1, AVERAGE_FILTER_3D_POSTFIX, 2 * parameters.X_AVG_FILTER_RADIUS + 1);
	
	sprintf( MLP_FILENAME,"%s%s", MLP_BASENAME, MLP_FILE_EXTENSION );
	sprintf( WEPL_FILENAME,"%s%s", WEPL_BASENAME, WEPL_FILE_EXTENSION );
	sprintf( HISTORIES_FILENAME,"%s%s", HISTORIES_BASENAME, HISTORIES_FILE_EXTENSION);
	sprintf( VOXELS_PER_PATH_FILENAME,"%s%s", VOXELS_PER_PATH_BASENAME, VOXELS_PER_PATH_FILE_EXTENSION );	
	sprintf( AVG_CHORDS_FILENAME,"%s%s", AVG_CHORDS_BASENAME, AVG_CHORDS_FILE_EXTENSION );	
	
	printf("HULL_FILENAME = %s\n\n", HULL_FILENAME );	
	printf("HULL_MEDIAN_2D_FILENAME = %s\n\n", HULL_MEDIAN_2D_FILENAME );	
	printf("HULL_MEDIAN_3D_FILENAME = %s\n\n", HULL_MEDIAN_3D_FILENAME );	
	printf("HULL_AVG_2D_FILENAME = %s\n\n", HULL_AVG_2D_FILENAME );	
	printf("HULL_AVG_3D_FILENAME = %s\n\n", HULL_AVG_3D_FILENAME );	
	printf("HULL_COMBO_2D_FILENAME = %s\n\n", HULL_COMBO_2D_FILENAME );	
	printf("HULL_COMBO_3D_FILENAME = %s\n\n", HULL_COMBO_3D_FILENAME );	
	
	printf("FBP_FILENAME = %s\n\n", FBP_FILENAME );
	printf("FBP_MEDIAN_2D_FILENAME = %s\n\n", FBP_MEDIAN_2D_FILENAME );	
	printf("FBP_MEDIAN_3D_FILENAME = %s\n\n", FBP_MEDIAN_3D_FILENAME );
	printf("FBP_AVG_2D_FILENAME = %s\n\n", FBP_AVG_2D_FILENAME );
	printf("FBP_AVG_3D_FILENAME = %s\n\n", FBP_AVG_3D_FILENAME );
	printf("FBP_COMBO_2D_FILENAME = %s\n\n", FBP_COMBO_2D_FILENAME );
	printf("FBP_COMBO_3D_FILENAME = %s\n\n", FBP_COMBO_3D_FILENAME );
	
	printf("X_0_FILENAME = %s\n\n", X_0_FILENAME );	
	printf("X_0_MEDIAN_2D_FILENAME = %s\n\n", X_0_MEDIAN_2D_FILENAME );	
	printf("X_0_MEDIAN_3D_FILENAME = %s\n\n", X_0_MEDIAN_3D_FILENAME );	
	printf("X_0_AVG_2D_FILENAME = %s\n\n", X_0_AVG_2D_FILENAME );	
	printf("X_0_AVG_3D_FILENAME = %s\n\n", X_0_AVG_3D_FILENAME );	
	printf("X_0_COMBO_2D_FILENAME = %s\n\n", X_0_COMBO_2D_FILENAME );	
	printf("X_0_COMBO_3D_FILENAME = %s\n\n", X_0_COMBO_3D_FILENAME );	
	
	printf("X_FILENAME = %s\n\n", X_FILENAME_BASE );
	printf("X_MEDIAN_2D_FILENAME = %s\n\n", X_MEDIAN_2D_FILENAME_BASE );
	printf("X_MEDIAN_3D_FILENAME = %s\n\n", X_MEDIAN_3D_FILENAME_BASE );
	printf("X_AVG_2D_FILENAME = %s\n\n", X_AVG_2D_FILENAME_BASE );
	printf("X_AVG_3D_FILENAME = %s\n\n", X_AVG_3D_FILENAME_BASE );
	printf("X_COMBO_2D_FILENAME = %s\n\n", X_COMBO_2D_FILENAME_BASE );
	printf("X_COMBO_3D_FILENAME = %s\n\n", X_COMBO_3D_FILENAME_BASE );

	printf("MLP_FILENAME = %s\n\n", MLP_FILENAME );	
	printf("WEPL_FILENAME = %s\n\n", WEPL_FILENAME );	
	printf("HISTORIES_FILENAME = %s\n", HISTORIES_FILENAME );	
	printf("VOXELS_PER_PATH_FILENAME = %s\n\n", VOXELS_PER_PATH_FILENAME );	
	printf("AVG_CHORDS_FILENAME = %s\n\n", AVG_CHORDS_FILENAME );	
	
	print_section_exit("Finished setting file names of input/output data files", "====>" );
}
void set_IO_filepaths()
{
	//----------------------------------------------------------------------------------------------------------------------------------------------------------//
	//----------------------------- Set paths to preprocessing and reconstruction data using associated directory and file names -------------------------------//
	//----------------------------------------------------------------------------------------------------------------------------------------------------------//	
	print_section_header( "File names of preprocessing data generated as output", '*' );	

	HULL_PATH = (char*) calloc( strlen(PREPROCESSING_DIR) + strlen(HULL_FILENAME) + 1, sizeof(char) );
	HULL_MEDIAN_2D_PATH = (char*) calloc( strlen(PREPROCESSING_DIR) + strlen(HULL_MEDIAN_2D_FILENAME) + 1, sizeof(char) );
	HULL_MEDIAN_3D_PATH = (char*) calloc( strlen(PREPROCESSING_DIR) + strlen(HULL_MEDIAN_3D_FILENAME) + 1, sizeof(char) );
	HULL_AVG_2D_PATH = (char*) calloc( strlen(PREPROCESSING_DIR) + strlen(HULL_AVG_2D_FILENAME) + 1, sizeof(char) );
	HULL_AVG_3D_PATH = (char*) calloc( strlen(PREPROCESSING_DIR) + strlen(HULL_AVG_3D_FILENAME) + 1, sizeof(char) );
	HULL_COMBO_2D_PATH = (char*) calloc( strlen(PREPROCESSING_DIR) + strlen(HULL_COMBO_2D_FILENAME) + 1, sizeof(char) );
	HULL_COMBO_3D_PATH = (char*) calloc( strlen(PREPROCESSING_DIR) + strlen(HULL_COMBO_3D_FILENAME) + 1, sizeof(char) );
			
	FBP_PATH = (char*) calloc( strlen(PREPROCESSING_DIR) + strlen(FBP_FILENAME) + 1, sizeof(char) );
	FBP_MEDIAN_2D_PATH = (char*) calloc( strlen(PREPROCESSING_DIR) + strlen(FBP_MEDIAN_2D_FILENAME) + 1, sizeof(char) );
	FBP_MEDIAN_3D_PATH = (char*) calloc( strlen(PREPROCESSING_DIR) + strlen(FBP_MEDIAN_3D_FILENAME) + 1, sizeof(char) );
	FBP_AVG_2D_PATH = (char*) calloc( strlen(PREPROCESSING_DIR) + strlen(FBP_AVG_2D_FILENAME) + 1, sizeof(char) );
	FBP_AVG_3D_PATH = (char*) calloc( strlen(PREPROCESSING_DIR) + strlen(FBP_AVG_3D_FILENAME) + 1, sizeof(char) );
	FBP_COMBO_2D_PATH = (char*) calloc( strlen(PREPROCESSING_DIR) + strlen(FBP_COMBO_2D_FILENAME) + 1, sizeof(char) );
	FBP_COMBO_3D_PATH = (char*) calloc( strlen(PREPROCESSING_DIR) + strlen(FBP_COMBO_3D_FILENAME) + 1, sizeof(char) );
	
	X_0_PATH = (char*) calloc( strlen(PREPROCESSING_DIR) + strlen(X_0_FILENAME) + 1, sizeof(char) );	
	X_0_MEDIAN_2D_PATH = (char*) calloc( strlen(PREPROCESSING_DIR) + strlen(X_0_MEDIAN_2D_FILENAME) + 1, sizeof(char) );
	X_0_MEDIAN_3D_PATH = (char*) calloc( strlen(PREPROCESSING_DIR) + strlen(X_0_MEDIAN_3D_FILENAME) + 1, sizeof(char) );
	X_0_AVG_2D_PATH = (char*) calloc( strlen(PREPROCESSING_DIR) + strlen(X_0_AVG_2D_FILENAME) + 1, sizeof(char) );
	X_0_AVG_3D_PATH = (char*) calloc( strlen(PREPROCESSING_DIR) + strlen(X_0_AVG_3D_FILENAME) + 1, sizeof(char) );
	X_0_COMBO_2D_PATH = (char*) calloc( strlen(PREPROCESSING_DIR) + strlen(X_0_COMBO_2D_FILENAME) + 1, sizeof(char) );
	X_0_COMBO_3D_PATH = (char*) calloc( strlen(PREPROCESSING_DIR) + strlen(X_0_COMBO_3D_FILENAME) + 1, sizeof(char) );
	
	X_PATH_BASE = (char*) calloc( strlen(RECONSTRUCTION_DIR) + strlen(X_FILENAME_BASE) + 1, sizeof(char) );
	X_MEDIAN_2D_PATH_BASE = (char*) calloc( strlen(RECONSTRUCTION_DIR) + strlen(X_MEDIAN_2D_FILENAME_BASE) + 1, sizeof(char) );
	X_MEDIAN_3D_PATH_BASE = (char*) calloc( strlen(RECONSTRUCTION_DIR) + strlen(X_MEDIAN_3D_FILENAME_BASE) + 1, sizeof(char) );
	X_AVG_2D_PATH_BASE = (char*) calloc( strlen(RECONSTRUCTION_DIR) + strlen(X_AVG_2D_FILENAME_BASE) + 1, sizeof(char) );
	X_AVG_3D_PATH_BASE = (char*) calloc( strlen(RECONSTRUCTION_DIR) + strlen(X_AVG_3D_FILENAME_BASE) + 1, sizeof(char) );
	X_COMBO_2D_PATH_BASE = (char*) calloc( strlen(RECONSTRUCTION_DIR) + strlen(X_COMBO_2D_FILENAME_BASE) + 1, sizeof(char) );
	X_COMBO_3D_PATH_BASE = (char*) calloc( strlen(RECONSTRUCTION_DIR) + strlen(X_COMBO_3D_FILENAME_BASE) + 1, sizeof(char) );

	MLP_PATH = (char*) calloc( strlen(PREPROCESSING_DIR) + strlen(MLP_FILENAME) + 1, sizeof(char) );
	WEPL_PATH = (char*) calloc( strlen(PREPROCESSING_DIR) + strlen(WEPL_FILENAME) + 1, sizeof(char) );
	HISTORIES_PATH = (char*) calloc( strlen(PREPROCESSING_DIR) + strlen(HISTORIES_FILENAME) + 1, sizeof(char) );
	VOXELS_PER_PATH_PATH = (char*) calloc( strlen(PREPROCESSING_DIR) + strlen(VOXELS_PER_PATH_FILENAME) + 1, sizeof(char) );
	AVG_CHORDS_PATH = (char*) calloc( strlen(PREPROCESSING_DIR) + strlen(AVG_CHORDS_FILENAME) + 1, sizeof(char) );
	
	sprintf( HULL_PATH,"%s\\%s", PREPROCESSING_DIR, HULL_FILENAME );
	sprintf( HULL_MEDIAN_2D_PATH,"%s\\%s", PREPROCESSING_DIR, HULL_MEDIAN_2D_FILENAME );
	sprintf( HULL_MEDIAN_3D_PATH,"%s\\%s", PREPROCESSING_DIR, HULL_MEDIAN_3D_FILENAME );
	sprintf( HULL_AVG_2D_PATH,"%s\\%s", PREPROCESSING_DIR, HULL_AVG_2D_FILENAME );
	sprintf( HULL_AVG_3D_PATH,"%s\\%s", PREPROCESSING_DIR, HULL_AVG_3D_FILENAME );
	sprintf( HULL_COMBO_2D_PATH,"%s\\%s", PREPROCESSING_DIR, HULL_COMBO_2D_FILENAME );
	sprintf( HULL_COMBO_3D_PATH,"%s\\%s", PREPROCESSING_DIR, HULL_COMBO_3D_FILENAME );
	
	sprintf( FBP_PATH,"%s\\%s", PREPROCESSING_DIR, FBP_FILENAME );
	sprintf( FBP_MEDIAN_2D_PATH,"%s\\%s", PREPROCESSING_DIR, FBP_MEDIAN_2D_FILENAME );
	sprintf( FBP_MEDIAN_3D_PATH,"%s\\%s", PREPROCESSING_DIR, FBP_MEDIAN_3D_FILENAME );
	sprintf( FBP_AVG_2D_PATH,"%s\\%s", PREPROCESSING_DIR, FBP_AVG_2D_FILENAME );
	sprintf( FBP_AVG_3D_PATH,"%s\\%s", PREPROCESSING_DIR, FBP_AVG_3D_FILENAME );
	sprintf( FBP_COMBO_2D_PATH,"%s\\%s", PREPROCESSING_DIR, FBP_COMBO_2D_FILENAME );
	sprintf( FBP_COMBO_3D_PATH,"%s\\%s", PREPROCESSING_DIR, FBP_COMBO_3D_FILENAME );
	
	sprintf( X_0_PATH, "%s\\%s", PREPROCESSING_DIR, X_0_FILENAME );
	sprintf( X_0_MEDIAN_2D_PATH, "%s\\%s", PREPROCESSING_DIR, X_0_MEDIAN_2D_FILENAME );
	sprintf( X_0_MEDIAN_3D_PATH, "%s\\%s", PREPROCESSING_DIR, X_0_MEDIAN_3D_FILENAME );
	sprintf( X_0_AVG_2D_PATH, "%s\\%s", PREPROCESSING_DIR, X_0_AVG_2D_FILENAME );
	sprintf( X_0_AVG_3D_PATH, "%s\\%s", PREPROCESSING_DIR, X_0_AVG_3D_FILENAME );
	sprintf( X_0_COMBO_2D_PATH, "%s\\%s", PREPROCESSING_DIR, X_0_COMBO_2D_FILENAME );
	sprintf( X_0_COMBO_3D_PATH, "%s\\%s", PREPROCESSING_DIR, X_0_COMBO_3D_FILENAME );
		
	sprintf( X_PATH_BASE,"%s\\%s", RECONSTRUCTION_DIR, X_FILENAME_BASE);
	sprintf( X_MEDIAN_2D_PATH_BASE,"%s\\%s", RECONSTRUCTION_DIR, X_MEDIAN_2D_FILENAME_BASE);
	sprintf( X_MEDIAN_3D_PATH_BASE,"%s\\%s", RECONSTRUCTION_DIR, X_MEDIAN_3D_FILENAME_BASE);
	sprintf( X_AVG_2D_PATH_BASE,"%s\\%s", RECONSTRUCTION_DIR, X_AVG_2D_FILENAME_BASE);
	sprintf( X_AVG_3D_PATH_BASE,"%s\\%s", RECONSTRUCTION_DIR, X_AVG_3D_FILENAME_BASE);
	sprintf( X_COMBO_2D_PATH_BASE,"%s\\%s", RECONSTRUCTION_DIR, X_COMBO_2D_FILENAME_BASE);
	sprintf( X_COMBO_3D_PATH_BASE,"%s\\%s", RECONSTRUCTION_DIR, X_COMBO_3D_FILENAME_BASE);
	
	sprintf(MLP_PATH,"%s\\%s", PREPROCESSING_DIR, MLP_FILENAME );
	sprintf(WEPL_PATH,"%s\\%s", PREPROCESSING_DIR, WEPL_FILENAME );
	sprintf(HISTORIES_PATH,"%s\\%s", PREPROCESSING_DIR, HISTORIES_FILENAME );
	sprintf(VOXELS_PER_PATH_PATH,"%s\\%s", PREPROCESSING_DIR, VOXELS_PER_PATH_FILENAME );
	sprintf(AVG_CHORDS_PATH,"%s\\%s", PREPROCESSING_DIR, AVG_CHORDS_FILENAME );
	
	HULL_2_USE_FILENAME = HULL_FILENAME;
	HULL_2_USE_PATH = HULL_PATH;
	FBP_2_USE_FILENAME = FBP_FILENAME;
	FBP_2_USE_PATH = FBP_PATH;
	X_0_2_USE_FILENAME = X_0_AVG_2D_FILENAME;
	X_0_2_USE_PATH = X_0_AVG_2D_PATH;
	X_2_USE_FILENAME_BASE = X_FILENAME_BASE;
	X_2_USE_PATH_BASE = X_PATH_BASE;
	
	printf("HULL_PATH = %s\n\n", HULL_PATH );	
	printf("HULL_MEDIAN_2D_PATH = %s\n\n", HULL_MEDIAN_2D_PATH );	
	printf("HULL_MEDIAN_3D_PATH = %s\n\n", HULL_MEDIAN_3D_PATH );	
	printf("HULL_AVG_2D_PATH = %s\n\n", HULL_AVG_2D_PATH );	
	printf("HULL_AVG_3D_PATH = %s\n\n", HULL_AVG_3D_PATH );	
	printf("HULL_COMBO_2D_PATH = %s\n\n", HULL_COMBO_2D_PATH );	
	printf("HULL_COMBO_3D_PATH = %s\n\n", HULL_COMBO_3D_PATH );	
	
	printf("FBP_PATH = %s\n\n", FBP_PATH );
	printf("FBP_MEDIAN_2D_PATH = %s\n\n", FBP_MEDIAN_2D_PATH );
	printf("FBP_MEDIAN_3D_PATH = %s\n\n", FBP_MEDIAN_3D_PATH );
	printf("FBP_AVG_2D_PATH = %s\n\n", FBP_AVG_2D_PATH );
	printf("FBP_AVG_3D_PATH = %s\n\n", FBP_AVG_3D_PATH );
	printf("FBP_COMBO_2D_PATH = %s\n\n", FBP_COMBO_2D_PATH );
	printf("FBP_COMBO_3D_PATH = %s\n\n", FBP_COMBO_3D_PATH );
	
	printf("X_0_PATH = %s\n\n", X_0_PATH );	
	printf("X_0_MEDIAN_2D_PATH = %s\n\n", X_0_MEDIAN_2D_PATH );	
	printf("X_0_MEDIAN_3D_PATH = %s\n\n", X_0_MEDIAN_3D_PATH );	
	printf("X_0_AVG_2D_PATH = %s\n\n", X_0_AVG_2D_PATH );	
	printf("X_0_AVG_3D_PATH = %s\n\n", X_0_AVG_3D_PATH );	
	printf("X_0_COMBO_2D_PATH = %s\n\n", X_0_COMBO_2D_PATH );	
	printf("X_0_COMBO_3D_PATH = %s\n\n", X_0_COMBO_3D_PATH );	
	
	printf("X_PATH = %s\n\n", X_PATH_BASE );
	printf("X_MEDIAN_2D_PATH = %s\n\n", X_MEDIAN_2D_PATH_BASE );
	printf("X_MEDIAN_3D_PATH = %s\n\n", X_MEDIAN_3D_PATH_BASE );
	printf("X_AVG_2D_PATH = %s\n\n", X_AVG_2D_PATH_BASE );
	printf("X_AVG_3D_PATH = %s\n\n", X_AVG_3D_PATH_BASE );	
	printf("X_COMBO_2D_PATH = %s\n\n", X_COMBO_2D_PATH_BASE );
	printf("X_COMBO_3D_PATH = %s\n\n", X_COMBO_3D_PATH_BASE );
	
	printf("MLP_PATH = %s\n\n", MLP_PATH );	
	printf("WEPL_PATH = %s\n\n", WEPL_PATH );	
	printf("HISTORIES_PATH = %s\n", HISTORIES_PATH );	
	printf("VOXELS_PER_PATH_PATH = %s\n\n", VOXELS_PER_PATH_PATH );	
	printf("AVG_CHORDS_PATH = %s\n\n", AVG_CHORDS_PATH );	
	
	print_section_exit("Finished setting paths to where the input/output data are to be written", "====>" );
}
void set_images_2_use()
{
	print_section_header( "Setting image file names and paths based on specified filtering options...", '*' );

	// Hull image specification
	if( parameters.MEDIAN_FILTER_HULL && parameters.AVG_FILTER_HULL )
	{
		HULL_2_USE_FILENAME = HULL_COMBO_2D_FILENAME;
		HULL_2_USE_PATH = HULL_COMBO_2D_PATH;
	}
	else if( parameters.MEDIAN_FILTER_HULL)
	{
		HULL_2_USE_FILENAME = HULL_MEDIAN_2D_FILENAME;
		HULL_2_USE_PATH = HULL_MEDIAN_2D_PATH;
	}
	else if( parameters.AVG_FILTER_HULL)
	{
		HULL_2_USE_FILENAME = HULL_AVG_2D_FILENAME;
		HULL_2_USE_PATH = HULL_AVG_2D_PATH;
	}
	else
	{
		HULL_2_USE_FILENAME = HULL_FILENAME;
		HULL_2_USE_PATH = HULL_PATH;
	}
	print_section_separator('~');
	printf("HULL_2_USE_FILENAME = %s\n", HULL_2_USE_FILENAME );
	printf("HULL_2_USE_PATH = %s\n", HULL_2_USE_PATH );
	print_section_separator('~');

	// FBP image specification
	if( parameters.MEDIAN_FILTER_FBP && parameters.AVG_FILTER_FBP )
	{
		FBP_2_USE_FILENAME = FBP_COMBO_2D_FILENAME;
		FBP_2_USE_PATH = FBP_COMBO_2D_PATH;
	}
	else if( parameters.MEDIAN_FILTER_FBP)
	{
		FBP_2_USE_FILENAME = FBP_MEDIAN_2D_FILENAME;
		FBP_2_USE_PATH = FBP_MEDIAN_2D_PATH;
	}
	else if( parameters.AVG_FILTER_FBP)
	{
		FBP_2_USE_FILENAME = FBP_AVG_2D_FILENAME;
		FBP_2_USE_PATH = FBP_AVG_2D_PATH;
	}
	else
	{
		FBP_2_USE_FILENAME = FBP_FILENAME;
		FBP_2_USE_PATH = FBP_PATH;
	}
	print_section_separator('~');
	printf("FBP_2_USE_FILENAME = %s\n", FBP_2_USE_FILENAME );
	printf("FBP_2_USE_PATH = %s\n", FBP_2_USE_PATH );
	print_section_separator('~');

	// x_0 image specification
	if( parameters.MEDIAN_FILTER_X_0 && parameters.AVG_FILTER_X_0 )
	{
		X_0_2_USE_FILENAME = X_0_COMBO_2D_FILENAME;
		X_0_2_USE_PATH = X_0_COMBO_2D_PATH;
	}
	else if( parameters.MEDIAN_FILTER_X_0)
	{
		X_0_2_USE_FILENAME = X_0_MEDIAN_2D_FILENAME;
		X_0_2_USE_PATH = X_0_MEDIAN_2D_PATH;
	}
	else if( parameters.AVG_FILTER_X_0)
	{
		X_0_2_USE_FILENAME = X_0_AVG_2D_FILENAME;
		X_0_2_USE_PATH = X_0_AVG_2D_PATH;
	}
	else
	{
		X_0_2_USE_FILENAME = X_0_FILENAME;
		X_0_2_USE_PATH = X_0_PATH;
	}
	print_section_separator('~');
	printf("X_0_2_USE_FILENAME = %s\n", X_0_2_USE_FILENAME );
	printf("X_0_2_USE_PATH = %s\n", X_0_2_USE_PATH );
	print_section_separator('~');

	//// x_k image specification
	//if( parameters.MEDIAN_FILTER_X_K && parameters.AVG_FILTER_X_K )
	//{
	//	X_K_2_USE_FILENAME = X_K_COMBO_2D_FILENAME;
	//	X_K_2_USE_PATH = X_K_COMBO_2D_PATH;
	//}
	//else if( parameters.MEDIAN_FILTER_X_K)
	//{
	//	X_K_2_USE_FILENAME = X_K_MEDIAN_2D_FILENAME;
	//	X_K_2_USE_PATH = X_K_MEDIAN_2D_PATH;
	//}
	//else if( parameters.AVG_FILTER_X_K)
	//{
	//	X_K_2_USE_FILENAME = X_K_AVG_2D_FILENAME;
	//	X_K_2_USE_PATH = X_K_AVG_2D_PATH;
	//}
	//else
	//{
	//	X_K_2_USE_FILENAME = X_K_FILENAME;
	//	X_K_2_USE_PATH = X_K_PATH;
	//}
	//print_section_separator('~');
	//printf("X_K_2_USE_FILENAME = %s\n", X_K_2_USE_FILENAME );
	//printf("X_K_2_USE_PATH = %s\n", X_K_2_USE_PATH );
	//print_section_separator('~');

	// x image specification
	if( parameters.MEDIAN_FILTER_X && parameters.AVG_FILTER_X )
	{
		X_2_USE_FILENAME_BASE = X_COMBO_2D_FILENAME_BASE;
		X_2_USE_PATH_BASE = X_COMBO_2D_PATH_BASE;
	}
	else if( parameters.MEDIAN_FILTER_X)
	{
		X_2_USE_FILENAME_BASE = X_MEDIAN_2D_FILENAME_BASE;
		X_2_USE_PATH_BASE = X_MEDIAN_2D_PATH_BASE;
	}
	else if( parameters.AVG_FILTER_X)
	{
		X_2_USE_FILENAME_BASE = X_AVG_2D_FILENAME_BASE;
		X_2_USE_PATH_BASE = X_AVG_2D_PATH_BASE;
	}
	else
	{
		X_2_USE_FILENAME_BASE = X_FILENAME_BASE;
		X_2_USE_PATH_BASE = X_PATH_BASE;
	}
	print_section_separator('~');
	printf("X_2_USE_FILENAME_BASE = %s\n", X_2_USE_FILENAME_BASE );
	printf("X_2_USE_PATH_BASE = %s\n", X_2_USE_PATH_BASE );
	print_section_separator('~');
}
void existing_data_check()
{
	//if( parameters.PERFORM_RECONSTRUCTION && !parameters.PREPROCESS_OVERWRITE_OK )
	//{
		//char* existing_data_dir = "D:\\pCT_Data\\Output\\CTP404\\CTP404_merged\\";
		HULL_EXISTS = file_exists3(HULL_2_USE_PATH);
		FBP_EXISTS = file_exists3(FBP_2_USE_PATH);
		X_0_EXISTS = file_exists3(X_0_2_USE_PATH);
		MLP_EXISTS = file_exists3(MLP_PATH);
		WEPL_EXISTS = file_exists3(WEPL_PATH);
		VOXELS_PER_PATH_EXISTS = file_exists3(VOXELS_PER_PATH_PATH);
		AVG_CHORD_LENGTHS_EXISTS = file_exists3(AVG_CHORDS_PATH);
		HISTORIES_EXISTS = file_exists3(HISTORIES_PATH);

		//X_K_EXISTS = file_exists3(X_K_2_USE_PATH);
		// HULL_EXISTS, FBP_EXISTS, X_0_EXISTS, X_K_EXISTS, X_EXISTS, MLP_EXISTS, WEPL_EXISTS, VOXELS_PER_PATH_EXISTS, AVG_CHORD_LENGTHS_EXISTS, HISTORIES_EXISTS;
		uint NUM_X_EXISTS = 0;
		//X_2_USE_PATH_BASE = (char*)calloc(256,sizeof(char));
		//sprintf(X_2_USE_PATH_BASE, "%s%s", RECONSTRUCTION_DIR, X_BASENAME);
		char x_existing_check[256];
		//sprintf(X_FILE_EXTENSION, ".txt");
		sprintf( x_existing_check, "%s_%d%s", X_2_USE_PATH_BASE, NUM_X_EXISTS + 1, X_FILE_EXTENSION );
		//cout << x_existing_check << endl;
		while( file_exists3(x_existing_check) )
			sprintf( x_existing_check, "%s_%d%s", X_2_USE_PATH_BASE, ++NUM_X_EXISTS + 1, X_FILE_EXTENSION );

		if( NUM_X_EXISTS > 0 )
			X_EXISTS = true;
		//cout << NUM_X_EXISTS << endl;
		//cout << HULL_EXISTS << endl;
		//cout << FBP_EXISTS << endl;
		//cout << X_0_EXISTS << endl;
		//cout << X_K_EXISTS << endl;
		// MLP_PATH, *WEPL_PATH, *HISTORIES_PATH
		
		//cout << MLP_EXISTS << endl;
		//cout << WEPL_EXISTS << endl;
		//cout << HISTORIES_EXISTS << endl;
	//}
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