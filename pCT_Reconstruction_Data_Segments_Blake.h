//#ifndef PCT_RECONSTRUCTION_H
//#define PCT_RECONSTRUCTION_H
#pragma once
/*****************************************************************************************************************************************************************************************************/
/*****************************************************************************************************************************************************************************************************/
/******************************************************************************* Header for pCT reconstruction program *******************************************************************************/
/*****************************************************************************************************************************************************************************************************/
/*****************************************************************************************************************************************************************************************************/
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>

#include <algorithm>    // std::transform
//#include <array>
#include <cmath>
#include <cstdarg>		// va_list, va_arg, va_start, va_end, va_copy
#include <cstdio>		// printf, sprintf,  
#include <cstdlib>		// rand, srand
#include <cstring>
#include <ctime>		// clock(), time() 
#include <fstream>
#include <functional>	// std::multiplies, std::plus, std::function, std::negate
#include <iostream>
#include <limits>
#include <map>
#include <new>			 
#include <numeric>		// inner_product, partial_sum, adjacent_difference, accumulate
#include <omp.h>		// OpenMP
#include <sstream>
#include <string>
#include "sys/types.h"	// stat f
#include "sys/stat.h"	// stat functions
#include <typeinfo>		//operator typeid
#include <utility> // for std::move
#include <vector>

#if defined(_WIN32) || defined(_WIN64)
	#include "Shlwapi.h"
	#include <windows.h>
#else
	#include <unistd.h>
#endif
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//-------------------------------------------------------------------------------- Execution and early exit options ----------------------------------------------------------------------------------/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
using std::cout;
using std::endl;
typedef unsigned long long ULL;
typedef unsigned int UINT;
/*****************************************************************************************************************************************************************************************************/
/*****************************************************************************************************************************************************************************************************/
/*********************************************************************************** Preprocessing usage options *************************************************************************************/
/*****************************************************************************************************************************************************************************************************/
/*****************************************************************************************************************************************************************************************************/
enum DATA_FORMATS			{ OLD_FORMAT, VERSION_0, VERSION_1				};				// Define the data formats that are supported
enum SCAN_TYPES				{ EXPERIMENTAL, SIMULATED_G, SIMULATED_T		};				// Experimental or simulated (GEANT4 or TOPAS) data
enum DISK_WRITE_MODE		{ TEXT, BINARY									};				// Experimental or simulated data
enum BIN_ANALYSIS_TYPE		{ MEANS, COUNTS, MEMBERS						};				// Choices for what information about the binned data is desired 
enum BIN_ANALYSIS_FOR		{ ALL_BINS, SPECIFIC_BINS						};				// Choices for which bins the desired data should come from
enum BIN_ORGANIZATION		{ BY_BIN, BY_HISTORY							};				// Binned data is either organized in order by bin or by history w/ bin # specified separately
enum BIN_ANALYSIS_OF		{ WEPLS, ANGLES, POSITIONS, BIN_NUMS			};				// Choices for which type of binned data is desired
enum FILTER_TYPES			{ RAM_LAK, SHEPP_LOGAN, NONE					};				// Define the types of filters that are available for use in FBP
enum HULL_TYPES				{ SC_HULL, MSC_HULL, SM_HULL, FBP_HULL			};				// Define valid choices for which hull to use in MLP calculations
enum INITIAL_ITERATE		{ X_HULL, FBP_IMAGE, HYBRID, ZEROS, IMPORT		};				// Define valid choices for which hull to use in MLP calculations
enum PROJECTION_ALGORITHMS	{ ART, SART, DROP, BIP, SAP						};				// Define valid choices for iterative projection algorithm to use
enum TX_OPTIONS				{ FULL_TX, PARTIAL_TX, PARTIAL_TX_PREALLOCATED	};				// Define valid choices for the host->GPU data transfer method
enum ENDPOINTS_ALGORITHMS	{ USE_BOOL_ARRAY, NO_BOOL_ARRAY					};				// Define the method used to identify protons that miss/hit the hull in MLP endpoints calculations
enum MLP_ALGORITHMS			{ STANDARD, TABULATED							};				// Define whether standard explicit calculations or lookup tables are used in MLP calculations
enum LOG_ENTRIES			{ OBJECT_L, SCAN_TYPE_L, RUN_DATE_L, RUN_NUMBER_L,				// Define the headings of each column in the execution log 
							ACQUIRED_BY_L, PROJECTION_DATA_DATE_L, CALIBRATED_BY_L, 				
							PREPROCESS_DATE_L, PREPROCESSED_BY_L, RECONSTRUCTION_DATE_L, 
							RECONSTRUCTED_BY_L, COMMENTS_L					};
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
struct generic_IO_container
{
	char* key;
	unsigned int input_type_ID;
	int integer_input;		// type_ID = 1
	double double_input;	// type_ID = 2
	char string_input[512];	// type_ID = 3
};
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//-------------------------------------------------------------------------------- Execution and early exit options ----------------------------------------------------------------------------------/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
const bool RUN_ON						= false;												// Turn preprocessing on/off (T/F) to enter individual function testing without commenting
const bool TESTING_ON					= false;												// Write output to "testing" directory (T) or to organized dat directory (F)
const bool OVERWRITING_OK				= false;											// Permit output data to be written to an existing directory (T) or enforce writing to new unique directory (F)
const bool EXIT_AFTER_BINNING			= false;											// Exit program early after completing data read and initial processing
const bool EXIT_AFTER_HULLS				= false;											// Exit program early after completing hull-detection
const bool EXIT_AFTER_CUTS				= false;											// Exit program early after completing statistical cuts
const bool EXIT_AFTER_SINOGRAM			= false;											// Exit program early after completing the construction of the sinogram
const bool EXIT_AFTER_FBP				= false;											// Exit program early after completing FBP
/*****************************************************************************************************************************************************************************************************/
/*****************************************************************************************************************************************************************************************************/
/****************************************************************************** Input/output specifications and options ******************************************************************************/
/*****************************************************************************************************************************************************************************************************/
/*****************************************************************************************************************************************************************************************************/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//------------------------------------------------------------------------------ Path to the input/output directories --------------------------------------------------------------------------------/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
const DATA_FORMATS			DATA_FORMAT				= VERSION_0;							// Specify which data format to use for this run
const SCAN_TYPES			SCAN_TYPE				= SIMULATED_G;			  				// Specifies which of the defined filters will be used in FBP
const FILTER_TYPES			FBP_FILTER				= SHEPP_LOGAN;			  				// Specifies which of the defined filters will be used in FBP
const HULL_TYPES			MLP_HULL				= MSC_HULL;								// Specify which of the HULL_TYPES to use in this run's MLP calculations
const TX_OPTIONS			ENDPOINTS_TX_MODE		= PARTIAL_TX_PREALLOCATED;				// Specifies GPU data tx mode for MLP endpoints as all data (FULL_TX), portions of data (PARTIAL_TX), or portions of data w/ reused GPU arrays (PARTIAL_TX_PREALLOCATED)
const TX_OPTIONS			DROP_TX_MODE			= FULL_TX;				// Specifies GPU data tx mode for MLP+DROP as all data (FULL_TX), portions of data (PARTIAL_TX), or portions of data w/ reused GPU arrays (PARTIAL_TX_PREALLOCATED)
const ENDPOINTS_ALGORITHMS	ENDPOINTS_ALG			= NO_BOOL_ARRAY;						// Specifies if boolean array is used to store whether a proton hit/missed the hull (BOOL) or uses the 1st MLP voxel (NO_BOOL_ARRAY)
const MLP_ALGORITHMS		MLP_ALGORITHM			= TABULATED;							// Specifies whether calculations are performed explicitly (STANDARD) or if lookup tables are used for MLP calculations (TABULATED)
const INITIAL_ITERATE		X_0						= HYBRID;								// Specify which of the HULL_TYPES to use in this run's MLP calculations
const PROJECTION_ALGORITHMS	PROJECTION_ALGORITHM	= DROP;									// Specify which of the projection algorithms to use for image reconstruction
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//-------------------------------------------------------------------------------- Execution and early exit options ----------------------------------------------------------------------------------/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//const char OUTPUT_DIRECTORY[]		= "C://Users//Blake//Documents//Visual Studio 2010//Projects//pCT_Reconstruction_R01//pCT_Reconstruction_R01//";		// Blake's Laptop
//const char INPUT_DIRECTORY[]		= "//home//share//organized_data//";																					// Workstation #2
//const char OUTPUT_DIRECTORY[]		= "//home//share//reconstruction_data//";																				// Workstation #2
//const char INPUT_DIRECTORY[]		= "//local//pCT_data//organized_data//";																				// WHartnell
//const char OUTPUT_DIRECTORY[]		= "//local//pCT_data//reconstruction_data//";																			// WHartnell
//const char INPUT_DIRECTORY[]		= "//local//pCT_data//organized_data//";																				// JPertwee
//const char OUTPUT_DIRECTORY[]		= "//local//pCT_data//reconstruction_data//";																			// JPertwee
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//const char INPUT_FROM[]			= "input_CTP404_4M//Simulated//14-10-28";																					// CTP404_Sensitom_4M
//const char INPUT_FROM[]			= "CTP404_Sensitom//Experimental//15-05-16//0061//Output//15-06-25";														// CTP404_Sensitom
//const char INPUT_FROM[]			= "Edge_Phantom//Experimental//15-05-16//0057//Output//15-05-25";															// Edge_Phantom
//const char INPUT_FROM[]			= "Head_Phantom//Experimental//15-05-16//0059_Sup//Output//15-06-25";														// Head_Phantom
//const char INPUT_FROM[]			= "Head_Phantom//Experimental//15-05-16//0060_inf//Output//15-06-25";														// Head_Phantom
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//const char OUTPUT_TO[]		= "input_CTP404_4M//Simulated//14-10-28";																					// CTP404_Sensitom_4M
//const char OUTPUT_TO[]		= "CTP404_Sensitom//Experimental//15-05-16//0061";																		// CTP404_Sensitom
//const char OUTPUT_TO[]			= "Edge_Phantom//Experimental//15-05-16//0057";																			// Edge_Phantom
//const char OUTPUT_TO[]		= "Head_Phantom//Experimental//15-05-16//0059_Sup";																		// Head_Phantom
//const char OUTPUT_TO[]		= "Head_Phantom//Experimental//15-05-16//0060_Inf";																		// Head_Phantom
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
const char PHANTOM[]				= "CTP404_Sensitom_4M";
//const char PHANTOM[]				= "CTP404_Sensitom";
//const char PHANTOM[]				= "Edge_Phantom";
//const char PHANTOM[]				= "Head_Phantom";
const char RUN_NUMBER[]				= "0001";
//const char RUN_NUMBER[]				= "0061";
//const char RUN_NUMBER[]				= "0057";
//const char RUN_NUMBER[]				= "0059_Sup";
//const char RUN_NUMBER[]				= "0060_Inf";
const char RUN_DATE[]				= "14-10-28";
//const char RUN_DATE[]				= "15-05-16";
const char PREPROCESSED_DATE[]		= "14-10-28";
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
const char KODIAK_RECON_DIR[]		= "//data/ion//pCT_data//reconstruction_data//";																			// Kodiak
const char WS2_RECON_DIR[]			= "//home//share//reconstruction_data//";																				// Workstation #2
const char WHARTNELL_RECON_DIR[]	= "//local//pCT_Data//reconstruction_data//";																			// WHartnell
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/	
//char GLOBAL_RESULTS_PATH[]		= {"C://Users//Blake//Documents//Visual Studio 2010//Projects//pCT_Reconstruction_R01//pCT_Reconstruction_R01"};
//char GLOBAL_RESULTS_PATH[]		= {"//home//share//reconstruction_data"};
//char GLOBAL_RESULTS_PATH[]		= {"//local//reconstruction_data"};
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/	

const char KODIAK_CODE_DIR[]		= "//data/ion//pCT_code//Reconstruction//";
const char WS2_CODE_DIR[]			= "//home//share//pCT_code//";
const char WHARTNELL_CODE_DIR[]		= "//local//pCT_code//";
const char JPERTWEE_CODE_DIR[]		= "//local//pCT_code//";
//const char KODIAK_SCP_BASE[]		= "scp -r schultze@kodiak:";
const char KODIAK_LOGIN[]			= "schultze@kodiak:";
const char WHARTNELL_LOGIN[]		= "schultze@whartnell:";
const char JPERTWEE_LOGIN[]			= "schultze@kodiak:";
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
char INPUT_DIRECTORY[256];
char OUTPUT_DIRECTORY[256];
char INPUT_FROM[256];
char OUTPUT_TO[256];
char DATA_TYPE[32];
char EXECUTION_DATE[9];																		// Specify which of the projection algorithms to use for image reconstruction						

char CONFIG_DIRECTORY[256];
char OUTPUT_TO_UNIQUE[256];
char KODIAK_OUTPUT_PATH[256];
char WS2_OUTPUT_PATH[256];
char WHARTNELL_OUTPUT_PATH[256];
char IMPORT_FBP_PATH[256];
char INPUT_ITERATE_PATH[256];																// Path 



//char* OBJECT, *SCAN_TYPE, *RUN_DATE, *RUN_NUMBER, *PROJECTION_DATA_DATE, *PREPROCESS_DATE, *RECONSTRUCTION_DATE;
char* USER_NAME;
char* PATH_2_PCT_DATA_DIR, *DATA_TYPE_DIR, *PROJECTION_DATA_DIR, *PREPROCESSING_DIR, *RECONSTRUCTION_DIR;
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//--------------------------------------------------------------------------- Iterative Image Reconstruction Parameters ------------------------------------------------------------------------------/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//------------------------------------------------------------------------- Host/GPU computation and structure information ---------------------------------------------------------------------------/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
#define MAX_INTERSECTIONS				1280												// Limit on the # of intersections expected for proton's MLP; = # voxels along image diagonal
#define DROP_BLOCK_SIZE					3200												// # of histories in each DROP block, i.e., # of histories used per image update
#define THREADS_PER_BLOCK				1024												// # of threads per GPU block for preprocessing kernels
#define ENDPOINTS_PER_BLOCK 			320													// # of threads per GPU block for collect_MLP_endpoints_GPU kernel
#define HISTORIES_PER_BLOCK 			320													// # of threads per GPU block for block_update_GPU kernel
#define ENDPOINTS_PER_THREAD 			1													// # of MLP endpoints each thread is responsible for calculating in collect_MLP_endpoints_GPU kernel
#define HISTORIES_PER_THREAD 			1													// # of histories each thread is responsible for in MLP/DROP kernel block_update_GPU
#define VOXELS_PER_THREAD 				1													// # of voxels each thread is responsible for updating for reconstruction image initialization/updates
#define MAX_GPU_HISTORIES				4000000												// [#] Number of histories to process on the GPU at a time for preprocessing, based on GPU capacity
#define MAX_CUTS_HISTORIES				1500000												// [#] Number of histories to process on the GPU at a time for statistical cuts, based on GPU capacity
#define MAX_ENDPOINTS_HISTORIES			10240000											// [#] Number of histories to process on the GPU at a time for MLP endpoints, based on GPU capacity
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//------------------------------------------------------------------------------ Testing configurations/options controls ---------------------------------------------------------------------------------/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//----------------------------------------------------------------------------- Input data specification configurations ----------------------------------------------------------------------------------/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
unsigned int PHANTOM_NAME_SIZE;
unsigned int DATA_SOURCE_SIZE;
unsigned int PREPARED_BY_SIZE;
unsigned int SKIP_2_DATA_SIZE;
unsigned int VERSION_ID;
unsigned int PROJECTION_INTERVAL;
const bool BINARY_ENCODING				= true;												// Input data provided in binary (T) encoded files or ASCI text files (F)
const bool SINGLE_DATA_FILE				= false;											// Individual file for each gantry angle (T) or single data file for all data (F)
const bool SSD_IN_MM					= true;												// SSD distances from rotation axis given in mm (T) or cm (F)
const bool DATA_IN_MM					= true;												// Input data given in mm (T) or cm (F)
const bool MICAH_SIM					= false;											// Specify whether the input data is from Micah's simulator (T) or not (F)
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//--------------------------------------------------------------------------------- Preprocessing option configurations ----------------------------------------------------------------------------------/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
const bool DEBUG_TEXT_ON				= true;												// Provide (T) or suppress (F) print statements to console during execution
const bool SAMPLE_STD_DEV				= true;												// Use sample/population standard deviation (T/F) in statistical cuts (i.e. divisor is N/N-1)
const bool FBP_ON						= true;												// Turn FBP on (T) or off (F)
const bool MEDIAN_FILTER_FBP			= true;												// Apply median filter to FBP (T) or not (F)
const bool AVG_FILTER_FBP				= false;											// Apply averaging filter to FBP (T) or not (F)
const bool IMPORT_FILTERED_FBP			= false;											// Import and use filtered FBP image (T) or not (F)
const bool SC_ON						= false;											// Turn Space Carving on (T) or off (F)
const bool MSC_ON						= true;												// Turn Modified Space Carving on (T) or off (F)
const bool SM_ON						= false;											// Turn Space Modeling on (T) or off (F)
const bool AVG_FILTER_HULL				= true;												// Apply averaging filter to hull (T) or not (F)
const bool AVG_FILTER_ITERATE			= false;											// Apply averaging filter to initial iterate (T) or not (F)
const bool DIRECT_IMAGE_RECONSTRUCTION	= false;											// Begin image reconstruction immediately upon execution (T) or perform preprocessing first (F)
const bool COUNT_0_WEPLS				= false;											// Count the number of histories with WEPL = 0 (T) or not (F)
const bool MLP_FILE_EXISTS				= false;											// Specifies whether the file containing the MLP data for this object/settings already exists (T) or not (F)
const bool MLP_ENDPOINTS_FILE_EXISTS	= false;											// Specifies whether the file containing the MLP endpoint data for this object/settings already exists (T) or not (F)
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//------------------------------------------------------------------------------------ Output option configurations --------------------------------------------------------------------------------------/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
const bool WRITE_SC_HULL			= true;													// Write SC hull to disk (T) or not (F)
const bool WRITE_MSC_COUNTS			= true;													// Write MSC counts array to disk (T) or not (F) before performing edge detection 
const bool WRITE_MSC_HULL			= true;													// Write MSC hull to disk (T) or not (F)
const bool WRITE_SM_COUNTS			= true;													// Write SM counts array to disk (T) or not (F) before performing edge detection 
const bool WRITE_SM_HULL			= true;													// Write SM hull to disk (T) or not (F)
const bool WRITE_FBP_IMAGE			= true;													// Write FBP image before thresholding to disk (T) or not (F)
const bool WRITE_FBP_HULL			= true;													// Write FBP hull to disk (T) or not (F)
const bool WRITE_AVG_FBP			= true;													// Write average filtered FBP image before thresholding to disk (T) or not (F)
const bool WRITE_MEDIAN_FBP			= true;													// Write median filtered FBP image to disk (T) or not (F)
const bool WRITE_FILTERED_HULL		= true;													// Write average filtered FBP image to disk (T) or not (F)
const bool WRITE_X_HULL				= true;													// Write the hull selected to be used in MLP calculations to disk (T) or not (F)
const bool WRITE_X_0				= true;													// Write the hull selected to be used in MLP calculations to disk (T) or not (F)
const bool WRITE_X_KI				= true;													// Write the hull selected to be used in MLP calculations to disk (T) or not (F)
const bool WRITE_X					= false;												// Write the reconstructed image to disk (T) or not (F)
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//--------------------------------------------------------------------------- Binned data analysis options and configurations ----------------------------------------------------------------------------/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
const bool WRITE_BIN_WEPLS			= false;												// Write WEPLs for each bin to disk (T) for WEPL distribution analysis, or do not (F)
const bool WRITE_WEPL_DISTS			= false;												// Write mean WEPL values to disk (T) or not (F): t bin = columns, v bin = rows, 1 angle per file
const bool WRITE_SSD_ANGLES			= false;												// Write angles for each proton through entry/exit tracker planes to disk (T), or do not (F) 
/*****************************************************************************************************************************************************************************************************/
/*****************************************************************************************************************************************************************************************************/
/************************************************************************************** Preprocessing Constants **************************************************************************************/
/*****************************************************************************************************************************************************************************************************/
/*****************************************************************************************************************************************************************************************************/						
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//------------------------------------------------------ Scanning and detector system (source distance, tracking plane dimensions) configurations --------------------------------------------------------/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
#define GANTRY_ANGLE_INTERVAL		4.0														// [degrees] Angle between successive projection angles 
#define GANTRY_ANGLES				static_cast<int>( 360 / GANTRY_ANGLE_INTERVAL )			// [#] Total number of projection angles
#define NUM_SCANS					1														// [#] Total number of scans
#define NUM_FILES					( NUM_SCANS * GANTRY_ANGLES )							// [#] 1 file per gantry angle per translation
#define SSD_T_SIZE					35.0													// [cm] Length of SSD in t (lateral) direction
#define SSD_V_SIZE					10.0													// [cm] Length of SSD in v (vertical) direction
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//-------------------------------------------------------------- Binning (for statistical analysis) and sinogram (for FBP) configurations ----------------------------------------------------------------/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
#define T_SHIFT						0.0														// [cm] Amount by which to shift all t coordinates on input
#define U_SHIFT						0.0														// [cm] Amount by which to shift all u coordinates on input
#define V_SHIFT						0.0														// [cm] Amount by which to shift all v coordinates on input
#define T_BIN_SIZE					0.1														// [cm] Distance between adjacent bins in t (lateral) direction
#define T_BINS						static_cast<int>( SSD_T_SIZE / T_BIN_SIZE + 0.5 )		// [#] Number of bins (i.e. quantization levels) for t (lateral) direction 
#define V_BIN_SIZE					0.25													// [cm] Distance between adjacent bins in v (vertical) direction
#define V_BINS						static_cast<int>( SSD_V_SIZE/ V_BIN_SIZE + 0.5 )		// [#] Number of bins (i.e. quantization levels) for v (vertical) direction 
#define ANGULAR_BIN_SIZE			4.0														// [degrees] Angle between adjacent bins in angular (rotation) direction
#define ANGULAR_BINS				static_cast<int>( 360 / ANGULAR_BIN_SIZE + 0.5 )		// [#] Number of bins (i.e. quantization levels) for path angle 
#define NUM_BINS					( ANGULAR_BINS * T_BINS * V_BINS )						// [#] Total number of bins corresponding to possible 3-tuples [ANGULAR_BIN, T_BIN, V_BIN]
#define SIGMAS_TO_KEEP				3														// [#] Number of standard deviations from mean to allow before cutting the history 
#define RAM_LAK_TAU					2/ROOT_TWO * T_BIN_SIZE									// Defines tau in Ram-Lak filter calculation, estimated from largest frequency in slice 
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//------------------------------------------------------------------------------- Reconstruction cylinder configurations ---------------------------------------------------------------------------------/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//-------------------------------------------------------------------------------- Reconstruction image configurations ----------------------------------------------------------------------------------/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
#define RECON_CYL_RADIUS			12.0													// [cm] Radius of reconstruction cylinder
#define RECON_CYL_DIAMETER			(2 * RECON_CYL_RADIUS)									// [cm] Diameter of reconstruction cylinder
#define RECON_CYL_HEIGHT			(SSD_V_SIZE - 1.0)										// [cm] Height of reconstruction cylinder
#define IMAGE_WIDTH					RECON_CYL_DIAMETER										// [cm] Distance between left and right edges of each slice in image
#define IMAGE_HEIGHT				RECON_CYL_DIAMETER										// [cm] Distance between top and bottom edges of each slice in image
#define IMAGE_THICKNESS				( SLICES * SLICE_THICKNESS )							// [cm] Distance between bottom of bottom slice and top of the top slice of image
#define COLUMNS						static_cast<int>(RECON_CYL_DIAMETER / VOXEL_WIDTH)		// [#] Number of voxels in the x direction (i.e., number of columns) of image
#define ROWS						static_cast<int>(RECON_CYL_DIAMETER / VOXEL_HEIGHT)		// [#] Number of voxels in the y direction (i.e., number of rows) of image
#define SLICES						static_cast<int>( RECON_CYL_HEIGHT / SLICE_THICKNESS)	// [#] Number of voxels in the z direction (i.e., number of slices) of image
#define NUM_VOXELS					(COLUMNS * ROWS * SLICES)								// [#] Total number of voxels (i.e. 3-tuples [column, row, slice]) in image
#define VOXEL_WIDTH					0.1														// [cm] Distance between left and right edges of each voxel in image
#define VOXEL_HEIGHT				0.1														// [cm] Distance between top and bottom edges of each voxel in image
#define VOXEL_THICKNESS				0.25													// [cm] Distance between top and bottom of each slice in image
#define SLICE_THICKNESS				0.25													// [cm] Distance between top and bottom of each slice in image
#define X_ZERO_COORDINATE			-RECON_CYL_RADIUS										// [cm] x-coordinate corresponding to front edge of 1st voxel (i.e. column) in image space
#define Y_ZERO_COORDINATE			RECON_CYL_RADIUS										// [cm] y-coordinate corresponding to front edge of 1st voxel (i.e. row) in image space
#define Z_ZERO_COORDINATE			(RECON_CYL_HEIGHT / 2)									// [cm] z-coordinate corresponding to front edge of 1st voxel (i.e. slice) in image space
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//------------------------------------------------------------------------------------ Hull-Detection Parameters -------------------------------------------------------------------------------------/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
#define SC_LOWER_THRESHOLD			0.0														// [cm] If WEPL >= SC_LOWER_THRESHOLD, SC assumes the proton missed the object
#define SC_UPPER_THRESHOLD			0.0														// [cm] If WEPL <= SC_UPPER_THRESHOLD, SC assumes the proton missed the object
#define MSC_UPPER_THRESHOLD			0.0														// [cm] If WEPL >= MSC_LOWER_THRESHOLD, MSC assumes the proton missed the object
#define MSC_LOWER_THRESHOLD			-10.0													// [cm] If WEPL <= MSC_UPPER_THRESHOLD, MSC assumes the proton missed the object
#define MSC_DIFF_THRESH				75														// [#] Threshold on difference in counts between adjacent voxels used by MSC for edge detection
#define SM_LOWER_THRESHOLD			6.0														// [cm] If WEPL >= SM_THRESHOLD, SM assumes the proton passed through the object
#define SM_UPPER_THRESHOLD			21.0													// [cm] If WEPL > SM_UPPER_THRESHOLD, SM ignores this history
#define SM_SCALE_THRESHOLD			1.0														// [cm] Threshold scaling factor used by SM to adjust edge detection sensitivity
#define FBP_HULL_THRESHOLD			0.6														// [cm] RSP threshold used to generate FBP_hull from FBP_image
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//---------------------------------------------------------------------------- Average/median filtering options/parameters ---------------------------------------------------------------------------/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
UINT HULL_AVG_FILTER_RADIUS			= 0;													// [#] Radius of the average filter to apply to hull image
UINT FBP_MED_FILTER_RADIUS			= 3;													// [#] Radius of the median filter to apply to FBP image
UINT FBP_AVG_FILTER_RADIUS			= 1;													// [#] Radius of the average filter to apply to FBP image
UINT ITERATE_AVG_FILTER_RADIUS			= 3;													// [#] Radius of the average filter to apply to initial iterate
double HULL_AVG_FILTER_THRESHOLD	= 0.1;													// [#] Threshold applied to average filtered hull separating voxels to include/exclude from hull (i.e. set to 0/1)
double FBP_AVG_FILTER_THRESHOLD		= 0.1;													// [#] Threshold applied to average filtered FBP separating voxels to include/exclude from FBP hull (i.e. set to 0/1)
double ITERATE_AVG_FILTER_THRESHOLD	= 0.1;													// [#] Threshold applied to average filtered initial iterate below which a voxel is excluded from reconstruction (i.e. set to 0)
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//----------------------------------------------------------------------------------------- MLP Parameters -------------------------------------------------------------------------------------------/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
float LAMBDA						= 0.0005;												// Relaxation parameter to use in image iterative projection reconstruction algorithms	
int PSI_SIGN						= 1;													// Sign of the perturbation used in robust reconstruction imposing Tikhonov, ridge regression, total least square, minmax, maxmax, 
double ETA							= 2.5;													// Radius of bounded region in which a solution is sought, commonly set based on the bound of expected error in measurements
#define ITERATIONS					6														// # of iterations through the entire set of histories to perform in iterative image reconstruction
#define BOUND_IMAGE					1														// If any voxel in the image exceeds 2.0, set it to exactly 2.0
const bool TVS_ON					= true;													// Perform total variation superiorization (TVS) (T) or not (F)
const bool TVS_FIRST				= true;													// Perform TVS before DROP updates (T) or after image is updated by DROP block (F)
const bool TVS_PARALLEL				= false;												// Use the parallel implementation of TVS (T) or the host only version (F)
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//----------------------------------------------------------------------------------- Tabulated data file names --------------------------------------------------------------------------------------/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
#define TRIG_TABLE_MIN				-2 * PI													// [radians] Minimum angle contained in the sin/cos lookup table used for MLP
#define TRIG_TABLE_MAX				4 * PI													// [radians] Maximum angle contained in the sin/cos lookup table used for MLP
#define TRIG_TABLE_RANGE			(TRIG_TABLE_MAX - TRIG_TABLE_MIN)						// [radians] Range of angles contained in the sin/cos lookup table used for MLP
#define TRIG_TABLE_STEP				(0.001 * ANGLE_TO_RADIANS)								// [radians] Step size in radians between elements of sin/cos lookup table used for MLP
#define TRIG_TABLE_ELEMENTS			static_cast<int>(TRIG_TABLE_RANGE / TRIG_TABLE_STEP)	// [#] # of elements contained in the sin/cos lookup table used for MLP
#define COEFF_TABLE_RANGE			40.0													// [cm] Range of depths u contained in the polynomial lookup tables used for MLP
#define COEFF_TABLE_STEP			0.00005													// [cm] Step of 1/1000 cm for elements of the polynomial lookup tables used for MLP
#define COEFF_TABLE_SHIFT			static_cast<int>(MLP_U_STEP / COEFF_TABLE_STEP)			// [cm] Step of 1/1000 cm for elements of the polynomial lookup tables used for MLP
#define COEFF_TABLE_ELEMENTS		static_cast<int>(COEFF_TABLE_RANGE / COEFF_TABLE_STEP)	// [#] # of elements contained in the polynomial lookup tables used for MLP
#define POLY_TABLE_RANGE			40.0													// [cm] Range of depths u contained in the polynomial lookup tables used for MLP
#define POLY_TABLE_STEP				0.00005													// [cm] Step of 1/1000 cm for elements of the polynomial lookup tables used for MLP
#define POLY_TABLE_SHIFT			static_cast<int>(MLP_U_STEP / POLY_TABLE_STEP)			// [cm] Step of 1/1000 cm for elements of the polynomial lookup tables used for MLP
#define POLY_TABLE_ELEMENTS			static_cast<int>(POLY_TABLE_RANGE / POLY_TABLE_STEP)	// [#] # of elements contained in the polynomial lookup tables used for MLP
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//----------------------------------------------------------------------- Memory allocation size for arrays (binning, image) -------------------------------------------------------------------------/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
#define SIZE_BINS_CHAR				( NUM_BINS   * sizeof(char)		)						// Amount of memory required for a character array used for binning
#define SIZE_BINS_BOOL				( NUM_BINS   * sizeof(bool)		)						// Amount of memory required for a boolean array used for binning
#define SIZE_BINS_INT				( NUM_BINS   * sizeof(int)		)						// Amount of memory required for a integer array used for binning
#define SIZE_BINS_UINT				( NUM_BINS   * sizeof(UINT)		)						// Amount of memory required for a integer array used for binning
#define SIZE_BINS_FLOAT				( NUM_BINS	 * sizeof(float)	)						// Amount of memory required for a floating point array used for binning
#define SIZE_IMAGE_CHAR				( NUM_VOXELS * sizeof(char)		)						// Amount of memory required for a character array used for binning
#define SIZE_IMAGE_BOOL				( NUM_VOXELS * sizeof(bool)		)						// Amount of memory required for a boolean array used for binning
#define SIZE_IMAGE_INT				( NUM_VOXELS * sizeof(int)		)						// Amount of memory required for a integer array used for binning
#define SIZE_IMAGE_UINT				( NUM_VOXELS * sizeof(UINT)		)						// Amount of memory required for a integer array used for binning
#define SIZE_IMAGE_FLOAT			( NUM_VOXELS * sizeof(float)	)						// Amount of memory required for a floating point array used for binning
#define SIZE_IMAGE_DOUBLE			( NUM_VOXELS * sizeof(double)	)						// Amount of memory required for a floating point array used for binning
/*****************************************************************************************************************************************************************************************************/
/*****************************************************************************************************************************************************************************************************/
/************************************************************************************** Precalculated Constants **************************************************************************************/
/*****************************************************************************************************************************************************************************************************/
/*****************************************************************************************************************************************************************************************************/						
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//---------------------------------------------------------- Preprocessing/reconstruction parameters/options and beam/material properties ------------------------------------------------------------/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
#define BYTES_PER_HISTORY			48														// [bytes] Data size of each history, 44 for actual data and 4 empty bytes, for old data format
#define X_INCREASING_DIRECTION		RIGHT													// [#] specifies direction (LEFT/RIGHT) along x-axis in which voxel #s increase
#define Y_INCREASING_DIRECTION		DOWN													// [#] specifies direction (UP/DOWN) along y-axis in which voxel #s increase
#define Z_INCREASING_DIRECTION		DOWN													// [#] specifies direction (UP/DOWN) along z-axis in which voxel #s increase
#define SOURCE_RADIUS				260.7													// [cm] Distance  to source/scatterer 
#define BEAM_ENERGY					200														// [MeV] Nominal energy of the proton beam 
#define E_0							13.6													// [MeV/c] empirical constant
#define X0							36.08													// [cm] radiation length
#define RSP_AIR						0.00113													// [cm/cm] Approximate RSP of air
#define MAX_ITERATIONS				15														// [#] Max # of iterations ever used in image reconstruction so execution times file has enough columns for iteration times
#define MLP_U_STEP					( VOXEL_WIDTH / 2)										// Size of the step taken along u direction during MLP; depth difference between successive MLP points
#define A_0							7.457E-6												// Coefficient of x^0 of 5th order polynomial fit of [1 / ( beta^2(u)*p^2(u) )] term of MLP scattering matrices Sigma 1/2 for 200 MeV beam
#define A_1							4.548E-7												// Coefficient of x^1 of 5th order polynomial fit of [1 / ( beta^2(u)*p^2(u) )] term of MLP scattering matrices Sigma 1/2 for 200 MeV beam
#define A_2							-5.777E-8												// Coefficient of x^2 of 5th order polynomial fit of [1 / ( beta^2(u)*p^2(u) )] term of MLP scattering matrices Sigma 1/2 for 200 MeV beam
#define A_3							1.301E-9												// Coefficient of x^3 of 5th order polynomial fit of [1 / ( beta^2(u)*p^2(u) )] term of MLP scattering matrices Sigma 1/2 for 200 MeV beam
#define A_4							-9.228E-10												// Coefficient of x^4 of 5th order polynomial fit of [1 / ( beta^2(u)*p^2(u) )] term of MLP scattering matrices Sigma 1/2 for 200 MeV beam
#define A_5							2.687E-11												// Coefficient of x^5 of 5th order polynomial fit of [1 / ( beta^2(u)*p^2(u) )] term of MLP scattering matrices Sigma 1/2 for 200 MeV beam
#define A_0_OVER_2					A_0/2													// Precalculated value of A_0/2 used in MLP routine so this constant does not need to be repeatedly calculated explicitly
#define A_0_OVER_3					A_0/3													// Precalculated value of A_0/3 used in MLP routine so this constant does not need to be repeatedly calculated explicitly
#define A_1_OVER_2					A_1/2													// Precalculated value of A_1/2 used in MLP routine so this constant does not need to be repeatedly calculated explicitly
#define A_1_OVER_3					A_1/3													// Precalculated value of A_1/3 used in MLP routine so this constant does not need to be repeatedly calculated explicitly
#define A_1_OVER_4					A_1/4													// Precalculated value of A_1/4 used in MLP routine so this constant does not need to be repeatedly calculated explicitly
#define A_1_OVER_6					A_1/6													// Precalculated value of A_1/6 used in MLP routine so this constant does not need to be repeatedly calculated explicitly
#define A_1_OVER_12					A_1/12													// Precalculated value of A_1/12 used in MLP routine so this constant does not need to be repeatedly calculated explicitly
#define A_2_OVER_3					A_2/3													// Precalculated value of A_2/3 used in MLP routine so this constant does not need to be repeatedly calculated explicitly
#define A_2_OVER_4					A_2/4													// Precalculated value of A_2/4 used in MLP routine so this constant does not need to be repeatedly calculated explicitly
#define A_2_OVER_5					A_2/5													// Precalculated value of A_2/5 used in MLP routine so this constant does not need to be repeatedly calculated explicitly
#define A_2_OVER_12					A_2/12													// Precalculated value of A_2/12 used in MLP routine so this constant does not need to be repeatedly calculated explicitly
#define A_2_OVER_30					A_2/30													// Precalculated value of A_2/30 used in MLP routine so this constant does not need to be repeatedly calculated explicitly
#define A_3_OVER_4					A_3/4													// Precalculated value of A_3/4 used in MLP routine so this constant does not need to be repeatedly calculated explicitly
#define A_3_OVER_5					A_3/5													// Precalculated value of A_3/5 used in MLP routine so this constant does not need to be repeatedly calculated explicitly
#define A_3_OVER_6					A_3/6													// Precalculated value of A_3/6 used in MLP routine so this constant does not need to be repeatedly calculated explicitly		
#define A_3_OVER_20					A_3/20													// Precalculated value of A_3/20 used in MLP routine so this constant does not need to be repeatedly calculated explicitly
#define A_3_OVER_60					A_3/60													// Precalculated value of A_3/60 used in MLP routine so this constant does not need to be repeatedly calculated explicitly
#define A_4_OVER_5					A_4/5													// Precalculated value of A_4/5 used in MLP routine so this constant does not need to be repeatedly calculated explicitly
#define A_4_OVER_6					A_4/6													// Precalculated value of A_4/6 used in MLP routine so this constant does not need to be repeatedly calculated explicitly
#define A_4_OVER_7					A_4/7													// Precalculated value of A_4/7 used in MLP routine so this constant does not need to be repeatedly calculated explicitly
#define A_4_OVER_30					A_4/30													// Precalculated value of A_4/30 used in MLP routine so this constant does not need to be repeatedly calculated explicitly
#define A_4_OVER_105				A_4/105													// Precalculated value of A_4/105 used in MLP routine so this constant does not need to be repeatedly calculated explicitly
#define A_5_OVER_6					A_5/6													// Precalculated value of A_5/6 used in MLP routine so this constant does not need to be repeatedly calculated explicitly
#define A_5_OVER_7					A_5/7													// Precalculated value of A_5/7 used in MLP routine so this constant does not need to be repeatedly calculated explicitly
#define A_5_OVER_8					A_5/8													// Precalculated value of A_5/8 used in MLP routine so this constant does not need to be repeatedly calculated explicitly			
#define A_5_OVER_42					A_5/42													// Precalculated value of A_5/42 used in MLP routine so this constant does not need to be repeatedly calculated explicitly			
#define A_5_OVER_168				A_5/168													// Precalculated value of A_5/168 used in MLP routine so this constant does not need to be repeatedly calculated explicitly
#define VOXEL_ALLOWANCE				1.0e-7													// [cm] Distance from voxel edge below which the edge is considered to have been reached
#define TV_THRESHOLD				(1/10000)												// [#] Value of TV difference ratio |TV_y - TV_y_previous| / TV_y between successive betas where beta is not decreased more
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//-------------------------------------------------------------------------- Mathematical constants and unit conversions -----------------------------------------------------------------------------/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
#define PHI							((1 + sqrt(5.0) ) / 2)									// [#] Positive golden ratio, positive solution of PHI^2-PHI-1 = 0; also PHI = a/b when a/b = (a + b) / a 
#define PHI_NEGATIVE				((1 - sqrt(5.0) ) / 2)									// [#] Negative golden ratio, negative solution of PHI^2-PHI-1 = 0; 
#define PI_OVER_4					( atan( 1.0 ) )											// [radians] 1*pi/4 radians =   pi/4 radians = 45 degrees
#define PI_OVER_2					( 2 * atan( 1.0 ) )										// [radians] 2*pi/4 radians =   pi/2 radians = 90 degrees
#define THREE_PI_OVER_4				( 3 * atan( 1.0 ) )										// [radians] 3*pi/4 radians = 3*pi/4 radians = 135 degrees
#define PI							( 4 * atan( 1.0 ) )										// [radians] 4*pi/4 radians =   pi   radians = 180 degrees
#define FIVE_PI_OVER_4				( 5 * atan( 1.0 ) )										// [radians] 5*pi/4 radians = 5*pi/4 radians = 225 degrees
#define SIX_PI_OVER_4				( 5 * atan( 1.0 ) )										// [radians] 6*pi/4 radians = 3*pi/2 radians = 270 degrees
#define SEVEN_PI_OVER_4				( 7 * atan( 1.0 ) )										// [radians] 7*pi/4 radians = 7*pi/4 radians = 315 degrees
#define TWO_PI						( 8 * atan( 1.0 ) )										// [radians] 8*pi/4 radians = 2*pi   radians = 360 degrees = 0 degrees
#define ANGLE_TO_RADIANS			( PI/180.0 )											// [radians/degree] Convertion from angle to radians
#define RADIANS_TO_ANGLE			( 180.0/PI )											// [degrees/radian] Convertion from radians to angle
#define ROOT_TWO					sqrtf(2.0)												// [#] 2^(1/2) = Square root of 2 
#define MM_TO_CM					0.1														// [cm/mm] 10 [mm] = 1 [cm] => 1 [mm] = 0.1 [cm]
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//------------------------------------------------------------------------------------ Precalculated Constants ---------------------------------------------------------------------------------------/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
#define START						true													// [bool] Used as an alias for true for starting timer
#define STOP						false													// [bool] Used as an alias for false for stopping timer
#define RIGHT						1														// [#] Specifies that moving right corresponds with an increase in x position, used in voxel walk 
#define LEFT						-1														// [#] Specifies that moving left corresponds with a decrease in x position, used in voxel walk 
#define UP							1														// [#] Specifies that moving up corresponds with an increase in y/z position, used in voxel walk 
#define DOWN						-1														// [#] Specifies that moving down corresponds with a decrease in y/z position, used in voxel walk 
#define BOOL_FORMAT					"%d"													// Specifies format to use for writing/printing boolean data using {s/sn/f/v/vn}printf statements
#define CHAR_FORMAT					"%c"													// Specifies format to use for writing/printing character data using {s/sn/f/v/vn}printf statements
#define INT_FORMAT					"%d"													// Specifies format to use for writing/printing integer data using {s/sn/f/v/vn}printf statements
#define FLOAT_FORMAT				"%f"													// Specifies format to use for writing/printing floating point data using {s/sn/f/v/vn}printf statements
#define STRING_FORMAT				"%s"													// Specifies format to use for writing/printing strings data using {s/sn/f/v/vn}printf statements
#define CONSOLE_WINDOW_WIDTH		80														// [#] Terminal window character width
#define DARK						0														// [#] Color prompt value specifying the usage of dark colors
#define LIGHT						1														// [#] Color prompt value specifying the usage of light colors
#define BLACK						30														// [#] Color prompt value specifying the usage of black color
#define RED							31														// [#] Color prompt value specifying the usage of red color
#define GREEN						32														// [#] Color prompt value specifying the usage of green color
#define BROWN						33														// [#] Color prompt value specifying the usage of brown color
#define BLUE						34														// [#] Color prompt value specifying the usage of blue color
#define PURPLE						35														// [#] Color prompt value specifying the usage of purple color
#define CYAN						36														// [#] Color prompt value specifying the usage of cyan color
const char SECTION_EXIT_STRING[]	= "====>";												// Character string to place preceding task completion statement printed to console window
const char OWNER_ACCESS[]			= "744";												// Permissions to give owner rwx permissions but all other users only r permission to a folder/file
const char GROUP_ACCESS[]			= "774";												// Permissions to give owner and group rwx permissions but all other users only r permission to a folder/file
const char GLOBAL_ACCESS[]			= "777";												// Permissions to give everuone rwx permissions to a folder/file
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*--------------------------------------------------------------- Configuration and execution logging file names ----------------------------------------------------------*/
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
const char INPUT_DATA_BASENAME[]		= "projection";										// Base name of projection data files projection_xxx.bin for each gantry angle xxx
const char INPUT_DATA_EXTENSION[]		= ".bin";											// File extension of projection data files projection_xxx.bin for each gantry angle xxx
const char CONFIG_FILENAME[]			= "settings.cfg";									// Name of the file used to control the program options/parameters as key=value pairs
const char CONFIG_OUT_FILENAME[]		= "settings_log.cfg";								// Name of the file used to control the program options/parameters as key=value pairs
const char LOG_FILENAME[]				= "log.csv";										// Name of the file logging the execution information associated with each data set generated
const char STDOUT_FILENAME[]			= "stdout.txt";										// Name of the file where the standard output stream stdout is redirected
const char STDIN_FILENAME[]				= "stdin.txt";										// Name of the file where the standard input stream stdin is redirected
const char STDERR_FILENAME[]			= "stderr.txt";										// Name of the file where the standard error stream stderr is redirected
const char EXECUTION_TIMES_BASENAME[]	= "execution_times";								// Base name of global .csv and run specific .txt files specifying the execution times for various portions of preprocessing/recosntruction
const char SIN_TABLE_FILENAME[]			= "sin_table.bin";									// Prefix of the file containing the tabulated values of sine function for angles [0, 2PI]
const char COS_TABLE_FILENAME[]			= "cos_table.bin";									// Prefix of the file containing the tabulated values of cosine function for angles [0, 2PI]
const char COEFFICIENT_FILENAME[]		= "coefficient.bin";								// Prefix of the file containing the tabulated values of the scattering coefficient for u_2-u_1/u_1 values in increments of 0.001
const char POLY_1_2_FILENAME[]			= "poly_1_2.bin";									// Prefix of the file containing the tabulated values of the polynomial with coefficients {1,2,3,4,5,6} in increments of 0.001
const char POLY_2_3_FILENAME[]			= "poly_2_3.bin";									// Prefix of the file containing the tabulated values of the polynomial with coefficients {2,3,4,5,6,7} in increments of 0.001
const char POLY_3_4_FILENAME[]			= "poly_3_4.bin";									// Prefix of the file containing the tabulated values of the polynomial with coefficients {3,4,5,6,7,8} in increments of 0.001
const char POLY_2_6_FILENAME[]			= "poly_2_6.bin";									// Prefix of the file containing the tabulated values of the polynomial with coefficients {2,6,12,20,30,42} in increments of 0.001
const char POLY_3_12_FILENAME[]			= "poly_3_12.bin";									// Prefix of the file containing the tabulated values of the polynomial with coefficients {3,12,30,60,105,168} in increments of 0.001
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
const std::string workstation_1_hostname("tardis-student1.ecs.baylor.edu");					// Host name of workstation #1 at BRIC ay Baylor
const std::string workstation_2_hostname("tardis-student2.ecs.baylor.edu");					// Host name of workstation #2 at BRIC ay Baylor
const std::string kodiak_hostname("n130");													// Host name of Kodiak cluster's master node at BRIC ay Baylor
const std::string whartnell_hostname("whartnell");											// Host name of Tardis cluster's master node at BRIC ay Baylor
const std::string whartnell_ID("ecsn001");													// Host ID # of Tardis cluster's master node at BRIC ay Baylor
const std::string jpertwee_ID("ecsn003");													// Host ID # of Tardis cluster's compute node JPertwee at BRIC ay Baylor
const std::string jpertwee_hostname("jpertwee");											// Host name of Tardis cluster's compute node JPertwee at BRIC ay Baylor
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//------------------------------------------------------------------------------------ Bash commands ---------------------------------------------------------------------------------------/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
const char HOSTNAME_CMD[]			= "uname -n";											// Bash command to acquire the hostname
const char USERNAME_CMD[]			= "id -un";												// Bash command to acquire the user name logged in
const char USERID_CMD[]				= "id -u";												// Bash command to acquire the user ID logged in
const char GROUPNAME_CMD[]			= "id -gn";												// Bash command to acquire the group name of the user logged in
const char GROUPID_CMD[]			= "id -g";												// Bash command to acquire the group ID of the user logged in
const char GROUPNAMES_CMD[]			= "id -Gn";												// Bash command to acquire the all group names of the user logged in
const char GROUPIDS_CMD[]			= "id -G";												// Bash command to acquire the all group IDs of the user logged in
const char HOSTNAME_ENVVAR[]		= "echo $HOSTNAME";										// Bash command to return the host name environment variable
const char HOME_DIR_ENVVAR[]		= "echo $HOME";											// Bash command to return the home directory environment variable of user logged in
const char USERNAME_ENVVAR[]		= "echo $USER";											// Bash command to return the user name environment variable of user logged in
const char USERID_ENVVAR[]			= "echo $UID";											// Bash command to return the user ID environment variable of user logged in
const char USERID_EFFECT_ENVVAR[]	= "echo $EUID";											// Bash command to return the effective user ID environment variable of user logged in
const char GROUPS_ENVVAR[]			= "echo $GROUPS";										// Bash command to return the group names environment variable of user logged in
const char OSTYPE_ENVVAR[]			= "echo $OSTYPE";										// Bash command to return the operating system environment variable on system user is logged in to
const char PATH_ENVVAR[]			= "echo $PATH";											// Bash command to return the path environment variable for the various compilers/applications for user logged in
const char WORK_DIR_ENVVAR[]		= "echo $PWD";											// Bash command to return the current directory environment variable of user logged in
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//------------------------------------------------------------------------------------ Output folder names/paths associated with pCT data format ---------------------------------------------------------------------------------------/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
const char PCT_PATH_GLOBAL[]		= "//data//ion";										// Path to the pCT data/code in permanent storage on the network drive mounted and accessible by all Kodiak/Tardis nodes at Baylor
const char PCT_PATH_TARDIS[]		= "C://Users//Blake//Documents//Education//Research//pCT";											// Path to the pCT data/code stored locally for usage on the Tardis compute nodes at Baylor
//const char PCT_PATH_TARDIS[]		= "//local";											// Path to the pCT data/code stored locally for usage on the Tardis compute nodes at Baylor
const char PCT_DATA_FOLDER[]		= "pCT_data";											// Name of folder containing all of the pCT data
const char PCT_CODE_FOLDER[]		= "pCT_code";											// Name of folder containing all of the pCT code
const char RAW_DATA_FOLDER[]		= "raw_data";											// Name of folder in pCT_data directory containing the raw experimental data
const char PROCESSED_DATA_FOLDER[]	= "processed_data";										// Name of folder in pCT_data directory containing the preprocessed raw data
const char PROJECTION_DATA_FOLDER[]	= "projection_data";									// Name of folder in pCT_data directory containing the projection data used as input to reconstruction
const char RECON_DATA_FOLDER[]		= "reconstruction_data";								// Name of folder in pCT_data directory containing the reconstruction data/image
const char ORGANIZED_DATA_FOLDER[]	= "organized_data";										// Name of folder in pCT_data containing the organized data 
const char EXPERIMENTAL_FOLDER[]	= "Experimental";										// Name of folder in the organized_data directory containing experimental data 
const char SIMULATED_FOLDER[]		= "Simulated";											// Name of folder in the organized_data directory containing simulated data
const char INPUT_FOLDER[]			= "Input";												// Name of folder in the organized_data directory containing raw experimental data
const char OUTPUT_FOLDER[]			= "Output";												// Name of folder in the organized_data directory containing the projection data
const char RECONSTRUCTION_FOLDER[]	= "Reconstruction";										// Name of folder in the organized_data directory containing the reconstruction data	
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//------------------------------------------------------------------------ Column header names used in execution times files -------------------------------------------------------------------------/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
const char TESTED_BY_STRING[]		= "Blake Schultze";										// String to write to execution times files specifying the person that executed the reconstruction program
const char FULL_TX_STRING[]			= "FULL_TX";											// String to write to execution times files when full data transfer is used for MLP endpoints/MLP+DROP 
const char PARTIAL_TX_STRING[]		= "PARTIAL_TX";											// String to write to execution times files when partial data transfer is used for MLP endpoints/MLP+DROP 
const char PARTIAL_TX_PRE_STRING[]	= "PARTIAL_TX_PREALLOCATED";							// String to write to execution times files when partial data transfer reusing GPU arrays is used for MLP endpoints/MLP+DROP 
const char BOOL_ARRAY_STRING[]		= "BOOL_ARRAY";											// String to write to execution times files when boolean array is used to identify reconstruction histories in MLP endpoint routine
const char NO_BOOL_ARRAY_STRING[]	= "NO_BOOL_ARRAY";										// String to write to execution times files when no boolean array is used to identify reconstruction histories in MLP endpoint routine
const char STANDARD_STRING[]		= "STANDARD";											// String to write to execution times files when standard MLP routine is used for reconstruction
const char TABULATED_STRING[]		= "TABULATED";											// String to write to execution times files when MLP routine uses tabulated values for reconstruction
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//---------------------------------------------------------------------------------------- Output filenames ------------------------------------------------------------------------------------------/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
const char SC_HULL_FILENAME[]			= "SC_hull";										// Filename of the space/silhouette carving hull image
const char MSC_HULL_FILENAME[]			= "MSC_hull";										// Filename of the modified space/silhouette carving hull image
const char SM_HULL_FILENAME[]			= "SM_hull";										// Filename of the space/silhouette modeling hull image
const char FBP_HULL_FILENAME[]			= "FBP_hull";										// Filename of the filtered back projection hull image
const char SM_COUNTS_FILENAME[]			= "SM_counts";										// Filename of the space/silhouette carving hull image
const char MSC_COUNTS_FILENAME[]		= "MSC_counts";										// Filename of the space/silhouette carving hull image
const char HULL_FILENAME[]				= "hull";											// Filename of the hull image selected from the available hulls that is used in reconstruction 
const char HULL_AVG_FILTER_FILENAME[]	= "hull_avg_filtered";								// Filename of the hull image selected from the available hulls and average filtered that is used in reconstruction 
const char HULL_MED_FILTER_FILENAME[]	= "hull_median_filtered";							// Filename of the hull image selected from the available hulls and median filtered that is used in reconstruction 
const char SINOGRAM_FILENAME[]			= "sinogram";										// Filename of the sinogram image
const char FBP_FILENAME[]				= "FBP";											// Filename of the filtered back projection image
const char X_0_FILENAME[]				= "x_0";											// Filename of the initial iterate used in image reconstruction
const char X_FILENAME[]					= "x";
const char MLP_PATHS_FILENAME[]			= "MLPs";
const char MLP_PATHS_ERROR_FILENAME[]	= "MLP_error";
const char MLP_ENDPOINTS_FILENAME[]		= "MLP_endpoints";
const char INPUT_ITERATE_FILENAME[]		= "FBP_med7.bin";
const char IMPORT_FBP_FILENAME[]		= "FBP_med";
const char INPUT_HULL_FILENAME[]		= "input_hull.bin";
const char BIN_COUNTS_FILENAME[]		= "bin_counts_h_pre";
const char MEAN_WEPL_FILENAME[]			= "mean_WEPL_h";
const char MEAN_REL_UT_FILENAME[]		= "mean_rel_ut_angle_h";
const char MEAN_REL_UV_FILENAME[]		= "mean_rel_uv_angle_h";
const char STDDEV_REL_UT_FILENAME[]		= "stddev_rel_ut_angle_h";
const char STDDEV_REL_UV_FILENAME[]		= "stddev_rel_uv_angle_h";
const char STDDEV_WEPL_FILENAME[]		= "stddev_WEPL_h";
const char BIN_COUNTS_PRE_FILENAME[]	= "bin_counts_pre";
const char SINOGRAM_PRE_FILENAME[]		= "sinogram_pre";
const char BIN_COUNTS_POST_FILENAME[]	= "bin_counts_post";
const char FBP_AFTER_FILENAME[]			= "FBP_after";
const char FBP_IMAGE_FILTER_FILENAME[]	= "FBP_image_filtered";
const char FBP_MED_FILTER_FILENAME[]	= "FBP_median_filtered";
/*****************************************************************************************************************************************************************************************************/
/*****************************************************************************************************************************************************************************************************/
/****************************************************************************************** Global Variables *****************************************************************************************/
/*****************************************************************************************************************************************************************************************************/
/*****************************************************************************************************************************************************************************************************/
unsigned int num_run_arguments;
char** run_arguments;
char print_statement[256];
cudaError_t cudaStatus;	
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
// CTP_404
// 20367505: 32955315 -> 32955313									// No average filtering on hull
// 20367499: 32955306 ->32955301									// No average filtering on hull
// 20573129: 5143282->5143291										// r=1
// 20648251: 33409572->33409567										// r=2
// 20648257: 33409582 -> 33409577/33409603//5162071;				// r=2
// 20764061: 33596956->33596939/33596977							// r=3 

// CTP_404M
// 20153778: 5038452-> 5038457										// r=1
//ULL NUM_RECON_HISTORIES = 20574733
ULL NUM_RECON_HISTORIES =105642524;
ULL PRIME_OFFSET = 26410633;
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//-------------------------------------------------- Declaration of arrays number of histories per file, projection, angle, total, and translation ---------------------------------------------------/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
int total_histories = 0, recon_vol_histories = 0, maximum_histories_per_file = 0;
int* histories_per_projection, * histories_per_gantry_angle, * histories_per_file;
int* recon_vol_histories_per_projection;
int histories_per_scan[NUM_SCANS];
int post_cut_histories = 0;
int reconstruction_histories = 0;
int zero_WEPL = 0;
int zero_WEPL_files = 0;
double percentage_pass_each_intersection_cut, percentage_pass_intersection_cuts, percentage_pass_statistical_cuts, percentage_pass_hull_cuts;
ULL* history_sequence;
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//--------------------------------------------------------- Declaration of array used to store tracking plane distances from rotation axis -----------------------------------------------------------/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
std::vector<float> projection_angles;
float SSD_u_Positions[8];
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//----------------------------------------------------------------------------------- Execution timing variables -------------------------------------------------------------------------------------/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
clock_t program_start, program_end, pause_cycles = 0;
clock_t begin_endpoints = 0, begin_init_image = 0, begin_tables = 0, begin_DROP_iteration = 0, begin_DROP = 0, begin_update_calcs = 0, begin_update_image = 0, begin_data_reads = 0, begin_preprocessing = 0, begin_reconstruction = 0, begin_program = 0;
double execution_time_endpoints = 0, execution_time_init_image = 0, execution_time_DROP_iteration = 0, execution_time_DROP = 0, execution_time_update_calcs = 0, execution_time_update_image = 0, execution_time_tables = 0;
double execution_time_data_reads = 0, execution_time_preprocessing = 0, execution_time_reconstruction = 0, execution_time_program = 0; 
std::vector<double> execution_times_DROP_iterations;
FILE* execution_times_file;
/*****************************************************************************************************************************************************************************************************/
/*****************************************************************************************************************************************************************************************************/
/************************************************************************************ Global Array Declerations **************************************************************************************/
/*****************************************************************************************************************************************************************************************************/
/*****************************************************************************************************************************************************************************************************/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//----------------------------------------------------------------- Declaration of image arrays for use on host(_h) or device (_d) -------------------------------------------------------------------/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
FILE* sin_table_file, * cos_table_file, * scattering_table_file, * poly_1_2_file, * poly_2_3_file, * poly_3_4_file, * poly_2_6_file, * poly_3_12_file;
double* sin_table_h, * cos_table_h, * scattering_table_h, * poly_1_2_h, * poly_2_3_h, * poly_3_4_h, * poly_2_6_h, * poly_3_12_h;
double* sin_table_d, * cos_table_d, * scattering_table_d, * poly_1_2_d, * poly_2_3_d, * poly_3_4_d, * poly_2_6_d, * poly_3_12_d;
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//------------------------------------------------------------ Declaration of arrays for storage of input data for use on the host (_h) --------------------------------------------------------------/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
int* gantry_angle_h, * bin_num_h, * bin_counts_h;
bool* missed_recon_volume_h, * failed_cuts_h;
float* t_in_1_h, * t_in_2_h, * t_out_1_h, * t_out_2_h;
float* u_in_1_h, * u_in_2_h, * u_out_1_h, * u_out_2_h;
float* v_in_1_h, * v_in_2_h, * v_out_1_h, * v_out_2_h;
float* ut_entry_angle_h, * ut_exit_angle_h;
float* uv_entry_angle_h, * uv_exit_angle_h;
float* x_entry_h, * y_entry_h, * z_entry_h;
float* x_exit_h, * y_exit_h, * z_exit_h;
float* xy_entry_angle_h, * xy_exit_angle_h;
float* xz_entry_angle_h, * xz_exit_angle_h;
float* WEPL_h;
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//----------------------------------------------------------- Declaration of arrays for storage of input data for use on the device (_d) -------------------------------------------------------------/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
int* gantry_angle_d, * bin_num_d, * bin_counts_d;
bool* missed_recon_volume_d, * failed_cuts_d;
float* t_in_1_d, * t_in_2_d, * t_out_1_d, * t_out_2_d;
float* u_in_1_d, * u_in_2_d, * u_out_1_d, * u_out_2_d;
float* v_in_1_d, * v_in_2_d, * v_out_1_d, * v_out_2_d;
float* ut_entry_angle_d, * ut_exit_angle_d;
float* uv_entry_angle_d, * uv_exit_angle_d;
float* x_entry_d, * y_entry_d, * z_entry_d;
float* x_exit_d, * y_exit_d, * z_exit_d;
float* xy_entry_angle_d, * xy_exit_angle_d;
float* xz_entry_angle_d, * xz_exit_angle_d;
float* WEPL_d;
unsigned int* first_MLP_voxel_d;
int* voxel_x_d, voxel_y_d, voxel_z_d;
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//--------------------------------------------------------- Declaration of statistical analysis arrays for use on host(_h) or device (_d) ------------------------------------------------------------/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
float* ut_entry_angle, * uv_entry_angle, * ut_exit_angle, * uv_exit_angle; 
float* mean_WEPL_h, * mean_WEPL_d;
float* mean_energy_h, * mean_energy_d;
float* mean_rel_ut_angle_h, * mean_rel_ut_angle_d;
float* mean_rel_uv_angle_h, * mean_rel_uv_angle_d;
float* mean_total_rel_angle_h, * mean_total_rel_angle_d;
float* stddev_rel_ut_angle_h, * stddev_rel_ut_angle_d;
float* stddev_rel_uv_angle_h, * stddev_rel_uv_angle_d;
float* stddev_WEPL_h, * stddev_WEPL_d;
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//------------------------------------------------------- Declaration of pre/post filter sinogram for FBP for use on host(_h) or device (_d) ---------------------------------------------------------/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
float* sinogram_h, * sinogram_d;
float* sinogram_filtered_h, * sinogram_filtered_d;
bool* SC_hull_h, * SC_hull_d;
bool* MSC_hull_h, * MSC_hull_d;
bool* SM_hull_h, * SM_hull_d;
bool* FBP_hull_h, * FBP_hull_d;
bool* hull_h, * hull_d;
int* MSC_counts_h, * MSC_counts_d;
int* SM_counts_h, * SM_counts_d;
int* MLP_test_image_h, * MLP_test_image_d;
float* FBP_image_h, * FBP_image_d;
float* FBP_image_filtered_h, * FBP_image_filtered_d;
float* FBP_median_filtered_h, * FBP_median_filtered_d;
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//------------------------------------------------------------------ Declaration of DROP arrays for use on host(_h) or device (_d) -------------------------------------------------------------------/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
bool* traversed_hull_h, * traversed_hull_d;													// Indicates whether the protons entered and exited (traversed) the object hull and should be used in image reconstruction
float* x_h, * x_d;																			// The target image x containing RSP values for each voxel in the object and 0 for each voxel outside the object																// 
float* x_update_h, * x_update_d;															// Update value calculated each iteration which is then applied to the image x
unsigned int* S_h, * S_d;																	// Counts of how many times each voxel was intersected by a proton in the current DROP block
unsigned int* MLP_h, * MLP_d;																// Voxels intersected along the MLP path of the current proton
unsigned int* MLP_block_h, * MLP_block_d;													// Voxels intersected along the MLP paths of protons in the current DROP block
float* A_ij_h, * A_ij_d;																	// Chord length of the intersection of proton i with voxel j, i.e., the ith row and jth column of A = A(i,j)
double* norm_Ai;																			// L2 norm of row i of A matrix, i.e., Ai 

float* G_x_h, * G_y_h, * G_norm_h, * G_h, * v_h, * y_h;
float* G_x_d, * G_y_d, * G_norm_d, * G_d, * v_d, * y_d;
float BETA = 1.0;
float* TV_x_h, * TV_y_h;
float* TV_x_d, * TV_y_d;
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//--------------------------------------------- Declaration of vectors used to accumulate data from histories that have passed currently applied cuts ------------------------------------------------/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
std::vector<int>	bin_num_vector;			
std::vector<int>	gantry_angle_vector;	
std::vector<float>	WEPL_vector;		
std::vector<float>	x_entry_vector;		
std::vector<float>	y_entry_vector;		
std::vector<float>	z_entry_vector;		
std::vector<float>	x_exit_vector;			
std::vector<float>	y_exit_vector;			
std::vector<float>	z_exit_vector;			
std::vector<float>	xy_entry_angle_vector;	
std::vector<float>	xz_entry_angle_vector;	
std::vector<float>	xy_exit_angle_vector;	
std::vector<float>	xz_exit_angle_vector;	
std::vector<UINT>	first_MLP_voxel_vector;
std::vector<int>	voxel_x_vector;
std::vector<int>	voxel_y_vector;
std::vector<int>	voxel_z_vector;
/*****************************************************************************************************************************************************************************************************/
/*****************************************************************************************************************************************************************************************************/
/*********************************************************************************** End of Parameter Definitions ************************************************************************************/
/*****************************************************************************************************************************************************************************************************/
/*****************************************************************************************************************************************************************************************************/
//#endif // _PCT_RECONSTRUCTION_H_
