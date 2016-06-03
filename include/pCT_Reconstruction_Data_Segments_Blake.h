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
#include <chrono>		// chronology: system_clock::now().time_since_epoch().count(), duration, duration_cast
#include <cmath>
#include <cstdarg>		// variable arguments: va_list, va_arg, va_start, va_end, va_copy
#include <cstdio>		// printf, sprintf,  
#include <cstdlib>		// standard library: rand, srand
#include <cstring>
#include <ctime>		// clock(), time() 
#include <fstream>
#include <functional>	// std::multiplies, std::plus, std::function, std::negate
#include <iostream>
#include <limits>		// Numeric limits of fundamental data types
#include <map>			// Mapping provides ability to access an element's mapped value using the element's unique key value identifier 
#include <new>			// dynamic memory allocation/destruction
#include <numeric>		// inner_product, partial_sum, adjacent_difference, accumulate
#include <omp.h>		// OpenMP
#include <random>		// uniform_int_distribution, uniform_real_distribution, bernoulli_distribution, binomial_distribution, geometric_distribution
#include <sstream>		// string stream
#include <string>
#include "sys/types.h"	// stat f
#include "sys/stat.h"	// stat functions
#include <typeinfo>		//operator typeid
#include <utility>		// for std::move
#include <vector>

#if defined(_WIN32) || defined(_WIN64)
	#include "Shlwapi.h"
	#include <windows.h>
	#include <direct.h>
#else
	#include <unistd.h>
#endif
#ifdef __GNUG__

    #include <cxxabi.h>
    #include <cstdlib>
    #include <memory>

    template< typename T > std::string type_name()
    {
        int status ;
        std::unique_ptr< char[], decltype(&std::free) > buffer(
            __cxxabiv1::__cxa_demangle( typeid(T).name(), nullptr, 0, &status ), &std::free ) ;
        return status==0 ? buffer.get() : "__cxa_demangle error" ;
    }

#else // !defined __GNUG__

    template< typename T > std::string type_name() { return typeid(T).name() ; }

#endif //__GNUG__


/*****************************************************************************************************************************************************************************************************/
/*****************************************************************************************************************************************************************************************************/
/*********************************************************************************** Preprocessing usage options *************************************************************************************/
/*****************************************************************************************************************************************************************************************************/
/*****************************************************************************************************************************************************************************************************/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//----------------------------------------------------------------------- Namespace selections and typedef (data type alias) definitions  ----------------------------------------------------------------------/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//using namespace std;
using std::cout;
using std::endl;
using std::chrono::system_clock;
typedef unsigned long long ULL;
typedef unsigned int UINT;
typedef unsigned int uint;
typedef std::vector<UINT>::iterator block_iterator;
#define ON						(0==0)									// [T/F] Definition of boolean "ON" as equivalent to 'true'
#define OFF						(!ON)									// [T/F] Definition of boolean "OFF" as equivalent to 'false' (technically 'not true')
#define START					true									// [T/F] Used as an alias for true for starting timer
#define STOP					false									// [T/F] Used as an alias for false for stopping timer
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//----------------------------------------------------------------------- Struct definitions and global variable instantiations ----------------------------------------------------------------------/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
#define SOME_ENUM(DO) \
    DO(Foo) \
    DO(Bar) \
    DO(Baz)

#define MAKE_ENUM(VAR) VAR,
enum MetaSyntacticVariable{
    SOME_ENUM(MAKE_ENUM)
};

#define MAKE_CSTRINGS(VAR) #VAR,
const char* const MetaSyntacticVariableNames[] = {
    SOME_ENUM(MAKE_CSTRINGS)
};
	
enum DATA_FORMATS			{ OLD_FORMAT, VERSION_0, VERSION_1, VERSION_0_W_ANGLES, END_DATA_FORMATS							};				// Define the data formats that are supported
enum SCAN_TYPES				{ EXPERIMENTAL, SIMULATED_G, SIMULATED_T, END_SCAN_TYPES						};				// Experimental or simulated (GEANT4 or TOPAS) data
enum FILE_TYPES				{ TEXT, BINARY, END_FILE_TYPES													};				// Experimental or simulated data
enum RAND_GENERATORS		{ DEFAULT_RAND, MINSTD_RAND, MINSTD_RAND0, MT19937,											// Defines the available random number generator engines 
								MT19937_64, RANLUX24, RANLUX48, KNUTH_B, END_RAND_GENERATORS				};				// ...
enum IMAGE_BASES			{ VOXELS, BLOBS, END_IMAGE_BASES												};				// Defines whether images are formed using voxels or blobs as the basis elements
enum BIN_ANALYSIS_TYPE		{ MEANS, COUNTS, MEMBERS, END_BIN_ANALYSIS_TYPE									};				// Choices for what information about the binned data is desired 
enum BIN_ANALYSIS_FOR		{ ALL_BINS, SPECIFIC_BINS, END_BIN_ANALYSIS_FOR									};				// Choices for which bins the desired data should come from
enum BIN_ORGANIZATION		{ BY_BIN, BY_HISTORY, END_BIN_ORGANIZATION										};				// Binned data is either organized in order by bin or by history w/ bin # specified separately
enum BIN_ANALYSIS_OF		{ WEPLS, ANGLES, POSITIONS, BIN_NUMS, END_BIN_ANALYSIS_OF						};				// Choices for which type of binned data is desired
enum FBP_FILTER_TYPES		{ RAM_LAK, SHEPP_LOGAN, UNFILTERED, END_FBP_FILTER_TYPES						};				// Define the types of filters that are available for use in FBP
enum IMAGE_FILTERING_OPTIONS{ NO_FILTER, MEDIAN, AVERAGE, MED_2_AVG, AVG_2_MED, END_IMAGE_FILTERING_OPTIONS	};
enum HULL_TYPES				{ SC_HULL, MSC_HULL, SM_HULL, FBP_HULL, END_HULL_TYPES							};				// Define valid choices for which hull to use in MLP calculations
enum INITIAL_ITERATE		{ X_HULL, FBP_IMAGE, HYBRID, ZEROS, IMPORT, END_INITIAL_ITERATE					};				// Define valid choices for which hull to use in MLP calculations
enum TX_OPTIONS				{ FULL_TX, PARTIAL_TX, PARTIAL_TX_PREALLOCATED, END_TX_OPTIONS					};				// Define valid choices for the host->GPU data transfer method
enum ENDPOINTS_ALGORITHMS	{ YES_BOOL, NO_BOOL, END_ENDPOINTS_ALGORITHMS									};				// Define the method used to identify protons that miss/hit the hull in MLP endpoints calculations
enum MLP_ALGORITHMS			{ STANDARD, TABULATED, END_MLP_ALGORITHMS										};				// Define whether standard explicit calculations or lookup tables are used in MLP calculations
enum PROJECTION_ALGORITHMS	{ ART, SART, DROP, BIP, SAP, ROBUSTA, ROBUSTB, END_PROJECTION_ALGORITHMS		};				// Define valid choices for iterative projection algorithm to use
enum S_CURVES				{ SIGMOID, TANH, ATAN, ERF, LIN_OVER_ROOT, END_S_CURVES							};				// Define valid choices for iterative projection algorithm to use
enum ROBUST_METHODS			{ OLS, TLS, TIKHONOV, RIDGE, MINMIN, MINMAX, END_ROBUST_METHODS					};				// Defines the robust regression methods available for robust image reconstruction
enum STRUCTURAL_ELEMENTS	{ SQUARE, DISK, END_STRUCTURAL_ELEMENTS											};				// Defines the structural elements available to the morphological operators
enum MORPHOLOGICAL_OPS		{ ERODE, DILATE, OPEN, CLOSE, END_MORPHOLOGICAL_OPS								};				// Defines the mathematical morphology operators available for image processing 
enum LOG_ENTRIES			{ OBJECT_L, SCAN_TYPE_L, RUN_DATE_L, RUN_NUMBER_L,											// Define the headings of each column in the execution log 
							ACQUIRED_BY_L, PROJECTION_DATA_DATE_L, CALIBRATED_BY_L, 									// ...	 	
							PREPROCESS_DATE_L, PREPROCESSED_BY_L, RECONSTRUCTION_DATE_L,								// ...
							RECONSTRUCTED_BY_L, COMMENTS_L, END_LOG_ENTRIES									};				// ...
enum CODE_SOURCES			{ LOCAL, GLOBAL, USER_HOME, GROUP_HOME, END_CODE_SOURCES						};				// Define the data formats that are supported
enum HISTORY_ORDERING		{ SEQUENTIAL, PRIME_PERMUTATION, END_HISTORY_ORDERING							};				// Define the data formats that are supported
enum BLOCK_ORDERING			{ CYCLIC, ROTATE_LEFT, ROTATE_RIGHT, RANDOMLY_SHUFFLE, END_BLOCK_ORDERING					};				// Define the data formats that are supported
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//----------------------------------------------------------------------- Struct definitions and global variable instantiations ----------------------------------------------------------------------/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
// Value of a key/value pair having unknown target data type read from disk and interpreted as all 3 primary data types, allowing this to be imposed later
struct generic_IO_container
{
	char* key;						// Stores key string of a key/value pair read from file
	unsigned int input_type_ID;		// ID (1,2,3) corresponding to data type assumed based on value format (e.g. w or w/o '"'s, '.', etc.)
	int integer_input;				// type_ID = 1
	double double_input;			// type_ID = 2
	char string_input[512];			// type_ID = 3
};
generic_IO_container config_file_input;
// Container for all config file specified configurations allowing these to be transferred to the GPU with a single cudamemcpy command statement
// 8 UI, 18D, 6 C*
struct configurations
{
	double lambda;
};
configurations parameter_container;
configurations *configurations_h = &parameter_container;
configurations *configurations_d;
/*****************************************************************************************************************************************************************************************************/
/*****************************************************************************************************************************************************************************************************/
/*********************************************************************************** Preprocessing usage options *************************************************************************************/
/*****************************************************************************************************************************************************************************************************/
/*****************************************************************************************************************************************************************************************************/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//------------------------------------------------------------------------------ Testing configurations/options controls ---------------------------------------------------------------------------------/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
const RAND_GENERATORS		RAND_ENGINE				= DEFAULT_RAND;					// Specify the random number generator engine to use
const RAND_GENERATORS		TVS_RAND_ENGINE			= DEFAULT_RAND;					// Specify the random number generator engine to use
//const SCAN_TYPES			SCAN_TYPE				= SIMULATED_G;			  		// Specifies which of the defined filters will be used in FBP
////const SCAN_TYPES			SCAN_TYPE				= EXPERIMENTAL;			  		// Specifies which of the defined filters will be used in FBP
const FILE_TYPES			FILE_TYPE				= BINARY;						// Experimental or simulated data
const DATA_FORMATS			DATA_FORMAT				= VERSION_0;					// Specify which data format to use for this run
const IMAGE_BASES			IMAGE_BASIS				= VOXELS;						// Specifies which basis is used to construct the images
const FBP_FILTER_TYPES		SINOGRAM_FILTER			= SHEPP_LOGAN;			  		// Specifies which of the defined filters will be used in FBP
const IMAGE_FILTERING_OPTIONS	FBP_FILTER			= MEDIAN;			  		// Specifies which of the defined filters will be used in FBP
const IMAGE_FILTERING_OPTIONS	HULL_FILTER			= NO_FILTER;			  		// Specifies which of the defined filters will be used in FBP
const IMAGE_FILTERING_OPTIONS	X_0_FILTER			= NO_FILTER;			  		// Specifies which of the defined filters will be used in FBP
const HULL_TYPES			ENDPOINTS_HULL			= MSC_HULL;						// Specify which of the HULL_TYPES to use in this run's MLP calculations
const INITIAL_ITERATE		X_0						= HYBRID;						// Specify which of the HULL_TYPES to use in this run's MLP calculations
const ENDPOINTS_ALGORITHMS	ENDPOINTS_ALG			= NO_BOOL;						// Specifies if boolean array is used to store whether a proton hit/missed the hull (BOOL) or uses the 1st MLP voxel (NO_BOOL)
const TX_OPTIONS			ENDPOINTS_TX_MODE		= PARTIAL_TX_PREALLOCATED;		// Specifies GPU data tx mode for MLP endpoints as all data (FULL_TX), portions of data (PARTIAL_TX), or portions of data w/ reused GPU arrays (PARTIAL_TX_PREALLOCATED)
const MLP_ALGORITHMS		MLP_ALGORITHM			= TABULATED;					// Specifies whether calculations are performed explicitly (STANDARD) or if lookup tables are used for MLP calculations (TABULATED)
const PROJECTION_ALGORITHMS	PROJECTION_ALGORITHM	= DROP;							// Specify which of the projection algorithms to use for image reconstruction
const TX_OPTIONS			RECON_TX_MODE			= FULL_TX;						// Specifies GPU data tx mode for MLP+DROP as all data (FULL_TX), portions of data (PARTIAL_TX), or portions of data w/ reused GPU arrays (PARTIAL_TX_PREALLOCATED)
const TX_OPTIONS			DROP_TX_MODE			= FULL_TX;						// Specifies GPU data tx mode for MLP+DROP as all data (FULL_TX), portions of data (PARTIAL_TX), or portions of data w/ reused GPU arrays (PARTIAL_TX_PREALLOCATED)
const S_CURVES				S_CURVE					= SIGMOID;						// Specify S-curve to use to scale updates applied to voxels around the object boundary
const ROBUST_METHODS		ROBUST_METHOD			= TIKHONOV;						// Specifies robust method used in robust image reconstruction
const CODE_SOURCES			CODE_SOURCE				= LOCAL;						// Specify the random number generator engine to use
const HISTORY_ORDERING		RECON_HISTORY_ORDERING	= SEQUENTIAL;
const BLOCK_ORDERING		DROP_BLOCK_ORDER		= CYCLIC;
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//-------------------------------------------------------------------------------- Execution and early exit options ----------------------------------------------------------------------------------/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//const bool RECONSTRUCTION_ON			= false;									// Specify whether execution proceeds with reconstruction (T) or with test_func() for independent code testing (F)
const bool RUN_ON						= true;										// Specify whether execution proceeds with reconstruction (T) or with test_func() for independent code testing (F)
const bool FUNCTION_TESTING				= OFF;										// Write output to "testing" directory (T) or to organized dat directory (F)
const bool USING_RSYNC					= false;									// Specify whether execution proceeds with reconstruction (T) or with test_func() for independent code testing (F)
const bool SHARE_OUTPUT_DATA			= ON;										// Specify whether data should be copied to shared (T) or personal (F) the network-attached storage device
const bool USE_GIT_CODE					= true;
const bool USE_GROUP_CODE				= false;
const bool EXIT_AFTER_BINNING			= false;									// Exit program early after completing data read and initial processing
const bool EXIT_AFTER_HULLS				= false;									// Exit program early after completing hull-detection
const bool EXIT_AFTER_CUTS				= false;									// Exit program early after completing statistical cuts
const bool EXIT_AFTER_SINOGRAM			= false;									// Exit program early after completing the construction of the sinogram
const bool EXIT_AFTER_FBP				= false;									// Exit program early after completing FBP
const bool EXIT_AFTER_X_O				= false;									// Exit program early after completing FBP
const bool CLOSE_AFTER_EXECUTION		= true;										// Exit program early after completing FBP
const bool DEBUG_TEXT_ON				= true;										// Provide (T) or suppress (F) print statements to console during execution
const bool PRINT_ALL_PATHS				= false;
const bool PRINT_CHMOD_CHANGES_ONLY		= false;

const bool OVERWRITING_OK				= false;									// Allow output to 
const bool TESTING_ON					= false;									// Write output to "testing" directory (T) or to organized dat directory (F)
const bool CUTS_TESTING_ON				= true;										// Validating proper FBP filtering name output directory (T) or not (F)
const bool RECON_VOLUME_TESTING_ON		= true;									// Validating proper FBP filtering name output directory (T) or not (F)
const bool RECON_PARAMETER_TESTING_ON	= true;										// Use value of variables relevant to block testing to name output directory (T) or not (F)
const bool FBP_TESTING_ON				= false;									// Validating proper FBP filtering name output directory (T) or not (F)
const bool FILTER_TESTING_ON			= false;										// Validating proper FBP filtering name output directory (T) or not (F)
const bool AIR_THRESH_TESTING_ON		= true;										// Use value of variables relevant to block testing to name output directory (T) or not (F)
const bool MLP_LENGTH_TESTING_ON		= false;									// Use value of variables relevant to block testing to name output directory (T) or not (F)
const bool BLOCK_TESTING_ON				= false;										// Use value of variables relevant to block testing to name output directory (T) or not (F)
const bool NTVS_TESTING_ON				= false;									// Use value of variables relevant to block testing to name output directory (T) or not (F)
const bool OLD_TVS_TESTING_ON			= false;									// Use value of variables relevant to block testing to name output directory (T) or not (F)
const bool OLD_TVS_COMPARISON_TESTING_ON= false;									// Use value of variables relevant to block testing to name output directory (T) or not (F)
const bool WITH_OPTIMAL_NTVS_TESTING_ON	= true;										// Use value of variables relevant to block testing to name output directory (T) or not (F)
const bool S_CURVE_TESTING_ON			= false;									// Use value of variables relevant to block testing to name output directory (T) or not (F)
const bool ANGULAR_BIN_TESTING_ON		= true;										// Use value of variables relevant to block testing to name output directory (T) or not (F)

//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//--------------------------------------------------------------------------------- Preprocessing option configurations ----------------------------------------------------------------------------------/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
const bool SAMPLE_STD_DEV				= true;										// Use sample/population standard deviation (T/F) in statistical cuts (i.e. divisor is N/N-1)
const bool FBP_ON						= true;										// Turn FBP on (T) or off (F)
const bool IMPORT_FILTERED_FBP			= false;									// Import FBP image that was filtered externally from disk (T) or not (F)
const bool SC_ON						= false;									// Turn Space Carving on (T) or off (F)
const bool MSC_ON						= true;										// Turn Modified Space Carving on (T) or off (F)
const bool SM_ON						= false;									// Turn Space Modeling on (T) or off (F)
const bool COUNT_0_WEPLS				= false;									// Count the number of histories with WEPL = 0 (T) or not (F)
const bool DIRECT_IMAGE_RECONSTRUCTION	= false;									// Begin execution with reconstruction by importing existing preprocessing data from disk (T) or generate preprocessing data first (F)
const bool MLP_FILE_EXISTS				= false;									// Specifies whether a file with the relevant MLP data exists (T) or not (F)
const bool MLP_ENDPOINTS_FILE_EXISTS	= false;									// Specifies whether a file with the relevant MLP endpoint data exists (T) or not (F)
const bool IMPORT_MLP_LOOKUP_TABLES		= false;									// Import MLP lookup tables instead of generating them at run time
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//---------- Control if mutually exclusive code is selectively compiled w/ precompiler directives (PCDs) or conditionally executed via branching -----------------------------------------------------/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/	

// Precompiler directive controls
const bool DROP_BRANCHING = true;
const bool PCD_DROP = true && !DROP_BRANCHING;

// Branching controls
const bool PCD_DROP_FULL_TX = true;
const bool PCD_DROP_PARTIAL_TX = true && !PCD_DROP_PARTIAL_TX;
const bool PCD_DROP_PARTIAL_TX_PREALLOCATED = true && !PCD_DROP_PARTIAL_TX;

/*****************************************************************************************************************************************************************************************************/
/*****************************************************************************************************************************************************************************************************/
/****************************************************************************** Input/output specifications and options ******************************************************************************/
/*****************************************************************************************************************************************************************************************************/
/*****************************************************************************************************************************************************************************************************/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//------------------------------------------------------------ Definition of character arrays for path variables known at compile time ---------------------------------------------------------------/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
const char INPUT_DIRECTORY[]	= "//local//pCT_data//organized_data//";
const char OUTPUT_DIRECTORY[]	= "//local//pCT_data//reconstruction_data//";

//const SCAN_TYPES				SCAN_TYPE				= SIMULATED_G;			  		// Specifies which of the defined filters will be used in FBP
//const char INPUT_FOLDER[]		= "CTP404_Sensitom//Simulated//G_15-05-24//0001//Output//15-05-24";
//const char OUTPUT_FOLDER[]      = "CTP404_Sensitom//Simulated//G_15-05-24//0001//Output//15-05-24";
//const char PHANTOM_NAME[]		= "CTP404_Sensitom";
//const char RUN_DATE[]			= "15-05-24";
//const char RUN_NUMBER[]			= "0001";
//const char PREPROCESS_DATE[]	= "15-05-24";

const SCAN_TYPES				SCAN_TYPE				= EXPERIMENTAL;			  		// Specifies which of the defined filters will be used in FBP
//const char INPUT_FOLDER[]		= "Water//Experimental//16-04-23//0069_Cont//Output//16-05-24.B";
//const char OUTPUT_FOLDER[]		= "Water//Experimental//16-04-23//0069_Cont//Output//16-05-24.B";
//const char INPUT_FOLDER[]		= "Water//Experimental//15-05-16//0058//Output//15-06-25";
//const char OUTPUT_FOLDER[]		= "Water//Experimental//15-05-16//0058//Output//15-06-25";
const char INPUT_FOLDER[]		= "Water//Experimental//15-05-16//0058//Output//16-03-14";
const char OUTPUT_FOLDER[]		= "Water//Experimental//15-05-16//0058//Output//16-03-14";
//const char INPUT_FOLDER[]		= "Water//Experimental//16-04-23//0069_Cont//Output//16-04-23";
//const char OUTPUT_FOLDER[]		= "Water//Experimental//16-04-23//0069_Cont//Output//16-04-23";

const char PHANTOM_NAME[]		= "Water";
const char RUN_NUMBER[]			= "0058";
const char RUN_DATE[]			= "15-05-16";
const char PREPROCESS_DATE[]	= "16-03-14";

//const char RUN_NUMBER[]			= "0069_Cont";
//const char RUN_DATE[]			= "16-04-23";
//const char PREPROCESS_DATE[]	= "16-05-24.B";

//const char RUN_NUMBER[]			= "0058";
//const char RUN_DATE[]			= "15-05-16";
//const char PREPROCESS_DATE[]	= "16-04-23";
//const char PREPROCESS_DATE[]	= "15-06-25";

const bool USE_CONT_ANGLES			= false;
const bool UPDATED_FBP			= true;

//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//--------------------------------------------------------------------------- Iterative Image Reconstruction Parameters ------------------------------------------------------------------------------/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//----------------------------------------------------------------------------- Input data specification configurations ----------------------------------------------------------------------------------/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
const bool BINARY_ENCODING	   = true;									// Input data provided in binary (T) encoded files or ASCI text files (F)
const bool SINGLE_DATA_FILE    = false;									// Individual file for each gantry angle (T) or single data file for all data (F)
const bool SSD_IN_MM		   = true;									// SSD distances from rotation axis given in mm (T) or cm (F)
const bool DATA_IN_MM		   = true;									// Input data given in mm (T) or cm (F)
const bool MICAH_SIM		   = false;									// Specify whether the input data is from Micah's simulator (T) or not (F)
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//------------------------------------------------------------------------------------ Output option configurations --------------------------------------------------------------------------------------/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
const bool WRITE_BIN_WEPLS		= false;								// Write WEPLs for each bin to disk (T) for WEPL distribution analysis, or do not (F)
const bool WRITE_WEPL_DISTS		= false;								// Write mean WEPL values to disk (T) or not (F): t bin = columns, v bin = rows, 1 angle per file
const bool WRITE_SSD_ANGLES		= false;								// Write angles for each proton through entry/exit tracker planes to disk (T), or do not (F) 
const bool WRITE_SC_HULL		= false;									// Write SC hull to disk (T) or not (F)
const bool WRITE_MSC_COUNTS		= true;									// Write MSC counts array to disk (T) or not (F) before performing edge detection 
const bool WRITE_MSC_HULL		= false;									// Write MSC hull to disk (T) or not (F)
const bool WRITE_SM_COUNTS		= false;									// Write SM counts array to disk (T) or not (F) before performing edge detection 
const bool WRITE_SM_HULL		= false;									// Write SM hull to disk (T) or not (F)
const bool WRITE_FBP_IMAGE		= true;									// Write FBP image before thresholding to disk (T) or not (F)
const bool WRITE_FBP_HULL		= false;									// Write FBP hull to disk (T) or not (F)
const bool WRITE_AVG_FBP		= false;								// Write average filtered FBP image before thresholding to disk (T) or not (F)
const bool WRITE_MEDIAN_FBP		= true;									// Write median filtered FBP image to disk (T) or not (F)
const bool WRITE_FILTERED_HULL	= false;									// Write average filtered FBP image to disk (T) or not (F)
const bool WRITE_X_HULL			= true;									// Write the hull selected to be used in MLP calculations to disk (T) or not (F)
const bool WRITE_X_0			= true;									// Write the hull selected to be used in MLP calculations to disk (T) or not (F)
const bool WRITE_X_KI			= true;									// Write the hull selected to be used in MLP calculations to disk (T) or not (F)
const bool WRITE_X				= false;								// Write the reconstructed image to disk (T) or not (F)
const bool WRITE_MLP_TABLES		= false;								// Write angles for each proton through entry/exit tracker planes to disk (T), or do not (F) 
/*****************************************************************************************************************************************************************************************************/
/*****************************************************************************************************************************************************************************************************/
/************************************************************************************** Preprocessing Constants **************************************************************************************/
/*****************************************************************************************************************************************************************************************************/
/*****************************************************************************************************************************************************************************************************/			
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//------------------------------------------------------------------------- Host/GPU computation and structure information ---------------------------------------------------------------------------/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
#define THREADS_PER_BLOCK		1024									// [#] # of threads per GPU block for preprocessing kernels
#define ENDPOINTS_PER_BLOCK 	320										// [#] # of threads per GPU block for collect_MLP_endpoints_GPU kernel
#define HISTORIES_PER_BLOCK 	640									// [#] # of threads per GPU block for block_update_GPU kernel
#define ENDPOINTS_PER_THREAD 	1										// [#] # of MLP endpoints each thread is responsible for calculating in collect_MLP_endpoints_GPU kernel
#define HISTORIES_PER_THREAD 	1										// [#] # of histories each thread is responsible for in MLP/DROP kernel block_update_GPU
#define VOXELS_PER_THREAD 		1										// [#] # of voxels each thread is responsible for updating for reconstruction image initialization/updates
#define MAX_GPU_HISTORIES		4000000									// [#] Number of histories to process on the GPU at a time for preprocessing, based on GPU capacity
#define MAX_CUTS_HISTORIES		1500000									// [#] Number of histories to process on the GPU at a time for statistical cuts, based on GPU capacity
#define MAX_ENDPOINTS_HISTORIES 10240000								// [#] Number of histories to process on the GPU at a time for MLP endpoints, based on GPU capacity
#define MAX_INTERSECTIONS		1280									// [#] Limit on the # of intersections expected for proton's MLP; = # voxels along image diagonal
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//------------------------------------------------------ Scanning and detector system (source distance, tracking plane dimensions) configurations --------------------------------------------------------/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
#define GANTRY_ANGLE_INTERVAL	4.0										// [degrees] Angle between successive projection angles 
#define GANTRY_ANGLES			int( 360 / GANTRY_ANGLE_INTERVAL )		// [#] Total number of projection angles
#define NUM_SCANS				1										// [#] Total number of scans
#define NUM_FILES				( NUM_SCANS * GANTRY_ANGLES )			// [#] 1 file per gantry angle per translation
#define SSD_T_SIZE				35.0									// [cm] Length of SSD in t (lateral) direction
#define SSD_V_SIZE				9.0										// [cm] Length of SSD in v (vertical) direction
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//-------------------------------------------------------------- Binning (for statistical analysis) and sinogram (for FBP) configurations ----------------------------------------------------------------/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
#define T_SHIFT					0.0										// [cm] Amount by which to shift all t coordinates on input
#define U_SHIFT					0.0										// [cm] Amount by which to shift all u coordinates on input
#define V_SHIFT					0.0										// [cm] Amount by which to shift all v coordinates on input
//#define T_SHIFT				   2.05									// [cm] Amount by which to shift all t coordinates on input
//#define U_SHIFT				   -0.16								// [cm] Amount by which to shift all u coordinates on input
#define T_BIN_SIZE				0.1										// [cm] Distance between adjacent bins in t (lateral) direction
#define T_BINS					int( SSD_T_SIZE / T_BIN_SIZE + 0.5 )	// [#] Number of bins (i.e. quantization levels) for t (lateral) direction 
#define V_BIN_SIZE				0.25									// [cm] Distance between adjacent bins in v (vertical) direction
#define V_BINS					int( SSD_V_SIZE/ V_BIN_SIZE + 0.5 )		// [#] Number of bins (i.e. quantization levels) for v (vertical) direction 
#define ANGULAR_BIN_SIZE		4.0										// [degrees] Angle between adjacent bins in angular (rotation) direction
#define ANGULAR_BINS			int( 360 / ANGULAR_BIN_SIZE + 0.5 )		// [#] Number of bins (i.e. quantization levels) for path angle 
#define NUM_BINS				( ANGULAR_BINS * T_BINS * V_BINS )		// [#] Total number of bins corresponding to possible 3-tuples [ANGULAR_BIN, T_BIN, V_BIN]
#define SIGMAS_TO_KEEP			2										// [#] Number of standard deviations from mean to allow before cutting the history 
#define RAM_LAK_TAU				2/ROOT_TWO * T_BIN_SIZE					// [#] Defines tau in Ram-Lak filter calculation, estimated from largest frequency in slice 
#define FBP_THRESHOLD			0.6										// [cm] RSP threshold used to generate FBP_hull from FBP_image
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//------------------------------------------------------------------------------- Reconstruction cylinder configurations ---------------------------------------------------------------------------------/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
#define RECON_CYL_RADIUS		8.0									// [cm] Radius of reconstruction cylinder
#define RECON_CYL_DIAMETER		( 2 * RECON_CYL_RADIUS )				// [cm] Diameter of reconstruction cylinder
#define RECON_CYL_HEIGHT		(SSD_V_SIZE - 1.0)						// [cm] Height of reconstruction cylinder
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//-------------------------------------------------------------------------------- Reconstruction image configurations ----------------------------------------------------------------------------------/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//DISCRETIZATION_SIZES
//DISCRETIZATION_DIMENSIONS
//SIZES_DIMENSIONS
#define IMAGE_WIDTH				RECON_CYL_DIAMETER						// [cm] Distance between left and right edges of each slice in image
#define IMAGE_HEIGHT			RECON_CYL_DIAMETER						// [cm] Distance between top and bottom edges of each slice in image
#define IMAGE_THICKNESS			( SLICES * SLICE_THICKNESS )			// [cm] Distance between bottom of bottom slice and top of the top slice of image
#define COLUMNS					int(RECON_CYL_DIAMETER/VOXEL_WIDTH)		// [#] Number of voxels in the x direction (i.e., number of columns) of image
#define ROWS					int(RECON_CYL_DIAMETER/VOXEL_HEIGHT)	// [#] Number of voxels in the y direction (i.e., number of rows) of image
#define SLICES					int( RECON_CYL_HEIGHT / SLICE_THICKNESS)// [#] Number of voxels in the z direction (i.e., number of slices) of image
#define NUM_VOXELS				( COLUMNS * ROWS * SLICES )				// [#] Total number of voxels (i.e. 3-tuples [column, row, slice]) in image
#define VOXEL_WIDTH				0.1										// [cm] Distance between left and right edges of each voxel in image
#define VOXEL_HEIGHT			0.1										// [cm] Distance between top and bottom edges of each voxel in image
#define VOXEL_THICKNESS			0.25									// [cm] Distance between top and bottom of each slice in image
#define SLICE_THICKNESS			0.25									// [cm] Distance between top and bottom of each slice in image
#define VOXEL_DIAMETER			sqrt(pow(VOXEL_WIDTH, 2) + pow(VOXEL_HEIGHT, 2) + pow(VOXEL_THICKNESS, 2))
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//------------------------------------------------------------------------------------ Hull-Detection Parameters -------------------------------------------------------------------------------------/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
#define SC_LOWER_THRESHOLD		0.0										// [cm] If WEPL >= SC_LOWER_THRESHOLD, SC assumes the proton missed the object
#define SC_UPPER_THRESHOLD		0.0										// [cm] If WEPL <= SC_UPPER_THRESHOLD, SC assumes the proton missed the object
#define MSC_UPPER_THRESHOLD		0.0										// [cm] If WEPL >= MSC_LOWER_THRESHOLD, MSC assumes the proton missed the object
#define MSC_LOWER_THRESHOLD		-10.0									// [cm] If WEPL <= MSC_UPPER_THRESHOLD, MSC assumes the proton missed the object
#define MSC_DIFF_THRESH			150										// [#] Threshold on difference in counts between adjacent voxels used by MSC for edge detection
#define SM_LOWER_THRESHOLD		6.0										// [cm] If WEPL >= SM_THRESHOLD, SM assumes the proton passed through the object
#define SM_UPPER_THRESHOLD		21.0									// [cm] If WEPL > SM_UPPER_THRESHOLD, SM ignores this history
#define SM_SCALE_THRESHOLD		1.0										// [cm] Threshold scaling factor used by SM to adjust edge detection sensitivity
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//----------------------------------------------------------------------------------------- MLP Parameters -------------------------------------------------------------------------------------------/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
#define IGNORE_SHORT_MLP		ON										// [T/F] Remove proton histories with short MLP paths from use in reconstruction (ON) or not (OFF)
#define MIN_MLP_LENGTH			30										// [#] Minimum # of intersections required to use in reconstruction so proton's skimming object are ignored
//#define MLP_U_STEP				( 3 * VOXEL_WIDTH / 4)						// [cm] Size of the step taken along u direction during MLP; depth difference between successive MLP points
#define MLP_U_STEP				( VOXEL_WIDTH / 2)						// [cm] Size of the step taken along u direction during MLP; depth difference between successive MLP points
const int max_path_elements		= int(sqrt(double( ROWS^2 + COLUMNS^2 + SLICES^2))); // Defines size of GPU array used to store a proton history's MLP voxel #s 
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//----------------------------------------------------------------- Iterative projection method (feasibility seeking) settings/parameters ------------------------------------------------------------/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
ULL PRIME_OFFSET					= 26410633;							// [#] Separation between successive histories used in ordering histories for reconstruction
double ETA							= 2.5;								// [#] Coefficient of perturbation used in robust methods
unsigned int METHOD					= 1;								// [#] Integer indicating the desired robust method to use (deprecated, non in use)
int PSI_SIGN						= 1;								// [#] Use a positive (1) or negative (-1) perturbation in robust methods
#define ITERATIONS					10									// [#] # of iterations through the entire set of histories to perform in iterative image reconstruction
#define DROP_BLOCK_SIZE				51200								// [#] # of histories in each DROP block, i.e., # of histories used per image update
//#define LAMBDA					0.00015								// [#] Relaxation parameter to use in image iterative projection reconstruction algorithms	
float LAMBDA						= 0.0005;								// [#] Relaxation parameter to use in image iterative projection reconstruction algorithms	
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//----------------------------------------------------------------------- Total variation superiorization (TVS) options/parameters -------------------------------------------------------------------/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
#define TVS_ON						ON									// [T/F] Perform total variation superiorization (TVS) during reconstruction
#define TVS_OLD						OFF									// [T/F] Perform original version of total variation superiorization (TVS)
#define NTVS_ON						(TVS_ON && !TVS_OLD)				// [T/F] Perform total variation superiorization (TVS) during reconstruction
//#define NTVS_ON					OFF									// [T/F] Perform total variation superiorization (TVS) during reconstruction
#define TVS_FIRST					ON									// [T/F] Perform TVS before (ON) or after (OFF) feasibility seeking during reconstruction
#define TVS_PARALLEL				OFF									// [T/F] Use the sequential (OFF) or parallel (ON) implementation of TVS
#define TVS_CONDITIONED				ON									// [T/F] Verify TVS perturbation improves total variation TV (ON) or not (OFF)
#define TVS_MIN_ETA					1E-40								// [#] Specify minimum perturbation coefficient to include in precalculated coefficient array 
#define TV_THRESHOLD				(1/1000)							// [#] Value of TV difference ratio |TV_y - TV_y_previous| / TV_y between successive betas where beta is not decreased more
#define A							0.85								// [#] Perturbation coefficient generation kernel value: BETA_K_N = A^L
#define L_0							0	                                // [#,>=0] Initial value of L used in calculating the perturbation coefficient: A^L
#define PERTURB_DOWN_FACTOR			(1/A - 1)							// [#] Used in scaling perturbation to yield image w/ reduced perturbation from image previously perturbed w/ larger perturbation
int L								= 0;								// [#] Variable storing perturbation coefficient kernel exponent L used in calculating the perturbation coefficient: A^L
float BETA_0						= 1.0;								// [#] Inital value of TVS perturbation coefficient
float BETA							= BETA_0;							// [#] TVS perturbation coefficient 
float BETA_K_N						= powf(A, L);						// [#] Value of BETA used in classical TVS as perturbation coefficient
UINT TVS_REPETITIONS				= 5;								// [#] Specifies # of times to perform TVS for each iteration of DROP

const bool RECONSTRUCT_X_0 = false;
#define X0_ITERATIONS					5									// [#] # of iterations through the entire set of histories to perform in iterative image reconstruction
float X0_LAMBDA						= 0.0002;							// [#] Relaxation parameter to use in image iterative projection reconstruction algorithms	
#define X0_DROP_BLOCK_SIZE				25600								// [#] # of histories in each DROP block, i.e., # of histories used per image update
#define X0_BOUND_IMAGE					OFF									// [T/F] If any voxel in the image exceeds 2.0, set it to exactly 2.0
#define X0_TVS_ON						ON									// [T/F] Perform total variation superiorization (TVS) during reconstruction
#define X0_TVS_OLD						OFF									// [T/F] Perform original version of total variation superiorization (TVS)
#define X0_NTVS_ON						(TVS_ON && !TVS_OLD)				// [T/F] Perform total variation superiorization (TVS) during reconstruction
//#define NTVS_ON					OFF									// [T/F] Perform total variation superiorization (TVS) during reconstruction
#define X0_TVS_FIRST					OFF									// [T/F] Perform TVS before (ON) or after (OFF) feasibility seeking during reconstruction
#define X0_TVS_PARALLEL				OFF									// [T/F] Use the sequential (OFF) or parallel (ON) implementation of TVS
#define X0_TVS_CONDITIONED				OFF									// [T/F] Verify TVS perturbation improves total variation TV (ON) or not (OFF)
#define X0_A							0.75								// [#] Perturbation coefficient generation kernel value: BETA_K_N = A^L
UINT X0_TVS_REPETITIONS				= 5;								// [#] Specifies # of times to perform TVS for each iteration of DROP
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//---------------------------------------------------------------------------- Average/median filtering options/parameters ---------------------------------------------------------------------------/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
const bool IDENTIFY_X_0_AIR		= false;
double X_0_AIR_THRESHOLD				= 0.2;
#define IDENTIFY_X_N_AIR	OFF
#define X_N_AIR_THRESHOLD		0.1
#define BOUND_IMAGE					OFF									// [T/F] If any voxel in the image exceeds 2.0, set it to exactly 2.0
#define S_CURVE_ON					OFF									// [T/F] Turn on application of S-curve scaling of updates of voxels near the boundary
#define SIGMOID_STEEPNESS			1.25								// [#] Scaling factor 'k' of logistic curve: 1 / (1 + exp[k(LOGISTIC_MID_SHIFT - voxel)])
#define SIGMOID_MID_SHIFT			10									// [#] x-coordinate where the signoid curve is half of its maximum value
#define DUAL_SIDED_S_CURVE			ON									// [T/F] Apply a single-sided (OFF) or double-sided (ON) s-curve attenuation of voxel update values
const bool AVG_FILTER_FBP			= false;							// [T/F] Apply averaging filter to FBP (T) or not (F)										// Turn Space Modeling on (T) or off (F)
const bool AVG_FILTER_HULL			= false;								// [T/F] Apply averaging filter to hull (T) or not (F)
const bool AVG_FILTER_X_0			= false;							// [T/F] Apply averaging filter to initial iterate (T) or not (F)
const bool MEDIAN_FILTER_FBP		= true;								// [T/F] Apply median filter to FBP (T) or not (F)
const bool MEDIAN_FILTER_HULL		= false;							// [T/F] Apply median filter to hull (T) or not (F)
const bool MEDIAN_FILTER_X_0		= false;							// [T/F] Apply averaging filter to initial iterate (T) or not (F)
UINT FBP_AVG_FILTER_RADIUS			= 0;								// [#] Radius of the average filter to apply to FBP image
UINT HULL_AVG_FILTER_RADIUS			= 0;								// [#] Radius of the average filter to apply to hull image
UINT X_0_AVG_FILTER_RADIUS			= 0;								// [#] Radius of the average filter to apply to initial iterate
UINT FBP_MED_FILTER_RADIUS			= 3;								// [#] Radius of the median filter to apply to hull image
UINT HULL_MED_FILTER_RADIUS			= 0;								// [#] Radius of the median filter to apply to FBP image
UINT X_0_MED_FILTER_RADIUS			= 0;								// [#] Radius of the median filter to apply to initial iterate
double HULL_AVG_FILTER_THRESHOLD	= 0.1;								// [#] Threshold applied to average filtered hull separating voxels to include/exclude from hull (i.e. set to 0/1)
double FBP_AVG_FILTER_THRESHOLD		= 0.1;								// [#] Threshold applied to average filtered FBP separating voxels to include/exclude from FBP hull (i.e. set to 0/1)
double X_0_AVG_FILTER_THRESHOLD		= 0.1;								// [#] Threshold applied to average filtered initial iterate below which a voxel is excluded from reconstruction (i.e. set to 0)
/*****************************************************************************************************************************************************************************************************/
/*****************************************************************************************************************************************************************************************************/
/********************************************************************************************* Constants *********************************************************************************************/
/*****************************************************************************************************************************************************************************************************/
/*****************************************************************************************************************************************************************************************************/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//------------------------------------------------------------------------------------------- Filenames ----------------------------------------------------------------------------------------------/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
const char TESTED_BY_CSTRING[]		= "Blake Schultze";					// [string] Name written to the execution log specifying the user that generated the data 
//const char SECTION_EXIT_CSTRING[]	= {"====>"};						// [string] String prefix of task completion console text notifications using section_exit function
const char ON_CSTRING[]				= "ON";								// [string] String used s5,12pecifying an optional execution procedure is on (boolean variable=true)
const char OFF_CSTRING[]				= "OFF";							// [string] String used specifying an optional execution procedure is off (boolean variable=true)
const char KODIAK_SERVER_NAME[]		= "kodiak.baylor.edu";				// [string] String to use for the name of the Kodiak master node
const char ECSN1_SERVER_NAME[]		= "ecsn001";						// [string] String to use for the server name of the Tardis compute node ecsn001/WHartnell
const char ECSN2_SERVER_NAME[]		= "ecsn002";						// [string] String to use for the server name of the Tardis compute node ecsn002/PTroughton
const char ECSN3_SERVER_NAME[]		= "ecsn003";						// [string] String to use for the server name of the Tardis compute node ecsn003/JPertwee
const char ECSN4_SERVER_NAME[]		= "ecsn004";						// [string] String to use for the server name of the Tardis compute node ecsn004/TBaker
const char ECSN5_SERVER_NAME[]		= "ecsn005";						// [string] String to use for the server name of the Tardis compute node ecsn005/PDavison
const char WS1_SERVER_NAME[]		= "tardis-student1.ecs.baylor.edu";	// [string] String to use for the server name of Workstation #1
const char WS2_SERVER_NAME[]		= "tardis-student2.ecs.baylor.edu";	// [string] String to use for the server name of Workstation #2
const char KODIAK_HOSTNAME_CSTRING[]	= "Kodiak";							// [string] String to use for the name of the Kodiak cluster
const char ECSN1_HOSTNAME_CSTRING[]	= "WHartnell";						// [string] String to use for the name of the Tardis' compute node ecsn001/WHartnell
const char ECSN2_HOSTNAME_CSTRING[]	= "PTRoughton";						// [string] String to use for the name of the Tardis' compute node ecsn002/PTroughton
const char ECSN3_HOSTNAME_CSTRING[]	= "JPertwee";						// [string] String to use for the name of the Tardis' compute node ecsn003/JPertwee
const char ECSN4_HOSTNAME_CSTRING[]	= "TBaker";							// [string] String to use for the name of the Tardis' compute node ecsn004/TBaker
const char ECSN5_HOSTNAME_CSTRING[]	= "PDavison";						// [string] String to use for the name of the Tardis' compute node ecsn005/PDavison
const char WS_HOSTNAME_CSTRING[]		= "Workstation";					// [string] String to use for the name of the workstations (#1/#2) 
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//--------------------------------------------------------------------- Hostnames ($HOSTNAME) of remote servers and cluster nodes  -------------------------------------------------------------------/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
std::string kodiak_ID				( "n130"							);	// Hostname ID ($HOSTNAME) of Kodiak head node
std::string whartnell_ID			( "ecsn001"							);	// Hostname ID ($HOSTNAME) of WHartnell compute node
std::string ptroughton_ID			( "ecsn002"							);	// Hostname ID ($HOSTNAME) of PTRoughton compute node
std::string jpertwee_ID				( "ecsn003"							);	// Hostname ID ($HOSTNAME) of JPertwee compute node
std::string tbaker_ID				( "ecsn004"							);	// Hostname ID ($HOSTNAME) of TBaker compute node
std::string pdavison_ID				( "ecsn005"							);	// Hostname ID ($HOSTNAME) of PDavison compute node
std::string whartnell_hostname		( "whartnell"						);	// Hostname ID ($HOSTNAME) of WHartnell compute node
std::string workstation_2_hostname	( "tardis-student2.ecs.baylor.edu"	);	// Hostname ID ($HOSTNAME) of WHartnell compute node
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//--------------------------------------------- Secure copy (scp) and secure shell (ssh) login commands for local/remote communuications/operations --------------------------------------------------/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
const char BAYLOR_USERNAME[]		= "schultzeb";											// User name on Baylor/ECS accounts
const char KODIAK_USERNAME[]		= "schultze";											// User name on Kodiak/Tardis cluster head/compute nodes
const char TARDIS_USERNAME[]		= "schultze";											// User name on Kodiak/Tardis cluster head/compute nodes
const char RECON_GROUP_HOME_DIR[]	= "recon";											// User name on Kodiak/Tardis cluster head/compute nodes
const char RECON_GROUP_USERNAME[]	= "ionrecon";											// User name on Kodiak/Tardis cluster head/compute nodes
const char GIT_ACCOUNT[]			= "BlakeSchultze";											// User name on Kodiak/Tardis cluster head/compute nodes
const char GIT_REPOSITORY[]			= "pCT_Reconstruction";											// User name on Kodiak/Tardis cluster head/compute nodes
const char RECON_PROGRAM_NAME[]		= "pCT_Reconstruction";									// Name of pCT reconstruction program
const char RECON_GROUP_NAME[]		= "recon";												// Name of pCT reconstruction program
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//------------------------------------------------------------ Input/Output data folder names/paths associated with pCT data format ------------------------------------------------------------------/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
const char PCT_PARENT_DIR[]			= "//ion";														// Path to the pCT data/code in permanent storage on the network drive mounted and accessible by all Kodiak/Tardis nodes at Baylor
const char TARDIS_PARENT_DIR[]		= "//local";													// Path to the pCT data/code stored locally for usage on the Tardis compute nodes at Baylor
const char PCT_DATA_FOLDER[]		= "pCT_data";											// Name of folder containing all of the pCT data
const char PCT_CODE_FOLDER[]		= "pCT_code";											// Name of folder containing all of the pCT code
const char HOME_FOLDER[]			= "home";												// Name of folder containing all of the pCT code
const char GIT_FOLDER[]				= "git";												// Name of folder in the organized_data directory containing the reconstruction data	
const char SRC_CODE_FOLDER[]		= "src";												// Name of folder containing all of the pCT code
const char INCLUDE_CODE_FOLDER[]	= "include";											// Name of folder containing all of the pCT code
const char RAW_DATA_FOLDER[]		= "raw_data";											// Name of folder in pCT_data directory containing the raw experimental data
const char PROCESSED_DATA_FOLDER[]	= "processed_data";										// Name of folder in pCT_data directory containing the preprocessed raw data
const char PROJECTION_DATA_FOLDER[]	= "projection_data";									// Name of folder in pCT_data directory containing the projection data used as input to reconstruction
const char RECON_DATA_FOLDER[]		= "reconstruction_data";								// Name of folder in pCT_data directory containing the reconstruction data/image
const char ORGANIZED_DATA_FOLDER[]	= "organized_data";										// Name of folder in pCT_data containing the organized data 
const char EXPERIMENTAL_FOLDER[]	= "Experimental";										// Name of folder in the organized_data directory containing experimental data 
const char SIMULATED_FOLDER[]		= "Simulated";											// Name of folder in the organized_data directory containing simulated data
const char GEANT4_DIR_PREFIX[]		= "G_";													// Prefix of date folder names in the case of GEANT4 simulated data 
const char TOPAS_DIR_PREFIX[]		= "T_";													// Prefix of date folder names in the case of TOPAS simulated data 
const char RAW_LINKS_FOLDER[]		= "Input";												// Name of folder in the organized_data directory containing raw experimental data
const char PROJECTION_LINKS_FOLDER[]= "Output";												// Name of folder in the organized_data directory containing the projection data
const char RECONSTRUCTION_FOLDER[]	= "Reconstruction";										// Name of folder in the organized_data directory containing the reconstruction data	
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//----------------------------------------------------------------------------------------- Bash commands/options --------------------------------------------------------------------------------------------/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
const char BASH_CHANGE_DIR[]		= "cd";													// Command to secure copy data/directories between clusters/nodes
const char BASH_ECHO_CMD[]			= "echo -e";											// Command to secure copy data/directories between clusters/nodes
const char BASH_SECURE_COPY[]		= "scp -rCp -c blowfish";								// Command to secure copy data/directories between clusters/nodes
const char BASH_COPY_DIR[]			= "cp -apRv";											// Command to secure copy data/directories between clusters/nodes
const char BASH_COPY_DIR_SILENT[]	= "cp -apR";											// Command to secure copy data/directories between clusters/nodes
const char BASH_COPY_FILE[]			= "cp -v";												// Command to secure copy data/directories between clusters/nodes
const char BASH_COPY_FILE_SILENT[]	= "cp";													// Command to secure copy data/directories between clusters/nodes
const char BASH_MKDIR_CHAIN[]		= "mkdir -p";											// Command to secure copy data/directories between clusters/nodes
const char BASH_SET_FULL_ACCESS[]	= "chmod -R 777";										// Command to change file permissions to rwx for everyone
const char BASH_SET_PERMISSIONS_SILENT[]	= "chmod -R";											// Command to change file permissions to rwx for everyone
const char BASH_SET_PERMISSIONS[]	= "chmod -Rv";											// Command to change file permissions to rwx for everyone
const char BASH_CHANGE_PERMISSIONS[]= "chmod -Rc";											// Command to change file permissions to rwx for everyone
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
const char WORKING_DIR_ENVVAR[]		= "echo $PWD";											// Bash command to return the current directory environment variable of user logged in
const char OWNER_ACCESS[]			= "744";												// Permissions to give owner rwx permissions but all other users only r permission to a folder/file
const char GROUP_ACCESS[]			= "774";												// Permissions to give owner and group rwx permissions but all other users only r permission to a folder/file
const char GLOBAL_ACCESS[]			= "777";												// Permissions to give everyone rwx permissions to a folder/file
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*--------------------------------------------------------------- Configuration and execution logging file names ----------------------------------------------------------*/
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
const char PROJECTION_DATA_BASENAME[]	= "projection";										// Base name of projection data files projection_xxx.bin for each gantry angle xxx
const char PROJECTION_DATA_EXTENSION[]	= ".bin";											// File extension of projection data files projection_xxx.bin for each gantry angle xxx
const char CONFIG_FILENAME[]			= "settings.cfg";									// Name of the file used to control the program options/parameters as key=value pairs
const char CONFIG_OUT_FILENAME[]		= "settings_log.cfg";								// Name of the file used to control the program options/parameters as key=value pairs
const char STDOUT_FILENAME[]			= "stdout.txt";										// Name of the file where the standard output stream stdout is redirected
const char STDIN_FILENAME[]				= "stdin.txt";										// Name of the file where the standard input stream stdin is redirected
const char STDERR_FILENAME[]			= "stderr.txt";										// Name of the file where the standard error stream stderr is redirected
const char EXECUTION_LOG_BASENAME[]		= "execution_log";									// Base name of global .csv and run specific .txt files specifying the execution times for various portions of preprocessing/recosntruction
const char SIN_TABLE_FILENAME[]			= "sin_table.bin";									// Prefix of the file containing the tabulated values of sine function for angles [0, 2PI]
const char COS_TABLE_FILENAME[]			= "cos_table.bin";									// Prefix of the file containing the tabulated values of cosine function for angles [0, 2PI]
const char COEFFICIENT_FILENAME[]		= "coefficient.bin";								// Prefix of the file containing the tabulated values of the scattering coefficient for u_2-u_1/u_1 values in increments of 0.001
const char POLY_1_2_FILENAME[]			= "poly_1_2.bin";									// Prefix of the file containing the tabulated values of the polynomial with coefficients {1,2,3,4,5,6} in increments of 0.001
const char POLY_2_3_FILENAME[]			= "poly_2_3.bin";									// Prefix of the file containing the tabulated values of the polynomial with coefficients {2,3,4,5,6,7} in increments of 0.001
const char POLY_3_4_FILENAME[]			= "poly_3_4.bin";									// Prefix of the file containing the tabulated values of the polynomial with coefficients {3,4,5,6,7,8} in increments of 0.001
const char POLY_2_6_FILENAME[]			= "poly_2_6.bin";									// Prefix of the file containing the tabulated values of the polynomial with coefficients {2,6,12,20,30,42} in increments of 0.001
const char POLY_3_12_FILENAME[]			= "poly_3_12.bin";									// Prefix of the file containing the tabulated values of the polynomial with coefficients {3,12,30,60,105,168} in increments of 0.001
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//------------------------------------------------------------------------------------------- Filenames ----------------------------------------------------------------------------------------------/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
const char SC_HULL_FILENAME[]			= "SC_hull";
const char MSC_HULL_FILENAME[]			= "MSC_hull";
const char SM_HULL_FILENAME[]			= "SM_hull";
const char FBP_HULL_FILENAME[]			= "FBP_hull";
const char SM_COUNTS_FILENAME[]			= "SM_counts";
const char MSC_COUNTS_FILENAME[]		= "MSC_counts";
const char HULL_FILENAME[]				= "hull";
const char HULL_AVG_FILTER_FILENAME[]	= "hull_avg_filtered";
const char HULL_MED_FILTER_FILENAME[]	= "hull_median_filtered";
const char INPUT_HULL_FILENAME[]		= "input_hull.bin";
const char SINOGRAM_FILENAME[]			= "sinogram";
const char SINOGRAM_PRE_FILENAME[]		= "sinogram_pre";
const char FBP_FILENAME[]				= "FBP";
const char FBP_AFTER_FILENAME[]			= "FBP_after";
const char FBP_IMAGE_FILTER_FILENAME[]	= "FBP_image_filtered";
const char FBP_MED_FILTER_FILENAME[]	= "FBP_median_filtered";
const char FBP_AVG_FILTER_FILENAME[]	= "FBP_avg_filtered";
const char INPUT_ITERATE_FILENAME[]		= "FBP_med7.bin";
const char IMPORT_FBP_FILENAME[]		= "FBP_med";
const char X_0_FILENAME[]				= "x_0";
const char X_FILENAME[]					= "x";
const char BIN_COUNTS_PRE_FILENAME[]	= "bin_counts_pre";
const char BIN_COUNTS_FILENAME[]		= "bin_counts_h_pre";
const char BIN_COUNTS_POST_FILENAME[]	= "bin_counts_post";
const char MEAN_WEPL_FILENAME[]			= "mean_WEPL_h";
const char MEAN_REL_UT_FILENAME[]		= "mean_rel_ut_angle_h";
const char MEAN_REL_UV_FILENAME[]		= "mean_rel_uv_angle_h";
const char STDDEV_REL_UT_FILENAME[]		= "stddev_rel_ut_angle_h";
const char STDDEV_REL_UV_FILENAME[]		= "stddev_rel_uv_angle_h";
const char STDDEV_WEPL_FILENAME[]		= "stddev_WEPL_h";
const char MLP_PATHS_FILENAME[]			= "MLP_paths";
const char MLP_PATHS_ERROR_FILENAME[]	= "MLP_path_error";
const char MLP_ENDPOINTS_FILENAME[]		= "MLP_endpoints";
const char TV_MEASUREMENTS_FILENAME[]	= "TV_measurements";	
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//----------------------------------------------- Strings corresponding to enum type members used in naming column header names in execution times files ---------------------------------------------/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
const char OLD_FORMAT_CSTRING[]		= "OLD_FORMAT";						// String for enum "DATA_FORMATS" member "STANDARD"
const char VERSION_0_CSTRING[]		= "VERSION_0";						// String for enum "DATA_FORMATS" member "STANDARD"
const char VERSION_1_CSTRING[]		= "VERSION_1";						// String for enum "DATA_FORMATS" member "STANDARD"
const char EXPERIMENTAL_CSTRING[]	= "EXPERIMENTAL";					// String for enum "SCAN_TYPES" member "STANDARD"
const char SIMULATED_G_CSTRING[]		= "SIMULATED_G";					// String for enum "SCAN_TYPES" member "STANDARD"
const char SIMULATED_T_CSTRING[]		= "SIMULATED_T";					// String for enum "SCAN_TYPES" member "STANDARD"
const char TEXT_CSTRING[]			= "TEXT";							// String for enum "FILE_TYPES" member "STANDARD"
const char BINARY_CSTRING[]			= "BINARY";							// String for enum "FILE_TYPES" member "STANDARD"
const char DEFAULT_RAND_CSTRING[]	= "DEFAULT_RAND";					// String for enum "RAND_GENERATORS" member "STANDARD"
const char MINSTD_RAND_CSTRING[]		= "MINSTD_RAND";					// String for enum "RAND_GENERATORS" member "STANDARD"
const char MINSTD_RAND0_CSTRING[]	= "MINSTD_RAND0";					// String for enum "RAND_GENERATORS" member "STANDARD"
const char MT19937_CSTRING[]			= "MT19937";						// String for enum "RAND_GENERATORS" member "STANDARD"
const char MT19937_64_CSTRING[]		= "MT19937_64";						// String for enum "RAND_GENERATORS" member "STANDARD"
const char RANLUX24_CSTRING[]		= "RANLUX24";						// String for enum "RAND_GENERATORS" member "STANDARD"
const char RANLUX48_CSTRING[]		= "RANLUX48";						// String for enum "RAND_GENERATORS" member "STANDARD"
const char KNUTH_B_CSTRING[]			= "KNUTH_B";						// String for enum "RAND_GENERATORS" member "STANDARD"
const char VOXELS_CSTRING[]			= "VOXELS";							// String for enum "IMAGE_BASES" member "STANDARD"
const char BLOBS_CSTRING[]			= "BLOBS";							// String for enum "IMAGE_BASES" member "STANDARD"
const char MEANS_CSTRING[]			= "MEANS";							// String for enum "BIN_ANALYSIS_TYPE" member "STANDARD"
const char COUNTS_CSTRING[]			= "COUNTS";							// String for enum "BIN_ANALYSIS_TYPE" member "STANDARD"
const char MEMBERS_CSTRING[]			= "MEMBERS";						// String for enum "BIN_ANALYSIS_TYPE" member "STANDARD"
const char ALL_BINS_CSTRING[]		= "ALL_BINS";						// String for enum "BIN_ANALYSIS_FOR" member "STANDARD"
const char SPECIFIC_BINS_CSTRING[]	= "SPECIFIC_BINS";					// String for enum "BIN_ANALYSIS_FOR" member "STANDARD"
const char BY_BIN_CSTRING[]			= "BY_BIN";							// String for enum "BIN_ORGANIZATION" member "STANDARD"
const char BY_HISTORY_CSTRING[]		= "BY_HISTORY";						// String for enum "BIN_ORGANIZATION" member "STANDARD"
const char WEPLS_CSTRING[]			= "WEPLS";							// String for enum "BIN_ANALYSIS_OF" member "STANDARD"
const char ANGLES_CSTRING[]			= "ANGLES";							// String for enum "BIN_ANALYSIS_OF" member "STANDARD"
const char POSITIONS_CSTRING[]		= "POSITIONS";						// String for enum "BIN_ANALYSIS_OF" member "STANDARD"
const char BIN_NUMS_CSTRING[]		= "BIN_NUMS";						// String for enum "BIN_ANALYSIS_OF" member "STANDARD"
const char RAM_LAK_CSTRING[]			= "RAM_LAK";						// String for enum "SINOGRAM_FILTER_TYPES" member "STANDARD"
const char SHEPP_LOGAN_CSTRING[]		= "SHEPP_LOGAN";					// String for enum "SINOGRAM_FILTER_TYPES" member "STANDARD"
const char UNFILTERED_CSTRING[]		= "UNFILTERED";							// String for enum "SINOGRAM_FILTER_TYPES" member "STANDARD"
const char MEDIAN_FILTER_CSTRING[]	= "MEDIAN";							// String for enum "SINOGRAM_FILTER_TYPES" member "STANDARD"
const char AVERAGE_FILTER_CSTRING[]	= "AVERAGE";							// String for enum "SINOGRAM_FILTER_TYPES" member "STANDARD"
const char MED_2_AVG_FILTER_CSTRING[]= "MED_2_AVG";							// String for enum "SINOGRAM_FILTER_TYPES" member "STANDARD"
const char AVG_2_MED_FILTER_CSTRING[]= "AVG_2_MED";							// String for enum "SINOGRAM_FILTER_TYPES" member "STANDARD"
const char NO_FILTER_CSTRING[]		= "NO_FILTER";							// String for enum "SINOGRAM_FILTER_TYPES" member "STANDARD"
const char SC_HULL_CSTRING[]			= "SC_HULL";						// String for enum "HULL_TYPES" member "STANDARD"
const char MSC_HULL_CSTRING[]		= "MSC_HULL";						// String for enum "HULL_TYPES" member "STANDARD"
const char SM_HULL_CSTRING[]			= "SM_HULL";						// String for enum "HULL_TYPES" member "STANDARD"
const char FBP_HULL_CSTRING[]		= "FBP_HULL";						// String for enum "HULL_TYPES" member "STANDARD"
const char X_HULL_CSTRING[]			= "X_HULL";							// String for enum "HULL_TYPES" member "STANDARD"
const char HULL_CSTRING[]			= "HULL";							// String for enum "INITIAL_ITERATE" member "STANDARD"
const char FBP_IMAGE_CSTRING[]		= "FBP_IMAGE";						// String for enum "INITIAL_ITERATE" member "STANDARD"
const char HYBRID_CSTRING[]			= "HYBRID";							// String for enum "INITIAL_ITERATE" member "STANDARD"
const char ZEROS_CSTRING[]			= "ZEROS";							// String for enum "INITIAL_ITERATE" member "STANDARD"
const char IMPORT_CSTRING[]			= "IMPORT";							// String for enum "INITIAL_ITERATE" member "STANDARD"
const char FULL_TX_CSTRING[]			= "FULL_TX";						// String for enum "TX_OPTIONS" member "STANDARD"
const char PARTIAL_TX_CSTRING[]		= "PARTIAL_TX";						// String for enum "TX_OPTIONS" member "STANDARD"
const char PARTIAL_TX_PREALLOCATED_CSTRING[]= "PARTIAL_TX_PREALLOCATED";	// String for enum "TX_OPTIONS" member "STANDARD"
const char BOOL_CSTRING[]			= "BOOL";							// String for enum "ENDPOINTS_ALGORITHMS" member "STANDARD"
const char YES_BOOL_CSTRING[]		= "YES_BOOL";						// String for enum "ENDPOINTS_ALGORITHMS" member "STANDARD"
const char NO_BOOL_CSTRING[]			= "NO_BOOL";						// String for enum "ENDPOINTS_ALGORITHMS" member "STANDARD"	
const char STANDARD_CSTRING[]		= "STANDARD";						// String for enum "MLP_ALGORITHMS" member "STANDARD"
const char TABULATED_CSTRING[]		= "TABULATED";						// String for enum "MLP_ALGORITHMS" member "STANDARD"	
const char ART_CSTRING[]				= "ART";							// String for enum "PROJECTION_ALGORITHMS" member "STANDARD"
const char SART_CSTRING[]			= "SART";							// String for enum "PROJECTION_ALGORITHMS" member "STANDARD"
const char DROP_CSTRING[]			= "DROP";							// String for enum "PROJECTION_ALGORITHMS" member "STANDARD"
const char BIP_CSTRING[]				= "BIP";							// String for enum "PROJECTION_ALGORITHMS" member "STANDARD"
const char SAP_CSTRING[]				= "SAP";							// String for enum "PROJECTION_ALGORITHMS" member "STANDARD"
const char ROBUSTA_CSTRING[]			= "ROBUSTA";						// String for enum "PROJECTION_ALGORITHMS" member "STANDARD"
const char ROBUSTB_CSTRING[]			= "ROBUSTB";						// String for enum "PROJECTION_ALGORITHMS" member "STANDARD"
const char SIGMOID_CSTRING[]			= "SIGMOID";						// String for enum "S_CURVES" member "STANDARD"
const char TANH_CSTRING[]			= "TANH";							// String for enum "S_CURVES" member "STANDARD"
const char ATAN_CSTRING[]			= "ATAN";							// String for enum "S_CURVES" member "STANDARD"
const char ERF_CSTRING[]				= "ERF";							// String for enum "S_CURVES" member "STANDARD"
const char LIN_OVER_ROOT_CSTRING[]	= "LIN_OVER_ROOT";					// String for enum "S_CURVES" member "STANDARD" 
const char OLS_CSTRING[]				= "OLS";							// String for enum "ROBUST_METHODS" member "OLS" 
const char TLS_CSTRING[]				= "TLS";							// String for enum "ROBUST_METHODS" member "TLS" 
const char TIKHONOV_CSTRING[]		= "TIKHONOV";						// String for enum "ROBUST_METHODS" member "STANDARD" 
const char RIDGE_CSTRING[]			= "RIDGE";							// String for enum "ROBUST_METHODS" member "RIDGE" 
const char MINMIN_CSTRING[]			= "MINMIN";							// String for enum "ROBUST_METHODS" member "MINMIN" 
const char MINMAX_CSTRING[]			= "MINMAX";							// String for enum "ROBUST_METHODS" member "MINMAX" 
const char SQUARE_CSTRING[]			= "SQUARE";							// String for enum "STRUCTURAL_ELEMENTS" member "SQUARE"  
const char DISK_CSTRING[]			= "DISK";							// String for enum "STRUCTURAL_ELEMENTS" member "DISK" 								
const char EROSION_CSTRING[]			= "EROSION";						// String for enum "MORPHOLOGICAL_OPS" member "EROSION" 
const char DILATION_CSTRING[]		= "DILATION";						// String for enum "MORPHOLOGICAL_OPS" member "DILATION" 
const char OPENING_CSTRING[]			= "OPENING";						// String for enum "MORPHOLOGICAL_OPS" member "OPENING" 
const char CLOSING_CSTRING[]			= "CLOSING";						// String for enum "MORPHOLOGICAL_OPS" member "CLOSING" 			
const char OBJECT_L_CSTRING[]		= "OBJECT_L";						// String for enum "LOG_ENTRIES" member "OBJECT_L"  
const char SCAN_TYPE_L_CSTRING[]		= "SCAN_TYPE_L";					// String for enum "LOG_ENTRIES" member "SCAN_TYPE_L"  
const char RUN_DATE_L_CSTRING[]		= "RUN_DATE_L";						// String for enum "LOG_ENTRIES" member "RUN_DATE_L"  
const char RUN_NUMBER_L_CSTRING[]	= "RUN_NUMBER_L";					// String for enum "LOG_ENTRIES" member "RUN_NUMBER_L" 			
std::string ON_STRING("ON");
std::string OFF_STRING("OFF");
	//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//---------------------------------------------------------------------------------------- Output filenames ------------------------------------------------------------------------------------------/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
const int MAX_ITERATIONS			= 15;										// [#] Max # of feasibility seeking iterations (specifies # of iteration execution time columns needed in execution log)
#define E_0						13.6											// [MeV/c] empirical constant
#define X0						36.08											// [cm] radiation length
#define RSP_AIR					0.00113											// [cm/cm] Approximate RSP of air
#define BEAM_ENERGY				200												// [MeV] Initial energy of proton beam 
#define SOURCE_RADIUS			260.7											// [cm] Distance  to source/scatterer 

// Coefficients of 5th order polynomial fit to the term [1 / ( beta^2(u)*p^2(u) )] present in scattering covariance matrices Sigma 1/2 for:
// 200 MeV protons
#define A_0						7.457  * pow( (float)10, (float)-6.0  )			// Coefficient of polynomial fit term x^0
#define A_1						4.548  * pow( (float)10, (float)-7.0  )			// Coefficient of polynomial fit term x^1
#define A_2						-5.777 * pow( (float)10, (float)-8.0  )			// Coefficient of polynomial fit term x^2
#define A_3						1.301  * pow( (float)10, (float)-8.0  )			// Coefficient of polynomial fit term x^3
#define A_4						-9.228 * pow( (float)10, (float)-10.0 )			// Coefficient of polynomial fit term x^4
#define A_5						2.687  * pow( (float)10, (float)-11.0 )			// Coefficient of polynomial fit term x^5

//// 200 MeV protons
//#define A_0						202.20574
//#define A_1						-7.6174839
//#define A_2						0.9413194
//#define A_3						-0.1141406
//#define A_4						0.0055340
//#define A_5						-0.0000972

// Common fractional values of A_i coefficients appearing in polynomial expansions of MLP calculations, precalculating saves time
#define A_0_OVER_2				A_0/2 
#define A_0_OVER_3				A_0/3
#define A_1_OVER_2				A_1/2
#define A_1_OVER_3				A_1/3
#define A_1_OVER_4				A_1/4
#define A_1_OVER_6				A_1/6
#define A_1_OVER_12				A_1/12
#define A_2_OVER_3				A_2/3
#define A_2_OVER_4				A_2/4
#define A_2_OVER_5				A_2/5
#define A_2_OVER_12				A_2/12
#define A_2_OVER_30				A_2/30
#define A_3_OVER_4				A_3/4
#define A_3_OVER_5				A_3/5
#define A_3_OVER_6				A_3/6
#define A_3_OVER_20				A_3/20
#define A_3_OVER_60				A_3/60
#define A_4_OVER_5				A_4/5
#define A_4_OVER_6				A_4/6
#define A_4_OVER_7				A_4/7
#define A_4_OVER_30				A_4/30
#define A_4_OVER_105			A_4/105
#define A_5_OVER_6				A_5/6
#define A_5_OVER_7				A_5/7
#define A_5_OVER_8				A_5/8
#define A_5_OVER_42				A_5/42
#define A_5_OVER_168			A_5/168
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//----------------------------------------------------------------------------------- Tabulated data file names --------------------------------------------------------------------------------------/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
#define TRIG_TABLE_MIN			-2 * PI															// [radians] Minimum angle contained in the sin/cos lookup table used for MLP
#define TRIG_TABLE_MAX			4 * PI															// [radians] Maximum angle contained in the sin/cos lookup table used for MLP
#define TRIG_TABLE_RANGE		(TRIG_TABLE_MAX - TRIG_TABLE_MIN)								// [radians] Range of angles contained in the sin/cos lookup table used for MLP
#define TRIG_TABLE_STEP			(0.001 * ANGLE_TO_RADIANS)										// [radians] Step size in radians between elements of sin/cos lookup table used for MLP
#define TRIG_TABLE_ELEMENTS		static_cast<int>(TRIG_TABLE_RANGE / TRIG_TABLE_STEP + 0.5)		// [#] # of elements contained in the sin/cos lookup table used for MLP
#define COEFF_TABLE_RANGE		40.0															// [cm] Range of depths u contained in the polynomial lookup tables used for MLP
#define COEFF_TABLE_STEP		0.00005															// [cm] Step of 1/1000 cm for elements of the polynomial lookup tables used for MLP
#define COEFF_TABLE_SHIFT		static_cast<int>(MLP_U_STEP / COEFF_TABLE_STEP  + 0.5 )			// [cm] Step of 1/1000 cm for elements of the polynomial lookup tables used for MLP
#define COEFF_TABLE_ELEMENTS	static_cast<int>(COEFF_TABLE_RANGE / COEFF_TABLE_STEP + 0.5 )	// [#] # of elements contained in the polynomial lookup tables used for MLP
#define POLY_TABLE_RANGE		40.0															// [cm] Range of depths u contained in the polynomial lookup tables used for MLP
#define POLY_TABLE_STEP			0.00005															// [cm] Step of 1/1000 cm for elements of the polynomial lookup tables used for MLP
#define POLY_TABLE_SHIFT		static_cast<int>(MLP_U_STEP / POLY_TABLE_STEP  + 0.5 )			// [cm] Step of 1/1000 cm for elements of the polynomial lookup tables used for MLP
#define POLY_TABLE_ELEMENTS		static_cast<int>(POLY_TABLE_RANGE / POLY_TABLE_STEP + 0.5 )		// [#] # of elements contained in the polynomial lookup tables used for MLP
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//----------------------------------------------------------------------- Memory allocation size for arrays (binning, image) -------------------------------------------------------------------------/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
#define SIZE_BINS_CHAR			( NUM_BINS   * sizeof(char)	 )			// Amount of memory required for a character array used for binning
#define SIZE_BINS_BOOL			( NUM_BINS   * sizeof(bool)	 )			// Amount of memory required for a boolean array used for binning
#define SIZE_BINS_INT			( NUM_BINS   * sizeof(int)	 )			// Amount of memory required for a integer array used for binning
#define SIZE_BINS_UINT			( NUM_BINS   * sizeof(unsigned int)	 )	// Amount of memory required for a integer array used for binning
#define SIZE_BINS_FLOAT			( NUM_BINS	 * sizeof(float) )			// Amount of memory required for a floating point array used for binning
#define SIZE_IMAGE_CHAR			( NUM_VOXELS * sizeof(char)	 )			// Amount of memory required for a character array used for binning
#define SIZE_IMAGE_BOOL			( NUM_VOXELS * sizeof(bool)	 )			// Amount of memory required for a boolean array used for binning
#define SIZE_IMAGE_INT			( NUM_VOXELS * sizeof(int)	 )			// Amount of memory required for a integer array used for binning
#define SIZE_IMAGE_UINT			( NUM_VOXELS * sizeof(unsigned int)	 )	// Amount of memory required for a integer array used for binning
#define SIZE_IMAGE_FLOAT		( NUM_VOXELS * sizeof(float) )			// Amount of memory required for a floating point array used for binning
#define SIZE_IMAGE_DOUBLE		( NUM_VOXELS * sizeof(double) )			// Amount of memory required for a floating point array used for binning
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//------------------------------------------------------------------------------------ Precalculated Constants ---------------------------------------------------------------------------------------/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/

#define BYTES_PER_HISTORY		48										// [bytes] Data size of each history, 44 for actual data and 4 empty bytes, for old data format (OLD_VERSION)
#define PHI						((1 + sqrt(5.0) ) / 2)					// [#] Positive golden ratio, positive solution of PHI^2-PHI-1 = 0; also PHI = a/b when a/b = (a + b) / a 
#define PHI_NEGATIVE			((1 - sqrt(5.0) ) / 2)					// [#] Negative golden ratio, negative solution of PHI^2-PHI-1 = 0; 
#define PI_OVER_4				( atan( 1.0 ) )							// [radians] 1*pi/4 radians =   pi/4 radians = 45 degrees
#define PI_OVER_2				( 2 * atan( 1.0 ) )						// [radians] 2*pi/4 radians =   pi/2 radians = 90 degrees
#define THREE_PI_OVER_4			( 3 * atan( 1.0 ) )						// [radians] 3*pi/4 radians = 3*pi/4 radians = 135 degrees
#define PI						( 4 * atan( 1.0 ) )						// [radians] 4*pi/4 radians =   pi   radians = 180 degrees
#define FIVE_PI_OVER_4			( 5 * atan( 1.0 ) )						// [radians] 5*pi/4 radians = 5*pi/4 radians = 225 degrees
#define SIX_PI_OVER_4			( 5 * atan( 1.0 ) )						// [radians] 6*pi/4 radians = 3*pi/2 radians = 270 degrees
#define SEVEN_PI_OVER_4			( 7 * atan( 1.0 ) )						// [radians] 7*pi/4 radians = 7*pi/4 radians = 315 degrees
#define TWO_PI					( 8 * atan( 1.0 ) )						// [radians] 8*pi/4 radians = 2*pi   radians = 360 degrees = 0 degrees
#define ROOT_PI_OVER_TWO		(sqrt(PI)/2)							// [#] Square root of pi divided by 2
#define PI_OVER_TWO				(PI/2)									// [#] Square root of pi divided by 2
#define TWO_OVER_PI				(2/PI)									// [#] Square root of pi divided by 2
#define ANGLE_TO_RADIANS		( PI/180.0 )							// [radians/degree] Multiplicative factor used to convert from angle to radians
#define RADIANS_TO_ANGLE		( 180.0/PI )							// [degrees/radian] Multiplicative factor used to convert from radians to angle
#define ROOT_TWO				sqrtf(2.0)								// [#] 2^(1/2) = square root of 2 
#define MM_TO_CM				0.1										// [cm/mm] Multiplicative factor used to convert from mm to cm: 10 [mm] = 1 [cm] => 1 [mm] = 0.1 [cm]
#define CM_TO_MM				10.0									// [mm/cm] Multiplicative factor used to convert from cm to mm: 1 [cm] = 10 [mm]
#define VOXEL_ALLOWANCE			1.0e-7									// [cm] Distance from a voxel edge a point must come within before considered to be on the edge
#define RIGHT					1										// [#] Specifies that moving right corresponds with an increase in x position, used in voxel walk 
#define LEFT					-1										// [#] Specifies that moving left corresponds with a decrease in x position, used in voxel walk 
#define UP						1										// [#] Specifies that moving up corresponds with an increase in y/z position, used in voxel walk 
#define DOWN					-1										// [#] Specifies that moving down corresponds with a decrease in y/z position, used in voxel walk 
#define CONSOLE_WINDOW_WIDTH	80										// [#] Specifies character width of stdout console window
#define MAJOR_SECTION_SEPARATOR	'*'										// [character] Specifies character to use in major section separator
#define MINOR_SECTION_SEPARATOR	'-'										// [character] Specifies character to use in minor section separator
#define SECTION_EXIT_CSTRING		"======>"								// [string] String prefix of task completion console text notifications using section_exit function
#define DARK					0										// [#] Integer encoding of 'dark' text color shading option used in printing colored text to stdout (console window)
#define LIGHT					1										// [#] Integer encoding of 'light' text color shading option used in printing colored text to stdout (console window) 	
#define BLACK					30										// [#] Integer encoding of 'black' text color used in printing colored text to stdout (console window)
#define RED						31										// [#] Integer encoding of 'red' text color used in printing colored text to stdout (console window)
#define GREEN					32										// [#] Integer encoding of 'green' text color used in printing colored text to stdout (console window)
#define BROWN					33										// [#] Integer encoding of 'brown' text color used in printing colored text to stdout (console window)
#define BLUE					34										// [#] Integer encoding of 'blue' text color used in printing colored text to stdout (console window)
#define PURPLE					35										// [#] Integer encoding of 'purple' text color used in printing colored text to stdout (console window)
#define CYAN					36										// [#] Integer encoding of 'cyan' text color used in printing colored text to stdout (console window)
#define BLACK_TEXT				"0;30"									// [string] Integer encoding of 'black' text color used in printing colored text to stdout (console window)
#define GRAY_TEXT				"1;30"									// [string] Integer encoding of 'black' text color used in printing colored text to stdout (console window)
#define RED_TEXT				"0;31"									// [string] Integer encoding of 'red' text color used in printing colored text to stdout (console window)
#define LIGHT_RED_TEXT			"1;31"									// [string] Integer encoding of 'red' text color used in printing colored text to stdout (console window)
#define GREEN_TEXT				"0;32"									// [string] Integer encoding of 'green' text color used in printing colored text to stdout (console window)
#define LIGHT_GREEN_TEXT		"1;32"									// [string] Integer encoding of 'green' text color used in printing colored text to stdout (console window)
#define BROWN_TEXT				"0;33"									// [string] Integer encoding of 'brown' text color used in printing colored text to stdout (console window)
#define YELLOW_TEXT				"1;33"									// [string] Integer encoding of 'brown' text color used in printing colored text to stdout (console window)
#define BLUE_TEXT				"0;34"									// [string] Integer encoding of 'blue' text color used in printing colored text to stdout (console window)
#define LIGHT_BLUE_TEXT			"1;34"									// [string] Integer encoding of 'blue' text color used in printing colored text to stdout (console window)
#define PURPLE_TEXT				"0;35"									// [string] Integer encoding of 'purple' text color used in printing colored text to stdout (console window)
#define LIGHT_PURPLE_TEXT		"1;35"									// [string] Integer encoding of 'purple' text color used in printing colored text to stdout (console window)
#define CYAN_TEXT				"0;36"									// [string] Integer encoding of 'cyan' text color used in printing colored text to stdout (console window)
#define LIGHT_CYAN_TEXT			"1;36"									// [string] Integer encoding of 'cyan' text color used in printing colored text to stdout (console window)
#define LIGHT_GRAY_TEXT			"0;37"									// [string] Integer encoding of 'cyan' text color used in printing colored text to stdout (console window)
#define WHITE_TEXT				"1;37"									// [string] Integer encoding of 'cyan' text color used in printing colored text to stdout (console window)
#define GRAY_BACKGROUND			"5;40"									// [string] Integer encoding of 'black' text color used in printing colored text to stdout (console window)
#define BLACK_BACKGROUND		"6;40"									// [string] Integer encoding of 'black' text color used in printing colored text to stdout (console window)
#define LIGHT_RED_BACKGROUND	"5;41"									// [string] Integer encoding of 'red' text color used in printing colored text to stdout (console window)
#define RED_BACKGROUND			"6;41"									// [string] Integer encoding of 'red' text color used in printing colored text to stdout (console window)
#define LIGHT_GREEN_BACKGROUND	"5;42"									// [string] Integer encoding of 'green' text color used in printing colored text to stdout (console window)
#define GREEN_BACKGROUND		"6;42"									// [string] Integer encoding of 'green' text color used in printing colored text to stdout (console window)
#define BROWN_BACKGROUND		"5;43"									// [string] Integer encoding of 'brown' text color used in printing colored text to stdout (console window)
#define YELLOW_BACKGROUND		"6;43"									// [string] Integer encoding of 'brown' text color used in printing colored text to stdout (console window)
#define LIGHT_BLUE_BACKGROUND	"5;44"									// [string] Integer encoding of 'blue' text color used in printing colored text to stdout (console window)
#define BLUE_BACKGROUND			"6;44"									// [string] Integer encoding of 'blue' text color used in printing colored text to stdout (console window)
#define LIGHT_PURPLE_BACKGROUND	"5;45"									// [string] Integer encoding of 'purple' text color used in printing colored text to stdout (console window)
#define PURPLE_BACKGROUND		"6;45"									// [string] Integer encoding of 'purple' text color used in printing colored text to stdout (console window)
#define LIGHT_CYAN_BACKGROUND	"5;46"									// [string] Integer encoding of 'cyan' text color used in printing colored text to stdout (console window)
#define CYAN_BACKGROUND			"6;46"									// [string] Integer encoding of 'cyan' text color used in printing colored text to stdout (console window)
#define WHITE_BACKGROUND		"5;47"									// [string] Integer encoding of 'cyan' text color used in printing colored text to stdout (console window)
#define LIGHT_GRAY_BACKGROUND	"6;47"									// [string] Integer encoding of 'cyan' text color used in printing colored text to stdout (console window)
#define UNDERLINE_TEXT			";4"									// [string] Integer encoding specifying text be underlined when printing colored text to stdout (console window)
#define DONT_UNDERLINE_TEXT		""										// [string] Empty encoding specifying text NOT be underlined when printing colored text to stdout (console window)
#define OPEN_COLOR_ESCAPE_SEQ	"\033["								// [string] Escape sequence setting removing color from subsequent console output text
#define CLOSE_COLOR_ESCAPE_SEQ	"\033[m"								// [string] Escape sequence setting removing color from subsequent console output text
#define NOCOLOR					"\033[0m"								// [string] Escape sequence setting removing color from subsequent console output text
//#define BLACK					"\033[0;30m"							// [string] Escape sequence setting the color of subsequent console output text to black
#define DARKGRAY				"\033[1;30m"							// [string] Escape sequence setting the color of subsequent console output text to dark gray
//#define RED					"\033[0;31m"							// [string] Escape sequence setting the color of subsequent console output text to red
#define LIGHTRED				"\033[1;31m"							// [string] Escape sequence setting the color of subsequent console output text to light red
//#define GREEN					"\033[0;32m"							// [string] Escape sequence setting the color of subsequent console output text to green
#define LIGHTGREEN				"\033[1;32m"							// [string] Escape sequence setting the color of subsequent console output text to light green
//#define BROWN					"\033[1;33m"							// [string] Escape sequence setting the color of subsequent console output text to brown
//#define YELLOW				"\033[0;33m"							// [string] Escape sequence setting the color of subsequent console output text to yellow
//#define BLUE					"\033[0;34m"							// [string] Escape sequence setting the color of subsequent console output text to blue
#define LIGHTBLUE				"\033[1;34m"							// [string] Escape sequence setting the color of subsequent console output text to light blue
//#define PURPLE				"\033[0;35m"							// [string] Escape sequence setting the color of subsequent console output text to purple
#define LIGHTPURPLE				"\033[1;35m"							// [string] Escape sequence setting the color of subsequent console output text to light purple
//#define CYAN					"\033[0;36m"							// [string] Escape sequence setting the color of subsequent console output text to cyan
#define LIGHTCYAN				"\033[1;36m"							// [string] Escape sequence setting the color of subsequent console output text to light cyan
#define LIGHTGRAY				"\033[0;37m"							// [string] Escape sequence setting the color of subsequent console output text to light gray
//#define WHITE					"\033[1;37m"							// [string] Escape sequence setting the color of subsequent console output text to white
#define X_INCREASING_DIRECTION	RIGHT									// [#] specifies direction (LEFT/RIGHT) along x-axis in which voxel #s increase
#define Y_INCREASING_DIRECTION	DOWN									// [#] specifies direction (UP/DOWN) along y-axis in which voxel #s increase
#define Z_INCREASING_DIRECTION	DOWN									// [#] specifies direction (UP/DOWN) along z-axis in which voxel #s increase
#define X_ZERO_COORDINATE		-RECON_CYL_RADIUS						// [cm] x-coordinate corresponding to front edge of 1st voxel (i.e. column) in image space
#define Y_ZERO_COORDINATE		RECON_CYL_RADIUS						// [cm] y-coordinate corresponding to front edge of 1st voxel (i.e. row) in image space
#define Z_ZERO_COORDINATE		RECON_CYL_HEIGHT/2						// [cm] z-coordinate corresponding to front edge of 1st voxel (i.e. slice) in image space
#define PRINT_TV				true									// [bool] Print total variation measurement
#define DONT_PRINT_TV			false									// [bool] Dont print total variation measurement
#define CHAR_ID_CHAR			'c'
#define BOOL_ID_CHAR			'b'
#define INT_ID_CHAR				'i'
#define UINT_ID_CHAR			'j'
#define STRING_ID_CHAR			's'
#define FLOAT_ID_CHAR			'f'
#define DOUBLE_ID_CHAR			'd'
#define BOOL_FORMAT				"%d"									// [string] Specifies format to use for writing/printing boolean data using {s/sn/f/v/vn}printf statements
#define CHAR_FORMAT				"%c"									// [string] Specifies format to use for writing/printing character data using {s/sn/f/v/vn}printf statements
#define INT_FORMAT				"%d"									// [string] Specifies format to use for writing/printing integer data using {s/sn/f/v/vn}printf statements
#define FLOAT_FORMAT			"%6.6lf"								// [string] Specifies format to use for writing/printing floating point data using {s/sn/f/v/vn}printf statements
#define STRING_FORMAT			"%s"									// [string] Specifies format to use for writing/printing strings data using {s/sn/f/v/vn}printf statements
#define GIT_COMMIT_DATE_CSTRING_LENGTH		30
#define print_type_name(var) ( std::cout << type_name(var) << endl )
/*****************************************************************************************************************************************************************************************************/
/*****************************************************************************************************************************************************************************************************/
/********************************************************************************* Preprocessing Array Declerations **********************************************************************************/
/*****************************************************************************************************************************************************************************************************/
/*****************************************************************************************************************************************************************************************************/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//----------------------------------------------------------------------- Main function argument and generic reusable variables ----------------------------------------------------------------------/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
// Generic reused variables
cudaError_t cudaStatus;	
char print_statement[1024];
char system_command[512];
char bash_command[512];

// Global execution and IO related variables
unsigned int NUM_RUN_ARGUMENTS;
char** RUN_ARGUMENTS;
char* CONFIG_DIRECTORY;
std::map<std::string, unsigned int> EXECUTION_LOG_SWITCHMAP;
bool CONTINUOUS_DATA = false;
unsigned int PHANTOM_NAME_SIZE;
unsigned int DATA_SOURCE_SIZE;
unsigned int PREPARED_BY_SIZE;
unsigned int SKIP_2_DATA_SIZE;
unsigned int VERSION_ID;
unsigned int PROJECTION_INTERVAL;
std::vector<std::string> OUTPUT_FILE_LIST;
std::vector<std::string> IMAGE_LIST;
std::vector<std::string> LOCAL_DATA_FILE_LIST;
std::vector<std::string> GLOBAL_DATA_FILE_LIST;

//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//--------------------------------------------------------- Declaration of character arrays for path variables assigned at execution time ------------------------------------------------------------/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//-------------------------- string_assigments()
// set_ssh_server_login_strings()
char KODIAK_SSH_LOGIN[64];												// ssh command addressing username@server for Kodiak e.g. "schultze@kodiak.baylor.edu:"
char WHARTNELL_SSH_LOGIN[64];											// ssh command addressing username@server for WHartnell e.g. "schultze@kodiak:"
char PTROUGHTON_SSH_LOGIN[64];											// ssh command addressing username@server for PTRoughton e.g. "schultze@ptroughton:"
char JPERTWEE_SSH_LOGIN[64];											// ssh command addressing username@server for JPertwee e.g. "schultze@jpertwee:"
char TBAKER_SSH_LOGIN[64];												// ssh command addressing username@server for TBaker e.g. "schultze@tbaker:"
char PDAVISON_SSH_LOGIN[64];											// ssh command addressing username@server for PDavison e.g. "schultze@pdavison:"
char WS1_SSH_LOGIN[64];											// ssh command addressing username@server for PDavison e.g. "schultze@pdavison:"
char WS2_SSH_LOGIN[64];											// ssh command addressing username@server for PDavison e.g. "schultze@pdavison:"

// set_enum_strings(): Character arrays used to write enum variable options selected to execution logs
char SCAN_TYPE_CSTRING[32];
char SINOGRAM_FILTER_CSTRING[32];
char ENDPOINTS_HULL_CSTRING[32];
char ENDPOINTS_ALG_CSTRING[32];
char ENDPOINTS_TX_MODE_CSTRING[32];
char MLP_ALGORITHM_CSTRING[32];
char X_0_CSTRING[32];
char PROJECTION_ALGORITHM_CSTRING[32];
char RECON_TX_MODE_CSTRING[32];
char ROBUST_METHOD_CSTRING[32];
char S_CURVE_CSTRING[32];

// set_procedures_on_off_strings(): Character arrays used to write whether the corresponding optional procedures are ON/OFF to execution logs
char SAMPLE_STD_DEV_CSTRING[8];
char FBP_FILTER_CSTRING[32];
char HULL_FILTER_CSTRING[32];
char X_0_FILTER_CSTRING[32];
char AVG_FILTER_FBP_CSTRING[8];
char AVG_FILTER_HULL_CSTRING[8];
char AVG_FILTER_X_0_CSTRING[8];
char MEDIAN_FILTER_FBP_CSTRING[8];
char MEDIAN_FILTER_HULL_CSTRING[8];
char MEDIAN_FILTER_X_0_CSTRING[8];
char IGNORE_SHORT_MLP_CSTRING[8];
char BOUND_IMAGE_CSTRING[8];
char IDENTIFY_X_0_AIR_CSTRING[8];
char IDENTIFY_X_N_AIR_CSTRING[8];
char S_CURVE_ON_CSTRING[8];	
char DUAL_SIDED_S_CURVE_CSTRING[8];	
char TVS_ON_CSTRING[8];	
char TVS_FIRST_CSTRING[8];	
char TVS_PARALLEL_CSTRING[8];	
char TVS_CONDITIONED_CSTRING[8];	

char EXECUTION_DATE[9];
char EXECUTION_YY_MM_DD[9];
char EXECUTION_DATE_TIME[128];
char EXECUTION_TIME_GMT[9];
char EXECUTION_TIME_LOCAL[9];

//-------------------------- IO_setup()
// set_compute_node()
char CURRENT_COMPUTE_NODE[32];
char CURRENT_COMPUTE_NODE_ALIAS[32];	

// set_user_strings()
char USERNAME[32];
char USE_TARDIS_USERNAME[32];	
char USE_KODIAK_USERNAME[32];	
char USE_BAYLOR_USERNAME[32];	
char USE_HOME_DIR_USERNAME[32];	
char USE_CODE_OWNER_NAME[32];	
char USE_RCODE_OWNER_NAME[32];	

// set_compute_system_directories()
char COMMON_ORG_DATA_SUBDIRECTORY[256];
char COMMON_RECON_DATA_SUBDIRECTORY[256];
char COMMON_RCODE_SUBDIRECTORY[256];
char COMMON_GIT_CODE_SUBDIRECTORY[256];
char PCT_DATA_DIR_SET[256];
char PCT_ORG_DATA_DIR_SET[256];
char PCT_RECON_DIR_SET[256];
char PCT_CODE_PARENT_SET[256];
char PCT_RCODE_PARENT_SET[256];
char PCT_GIT_RCODE_PARENT_SET[256];
char TARDIS_DATA_DIR_SET[256];
char TARDIS_ORG_DATA_DIR_SET[256];
char TARDIS_RECON_DIR_SET[256];
char TARDIS_CODE_PARENT_SET[256];
char TARDIS_RCODE_PARENT_SET[256];
char TARDIS_GIT_RCODE_PARENT_SET[256];
char SHARED_HOME_DIR_SET[256];
char SHARED_DATA_DIR_SET[256];
char SHARED_ORG_DATA_DIR_SET[256];
char SHARED_RECON_DIR_SET[256];
char SHARED_CODE_PARENT_SET[256];
char SHARED_RCODE_PARENT_SET[256];
char SHARED_GIT_RCODE_PARENT_SET[256];
char MY_HOME_DIR_SET[256];
char MY_DATA_DIR_SET[256];
char MY_ORG_DATA_DIR_SET[256];
char MY_RECON_DIR_SET[256];
char MY_CODE_PARENT_SET[256];
char MY_RCODE_PARENT_SET[256];
char MY_GIT_RCODE_PARENT_SET[256];
char WS2_CODE_PARENT_SET[256];
char WS2_RECON_DIR_SET[256];
char MYLAPTOP_RECON_DIR_SET[256];
char MYLAPTOP_ORG_DATA_DIR_SET[256];

// set_git_branch_info()
char CURRENT_CODE_PARENT[256];
char CURRENT_RCODE_PARENT[256];
char GIT_COMMIT_HASH_CSTRING[128];
char GIT_BRANCH_INFO_CSTRING[128];
char GIT_REPO_PATH[256];
char GIT_BRANCH_NAME[256];
char GIT_COMMIT_HASH[256];
char GIT_COMMIT_DATE[256];
char GIT_LOG_INFO[256];
char GIT_REPO_INFO[256];

// set_and_make_output_folder()
char CURRENT_DATA_DIR[256];
char CURRENT_RECON_DIR[256];
char INPUT_DIRECTORY_SET[256];
char OUTPUT_DIRECTORY_SET[256];
char OUTPUT_FOLDER_UNIQUE[256];

// set_IO_folder_names()
char INPUT_FOLDER_SET[256];
char OUTPUT_FOLDER_SET[256];
char LOCAL_INPUT_DATA_PATH[256];
char LOCAL_OUTPUT_DATA_PATH[256];
char GLOBAL_INPUT_DATA_PATH[256];
char GLOBAL_OUTPUT_DATA_PATH[256];
//char GLOBAL_OUTPUT_FOLDER_DESTINATION[256];
char GLOBAL_EXECUTION_LOG_PATH[256];
char LOCAL_EXECUTION_LOG_PATH[256];
char LOCAL_EXECUTION_INFO_PATH[256];
char GLOBAL_EXECUTION_INFO_PATH[512];
char LOCAL_TV_MEASUREMENTS_PATH[256];
char GLOBAL_TV_MEASUREMENTS_PATH[256];
char INPUT_ITERATE_PATH[256];
char IMPORT_FBP_PATH[256];

// set_source_code_paths()
char EXECUTED_CODE_DIR[256];
char LOCAL_OUTPUT_CODE_DIR[256];
char GLOBAL_OUTPUT_CODE_DIR[256];
char EXECUTED_SRC_CODE_PATH[256];
char EXECUTED_INCLUDE_CODE_PATH[256];
char LOCAL_OUTPUT_SRC_CODE_PATH[256];
char LOCAL_OUTPUT_INCLUDE_CODE_PATH[256];
char GLOBAL_OUTPUT_SRC_CODE_PATH[256];
char GLOBAL_OUTPUT_INCLUDE_CODE_PATH[256];
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//----------------------------------------------------------------------------------- Execution timing variables -------------------------------------------------------------------------------------/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
clock_t program_start, program_end, pause_cycles = 0;
clock_t begin_endpoints = 0, begin_init_image = 0, begin_tables = 0, begin_DROP_iteration = 0, begin_DROP = 0, begin_TVS_iteration = 0, begin_TVS = 0, begin_update_calcs = 0, begin_update_image = 0, begin_data_reads = 0, begin_preprocessing = 0, begin_reconstruction = 0, begin_program = 0, begin_data_tx = 0;
double execution_time_endpoints = 0, execution_time_init_image = 0, execution_time_DROP_iteration = 0, execution_time_TVS_iteration = 0, execution_time_DROP = 0, execution_time_TVS = 0, execution_time_update_calcs = 0, execution_time_update_image = 0, execution_time_tables = 0;
double execution_time_data_reads = 0, execution_time_preprocessing = 0, execution_time_reconstruction = 0, execution_time_program = 0,  execution_time_data_tx = 0; 
std::vector<double> execution_times_DROP_iterations;
std::vector<double> execution_times_TVS_iterations;
FILE* execution_log_file;
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//-------------------------------------------------------- Declaration of variables/arrays used to record information related to # of histories ------------------------------------------------------/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
int total_histories = 0, recon_vol_histories = 0, maximum_histories_per_file = 0;
int* histories_per_projection, * histories_per_gantry_angle, * histories_per_file;
int* recon_vol_histories_per_projection;
int histories_per_scan[NUM_SCANS];
int post_cut_histories = 0;
int reconstruction_histories = 0;

double percentage_pass_each_intersection_cut, percentage_pass_intersection_cuts, percentage_pass_statistical_cuts, percentage_pass_hull_cuts;
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//--------------------------------------------------------- Declaration of array used to store tracking plane distances from rotation axis -----------------------------------------------------------/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
std::vector<float> projection_angles;
float SSD_u_Positions[8];
float* ut_entry_angle, * uv_entry_angle, * ut_exit_angle, * uv_exit_angle; 
int zero_WEPL = 0;
int zero_WEPL_files = 0;
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

float* actual_projection_angles_h, * actual_projection_angles_d;

//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//--------------------------------------------------------- Declaration of statistical analysis arrays for use on host(_h) or device (_d) ------------------------------------------------------------/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
float* mean_WEPL_h, * mean_WEPL_d;
float* mean_energy_h, * mean_energy_d;
float* mean_rel_ut_angle_h, * mean_rel_ut_angle_d;
float* mean_rel_uv_angle_h, * mean_rel_uv_angle_d;
float* mean_total_rel_angle_h, * mean_total_rel_angle_d;
float* stddev_rel_ut_angle_h, * stddev_rel_ut_angle_d;
float* stddev_rel_uv_angle_h, * stddev_rel_uv_angle_d;
float* stddev_WEPL_h, * stddev_WEPL_d;
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//--------------------------------------------------------------------- Declaration of host(_h) and device (_d) image arrays -------------------------------------------------------------------------/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
float* sinogram_h, * sinogram_d;
float* sinogram_filtered_h, * sinogram_filtered_d;
bool* SC_hull_h, * SC_hull_d;
bool* MSC_hull_h, * MSC_hull_d;
bool* SM_hull_h, * SM_hull_d;
bool* FBP_hull_h, * FBP_hull_d;
bool* hull_h, * hull_d;
uint* hull_voxels_h, * hull_voxels_d;
std::vector<uint> hull_voxels_vector;
int* MSC_counts_h, * MSC_counts_d;
int* SM_counts_h, * SM_counts_d;
int* MLP_test_image_h, * MLP_test_image_d;
float* FBP_image_h, * FBP_image_d;
float* FBP_image_filtered_h, * FBP_image_filtered_d;
float* FBP_median_filtered_h, * FBP_median_filtered_d;
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//------------------------------------------------------- Declaration of host(_h) and device (_d) arrays used for MLP and reconstruction -------------------------------------------------------------/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
int num_voxel_scales;
double* voxel_scales;
bool* intersected_hull_h, * intersected_hull_d;
ULL* history_sequence;
unsigned int* num_voxel_intersections_h, * num_voxel_intersections_d;
unsigned int* S_h, * S_d;
unsigned int* block_voxels_h, *block_voxels_d;
unsigned int* block_counts_h, * block_counts_d;
double* x_update_h;
float* x_update_d;
double* norm_Ai;
float* x_h, * x_d;
float* x_before_TVS_h, * x_before_TVS_d;
float* x_TVS_h, * x_TVS_d;
unsigned int* global_a_i;
uint* DROP_blocks_ordered_h, DROP_blocks_ordered_d;
uint* DROP_block_sizes_ordered_h, DROP_block_order_d;
UINT DROP_last_block_size, num_DROP_blocks;
std::vector<UINT> DROP_block_sizes;
std::vector<UINT> DROP_block_order;
std::vector<UINT> DROP_block_start_positions;
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//-------------------------------------------------- Declaration of host(_h) and device (_d) arrays used in total variation superiorization (TVS) ----------------------------------------------------/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
int TVS_ETA_SEQUENCE_LENGTH;
float* G_x_h, * G_y_h, * G_norm_h, * G_h, * v_h, * y_h;
float* G_x_d, * G_y_d, * G_norm_d, * G_d, * v_d, * y_d;
float* TV_x_h, * TV_y_h;
float* TV_x_d, * TV_y_d;
float* TVS_eta_sequence_h, * TVS_eta_sequence_d;
float TV_x_final;
std::vector<float> TV_x_values;
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
//----------------------------------------------------------------- Declaration of image arrays for use on host(_h) or device (_d) -------------------------------------------------------------------/
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
FILE* sin_table_file, * cos_table_file, * scattering_table_file, * poly_1_2_file, * poly_2_3_file, * poly_3_4_file, * poly_2_6_file, * poly_3_12_file;
double* sin_table_h, * cos_table_h, * scattering_table_h, * poly_1_2_h, * poly_2_3_h, * poly_3_4_h, * poly_2_6_h, * poly_3_12_h;
double* sin_table_d, * cos_table_d, * scattering_table_d, * poly_1_2_d, * poly_2_3_d, * poly_3_4_d, * poly_2_6_d, * poly_3_12_d;
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
std::vector<float>	actual_projection_angles;	
std::vector<unsigned int> first_MLP_voxel_vector;
std::vector<int> voxel_x_vector;
std::vector<int> voxel_y_vector;
std::vector<int> voxel_z_vector;

std::vector<int>	bin_num_vector_ordered;			
std::vector<int>	gantry_angle_vector_ordered;	
std::vector<float>	WEPL_vector_ordered;		
std::vector<float>	x_entry_vector_ordered;		
std::vector<float>	y_entry_vector_ordered;		
std::vector<float>	z_entry_vector_ordered;		
std::vector<float>	x_exit_vector_ordered;			
std::vector<float>	y_exit_vector_ordered;			
std::vector<float>	z_exit_vector_ordered;			
std::vector<float>	xy_entry_angle_vector_ordered;	
std::vector<float>	xz_entry_angle_vector_ordered;	
std::vector<float>	xy_exit_angle_vector_ordered;	
std::vector<float>	xz_exit_angle_vector_ordered;	
std::vector<unsigned int> first_MLP_voxel_vector_ordered;

std::vector<int>	bin_num_vector_reconstruction;			
std::vector<int>	gantry_angle_vector_reconstruction;	
std::vector<float>	WEPL_vector_reconstruction;		
std::vector<float>	x_entry_vector_reconstruction;		
std::vector<float>	y_entry_vector_reconstruction;		
std::vector<float>	z_entry_vector_reconstruction;		
std::vector<float>	x_exit_vector_reconstruction;			
std::vector<float>	y_exit_vector_reconstruction;			
std::vector<float>	z_exit_vector_reconstruction;			
std::vector<float>	xy_entry_angle_vector_reconstruction;	
std::vector<float>	xz_entry_angle_vector_reconstruction;	
std::vector<float>	xy_exit_angle_vector_reconstruction;	
std::vector<float>	xz_exit_angle_vector_reconstruction;	
std::vector<unsigned int> first_MLP_voxel_vector_reconstruction;

/*****************************************************************************************************************************************************************************************************/
/*****************************************************************************************************************************************************************************************************/
/*********************************************************************************** End of Parameter Definitions ************************************************************************************/
/*****************************************************************************************************************************************************************************************************/
/*****************************************************************************************************************************************************************************************************/
//#endif // _PCT_RECONSTRUCTION_H_
