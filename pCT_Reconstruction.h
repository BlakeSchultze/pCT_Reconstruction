//#ifndef PCT_RECONSTRUCTION_H
//#define PCT_RECONSTRUCTION_H
#pragma once
/***************************************************************************************************************************************************************************/
/********************************************************************* Header for pCT reconstruction program ***************************************************************/
/***************************************************************************************************************************************************************************/
//#include <Windows4Root.h>
//#include "w32pragma.h"
//#include <TROOT.h>
//#include <TMath.h>

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>


#include <algorithm>    // std::transform
#include <array>
//#include <boost/lambda/lambda.hpp>
#include <cmath>
#include <cstdarg>		// va_list, va_arg, va_start, va_end, va_copy
#include <cstdio>		// printf, sprintf,  
#include <cstdlib>		// rand, srand
#include <ctime>		// clock(), time() 
#include <fstream>
#include <functional>	// std::multiplies, std::plus, std::function, std::negate
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <map>
#include <new>			 
#include <numeric>		// inner_product, partial_sum, adjacent_difference, accumulate
//#include <omp.h>		// OpenMP
#include <sstream>
#include <stdexcept>
#include <string>
#include "sys/types.h"	// stat f
#include "sys/stat.h"	// stat functions
#include <tuple>
#include <typeinfo>		//operator typeid
#include <type_traits>	// is_pod
#include <utility>		// for std::move
#include <vector>
#if defined(_WIN32) || defined(_WIN64)
	#include <windows.h>
	#include "Shlwapi.h"	// Requires: Shlwapi.lib, Shlwapi.h, Shlwapi.dll (version 4.71 or later)
#endif
/***************************************************************************************************************************************************************************/
/************************************************************** Global typedef and namespace usage definitions *************************************************************/
/***************************************************************************************************************************************************************************/
//using namespace std::placeholders; 
//using namespace std;
using std::cout;
using std::endl;
typedef unsigned long long ULL;
typedef unsigned int uint;
/***************************************************************************************************************************************************************************/
/************************************************************************ User defined data types **********************************************************************/
/***************************************************************************************************************************************************************************/
struct generic_IO_container
{
//generic_IO_container(): string_input(""){};
	char* key;
	unsigned int input_type_ID;
	int integer_input;		// type_ID = 1
	double double_input;	// type_ID = 2
	char* string_input;	// type_ID = 3
};

// Container for all config file specified parameters allowing these to be transffered to GPU with single statements
// 8 UI, 18D, 6 C*
struct configurations
{
	char INPUT_DIRECTORY_D[256], OUTPUT_DIRECTORY_D[256], INPUT_FOLDER_D[256], OUTPUT_FOLDER_D[256], INPUT_BASE_NAME_D[32], FILE_EXTENSION_D[4];
	unsigned int GANTRY_ANGLES_D, NUM_SCANS_D, T_BINS_D, V_BINS_D, COLUMNS_D, ROWS_D, SLICES_D, SIGMAS_TO_KEEP_D;
	double SSD_T_SIZE_D, SSD_V_SIZE_D, T_SHIFT_D, U_SHIFT_D, T_BIN_SIZE_D, V_BIN_SIZE_D,  ANGULAR_BIN_SIZE_D, RECON_CYL_RADIUS_D, RECON_CYL_HEIGHT_D;
	double IMAGE_WIDTH_D, IMAGE_HEIGHT_D, IMAGE_THICKNESS_D, VOXEL_WIDTH_D, VOXEL_HEIGHT_D, VOXEL_THICKNESS_D, LAMBDA_D, LAMBDA;
	
	//configurations() : GANTRY_ANGLES_D(), NUM_SCANS_D(), T_BINS_D(), V_BINS_D(), COLUMNS_D(), ROWS_D(), SLICES_D(), SIGMAS_TO_KEEP_D(),
	//	SSD_T_SIZE_D(), SSD_V_SIZE_D(), T_SHIFT_D(), U_SHIFT_D(), T_BIN_SIZE_D(), V_BIN_SIZE_D(), ANGULAR_BIN_SIZE_D(),	
	//	RECON_CYL_RADIUS_D(), RECON_CYL_HEIGHT_D(), IMAGE_WIDTH_D(), IMAGE_HEIGHT_D(), IMAGE_THICKNESS_D(), VOXEL_WIDTH_D(), 
	//	VOXEL_HEIGHT_D(), VOXEL_THICKNESS_D(), LAMBDA_D(), LAMBDA() {};
	configurations(
		uint gantry_angles= 90, uint num_scans = 1, uint t_bins = 350, uint v_bins = 36, uint columns = 200, uint rows = 200, uint slices =  32, uint sigmas_to_keep =  3,
		double ssd_t_size =  35.0, double ssd_v_size =  9.0, double t_shift = 0.0, double u_shift = 0.0, double t_bin_size = 0.1, double v_bin_size = 0.25, double angular_bin_size = 4.0, double recon_cyl_radius = 10.0, 
		double recon_cyl_height = 8.0, double image_width =  20.0, double image_height = 20.0, double image_thickness = 8.0, double voxel_width = 20.0/200, double voxel_height =  20.0/200, double voxel_thickness =  0.25, 
		double lambda = 0.0001,  double lambda2 = 0.0001
	): 	
		GANTRY_ANGLES_D(gantry_angles), NUM_SCANS_D(num_scans), T_BINS_D(t_bins), V_BINS_D(v_bins), COLUMNS_D(columns), ROWS_D(rows), SLICES_D(slices), SIGMAS_TO_KEEP_D(sigmas_to_keep),
		SSD_T_SIZE_D(ssd_t_size), SSD_V_SIZE_D(ssd_v_size), T_SHIFT_D(t_shift), U_SHIFT_D(u_shift), T_BIN_SIZE_D(t_bin_size), V_BIN_SIZE_D(v_bin_size), ANGULAR_BIN_SIZE_D(angular_bin_size),	
		RECON_CYL_RADIUS_D(recon_cyl_radius), RECON_CYL_HEIGHT_D(recon_cyl_height), IMAGE_WIDTH_D(image_width), IMAGE_HEIGHT_D(image_height), IMAGE_THICKNESS_D(image_thickness), VOXEL_WIDTH_D(voxel_width), 
		VOXEL_HEIGHT_D(voxel_height), VOXEL_THICKNESS_D(voxel_thickness), LAMBDA_D(lambda), LAMBDA(lambda2) {};
};
enum SCAN_TYPE{ EXPERIMENTAL, SIMULATED};									// Experimental or simulated data
enum DATA_FORMATS { OLD_FORMAT, VERSION_0, VERSION_1 };						// Define the data formats that are supported
enum BIN_ANALYSIS_TYPE { MEANS, COUNTS, MEMBERS };							// Choices for what information about the binned data is desired 
enum BIN_ANALYSIS_FOR { ALL_BINS, SPECIFIC_BINS };							// Choices for which bins the desired data should come from
enum BIN_ORGANIZATION { BY_BIN, BY_HISTORY };								// Binned data is either organized in order by bin or by history w/ bin # specified separately
enum BIN_ANALYSIS_OF { WEPLS, ANGLES, POSITIONS, BIN_NUMS };				// Choices for which type of binned data is desired
enum FILTER_TYPES {RAM_LAK, SHEPP_LOGAN, NONE};								// Define the types of filters that are available for use in FBP
enum IMAGE_DEFINED_BY { SIZE_VOXELS, DIMENSIONS_VOXELS, SIZE_DIMENSIONS};	// Image size defined by 2 of voxel dimenensions, image dimensions, and image discretization
enum  HULL_TYPES {SC_HULL, MSC_HULL, SM_HULL, FBP_HULL };					// Define valid choices for which hull to use in MLP calculations
enum  INITIAL_ITERATE { X_HULL, FBP_IMAGE, HYBRID, ZEROS, IMPORT };			// Define valid choices for which hull to use in MLP calculations
enum RECON_ALGORITHMS { ART, DROP, BIP, SAP, ROBUST1, ROBUST2 };			// Define valid choices for iterative projection algorithm to use
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*--------------------------------------------------------------------- Preprocessing option parameters -------------------------------------------------------------------*/
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
const bool SAMPLE_STD_DEV		= true;							// Use sample/population standard deviation (T/F) in statistical cuts (i.e. divisor is N/N-1)
bool FBP_ON						= true;							// Turn FBP on (T) or off (F)
bool AVG_FILTER_FBP				= false;						// Apply averaging filter to initial iterate (T) or not (F)
bool MEDIAN_FILTER_FBP			= false; 
bool IMPORT_FILTERED_FBP		= false;
bool SC_ON						= false;						// Turn Space Carving on (T) or off (F)
bool MSC_ON						= true;							// Turn Modified Space Carving on (T) or off (F)
bool SM_ON						= false;						// Turn Space Modeling on (T) or off (F)
bool AVG_FILTER_HULL			= true;							// Apply averaging filter to hull (T) or not (F)
bool MLP_FILE_EXISTS			= true;
bool MLP_ENDPOINTS_FILE_EXISTS	= true;
bool REPERFORM_PREPROCESSING	= false;
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------- Input/output specifications and options ----------------------------------------------------------------*/
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/

/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*------------------------------------------------------------------ Path to the input/output directories -----------------------------------------------------------------*/
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
SCAN_TYPE DATA_TYPE = SIMULATED;
char PCT_DIRECTORY[]			= "D:\\pCT_Data\\";
char OBJECT[]					= "D:\\pCT_Data\\";
char RUN_DATE[]					= "D:\\pCT_Data\\";
char RUN_NUMBER[]				= "D:\\pCT_Data\\";
char PREPROCESS_DATE[]			= "D:\\pCT_Data\\";
char RECON_DATE[]				= "D:\\pCT_Data\\";
char PCT_IMAGES[]				= "D:\\pCT_Data\\";
char REFERENCE_IMAGES[]			= "D:\\pCT_Data\\";
char TEST_OUTPUT_FILE[]			= "export_testing.cfg";

char INPUT_BASE_NAME[]			= "projection";							// Prefix of the input data set filename
char FILE_EXTENSION[]			= ".bin";								// File extension for the input data
char CONFIG_DIRECTORY[]			= "C:\\Users\\Blake\\Documents\\Visual Studio 2010\\Projects\\robust_pct\\robust_pct\\";
char* CONFIG_FILE_NAME			= TEST_OUTPUT_FILE;
//char CONFIG_FILE_NAME[]		= "config_testing.cfg";
char EXPORT_FILE_NAME[]			= "export_testing.cfg";
char OUTPUT_FOLDER[]			= "CTP404\\input_CTP404_4M\\Robust2\\ETA0001\\";
char OUTPUT_DIRECTORY[]			= "D:\\pCT_Data\\Output\\";
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*------------------------------------------- Name of the folder where the input data resides and output data is to be written --------------------------------------------*/
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
char INPUT_FOLDER[]				= "CTP404\\input_CTP404_4M";
char INPUT_DIRECTORY[]			= "D:\\pCT_Data\\Input\\";
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*----------------------------------------------------------------- Input data specification parameters -------------------------------------------------------------------*/
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
DATA_FORMATS DATA_FORMAT		= VERSION_0;							// Specify which data format to use for this run
unsigned int PHANTOM_NAME_SIZE;
unsigned int DATA_SOURCE_SIZE;
unsigned int PREPARED_BY_SIZE;
unsigned int MAGIC_NUMBER_CHECK = int('DTCP');
unsigned int SKIP_2_DATA_SIZE;
unsigned int VERSION_ID;
unsigned int PROJECTION_INTERVAL;
bool BINARY_ENCODING			= true;									// Input data provided in binary (T) encoded files or ASCI text files (F)
bool SINGLE_DATA_FILE			= false;								// Individual file for each gantry angle (T) or single data file for all data (F)
bool SSD_IN_MM					= true;									// SSD distances from rotation axis given in mm (T) or cm (F)
bool DATA_IN_MM					= true;									// Input data given in mm (T) or cm (F)
char IMPORT_FBP_PATH[256];
char INPUT_ITERATE_PATH[256];
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*----------------------------------------------------------------------- Output option parameters ------------------------------------------------------------------------*/
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
bool WRITE_SC_HULL				= true;									// Write SC hull to disk (T) or not (F)
bool WRITE_MSC_COUNTS			= true;									// Write MSC counts array to disk (T) or not (F) before performing edge detection 
bool WRITE_MSC_HULL				= true;									// Write MSC hull to disk (T) or not (F)
bool WRITE_SM_COUNTS			= true;									// Write SM counts array to disk (T) or not (F) before performing edge detection 
bool WRITE_SM_HULL				= true;									// Write SM hull to disk (T) or not (F)
bool WRITE_FBP_IMAGE			= true;									// Write FBP image before thresholding to disk (T) or not (F)
bool WRITE_FBP_HULL				= true;									// Write FBP hull to disk (T) or not (F)
bool WRITE_AVG_FBP				= true;									// Write average filtered FBP image before thresholding to disk (T) or not (F)
bool WRITE_MEDIAN_FBP			= false;								// Write median filtered FBP image to disk (T) or not (F)
bool WRITE_FILTERED_HULL		= true;									// Write average filtered FBP image to disk (T) or not (F)
bool WRITE_X_HULL				= true;									// Write the hull selected to be used in MLP calculations to disk (T) or not (F)
bool WRITE_X_K0					= true;									// Write the hull selected to be used in MLP calculations to disk (T) or not (F)
bool WRITE_X_KI					= true;									// Write the hull selected to be used in MLP calculations to disk (T) or not (F)
bool WRITE_X					= true;									// Write the reructed image to disk (T) or not (F)
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*-------------------------------------------------------------- Binned data analysis options and parameters --------------------------------------------------------------*/
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
bool WRITE_BIN_WEPLS			= false;								// Write WEPLs for each bin to disk (T) for WEPL distribution analysis, or do not (F)
bool WRITE_WEPL_DISTS			= false;								// Write mean WEPL values to disk (T) or not (F): t bin = columns, v bin = rows, 1 angle per file
bool WRITE_SSD_ANGLES			= false;								// Write angles for each proton through entry/exit tracker planes to disk (T), or do not (F) 
/***************************************************************************************************************************************************************************/
/************************************************************************* Preprocessing Constants *************************************************************************/
/***************************************************************************************************************************************************************************/
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*--------------------------------------- Scanning and detector system	(source distance, tracking plane dimensions) parameters -------------------------------------------*/
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
#define GANTRY_ANGLE_INTERVAL	4.0										// [degrees] Angle between successive projection angles 
#define GANTRY_ANGLES			int( 360 / GANTRY_ANGLE_INTERVAL )		// [#] Total number of projection angles
#define NUM_SCANS				1										// [#] Total number of scans
#define NUM_FILES				( NUM_SCANS * GANTRY_ANGLES )			// [#] 1 file per gantry angle per translation
#define SSD_T_SIZE				35.0									// [cm] Length of SSD in t (lateral) direction
#define SSD_V_SIZE				9.0										// [cm] Length of SSD in v (vertical) direction
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*------------------------------------------------ Binning (for statistical analysis) and sinogram (for FBP) parameters ---------------------------------------------------*/
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
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
#define SIGMAS_TO_KEEP			3										// [#] Number of standard deviations from mean to allow before cutting the history 
const FILTER_TYPES FBP_FILTER	= SHEPP_LOGAN;		  					// Specifies which of the defined filters will be used in FBP
unsigned int FBP_AVG_RADIUS		= 1;
double FBP_AVG_THRESHOLD		= 0.1;
unsigned int FBP_MEDIAN_RADIUS	= 3;
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*------------------------------------------------------------------ Reconstruction cylinder parameters -------------------------------------------------------------------*/
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
#define RECON_CYL_RADIUS		10.0									// [cm] Radius of reconstruction cylinder
#define RECON_CYL_DIAMETER		( 2 * RECON_CYL_RADIUS )				// [cm] Diameter of reconstruction cylinder
#define RECON_CYL_HEIGHT		(SSD_V_SIZE - 1.0)						// [cm] Height of reconstruction cylinder
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*--------------------------------------------------------------------- Reconstruction image parameters -------------------------------------------------------------------*/
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
IMAGE_DEFINED_BY IMAGE_SIZING	= SIZE_DIMENSIONS;
#define IMAGE_WIDTH				RECON_CYL_DIAMETER						// [cm] Distance between left and right edges of each slice in image
#define IMAGE_HEIGHT			RECON_CYL_DIAMETER						// [cm] Distance between top and bottom edges of each slice in image
#define IMAGE_THICKNESS			( SLICES * SLICE_THICKNESS )			// [cm] Distance between bottom of bottom slice and top of the top slice of image
#define COLUMNS					200										// [#] Number of voxels in the x direction (i.e., number of columns) of image
#define ROWS					200										// [#] Number of voxels in the y direction (i.e., number of rows) of image
#define SLICES					int( RECON_CYL_HEIGHT / SLICE_THICKNESS)// [#] Number of voxels in the z direction (i.e., number of slices) of image
#define NUM_VOXELS				( COLUMNS * ROWS * SLICES )				// [#] Total number of voxels (i.e. 3-tuples [column, row, slice]) in image
#define VOXEL_WIDTH				( RECON_CYL_DIAMETER / COLUMNS )		// [cm] Distance between left and right edges of each voxel in image
#define VOXEL_HEIGHT			( RECON_CYL_DIAMETER / ROWS )			// [cm] Distance between top and bottom edges of each voxel in image
#define VOXEL_THICKNESS			( IMAGE_THICKNESS / SLICES )			// [cm] Distance between top and bottom of each slice in image
#define SLICE_THICKNESS			0.25									// [cm] Distance between top and bottom of each slice in image
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*----------------------------------------------------------------------- Hull-Detection Parameters -----------------------------------------------------------------------*/
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
#define MSC_DIFF_THRESH			50										// [#] Threshold on difference in counts between adjacent voxels used by MSC for edge detection
#define SC_THRESHOLD			0.0										// [cm] If WEPL < SC_THRESHOLD, SC assumes the proton missed the object
#define MSC_THRESHOLD			0.0										// [cm] If WEPL < MSC_THRESHOLD, MSC assumes the proton missed the object
#define SM_LOWER_THRESHOLD		6.0										// [cm] If WEPL >= SM_THRESHOLD, SM assumes the proton passed through the object
#define SM_UPPER_THRESHOLD		21.0									// [cm] If WEPL > SM_UPPER_THRESHOLD, SM ignores this history
#define SM_SCALE_THRESHOLD		1.0										// [cm] Threshold scaling factor used by SM to adjust edge detection sensitivity
#define HULL_FILTER_THRESHOLD	0.1										// [#] Threshold ([0.0, 1.0]) used by averaging filter to identify voxels belonging to the hull
#define HULL_FILTER_RADIUS		1										// [#] Averaging filter neighborhood radius in: [voxel - AVG_FILTER_SIZE, voxel + AVG_FILTER_RADIUS]
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------- MLP Parameters ----------------------------------------------------------------------------*/
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
HULL_TYPES						MLP_HULL = MSC_HULL;					// Specify which of the HULL_TYPES to use in this run's MLP calculations
#define VOXEL_STEP_SIZE			( VOXEL_WIDTH / 2 )						// [cm] Length of the step taken along the path, i.e. change in depth per step for
#define MLP_U_STEP				( VOXEL_WIDTH / 2)						// Size of the step taken along u direction during MLP; depth difference between successive MLP points
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*-------------------------------------------------------------- Iterative Image Reconstruction Parameters ----------------------------------------------------------------*/
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
bool AVG_FILTER_ITERATE			= false;								// Apply averaging filter to initial iterate (T) or not (F)
bool PREPROCESSING_REQUIRED		= false;
uint ITERATE_FILTER_RADIUS		= 3;
double ITERATE_FILTER_THRESHOLD = 0.1;
INITIAL_ITERATE	X_K0			= HYBRID;								// Specify which of the HULL_TYPES to use in this run's MLP calculations
RECON_ALGORITHMS RECONSTRUCTION = DROP;									// Specify which of the projection algorithms to use for image reconstruction
uint reconstruction_histories	= 0;
double LAMBDA					= 0.0001;
double ETA                      = 0.0001;
unsigned int METHOD             = 1;
int PSI_SIGN                    = 1;									// Relaxation parameter to use in image iterative projection reconstruction algorithms					
//#define LAMBDA				0.0001									// Relaxation parameter to use in image iterative projection reconstruction algorithms	
#define ITERATIONS				12										// # of iterations through the entire set of histories to perform in iterative image reconstruction
#define BLOCK_SIZE				60										// # of paths to use for each update: ART = 1, 
#define CONSTANT_CHORD_NORM		pow(VOXEL_WIDTH, 2.0)
double CONSTANT_LAMBDA_SCALE	= VOXEL_WIDTH * LAMBDA;
//#define CONSTANT_LAMBDA_SCALE	VOXEL_WIDTH * LAMBDA
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------- Memory allocation size for arrays (binning, image) -----------------------------------------------------------*/
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
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
/***************************************************************************************************************************************************************************/
/******************************************************************************* Constants *********************************************************************************/
/***************************************************************************************************************************************************************************/
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*------------------------------------------------------------ Host/GPU computation and structure information -------------------------------------------------------------*/
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
#define BYTES_PER_HISTORY		48										// [bytes] Data size of each history, 44 for actual data and 4 empty bytes, for old data format
#define MAX_GPU_HISTORIES		1500000									// [#] Number of histories to process on the GPU at a time, based on GPU capacity
#define MAX_CUTS_HISTORIES		1500000
#define THREADS_PER_BLOCK		1024									// [#] Number of threads assigned to each block on the GPU
#define STRIDE					5
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*------------------------------------------------------------------- FBP and FBP Filtering parameters --------------------------------------------------------------------*/
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
#define SOURCE_RADIUS			260.7									// [cm] Distance  to source/scatterer 
#define RAM_LAK_TAU				2/ROOT_TWO * T_BIN_SIZE					// Defines tau in Ram-Lak filter calculation, estimated from largest frequency in slice 
#define FBP_THRESHOLD			0.6										// [cm] RSP threshold used to generate FBP_hull from FBP_image
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------- Output filenames ----------------------------------------------------------------------------*/
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
char SC_HULL_FILENAME[]			= "x_SC";
char MSC_HULL_FILENAME[]		= "x_MSC";
char SM_HULL_FILENAME[]			= "x_SM";
char FBP_HULL_FILENAME[]		= "x_FBP";
char FBP_IMAGE_FILENAME[]		= "FBP_image";
char* MLP_PATHS_FILENAME		= "MLP_paths";
char* MLP_ENDPOINTS_FILENAME	= "MLP_endpoints";
char* INPUT_ITERATE_FILENAME	= "FBP_med7.bin";
char* IMPORT_FBP_FILENAME		= "FBP_med";
char* INPUT_HULL_FILENAME		= "hull.txt";
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------- MLP Parameters ----------------------------------------------------------------------------*/
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
#define E_0						13.6									// [MeV/c] empirical constant
#define X_0						36.08									// [cm] radiation length
#define RSP_AIR					0.00113									// [cm/cm] Approximate RSP of air
#define MAX_INTERSECTIONS		1000									// Limit on the # of intersections expected for proton's MLP; = # voxels along image diagonal

// Coefficients of 5th order polynomial fit to the term [1 / ( beta^2(u)*p^2(u) )] present in scattering covariance matrices Sigma 1/2 for:
#define BEAM_ENERGY				200
// 200 MeV protons
#define A_0						7.457  * pow( 10, -6.0  )
#define A_1						4.548  * pow( 10, -7.0  )
#define A_2						-5.777 * pow( 10, -8.0  )
#define A_3						1.301  * pow( 10, -8.0  )
#define A_4						-9.228 * pow( 10, -10.0 )
#define A_5						2.687  * pow( 10, -11.0 )

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
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*--------------------------------------------------------------------- Reconstruction image parameters -------------------------------------------------------------------*/
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
#define X_ZERO_COORDINATE		-RECON_CYL_RADIUS						// [cm] x-coordinate corresponding to front edge of 1st voxel (i.e. column) in image space
#define Y_ZERO_COORDINATE		RECON_CYL_RADIUS						// [cm] y-coordinate corresponding to front edge of 1st voxel (i.e. row) in image space
#define Z_ZERO_COORDINATE		RECON_CYL_HEIGHT/2						// [cm] z-coordinate corresponding to front edge of 1st voxel (i.e. slice) in image space
#define X_INCREASING_DIRECTION	RIGHT									// [#] specifies direction (LEFT/RIGHT) along x-axis in which voxel #s increase
#define Y_INCREASING_DIRECTION	DOWN									// [#] specifies direction (UP/DOWN) along y-axis in which voxel #s increase
#define Z_INCREASING_DIRECTION	DOWN									// [#] specifies direction (UP/DOWN) along z-axis in which voxel #s increase
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------ Precalculated Constants ------------------------------------------------------------------------*/
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
#define PHI						((1 + sqrt(5.0) ) / 2)					// Positive golden ratio, positive solution of PHI^2-PHI-1 = 0; also PHI = a/b when a/b = (a + b) / a 
#define PHI_NEGATIVE			((1 - sqrt(5.0) ) / 2)					// Negative golden ratio, negative solution of PHI^2-PHI-1 = 0; 
#define PI_OVER_4				( atan( 1.0 ) )							// 1*pi/4 radians =   pi/4 radians = 45 degrees
#define PI_OVER_2				( 2 * atan( 1.0 ) )						// 2*pi/4 radians =   pi/2 radians = 90 degrees
#define THREE_PI_OVER_4			( 3 * atan( 1.0 ) )						// 3*pi/4 radians = 3*pi/4 radians = 135 degrees
#define PI						( 4 * atan( 1.0 ) )						// 4*pi/4 radians =   pi   radians = 180 degrees
#define FIVE_PI_OVER_4			( 5 * atan( 1.0 ) )						// 5*pi/4 radians = 5*pi/4 radians = 225 degrees
#define SIX_PI_OVER_4			( 5 * atan( 1.0 ) )						// 6*pi/4 radians = 3*pi/2 radians = 270 degrees
#define SEVEN_PI_OVER_4			( 7 * atan( 1.0 ) )						// 7*pi/4 radians = 7*pi/4 radians = 315 degrees
#define TWO_PI					( 8 * atan( 1.0 ) )						// 8*pi/4 radians = 2*pi   radians = 360 degrees = 0 degrees
#define ANGLE_TO_RADIANS		( PI/180.0 )							// Convertion from angle to radians
#define RADIANS_TO_ANGLE		( 180.0/PI )							// Convertion from radians to angle
#define ROOT_TWO				sqrtf(2.0)								// 2^(1/2) = Square root of 2 
#define MM_TO_CM				0.1										// 10 [mm] = 1 [cm] => 1 [mm] = 0.1 [cm]
#define VOXEL_ALLOWANCE			1.0e-7
#define START					true									// Used as an alias for true for starting timer
#define STOP					false									// Used as an alias for false for stopping timer
#define RIGHT					1										// Specifies that moving right corresponds with an increase in x position, used in voxel walk 
#define LEFT					-1										// Specifies that moving left corresponds with a decrease in x position, used in voxel walk 
#define UP						1										// Specifies that moving up corresponds with an increase in y/z position, used in voxel walk 
#define DOWN					-1										// Specifies that moving down corresponds with a decrease in y/z position, used in voxel walk 
#define BOOL_FORMAT				"%d"									// Specifies format to use for writing/printing boolean data using {s/sn/f/v/vn}printf statements
#define CHAR_FORMAT				"%c"									// Specifies format to use for writing/printing character data using {s/sn/f/v/vn}printf statements
#define INT_FORMAT				"%d"									// Specifies format to use for writing/printing integer data using {s/sn/f/v/vn}printf statements
#define FLOAT_FORMAT			"%f"									// Specifies format to use for writing/printing floating point data using {s/sn/f/v/vn}printf statements
#define STRING_FORMAT			"%s"									// Specifies format to use for writing/printing strings data using {s/sn/f/v/vn}printf statements
/***************************************************************************************************************************************************************************/
/****************************************************************** Global Variable and Array Declerations *****************************************************************/
/***************************************************************************************************************************************************************************/
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*--------------------------------------------- Preprocessing and reconstruction configuration/parameter container definitions --------------------------------------------*/
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
configurations parameters;
configurations parameter_container;
configurations *parameters_h = &parameter_container;
configurations *parameters_d;

std::map<std::string,unsigned int> switchmap;
unsigned int num_run_arguments;
char** run_arguments;
//char* CONFIG_DIRECTORY;
char* input_directory;
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*------------------------------------------------------ Reconstruction history ordering and iterate update parameters ----------------------------------------------------*/
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
double lambda;
int parameter;
int num_voxel_scales;
double* voxel_scales;

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
ULL NUM_RECON_HISTORIES			= 20153778;
ULL PRIME_OFFSET				= 5038457;
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*------------------------------------- Declaration of arrays number of histories per file, projection, angle, total, and translation -------------------------------------*/
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
int total_histories = 0, recon_vol_histories = 0, maximum_histories_per_file = 0;
int* histories_per_projection, * histories_per_gantry_angle, * histories_per_file;
int* recon_vol_histories_per_projection;
int histories_per_scan[NUM_SCANS];
int post_cut_histories = 0;
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*--------------------------------------------- Declaration of array used to store tracking plane distances from rotation axis --------------------------------------------*/
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
std::vector<double> projection_angles;
float SSD_u_Positions[8];
float* ut_entry_angle, * uv_entry_angle, * ut_exit_angle, * uv_exit_angle; 
int zero_WEPL = 0;
int zero_WEPL_files = 0;
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*------------------------------------------------ Declaration of arrays for storage of input data for use on the host (_h) -----------------------------------------------*/
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
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
float* relative_ut_angle_h, * relative_uv_angle_h;
float* WEPL_h;
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------- Declaration of arrays for storage of input data for use on the device (_d) -----------------------------------------------*/
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
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
float* relative_ut_angle_d, * relative_uv_angle_d;
float* WEPL_d;
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*--------------------------------------------- Declaration of statistical analysis arrays for use on host(_h) or device (_d) ---------------------------------------------*/
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
float* mean_WEPL_h, * mean_WEPL_d;
float* mean_energy_h, * mean_energy_d;
float* mean_rel_ut_angle_h, * mean_rel_ut_angle_d;
float* mean_rel_uv_angle_h, * mean_rel_uv_angle_d;
float* mean_total_rel_angle_h, * mean_total_rel_angle_d;
float* stddev_rel_ut_angle_h, * stddev_rel_ut_angle_d;
float* stddev_rel_uv_angle_h, * stddev_rel_uv_angle_d;
float* stddev_WEPL_h, * stddev_WEPL_d;
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*------------------------------------------- Declaration of pre/post filter sinogram for FBP for use on host(_h) or device (_d) ------------------------------------------*/
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
float* sinogram_h, * sinogram_d;
float* sinogram_filtered_h, * sinogram_filtered_d;
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------- Declaration of image arrays for use on host(_h) or device (_d) -----------------------------------------------------*/
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
bool* SC_hull_h, * SC_hull_d;
bool* MSC_hull_h, * MSC_hull_d;
bool* SM_hull_h, * SM_hull_d;
bool* FBP_hull_h, * FBP_hull_d;
bool* x_hull_h, * x_hull_d;
int* MSC_counts_h, * MSC_counts_d;
int* SM_counts_h, * SM_counts_d;
int* MLP_test_image_h, * MLP_test_image_d;
float* FBP_image_h, * FBP_image_d;
float* FBP_image_filtered_h, * FBP_image_filtered_d;
float* FBP_median_filtered_h, * FBP_median_filtered_d;
double* x_update_h, * x_update_d;
unsigned int* num_voxel_intersections_h, * num_voxel_intersections_d;
unsigned int* intersection_counts_h, * intersection_counts_d;
unsigned int* block_voxels_h, *block_voxels_d;
unsigned int* block_counts_h, * block_counts_d;
double* norm_Ai;
float* x_h, * x_d;
ULL* history_sequence;
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*--------------------------------- Declaration of vectors used to accumulate data from histories that have passed currently applied cuts ---------------------------------*/
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
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
//std::vector<float>	relative_ut_angle_vector;	
//std::vector<float>	relative_uv_angle_vector;
std::vector<int> voxel_x_vector;
std::vector<int> voxel_y_vector;
std::vector<int> voxel_z_vector;
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------- Execution timer variables ------------------------------------------------------------------------*/
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
clock_t program_start, program_end, pause_cycles = 0;
/***************************************************************************************************************************************************************************/
/************************************************************************* For Use In Development **************************************************************************/
/***************************************************************************************************************************************************************************/
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*-------------------------------------------------------------------- Execution and early exit options -------------------------------------------------------------------*/
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
bool DEBUG_TEXT_ON				= true;							// Provide (T) or suppress (F) print statements to console during execution
bool RUN_ON						= false;						// Turn preprocessing on/off (T/F) to enter individual function testing without commenting
bool EXIT_AFTER_BINNING			= false;						// Exit program early after completing data read and initial processing
bool EXIT_AFTER_HULLS			= true;							// Exit program early after completing hull-detection
bool EXIT_AFTER_CUTS			= false;						// Exit program early after completing statistical cuts
bool EXIT_AFTER_SINOGRAM		= false;						// Exit program early after completing the ruction of the sinogram
bool EXIT_AFTER_FBP				= false;						// Exit program early after completing FBP
/***************************************************************************************************************************************************************************/
/************************************************************************ End of Parameter Definitions *********************************************************************/
/***************************************************************************************************************************************************************************/
//#endif // _PCT_RECONSTRUCTION_H_