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
//#include <C:\Users\Blake\Documents\GitHub\pCT_Reconstruction\Configurations.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>

//#include <boost/lambda/lambda.hpp>
//#include <boost/filesystem/operations.hpp>
//#include <boost/filesystem/path.hpp>

#include <algorithm>    // std::transform
#include <array>
#include <bitset>
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
	unsigned int input_type_ID;
	bool boolean_input;						// type_ID = 1
	int integer_input;						// type_ID = 2
	double double_input;					// type_ID = 3
	char* string_input;						// type_ID = 4
	char* key;
};

enum SCAN_TYPES { EXPERIMENTAL, SIMULATED_G, SIMULATED_T };					// Experimental or simulated data
enum DATA_FORMATS { OLD_FORMAT, VERSION_0, VERSION_1 };						// Define the data formats that are supported
enum DISK_WRITE_MODE { TEXT, BINARY };										// Experimental or simulated data
enum BIN_ANALYSIS_TYPE { MEANS, COUNTS, MEMBERS };							// Choices for what information about the binned data is desired 
enum BIN_ANALYSIS_FOR { ALL_BINS, SPECIFIC_BINS };							// Choices for which bins the desired data should come from
enum BIN_ORGANIZATION { BY_BIN, BY_HISTORY };								// Binned data is either organized in order by bin or by history w/ bin # specified separately
enum BIN_ANALYSIS_OF { WEPLS, ANGLES, POSITIONS, BIN_NUMS };				// Choices for which type of binned data is desired
enum FILTER_TYPES {RAM_LAK, SHEPP_LOGAN, NONE};								// Define the types of filters that are available for use in FBP
enum IMAGE_DEFINED_BY { SIZE_VOXELS, DIMENSIONS_VOXELS, SIZE_DIMENSIONS};	// Image size defined by 2 of voxel dimenensions, image dimensions, and image discretization
enum HULL_TYPES {SC_HULL, MSC_HULL, SM_HULL, FBP_HULL, IMPORT_HULL };		// Define valid choices for which hull to use in MLP calculations
enum INITIAL_ITERATE { X_HULL, X_FBP, HYBRID, ZEROS, IMPORT_X_0 };			// Define valid choices for which hull to use in MLP calculations
enum RECON_ALGORITHMS { ART, DROP, BIP, SAP, ROBUST1, ROBUST2 };			// Define valid choices for iterative projection algorithm to use
/***************************************************************************************************************************************************************************/
/********************************************************************* Host only configuration parameters ******************************************************************/
/***************************************************************************************************************************************************************************/
SCAN_TYPES DATA_TYPE			= SIMULATED_G;							// Specify the type of input data: EXPERIMENTAL, SIMULATED_G, SIMULATED_T
HULL_TYPES HULL					= MSC_HULL;								// Specify which of the HULL_TYPES to use in this run's MLP calculations
const FILTER_TYPES FBP_FILTER	= SHEPP_LOGAN;		  					// Specifies which of the defined filters will be used in FBP
INITIAL_ITERATE	X_0				= INITIAL_ITERATE(1);					// Specify which of the HULL_TYPES to use in this run's MLP calculations
RECON_ALGORITHMS RECONSTRUCTION = DROP;									// Specify which of the projection algorithms to use for image reconstruction
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*--------------------------------------------------------------------- Preprocessing option parameters -------------------------------------------------------------------*/
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
char * OBJECT, * SCAN_TYPE, * RUN_DATE, * RUN_NUMBER, * PROJECTION_DATA_DATE, * PREPROCESS_DATE, * RECONSTRUCTION_DATE;
char * PATH_2_PCT_DATA_DIR, * DATA_TYPE_DIR, * PROJECTION_DATA_DIR, * PREPROCESSING_DIR, * RECONSTRUCTION_DIR;
char * HULL_PATH, * FBP_PATH, * X_0_PATH, * MLP_PATH, * RECON_HISTORIES_PATH, * X_PATH;
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*--------------------------------------------------------------------- Preprocessing option parameters -------------------------------------------------------------------*/
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
bool FBP_ON						= true;									// Turn FBP on (T) or off (F)
bool AVG_FILTER_FBP				= false;								// Apply averaging filter to initial iterate (T) or not (F)
bool MEDIAN_FILTER_FBP			= false; 
bool IMPORT_FILTERED_FBP		= false;
bool SC_ON						= false;								// Turn Space Carving on (T) or off (F)
bool MSC_ON						= true;									// Turn Modified Space Carving on (T) or off (F)
bool SM_ON						= false;								// Turn Space Modeling on (T) or off (F)
bool AVG_FILTER_HULL			= true;									// Apply averaging filter to hull (T) or not (F)
bool AVG_FILTER_ITERATE			= false;								// Apply averaging filter to initial iterate (T) or not (F)
bool MLP_FILE_EXISTS			= false;
bool HISTORIES_FILE_EXISTS		= false;
bool REPERFORM_PREPROCESSING	= false;
bool PREPROCESSING_REQUIRED		= false;
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*----------------------------------------------------------------------- Output option parameters ------------------------------------------------------------------------*/
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
bool WRITE_MSC_COUNTS			= true;									// Write MSC counts array to disk (T) or not (F) before performing edge detection 
bool WRITE_SM_COUNTS			= true;									// Write SM counts array to disk (T) or not (F) before performing edge detection 
bool WRITE_X_FBP				= true;									// Write FBP image before thresholding to disk (T) or not (F)
bool WRITE_FBP_HULL				= true;									// Write FBP hull to disk (T) or not (F)
bool WRITE_AVG_FBP				= true;									// Write average filtered FBP image before thresholding to disk (T) or not (F)
bool WRITE_MEDIAN_FBP			= false;								// Write median filtered FBP image to disk (T) or not (F)
bool WRITE_BIN_WEPLS			= false;								// Write WEPLs for each bin to disk (T) for WEPL distribution analysis, or do not (F)
bool WRITE_WEPL_DISTS			= false;								// Write mean WEPL values to disk (T) or not (F): t bin = columns, v bin = rows, 1 angle per file
bool WRITE_SSD_ANGLES			= false;								// Write angles for each proton through entry/exit tracker planes to disk (T), or do not (F) 
/***************************************************************************************************************************************************************************/
/****************************************************************** GPU accessed configuration parameters ******************************************************************/
/***************************************************************************************************************************************************************************/
#define GANTRY_ANGLE_INTERVAL	4.0										// [degrees] Angle between successive projection angles 
#define SSD_T_SIZE				35.0									// [cm] Length of SSD in t (lateral) direction
#define SSD_V_SIZE				9.0										// [cm] Length of SSD in v (vertical) direction
#define T_SHIFT					0.0										// [cm] Amount by which to shift all t coordinates on input
#define U_SHIFT					0.0										// [cm] Amount by which to shift all u coordinates on input
#define V_SHIFT					0.0										// [cm] Amount by which to shift all v coordinates on input
#define T_BIN_SIZE				0.1										// [cm] Distance between adjacent bins in t (lateral) direction
#define V_BIN_SIZE				0.25									// [cm] Distance between adjacent bins in v (vertical) direction
#define ANGULAR_BIN_SIZE		4.0										// [degrees] Angle between adjacent bins in angular (rotation) direction
#define RECON_CYL_RADIUS		10.0									// [cm] Radius of reconstruction cylinder
#define COLUMNS					200										// [#] Number of voxels in the x direction (i.e., number of columns) of image
#define ROWS					200										// [#] Number of voxels in the y direction (i.e., number of rows) of image
#define SLICE_THICKNESS			0.25									// [cm] Distance between top and bottom of each slice in image
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*--------------------------------------------------------------------- Preprocessing option parameters -------------------------------------------------------------------*/
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
//#define LAMBDA				0.0001									// Relaxation parameter to use in image iterative projection reconstruction algorithms	
double LAMBDA					= 0.0001;								// Relaxation parameter to use in image iterative projection reconstruction algorithms		
double ETA                      = 0.0001;
int PSI_SIGN                    = 1;	
unsigned int ITERATIONS			= 12;									// # of iterations through the entire set of histories to perform in iterative image reconstruction
unsigned int BLOCK_SIZE			= 60;									// # of paths to use for each update: ART = 1, 
unsigned int HULL_FILTER_RADIUS = 1;									// [#] Averaging filter neighborhood radius in: [voxel - AVG_FILTER_SIZE, voxel + AVG_FILTER_RADIUS]
unsigned int X_0_FILTER_RADIUS	= 3;
unsigned int FBP_AVG_RADIUS		= 1;
int FBP_MEDIAN_RADIUS			= 3;									
double HULL_FILTER_THRESHOLD	= 0.1;									// [#] Threshold ([0.0, 1.0]) used by averaging filter to identify voxels belonging to the hull
double FBP_AVG_THRESHOLD		= 0.1;
double X_0_FILTER_THRESHOLD		= 0.1;
unsigned int NUM_SCANS			= 1;									// [#] Total number of scans of same object
unsigned int MAX_GPU_HISTORIES	= 1500000;								// [#] Number of histories to process on the GPU at a time, based on GPU capacity
unsigned int MAX_CUTS_HISTORIES = 1500000;	
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*----------------------------------------------------------------------- Hull-Detection Parameters -----------------------------------------------------------------------*/
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
#define MSC_DIFF_THRESH			50										// [#] Threshold on difference in counts between adjacent voxels used by MSC for edge detection
#define SC_THRESHOLD			0.0										// [cm] If WEPL < SC_THRESHOLD, SC assumes the proton missed the object
#define MSC_THRESHOLD			0.0										// [cm] If WEPL < MSC_THRESHOLD, MSC assumes the proton missed the object
#define SM_LOWER_THRESHOLD		6.0										// [cm] If WEPL >= SM_THRESHOLD, SM assumes the proton passed through the object
#define SM_UPPER_THRESHOLD		21.0									// [cm] If WEPL > SM_UPPER_THRESHOLD, SM ignores this history
#define SM_SCALE_THRESHOLD		1.0										// [cm] Threshold scaling factor used by SM to adjust edge detection sensitivity
/***************************************************************************************************************************************************************************/
/***************************************************** Derived values from configuration parameters using constant expressions *********************************************/
/***************************************************************************************************************************************************************************/
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*--------------------------------------------------------------------- Reconstruction image parameters -------------------------------------------------------------------*/
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
#define GANTRY_ANGLES			uint( 360 / GANTRY_ANGLE_INTERVAL )		// [#] Total number of projection angles
#define NUM_FILES				( NUM_SCANS * GANTRY_ANGLES )			// [#] 1 file per gantry angle per translation
#define T_BINS					uint( SSD_T_SIZE / T_BIN_SIZE + 0.5 )	// [#] Number of bins (i.e. quantization levels) for t (lateral) direction 
#define V_BINS					uint( SSD_V_SIZE/ V_BIN_SIZE + 0.5 )		// [#] Number of bins (i.e. quantization levels) for v (vertical) direction 
#define ANGULAR_BINS			uint( 360 / ANGULAR_BIN_SIZE + 0.5 )		// [#] Number of bins (i.e. quantization levels) for path angle 
#define NUM_BINS				( ANGULAR_BINS * T_BINS * V_BINS )		// [#] Total number of bins corresponding to possible 3-tuples [ANGULAR_BIN, T_BIN, V_BIN]
#define RECON_CYL_HEIGHT		(SSD_V_SIZE - 1.0)						// [cm] Height of reconstruction cylinder
#define RECON_CYL_DIAMETER		( 2 * RECON_CYL_RADIUS )				// [cm] Diameter of reconstruction cylinder
#define SLICES					uint( RECON_CYL_HEIGHT / SLICE_THICKNESS)// [#] Number of voxels in the z direction (i.e., number of slices) of image
#define NUM_VOXELS				( COLUMNS * ROWS * SLICES )				// [#] Total number of voxels (i.e. 3-tuples [column, row, slice]) in image
#define IMAGE_WIDTH				RECON_CYL_DIAMETER						// [cm] Distance between left and right edges of each slice in image
#define IMAGE_HEIGHT			RECON_CYL_DIAMETER						// [cm] Distance between top and bottom edges of each slice in image
#define IMAGE_THICKNESS			RECON_CYL_HEIGHT						// [cm] Distance between bottom of bottom slice and top of the top slice of image
#define VOXEL_WIDTH				( IMAGE_WIDTH / COLUMNS )				// [cm] Distance between left and right edges of each voxel in image
#define VOXEL_HEIGHT			( IMAGE_HEIGHT / ROWS )					// [cm] Distance between top and bottom edges of each voxel in image
#define VOXEL_THICKNESS			( IMAGE_THICKNESS / SLICES )			// [cm] Distance between top and bottom of each slice in image
#define X_ZERO_COORDINATE		-RECON_CYL_RADIUS						// [cm] x-coordinate corresponding to left edge of 1st voxel (i.e. column) in image space
#define Y_ZERO_COORDINATE		RECON_CYL_RADIUS						// [cm] y-coordinate corresponding to top edge of 1st voxel (i.e. row) in image space
#define Z_ZERO_COORDINATE		RECON_CYL_HEIGHT/2						// [cm] z-coordinate corresponding to top edge of 1st voxel (i.e. slice) in image space
#define RAM_LAK_TAU				2/ROOT_TWO * T_BIN_SIZE					// Defines tau in Ram-Lak filter calculation, estimated from largest frequency in slice 
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------- Memory allocation size for arrays (binning, image) -----------------------------------------------------------*/
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
#define SIZE_BINS_CHAR			( NUM_BINS   * sizeof(char)	 )			// Amount of memory required for a character array used for binning
#define SIZE_BINS_BOOL			( NUM_BINS   * sizeof(bool)	 )			// Amount of memory required for a boolean array used for binning
#define SIZE_BINS_INT			( NUM_BINS   * sizeof(int)	 )			// Amount of memory required for a integer array used for binning
#define SIZE_BINS_UINT			( NUM_BINS   * sizeof(uint)	 )			// Amount of memory required for a integer array used for binning
#define SIZE_BINS_FLOAT			( NUM_BINS	 * sizeof(float) )			// Amount of memory required for a floating point array used for binning
#define SIZE_IMAGE_CHAR			( NUM_VOXELS * sizeof(char)	 )			// Amount of memory required for a character array used for binning
#define SIZE_IMAGE_BOOL			( NUM_VOXELS * sizeof(bool)	 )			// Amount of memory required for a boolean array used for binning
#define SIZE_IMAGE_INT			( NUM_VOXELS * sizeof(int)	 )			// Amount of memory required for a integer array used for binning
#define SIZE_IMAGE_UINT			( NUM_VOXELS * sizeof(unsigned int)	 )	// Amount of memory required for a integer array used for binning
#define SIZE_IMAGE_FLOAT		( NUM_VOXELS * sizeof(float) )			// Amount of memory required for a floating point array used for binning
#define SIZE_IMAGE_DOUBLE		( NUM_VOXELS * sizeof(double) )			// Amount of memory required for a floating point array used for binning
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*-------------------------------------------------------------- Iterative Image Reconstruction Parameters ----------------------------------------------------------------*/
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
#define VOXEL_STEP_SIZE			( VOXEL_WIDTH / 2 )						// [cm] Length of the step taken along the path, i.e. change in depth per step for
#define MLP_U_STEP				( VOXEL_WIDTH / 2)						// Size of the step taken along u direction during MLP; depth difference between successive MLP points
#define CONSTANT_CHORD_NORM		pow(VOXEL_WIDTH, 2.0)
double CONSTANT_LAMBDA_SCALE	= VOXEL_WIDTH * LAMBDA;
/***************************************************************************************************************************************************************************/
/******************************************************************************* Constants *********************************************************************************/
/***************************************************************************************************************************************************************************/
const bool SAMPLE_STD_DEV		= true;									// Use sample/population standard deviation (T/F) in statistical cuts (i.e. divisor is N/N-1)
unsigned int MAGIC_NUMBER_CHECK = int('DTCP');
#define STRIDE					5
#define SIGMAS_2_KEEP			3										// [#] Number of standard deviations from mean to allow before cutting the history 
#define THREADS_PER_BLOCK		1024									// [#] Number of threads assigned to each block on the GPU
#define BYTES_PER_HISTORY		48										// [bytes] Data size of each history, 44 for actual data and 4 empty bytes, for old data format
#define SOURCE_RADIUS			260.7									// [cm] Distance  to source/scatterer 
#define FBP_THRESHOLD			0.6										// [cm] RSP threshold used to generate FBP_hull from x_FBP
#define MAX_INTERSECTIONS		1000									// Limit on the # of intersections expected for proton's MLP; = # voxels along image diagonal
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