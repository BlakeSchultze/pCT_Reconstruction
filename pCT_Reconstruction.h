//#ifndef PCT_RECONSTRUCTION_H
//#define PCT_RECONSTRUCTION_H
#pragma once
/***************************************************************************************************************************************************************************/
/********************************************************************* Header for pCT reconstruction program ***************************************************************/
/***************************************************************************************************************************************************************************/
#include <Windows4Root.h>
//#include "w32pragma.h"
//#include <TROOT.h>
#include <TMath.h>

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>


#include <algorithm>    // std::transform
#include <array>
#include <cmath>
#include <cstdarg>		// va_list, va_arg, va_start, va_end, va_copy
#include <cstdio>		// printf, sprintf,  
#include <cstdlib>		// rand, srand
#include <ctime>		// clock(), time() 
#include <fstream>
#include <functional>	// std::multiplies, std::plus, std::function, std::negate
#include <initializer_list>
#include <iostream>
#include <new>			 
#include <numeric>		// inner_product, partial_sum, adjacent_difference, accumulate
#include <omp.h>		// OpenMP
#include <sstream>
#include <typeinfo>		//operator typeid
#include <type_traits>	// is_pod
#include <vector>

using namespace std::placeholders; 
//using namespace std;
using std::cout;
using std::endl;
/***************************************************************************************************************************************************************************/
/***************************************************************************************************************************************************************************/
/************************************************************************ Preprocessing usage options **********************************************************************/
/***************************************************************************************************************************************************************************/
/***************************************************************************************************************************************************************************/

/***************************************************************************************************************************************************************************/
/********************************************************************* Execution and early exit options ********************************************************************/
/***************************************************************************************************************************************************************************/
const bool RUN_ON			   = true;									// Turn preprocessing on/off (T/F) to enter individual function testing without commenting
const bool EXIT_AFTER_BINNING  = false;									// Exit program early after completing data read and initial processing
const bool EXIT_AFTER_HULLS    = false;									// Exit program early after completing hull-detection
const bool EXIT_AFTER_CUTS     = false;									// Exit program early after completing statistical cuts
const bool EXIT_AFTER_SINOGRAM = false;									// Exit program early after completing the construction of the sinogram
const bool EXIT_AFTER_FBP	   = false;									// Exit program early after completing FBP
/***************************************************************************************************************************************************************************/
/********************************************************************** Preprocessing option parameters ********************************************************************/
/***************************************************************************************************************************************************************************/
const bool DEBUG_TEXT_ON	   = true;									// Provide (T) or suppress (F) print statements to console during execution
const bool SAMPLE_STD_DEV	   = true;									// Use sample/population standard deviation (T/F) in statistical cuts (i.e. divisor is N/N-1)
const bool FBP_ON			   = false;									// Turn FBP on (T) or off (F)
const bool SC_ON			   = false;									// Turn Space Carving on (T) or off (F)
const bool MSC_ON			   = true;									// Turn Modified Space Carving on (T) or off (F)
const bool SM_ON			   = false;									// Turn Space Modeling on (T) or off (F)
const bool HULL_FILTER_ON	   = false;									// Apply averaging filter to hull (T) or not (F)
const bool COUNT_0_WEPLS	   = false;									// Count the number of histories with WEPL = 0 (T) or not (F)
const bool REALLOCATE		   = false;
/***************************************************************************************************************************************************************************/
/***************************************************************** Input/output specifications and options *****************************************************************/
/***************************************************************************************************************************************************************************/

/***************************************************************************************************************************************************************************/
/******************************************************************* Path to the input/output directories ******************************************************************/
/***************************************************************************************************************************************************************************/
const char INPUT_DIRECTORY[]   = "C:\\Users\\Blake\\Documents\\Visual Studio 2010\\Projects\\pCT_Reconstruction\\Input\\";
const char OUTPUT_DIRECTORY[]  = "C:\\Users\\Blake\\Documents\\Visual Studio 2010\\Projects\\pCT_Reconstruction\\Output\\";
/***************************************************************************************************************************************************************************/
/******************************************** Name of the folder where the input data resides and output data is to be written *********************************************/
/***************************************************************************************************************************************************************************/
const char INPUT_FOLDER[]	   = "input_CTP404";
const char OUTPUT_FOLDER[]	   = "input_CTP404";
//const char INPUT_FOLDER[]	   = "input_water_GeantNONUC";
//const char OUTPUT_FOLDER[]	   = "input_water_GeantNONUC";
//const char INPUT_FOLDER[]	   = "input_water_Geant500000";
//const char OUTPUT_FOLDER[]   = "input_water_Geant500000";
//const char INPUT_FOLDER[]	   = "waterPhantom";
//const char OUTPUT_FOLDER[]   = "waterPhantom";
//const char INPUT_FOLDER[]	   = "catphan";
//const char OUTPUT_FOLDER[]   = "catphan";
//const char INPUT_FOLDER[]	   = "DetectData";
//const char OUTPUT_FOLDER[]   = "DetectData";
//const char INPUT_FOLDER[]	   = "Rat_Scan2";
//const char OUTPUT_FOLDER[]   = "Rat_Scan2"; 
//const char INPUT_FOLDER[]	   = "DetectDataWeplNoisy1";
//const char OUTPUT_FOLDER[]   = "DetectDataWeplNoisy1";
//const char INPUT_FOLDER[]	   = "input_noisefloor40";
//const char OUTPUT_FOLDER[]   = "input_noisefloor40";
/***************************************************************************************************************************************************************************/
/**************************************** Prefix of the input data set filename (_trans%d_%03d.txt (or .dat) will be added to this) ****************************************/
/***************************************************************************************************************************************************************************/
const char INPUT_BASE_NAME[]   = "projection";							// waterPhantom, catphan, input_water_Geant500000
//const char INPUT_BASE_NAME[] = "simdata";								// Simulated data sets generated by Micah: DetectData, DetectDataWeplNoisy1, NoisyUniform1,...
//const char INPUT_BASE_NAME[] = "rat_scan2_shift";						// Anesthetized rat held in restraints
//const char INPUT_BASE_NAME[] = "ped_scan1";							// Anthropomorphic pediatric head phantom (Model 715-HN, CIRS1)
/***************************************************************************************************************************************************************************/
/******************************************************************** File extension for the input data ********************************************************************/
/***************************************************************************************************************************************************************************/
const char FILE_EXTENSION[]	   = ".bin";								// Binary file extension
//const char FILE_EXTENSION[]  = ".dat";								// Generic data file extension, independent of encoding (various encodings can be used)
//const char FILE_EXTENSION[]  = ".txt";								// ASCII text file extension
/***************************************************************************************************************************************************************************/
/****************************************************************** Input data specification parameters ********************************************************************/
/***************************************************************************************************************************************************************************/
enum DATA_FORMATS { OLD_FORMAT, VERSION_0, VERSION_1 };					// Define the data formats that are supported
const DATA_FORMATS				DATA_FORMAT = VERSION_0;				// Specify which data format to use for this run
const bool BINARY_ENCODING	   = true;									// Input data provided in binary (T) encoded files or ASCI text files (F)
const bool SINGLE_DATA_FILE    = false;									// Individual file for each gantry angle (T) or single data file for all data (F)
const bool SSD_IN_MM		   = true;									// SSD distances from rotation axis given in mm (T) or cm (F)
const bool DATA_IN_MM		   = true;									// Input data given in mm (T) or cm (F)
const bool MICAH_SIM		   = false;									// Specify whether the input data is from Micah's simulator (T) or not (F)
/***************************************************************************************************************************************************************************/
/**************************************************************************** Output filenames *****************************************************************************/
/***************************************************************************************************************************************************************************/
const char SC_HULL_FILENAME[]  = "x_SC";
const char MSC_HULL_FILENAME[] = "x_MSC";
const char SM_HULL_FILENAME[]  = "x_SM";
const char FBP_HULL_FILENAME[] = "x_FBP";
const char FBP_IMAGE_FILENAME[]= "FBP_image";
const char MLP_PATH_FILENAME[] = "MLP_paths";
/***************************************************************************************************************************************************************************/
/************************************************************************ Output option parameters *************************************************************************/
/***************************************************************************************************************************************************************************/
const bool WRITE_SC_HULL	   = true;									// Write SC hull to disk (T) or not (F)
const bool WRITE_MSC_COUNTS    = true;									// Write MSC counts array to disk (T) or not (F) before performing edge detection 
const bool WRITE_MSC_HULL	   = true;									// Write MSC hull to disk (T) or not (F)
const bool WRITE_SM_COUNTS	   = true;									// Write SM counts array to disk (T) or not (F) before performing edge detection 
const bool WRITE_SM_HULL	   = true;									// Write SM hull to disk (T) or not (F)
const bool WRITE_FBP_IMAGE	   = true;									// Write FBP image before thresholding to disk (T) or not (F)
const bool WRITE_FBP_HULL	   = true;									// Write FBP hull to disk (T) or not (F)
const bool WRITE_FILTERED_HULL = true;									// Write average filtered hull to disk (T) or not (F)
const bool WRITE_X_HULL		   = true;									// Write the hull selected to be used in MLP calculations to disk (T) or not (F)
const bool WRITE_X_K0		   = true;									// Write the hull selected to be used in MLP calculations to disk (T) or not (F)
const bool WRITE_X_KI		   = true;									// Write the hull selected to be used in MLP calculations to disk (T) or not (F)
const bool WRITE_X			   = true;									// Write the reconstructed image to disk (T) or not (F)
/***************************************************************************************************************************************************************************/
/*************************************************************** Binned data analysis options and parameters ***************************************************************/
/***************************************************************************************************************************************************************************/
enum BIN_ANALYSIS_TYPE { MEANS, COUNTS, MEMBERS };						// Choices for what information about the binned data is desired 
enum BIN_ANALYSIS_FOR { ALL_BINS, SPECIFIC_BINS };						// Choices for which bins the desired data should come from
enum BIN_ORGANIZATION { BY_BIN, BY_HISTORY };							// Binned data is either organized in order by bin or by history w/ bin # specified separately
enum BIN_ANALYSIS_OF { WEPLS, ANGLES, POSITIONS, BIN_NUMS };			// Choices for which type of binned data is desired
const bool WRITE_BIN_WEPLS	   = false;									// Write WEPLs for each bin to disk (T) for WEPL distribution analysis, or do not (F)
const bool WRITE_WEPL_DISTS	   = false;									// Write mean WEPL values to disk (T) or not (F): t bin = columns, v bin = rows, 1 angle per file
const bool WRITE_SSD_ANGLES    = false;									// Write angles for each proton through entry/exit tracker planes to disk (T), or do not (F) 
/***************************************************************************************************************************************************************************/
/***************************************************************************************************************************************************************************/
/************************************************************************* Preprocessing Constants *************************************************************************/
/***************************************************************************************************************************************************************************/
/***************************************************************************************************************************************************************************/

/***************************************************************************************************************************************************************************/
/************************************************************* Host/GPU computation and structure information **************************************************************/
/***************************************************************************************************************************************************************************/
#define BYTES_PER_HISTORY		48										// [bytes] Data size of each history, 44 for actual data and 4 empty bytes, for old data format
#define MAX_GPU_HISTORIES		1100000									// [#] Number of histories to process on the GPU at a time, based on GPU capacity
#define THREADS_PER_BLOCK		1024									// [#] Number of threads assigned to each block on the GPU
/***************************************************************************************************************************************************************************/
/**************************************** Scanning and detector system	(source distance, tracking plane dimensions) parameters ********************************************/
/***************************************************************************************************************************************************************************/
#define SOURCE_RADIUS			260.7									// [cm] Distance  to source/scatterer 
#define GANTRY_ANGLE_INTERVAL	4.0										// [degrees] Angle between successive projection angles 
#define GANTRY_ANGLES			int( 360 / GANTRY_ANGLE_INTERVAL )		// [#] Total number of projection angles
#define NUM_SCANS				1										// [#] Total number of scans
#define NUM_FILES				( NUM_SCANS * GANTRY_ANGLES )			// [#] 1 file per gantry angle per translation
#define SSD_T_SIZE				35.0									// [cm] Length of SSD in t (lateral) direction
#define SSD_V_SIZE				9.0										// [cm] Length of SSD in v (vertical) direction
/***************************************************************************************************************************************************************************/
/************************************************* Binning (for statistical analysis) and sinogram (for FBP) parameters ****************************************************/
/***************************************************************************************************************************************************************************/
#define T_SHIFT					0.0										// [cm] Amount by which to shift all t coordinates on input
#define U_SHIFT					0.0										// [cm] Amount by which to shift all v coordinates on input
//#define T_SHIFT				   2.05									// [cm] Amount by which to shift all t coordinates on input
//#define U_SHIFT				   -0.16								// [cm] Amount by which to shift all v coordinates on input
#define T_BIN_SIZE				0.1										// [cm] Distance between adjacent bins in t (lateral) direction
#define T_BINS					int( SSD_T_SIZE / T_BIN_SIZE + 0.5 )	// [#] Number of bins (i.e. quantization levels) for t (lateral) direction 
#define V_BIN_SIZE				0.25									// [cm] Distance between adjacent bins in v (vertical) direction
#define V_BINS					int( SSD_V_SIZE / V_BIN_SIZE + 0.5 )	// [#] Number of bins (i.e. quantization levels) for v (vertical) direction 
#define ANGULAR_BIN_SIZE		4.0										// [degrees] Angle between adjacent bins in angular (rotation) direction
#define ANGULAR_BINS			int( 360 / ANGULAR_BIN_SIZE + 0.5 )		// [#] Number of bins (i.e. quantization levels) for path angle 
#define NUM_BINS				( ANGULAR_BINS * T_BINS * V_BINS )		// [#] Total number of bins corresponding to possible 3-tuples [ANGULAR_BIN, T_BIN, V_BIN]
#define SIGMAS_TO_KEEP			3										// [#] Number of standard deviations from mean to allow before cutting the history 
enum FILTER_TYPES {RAM_LAK, SHEPP_LOGAN, NONE};							// Define the types of filters that are available for use in FBP
const FILTER_TYPES				FBP_FILTER = SHEPP_LOGAN;			  	// Specifies which of the defined filters will be used in FBP
#define RAM_LAK_TAU				2/ROOT_TWO * T_BIN_SIZE					// Defines tau in Ram-Lak filter calculation, estimated from largest frequency in slice 
#define AVG_FILTER_THRESHOLD	0.1										// [#] Threshold ([0.0, 1.0]) used by averaging filter to identify voxels belonging to the hull
#define AVG_FILTER_RADIUS		2										// [#] Averaging filter neighborhood radius in: [voxel - AVG_FILTER_SIZE, voxel + AVG_FILTER_RADIUS]
#define FBP_THRESHOLD			0.6										// [cm] RSP threshold used to generate FBP_hull from FBP_image
/***************************************************************************************************************************************************************************/
/******************************************************************* Reconstruction cylinder parameters ********************************************************************/
/***************************************************************************************************************************************************************************/
#define RECON_CYL_RADIUS		10.0										// [cm] Radius of reconstruction cylinder
#define RECON_CYL_DIAMETER		( 2 * RECON_CYL_RADIUS )					// [cm] Diameter of reconstruction cylinder
#define RECON_CYL_HEIGHT		(SSD_V_SIZE - 1.0)						// [cm] Height of reconstruction cylinder
/***************************************************************************************************************************************************************************/
/********************************************************************	Reconstruction image parameters ********************************************************************/
/***************************************************************************************************************************************************************************/
#define IMAGE_WIDTH				RECON_CYL_DIAMETER				// [cm] Distance between left and right edges of each slice in image
#define IMAGE_HEIGHT			RECON_CYL_DIAMETER					// [cm] Distance between top and bottom edges of each slice in image
#define IMAGE_THICKNESS			( SLICES * SLICE_THICKNESS )			// [cm] Distance between bottom of bottom slice and top of the top slice of image
#define COLUMNS					200										// [#] Number of voxels in the x direction (i.e., number of columns) of image
#define ROWS					200										// [#] Number of voxels in the y direction (i.e., number of rows) of image
#define SLICES					int( RECON_CYL_HEIGHT / SLICE_THICKNESS )// [#] Number of voxels in the z direction (i.e., number of slices) of image
#define NUM_VOXELS					( COLUMNS * ROWS * SLICES )				// [#] Total number of voxels (i.e. 3-tuples [column, row, slice]) in image
#define VOXEL_WIDTH				( RECON_CYL_DIAMETER / COLUMNS )	// [cm] Distance between left and right edges of each voxel in image
#define VOXEL_HEIGHT				( RECON_CYL_DIAMETER / ROWS )		// [cm] Distance between top and bottom edges of each voxel in image
#define VOXEL_THICKNESS			( IMAGE_THICKNESS / SLICES )		// [cm] Distance between top and bottom of each slice in image
#define SLICE_THICKNESS			0.25								// [cm] Distance between top and bottom of each slice in image

//#define RECON_CYL_RADIUS		5.0										// [cm] Radius of reconstruction cylinder
//#define RECON_CYL_HEIGHT		5.0										// [cm] Height of reconstruction cylinder
//#define COLUMNS					100										// [#] Number of voxels in the x direction (i.e., number of columns) of image
//#define ROWS					100										// [#] Number of voxels in the y direction (i.e., number of rows) of image
//#define SLICES					5
//#define VOXEL_WIDTH				0.1										// [cm] Distance between left and right edges of each voxel in image
//#define VOXEL_HEIGHT			0.1										// [cm] Distance between top and bottom edges of each voxel in image
//#define VOXEL_THICKNESS			1.0										// [cm] Distance between top and bottom of each slice in image
//#define SLICE_THICKNESS			1.0										// [cm] Distance between top and bottom of each slice in image
/***************************************************************************************************************************************************************************/
/************************************************************************ Hull-Detection Parameters ************************************************************************/
/***************************************************************************************************************************************************************************/
#define MSC_DIFF_THRESH			50										// [#] Threshold on difference in counts between adjacent voxels used by MSC for edge detection
#define SC_THRESHOLD			0.0										// [cm] If WEPL < SC_THRESHOLD, SC assumes the proton missed the object
#define MSC_THRESHOLD			0.0										// [cm] If WEPL < MSC_THRESHOLD, MSC assumes the proton missed the object
#define SM_LOWER_THRESHOLD		6.0										// [cm] If WEPL >= SM_THRESHOLD, SM assumes the proton passed through the object
#define SM_UPPER_THRESHOLD		21.0										// [cm] If WEPL > SM_UPPER_THRESHOLD, SM ignores this history
#define SM_SCALE_THRESHOLD		1.0										// [cm] Threshold scaling factor used by SM to adjust edge detection sensitivity
/***************************************************************************************************************************************************************************/
/****************************************************************************** MLP Parameters *****************************************************************************/
/***************************************************************************************************************************************************************************/
enum  HULL_TYPES {SC_HULL, MSC_HULL, SM_HULL, FBP_HULL };				// Define valid choices for which hull to use in MLP calculations
const HULL_TYPES				MLP_HULL = MSC_HULL;					// Specify which of the HULL_TYPES to use in this run's MLP calculations
#define E_0						13.6									// [MeV/c] empirical constant
#define X_0						36.1									// [cm] radiation length
#define RSP_AIR					0.00113									// [cm/cm] Approximate RSP of air
#define VOXEL_STEP_SIZE			( VOXEL_WIDTH / 2 )						// [cm] Length of the step taken along the path, i.e. change in depth per step for
#define MAX_INTERSECTIONS		1000									// Limit on the # of intersections expected for proton's MLP; = # voxels along image diagonal
#define MLP_u_step (min(VOXEL_WIDTH, VOXEL_HEIGHT) / 2) 
const int max_path_elements = int(sqrt(double( ROWS^2 + COLUMNS^2 + SLICES^2)));
// 200 MeV coefficients
#define A_0						7.457  * pow( 10, -6.0  ) 
#define A_1						4.548  * pow( 10, -7.0  )
#define A_2						-5.777 * pow( 10, -8.0  )
#define A_3						1.301  * pow( 10, -8.0  )
#define A_4						-9.228 * pow( 10, -10.0 )
#define A_5						2.687  * pow( 10, -11.0 )
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
/***************************************************************************************************************************************************************************/
/*************************************************************** Iterative Image Reconstruction Parameters *****************************************************************/
/***************************************************************************************************************************************************************************/
enum  INITIAL_ITERATE { X_HULL, FBP_IMAGE, HYBRID, ZEROS };					// Define valid choices for which hull to use in MLP calculations
const INITIAL_ITERATE			X_K0 = X_HULL;							// Specify which of the HULL_TYPES to use in this run's MLP calculations
#define LAMBDA					0.0001										// Relaxation parameter to use in image iterative projection reconstruction algorithms
enum PROJECTION_ALGORITHMS { ART, SART, DROP, BIP, SAP };				// Define valid choices for iterative projection algorithm to use
const PROJECTION_ALGORITHMS		PROJECTION_ALGORITHM = ART;				// Specify which of the projection algorithms to use for image reconstruction
#define BLOCK_SIZE				1										// # of paths to use for each update: ART = 1, 
/***************************************************************************************************************************************************************************/
/*********************************************************** Memory allocation size for arrays (binning, image) ************************************************************/
/***************************************************************************************************************************************************************************/
#define SIZE_BINS_CHAR			( NUM_BINS   * sizeof(char)	 )			// Amount of memory required for a character array used for binning
#define SIZE_BINS_BOOL			( NUM_BINS   * sizeof(bool)	 )			// Amount of memory required for a boolean array used for binning
#define SIZE_BINS_INT			( NUM_BINS   * sizeof(int)	 )			// Amount of memory required for a integer array used for binning
#define SIZE_BINS_FLOAT			( NUM_BINS	 * sizeof(float) )			// Amount of memory required for a floating point array used for binning
#define SIZE_IMAGE_CHAR			( NUM_VOXELS * sizeof(char)	 )			// Amount of memory required for a character array used for binning
#define SIZE_IMAGE_BOOL			( NUM_VOXELS * sizeof(bool)	 )			// Amount of memory required for a boolean array used for binning
#define SIZE_IMAGE_INT			( NUM_VOXELS * sizeof(int)	 )			// Amount of memory required for a integer array used for binning
#define SIZE_IMAGE_FLOAT		( NUM_VOXELS * sizeof(float) )			// Amount of memory required for a floating point array used for binning
/***************************************************************************************************************************************************************************/
/************************************************************************* Precalculated Constants *************************************************************************/
/***************************************************************************************************************************************************************************/
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
#define BOOL_FORMAT				"%d";									// Specifies format to use for writing/printing boolean data using {s/sn/f/v/vn}printf statements
#define CHAR_FORMAT				"%c";									// Specifies format to use for writing/printing character data using {s/sn/f/v/vn}printf statements
#define INT_FORMAT				"%d";									// Specifies format to use for writing/printing integer data using {s/sn/f/v/vn}printf statements
#define FLOAT_FORMAT			"%f";									// Specifies format to use for writing/printing floating point data using {s/sn/f/v/vn}printf statements
#define STRING_FORMAT			"%s";									// Specifies format to use for writing/printing strings data using {s/sn/f/v/vn}printf statements
/***************************************************************************************************************************************************************************/
/***************************************************************************************************************************************************************************/
/********************************************************************* Preprocessing Array Declerations ********************************************************************/
/***************************************************************************************************************************************************************************/
/***************************************************************************************************************************************************************************/

/***************************************************************************************************************************************************************************/
/************************************** Declaration of arrays number of histories per file, projection, angle, total, and translation **************************************/
/***************************************************************************************************************************************************************************/
int total_histories = 0, recon_vol_histories = 0, maximum_histories_per_file = 0;
int* histories_per_projection, * histories_per_gantry_angle, * histories_per_file;
int* recon_vol_histories_per_projection;
int histories_per_scan[NUM_SCANS];
int post_cut_histories = 0;
/***************************************************************************************************************************************************************************/
/********************************************** Declaration of array used to store tracking plane distances from rotation axis *********************************************/
/***************************************************************************************************************************************************************************/
float SSD_u_Positions[8];
float* ut_entry_angle, * uv_entry_angle, * ut_exit_angle, * uv_exit_angle; 
int zero_WEPL = 0;
int zero_WEPL_files = 0;
/***************************************************************************************************************************************************************************/
/************************************************* Declaration of arrays for storage of input data for use on the host (_h) ************************************************/
/***************************************************************************************************************************************************************************/
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
/***************************************************************************************************************************************************************************/
/*********************************************** Declaration of arrays for storage of input data for use on the device (_d) ************************************************/
/***************************************************************************************************************************************************************************/
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
/***************************************************************************************************************************************************************************/
/********************************************** Declaration of statistical analysis arrays for use on host(_h) or device (_d) **********************************************/
/***************************************************************************************************************************************************************************/
float* mean_WEPL_h, * mean_WEPL_d;
float* mean_energy_h, * mean_energy_d;
float* mean_rel_ut_angle_h, * mean_rel_ut_angle_d;
float* mean_rel_uv_angle_h, * mean_rel_uv_angle_d;
float* mean_total_rel_angle_h, * mean_total_rel_angle_d;
float* stddev_rel_ut_angle_h, * stddev_rel_ut_angle_d;
float* stddev_rel_uv_angle_h, * stddev_rel_uv_angle_d;
float* stddev_WEPL_h, * stddev_WEPL_d;
/***************************************************************************************************************************************************************************/
/******************************************** Declaration of pre/post filter sinogram for FBP for use on host(_h) or device (_d) *******************************************/
/***************************************************************************************************************************************************************************/
float* sinogram_h, * sinogram_d;
float* sinogram_filtered_h, * sinogram_filtered_d;
/***************************************************************************************************************************************************************************/
/***************************************************** Declaration of image arrays for use on host(_h) or device (_d) ******************************************************/
/***************************************************************************************************************************************************************************/
bool* SC_hull_h, * SC_hull_d;
bool* MSC_hull_h, * MSC_hull_d;
bool* SM_hull_h, * SM_hull_d;
bool* FBP_hull_h, * FBP_hull_d;
bool* x_hull_h, * x_hull_d;
int* MSC_counts_h, * MSC_counts_d;
int* SM_counts_h, * SM_counts_d;
int* MLP_test_image_h, * MLP_test_image_d;
float* FBP_image_h, * FBP_image_d;
double* x_update_h, *x_update_d;
float* x_h, * x_d;
/***************************************************************************************************************************************************************************/
/********************************** Declaration of vectors used to accumulate data from histories that have passed currently applied cuts **********************************/
/***************************************************************************************************************************************************************************/
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
/***************************************************************************************************************************************************************************/
/********************************** Declaration of arrays used to accumulate data from histories that have passed currently applied cuts ***********************************/
/***************************************************************************************************************************************************************************/
int*	bin_index;
float*	bin_WEPL;
int*	bin_num;			
int*	gantry_angle;	
float*	WEPL;		
float*	x_entry;		
float*	y_entry;		
float*	z_entry;		
float*	x_exit;			
float*	y_exit;			
float*	z_exit;			
float*	xy_entry_angle;	
float*	xz_entry_angle;	
float*	xy_exit_angle;	
float*	xz_exit_angle;	
float*	relative_ut_angle;	
float*	relative_uv_angle;
/***************************************************************************************************************************************************************************/
/*********************************************************************** Execution timer variables *************************************************************************/
/***************************************************************************************************************************************************************************/
clock_t start_time, end_time, execution_clock_cycles;
double execution_time;
/***************************************************************************************************************************************************************************/
/***************************************************************************************************************************************************************************/
/************************************************************************* For Use In Development **************************************************************************/
/***************************************************************************************************************************************************************************/
/***************************************************************************************************************************************************************************/

/***************************************************************************************************************************************************************************/
/************************************************************************ MLP Test Image Parameters ************************************************************************/
/***************************************************************************************************************************************************************************/
int* test_image_h, test_image_d;


int MLP_IMAGE_COLUMNS = 100, MLP_IMAGE_ROWS = 100, MLP_IMAGE_SLICES = 5;
int MLP_IMAGE_VOXELS = MLP_IMAGE_COLUMNS * MLP_IMAGE_ROWS * MLP_IMAGE_SLICES;
int MLP_IMAGE_SIZE = MLP_IMAGE_VOXELS * sizeof(int);

int MLP_IMAGE_RECON_CYL_RADIUS_VOXELS = 50;
int MLP_IMAGE_RECON_CYL_HEIGHT_VOXELS = 5;
int MLP_PHANTOM_A_VOXELS = 15, MLP_PHANTOM_B_VOXELS = 25;

double MLP_IMAGE_VOXEL_WIDTH = 0.1;
double MLP_IMAGE_VOXEL_HEIGHT = 0.1;
double MLP_IMAGE_VOXEL_THICKNESS = 1.0; 

double MLP_IMAGE_RECON_CYL_RADIUS = MLP_IMAGE_RECON_CYL_RADIUS_VOXELS * MLP_IMAGE_VOXEL_WIDTH;
double MLP_IMAGE_RECON_CYL_HEIGHT = MLP_IMAGE_RECON_CYL_HEIGHT_VOXELS * MLP_IMAGE_VOXEL_THICKNESS;
double MLP_PHANTOM_A = MLP_PHANTOM_A_VOXELS * MLP_IMAGE_VOXEL_WIDTH;
double MLP_PHANTOM_B = MLP_PHANTOM_B_VOXELS * MLP_IMAGE_VOXEL_HEIGHT;

double MLP_IMAGE_WIDTH = MLP_IMAGE_COLUMNS * MLP_IMAGE_VOXEL_WIDTH;
double MLP_IMAGE_HEIGHT = MLP_IMAGE_ROWS * MLP_IMAGE_VOXEL_HEIGHT;
double MLP_IMAGE_THICKNESS = MLP_IMAGE_SLICES * MLP_IMAGE_VOXEL_THICKNESS;
/***************************************************************************************************************************************************************************/
/***************************************************************************************************************************************************************************/
/************************************************************************ End of Parameter Definitions *********************************************************************/
/***************************************************************************************************************************************************************************/
/***************************************************************************************************************************************************************************/
//#endif // _PCT_RECONSTRUCTION_H_
///***************************************************************************************************************************************************************************/
///********************************************************* tv conversions and energy/WEPL calibration parameters ***********************************************************/
///***************************************************************************************************************************************************************************/
//// Vladimir
//double p0, p1, p2, p3, p4;												// t/v fit parameters
//static double Estage[5] = { 25.85, 29.04, 34.43, 47.8, 52.63 };			// Calibration coefficient = avg E deposited in stage N by 200MeV p+; used in ADC->E (MeV) conversion
//// Reco.C
//double ped[5] = {9.645, -20.484, -201.987, 62.966, -7.747};				// ADC Pedestals (Celeste data) Reco.C
//double ucal = 216.9 + 40;	
//
////pctroot.cpp
// const unsigned char pattern_run_header_identifier[3] = {0xD2, 0x55, 0x4E};     // 1R U N
//
//// sensor.C/edisplay.C
//// approx position for the calorimeter entrance for runs >= 51
//Double_t ut[4] = {-211.80, -161.80, 161.80, 211.80};
//Double_t uv[4] = {-217.70, -167.6,  167.6,  217.70};
//
////sensor.C/Sensor.C
//Double_t tPin[4] = {215.24, 211.25, -211.25, -215.24};   // T coordinate of alignment pin per layer
//      //Double_t tDir[4] = {-1.0, -1.0, 1.0, 1.0};               // T direction of increasing strip number per layer
//Double_t vPin[4] = {0.055,0.055,0.055,0.055};
//      //int vBoardLayer_[4] = {6,4,2,3};  	      // fpga to V-board translation; this will change if spares are swapped
//      Double_t firstStripV[7][2] = {		// Distance from the alignment pin to the first strip
//         {-43.7193, -43.716},             // Board V0 doesn't exist
//         {-43.7193, -43.716},
//         {-43.7193, -43.716},
//         {-43.7193, -43.716},
//         {-43.7193, -43.716},	            // These numbers are actually from V4
//         {-43.7193, -43.716},
//         {-43.5855, -43.5865}
//	  };
//// edisplay.C
//struct Geo {
//      Double_t halfsize;
//      Double_t tpin[2];
//      Double_t firstStrip[4];    // position of the first strip of the t-sensors relative the pin
//      Geo() {
//         halfsize = 43.662;
//         tpin[0] = 215.24;
//         tpin[1] = 211.25;
//         firstStrip[0] = 38.58;
//         firstStrip[1] = 126.85;
//         firstStrip[2] = 215.11;
//         firstStrip[3] = 303.37;
//      }
//   } geo;
//
//// StripHit.h
//Int_t vstrip[4][2*384];    // particle eye view: layer 0: left part 384..768, right part: 0..383
//   Int_t tstrip[4][4*384];
//   Int_t* stripFPGA[12];