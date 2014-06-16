//THIS FILE CONTAINS DEFINITION OF VARIABLES (RWS)

#ifndef _PCT_RECONSTRUCTION_H_
#define _PCT_RECONSTRUCTION_H_

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <omp.h>
#include <time.h>
#include <vector>
#include <math.h>
#include <new>

using namespace std;

/*********************************************************************************************************************************************************/
/*************************************************************** Preprocessing Usage Options *************************************************************/
/*********************************************************************************************************************************************************/

/************************************************************* Preprocessing Option Parameters ***********************************************************/
const bool BINARY_ENCODING = true;		// Input data provided in binary (T) encoded files or ASCI text files (F)
const bool SINGLE_DATA_FILE = false;	// Individual file for each gantry angle (T) or single data file for all data (F)
const bool DEBUG_TEXT_ON = true;		// Provide (T) or suppress (F) print statements to console during execution
const bool CONFIG_FILE = false;			// Tracking plane distances to rotation axis read from config file (T) or defined manually (F)
const bool SAMPLE_STD_DEV = true;		// Use sample (T) or population (F) standard deviation in statistical analysis (i.e. divide cumulative error by N/N-1)
const bool SSD_IN_MM = true;			// SSD distances from rotation axis given in mm (T) or cm (F)
const bool DATA_IN_MM = true;			// Input data given in mm (T) or cm (F)
const bool FBP_ON = true;				// Turn FBP on (T) or off (F)
const bool SC_ON = true;				// Turn Space Carving on (T) or off (F)
const bool MSC_ON = true;				// Turn Modified Space Carving on (T) or off (F)
const bool SM_ON = true;				// Turn Space Modeling on (T) or off (F)

/************************************************************* Input Data Format Specification ***********************************************************/
const bool DATA_FORMAT			// Input data in format used prior to specification of Version 0
const bool VERSION_0 = true;			// Input data in Version 0 format
const bool VERSION_1 = false;			// Input data in Version 1 format
/*********************************************************************************************************************************************************/
/************************************************************ Preprocessing Path Information *************************************************************/
/*********************************************************************************************************************************************************/

/********************************************************* Path to the input/output directories **********************************************************/
const char input_directory[] = "C:\\Users\\Blake\\Documents\\Visual Studio 2010\\Projects\\pCT_Reconstruction\\Input\\";
const char output_directory[] = "C:\\Users\\Blake\\Documents\\Visual Studio 2010\\Projects\\pCT_Reconstruction\\Output\\";

/*********************************** Name of the folder where the input data resides and output data is to be written ************************************/
const char input_folder[] = "waterPhantom";
const char output_folder[] = "waterPhantom";
//const char input_folder[] = "catphan";
//const char output_folder[] = "catphan";
//const char input_folder[] = "DetectData";
//const char output_folder[] = "DetectData";
//const char input_folder[] = "Rat_Scan2";
//const char output_folder[] = "Rat_Scan2";
//const char input_folder[] = "sim_noerror";
//const char output_folder[] = "sim_noerror";
//const char input_folder[] = "sim_error1";
//const char output_folder[] = "sim_error1";
//const char input_folder[] = "DetectDataWeplNoisy1";
//const char output_folder[] = "DetectDataWeplNoisy1";
//const char input_folder[] = "NoisyUniform1";
//const char output_folder[] = "NoisyUniform1";
//const char input_folder[] = "NoisyUniform2";
//const char output_folder[] = "NoisyUniform2";
//const char input_folder[] = "NoisyUniform3";
//const char output_folder[] = "NoisyUniform3";
//const char input_folder[] = "input_noisefloor40";
//const char output_folder[] = "input_noisefloor40";
//const char input_folder[] = "Simulated_Data\\9-21";
//const char output_folder[] = "Simulated_Data\\9-21";

/******************************* Prefix of the input data set filename (_trans%d_%03d.txt (or .dat) will be added to this) *******************************/
const char input_base_name[] = "projection";			//waterPhantom, catphan
//const char input_base_name[] = "simdata";				//DetectData files
//const char input_base_name[] = "rat_scan2_shift";
//const char input_base_name[] = "ped_scan1";			//  anthropomorphic pediatric head phantom (Model 715-HN, CIRS1)

const char file_extension[] = ".bin";					// Binary file extension
//const char file_extension[] = ".dat";					// Generic data file extension, independent of encoding (various encodings can be used)
/*********************************************************************************************************************************************************/
/****************************************************************** Preprocessing Constants **************************************************************/
/*********************************************************************************************************************************************************/

/*************************************************************** Precalculated Constants *****************************************************************/
#define PI_OVER_4			( atanf( 1.0 ) )
#define PI_OVER_2			( 2 * atanf( 1.0 ) )
#define THREE_PI_OVER_4		( 3 * atanf( 1.0 ) )
#define PI					( 4 * atanf( 1.0 ) )
#define FIVE_PI_OVER_4		( 5 * atanf( 1.0 ) )
#define SEVEN_PI_OVER_4		( 7 * atanf( 1.0 ) )
#define TWO_PI				( 8 * atanf( 1.0 ) )
#define ANGLE_TO_RADIANS	( PI/180.0 )					// Convertion from angle to radians
#define RADIANS_TO_ANGLE	( 180.0/PI )					// Convertion from radians to angle
#define ROOT_TWO			sqrtf(2.0)

/****************************************************** Host/GPU computation and structure information ***************************************************/
#define BYTES_PER_HISTORY 48								// [bytes] Size of data associated with each history, 44 for actual data and 4 empty bytes
#define MAX_GPU_HISTORIES 300000							// [#] Number of histories to process on the GPU at a time, based on GPU capacity
#define THREADS_PER_BLOCK 512								// [#] Number of threads assigned to each block on the GPU

/*********************************************************** Reconstruction cylinder parameters **********************************************************/
#define RECON_CYL_RADIUS 10.0								// [cm] Radius of reconstruction cylinder
#define RECON_CYL_DIAMETER ( 2 * RECON_CYL_RADIUS )			// [cm] Diameter of reconstruction cylinder
#define RECON_CYL_HEIGHT (SSD_V_SIZE - 1.0)					// [cm] Height of reconstruction cylinder

/************************************************************* Reconstruction image parameters ***********************************************************/
#define IMAGE_WIDTH ( COLUMNS * VOXEL_WIDTH )				// [cm] 
#define IMAGE_HEIGHT ( ROWS * VOXEL_HEIGHT )				// [cm]
#define IMAGE_THICKNESS ( SLICES * SLICE_THICKNESS )		// [cm]
#define COLUMNS 200											// [#]
#define ROWS 200											// [#]
#define SLICES int( RECON_CYL_HEIGHT / SLICE_THICKNESS )	// [#]
#define VOXELS ( COLUMNS * ROWS * SLICES )					// [#]
#define VOXEL_WIDTH ( RECON_CYL_DIAMETER / COLUMNS )		// [cm]
#define VOXEL_HEIGHT ( RECON_CYL_DIAMETER / ROWS )			// [cm]
#define VOXEL_THICKNESS (IMAGE_THICKNESS / SLICES)			// [cm]
#define SLICE_THICKNESS 0.25								// [cm]
#define VOXEL_STEP_SIZE ( VOXEL_WIDTH / 2 )					// [cm]

/********************************** Scanning and detector system (source distance, tracking plane dimensions) parameters *********************************/
#define SOURCE_RADIUS 265.7									// [cm] distance  to source/scatterer
#define GANTRY_ANGLE_INTERVAL 6.0							// [degrees]
#define GANTRY_ANGLES int( 360.0 / GANTRY_ANGLE_INTERVAL )	// [#] number of projection angles
#define NUM_SCANS 1											// [#]
#define NUM_FILES ( NUM_SCANS * GANTRY_ANGLES )				// [#] 1 file per gantry angle per translation
#define SSD_T_SIZE 18.0										// [cm] length of SSD 
#define SSD_V_SIZE 9.0										// [cm] length of SSD 

/******************************************* Binning (for Statistical analysis) and sinogram (for FBP) parameters ****************************************/
#define T_BIN_SIZE 0.1										// [cm]
#define T_BINS int( SSD_T_SIZE / T_BIN_SIZE + 0.5 )			// [#]
#define V_BIN_SIZE 0.25										// [cm]
#define V_BINS int( SSD_V_SIZE / V_BIN_SIZE + 0.5 )			// [#]
#define ANGULAR_BIN_SIZE 6.0								// [degrees]
#define ANGULAR_BINS int( 360.0 / ANGULAR_BIN_SIZE + 0.5 )	// [#]
#define NUM_BINS ( ANGULAR_BINS * T_BINS * V_BINS )			// [#]
#define SIGMAS_TO_KEEP 3									// [#]
#define FILTER_NUM 1 // Ram-Lak = 0, Shepp-Logan = 1
#define FBP_THRESHOLD 0.6

/************************************************************* Hull-Detection Parameters *****************************************************************/
//#define WEPL_CUT_ALLOWANCE 0.1
#define RESTRICTED_ANGLES 1
#define MSC_DIFF_THRESH 50  
#define SC_THRESHOLD 0.0 
#define MSC_THRESHOLD 1.0 
#define SM_THRESHOLD_MULTIPLIER 1.0
#define SM_LOWER_THRESHOLD 6.0 
#define SM_UPPER_THRESHOLD 21.0

/********************************************************************* MLP Parameters ********************************************************************/
#define E_0 13.6	// [MeV/c] empirical constant
#define X_0 36.1	// [cm] radiation length

// 200 MeV coefficients
double A_0 = (  7.457 * pow( 10, -6.0  ) );
double A_1 = (  4.548 * pow( 10, -7.0  ) );
double A_2 = ( -5.777 * pow( 10, -8.0  ) );
double A_3 = (  1.301 * pow( 10, -8.0  ) );
double A_4 = ( -9.228 * pow( 10, -10.0 ) );
double A_5 = (  2.687 * pow( 10, -11.0 ) );

/************************************************** Memory allocation size for arrays (binning, image) ***************************************************/
#define MEM_SIZE_BINS_FLOATS	( NUM_BINS * sizeof(float)	)
#define MEM_SIZE_BINS_INTS		( NUM_BINS * sizeof(int)	)
#define MEM_SIZE_BINS_BOOL		( NUM_BINS * sizeof(bool)	)
#define MEM_SIZE_BINS_CHAR		( NUM_BINS * sizeof(char)	)
#define MEM_SIZE_IMAGE_INT		( VOXELS * sizeof(int)		)
#define MEM_SIZE_IMAGE_FLOAT	( VOXELS * sizeof(float)	)
#define MEM_SIZE_IMAGE_BOOL		( VOXELS * sizeof(bool)		)
#define MEM_SIZE_IMAGE_CHAR		( VOXELS * sizeof(bool)		)

/*********************************************************************************************************************************************************/
/************************************************************ Preprocessing Array Declerations ***********************************************************/
/*********************************************************************************************************************************************************/

/***************************** Declaration of arrays number of histories per file, projection, angle, total, and translation *****************************/
int total_histories = 0, recon_vol_histories = 0, maximum_histories_per_file = 0;
int* histories_per_projection, * histories_per_gantry_angle, * histories_per_file;
int* recon_vol_histories_per_projection;
int histories_per_scan[NUM_SCANS];
int post_cut_histories = 0;

/************************************ Declaration of array used to store tracking plane distances from rotation axis *************************************/
float SSD_u_Positions[8];

/************************************** Declaration of arrays for storage of input data for use on the host (_h) *****************************************/
int* gantry_angle_h, * bin_num_h, * bin_counts_h;
bool* traversed_recon_volume_h, * passed_cuts_h;
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

/************************************* Declaration of arrays for storage of input data for use on the device (_d) ****************************************/
int* gantry_angle_d, * bin_num_d, * bin_counts_d;
bool* traversed_recon_volume_d, * passed_cuts_d;
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

/************************************ Declaration of statistical analysis arrays for use on host(_h) or device (_d) **************************************/
float* mean_WEPL_h, * mean_WEPL_d;
float* mean_rel_ut_angle_h, * mean_rel_ut_angle_d;
float* mean_rel_uv_angle_h, * mean_rel_uv_angle_d;
float* stddev_rel_ut_angle_h, * stddev_rel_ut_angle_d;
float* stddev_rel_uv_angle_h, * stddev_rel_uv_angle_d;
float* stddev_WEPL_h, * stddev_WEPL_d;

/********************************** Declaration of pre/post filter sinogram for FBP for use on host(_h) or device (_d) ***********************************/
float* sinogram_h, * sinogram_d;
float* sinogram_filtered_h, * sinogram_filtered_d;

/****************************************** Declaration of image arrays for use on host(_h) or device (_d) ***********************************************/
float* X_h, * X_d;
bool* SC_image_h, * SC_image_d;
bool* SC2_image_h, * SC2_image_d;
int* MSC_image_h, * MSC_image_d;
int* SM_image_h, * SM_image_d;
int* FBP_object_h, * FBP_object_d;
int* MLP_test_image_h, * MLP_test_image_d;

/************************ Declaration of vectors used to accumulate data from histories that have passed currently applied cuts **************************/		
vector<int>	bin_num_vector;			
vector<int>	gantry_angle_vector;	
vector<float> WEPL_vector;		
vector<float> x_entry_vector;		
vector<float> y_entry_vector;		
vector<float> z_entry_vector;		
vector<float> x_exit_vector;			
vector<float> y_exit_vector;			
vector<float> z_exit_vector;			
vector<float> xy_entry_angle_vector;	
vector<float> xz_entry_angle_vector;	
vector<float> xy_exit_angle_vector;	
vector<float> xz_exit_angle_vector;	
vector<float> relative_ut_angle_vector;	
vector<float> relative_uv_angle_vector;

/*********************************************************************************************************************************************************/
/************************************************************* For Use In Development  *******************************************************************/
/*********************************************************************************************************************************************************/

/************************************************************ MLP Test Image Parameters ******************************************************************/
#define MLP_u_step (min(VOXEL_WIDTH, VOXEL_HEIGHT) / 2) 
int MLP_IMAGE_COLUMNS = 100, MLP_IMAGE_ROWS = 100, MLP_IMAGE_SLICES = 5;
int MLP_IMAGE_VOXELS = MLP_IMAGE_COLUMNS * MLP_IMAGE_ROWS * MLP_IMAGE_SLICES;
int MLP_IMAGE_SIZE = MLP_IMAGE_VOXELS * sizeof(int);

int MLP_IMAGE_RECON_CYL_RADIUS_VOXELS = 40;
int MLP_IMAGE_RECON_CYL_HEIGHT_VOXELS = 5;
int MLP_PHANTOM_A_VOXELS = 15, MLP_PHANTOM_B_VOXELS = 25;

float MLP_IMAGE_VOXEL_WIDTH = 0.1;
float MLP_IMAGE_VOXEL_HEIGHT = 0.1;
float MLP_IMAGE_VOXEL_THICKNESS = 1.0;

float MLP_IMAGE_RECON_CYL_RADIUS = MLP_IMAGE_RECON_CYL_RADIUS_VOXELS * MLP_IMAGE_VOXEL_WIDTH;
float MLP_IMAGE_RECON_CYL_HEIGHT = MLP_IMAGE_RECON_CYL_HEIGHT_VOXELS * MLP_IMAGE_VOXEL_THICKNESS;
float MLP_PHANTOM_A = MLP_PHANTOM_A_VOXELS * MLP_IMAGE_VOXEL_WIDTH;
float MLP_PHANTOM_B = MLP_PHANTOM_B_VOXELS * MLP_IMAGE_VOXEL_HEIGHT;

float MLP_IMAGE_WIDTH = MLP_IMAGE_COLUMNS * MLP_IMAGE_VOXEL_WIDTH;
float MLP_IMAGE_HEIGHT = MLP_IMAGE_ROWS * MLP_IMAGE_VOXEL_HEIGHT;
float MLP_IMAGE_THICKNESS = MLP_IMAGE_SLICES * MLP_IMAGE_VOXEL_THICKNESS;

#endif // _PCT_RECONSTRUCTION_H_
