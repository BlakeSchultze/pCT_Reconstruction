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

// Data Input/Output Directory Paths and File Naming Prefix
//const char input_dir[] = "C:\\Users\\Blake\\Documents\\Visual Studio 2010\\Projects\\pCT_Reconstruction\\Input\\Rat_Scan2";
//const char output_dir[] = "C:\\Users\\Blake\\Documents\\Visual Studio 2010\\Projects\\pCT_Reconstruction\\Output\\Rat_Scan2";
//const char input_dir[] = "C:/Users/Blake/Documents/Visual Studio 2010/Projects/pCT_Reconstruction/Input/Simulated_Data/9-21";
//const char output_dir[] = "C:/Users/Blake/Documents/Visual Studio 2010/Projects/pCT_Reconstruction/Output/Simulated_Data/9-21";
//const char input_dir[] = "C:/Users/Blake/Documents/Visual Studio 2010/Projects/pCT_Reconstruction/Input/sim_noerror";
//const char output_dir[] = "C:/Users/Blake/Documents/Visual Studio 2010/Projects/pCT_Reconstruction/Output/sim_noerror";
//const char input_dir[] = "C:/Users/Blake/Documents/Visual Studio 2010/Projects/pCT_Reconstruction/Input/sim_error1";
//const char output_dir[] = "C:/Users/Blake/Documents/Visual Studio 2010/Projects/pCT_Reconstruction/Output/sim_error1";
const char input_dir[] = "C:\\Users\\Blake\\Documents\\Visual Studio 2010\\Projects\\pCT_Reconstruction\\Input\\DetectData";
const char output_dir[] = "C:\\Users\\Blake\\Documents\\Visual Studio 2010\\Projects\\pCT_Reconstruction\\Output\\DetectData";
//const char input_dir[] = "C:/Users/Blake/Documents/Visual Studio 2010/Projects/pCT_Reconstruction/Input/DetectDataWeplNoisy1";
//const char output_dir[] = "C:/Users/Blake/Documents/Visual Studio 2010/Projects/pCT_Reconstruction/Output/DetectDataWeplNoisy1";
//const char input_dir[] = "C:/Users/Blake/Documents/Visual Studio 2010/Projects/pCT_Reconstruction/Input/NoisyUniform1";
//const char output_dir[] = "C:/Users/Blake/Documents/Visual Studio 2010/Projects/pCT_Reconstruction/Output/NoisyUniform1";
//const char input_dir[] = "C:/Users/Blake/Documents/Visual Studio 2010/Projects/pCT_Reconstruction/Input/NoisyUniform2";
//const char output_dir[] = "C:/Users/Blake/Documents/Visual Studio 2010/Projects/pCT_Reconstruction/Output/NoisyUniform2";
//const char input_dir[] = "C:/Users/Blake/Documents/Visual Studio 2010/Projects/pCT_Reconstruction/Input/NoisyUniform3";
//const char output_dir[] = "C:/Users/Blake/Documents/Visual Studio 2010/Projects/pCT_Reconstruction/Output/NoisyUniform3";
//const char input_dir[] = "C:/Users/Blake/Documents/Visual Studio 2010/Projects/pCT_Reconstruction/Input/input_noisefloor40";
//const char output_dir[] = "C:/Users/Blake/Documents/Visual Studio 2010/Projects/pCT_Reconstruction/Output/input_noisefloor40";

//const char input_base_name[] = "rat_scan2_shift"; // _trans%d_%03d.txt (or .dat) will be added to this
//const char input_base_name[] = "ped_scan1"; // _trans%d_%03d.txt (or .dat) will be added to this
char input_base_name[] = "simdata"; // _trans%d_%03d.txt (or .dat) will be added to this 

// Preprocessing Option Parameters
const bool BINARY_DATA_FILES = true; // use binary encoded files instead of text files for raw data input
const bool SINGLE_DATA_FILE = false; // true = 1 file for all gantry angles; false = 1 file per gantry angle
const bool DEBUG_TEXT_ON = true; // include printf() statements in execution
const bool CONFIG_FILE = false; // 
const bool SAMPLE_STD_DEV = true;
const bool SSD_IN_MM = true;

const bool SIM = true;
const bool NOISY = false;
const bool RAT = false;
const bool PED = false;

// Precalculated Constants
#define PI_OVER_4			( atanf( 1.0 ) )
#define PI_OVER_2			( 2 * atanf( 1.0 ) )
#define THREE_PI_OVER_4		( 3 * atanf( 1.0 ) )
#define PI					( 4 * atanf( 1.0 ) )
#define FIVE_PI_OVER_4		( 5 * atanf( 1.0 ) )
#define SEVEN_PI_OVER_4		( 7 * atanf( 1.0 ) )
#define TWO_PI				( 8 * atanf( 1.0 ) )
#define ANGLE_TO_RADIANS	( PI/180.0 )
#define RADIANS_TO_ANGLE	( 180.0/PI )
#define ROOT_TWO			sqrtf(2.0)

// Scan Parameters
#define GANTRY_ANGLE_INTERVAL 4.0 //degrees
#define GANTRY_ANGLES int( 360.0 / GANTRY_ANGLE_INTERVAL )
#define NUM_FILES ( NUM_SCANS * GANTRY_ANGLES ) //1 file per gantry angle per translation

// Detector Parameters
#define SOURCE_RADIUS 265.7 // cm
// Simulated Data
#define SSD_T_SIZE 20.0 // length of SSD cm
#define SSD_V_SIZE 10.6 // length of SSD cm
// Pediatric Head Phantom
//#define SSD_T_SIZE 18.0 // length of SSD cm
//#define SSD_V_SIZE 9.0 // length of SSD cm

// FBP and Statistical Binning Parameters
#define T_BIN_SIZE 0.1 // cm
#define T_BINS int( SSD_T_SIZE / T_BIN_SIZE + 0.5 )
#define V_BIN_SIZE 0.5 // cm
#define V_BINS int( SSD_V_SIZE / V_BIN_SIZE + 0.5 )
#define ANGULAR_BIN_SIZE 4.0 //degrees
#define ANGULAR_BINS int( 360.0 / ANGULAR_BIN_SIZE + 0.5 )
#define NUM_BINS ( ANGULAR_BINS * T_BINS * V_BINS )

// Reconstruction Cylinder Parameters
#define RECON_CYL_RADIUS 10.0 // cm
#define RECON_CYL_DIAMETER ( 2 * RECON_CYL_RADIUS ) // cm
#define RECON_CYL_HEIGHT (SSD_V_SIZE - 1.0) // cm

//// Image Parameters
//#define IMAGE_WIDTH (RECON_CYL_DIAMETER + 1.0)
//#define IMAGE_HEIGHT (RECON_CYL_DIAMETER + 1.0)
//#define IMAGE_THICKNESS (RECON_CYL_HEIGHT + 0.6)
//#define VOXELS_X 200
//#define VOXELS_Y 200
//#define VOXELS_Z 32
//#define VOXELS (VOXELS_X * VOXELS_Y * VOXELS_Z)
//#define VOXEL_WIDTH (IMAGE_WIDTH / VOXELS_X) // cm
//#define VOXEL_HEIGHT (IMAGE_HEIGHT / VOXELS_Y) // cm
//#define VOXEL_THICKNESS (IMAGE_THICKNESS / VOXELS_Z) // cm
//#define SLICE_THICKNESS 0.3// cm
//#define VOXEL_STEP_SIZE ( VOXEL_WIDTH / 2 ) // cm

// Image Parameters
#define IMAGE_WIDTH ( VOXELS_X * VOXEL_WIDTH )
#define IMAGE_HEIGHT ( VOXELS_Y * VOXEL_HEIGHT )
#define IMAGE_THICKNESS ( VOXELS_Z * SLICE_THICKNESS )
#define VOXELS_X 200
#define VOXELS_Y 200
#define VOXELS_Z int( RECON_CYL_HEIGHT / SLICE_THICKNESS )
#define VOXELS ( VOXELS_X * VOXELS_Y * VOXELS_Z )
#define VOXEL_WIDTH ( RECON_CYL_DIAMETER / VOXELS_X ) // cm
#define VOXEL_HEIGHT ( RECON_CYL_DIAMETER / VOXELS_Y ) // cm
#define VOXEL_THICKNESS (IMAGE_THICKNESS / VOXELS_Z) // cm
#define SLICE_THICKNESS 0.3// cm
#define VOXEL_STEP_SIZE ( VOXEL_WIDTH / 2 ) // cm

#define IMAGE_INT_MEM_SIZE ( VOXELS * sizeof(int) )
#define IMAGE_FLOAT_MEM_SIZE ( VOXELS * sizeof(float) )
#define IMAGE_BOOL_MEM_SIZE ( VOXELS * sizeof(bool) )

#define BYTES_PER_HISTORY 48
#define NUM_SCANS 1
#define MAX_GPU_HISTORIES 300000
#define THREADS_PER_BLOCK 512
//#define WEPL_CUT_ALLOWANCE 0.1
#define MSC_DIFF_THRESH 50 //DetectData,DetectDataNoisy
#define POST_CUT_MSC_DIFF_THRESH 50 //DetectData,DetectDataNoisy
//DetectData
#define SC_THRESHOLD 1.0 
#define MSC_THRESHOLD 1.0 
#define POST_CUT_MSC_THRESHOLD 1.0 
#define BIN_CARVE_THRESHOLD 1.0 
////DetectDataNoisy
//#define SC_THRESHOLD 0.0 
//#define MSC_THRESHOLD 0.0 
//#define POST_CUT_MSC_THRESHOLD 0.0
//#define BIN_CARVE_THRESHOLD 0.0
//Ped
//#define SC_THRESHOLD 2.0 
//#define MSC_THRESHOLD 1.0 
//#define POST_CUT_MSC_THRESHOLD 2.0
//#define BIN_CARVE_THRESHOLD 1.0
#define SM_LOWER_THRESHOLD 6.0 //DetectData,DetectDataNoisy
#define SM_UPPER_THRESHOLD 21.0 //DetectData,DetectDataNoisy
#define BIN_MODEL_THRESHOLD 6.0
#define FBP_THRESHOLD 0.6
#define SIGMAS_TO_KEEP 3
// Ram-Lak = 0, Shepp-Logan = 1
#define FILTER_NUM 1
// MLP Parameters
#define MLP_u_step (min(VOXEL_WIDTH, VOXEL_HEIGHT) / 2) 
#define E_0 13.6
#define X_0 36.1
// 200 MeV Coefficients
//#define a_0 (7.457*pow(10.0,-6.0))
//#define a_1 (4.548*pow(10.0,-7.0))
//#define a_2 (-5.777*pow(10.0,-8.0))
//#define a_3 (1.301*pow(10.0,-8.0))
//#define a_4 (-9.228*pow(10.0,-10.0))
//#define a_5 (2.687*pow(10.0,-11.0))

double a_0 = (7.457*pow(10,-6.0));
double a_1 = (4.548*pow(10,-7.0));
double a_2 = (-5.777*pow(10,-8.0));
double a_3 = (1.301*pow(10,-8.0));
double a_4 = (-9.228*pow(10,-10.0));
double a_5 = (2.687*pow(10,-11.0));

// Number of Histories Per File, Projection, Angle, Total, and Translation
int total_histories = 0, recon_vol_histories = 0, maximum_histories_per_file = 0;
int* histories_per_projection, * histories_per_gantry_angle, * histories_per_file;
int* recon_vol_histories_per_projection;
int histories_per_scan[NUM_SCANS];
int post_cut_histories = 0;

// Binning Array Sizes by Type
unsigned int MEM_SIZE_BINS_FLOATS = NUM_BINS * sizeof(float);
unsigned int MEM_SIZE_BINS_INTS = NUM_BINS * sizeof(int);
unsigned int MEM_SIZE_BINS_BOOL = NUM_BINS * sizeof(bool);
unsigned int MEM_SIZE_BINS_CHAR = NUM_BINS * sizeof(char);

int valid_array_position;
int SSD_INTERSECTION_NUMS;
float SSD_u_Positions[8];

// Host Arrays for Data Storage During Iterative Data Reading 
int* gantry_angle_h, * bin_num_h, * bin_counts_h, * new_value_h;
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

// GPU Arrays for Data Storage During Iterative Data Reading 
int* gantry_angle_d, * bin_num_d, * bin_counts_d, * new_value_d;
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

// Valid Data Storage Arrays
int* valid_bin_num;
float* valid_WEPL;
float* valid_x_entry, * valid_y_entry, * valid_z_entry;
float* valid_x_exit, * valid_y_exit, * valid_z_exit;
float* valid_xy_entry_angle, * valid_xz_entry_angle;
float* valid_xy_exit_angle, * valid_xz_exit_angle;

// Statistical Data Arrays
float* mean_WEPL_h, * mean_WEPL_d;
float* mean_rel_ut_angle_h, * mean_rel_ut_angle_d;
float* mean_rel_uv_angle_h, * mean_rel_uv_angle_d;
float* stddev_rel_ut_angle_h, * stddev_rel_ut_angle_d;
float* stddev_rel_uv_angle_h, * stddev_rel_uv_angle_d;
float* stddev_WEPL_h, * stddev_WEPL_d;

// Pre and Post Filter Sinograms for FBP
float* sinogram_h, * sinogram_d;
float* sinogram_filtered_h, * sinogram_filtered_d;

// Image Arrays
float* X_h, * X_d;
int* SC_image_h, * SC_image_d;
int* MSC_image_h, * MSC_image_d;
int* SM_image_h, * SM_image_d;
int* bin_carve_image_h, * bin_carve_image_d;
int* bin_model_image_h, * bin_model_image_d;
int* bin_MSC_image_h, * bin_MSC_image_d;
int* post_cut_SC_image_h, * post_cut_SC_image_d;
int* post_cut_MSC_image_h, * post_cut_MSC_image_d;
int* post_cut_SM_image_h, * post_cut_SM_image_d;
int* PCSC_image_h, * PCSC_image_d;
int* FBP_object_h, * FBP_object_d;
int* MLP_test_image_h, * MLP_test_image_d;

// Temporary Storage Vectors			
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

// MLP Test Image Parameters
int MLP_IMAGE_VOXELS_X = 100, MLP_IMAGE_VOXELS_Y = 100, MLP_IMAGE_VOXELS_Z = 5;
int MLP_IMAGE_VOXELS = MLP_IMAGE_VOXELS_X * MLP_IMAGE_VOXELS_Y * MLP_IMAGE_VOXELS_Z;
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

float MLP_IMAGE_WIDTH = MLP_IMAGE_VOXELS_X * MLP_IMAGE_VOXEL_WIDTH;
float MLP_IMAGE_HEIGHT = MLP_IMAGE_VOXELS_Y * MLP_IMAGE_VOXEL_HEIGHT;
float MLP_IMAGE_THICKNESS = MLP_IMAGE_VOXELS_Z * MLP_IMAGE_VOXEL_THICKNESS;

#endif // _PCT_RECONSTRUCTION_H_
