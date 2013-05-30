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
////rat2		C:\Users\Blake\Documents\Visual Studio 2010\Projects\pCT_Reconstruction\Input\Rat_Scan2
//const char input_dir[] = "C:/Users/Blake/Documents/Visual Studio 2010/Projects/pCT_Reconstruction/Input/Rat_Scan2/";
//const char output_dir[] = "C:/Users/Blake/Documents/Visual Studio 2010/Projects/pCT_Reconstruction/Output/Rat_Scan2/";
//const char input_base_name[] = "rat_scan2_shift"; // _trans%d_%03d.txt (or .dat) will be added to this
////Simulated Data		C:\Users\Blake\Documents\Visual Studio 2010\Projects\pCT_Reconstruction\Input\Simulated_Data\9-21
//const char input_dir[] = "C:/Users/Blake/Documents/Visual Studio 2010/Projects/pCT_Reconstruction/Input/Simulated_Data/9-21";
//const char output_dir[] = "C:/Users/Blake/Documents/Visual Studio 2010/Projects/pCT_Reconstruction/Output/Simulated_Data/9-21";
//const char input_dir[] = "C:/Users/Blake/Documents/Visual Studio 2010/Projects/pCT_Reconstruction/Input/sim_noerror";
//const char output_dir[] = "C:/Users/Blake/Documents/Visual Studio 2010/Projects/pCT_Reconstruction/Output/sim_noerror";
//const char input_dir[] = "C:/Users/Blake/Documents/Visual Studio 2010/Projects/pCT_Reconstruction/Input/sim_error1";
//const char output_dir[] = "C:/Users/Blake/Documents/Visual Studio 2010/Projects/pCT_Reconstruction/Output/sim_error1";
//C:\Users\Blake\Documents\Visual Studio 2010\Projects\pCT_Reconstruction\Data\DetectData
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

//const char input_base_name[] = "ped_scan1"; // _trans%d_%03d.txt (or .dat) will be added to this
char input_base_name[] = "simdata"; // _trans%d_%03d.txt (or .dat) will be added to this 

// Preprocessing Option Parameters
const bool binary_data_files = true; // use binary encoded files instead of text files for raw data input
const bool single_data_file = false;
const bool debug_text_on = true;
#define DEBUG_TEXT_ON 1
int something;
#define something_else something

// Constants
#define PI ( 4 * atanf( 1.0 ) )
#define ROOT_TWO sqrtf(2.0)

// Scan Parameters
#define GANTRY_ANGLE_INTERVAL 4.0 //degrees
#define GANTRY_ANGLES int( 360.0 / GANTRY_ANGLE_INTERVAL )
#define TRANSLATIONS 1
#define NUM_FILES = (TRANSLATIONS * GANTRY_ANGLES)  //1 file per gantry angle per translation

// Detector Parameters
#define SOURCE_RADIUS 265.7 // cm
#define SSD_T_SIZE 20 // length of SSD cm
#define SSD_V_SIZE 10.6 // length of SSD cm

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

// Image Parameters
#define IMAGE_WIDTH (RECON_CYL_DIAMETER + 1.0)
#define IMAGE_HEIGHT (RECON_CYL_DIAMETER + 1.0)
#define IMAGE_THICKNESS (RECON_CYL_HEIGHT + 0.6)
#define VOXELS_X 200
#define VOXELS_Y 200
#define VOXELS_Z 32
#define VOXELS (VOXELS_X * VOXELS_Y * VOXELS_Z)
#define VOXEL_WIDTH (IMAGE_WIDTH / VOXELS_X) // cm
#define VOXEL_HEIGHT (IMAGE_HEIGHT / VOXELS_Y) // cm
#define VOXEL_THICKNESS (IMAGE_THICKNESS / VOXELS_Z) // cm
#define SLICE_THICKNESS 0.3// cm
#define VOXEL_STEP_SIZE ( VOXEL_WIDTH / 2 ) // cm

// Memory Allocation Sizes for Image Types
#define IMAGE_INT_MEM_SIZE (VOXELS * sizeof(int))
#define IMAGE_FLOAT_MEM_SIZE (VOXELS * sizeof(float))

// Computational Parameters
#define BYTES_PER_HISTORY 48
#define MAX_GPU_HISTORIES 200000
#define THREADS_PER_BLOCK 256
//#define SPACE_CARVE_DIFF_THRESH 1000 //RatScan2
//#define SPACE_CARVE_DIFF_THRESH 50
//#define SPACE_CARVE_DIFF_THRESH 50 //Uniform2
//#define SPACE_CARVE_DIFF_THRESH 50 //Uniform1
#define SPACE_CARVE_DIFF_THRESH 50 //DetectDataNoisy
//#define SPACE_CARVE_DIFF_THRESH 50 //DetectData
#define PURE_SPACE_CARVE_THRESHOLD 0.0
#define SPACE_CARVE_THRESHOLD 0.0
#define SPACE_CARVE_INTERSECTIONS_THRESHOLD 1300
#define SPACE_MODEL_LOWER_THRESHOLD 4
#define SPACE_MODEL_UPPER_THRESHOLD 100
#define SPACE_MODEL_INTERSECTIONS_THRESHOLD 75000
#define FBP_THRESHOLD 0.6
#define SIGMAS_TO_KEEP 3

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

// Coefficients of 5th Degree Polynomial Fit to 1/[Beta(u)*p(u)] for 200 MeV Proton Beam
double a_0 = 7.457 * pow( 10.0, -6.0 );
double a_1 = 4.548 * pow( 10.0, -7.0 );
double a_2 = -5.777 * pow( 10.0, -8.0 );
double a_3 = 1.301 * pow( 10.0, -8.0 );
double a_4 = -9.228 * pow( 10.0, -10.0 );
double a_5 = 2.687 * pow( 10.0, -11.0 );

//#define azero (7.457*powf(10,-6))
//#define aone (4.548*powf(10,-7))
//#define atwo (-5.777*powf(10,-8))
//#define athree (1.301*powf(10,-8))
//#define afour (-9.228*powf(10,-10))
//#define afive (2.687*powf(10,-11))

double azero = 7.457 * pow( 10.0, -6.0 );
double aone  = 4.548 * pow( 10.0, -7.0 );
double atwo = -5.777 * pow( 10.0, -8.0 );
double athree = 1.301 * pow( 10.0, -8.0 );
double afour  = -9.228 * pow( 10.0, -10.0 );
double afive  = 2.687 * pow( 10.0, -11.0 );

//#define a_0 1
//#define a_1 1
//#define a_2 1
//#define a_3 1
//#define a_4 1
//#define a_5 1

// Number of Histories Per File, Projection, Angle, Total, and Translation
int total_histories = 0, recon_vol_histories = 0, maximum_histories_per_file = 0;
int* histories_per_projection, * histories_per_gantry_angle, * histories_per_file;
int* recon_vol_histories_per_projection;
int histories_per_translation[TRANSLATIONS];

// Binning Array Sizes by Type
unsigned int mem_size_bins_floats = NUM_BINS * sizeof(float);
unsigned int mem_size_bins_ints = NUM_BINS * sizeof(int);
unsigned int mem_size_bins_bool = NUM_BINS * sizeof(bool);
unsigned int mem_size_bins_char = NUM_BINS * sizeof(char);

int valid_array_position;
int SSD_INTERSECTION_NUMS;
float SSD_u_Positions[8];

// Host Arrays for Data Storage During Iterative Data Reading 
int* gantry_angle_h, * bin_num_h, * bin_counts_h;
bool* traversed_recon_volume_h;
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

// GPU Arrays for Data Storage During Iterative Data Reading 
int* gantry_angle_d, * bin_num_d, * bin_counts_d;
bool* traversed_recon_volume_d;
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
int* pure_space_carve_object_h, * pure_space_carve_object_d;
int* space_carve_object_h, * space_carve_object_d;
int* space_model_object_h, * space_model_object_d;
int* FBP_object_h, * FBP_object_d;
int* MLP_test_image_h, * MLP_test_image_d;

// Temporary Storage Vectors			
vector<int>	bin_num_vector;			
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

//#define IMAGE_WIDTH 200
//#define IMAGE_HEIGHT 200
//#define IMAGE_THICKNESS RECON_CYL_HEIGHT
//#define IMAGE_SLICES int( RECON_CYL_HEIGHT / SLICE_THICKNESS )
//#define IMAGE_VOXELS IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_SLICES
//#define VOXEL_WIDTH ( RECON_CYL_DIAMETER / IMAGE_WIDTH ) // cm
//#define VOXEL_HEIGHT ( RECON_CYL_DIAMETER / IMAGE_HEIGHT ) // cm
//#define VOXEL_THICKNESS SLICE_THICKNESS
//#define VOXEL_STEP_SIZE ( VOXEL_WIDTH / 2 ) // cm
//#define IMAGE_INT_MEM_SIZE IMAGE_VOXELS * sizeof(int)
//#define IMAGE_FLOAT_MEM_SIZE IMAGE_VOXELS * sizeof(float)

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
