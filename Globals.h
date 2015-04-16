#pragma once

//#include <C:\Users\Blake\Documents\GitHub\pCT_Reconstruction\pCT_Reconstruction.h>

typedef unsigned long long ULL;
typedef unsigned int uint;

/***************************************************************************************************************************************************************************/
/****************************************************************** Global Variable and Array Declerations *****************************************************************/
/***************************************************************************************************************************************************************************/
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*--------------------------------------------- Preprocessing and reconstruction configuration/parameter container definitions --------------------------------------------*/
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
char EXECUTION_DATE[9];
bool CONFIG_PATH_PASSED = false;			// [T/F] Path to "settings.cfg" passed as command line argument [T] or inferred from current directory [F]
unsigned int num_run_arguments;
int num_parameters_2_change; 
char** run_arguments;
std::stringstream buffer;

int GENERATION_DATE, CALIBRATION_DATE;
uint PHANTOM_NAME_SIZE, DATA_SOURCE_SIZE, ACQUIRED_BY_SIZE, CALIBRATED_BY_SIZE, SKIP_2_DATA_SIZE, VERSION_ID, PROJECTION_INTERVAL;
float PROJECTION_ANGLE, BEAM_ENERGY_IN;
char* PHANTOM_NAME, * DATA_SOURCE, * ACQUIRED_BY, * CALIBRATED_BY, * PREPROCESSED_BY, * RECONSTRUCTED_BY, * CONFIG_LINK, * COMMENTS;
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*------------------------------------------------------ Reconstruction history ordering and iterate update parameters ----------------------------------------------------*/
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
DATA_FORMATS DATA_FORMAT;												// Specify which data format to use for this run
ULL NUM_RECON_HISTORIES			= 20153778;
ULL PRIME_OFFSET				= 5038457;
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*------------------------------------- Declaration of arrays number of histories per file, projection, angle, total, and translation -------------------------------------*/
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
int total_histories = 0, recon_vol_histories = 0, maximum_histories_per_file = 0;
int* histories_per_projection, * histories_per_gantry_angle, * histories_per_file;
int* recon_vol_histories_per_projection;
int* histories_per_scan;
uint post_cut_histories = 0;
uint reconstruction_histories	= 0;
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
int* gantry_angle_h, * bin_number_h, * bin_counts_h;
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
int* gantry_angle_d, * bin_number_d, * bin_counts_d;
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
bool* hull_h, * hull_d;
int* MSC_counts_h, * MSC_counts_d;
int* SM_counts_h, * SM_counts_d;
int* MLP_test_image_h, * MLP_test_image_d;
float* x_FBP_h, * x_FBP_d;
float* x_FBP_filtered_h, * x_FBP_filtered_d;
float* FBP_median_filtered_2D_h, * FBP_median_filtered_2D_d;
float* FBP_median_filtered_3D_h, * FBP_median_filtered_3D_d;

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
std::vector<int>	bin_number_vector;			
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