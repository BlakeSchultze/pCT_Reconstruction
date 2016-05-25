// //#ifndef PCT_RECONSTRUCTION_CU
//#define PCT_RECONSTRUCTION_CU
//#pragma once
/***********************************************************************************************************************************************************************************************************************/
/********************************************************************************** Proton CT Preprocessing and Image Reconstruction Code ******************************************************************************/
/***********************************************************************************************************************************************************************************************************************/
#include "../include/pCT_Reconstruction_Data_Segments_Blake.h"
//#include "pCT_Reconstruction_Data_Segments_Blake.h"
//#include "...\include/pCT_Reconstruction_Data_Segments_Blake.h"
//#include "C:\Users\Blake\Documents\GitHub\Baylor_ICTHUS\pCT_Reconstruction\include\pCT_Reconstruction_Data_Segments_Blake.h"
//#include "..\include\pCT_Reconstruction_Data_Segments_Blake.h"

//#define PROFILER 11
// Includes, CUDA project\
//#include <cutil_inline.h>

// Includes, kernels
//#include "pCT_Reconstruction_GPU.cu"
/***********************************************************************************************************************************************************************************************************************/
/********************************************************************************************** Host functions declarations ********************************************************************************************/
/***********************************************************************************************************************************************************************************************************************/
void write_PNG(const char*, float*);

bool file_exists (const char* );
bool file_exists2 (const char* file_location) { return static_cast<bool>( std::ifstream(file_location) ); };
bool file_exists3 (const char* file_location) { return std::ifstream(file_location).good(); };
bool directory_exists(char* );
bool mkdir(char*);
unsigned int create_unique_dir( char* );
unsigned int create_unique_dir( const char*, const char*, char* );
bool input_directory_exists();
void copy_data(const char*, const char*, const char*);
void copy_folder_contents(const char*, const char*, const char*);
void set_file_permissions(const char*, const char*);

// Execution Control Functions
void timer( bool, clock_t&, clock_t&, double&, const char*);
double timer( bool, clock_t&, const char*);
void exit_program_if( bool);
void pause_execution();
void exit_program_if( bool, const char* );

// Host helper functions		
template< typename T > std::string type_name( const T& ) { return type_name<T>() ; }
void create_random_number_generator_engines();
double randn(double, double);
int randi(int, int);
int randi(RAND_GENERATORS, int, int);
void rand_check(uint);
template< typename T, typename T2> T max_n( int, T2, ...);
template< typename T, typename T2> T min_n( int, T2, ...);
template<typename T> T* sequential_numbers( int, int );
void bin_2_indexes( int, int&, int&, int& );
inline const char * bool_2_string( bool b ){ return b ? "true" : "false"; }
std::string terminal_response(char*);
std::string terminal_response(const char*);
char((&terminal_response( char*, char(&)[256]))[256]);
char((&terminal_response( const char*, char(&)[256]))[256]);
char((&current_MMDD( char(&)[5]))[5]);
char((&current_MMDDYYYY( char(&)[9]))[9]);
char((&current_YY_MM_DD( char(&)[9]))[9]);

// Image Initialization/Construction Functions
template<typename T> void initialize_host_image( T*& );
template<typename T> void add_ellipse( T*&, int, double, double, double, double, T );
template<typename T> void add_circle( T*&, int, double, double, double, T );
template<typename O> void import_image( O*&, char* );
template<typename T, typename T2> void averaging_filter( T*&, T2*&, int, bool, double );
template<typename T> void median_filter_2D( T*&, unsigned int );
template<typename T> void median_filter_2D( T*&, T*&, unsigned int );
template< typename T, typename L, typename R> T discrete_dot_product( L*&, R*&, unsigned int*&, unsigned int );
template< typename T, typename RHS> T scalar_dot_product( double, RHS*&, unsigned int*&, unsigned int );
double scalar_dot_product2( double, float*&, unsigned int*&, unsigned int );

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

// Memory transfers and allocations/deallocations
void initial_processing_memory_clean();
void reserve_vector_capacity(); 
void resize_vectors( unsigned int );
void shrink_vectors();
void data_shift_vectors( unsigned int, unsigned int );
void post_cut_memory_clean(); 

// Program startup task Functions
void set_ssh_server_login_strings();
void set_enum_strings();
void set_procedures_on_off_strings();
void check_4_missing_input();
bool verify_input_data();
void set_compute_node();
void set_user_strings();
void set_compute_system_directories();
void set_git_branch_info();
void set_and_make_output_folder();
void set_IO_folder_names();
void set_source_code_paths();
void print_paths();
void string_assigments();
void IO_setup();
void program_startup_tasks();

// Execution Control Functions
void define_execution_log_order();
void init_execution_log_csv();
void execution_log_2_txt();
void execution_log_2_csv();
void cp_output_2_kodiak();
void scp_output_2_kodiak();
void write_TV_measurements();
void apply_permissions();
void program_completion_tasks();

// Preprocessing setup and initializations 
void initializations();
void count_histories();	
void count_histories_old();
void count_histories_v0();
void count_histories_v02();
void count_histories_v1();

// Preprocessing functions
void combine_data_sets();
void read_data_chunk( const int, const int, const int );
void read_data_chunk_old( const int, const int, const int );
void read_data_chunk_v0( const int, const int, const int );
void read_data_chunk_v02( const int, const int, const int );
void read_data_chunk_v1( const int, const int, const int );
void apply_tuv_shifts( unsigned int );
void convert_mm_2_cm( unsigned int );
void recon_volume_intersections( const int );
void binning( const int );
void import_and_process_data();

void calculate_means();
void initialize_stddev();
void sum_squared_deviations( const int, const int );
void calculate_standard_deviations();
void statistical_cuts( const int, const int );
void statistical_calculations_and_cuts();
void initialize_sinogram();
void construct_sinogram();
void FBP();
void FBP_image_2_hull();
void filter();
void backprojection();
void generate_preprocessing_images();
		
// Hull-Detection 
void hull_initializations();
template<typename T> void initialize_hull( T*&, T*& );
void hull_detection( const int );
void hull_detection_finish();
void SC( const int );
void MSC( const int );
void MSC_edge_detection();
void SM( const int );
void SM_edge_detection();
void SM_edge_detection_2();
void hull_conversion_int_2_bool( int* );
void hull_selection();
void preprocessing();
		
// Image Reconstruction
void initial_iterate_generate_hybrid();
void reconstruct_initial_iterate();
void define_initial_iterate();
void create_hull_image_hybrid();
void generate_history_sequence(ULL, ULL, ULL* );
void verify_history_sequence(ULL, ULL, ULL* );
void print_history_sequence(ULL*, ULL, ULL);

bool is_valid_reconstruction_history(const int, const int);
void reconstruction_cuts_allocations(const int);
void reconstruction_cuts_host_2_device(const int, const int);
void reconstruction_cuts_device_2_host(const int, const int);
void reconstruction_cuts_deallocations();
void reconstruction_cuts_full_tx( const int );
void reconstruction_cuts_partial_tx(const int, const int);
void reconstruction_cuts_partial_tx_preallocated(const int, const int);

void reconstruction_cuts_allocations_nobool(const int);
void reconstruction_cuts_host_2_device_nobool(const int, const int);
void reconstruction_cuts_device_2_host_nobool(const int, const int);
void reconstruction_cuts_deallocations_nobool();
void reconstruction_cuts_full_tx_nobool( const int );
void reconstruction_cuts_partial_tx_nobool(const int, const int);
void reconstruction_cuts_partial_tx_preallocated_nobool(const int, const int);
void reconstruction_cuts();

void generate_trig_tables();
void generate_scattering_coefficient_table();
void generate_polynomial_tables();
void import_trig_tables();
void import_scattering_coefficient_table();
void import_polynomial_tables();
void MLP_lookup_table_2_GPU();
void setup_MLP_lookup_tables();
void free_MLP_lookup_tables();

void DROP_setup_update_arrays();
void DROP_free_update_arrays();
void DROP_allocations(const int);
void DROP_host_2_device(const int,const int);
void DROP_deallocations();
void DROP_full_tx_iteration(const int, const int);
void DROP_partial_tx_iteration( const int, const int);
void DROP_partial_tx_preallocated_iteration( const int, const int);
void DROP_full_tx_iteration(const int, const int, double);
void DROP_partial_tx_iteration( const int, const int, double);
void DROP_partial_tx_preallocated_iteration( const int, const int, double);
void DROP_GPU(const unsigned int);
void DROP_GPU(const unsigned int, const int, double);
void x_host_2_GPU();
void x_GPU_2_host();

void generate_TVS_eta_sequence();
template<typename T> float calculate_total_variation( T*, bool  );
void allocate_perturbation_arrays( bool);
void deallocate_perturbation_arrays( bool);
template<typename T> void generate_perturbation_array( T* );
template<typename T, typename P> void apply_TVS_perturbation( T*, T*, P*, float, bool*);
template<typename T> void iteratively_perturb_image(T*, bool*, UINT);
template<typename T> void iteratively_perturb_image_in_place(T*, bool*, UINT);
template<typename T> void iteratively_perturb_image_GPU( T*, bool*, UINT);
template<typename T> void iteratively_perturb_image_in_place_GPU( T*, bool*, UINT);
template<typename T> void iteratively_perturb_image_unconditional(T*, bool*, UINT);
template<typename T> void iteratively_perturb_image_unconditional_GPU( T*, bool*, UINT);
void NTVS_iteration(const int);
void image_reconstruction();  

//Console Window Print Statement Functions
std::string color_encoding_statement(const char*, const char*, const char*);
std::string echo_statement(const char*, const char*, const char*, const char*);
std::string echo_statement(std::string, const char*, const char*, const char*);
void print_multiline_bash_results(const char*, const char*, const char*, const char*);
void change_text_color(const char*, const char*, const char*, bool);
std::string colored_text(const char*, const char*, const char*, const char*);
std::string colored_text(std::string, const char*, const char*, const char*);
void print_colored_text(const char*, const char*, const char*, const char*);
void print_colored_text(std::string, const char*, const char*, const char*);
void print_section_separator(const char, const char*, const char*, const char*);
void print_section_separator(const char, const char*, const char*, const char*);
void print_section_header( const char*, const char, const char*, const char*, const char*, const char* );
void print_section_exit( const char*, const char*, const char*, const char*, const char*, const char*);
template<typename T> char print_format_identification( T);
template<typename T> void print_labeled_value(const char*, T, const char*, const char*, const char*, const char* );
std::string change_text_color_cmd(const char*, const char*, const char*, bool);

// Write arrays/vectors to file(s)
void binary_2_ASCII();
template<typename T> void array_2_disk( const char*, const char*, const char*, T*, const int, const int, const int, const int, const bool );
template<typename T> void vector_2_disk( const char*, const char*, const char*, std::vector<T>, const int, const int, const int, const bool );
template<typename T> void t_bins_2_disk( FILE*, const std::vector<int>&, const std::vector<T>&, const BIN_ANALYSIS_TYPE, const int );
template<typename T> void bins_2_disk( const char*, const std::vector<int>&, const std::vector<T>&, const BIN_ANALYSIS_TYPE, const BIN_ANALYSIS_FOR, const BIN_ORGANIZATION, ... );
template<typename T> void t_bins_2_disk( FILE*, int*&, T*&, const unsigned int, const BIN_ANALYSIS_TYPE, const BIN_ORGANIZATION, int );
template<typename T> void bins_2_disk( const char*, int*&, T*&, const int, const BIN_ANALYSIS_TYPE, const BIN_ANALYSIS_FOR, const BIN_ORGANIZATION, ... );

// New routine test functions
void NTVS_timing_analysis();
void test_func();
void test_func2( std::vector<int>&, std::vector<double>&);

/***********************************************************************************************************************************************************************************************************************/
/****************************************************************************************** Device (GPU) function declarations *****************************************************************************************/
/***********************************************************************************************************************************************************************************************************************/

// Device helper functions
cudaError_t CUDA_error_check( char* );
cudaError_t CUDA_error_check_and_sync_device( char*, char* );

//template<typename H, typename D> __global__ void averaging_filter_GPU( H*, D*, int, bool, double );
template<typename D> __global__ void median_filter_GPU( D*, D*, int, bool, double );
template<typename D> __global__ void averaging_filter_GPU( D*, D*, int, bool, double );
template<typename D> __global__ void apply_averaging_filter_GPU( D*, D* );

// Image position/voxel calculation functions
__device__ int calculate_voxel_GPU( double, double, double );
__device__ int positions_2_voxels_GPU(const double, const double, const double, int&, int&, int& );
__device__ int position_2_voxel_GPU( double, double, double );
__device__ void voxel_2_3D_voxels_GPU( int, int&, int&, int& );
__device__ double voxel_2_position_GPU( int, double, int, int );
__device__ void voxel_2_positions_GPU( int, double&, double&, double& );
__device__ double voxel_2_radius_squared_GPU( int );

// Voxel walk algorithm functions
__device__ double distance_remaining_GPU( double, double, int, int, double, int );
__device__ double edge_coordinate_GPU( double, int, double, int, int );
__device__ double path_projection_GPU( double, double, double, int, double, int, int );
__device__ double corresponding_coordinate_GPU( double, double, double, double );
__device__ void take_2D_step_GPU( const int, const int, const int, const double, const double, const double, const double, const double, const double, const double, const double, const double, double&, double&, double&, int&, int&, int&, int&, double&, double&, double& );
__device__ void take_3D_step_GPU( const int, const int, const int, const double, const double, const double, const double, const double, const double, const double, const double, const double, double&, double&, double&, int&, int&, int&, int&, double&, double&, double& );

// Preprocessing routines
__device__ bool calculate_intercepts( double, double, double, double&, double& );
__global__ void recon_volume_intersections_GPU( int, int*, bool*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float* );
__global__ void binning_GPU( int, int*, int*, bool*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float* );
__global__ void calculate_means_GPU( int*, float*, float*, float* );
__global__ void sum_squared_deviations_GPU( int, int*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*  );
__global__ void calculate_standard_deviations_GPU( int*, float*, float*, float* );
__global__ void statistical_cuts_GPU( int, int*, int*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, bool* );
__global__ void construct_sinogram_GPU( int*, float* );
__global__ void filter_GPU( float*, float* );
__global__ void backprojection_GPU( float*, float* );
__global__ void FBP_image_2_hull_GPU( float*, bool* );

// Hull-Detection 
template<typename T> __global__ void initialize_hull_GPU( T* );
__global__ void SC_GPU( const int, bool*, int*, bool*, float*, float*, float*, float*, float*, float*, float* );
__global__ void MSC_GPU( const int, int*, int*, bool*, float*, float*, float*, float*, float*, float*, float* );
__global__ void SM_GPU( const int, int*, int*, bool*, float*, float*, float*, float*, float*, float*, float* );
__global__ void MSC_edge_detection_GPU( int* );
__global__ void MSC_edge_detection_GPU( int*, int* );
__global__ void SM_edge_detection_GPU( int*, int* );
__global__ void SM_edge_detection_GPU_2( int*, int* );
__global__ void carve_differences( int*, int* );

__global__ void create_hull_image_hybrid_GPU( bool*&, float*& );
template<typename O> __device__ bool find_MLP_endpoints_GPU( O*, double, double, double, double, double, double&, double&, double&, int&, int&, int&, bool);                                            			//*
//__device__ bool find_MLP_endpoints_GPU( bool*, double, double, double, double, double, double&, double&, double&, int&, int&, int&, bool); 
__device__ void find_MLP_path_GPU(float*, double, unsigned int, double, double, double, double, double, double, double, double, double, double, double, double, unsigned int*, int&, double&, double&, double& );                        		//*
//__device__ void find_MLP_path_GPU(float*, double, unsigned int, double, double, double, double, double, double, double, double, double, double, double, float, unsigned int*, int&, double&, double&, double& );                        		//*
__device__ void find_MLP_path_GPU_tabulated(float*, double, unsigned int, double, double, double, double, double, double, double, double, double, double, double, float, unsigned int*, int&, double&, double&, double&, double* sin_table, double* cos_table, double* scattering_table, double* poly_1_2, double* poly_2_3, double* poly_3_4, double* poly_2_6, double* poly_3_12);
//__device__ void find_MLP_path_GPU_tabulated(float*, double, unsigned int, double, double, double, double, double, double, double, double, double, double, float, unsigned int*, float&, int&, double*, double*, double*, double*, double*, double*, double*, double*);
__global__ void collect_MLP_endpoints_GPU(bool*, unsigned int*, bool*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float* , int, int );                               	//*
__global__ void collect_MLP_endpoints_GPU_nobool(unsigned int*, bool*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float* , int, int );                               	//*
__device__ void DROP_block_update_GPU(int, int, unsigned int*, unsigned int*, float*, float*, double, double, double, double );
__global__ void calculate_x_update_GPU(float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, unsigned int*, float*, unsigned int*, int, int, double, double*, double*, double*, double*, double*, double*, double*, double*);
__global__ void block_update_GPU(float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, unsigned int*, float*, unsigned int*, int, int, double, double*, double*, double*, double*, double*, double*, double*, double*);
//__global__ void DROP_block_update_GPU(float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, unsigned int*, float*, unsigned int*, int, int, double, double*, double*, double*, double*, double*, double*, double*, double*);
__global__ void init_image_GPU(float*, unsigned int*); 
__global__ void image_update_GPU(float*, double*, unsigned int*); 
__device__ double EffectiveChordLength_GPU(double, double);                                                                                                                                              			//*

__device__ float sigmoid_GPU(UINT);
__device__ float erf_GPU(UINT);
__device__ float atan_GPU(UINT);
__device__ float tanh_GPU(UINT);
__device__ float linear_over_root_GPU(UINT);
__device__ float s_curve_scale_GPU(UINT, UINT);

template<typename T> __device__ float calculate_total_variation_GPU( T* );
template<typename T> __global__ void generate_perturbation_array_GPU( float*, float*, float*, float*, float*, T* );
template<typename T> __device__ void generate_perturbation_array_GPU( float*, float*, float*, float*, float*, T*, int, int, int );

// New routine test functions
__global__ void test_func_GPU( int* );
__global__ void test_func_device( double*, double*, double* );

/***********************************************************************************************************************************************************************************************************************/
/***************************************************************************************************** Program Main ****************************************************************************************************/
/***********************************************************************************************************************************************************************************************************************/
int main(int NUM_RUN_ARGUMENTS, char** RUN_ARGUMENTS)
{
	if( FUNCTION_TESTING )
		test_func();			
	else
	{
		program_startup_tasks();
		preprocessing();
		image_reconstruction(); // Write to the external file	 	
		program_completion_tasks();
	}
	exit_program_if(true);
}
/***********************************************************************************************************************************************************************************************************************/
/********************************************************************************************** Execution Control Functions ********************************************************************************************/
/***********************************************************************************************************************************************************************************************************************/
void timer( bool start, clock_t& start_time, clock_t& end_time, double& execution_time, const char* statement )
{
	if( start )
		start_time = clock();
	else if( !start )
	{
		end_time = clock();
		clock_t execution_clock_cycles = (end_time - start_time);
		execution_time = static_cast<double>( execution_clock_cycles) / CLOCKS_PER_SEC;
		sprintf(print_statement, "Total execution time %s: %3f [seconds]\n", statement, execution_time );	
		print_colored_text( print_statement, YELLOW_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT );
	}
	else
		puts("ERROR: Invalid timer control parameter passed");
}
double timer( bool start, clock_t& start_time, const char* statement )
{
	double execution_time = 0.0;
	if( start )
		start_time = clock();
	else
	{
		clock_t end_time = clock();
		clock_t execution_clock_cycles = (end_time - start_time);
		execution_time = static_cast<double>( execution_clock_cycles) / CLOCKS_PER_SEC;
		sprintf(print_statement, "Total execution time %s: %3f [seconds]\n", statement, execution_time );	
		print_colored_text( print_statement, YELLOW_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT );
	}
	return execution_time;
}
void pause_execution()
{
	clock_t pause_start, pause_end;
	pause_start = clock();
	//char user_response[20];
	puts("Execution paused.  Hit enter to continue execution.\n");
	 //Clean the stream and ask for input
	//std::cin.ignore ( std::numeric_limits<std::streamsize>::max(), '\n' );
	std::cin.get();

	pause_end = clock();
	pause_cycles += pause_end - pause_start;
}
void exit_program_if( bool early_exit, const char* statement)
{
	/************************************************************************************************************************************************************/
	/* Program has finished execution. Require the user to hit enter to terminate the program and close the terminal/console window	and write execution times to disk							*/ 															
	/************************************************************************************************************************************************************/	
	if( early_exit )
	{
		char user_response[20];
		double execution_time;
		timer( STOP, program_start, program_end, execution_time, statement );
		if( !CLOSE_AFTER_EXECUTION )
		{
			sprintf(print_statement, "Press 'ENTER' to exit program...");
			print_colored_text(print_statement, GREEN_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );
			fgets(user_response, sizeof(user_response), stdin);
		}
		exit(1);
	}
}
void exit_program_if( bool early_exit)
{
	/************************************************************************************************************************************************************/
	/* Program has finished execution. Require the user to hit enter to terminate the program and close the terminal/console window	and write execution times to disk							*/ 															
	/************************************************************************************************************************************************************/	
	if( early_exit )
	{
		char user_response[20];
		double execution_time;
		sprintf( print_statement, "Program execution completed");
		print_section_header( print_statement, MAJOR_SECTION_SEPARATOR, WHITE_TEXT, WHITE_TEXT, RED_BACKGROUND, DONT_UNDERLINE_TEXT );
		timer( STOP, program_start, program_end, execution_time, "" );
		if( !CLOSE_AFTER_EXECUTION )
		{
			sprintf(print_statement, "Press 'ENTER' to exit program...");
			print_colored_text(print_statement, GREEN_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );
			fgets(user_response, sizeof(user_response), stdin);
		}
		exit(1);
	}
}
/***********************************************************************************************************************************************************************************************************************/
/******************************************************************************* Reading/Setting Run Settings, Parameters, and Configurations **************************************************************************/
/***********************************************************************************************************************************************************************************************************************/
char((&current_MMDD( char(&date_MMDD)[5]))[5])
{
	time_t rawtime;
	time (&rawtime);
	struct tm * timeinfo = gmtime (&rawtime);
	strftime (date_MMDD,11,"%m%d", timeinfo);
	return date_MMDD;
}
char((&current_MMDDYYYY( char(&date_MMDDYYYY)[9]))[9])
{
	time_t rawtime;
	time (&rawtime);
	struct tm * timeinfo = gmtime (&rawtime);
	strftime (date_MMDDYYYY,11,"%m%d%Y", timeinfo);
	return date_MMDDYYYY;
}
char((&current_YY_MM_DD( char(&date_YY_MM_DD)[9]))[9])
{
	time_t rawtime;
	time (&rawtime);
	struct tm * timeinfo = gmtime (&rawtime);
	strftime (date_YY_MM_DD,11,"%y-%m-%d", timeinfo);
	return date_YY_MM_DD;
}
char((&current_date_time_formatted( char(&current_date_time)[128]))[128])
{
	char current_date[32], time_local[32], time_GMT[32];
	time_t rawtime;
	time (&rawtime);
	struct tm * GMT_timeinfo, * local_timeinfo = localtime (&rawtime);
	strftime (current_date, 80,"%m-%d-%Y (%a)", local_timeinfo);
	strftime (time_local, 80,"%X %Z", local_timeinfo);
	GMT_timeinfo = gmtime (&rawtime);
	strftime (time_GMT, 80,"%X ", GMT_timeinfo);
	
	sprintf(current_date_time, "%s %s (%s GMT)", current_date, time_local, time_GMT);
	return current_date_time;
}
bool file_exists ( const char* file_path) 
{
    #if defined(_WIN32) || defined(_WIN64)
		return false;
		//return file_path && ( PathFileExists(file_path) != 0 );
    #else
		if( access( file_path, F_OK ) != -1 )
			return true;
		else 
			return false;
		//struct stat sb;
		//return file_path && (stat (file_path, &sb) == 0 );
   #endif
} 
bool file_exists_OS_independent ( const char* file_path)
{
	//char system_command[256];
	//sprintf("cd %s", 
	return true;
}
bool directory_exists(char* dir_name )
{
	sprintf(system_command, "%s %s", BASH_CHANGE_DIR, dir_name);
	return !system( system_command );
}
bool input_directory_exists()
{
	char input_folder_name[256];
	sprintf(input_folder_name, "%s%s/%s_%03d%s", INPUT_DIRECTORY, INPUT_FOLDER );
	if(!USING_RSYNC || directory_exists(input_folder_name))
		return true;
	else 
		return false;
}
bool mkdir(char* directory)
{
	char mkdir_command[256];
	#if defined(_WIN32) || defined(_WIN64)
		sprintf(mkdir_command, "mkdir \"%s\"", directory);
	#else
		sprintf(mkdir_command, "mkdir -p \"%s\"", directory);
	#endif
	return !system(mkdir_command);
}
unsigned int create_unique_dir( char* dir_name )
{
	unsigned int i = 0;
	char dir_2_make[256];
	sprintf(dir_2_make, "%s", dir_name);
	while( directory_exists(dir_2_make) )
		sprintf(dir_2_make, "%s_%u", dir_name, ++i);
	mkdir( dir_2_make );
	return i;
}
unsigned int create_unique_dir( const char* search_dir, const char* output_dir, char* dir_name )
{
	unsigned int i = 0;
	char storage_dir_2_make[256];
	char output_dir_2_make[256];
	sprintf(storage_dir_2_make, "%s//%s", search_dir, dir_name );
	sprintf(output_dir_2_make, "%s%s", output_dir, dir_name );
	change_text_color( BLACK_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT, false);
	while( directory_exists(storage_dir_2_make) )
	{
		sprintf(storage_dir_2_make, "%s//%s_%u", search_dir, dir_name, ++i);
		sprintf(output_dir_2_make, "%s%s_%u", output_dir, dir_name, i);
	}
	print_colored_text("Generating unique output data folder name and creating the corresponding directory on the compute node and the network-attached storage device...", CYAN_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );	
	sprintf(system_command, "%s \"%s\"; %s \"%s\"", BASH_MKDIR_CHAIN, storage_dir_2_make, BASH_MKDIR_CHAIN, output_dir_2_make);
	system( system_command );
	return i;
}
void copy_data(const char* command, const char* source, const char* destination )
{
	sprintf(bash_command, "%s \"%s\" \"%s\"", command, source, destination);
	print_multiline_bash_results(bash_command, LIGHT_PURPLE_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT);
}
void copy_folder_contents(const char* command, const char* source, const char* destination )
{
	sprintf(bash_command, "%s %s//* %s", command, source, destination);
	print_multiline_bash_results(bash_command, LIGHT_PURPLE_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT);
}
void set_file_permissions(const char* path, const char* permission)
{
	if(PRINT_CHMOD_CHANGES_ONLY)
		sprintf(bash_command, "%s %s \"%s\"", BASH_CHANGE_PERMISSIONS, permission, path);
	else
		sprintf(bash_command, "%s %s \"%s\"", BASH_SET_PERMISSIONS, permission, path);
	print_multiline_bash_results(bash_command, LIGHT_PURPLE_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT);
}
/***********************************************************************************************************************************************************************************************************************/
/********************************************************************************** Program Startup Tasks **************************************************************************************************************/
/***********************************************************************************************************************************************************************************************************************/
void check_4_missing_input()
{
	char local_input_dir[256];
	char global_input_dir[256];
	char cp_command[256];
	char mkdir_command[256];
	print_colored_text("Verifying input data exists on compute node's local drive...", CYAN_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );
	if(!USING_RSYNC)
	{
		sprintf(local_input_dir, "%s%s", INPUT_DIRECTORY, INPUT_FOLDER );
		sprintf(global_input_dir, "%s//%s", PCT_ORG_DATA_DIR_SET, INPUT_FOLDER );
		print_colored_text("Input data directory:", GREEN_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );
		print_colored_text(local_input_dir, LIGHT_PURPLE_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT );
		
		bool verified_data = verify_input_data();
		if(!directory_exists(local_input_dir))
		{
			sprintf(mkdir_command, "%s %s", BASH_MKDIR_CHAIN, local_input_dir);
			system(mkdir_command);
			if(!directory_exists(global_input_dir))
			{
				sprintf(print_statement, "Missing input projection data not found in %s", PCT_ORG_DATA_DIR_SET );
				print_colored_text(print_statement, RED_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );	
				exit_program_if(true);
			}
		}
		if(!verified_data)
		{
		
			sprintf(cp_command, "%s %s//* %s%s", BASH_COPY_DIR, global_input_dir, INPUT_DIRECTORY, INPUT_FOLDER );
			system(cp_command);		
		}
		print_section_exit( "Finished input data verification", SECTION_EXIT_CSTRING, RED_TEXT, RED_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );	
	}
}
bool verify_input_data()
{
	char input_file_name[256];
	if(!USING_RSYNC)
	{
		//bool missing_files = false;
		print_colored_text("Verifying existence of each input projection data file on compute node's local drive...", CYAN_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );
		for( unsigned int gantry_angle = 0; gantry_angle < 360; gantry_angle += int(GANTRY_ANGLE_INTERVAL) )
		{		
			sprintf(input_file_name, "%s%s/%s_%03d%s", INPUT_DIRECTORY, INPUT_FOLDER, PROJECTION_DATA_BASENAME, gantry_angle, PROJECTION_DATA_EXTENSION  );
			//missing_files &= file_exists(input_file_name);
			if(!file_exists(input_file_name))
				return false;
		}
	}
	return true;
}
void set_compute_node()
{
	std::string terminal_string = terminal_response(HOSTNAME_CMD);
	terminal_string.pop_back();
	sprintf(CURRENT_COMPUTE_NODE, "%s", terminal_string.c_str() );
	print_colored_text("Querying the current compute node...", CYAN_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );
			
	if( terminal_string.compare(kodiak_ID) == 0 )
		sprintf( CURRENT_COMPUTE_NODE_ALIAS, "%s", KODIAK_HOSTNAME_CSTRING);
	else if( ( terminal_string.compare(whartnell_ID) ) == 0 || ( terminal_string.compare(whartnell_hostname) == 0 ) )
		sprintf( CURRENT_COMPUTE_NODE_ALIAS, "%s", ECSN1_HOSTNAME_CSTRING);
	else if( terminal_string.compare(ptroughton_ID) == 0 )
		sprintf( CURRENT_COMPUTE_NODE_ALIAS, "%s", ECSN2_HOSTNAME_CSTRING);
	else if( terminal_string.compare(jpertwee_ID) == 0 )
		sprintf( CURRENT_COMPUTE_NODE_ALIAS, "%s", ECSN3_HOSTNAME_CSTRING);
	else if( terminal_string.compare(tbaker_ID) == 0 )
		sprintf( CURRENT_COMPUTE_NODE_ALIAS, "%s", ECSN4_HOSTNAME_CSTRING);
	else if( terminal_string.compare(pdavison_ID) == 0 )
		sprintf( CURRENT_COMPUTE_NODE_ALIAS, "%s", ECSN5_HOSTNAME_CSTRING);
	else if( terminal_string.compare(workstation_2_hostname) == 0 )
		sprintf( CURRENT_COMPUTE_NODE_ALIAS, "%s", WS_HOSTNAME_CSTRING);
	else
		sprintf( CURRENT_COMPUTE_NODE_ALIAS, "Unknown Host");
	std::string compute_node_string = colored_text(CURRENT_COMPUTE_NODE, LIGHT_PURPLE_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );
	std::string compute_node_alias_string = colored_text(CURRENT_COMPUTE_NODE, LIGHT_PURPLE_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );
	print_labeled_value("Current compute node =", CURRENT_COMPUTE_NODE, GREEN_TEXT, LIGHT_PURPLE_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT);
	print_labeled_value("Current compute node alias =", CURRENT_COMPUTE_NODE_ALIAS, GREEN_TEXT, LIGHT_PURPLE_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT);
}
void set_user_strings()
{
	print_colored_text("Setting the usernames for logins and setting directory/file ownership and the user/group owning the git code and the network-attached storage device's home directory", CYAN_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );
	std::string username_string = terminal_response(USERNAME_CMD);
	username_string.pop_back();
	sprintf(USERNAME, "%s", username_string.c_str());
	//sprintf( USE_TARDIS_USERNAME, "%s", TARDIS_USERNAME);
	//sprintf( USE_KODIAK_USERNAME, "%s", KODIAK_USERNAME);
	sprintf( USE_TARDIS_USERNAME, "%s", USERNAME);
	sprintf( USE_KODIAK_USERNAME, "%s", USERNAME);
	sprintf( USE_BAYLOR_USERNAME, "%s", BAYLOR_USERNAME);
	if(SHARE_OUTPUT_DATA)
		sprintf( USE_HOME_DIR_USERNAME, "%s", RECON_GROUP_HOME_DIR);
	else
		sprintf( USE_HOME_DIR_USERNAME, "%s", USERNAME);
	//sprintf( USE_CODE_OWNER_NAME, "%s", KODIAK_USERNAME);	
	if(USE_GROUP_CODE)
		sprintf( USE_RCODE_OWNER_NAME, "%s", RECON_GROUP_USERNAME);	
	else
		sprintf( USE_RCODE_OWNER_NAME, "%s", USERNAME);	
}
void set_compute_system_directories()
{
	print_colored_text( "Setting the IO data/code directories/subdirectories for the compute node, network-attached storage device, and the current user/group home directories", CYAN_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT );		
	sprintf(COMMON_ORG_DATA_SUBDIRECTORY, "%s//%s", PCT_DATA_FOLDER, ORGANIZED_DATA_FOLDER);
	sprintf(COMMON_RECON_DATA_SUBDIRECTORY, "%s//%s", PCT_DATA_FOLDER, RECON_DATA_FOLDER);
	sprintf(COMMON_RCODE_SUBDIRECTORY, "%s//%s//%s", PCT_CODE_FOLDER, RECONSTRUCTION_FOLDER, USE_RCODE_OWNER_NAME);
	sprintf(COMMON_GIT_CODE_SUBDIRECTORY, "%s//%s//%s", GIT_FOLDER, GIT_ACCOUNT, GIT_REPOSITORY);	
		
	sprintf(PCT_DATA_DIR_SET, "%s//%s", PCT_PARENT_DIR, PCT_DATA_FOLDER);
	sprintf(PCT_ORG_DATA_DIR_SET, "%s//%s", PCT_PARENT_DIR, COMMON_ORG_DATA_SUBDIRECTORY);
	sprintf(PCT_RECON_DIR_SET, "%s//%s", PCT_PARENT_DIR, COMMON_RECON_DATA_SUBDIRECTORY);	
	sprintf(PCT_CODE_PARENT_SET, "%s//%s", PCT_PARENT_DIR, PCT_CODE_FOLDER);
	sprintf(PCT_RCODE_PARENT_SET, "%s//%s", PCT_PARENT_DIR, COMMON_RCODE_SUBDIRECTORY);
	sprintf(PCT_GIT_RCODE_PARENT_SET, "%s//%s", PCT_RCODE_PARENT_SET, COMMON_GIT_CODE_SUBDIRECTORY);
	
	sprintf(TARDIS_DATA_DIR_SET, "%s//%s", TARDIS_PARENT_DIR, PCT_DATA_FOLDER);
	sprintf(TARDIS_ORG_DATA_DIR_SET, "%s//%s", TARDIS_DATA_DIR_SET, ORGANIZED_DATA_FOLDER);
	sprintf(TARDIS_RECON_DIR_SET, "%s//%s", TARDIS_DATA_DIR_SET, RECON_DATA_FOLDER);	
	sprintf(TARDIS_CODE_PARENT_SET, "%s//%s", TARDIS_PARENT_DIR, PCT_CODE_FOLDER);
	sprintf(TARDIS_RCODE_PARENT_SET, "%s//%s", TARDIS_PARENT_DIR, COMMON_RCODE_SUBDIRECTORY);
	sprintf(TARDIS_GIT_RCODE_PARENT_SET, "%s//%s", TARDIS_RCODE_PARENT_SET, COMMON_GIT_CODE_SUBDIRECTORY);
	
	sprintf(SHARED_HOME_DIR_SET, "%s//%s//%s", PCT_PARENT_DIR, HOME_FOLDER, RECON_GROUP_NAME);
	sprintf(SHARED_DATA_DIR_SET, "%s//%s", SHARED_HOME_DIR_SET, PCT_DATA_FOLDER);
	sprintf(SHARED_ORG_DATA_DIR_SET, "%s//%s", SHARED_DATA_DIR_SET, ORGANIZED_DATA_FOLDER);
	sprintf(SHARED_RECON_DIR_SET, "%s//%s", SHARED_DATA_DIR_SET, RECON_DATA_FOLDER);	
	sprintf(SHARED_CODE_PARENT_SET, "%s//%s", SHARED_HOME_DIR_SET, PCT_CODE_FOLDER);
	sprintf(SHARED_RCODE_PARENT_SET, "%s//%s", SHARED_HOME_DIR_SET, COMMON_RCODE_SUBDIRECTORY);
	sprintf(SHARED_GIT_RCODE_PARENT_SET, "%s//%s", SHARED_RCODE_PARENT_SET, COMMON_GIT_CODE_SUBDIRECTORY);
	
	sprintf(MY_HOME_DIR_SET, "%s//%s//%s", PCT_PARENT_DIR, HOME_FOLDER, USERNAME);
	sprintf(MY_DATA_DIR_SET, "%s//%s", MY_HOME_DIR_SET, PCT_DATA_FOLDER);
	sprintf(MY_ORG_DATA_DIR_SET, "%s//%s", MY_DATA_DIR_SET, ORGANIZED_DATA_FOLDER);
	sprintf(MY_RECON_DIR_SET, "%s//%s", MY_DATA_DIR_SET, RECON_DATA_FOLDER);	
	sprintf(MY_CODE_PARENT_SET, "%s//%s", MY_HOME_DIR_SET, PCT_CODE_FOLDER);
	sprintf(MY_RCODE_PARENT_SET, "%s//%s", MY_HOME_DIR_SET, COMMON_RCODE_SUBDIRECTORY);
	sprintf(MY_GIT_RCODE_PARENT_SET, "%s//%s", MY_RCODE_PARENT_SET, COMMON_GIT_CODE_SUBDIRECTORY);
	
	//sprintf(WS2_RECON_DIR_SET, "%s//%s", PCT_PARENT_DIR, PCT_CODE_FOLDER);
	//sprintf(WS2_CODE_PARENT_SET, "%s//%s", PCT_DATA_DIR_SET, RECON_DATA_FOLDER);	
	//sprintf(MYLAPTOP_ORG_DATA_DIR_SET, "%s//%s", PCT_DATA_DIR_SET, ORGANIZED_DATA_FOLDER);
	//sprintf(MYLAPTOP_RECON_DIR_SET, "%s//%s", PCT_DATA_DIR_SET, RECON_DATA_FOLDER);		
}
void set_git_branch_info()
{		
	if(SHARE_OUTPUT_DATA)
	{
		sprintf( CURRENT_CODE_PARENT, "%s", SHARED_CODE_PARENT_SET);
		sprintf( CURRENT_RCODE_PARENT, "%s", SHARED_RCODE_PARENT_SET);
	}
	else
	{
		sprintf( CURRENT_CODE_PARENT, "%s", MY_CODE_PARENT_SET);
		sprintf( CURRENT_RCODE_PARENT, "%s", MY_RCODE_PARENT_SET);
	}
	//////////////////////////////////////////////////////
	char git_branch_name_command[256];
	char git_commit_hash_command[256];
	char git_commit_date_command[256];
	print_colored_text("Querying git repository info...", CYAN_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );
	sprintf(GIT_REPO_PATH, "%s//%s", TARDIS_RCODE_PARENT_SET, COMMON_GIT_CODE_SUBDIRECTORY);	
	sprintf(git_branch_name_command, "cd %s; git rev-parse --abbrev-ref HEAD", GIT_REPO_PATH);
	sprintf(git_commit_hash_command, "cd %s; git rev-parse HEAD", GIT_REPO_PATH);
	sprintf(git_commit_date_command, "cd %s; git log --pretty=format:\"%%cd\"", GIT_REPO_PATH);
	
	std::string git_branch_name_string = terminal_response(git_branch_name_command);
	std::string git_commit_hash_string = terminal_response(git_commit_hash_command);
	std::string git_commit_dates_string = terminal_response(git_commit_date_command);
	std::string git_commit_date_string = git_commit_dates_string.substr(0, GIT_COMMIT_DATE_CSTRING_LENGTH);	
	git_branch_name_string.pop_back();
	git_commit_hash_string.pop_back();
	git_commit_date_string.pop_back();	
	sprintf(GIT_BRANCH_NAME, "%s", git_branch_name_string.c_str());
	sprintf(GIT_COMMIT_HASH, "%s", git_commit_hash_string.c_str());
	sprintf(GIT_COMMIT_DATE, "%s", git_commit_date_string.c_str());
	sprintf(GIT_REPO_INFO, "%s : %s (%s)", git_branch_name_string.c_str(), git_commit_hash_string.c_str(), git_commit_date_string.c_str());
	
	// Colorize and print the git info strings 
	std::string git_account_string_colored = colored_text(GIT_ACCOUNT, LIGHT_PURPLE_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT );
	std::string git_repository_string_colored = colored_text(GIT_REPOSITORY, LIGHT_PURPLE_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT );
	std::string git_branch_name_string_colored = colored_text(git_branch_name_string, LIGHT_PURPLE_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT );
	std::string git_commit_hash_string_colored = colored_text(git_commit_hash_string, LIGHT_PURPLE_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT );
	std::string git_commit_date_string_colored = colored_text(git_commit_date_string, LIGHT_PURPLE_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT );
	print_labeled_value("Current git account =", git_account_string_colored.c_str(), GREEN_TEXT, LIGHT_PURPLE_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT);
	print_labeled_value("Current git repository =", git_repository_string_colored.c_str(), GREEN_TEXT, LIGHT_PURPLE_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT);
	print_labeled_value("Current git branch =", git_branch_name_string_colored.c_str(), GREEN_TEXT, LIGHT_PURPLE_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT);
	print_labeled_value("Current git commit hash =", git_commit_hash_string_colored.c_str(), GREEN_TEXT, LIGHT_PURPLE_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT);
	print_labeled_value("Current git commit date =", git_commit_date_string_colored.c_str(), GREEN_TEXT, LIGHT_PURPLE_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT);
}
void set_and_make_output_folder()
{
	char folder_name[256];
	unsigned int i = 0;
	bool naming_applied = false;
	std::string color_command = change_text_color_cmd( BLACK_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT, false);
	sprintf(OUTPUT_FOLDER_UNIQUE, "%s//", OUTPUT_FOLDER );
	
	if(SHARE_OUTPUT_DATA)
	{
		sprintf( CURRENT_DATA_DIR, "%s", SHARED_DATA_DIR_SET);
		sprintf( CURRENT_RECON_DIR, "%s", SHARED_RECON_DIR_SET);
	}
	else
	{
		sprintf( CURRENT_DATA_DIR, "%s", MY_DATA_DIR_SET);
		sprintf( CURRENT_RECON_DIR, "%s", MY_RECON_DIR_SET);
	}
	sprintf(INPUT_DIRECTORY_SET, "%s", TARDIS_ORG_DATA_DIR_SET);
	sprintf(OUTPUT_DIRECTORY_SET, "%s", TARDIS_RECON_DIR_SET);
	if( BLOCK_TESTING_ON )
	{
		sprintf(OUTPUT_FOLDER_UNIQUE, "%sB_%d_L_%3f", OUTPUT_FOLDER_UNIQUE, DROP_BLOCK_SIZE, LAMBDA );
		naming_applied = true;
	}
	if (MLP_LENGTH_TESTING_ON)
	{
		if(naming_applied)
			sprintf(OUTPUT_FOLDER_UNIQUE, "%s_I_%d_N_%d", OUTPUT_FOLDER_UNIQUE, IGNORE_SHORT_MLP, MIN_MLP_LENGTH);
		else
			sprintf(OUTPUT_FOLDER_UNIQUE, "%sI_%d_N_%d", OUTPUT_FOLDER_UNIQUE, IGNORE_SHORT_MLP, MIN_MLP_LENGTH);
		naming_applied = true;
	}
	if( S_CURVE_TESTING_ON )
	{
		if(naming_applied)
			sprintf(OUTPUT_FOLDER_UNIQUE, "%s_k_%3f_x0_%d", OUTPUT_FOLDER_UNIQUE, SIGMOID_STEEPNESS, SIGMOID_MID_SHIFT );	
		else
			sprintf(OUTPUT_FOLDER_UNIQUE, "%sk_%3f_x0_%d", OUTPUT_FOLDER_UNIQUE, SIGMOID_STEEPNESS, SIGMOID_MID_SHIFT );	
		naming_applied = true;
	}
	if( RECON_PARAMETER_TESTING_ON )
	{
		if(naming_applied)
			sprintf(OUTPUT_FOLDER_UNIQUE, "%s_B_%d_L_%6.5f_Hr_%d_Fr_%d", OUTPUT_FOLDER_UNIQUE, DROP_BLOCK_SIZE, LAMBDA, HULL_AVG_FILTER_RADIUS, FBP_MED_FILTER_RADIUS);
		else
			sprintf(OUTPUT_FOLDER_UNIQUE, "%sB_%d_L_%6.5f_Hr_%d_Fr_%d", OUTPUT_FOLDER_UNIQUE, DROP_BLOCK_SIZE, LAMBDA, HULL_AVG_FILTER_RADIUS, FBP_MED_FILTER_RADIUS);
		//sprintf(OUTPUT_FOLDER_UNIQUE, "%s//TV_%d_A_%3f_L0_%d_Nk_%d", OUTPUT_FOLDER, TVS_CONDITIONED, A, L_0, TVS_REPETITIONS );
		naming_applied = true;
	}
	if( FBP_TESTING_ON )
	{
		if(naming_applied)
			sprintf(OUTPUT_FOLDER_UNIQUE, "%s_FBP_MED_FILTER_TESTING", OUTPUT_FOLDER_UNIQUE);		
		else
			sprintf(OUTPUT_FOLDER_UNIQUE, "%sFBP_MED_FILTER_TESTING", OUTPUT_FOLDER_UNIQUE);		
		naming_applied = true;
	}
	if( FILTER_TESTING_ON )
	{
		if(naming_applied)
			sprintf(OUTPUT_FOLDER_UNIQUE, "%s_Hull_r_%d_FBP_r_%d", OUTPUT_FOLDER_UNIQUE, HULL_AVG_FILTER_RADIUS, FBP_MED_FILTER_RADIUS);		
		else
			sprintf(OUTPUT_FOLDER_UNIQUE, "%sHull_r_%d_FBP_r_%d", OUTPUT_FOLDER_UNIQUE, HULL_AVG_FILTER_RADIUS, FBP_MED_FILTER_RADIUS);		
		naming_applied = true;
	}
	if( NTVS_TESTING_ON )
	{
		if(naming_applied)
			sprintf(OUTPUT_FOLDER_UNIQUE, "%s_TV_%d_A_%3f_L0_%d_Nk_%d", OUTPUT_FOLDER_UNIQUE, TVS_CONDITIONED, A, L_0, TVS_REPETITIONS );
		else
			sprintf(OUTPUT_FOLDER_UNIQUE, "%sTV_%d_A_%3f_L0_%d_Nk_%d", OUTPUT_FOLDER_UNIQUE, TVS_CONDITIONED, A, L_0, TVS_REPETITIONS );
		//sprintf(OUTPUT_FOLDER_UNIQUE, "%s//TV_%d_A_%3f_L0_%d_Nk_%d", OUTPUT_FOLDER, TVS_CONDITIONED, A, L_0, TVS_REPETITIONS );
		naming_applied = true;
	}
	else if( OLD_TVS_TESTING_ON )
	{
		#if TVS_OLD
			TVS_REPETITIONS = 1;
		#endif
		if(naming_applied)
			sprintf(OUTPUT_FOLDER_UNIQUE, "%s_OLD_TV_%d_A_%3f_L0_%d_Nk_%d", OUTPUT_FOLDER_UNIQUE, TVS_CONDITIONED, A, L_0, TVS_REPETITIONS );
		else
			sprintf(OUTPUT_FOLDER_UNIQUE, "%sOLD_TV_%d_A_%3f_L0_%d_Nk_%d", OUTPUT_FOLDER_UNIQUE, TVS_CONDITIONED, A, L_0, TVS_REPETITIONS );
		//sprintf(OUTPUT_FOLDER_UNIQUE, "%s//TV_%d_A_%3f_L0_%d_Nk_%d", OUTPUT_FOLDER, TVS_CONDITIONED, A, L_0, TVS_REPETITIONS );
		naming_applied = true;
	}
	else if( WITH_OPTIMAL_NTVS_TESTING_ON )
	{
		if(naming_applied)
			sprintf(OUTPUT_FOLDER_UNIQUE, "%s_A_%3.2f_Nk_%d", OUTPUT_FOLDER_UNIQUE, A, TVS_REPETITIONS );
		else
			sprintf(OUTPUT_FOLDER_UNIQUE, "%sA_%3.2f_Nk_%d", OUTPUT_FOLDER_UNIQUE, A, TVS_REPETITIONS );
		//sprintf(OUTPUT_FOLDER_UNIQUE, "%s//TV_%d_A_%3f_L0_%d_Nk_%d", OUTPUT_FOLDER, TVS_CONDITIONED, A, L_0, TVS_REPETITIONS );
		naming_applied = true;
	}	
	if( AIR_THRESH_TESTING_ON )
	{
		if(naming_applied)
			sprintf(OUTPUT_FOLDER_UNIQUE, "%s_AIR_%d.%d_I_%3.2f_U_%3.2f", OUTPUT_FOLDER_UNIQUE, IDENTIFY_X_0_AIR, IDENTIFY_X_N_AIR, X_0_AIR_THRESHOLD, X_N_AIR_THRESHOLD );
		else
			sprintf(OUTPUT_FOLDER_UNIQUE, "%sAIR_%d.%d_I_%3.2f_U_%3.2f", OUTPUT_FOLDER_UNIQUE, IDENTIFY_X_0_AIR, IDENTIFY_X_N_AIR, X_0_AIR_THRESHOLD, X_N_AIR_THRESHOLD );
		//sprintf(OUTPUT_FOLDER_UNIQUE, "%s//TV_%d_A_%3f_L0_%d_Nk_%d", OUTPUT_FOLDER, TVS_CONDITIONED, A, L_0, TVS_REPETITIONS );
		naming_applied = true;
	}
	if( OLD_TVS_COMPARISON_TESTING_ON )
	{
		#if TVS_OLD
			TVS_REPETITIONS = 1;
		#endif
		if(naming_applied)
			sprintf(OUTPUT_FOLDER_UNIQUE, "%s_TV_%d_A_%3f_L0_%d_Nk_%d_compared", OUTPUT_FOLDER_UNIQUE, TVS_CONDITIONED, A, L_0, TVS_REPETITIONS );
		else
			sprintf(OUTPUT_FOLDER_UNIQUE, "%sTV_%d_A_%3f_L0_%d_Nk_%d_compared", OUTPUT_FOLDER_UNIQUE, TVS_CONDITIONED, A, L_0, TVS_REPETITIONS );
		//sprintf(OUTPUT_FOLDER_UNIQUE, "%s//TV_%d_A_%3f_L0_%d_Nk_%d", OUTPUT_FOLDER, TVS_CONDITIONED, A, L_0, TVS_REPETITIONS );
		naming_applied = true;
	}
	
	if(!naming_applied)
		sprintf(OUTPUT_FOLDER_UNIQUE, "%s//%s", OUTPUT_FOLDER, EXECUTION_YY_MM_DD );	// EXECUTION_DATE
	
	if( OVERWRITING_OK )
	{
		sprintf(folder_name, "%s%s", OUTPUT_DIRECTORY, OUTPUT_FOLDER_UNIQUE );
		while( directory_exists(folder_name ) )
			sprintf(folder_name, "%s%s_%u", OUTPUT_DIRECTORY, OUTPUT_FOLDER_UNIQUE, ++i );
		sprintf(folder_name, "%s%s_%u", OUTPUT_DIRECTORY, OUTPUT_FOLDER_UNIQUE, --i );
		mkdir( folder_name );
	}
	else
		i = create_unique_dir( CURRENT_RECON_DIR, OUTPUT_DIRECTORY, OUTPUT_FOLDER_UNIQUE );	
	if( i != 0 )
		sprintf(OUTPUT_FOLDER_UNIQUE, "%s_%u", OUTPUT_FOLDER_UNIQUE, i );	
	print_colored_text("Writing output data/images to:", GREEN_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );
	print_colored_text(OUTPUT_FOLDER_UNIQUE, LIGHT_PURPLE_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT );
	print_section_exit("Finished assigning and creating output data directory", SECTION_EXIT_CSTRING, RED_TEXT, RED_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );		
}
void set_IO_folder_names()
{
	print_colored_text("Setting the specific IO directory/folder names and paths to where input/output data/code will the read from and written to", CYAN_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT );	
	if(SCAN_TYPE == EXPERIMENTAL)
	{
		sprintf(INPUT_FOLDER_SET, "%s//%s//%s//%s//%s//%s", PHANTOM_NAME, EXPERIMENTAL_FOLDER, RUN_DATE, RUN_NUMBER, PROJECTION_LINKS_FOLDER, PREPROCESS_DATE);
		sprintf(OUTPUT_FOLDER_SET, "%s//%s//%s//%s//%s//%s", PHANTOM_NAME, EXPERIMENTAL_FOLDER, RUN_DATE, RUN_NUMBER, PROJECTION_LINKS_FOLDER, PREPROCESS_DATE);
		
	}
	else if(SCAN_TYPE == SIMULATED_G)
	{
		sprintf(INPUT_FOLDER_SET, "%s//%s//%s%s//%s//%s//%s", PHANTOM_NAME, SIMULATED_FOLDER, GEANT4_DIR_PREFIX, RUN_DATE, RUN_NUMBER, PROJECTION_LINKS_FOLDER, PREPROCESS_DATE);
		sprintf(OUTPUT_FOLDER_SET, "%s//%s//%s%s//%s//%s//%s", PHANTOM_NAME, SIMULATED_FOLDER, GEANT4_DIR_PREFIX, RUN_DATE, RUN_NUMBER, PROJECTION_LINKS_FOLDER, PREPROCESS_DATE);
		
	}
	else if(SCAN_TYPE == SIMULATED_T)
	{
		sprintf(INPUT_FOLDER_SET, "%s//%s//%s%//%s//%s//%s", PHANTOM_NAME, SIMULATED_FOLDER, TOPAS_DIR_PREFIX, RUN_DATE, RUN_NUMBER, PROJECTION_LINKS_FOLDER, PREPROCESS_DATE);
		sprintf(OUTPUT_FOLDER_SET, "%s//%s//%s%//%s//%s//%s", PHANTOM_NAME, SIMULATED_FOLDER, TOPAS_DIR_PREFIX, RUN_DATE, RUN_NUMBER, PROJECTION_LINKS_FOLDER, PREPROCESS_DATE);		
	}
	else
	{
		print_section_header("ERROR", MAJOR_SECTION_SEPARATOR, WHITE_TEXT, WHITE_TEXT, RED_BACKGROUND, DONT_UNDERLINE_TEXT);
		print_section_exit("Invalid scan type specified", SECTION_EXIT_CSTRING, RED_TEXT, RED_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT);
		exit_program_if(true);
	}	
	sprintf(LOCAL_INPUT_DATA_PATH, "%s//%s", INPUT_DIRECTORY_SET, INPUT_FOLDER);
	sprintf(LOCAL_OUTPUT_DATA_PATH, "%s//%s", OUTPUT_DIRECTORY_SET, OUTPUT_FOLDER_UNIQUE);
	sprintf(GLOBAL_INPUT_DATA_PATH, "%s//%s", PCT_ORG_DATA_DIR_SET, INPUT_FOLDER);
	sprintf(GLOBAL_OUTPUT_DATA_PATH, "%s//%s", CURRENT_RECON_DIR, OUTPUT_FOLDER_UNIQUE);
	//sprintf(GLOBAL_OUTPUT_FOLDER_DESTINATION, "%s//%s", CURRENT_RECON_DIR, OUTPUT_FOLDER);	
	sprintf(LOCAL_EXECUTION_LOG_PATH, "%s//%s.csv", OUTPUT_DIRECTORY_SET, EXECUTION_LOG_BASENAME);	
	sprintf(GLOBAL_EXECUTION_LOG_PATH, "%s//%s.csv", CURRENT_RECON_DIR, EXECUTION_LOG_BASENAME );
	sprintf(LOCAL_EXECUTION_INFO_PATH, "%s//%s//%s.txt", OUTPUT_DIRECTORY_SET, OUTPUT_FOLDER_UNIQUE, EXECUTION_LOG_BASENAME);
	sprintf(GLOBAL_EXECUTION_INFO_PATH, "%s//%s//%s.txt", CURRENT_RECON_DIR, OUTPUT_FOLDER_UNIQUE, EXECUTION_LOG_BASENAME);
	sprintf(LOCAL_TV_MEASUREMENTS_PATH, "%s//%s//%s.txt", OUTPUT_DIRECTORY_SET, OUTPUT_FOLDER_UNIQUE, TV_MEASUREMENTS_FILENAME);	
	sprintf(GLOBAL_TV_MEASUREMENTS_PATH, "%s//%s//%s.txt", CURRENT_RECON_DIR, OUTPUT_FOLDER_UNIQUE, TV_MEASUREMENTS_FILENAME);	
	sprintf(INPUT_ITERATE_PATH, "%s//%s//%s", OUTPUT_DIRECTORY_SET, OUTPUT_FOLDER_UNIQUE, INPUT_ITERATE_FILENAME );
}
void set_source_code_paths()
{
	print_colored_text("Setting paths to the executed code and its copies written to the current compute nodes and the network-attached storage device ", CYAN_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT );	
	switch(CODE_SOURCE)
	{
		case LOCAL:			sprintf(EXECUTED_CODE_DIR, "%s", TARDIS_RCODE_PARENT_SET);		break;
		case GLOBAL:		sprintf(EXECUTED_CODE_DIR, "%s", PCT_RCODE_PARENT_SET);			break;
		case USER_HOME:		sprintf(EXECUTED_CODE_DIR, "%s", SHARED_RCODE_PARENT_SET);		break;
		case GROUP_HOME:	sprintf(EXECUTED_CODE_DIR, "%s", MY_RCODE_PARENT_SET);			break;
	}
	if(USE_GIT_CODE)
		sprintf(EXECUTED_CODE_DIR, "%s//%s", EXECUTED_CODE_DIR, COMMON_GIT_CODE_SUBDIRECTORY);
	sprintf(LOCAL_OUTPUT_CODE_DIR, "%s//%s", LOCAL_OUTPUT_DATA_PATH, SRC_CODE_FOLDER);
	sprintf(GLOBAL_OUTPUT_CODE_DIR, "%s//%s", GLOBAL_OUTPUT_DATA_PATH, SRC_CODE_FOLDER);
	
	sprintf(EXECUTED_SRC_CODE_PATH, "%s//%s", EXECUTED_CODE_DIR, SRC_CODE_FOLDER);
	sprintf(EXECUTED_INCLUDE_CODE_PATH, "%s//%s", EXECUTED_CODE_DIR, INCLUDE_CODE_FOLDER);
	sprintf(LOCAL_OUTPUT_SRC_CODE_PATH, "%s//%s", LOCAL_OUTPUT_DATA_PATH, SRC_CODE_FOLDER);
	sprintf(LOCAL_OUTPUT_INCLUDE_CODE_PATH, "%s//%s", LOCAL_OUTPUT_DATA_PATH, INCLUDE_CODE_FOLDER);
	sprintf(GLOBAL_OUTPUT_SRC_CODE_PATH, "%s//%s", GLOBAL_OUTPUT_DATA_PATH, SRC_CODE_FOLDER);
	sprintf(GLOBAL_OUTPUT_INCLUDE_CODE_PATH, "%s//%s", GLOBAL_OUTPUT_DATA_PATH, INCLUDE_CODE_FOLDER);	
}
void set_ssh_server_login_strings()
{
	sprintf(KODIAK_SSH_LOGIN, "%s@%s:", USE_KODIAK_USERNAME, KODIAK_SERVER_NAME);
	sprintf(WHARTNELL_SSH_LOGIN, "%s@%s:", USE_TARDIS_USERNAME, ECSN1_SERVER_NAME);
	sprintf(PTROUGHTON_SSH_LOGIN, "%s@%s:", USE_TARDIS_USERNAME, ECSN2_SERVER_NAME);
	sprintf(JPERTWEE_SSH_LOGIN, "%s@%s:", USE_TARDIS_USERNAME, ECSN3_SERVER_NAME);
	sprintf(TBAKER_SSH_LOGIN, "%s@%s:", USE_TARDIS_USERNAME, ECSN4_SERVER_NAME);
	sprintf(PDAVISON_SSH_LOGIN, "%s@%s:", USE_TARDIS_USERNAME, ECSN5_SERVER_NAME);
	sprintf(WS1_SSH_LOGIN, "%s@%s:", USE_TARDIS_USERNAME, WS1_SERVER_NAME);
	sprintf(WS2_SSH_LOGIN, "%s@%s:", USE_TARDIS_USERNAME, WS2_SERVER_NAME);
}
void set_enum_strings()
{
	print_colored_text( "Assigning strings corresponding to enum variable values...", CYAN_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );	
	switch( SCAN_TYPE )
	{
		case EXPERIMENTAL:				sprintf(SCAN_TYPE_CSTRING, "%s", EXPERIMENTAL_CSTRING);						break;
		case SIMULATED_G:				sprintf(SCAN_TYPE_CSTRING, "%s", SIMULATED_G_CSTRING);						break;
		case SIMULATED_T:				sprintf(SCAN_TYPE_CSTRING, "%s", SIMULATED_T_CSTRING);						break;		
	}
	switch( SINOGRAM_FILTER )
	{
		case RAM_LAK:					sprintf(SINOGRAM_FILTER_CSTRING, "%s", RAM_LAK_CSTRING);						break;
		case SHEPP_LOGAN:				sprintf(SINOGRAM_FILTER_CSTRING, "%s", SHEPP_LOGAN_CSTRING);					break;
		case UNFILTERED:				sprintf(SINOGRAM_FILTER_CSTRING, "%s", UNFILTERED_CSTRING);					break;		
	}
	switch( FBP_FILTER )
	{
		case NO_FILTER:					sprintf(FBP_FILTER_CSTRING, "%s", NO_FILTER_CSTRING);							break;
		case MEDIAN:					sprintf(FBP_FILTER_CSTRING, "%s", MEDIAN_FILTER_CSTRING);						break;
		case AVERAGE:					sprintf(FBP_FILTER_CSTRING, "%s", AVERAGE_FILTER_CSTRING);					break;
		case MED_2_AVG:					sprintf(FBP_FILTER_CSTRING, "%s", MED_2_AVG_FILTER_CSTRING);					break;
		case AVG_2_MED:					sprintf(FBP_FILTER_CSTRING, "%s", AVG_2_MED_FILTER_CSTRING);					break;		
	}
	switch( HULL_FILTER )
	{
		case NO_FILTER:					sprintf(HULL_FILTER_CSTRING, "%s", NO_FILTER_CSTRING);						break;
		case MEDIAN:					sprintf(HULL_FILTER_CSTRING, "%s", MEDIAN_FILTER_CSTRING);					break;
		case AVERAGE:					sprintf(HULL_FILTER_CSTRING, "%s", AVERAGE_FILTER_CSTRING);					break;
		case MED_2_AVG:					sprintf(HULL_FILTER_CSTRING, "%s", MED_2_AVG_FILTER_CSTRING);					break;
		case AVG_2_MED:					sprintf(HULL_FILTER_CSTRING, "%s", AVG_2_MED_FILTER_CSTRING);					break;		
	}
	switch( X_0_FILTER )
	{
		case NO_FILTER:					sprintf(X_0_FILTER_CSTRING, "%s", NO_FILTER_CSTRING);							break;
		case MEDIAN:					sprintf(X_0_FILTER_CSTRING, "%s", MEDIAN_FILTER_CSTRING);						break;
		case AVERAGE:					sprintf(X_0_FILTER_CSTRING, "%s", AVERAGE_FILTER_CSTRING);					break;
		case MED_2_AVG:					sprintf(X_0_FILTER_CSTRING, "%s", MED_2_AVG_FILTER_CSTRING);					break;
		case AVG_2_MED:					sprintf(X_0_FILTER_CSTRING, "%s", AVG_2_MED_FILTER_CSTRING);					break;		
	}
	switch( ENDPOINTS_HULL )
	{
		case SC_HULL:					sprintf(ENDPOINTS_HULL_CSTRING, "%s", SC_HULL_CSTRING);						break;
		case MSC_HULL:					sprintf(ENDPOINTS_HULL_CSTRING, "%s", MSC_HULL_CSTRING);						break;
		case SM_HULL:					sprintf(ENDPOINTS_HULL_CSTRING, "%s", SM_HULL_CSTRING);						break;
		case FBP_HULL:					sprintf(ENDPOINTS_HULL_CSTRING, "%s", FBP_HULL_CSTRING);						break;		
	}
	switch( ENDPOINTS_ALG )
	{
		case YES_BOOL:					sprintf(ENDPOINTS_ALG_CSTRING, "%s", BOOL_CSTRING);							break;
		case NO_BOOL:					sprintf(ENDPOINTS_ALG_CSTRING, "%s", NO_BOOL_CSTRING);						break;
	}
	switch( ENDPOINTS_TX_MODE )
	{
		case FULL_TX:					sprintf(ENDPOINTS_TX_MODE_CSTRING, "%s", FULL_TX_CSTRING);					break;
		case PARTIAL_TX:				sprintf(ENDPOINTS_TX_MODE_CSTRING, "%s", PARTIAL_TX_CSTRING);					break;
		case PARTIAL_TX_PREALLOCATED:	sprintf(ENDPOINTS_TX_MODE_CSTRING, "%s", PARTIAL_TX_PREALLOCATED_CSTRING);	break;		
	}
	switch( MLP_ALGORITHM )
	{
		case TABULATED:					sprintf(MLP_ALGORITHM_CSTRING, "%s", TABULATED_CSTRING);						break;
		case STANDARD:					sprintf(MLP_ALGORITHM_CSTRING, "%s", STANDARD_CSTRING);						break;
	}
	switch( X_0 )
	{ 
		case X_HULL:					sprintf(X_0_CSTRING, "%s", HULL_CSTRING);										break;
		case FBP_IMAGE:					sprintf(X_0_CSTRING, "%s", FBP_IMAGE_CSTRING);								break;
		case HYBRID:					sprintf(X_0_CSTRING, "%s", HYBRID_CSTRING);									break;
		case ZEROS:						sprintf(X_0_CSTRING, "%s", ZEROS_CSTRING);									break;
		case IMPORT:					sprintf(X_0_CSTRING, "%s", IMPORT_CSTRING);									break;
	}
	switch( PROJECTION_ALGORITHM )
	{
		case ART:						sprintf(PROJECTION_ALGORITHM_CSTRING, "%s", ART_CSTRING);						break;
		case SART:						sprintf(PROJECTION_ALGORITHM_CSTRING, "%s", SART_CSTRING);					break;
		case DROP:						sprintf(PROJECTION_ALGORITHM_CSTRING, "%s", DROP_CSTRING);					break;
		case BIP:						sprintf(PROJECTION_ALGORITHM_CSTRING, "%s", BIP_CSTRING);						break;
		case SAP:						sprintf(PROJECTION_ALGORITHM_CSTRING, "%s", SAP_CSTRING);						break;
		case ROBUSTA:					sprintf(PROJECTION_ALGORITHM_CSTRING, "%s", ROBUSTA_CSTRING);					break;
		case ROBUSTB:					sprintf(PROJECTION_ALGORITHM_CSTRING, "%s", ROBUSTB_CSTRING);					break;
	}
	switch( RECON_TX_MODE )
	{
		case FULL_TX:					sprintf(RECON_TX_MODE_CSTRING, "%s", FULL_TX_CSTRING);						break;
		case PARTIAL_TX:				sprintf(RECON_TX_MODE_CSTRING, "%s", PARTIAL_TX_CSTRING);						break;
		case PARTIAL_TX_PREALLOCATED:	sprintf(RECON_TX_MODE_CSTRING, "%s", PARTIAL_TX_PREALLOCATED_CSTRING);		break;
	}
	switch( ROBUST_METHOD )
	{
		case OLS:						sprintf(ROBUST_METHOD_CSTRING, "%s", OLS_CSTRING);							break;
		case TLS:						sprintf(ROBUST_METHOD_CSTRING, "%s", TLS_CSTRING);							break;
		case TIKHONOV:					sprintf(ROBUST_METHOD_CSTRING, "%s", TIKHONOV_CSTRING);						break;
		case RIDGE:						sprintf(ROBUST_METHOD_CSTRING, "%s", RIDGE_CSTRING);							break;
		case MINMIN:					sprintf(ROBUST_METHOD_CSTRING, "%s", MINMIN_CSTRING);							break;
		case MINMAX:					sprintf(ROBUST_METHOD_CSTRING, "%s", MINMAX_CSTRING);							break;
	}
	switch( S_CURVE )
	{
		case SIGMOID:					sprintf(S_CURVE_CSTRING, "%s", SIGMOID_CSTRING);								break;
		case TANH:						sprintf(S_CURVE_CSTRING, "%s", TANH_CSTRING);									break;
		case ATAN:						sprintf(S_CURVE_CSTRING, "%s", ATAN_CSTRING);									break;
		case ERF:						sprintf(S_CURVE_CSTRING, "%s", ERF_CSTRING);									break;
		case LIN_OVER_ROOT:				sprintf(S_CURVE_CSTRING, "%s", LIN_OVER_ROOT_CSTRING);						break;		
	}

}
void set_procedure_on_off_string(const bool procedure_on_off, char* procedure_on_off_string)
{
	if(procedure_on_off)
		sprintf(procedure_on_off_string, "%s", ON_CSTRING);
	else
		sprintf(procedure_on_off_string, "%s", OFF_CSTRING);	
}
void set_procedures_on_off_strings()
{
	print_colored_text( "Assigning strings corresponding to boolean variable values...", CYAN_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );	
	//	
	if(SAMPLE_STD_DEV)
		sprintf(SAMPLE_STD_DEV_CSTRING, "%s", ON_CSTRING);
	else
		sprintf(SAMPLE_STD_DEV_CSTRING, "%s", OFF_CSTRING);
	//	
	if(AVG_FILTER_FBP)
		sprintf(AVG_FILTER_FBP_CSTRING, "%s", ON_CSTRING);
	else
		sprintf(AVG_FILTER_FBP_CSTRING, "%s", OFF_CSTRING);
	//
	if(AVG_FILTER_HULL)
		sprintf(AVG_FILTER_HULL_CSTRING, "%s", ON_CSTRING);
	else
		sprintf(AVG_FILTER_HULL_CSTRING, "%s", OFF_CSTRING);
	//
	if(AVG_FILTER_X_0)
		sprintf(AVG_FILTER_X_0_CSTRING, "%s", ON_CSTRING);
	else
		sprintf(AVG_FILTER_X_0_CSTRING, "%s", OFF_CSTRING);
	//
	if(MEDIAN_FILTER_FBP)
		sprintf(MEDIAN_FILTER_FBP_CSTRING, "%s", ON_CSTRING);
	else
		sprintf(MEDIAN_FILTER_FBP_CSTRING, "%s", OFF_CSTRING);
	//
	if(MEDIAN_FILTER_HULL)
		sprintf(MEDIAN_FILTER_HULL_CSTRING, "%s", ON_CSTRING);
	else
		sprintf(MEDIAN_FILTER_HULL_CSTRING, "%s", OFF_CSTRING);
	//
	if(MEDIAN_FILTER_X_0)
		sprintf(MEDIAN_FILTER_X_0_CSTRING, "%s", ON_CSTRING);
	else
		sprintf(MEDIAN_FILTER_X_0_CSTRING, "%s", OFF_CSTRING);
	//
	if(IGNORE_SHORT_MLP == 1)
		sprintf(IGNORE_SHORT_MLP_CSTRING, "%s", ON_CSTRING);
	else
		sprintf(IGNORE_SHORT_MLP_CSTRING, "%s", OFF_CSTRING);
	//
	if(BOUND_IMAGE == 1)
		sprintf(BOUND_IMAGE_CSTRING, "%s", ON_CSTRING);
	else
		sprintf(BOUND_IMAGE_CSTRING, "%s", OFF_CSTRING);	
	//
	if(IDENTIFY_X_0_AIR)
		sprintf(IDENTIFY_X_0_AIR_CSTRING, "%s", ON_CSTRING);
	else
		sprintf(IDENTIFY_X_0_AIR_CSTRING, "%s", OFF_CSTRING);	
	//
	if(IDENTIFY_X_N_AIR)
		sprintf(IDENTIFY_X_N_AIR_CSTRING, "%s", ON_CSTRING);
	else
		sprintf(IDENTIFY_X_N_AIR_CSTRING, "%s", OFF_CSTRING);	
	//
	if(S_CURVE_ON == 1)
		sprintf(S_CURVE_ON_CSTRING, "%s", ON_CSTRING);
	else
		sprintf(S_CURVE_ON_CSTRING, "%s", OFF_CSTRING);
	//
	if(DUAL_SIDED_S_CURVE == 1)
		sprintf(DUAL_SIDED_S_CURVE_CSTRING, "%s", ON_CSTRING);
	else
		sprintf(DUAL_SIDED_S_CURVE_CSTRING, "%s", OFF_CSTRING);
	//
	if(TVS_ON == 1)
		sprintf(TVS_ON_CSTRING, "%s", ON_CSTRING);
	else
		sprintf(TVS_ON_CSTRING, "%s", OFF_CSTRING);
	//
	if(TVS_FIRST == 1)
		sprintf(TVS_FIRST_CSTRING, "%s", ON_CSTRING);
	else
		sprintf(TVS_FIRST_CSTRING, "%s", OFF_CSTRING);
	//
	if(TVS_PARALLEL == 1)
		sprintf(TVS_PARALLEL_CSTRING, "%s", ON_CSTRING);
	else
		sprintf(TVS_PARALLEL_CSTRING, "%s", OFF_CSTRING);
	//
	if(TVS_CONDITIONED == 1)
		sprintf(TVS_CONDITIONED_CSTRING, "%s", ON_CSTRING);
	else
		sprintf(TVS_CONDITIONED_CSTRING, "%s", OFF_CSTRING);		
}
void string_assigments()
{
	print_section_header( "Setting parameter dependent string variables", MINOR_SECTION_SEPARATOR, LIGHT_CYAN_TEXT, YELLOW_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT );
	set_ssh_server_login_strings();
	set_enum_strings();
	set_procedures_on_off_strings();
	current_MMDDYYYY(EXECUTION_DATE);
	current_YY_MM_DD(EXECUTION_YY_MM_DD);	
	current_date_time_formatted(EXECUTION_DATE_TIME);
	//sprintf( INPUT_ITERATE_PATH, "%s%s/%s", OUTPUT_DIRECTORY, OUTPUT_FOLDER_UNIQUE, INPUT_ITERATE_FILENAME );
	//std::string str("execution_log.txt, FBP_median_filtered.txt, hull.txt, TV_measurements.txt, x_0.txt, x_10.txt, x_11.txt, x_12.txt, x_1.txt, x_2.txt, x_3.txt, x_4.txt, x_5.txt, x_6.txt, x_7.txt, x_8.txt, x_9.txt");
}
void IO_setup()
{
	print_section_header( "Determining the current user/group, source of the code and the compute node its executed on, and establishing the appropriate input/output directory/file names and paths", MINOR_SECTION_SEPARATOR, LIGHT_CYAN_TEXT, YELLOW_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT );
	if(!USING_RSYNC)
		check_4_missing_input();
	set_compute_node();
	set_user_strings();
	set_compute_system_directories();
	set_git_branch_info();
	set_and_make_output_folder();	
	set_IO_folder_names();
	set_source_code_paths();
}
void print_paths()
{	
	print_section_header("Permanent paths to the shared and private data/code directories on the networked-attached storage device and the compute nodes", MINOR_SECTION_SEPARATOR, LIGHT_BLUE_TEXT, YELLOW_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT );
	print_labeled_value("USERNAME =", USERNAME, GREEN_TEXT, LIGHT_PURPLE_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT);
	print_labeled_value("COMMON_ORG_DATA_SUBDIRECTORY =", COMMON_ORG_DATA_SUBDIRECTORY, GREEN_TEXT, LIGHT_PURPLE_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT);
	print_labeled_value("COMMON_RECON_DATA_SUBDIRECTORY =", COMMON_RECON_DATA_SUBDIRECTORY, GREEN_TEXT, LIGHT_PURPLE_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT);
	print_labeled_value("COMMON_RCODE_SUBDIRECTORY =", COMMON_RCODE_SUBDIRECTORY, GREEN_TEXT, LIGHT_PURPLE_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT);
	print_labeled_value("COMMON_GIT_CODE_SUBDIRECTORY =", COMMON_GIT_CODE_SUBDIRECTORY, GREEN_TEXT, LIGHT_PURPLE_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT);
	print_labeled_value("PCT_DATA_DIR_SET =", PCT_DATA_DIR_SET, GREEN_TEXT, LIGHT_PURPLE_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT);
	print_labeled_value("PCT_ORG_DATA_DIR_SET =", PCT_ORG_DATA_DIR_SET, GREEN_TEXT, LIGHT_PURPLE_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT);
	print_labeled_value("PCT_RECON_DIR_SET =", PCT_RECON_DIR_SET, GREEN_TEXT, LIGHT_PURPLE_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT);
	print_labeled_value("PCT_CODE_PARENT_SET =", PCT_CODE_PARENT_SET, GREEN_TEXT, LIGHT_PURPLE_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT);
	print_labeled_value("PCT_RCODE_PARENT_SET =", PCT_RCODE_PARENT_SET, GREEN_TEXT, LIGHT_PURPLE_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT);
	print_labeled_value("PCT_GIT_RCODE_PARENT_SET =", PCT_GIT_RCODE_PARENT_SET, GREEN_TEXT, LIGHT_PURPLE_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT);
	print_labeled_value("TARDIS_DATA_DIR_SET =", TARDIS_DATA_DIR_SET, GREEN_TEXT, LIGHT_PURPLE_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT);
	print_labeled_value("TARDIS_ORG_DATA_DIR_SET =", TARDIS_ORG_DATA_DIR_SET, GREEN_TEXT, LIGHT_PURPLE_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT);
	print_labeled_value("TARDIS_RECON_DIR_SET =", TARDIS_RECON_DIR_SET, GREEN_TEXT, LIGHT_PURPLE_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT);
	print_labeled_value("TARDIS_CODE_PARENT_SET =", TARDIS_CODE_PARENT_SET, GREEN_TEXT, LIGHT_PURPLE_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT);
	print_labeled_value("TARDIS_RCODE_PARENT_SET =", TARDIS_RCODE_PARENT_SET, GREEN_TEXT, LIGHT_PURPLE_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT);
	print_labeled_value("TARDIS_GIT_RCODE_PARENT_SET =", TARDIS_GIT_RCODE_PARENT_SET, GREEN_TEXT, LIGHT_PURPLE_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT);
	print_labeled_value("SHARED_HOME_DIR_SET =", SHARED_HOME_DIR_SET, GREEN_TEXT, LIGHT_PURPLE_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT);
	print_labeled_value("SHARED_DATA_DIR_SET =", SHARED_DATA_DIR_SET, GREEN_TEXT, LIGHT_PURPLE_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT);
	print_labeled_value("SHARED_ORG_DATA_DIR_SET =", SHARED_ORG_DATA_DIR_SET, GREEN_TEXT, LIGHT_PURPLE_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT);
	print_labeled_value("SHARED_RECON_DIR_SET =", SHARED_RECON_DIR_SET, GREEN_TEXT, LIGHT_PURPLE_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT);
	print_labeled_value("SHARED_CODE_PARENT_SET =", SHARED_CODE_PARENT_SET, GREEN_TEXT, LIGHT_PURPLE_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT);
	print_labeled_value("SHARED_RCODE_PARENT_SET =", SHARED_RCODE_PARENT_SET, GREEN_TEXT, LIGHT_PURPLE_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT);
	print_labeled_value("SHARED_GIT_RCODE_PARENT_SET =", SHARED_GIT_RCODE_PARENT_SET, GREEN_TEXT, LIGHT_PURPLE_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT);
	print_labeled_value("MY_HOME_DIR_SET =", MY_HOME_DIR_SET, GREEN_TEXT, LIGHT_PURPLE_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT);
	print_labeled_value("MY_DATA_DIR_SET =", MY_DATA_DIR_SET, GREEN_TEXT, LIGHT_PURPLE_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT);
	print_labeled_value("MY_ORG_DATA_DIR_SET =", MY_ORG_DATA_DIR_SET, GREEN_TEXT, LIGHT_PURPLE_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT);
	print_labeled_value("MY_RECON_DIR_SET =", MY_RECON_DIR_SET, GREEN_TEXT, LIGHT_PURPLE_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT);
	print_labeled_value("MY_CODE_PARENT_SET =", MY_CODE_PARENT_SET, GREEN_TEXT, LIGHT_PURPLE_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT);
	print_labeled_value("MY_RCODE_PARENT_SET =", MY_RCODE_PARENT_SET, GREEN_TEXT, LIGHT_PURPLE_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT);
	print_labeled_value("MY_GIT_RCODE_PARENT_SET =", MY_GIT_RCODE_PARENT_SET, GREEN_TEXT, LIGHT_PURPLE_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT);
	
	print_section_header("Paths to the data/code directories specific to the current execution on the networked-attached storage device and the compute nodes", MINOR_SECTION_SEPARATOR, LIGHT_BLUE_TEXT, YELLOW_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT );
	print_colored_text( "Parent directory/folder of the input/output data for the current execution and compute node...", CYAN_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );		
	print_labeled_value("INPUT_FOLDER_SET =", INPUT_FOLDER_SET, GREEN_TEXT, LIGHT_PURPLE_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT);
	print_labeled_value("OUTPUT_FOLDER_SET =", OUTPUT_FOLDER_SET, GREEN_TEXT, LIGHT_PURPLE_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT);
	print_labeled_value("INPUT_DIRECTORY_SET =", INPUT_DIRECTORY_SET, GREEN_TEXT, LIGHT_PURPLE_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT);
	print_labeled_value("OUTPUT_DIRECTORY_SET =", OUTPUT_DIRECTORY_SET, GREEN_TEXT, LIGHT_PURPLE_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT);
	
	print_colored_text( "Paths to the input/output data on the compute node and the network-attached storage device...", CYAN_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );		
	print_labeled_value("LOCAL_INPUT_DATA_PATH =", LOCAL_INPUT_DATA_PATH, GREEN_TEXT, LIGHT_PURPLE_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT);
	print_labeled_value("LOCAL_OUTPUT_DATA_PATH =", LOCAL_OUTPUT_DATA_PATH, GREEN_TEXT, LIGHT_PURPLE_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT);
	print_labeled_value("GLOBAL_INPUT_DATA_PATH =", GLOBAL_INPUT_DATA_PATH, GREEN_TEXT, LIGHT_PURPLE_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT);
	print_labeled_value("GLOBAL_OUTPUT_DATA_PATH =", GLOBAL_OUTPUT_DATA_PATH, GREEN_TEXT, LIGHT_PURPLE_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT);
	
	print_section_header("Paths to the data/code files specific to the current execution on the networked-attached storage device and the compute nodes", MINOR_SECTION_SEPARATOR, LIGHT_BLUE_TEXT, YELLOW_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT );
	print_colored_text( "Paths to the TXT/CSV execution logs on the compute node and the network-attached storage device...", CYAN_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );		
	print_labeled_value("GLOBAL_EXECUTION_LOG_PATH =", GLOBAL_EXECUTION_LOG_PATH, GREEN_TEXT, LIGHT_PURPLE_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT);
	print_labeled_value("LOCAL_EXECUTION_LOG_PATH =", LOCAL_EXECUTION_LOG_PATH, GREEN_TEXT, LIGHT_PURPLE_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT);
	print_labeled_value("GLOBAL_EXECUTION_INFO_PATH =", GLOBAL_EXECUTION_INFO_PATH, GREEN_TEXT, LIGHT_PURPLE_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT);
	print_labeled_value("LOCAL_EXECUTION_INFO_PATH =", LOCAL_EXECUTION_INFO_PATH, GREEN_TEXT, LIGHT_PURPLE_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT);
	print_labeled_value("LOCAL_TV_MEASUREMENTS_PATH =", LOCAL_TV_MEASUREMENTS_PATH, GREEN_TEXT, LIGHT_PURPLE_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT);
	print_labeled_value("GLOBAL_TV_MEASUREMENTS_PATH =", GLOBAL_TV_MEASUREMENTS_PATH, GREEN_TEXT, LIGHT_PURPLE_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT);

	print_colored_text( "Paths to the currently executed code on the compute node and the destination for its copies on the network-attached storage device...", CYAN_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );		
	print_labeled_value("EXECUTED_SRC_CODE_PATH =", EXECUTED_SRC_CODE_PATH, GREEN_TEXT, LIGHT_PURPLE_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT);
	print_labeled_value("EXECUTED_INCLUDE_CODE_PATH =", EXECUTED_INCLUDE_CODE_PATH, GREEN_TEXT, LIGHT_PURPLE_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT);
	print_labeled_value("LOCAL_OUTPUT_SRC_CODE_PATH =", LOCAL_OUTPUT_SRC_CODE_PATH, GREEN_TEXT, LIGHT_PURPLE_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT);	
	print_labeled_value("LOCAL_OUTPUT_INCLUDE_CODE_PATH =", LOCAL_OUTPUT_INCLUDE_CODE_PATH, GREEN_TEXT, LIGHT_PURPLE_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT);
	print_labeled_value("GLOBAL_OUTPUT_SRC_CODE_PATH =", GLOBAL_OUTPUT_SRC_CODE_PATH, GREEN_TEXT, LIGHT_PURPLE_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT);
	print_labeled_value("GLOBAL_OUTPUT_INCLUDE_CODE_PATH =", GLOBAL_OUTPUT_INCLUDE_CODE_PATH, GREEN_TEXT, LIGHT_PURPLE_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT);	
	print_labeled_value("INPUT_ITERATE_PATH =", INPUT_ITERATE_PATH, GREEN_TEXT, LIGHT_PURPLE_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT);	
}
void program_startup_tasks()
{
	/********************************************************************************************************************************************************/
	/* Perform program initializations, assignments, and other program startup tasks and start the execution timing clock									*/
	/********************************************************************************************************************************************************/
	print_section_header( "Determining version of reconstruction code, verifying input data, querying current compute node, and assigning output data directory", MAJOR_SECTION_SEPARATOR, LIGHT_GREEN_TEXT, YELLOW_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT );
	timer( START, begin_program, "for entire program");
	string_assigments();
	IO_setup();	
	if(PRINT_ALL_PATHS)
		print_paths();
}
/***********************************************************************************************************************************************************************************************************************/
/************************************************************************ Program exit/output data management tasks ****************************************************************************************************/
/***********************************************************************************************************************************************************************************************************************/
void define_execution_log_order()
{
	// Generate mapping of all possible keys to integer ID so key can be used to control switch statement
	EXECUTION_LOG_SWITCHMAP.insert( std::pair<std::string,unsigned int>(std::string("INPUT_DIRECTORY"), 1));
	EXECUTION_LOG_SWITCHMAP.insert( std::pair<std::string,unsigned int>(std::string("OUTPUT_DIRECTORY"), 2));
	EXECUTION_LOG_SWITCHMAP.insert( std::pair<std::string,unsigned int>(std::string("INPUT_FOLDER"), 3));
	EXECUTION_LOG_SWITCHMAP.insert( std::pair<std::string,unsigned int>(std::string("OUTPUT_FOLDER"), 4));
	EXECUTION_LOG_SWITCHMAP.insert( std::pair<std::string,unsigned int>(std::string("PROJECTION_DATA_BASENAME"), 5));
	EXECUTION_LOG_SWITCHMAP.insert( std::pair<std::string,unsigned int>(std::string("PROJECTION_DATA_EXTENSION"), 6));
	EXECUTION_LOG_SWITCHMAP.insert( std::pair<std::string,unsigned int>(std::string("GANTRY_ANGLES"), 7));
	EXECUTION_LOG_SWITCHMAP.insert( std::pair<std::string,unsigned int>(std::string("NUM_SCANS"), 8));
	EXECUTION_LOG_SWITCHMAP.insert( std::pair<std::string,unsigned int>(std::string("SSD_T_SIZE"), 9));
	EXECUTION_LOG_SWITCHMAP.insert( std::pair<std::string,unsigned int>(std::string("SSD_V_SIZE"), 10));
	EXECUTION_LOG_SWITCHMAP.insert( std::pair<std::string,unsigned int>(std::string("T_SHIFT"), 11));
	EXECUTION_LOG_SWITCHMAP.insert( std::pair<std::string,unsigned int>(std::string("U_SHIFT"), 12));
	EXECUTION_LOG_SWITCHMAP.insert( std::pair<std::string,unsigned int>(std::string("T_BIN_SIZE"), 13));
	EXECUTION_LOG_SWITCHMAP.insert( std::pair<std::string,unsigned int>(std::string("T_BINS"), 14));
	EXECUTION_LOG_SWITCHMAP.insert( std::pair<std::string,unsigned int>(std::string("V_BIN_SIZE"), 15));
	EXECUTION_LOG_SWITCHMAP.insert( std::pair<std::string,unsigned int>(std::string("V_BINS"), 16));
	EXECUTION_LOG_SWITCHMAP.insert( std::pair<std::string,unsigned int>(std::string("ANGULAR_BIN_SIZE"), 17));
	EXECUTION_LOG_SWITCHMAP.insert( std::pair<std::string,unsigned int>(std::string("SIGMAS_TO_KEEP"), 18));
	EXECUTION_LOG_SWITCHMAP.insert( std::pair<std::string,unsigned int>(std::string("RECON_CYL_RADIUS"), 19));
	EXECUTION_LOG_SWITCHMAP.insert( std::pair<std::string,unsigned int>(std::string("RECON_CYL_HEIGHT"), 20));
	EXECUTION_LOG_SWITCHMAP.insert( std::pair<std::string,unsigned int>(std::string("IMAGE_WIDTH"), 21));
	EXECUTION_LOG_SWITCHMAP.insert( std::pair<std::string,unsigned int>(std::string("IMAGE_HEIGHT"), 22));
	EXECUTION_LOG_SWITCHMAP.insert( std::pair<std::string,unsigned int>(std::string("IMAGE_THICKNESS"), 23));
	EXECUTION_LOG_SWITCHMAP.insert( std::pair<std::string,unsigned int>(std::string("COLUMNS"), 24));
	EXECUTION_LOG_SWITCHMAP.insert( std::pair<std::string,unsigned int>(std::string("ROWS"), 25));
	EXECUTION_LOG_SWITCHMAP.insert( std::pair<std::string,unsigned int>(std::string("SLICES"), 26));
	EXECUTION_LOG_SWITCHMAP.insert( std::pair<std::string,unsigned int>(std::string("VOXEL_WIDTH"), 27));
	EXECUTION_LOG_SWITCHMAP.insert( std::pair<std::string,unsigned int>(std::string("VOXEL_HEIGHT"), 28));
	EXECUTION_LOG_SWITCHMAP.insert( std::pair<std::string,unsigned int>(std::string("VOXEL_THICKNESS"), 29));
	EXECUTION_LOG_SWITCHMAP.insert( std::pair<std::string,unsigned int>(std::string("LAMBDA"), 30));
	EXECUTION_LOG_SWITCHMAP.insert( std::pair<std::string,unsigned int>(std::string("parameter"), 31));

}
void execution_log_2_txt()
{
	int i = 0;
	print_colored_text("Execution options/parameters written to local execution log at:", CYAN_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );
	print_colored_text(LOCAL_EXECUTION_INFO_PATH, LIGHT_PURPLE_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT );
	
	FILE* execution_log_file = fopen( LOCAL_EXECUTION_INFO_PATH, "w" );
	fprintf(execution_log_file, "Execution Date/Time = %s\n",				EXECUTION_DATE_TIME					);	// 1
	fprintf(execution_log_file, "Execution Host = %s\n",					CURRENT_COMPUTE_NODE_ALIAS				);	// 2	
	fprintf(execution_log_file, "Executed Git Branch : Commit Hash (Commit Date) = %s\n",		GIT_REPO_INFO				);	// 3
	fprintf(execution_log_file, "Executed By = %s\n",						TESTED_BY_CSTRING					);	// 4
	fprintf(execution_log_file, "INPUT_DIRECTORY = %s\n",					INPUT_DIRECTORY						);	// 5
	fprintf(execution_log_file, "INPUT_FOLDER = %s\n",						INPUT_FOLDER						);	// 6
	fprintf(execution_log_file, "OUTPUT_DIRECTORY = %s\n",					OUTPUT_DIRECTORY					);	// 7
	fprintf(execution_log_file, "OUTPUT_FOLDER_UNIQUE = %s\n",				OUTPUT_FOLDER_UNIQUE				);	// 8
	fprintf(execution_log_file, "total_histories = %d\n",					total_histories						);	// 8
	fprintf(execution_log_file, "recon_vol_histories = %d\n",				recon_vol_histories					);	// 8
	fprintf(execution_log_file, "post_cut_histories = %d\n",				post_cut_histories					);	// 8
	fprintf(execution_log_file, "reconstruction_histories = %d\n",			reconstruction_histories			);	// 8	
	fprintf(execution_log_file, "execution_time_data_reads = %6.6lf\n",		execution_time_data_reads			);	// 9
	fprintf(execution_log_file, "execution_time_preprocessing = %6.6lf\n",	execution_time_preprocessing		);	// 10
	fprintf(execution_log_file, "execution_time_endpoints = %6.6lf\n",		execution_time_endpoints			);	// 11
	fprintf(execution_log_file, "execution_time_tables = %6.6lf\n",			execution_time_tables				);	// 12
	fprintf(execution_log_file, "execution_time_init_image = %6.6lf\n",		execution_time_init_image			);	// 13
	fprintf(execution_log_file, "execution_time_DROP = %6.6lf\n",			execution_time_DROP					);	// 14
	for( i = 0; i < execution_times_DROP_iterations.size(); i++ )
		fprintf(execution_log_file, "execution_times_DROP_iterations %d = %6.6lf\n", i, execution_times_DROP_iterations[i]	);	// 15-26																									
	fprintf(execution_log_file, "execution_time_reconstruction = %6.6lf\n",	execution_time_reconstruction		);	// 27
	fprintf(execution_log_file, "execution_time_program = %6.6lf\n",		execution_time_program				);	// 28

	fprintf(execution_log_file, "THREADS_PER_BLOCK = %d\n",					THREADS_PER_BLOCK					);	// 29	
	fprintf(execution_log_file, "ENDPOINTS_PER_BLOCK = %d\n",				ENDPOINTS_PER_BLOCK					);	// 30
	fprintf(execution_log_file, "HISTORIES_PER_BLOCK = %d\n",				HISTORIES_PER_BLOCK					);	// 31
	fprintf(execution_log_file, "ENDPOINTS_PER_THREAD = %d\n",				ENDPOINTS_PER_THREAD				);	// 32
	fprintf(execution_log_file, "HISTORIES_PER_THREAD = %d\n",				HISTORIES_PER_THREAD				);	// 33
	fprintf(execution_log_file, "VOXELS_PER_THREAD = %d\n",					VOXELS_PER_THREAD					);	// 34
	fprintf(execution_log_file, "MAX_GPU_HISTORIES = %d\n",					MAX_GPU_HISTORIES					);	// 35
	fprintf(execution_log_file, "MAX_CUTS_HISTORIES = %d\n",				MAX_CUTS_HISTORIES					);	// 36
	fprintf(execution_log_file, "MAX_ENDPOINTS_HISTORIES = %d\n",			MAX_ENDPOINTS_HISTORIES				);	// 37
				
	fprintf(execution_log_file, "NUM_SCANS = %d\n",							NUM_SCANS							);	// 38
	fprintf(execution_log_file, "GANTRY_ANGLE_INTERVAL = %6.6lf\n",			GANTRY_ANGLE_INTERVAL				);	// 39	
	fprintf(execution_log_file, "SCAN_TYPE = %s\n",							SCAN_TYPE_CSTRING					);	// 40
	fprintf(execution_log_file, "T_SHIFT = %6.6lf\n",						T_SHIFT								);	// 41
	fprintf(execution_log_file, "U_SHIFT = %6.6lf\n",						U_SHIFT								);	// 42
	fprintf(execution_log_file, "V_SHIFT = %6.6lf\n",						V_SHIFT								);	// 43

	fprintf(execution_log_file, "SSD_T_SIZE = %6.6lf\n",					SSD_T_SIZE							);	// 44
	fprintf(execution_log_file, "SSD_V_SIZE = %6.6lf\n",					SSD_V_SIZE							);	// 45
	fprintf(execution_log_file, "T_BIN_SIZE = %6.6lf\n",					T_BIN_SIZE							);	// 46
	fprintf(execution_log_file, "V_BIN_SIZE = %6.6lf\n",					V_BIN_SIZE							);	// 47
	fprintf(execution_log_file, "ANGULAR_BIN_SIZE = %6.6lf\n",				ANGULAR_BIN_SIZE					);	// 48
	fprintf(execution_log_file, "SIGMAS_TO_KEEP = %d\n",					SIGMAS_TO_KEEP						);	// 49
	fprintf(execution_log_file, "SAMPLE_STD_DEV = %s\n",					SAMPLE_STD_DEV_CSTRING				);	// 50
	//fprintf(execution_log_file, "SAMPLE_STD_DEV = %d\n",						SAMPLE_STD_DEV					);	// 50

	fprintf(execution_log_file, "RECON_CYL_RADIUS = %6.6lf\n",				RECON_CYL_RADIUS					);	// 51
	fprintf(execution_log_file, "RECON_CYL_HEIGHT = %6.6lf\n",				RECON_CYL_HEIGHT					);	// 52

	fprintf(execution_log_file, "COLUMNS = %d\n",							COLUMNS								);	// 53
	fprintf(execution_log_file, "ROWS = %d\n",								ROWS								);	// 54
	fprintf(execution_log_file, "SLICES = %d\n",							SLICES								);	// 55
	fprintf(execution_log_file, "VOXEL_WIDTH = %6.6lf\n",					VOXEL_WIDTH							);	// 56
	fprintf(execution_log_file, "VOXEL_HEIGHT = %6.6lf\n",					VOXEL_HEIGHT						);	// 57
	fprintf(execution_log_file, "VOXEL_THICKNESS = %6.6lf\n",				VOXEL_THICKNESS						);	// 58
	fprintf(execution_log_file, "IMAGE_WIDTH = %6.6lf\n",					IMAGE_WIDTH							);	// 59
	fprintf(execution_log_file, "IMAGE_HEIGHT = %6.6lf\n",					IMAGE_HEIGHT						);	// 60
	fprintf(execution_log_file, "IMAGE_THICKNESS = %6.6lf\n",				IMAGE_THICKNESS						);	// 61
		
	fprintf(execution_log_file, "SC_LOWER_THRESHOLD = %6.6lf\n",			SC_LOWER_THRESHOLD					);	// 62
	fprintf(execution_log_file, "SC_UPPER_THRESHOLD = %6.6lf\n",			SC_UPPER_THRESHOLD					);	// 63
	fprintf(execution_log_file, "MSC_LOWER_THRESHOLD = %6.6lf\n",			MSC_LOWER_THRESHOLD					);	// 65
	fprintf(execution_log_file, "MSC_UPPER_THRESHOLD = %6.6lf\n",			MSC_UPPER_THRESHOLD					);	// 64
	fprintf(execution_log_file, "MSC_DIFF_THRESH = %d\n",					MSC_DIFF_THRESH						);	// 66
	fprintf(execution_log_file, "SM_LOWER_THRESHOLD = %6.6lf\n",			SM_LOWER_THRESHOLD					);	// 67
	fprintf(execution_log_file, "SM_UPPER_THRESHOLD = %6.6lf\n",			SM_UPPER_THRESHOLD					);	// 68
	fprintf(execution_log_file, "SM_SCALE_THRESHOLD = %6.6lf\n",			SM_SCALE_THRESHOLD					);	// 69

	fprintf(execution_log_file, "SINOGRAM_FILTER = %s\n",						SINOGRAM_FILTER_CSTRING					);	// 70
	fprintf(execution_log_file, "AVG_FILTER_FBP = %s\n",					AVG_FILTER_FBP_CSTRING				);	// 71
	fprintf(execution_log_file, "AVG_FILTER_HULL = %s\n",					AVG_FILTER_HULL_CSTRING				);	// 72
	fprintf(execution_log_file, "AVG_FILTER_X_0 = %s\n",					AVG_FILTER_X_0_CSTRING				);	// 73
	fprintf(execution_log_file, "MEDIAN_FILTER_FBP = %s\n",					MEDIAN_FILTER_FBP_CSTRING			);	// 74
	fprintf(execution_log_file, "MEDIAN_FILTER_HULL = %s\n",				MEDIAN_FILTER_HULL_CSTRING			);	// 75
	fprintf(execution_log_file, "MEDIAN_FILTER_X_0 = %s\n",					MEDIAN_FILTER_X_0_CSTRING			);	// 76
	//fprintf(execution_log_file, "AVG_FILTER_FBP = %d\n",						AVG_FILTER_FBP					);	// 69
	//fprintf(execution_log_file, "AVG_FILTER_HULL = %d\n",						AVG_FILTER_HULL					);	// 70
	//fprintf(execution_log_file, "AVG_FILTER_X_0 = %d\n",						AVG_FILTER_X_0					);	// 71
	//fprintf(execution_log_file, "MEDIAN_FILTER_FBP = %d\n",					MEDIAN_FILTER_FBP					);	// 72
	//fprintf(execution_log_file, "MEDIAN_FILTER_HULL = %d\n",					MEDIAN_FILTER_HULL				);	// 73
	//fprintf(execution_log_file, "MEDIAN_FILTER_X_0 = %d\n",					MEDIAN_FILTER_X_0					);	// 74

	fprintf(execution_log_file, "FBP_AVG_FILTER_RADIUS = %d\n",				FBP_AVG_FILTER_RADIUS				);	// 77	
	fprintf(execution_log_file, "FBP_MED_FILTER_RADIUS = %d\n",				FBP_MED_FILTER_RADIUS				);	// 78
	fprintf(execution_log_file, "FBP_AVG_FILTER_THRESHOLD = %6.6lf\n",		FBP_AVG_FILTER_THRESHOLD			);	// 79	
	fprintf(execution_log_file, "HULL_AVG_FILTER_RADIUS = %d\n",			HULL_AVG_FILTER_RADIUS				);	// 80
	fprintf(execution_log_file, "HULL_MED_FILTER_RADIUS = %d\n",			HULL_MED_FILTER_RADIUS				);	// 81
	fprintf(execution_log_file, "HULL_AVG_FILTER_THRESHOLD = %6.6lf\n",		HULL_AVG_FILTER_THRESHOLD			);	// 82
	fprintf(execution_log_file, "X_0_AVG_FILTER_RADIUS = %d\n",				X_0_AVG_FILTER_RADIUS				);	// 83
	fprintf(execution_log_file, "X_0_MED_FILTER_RADIUS = %d\n",				X_0_MED_FILTER_RADIUS				);	// 84
	fprintf(execution_log_file, "X_0_AVG_FILTER_THRESHOLD = %6.6lf\n",		X_0_AVG_FILTER_THRESHOLD			);	// 85
	
	fprintf(execution_log_file, "TRIG_TABLE_MIN = %6.6lf\n",				TRIG_TABLE_MIN						);	// 86
	fprintf(execution_log_file, "TRIG_TABLE_MAX = %6.6lf\n",				TRIG_TABLE_MAX						);	// 87
	fprintf(execution_log_file, "TRIG_TABLE_STEP = %6.6lf\n",				TRIG_TABLE_STEP						);	// 88
	fprintf(execution_log_file, "COEFF_TABLE_RANGE = %6.6lf\n",				COEFF_TABLE_RANGE					);	// 89
	fprintf(execution_log_file, "COEFF_TABLE_STEP = %6.6lf\n",				COEFF_TABLE_STEP					);	// 90
	fprintf(execution_log_file, "POLY_TABLE_RANGE = %6.6lf\n",				POLY_TABLE_RANGE					);	// 91
	fprintf(execution_log_file, "POLY_TABLE_STEP = %6.6lf\n",				POLY_TABLE_STEP						);	// 92
	
	fprintf(execution_log_file, "ENDPOINTS_ALG = %s\n",						ENDPOINTS_ALG_CSTRING				);	// 93
	fprintf(execution_log_file, "ENDPOINTS_TX_MODE = %s\n",					ENDPOINTS_TX_MODE_CSTRING			);	// 94
	fprintf(execution_log_file, "ENDPOINTS_HULL = %s\n",					ENDPOINTS_HULL_CSTRING				);	// 95
	fprintf(execution_log_file, "MLP_ALGORITHM = %s\n",						MLP_ALGORITHM_CSTRING				);	// 96
	//fprintf(execution_log_file, "IGNORE_SHORT_MLP = %d\n",					IGNORE_SHORT_MLP					);	// 95
	fprintf(execution_log_file, "IGNORE_SHORT_MLP = %s\n",					IGNORE_SHORT_MLP_CSTRING				);	// 97
	fprintf(execution_log_file, "MIN_MLP_LENGTH = %d\n",					MIN_MLP_LENGTH						);	// 98
	fprintf(execution_log_file, "MLP_U_STEP = %6.6lf\n",					MLP_U_STEP							);	// 99
	
	fprintf(execution_log_file, "X_0_CSTRING = %s\n",						X_0_CSTRING							);	// 100
	fprintf(execution_log_file, "PROJECTION_ALGORITHM_CSTRING = %s\n",		PROJECTION_ALGORITHM_CSTRING			);	// 101
	fprintf(execution_log_file, "RECON_TX_MODE = %s\n",						RECON_TX_MODE_CSTRING				);	// 102
	fprintf(execution_log_file, "ITERATIONS = %d\n",						ITERATIONS							);	// 103
	fprintf(execution_log_file, "DROP_BLOCK_SIZE = %d\n",					DROP_BLOCK_SIZE						);	// 104
	fprintf(execution_log_file, "LAMBDA = %6.6lf\n",						LAMBDA								);	// 105
	//fprintf(execution_log_file, "BOUND_IMAGE = %s\n",						BOUND_IMAGE_CSTRING					);	// 106
	//fprintf(execution_log_file, "BOUND_IMAGE = %d\n",							BOUND_IMAGE						);	// 104
	
	fprintf(execution_log_file, "ROBUST_METHOD = %s\n",						ROBUST_METHOD_CSTRING				);	// 107
	fprintf(execution_log_file, "ETA = %6.6lf\n",							ETA									);	// 108
	fprintf(execution_log_file, "PSI_SIGN = %d\n",							PSI_SIGN							);	// 109
	
	fprintf(execution_log_file, "BOUND_IMAGE = %s\n",						BOUND_IMAGE_CSTRING					);	// 106
	fprintf(execution_log_file, "IDENTIFY_X_0_AIR = %s\n",					IDENTIFY_X_0_AIR_CSTRING			);	// 106
	fprintf(execution_log_file, "X_0_AIR_THRESHOLD = %6.6lf\n",				X_0_AIR_THRESHOLD					);	// 108
	fprintf(execution_log_file, "IDENTIFY_X_N_AIR = %s\n",					IDENTIFY_X_N_AIR_CSTRING			);	// 106
	fprintf(execution_log_file, "X_N_AIR_THRESHOLD = %6.6lf\n",				X_N_AIR_THRESHOLD					);	// 108

	fprintf(execution_log_file, "S_CURVE = %s\n",							S_CURVE_CSTRING						);	// 110
	fprintf(execution_log_file, "S_CURVE_ON = %s\n",						S_CURVE_ON_CSTRING					);	// 111
	fprintf(execution_log_file, "DUAL_SIDED_S_CURVE = %s\n",				DUAL_SIDED_S_CURVE_CSTRING			);	// 112
	//fprintf(execution_log_file, "S_CURVE_ON = %d\n",							S_CURVE_ON						);	// 109
	//fprintf(execution_log_file, "DUAL_SIDED_S_CURVE = %d\n",					DUAL_SIDED_S_CURVE				);	// 110
	fprintf(execution_log_file, "SIGMOID_STEEPNESS = %6.6lf\n",				SIGMOID_STEEPNESS					);	// 113
	fprintf(execution_log_file, "SIGMOID_MID_SHIFT = %6.6lf\n",				SIGMOID_MID_SHIFT					);	// 114
		
	fprintf(execution_log_file, "TVS_ON = %s\n",							TVS_ON_CSTRING						);	// 115
	fprintf(execution_log_file, "TVS_FIRST = %s\n",							TVS_FIRST_CSTRING					);	// 116
	fprintf(execution_log_file, "TVS_PARALLEL = %s\n",						TVS_PARALLEL_CSTRING					);	// 117
	fprintf(execution_log_file, "TVS_CONDITIONED = %s\n",					TVS_CONDITIONED_CSTRING				);	// 118
	fprintf(execution_log_file, "TVS_REPETITIONS = %d\n",					TVS_REPETITIONS						);	// 119
	fprintf(execution_log_file, "BETA_0 = %6.6lf\n",						BETA_0								);	// 120
	fprintf(execution_log_file, "A = %6.6lf\n",								A									);	// 121
	fprintf(execution_log_file, "L_0 = %d\n",								L_0									);	// 122
	fprintf(execution_log_file, "\n"																				);	// end line, go to beginning of next entry
	fclose(execution_log_file);
	
	OUTPUT_FILE_LIST.push_back(std::string(LOCAL_EXECUTION_INFO_PATH));
	LOCAL_DATA_FILE_LIST.push_back(std::string(LOCAL_EXECUTION_INFO_PATH));	
	print_section_exit( "Finished writing execution info to local file", SECTION_EXIT_CSTRING, RED_TEXT, RED_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );
}
void execution_log_2_csv()
{
	int i = 0;
	//char execution_log_path[256];
	//sprintf(execution_log_path, "%s%s.csv", OUTPUT_DIRECTORY, EXECUTION_LOG_BASENAME);
	if(!file_exists (GLOBAL_EXECUTION_LOG_PATH))
		init_execution_log_csv();
	print_colored_text("Copying execution log from the network-attached storage device to the compute node so an entry can be added for the current execution...", CYAN_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );
	copy_data(BASH_COPY_FILE, GLOBAL_EXECUTION_LOG_PATH, LOCAL_EXECUTION_LOG_PATH);
	if( !file_exists (LOCAL_EXECUTION_LOG_PATH))
		init_execution_log_csv();
	
	print_colored_text("Writing parameters/options for the current execution to global .csv execution log at:", CYAN_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );
	//print_colored_text(execution_log_path, LIGHT_PURPLE_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );
	print_colored_text(LOCAL_EXECUTION_LOG_PATH, LIGHT_PURPLE_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT );
		
	FILE* execution_log_file = fopen( LOCAL_EXECUTION_LOG_PATH, "a+" );
	fprintf(execution_log_file, "%s, ",				EXECUTION_DATE_TIME					);	// 1
	fprintf(execution_log_file, "%s, ",				CURRENT_COMPUTE_NODE_ALIAS			);	// 2	
	fprintf(execution_log_file, "%s, ",				GIT_REPO_INFO						);	// 3
	fprintf(execution_log_file, "%s, ",				TESTED_BY_CSTRING					);	// 4
	fprintf(execution_log_file, "%s, ",				INPUT_DIRECTORY						);	// 5
	fprintf(execution_log_file, "%s, ",				INPUT_FOLDER						);	// 6
	fprintf(execution_log_file, "%s, ",				OUTPUT_DIRECTORY					);	// 7
	fprintf(execution_log_file, "%s, ",				OUTPUT_FOLDER_UNIQUE				);	// 8
	fprintf(execution_log_file, "%d, ",				total_histories						);	// 9
	fprintf(execution_log_file, "%d, ",				recon_vol_histories					);	// 10
	fprintf(execution_log_file, "%d, ",				post_cut_histories					);	// 11
	fprintf(execution_log_file, "%d, ",				reconstruction_histories			);	// 12
	fprintf(execution_log_file, "%6.6lf, ",			execution_time_data_reads			);	// 13
	fprintf(execution_log_file, "%6.6lf, ",			execution_time_preprocessing		);	// 14
	fprintf(execution_log_file, "%6.6lf, ",			execution_time_endpoints			);	// 15
	fprintf(execution_log_file, "%6.6lf, ",			execution_time_tables				);	// 16
	fprintf(execution_log_file, "%6.6lf, ",			execution_time_init_image			);	// 17
	fprintf(execution_log_file, "%6.6lf, ",			execution_time_DROP					);	// 18
	for( i = 0; i < execution_times_DROP_iterations.size(); i++ )
		fprintf(execution_log_file, " %6.6lf, ", execution_times_DROP_iterations[i]		);	// 19-33
	for( ; i < MAX_ITERATIONS; i++ )
		fprintf(execution_log_file, ", "												);													
	fprintf(execution_log_file, "%6.6lf, ",			execution_time_reconstruction		);	// 34
	fprintf(execution_log_file, "%6.6lf, ",			execution_time_program				);	// 35
	
	fprintf(execution_log_file, "%d, ",				THREADS_PER_BLOCK					);	// 36	
	fprintf(execution_log_file, "%d, ",				ENDPOINTS_PER_BLOCK					);	// 37
	fprintf(execution_log_file, "%d, ",				HISTORIES_PER_BLOCK					);	// 38
	fprintf(execution_log_file, "%d, ",				ENDPOINTS_PER_THREAD				);	// 39
	fprintf(execution_log_file, "%d, ",				HISTORIES_PER_THREAD				);	// 40
	fprintf(execution_log_file, "%d, ",				VOXELS_PER_THREAD					);	// 41
	fprintf(execution_log_file, "%d, ",				MAX_GPU_HISTORIES					);	// 42
	fprintf(execution_log_file, "%d, ",				MAX_CUTS_HISTORIES					);	// 43
	fprintf(execution_log_file, "%d, ",				MAX_ENDPOINTS_HISTORIES				);	// 44
				
	fprintf(execution_log_file, "%d, ",				NUM_SCANS							);	// 45
	fprintf(execution_log_file, "%6.6lf, ",			GANTRY_ANGLE_INTERVAL				);	// 46	
	fprintf(execution_log_file, "%s, ",				SCAN_TYPE_CSTRING					);	// 47
	fprintf(execution_log_file, "%6.6lf, ",			T_SHIFT								);	// 48
	fprintf(execution_log_file, "%6.6lf, ",			U_SHIFT								);	// 49
	fprintf(execution_log_file, "%6.6lf, ",			V_SHIFT								);	// 50
	
	fprintf(execution_log_file, "%6.6lf, ",			SSD_T_SIZE							);	// 51
	fprintf(execution_log_file, "%6.6lf, ",			SSD_V_SIZE							);	// 52
	fprintf(execution_log_file, "%6.6lf, ",			T_BIN_SIZE							);	// 53
	fprintf(execution_log_file, "%6.6lf, ",			V_BIN_SIZE							);	// 54
	fprintf(execution_log_file, "%6.6lf, ",			ANGULAR_BIN_SIZE					);	// 55
	fprintf(execution_log_file, "%d, ",				SIGMAS_TO_KEEP						);	// 56
	fprintf(execution_log_file, "%s, ",				SAMPLE_STD_DEV_CSTRING				);	// 57
	
	fprintf(execution_log_file, "%6.6lf, ",			RECON_CYL_RADIUS					);	// 58
	fprintf(execution_log_file, "%6.6lf, ",			RECON_CYL_HEIGHT					);	// 59
	fprintf(execution_log_file, "%d, ",				COLUMNS							);	// 60
	fprintf(execution_log_file, "%d, ",				ROWS								);	// 61
	fprintf(execution_log_file, "%d, ",				SLICES								);	// 62
	fprintf(execution_log_file, "%6.6lf, ",			VOXEL_WIDTH							);	// 63
	fprintf(execution_log_file, "%6.6lf, ",			VOXEL_HEIGHT						);	// 64
	fprintf(execution_log_file, "%6.6lf, ",			VOXEL_THICKNESS						);	// 65
	fprintf(execution_log_file, "%6.6lf, ",			IMAGE_WIDTH							);	// 66
	fprintf(execution_log_file, "%6.6lf, ",			IMAGE_HEIGHT						);	// 67
	fprintf(execution_log_file, "%6.6lf, ",			IMAGE_THICKNESS						);	// 68
		
	fprintf(execution_log_file, "%6.6lf, ",			SC_LOWER_THRESHOLD					);	// 69
	fprintf(execution_log_file, "%6.6lf, ",			SC_UPPER_THRESHOLD					);	// 70
	fprintf(execution_log_file, "%6.6lf, ",			MSC_LOWER_THRESHOLD					);	// 71
	fprintf(execution_log_file, "%6.6lf, ",			MSC_UPPER_THRESHOLD					);	// 72
	fprintf(execution_log_file, "%d, ",				MSC_DIFF_THRESH						);	// 73
	fprintf(execution_log_file, "%6.6lf, ",			SM_LOWER_THRESHOLD					);	// 74
	fprintf(execution_log_file, "%6.6lf, ",			SM_UPPER_THRESHOLD					);	// 75
	fprintf(execution_log_file, "%6.6lf, ",			SM_SCALE_THRESHOLD					);	// 76

	fprintf(execution_log_file, "%s, ",				SINOGRAM_FILTER_CSTRING				);	// 77
	fprintf(execution_log_file, "%s, ",				AVG_FILTER_FBP_CSTRING				);	// 78
	fprintf(execution_log_file, "%s, ",				AVG_FILTER_HULL_CSTRING				);	// 79
	fprintf(execution_log_file, "%s, ",				AVG_FILTER_X_0_CSTRING				);	// 80
	fprintf(execution_log_file, "%s, ",				MEDIAN_FILTER_FBP_CSTRING			);	// 81
	fprintf(execution_log_file, "%s, ",				MEDIAN_FILTER_HULL_CSTRING			);	// 82
	fprintf(execution_log_file, "%s, ",				MEDIAN_FILTER_X_0_CSTRING			);	// 83
	
	fprintf(execution_log_file, "%d, ",				FBP_AVG_FILTER_RADIUS				);	// 84
	fprintf(execution_log_file, "%d, ",				FBP_MED_FILTER_RADIUS				);	// 85
	fprintf(execution_log_file, "%6.6lf, ",			FBP_AVG_FILTER_THRESHOLD			);	// 86
	fprintf(execution_log_file, "%d, ",				HULL_AVG_FILTER_RADIUS				);	// 87
	fprintf(execution_log_file, "%d, ",				HULL_MED_FILTER_RADIUS				);	// 88
	fprintf(execution_log_file, "%6.6lf, ",			HULL_AVG_FILTER_THRESHOLD			);	// 89
	fprintf(execution_log_file, "%d, ",				X_0_AVG_FILTER_RADIUS				);	// 90
	fprintf(execution_log_file, "%d, ",				X_0_MED_FILTER_RADIUS				);	// 91
	fprintf(execution_log_file, "%6.6lf, ",			X_0_AVG_FILTER_THRESHOLD			);	// 92

	fprintf(execution_log_file, "%6.6lf, ",			TRIG_TABLE_MIN						);	// 93
	fprintf(execution_log_file, "%6.6lf, ",			TRIG_TABLE_MAX						);	// 94
	fprintf(execution_log_file, "%6.6lf, ",			TRIG_TABLE_STEP						);	// 95
	fprintf(execution_log_file, "%6.6lf, ",			COEFF_TABLE_RANGE					);	// 96
	fprintf(execution_log_file, "%6.6lf, ",			COEFF_TABLE_STEP					);	// 97
	fprintf(execution_log_file, "%6.6lf, ",			POLY_TABLE_RANGE					);	// 98
	fprintf(execution_log_file, "%6.6lf, ",			POLY_TABLE_STEP						);	// 99

	fprintf(execution_log_file, "%s, ",				ENDPOINTS_ALG_CSTRING				);	// 100
	fprintf(execution_log_file, "%s, ",				ENDPOINTS_TX_MODE_CSTRING			);	// 101
	fprintf(execution_log_file, "%s, ",				ENDPOINTS_HULL_CSTRING				);	// 102
	fprintf(execution_log_file, "%s, ",				MLP_ALGORITHM_CSTRING				);	// 103
	fprintf(execution_log_file, "%s, ",				IGNORE_SHORT_MLP_CSTRING			);	// 104
	fprintf(execution_log_file, "%d, ",				MIN_MLP_LENGTH						);	// 105
	fprintf(execution_log_file, "%6.6lf, ",			MLP_U_STEP							);	// 106
	
	fprintf(execution_log_file, "%s, ",				X_0_CSTRING							);	// 107
	fprintf(execution_log_file, "%s, ",				PROJECTION_ALGORITHM_CSTRING		);	// 108
	fprintf(execution_log_file, "%s, ",				RECON_TX_MODE_CSTRING				);	// 109
	fprintf(execution_log_file, "%d, ",				ITERATIONS							);	// 110
	fprintf(execution_log_file, "%d, ",				DROP_BLOCK_SIZE						);	// 111
	fprintf(execution_log_file, "%6.6lf, ",			LAMBDA								);	// 112
	
	fprintf(execution_log_file, "%s, ",				ROBUST_METHOD_CSTRING				);	// 113
	fprintf(execution_log_file, "%6.6lf, ",			ETA									);	// 114
	fprintf(execution_log_file, "%d, ",				PSI_SIGN							);	// 115

	fprintf(execution_log_file, "%s, ",				BOUND_IMAGE_CSTRING					);	// 116
	fprintf(execution_log_file, "%s, ",				IDENTIFY_X_0_AIR_CSTRING			);	// 117
	fprintf(execution_log_file, "%6.6lf, ",			X_0_AIR_THRESHOLD					);	// 118
	fprintf(execution_log_file, "%s, ",				IDENTIFY_X_N_AIR_CSTRING			);	// 119
	fprintf(execution_log_file, "%6.6lf, ",			X_N_AIR_THRESHOLD					);	// 120
		
	fprintf(execution_log_file, "%s, ",				S_CURVE_CSTRING						);	// 121
	fprintf(execution_log_file, "%s, ",				S_CURVE_ON_CSTRING					);	// 122
	fprintf(execution_log_file, "%s, ",				DUAL_SIDED_S_CURVE_CSTRING			);	// 123
	fprintf(execution_log_file, "%6.6lf, ",			SIGMOID_STEEPNESS					);	// 124
	fprintf(execution_log_file, "%6.6lf, ",			SIGMOID_MID_SHIFT					);	// 125
	
	fprintf(execution_log_file, "%s, ",				TVS_ON_CSTRING						);	// 126
	fprintf(execution_log_file, "%s, ",				TVS_FIRST_CSTRING					);	// 127
	fprintf(execution_log_file, "%s, ",				TVS_PARALLEL_CSTRING				);	// 128
	fprintf(execution_log_file, "%s, ",				TVS_CONDITIONED_CSTRING				);	// 129
	fprintf(execution_log_file, "%d, ",				TVS_REPETITIONS						);	// 130
	fprintf(execution_log_file, "%6.6lf, ",			BETA_0								);	// 131
	fprintf(execution_log_file, "%6.6lf, ",			A									);	// 132
	fprintf(execution_log_file, "%d, ",				L_0									);	// 133
	fprintf(execution_log_file, "\n"													);	// end line, go to beginning of next entry
	fclose(execution_log_file);
	print_colored_text("Copying updated execution log back to the network-attached storage device...", CYAN_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );
	copy_data(BASH_COPY_FILE, LOCAL_EXECUTION_LOG_PATH, GLOBAL_EXECUTION_LOG_PATH);		
	print_section_exit( "Finshed updating global execution log with current execution info", SECTION_EXIT_CSTRING, RED_TEXT, RED_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );	
	GLOBAL_DATA_FILE_LIST.push_back(std::string(GLOBAL_EXECUTION_LOG_PATH));		
	OUTPUT_FILE_LIST.push_back(std::string(LOCAL_EXECUTION_LOG_PATH));
	LOCAL_DATA_FILE_LIST.push_back(std::string(LOCAL_EXECUTION_LOG_PATH));	
}
void init_execution_log_csv()
{
	print_colored_text("Global execution log does not exist; proceeding to create/initialize global execution log and writing to the network-attached storage device...", CYAN_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );	
	FILE* execution_log_file = fopen( GLOBAL_EXECUTION_LOG_PATH, "w"	);
	fprintf(execution_log_file, "Execution Properties: Date/Time (EXECUTION_DATE_TIME), "			);	// 2
	fprintf(execution_log_file, "Execution Properties: Compute Node (CURRENT_COMPUTE_NODE_ALIAS), "				);	// 3	
	fprintf(execution_log_file, "Execution Properties: Git Branch : Commit Hash (Commit Date) (GIT_REPO_INFO), "	);	// 3
	fprintf(execution_log_file, "Execution Properties: Executed By (TESTED_BY_CSTRING), "					);	// 4
	fprintf(execution_log_file, "Data Paths: Input Data Directory (INPUT_DIRECTORY), "				);	// 5
	fprintf(execution_log_file, "Data Paths: Input Data Folder (INPUT_FOLDER), "				);	// 6
	fprintf(execution_log_file, "Data Paths: Output Data/Image Directory (OUTPUT_DIRECTORY), "			);	// 7
	fprintf(execution_log_file, "Data Paths: Output Data/Image Folder (OUTPUT_FOLDER_UNIQUE), "		);	// 8
	fprintf(execution_log_file, "# Proton Histories: Read From Disk (total_histories), "				);	// 8
	fprintf(execution_log_file, "# Proton Histories: Traversed Reconstruction Volume (recon_vol_histories), "			);	// 8
	fprintf(execution_log_file, "# Proton Histories: Survived Statistical Cuts (post_cut_histories), "			);	// 8
	fprintf(execution_log_file, "# Proton Histories: Used in Reconstruction (reconstruction_histories), "	);	// 8	
	fprintf(execution_log_file, "Execution Time: Import Data ((execution_time_data_reads)), "					);	// 9
	fprintf(execution_log_file, "Execution Time: Preprocessing (execution_time_preprocessing), "				);	// 10
	fprintf(execution_log_file, "Execution Time: MLP endpoints (execution_time_endpoints), "					);	// 11
	fprintf(execution_log_file, "Execution Time: MLP Lookup Tables (execution_time_tables), "						);	// 12
	fprintf(execution_log_file, "Execution Time: Image Initialization (execution_time_init_image), "					);	// 13
	fprintf(execution_log_file, "Execution Time: DROP_total (execution_time_DROP), "					);	// 14
	for( int i = 1; i <= MAX_ITERATIONS; i++ )
		fprintf(execution_log_file, "Execution Time: Iteration %d (execution_times_DROP_iterations[%d]), ", i, i - 1		);	// 15-26													
	fprintf(execution_log_file, "Execution Time: Reconstruction (execution_time_reconstruction), "				);	// 27
	fprintf(execution_log_file, "Execution Time: Complete Program (execution_time_program), "						);	// 28

	fprintf(execution_log_file, "GPU Configuration: # Threads / GPU Block (THREADS_PER_BLOCK), "			);	// 29	
	fprintf(execution_log_file, "GPU Configuration: # MLP Endpoints Each GPU Block is Responsible for (ENDPOINTS_PER_BLOCK), "			);	// 30
	fprintf(execution_log_file, "GPU Configuration: # Histories Each GPU Block is Responsible for (HISTORIES_PER_BLOCK), "			);	// 31
	fprintf(execution_log_file, "GPU Configuration: # MLP Endpoints Each Thread is Responsible for (ENDPOINTS_PER_THREAD), "		);	// 32
	fprintf(execution_log_file, "GPU Configuration: # Histories Each Thread is Responsible for (HISTORIES_PER_THREAD), "		);	// 33
	fprintf(execution_log_file, "GPU Configuration: # Voxels Each Thread is Responsible for (VOXELS_PER_THREAD), "			);	// 34
	fprintf(execution_log_file, "GPU Configuration: Max # Histories Per Typical GPU Kernel Launch (MAX_GPU_HISTORIES), "			);	// 35
	fprintf(execution_log_file, "GPU Configuration: Max # Histories Per Statistical Cuts GPU Kernel Launch for (MAX_CUTS_HISTORIES), "			);	// 36
	fprintf(execution_log_file, "GPU Configuration: Max # Histories Per MLP Endpoints GPU Kernel Launch for (MAX_ENDPOINTS_HISTORIES), "		);	// 37
				
	fprintf(execution_log_file, "Scan Properties: # Object Scan Repetitions (NUM_SCANS), "					);	// 38
	fprintf(execution_log_file, "Scan Properties: Gantry Angle Increments (GANTRY_ANGLE_INTERVAL), "		);	// 39	
	fprintf(execution_log_file, "Scan Properties: Simulated/Experimental Scan (SCAN_TYPE), "					);	// 40
	fprintf(execution_log_file, "Scan Properties: Object Off-Center in t by [cm] (T_SHIFT), "						);	// 41
	fprintf(execution_log_file, "Scan Properties: Object Off-Center in u by [cm] (U_SHIFT), "						);	// 42
	fprintf(execution_log_file, "Scan Properties: Object Off-Center in v by [cm] (V_SHIFT), "						);	// 43
	
	fprintf(execution_log_file, "Scan Properties: t-Coordinate Range [cm] (SSD_T_SIZE), "					);	// 44
	fprintf(execution_log_file, "Scan Properties: v-Coordinate Range [cm] (SSD_V_SIZE), "					);	// 45
	fprintf(execution_log_file, "Statistical Cuts: t-Bin Size [cm] (T_BIN_SIZE), "					);	// 46
	fprintf(execution_log_file, "Statistical Cuts: v-Bin Size [cm] (V_BIN_SIZE), "					);	// 47
	fprintf(execution_log_file, "Statistical Cuts: Path Angle-Bin Size [degrees] (ANGULAR_BIN_SIZE), "			);	// 48
	fprintf(execution_log_file, "Statistical Cuts: # Standard Deviations Accepted (SIGMAS_TO_KEEP), "				);	// 49
	fprintf(execution_log_file, "Statistical Cuts: Use Sample Standard Deviation (SAMPLE_STD_DEV), "				);	// 50
	
	fprintf(execution_log_file, "Reconstruction Volume: Cylinder Radius [cm] (RECON_CYL_RADIUS), "			);	// 51
	fprintf(execution_log_file, "Reconstruction Volume: Cylinder Height [cm] (RECON_CYL_HEIGHT), "			);	// 52
	fprintf(execution_log_file, "Reconstructed Images: # Columns (COLUMNS), "						);	// 53
	fprintf(execution_log_file, "Reconstructed Images: # Rows (ROWS), "						);	// 54
	fprintf(execution_log_file, "Reconstructed Images: # Slices (SLICES), "						);	// 55
	fprintf(execution_log_file, "Reconstructed Images: Voxel Width [cm] (VOXEL_WIDTH), "					);	// 56
	fprintf(execution_log_file, "Reconstructed Images: Voxel Height [cm] (VOXEL_HEIGHT), "				);	// 57
	fprintf(execution_log_file, "Reconstructed Images: Voxel Thickness [cm] (VOXEL_THICKNESS), "				);	// 58
	fprintf(execution_log_file, "Reconstructed Images: Image Width [cm] (IMAGE_WIDTH), "					);	// 59
	fprintf(execution_log_file, "Reconstructed Images: Image Height [cm] (IMAGE_HEIGHT), "				);	// 60
	fprintf(execution_log_file, "Reconstructed Images: Image Thickness [cm] (IMAGE_THICKNESS), "				);	// 61
		
	fprintf(execution_log_file, "Hull Detection (SC): Skip if WEPL Below (SC_LOWER_THRESHOLD), "			);	// 62
	fprintf(execution_log_file, "Hull Detection (SC): Skip if WEPL Above (SC_UPPER_THRESHOLD), "			);	// 63
	fprintf(execution_log_file, "Hull Detection (MSC): Skip if WEPL Below (MSC_LOWER_THRESHOLD), "			);	// 65
	fprintf(execution_log_file, "Hull Detection (MSC): Skip if WEPL Above (MSC_UPPER_THRESHOLD), "			);	// 64
	fprintf(execution_log_file, "Hull Detection (MSC): In Hull if ID'ed Count Exceeds Neighbors by (MSC_DIFF_THRESH), "				);	// 66
	fprintf(execution_log_file, "Hull Detection (SM): Skip if WEPL Below (SM_LOWER_THRESHOLD), "			);	// 67
	fprintf(execution_log_file, "Hull Detection (SM): Skip if WEPL Above (SM_UPPER_THRESHOLD), "			);	// 68
	fprintf(execution_log_file, "Hull Detection (SM): Hull Detection (SM): Scale ID Count Threshold by (SM_SCALE_THRESHOLD), "			);	// 69

	fprintf(execution_log_file, "Image Filtering: FBP Filter Name (SINOGRAM_FILTER), "					);	// 70
	fprintf(execution_log_file, "Image Filtering: Average Filter FBP (AVG_FILTER_FBP), "				);	// 71
	fprintf(execution_log_file, "Image Filtering: Average Filter Hull (AVG_FILTER_HULL), "				);	// 72
	fprintf(execution_log_file, "Image Filtering: Average Filter Initial Iterate (AVG_FILTER_X_0), "				);	// 73
	fprintf(execution_log_file, "Image Filtering: Median Filter FBP (MEDIAN_FILTER_FBP), "			);	// 74
	fprintf(execution_log_file, "Image Filtering: Median Filter Hull (MEDIAN_FILTER_HULL), "			);	// 75
	fprintf(execution_log_file, "Image Filtering: Median Filter Initial Iterate (MEDIAN_FILTER_X_0), "			);	// 76

	fprintf(execution_log_file, "Image Filtering: FBP Average Filter Radius (FBP_AVG_FILTER_RADIUS), "		);	// 77	
	fprintf(execution_log_file, "Image Filtering: FBP Median Filter Radius (FBP_MED_FILTER_RADIUS), "		);	// 78
	fprintf(execution_log_file, "Image Filtering: If Thresholding on, FBP Averages < Threshold -> 0 (FBP_AVG_FILTER_THRESHOLD), "	);	// 79	
	fprintf(execution_log_file, "Image Filtering: Hull Average Filter Radius (HULL_AVG_FILTER_RADIUS), "		);	// 80
	fprintf(execution_log_file, "Image Filtering: Hull Median Filter Radius (HULL_MED_FILTER_RADIUS), "		);	// 81
	fprintf(execution_log_file, "Image Filtering: If Thresholding on, Hull Averages < Threshold -> 0 (HULL_AVG_FILTER_THRESHOLD), "	);	// 82
	fprintf(execution_log_file, "Image Filtering: Initial Iterate Average Filter Radius (X_0_AVG_FILTER_RADIUS), "		);	// 83
	fprintf(execution_log_file, "Image Filtering: Initial Iterate Median Filter Radius (X_0_MED_FILTER_RADIUS), "		);	// 84
	fprintf(execution_log_file, "Image Filtering: If Thresholding on, Initial Iterate Averages < Threshold -> 0 (X_0_AVG_FILTER_THRESHOLD) "	);	// 85
	
	fprintf(execution_log_file, "MLP Lookup Tables: Sin/Cos Table Min Angle [radians] (TRIG_TABLE_MIN), "				);	// 86
	fprintf(execution_log_file, "MLP Lookup Tables: Sin/Cos Table Max Angle [radians] (TRIG_TABLE_MAX), "				);	// 87
	fprintf(execution_log_file, "MLP Lookup Tables: Sin/Cos Table Angle Resolution [radians] (TRIG_TABLE_STEP), "				);	// 88
	fprintf(execution_log_file, "MLP Lookup Tables: Coefficient Table Depth Range [cm] (COEFF_TABLE_RANGE), "			);	// 89
	fprintf(execution_log_file, "MLP Lookup Tables: Coefficient Table Depth Resolution [cm] (COEFF_TABLE_STEP), "			);	// 90
	fprintf(execution_log_file, "MLP Lookup Tables: Polynomial Tables Depth Range [cm] (POLY_TABLE_RANGE), "			);	// 91
	fprintf(execution_log_file, "MLP Lookup Tables: Polynomial Tables Depth Resolution [cm] (POLY_TABLE_STEP), "				);	// 92
	
	fprintf(execution_log_file, "MLP Endpoints: Algorithm (ENDPOINTS_ALG_CSTRING), "				);	// 93
	fprintf(execution_log_file, "MLP Endpoints: Data Transfer Mode (ENDPOINTS_TX_MODE_CSTRING), "			);	// 94
	fprintf(execution_log_file, "MLP Endpoints: Hull Used (ENDPOINTS_HULL_CSTRING), "				);	// 95
	fprintf(execution_log_file, "MLP: Algorithm (MLP_ALGORITHM_CSTRING), "				);	// 96
	fprintf(execution_log_file, "MLP: Ignore Short MLPs ON/OFF (IGNORE_SHORT_MLP_CSTRING), "			);	// 97
	fprintf(execution_log_file, "MLP: Ignore MLP Paths w/ # Voxels Below (MIN_MLP_LENGTH), "				);	// 98
	fprintf(execution_log_file, "MLP: Calculate MLP at Depths Separated by [cm] (MLP_U_STEP), "					);	// 99
	
	fprintf(execution_log_file, "Reconstruction: Initial Iterate Type (X_0_CSTRING), "					);	// 100
	fprintf(execution_log_file, "Reconstruction: Iterative Projection Algorithm (PROJECTION_ALGORITHM_CSTRING), "	);	// 101
	fprintf(execution_log_file, "Reconstruction: Data Transfer Mode (RECON_TX_MODE), "				);	// 102
	fprintf(execution_log_file, "Reconstruction: # Iterations (ITERATIONS), "					);	// 103
	fprintf(execution_log_file, "Reconstruction: DROP Block Size (DROP_BLOCK_SIZE), "				);	// 104
	fprintf(execution_log_file, "Reconstruction: Relaxation Parameter (LAMBDA), "						);	// 105
	//fprintf(execution_log_file, "Reconstruction: Bound Image (BOUND_IMAGE), "					);	// 106
	
	fprintf(execution_log_file, "Robust Reconstruction: Algorithm (ROBUST_METHOD), "				);	// 107
	fprintf(execution_log_file, "Robust Reconstruction: Perturbation Magnitude (ETA), "							);	// 108
	fprintf(execution_log_file, "Robust Reconstruction: Perturbation Sign (PSI_SIGN), "					);	// 109
	
	fprintf(execution_log_file, "Reconstruction: Bound Image (BOUND_IMAGE), "					);	// 106
	fprintf(execution_log_file, "Reconstruction: Identify Air Pockets in Initial Iterate (IDENTIFY_X_0_AIR_CSTRING), "					);	// 106
	fprintf(execution_log_file, "Reconstruction: RSP Threshold Applied to Initial Iterate (X_0_AIR_THRESHOLD), "					);	// 106
	fprintf(execution_log_file, "Reconstruction: Identify Air Pockets in Each Iterate (IDENTIFY_X_N_AIR_CSTRING), "					);	// 106
	fprintf(execution_log_file, "Reconstruction: RSP Threshold Applied to Each Iterate (X_N_AIR_THRESHOLD), "					);	// 106
	
	fprintf(execution_log_file, "S-Curve Attenutation: Method (S_CURVE), "						);	// 110
	fprintf(execution_log_file, "S-Curve Attenutation: ON/OFF (S_CURVE_ON), "					);	// 111
	fprintf(execution_log_file, "S-Curve Attenutation: Attenuate Exit ON/OFF (DUAL_SIDED_S_CURVE), "			);	// 112
	fprintf(execution_log_file, "S-Curve Attenutation: Sigmoid Curve Steepness (SIGMOID_STEEPNESS), "			);	// 113
	fprintf(execution_log_file, "S-Curve Attenutation: Sigmoid Curve Midpoint (SIGMOID_MID_SHIFT), "			);	// 114
	
	fprintf(execution_log_file, "Total Variation Superiorization (TVS): ON/OFF (TVS_ON), "						);	// 115
	fprintf(execution_log_file, "Total Variation Superiorization (TVS): TVS Before Feasibility Seeking (TVS_FIRST), "					);	// 116
	fprintf(execution_log_file, "Total Variation Superiorization (TVS): Use Parallel Version (TVS_PARALLEL), "				);	// 117
	fprintf(execution_log_file, "Total Variation Superiorization (TVS): Require Total Variation (TV) Reduction (TVS_CONDITIONED), "				);	// 118
	fprintf(execution_log_file, "Total Variation Superiorization (TVS): # TVS Repetetions (TVS_REPETITIONS), "				);	// 119
	fprintf(execution_log_file, "Total Variation Superiorization (TVS): Perturbation Coefficient Initial Value (BETA_0), "						);	// 120
	fprintf(execution_log_file, "Total Variation Superiorization (TVS): Perturbation Coefficient Kernel (A), "							);	// 121
	fprintf(execution_log_file, "Total Variation Superiorization (TVS): Perturbation Coefficient Initial Kernel Power (L_0), "							);	// 122
	fprintf(execution_log_file, "\n"							);	// end line, go to beginning of next entry
	fclose(execution_log_file);

	print_section_exit( "Finished initializing global execution log", SECTION_EXIT_CSTRING, RED_TEXT, RED_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );	
}
void write_TV_measurements()
{
	print_section_header("Writing total variation measurements to the local output data directory", MINOR_SECTION_SEPARATOR, LIGHT_CYAN_TEXT, YELLOW_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT );	
	sprintf(print_statement, "Writing %d total variation (TV) measurements to:", TV_x_values.size());
	print_colored_text(print_statement, CYAN_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );
	print_colored_text(LOCAL_TV_MEASUREMENTS_PATH, LIGHT_PURPLE_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT );
	FILE* TV_x_values_file = fopen( LOCAL_TV_MEASUREMENTS_PATH, "w" );
	for( int i = 0; i < TV_x_values.size(); i++ )
		fprintf(TV_x_values_file, "%6.6lf\n",	TV_x_values[i] );	// 1
	fclose(TV_x_values_file);
	OUTPUT_FILE_LIST.push_back(std::string(LOCAL_TV_MEASUREMENTS_PATH));
	LOCAL_DATA_FILE_LIST.push_back(std::string(LOCAL_TV_MEASUREMENTS_PATH));	
}
void apply_permissions()
{
	//std::string current_compute_node_alias_str = colored_text(CURRENT_COMPUTE_NODE_ALIAS, LIGHT_PURPLE_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT );
	//sprintf(print_statement, "Setting file permissions of output data/images on %s and the network-attached storage device", current_compute_node_alias_str.c_str());
	sprintf(print_statement, "Setting file permissions of output data/images on %s and the network-attached storage device", CURRENT_COMPUTE_NODE_ALIAS);	
	print_section_header(print_statement, MINOR_SECTION_SEPARATOR, LIGHT_CYAN_TEXT, BROWN_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT );
	std::string terminal_string = terminal_response(HOSTNAME_CMD);
	terminal_string.pop_back();
	if( terminal_string.compare(workstation_2_hostname) == 0 )
		set_file_permissions( "/home/share/*", GLOBAL_ACCESS);
	else
	{
		set_file_permissions(LOCAL_EXECUTION_LOG_PATH, GLOBAL_ACCESS);
		set_file_permissions(GLOBAL_EXECUTION_LOG_PATH, GLOBAL_ACCESS);
		set_file_permissions(LOCAL_OUTPUT_DATA_PATH, GLOBAL_ACCESS);
		set_file_permissions(GLOBAL_OUTPUT_DATA_PATH, GLOBAL_ACCESS);	
	}
	print_section_exit( "Finished applying file permissions", SECTION_EXIT_CSTRING, RED_TEXT, RED_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );	
}
void cp_output_2_kodiak()
{
	//print_section_header( "Copying executed code, output data files and images, and the updated execution log to the network-attached storage device", MAJOR_SECTION_SEPARATOR, GREEN_TEXT, YELLOW_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT );
	/********************************************************************************************************************************************************/
	/*								Copy the execution log with newly added entry and overwrite the existing execution log on Kodiak						*/
	/********************************************************************************************************************************************************/				
	print_section_header("Updating global execution log from the network-attached storage device", MINOR_SECTION_SEPARATOR, LIGHT_CYAN_TEXT, YELLOW_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT );	
	execution_log_2_csv();	// Add an entry for this execution to the global execution log on current computation node
	/********************************************************************************************************************************************************/
	/*							Copy directory containing newly generated preprocessing data and reconstructed images to Kodiak								*/
	/********************************************************************************************************************************************************/				
	print_section_header("Copying reconstruction results to the network-attached storage device", MINOR_SECTION_SEPARATOR, LIGHT_CYAN_TEXT, YELLOW_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT );	
	copy_folder_contents(BASH_COPY_DIR, LOCAL_OUTPUT_DATA_PATH, GLOBAL_OUTPUT_DATA_PATH);
	print_section_exit( "Finshed copying local output data to the network-attached storage device", SECTION_EXIT_CSTRING, RED_TEXT, RED_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );	
	/********************************************************************************************************************************************************/
	/*							Copy directory containing reconstruction code compiled/executed to generate reconstruction results to Kodiak								*/
	/********************************************************************************************************************************************************/				
	print_section_header("Copying executed code to the network-attached storage device...", MINOR_SECTION_SEPARATOR, LIGHT_CYAN_TEXT, YELLOW_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT );	
	copy_data(BASH_COPY_DIR, EXECUTED_SRC_CODE_PATH, GLOBAL_OUTPUT_SRC_CODE_PATH);		
	copy_data(BASH_COPY_DIR, EXECUTED_INCLUDE_CODE_PATH, GLOBAL_OUTPUT_INCLUDE_CODE_PATH);		
	print_section_exit( "cp data transfers complete", SECTION_EXIT_CSTRING, RED_TEXT, RED_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );			
}
void scp_output_2_kodiak()
{
	char scp_command[256];								
	char execution_log_path[256];
	print_section_header( "Updating execution log and copying it and output data to the network-attached storage device", MINOR_SECTION_SEPARATOR, LIGHT_CYAN_TEXT, YELLOW_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT );
	/********************************************************************************************************************************************************/
	/*							Copy directory containing newly generated preprocessing data and reconstructed images to Kodiak								*/
	/********************************************************************************************************************************************************/				
	print_colored_text("Copying (scp) reconstruction results to the network-attached storage device...", CYAN_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );	
	sprintf(scp_command, "%s %s%s %s//%s//", BASH_SECURE_COPY, OUTPUT_DIRECTORY, OUTPUT_FOLDER_UNIQUE, KODIAK_SSH_LOGIN, CURRENT_RECON_DIR);		
	change_text_color( LIGHT_PURPLE_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT, false);
	system(scp_command);
	change_text_color( LIGHT_PURPLE_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT, true);
	/********************************************************************************************************************************************************/
	/*								Copy the execution log with newly added entry and overwrite the existing execution log on Kodiak						*/
	/********************************************************************************************************************************************************/				
	print_colored_text("Copying (scp) updated execution log to the network-attached storage device...", CYAN_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );
	sprintf(execution_log_path, "%s%s.csv", OUTPUT_DIRECTORY, EXECUTION_LOG_BASENAME);
	sprintf(scp_command, "%s %s %s%s//", BASH_SECURE_COPY, execution_log_path, KODIAK_SSH_LOGIN, CURRENT_RECON_DIR);
	change_text_color( LIGHT_PURPLE_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT, false);
	system(scp_command);
	change_text_color( LIGHT_PURPLE_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT, true);
	/********************************************************************************************************************************************************/
	/*							Copy directory containing reconstruction code compiled/executed to generate reconstruction results to Kodiak								*/
	/********************************************************************************************************************************************************/				
	print_section_header( "Copying reconstruction code and reconstructed data/images to the network-attached storage device", MINOR_SECTION_SEPARATOR, LIGHT_CYAN_TEXT, YELLOW_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT );
	print_colored_text("Copying (scp) code compiled/executed to generate reconstruction results to the network-attached storage device...", CYAN_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );	
	sprintf(scp_command, "%s %s//%s %s%s//%s", BASH_SECURE_COPY, TARDIS_RCODE_PARENT_SET, INCLUDE_CODE_FOLDER, KODIAK_SSH_LOGIN, CURRENT_RECON_DIR, OUTPUT_FOLDER_UNIQUE);		
	change_text_color( LIGHT_PURPLE_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT, false);
	system(scp_command);
	sprintf(scp_command, "%s %s//%s %s%s//%s", BASH_SECURE_COPY, TARDIS_RCODE_PARENT_SET, SRC_CODE_FOLDER, KODIAK_SSH_LOGIN, CURRENT_RECON_DIR, OUTPUT_FOLDER_UNIQUE);		
	system(scp_command);
	change_text_color( LIGHT_PURPLE_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT, true);
	/********************************************************************************************************************************************************/
	/*							Copy directory containing reconstruction code compiled/executed to generate reconstruction results to Kodiak								*/
	/********************************************************************************************************************************************************/					
	print_section_exit( "scp data transfers complete", SECTION_EXIT_CSTRING, RED_TEXT, RED_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );			
}
void program_completion_tasks()
{
		/********************************************************************************************************************************************************/
		/*																PROGRAM COMPLETION TASKS																*/
		/********************************************************************************************************************************************************/				
		if( WRITE_X ) 
			array_2_disk(X_FILENAME, OUTPUT_DIRECTORY, OUTPUT_FOLDER_UNIQUE, x_h, COLUMNS, ROWS, SLICES, NUM_VOXELS, true );
		execution_time_program = timer( STOP, begin_program, "for entire program");	
		/********************************************************************************************************************************************************/
		/*																PROGRAM COMPLETION TASKS																*/
		/********************************************************************************************************************************************************/				
		print_section_header( "Output data management tasks", MAJOR_SECTION_SEPARATOR, LIGHT_GREEN_TEXT, BROWN_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT );
		timer( START, begin_data_tx, "for data management tasks");	
		write_TV_measurements();
		execution_log_2_txt();										// Write execution options/settings and timing .txt to output recon directory
		//execution_log_2_csv();										// Add an entry for this execution to the global execution log on current computation node
		//apply_permissions();										// Change file permissions of newly generated reconstruction data/images
		cp_output_2_kodiak();										// Copy newly generated reconstruction output directory and contents to /ion
		//scp_output_2_kodiak();									// Copy newly generated reconstruction output directory and contents to Kodiak
		apply_permissions();										// Change file permissions of newly generated reconstruction data/images
		execution_time_data_tx = timer( STOP, begin_data_tx, "for data management tasks");	
		/*for(int i = 0; i < OUTPUT_FILE_LIST.size(); i++)
			cout << OUTPUT_FILE_LIST[i] << endl;
		cout  << endl;
		for(int i = 0; i < LOCAL_DATA_FILE_LIST.size(); i++)
			cout << LOCAL_DATA_FILE_LIST[i] << endl;
		cout  << endl;
		for(int i = 0; i < GLOBAL_DATA_FILE_LIST.size(); i++)
			cout << GLOBAL_DATA_FILE_LIST[i] << endl;	
		cout  << endl;*/
		//resize_vectors(0);
		//shrink_vectors();
		//first_MLP_voxel_vector.resize(0);	
		//first_MLP_voxel_vector.shrink_to_fit();

		//free(bin_num_h);
		//free(WEPL_h);
		//free(xy_entry_angle_h);
		//free(xz_entry_angle_h);
		//free(xy_exit_angle_h);
		//free(xz_exit_angle_h);
		//free(failed_cuts_h);
		//}
}
/***********************************************************************************************************************************************************************************************************************/
/************************************************************************************** Memory Transfers, Maintenance, and Cleaning ************************************************************************************/
/***********************************************************************************************************************************************************************************************************************/
void initializations()
{
	print_colored_text("Allocating statistical analysis arrays on host/GPU...", CYAN_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT);

	bin_counts_h		  = (int*)	 calloc( NUM_BINS, sizeof(int)	 );
	mean_WEPL_h			  = (float*) calloc( NUM_BINS, sizeof(float) );
	mean_rel_ut_angle_h	  = (float*) calloc( NUM_BINS, sizeof(float) );
	mean_rel_uv_angle_h	  = (float*) calloc( NUM_BINS, sizeof(float) );
	
	if( ( bin_counts_h == NULL ) || (mean_WEPL_h == NULL) || (mean_rel_ut_angle_h == NULL) || (mean_rel_uv_angle_h == NULL) )
	{
		puts("std dev allocation error\n");
		exit(1);
	}

	cudaMalloc((void**) &bin_counts_d,			SIZE_BINS_INT );
	cudaMalloc((void**) &mean_WEPL_d,			SIZE_BINS_FLOAT );
	cudaMalloc((void**) &mean_rel_ut_angle_d,	SIZE_BINS_FLOAT );
	cudaMalloc((void**) &mean_rel_uv_angle_d,	SIZE_BINS_FLOAT );

	cudaMemcpy( bin_counts_d,			bin_counts_h,			SIZE_BINS_INT,		cudaMemcpyHostToDevice );
	cudaMemcpy( mean_WEPL_d,			mean_WEPL_h,			SIZE_BINS_FLOAT,	cudaMemcpyHostToDevice );
	cudaMemcpy( mean_rel_ut_angle_d,	mean_rel_ut_angle_h,	SIZE_BINS_FLOAT,	cudaMemcpyHostToDevice );
	cudaMemcpy( mean_rel_uv_angle_d,	mean_rel_uv_angle_h,	SIZE_BINS_FLOAT,	cudaMemcpyHostToDevice );
}
void reserve_vector_capacity()
{
	// Reserve enough memory for vectors to hold all histories.  If a vector grows to the point where the next memory address is already allocated to another
	// object, the system must first move the vector to a new location in memory which can hold the existing vector and new element.  The eventual size of these
	// vectors is quite large and the possibility of this happening is high for one or more vectors and it can happen multiple times as the vector grows.  Moving 
	// a vector and its contents is a time consuming process, especially as it becomes large, so we reserve enough memory to guarantee this does not happen.
	print_colored_text( "Reserving memory for preprocessing/reconstruction vectors...", CYAN_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );
	bin_num_vector.reserve( total_histories );
	//gantry_angle_vector.reserve( total_histories );
	WEPL_vector.reserve( total_histories );
	x_entry_vector.reserve( total_histories );
	y_entry_vector.reserve( total_histories );
	z_entry_vector.reserve( total_histories );
	x_exit_vector.reserve( total_histories );
	y_exit_vector.reserve( total_histories );
	z_exit_vector.reserve( total_histories );
	xy_entry_angle_vector.reserve( total_histories );
	xz_entry_angle_vector.reserve( total_histories );
	xy_exit_angle_vector.reserve( total_histories );
	xz_exit_angle_vector.reserve( total_histories );
}
void initial_processing_memory_clean()
{
	//clear_input_memory
	//free( missed_recon_volume_h );
	free( gantry_angle_h );
	cudaFree( x_entry_d );
	cudaFree( y_entry_d );
	cudaFree( z_entry_d );
	cudaFree( x_exit_d );
	cudaFree( y_exit_d );
	cudaFree( z_exit_d );
	cudaFree( missed_recon_volume_d );
	cudaFree( bin_num_d );
	cudaFree( WEPL_d);
}
void resize_vectors( unsigned int new_size )
{
	bin_num_vector.resize( new_size );
	//gantry_angle_vector.resize( new_size );
	WEPL_vector.resize( new_size );
	x_entry_vector.resize( new_size );	
	y_entry_vector.resize( new_size );	
	z_entry_vector.resize( new_size );
	x_exit_vector.resize( new_size );
	y_exit_vector.resize( new_size );
	z_exit_vector.resize( new_size );
	xy_entry_angle_vector.resize( new_size );	
	xz_entry_angle_vector.resize( new_size );	
	xy_exit_angle_vector.resize( new_size );
	xz_exit_angle_vector.resize( new_size );
}
void shrink_vectors()
{
	bin_num_vector.shrink_to_fit();
	gantry_angle_vector.shrink_to_fit();
	WEPL_vector.shrink_to_fit();
	x_entry_vector.shrink_to_fit();	
	y_entry_vector.shrink_to_fit();	
	z_entry_vector.shrink_to_fit();	
	x_exit_vector.shrink_to_fit();	
	y_exit_vector.shrink_to_fit();	
	z_exit_vector.shrink_to_fit();	
	xy_entry_angle_vector.shrink_to_fit();	
	xz_entry_angle_vector.shrink_to_fit();	
	xy_exit_angle_vector.shrink_to_fit();	
	xz_exit_angle_vector.shrink_to_fit();	
}
void data_shift_vectors( unsigned int read_index, unsigned int write_index )
{
	bin_num_vector[write_index] = bin_num_vector[read_index];
	//gantry_angle_vector[write_index]  = gantry_angle_vector[read_index];
	WEPL_vector[write_index] = WEPL_vector[read_index];
	x_entry_vector[write_index] = x_entry_vector[read_index];
	y_entry_vector[write_index] = y_entry_vector[read_index];
	z_entry_vector[write_index] = z_entry_vector[read_index];
	x_exit_vector[write_index] = x_exit_vector[read_index];
	y_exit_vector[write_index] = y_exit_vector[read_index];
	z_exit_vector[write_index] = z_exit_vector[read_index];
	xy_entry_angle_vector[write_index] = xy_entry_angle_vector[read_index];
	xz_entry_angle_vector[write_index] = xz_entry_angle_vector[read_index];
	xy_exit_angle_vector[write_index] = xy_exit_angle_vector[read_index];
	xz_exit_angle_vector[write_index] = xz_exit_angle_vector[read_index];
}
void initialize_stddev()
{	
	stddev_rel_ut_angle_h = (float*) calloc( NUM_BINS, sizeof(float) );	
	stddev_rel_uv_angle_h = (float*) calloc( NUM_BINS, sizeof(float) );	
	stddev_WEPL_h		  = (float*) calloc( NUM_BINS, sizeof(float) );
	if( ( stddev_rel_ut_angle_h == NULL ) || (stddev_rel_uv_angle_h == NULL) || (stddev_WEPL_h == NULL) )
	{
		puts("std dev allocation error\n");
		exit(1);
	}
	cudaMalloc((void**) &stddev_rel_ut_angle_d,	SIZE_BINS_FLOAT );
	cudaMalloc((void**) &stddev_rel_uv_angle_d,	SIZE_BINS_FLOAT );
	cudaMalloc((void**) &stddev_WEPL_d,			SIZE_BINS_FLOAT );

	cudaMemcpy( stddev_rel_ut_angle_d,	stddev_rel_ut_angle_h,	SIZE_BINS_FLOAT,	cudaMemcpyHostToDevice );
	cudaMemcpy( stddev_rel_uv_angle_d,	stddev_rel_uv_angle_h,	SIZE_BINS_FLOAT,	cudaMemcpyHostToDevice );
	cudaMemcpy( stddev_WEPL_d,			stddev_WEPL_h,			SIZE_BINS_FLOAT,	cudaMemcpyHostToDevice );
}
void post_cut_memory_clean()
{
	//char statement[] = "Freeing unnecessary memory, resizing vectors, and shrinking vectors to fit just the remaining histories...";
	sprintf(print_statement, "Freeing unnecessary memory, resizing vectors, and shrinking vectors to fit just the remaining histories...");
	print_colored_text( print_statement, CYAN_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );
	//free(failed_cuts_h );
	free(stddev_rel_ut_angle_h);
	free(stddev_rel_uv_angle_h);
	free(stddev_WEPL_h);

	//cudaFree( failed_cuts_d );
	//cudaFree( bin_num_d );
	//cudaFree( WEPL_d );
	//cudaFree( xy_entry_angle_d );
	//cudaFree( xz_entry_angle_d );
	//cudaFree( xy_exit_angle_d );
	//cudaFree( xz_exit_angle_d );

	cudaFree( mean_rel_ut_angle_d );
	cudaFree( mean_rel_uv_angle_d );
	cudaFree( mean_WEPL_d );
	cudaFree( stddev_rel_ut_angle_d );
	cudaFree( stddev_rel_uv_angle_d );
	cudaFree( stddev_WEPL_d );
}
/***********************************************************************************************************************************************************************************************************************/
/**************************************************************************************** Preprocessing setup and initializations **************************************************************************************/
/***********************************************************************************************************************************************************************************************************************/
void count_histories()
{
	//char statement[256];
	print_colored_text("Counting proton histories...", CYAN_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );		
	for( int scan_number = 0; scan_number < NUM_SCANS; scan_number++ )
		histories_per_scan[scan_number] = 0;

	histories_per_file =				 (int*) calloc( NUM_SCANS * GANTRY_ANGLES, sizeof(int) );
	histories_per_gantry_angle =		 (int*) calloc( GANTRY_ANGLES, sizeof(int) );
	recon_vol_histories_per_projection = (int*) calloc( GANTRY_ANGLES, sizeof(int) );

	print_colored_text("Counting proton histories...", CYAN_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );		
	switch( DATA_FORMAT )
	{
		case OLD_FORMAT : count_histories_old();	break;
		case VERSION_0  : count_histories_v0();		break;
		case VERSION_1  : count_histories_v1();		break;
	}
	/*if( DEBUG_TEXT_ON )
	{
		for( int file_number = 0, gantry_position_number = 0; file_number < (NUM_SCANS * GANTRY_ANGLES); file_number++, gantry_position_number++ )
		{
			if( file_number % NUM_SCANS == 0 )
				printf("There are a total of %d histories from gantry angle %d\n", histories_per_gantry_angle[gantry_position_number], int(gantry_position_number* GANTRY_ANGLE_INTERVAL) );			
			printf("------> %d Histories are From Scan Number %d\n", histories_per_file[file_number], (file_number % NUM_SCANS) + 1 );
			
		}
		for( int scan_number = 0; scan_number < NUM_SCANS; scan_number++ )
			printf("There are a total of %d histories in Scan Number %d \n", histories_per_scan[scan_number], scan_number + 1);
		printf("There are a total of %d histories\n", total_histories);
	}*/
	//print_section_exit( "Finished counting proton histories", SECTION_EXIT_CSTRING, RED_TEXT, RED_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );
}
void count_histories_old()
{
	//char user_response[20];
	char data_filename[128];
	int file_size, num_histories, file_number = 0, gantry_position_number = 0;
	for( int gantry_angle = 0; gantry_angle < 360; gantry_angle += int(GANTRY_ANGLE_INTERVAL), gantry_position_number++ )
	{
		for( int scan_number = 1; scan_number <= NUM_SCANS; scan_number++, file_number++ )
		{
			
			sprintf( data_filename, "%s%s/%s_trans%d_%03d%s", INPUT_DIRECTORY, INPUT_FOLDER, PROJECTION_DATA_BASENAME, scan_number, gantry_angle, PROJECTION_DATA_EXTENSION );
			FILE *data_file = fopen(data_filename, "rb");
			if( data_file == NULL )
			{
				fputs( "Error Opening Data File:  Check that the directories are properly named.", stderr ); 
				exit_program_if(true);
			}
			fseek( data_file, 0, SEEK_END );
			file_size = ftell( data_file );
			if( BINARY_ENCODING )
			{
				if( file_size % BYTES_PER_HISTORY ) 
				{
					printf("ERROR! Problem with bytes_per_history!\n");
					exit_program_if(true);
				}
				num_histories = file_size / BYTES_PER_HISTORY;	
			}
			else
				num_histories = file_size;							
			fclose(data_file);
			histories_per_file[file_number] = num_histories;
			histories_per_gantry_angle[gantry_position_number] += num_histories;
			histories_per_scan[scan_number-1] += num_histories;
			total_histories += num_histories;
			
			if( DEBUG_TEXT_ON )
				printf("There are %d Histories for Gantry Angle %d From Scan Number %d\n", num_histories, gantry_angle, scan_number);
		}
	}
}
void count_histories_v0()
{
	char data_filename[256];
	float projection_angle;
	unsigned int num_histories, file_number = 0, gantry_position_number = 0;
	char magic_number_string[5];
	for( unsigned int gantry_angle = 0; gantry_angle < 360; gantry_angle += int(GANTRY_ANGLE_INTERVAL), gantry_position_number++ )
	{
		for( unsigned int scan_number = 1; scan_number <= NUM_SCANS; scan_number++, file_number++ )
		{
			sprintf(data_filename, "%s%s/%s_%03d%s", INPUT_DIRECTORY, INPUT_FOLDER, PROJECTION_DATA_BASENAME, gantry_angle, PROJECTION_DATA_EXTENSION  );
			/*
			Contains the following headers:
				Magic number identifier: "PCTD" (4-byte string)
				Format version identifier (integer)
				Number of events in file (integer)
				Projection angle (float | degrees)
				Beam energy (float | MeV)
				Acquisition/generation date (integer | Unix time)
				Pre-process date (integer | Unix time)
				Phantom name or description (variable length string)
				Data source (variable length string)
				Prepared by (variable length string)
				* Note on variable length strings: each variable length string should be preceded with an integer containing the number of characters in the string.			
			*/
			FILE* data_file = fopen(data_filename, "rb");
			if( data_file == NULL )
			{
				fputs( "Error Opening Data File:  Check that the directories are properly named.", stderr ); 
				exit_program_if(true);
			}
			fread(&magic_number_string, 4, 1, data_file );
			magic_number_string[4] = '\0';
			if( strcmp( magic_number_string, "PCTD" ) != 0 ) 
			{
				//puts(magic_number);
				puts("Error: unknown file type (should be PCTD)!\n");
				exit_program_if(true);
			}
			fread(&VERSION_ID, sizeof(int), 1, data_file );			
			if( VERSION_ID == 0 )
			{
				fread(&num_histories, sizeof(int), 1, data_file );
				histories_per_file[file_number] = num_histories;
				histories_per_gantry_angle[gantry_position_number] += num_histories;
				histories_per_scan[scan_number-1] += num_histories;
				total_histories += num_histories;
			
				fread(&projection_angle, sizeof(float), 1, data_file );
				projection_angles.push_back(projection_angle);

				fseek( data_file, 2 * sizeof(int) + sizeof(float), SEEK_CUR );
				fread(&PHANTOM_NAME_SIZE, sizeof(int), 1, data_file );

				fseek( data_file, PHANTOM_NAME_SIZE, SEEK_CUR );
				fread(&DATA_SOURCE_SIZE, sizeof(int), 1, data_file );

				fseek( data_file, DATA_SOURCE_SIZE, SEEK_CUR );
				fread(&PREPARED_BY_SIZE, sizeof(int), 1, data_file );

				fseek( data_file, PREPARED_BY_SIZE, SEEK_CUR );
				fclose(data_file);
				SKIP_2_DATA_SIZE = 4 + 7 * sizeof(int) + 2 * sizeof(float) + PHANTOM_NAME_SIZE + DATA_SOURCE_SIZE + PREPARED_BY_SIZE;
			}
			else if( VERSION_ID == 1 )
			{
				fread(&num_histories, sizeof(int), 1, data_file );
				//if( DEBUG_TEXT_ON )
				//	printf("There are %d Histories for Gantry Angle %d From Scan Number %d\n", num_histories, gantry_angle, scan_number);
				histories_per_file[file_number] = num_histories;
				histories_per_gantry_angle[gantry_position_number] += num_histories;
				histories_per_scan[scan_number-1] += num_histories;
				total_histories += num_histories;
			
				fread(&projection_angle, sizeof(float), 1, data_file );
				projection_angles.push_back(projection_angle);

				fseek( data_file, 2 * sizeof(int) + sizeof(float), SEEK_CUR );
				fread(&PHANTOM_NAME_SIZE, sizeof(int), 1, data_file );

				fseek( data_file, PHANTOM_NAME_SIZE, SEEK_CUR );
				fread(&DATA_SOURCE_SIZE, sizeof(int), 1, data_file );

				fseek( data_file, DATA_SOURCE_SIZE, SEEK_CUR );
				fread(&PREPARED_BY_SIZE, sizeof(int), 1, data_file );

				fseek( data_file, PREPARED_BY_SIZE, SEEK_CUR );
				fclose(data_file);
				SKIP_2_DATA_SIZE = 4 + 7 * sizeof(int) + 2 * sizeof(float) + PHANTOM_NAME_SIZE + DATA_SOURCE_SIZE + PREPARED_BY_SIZE;
			}
			else 
			{
				printf("ERROR: Data format is not Version (%d)!\n", VERSION_ID);
				exit_program_if(true);
			}						
		}
	}
}
void count_histories_v02()
{
	char data_filename[256];
	int num_histories, file_number = 0, gantry_position_number = 0;
	for( int gantry_angle = 0; gantry_angle < 360; gantry_angle += int(GANTRY_ANGLE_INTERVAL), gantry_position_number++ )
	{
		for( int scan_number = 1; scan_number <= NUM_SCANS; scan_number++, file_number++ )
		{
			sprintf(data_filename, "%s%s/%s_%03d%s", INPUT_DIRECTORY, INPUT_FOLDER, PROJECTION_DATA_BASENAME, gantry_angle, PROJECTION_DATA_EXTENSION  );
			std::ifstream data_file(data_filename, std::ios::binary);
			if( data_file == NULL )
			{
				fputs( "File not found:  Check that the directories and files are properly named.", stderr ); 
				exit_program_if(true);
			}
			char magic_number[5];
			data_file.read(magic_number, 4);
			magic_number[4] = '\0';
			if( strcmp(magic_number, "PCTD") ) {
				puts("Error: unknown file type (should be PCTD)!\n");
				exit_program_if(true);
			}
			int version_id;
			data_file.read((char*)&version_id, sizeof(int));
			if( version_id == 0 )
			{
				data_file.read((char*)&num_histories, sizeof(int));						
				data_file.close();
				histories_per_file[file_number] = num_histories;
				histories_per_gantry_angle[gantry_position_number] += num_histories;
				histories_per_scan[scan_number-1] += num_histories;
				total_histories += num_histories;			
				printf("There are %d Histories for Gantry Angle %d From Scan Number %d\n", num_histories, gantry_angle, scan_number);
			}
			else 
			{
				printf("ERROR: Data format is not Version (%d)!\n", version_id);
				exit_program_if(true);
			}			
		}
	}
}
void count_histories_v1()
{
	//char user_response[20];
	char data_filename[256];
	int num_histories, file_number = 0, gantry_position_number = 0;
	for( int gantry_angle = 0; gantry_angle < 360; gantry_angle += int(GANTRY_ANGLE_INTERVAL), gantry_position_number++ )
	{
		for( int scan_number = 1; scan_number <= NUM_SCANS; scan_number++, file_number++ )
		{
			sprintf(data_filename, "%s%s/%s_%03d%s", INPUT_DIRECTORY, INPUT_FOLDER, PROJECTION_DATA_BASENAME, gantry_angle, PROJECTION_DATA_EXTENSION  );
			std::ifstream data_file(data_filename, std::ios::binary);
			if( data_file == NULL )
			{
				fputs( "File not found:  Check that the directories and files are properly named.", stderr ); 
				exit_program_if(true);
			}
			char magic_number[5];
			data_file.read(magic_number, 4);
			magic_number[4] = '\0';
			if( strcmp(magic_number, "PCTD") ) {
				puts("Error: unknown file type (should be PCTD)!\n");
				exit_program_if(true);
			}
			int version_id;
			data_file.read((char*)&version_id, sizeof(int));
			if( version_id == 1 )
			{
				data_file.read((char*)&num_histories, sizeof(int));						
				data_file.close();
				histories_per_file[file_number] = num_histories;
				histories_per_gantry_angle[gantry_position_number] += num_histories;
				histories_per_scan[scan_number-1] += num_histories;
				total_histories += num_histories;
			
				if( DEBUG_TEXT_ON )
					printf("There are %d Histories for Gantry Angle %d From Scan Number %d\n", num_histories, gantry_angle, scan_number);
			}
			else 
			{
				printf("ERROR: Data format is not Version 1 (Version %d detected)!\n", version_id);
				exit_program_if(true);
			}			
		}
	}
}
/***********************************************************************************************************************************************************************************************************************/
/******************************************************************************************* Image initialization/Construction *****************************************************************************************/
/***********************************************************************************************************************************************************************************************************************/
template<typename T> void initialize_host_image( T*& image )
{
	image = (T*)calloc( NUM_VOXELS, sizeof(T));
}
template<typename T> void add_ellipse( T*& image, int slice, double x_center, double y_center, double semi_major_axis, double semi_minor_axis, T value )
{
	double x, y;
	for( int row = 0; row < ROWS; row++ )
	{
		for( int column = 0; column < COLUMNS; column++ )
		{
			x = ( column - COLUMNS/2 + 0.5) * VOXEL_WIDTH;
			y = ( ROWS/2 - row - 0.5 ) * VOXEL_HEIGHT;
			if( pow( ( x - x_center) / semi_major_axis, 2 ) + pow( ( y - y_center )  / semi_minor_axis, 2 ) <= 1 )
				image[slice * COLUMNS * ROWS + row * COLUMNS + column] = value;
		}
	}
}
template<typename T> void add_circle( T*& image, int slice, double x_center, double y_center, double radius, T value )
{
	double x, y;
	for( int row = 0; row < ROWS; row++ )
	{
		for( int column = 0; column < COLUMNS; column++ )
		{
			x = ( column - COLUMNS/2 + 0.5) * VOXEL_WIDTH;
			//x_center = ( center_column - COLUMNS/2 + 0.5) * VOXEL_WIDTH;
			y = ( ROWS/2 - row - 0.5 ) * VOXEL_HEIGHT;
			//y_center = ( center_row - COLUMNS/2 + 0.5) * VOXEL_WIDTH;
			if( pow( (x - x_center), 2 ) + pow( (y - y_center), 2 ) <= pow( radius, 2) )
				image[slice * COLUMNS * ROWS + row * COLUMNS + column] = value;
		}
	}
}	
template<typename O> void import_image( O*& import_into, char* filename )
{
	FILE* input_file = fopen(filename, "rb" );
	O* temp = (O*)calloc(NUM_VOXELS, sizeof(O) );
	fread(temp, sizeof(O), NUM_VOXELS, input_file );
	free(import_into);
	import_into = temp;
}
template<typename O> void import_text_image( O*& import_into, char* filename )
{
	std::ifstream input_file;
	input_file.open(filename);		
	for(int i=0; i < NUM_VOXELS; i++)
		input_file >> import_into[i];
	input_file.close();		
}
/***********************************************************************************************************************************************************************************************************************/
/************************************************************************************** Data importation, initial cuts, and binning ************************************************************************************/
/***********************************************************************************************************************************************************************************************************************/
void combine_data_sets()
{
	char input_filename1[256];
	char input_filename2[256];
	char output_filename[256];
	const char INPUT_FOLDER1[]	   = "input_CTP404";
	const char INPUT_FOLDER2[]	   = "CTP404_4M";
	const char MERGED_FOLDER[]	   = "my_merged";
	//unsigned int gantry_position, gantry_angle, scan_number, file_histories, array_index = 0, histories_read = 0; //Warning: Not used in function

	char magic_number1[4], magic_number2[4];
	int version_id1, version_id2;
	int file_histories1, file_histories2, combined_histories;

	float projection_angle1, beam_energy1;
	int generation_date1, preprocess_date1;
	int phantom_name_size1, data_source_size1, prepared_by_size1;
	char *phantom_name1, *data_source1, *prepared_by1;
	
	float projection_angle2, beam_energy2;
	int generation_date2, preprocess_date2;
	int phantom_name_size2, data_source_size2, prepared_by_size2;
	char *phantom_name2, *data_source2, *prepared_by2;

	float* t_in_1_h1, * t_in_1_h2, *t_in_2_h1, * t_in_2_h2; 
	float* t_out_1_h1, * t_out_1_h2, * t_out_2_h1, * t_out_2_h2;
	float* v_in_1_h1, * v_in_1_h2, * v_in_2_h1, * v_in_2_h2;
	float* v_out_1_h1, * v_out_1_h2, * v_out_2_h1, * v_out_2_h2;
	float* u_in_1_h1, * u_in_1_h2, * u_in_2_h1, * u_in_2_h2;
	float* u_out_1_h1, * u_out_1_h2, * u_out_2_h1, * u_out_2_h2;
	float* WEPL_h1, * WEPL_h2;

	for( unsigned int gantry_angle = 0; gantry_angle < 360; gantry_angle += int(GANTRY_ANGLE_INTERVAL) )
	{	
		cout << gantry_angle << endl;
		sprintf(input_filename1, "%s%s/%s_%03d%s", INPUT_DIRECTORY, INPUT_FOLDER1, PROJECTION_DATA_BASENAME, gantry_angle, PROJECTION_DATA_EXTENSION );
		sprintf(input_filename2, "%s%s/%s_%03d%s", INPUT_DIRECTORY, INPUT_FOLDER2, PROJECTION_DATA_BASENAME, gantry_angle, PROJECTION_DATA_EXTENSION );
		sprintf(output_filename, "%s%s/%s_%03d%s", INPUT_DIRECTORY, MERGED_FOLDER, PROJECTION_DATA_BASENAME, gantry_angle, PROJECTION_DATA_EXTENSION );

		printf("%s\n", input_filename1 );
		printf("%s\n", input_filename2 );
		printf("%s\n", output_filename );

		FILE* input_file1 = fopen(input_filename1, "rb");
		FILE* input_file2 = fopen(input_filename2, "rb");
		FILE* output_file = fopen(output_filename, "wb");

		if( (input_file1 == NULL) ||  (input_file2 == NULL)  || (output_file == NULL)  )
		{
			fputs( "Error Opening Data File:  Check that the directories are properly named.", stderr ); 
			exit_program_if(true);
		}

		fread(&magic_number1, sizeof(char), 4, input_file1 );
		fread(&magic_number2, sizeof(char), 4, input_file2 );
		fwrite( &magic_number1, sizeof(char), 4, output_file );
		//if( magic_number != MAGIC_NUMBER_CHECK ) 
		//{
		//	puts("Error: unknown file type (should be PCTD)!\n");
		//	exit_program_if(true);
		//}

		fread(&version_id1, sizeof(int), 1, input_file1 );
		fread(&version_id2, sizeof(int), 1, input_file2 );
		fwrite( &version_id1, sizeof(int), 1, output_file );

		fread(&file_histories1, sizeof(int), 1, input_file1 );
		fread(&file_histories2, sizeof(int), 1, input_file2 );
		combined_histories = file_histories1 + file_histories2;
		fwrite( &combined_histories, sizeof(int), 1, output_file );

		puts("Reading headers from files...\n");
	
		fread(&projection_angle1, sizeof(float), 1, input_file1 );
		fread(&projection_angle2, sizeof(float), 1, input_file2 );
		fwrite( &projection_angle1, sizeof(float), 1, output_file );
			
		fread(&beam_energy1, sizeof(float), 1, input_file1 );
		fread(&beam_energy2, sizeof(float), 1, input_file2 );
		fwrite( &beam_energy1, sizeof(float), 1, output_file );

		fread(&generation_date1, sizeof(int), 1, input_file1 );
		fread(&generation_date2, sizeof(int), 1, input_file2 );
		fwrite( &generation_date1, sizeof(int), 1, output_file );

		fread(&preprocess_date1, sizeof(int), 1, input_file1 );
		fread(&preprocess_date2, sizeof(int), 1, input_file2 );
		fwrite( &preprocess_date1, sizeof(int), 1, output_file );

		fread(&phantom_name_size1, sizeof(int), 1, input_file1 );
		fread(&phantom_name_size2, sizeof(int), 1, input_file2 );
		fwrite( &phantom_name_size1, sizeof(int), 1, output_file );

		phantom_name1 = (char*)malloc(phantom_name_size1);
		phantom_name2 = (char*)malloc(phantom_name_size2);

		fread(phantom_name1, phantom_name_size1, 1, input_file1 );
		fread(phantom_name2, phantom_name_size2, 1, input_file2 );
		fwrite( phantom_name1, phantom_name_size1, 1, output_file );

		fread(&data_source_size1, sizeof(int), 1, input_file1 );
		fread(&data_source_size2, sizeof(int), 1, input_file2 );
		fwrite( &data_source_size1, sizeof(int), 1, output_file );

		data_source1 = (char*)malloc(data_source_size1);
		data_source2 = (char*)malloc(data_source_size2);

		fread(data_source1, data_source_size1, 1, input_file1 );
		fread(data_source2, data_source_size2, 1, input_file2 );
		fwrite( &data_source1, data_source_size1, 1, output_file );

		fread(&prepared_by_size1, sizeof(int), 1, input_file1 );
		fread(&prepared_by_size2, sizeof(int), 1, input_file2 );
		fwrite( &prepared_by_size1, sizeof(int), 1, output_file );

		prepared_by1 = (char*)malloc(prepared_by_size1);
		prepared_by2 = (char*)malloc(prepared_by_size2);

		fread(prepared_by1, prepared_by_size1, 1, input_file1 );
		fread(prepared_by2, prepared_by_size2, 1, input_file2 );
		fwrite( &prepared_by1, prepared_by_size1, 1, output_file );

		puts("Reading data from files...\n");

		t_in_1_h1 = (float*)calloc( file_histories1, sizeof(float ) );
		t_in_1_h2 = (float*)calloc( file_histories2, sizeof(float ) );
		t_in_2_h1 = (float*)calloc( file_histories1, sizeof(float ) );
		t_in_2_h2 = (float*)calloc( file_histories2, sizeof(float ) );
		t_out_1_h1 = (float*)calloc( file_histories1, sizeof(float ) );
		t_out_1_h2 = (float*)calloc( file_histories2, sizeof(float ) );
		t_out_2_h1 = (float*)calloc( file_histories1, sizeof(float ) );
		t_out_2_h2 = (float*)calloc( file_histories2, sizeof(float ) );
		v_in_1_h1 = (float*)calloc( file_histories1, sizeof(float ) );
		v_in_1_h2 = (float*)calloc( file_histories2, sizeof(float ) );		
		v_in_2_h1 = (float*)calloc( file_histories1, sizeof(float ) );
		v_in_2_h2 = (float*)calloc( file_histories2, sizeof(float ) );
		v_out_1_h1 = (float*)calloc( file_histories1, sizeof(float ) );
		v_out_1_h2 = (float*)calloc( file_histories2, sizeof(float ) );
		v_out_2_h1 = (float*)calloc( file_histories1, sizeof(float ) );
		v_out_2_h2 = (float*)calloc( file_histories2, sizeof(float ) );
		u_in_1_h1 = (float*)calloc( file_histories1, sizeof(float ) );
		u_in_1_h2 = (float*)calloc( file_histories2, sizeof(float ) );
		u_in_2_h1 = (float*)calloc( file_histories1, sizeof(float ) );
		u_in_2_h2 = (float*)calloc( file_histories2, sizeof(float ) );
		u_out_1_h1 = (float*)calloc( file_histories1, sizeof(float ) );
		u_out_1_h2 = (float*)calloc( file_histories2, sizeof(float ) );
		u_out_2_h1 = (float*)calloc( file_histories1, sizeof(float ) );
		u_out_2_h2 = (float*)calloc( file_histories2, sizeof(float ) );
		WEPL_h1 = (float*)calloc( file_histories1, sizeof(float ) );
		WEPL_h2 = (float*)calloc( file_histories2, sizeof(float ) );

		fread( t_in_1_h1,  sizeof(float), file_histories1, input_file1 );
		fread( t_in_2_h1,  sizeof(float), file_histories1, input_file1 );
		fread( t_out_1_h1,  sizeof(float), file_histories1, input_file1 );
		fread( t_out_2_h1,  sizeof(float), file_histories1, input_file1 );
		fread( v_in_1_h1,  sizeof(float), file_histories1, input_file1 );
		fread( v_in_2_h1,  sizeof(float), file_histories1, input_file1 );
		fread( v_out_1_h1,  sizeof(float), file_histories1, input_file1 );
		fread( v_out_2_h1,  sizeof(float), file_histories1, input_file1 );
		fread( u_in_1_h1,  sizeof(float), file_histories1, input_file1 );
		fread( u_in_2_h1,  sizeof(float), file_histories1, input_file1 );
		fread( u_out_1_h1,  sizeof(float), file_histories1, input_file1 );
		fread( u_out_2_h1,  sizeof(float), file_histories1, input_file1 );
		fread( WEPL_h1,  sizeof(float), file_histories1, input_file1 );

		fread( t_in_1_h2,  sizeof(float), file_histories2, input_file2 );
		fread( t_in_2_h2,  sizeof(float), file_histories2, input_file2 );
		fread( t_out_1_h2,  sizeof(float), file_histories2, input_file2 );
		fread( t_out_2_h2,  sizeof(float), file_histories2, input_file2 );
		fread( v_in_1_h2,  sizeof(float), file_histories2, input_file2 );
		fread( v_in_2_h2,  sizeof(float), file_histories2, input_file2 );
		fread( v_out_1_h2,  sizeof(float), file_histories2, input_file2 );
		fread( v_out_2_h2,  sizeof(float), file_histories2, input_file2 );
		fread( u_in_1_h2,  sizeof(float), file_histories2, input_file2 );
		fread( u_in_2_h2,  sizeof(float), file_histories2, input_file2 );
		fread( u_out_1_h2,  sizeof(float), file_histories2, input_file2 );
		fread( u_out_2_h2,  sizeof(float), file_histories2, input_file2 );
		fread( WEPL_h2,  sizeof(float), file_histories2, input_file2 );

		fwrite( t_in_1_h1, sizeof(float), file_histories1, output_file );
		fwrite( t_in_1_h2, sizeof(float), file_histories2, output_file );		
		fwrite( t_in_2_h1, sizeof(float), file_histories1, output_file );
		fwrite( t_in_2_h2, sizeof(float), file_histories2, output_file );		
		fwrite( t_out_1_h1, sizeof(float), file_histories1, output_file );
		fwrite( t_out_1_h2, sizeof(float), file_histories2, output_file );		
		fwrite( t_out_2_h1, sizeof(float), file_histories1, output_file );
		fwrite( t_out_2_h2, sizeof(float), file_histories2, output_file );	

		fwrite( v_in_1_h1, sizeof(float), file_histories1, output_file );
		fwrite( v_in_1_h2, sizeof(float), file_histories2, output_file );		
		fwrite( v_in_2_h1, sizeof(float), file_histories1, output_file );
		fwrite( v_in_2_h2, sizeof(float), file_histories2, output_file );		
		fwrite( v_out_1_h1, sizeof(float), file_histories1, output_file );
		fwrite( v_out_1_h2, sizeof(float), file_histories2, output_file );		
		fwrite( v_out_2_h1, sizeof(float), file_histories1, output_file );
		fwrite( v_out_2_h2, sizeof(float), file_histories2, output_file );	

		fwrite( u_in_1_h1, sizeof(float), file_histories1, output_file );
		fwrite( u_in_1_h2, sizeof(float), file_histories2, output_file );		
		fwrite( u_in_2_h1, sizeof(float), file_histories1, output_file );
		fwrite( u_in_2_h2, sizeof(float), file_histories2, output_file );	
		fwrite( u_out_1_h1, sizeof(float), file_histories1, output_file );
		fwrite( u_out_1_h2, sizeof(float), file_histories2, output_file );	
		fwrite( u_out_2_h1, sizeof(float), file_histories1, output_file );
		fwrite( u_out_2_h2, sizeof(float), file_histories2, output_file );	

		fwrite( WEPL_h1, sizeof(float), file_histories1, output_file );
		fwrite( WEPL_h2, sizeof(float), file_histories2, output_file );
		
		free( t_in_1_h1 );
		free( t_in_1_h2 );
		free( t_in_2_h1 );
		free( t_in_2_h2 );
		free( t_out_1_h1 );
		free( t_out_1_h2 );
		free( t_out_2_h1 );
		free( t_out_2_h2 );

		free( v_in_1_h1 );
		free( v_in_1_h2 );
		free( v_in_2_h1 );
		free( v_in_2_h2 );
		free( v_out_1_h1 );
		free( v_out_1_h2 );
		free( v_out_2_h1 );
		free( v_out_2_h2 );

		free( u_in_1_h1 );
		free( u_in_1_h2 );
		free( u_in_2_h1 );
		free( u_in_2_h2 );
		free( u_out_1_h1 );
		free( u_out_1_h2 );
		free( u_out_2_h1 );
		free( u_out_2_h2 );

		free( WEPL_h1 );
		free( WEPL_h2 );

		fclose(input_file1);						
		fclose(input_file2);	
		fclose(output_file);	

		puts("Finished");
		pause_execution();
	}

}
void convert_mm_2_cm( unsigned int num_histories )
{
	for( unsigned int i = 0; i < num_histories; i++ ) 
	{
		// Convert the input data from mm to cm
		v_in_1_h[i]	 *= MM_TO_CM;
		v_in_2_h[i]	 *= MM_TO_CM;
		v_out_1_h[i] *= MM_TO_CM;
		v_out_2_h[i] *= MM_TO_CM;
		t_in_1_h[i]	 *= MM_TO_CM;
		t_in_2_h[i]	 *= MM_TO_CM;
		t_out_1_h[i] *= MM_TO_CM;
		t_out_2_h[i] *= MM_TO_CM;
		u_in_1_h[i]	 *= MM_TO_CM;
		u_in_2_h[i]	 *= MM_TO_CM;
		u_out_1_h[i] *= MM_TO_CM;
		u_out_2_h[i] *= MM_TO_CM;
		WEPL_h[i]	 *= MM_TO_CM;
		if( COUNT_0_WEPLS && WEPL_h[i] == 0 )
		{
			zero_WEPL++;
			zero_WEPL_files++;
		}
	}
	if( COUNT_0_WEPLS )
	{
		std::cout << "Histories in " << gantry_angle_h[0] << "with WEPL = 0 :" << zero_WEPL_files << std::endl;
		zero_WEPL_files = 0;
	}
}
void apply_tuv_shifts( unsigned int num_histories)
{
	for( unsigned int i = 0; i < num_histories; i++ ) 
	{
		// Correct for any shifts in u/t coordinates
		t_in_1_h[i]	 += T_SHIFT;
		t_in_2_h[i]	 += T_SHIFT;
		t_out_1_h[i] += T_SHIFT;
		t_out_2_h[i] += T_SHIFT;
		u_in_1_h[i]	 += U_SHIFT;
		u_in_2_h[i]	 += U_SHIFT;
		u_out_1_h[i] += U_SHIFT;
		u_out_2_h[i] += U_SHIFT;
		v_in_1_h[i]	 += V_SHIFT;
		v_in_2_h[i]	 += V_SHIFT;
		v_out_1_h[i] += V_SHIFT;
		v_out_2_h[i] += V_SHIFT;
		if( WRITE_SSD_ANGLES )
		{
			ut_entry_angle[i] = atan2( t_in_2_h[i] - t_in_1_h[i], u_in_2_h[i] - u_in_1_h[i] );	
			uv_entry_angle[i] = atan2( v_in_2_h[i] - v_in_1_h[i], u_in_2_h[i] - u_in_1_h[i] );	
			ut_exit_angle[i] = atan2( t_out_2_h[i] - t_out_1_h[i], u_out_2_h[i] - u_out_1_h[i] );	
			uv_exit_angle[i] = atan2( v_out_2_h[i] - v_out_1_h[i], u_out_2_h[i] - u_out_1_h[i] );	
		}
	}
	if( WRITE_SSD_ANGLES )
	{
		char data_filename[256];
		sprintf(data_filename, "%s_%03d%s", "ut_entry_angle", gantry_angle_h[0], ".txt" );
		array_2_disk( data_filename, OUTPUT_DIRECTORY, OUTPUT_FOLDER_UNIQUE, ut_entry_angle, COLUMNS, ROWS, SLICES, num_histories, true );
		sprintf(data_filename, "%s_%03d%s", "uv_entry_angle", gantry_angle_h[0], ".txt" );
		char ut_entry_angle[] = {"ut_entry_angle"};
		array_2_disk( ut_entry_angle, OUTPUT_DIRECTORY, OUTPUT_FOLDER_UNIQUE, uv_entry_angle, COLUMNS, ROWS, SLICES, num_histories, true );
		sprintf(data_filename, "%s_%03d%s", "ut_exit_angle", gantry_angle_h[0], ".txt" );
		array_2_disk( ut_entry_angle, OUTPUT_DIRECTORY, OUTPUT_FOLDER_UNIQUE, ut_exit_angle, COLUMNS, ROWS, SLICES, num_histories, true );
		sprintf(data_filename, "%s_%03d%s", "uv_exit_angle", gantry_angle_h[0], ".txt" );
		array_2_disk( ut_entry_angle, OUTPUT_DIRECTORY, OUTPUT_FOLDER_UNIQUE, uv_exit_angle, COLUMNS, ROWS, SLICES, num_histories, true );
	}
}
void read_data_chunk( const int num_histories, const int start_file_num, const int end_file_num )
{
	// The GPU cannot process all the histories at once, so they are broken up into chunks that can fit on the GPU.  As we iterate 
	// through the data one chunk at a time, we determine which histories enter the reconstruction volume and if they belong to a 
	// valid bin (i.e. t, v, and angular bin number is greater than zero and less than max).  If both are true, we push the bin
	// number, WEPL, and relative entry/exit ut/uv angles to the back of their corresponding std::vector.
	
	unsigned int size_floats = sizeof(float) * num_histories;
	unsigned int size_ints = sizeof(int) * num_histories;

	t_in_1_h		= (float*) malloc(size_floats);
	t_in_2_h		= (float*) malloc(size_floats);
	t_out_1_h		= (float*) malloc(size_floats);
	t_out_2_h		= (float*) malloc(size_floats);
	u_in_1_h		= (float*) malloc(size_floats);
	u_in_2_h		= (float*) malloc(size_floats);
	u_out_1_h		= (float*) malloc(size_floats);
	u_out_2_h		= (float*) malloc(size_floats);
	v_in_1_h		= (float*) malloc(size_floats);
	v_in_2_h		= (float*) malloc(size_floats);
	v_out_1_h		= (float*) malloc(size_floats);
	v_out_2_h		= (float*) malloc(size_floats);		
	WEPL_h			= (float*) malloc(size_floats);
	gantry_angle_h	= (int*)   malloc(size_ints);

	if( WRITE_SSD_ANGLES )
	{
		ut_entry_angle	= (float*) malloc(size_floats);
		uv_entry_angle	= (float*) malloc(size_floats);
		ut_exit_angle	= (float*) malloc(size_floats);
		uv_exit_angle	= (float*) malloc(size_floats);
	}
	switch( DATA_FORMAT )
	{
		case OLD_FORMAT : read_data_chunk_old( num_histories, start_file_num, end_file_num - 1 );	break;
		case VERSION_0  : read_data_chunk_v0(  num_histories, start_file_num, end_file_num - 1 );	break;
		case VERSION_1  : read_data_chunk_v1(  num_histories, start_file_num, end_file_num - 1 );
	}
}
void read_data_chunk_old( const int num_histories, const int start_file_num, const int end_file_num )
{
	int array_index = 0, gantry_position, gantry_angle, scan_number, scan_histories;
	float v_data[4], t_data[4], WEPL_data, gantry_angle_data, dummy_data;
	char tracker_plane[4];
	char data_filename[128];
	FILE* data_file;

	for( int file_num = start_file_num; file_num <= end_file_num; file_num++ )
	{
		gantry_position = file_num / NUM_SCANS;
		gantry_angle = int(gantry_position * GANTRY_ANGLE_INTERVAL);
		scan_number = file_num % NUM_SCANS + 1;
		scan_histories = histories_per_file[file_num];

		printf("Reading File for Gantry Angle %d from Scan Number %d...\n", gantry_angle, scan_number );
		sprintf( data_filename, "%s%s/%s_trans%d_%03d%s", INPUT_DIRECTORY, INPUT_FOLDER, PROJECTION_DATA_BASENAME, scan_number, gantry_angle, PROJECTION_DATA_EXTENSION );
		data_file = fopen( data_filename, "rb" );	

		for( int history = 0; history < scan_histories; history++, array_index++ ) 
		{
			fread(&v_data,				sizeof(float),	4, data_file);
			fread(&t_data,				sizeof(float),	4, data_file);
			fread(&tracker_plane,		sizeof(char),	4, data_file);
			fread(&WEPL_data,			sizeof(float),	1, data_file);
			fread(&gantry_angle_data,	sizeof(float),	1, data_file);
			fread(&dummy_data,			sizeof(float),	1, data_file); // dummy read because each event has an extra 4 bytes, for some reason
			if( DATA_IN_MM )
			{
				// Convert the input data from mm to cm
				v_in_1_h[array_index]	= v_data[0] * MM_TO_CM;;
				v_in_2_h[array_index]	= v_data[1] * MM_TO_CM;;
				v_out_1_h[array_index]	= v_data[2] * MM_TO_CM;;
				v_out_2_h[array_index]	= v_data[3] * MM_TO_CM;;
				t_in_1_h[array_index]	= t_data[0] * MM_TO_CM;;
				t_in_2_h[array_index]	= t_data[1] * MM_TO_CM;;
				t_out_1_h[array_index]	= t_data[2] * MM_TO_CM;;
				t_out_2_h[array_index]	= t_data[3] * MM_TO_CM;;
				WEPL_h[array_index]		= WEPL_data * MM_TO_CM;;
			}
			else
			{
				v_in_1_h[array_index]	= v_data[0];
				v_in_2_h[array_index]	= v_data[1];
				v_out_1_h[array_index]	= v_data[2];
				v_out_2_h[array_index]	= v_data[3];
				t_in_1_h[array_index]	= t_data[0];
				t_in_2_h[array_index]	= t_data[1];
				t_out_1_h[array_index]	= t_data[2];
				t_out_2_h[array_index]	= t_data[3];
				WEPL_h[array_index]		= WEPL_data;
			}
			if( !MICAH_SIM )
			{
				u_in_1_h[array_index]	= SSD_u_Positions[int(tracker_plane[0])];
				u_in_2_h[array_index]	= SSD_u_Positions[int(tracker_plane[1])];
				u_out_1_h[array_index]	= SSD_u_Positions[int(tracker_plane[2])];
				u_out_2_h[array_index]	= SSD_u_Positions[int(tracker_plane[3])];
			}
			else
			{
				u_in_1_h[array_index]	= SSD_u_Positions[0];
				u_in_2_h[array_index]	= SSD_u_Positions[2];
				u_out_1_h[array_index]	= SSD_u_Positions[4];
				u_out_2_h[array_index]	= SSD_u_Positions[6];
			}
			if( SSD_IN_MM )
			{
				// Convert the tracking plane positions from mm to cm
				u_in_1_h[array_index]	*= MM_TO_CM;;
				u_in_2_h[array_index]	*= MM_TO_CM;;
				u_out_1_h[array_index]	*= MM_TO_CM;;
				u_out_2_h[array_index]	*= MM_TO_CM;;
			}
			gantry_angle_h[array_index] = int(gantry_angle_data);
		}
		fclose(data_file);		
	}
}
void read_data_chunk_v0( const int num_histories, const int start_file_num, const int end_file_num )
{	
	/*
	Event data:
	Data is be stored with all of one type in a consecutive row, meaning the first entries will be N t0 values, where N is the number of events in the file. Next will be N t1 values, etc. This more closely matches the data structure in memory.
	Detector coordinates in mm relative to a phantom center, given in the detector coordinate system:
		t0 (float * N)
		t1 (float * N)
		t2 (float * N)
		t3 (float * N)
		v0 (float * N)
		v1 (float * N)
		v2 (float * N)
		v3 (float * N)
		u0 (float * N)
		u1 (float * N)
		u2 (float * N)
		u3 (float * N)
		WEPL in mm (float * N)
	*/
	char data_filename[128];//, statement[256];
	unsigned int gantry_position, gantry_angle, scan_number, file_histories, array_index = 0, histories_read = 0;

	sprintf(print_statement, "%d histories to be read from %d files", num_histories, end_file_num - start_file_num + 1 );
	print_colored_text( print_statement, YELLOW_TEXT, BLACK_BACKGROUND, UNDERLINE_TEXT );	
		
	for( unsigned int file_num = start_file_num; file_num <= end_file_num; file_num++ )
	{	
		gantry_position = file_num / NUM_SCANS;
		gantry_angle = int(gantry_position * GANTRY_ANGLE_INTERVAL);
		scan_number = file_num % NUM_SCANS + 1;
		file_histories = histories_per_file[file_num];
		
		sprintf(data_filename, "%s%s/%s_%03d%s", INPUT_DIRECTORY, INPUT_FOLDER, PROJECTION_DATA_BASENAME, gantry_angle, PROJECTION_DATA_EXTENSION );
		FILE* data_file = fopen(data_filename, "rb");
		if( data_file == NULL )
		{
			fputs( "Error Opening Data File:  Check that the directories are properly named.\n", stderr ); 
			exit_program_if(true);
		}
		if( VERSION_ID == 0 )
		{
			//printf("\t");
			sprintf(print_statement, "\tReading %d histories for gantry angle %d from scan number %d...", file_histories, gantry_angle, scan_number );			
			print_colored_text( print_statement, CYAN_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );	
			fseek( data_file, SKIP_2_DATA_SIZE, SEEK_SET );

			fread( &t_in_1_h[histories_read],  sizeof(float), file_histories, data_file );
			fread( &t_in_2_h[histories_read],  sizeof(float), file_histories, data_file );
			fread( &t_out_1_h[histories_read], sizeof(float), file_histories, data_file );
			fread( &t_out_2_h[histories_read], sizeof(float), file_histories, data_file );
			fread( &v_in_1_h[histories_read],  sizeof(float), file_histories, data_file );
			fread( &v_in_2_h[histories_read],  sizeof(float), file_histories, data_file );
			fread( &v_out_1_h[histories_read], sizeof(float), file_histories, data_file );
			fread( &v_out_2_h[histories_read], sizeof(float), file_histories, data_file );
			fread( &u_in_1_h[histories_read],  sizeof(float), file_histories, data_file );
			fread( &u_in_2_h[histories_read],  sizeof(float), file_histories, data_file );
			fread( &u_out_1_h[histories_read], sizeof(float), file_histories, data_file );
			fread( &u_out_2_h[histories_read], sizeof(float), file_histories, data_file );
			fread( &WEPL_h[histories_read],    sizeof(float), file_histories, data_file );
			fclose(data_file);

			histories_read += file_histories;
			for( unsigned int i = 0; i < file_histories; i++, array_index++ ) 
				gantry_angle_h[array_index] = int(projection_angles[file_num]);							
		}
		else if( VERSION_ID == 1 )
		{
			sprintf(print_statement, "\tReading %d histories for gantry angle %d from scan number %d...", file_histories, gantry_angle, scan_number );			
			print_colored_text( print_statement, CYAN_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );	
			fseek( data_file, SKIP_2_DATA_SIZE, SEEK_SET );

			fread( &t_in_1_h[histories_read],  sizeof(float), file_histories, data_file );
			fread( &t_in_2_h[histories_read],  sizeof(float), file_histories, data_file );
			fread( &t_out_1_h[histories_read], sizeof(float), file_histories, data_file );
			fread( &t_out_2_h[histories_read], sizeof(float), file_histories, data_file );
			fread( &v_in_1_h[histories_read],  sizeof(float), file_histories, data_file );
			fread( &v_in_2_h[histories_read],  sizeof(float), file_histories, data_file );
			fread( &v_out_1_h[histories_read], sizeof(float), file_histories, data_file );
			fread( &v_out_2_h[histories_read], sizeof(float), file_histories, data_file );
			fread( &u_in_1_h[histories_read],  sizeof(float), file_histories, data_file );
			fread( &u_in_2_h[histories_read],  sizeof(float), file_histories, data_file );
			fread( &u_out_1_h[histories_read], sizeof(float), file_histories, data_file );
			fread( &u_out_2_h[histories_read], sizeof(float), file_histories, data_file );
			fread( &WEPL_h[histories_read],    sizeof(float), file_histories, data_file );
			fclose(data_file);

			histories_read += file_histories;
			for( unsigned int i = 0; i < file_histories; i++, array_index++ ) 
				gantry_angle_h[array_index] = int(projection_angles[file_num]);							
		}
	}
	if( COUNT_0_WEPLS )
	{
		std::cout << "Histories in " << gantry_angle_h[0] << "with WEPL = 0 :" << zero_WEPL_files << std::endl;
		zero_WEPL_files = 0;
	}
	if( DATA_IN_MM )
		convert_mm_2_cm( num_histories );
	if( T_SHIFT != 0.0	||  U_SHIFT != 0.0 ||  V_SHIFT != 0.0)
		apply_tuv_shifts( num_histories );
}
void read_data_chunk_v02( const int num_histories, const int start_file_num, const int end_file_num )
{
	/*
	Contains the following headers:
		Magic number identifier: "PCTD" (4-byte string)
		Format version identifier (integer)
		Number of events in file (integer)
		Projection angle (float | degrees)
		Beam energy (float | MeV)
		Acquisition/generation date (integer | Unix time)
		Pre-process date (integer | Unix time)
		Phantom name or description (variable length string)
		Data source (variable length string)
		Prepared by (variable length string)
	* Note on variable length strings: each variable length string should be preceded with an integer containing the number of characters in the string.
	
	Event data:
	Data is be stored with all of one type in a consecutive row, meaning the first entries will be N t0 values, where N is the number of events in the file. Next will be N t1 values, etc. This more closely matches the data structure in memory.
	Detector coordinates in mm relative to a phantom center, given in the detector coordinate system:
		t0 (float * N)
		t1 (float * N)
		t2 (float * N)
		t3 (float * N)
		v0 (float * N)
		v1 (float * N)
		v2 (float * N)
		v3 (float * N)
		u0 (float * N)
		u1 (float * N)
		u2 (float * N)
		u3 (float * N)
		WEPL in mm (float * N)
	*/
	//char user_response[20];
	char data_filename[128];
	int array_index = 0, histories_read = 0;
	for( int file_num = start_file_num; file_num <= end_file_num; file_num++ )
	{
		int gantry_position = file_num / NUM_SCANS;
		int gantry_angle = int(gantry_position * GANTRY_ANGLE_INTERVAL);
		int scan_number = file_num % NUM_SCANS + 1;
		//int scan_histories = histories_per_file[file_num];

		printf("Reading File for Gantry Angle %d from Scan Number %d...\n", gantry_angle, scan_number );
		sprintf(data_filename, "%s%s/%s_%03d%s", INPUT_DIRECTORY, INPUT_FOLDER, PROJECTION_DATA_BASENAME, gantry_angle, PROJECTION_DATA_EXTENSION );	
		std::ifstream data_file(data_filename, std::ios::binary);
		if( data_file == NULL )
		{
			fputs( "File not found:  Check that the directories and files are properly named.", stderr ); 
			exit_program_if(true);
		}
		char magic_number[5];
		data_file.read(magic_number, 4);
		magic_number[4] = '\0';
		if( strcmp(magic_number, "PCTD") ) {
			puts("Error: unknown file type (should be PCTD)!\n");
			exit_program_if(true);
		}
		int version_id;
		data_file.read((char*)&version_id, sizeof(int));
		if( version_id == 0 )
		{
			int file_histories;
			data_file.read((char*)&file_histories, sizeof(int));
	
			puts("Reading headers from file...\n");
	
			float projection_angle, beam_energy;
			int generation_date, preprocess_date;
			int phantom_name_size, data_source_size, prepared_by_size;
			char *phantom_name, *data_source, *prepared_by;
	
			data_file.read((char*)&projection_angle, sizeof(float));
			data_file.read((char*)&beam_energy, sizeof(float));
			data_file.read((char*)&generation_date, sizeof(int));
			data_file.read((char*)&preprocess_date, sizeof(int));
			data_file.read((char*)&phantom_name_size, sizeof(int));
			phantom_name = (char*)malloc(phantom_name_size);
			data_file.read(phantom_name, phantom_name_size);
			data_file.read((char*)&data_source_size, sizeof(int));
			data_source = (char*)malloc(data_source_size);
			data_file.read(data_source, data_source_size);
			data_file.read((char*)&prepared_by_size, sizeof(int));
			prepared_by = (char*)malloc(prepared_by_size);
			data_file.read(prepared_by, prepared_by_size);
	
			printf("Loading %d histories from file\n", file_histories);
	
			int data_size = file_histories * sizeof(float);
	
			data_file.read((char*)&t_in_1_h[histories_read], data_size);
			data_file.read((char*)&t_in_2_h[histories_read], data_size);
			data_file.read((char*)&t_out_1_h[histories_read], data_size);
			data_file.read((char*)&t_out_2_h[histories_read], data_size);
			data_file.read((char*)&v_in_1_h[histories_read], data_size);
			data_file.read((char*)&v_in_2_h[histories_read], data_size);
			data_file.read((char*)&v_out_1_h[histories_read], data_size);
			data_file.read((char*)&v_out_2_h[histories_read], data_size);
			data_file.read((char*)&u_in_1_h[histories_read], data_size);
			data_file.read((char*)&u_in_2_h[histories_read], data_size);
			data_file.read((char*)&u_out_1_h[histories_read], data_size);
			data_file.read((char*)&u_out_2_h[histories_read], data_size);
			data_file.read((char*)&WEPL_h[histories_read], data_size);
	
			//double max_v = 0;
			//double min_v = 0;
			//double max_WEPL = 0;
			//double min_WEPL = 0;
			//float v_data[4], t_data[4], WEPL_data, gantry_angle_data, dummy_data;
			for( unsigned int i = 0; i < file_histories; i++, array_index++ ) 
			{
				if( DATA_IN_MM )
				{
					// Convert the input data from mm to cm
					v_in_1_h[array_index]		*= MM_TO_CM;
					v_in_2_h[array_index]		*= MM_TO_CM;
					v_out_1_h[array_index]	*= MM_TO_CM;
					v_out_2_h[array_index]	*= MM_TO_CM;
					t_in_1_h[array_index]		*= MM_TO_CM;
					t_in_2_h[array_index]		*= MM_TO_CM; 
					t_out_1_h[array_index]	*= MM_TO_CM; 
					t_out_2_h[array_index]	*= MM_TO_CM;
					WEPL_h[array_index]		*= MM_TO_CM;
					//if( WEPL_h[array_index] < 0 )
						//printf("WEPL[%d] = %3f\n", i, WEPL_h[array_index] );
					u_in_1_h[array_index]		*= MM_TO_CM;
					u_in_2_h[array_index]		*= MM_TO_CM;
					u_out_1_h[array_index]	*= MM_TO_CM;
					u_out_2_h[array_index]	*= MM_TO_CM;
					/*if( (v_in_1_h[array_index]) > max_v )
						max_v = v_in_1_h[array_index];
					if( (v_in_2_h[array_index]) > max_v )
						max_v = v_in_2_h[array_index];
					if( (v_out_1_h[array_index]) > max_v )
						max_v = v_out_1_h[array_index];
					if( (v_out_2_h[array_index]) > max_v )
						max_v = v_out_2_h[array_index];
					
					if( (v_in_1_h[array_index]) < min_v )
						min_v = v_in_1_h[array_index];
					if( (v_in_2_h[array_index]) < min_v )
						min_v = v_in_2_h[array_index];
					if( (v_out_1_h[array_index]) < min_v )
						min_v = v_out_1_h[array_index];
					if( (v_out_2_h[array_index]) < min_v )
						min_v = v_out_2_h[array_index];

					if( (WEPL_h[array_index]) > max_WEPL )
						max_WEPL = WEPL_h[array_index];
					if( (WEPL_h[array_index]) < min_WEPL )
						min_WEPL = WEPL_h[array_index];*/
				}
				gantry_angle_h[array_index] = static_cast<int>(projection_angle);
				//gantry_angle_h[array_index] = (int(projection_angle) + 270)%360;

			}
			//printf("max_v = %3f\n", max_v );
			//printf("min_v = %3f\n", min_v );
			//printf("max_WEPL = %3f\n", max_WEPL );
			//printf("min_WEPL = %3f\n", min_WEPL );
			data_file.close();
			histories_read += file_histories;
		}
	}
	//printf("gantry_angle_h[0] = %d\n", gantry_angle_h[0] );
	//printf("t_in_1_h[0] = %3f\n", t_in_1_h[0] );
}
void read_data_chunk_v1( const int num_histories, const int start_file_num, const int end_file_num )
{
	/*
	Contains the following headers:
		Magic number identifier: "PCTD" (4-byte string)
		Format version identifier (integer)
		Number of events in file (integer)
		Projection angle (float | degrees)
		Beam energy (float | MeV)
		Acquisition/generation date (integer | Unix time)
		Pre-process date (integer | Unix time)
		Phantom name or description (variable length string)
		Data source (variable length string)
		Prepared by (variable length string)
	* Note on variable length strings: each variable length string should be preceded with an integer containing the number of characters in the string.
	
	Event data:
	Data is be stored with all of one type in a consecutive row, meaning the first entries will be N t0 values, where N is the number of events in the file. Next will be N t1 values, etc. This more closely matches the data structure in memory.
	Detector coordinates in mm relative to a phantom center, given in the detector coordinate system:
		t0 (float * N)
		t1 (float * N)
		t2 (float * N)
		t3 (float * N)
		v0 (float * N)
		v1 (float * N)
		v2 (float * N)
		v3 (float * N)
		u0 (float * N)
		u1 (float * N)
		u2 (float * N)
		u3 (float * N)
		WEPL in mm (float * N)
	*/
	//char user_response[20];
	char data_filename[128];
	//int array_index = 0;
	for( int file_num = start_file_num; file_num <= end_file_num; file_num++ )
	{
		int gantry_position = file_num / NUM_SCANS;
		int gantry_angle = int(gantry_position * GANTRY_ANGLE_INTERVAL);
		int scan_number = file_num % NUM_SCANS + 1;
		//int scan_histories = histories_per_file[file_num];

		printf("Reading File for Gantry Angle %d from Scan Number %d...\n", gantry_angle, scan_number );
		sprintf(data_filename, "%s%s/%s_%03d%s", INPUT_DIRECTORY, INPUT_FOLDER, PROJECTION_DATA_BASENAME, gantry_angle, PROJECTION_DATA_EXTENSION );	
		std::ifstream data_file(data_filename, std::ios::binary);
		if( data_file == NULL )
		{
			fputs( "File not found:  Check that the directories and files are properly named.", stderr ); 
			exit_program_if(true);
		}
		char magic_number[5];
		data_file.read(magic_number, 4);
		magic_number[4] = '\0';
		if( strcmp(magic_number, "PCTD") ) {
			puts("Error: unknown file type (should be PCTD)!\n");
			exit_program_if(true);
		}
		int version_id;
		data_file.read((char*)&version_id, sizeof(int));
		if( version_id == 1 )
		{
			int num_histories;
			data_file.read((char*)&num_histories, sizeof(int));
	
			puts("Reading headers from file...\n");
	
			float projection_angle, beam_energy;
			int generation_date, preprocess_date;
			int phantom_name_size, data_source_size, prepared_by_size;
			char *phantom_name, *data_source, *prepared_by;
	
			data_file.read((char*)&projection_angle, sizeof(float));
			data_file.read((char*)&beam_energy, sizeof(float));
			data_file.read((char*)&generation_date, sizeof(int));
			data_file.read((char*)&preprocess_date, sizeof(int));
			data_file.read((char*)&phantom_name_size, sizeof(int));
			phantom_name = (char*)malloc(phantom_name_size);
			data_file.read(phantom_name, phantom_name_size);
			data_file.read((char*)&data_source_size, sizeof(int));
			data_source = (char*)malloc(data_source_size);
			data_file.read(data_source, data_source_size);
			data_file.read((char*)&prepared_by_size, sizeof(int));
			prepared_by = (char*)malloc(prepared_by_size);
			data_file.read(prepared_by, prepared_by_size);
	
			printf("Loading %d histories from file\n", num_histories);
	
			int data_size = num_histories * sizeof(float);
	
			data_file.read((char*)t_in_1_h, data_size);
			data_file.read((char*)t_in_2_h, data_size);
			data_file.read((char*)t_out_1_h, data_size);
			data_file.read((char*)t_out_2_h, data_size);
			data_file.read((char*)v_in_1_h, data_size);
			data_file.read((char*)v_in_2_h, data_size);
			data_file.read((char*)v_out_1_h, data_size);
			data_file.read((char*)v_out_2_h, data_size);
			data_file.read((char*)u_in_1_h, data_size);
			data_file.read((char*)u_in_2_h, data_size);
			data_file.read((char*)u_out_1_h, data_size);
			data_file.read((char*)u_out_2_h, data_size);
			data_file.read((char*)WEPL_h, data_size);
	
			//float v_data[4], t_data[4], WEPL_data, gantry_angle_data, dummy_data;
			for( unsigned int i = 0; i < num_histories; i++ ) 
			{
				if( DATA_IN_MM )
				{
					// Convert the input data from mm to cm
					v_in_1_h[i]		*= MM_TO_CM;
					v_in_2_h[i]		*= MM_TO_CM;
					v_out_1_h[i]	*= MM_TO_CM;
					v_out_2_h[i]	*= MM_TO_CM;
					t_in_1_h[i]		*= MM_TO_CM;
					t_in_2_h[i]		*= MM_TO_CM; 
					t_out_1_h[i]	*= MM_TO_CM; 
					t_out_2_h[i]	*= MM_TO_CM;
					WEPL_h[i]		*= MM_TO_CM;
					if( WEPL_h[i] < 0 )
						printf("WEPL[%d] = %3f\n", i, WEPL_h[i] );
					u_in_1_h[i]		*= MM_TO_CM;
					u_in_2_h[i]		*= MM_TO_CM;
					u_out_1_h[i]	*= MM_TO_CM;
					u_out_2_h[i]	*= MM_TO_CM;
				}
				gantry_angle_h[i] = int(projection_angle);
			}
			data_file.close();
		}
	}
}
void recon_volume_intersections( const int num_histories )
{
	unsigned int size_floats = sizeof(float) * num_histories;
	unsigned int size_ints = sizeof(int) * num_histories;
	unsigned int size_bool = sizeof(bool) * num_histories;

	// Allocate GPU memory
	cudaMalloc((void**) &t_in_1_d,				size_floats);
	cudaMalloc((void**) &t_in_2_d,				size_floats);
	cudaMalloc((void**) &t_out_1_d,				size_floats);
	cudaMalloc((void**) &t_out_2_d,				size_floats);
	cudaMalloc((void**) &u_in_1_d,				size_floats);
	cudaMalloc((void**) &u_in_2_d,				size_floats);
	cudaMalloc((void**) &u_out_1_d,				size_floats);
	cudaMalloc((void**) &u_out_2_d,				size_floats);
	cudaMalloc((void**) &v_in_1_d,				size_floats);
	cudaMalloc((void**) &v_in_2_d,				size_floats);
	cudaMalloc((void**) &v_out_1_d,				size_floats);
	cudaMalloc((void**) &v_out_2_d,				size_floats);		
	cudaMalloc((void**) &gantry_angle_d,		size_ints);

	cudaMalloc((void**) &x_entry_d,				size_floats);
	cudaMalloc((void**) &y_entry_d,				size_floats);
	cudaMalloc((void**) &z_entry_d,				size_floats);
	cudaMalloc((void**) &x_exit_d,				size_floats);
	cudaMalloc((void**) &y_exit_d,				size_floats);
	cudaMalloc((void**) &z_exit_d,				size_floats);
	cudaMalloc((void**) &xy_entry_angle_d,		size_floats);	
	cudaMalloc((void**) &xz_entry_angle_d,		size_floats);
	cudaMalloc((void**) &xy_exit_angle_d,		size_floats);
	cudaMalloc((void**) &xz_exit_angle_d,		size_floats);
	cudaMalloc((void**) &missed_recon_volume_d,	size_bool);	

	cudaMemcpy(t_in_1_d,		t_in_1_h,		size_floats, cudaMemcpyHostToDevice) ;
	cudaMemcpy(t_in_2_d,		t_in_2_h,		size_floats, cudaMemcpyHostToDevice) ;
	cudaMemcpy(t_out_1_d,		t_out_1_h,		size_floats, cudaMemcpyHostToDevice) ;
	cudaMemcpy(t_out_2_d,		t_out_2_h,		size_floats, cudaMemcpyHostToDevice) ;
	cudaMemcpy(u_in_1_d,		u_in_1_h,		size_floats, cudaMemcpyHostToDevice) ;
	cudaMemcpy(u_in_2_d,		u_in_2_h,		size_floats, cudaMemcpyHostToDevice) ;
	cudaMemcpy(u_out_1_d,		u_out_1_h,		size_floats, cudaMemcpyHostToDevice) ;
	cudaMemcpy(u_out_2_d,		u_out_2_h,		size_floats, cudaMemcpyHostToDevice) ;
	cudaMemcpy(v_in_1_d,		v_in_1_h,		size_floats, cudaMemcpyHostToDevice) ;
	cudaMemcpy(v_in_2_d,		v_in_2_h,		size_floats, cudaMemcpyHostToDevice) ;
	cudaMemcpy(v_out_1_d,		v_out_1_h,		size_floats, cudaMemcpyHostToDevice) ;
	cudaMemcpy(v_out_2_d,		v_out_2_h,		size_floats, cudaMemcpyHostToDevice) ;
	cudaMemcpy(gantry_angle_d,	gantry_angle_h,	size_ints,   cudaMemcpyHostToDevice) ;

	dim3 dimBlock(THREADS_PER_BLOCK);
	dim3 dimGrid((int)(num_histories/THREADS_PER_BLOCK)+1);
	recon_volume_intersections_GPU<<<dimGrid, dimBlock>>>
	(
		num_histories, gantry_angle_d, missed_recon_volume_d,
		t_in_1_d, t_in_2_d, t_out_1_d, t_out_2_d,
		u_in_1_d, u_in_2_d, u_out_1_d, u_out_2_d,
		v_in_1_d, v_in_2_d, v_out_1_d, v_out_2_d, 	
		x_entry_d, y_entry_d, z_entry_d, x_exit_d, y_exit_d, z_exit_d, 		
		xy_entry_angle_d, xz_entry_angle_d, xy_exit_angle_d, xz_exit_angle_d
	);

	free(t_in_1_h);
	free(t_in_2_h);
	free(t_out_1_h);
	free(t_out_2_h);
	free(v_in_1_h);
	free(v_in_2_h);
	free(v_out_1_h);
	free(v_out_2_h);
	free(u_in_1_h);
	free(u_in_2_h);
	free(u_out_1_h);
	free(u_out_2_h);
	// Host memory not freed


	cudaFree(t_in_1_d);
	cudaFree(t_in_2_d);
	cudaFree(t_out_1_d);
	cudaFree(t_out_2_d);
	cudaFree(v_in_1_d);
	cudaFree(v_in_2_d);
	cudaFree(v_out_1_d);
	cudaFree(v_out_2_d);
	cudaFree(u_in_1_d);
	cudaFree(u_in_2_d);
	cudaFree(u_out_1_d);
	cudaFree(u_out_2_d);	
	cudaFree(gantry_angle_d);
	/* 
		Device memory allocated but not freed here
		x_entry_d;
		y_entry_d;
		z_entry_d;
		x_exit_d;
		y_exit_d;
		z_exit_d;
		xy_entry_angle_d;
		xz_entry_angle_d;
		xy_exit_angle_d;
		xz_exit_angle_d;
		missed_recon_volume_d;
	*/
}
__global__ void recon_volume_intersections_GPU
(
	int num_histories, int* gantry_angle, bool* missed_recon_volume, float* t_in_1, float* t_in_2, float* t_out_1, float* t_out_2, float* u_in_1, float* u_in_2, 
	float* u_out_1, float* u_out_2, float* v_in_1, float* v_in_2, float* v_out_1, float* v_out_2, float* x_entry, float* y_entry, float* z_entry, float* x_exit, 
	float* y_exit, float* z_exit, float* xy_entry_angle, float* xz_entry_angle, float* xy_exit_angle, float* xz_exit_angle
)
{
	/************************************************************************************************************************************************************/
	/*		Determine if the proton path passes through the reconstruction volume (i.e. intersects the reconstruction cylinder twice) and if it does, determine	*/ 
	/*	the x, y, and z positions in the global/object coordinate system where the proton enters and exits the reconstruction volume.  The origin of the object */
	/*	coordinate system is defined to be at the center of the reconstruction cylinder so that its volume is bounded by:										*/
	/*																																							*/
	/*													-RECON_CYL_RADIUS	<= x <= RECON_CYL_RADIUS															*/
	/*													-RECON_CYL_RADIUS	<= y <= RECON_CYL_RADIUS															*/
	/*													-RECON_CYL_HEIGHT/2 <= z <= RECON_CYL_HEIGHT/2															*/																									
	/*																																							*/
	/*		First, the coordinates of the points where the proton path intersected the entry/exit detectors must be calculated.  Since the detectors records	*/ 
	/*	data in the detector coordinate system, data in the utv coordinate system must be converted into the global/object coordinate system.  The coordinate	*/
	/*	transformation can be accomplished using a rotation matrix with an angle of rotation determined by the angle between the two coordinate systems, which  */ 
	/*	is the gantry_angle, in this case:																														*/
	/*																																							*/
	/*	Rotate ut-coordinate system to xy-coordinate system							Rotate xy-coordinate system to ut-coordinate system							*/
	/*		x = cos( gantry_angle ) * u - sin( gantry_angle ) * t						u = cos( gantry_angle ) * x + sin( gantry_angle ) * y					*/
	/*		y = sin( gantry_angle ) * u + cos( gantry_angle ) * t						t = cos( gantry_angle ) * y - sin( gantry_angle ) * x					*/
	/************************************************************************************************************************************************************/
			
	int i = threadIdx.x + blockIdx.x * THREADS_PER_BLOCK;
	if( i < num_histories )
	{
		double rotation_angle_radians = gantry_angle[i] * ANGLE_TO_RADIANS;
		/********************************************************************************************************************************************************/
		/************************************************************ Check entry information *******************************************************************/
		/********************************************************************************************************************************************************/

		/********************************************************************************************************************************************************/
		/* Determine if the proton path enters the reconstruction volume.  The proton path is defined using the angle and position of the proton as it passed	*/
		/* through the SSD closest to the object.  Since the reconstruction cylinder is symmetric about the rotation axis, we find a proton's intersection 		*/
		/* points in the ut plane and then rotate these points into the xy plane.  Since a proton very likely has a small angle in ut plane, this allows us to 	*/
		/* overcome numerical instabilities that occur at near vertical angles which would occur for gantry angles near 90/270 degrees.  However, if a path is 	*/
		/* between [45,135] or [225,315], calculations are performed in a rotated coordinate system to avoid these numerical issues								*/
		/********************************************************************************************************************************************************/
		double ut_entry_angle = atan2( t_in_2[i] - t_in_1[i], u_in_2[i] - u_in_1[i] );
		//ut_entry_angle += PI;
		double u_entry, t_entry;
		
		// Calculate if and where proton enters reconstruction volume; u_entry/t_entry passed by reference so they hold the entry point upon function returns
		bool entered = calculate_intercepts( u_in_2[i], t_in_2[i], ut_entry_angle, u_entry, t_entry );
		
		xy_entry_angle[i] = ut_entry_angle + rotation_angle_radians;

		// Rotate exit detector positions
		x_entry[i] = ( cos( rotation_angle_radians ) * u_entry ) - ( sin( rotation_angle_radians ) * t_entry );
		y_entry[i] = ( sin( rotation_angle_radians ) * u_entry ) + ( cos( rotation_angle_radians ) * t_entry );
		/********************************************************************************************************************************************************/
		/************************************************************* Check exit information *******************************************************************/
		/********************************************************************************************************************************************************/
		double ut_exit_angle = atan2( t_out_2[i] - t_out_1[i], u_out_2[i] - u_out_1[i] );
		double u_exit, t_exit;
		
		// Calculate if and where proton exits reconstruction volume; u_exit/t_exit passed by reference so they hold the exit point upon function returns
		bool exited = calculate_intercepts( u_out_1[i], t_out_1[i], ut_exit_angle, u_exit, t_exit );

		xy_exit_angle[i] = ut_exit_angle + rotation_angle_radians;

		// Rotate exit detector positions
		x_exit[i] = ( cos( rotation_angle_radians ) * u_exit ) - ( sin( rotation_angle_radians ) * t_exit );
		y_exit[i] = ( sin( rotation_angle_radians ) * u_exit ) + ( cos( rotation_angle_radians ) * t_exit );
		/********************************************************************************************************************************************************/
		/************************************************************* Check z(v) information *******************************************************************/
		/********************************************************************************************************************************************************/
		
		// Relevant angles/slopes in radians for entry and exit in the uv plane
		double uv_entry_slope = ( v_in_2[i] - v_in_1[i] ) / ( u_in_2[i] - u_in_1[i] );
		double uv_exit_slope = ( v_out_2[i] - v_out_1[i] ) / ( u_out_2[i] - u_out_1[i] );
		
		xz_entry_angle[i] = atan2( v_in_2[i] - v_in_1[i], u_in_2[i] - u_in_1[i] );
		xz_exit_angle[i] = atan2( v_out_2[i] - v_out_1[i],  u_out_2[i] - u_out_1[i] );

		/********************************************************************************************************************************************************/
		/* Calculate the u coordinate for the entry and exit points of the reconstruction volume and then use the uv slope calculated from the detector entry	*/
		/* and exit positions to determine the z position of the proton as it entered and exited the reconstruction volume, respectively.  The u-coordinate of  */
		/* the entry and exit points of the reconsruction cylinder can be found using the x/y entry/exit points just calculated and the inverse rotation		*/
		/*																																						*/
		/*											u = cos( gantry_angle ) * x + sin( gantry_angle ) * y														*/
		/********************************************************************************************************************************************************/
		u_entry = ( cos( rotation_angle_radians ) * x_entry[i] ) + ( sin( rotation_angle_radians ) * y_entry[i] );
		u_exit = ( cos(rotation_angle_radians) * x_exit[i] ) + ( sin(rotation_angle_radians) * y_exit[i] );
		z_entry[i] = v_in_2[i] + uv_entry_slope * ( u_entry - u_in_2[i] );
		z_exit[i] = v_out_1[i] - uv_exit_slope * ( u_out_1[i] - u_exit );

		/********************************************************************************************************************************************************/
		/* Even if the proton path intersected the circle defining the boundary of the cylinder in xy plane twice, it may not have actually passed through the	*/
		/* reconstruction volume or may have only passed through part way.  If |z_entry|> RECON_CYL_HEIGHT/2, then data is erroneous since the source			*/
		/* is around z=0 and we do not want to use this history.  If |z_entry| < RECON_CYL_HEIGHT/2 and |z_exit| > RECON_CYL_HEIGHT/2 then we want to use the	*/ 
		/* history but the x_exit and y_exit positions need to be calculated again based on how far through the cylinder the proton passed before exiting		*/
		/********************************************************************************************************************************************************/
		if( entered && exited )
		{
			if( ( abs(z_entry[i]) < RECON_CYL_HEIGHT * 0.5 ) && ( abs(z_exit[i]) > RECON_CYL_HEIGHT * 0.5 ) )
			{
				double recon_cyl_fraction = abs( ( ( (z_exit[i] >= 0) - (z_exit[i] < 0) ) * RECON_CYL_HEIGHT * 0.5 - z_entry[i] ) / ( z_exit[i] - z_entry[i] ) );
				x_exit[i] = x_entry[i] + recon_cyl_fraction * ( x_exit[i] - x_entry[i] );
				y_exit[i] = y_entry[i] + recon_cyl_fraction * ( y_exit[i] - y_entry[i] );
				z_exit[i] = ( (z_exit[i] >= 0) - (z_exit[i] < 0) ) * RECON_CYL_HEIGHT * 0.5;
			}
			else if( abs(z_entry[i]) > RECON_CYL_HEIGHT * 0.5 )
			{
				entered = false;
				exited = false;
			}
			if( ( abs(z_entry[i]) > RECON_CYL_HEIGHT * 0.5 ) && ( abs(z_exit[i]) < RECON_CYL_HEIGHT * 0.5 ) )
			{
				double recon_cyl_fraction = abs( ( ( (z_exit[i] >= 0) - (z_exit[i] < 0) ) * RECON_CYL_HEIGHT * 0.5 - z_exit[i] ) / ( z_exit[i] - z_entry[i] ) );
				x_entry[i] = x_exit[i] + recon_cyl_fraction * ( x_exit[i] - x_entry[i] );
				y_entry[i] = y_exit[i] + recon_cyl_fraction * ( y_exit[i] - y_entry[i] );
				z_entry[i] = ( (z_entry[i] >= 0) - (z_entry[i] < 0) ) * RECON_CYL_HEIGHT * 0.5;
			}
			/****************************************************************************************************************************************************/ 
			/* Check the measurement locations. Do not allow more than 5 cm difference in entry and exit in t and v. This gets									*/
			/* rid of spurious events.																															*/
			/****************************************************************************************************************************************************/
			if( ( abs(t_out_1[i] - t_in_2[i]) > 5 ) || ( abs(v_out_1[i] - v_in_2[i]) > 5 ) )
			{
				entered = false;
				exited = false;
			}
		}

		// Proton passed through the reconstruction volume only if it both entered and exited the reconstruction cylinder
		missed_recon_volume[i] = !entered || !exited;
	}	
}
__device__ bool calculate_intercepts( double u, double t, double ut_angle, double& u_intercept, double& t_intercept )
{
	/************************************************************************************************************************************************************/
	/*	If a proton passes through the reconstruction volume, then the line defining its path in the xy-plane will intersect the circle defining the boundary	*/
	/* of the reconstruction cylinder in the xy-plane twice.  We can determine if the proton path passes through the reconstruction volume by equating the		*/
	/* equations of the proton path and the circle.  This produces a second order polynomial which we must solve:												*/
	/*																																							*/
	/* 															 f(x)_proton = f(x)_cylinder																	*/
	/* 																	mx+b = sqrt(r^2 - x^2)																	*/
	/* 													 m^2x^2 + 2mbx + b^2 = r^2 - x^2																		*/
	/* 									   (m^2 + 1)x^2 + 2mbx + (b^2 - r^2) = 0																				*/
	/* 														   ax^2 + bx + c = 0																				*/
	/* 																   =>  a = m^2 + 1																			*/
	/* 																	   b = 2mb																				*/
	/* 																	   c = b^2 - r^2																		*/
	/* 																																							*/
	/* 		We can solve this using the quadratic formula ([-b +/- sqrt(b^2-4ac)]/2a).  If the proton passed through the reconstruction volume, then the		*/
	/* 	determinant will be greater than zero ( b^2-4ac > 0 ) and the quadratic formula will return two unique points of intersection.  The intersection point	*/
	/*	closest to where the proton entry/exit path intersects the entry/exit detector plane is then the entry/exit point.  If the determinant <= 0, then the	*/
	/*	proton path does not go through the reconstruction volume and we need not determine intersection coordinates.											*/
	/*																																							*/
	/* 		If the exit/entry path travels through the cone bounded by y=|x| && y=-|x| the x_coordinates will be small and the difference between the entry and */
	/*	exit x-coordinates will approach zero, causing instabilities in trig functions and slope calculations ( x difference in denominator). To overcome these */ 
	/*	innaccurate calculations, coordinates for these proton paths will be rotated PI/2 radians (90 degrees) prior to calculations and rotated back when they	*/ 
	/*	are completed using a rotation matrix transformation again:																								*/
	/* 																																							*/
	/* 					Positive Rotation By 90 Degrees											Negative Rotation By 90 Degree									*/
	/* 						x' = cos( 90 ) * x - sin( 90 ) * y = -y									x' = cos( 90 ) * x + sin( 90 ) * y = y						*/
	/* 						y' = sin( 90 ) * x + cos( 90 ) * y = x									y' = cos( 90 ) * y - sin( 90 ) * x = -x						*/
	/************************************************************************************************************************************************************/

	// Determine if entry points should be rotated
	bool entry_in_cone = ( (ut_angle > PI_OVER_4) && (ut_angle < THREE_PI_OVER_4) ) || ( (ut_angle > FIVE_PI_OVER_4) && (ut_angle < SEVEN_PI_OVER_4) );


	// Rotate u and t by 90 degrees, if necessary
	double u_temp;
	if( entry_in_cone )
	{
		u_temp = u;	
		u = -t;
		t = u_temp;
		ut_angle += PI_OVER_2;
	}
	double m = tan( ut_angle );											// proton entry path slope
	double b_in = t - m * u;											// proton entry path y-intercept

	// Quadratic formula coefficients
	double a = 1 + powf(m, 2);											// x^2 coefficient 
	double b = 2 * m * b_in;											// x coefficient
	double c = powf(b_in, 2) - powf(RECON_CYL_RADIUS, 2 );				// 1 coefficient
	double entry_discriminant = powf(b, 2) - (4 * a * c);				// Quadratic formula discriminant		
	bool intersected = ( entry_discriminant > 0 );						// Proton path intersected twice

	/************************************************************************************************************************************************************/
	/* Find both intersection points of the circle; closest one to the SSDs is the desired intersection point.  Notice that x_intercept_2 = (-b - sqrt())/2a	*/
	/* has the negative sign pulled out and the proceding equations are modified as necessary, e.g.:															*/
	/*																																							*/
	/*														x_intercept_2 = -x_real_2																			*/
	/*														y_intercept_2 = -y_real_2																			*/
	/*												   squared_distance_2 = sqd_real_2																			*/
	/* since									 (x_intercept_2 + x_in)^2 = (-x_intercept_2 - x_in)^2 = (x_real_2 - x_in)^2 (same for y term)					*/
	/*																																							*/
	/* This negation is also considered when assigning x_entry/y_entry using -x_intercept_2/y_intercept_2 *(TRUE/FALSE = 1/0)									*/
	/************************************************************************************************************************************************************/
	if( intersected )
	{
		double u_intercept_1		= ( sqrt(entry_discriminant) - b ) / ( 2 * a );
		double u_intercept_2		= ( sqrt(entry_discriminant) + b ) / ( 2 * a );
		double t_intercept_1		= m * u_intercept_1 + b_in;
		double t_intercept_2		= m * u_intercept_2 - b_in;
		double squared_distance_1	= powf( u_intercept_1 - u, 2 ) + powf( t_intercept_1 - t, 2 );
		double squared_distance_2	= powf( u_intercept_2 + u, 2 ) + powf( t_intercept_2 + t, 2 );
		u_intercept					= u_intercept_1 * ( squared_distance_1 <= squared_distance_2 ) - u_intercept_2 * ( squared_distance_1 > squared_distance_2 );
		t_intercept					= t_intercept_1 * ( squared_distance_1 <= squared_distance_2 ) - t_intercept_2 * ( squared_distance_1 > squared_distance_2 );
	}
	// Unrotate by 90 degrees, if necessary
	if( entry_in_cone )
	{
		u_temp = u_intercept;
		u_intercept = t_intercept;
		t_intercept = -u_temp;
		ut_angle -= PI_OVER_2;
	}

	return intersected;
}
void binning( const int num_histories )
{
	unsigned int size_floats	= sizeof(float) * num_histories;
	unsigned int size_ints		= sizeof(int) * num_histories;
	unsigned int size_bool		= sizeof(bool) * num_histories;

	missed_recon_volume_h		= (bool*)  calloc( num_histories, sizeof(bool)	);	
	bin_num_h					= (int*)   calloc( num_histories, sizeof(int)   );
	x_entry_h					= (float*) calloc( num_histories, sizeof(float) );
	y_entry_h					= (float*) calloc( num_histories, sizeof(float) );
	z_entry_h					= (float*) calloc( num_histories, sizeof(float) );
	x_exit_h					= (float*) calloc( num_histories, sizeof(float) );
	y_exit_h					= (float*) calloc( num_histories, sizeof(float) );
	z_exit_h					= (float*) calloc( num_histories, sizeof(float) );	
	xy_entry_angle_h			= (float*) calloc( num_histories, sizeof(float) );	
	xz_entry_angle_h			= (float*) calloc( num_histories, sizeof(float) );
	xy_exit_angle_h				= (float*) calloc( num_histories, sizeof(float) );
	xz_exit_angle_h				= (float*) calloc( num_histories, sizeof(float) );

	cudaMalloc((void**) &WEPL_d,	size_floats);
	cudaMalloc((void**) &bin_num_d,	size_ints );

	cudaMemcpy( WEPL_d,		WEPL_h,		size_floats,	cudaMemcpyHostToDevice) ;
	cudaMemcpy( bin_num_d,	bin_num_h,	size_ints,		cudaMemcpyHostToDevice );

	dim3 dimBlock( THREADS_PER_BLOCK );
	dim3 dimGrid( (int)( num_histories/THREADS_PER_BLOCK ) + 1 );
	binning_GPU<<<dimGrid, dimBlock>>>
	( 
		num_histories, bin_counts_d, bin_num_d, missed_recon_volume_d,
		x_entry_d, y_entry_d, z_entry_d, x_exit_d, y_exit_d, z_exit_d, 
		mean_WEPL_d, mean_rel_ut_angle_d, mean_rel_uv_angle_d, WEPL_d, 
		xy_entry_angle_d, xz_entry_angle_d, xy_exit_angle_d, xz_exit_angle_d
	);
	cudaMemcpy( missed_recon_volume_h,		missed_recon_volume_d,		size_bool,		cudaMemcpyDeviceToHost );
	cudaMemcpy( bin_num_h,					bin_num_d,					size_ints,		cudaMemcpyDeviceToHost );
	cudaMemcpy( x_entry_h,					x_entry_d,					size_floats,	cudaMemcpyDeviceToHost );
	cudaMemcpy( y_entry_h,					y_entry_d,					size_floats,	cudaMemcpyDeviceToHost );
	cudaMemcpy( z_entry_h,					z_entry_d,					size_floats,	cudaMemcpyDeviceToHost );
	cudaMemcpy( x_exit_h,					x_exit_d,					size_floats,	cudaMemcpyDeviceToHost );
	cudaMemcpy( y_exit_h,					y_exit_d,					size_floats,	cudaMemcpyDeviceToHost );
	cudaMemcpy( z_exit_h,					z_exit_d,					size_floats,	cudaMemcpyDeviceToHost );
	cudaMemcpy( xy_entry_angle_h,			xy_entry_angle_d,			size_floats,	cudaMemcpyDeviceToHost );
	cudaMemcpy( xz_entry_angle_h,			xz_entry_angle_d,			size_floats,	cudaMemcpyDeviceToHost );
	cudaMemcpy( xy_exit_angle_h,			xy_exit_angle_d,			size_floats,	cudaMemcpyDeviceToHost );
	cudaMemcpy( xz_exit_angle_h,			xz_exit_angle_d,			size_floats,	cudaMemcpyDeviceToHost );

	char data_filename[128];
	if( WRITE_BIN_WEPLS )
	{
		sprintf(data_filename, "%s_%03d%s", "bin_numbers", gantry_angle_h[0], ".txt" );
		array_2_disk( data_filename, OUTPUT_DIRECTORY, OUTPUT_FOLDER_UNIQUE, bin_num_h, COLUMNS, ROWS, SLICES, num_histories, true );
	}

	// Push data from valid histories  (i.e. missed_recon_volume = FALSE) onto the end of each vector
	int offset = 0;
	for( unsigned int i = 0; i < num_histories; i++ )
	{
		if( !missed_recon_volume_h[i] && ( bin_num_h[i] >= 0 ) )
		{
			bin_num_vector.push_back( bin_num_h[i] );
			//gantry_angle_vector.push_back( gantry_angle_h[i] );
			WEPL_vector.push_back( WEPL_h[i] );
			x_entry_vector.push_back( x_entry_h[i] );
			y_entry_vector.push_back( y_entry_h[i] );
			z_entry_vector.push_back( z_entry_h[i] );
			x_exit_vector.push_back( x_exit_h[i] );
			y_exit_vector.push_back( y_exit_h[i] );
			z_exit_vector.push_back( z_exit_h[i] );
			xy_entry_angle_vector.push_back( xy_entry_angle_h[i] );
			xz_entry_angle_vector.push_back( xz_entry_angle_h[i] );
			xy_exit_angle_vector.push_back( xy_exit_angle_h[i] );
			xz_exit_angle_vector.push_back( xz_exit_angle_h[i] );
			offset++;
			recon_vol_histories++;
		}
	}
	//char statement[256];
	percentage_pass_each_intersection_cut = (double) offset / num_histories * 100.0;
	sprintf(print_statement, "------> %d out of %d (%4.2f%%) histories passed intersection cuts", offset, num_histories, percentage_pass_each_intersection_cut );
	print_colored_text(print_statement, GREEN_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT );
	
	free( missed_recon_volume_h ); 
	free( bin_num_h );
	free( x_entry_h );
	free( y_entry_h );
	free( z_entry_h );
	free( x_exit_h );
	free( y_exit_h );
	free( z_exit_h );
	free( xy_entry_angle_h );
	free( xz_entry_angle_h );
	free( xy_exit_angle_h );
	free( xz_exit_angle_h );
	/* 
		Host memory allocated but not freed here
		N/A
	*/

	cudaFree( xy_entry_angle_d );
	cudaFree( xz_entry_angle_d );
	cudaFree( xy_exit_angle_d );
	cudaFree( xz_exit_angle_d );
	/* 
		Device memory allocated but not freed here
		WEPL_d;
		bin_num_d;
	*/
}
__global__ void binning_GPU
( 
	int num_histories, int* bin_counts, int* bin_num, bool* missed_recon_volume, 
	float* x_entry, float* y_entry, float* z_entry, float* x_exit, float* y_exit, float* z_exit, 
	float* mean_WEPL, float* mean_rel_ut_angle, float* mean_rel_uv_angle, float* WEPL, 
	float* xy_entry_angle, float* xz_entry_angle, float* xy_exit_angle, float* xz_exit_angle
)
{
	int i = threadIdx.x + blockIdx.x * THREADS_PER_BLOCK;
	if( i < num_histories )
	{		
		/********************************************************************************************************************/ 
		/*	Bin histories according to angle/t/v.  The value of t varies along the path, so use the average value, which	*/
		/*	occurs at the midpoint of the chord connecting the entry and exit of the reconstruction volume since the		*/
		/*	orientation of the chord is symmetric about the midpoint (drawing included in documentation).					*/
		/********************************************************************************************************************/ 
		double x_midpath, y_midpath, z_midpath, path_angle;
		int angle_bin, t_bin, v_bin;
		double angle, t, v;
		double rel_ut_angle, rel_uv_angle;

		// Calculate midpoint of chord connecting entry and exit
		x_midpath = ( x_entry[i] + x_exit[i] ) / 2;
		y_midpath = ( y_entry[i] + y_exit[i] ) / 2;
		z_midpath = ( z_entry[i] + z_exit[i] ) / 2;

		// Calculate path angle and determine which angular bin is closest
		path_angle = atan2( ( y_exit[i] - y_entry[i] ) , ( x_exit[i] - x_entry[i] ) );
		if( path_angle < 0 )
			path_angle += 2*PI;
		angle_bin = int( ( path_angle * RADIANS_TO_ANGLE / ANGULAR_BIN_SIZE ) + 0.5) % ANGULAR_BINS;	
		angle = angle_bin * ANGULAR_BIN_SIZE * ANGLE_TO_RADIANS;

		// Calculate t/v of midpoint and find t/v bin closest to this value
		t = y_midpath * cos(angle) - x_midpath * sin(angle);
		t_bin = int( (t / T_BIN_SIZE ) + T_BINS/2);			
			
		v = z_midpath;
		v_bin = int( (v / V_BIN_SIZE ) + V_BINS/2);
		
		// For histories with valid angular/t/v bin #, calculate bin #, add to its count and WEPL/relative angle sums
		if( (t_bin >= 0) && (v_bin >= 0) && (t_bin < T_BINS) && (v_bin < V_BINS) )
		{
			bin_num[i] = t_bin + angle_bin * T_BINS + v_bin * T_BINS * ANGULAR_BINS;
			if( !missed_recon_volume[i] )
			{
				//xy_entry_angle[i]
				//xz_entry_angle[i]
				//xy_exit_angle[i]
				//xz_exit_angle[i]
				rel_ut_angle = xy_exit_angle[i] - xy_entry_angle[i];
				if( rel_ut_angle > PI )
					rel_ut_angle -= 2 * PI;
				if( rel_ut_angle < -PI )
					rel_ut_angle += 2 * PI;
				rel_uv_angle = xz_exit_angle[i] - xz_entry_angle[i];
				if( rel_uv_angle > PI )
					rel_uv_angle -= 2 * PI;
				if( rel_uv_angle < -PI )
					rel_uv_angle += 2 * PI;
				atomicAdd( &bin_counts[bin_num[i]], 1 );
				atomicAdd( &mean_WEPL[bin_num[i]], WEPL[i] );
				atomicAdd( &mean_rel_ut_angle[bin_num[i]], rel_ut_angle );
				atomicAdd( &mean_rel_uv_angle[bin_num[i]], rel_uv_angle );
				//atomicAdd( &mean_rel_ut_angle[bin_num[i]], relative_ut_angle[i] );
				//atomicAdd( &mean_rel_uv_angle[bin_num[i]], relative_uv_angle[i] );
			}
			//else
				//bin_num[i] = -1;
		}
	}
}
void import_and_process_data()
{
	/********************************************************************************************************************************************************/
	/* Iteratively Read and Process Data One Chunk at a Time. There are at Most	MAX_GPU_HISTORIES Per Chunk (i.e. Iteration). On Each Iteration:			*/
	/*	(1) Read data from file																																*/
	/*	(2) Determine which histories traverse the reconstruction volume and store this	information in a boolean array										*/
	/*	(3) Determine which bin each history belongs to																										*/
	/*	(4) Use the boolean array to determine which histories to keep and then push the intermediate data from these histories onto the permanent 			*/
	/*		storage std::vectors																															*/
	/*	(5) Free up temporary host/GPU array memory allocated during iteration																				*/
	/********************************************************************************************************************************************************/
	print_section_header( "Count proton histories and initialize preprocessing host/GPU arrays/vectors", MINOR_SECTION_SEPARATOR, LIGHT_CYAN_TEXT, YELLOW_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT );
	hull_initializations();			// Initialize hull detection images and transfer them to the GPU (performed if SC_ON, MSC_ON, or SM_ON is true)		
	initializations();				// allocate and initialize host and GPU memory for statistical
	count_histories();				// count the number of histories per file, per scan, total, etc.
	reserve_vector_capacity();		// Reserve enough memory so vectors don't grow into another reserved memory space, wasting time since they must be moved
	print_section_exit( "Finished counting proton histories and initialization of arrays/vectors", SECTION_EXIT_CSTRING, RED_TEXT, RED_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );
	
	print_section_header( "Importing and processing proton history data...", MINOR_SECTION_SEPARATOR, LIGHT_CYAN_TEXT, YELLOW_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT );
	print_colored_text( "Iteratively reading data from hard disk...", CYAN_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );	
	print_colored_text( "Removing proton histories that don't pass through the reconstruction volume...", CYAN_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );	
	print_colored_text( "Binning the data from those that did...", CYAN_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );	
	timer( START, begin_data_reads, "for reading data, coordinate conversions/intersections, hull detection counts, and binning");
	int start_file_num = 0, end_file_num = 0, histories_2_process = 0;
	while( start_file_num != NUM_FILES )
	{
		while( end_file_num < NUM_FILES )
		{
			if( histories_2_process + histories_per_file[end_file_num] < MAX_GPU_HISTORIES )
				histories_2_process += histories_per_file[end_file_num];
			else
				break;
			end_file_num++;
		}
		read_data_chunk( histories_2_process, start_file_num, end_file_num );
		recon_volume_intersections( histories_2_process );
		binning( histories_2_process );
		hull_detection( histories_2_process );
		initial_processing_memory_clean();			
		start_file_num = end_file_num;
		histories_2_process = 0;
	}		
	percentage_pass_intersection_cuts = (double) recon_vol_histories / total_histories * 100;
	sprintf(print_statement, "======> %d out of %d (%4.2f%%) histories traversed the reconstruction volume", recon_vol_histories, total_histories, percentage_pass_intersection_cuts );
	//shrink_vectors( recon_vol_histories );	// Shrink vector capacities to their size = # histories remaining reconstruction volume intersection cuts
	print_colored_text( print_statement, GREEN_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT );

	print_section_exit( "Finished importing/processing input data and accumulating hull intersection info", SECTION_EXIT_CSTRING, RED_TEXT, RED_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );	
	execution_time_data_reads = timer( STOP, begin_data_reads, "for reading data, coordinate conversions/intersections, hull detection counts, and binning");	
}		
/***********************************************************************************************************************************************************************************************************************/
/******************************************************************************************** Statistical analysis and cuts ********************************************************************************************/
/***********************************************************************************************************************************************************************************************************************/
void calculate_means()
{
	print_colored_text( "Calculating the Mean for Each Bin Before Cuts...", CYAN_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );	
	//cudaMemcpy( mean_WEPL_h,	mean_WEPL_d,	SIZE_BINS_FLOAT, cudaMemcpyDeviceToHost );
	//	int* empty_parameter;
	//	bins_2_disk( "WEPL_dist_pre_test2", empty_parameter, mean_WEPL_h, NUM_BINS, MEANS, ALL_BINS, BY_BIN );

	dim3 dimBlock( T_BINS );
	dim3 dimGrid( V_BINS, ANGULAR_BINS );   
	calculate_means_GPU<<< dimGrid, dimBlock >>>
	( 
		bin_counts_d, mean_WEPL_d, mean_rel_ut_angle_d, mean_rel_uv_angle_d
	);

	if( WRITE_WEPL_DISTS )
	{
		cudaMemcpy( mean_WEPL_h,	mean_WEPL_d,	SIZE_BINS_FLOAT, cudaMemcpyDeviceToHost );
		//int* empty_parameter; //Warning: declared but not used
		//bins_2_disk( "WEPL_dist_pre_test2", empty_parameter, mean_WEPL_h, NUM_BINS, MEANS, ALL_BINS, BY_BIN );
	}
	bin_counts_h		  = (int*)	 calloc( NUM_BINS, sizeof(int) );
	cudaMemcpy(bin_counts_h, bin_counts_d, SIZE_BINS_INT, cudaMemcpyDeviceToHost) ;
	cudaMemcpy( mean_WEPL_h,	mean_WEPL_d,	SIZE_BINS_FLOAT, cudaMemcpyDeviceToHost );
	cudaMemcpy( mean_rel_ut_angle_h,	mean_rel_ut_angle_d,	SIZE_BINS_FLOAT, cudaMemcpyDeviceToHost );
	cudaMemcpy( mean_rel_uv_angle_h,	mean_rel_uv_angle_d,	SIZE_BINS_FLOAT, cudaMemcpyDeviceToHost );
	
	//array_2_disk(BIN_COUNTS_FILENAME, OUTPUT_DIRECTORY, OUTPUT_FOLDER_UNIQUE, bin_counts_h, T_BINS, ANGULAR_BINS, V_BINS, NUM_BINS, true );
	//array_2_disk(MEAN_WEPL_FILENAME, OUTPUT_DIRECTORY, OUTPUT_FOLDER_UNIQUE, mean_WEPL_h, T_BINS, ANGULAR_BINS, V_BINS, NUM_BINS, true );
	//array_2_disk(MEAN_REL_UT_FILENAME, OUTPUT_DIRECTORY, OUTPUT_FOLDER_UNIQUE, mean_rel_ut_angle_h, T_BINS, ANGULAR_BINS, V_BINS, NUM_BINS, true );
	//array_2_disk(MEAN_REL_UV_FILENAME, OUTPUT_DIRECTORY, OUTPUT_FOLDER_UNIQUE, mean_rel_uv_angle_h, T_BINS, ANGULAR_BINS, V_BINS, NUM_BINS, true );
	
	free(bin_counts_h);
	free(mean_WEPL_h);
	free(mean_rel_ut_angle_h);
	free(mean_rel_uv_angle_h);
}
__global__ void calculate_means_GPU( int* bin_counts, float* mean_WEPL, float* mean_rel_ut_angle, float* mean_rel_uv_angle )
{
	int v = blockIdx.x, angle = blockIdx.y, t = threadIdx.x;
	int bin = t + angle * T_BINS + v * T_BINS * ANGULAR_BINS;
	if( bin_counts[bin] > 0 )
	{
		mean_WEPL[bin] /= bin_counts[bin];		
		mean_rel_ut_angle[bin] /= bin_counts[bin];
		mean_rel_uv_angle[bin] /= bin_counts[bin];
	}
}
void sum_squared_deviations( const int start_position, const int num_histories )
{
	unsigned int size_floats = sizeof(float) * num_histories;
	unsigned int size_ints = sizeof(int) * num_histories;

	cudaMalloc((void**) &bin_num_d,				size_ints);
	cudaMalloc((void**) &WEPL_d,				size_floats);
	cudaMalloc((void**) &xy_entry_angle_d,		size_floats);
	cudaMalloc((void**) &xz_entry_angle_d,		size_floats);
	cudaMalloc((void**) &xy_exit_angle_d,		size_floats);
	cudaMalloc((void**) &xz_exit_angle_d,		size_floats);

	cudaMemcpy( bin_num_d,				&bin_num_vector[start_position],			size_ints, cudaMemcpyHostToDevice);
	cudaMemcpy( WEPL_d,					&WEPL_vector[start_position],				size_floats, cudaMemcpyHostToDevice);
	cudaMemcpy( xy_entry_angle_d,		&xy_entry_angle_vector[start_position],		size_floats, cudaMemcpyHostToDevice);
	cudaMemcpy( xz_entry_angle_d,		&xz_entry_angle_vector[start_position],		size_floats, cudaMemcpyHostToDevice);
	cudaMemcpy( xy_exit_angle_d,		&xy_exit_angle_vector[start_position],		size_floats, cudaMemcpyHostToDevice);
	cudaMemcpy( xz_exit_angle_d,		&xz_exit_angle_vector[start_position],		size_floats, cudaMemcpyHostToDevice);

	dim3 dimBlock(THREADS_PER_BLOCK);
	dim3 dimGrid((int)(num_histories/THREADS_PER_BLOCK)+1);
	sum_squared_deviations_GPU<<<dimGrid, dimBlock>>>
	( 
		num_histories, bin_num_d, mean_WEPL_d, mean_rel_ut_angle_d, mean_rel_uv_angle_d, 
		WEPL_d, xy_entry_angle_d, xz_entry_angle_d,  xy_exit_angle_d, xz_exit_angle_d,
		stddev_WEPL_d, stddev_rel_ut_angle_d, stddev_rel_uv_angle_d
	);
	cudaFree( bin_num_d );
	cudaFree( WEPL_d );
	cudaFree( xy_entry_angle_d );
	cudaFree( xz_entry_angle_d );
	cudaFree( xy_exit_angle_d );
	cudaFree( xz_exit_angle_d );
}
__global__ void sum_squared_deviations_GPU
( 
	int num_histories, int* bin_num, float* mean_WEPL, float* mean_rel_ut_angle, float* mean_rel_uv_angle,  
	float* WEPL, float* xy_entry_angle, float* xz_entry_angle, float* xy_exit_angle, float* xz_exit_angle,
	float* stddev_WEPL, float* stddev_rel_ut_angle, float* stddev_rel_uv_angle
)
{
	int i = threadIdx.x + blockIdx.x * THREADS_PER_BLOCK;
	if( i < num_histories )
	{
		double rel_ut_angle = xy_exit_angle[i] - xy_entry_angle[i];
		if( rel_ut_angle > PI )
			rel_ut_angle -= 2 * PI;
		if( rel_ut_angle < -PI )
			rel_ut_angle += 2 * PI;
		double rel_uv_angle = xz_exit_angle[i] - xz_entry_angle[i];
		if( rel_uv_angle > PI )
			rel_uv_angle -= 2 * PI;
		if( rel_uv_angle < -PI )
			rel_uv_angle += 2 * PI;
		double WEPL_difference = WEPL[i] - mean_WEPL[bin_num[i]];
		double rel_ut_angle_difference = rel_ut_angle - mean_rel_ut_angle[bin_num[i]];
		double rel_uv_angle_difference = rel_uv_angle - mean_rel_uv_angle[bin_num[i]];

		atomicAdd( &stddev_WEPL[bin_num[i]], powf( WEPL_difference, 2 ) );
		atomicAdd( &stddev_rel_ut_angle[bin_num[i]], powf( rel_ut_angle_difference, 2 ) );
		atomicAdd( &stddev_rel_uv_angle[bin_num[i]], powf( rel_uv_angle_difference, 2 ) );
	}
}
void calculate_standard_deviations()
{
	char statement[] = "Calculating standard deviations for each bin...";		
	print_colored_text( statement, CYAN_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );	
	dim3 dimBlock( T_BINS );
	dim3 dimGrid( V_BINS, ANGULAR_BINS );   
	calculate_standard_deviations_GPU<<< dimGrid, dimBlock >>>
	( 
		bin_counts_d, stddev_WEPL_d, stddev_rel_ut_angle_d, stddev_rel_uv_angle_d
	);
	cudaMemcpy( stddev_rel_ut_angle_h,	stddev_rel_ut_angle_d,	SIZE_BINS_FLOAT,	cudaMemcpyDeviceToHost );
	cudaMemcpy( stddev_rel_uv_angle_h,	stddev_rel_uv_angle_d,	SIZE_BINS_FLOAT,	cudaMemcpyDeviceToHost );
	cudaMemcpy( stddev_WEPL_h,			stddev_WEPL_d,			SIZE_BINS_FLOAT,	cudaMemcpyDeviceToHost );

	//array_2_disk(STDDEV_REL_UT_FILENAME, OUTPUT_DIRECTORY, OUTPUT_FOLDER_UNIQUE, stddev_rel_ut_angle_h, T_BINS, ANGULAR_BINS, V_BINS, NUM_BINS, true );
	//array_2_disk(STDDEV_REL_UV_FILENAME, OUTPUT_DIRECTORY, OUTPUT_FOLDER_UNIQUE, stddev_rel_uv_angle_h, T_BINS, ANGULAR_BINS, V_BINS, NUM_BINS, true );
	//array_2_disk(STDDEV_WEPL_FILENAME, OUTPUT_DIRECTORY, OUTPUT_FOLDER_UNIQUE, stddev_WEPL_h, T_BINS, ANGULAR_BINS, V_BINS, NUM_BINS, true );
	//cudaFree( bin_counts_d );
}
__global__ void calculate_standard_deviations_GPU( int* bin_counts, float* stddev_WEPL, float* stddev_rel_ut_angle, float* stddev_rel_uv_angle )
{
	int v = blockIdx.x, angle = blockIdx.y, t = threadIdx.x;
	int bin = t + angle * T_BINS + v * T_BINS * ANGULAR_BINS;
	if( bin_counts[bin] > 1 )
	{
		// SAMPLE_STD_DEV = true/false = 1/0 => std_dev = SUM{i = 1 -> N} [ ( mu - x_i)^2 / ( N - 1/0 ) ]
		stddev_WEPL[bin] = sqrtf( stddev_WEPL[bin] / ( bin_counts[bin] - SAMPLE_STD_DEV ) );		
		stddev_rel_ut_angle[bin] = sqrtf( stddev_rel_ut_angle[bin] / ( bin_counts[bin] - SAMPLE_STD_DEV ) );
		stddev_rel_uv_angle[bin] = sqrtf( stddev_rel_uv_angle[bin] / ( bin_counts[bin] - SAMPLE_STD_DEV ) );
	}
	syncthreads();
	bin_counts[bin] = 0;
}
void statistical_allocations( const int num_histories)
{
	unsigned int size_floats = sizeof(float) * num_histories;
	unsigned int size_ints = sizeof(int) * num_histories;
	unsigned int size_bools = sizeof(bool) * num_histories;

	failed_cuts_h = (bool*) calloc ( num_histories, sizeof(bool) );
	
	cudaMalloc( (void**) &bin_num_d,			size_ints );
	cudaMalloc( (void**) &WEPL_d,				size_floats );
	cudaMalloc( (void**) &xy_entry_angle_d,		size_floats );
	cudaMalloc( (void**) &xz_entry_angle_d,		size_floats );
	cudaMalloc( (void**) &xy_exit_angle_d,		size_floats );
	cudaMalloc( (void**) &xz_exit_angle_d,		size_floats );
	cudaMalloc( (void**) &failed_cuts_d,		size_bools );

	//char error_statement[] = {"ERROR: reconstruction_cuts GPU array allocations caused...\n"};
	//cudaError_t cudaStatus = CUDA_error_check( error_statement );
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) 
		printf("reconstruction_cuts_allocations Error: %s\n", cudaGetErrorString(cudaStatus));
}
void statistical_host_2_device(const int start_position, const int num_histories) 
{
	unsigned int size_ints		= sizeof(int) * num_histories;
	unsigned int size_floats	= sizeof(float) * num_histories;
	unsigned int size_bools		= sizeof(bool) * num_histories;

	cudaMemcpy( bin_num_d,			&bin_num_vector[start_position],			size_ints,		cudaMemcpyHostToDevice );
	cudaMemcpy( WEPL_d,				&WEPL_vector[start_position],				size_floats,	cudaMemcpyHostToDevice );
	cudaMemcpy( xy_entry_angle_d,	&xy_entry_angle_vector[start_position],		size_floats,	cudaMemcpyHostToDevice );
	cudaMemcpy( xz_entry_angle_d,	&xz_entry_angle_vector[start_position],		size_floats,	cudaMemcpyHostToDevice );
	cudaMemcpy( xy_exit_angle_d,	&xy_exit_angle_vector[start_position],		size_floats,	cudaMemcpyHostToDevice );
	cudaMemcpy( xz_exit_angle_d,	&xz_exit_angle_vector[start_position],		size_floats,	cudaMemcpyHostToDevice );
	cudaMemcpy( failed_cuts_d,		failed_cuts_h,								size_bools,		cudaMemcpyHostToDevice );

	//cudaMemcpy( bin_num_d,				&bin_num[start_position],			size_ints, cudaMemcpyHostToDevice);
	//cudaMemcpy( WEPL_d,					&WEPL[start_position],				size_floats, cudaMemcpyHostToDevice);
	//cudaMemcpy( xy_entry_angle_d,		&xy_entry_angle[start_position],		size_floats, cudaMemcpyHostToDevice);
	//cudaMemcpy( xz_entry_angle_d,		&xz_entry_angle[start_position],		size_floats, cudaMemcpyHostToDevice);
	//cudaMemcpy( xy_exit_angle_d,		&xy_exit_angle[start_position],		size_floats, cudaMemcpyHostToDevice);
	//cudaMemcpy( xz_exit_angle_d,		&xz_exit_angle[start_position],		size_floats, cudaMemcpyHostToDevice);

	//char error_statement[] = {"ERROR: reconstruction_cuts host->GPU data transfer caused...\n"};
	//cudaError_t cudaStatus =  CUDA_error_check( error_statement );
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) 
		printf("reconstruction_cuts_host_2_device Error: %s\n", cudaGetErrorString(cudaStatus));	
}
void statistical_cuts_device_2_host(const int start_position, const int num_histories) 
{
	unsigned int size_floats		= sizeof(float) * num_histories;
	unsigned int size_ints			= sizeof(int) * num_histories;
	unsigned int size_bool			= sizeof(bool) * num_histories;

	cudaMemcpy(&first_MLP_voxel_vector[start_position], first_MLP_voxel_d, size_ints, cudaMemcpyDeviceToHost);
	cudaMemcpy(&x_entry_vector[start_position], x_entry_d, size_floats, cudaMemcpyDeviceToHost);
	cudaMemcpy(&y_entry_vector[start_position], y_entry_d, size_floats, cudaMemcpyDeviceToHost);
	cudaMemcpy(&z_entry_vector[start_position], z_entry_d, size_floats, cudaMemcpyDeviceToHost);
	cudaMemcpy(&x_exit_vector[start_position], x_exit_d, size_floats, cudaMemcpyDeviceToHost);
	cudaMemcpy(&y_exit_vector[start_position], y_exit_d, size_floats, cudaMemcpyDeviceToHost);
	cudaMemcpy(&z_exit_vector[start_position], z_exit_d, size_floats, cudaMemcpyDeviceToHost);	
	cudaMemcpy(intersected_hull_h, intersected_hull_d, size_bool, cudaMemcpyDeviceToHost);
	
	//char error_statement[] = {"ERROR: reconstruction_cuts GPU->host data transfer caused...\n"};
	//cudaError_t cudaStatus =  CUDA_error_check( error_statement );
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) 
		printf("reconstruction_cuts_device_2_host Error: %s\n", cudaGetErrorString(cudaStatus));
}
void statistical_cuts_deallocations()
{
	cudaFree(bin_num_d);
	cudaFree(WEPL_d);
	cudaFree(xy_entry_angle_d);
	cudaFree(xz_entry_angle_d);
	cudaFree(xy_exit_angle_d);
	cudaFree(xz_exit_angle_d);
	cudaFree(failed_cuts_d);

	free(failed_cuts_h);
	/* 
		Host memory allocated but not freed here
		failed_cuts_h
	*/
	/* 
		Device memory allocated but not freed here
		bin_num_d;
		WEPL_d;
		xy_entry_angle_d
		xz_entry_angle_d
		xy_exit_angle_d
		xz_exit_angle_d
		failed_cuts_d
	*/
	
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) 
		printf("reconstruction_cuts_deallocations Error: %s\n", cudaGetErrorString(cudaStatus));
}
void statistical_cuts( const int start_position, const int num_histories )
{
	unsigned int size_floats = sizeof(float) * num_histories;
	unsigned int size_ints = sizeof(int) * num_histories;
	unsigned int size_bools = sizeof(bool) * num_histories;

	failed_cuts_h = (bool*) calloc ( num_histories, sizeof(bool) );
	
	cudaMalloc( (void**) &bin_num_d,			size_ints );
	cudaMalloc( (void**) &WEPL_d,				size_floats );
	cudaMalloc( (void**) &xy_entry_angle_d,		size_floats );
	cudaMalloc( (void**) &xz_entry_angle_d,		size_floats );
	cudaMalloc( (void**) &xy_exit_angle_d,		size_floats );
	cudaMalloc( (void**) &xz_exit_angle_d,		size_floats );
	cudaMalloc( (void**) &failed_cuts_d,		size_bools );

	cudaMemcpy( bin_num_d,				&bin_num_vector[start_position],			size_ints,		cudaMemcpyHostToDevice );
	cudaMemcpy( WEPL_d,					&WEPL_vector[start_position],				size_floats,	cudaMemcpyHostToDevice );
	cudaMemcpy( xy_entry_angle_d,		&xy_entry_angle_vector[start_position],		size_floats,	cudaMemcpyHostToDevice );
	cudaMemcpy( xz_entry_angle_d,		&xz_entry_angle_vector[start_position],		size_floats,	cudaMemcpyHostToDevice );
	cudaMemcpy( xy_exit_angle_d,		&xy_exit_angle_vector[start_position],		size_floats,	cudaMemcpyHostToDevice );
	cudaMemcpy( xz_exit_angle_d,		&xz_exit_angle_vector[start_position],		size_floats,	cudaMemcpyHostToDevice );
	cudaMemcpy( failed_cuts_d,			failed_cuts_h,								size_bools,		cudaMemcpyHostToDevice );

	//statistical_allocations(num_histories);
	//statistical_host_2_device(start_position, num_histories);
	
	dim3 dimBlock(THREADS_PER_BLOCK);
	dim3 dimGrid( int( num_histories / THREADS_PER_BLOCK ) + 1 );  
	statistical_cuts_GPU<<< dimGrid, dimBlock >>>
	( 
		num_histories, bin_counts_d, bin_num_d, sinogram_d, WEPL_d, 
		xy_entry_angle_d, xz_entry_angle_d, xy_exit_angle_d, xz_exit_angle_d, 
		mean_WEPL_d, mean_rel_ut_angle_d, mean_rel_uv_angle_d, 
		stddev_WEPL_d, stddev_rel_ut_angle_d, stddev_rel_uv_angle_d, 
		failed_cuts_d
	);
	cudaMemcpy( failed_cuts_h, failed_cuts_d, size_bools, cudaMemcpyDeviceToHost);

	// Shift valid data (i.e. failed_cuts = FALSE) to the left, overwriting data from histories that did not pass through the reconstruction volume
	// 
	for( unsigned int i = 0; i < num_histories; i++ )
	{
		if( !failed_cuts_h[i] )
		{
			bin_num_vector[post_cut_histories] = bin_num_vector[start_position + i];
			//gantry_angle_vector[post_cut_histories] = gantry_angle_vector[start_position + i];
			WEPL_vector[post_cut_histories] = WEPL_vector[start_position + i];
			x_entry_vector[post_cut_histories] = x_entry_vector[start_position + i];
			y_entry_vector[post_cut_histories] = y_entry_vector[start_position + i];
			z_entry_vector[post_cut_histories] = z_entry_vector[start_position + i];
			x_exit_vector[post_cut_histories] = x_exit_vector[start_position + i];
			y_exit_vector[post_cut_histories] = y_exit_vector[start_position + i];
			z_exit_vector[post_cut_histories] = z_exit_vector[start_position + i];
			xy_entry_angle_vector[post_cut_histories] = xy_entry_angle_vector[start_position + i];
			xz_entry_angle_vector[post_cut_histories] = xz_entry_angle_vector[start_position + i];
			xy_exit_angle_vector[post_cut_histories] = xy_exit_angle_vector[start_position + i];
			xz_exit_angle_vector[post_cut_histories] = xz_exit_angle_vector[start_position + i];
			post_cut_histories++;
		}
	}
	
	cudaFree(bin_num_d);
	cudaFree(WEPL_d);
	cudaFree(xy_entry_angle_d);
	cudaFree(xz_entry_angle_d);
	cudaFree(xy_exit_angle_d);
	cudaFree(xz_exit_angle_d);
	cudaFree(failed_cuts_d);

	free(failed_cuts_h);
	/* 
		Host memory allocated but not freed here
		failed_cuts_h
	*/
	/* 
		Device memory allocated but not freed here
		bin_num_d;
		WEPL_d;
		xy_entry_angle_d
		xz_entry_angle_d
		xy_exit_angle_d
		xz_exit_angle_d
		failed_cuts_d
	*/
}
__global__ void statistical_cuts_GPU
( 
	int num_histories, int* bin_counts, int* bin_num, float* sinogram, float* WEPL, 
	float* xy_entry_angle, float* xz_entry_angle, float* xy_exit_angle, float* xz_exit_angle, 
	float* mean_WEPL, float* mean_rel_ut_angle, float* mean_rel_uv_angle,
	float* stddev_WEPL, float* stddev_rel_ut_angle, float* stddev_rel_uv_angle, 
	bool* failed_cuts
)
{
	int i = threadIdx.x + blockIdx.x * THREADS_PER_BLOCK;
	if( i < num_histories )
	{
		double rel_ut_angle = xy_exit_angle[i] - xy_entry_angle[i];
		if( rel_ut_angle > PI )
			rel_ut_angle -= 2 * PI;
		if( rel_ut_angle < -PI )
			rel_ut_angle += 2 * PI;
		double rel_uv_angle = xz_exit_angle[i] - xz_entry_angle[i];
		if( rel_uv_angle > PI )
			rel_uv_angle -= 2 * PI;
		if( rel_uv_angle < -PI )
			rel_uv_angle += 2 * PI;
		bool passed_ut_cut = ( abs( rel_ut_angle - mean_rel_ut_angle[bin_num[i]] ) < ( SIGMAS_TO_KEEP * stddev_rel_ut_angle[bin_num[i]] ) );
		bool passed_uv_cut = ( abs( rel_uv_angle - mean_rel_uv_angle[bin_num[i]] ) < ( SIGMAS_TO_KEEP * stddev_rel_uv_angle[bin_num[i]] ) );
		//bool passed_uv_cut = true;
		bool passed_WEPL_cut = ( abs( mean_WEPL[bin_num[i]] - WEPL[i] ) <= ( SIGMAS_TO_KEEP * stddev_WEPL[bin_num[i]] ) );
		failed_cuts[i] = !passed_ut_cut || !passed_uv_cut || !passed_WEPL_cut;

		if( !failed_cuts[i] )
		{
			atomicAdd( &bin_counts[bin_num[i]], 1 );
			atomicAdd( &sinogram[bin_num[i]], WEPL[i] );			
		}
	}
}
void statistical_calculations_and_cuts()
{
	/********************************************************************************************************************************************************/
	/* Calculate the standard deviation in WEPL, relative ut-angle, and relative uv-angle for each bin.  Iterate through the valid history std::vectors one	*/
	/* chunk at a time, with at most MAX_GPU_HISTORIES per chunk, and calculate the difference between the mean WEPL and WEPL, mean relative ut-angle and	*/ 
	/* relative ut-angle, and mean relative uv-angle and relative uv-angle for each history. The standard deviation is then found by calculating the sum	*/
	/* of these differences for each bin and dividing it by the number of histories in the bin 																*/
	/********************************************************************************************************************************************************/
	sprintf( print_statement, "Calculating the statistical distribution for each of the WEPL and relative ut/uv angle data bins and removing statistical outliers");
	print_section_header( print_statement, MINOR_SECTION_SEPARATOR, LIGHT_CYAN_TEXT, YELLOW_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT );
	int remaining_histories = recon_vol_histories, histories_2_process;
	int start_position = 0;
	calculate_means();
	initialize_stddev();
	while( remaining_histories > 0 )
	{
		if( remaining_histories > MAX_CUTS_HISTORIES )
			histories_2_process = MAX_CUTS_HISTORIES;
		else
			histories_2_process = remaining_histories;
		sum_squared_deviations( start_position, histories_2_process );
		remaining_histories -= MAX_CUTS_HISTORIES;
		start_position		+= MAX_CUTS_HISTORIES;
	} 
	calculate_standard_deviations();
	/********************************************************************************************************************************************************/
	/* Iterate through the valid history vectors one chunk at a time, with at most MAX_GPU_HISTORIES per chunk, and perform statistical cuts				*/
	/********************************************************************************************************************************************************/
	remaining_histories = recon_vol_histories, start_position = 0;
	initialize_sinogram();

	print_colored_text( "Performing statistical cuts...", CYAN_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );	
	while( remaining_histories > 0 )
	{
		if( remaining_histories > MAX_CUTS_HISTORIES )
			histories_2_process = MAX_CUTS_HISTORIES;
		else
			histories_2_process = remaining_histories;
		statistical_cuts( start_position, histories_2_process );
		remaining_histories -= MAX_CUTS_HISTORIES;
		start_position		+= MAX_CUTS_HISTORIES;
	}	
	percentage_pass_statistical_cuts = (double) post_cut_histories / total_histories * 100;
	sprintf(print_statement, "------> %d out of the original %d (%4.2f%%) histories remain after statistical cuts", post_cut_histories, total_histories, percentage_pass_statistical_cuts );
	print_colored_text( print_statement, GREEN_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT );
	post_cut_memory_clean();
	resize_vectors( post_cut_histories );
	//shrink_vectors( post_cut_histories );
	print_section_exit( "Finished statistical cuts", SECTION_EXIT_CSTRING, RED_TEXT, RED_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );
	exit_program_if( EXIT_AFTER_CUTS, "through statistical data cuts" );
}		
/***********************************************************************************************************************************************************************************************************************/
/********************************************************************************************************* FBP *********************************************************************************************************/
/***********************************************************************************************************************************************************************************************************************/
void initialize_sinogram()
{
	char statement[] = "Allocating host and GPU memory and initializing sinogram...";		
	print_colored_text( statement, CYAN_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );	
	sinogram_h = (float*) calloc( NUM_BINS, sizeof(float) );
	if( sinogram_h == NULL )
	{
		puts("ERROR: Memory allocation for sinogram_filtered_h failed.");
		exit(1);
	}
	cudaMalloc((void**) &sinogram_d, SIZE_BINS_FLOAT );
	cudaMemcpy( sinogram_d ,	sinogram_h,	SIZE_BINS_FLOAT, cudaMemcpyHostToDevice );	
}
void construct_sinogram()
{
	/********************************************************************************************************************************************************/
	/* Generate the sinogram from remaining histories after cuts, perform filtered backprojection, and generate/define the hull/initial iterate to use		*/
	/********************************************************************************************************************************************************/
	print_colored_text( "Recalculating the mean WEPL for each bin and constructing the sinogram...", CYAN_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );	

	bin_counts_h		  = (int*)	 calloc( NUM_BINS, sizeof(int) );
	cudaMemcpy(bin_counts_h, bin_counts_d, SIZE_BINS_INT, cudaMemcpyDeviceToHost) ;	
	cudaMemcpy(sinogram_h,  sinogram_d, SIZE_BINS_FLOAT, cudaMemcpyDeviceToHost);
	
	//array_2_disk(SINOGRAM_PRE_FILENAME, OUTPUT_DIRECTORY, OUTPUT_FOLDER_UNIQUE, sinogram_h, T_BINS, ANGULAR_BINS, V_BINS, NUM_BINS, true );
	//array_2_disk( BIN_COUNTS_PRE_FILENAME, OUTPUT_DIRECTORY, OUTPUT_FOLDER_UNIQUE, bin_counts_h, T_BINS, ANGULAR_BINS, V_BINS, NUM_BINS, true );
	
	dim3 dimBlock( T_BINS );
	dim3 dimGrid( V_BINS, ANGULAR_BINS );   
	construct_sinogram_GPU<<< dimGrid, dimBlock >>>( bin_counts_d, sinogram_d );

	if( WRITE_WEPL_DISTS )
	{
		cudaMemcpy( sinogram_h,	sinogram_d,	SIZE_BINS_FLOAT, cudaMemcpyDeviceToHost );
		//int* empty_parameter; //Warning: declared but never used
		//bins_2_disk( "WEPL_dist_post_test2", empty_parameter, sinogram_h, NUM_BINS, MEANS, ALL_BINS, BY_BIN );
	}
	cudaMemcpy(sinogram_h,  sinogram_d, SIZE_BINS_FLOAT, cudaMemcpyDeviceToHost);
	cudaMemcpy(bin_counts_h, bin_counts_d, SIZE_BINS_INT, cudaMemcpyDeviceToHost) ;
	
	//array_2_disk(SINOGRAM_FILENAME, OUTPUT_DIRECTORY, OUTPUT_FOLDER_UNIQUE, sinogram_h, T_BINS, ANGULAR_BINS, V_BINS, NUM_BINS, true );
	//array_2_disk( BIN_COUNTS_POST_FILENAME, OUTPUT_DIRECTORY, OUTPUT_FOLDER_UNIQUE, bin_counts_h, T_BINS, ANGULAR_BINS, V_BINS, NUM_BINS, true );
	
	cudaFree(bin_counts_d);
	print_section_exit( "Finished generating sinogram", SECTION_EXIT_CSTRING, RED_TEXT, RED_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );	
	exit_program_if( EXIT_AFTER_SINOGRAM, "through sinogram generation" );	
}
__global__ void construct_sinogram_GPU( int* bin_counts, float* sinogram )
{
	int v = blockIdx.x, angle = blockIdx.y, t = threadIdx.x;
	int bin = t + angle * T_BINS + v * T_BINS * ANGULAR_BINS;
	if( bin_counts[bin] > 0 )
		sinogram[bin] /= bin_counts[bin];		
}
void FBP()
{
	// Filter the sinogram before backprojecting
	filter();

	free(sinogram_h);
	cudaFree(sinogram_d);
	sprintf(print_statement, "Performing backprojection...");		
	print_colored_text( print_statement, CYAN_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );	
	
	FBP_image_h = (float*) calloc( NUM_VOXELS, sizeof(float) );
	if( FBP_image_h == NULL ) 
	{
		printf("ERROR: Memory not allocated for FBP_image_h!\n");
		exit_program_if(true);
	}

	free(sinogram_filtered_h);
	cudaMalloc((void**) &FBP_image_d, SIZE_IMAGE_FLOAT );
	cudaMemcpy( FBP_image_d, FBP_image_h, SIZE_IMAGE_FLOAT, cudaMemcpyHostToDevice );

	dim3 dimBlock( SLICES );
	dim3 dimGrid( COLUMNS, ROWS );   
	backprojection_GPU<<< dimGrid, dimBlock >>>( sinogram_filtered_d, FBP_image_d );
	cudaFree(sinogram_filtered_d);

	if( WRITE_FBP_IMAGE || MEDIAN_FILTER_FBP || (X_0 == FBP_IMAGE) || (X_0 == HYBRID) )
	{
		print_colored_text( "Copying FBP image to host...", CYAN_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );		
		cudaMemcpy( FBP_image_h, FBP_image_d, SIZE_IMAGE_FLOAT, cudaMemcpyDeviceToHost );		
	}
	if( WRITE_FBP_IMAGE )
	{
		print_colored_text( "Writing FBP image to disk...", CYAN_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );		
		//cudaMemcpy( FBP_image_h, FBP_image_d, SIZE_IMAGE_FLOAT, cudaMemcpyDeviceToHost );
		array_2_disk( FBP_FILENAME, OUTPUT_DIRECTORY, OUTPUT_FOLDER_UNIQUE, FBP_image_h, COLUMNS, ROWS, SLICES, NUM_VOXELS, true );	
		write_PNG(FBP_FILENAME, FBP_image_h);		
	}

	if( IMPORT_FILTERED_FBP)
	{
		print_colored_text( "Importing FBP image from disk...", CYAN_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );		
		float* image = (float*)calloc( NUM_VOXELS, sizeof(float));
		sprintf(IMPORT_FBP_PATH,"%s%s/%s%d%s", OUTPUT_DIRECTORY, OUTPUT_FOLDER_UNIQUE, IMPORT_FBP_FILENAME, 2*FBP_MED_FILTER_RADIUS+1,".bin" );
		import_image( image, IMPORT_FBP_PATH );
		FBP_image_h = image;
	}
	else if( AVG_FILTER_FBP )
	{
		sprintf(print_statement, "Applying average filter to FBP image...");		
		print_colored_text( print_statement, CYAN_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );	
		cudaMalloc((void**) &FBP_image_filtered_d, SIZE_IMAGE_FLOAT );
		cudaMemcpy( FBP_image_filtered_d, FBP_image_filtered_h, SIZE_IMAGE_FLOAT, cudaMemcpyHostToDevice );
		FBP_image_filtered_h = FBP_image_h;
		//float* FBP_image_filtered_d;
		//averaging_filter( FBP_image_h, FBP_image_filtered_d, FBP_FILTER_RADIUS, false, FBP_AVG_FILTER_THRESHOLD );
		averaging_filter( FBP_image_filtered_h, FBP_image_filtered_d, FBP_AVG_FILTER_RADIUS, false, FBP_AVG_FILTER_THRESHOLD );		
		sprintf(print_statement, "Average filtering of FBP complete");		
		print_colored_text( print_statement, RED_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );	
		if( WRITE_AVG_FBP )
		{
			sprintf(print_statement, "Writing average filtered FBP image to disk...");		
			print_colored_text( print_statement, CYAN_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );	
			//cudaMemcpy(FBP_image_h, FBP_image_filtered_d, SIZE_IMAGE_FLOAT, cudaMemcpyDeviceToHost);
			//array_2_disk( "FBP_image_filtered", OUTPUT_DIRECTORY, OUTPUT_FOLDER_UNIQUE, FBP_image_h, COLUMNS, ROWS, SLICES, NUM_VOXELS, true );
			cudaMemcpy(FBP_image_filtered_h, FBP_image_filtered_d, SIZE_IMAGE_FLOAT, cudaMemcpyDeviceToHost) ;
			array_2_disk( FBP_AVG_FILTER_FILENAME, OUTPUT_DIRECTORY, OUTPUT_FOLDER_UNIQUE, FBP_image_filtered_h, COLUMNS, ROWS, SLICES, NUM_VOXELS, true );
			//FBP_image_h = FBP_image_filtered_h;
			write_PNG("FBP_avg_filtered", FBP_image_h);	
		}
		cudaFree(FBP_image_filtered_d);
	}
	else if( MEDIAN_FILTER_FBP && (FBP_MED_FILTER_RADIUS > 0) )
	{
		print_colored_text( "Applying median filter to FBP image...", CYAN_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );	
		FBP_median_filtered_h = (float*)calloc(NUM_VOXELS, sizeof(float));
		//cudaMemcpy( FBP_image_h, FBP_image_d, SIZE_IMAGE_FLOAT, cudaMemcpyDeviceToHost );
		//NTVS_timing_analysis();
		median_filter_2D( FBP_image_h, FBP_median_filtered_h, FBP_MED_FILTER_RADIUS );
		//NTVS_timing_analysis();
		//std::copy(FBP_median_filtered_h, FBP_median_filtered_h + NUM_VOXELS, FBP_image_h);
		//FBP_image_h = FBP_image_filtered_h;
		if( WRITE_MEDIAN_FBP )
		{
			print_colored_text( "Writing median filtered FBP image to disk...", CYAN_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );	
			array_2_disk( FBP_MED_FILTER_FILENAME, OUTPUT_DIRECTORY, OUTPUT_FOLDER_UNIQUE, FBP_image_h, COLUMNS, ROWS, SLICES, NUM_VOXELS, true );		
			write_PNG("FBP_med_filtered", FBP_image_h);	
	
		}
		//cudaMemcpy( FBP_image_h, FBP_image_d, SIZE_IMAGE_FLOAT, cudaMemcpyHostToDevice);
		
	}	
	// Generate FBP hull by thresholding FBP image
	if( ENDPOINTS_HULL == FBP_HULL )
		FBP_image_2_hull();

	// Discard FBP image unless it is to be used as the initial iterate x_0 in iterative image reconstruction
	if( X_0 != FBP_IMAGE && X_0 != HYBRID )
		free(FBP_image_h);
	
	//write_PNG("FBP", FBP_image_h);	
	cudaFree(FBP_image_d);
	print_section_exit( "Finished filtered backprojection", SECTION_EXIT_CSTRING, RED_TEXT, RED_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );	
}
void filter()
{
	char statement[] = "Filtering the sinogram...";	
	//sprintf(statement, "Writing filtered hull to disk...");		
	print_colored_text( statement, CYAN_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );	
			
	sinogram_filtered_h = (float*) calloc( NUM_BINS, sizeof(float) );
	if( sinogram_filtered_h == NULL )
	{
		puts("ERROR: Memory allocation for sinogram_filtered_h failed.");
		exit(1);
	}
	cudaMalloc((void**) &sinogram_filtered_d, SIZE_BINS_FLOAT);
	cudaMemcpy( sinogram_filtered_d, sinogram_filtered_h, SIZE_BINS_FLOAT, cudaMemcpyHostToDevice);

	dim3 dimBlock( T_BINS );
	dim3 dimGrid( V_BINS, ANGULAR_BINS );   	
	filter_GPU<<< dimGrid, dimBlock >>>( sinogram_d, sinogram_filtered_d );
}
__global__ void filter_GPU( float* sinogram, float* sinogram_filtered )
{		
	int v_bin = blockIdx.x, angle_bin = blockIdx.y, t_bin = threadIdx.x;
	int t_bin_ref, t_bin_sep, strip_index; 
	double filtered, t, scale_factor;
	double v = ( v_bin - V_BINS/2 ) * V_BIN_SIZE + V_BIN_SIZE/2.0;
	
	// Loop over strips for this strip
	for( t_bin_ref = 0; t_bin_ref < T_BINS; t_bin_ref++ )
	{
		t = ( t_bin_ref - T_BINS/2 ) * T_BIN_SIZE + T_BIN_SIZE/2.0;
		t_bin_sep = t_bin - t_bin_ref;
		// scale_factor = r . path = cos(theta_{r,path})
		scale_factor = SOURCE_RADIUS / sqrt( SOURCE_RADIUS * SOURCE_RADIUS + t * t + v * v );
		switch( SINOGRAM_FILTER )
		{
			case UNFILTERED: 
				break;
			case RAM_LAK:
				if( t_bin_sep == 0 )
					filtered = 1.0 / ( 4.0 * pow( RAM_LAK_TAU, 2.0 ) );
				else if( t_bin_sep % 2 == 0 )
					filtered = 0;
				else
					filtered = -1.0 / ( pow( RAM_LAK_TAU * PI * t_bin_sep, 2.0 ) );	
				break;
			case SHEPP_LOGAN:
				//filtered = pow( pow(T_BIN_SIZE * PI, 2.0) * ( 1.0 - pow(2 * t_bin_sep, 2.0) ), -1.0 );
				filtered = 1/((T_BIN_SIZE * PI*T_BIN_SIZE * PI) * ( 1.0 - (2 * t_bin_sep*2 * t_bin_sep) ));
		}
		strip_index = ( v_bin * ANGULAR_BINS * T_BINS ) + ( angle_bin * T_BINS );
		sinogram_filtered[strip_index + t_bin] += T_BIN_SIZE * sinogram[strip_index + t_bin_ref] * filtered * scale_factor;
	}
}
void backprojection()
{
	//// Check that we don't have any corruptions up until now
	//for( unsigned int i = 0; i < NUM_BINS; i++ )
	//	if( sinogram_filtered_h[i] != sinogram_filtered_h[i] )
	//		printf("We have a nan in bin #%d\n", i);

	double delta = ANGULAR_BIN_SIZE * ANGLE_TO_RADIANS;
	int voxel;
	double x, y, z;
	double u, t, v;
	double detector_number_t, detector_number_v;
	double eta, epsilon;
	double scale_factor;
	int t_bin, v_bin, bin, bin_below;
	// Loop over the voxels
	for( int slice = 0; slice < SLICES; slice++ )
	{
		for( int column = 0; column < COLUMNS; column++ )
		{

			for( int row = 0; row < ROWS; row++ )
			{
				voxel = column +  ( row * COLUMNS ) + ( slice * COLUMNS * ROWS);
				x = -RECON_CYL_RADIUS + ( column + 0.5 )* VOXEL_WIDTH;
				y = RECON_CYL_RADIUS - (row + 0.5) * VOXEL_HEIGHT;
				z = -RECON_CYL_HEIGHT / 2.0 + (slice + 0.5) * SLICE_THICKNESS;
				// If the voxel is outside the cylinder defining the reconstruction volume, set RSP to air
				if( ( x * x + y * y ) > ( RECON_CYL_RADIUS * RECON_CYL_RADIUS ) )
					FBP_image_h[voxel] = RSP_AIR;							
				else
				{	  
					// Sum over projection angles
					for( int angle_bin = 0; angle_bin < ANGULAR_BINS; angle_bin++ )
					{
						// Rotate the pixel position to the beam-detector coordinate system
						u = x * cos( angle_bin * delta ) + y * sin( angle_bin * delta );
						t = -x * sin( angle_bin * delta ) + y * cos( angle_bin * delta );
						v = z;

						// Project to find the detector number
						detector_number_t = ( t - u *( t / ( SOURCE_RADIUS + u ) ) ) / T_BIN_SIZE + T_BINS/2.0;
						t_bin = int( detector_number_t);
						if( t_bin > detector_number_t )
							t_bin -= 1;
						eta = detector_number_t - t_bin;

						// Now project v to get detector number in v axis
						detector_number_v = ( v - u * ( v / ( SOURCE_RADIUS + u ) ) ) / V_BIN_SIZE + V_BINS/2.0;
						v_bin = int( detector_number_v);
						if( v_bin > detector_number_v )
							v_bin -= 1;
						epsilon = detector_number_v - v_bin;

						// Calculate the fan beam scaling factor
						scale_factor = pow( SOURCE_RADIUS / ( SOURCE_RADIUS + u ), 2 );
		  
						// Compute the back-projection
						bin = t_bin + angle_bin * T_BINS + v_bin * ANGULAR_BINS * T_BINS;
						bin_below = bin + ( ANGULAR_BINS * T_BINS );

						// If in last v_vin, there is no bin below so only use adjacent bins
						if( v_bin == V_BINS - 1 || ( bin < 0 ) )
							FBP_image_h[voxel] += scale_factor * ( ( ( 1 - eta ) * sinogram_filtered_h[bin] ) + ( eta * sinogram_filtered_h[bin + 1] ) ) ;
					/*	if( t_bin < T_BINS - 1 )
								FBP_image_h[voxel] += scale_factor * ( ( ( 1 - eta ) * sinogram_filtered_h[bin] ) + ( eta * sinogram_filtered_h[bin + 1] ) );
							if( v_bin < V_BINS - 1 )
								FBP_image_h[voxel] += scale_factor * ( ( ( 1 - epsilon ) * sinogram_filtered_h[bin] ) + ( epsilon * sinogram_filtered_h[bin_below] ) );
							if( t_bin == T_BINS - 1 && v_bin == V_BINS - 1 )
								FBP_image_h[voxel] += scale_factor * sinogram_filtered_h[bin];*/
						else 
						{
							// Technically this is to be multiplied by delta as well, but since delta is constant, it is more accurate numerically to multiply result by delta instead
							FBP_image_h[voxel] += scale_factor * ( ( 1 - eta ) * ( 1 - epsilon ) * sinogram_filtered_h[bin] 
							+ eta * ( 1 - epsilon ) * sinogram_filtered_h[bin + 1]
							+ ( 1 - eta ) * epsilon * sinogram_filtered_h[bin_below]
							+ eta * epsilon * sinogram_filtered_h[bin_below + 1] );
						} 
					}
					FBP_image_h[voxel] *= delta;
				}
			}
		}
	}
}
__global__ void backprojection_GPU( float* sinogram_filtered, float* FBP_image )
{
	int row = blockIdx.y, column = blockIdx.x, slice = threadIdx.x;
	int voxel = slice * COLUMNS * ROWS + row * COLUMNS + column;	
	if ( voxel < NUM_VOXELS )
	{
		double delta = ANGULAR_BIN_SIZE * ANGLE_TO_RADIANS;
		double u, t, v;
		double detector_number_t, detector_number_v;
		double eta, epsilon;
		double scale_factor;
		int t_bin, v_bin, bin;
		double x = -RECON_CYL_RADIUS + ( column + 0.5 )* VOXEL_WIDTH;
		double y = RECON_CYL_RADIUS - (row + 0.5) * VOXEL_HEIGHT;
		double z = -RECON_CYL_HEIGHT / 2.0 + (slice + 0.5) * SLICE_THICKNESS;

		//// If the voxel is outside a cylinder contained in the reconstruction volume, set to air
		if( ( x * x + y * y ) > ( RECON_CYL_RADIUS * RECON_CYL_RADIUS ) )
			FBP_image[( slice * COLUMNS * ROWS) + ( row * COLUMNS ) + column] = RSP_AIR;							
		else
		{	  
			// Sum over projection angles
			for( int angle_bin = 0; angle_bin < ANGULAR_BINS; angle_bin++ )
			{
				// Rotate the pixel position to the beam-detector coordinate system
				u = x * cos( angle_bin * delta ) + y * sin( angle_bin * delta );
				t = -x * sin( angle_bin * delta ) + y * cos( angle_bin * delta );
				v = z;

				// Project to find the detector number
				detector_number_t = ( t - u *( t / ( SOURCE_RADIUS + u ) ) ) / T_BIN_SIZE + T_BINS/2.0;
				t_bin = int( detector_number_t);
				if( t_bin > detector_number_t )
					t_bin -= 1;
				eta = detector_number_t - t_bin;

				// Now project v to get detector number in v axis
				detector_number_v = ( v - u * ( v / ( SOURCE_RADIUS + u ) ) ) / V_BIN_SIZE + V_BINS/2.0;
				v_bin = int( detector_number_v);
				if( v_bin > detector_number_v )
					v_bin -= 1;
				epsilon = detector_number_v - v_bin;

				// Calculate the fan beam scaling factor
				scale_factor = powf( SOURCE_RADIUS / ( SOURCE_RADIUS + u ), 2 );
		  
				//bin_num[i] = t_bin + angle_bin * T_BINS + v_bin * T_BINS * ANGULAR_BINS;
				// Compute the back-projection
				bin = t_bin + angle_bin * T_BINS + v_bin * ANGULAR_BINS * T_BINS;
				// not sure why this won't compile without calculating the index ahead of time instead inside []s
				//int index = ANGULAR_BINS * T_BINS;

				//if( ( ( bin + ANGULAR_BINS * T_BINS + 1 ) >= NUM_BINS ) || ( bin < 0 ) );
				if( v_bin == V_BINS - 1 || ( bin < 0 ) )
					FBP_image[voxel] += scale_factor * ( ( ( 1 - eta ) * sinogram_filtered[bin] ) + ( eta * sinogram_filtered[bin + 1] ) ) ;
					//printf("The bin selected for this voxel does not exist!\n Slice: %d\n Column: %d\n Row: %d\n", slice, column, row);
				else 
				{
					// not sure why this won't compile without calculating the index ahead of time instead inside []s
					/*FBP_image[voxel] += delta * ( ( 1 - eta ) * ( 1 - epsilon ) * sinogram_filtered[bin] 
					+ eta * ( 1 - epsilon ) * sinogram_filtered[bin + 1]
					+ ( 1 - eta ) * epsilon * sinogram_filtered[bin + ANGULAR_BINS * T_BINS]
					+ eta * epsilon * sinogram_filtered[bin + ANGULAR_BINS * T_BINS + 1] ) * scale_factor;*/

					// Multilpying by the gantry angle interval for each gantry angle is equivalent to multiplying the final answer by 2*PI and is better numerically
					// so multiplying by delta each time should be replaced by FBP_image_h[voxel] *= 2 * PI after all contributions have been made, which is commented out below
					FBP_image[voxel] += scale_factor * ( ( 1 - eta ) * ( 1 - epsilon ) * sinogram_filtered[bin] 
					+ eta * ( 1 - epsilon ) * sinogram_filtered[bin + 1]
					+ ( 1 - eta ) * epsilon * sinogram_filtered[bin + ( ANGULAR_BINS * T_BINS)]
					+ eta * epsilon * sinogram_filtered[bin + ( ANGULAR_BINS * T_BINS) + 1] );
				}				
			}
			FBP_image[voxel] *= delta; 
		}
	}
}
void FBP_image_2_hull()
{
	char statement[] = "Performing thresholding on FBP image to generate FBP hull...";		
	print_colored_text( statement, CYAN_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );	
	
	FBP_hull_h = (bool*) calloc( COLUMNS * ROWS * SLICES, sizeof(bool) );
	initialize_hull( FBP_hull_h, FBP_hull_d );
	dim3 dimBlock( SLICES );
	dim3 dimGrid( COLUMNS, ROWS );   
	FBP_image_2_hull_GPU<<< dimGrid, dimBlock >>>( FBP_image_d, FBP_hull_d );	
	cudaMemcpy( FBP_hull_h, FBP_hull_d, SIZE_IMAGE_BOOL, cudaMemcpyDeviceToHost );
	
	if( WRITE_FBP_HULL )
		array_2_disk( FBP_HULL_FILENAME, OUTPUT_DIRECTORY, OUTPUT_FOLDER_UNIQUE, FBP_hull_h, COLUMNS, ROWS, SLICES, NUM_VOXELS, true );

	if( ENDPOINTS_HULL != FBP_HULL)	
		free(FBP_hull_h);	
	cudaFree(FBP_hull_d);
}
__global__ void FBP_image_2_hull_GPU( float* FBP_image, bool* FBP_hull )
{
	int row = blockIdx.y, column = blockIdx.x, slice = threadIdx.x;
	int voxel = slice * COLUMNS * ROWS + row * COLUMNS + column; 
	double x = -RECON_CYL_RADIUS + ( column + 0.5 )* VOXEL_WIDTH;
	double y = RECON_CYL_RADIUS - (row + 0.5) * VOXEL_HEIGHT;
	double d_squared = powf(x, 2) + powf(y, 2);
	if(FBP_image[voxel] > FBP_THRESHOLD && (d_squared < powf(RECON_CYL_RADIUS, 2) ) ) 
		FBP_hull[voxel] = true; 
	else
		FBP_hull[voxel] = false; 
}
/***********************************************************************************************************************************************************************************************************************/
/*************************************************************************************************** Hull-Detection ****************************************************************************************************/
/***********************************************************************************************************************************************************************************************************************/
void hull_detection( const int histories_2_process)
{
	if( SC_ON  ) 
		SC( histories_2_process );		
	if( MSC_ON )
		MSC( histories_2_process );
	if( SM_ON )
		SM( histories_2_process );   
}
__global__ void carve_differences( int* carve_differences, int* image )
{
	int row = blockIdx.y, column = blockIdx.x, slice = threadIdx.x;
	int voxel = column + row * COLUMNS + slice * COLUMNS * ROWS;
	if( (row != 0) && (row != ROWS - 1) && (column != 0) && (column != COLUMNS - 1) )
	{
		int difference, max_difference = 0;
		for( int current_row = row - 1; current_row <= row + 1; current_row++ )
		{
			for( int current_column = column - 1; current_column <= column + 1; current_column++ )
			{
				difference = image[voxel] - image[current_column + current_row * COLUMNS + slice * COLUMNS * ROWS];
				if( difference > max_difference )
					max_difference = difference;
			}
		}
		carve_differences[voxel] = max_difference;
	}
}
/***********************************************************************************************************************************************************************************************************************/
void hull_initializations()
{		
	print_colored_text( "Initializing hull detection arrays...", CYAN_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );
	if( SC_ON )
		initialize_hull( SC_hull_h, SC_hull_d );
	if( MSC_ON )
		initialize_hull( MSC_counts_h, MSC_counts_d );
	if( SM_ON )
		initialize_hull( SM_counts_h, SM_counts_d );
}
template<typename T> void initialize_hull( T*& hull_h, T*& hull_d )
{
	/* Allocate memory and initialize hull on the GPU.  Use the image and reconstruction cylinder configurations to determine the location of the perimeter of  */
	/* the reconstruction cylinder, which is centered on the origin (center) of the image.  Assign voxels inside the perimeter of the reconstruction volume */
	/* the value 1 and those outside 0.																														*/

	int image_size = NUM_VOXELS * sizeof(T);
	cudaMalloc((void**) &hull_d, image_size );

	dim3 dimBlock( SLICES );
	dim3 dimGrid( COLUMNS, ROWS );   
	initialize_hull_GPU<<< dimGrid, dimBlock >>>( hull_d );	
}
template<typename T> __global__ void initialize_hull_GPU( T* hull )
{
	int row = blockIdx.y, column = blockIdx.x, slice = threadIdx.x;
	int voxel = column + ( row * COLUMNS ) + ( slice * COLUMNS * ROWS );
	double x = ( column - COLUMNS/2 + 0.5) * VOXEL_WIDTH;
	double y = ( ROWS/2 - row - 0.5) * VOXEL_HEIGHT;
	if( powf(x, 2) + powf(y, 2) < powf(RECON_CYL_RADIUS, 2) )
		hull[voxel] = 1;
	else
		hull[voxel] = 0;
}
void SC( const int num_histories )
{
	dim3 dimBlock(THREADS_PER_BLOCK);
	dim3 dimGrid( (int)( num_histories / THREADS_PER_BLOCK ) + 1 );
	SC_GPU<<<dimGrid, dimBlock>>>
	(
		num_histories, SC_hull_d, bin_num_d, missed_recon_volume_d, WEPL_d,
		x_entry_d, y_entry_d, z_entry_d, x_exit_d, y_exit_d, z_exit_d
	);
	//pause_execution();
}
__global__ void SC_GPU
( 
	const int num_histories, bool* SC_hull, int* bin_num, bool* missed_recon_volume, float* WEPL,
	float* x_entry, float* y_entry, float* z_entry, float* x_exit, float* y_exit, float* z_exit
)
{// 15 doubles, 11 integers, 2 booleans
	int i = threadIdx.x + blockIdx.x * THREADS_PER_BLOCK;
	if( (i < num_histories) && !missed_recon_volume[i] && (WEPL[i] <= SC_UPPER_THRESHOLD) && (WEPL[i] >= SC_LOWER_THRESHOLD) )
	{
		/********************************************************************************************/
		/************************** Path Characteristic Parameters **********************************/
		/********************************************************************************************/
		int x_move_direction, y_move_direction, z_move_direction;
		double dy_dx, dz_dx, dz_dy, dx_dy, dx_dz, dy_dz;
		/********************************************************************************************/
		/**************************** Status Tracking Information ***********************************/
		/********************************************************************************************/
		double x = x_entry[i], y = y_entry[i], z = z_entry[i];
		double x_to_go, y_to_go, z_to_go;		
		//double x_extension, y_extension;	
		int voxel_x, voxel_y, voxel_z, voxel;
		int voxel_x_out, voxel_y_out, voxel_z_out, voxel_out; 
		bool end_walk;
		//bool debug_run = false;
		/********************************************************************************************/
		/******************** Initial Conditions and Movement Characteristics ***********************/
		/********************************************************************************************/
		x_move_direction = ( x_entry[i] <= x_exit[i] ) - ( x_entry[i] >= x_exit[i] );
		y_move_direction = ( y_entry[i] <= y_exit[i] ) - ( y_entry[i] >= y_exit[i] );
		z_move_direction = ( z_entry[i] <= z_exit[i] ) - ( z_entry[i] >= z_exit[i] );		

		voxel_x = calculate_voxel_GPU( X_ZERO_COORDINATE, x_entry[i], VOXEL_WIDTH );
		voxel_y = calculate_voxel_GPU( Y_ZERO_COORDINATE, y_entry[i], VOXEL_HEIGHT );
		voxel_z = calculate_voxel_GPU( Z_ZERO_COORDINATE, z_entry[i], VOXEL_THICKNESS );		
		voxel = voxel_x + voxel_y * COLUMNS + voxel_z * COLUMNS * ROWS;

		x_to_go = distance_remaining_GPU( X_ZERO_COORDINATE,	x, X_INCREASING_DIRECTION, x_move_direction, VOXEL_WIDTH,	 voxel_x );
		y_to_go = distance_remaining_GPU( Y_ZERO_COORDINATE,	y, Y_INCREASING_DIRECTION,  y_move_direction, VOXEL_HEIGHT,	 voxel_y );
		z_to_go = distance_remaining_GPU( Z_ZERO_COORDINATE,	z, Z_INCREASING_DIRECTION,  z_move_direction, VOXEL_THICKNESS, voxel_z );				
		/********************************************************************************************/
		/***************************** Path and Walk Information ************************************/
		/********************************************************************************************/
		// Slopes corresponging to each possible direction/reference.  Explicitly calculated inverses to avoid 1/# calculations later
		dy_dx = ( y_exit[i] - y_entry[i] ) / ( x_exit[i] - x_entry[i] );
		dz_dx = ( z_exit[i] - z_entry[i] ) / ( x_exit[i] - x_entry[i] );
		dz_dy = ( z_exit[i] - z_entry[i] ) / ( y_exit[i] - y_entry[i] );
		dx_dy = ( x_exit[i] - x_entry[i] ) / ( y_exit[i] - y_entry[i] );
		dx_dz = ( x_exit[i] - x_entry[i] ) / ( z_exit[i] - z_entry[i] );
		dy_dz = ( y_exit[i] - y_entry[i] ) / ( z_exit[i] - z_entry[i] );
		/********************************************************************************************/
		/************************* Initialize and Check Exit Conditions *****************************/
		/********************************************************************************************/
		voxel_x_out = calculate_voxel_GPU( X_ZERO_COORDINATE, x_exit[i], VOXEL_WIDTH );
		voxel_y_out = calculate_voxel_GPU( Y_ZERO_COORDINATE, y_exit[i], VOXEL_HEIGHT );
		voxel_z_out = calculate_voxel_GPU( Z_ZERO_COORDINATE, z_exit[i], VOXEL_THICKNESS );
		voxel_out = voxel_x_out + voxel_y_out * COLUMNS + voxel_z_out * COLUMNS * ROWS;

		end_walk = ( voxel == voxel_out ) || ( voxel_x >= COLUMNS ) || ( voxel_y >= ROWS ) || ( voxel_z >= SLICES );
		if( !end_walk )
			SC_hull[voxel] = 0;
		/********************************************************************************************/
		/*********************************** Voxel Walk Routine *************************************/
		/********************************************************************************************/
		if( z_move_direction != 0 )
		{
			//printf("z_exit[i] != z_entry[i]\n");
			while( !end_walk )
			{
				take_3D_step_GPU
				( 
					x_move_direction, y_move_direction, z_move_direction,
					dy_dx, dz_dx, dz_dy, 
					dx_dy, dx_dz, dy_dz, 
					x_entry[i], y_entry[i], z_entry[i], 
					x, y, z, 
					voxel_x, voxel_y, voxel_z, voxel,
					x_to_go, y_to_go, z_to_go
				);
				//voxel = voxel_x + voxel_y * COLUMNS + voxel_z * COLUMNS * ROWS;
				end_walk = ( voxel == voxel_out ) || ( voxel_x >= COLUMNS ) || ( voxel_y >= ROWS ) || ( voxel_z >= SLICES );
				if( !end_walk )
					SC_hull[voxel] = 0;
			}// end !end_walk 
		}
		else
		{
			//printf("z_exit[i] == z_entry[i]\n");
			while( !end_walk )
			{
				take_2D_step_GPU
				( 
					x_move_direction, y_move_direction, z_move_direction,
					dy_dx, dz_dx, dz_dy, 
					dx_dy, dx_dz, dy_dz, 
					x_entry[i], y_entry[i], z_entry[i], 
					x, y, z, 
					voxel_x, voxel_y, voxel_z, voxel,
					x_to_go, y_to_go, z_to_go
				);
				//voxel = voxel_x + voxel_y * COLUMNS + voxel_z * COLUMNS * ROWS;
				end_walk = ( voxel == voxel_out ) || ( voxel_x >= COLUMNS ) || ( voxel_y >= ROWS ) || ( voxel_z >= SLICES );
				if( !end_walk )
					SC_hull[voxel] = 0;
			}// end: while( !end_walk )
		}//end: else: z_start != z_end => z_start == z_end
	}// end: if( (i < num_histories) && !missed_recon_volume[i] && (WEPL[i] <= SC_THRESHOLD) )
}
/***********************************************************************************************************************************************************************************************************************/
void MSC( const int num_histories )
{
	dim3 dimBlock(THREADS_PER_BLOCK);
	dim3 dimGrid((int)(num_histories/THREADS_PER_BLOCK)+1);
	MSC_GPU<<<dimGrid, dimBlock>>>
	(
		num_histories, MSC_counts_d, bin_num_d, missed_recon_volume_d, WEPL_d,
		x_entry_d, y_entry_d, z_entry_d, x_exit_d, y_exit_d, z_exit_d
	);
}
__global__ void MSC_GPU
( 
	const int num_histories, int* MSC_counts, int* bin_num, bool* missed_recon_volume, float* WEPL,
	float* x_entry, float* y_entry, float* z_entry, float* x_exit, float* y_exit, float* z_exit
)
{
	int i = threadIdx.x + blockIdx.x * THREADS_PER_BLOCK;
	if( (i < num_histories) && !missed_recon_volume[i] && (WEPL[i] <= MSC_UPPER_THRESHOLD) && (WEPL[i] >= MSC_LOWER_THRESHOLD) )
	{
		/********************************************************************************************/
		/************************** Path Characteristic Parameters **********************************/
		/********************************************************************************************/
		int x_move_direction, y_move_direction, z_move_direction;
		double dy_dx, dz_dx, dz_dy, dx_dy, dx_dz, dy_dz;
		/********************************************************************************************/
		/**************************** Status Tracking Information ***********************************/
		/********************************************************************************************/
		double x = x_entry[i], y = y_entry[i], z = z_entry[i];
		double x_to_go, y_to_go, z_to_go;		
		//double x_extension, y_extension;	
		int voxel_x, voxel_y, voxel_z, voxel;
		int voxel_x_out, voxel_y_out, voxel_z_out, voxel_out; 
		bool end_walk;
		//bool debug_run = false;
		/********************************************************************************************/
		/******************** Initial Conditions and Movement Characteristics ***********************/
		/********************************************************************************************/
		x_move_direction = ( x_entry[i] <= x_exit[i] ) - ( x_entry[i] >= x_exit[i] );
		y_move_direction = ( y_entry[i] <= y_exit[i] ) - ( y_entry[i] >= y_exit[i] );
		z_move_direction = ( z_entry[i] <= z_exit[i] ) - ( z_entry[i] >= z_exit[i] );		

		voxel_x = calculate_voxel_GPU( X_ZERO_COORDINATE, x_entry[i], VOXEL_WIDTH );
		voxel_y = calculate_voxel_GPU( Y_ZERO_COORDINATE, y_entry[i], VOXEL_HEIGHT );
		voxel_z = calculate_voxel_GPU( Z_ZERO_COORDINATE, z_entry[i], VOXEL_THICKNESS );		
		voxel = voxel_x + voxel_y * COLUMNS + voxel_z * COLUMNS * ROWS;

		x_to_go = distance_remaining_GPU( X_ZERO_COORDINATE,	x, X_INCREASING_DIRECTION, x_move_direction, VOXEL_WIDTH,	 voxel_x );
		y_to_go = distance_remaining_GPU( Y_ZERO_COORDINATE,	y, Y_INCREASING_DIRECTION,  y_move_direction, VOXEL_HEIGHT,	 voxel_y );
		z_to_go = distance_remaining_GPU( Z_ZERO_COORDINATE,	z, Z_INCREASING_DIRECTION,  z_move_direction, VOXEL_THICKNESS, voxel_z );				
		/********************************************************************************************/
		/***************************** Path and Walk Information ************************************/
		/********************************************************************************************/
		// Slopes corresponging to each possible direction/reference.  Explicitly calculated inverses to avoid 1/# calculations later
		dy_dx = ( y_exit[i] - y_entry[i] ) / ( x_exit[i] - x_entry[i] );
		dz_dx = ( z_exit[i] - z_entry[i] ) / ( x_exit[i] - x_entry[i] );
		dz_dy = ( z_exit[i] - z_entry[i] ) / ( y_exit[i] - y_entry[i] );
		dx_dy = ( x_exit[i] - x_entry[i] ) / ( y_exit[i] - y_entry[i] );
		dx_dz = ( x_exit[i] - x_entry[i] ) / ( z_exit[i] - z_entry[i] );
		dy_dz = ( y_exit[i] - y_entry[i] ) / ( z_exit[i] - z_entry[i] );
		/********************************************************************************************/
		/************************* Initialize and Check Exit Conditions *****************************/
		/********************************************************************************************/
		voxel_x_out = calculate_voxel_GPU( X_ZERO_COORDINATE, x_exit[i], VOXEL_WIDTH );
		voxel_y_out = calculate_voxel_GPU( Y_ZERO_COORDINATE, y_exit[i], VOXEL_HEIGHT );
		voxel_z_out = calculate_voxel_GPU( Z_ZERO_COORDINATE, z_exit[i], VOXEL_THICKNESS );
		voxel_out = voxel_x_out + voxel_y_out * COLUMNS + voxel_z_out * COLUMNS * ROWS;

		end_walk = ( voxel == voxel_out ) || ( voxel_x >= COLUMNS ) || ( voxel_y >= ROWS ) || ( voxel_z >= SLICES );
		if( !end_walk )
			atomicAdd(&MSC_counts[voxel], 1);
		/********************************************************************************************/
		/*********************************** Voxel Walk Routine *************************************/
		/********************************************************************************************/
		if( z_move_direction != 0 )
		{
			//printf("z_exit[i] != z_entry[i]\n");
			while( !end_walk )
			{
				take_3D_step_GPU
				( 
					x_move_direction, y_move_direction, z_move_direction,
					dy_dx, dz_dx, dz_dy, dx_dy, dx_dz, dy_dz, 
					x_entry[i], y_entry[i], z_entry[i], 
					x, y, z, 
					voxel_x, voxel_y, voxel_z, voxel,
					x_to_go, y_to_go, z_to_go
				);
				end_walk = ( voxel == voxel_out ) || ( voxel_x >= COLUMNS ) || ( voxel_y >= ROWS ) || ( voxel_z >= SLICES );
				if( !end_walk )
					atomicAdd(&MSC_counts[voxel], 1);
			}// end !end_walk 
		}
		else
		{
			//printf("z_exit[i] == z_entry[i]\n");
			while( !end_walk )
			{
				take_2D_step_GPU
				( 
					x_move_direction, y_move_direction, z_move_direction,
					dy_dx, dz_dx, dz_dy, dx_dy, dx_dz, dy_dz, 
					x_entry[i], y_entry[i], z_entry[i], 
					x, y, z, 
					voxel_x, voxel_y, voxel_z, voxel,
					x_to_go, y_to_go, z_to_go
				);				
				end_walk = ( voxel == voxel_out ) || ( voxel_x >= COLUMNS ) || ( voxel_y >= ROWS ) || ( voxel_z >= SLICES );
				if( !end_walk )
					atomicAdd(&MSC_counts[voxel], 1);
			}// end: while( !end_walk )
		}//end: else: z_start != z_end => z_start == z_end
	}// end: if( (i < num_histories) && !missed_recon_volume[i] && (WEPL[i] <= MSC_UPPER_THRESHOLD) )
}
void MSC_edge_detection()
{
	print_colored_text( "Performing edge-detection on MSC_counts...", CYAN_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );	
	
	dim3 dimBlock( SLICES );
	dim3 dimGrid( COLUMNS, ROWS );   	
	MSC_edge_detection_GPU<<< dimGrid, dimBlock >>>( MSC_counts_d );
	//MSC_edge_detection_GPU<<< dimGrid, dimBlock >>>( MSC_counts_d, MSC_counts_output_d );
}
__global__ void MSC_edge_detection_GPU( int* MSC_counts )
{
	int row = blockIdx.y, column = blockIdx.x, slice = threadIdx.x;
	int voxel = column + row * COLUMNS + slice * COLUMNS * ROWS;
	float x = ( column - COLUMNS/2 + 0.5 ) * VOXEL_WIDTH;
	float y = ( ROWS/2 - row - 0.5 ) * VOXEL_HEIGHT;
	int difference, max_difference = 0;
	if( (row != 0) && (row != ROWS - 1) && (column != 0) && (column != COLUMNS - 1) )
	{		
		for( int current_row = row - 1; current_row <= row + 1; current_row++ )
		{
			for( int current_column = column - 1; current_column <= column + 1; current_column++ )
			{
				difference = MSC_counts[voxel] - MSC_counts[current_column + current_row * COLUMNS + slice * COLUMNS * ROWS];
				if( difference > max_difference )
					max_difference = difference;
			}
		}
	}
	__syncthreads();
	if( max_difference > MSC_DIFF_THRESH || MSC_counts[voxel] > MSC_DIFF_THRESH)
		MSC_counts[voxel] = 0;
	//if( MSC_counts[voxel] > MSC_DIFF_THRESH)
	//	MSC_counts[voxel] = 0;
	//if( max_difference > MSC_DIFF_THRESH )
	//	MSC_counts[voxel] = 0;
	else
		MSC_counts[voxel] = 1;
	if( powf(x, 2) + powf(y, 2) >= powf(RECON_CYL_RADIUS - max(VOXEL_WIDTH, VOXEL_HEIGHT)/2, 2 ) )
		MSC_counts[voxel] = 0;

}
__global__ void MSC_edge_detection_GPU( int* MSC_counts, int* MSC_counts_output )
{
	int row = blockIdx.y, column = blockIdx.x, slice = threadIdx.x;
	int voxel = column + row * COLUMNS + slice * COLUMNS * ROWS;
	float x = ( column - COLUMNS/2 + 0.5 ) * VOXEL_WIDTH;
	float y = ( ROWS/2 - row - 0.5 ) * VOXEL_HEIGHT;
	int difference, max_difference = 0;
	if( (row != 0) && (row != ROWS - 1) && (column != 0) && (column != COLUMNS - 1) )
	{		
		for( int current_row = row - 1; current_row <= row + 1; current_row++ )
		{
			for( int current_column = column - 1; current_column <= column + 1; current_column++ )
			{
				difference = MSC_counts[voxel] - MSC_counts[current_column + current_row * COLUMNS + slice * COLUMNS * ROWS];
				if( difference > max_difference )
					max_difference = difference;
			}
		}
	}
	__syncthreads();
	if( max_difference > MSC_DIFF_THRESH )
		MSC_counts_output[voxel] = 0;
	else
		MSC_counts_output[voxel] = 1;
	if( powf(x, 2) + powf(y, 2) >= powf(RECON_CYL_RADIUS - max(VOXEL_WIDTH, VOXEL_HEIGHT)/2, 2 ) )
		MSC_counts_output[voxel] = 0;

}
/***********************************************************************************************************************************************************************************************************************/
void SM( const int num_histories)
{
	dim3 dimBlock(THREADS_PER_BLOCK);
	dim3 dimGrid( (int)( num_histories / THREADS_PER_BLOCK ) + 1 );
	SM_GPU <<< dimGrid, dimBlock >>>
	(
		num_histories, SM_counts_d, bin_num_d, missed_recon_volume_d, WEPL_d,
		x_entry_d, y_entry_d, z_entry_d, x_exit_d, y_exit_d, z_exit_d
	);
}
__global__ void SM_GPU
( 
	const int num_histories, int* SM_counts, int* bin_num, bool* missed_recon_volume, float* WEPL,
	float* x_entry, float* y_entry, float* z_entry, float* x_exit, float* y_exit, float* z_exit
)
{
	int i = threadIdx.x + blockIdx.x * THREADS_PER_BLOCK;
	if( (i < num_histories) && !missed_recon_volume[i] && (WEPL[i] >= SM_LOWER_THRESHOLD) )
	{
		/********************************************************************************************/
		/************************** Path Characteristic Parameters **********************************/
		/********************************************************************************************/
		int x_move_direction, y_move_direction, z_move_direction;
		double dy_dx, dz_dx, dz_dy, dx_dy, dx_dz, dy_dz;
		/********************************************************************************************/
		/**************************** Status Tracking Information ***********************************/
		/********************************************************************************************/
		double x = x_entry[i], y = y_entry[i], z = z_entry[i];
		double x_to_go, y_to_go, z_to_go;		
		//double x_extension, y_extension;	
		int voxel_x, voxel_y, voxel_z, voxel;
		int voxel_x_out, voxel_y_out, voxel_z_out, voxel_out; 
		bool end_walk;
		//bool debug_run = false;
		/********************************************************************************************/
		/******************** Initial Conditions and Movement Characteristics ***********************/
		/********************************************************************************************/
		x_move_direction = ( x_entry[i] <= x_exit[i] ) - ( x_entry[i] >= x_exit[i] );
		y_move_direction = ( y_entry[i] <= y_exit[i] ) - ( y_entry[i] >= y_exit[i] );
		z_move_direction = ( z_entry[i] <= z_exit[i] ) - ( z_entry[i] >= z_exit[i] );		

		voxel_x = calculate_voxel_GPU( X_ZERO_COORDINATE, x_entry[i], VOXEL_WIDTH );
		voxel_y = calculate_voxel_GPU( Y_ZERO_COORDINATE, y_entry[i], VOXEL_HEIGHT );
		voxel_z = calculate_voxel_GPU( Z_ZERO_COORDINATE, z_entry[i], VOXEL_THICKNESS );		
		voxel = voxel_x + voxel_y * COLUMNS + voxel_z * COLUMNS * ROWS;

		x_to_go = distance_remaining_GPU( X_ZERO_COORDINATE,	x, X_INCREASING_DIRECTION, x_move_direction, VOXEL_WIDTH,	 voxel_x );
		y_to_go = distance_remaining_GPU( Y_ZERO_COORDINATE,	y, Y_INCREASING_DIRECTION,  y_move_direction, VOXEL_HEIGHT,	 voxel_y );
		z_to_go = distance_remaining_GPU( Z_ZERO_COORDINATE,	z, Z_INCREASING_DIRECTION,  z_move_direction, VOXEL_THICKNESS, voxel_z );				
		/********************************************************************************************/
		/***************************** Path and Walk Information ************************************/
		/********************************************************************************************/
		// Slopes corresponging to each possible direction/reference.  Explicitly calculated inverses to avoid 1/# calculations later
		dy_dx = ( y_exit[i] - y_entry[i] ) / ( x_exit[i] - x_entry[i] );
		dz_dx = ( z_exit[i] - z_entry[i] ) / ( x_exit[i] - x_entry[i] );
		dz_dy = ( z_exit[i] - z_entry[i] ) / ( y_exit[i] - y_entry[i] );
		dx_dy = ( x_exit[i] - x_entry[i] ) / ( y_exit[i] - y_entry[i] );
		dx_dz = ( x_exit[i] - x_entry[i] ) / ( z_exit[i] - z_entry[i] );
		dy_dz = ( y_exit[i] - y_entry[i] ) / ( z_exit[i] - z_entry[i] );
		/********************************************************************************************/
		/************************* Initialize and Check Exit Conditions *****************************/
		/********************************************************************************************/
		voxel_x_out = calculate_voxel_GPU( X_ZERO_COORDINATE, x_exit[i], VOXEL_WIDTH );
		voxel_y_out = calculate_voxel_GPU( Y_ZERO_COORDINATE, y_exit[i], VOXEL_HEIGHT );
		voxel_z_out = calculate_voxel_GPU( Z_ZERO_COORDINATE, z_exit[i], VOXEL_THICKNESS );
		voxel_out = voxel_x_out + voxel_y_out * COLUMNS + voxel_z_out * COLUMNS * ROWS;

		end_walk = ( voxel == voxel_out ) || ( voxel_x >= COLUMNS ) || ( voxel_y >= ROWS ) || ( voxel_z >= SLICES );
		if( !end_walk )
			atomicAdd(&SM_counts[voxel], 1);
		/********************************************************************************************/
		/*********************************** Voxel Walk Routine *************************************/
		/********************************************************************************************/
		if( z_move_direction != 0 )
		{
			//printf("z_exit[i] != z_entry[i]\n");
			while( !end_walk )
			{
				take_3D_step_GPU
				( 
					x_move_direction, y_move_direction, z_move_direction,
					dy_dx, dz_dx, dz_dy, dx_dy, dx_dz, dy_dz, 
					x_entry[i], y_entry[i], z_entry[i], 
					x, y, z, 
					voxel_x, voxel_y, voxel_z, voxel,
					x_to_go, y_to_go, z_to_go
				);
				end_walk = ( voxel == voxel_out ) || ( voxel_x >= COLUMNS ) || ( voxel_y >= ROWS ) || ( voxel_z >= SLICES );
				if( !end_walk )
					atomicAdd(&SM_counts[voxel], 1);
			}// end !end_walk 
		}
		else
		{
			//printf("z_exit[i] == z_entry[i]\n");
			while( !end_walk )
			{
				take_2D_step_GPU
				( 
					x_move_direction, y_move_direction, z_move_direction,
					dy_dx, dz_dx, dz_dy, dx_dy, dx_dz, dy_dz, 
					x_entry[i], y_entry[i], z_entry[i], 
					x, y, z, 
					voxel_x, voxel_y, voxel_z, voxel,
					x_to_go, y_to_go, z_to_go
				);				
				end_walk = ( voxel == voxel_out ) || ( voxel_x >= COLUMNS ) || ( voxel_y >= ROWS ) || ( voxel_z >= SLICES );
				if( !end_walk )
					atomicAdd(&SM_counts[voxel], 1);
			}// end: while( !end_walk )
		}//end: else: z_start != z_end => z_start == z_end
	}// end: if( (i < num_histories) && !missed_recon_volume[i] && (WEPL[i] <= MSC_UPPER_THRESHOLD) )
}
void SM_edge_detection()
{
	print_colored_text( "Performing edge-detection on SM_counts...", CYAN_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );	
	
	int* SM_differences_h = (int*) calloc( NUM_VOXELS, sizeof(int) );
	int* SM_differences_d;	
	cudaMalloc((void**) &SM_differences_d, SIZE_IMAGE_INT );
	cudaMemcpy( SM_differences_d, SM_differences_h, SIZE_IMAGE_INT, cudaMemcpyHostToDevice );

	dim3 dimBlock( SLICES );
	dim3 dimGrid( COLUMNS, ROWS );   
	carve_differences<<< dimGrid, dimBlock >>>( SM_differences_d, SM_counts_d );
	
	cudaMemcpy( SM_differences_h, SM_differences_d, SIZE_IMAGE_INT, cudaMemcpyDeviceToHost );

	int* SM_thresholds_h = (int*) calloc( SLICES, sizeof(int) );
	int voxel;	
	int max_difference = 0;
	for( int slice = 0; slice < SLICES; slice++ )
	{
		for( int pixel = 0; pixel < COLUMNS * ROWS; pixel++ )
		{
			voxel = pixel + slice * COLUMNS * ROWS;
			if( SM_differences_h[voxel] > max_difference )
			{
				max_difference = SM_differences_h[voxel];
				SM_thresholds_h[slice] = SM_counts_h[voxel];
			}
		}
		if( DEBUG_TEXT_ON )
		{
			//printf( "Slice %d : The maximum space_model difference = %d and the space_model threshold = %d\n", slice, max_difference, SM_thresholds_h[slice] );
		}
		max_difference = 0;
	}

	int* SM_thresholds_d;
	unsigned int threshold_size = SLICES * sizeof(int);
	cudaMalloc((void**) &SM_thresholds_d, threshold_size );
	cudaMemcpy( SM_thresholds_d, SM_thresholds_h, threshold_size, cudaMemcpyHostToDevice );

	SM_edge_detection_GPU<<< dimGrid, dimBlock >>>( SM_counts_d, SM_thresholds_d);
	
	//puts("SM hull-detection and edge-detection complete.");
	//sprintf(statement, "SM hull-detection and edge-detection complete.");		
	//print_colored_text( statement, RED_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );	
	
	//cudaMemcpy(SM_counts_h,  SM_counts_d,	 SIZE_IMAGE_INT,   cudaMemcpyDeviceToHost);
	//cudaFree( SM_counts_d );
	cudaFree( SM_differences_d );
	cudaFree( SM_thresholds_d );

	free(SM_differences_h);
	free(SM_thresholds_h);
	
	/*if( WRITE_SM_HULL )
		array_2_disk("x_SM", OUTPUT_DIRECTORY, OUTPUT_FOLDER_UNIQUE, SM_counts_h, COLUMNS, ROWS, SLICES, NUM_VOXELS, true );
	if( ENDPOINTS_HULL != SM_HULL)
		free(SM_counts_h);	*/
}
__global__ void SM_edge_detection_GPU( int* SM_counts, int* SM_threshold )
{
	int row = blockIdx.y, column = blockIdx.x, slice = threadIdx.x;
	float x = ( column - COLUMNS/2 + 0.5 ) * VOXEL_WIDTH;
	float y = ( ROWS/2 - row - 0.5 ) * VOXEL_HEIGHT;
	int voxel = column + row * COLUMNS + slice * COLUMNS * ROWS;
	if( voxel < NUM_VOXELS )
	{
		if( SM_counts[voxel] > SM_SCALE_THRESHOLD * SM_threshold[slice] )
			SM_counts[voxel] = 1;
		else
			SM_counts[voxel] = 0;
		if( powf(x, 2) + powf(y, 2) >= powf(RECON_CYL_RADIUS - max(VOXEL_WIDTH, VOXEL_HEIGHT)/2, 2 ) )
			SM_counts[voxel] = 0;
	}
}
void SM_edge_detection_2()
{
	print_colored_text( "Performing edge-detection on SM_counts...", CYAN_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );	
	
	// Copy the space modeled image from the GPU to the CPU and write it to file.
	cudaMemcpy(SM_counts_h,  SM_counts_d,	 SIZE_IMAGE_INT,   cudaMemcpyDeviceToHost);
	//array_2_disk(SM_COUNTS_FILENAME, OUTPUT_DIRECTORY, OUTPUT_FOLDER_UNIQUE, SM_counts_h, COLUMNS, ROWS, SLICES, NUM_VOXELS, false );

	int* SM_differences_h = (int*) calloc( NUM_VOXELS, sizeof(int) );
	int* SM_differences_d;
	cudaMalloc((void**) &SM_differences_d, SIZE_IMAGE_INT );
	cudaMemcpy( SM_differences_d, SM_differences_h, SIZE_IMAGE_INT, cudaMemcpyHostToDevice );

	dim3 dimBlock( SLICES );
	dim3 dimGrid( COLUMNS, ROWS );   

	carve_differences<<< dimGrid, dimBlock >>>( SM_differences_d, SM_counts_d );
	cudaMemcpy( SM_differences_h, SM_differences_d, SIZE_IMAGE_INT, cudaMemcpyDeviceToHost );

	int* SM_thresholds_h = (int*) calloc( SLICES, sizeof(int) );
	int voxel;	
	int max_difference = 0;
	for( int slice = 0; slice < SLICES; slice++ )
	{
		for( int pixel = 0; pixel < COLUMNS * ROWS; pixel++ )
		{
			voxel = pixel + slice * COLUMNS * ROWS;
			if( SM_differences_h[voxel] > max_difference )
			{
				max_difference = SM_differences_h[voxel];
				SM_thresholds_h[slice] = SM_counts_h[voxel];
			}
		}
		printf( "Slice %d : The maximum space_model difference = %d and the space_model threshold = %d\n", slice, max_difference, SM_thresholds_h[slice] );
		max_difference = 0;
	}

	int* SM_thresholds_d;
	unsigned int threshold_size = SLICES * sizeof(int);
	cudaMalloc((void**) &SM_thresholds_d, threshold_size );
	cudaMemcpy( SM_thresholds_d, SM_thresholds_h, threshold_size, cudaMemcpyHostToDevice );

	SM_edge_detection_GPU<<< dimGrid, dimBlock >>>( SM_counts_d, SM_thresholds_d);
	
	cudaMemcpy(SM_counts_h,  SM_counts_d,	 SIZE_IMAGE_INT,   cudaMemcpyDeviceToHost);
	cudaFree( SM_counts_d );
	cudaFree( SM_differences_d );
	cudaFree( SM_thresholds_d );

	free(SM_differences_h);
	free(SM_thresholds_h);
	
	//if( WRITE_SM_HULL )
	 	//array_2_disk(SM_HULL_FILENAME, OUTPUT_DIRECTORY, OUTPUT_FOLDER_UNIQUE, SM_counts_h, COLUMNS, ROWS, SLICES, NUM_VOXELS, true );
	if( ENDPOINTS_HULL != SM_HULL)
		free(SM_counts_h);	
}
__global__ void SM_edge_detection_GPU_2( int* SM_counts, int* SM_differences )
{
	int row = blockIdx.y, column = blockIdx.x, slice = threadIdx.x;
	int voxel = column + row * COLUMNS + slice * COLUMNS * ROWS;
	int difference, max_difference = 0;
	if( (row != 0) && (row != ROWS - 1) && (column != 0) && (column != COLUMNS - 1) )
	{
		for( int current_row = row - 1; current_row <= row + 1; current_row++ )
		{
			for( int current_column = column - 1; current_column <= column + 1; current_column++ )
			{
				difference = SM_counts[voxel] - SM_counts[current_column + current_row * COLUMNS + slice * COLUMNS * ROWS];
				if( difference > max_difference )
					max_difference = difference;
			}
		}
		SM_differences[voxel] = max_difference;
	}
	syncthreads();
	int slice_threshold;
	max_difference = 0;
	for( int pixel = 0; pixel < COLUMNS * ROWS; pixel++ )
	{
		voxel = pixel + slice * COLUMNS * ROWS;
		if( SM_differences[voxel] > max_difference )
		{
			max_difference = SM_differences[voxel];
			slice_threshold = SM_counts[voxel];
		}
	}
	syncthreads();
	float x = ( column - COLUMNS/2 + 0.5 ) * VOXEL_WIDTH;
	float y = ( ROWS/2 - row - 0.5 ) * VOXEL_HEIGHT;
	if( voxel < NUM_VOXELS )
	{
		if( SM_counts[voxel] > SM_SCALE_THRESHOLD * slice_threshold )
			SM_counts[voxel] = 1;
		else
			SM_counts[voxel] = 0;
		if( powf(x, 2) + powf(y, 2) >= powf(RECON_CYL_RADIUS - max(VOXEL_WIDTH, VOXEL_HEIGHT)/2, 2 ) )
			SM_counts[voxel] = 0;
	}
}
void hull_detection_finish()
{
	print_section_header( "Finishing hull detection and writing resulting images to disk...", MINOR_SECTION_SEPARATOR, LIGHT_CYAN_TEXT, YELLOW_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT );
	if( SC_ON )
	{
		SC_hull_h = (bool*) calloc( NUM_VOXELS, sizeof(bool) );
		cudaMemcpy(SC_hull_h,  SC_hull_d, SIZE_IMAGE_BOOL, cudaMemcpyDeviceToHost);
		if( WRITE_SC_HULL )
		{
			print_colored_text( "Writing SC hull to disk...", CYAN_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );	
			array_2_disk(SC_HULL_FILENAME, OUTPUT_DIRECTORY, OUTPUT_FOLDER_UNIQUE, SC_hull_h, COLUMNS, ROWS, SLICES, NUM_VOXELS, true );
		}
		if( ENDPOINTS_HULL != SC_HULL )
			free( SC_hull_h );
		cudaFree(SC_hull_d);
	}
	if( MSC_ON )
	{
		MSC_counts_h = (int*) calloc( NUM_VOXELS, sizeof(int) );
		if( WRITE_MSC_COUNTS )
		{		
			print_colored_text( "Writing MSC counts array to disk...", CYAN_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );	
			cudaMemcpy(MSC_counts_h,  MSC_counts_d, SIZE_IMAGE_INT, cudaMemcpyDeviceToHost);
			array_2_disk(MSC_COUNTS_FILENAME, OUTPUT_DIRECTORY, OUTPUT_FOLDER_UNIQUE, MSC_counts_h, COLUMNS, ROWS, SLICES, NUM_VOXELS, true );	
		}
		if( WRITE_MSC_HULL || (ENDPOINTS_HULL == MSC_HULL) )
		{
			MSC_edge_detection();
			cudaMemcpy(MSC_counts_h,  MSC_counts_d, SIZE_IMAGE_INT, cudaMemcpyDeviceToHost);
			if( WRITE_MSC_HULL )
			{
				print_colored_text( "Writing MSC hull to disk...", CYAN_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );	
				array_2_disk(MSC_HULL_FILENAME, OUTPUT_DIRECTORY, OUTPUT_FOLDER_UNIQUE, MSC_counts_h, COLUMNS, ROWS, SLICES, NUM_VOXELS, true );	
			}
			cudaFree(MSC_counts_d);
		}
		if( ENDPOINTS_HULL != MSC_HULL )
			free( MSC_counts_h );		
	}
	if( SM_ON )
	{
		SM_counts_h = (int*) calloc( NUM_VOXELS, sizeof(int) );
		if( WRITE_SM_COUNTS )
		{		
			print_colored_text( "Writing SM counts array to disk...", CYAN_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );	
			cudaMemcpy(SM_counts_h,  SM_counts_d, SIZE_IMAGE_INT, cudaMemcpyDeviceToHost);
			array_2_disk(SM_COUNTS_FILENAME, OUTPUT_DIRECTORY, OUTPUT_FOLDER_UNIQUE, SM_counts_h, COLUMNS, ROWS, SLICES, NUM_VOXELS, true );	
		}
		if( WRITE_SM_HULL || (ENDPOINTS_HULL == SM_HULL) )
		{
			SM_edge_detection();
			cudaMemcpy(SM_counts_h,  SM_counts_d, SIZE_IMAGE_INT, cudaMemcpyDeviceToHost);
			if( WRITE_SM_HULL )
			{
				print_colored_text( "Writing SM hull to disk...", CYAN_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );	
				array_2_disk(SM_HULL_FILENAME, OUTPUT_DIRECTORY, OUTPUT_FOLDER_UNIQUE, SM_counts_h, COLUMNS, ROWS, SLICES, NUM_VOXELS, true );	
			}
			cudaFree(SM_counts_d);
		}
		if( ENDPOINTS_HULL != SM_HULL )
			free( SM_counts_h );
	}
	print_section_exit( "Finished hull detection", SECTION_EXIT_CSTRING, RED_TEXT, RED_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );
	exit_program_if( EXIT_AFTER_HULLS, "through hull detection" );		
}
void hull_conversion_int_2_bool( int* int_hull )
{
	for( int voxel = 0; voxel < NUM_VOXELS; voxel++ )
	{
		if( int_hull[voxel] == 1 )
			hull_h[voxel] = true;
		else
			hull_h[voxel] = false;
	}
}
void hull_selection()
{
	print_colored_text( "Performing hull selection...", CYAN_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );				
	hull_h = (bool*) calloc( NUM_VOXELS, sizeof(bool) );
	switch( ENDPOINTS_HULL )
	{
		case SC_HULL  : hull_h = SC_hull_h;																									break;
		case MSC_HULL : hull_conversion_int_2_bool( MSC_counts_h );																			break;
						// std::transform( MSC_counts_h, MSC_counts_h + NUM_VOXELS, MSC_counts_h, hull_h, std::multiplies<int> () );		break;
		case SM_HULL  : hull_conversion_int_2_bool( SM_counts_h );																			break;
						// std::transform( SM_counts_h,  SM_counts_h + NUM_VOXELS,  SM_counts_h,  hull_h, std::multiplies<int> () );		break;
		case FBP_HULL : hull_h = FBP_hull_h;								
	}

	if( WRITE_X_HULL )
	{
		print_colored_text( "Writing selected hull to disk...", CYAN_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );	
		array_2_disk(HULL_FILENAME, OUTPUT_DIRECTORY, OUTPUT_FOLDER_UNIQUE, hull_h, COLUMNS, ROWS, SLICES, NUM_VOXELS, true );
	}
	// Allocate memory for and transfer hull to the GPU
	cudaMalloc((void**) &hull_d, SIZE_IMAGE_BOOL );
	cudaMemcpy( hull_d, hull_h, SIZE_IMAGE_BOOL, cudaMemcpyHostToDevice );

	if( AVG_FILTER_HULL && (HULL_AVG_FILTER_RADIUS > 0) )
	{
		print_colored_text( "Average filtering hull...", CYAN_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );	
		averaging_filter( hull_h, hull_d, HULL_AVG_FILTER_RADIUS, true, HULL_AVG_FILTER_THRESHOLD );
		if( WRITE_FILTERED_HULL )
		{
			print_colored_text( "Writing average filtered hull to disk...", CYAN_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );			
			//puts("Writing filtered hull to disk...");
			cudaMemcpy(hull_h, hull_d, SIZE_IMAGE_BOOL, cudaMemcpyDeviceToHost);
			array_2_disk( HULL_AVG_FILTER_FILENAME, OUTPUT_DIRECTORY, OUTPUT_FOLDER_UNIQUE, hull_h, COLUMNS, ROWS, SLICES, NUM_VOXELS, true );
		}
	}
	for(uint voxel = 0; voxel < NUM_VOXELS; voxel++)
	{	
		if(hull_h[voxel])
			hull_voxels_vector.push_back(voxel);
	}
	print_section_exit( "Finished hull selection", SECTION_EXIT_CSTRING, RED_TEXT, RED_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );	
}
void preprocessing()
{
	print_section_header( "Performing preprocessing", MAJOR_SECTION_SEPARATOR, LIGHT_GREEN_TEXT, BROWN_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT );
	timer( START, begin_preprocessing, "for preprocessing");
	import_and_process_data();
	hull_detection_finish();
	statistical_calculations_and_cuts();	
	generate_preprocessing_images();		
	execution_time_preprocessing = timer( STOP, begin_preprocessing, "for preprocessing");			
}
/***********************************************************************************************************************************************************************************************************************/
template<typename H, typename D> void averaging_filter( H*& image_h, D*& image_d, int radius, bool perform_threshold, double threshold_value )
{
	//bool is_hull = ( typeid(bool) == typeid(D) );
	D* new_value_d;
	int new_value_size = NUM_VOXELS * sizeof(D);
	cudaMalloc(&new_value_d, new_value_size );

	dim3 dimBlock( SLICES );
	dim3 dimGrid( COLUMNS, ROWS );   
	averaging_filter_GPU<<< dimGrid, dimBlock >>>( image_d, new_value_d, radius, perform_threshold, threshold_value );
	//apply_averaging_filter_GPU<<< dimGrid, dimBlock >>>( image_d, new_value_d );
	//cudaFree(new_value_d);
	cudaFree(image_d);
	image_d = new_value_d;
}
template<typename D> __global__ void averaging_filter_GPU( D* image, D* new_value, int radius, bool perform_threshold, double threshold_value )
{
	int voxel_x = blockIdx.x;
	int voxel_y = blockIdx.y;	
	int voxel_z = threadIdx.x;
	int voxel = voxel_x + voxel_y * COLUMNS + voxel_z * COLUMNS * ROWS;
	unsigned int left_edge = max( voxel_x - radius, 0 );
	unsigned int right_edge = min( voxel_x + radius, COLUMNS - 1);
	unsigned int top_edge = max( voxel_y - radius, 0 );
	unsigned int bottom_edge = min( voxel_y + radius, ROWS - 1);	
	int neighborhood_voxels = ( right_edge - left_edge + 1 ) * ( bottom_edge - top_edge + 1 );
	double sum_threshold = neighborhood_voxels * threshold_value;
	double sum = 0.0;
	// Determine neighborhood sum for voxels whose neighborhood is completely enclosed in image
	// Strip of size floor(AVG_FILTER_SIZE/2) around image perimeter must be ignored
	for( int column = left_edge; column <= right_edge; column++ )
		for( int row = top_edge; row <= bottom_edge; row++ )
			sum += image[column + (row * COLUMNS) + (voxel_z * COLUMNS * ROWS)];
	if( perform_threshold)
		new_value[voxel] = ( sum > sum_threshold );
	else
		new_value[voxel] = sum / neighborhood_voxels;
}
template<typename T> void median_filter_2D( T*& input_image, unsigned int radius )
{
	T* median_filtered_image = (T*) calloc( NUM_VOXELS, sizeof(T) );

	unsigned int neighborhood_voxels = (2*radius + 1 ) * (2*radius + 1 );
	unsigned int middle = neighborhood_voxels/2;
	unsigned int target_voxel, voxel;
	//T* neighborhood = (T*)calloc( neighborhood_voxels, sizeof(T));
	std::vector<T> neighborhood;
	for( unsigned int target_slice = 0; target_slice < SLICES; target_slice++ )
	{
		for( unsigned int target_column = radius; target_column < COLUMNS - radius; target_column++ )
		{
			for( unsigned int target_row = radius; target_row < ROWS - radius; target_row++ )
			{
				target_voxel = target_column + target_row * COLUMNS + target_slice * COLUMNS * ROWS;
				for( unsigned int column = target_column - radius; column <= target_column + radius; column++ )
				{
					for( unsigned int row = target_row - radius; row <=  target_row + radius; row++ )
					{
						voxel = column + row * COLUMNS + target_slice * COLUMNS * ROWS;
						neighborhood.push_back(input_image[voxel]);
						//neighborhood[i] = image_h[voxel2];
						//i++;
					}
				}
				std::sort( neighborhood.begin(), neighborhood.end());
				median_filtered_image[target_voxel] = neighborhood[middle];
				neighborhood.clear();
			}
		}
	}
	free(input_image);
	input_image = median_filtered_image;
	//input_image = (T*) calloc( NUM_VOXELS, sizeof(T) );
	//std::copy(median_filtered_image, median_filtered_image + NUM_VOXELS, input_image);
	

}
template<typename T> void median_filter_2D( T*& input_image, T*& median_filtered_image, unsigned int radius )
{
	//T* median_filtered_image = (T*) calloc( configurations.NUM_VOXELS, sizeof(T) );

	unsigned int neighborhood_voxels = (2*radius + 1 ) * (2*radius + 1 );
	unsigned int middle = neighborhood_voxels/2;
	unsigned int target_voxel, voxel;
	//T* neighborhood = (T*)calloc( neighborhood_voxels, sizeof(T));
	std::vector<T> neighborhood;
	for( unsigned int target_slice = 0; target_slice < SLICES; target_slice++ )
	{
		for( unsigned int target_column = radius; target_column < COLUMNS - radius; target_column++ )
		{
			for( unsigned int target_row = radius; target_row < ROWS - radius; target_row++ )
			{
				target_voxel = target_column + target_row * COLUMNS + target_slice * COLUMNS * ROWS;
				for( unsigned int column = target_column - radius; column <= target_column + radius; column++ )
				{
					for( unsigned int row = target_row - radius; row <=  target_row + radius; row++ )
					{
						voxel = column + row * COLUMNS + target_slice * COLUMNS * ROWS;
						neighborhood.push_back(input_image[voxel]);
						//neighborhood[i] = image_h[voxel2];
						//i++;
					}
				}
				std::sort( neighborhood.begin(), neighborhood.end());
				median_filtered_image[target_voxel] = neighborhood[middle];
				neighborhood.clear();
			}
		}
	}
	//T* temp = &input_image;
	//input_image = median_filtered_image;
	//free(temp);
	//free(input_image);
	//input_image = (T*) calloc( configurations.NUM_VOXELS, sizeof(T) );
	std::copy(median_filtered_image, median_filtered_image + NUM_VOXELS, input_image);
	free(median_filtered_image);
	//input_image = median_filtered_image;
}
template<typename D> __global__ void median_filter_GPU( D* image, D* new_value, int radius, bool perform_threshold, double threshold_value )
{
//	int voxel_x = blockIdx.x;
//	int voxel_y = blockIdx.y;	
//	int voxel_z = threadIdx.x;
//	int voxel = voxel_x + voxel_y * configurations->COLUMNS + voxel_z * configurations->COLUMNS * configurations->ROWS;
//	unsigned int left_edge = max( voxel_x - radius, 0 );
//	unsigned int right_edge = min( voxel_x + radius, configurations->COLUMNS - 1);
//	unsigned int top_edge = max( voxel_y - radius, 0 );
//	unsigned int bottom_edge = min( voxel_y + radius, configurations->ROWS - 1);	
//	int neighborhood_voxels = ( right_edge - left_edge + 1 ) * ( bottom_edge - top_edge + 1 );
//	double sum_threshold = neighborhood_voxels * threshold_value;
//	double sum = 0.0;
//	D new_element = image[voxel];
//	int middle = floor(neighborhood_voxels/2);
//
//	int count_up = 0;
//	int count_down = 0;
//	D current_value;
//	D* sorted = (D*)calloc( neighborhood_voxels, sizeof(D) );
//	//std::sort(
//	// Determine neighborhood sum for voxels whose neighborhood is completely enclosed in image
//	// Strip of size floor(AVG_FILTER_SIZE/2) around image perimeter must be ignored
//	for( int column = left_edge; column <= right_edge; column++ )
//	{
//		for( int row = top_edge; row <= bottom_edge; row++ )
//		{
//			current_value =  image[column + (row * configurations->COLUMNS) + (voxel_z * configurations->COLUMNS * configurations->ROWS)];
//			for( int column2 = left_edge; column2 <= right_edge; column2++ )
//			{
//				for( int row2 = top_edge; row2 <= bottom_edge; row2++ )
//				{
//					if(  image[column2 + (row2 * configurations->COLUMNS) + (voxel_z * configurations->COLUMNS * configurations->ROWS)] < current_value)
//						count++;
//				}
//			}
//			if( count == middle )
//				new_element = current_value;
//			count = 0;
//		}
//	}
//	new_value[voxel] = new_element;
}
template<typename T, typename T2> __global__ void apply_averaging_filter_GPU( T* image, T2* new_value )
{
	int voxel_x = blockIdx.x;
	int voxel_y = blockIdx.y;	
	int voxel_z = threadIdx.x;
	int voxel = voxel_x + voxel_y * COLUMNS + voxel_z * COLUMNS * ROWS;
	image[voxel] = new_value[voxel];
}
/**************************************************************************************************************************************************************************/
/**************************************************************************** MLP Endpoints Routines **********************************************************************/
/**************************************************************************************************************************************************************************/
template<typename O> __device__ bool find_MLP_endpoints_GPU
(
	O* image, double x_start, double y_start, double z_start, double xy_angle, double xz_angle, 
	double& x_object, double& y_object, double& z_object, int& voxel_x, int& voxel_y, int& voxel_z, bool entering
)
{
	/********************************************************************************************/
	/********************************* Voxel Walk Parameters ************************************/
	/********************************************************************************************/
	int x_move_direction, y_move_direction, z_move_direction;
	double delta_yx, delta_zx, delta_zy;
	/********************************************************************************************/
	/**************************** Status Tracking Information ***********************************/
	/********************************************************************************************/
	double x = x_start, y = y_start, z = z_start;
	double x_to_go, y_to_go, z_to_go;		
	double x_extension, y_extension;	
	int voxel; 
	bool hit_hull = false, end_walk, outside_image;
	/********************************************************************************************/
	/******************** Initial Conditions and Movement Characteristics ***********************/
	/********************************************************************************************/	
	if( !entering )
	{
		xy_angle += PI;
	}
	x_move_direction = ( cos(xy_angle) >= 0 ) - ( cos(xy_angle) <= 0 );
	y_move_direction = ( sin(xy_angle) >= 0 ) - ( sin(xy_angle) <= 0 );
	z_move_direction = ( sin(xz_angle) >= 0 ) - ( sin(xz_angle) <= 0 );
	if( x_move_direction < 0 )
		z_move_direction *= -1;

	voxel_x = calculate_voxel_GPU( X_ZERO_COORDINATE, x, VOXEL_WIDTH );
	voxel_y = calculate_voxel_GPU( Y_ZERO_COORDINATE, y, VOXEL_HEIGHT );
	voxel_z = calculate_voxel_GPU( Z_ZERO_COORDINATE, z, VOXEL_THICKNESS );

	x_to_go = distance_remaining_GPU( X_ZERO_COORDINATE, x, X_INCREASING_DIRECTION, x_move_direction, VOXEL_WIDTH, voxel_x );
	y_to_go = distance_remaining_GPU( Y_ZERO_COORDINATE, y, Y_INCREASING_DIRECTION, y_move_direction, VOXEL_HEIGHT, voxel_y );	
	z_to_go = distance_remaining_GPU( Z_ZERO_COORDINATE, z, Z_INCREASING_DIRECTION, z_move_direction, VOXEL_THICKNESS, voxel_z );

	voxel = voxel_x + voxel_y * COLUMNS + voxel_z * COLUMNS * ROWS;
	/********************************************************************************************/
	/***************************** Path and Walk Information ************************************/
	/********************************************************************************************/
	// Lengths/Distances as x is Incremented One Voxel tan( xy_hit_hull_angle )
	delta_yx = fabs(tan(xy_angle));
	delta_zx = fabs(tan(xz_angle));
	delta_zy = fabs( tan(xz_angle)/tan(xy_angle));

	double dy_dx = tan(xy_angle);
	double dz_dx = tan(xz_angle);
	double dz_dy = tan(xz_angle)/tan(xy_angle);

	double dx_dy = powf( tan(xy_angle), -1.0 );
	double dx_dz = powf( tan(xz_angle), -1.0 );
	double dy_dz = tanf(xy_angle)/tan(xz_angle);
	/********************************************************************************************/
	/************************* Initialize and Check Exit Conditions *****************************/
	/********************************************************************************************/
	outside_image = (voxel_x >= COLUMNS ) || (voxel_y >= ROWS ) || (voxel_z >= SLICES ) || (voxel_x < 0  ) || (voxel_y < 0 ) || (voxel_z < 0 );		
	if( !outside_image )
		hit_hull = (image[voxel] == 1);		
	end_walk = outside_image || hit_hull;
	/********************************************************************************************/
	/*********************************** Voxel Walk Routine *************************************/
	/********************************************************************************************/
	if( z_move_direction != 0 )
	{
		while( !end_walk )
		{
			// Change in z for Move to Voxel Edge in x and y
			x_extension = delta_zx * x_to_go;
			y_extension = delta_zy * y_to_go;
			if( (z_to_go <= x_extension  ) && (z_to_go <= y_extension) )
			{
				voxel_z -= z_move_direction;					
				z = edge_coordinate_GPU( Z_ZERO_COORDINATE, voxel_z, VOXEL_THICKNESS, Z_INCREASING_DIRECTION, z_move_direction );					
				x = corresponding_coordinate_GPU( dx_dz, z, z_start, x_start );
				y = corresponding_coordinate_GPU( dy_dz, z, z_start, y_start );
				x_to_go = distance_remaining_GPU( X_ZERO_COORDINATE, x, X_INCREASING_DIRECTION, x_move_direction, VOXEL_WIDTH, voxel_x );
				y_to_go = distance_remaining_GPU( Y_ZERO_COORDINATE, y, Y_INCREASING_DIRECTION, y_move_direction, VOXEL_HEIGHT, voxel_y );	
				z_to_go = VOXEL_THICKNESS;
			}
			//If Next Voxel Edge is in x or xy Diagonal
			else if( x_extension <= y_extension )
			{
				voxel_x += x_move_direction;
				x = edge_coordinate_GPU( X_ZERO_COORDINATE, voxel_x, VOXEL_WIDTH, X_INCREASING_DIRECTION, x_move_direction );
				y = corresponding_coordinate_GPU( dy_dx, x, x_start, y_start );
				z = corresponding_coordinate_GPU( dz_dx, x, x_start, z_start );
				x_to_go = VOXEL_WIDTH;
				y_to_go = distance_remaining_GPU( Y_ZERO_COORDINATE, y, Y_INCREASING_DIRECTION, y_move_direction, VOXEL_HEIGHT, voxel_y );
				z_to_go = distance_remaining_GPU( Z_ZERO_COORDINATE, z, Z_INCREASING_DIRECTION, z_move_direction, VOXEL_THICKNESS, voxel_z );
			}
			// Else Next Voxel Edge is in y
			else
			{
				voxel_y -= y_move_direction;					
				y = edge_coordinate_GPU( Y_ZERO_COORDINATE, voxel_y, VOXEL_HEIGHT, Y_INCREASING_DIRECTION, y_move_direction );
				x = corresponding_coordinate_GPU( dx_dy, y, y_start, x_start );
				z = corresponding_coordinate_GPU( dz_dy, y, y_start, z_start );
				x_to_go = distance_remaining_GPU( X_ZERO_COORDINATE, x, X_INCREASING_DIRECTION, x_move_direction, VOXEL_WIDTH, voxel_x );
				y_to_go = VOXEL_HEIGHT;					
				z_to_go = distance_remaining_GPU( Z_ZERO_COORDINATE, z, Z_INCREASING_DIRECTION, z_move_direction, VOXEL_THICKNESS, voxel_z );
			}
			// <= VOXEL_ALLOWANCE
			if( x_to_go == 0 )
			{
				x_to_go = VOXEL_WIDTH;
				voxel_x += x_move_direction;
			}
			if( y_to_go == 0 )
			{
				y_to_go = VOXEL_HEIGHT;
				voxel_y -= y_move_direction;
			}
			if( z_to_go == 0 )
			{
				z_to_go = VOXEL_THICKNESS;
				voxel_z -= z_move_direction;
			}
				
			voxel_z = max(voxel_z, 0 );
			voxel = voxel_x + voxel_y * COLUMNS + voxel_z * COLUMNS * ROWS;

			outside_image = (voxel_x >= COLUMNS ) || (voxel_y >= ROWS ) || (voxel_z >= SLICES ) || (voxel_x < 0  ) || (voxel_y < 0 ) || (voxel_z < 0 );		
			if( !outside_image )
				hit_hull = (image[voxel] == 1);	
			end_walk = outside_image || hit_hull;	
		}// end !end_walk 
	}
	else
	{
		//printf("z_end == z_start\n");
		while( !end_walk )
		{
			// Change in x for Move to Voxel Edge in y
			y_extension = y_to_go / delta_yx;
			//If Next Voxel Edge is in x or xy Diagonal
			if( x_to_go <= y_extension )
			{
				voxel_x += x_move_direction;					
				x = edge_coordinate_GPU( X_ZERO_COORDINATE, voxel_x, VOXEL_WIDTH, X_INCREASING_DIRECTION, x_move_direction );
				y = corresponding_coordinate_GPU( dy_dx, x, x_start, y_start );
				x_to_go = VOXEL_WIDTH;
				y_to_go = distance_remaining_GPU( Y_ZERO_COORDINATE, y, Y_INCREASING_DIRECTION, y_move_direction, VOXEL_HEIGHT, voxel_y );
			}
			// Else Next Voxel Edge is in y
			else
			{
				voxel_y -= y_move_direction;
				y = edge_coordinate_GPU( Y_ZERO_COORDINATE, voxel_y, VOXEL_HEIGHT, Z_INCREASING_DIRECTION, y_move_direction );
				x = corresponding_coordinate_GPU( dx_dy, y, y_start, x_start );
				x_to_go = distance_remaining_GPU( X_ZERO_COORDINATE, x, X_INCREASING_DIRECTION, x_move_direction, VOXEL_WIDTH, voxel_x );
				y_to_go = VOXEL_HEIGHT;
			}
			// <= VOXEL_ALLOWANCE
			if( x_to_go == 0 )
			{
				x_to_go = VOXEL_WIDTH;
				voxel_x += x_move_direction;
			}
			if( y_to_go == 0 )
			{
				y_to_go = VOXEL_HEIGHT;
				voxel_y -= y_move_direction;
			}
			voxel = voxel_x + voxel_y * COLUMNS + voxel_z * COLUMNS * ROWS;		
			outside_image = (voxel_x >= COLUMNS ) || (voxel_y >= ROWS ) || (voxel_z >= SLICES ) || (voxel_x < 0  ) || (voxel_y < 0 ) || (voxel_z < 0 );		
			if( !outside_image )
				hit_hull = (image[voxel] == 1);		
			end_walk = outside_image || hit_hull;	
		}// end: while( !end_walk )
	}//end: else: z_start != z_end => z_start == z_end
	if( hit_hull )
	{
		x_object = x;
		y_object = y;
		z_object = z;
	}
	return hit_hull;
}
__global__ void collect_MLP_endpoints_GPU
(
	bool* intersected_hull, unsigned int* first_MLP_voxel, bool* hull, float* x_entry, float* y_entry, float* z_entry, float* xy_entry_angle, float* xz_entry_angle, 
	float* x_exit, float* y_exit, float* z_exit, float* xy_exit_angle, float* xz_exit_angle, int start_proton_id, int num_histories
) 
{
  
	bool entered_object = false, exited_object = false;
	int voxel_x = 0, voxel_y = 0, voxel_z = 0, voxel_x_int = 0, voxel_y_int = 0, voxel_z_int = 0;
	double x_in_object = 0.0, y_in_object = 0.0, z_in_object = 0.0, x_out_object = 0.0, y_out_object = 0.0, z_out_object = 0.0;

	int proton_id = start_proton_id + threadIdx.x * ENDPOINTS_PER_THREAD + blockIdx.x * ENDPOINTS_PER_BLOCK* ENDPOINTS_PER_THREAD;
	if( proton_id < num_histories ) 
	{
		for( int history = 0; history < ENDPOINTS_PER_THREAD; history++)
		{
			if( proton_id < num_histories ) 
			{
	
				exited_object = find_MLP_endpoints_GPU( hull, x_exit[proton_id], y_exit[proton_id], z_exit[proton_id], xy_exit_angle[proton_id], xz_exit_angle[proton_id], x_out_object, y_out_object, z_out_object, voxel_x_int, voxel_y_int, voxel_z_int, false);
				entered_object = find_MLP_endpoints_GPU( hull, x_entry[proton_id], y_entry[proton_id], z_entry[proton_id], xy_entry_angle[proton_id], xz_entry_angle[proton_id], x_in_object, y_in_object, z_in_object, voxel_x, voxel_y, voxel_z, true);	
				
				//__syncthreads();
				if( entered_object && exited_object ) 
				{
		  
					intersected_hull[proton_id] = true;
					first_MLP_voxel[proton_id] = voxel_x + COLUMNS * voxel_y + ROWS * COLUMNS * voxel_z;
					x_entry[proton_id] = x_in_object;
					y_entry[proton_id] = y_in_object;
					z_entry[proton_id] = z_in_object;
					x_exit[proton_id] = x_out_object;
					y_exit[proton_id] = y_out_object;
					z_exit[proton_id] = z_out_object;
				}	  
				else
				{
					intersected_hull[proton_id] = false;
					first_MLP_voxel[proton_id] = NUM_VOXELS;			
				}
			}
			proton_id++;
		}
	}
}
__global__ void collect_MLP_endpoints_GPU_nobool
(
	unsigned int* first_MLP_voxel, bool* hull, float* x_entry, float* y_entry, float* z_entry, float* xy_entry_angle, float* xz_entry_angle, 
	float* x_exit, float* y_exit, float* z_exit, float* xy_exit_angle, float* xz_exit_angle, int start_proton_id, int num_histories
) 
{
	bool entered_object = false, exited_object = false;
	int voxel_x = 0, voxel_y = 0, voxel_z = 0, voxel_x_int = 0, voxel_y_int = 0, voxel_z_int = 0;
	double x_in_object = 0.0, y_in_object = 0.0, z_in_object = 0.0, x_out_object = 0.0, y_out_object = 0.0, z_out_object = 0.0;
	//float x_in_object = 0.0, y_in_object = 0.0, z_in_object = 0.0, x_out_object = 0.0, y_out_object = 0.0, z_out_object = 0.0;
	//int proton_id = start_proton_id + threadIdx.x * ENDPOINTS_PER_THREAD + blockIdx.x * ENDPOINTS_PER_BLOCK;
	int proton_id = start_proton_id + threadIdx.x * ENDPOINTS_PER_THREAD + blockIdx.x * ENDPOINTS_PER_BLOCK* ENDPOINTS_PER_THREAD;
	
	for( int history = 0; history < ENDPOINTS_PER_THREAD; history++)
	{
		if( proton_id < num_histories ) 
		{
			entered_object = find_MLP_endpoints_GPU( hull, x_entry[proton_id], y_entry[proton_id], z_entry[proton_id], xy_entry_angle[proton_id], xz_entry_angle[proton_id], x_in_object, y_in_object, z_in_object, voxel_x, voxel_y, voxel_z, true);	
			exited_object = find_MLP_endpoints_GPU( hull, x_exit[proton_id], y_exit[proton_id], z_exit[proton_id], xy_exit_angle[proton_id], xz_exit_angle[proton_id], x_out_object, y_out_object, z_out_object, voxel_x_int, voxel_y_int, voxel_z_int, false);
			
			if( entered_object && exited_object ) 
			{
				first_MLP_voxel[proton_id] = voxel_x + COLUMNS * voxel_y + ROWS * COLUMNS * voxel_z;
				x_entry[proton_id] = x_in_object;
				y_entry[proton_id] = y_in_object;
				z_entry[proton_id] = z_in_object;
				x_exit[proton_id] = x_out_object;
				y_exit[proton_id] = y_out_object;
				z_exit[proton_id] = z_out_object;
			}
			else
				first_MLP_voxel[proton_id] = NUM_VOXELS;			  
		}
		proton_id++;
	}
}
void reconstruction_cuts_hull_transfer()
{
	cudaFree(hull_d);
	cudaMalloc( (void**) &hull_d, NUM_VOXELS *sizeof(bool));	
	cudaMemcpy( hull_d, hull_h, NUM_VOXELS *sizeof(bool),cudaMemcpyHostToDevice );
}
void reconstruction_cuts_allocations( const int num_histories)
{
	unsigned int size_floats		= sizeof(float) * num_histories;
	unsigned int size_ints			= sizeof(int) * num_histories;
	
	#if (ENDPOINTS_ALG == YES_BOOL)	
		unsigned int size_bool			= sizeof(bool) * num_histories;
		intersected_hull_h = (bool*)calloc( num_histories, sizeof(bool) );
		cudaMalloc( (void**) &intersected_hull_d, 		size_bool );
	#endif	
	
	cudaMalloc( (void**) &x_entry_d,			size_floats );
	cudaMalloc( (void**) &y_entry_d,			size_floats );
	cudaMalloc( (void**) &z_entry_d,			size_floats );
	cudaMalloc( (void**) &x_exit_d,				size_floats );
	cudaMalloc( (void**) &y_exit_d,				size_floats );
	cudaMalloc( (void**) &z_exit_d,				size_floats );
	cudaMalloc( (void**) &xy_entry_angle_d,			size_floats );
	cudaMalloc( (void**) &xz_entry_angle_d,			size_floats );
	cudaMalloc( (void**) &xy_exit_angle_d,			size_floats );
	cudaMalloc( (void**) &xz_exit_angle_d,			size_floats );
	cudaMalloc( (void**) &first_MLP_voxel_d, 		size_ints );

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) 
		printf("reconstruction_cuts_allocations Error: %s\n", cudaGetErrorString(cudaStatus));

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Kernel!\n", cudaStatus);
}
void reconstruction_cuts_allocations_nobool( const int num_histories)
{
	unsigned int size_floats		= sizeof(float) * num_histories;
	unsigned int size_ints			= sizeof(int) * num_histories;

	cudaMalloc( (void**) &x_entry_d,			size_floats );
	cudaMalloc( (void**) &y_entry_d,			size_floats );
	cudaMalloc( (void**) &z_entry_d,			size_floats );
	cudaMalloc( (void**) &x_exit_d,				size_floats );
	cudaMalloc( (void**) &y_exit_d,				size_floats );
	cudaMalloc( (void**) &z_exit_d,				size_floats );
	cudaMalloc( (void**) &xy_entry_angle_d,			size_floats );
	cudaMalloc( (void**) &xz_entry_angle_d,			size_floats );
	cudaMalloc( (void**) &xy_exit_angle_d,			size_floats );
	cudaMalloc( (void**) &xz_exit_angle_d,			size_floats );
	cudaMalloc( (void**) &first_MLP_voxel_d, 		size_ints );
}
void reconstruction_cuts_host_2_device(const int start_position, const int num_histories) 
{
	unsigned int size_floats		= sizeof(float) * num_histories;
	
	#if (ENDPOINTS_ALG == YES_BOOL)	
		unsigned int size_bool			= sizeof(bool) * num_histories;
		cudaMemcpy( intersected_hull_d, 	intersected_hull_h, 					size_bool,		cudaMemcpyHostToDevice );  
	#endif	
	cudaMemcpy( x_entry_d,				&x_entry_vector[start_position],			size_floats,	cudaMemcpyHostToDevice );
	cudaMemcpy( y_entry_d,				&y_entry_vector[start_position],			size_floats,	cudaMemcpyHostToDevice );
	cudaMemcpy( z_entry_d,				&z_entry_vector[start_position],			size_floats,	cudaMemcpyHostToDevice );
	cudaMemcpy( x_exit_d,				&x_exit_vector[start_position],				size_floats,	cudaMemcpyHostToDevice );
	cudaMemcpy( y_exit_d,				&y_exit_vector[start_position],				size_floats,	cudaMemcpyHostToDevice );
	cudaMemcpy( z_exit_d,				&z_exit_vector[start_position],				size_floats,	cudaMemcpyHostToDevice );
	cudaMemcpy( xy_entry_angle_d,		&xy_entry_angle_vector[start_position],		size_floats,	cudaMemcpyHostToDevice );
	cudaMemcpy( xz_entry_angle_d,		&xz_entry_angle_vector[start_position],		size_floats,	cudaMemcpyHostToDevice );
	cudaMemcpy( xy_exit_angle_d,		&xy_exit_angle_vector[start_position],		size_floats,	cudaMemcpyHostToDevice );
	cudaMemcpy( xz_exit_angle_d,		&xz_exit_angle_vector[start_position],		size_floats,	cudaMemcpyHostToDevice );	

	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) 
			printf("econstruction_cuts_host_2_device Error: %s\n", cudaGetErrorString(cudaStatus));

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Kernel!\n", cudaStatus);	
}
void reconstruction_cuts_host_2_device_nobool(const int start_position, const int num_histories) 
{
	unsigned int size_floats		= sizeof(float) * num_histories;

	cudaMemcpy( x_entry_d,				&x_entry_vector[start_position],		size_floats,	cudaMemcpyHostToDevice );
	cudaMemcpy( y_entry_d,				&y_entry_vector[start_position],		size_floats,	cudaMemcpyHostToDevice );
	cudaMemcpy( z_entry_d,				&z_entry_vector[start_position],		size_floats,	cudaMemcpyHostToDevice );
	cudaMemcpy( x_exit_d,				&x_exit_vector[start_position],			size_floats,	cudaMemcpyHostToDevice );
	cudaMemcpy( y_exit_d,				&y_exit_vector[start_position],			size_floats,	cudaMemcpyHostToDevice );
	cudaMemcpy( z_exit_d,				&z_exit_vector[start_position],			size_floats,	cudaMemcpyHostToDevice );
	cudaMemcpy( xy_entry_angle_d,			&xy_entry_angle_vector[start_position],		size_floats,	cudaMemcpyHostToDevice );
	cudaMemcpy( xz_entry_angle_d,			&xz_entry_angle_vector[start_position],		size_floats,	cudaMemcpyHostToDevice );
	cudaMemcpy( xy_exit_angle_d,			&xy_exit_angle_vector[start_position],		size_floats,	cudaMemcpyHostToDevice );
	cudaMemcpy( xz_exit_angle_d,			&xz_exit_angle_vector[start_position],		size_floats,	cudaMemcpyHostToDevice );	

	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) 
			printf("Error: %s\n", cudaGetErrorString(cudaStatus));

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Kernel!\n", cudaStatus);
}
void reconstruction_cuts_device_2_host(const int start_position, const int num_histories) 
{
	unsigned int size_floats		= sizeof(float) * num_histories;
	unsigned int size_ints			= sizeof(int) * num_histories;
	
	cudaMemcpy(&first_MLP_voxel_vector[start_position], first_MLP_voxel_d, size_ints, cudaMemcpyDeviceToHost);
	cudaMemcpy(&x_entry_vector[start_position], x_entry_d, size_floats, cudaMemcpyDeviceToHost);
	cudaMemcpy(&y_entry_vector[start_position], y_entry_d, size_floats, cudaMemcpyDeviceToHost);
	cudaMemcpy(&z_entry_vector[start_position], z_entry_d, size_floats, cudaMemcpyDeviceToHost);
	cudaMemcpy(&x_exit_vector[start_position], x_exit_d, size_floats, cudaMemcpyDeviceToHost);
	cudaMemcpy(&y_exit_vector[start_position], y_exit_d, size_floats, cudaMemcpyDeviceToHost);
	cudaMemcpy(&z_exit_vector[start_position], z_exit_d, size_floats, cudaMemcpyDeviceToHost);	
	
	#if (ENDPOINTS_ALG == YES_BOOL)	
		unsigned int size_bool			= sizeof(bool) * num_histories;
		cudaMemcpy(intersected_hull_h, intersected_hull_d, size_bool, cudaMemcpyDeviceToHost);
	#endif	
	
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) 
		printf("reconstruction_cuts_device_2_host Error: %s\n", cudaGetErrorString(cudaStatus));

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Kernel!\n", cudaStatus);
}
void reconstruction_cuts_device_2_host_nobool(const int start_position, const int num_histories) 
{
	unsigned int size_floats		= sizeof(float) * num_histories;
	unsigned int size_ints			= sizeof(int) * num_histories;
  
	cudaMemcpy(&first_MLP_voxel_vector[start_position], first_MLP_voxel_d, size_ints, cudaMemcpyDeviceToHost);
	cudaMemcpy(&x_entry_vector[start_position], x_entry_d, size_floats, cudaMemcpyDeviceToHost);
	cudaMemcpy(&y_entry_vector[start_position], y_entry_d, size_floats, cudaMemcpyDeviceToHost);
	cudaMemcpy(&z_entry_vector[start_position], z_entry_d, size_floats, cudaMemcpyDeviceToHost);
	cudaMemcpy(&x_exit_vector[start_position], x_exit_d, size_floats, cudaMemcpyDeviceToHost);
	cudaMemcpy(&y_exit_vector[start_position], y_exit_d, size_floats, cudaMemcpyDeviceToHost);
	cudaMemcpy(&z_exit_vector[start_position], z_exit_d, size_floats, cudaMemcpyDeviceToHost);	
	
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) 
		printf("reconstruction_cuts_device_2_host Error: %s\n", cudaGetErrorString(cudaStatus));

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Kernel!\n", cudaStatus);
}
void reconstruction_cuts_deallocations()
{
	cudaFree( x_entry_d);
	cudaFree( y_entry_d);
	cudaFree( z_entry_d);
	cudaFree( x_exit_d);
	cudaFree( y_exit_d);
	cudaFree( z_exit_d);
	cudaFree( xy_entry_angle_d);
	cudaFree( xz_entry_angle_d);
	cudaFree( xy_exit_angle_d);
	cudaFree( xz_exit_angle_d );
	cudaFree( first_MLP_voxel_d);

	#if (ENDPOINTS_ALG == YES_BOOL)	
		free(intersected_hull_h);
		cudaFree( intersected_hull_d );
	#endif	
	
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) 
		printf("reconstruction_cuts_deallocations Error: %s\n", cudaGetErrorString(cudaStatus));

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Kernel!\n", cudaStatus);
}
void reconstruction_cuts_deallocations_nobool()
{
	cudaFree( x_entry_d);
	cudaFree( y_entry_d);
	cudaFree( z_entry_d);
	cudaFree( x_exit_d);
	cudaFree( y_exit_d);
	cudaFree( z_exit_d);
	cudaFree( xy_entry_angle_d);
	cudaFree( xz_entry_angle_d);
	cudaFree( xy_exit_angle_d);
	cudaFree( xz_exit_angle_d );
	cudaFree( first_MLP_voxel_d);
}
bool is_valid_reconstruction_history(const int start_position, const int index ) 
{
	#if (ENDPOINTS_ALG == YES_BOOL)	
		return intersected_hull_h[index];
	#elif (ENDPOINTS_ALG == NO_BOOL)
		reurn first_MLP_voxel_vector[start_position + index] != NUM_VOXELS 
	#endif 		
}
void reconstruction_cuts_full_tx( const int num_histories )
{
	// ENDPOINTS_TX_MODE = FULL_TX, ENDPOINTS_ALG = YES_BOOL
	cudaError_t cudaStatus;
	reconstruction_histories = 0;
	int remaining_histories = num_histories, histories_2_process, start_position = 0;
	//int num_blocks = static_cast<int>( (MAX_ENDPOINTS_HISTORIES - 1 + ENDPOINTS_PER_BLOCK*ENDPOINTS_PER_THREAD) / (ENDPOINTS_PER_BLOCK*ENDPOINTS_PER_THREAD));
	int num_blocks;
	while( remaining_histories > 0 )
	{
		if( remaining_histories > MAX_ENDPOINTS_HISTORIES )
			histories_2_process = MAX_ENDPOINTS_HISTORIES;
		else
			histories_2_process = remaining_histories;	

		num_blocks = static_cast<int>( (histories_2_process - 1 + ENDPOINTS_PER_BLOCK*ENDPOINTS_PER_THREAD) / (ENDPOINTS_PER_BLOCK*ENDPOINTS_PER_THREAD) );  
		#if (ENDPOINTS_ALG == YES_BOOL)	
			collect_MLP_endpoints_GPU<<< num_blocks, ENDPOINTS_PER_BLOCK >>>
			( 
				intersected_hull_d, first_MLP_voxel_d, hull_d, x_entry_d, y_entry_d, z_entry_d, xy_entry_angle_d, xz_entry_angle_d, 
				x_exit_d, y_exit_d, z_exit_d, xy_exit_angle_d, xz_exit_angle_d, start_position, num_histories
			);		
		#elif (ENDPOINTS_ALG == NO_BOOL)	
			collect_MLP_endpoints_GPU_nobool<<< num_blocks, ENDPOINTS_PER_BLOCK >>>
			( 
				first_MLP_voxel_d, hull_d, x_entry_d, y_entry_d, z_entry_d, xy_entry_angle_d, xz_entry_angle_d, 
				x_exit_d, y_exit_d, z_exit_d, xy_exit_angle_d, xz_exit_angle_d, start_position, num_histories
			);	
		#endif	
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) 
			printf("reconstruction_cuts_full_tx Error: %s\n", cudaGetErrorString(cudaStatus));

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess)
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Kernel!\n", cudaStatus);

		remaining_histories -= MAX_ENDPOINTS_HISTORIES;
		start_position		+= MAX_ENDPOINTS_HISTORIES;
	}
	//#if (ENDPOINTS_ALG == YES_BOOL)	
		reconstruction_cuts_device_2_host(0, num_histories);	    
	for( int i = 0; i < num_histories; i++ ) 
	{    
		if( is_valid_reconstruction_history(start_position, i)  ) 
		{
			first_MLP_voxel_vector[reconstruction_histories] = first_MLP_voxel_vector[i];
			data_shift_vectors( i + start_position, reconstruction_histories );
			reconstruction_histories++;
		}
	}
}
void reconstruction_cuts_full_tx_nobool( const int num_histories )
{
	// ENDPOINTS_TX_MODE = FULL_TX, ENDPOINTS_ALG = NO_BOOL 
	cudaError_t cudaStatus;
	reconstruction_histories = 0;
	int remaining_histories = num_histories, histories_2_process, start_position = 0;
	//int num_blocks = static_cast<int>( (MAX_ENDPOINTS_HISTORIES - 1 + ENDPOINTS_PER_BLOCK*ENDPOINTS_PER_THREAD) / (ENDPOINTS_PER_BLOCK*ENDPOINTS_PER_THREAD));
	int num_blocks;
	while( remaining_histories > 0 )
	{
		if( remaining_histories > MAX_ENDPOINTS_HISTORIES )
			histories_2_process = MAX_ENDPOINTS_HISTORIES;
		else
			histories_2_process = remaining_histories;	

		num_blocks = static_cast<int>( (histories_2_process - 1 + ENDPOINTS_PER_BLOCK*ENDPOINTS_PER_THREAD) / (ENDPOINTS_PER_BLOCK*ENDPOINTS_PER_THREAD) );  
		collect_MLP_endpoints_GPU_nobool<<< num_blocks, ENDPOINTS_PER_BLOCK >>>
		( 
			first_MLP_voxel_d, hull_d, x_entry_d, y_entry_d, z_entry_d, xy_entry_angle_d, xz_entry_angle_d, 
			x_exit_d, y_exit_d, z_exit_d, xy_exit_angle_d, xz_exit_angle_d, start_position, num_histories
		);		
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) 
			printf("Error: %s\n", cudaGetErrorString(cudaStatus));

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess)
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Kernel!\n", cudaStatus);

		remaining_histories -= MAX_ENDPOINTS_HISTORIES;
		start_position		+= MAX_ENDPOINTS_HISTORIES;
		//cout << "start_position = " << start_position << endl;
		//cout << "remaining_histories = " << remaining_histories << endl;	
	}
	reconstruction_cuts_device_2_host_nobool(0, num_histories);	    
	for( int i = 0; i < num_histories; i++ ) 
	{    
		if( first_MLP_voxel_vector[i] != NUM_VOXELS ) 
		{
			first_MLP_voxel_vector[reconstruction_histories] = first_MLP_voxel_vector[i];
			data_shift_vectors( i, reconstruction_histories );
			reconstruction_histories++;
		}
	}
}
void reconstruction_cuts_partial_tx(const int start_position, const int num_histories) 
{
	// ENDPOINTS_TX_MODE = PARTIAL_TX, ENDPOINTS_ALG = YES_BOOL
	reconstruction_cuts_allocations(num_histories);
	reconstruction_cuts_host_2_device( start_position, num_histories);

	dim3 dimBlock(ENDPOINTS_PER_BLOCK);
	int num_blocks = static_cast<int>( (num_histories - 1 + ENDPOINTS_PER_BLOCK*ENDPOINTS_PER_THREAD ) / (ENDPOINTS_PER_BLOCK*ENDPOINTS_PER_THREAD)  );
	#if (ENDPOINTS_ALG == YES_BOOL)	
		collect_MLP_endpoints_GPU<<< num_blocks, dimBlock >>>
		( 
			intersected_hull_d, first_MLP_voxel_d, hull_d, x_entry_d, y_entry_d, z_entry_d, xy_entry_angle_d, xz_entry_angle_d,  
			x_exit_d, y_exit_d, z_exit_d, xy_exit_angle_d, xz_exit_angle_d, 0, num_histories 
		);
	#elif (ENDPOINTS_ALG == NO_BOOL)	
		collect_MLP_endpoints_GPU_nobool<<< num_blocks, dimBlock >>>
		( 
			first_MLP_voxel_d, hull_d, x_entry_d, y_entry_d, z_entry_d, xy_entry_angle_d, xz_entry_angle_d, x_exit_d, y_exit_d, z_exit_d, xy_exit_angle_d, xz_exit_angle_d, 0, num_histories 
		);
	#endif
		
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) 
		printf("reconstruction_cuts_partial_tx Error: %s\n", cudaGetErrorString(cudaStatus));						  
	
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Kernel!\n", cudaStatus);

	reconstruction_cuts_device_2_host(start_position, num_histories);
	for( int i = 0; i < num_histories; i++ ) 
	{    
		if( is_valid_reconstruction_history(start_position, i)  ) 
		{
			first_MLP_voxel_vector[reconstruction_histories] = first_MLP_voxel_vector[ i + start_position ];
			data_shift_vectors( i + start_position, reconstruction_histories );
			reconstruction_histories++;
		}
	}	
	reconstruction_cuts_deallocations();
}
void reconstruction_cuts_partial_tx_nobool(const int start_position, const int num_histories) 
{ 
	// ENDPOINTS_TX_MODE = PARTIAL_TX, ENDPOINTS_ALG = NO_BOOL
	reconstruction_cuts_allocations_nobool(num_histories);
	reconstruction_cuts_host_2_device_nobool( start_position, num_histories);
			
	int num_blocks = static_cast<int>( (num_histories - 1 + ENDPOINTS_PER_BLOCK*ENDPOINTS_PER_THREAD ) / (ENDPOINTS_PER_BLOCK*ENDPOINTS_PER_THREAD)  );
	dim3 dimBlock(ENDPOINTS_PER_BLOCK);
	collect_MLP_endpoints_GPU_nobool<<< num_blocks, dimBlock >>>
	( 
		first_MLP_voxel_d, hull_d, x_entry_d, y_entry_d, z_entry_d, xy_entry_angle_d, xz_entry_angle_d, x_exit_d, y_exit_d, z_exit_d, xy_exit_angle_d, xz_exit_angle_d, 0, num_histories 
	);
		
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) 
		printf("Error: %s\n", cudaGetErrorString(cudaStatus));						  
	
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Kernel!\n", cudaStatus);
	

	reconstruction_cuts_device_2_host_nobool(start_position, num_histories);
	for( int i = 0; i < num_histories; i++ ) 
	{    
		if( first_MLP_voxel_vector[i + start_position] != NUM_VOXELS ) 
		{
			first_MLP_voxel_vector[reconstruction_histories] = first_MLP_voxel_vector[i + start_position];
			data_shift_vectors( i + start_position, reconstruction_histories );
			reconstruction_histories++;
		}
	}	
	reconstruction_cuts_deallocations_nobool();
}
void reconstruction_cuts_partial_tx_preallocated(const int start_position, const int num_histories) 
{ 
	// ENDPOINTS_TX_MODE = PARTIAL_TX_PREALLOCATED, ENDPOINTS_ALG = YES_BOOL
	reconstruction_cuts_host_2_device( start_position, num_histories);
	int num_blocks = static_cast<int>( (num_histories - 1 + ENDPOINTS_PER_BLOCK*ENDPOINTS_PER_THREAD ) / (ENDPOINTS_PER_BLOCK*ENDPOINTS_PER_THREAD)  );
	dim3 dimBlock(ENDPOINTS_PER_BLOCK);
	#if (ENDPOINTS_ALG == YES_BOOL)	
		collect_MLP_endpoints_GPU<<< num_blocks, dimBlock >>>
		( 
			intersected_hull_d, first_MLP_voxel_d, hull_d, x_entry_d, y_entry_d, z_entry_d, xy_entry_angle_d, xz_entry_angle_d, 
			x_exit_d, y_exit_d, z_exit_d, xy_exit_angle_d, xz_exit_angle_d, 0, num_histories 
		);
	#elif (ENDPOINTS_ALG == NO_BOOL)	
		collect_MLP_endpoints_GPU_nobool<<< num_blocks, dimBlock >>>
		( 
			first_MLP_voxel_d, hull_d, x_entry_d, y_entry_d, z_entry_d, xy_entry_angle_d, xz_entry_angle_d, x_exit_d, y_exit_d, z_exit_d, xy_exit_angle_d, xz_exit_angle_d, 0, num_histories 
		);
	#endif 
		
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) 
		printf("reconstruction_cuts_partial_tx_preallocated Error: %s\n", cudaGetErrorString(cudaStatus));						  
	
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Kernel!\n", cudaStatus);						

	reconstruction_cuts_device_2_host(start_position, num_histories);
	for( int i = 0; i < num_histories; i++ ) 
	{	    
		if( is_valid_reconstruction_history(start_position, i)  ) 
		{
			first_MLP_voxel_vector[reconstruction_histories] = first_MLP_voxel_vector[ i + start_position ];
			data_shift_vectors( i + start_position, reconstruction_histories );
			reconstruction_histories++;
		}
	}	
}
void reconstruction_cuts_partial_tx_preallocated_nobool(const int start_position, const int num_histories) 
{
	// ENDPOINTS_TX_MODE = PARTIAL_TX_PREALLOCATED, ENDPOINTS_ALG = NO_BOOL
	reconstruction_cuts_host_2_device_nobool( start_position, num_histories);
	int num_blocks = static_cast<int>( (num_histories - 1 + ENDPOINTS_PER_BLOCK*ENDPOINTS_PER_THREAD ) / (ENDPOINTS_PER_BLOCK*ENDPOINTS_PER_THREAD)  );
	dim3 dimBlock(ENDPOINTS_PER_BLOCK);
	collect_MLP_endpoints_GPU_nobool<<< num_blocks, dimBlock >>>
	( 
		first_MLP_voxel_d, hull_d, x_entry_d, y_entry_d, z_entry_d, xy_entry_angle_d, xz_entry_angle_d, x_exit_d, y_exit_d, z_exit_d, xy_exit_angle_d, xz_exit_angle_d, 0, num_histories 
	);
	
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) 
		printf("Error: %s\n", cudaGetErrorString(cudaStatus));						  
	
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Kernel!\n", cudaStatus);
	
	reconstruction_cuts_device_2_host_nobool(start_position, num_histories);	
	for( int i = 0; i < num_histories; i++ ) 
	{  
		if( first_MLP_voxel_vector[i + start_position] != NUM_VOXELS ) 
		{
			first_MLP_voxel_vector.at(reconstruction_histories) = first_MLP_voxel_vector.at( i + start_position );
			data_shift_vectors( i + start_position, reconstruction_histories );
			reconstruction_histories++;
		}
	}	
}
void reconstruction_cuts()
{
	timer( START, begin_endpoints, "for finding MLP endpoints");	
	int remaining_histories = post_cut_histories, start_position = 0, histories_2_process = 0;
	reconstruction_histories = 0;

	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) 
		printf("Before image reconstruction Error: %s\n", cudaGetErrorString(cudaStatus));
		
	//post_cut_histories = 100000000;
	first_MLP_voxel_vector.resize( post_cut_histories ); 
	reconstruction_cuts_hull_transfer();		// Free hull_d which may not be aligned well on the GPU and reallocate/transfer it to GPU again

	print_colored_text("Collecting MLP endpoints...", CYAN_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );
	if( ENDPOINTS_TX_MODE == FULL_TX )
	{
		print_colored_text("Identifying MLP endpoints with all data transferred to the GPU before the 1st kernel launch...", CYAN_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );
		if( ENDPOINTS_ALG == YES_BOOL )
		{
			print_colored_text( "Using boolean array to identify protons hitting/missing hull...", CYAN_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );
			reconstruction_cuts_allocations(post_cut_histories);
			reconstruction_cuts_host_2_device( 0, post_cut_histories );  
			reconstruction_cuts_full_tx( post_cut_histories );
			reconstruction_cuts_deallocations();
		}
		// 
		else if( ENDPOINTS_ALG == NO_BOOL )
		{
			print_colored_text("Using hull entry voxel # to identify protons hitting/missing hull...", CYAN_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );
			reconstruction_cuts_allocations_nobool(post_cut_histories);
			reconstruction_cuts_host_2_device_nobool( 0, post_cut_histories );  
			reconstruction_cuts_full_tx_nobool( post_cut_histories );
			reconstruction_cuts_deallocations_nobool();
		}
	}
	else if( ENDPOINTS_TX_MODE == PARTIAL_TX )
	{
		print_colored_text("Identifying MLP endpoints using partial data transfers with GPU arrays allocated/freed each kernel launch...", CYAN_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );
		if( ENDPOINTS_ALG == YES_BOOL )
		{
			print_colored_text("Using boolean array to identify protons hitting/missing hull...", CYAN_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );
			while( remaining_histories > 0 )
			{
				if( remaining_histories > MAX_ENDPOINTS_HISTORIES )
					histories_2_process = MAX_ENDPOINTS_HISTORIES;
				else
					histories_2_process = remaining_histories;		
				reconstruction_cuts_partial_tx( start_position, histories_2_process );
				remaining_histories -= MAX_ENDPOINTS_HISTORIES;
				start_position		+= MAX_ENDPOINTS_HISTORIES;
			}
		}
		else if( ENDPOINTS_ALG == NO_BOOL )
		{
			print_colored_text("Using hull entry voxel # to identify protons hitting/missing hull...", CYAN_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );
			while( remaining_histories > 0 )
			{
				if( remaining_histories > MAX_ENDPOINTS_HISTORIES )
					histories_2_process = MAX_ENDPOINTS_HISTORIES;
				else
					histories_2_process = remaining_histories;	
				reconstruction_cuts_partial_tx_nobool( start_position, histories_2_process );
				remaining_histories -= MAX_ENDPOINTS_HISTORIES;
				start_position		+= MAX_ENDPOINTS_HISTORIES;
			}	
		}
	}
	else if( ENDPOINTS_TX_MODE == PARTIAL_TX_PREALLOCATED )
	{
		print_colored_text("Identifying MLP endpoints using partial data transfers with preallocated GPU arrays reused each kernel launch...", CYAN_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );
		if( ENDPOINTS_ALG == YES_BOOL )
		{
			print_colored_text("Using boolean array to identify protons hitting/missing hull...", CYAN_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );
			reconstruction_cuts_allocations(MAX_ENDPOINTS_HISTORIES); 
			while( remaining_histories > 0 )
			{
				if( remaining_histories > MAX_ENDPOINTS_HISTORIES )
					histories_2_process = MAX_ENDPOINTS_HISTORIES;
				else
					histories_2_process = remaining_histories;			
				reconstruction_cuts_partial_tx_preallocated( start_position, histories_2_process );
				remaining_histories -= MAX_ENDPOINTS_HISTORIES;
				start_position		+= MAX_ENDPOINTS_HISTORIES;
			}
			reconstruction_cuts_deallocations();
		}
		else if( ENDPOINTS_ALG == NO_BOOL )
		{
			print_colored_text("Using hull entry voxel # to identify protons hitting/missing hull...", CYAN_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );
			reconstruction_cuts_allocations_nobool(MAX_ENDPOINTS_HISTORIES); 
			while( remaining_histories > 0 )
			{
				if( remaining_histories > MAX_ENDPOINTS_HISTORIES )
					histories_2_process = MAX_ENDPOINTS_HISTORIES;
				else
					histories_2_process = remaining_histories;	
				reconstruction_cuts_partial_tx_preallocated_nobool( start_position, histories_2_process );
				remaining_histories -= MAX_ENDPOINTS_HISTORIES;
				start_position		+= MAX_ENDPOINTS_HISTORIES;
			}	
			reconstruction_cuts_deallocations_nobool();
		}
	}
	sprintf(print_statement, "------> Protons that intersected the hull and will be used for reconstruction = %d", reconstruction_histories);
	print_colored_text( print_statement, GREEN_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT );
	print_section_exit( "Finished removing unnecessary reconstruction histories", SECTION_EXIT_CSTRING, RED_TEXT, RED_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );
	execution_time_endpoints = timer( STOP, begin_endpoints, "for finding MLP endpoints");
	
	// Reduce the size of the vectors to reconstruction_histories and shrink their capacity to match
	first_MLP_voxel_vector.resize( reconstruction_histories );
	//first_MLP_voxel_vector.shrink_to_fit();
	resize_vectors( reconstruction_histories );
	//shrink_vectors();

	for( int i = 0; i < reconstruction_histories; i++ )
	{
		if( first_MLP_voxel_vector[i] >= NUM_VOXELS + 1 )
			cout << "i = " << i << "voxel = " << first_MLP_voxel_vector[i] << endl;
	}
	
}
/***********************************************************************************************************************************************************************************************************************/
/****************************************************************************************** Generate/Export/Import MLP Lookup Tables ***********************************************************************************/
/***********************************************************************************************************************************************************************************************************************/
void generate_trig_tables()
{
	//printf("TRIG_TABLE_ELEMENTS = %d\n", TRIG_TABLE_ELEMENTS );
	double sin_term, cos_term, val;
	
	sin_table_h = (double*) calloc( TRIG_TABLE_ELEMENTS + 1, sizeof(double) );
	cos_table_h = (double*) calloc( TRIG_TABLE_ELEMENTS + 1, sizeof(double) );
	
	sin_table_file = fopen( SIN_TABLE_FILENAME, "wb" );
	cos_table_file = fopen( COS_TABLE_FILENAME, "wb" );
	/*for( float i = TRIG_TABLE_MIN; i <= TRIG_TABLE_MAX; i+= TRIG_TABLE_STEP )
	{
		sin_term = sin(i);
		cos_term = cos(i);
		fwrite( &sin_term, sizeof(float), 1, sin_table_file );
		fwrite( &cos_term, sizeof(float), 1, cos_table_file );
	}*/
	for( int i = 0; i <= TRIG_TABLE_ELEMENTS; i++ )
	{
		val =  TRIG_TABLE_MIN + i * TRIG_TABLE_STEP;
		sin_term = sin(val);
		cos_term = cos(val);
		sin_table_h[i] = sin_term;
		cos_table_h[i] = cos_term;
		fwrite( &sin_term, sizeof(double), 1, sin_table_file );
		fwrite( &cos_term, sizeof(double), 1, cos_table_file );
	}
	fclose(sin_table_file);
	fclose(cos_table_file);
}
void generate_scattering_coefficient_table()
{
	double scattering_coefficient;
	scattering_table_file = fopen( COEFFICIENT_FILENAME, "wb" );
	int i = 0;
	double depth = 0.0;
	scattering_table_h = (double*)calloc( COEFF_TABLE_ELEMENTS + 1, sizeof(double));
	for( int step_num = 0; step_num <= COEFF_TABLE_ELEMENTS; step_num++ )
	{
		depth = step_num * COEFF_TABLE_STEP;
		scattering_coefficient = pow( E_0 * ( 1 + 0.038 * log(depth / X0) ), 2.0 ) / X0;
		scattering_table_h[i] = scattering_coefficient;
		//fwrite( &scattering_coefficient, sizeof(float), 1, scattering_table_file );
		i++;
	}
	//for( float depth = 0.0; depth <= COEFF_TABLE_RANGE; depth+= COEFF_TABLE_STEP )
	//{
	//	scattering_coefficient = pow( E_0 * ( 1 + 0.038 * log(depth / X0) ), 2.0 ) / X0;
	//	scattering_table_h[i] = scattering_coefficient;
	//	//fwrite( &scattering_coefficient, sizeof(float), 1, scattering_table_file );
	//	i++;
	//}
	fwrite(scattering_table_h, sizeof(double), COEFF_TABLE_ELEMENTS + 1, scattering_table_file );
	fclose(scattering_table_file);
	//for( int step_num = 0; step_num <= COEFF_TABLE_ELEMENTS; step_num++ )
	//	cout << scattering_table_h[step_num] << endl;
	//cout << "elements = " << i << endl;
	//cout << "COEFF_TABLE_ELEMENTS = " << COEFF_TABLE_ELEMENTS << endl;
	//cout << scattering_table_h[i-1] << endl;
	//cout << (pow( E_0 * ( 1 + 0.038 * log(COEFF_TABLE_RANGE / X0) ), 2.0 ) / X0) << endl;
	
}
void generate_polynomial_tables()
{
	int i = 0;
	double du;
	//float poly_1_2_val, poly_2_3_val, poly_3_4_val, poly_2_6_val, poly_3_12_val;
	poly_1_2_h  = (double*) calloc( POLY_TABLE_ELEMENTS + 1, sizeof(double) );
	poly_2_3_h  = (double*) calloc( POLY_TABLE_ELEMENTS + 1, sizeof(double) );
	poly_3_4_h  = (double*) calloc( POLY_TABLE_ELEMENTS + 1, sizeof(double) );
	poly_2_6_h  = (double*) calloc( POLY_TABLE_ELEMENTS + 1, sizeof(double) );
	poly_3_12_h = (double*) calloc( POLY_TABLE_ELEMENTS + 1, sizeof(double) );

	poly_1_2_file  = fopen( POLY_1_2_FILENAME,  "wb" );
	poly_2_3_file  = fopen( POLY_2_3_FILENAME,  "wb" );
	poly_3_4_file  = fopen( POLY_3_4_FILENAME,  "wb" );
	poly_2_6_file  = fopen( POLY_2_6_FILENAME,  "wb" );
	poly_3_12_file = fopen( POLY_3_12_FILENAME, "wb" );
	for( int step_num = 0; step_num <= POLY_TABLE_ELEMENTS; step_num++ )
	{
		du = step_num * POLY_TABLE_STEP;
		//poly_1_2_val = A_0		   + du * (A_1_OVER_2  + du * (A_2_OVER_3  + du * (A_3_OVER_4  + du * (A_4_OVER_5   + du * A_5_OVER_6   ))));	// 1, 2, 3, 4, 5, 6
		//poly_2_3_val = A_0_OVER_2 + du * (A_1_OVER_3  + du * (A_2_OVER_4  + du * (A_3_OVER_5  + du * (A_4_OVER_6   + du * A_5_OVER_7   ))));	// 2, 3, 4, 5, 6, 7
		//poly_3_4_val = A_0_OVER_3 + du * (A_1_OVER_4  + du * (A_2_OVER_5  + du * (A_3_OVER_6  + du * (A_4_OVER_7   + du * A_5_OVER_8   ))));	// 3, 4, 5, 6, 7, 8
		//poly_2_6_val = A_0_OVER_2 + du * (A_1_OVER_6  + du * (A_2_OVER_12 + du * (A_3_OVER_20 + du * (A_4_OVER_30  + du * A_5_OVER_42  ))));	// 2, 6, 12, 20, 30, 42
		//poly_3_12_val = A_0_OVER_3 + du * (A_1_OVER_12 + du * (A_2_OVER_30 + du * (A_3_OVER_60 + du * (A_4_OVER_105 + du * A_5_OVER_168 ))));	// 3, 12, 30, 60, 105, 168		
		poly_1_2_h[step_num]  = du * ( A_0		   + du * (A_1_OVER_2  + du * (A_2_OVER_3  + du * (A_3_OVER_4  + du * (A_4_OVER_5   + du * A_5_OVER_6   )))) );	// 1, 2, 3, 4, 5, 6
		poly_2_3_h[step_num]  = pow(du, 2) * ( A_0_OVER_2 + du * (A_1_OVER_3  + du * (A_2_OVER_4  + du * (A_3_OVER_5  + du * (A_4_OVER_6   + du * A_5_OVER_7   )))) );	// 2, 3, 4, 5, 6, 7
		poly_3_4_h[step_num]  = pow(du, 3) * ( A_0_OVER_3 + du * (A_1_OVER_4  + du * (A_2_OVER_5  + du * (A_3_OVER_6  + du * (A_4_OVER_7   + du * A_5_OVER_8   )))) );	// 3, 4, 5, 6, 7, 8
		poly_2_6_h[step_num]  = pow(du, 2) * ( A_0_OVER_2 + du * (A_1_OVER_6  + du * (A_2_OVER_12 + du * (A_3_OVER_20 + du * (A_4_OVER_30  + du * A_5_OVER_42  )))) );	// 2, 6, 12, 20, 30, 42
		poly_3_12_h[step_num] = pow(du, 3) * ( A_0_OVER_3 + du * (A_1_OVER_12 + du * (A_2_OVER_30 + du * (A_3_OVER_60 + du * (A_4_OVER_105 + du * A_5_OVER_168 )))) );	// 3, 12, 30, 60, 105, 168		
		
		/*fwrite( &poly_1_2_h[step_num],  sizeof(float), 1, poly_1_2_file  );
		fwrite( &poly_2_3_h[step_num],  sizeof(float), 1, poly_2_3_file  );
		fwrite( &poly_3_4_h[step_num],  sizeof(float), 1, poly_3_4_file  );
		fwrite( &poly_2_6_h[step_num],  sizeof(float), 1, poly_2_6_file  );
		fwrite( &poly_3_12_h[step_num], sizeof(float), 1, poly_3_12_file );*/
		i++;
	}
	fwrite( poly_1_2_h,  sizeof(double), POLY_TABLE_ELEMENTS + 1, poly_1_2_file  );
	fwrite( poly_2_3_h,  sizeof(double), POLY_TABLE_ELEMENTS + 1, poly_2_3_file  );
	fwrite( poly_3_4_h,  sizeof(double), POLY_TABLE_ELEMENTS + 1, poly_3_4_file  );
	fwrite( poly_2_6_h,  sizeof(double), POLY_TABLE_ELEMENTS + 1, poly_2_6_file  );
	fwrite( poly_3_12_h, sizeof(double), POLY_TABLE_ELEMENTS + 1, poly_3_12_file );

	fclose( poly_1_2_file  );
	fclose( poly_2_3_file  );
	fclose( poly_3_4_file  );
	fclose( poly_2_6_file  );
	fclose( poly_3_12_file );
	//cout << "i = " << i << endl;														
	//cout << "POLY_TABLE_ELEMENTS = " << POLY_TABLE_ELEMENTS << endl;			
}
void import_trig_tables()
{
	sin_table_h = (double*) calloc( TRIG_TABLE_ELEMENTS + 1, sizeof(double) );
	cos_table_h = (double*) calloc( TRIG_TABLE_ELEMENTS + 1, sizeof(double) );

	sin_table_file = fopen( SIN_TABLE_FILENAME, "rb" );
	cos_table_file = fopen( COS_TABLE_FILENAME, "rb" );
	
	fread(sin_table_h, sizeof(double), TRIG_TABLE_ELEMENTS + 1, sin_table_file );
	fread(cos_table_h, sizeof(double), TRIG_TABLE_ELEMENTS + 1, cos_table_file );
	
	fclose(sin_table_file);
	fclose(cos_table_file);
}
void import_scattering_coefficient_table()
{
	scattering_table_h = (double*)calloc( COEFF_TABLE_ELEMENTS + 1, sizeof(double));
	scattering_table_file = fopen( COEFFICIENT_FILENAME, "rb" );
	fread(scattering_table_h, sizeof(double), COEFF_TABLE_ELEMENTS + 1, scattering_table_file );
	fclose(scattering_table_file);
}
void import_polynomial_tables()
{
	poly_1_2_h  = (double*) calloc( POLY_TABLE_ELEMENTS + 1, sizeof(double) );
	poly_2_3_h  = (double*) calloc( POLY_TABLE_ELEMENTS + 1, sizeof(double) );
	poly_3_4_h  = (double*) calloc( POLY_TABLE_ELEMENTS + 1, sizeof(double) );
	poly_2_6_h  = (double*) calloc( POLY_TABLE_ELEMENTS + 1, sizeof(double) );
	poly_3_12_h = (double*) calloc( POLY_TABLE_ELEMENTS + 1, sizeof(double) );

	poly_1_2_file  = fopen( POLY_1_2_FILENAME,  "rb" );
	poly_2_3_file  = fopen( POLY_2_3_FILENAME,  "rb" );
	poly_3_4_file  = fopen( POLY_3_4_FILENAME,  "rb" );
	poly_2_6_file  = fopen( POLY_2_6_FILENAME,  "rb" );
	poly_3_12_file = fopen( POLY_3_12_FILENAME, "rb" );

	fread( poly_1_2_h,  sizeof(double), POLY_TABLE_ELEMENTS + 1, poly_1_2_file  );
	fread( poly_2_3_h,  sizeof(double), POLY_TABLE_ELEMENTS + 1, poly_2_3_file  );
	fread( poly_3_4_h,  sizeof(double), POLY_TABLE_ELEMENTS + 1, poly_3_4_file  );
	fread( poly_2_6_h,  sizeof(double), POLY_TABLE_ELEMENTS + 1, poly_2_6_file  );
	fread( poly_3_12_h, sizeof(double), POLY_TABLE_ELEMENTS + 1, poly_3_12_file );

	fclose( poly_1_2_file  );
	fclose( poly_2_3_file  );
	fclose( poly_3_4_file  );
	fclose( poly_2_6_file  );
	fclose( poly_3_12_file );
}
void MLP_lookup_table_2_GPU()
{
	unsigned int size_trig_tables			= ( TRIG_TABLE_ELEMENTS	 + 1 ) * sizeof(double);
	unsigned int size_coefficient_tables	= ( COEFF_TABLE_ELEMENTS + 1 ) * sizeof(double);
	unsigned int size_poly_tables			= ( POLY_TABLE_ELEMENTS  + 1 ) * sizeof(double);

	cudaMalloc( (void**) &sin_table_d,			size_trig_tables		);
	cudaMalloc( (void**) &cos_table_d,			size_trig_tables		);
	cudaMalloc( (void**) &scattering_table_d,	size_coefficient_tables );
	cudaMalloc( (void**) &poly_1_2_d,			size_poly_tables		);
	cudaMalloc( (void**) &poly_2_3_d,			size_poly_tables		);
	cudaMalloc( (void**) &poly_3_4_d,			size_poly_tables		);	
	cudaMalloc( (void**) &poly_2_6_d,			size_poly_tables		);
	cudaMalloc( (void**) &poly_3_12_d,			size_poly_tables		);

	cudaMemcpy( sin_table_d,		sin_table_h,		size_trig_tables,			cudaMemcpyHostToDevice );
	cudaMemcpy( cos_table_d,		cos_table_h,		size_trig_tables,			cudaMemcpyHostToDevice );
	cudaMemcpy( scattering_table_d, scattering_table_h, size_coefficient_tables,	cudaMemcpyHostToDevice );
	cudaMemcpy( poly_1_2_d,			poly_1_2_h,			size_poly_tables,			cudaMemcpyHostToDevice );
	cudaMemcpy( poly_2_3_d,			poly_2_3_h,			size_poly_tables,			cudaMemcpyHostToDevice );
	cudaMemcpy( poly_3_4_d,			poly_3_4_h,			size_poly_tables,			cudaMemcpyHostToDevice );
	cudaMemcpy( poly_2_6_d,			poly_2_6_h,			size_poly_tables,			cudaMemcpyHostToDevice );
	cudaMemcpy( poly_3_12_d,		poly_3_12_h,		size_poly_tables,			cudaMemcpyHostToDevice );
}
void setup_MLP_lookup_tables()
{
	// Generate MLP lookup tables and transfer these to the GPU
	print_colored_text( "Setting up MLP lookup tables and transferring them to GPU...", CYAN_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );	
	timer( START, begin_tables, "for generating MLP lookup tables and transferring them to the GPU");		
	if(IMPORT_MLP_LOOKUP_TABLES)
	{
		print_colored_text( "Importing MLP lookup tables from disk...", CYAN_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );	
		import_trig_tables();
		import_scattering_coefficient_table();
		import_polynomial_tables();
	}
	else
	{
		print_colored_text( "Generating MLP lookup tables...", CYAN_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );	
		generate_trig_tables();
		generate_scattering_coefficient_table();
		generate_polynomial_tables();
	}
	MLP_lookup_table_2_GPU();
	execution_time_tables = timer( STOP, begin_tables, "for generating MLP lookup tables and transferring them to the GPU");	
	print_section_exit( "Finished setting up MLP lookup tables and transferring them to GPU", SECTION_EXIT_CSTRING, RED_TEXT, RED_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );	
}
void free_MLP_lookup_tables()
{
	free( sin_table_h);
	free( cos_table_h);
	free( scattering_table_h);
	free( poly_1_2_h);
	free( poly_2_3_h);
	free( poly_3_4_h);
	free( poly_2_6_h);
	free( poly_3_12_h);

	cudaFree( sin_table_d);
	cudaFree( cos_table_d);
	cudaFree( scattering_table_d);
	cudaFree( poly_1_2_d);
	cudaFree( poly_2_3_d);
	cudaFree( poly_3_4_d);
	cudaFree( poly_2_6_d);
	cudaFree( poly_3_12_d);

	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) 
		printf("free_MLP_lookup_tables Error: %s\n", cudaGetErrorString(cudaStatus));
	
}
/***********************************************************************************************************************************************************************************************************************/
/************************************************************************************************** Generate initial iterate *******************************************************************************************/
/***********************************************************************************************************************************************************************************************************************/
void create_hull_image_hybrid()
{
	dim3 dimBlock( SLICES );
	dim3 dimGrid( COLUMNS, ROWS );   

	create_hull_image_hybrid_GPU<<< dimGrid, dimBlock >>>( hull_d, FBP_image_d );
	cudaMemcpy( x_h, FBP_image_d, SIZE_IMAGE_FLOAT, cudaMemcpyDeviceToHost );
}
__global__ void create_hull_image_hybrid_GPU( bool*& hull, float*& FBP_image)
{
	int row = blockIdx.y, column = blockIdx.x, slice = threadIdx.x;
	int voxel = column + row * COLUMNS + slice * COLUMNS * ROWS;
	FBP_image[voxel] *= hull[voxel];
}
void initial_iterate_generate_hybrid()
{
	for( int voxel = 0; voxel < NUM_VOXELS; voxel++ )
	{
		if( hull_h[voxel] == true )
			x_h[voxel] = FBP_image_h[voxel];
		else
			x_h[voxel] = 0.0;
	}
}
void reconstruct_initial_iterate()
{
//	#define X0_ITERATIONS					12	
//float X0_LAMBDA						= 0.0002;				
//#define X0_DROP_BLOCK_SIZE				3200			
//#define X0_BOUND_IMAGE					OFF					
//#define X0_TVS_ON						ON					
//#define X0_TVS_OLD						OFF					
//#define X0_NTVS_ON						(TVS_ON && !TVS_OLD)	
////#define NTVS_ON					OFF								
//#define X0_TVS_FIRST					ON							
//#define X0_TVS_PARALLEL				OFF								
//#define X0_TVS_CONDITIONED				OFF							
//#define X0_A							0.75						
//UINT X0_TVS_REPETITIONS				= 5;							
	if( PROJECTION_ALGORITHM == DROP )
		DROP_GPU(reconstruction_histories, X0_ITERATIONS, X0_LAMBDA);
	//DROP_free_update_arrays();	
	
}
void define_initial_iterate()
{
	print_colored_text( "Generating initial iterate...", CYAN_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );		
	x_h = (float*) calloc( NUM_VOXELS, sizeof(float) );

	switch( X_0 )
	{
		case X_HULL		:	std::copy( hull_h, hull_h + NUM_VOXELS, x_h );															break;
		case FBP_IMAGE	:	x_h = FBP_image_h;																						break;
		case HYBRID		:	initial_iterate_generate_hybrid();																		break;
		case IMPORT		:	import_image( x_h, INPUT_ITERATE_PATH );																break;
		case ZEROS		:	break;
		default			:	puts("ERROR: Invalid initial iterate selected");
							exit(1);
	}
	//cudaMalloc((void**) &x_d, SIZE_IMAGE_FLOAT );
	//cudaMemcpy( x_d, x_h, SIZE_IMAGE_FLOAT, cudaMemcpyHostToDevice );
	if(IDENTIFY_X_0_AIR)
	{
		for(int voxel = 0; voxel < NUM_VOXELS; voxel++)
		{
			if(x_h[voxel] < X_0_AIR_THRESHOLD)
				x_h[voxel] = 0.0;
		}
	}
	if( WRITE_X_0 ) 
	{
		array_2_disk(X_0_FILENAME, OUTPUT_DIRECTORY, OUTPUT_FOLDER_UNIQUE, x_h, COLUMNS, ROWS, SLICES, NUM_VOXELS, true );
		write_PNG(X_0_FILENAME, x_h);
	}
	if(RECONSTRUCT_X_0)
		reconstruct_initial_iterate();
	exit_program_if( EXIT_AFTER_X_O, "through initial iterate generation" );
	print_section_exit( "Finished generating initial iterate", SECTION_EXIT_CSTRING, RED_TEXT, RED_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );	
}
void generate_preprocessing_images()
{
	print_section_header( "Generating sinogram, FBP, and initial iterate and selecting hull to use in image reconstruction", MINOR_SECTION_SEPARATOR, LIGHT_CYAN_TEXT, YELLOW_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT );
	construct_sinogram();
	if( FBP_ON )
		FBP();
	exit_program_if( EXIT_AFTER_FBP, "through FBP" );
	hull_selection();
	define_initial_iterate();
	if(RECONSTRUCT_X_0)
	{
		reconstruct_initial_iterate();
	}
}		
/***********************************************************************************************************************************************************************************************************************/
/******************************************************************************************* Generate history ordering for reconstruction ******************************************************************************/
/***********************************************************************************************************************************************************************************************************************/
void generate_history_sequence(ULL N, ULL offset_prime, ULL* sequence )
{
    sequence = (ULL*) calloc( N, sizeof(ULL));
	for( ULL i = 1; i < N; i++ )
        sequence[i] = ( sequence[i-1] + offset_prime ) % N;
}
void verify_history_sequence(ULL N, ULL offset_prime, ULL* sequence )
{
	for( ULL i = 1; i < N; i++ )
    {
        if(sequence[i] == 1)
        {
            printf("repeats at i = %llu\n", i);
            printf("history_sequence[i] = %llu\n", sequence[i]);
            break;
        }
        if(sequence[i] > N)
        {
            printf("exceeds at i = %llu\n", i);
            printf("history_sequence[i] = %llu\n", sequence[i]);
            break;
        }
    }
}
void print_history_sequence(ULL* sequence, ULL print_start, ULL print_end )
{
    for( ULL i = print_start; i < print_end; i++ )
		printf("history_sequence[i] = %llu\n", sequence[i]);
}
void apply_history_sequence(ULL N, ULL offset_prime, ULL* sequence)
{
	uint read_index;
	if(RECON_HISTORY_ORDERING == PRIME_PERMUTATION)
	{
		// 
		bin_num_vector_reconstruction.reserve(N);
		//gantry_angle_vector.reserve(gantry_angle_vector);
		WEPL_vector_reconstruction.reserve(N);
		x_entry_vector_reconstruction.reserve(N);
		y_entry_vector_reconstruction.reserve(N);
		z_entry_vector_reconstruction.reserve(N);
		x_exit_vector_reconstruction.reserve(N);
		y_exit_vector_reconstruction.reserve(N);
		z_exit_vector_reconstruction.reserve(N);
		xy_entry_angle_vector_reconstruction.reserve(N);
		xz_entry_angle_vector_reconstruction.reserve(N);
		xy_exit_angle_vector_reconstruction.reserve(N);
		xz_exit_angle_vector_reconstruction.reserve(N);
		first_MLP_voxel_vector_reconstruction.reserve(N);
	
		// 
		generate_history_sequence(N, offset_prime, sequence );
		
		for(int i = 0; i < N; i++ ) 
		{  
			read_index = sequence[i];
			bin_num_vector_reconstruction.push_back(bin_num_vector[read_index]);
			//gantry_angle_vector.push_back(gantry_angle_vector[read_index]);
			WEPL_vector_reconstruction.push_back(WEPL_vector[read_index]);
			x_entry_vector_reconstruction.push_back(x_entry_vector[read_index]);
			y_entry_vector_reconstruction.push_back(y_entry_vector[read_index]);
			z_entry_vector_reconstruction.push_back(z_entry_vector[read_index]);
			x_exit_vector_reconstruction.push_back(x_exit_vector[read_index]);
			y_exit_vector_reconstruction.push_back(y_exit_vector[read_index]);
			z_exit_vector_reconstruction.push_back(z_exit_vector[read_index]);
			xy_entry_angle_vector_reconstruction.push_back(xy_entry_angle_vector[read_index]);
			xz_entry_angle_vector_reconstruction.push_back(xz_entry_angle_vector[read_index]);
			xy_exit_angle_vector_reconstruction.push_back(xy_exit_angle_vector[read_index]);
			xz_exit_angle_vector_reconstruction.push_back(xz_exit_angle_vector[read_index]);
			first_MLP_voxel_vector_reconstruction.push_back(first_MLP_voxel_vector[read_index]);		
		}
	}
	else if(RECON_HISTORY_ORDERING == SEQUENTIAL)
	{
		bin_num_vector_reconstruction = bin_num_vector;
		//gantry_angle_vector = gantry_angle_vector;
		WEPL_vector_reconstruction = WEPL_vector;
		x_entry_vector_reconstruction = x_entry_vector;
		y_entry_vector_reconstruction = y_entry_vector;
		z_entry_vector_reconstruction = z_entry_vector;
		x_exit_vector_reconstruction = x_exit_vector;
		y_exit_vector_reconstruction = y_exit_vector;
		z_exit_vector_reconstruction = z_exit_vector;
		xy_entry_angle_vector_reconstruction = xy_entry_angle_vector;
		xz_entry_angle_vector_reconstruction = xz_entry_angle_vector;
		xy_exit_angle_vector_reconstruction = xy_exit_angle_vector;
		xz_exit_angle_vector_reconstruction = xz_exit_angle_vector;
		first_MLP_voxel_vector_reconstruction = first_MLP_voxel_vector;		
	}
//	bin_num_vector_ordered = ;			
//		gantry_angle_vector_ordered;	
//		WEPL_vector_ordered;		
//		x_entry_vector_ordered;		
//		y_entry_vector_ordered;		
//		z_entry_vector_ordered;		
//		x_exit_vector_ordered;			
//		y_exit_vector_ordered;			
//		z_exit_vector_ordered;			
//		xy_entry_angle_vector_ordered;	
//		xz_entry_angle_vector_ordered;	
//		xy_exit_angle_vector_ordered;	
//		xz_exit_angle_vector_ordered;	
//		first_MLP_voxel_vector_ordered;
//		voxel_x_vector_ordered;
//		voxel_y_vector_ordered;
//		voxel_z_vector_ordered;		
}
void shuffle_blocks()
{
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::shuffle (DROP_block_order.begin(),				DROP_block_order.end(),				std::default_random_engine(seed));
	std::shuffle (DROP_block_sizes.begin(),				DROP_block_sizes.end(),				std::default_random_engine(seed));
	std::shuffle (DROP_block_start_positions.begin(),	DROP_block_start_positions.end(),	std::default_random_engine(seed));
	/*std::srand ( seed );
	std::random_shuffle ( DROP_block_order.begin(), DROP_block_order.end() );
	std::srand ( seed );
	std::random_shuffle ( DROP_block_sizes.begin(), DROP_block_sizes.end() );
	std::srand ( seed );
	std::random_shuffle ( DROP_block_start_positions.begin(), DROP_block_start_positions.end() );
	*/
	//unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	//std::shuffle (DROP_block_order.begin(), DROP_block_order.end(), std::default_random_engine(seed));
	//std::shuffle (DROP_block_sizes.begin(), DROP_block_sizes.end(), std::default_random_engine(seed));
	//std::shuffle (DROP_block_start_positions.begin(), DROP_block_start_positions.end(), std::default_random_engine(seed));
}
void rotate_blocks_left()
{
	 std::rotate(DROP_block_order.begin(),				DROP_block_order.begin() + 1,			DROP_block_order.end());
	 std::rotate(DROP_block_sizes.begin(),				DROP_block_sizes.begin() + 1,			DROP_block_sizes.end());
	 std::rotate(DROP_block_start_positions.begin(),	DROP_block_start_positions.begin() + 1, DROP_block_start_positions.end());
}
void rotate_blocks_right()
{
	std::rotate(DROP_block_order.rbegin(),				DROP_block_order.rbegin() + 1,				DROP_block_order.rend());
	std::rotate(DROP_block_sizes.rbegin(),				DROP_block_sizes.rbegin() + 1,				DROP_block_sizes.rend());
	std::rotate(DROP_block_start_positions.rbegin(),	DROP_block_start_positions.rbegin() + 1,	DROP_block_start_positions.rend());
}
void print_DROP_block_info()
{
	for(int i = 0; i < num_DROP_blocks; i++)
	{
		cout << "i = " << i << ": " << DROP_block_order[i] << std::endl;
	}
	cout << endl;
	for(int i = 0; i < num_DROP_blocks; i++)
	{
		cout << "i = " << i << ": " << DROP_block_sizes[i] << std::endl;
	}
	cout << endl;
	for(int i = 0; i < num_DROP_blocks; i++)
	{
		cout << "i = " << i << ": " << DROP_block_start_positions[i] << std::endl;
	}
	cout << endl;	
}
void recon_DROP_initializations()
{
	DROP_last_block_size = reconstruction_histories % DROP_BLOCK_SIZE;
	num_DROP_blocks = static_cast<UINT>(ceil(reconstruction_histories / DROP_BLOCK_SIZE));
	
	// Construct temporary vectors with DROP block info 
	std::vector<UINT> DROP_block_sizes_constructor( num_DROP_blocks, DROP_BLOCK_SIZE);
	std::vector<UINT> DROP_block_order_constructor( num_DROP_blocks);
	std::vector<UINT> DROP_block_start_positions_constructor( num_DROP_blocks, DROP_BLOCK_SIZE);	
	DROP_block_sizes_constructor.back() = DROP_last_block_size;
	std::iota (DROP_block_order_constructor.begin(), DROP_block_order_constructor.end(), 0);
	DROP_block_start_positions_constructor.front() = 0;	
	for(int i = 1; i < num_DROP_blocks; i++)
		DROP_block_start_positions_constructor[i] += DROP_block_start_positions_constructor[i - 1];
	
	// Use temporary vectors to set DROP block info vectors
	DROP_block_sizes = DROP_block_sizes_constructor;
	DROP_block_order = DROP_block_order_constructor;
	DROP_block_start_positions = DROP_block_start_positions_constructor;

	/*std::cout << "Initial block info construction" << std::endl;
	print_DROP_block_info();
	
	
	std::cout << "Rotate left block info" << std::endl;
	rotate_blocks_left();
	print_DROP_block_info();
	if(exit_prompt( "Enter 'c' to continue execution, any other character exits program", 'c'))
		exit_program();
	
	std::cout << "Rotate right block info" << std::endl;
	rotate_blocks_right();
	print_DROP_block_info();
	if(exit_prompt( "Enter 'c' to continue execution, any other character exits program", 'c'))
		exit_program();
	
	std::cout << "Shuffled block info" << std::endl;
	shuffle_blocks();
	print_DROP_block_info();
	if(exit_prompt( "Enter 'c' to continue execution, any other character exits program", 'c'))
		exit_program();*/
	
	switch(DROP_BLOCK_ORDER)
	{
		case SEQUENTIAL:			break;
		case ROTATE_LEFT:		rotate_blocks_left();		break;
		case ROTATE_RIGHT:		rotate_blocks_right();		break;
		case RANDOMLY_SHUFFLE:		shuffle_blocks();			break;
	}
	//print_DROP_block_info();	
}
/**************************************************************************************************************************************************************************/
/******************************************* MLP and image update calculation/application functions ***********************************************************************/
/**************************************************************************************************************************************************************************/
__device__ double EffectiveChordLength_GPU(double abs_angle_t, double abs_angle_v)
{
	
	double eff_angle_t,eff_angle_v;
	
	eff_angle_t=abs_angle_t-((int)(abs_angle_t/(PI/2)))*(PI/2);
	
	eff_angle_v=fabs(abs_angle_v);
	
	// Get the effective chord in the t-u plane
	double step_fraction=MLP_U_STEP/VOXEL_WIDTH;
	double chord_length_2D=(1/3.0)*((step_fraction*step_fraction*sinf(2*eff_angle_t)-6)/(step_fraction*sinf(2*eff_angle_t)-2*(cosf(eff_angle_t)+sinf(eff_angle_t))) + step_fraction*step_fraction*sinf(2*eff_angle_t)/(2*(cosf(eff_angle_t)+sinf(eff_angle_t))));
	
	// Multiply this by the effective chord in the v-u plane
	double mean_pixel_width=VOXEL_WIDTH/(cosf(eff_angle_t)+sinf(eff_angle_t));
	double height_fraction=VOXEL_THICKNESS/mean_pixel_width;
	step_fraction=MLP_U_STEP/mean_pixel_width;
	double chord_length_3D=(1/3.0)*((step_fraction*step_fraction*sinf(2*eff_angle_v)-6*height_fraction)/(step_fraction*sinf(2*eff_angle_v)-2*(height_fraction*cosf(eff_angle_v)+sinf(eff_angle_v))) + step_fraction*step_fraction*sinf(2*eff_angle_v)/(2*(height_fraction*cosf(eff_angle_v)+sinf(eff_angle_v))));
	
	return VOXEL_WIDTH*chord_length_2D*chord_length_3D;
	 
}
__device__ void find_MLP_path_GPU 
(
	float* x, double b_i, unsigned int first_MLP_voxel_number, double x_in_object, double y_in_object, double z_in_object, 
	double x_out_object, double y_out_object, double z_out_object, double xy_entry_angle, double xz_entry_angle, double xy_exit_angle, double xz_exit_angle,
	double lambda, unsigned int* path, int& number_of_intersections, double& effective_chord_length ,double& a_i_dot_x_k_partially, double& a_i_dot_a_i_partially
) 
{
  
	//bool debug_output = false, MLP_image_output = false;
	//bool constant_chord_lengths = true;
	// MLP calculations variables
	number_of_intersections = 0;
	double u_0 = 0.0, u_1 = MLP_U_STEP,  u_2 = 0.0;
	double T_0[2] = {0.0, 0.0};
	double T_2[2] = {0.0, 0.0};
	double V_0[2] = {0.0, 0.0};
	double V_2[2] = {0.0, 0.0};
	double R_0[4] = { 1.0, 0.0, 0.0 , 1.0}; //a,b,c,d
	double R_1[4] = { 1.0, 0.0, 0.0 , 1.0}; //a,b,c,d
	double R_1T[4] = { 1.0, 0.0, 0.0 , 1.0};  //a,c,b,d

	double sigma_2_pre_1, sigma_2_pre_2, sigma_2_pre_3;
	double sigma_1_coefficient, sigma_t1, sigma_t1_theta1, sigma_theta1, determinant_Sigma_1, Sigma_1I[4];
	double common_sigma_2_term_1, common_sigma_2_term_2, common_sigma_2_term_3;
	double sigma_2_coefficient, sigma_t2, sigma_t2_theta2, sigma_theta2, determinant_Sigma_2, Sigma_2I[4]; 
	double first_term_common_13_1, first_term_common_13_2, first_term_common_24_1, first_term_common_24_2, first_term[4], determinant_first_term;
	double second_term_common_1, second_term_common_2, second_term_common_3, second_term_common_4, second_term[2];
	double t_1, v_1, x_1, y_1, z_1;
	double first_term_inversion_temp;
	
	//double a_i_dot_x_k = 0.0;
	//double a_i_dot_a_i = 0.0;
	
	//double theta_1, phi_1;
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	effective_chord_length = EffectiveChordLength_GPU( ( xy_entry_angle + xy_exit_angle ) / 2.0, ( xz_entry_angle + xz_exit_angle) / 2.0 );
	//double a_j_times_a_j = effective_chord_length * effective_chord_length;
	
	//double effective_chord_length = mean_chord_length( u_in_object, t_in_object, v_in_object, u_out_object, t_out_object, v_out_object );
	//double effective_chord_length = VOXEL_WIDTH;

	int voxel_x = 0, voxel_y = 0, voxel_z = 0;
	
	int voxel = first_MLP_voxel_number;
	
	path[number_of_intersections] = voxel;
	//path[number_of_intersections] = voxel;
	a_i_dot_x_k_partially = x[voxel];
	//a_i_dot_a_i += a_j_times_a_j;
	a_i_dot_a_i_partially = powf(effective_chord_length, 2);
	//atomicAdd(S[voxel], 1);
	//if(!constant_chord_lengths)
		//chord_lengths[num_intersections] = VOXEL_WIDTH;
	number_of_intersections++;
	//MLP_test_image_h[voxel] = 0;

	double u_in_object = ( cos( xy_entry_angle ) * x_in_object ) + ( sin( xy_entry_angle ) * y_in_object );
	double u_out_object = ( cos( xy_entry_angle ) * x_out_object ) + ( sin( xy_entry_angle ) * y_out_object );
	double t_in_object = ( cos( xy_entry_angle ) * y_in_object ) - ( sin( xy_entry_angle ) * x_in_object );
	double t_out_object = ( cos( xy_entry_angle ) * y_out_object ) - ( sin( xy_entry_angle ) * x_out_object );
	double v_in_object = z_in_object;
	double v_out_object = z_out_object;

	if( u_in_object > u_out_object )
	{
		//if( debug_output )
			//cout << "Switched directions" << endl;
		xy_entry_angle += PI;
		xy_exit_angle += PI;
		u_in_object = ( cos( xy_entry_angle ) * x_in_object ) + ( sin( xy_entry_angle ) * y_in_object );
		u_out_object = ( cos( xy_entry_angle ) * x_out_object ) + ( sin( xy_entry_angle ) * y_out_object );
		t_in_object = ( cos( xy_entry_angle ) * y_in_object ) - ( sin( xy_entry_angle ) * x_in_object );
		t_out_object = ( cos( xy_entry_angle ) * y_out_object ) - ( sin( xy_entry_angle ) * x_out_object );
		v_in_object = z_in_object;
		v_out_object = z_out_object;
	}
	T_0[0] = t_in_object;
	T_2[0] = t_out_object;
	T_2[1] = xy_exit_angle - xy_entry_angle;
	V_0[0] = v_in_object;
	V_2[0] = v_out_object;
	V_2[1] = xz_exit_angle - xz_entry_angle;
		
	u_0 = 0.0;
	u_1 = MLP_U_STEP;
	u_2 = abs(u_out_object - u_in_object);		
	//fgets(user_response, sizeof(user_response), stdin);

	//output_file.open(filename);						
				      
	//precalculated u_2 dependent terms (since u_2 does not change inside while loop)
	//u_2 terms
	sigma_2_pre_1 =  u_2*u_2*u_2 * ( A_0_OVER_3 + u_2 * ( A_1_OVER_12 + u_2 * ( A_2_OVER_30 + u_2 * (A_3_OVER_60 + u_2 * ( A_4_OVER_105 + u_2 * A_5_OVER_168 )))));;	//u_2^3 : 1/3, 1/12, 1/30, 1/60, 1/105, 1/168
	sigma_2_pre_2 =  u_2*u_2 * ( A_0_OVER_2 + u_2 * (A_1_OVER_6 + u_2 * (A_2_OVER_12 + u_2 * ( A_3_OVER_20 + u_2 * (A_4_OVER_30 + u_2 * A_5_OVER_42)))));	//u_2^2 : 1/2, 1/6, 1/12, 1/20, 1/30, 1/42
	sigma_2_pre_3 =  u_2 * ( A_0 +  u_2 * (A_1_OVER_2 +  u_2 * ( A_2_OVER_3 +  u_2 * ( A_3_OVER_4 +  u_2 * ( A_4_OVER_5 + u_2 * A_5_OVER_6 )))));			//u_2 : 1/1, 1/2, 1/3, 1/4, 1/5, 1/6

	while( u_1 < u_2 - MLP_U_STEP)
	//while( u_1 < u_2 - 0.001)
	{
		R_0[1] = u_1;
		R_1[1] = u_2 - u_1;
		R_1T[2] = u_2 - u_1;

		sigma_1_coefficient = powf( E_0 * ( 1 + 0.038 * log( (u_1 - u_0)/X0) ), 2.0 ) / X0;
		sigma_t1 =  sigma_1_coefficient * ( powf(u_1, 3.0) * ( A_0_OVER_3 + u_1 * (A_1_OVER_12 + u_1 * (A_2_OVER_30 + u_1 * (A_3_OVER_60 + u_1 * (A_4_OVER_105 + u_1 * A_5_OVER_168 ) )))) );	//u_1^3 : 1/3, 1/12, 1/30, 1/60, 1/105, 1/168
		sigma_t1_theta1 =  sigma_1_coefficient * ( powf(u_1, 2.0) * ( A_0_OVER_2 + u_1 * (A_1_OVER_6 + u_1 * (A_2_OVER_12 + u_1 * (A_3_OVER_20 + u_1 * (A_4_OVER_30 + u_1 * A_5_OVER_42))))) );	//u_1^2 : 1/2, 1/6, 1/12, 1/20, 1/30, 1/42															
		sigma_theta1 = sigma_1_coefficient * ( u_1 * ( A_0 + u_1 * (A_1_OVER_2 + u_1 * (A_2_OVER_3 + u_1 * (A_3_OVER_4 + u_1 * (A_4_OVER_5 + u_1 * A_5_OVER_6))))) );			//u_1 : 1/1, 1/2, 1/3, 1/4, 1/5, 1/6																	
		determinant_Sigma_1 = sigma_t1 * sigma_theta1 - powf( sigma_t1_theta1, 2 );//ad-bc
			
		Sigma_1I[0] = sigma_theta1 / determinant_Sigma_1;
		Sigma_1I[1] = -sigma_t1_theta1 / determinant_Sigma_1;
		Sigma_1I[2] = -sigma_t1_theta1 / determinant_Sigma_1;
		Sigma_1I[3] = sigma_t1 / determinant_Sigma_1;

		//sigma 2 terms
		sigma_2_coefficient = powf( E_0 * ( 1 + 0.038 * log( (u_2 - u_1)/X0) ), 2.0 ) / X0;
		common_sigma_2_term_1 = u_1 * ( A_0 + u_1 * (A_1_OVER_2 + u_1 * (A_2_OVER_3 + u_1 * (A_3_OVER_4 + u_1 * (A_4_OVER_5 + u_1 * A_5_OVER_6 )))));
		common_sigma_2_term_2 = powf(u_1, 2.0) * ( A_0_OVER_2 + u_1 * (A_1_OVER_3 + u_1 * (A_2_OVER_4 + u_1 * (A_3_OVER_5 + u_1 * (A_4_OVER_6 + u_1 * A_5_OVER_7 )))));
		common_sigma_2_term_3 = powf(u_1, 3.0) * ( A_0_OVER_3 + u_1 * (A_1_OVER_4 + u_1 * (A_2_OVER_5 + u_1 * (A_3_OVER_6 + u_1 * (A_4_OVER_7 + u_1 * A_5_OVER_8 )))));
		sigma_t2 =  sigma_2_coefficient * ( sigma_2_pre_1 - powf(u_2, 2.0) * common_sigma_2_term_1 + 2 * u_2 * common_sigma_2_term_2 - common_sigma_2_term_3 );
		sigma_t2_theta2 =  sigma_2_coefficient * ( sigma_2_pre_2 - u_2 * common_sigma_2_term_1 + common_sigma_2_term_2 );
		sigma_theta2 =  sigma_2_coefficient * ( sigma_2_pre_3 - common_sigma_2_term_1 );				
		determinant_Sigma_2 = sigma_t2 * sigma_theta2 - powf( sigma_t2_theta2, 2 );//ad-bc

		Sigma_2I[0] = sigma_theta2 / determinant_Sigma_2;
		Sigma_2I[1] = -sigma_t2_theta2 / determinant_Sigma_2;
		Sigma_2I[2] = -sigma_t2_theta2 / determinant_Sigma_2;
		Sigma_2I[3] = sigma_t2 / determinant_Sigma_2;

		// first_term_common_ij_k: i,j = rows common to, k = 1st/2nd of last 2 terms of 3 term summation in first_term calculation below
		first_term_common_13_1 = Sigma_2I[0] * R_1[0] + Sigma_2I[1] * R_1[2];
		first_term_common_13_2 = Sigma_2I[2] * R_1[0] + Sigma_2I[3] * R_1[2];
		first_term_common_24_1 = Sigma_2I[0] * R_1[1] + Sigma_2I[1] * R_1[3];
		first_term_common_24_2 = Sigma_2I[2] * R_1[1] + Sigma_2I[3] * R_1[3];

		first_term[0] = Sigma_1I[0] + R_1T[0] * first_term_common_13_1 + R_1T[1] * first_term_common_13_2;
		first_term[1] = Sigma_1I[1] + R_1T[0] * first_term_common_24_1 + R_1T[1] * first_term_common_24_2;
		first_term[2] = Sigma_1I[2] + R_1T[2] * first_term_common_13_1 + R_1T[3] * first_term_common_13_2;
		first_term[3] = Sigma_1I[3] + R_1T[2] * first_term_common_24_1 + R_1T[3] * first_term_common_24_2;


		determinant_first_term = first_term[0] * first_term[3] - first_term[1] * first_term[2];
		first_term_inversion_temp = first_term[0];
		first_term[0] = first_term[3] / determinant_first_term;
		first_term[1] = -first_term[1] / determinant_first_term;
		first_term[2] = -first_term[2] / determinant_first_term;
		first_term[3] = first_term_inversion_temp / determinant_first_term;

		// second_term_common_i: i = # of term of 4 term summation it is common to in second_term calculation below
		second_term_common_1 = R_0[0] * T_0[0] + R_0[1] * T_0[1];
		second_term_common_2 = R_0[2] * T_0[0] + R_0[3] * T_0[1];
		second_term_common_3 = Sigma_2I[0] * T_2[0] + Sigma_2I[1] * T_2[1];
		second_term_common_4 = Sigma_2I[2] * T_2[0] + Sigma_2I[3] * T_2[1];

		second_term[0] = Sigma_1I[0] * second_term_common_1 
						+ Sigma_1I[1] * second_term_common_2 
						+ R_1T[0] * second_term_common_3 
						+ R_1T[1] * second_term_common_4;
		second_term[1] = Sigma_1I[2] * second_term_common_1 
						+ Sigma_1I[3] * second_term_common_2 
						+ R_1T[2] * second_term_common_3 
						+ R_1T[3] * second_term_common_4;

		t_1 = first_term[0] * second_term[0] + first_term[1] * second_term[1];
		//cout << "t_1 = " << t_1 << endl;
		//double theta_1 = first_term[2] * second_term[0] + first_term[3] * second_term[1];

		// Do v MLP Now
		second_term_common_1 = R_0[0] * V_0[0] + R_0[1] * V_0[1];
		second_term_common_2 = R_0[2] * V_0[0] + R_0[3] * V_0[1];
		second_term_common_3 = Sigma_2I[0] * V_2[0] + Sigma_2I[1] * V_2[1];
		second_term_common_4 = Sigma_2I[2] * V_2[0] + Sigma_2I[3] * V_2[1];

		second_term[0]	= Sigma_1I[0] * second_term_common_1
						+ Sigma_1I[1] * second_term_common_2
						+ R_1T[0] * second_term_common_3
						+ R_1T[1] * second_term_common_4;
		second_term[1]	= Sigma_1I[2] * second_term_common_1
						+ Sigma_1I[3] * second_term_common_2
						+ R_1T[2] * second_term_common_3
						+ R_1T[3] * second_term_common_4;
		v_1 = first_term[0] * second_term[0] + first_term[1] * second_term[1];
		//double phi_1 = first_term[2] * second_term[0] + first_term[3] * second_term[1];

		// Rotate Coordinate From utv to xyz Coordinate System and Determine Which Voxel this Point on the MLP Path is in
		x_1 = ( cos( xy_entry_angle ) * (u_in_object + u_1) ) - ( sin( xy_entry_angle ) * t_1 );
		y_1 = ( sin( xy_entry_angle ) * (u_in_object + u_1) ) + ( cos( xy_entry_angle ) * t_1 );
		z_1 = v_1;

		voxel_x = calculate_voxel_GPU( X_ZERO_COORDINATE, x_1, VOXEL_WIDTH );
		voxel_y = calculate_voxel_GPU( Y_ZERO_COORDINATE, y_1, VOXEL_HEIGHT );
		voxel_z = calculate_voxel_GPU( Z_ZERO_COORDINATE, z_1, VOXEL_THICKNESS);
				
		voxel = voxel_x + voxel_y * COLUMNS + voxel_z * COLUMNS * ROWS;
		//cout << "voxel_x = " << voxel_x << "voxel_y = " << voxel_y << "voxel_z = " << voxel_z << "voxel = " << voxel <<endl;
		//fgets(user_response, sizeof(user_response), stdin);


		if( voxel != path[number_of_intersections - 1] )
		{
			//path[number_of_intersections] = voxel;
			path[number_of_intersections] = voxel;
			a_i_dot_x_k_partially += x[voxel];
			//a_i_dot_a_i += a_j_times_a_j;
			//atomicAdd(S[voxel], 1);
			//MLP_test_image_h[voxel] = 0;
			//if(!constant_chord_lengths)
				//chord_lengths[num_intersections] = effective_chord_length;						
			number_of_intersections++;
		}
		u_1 += MLP_U_STEP;
	}
	
	
	
	//update_value_history = effective_chord_length * (( b_i - a_i_dot_x_k ) /  a_i_dot_a_i) * lambda;
	
	//return update_value;
	
	
  
}
__device__ void find_MLP_path_GPU_tabulated
(
	float* x, double b_i, unsigned int first_MLP_voxel_number, double x_in_object, double y_in_object, double z_in_object, 
	double x_out_object, double y_out_object, double z_out_object, double xy_entry_angle, double xz_entry_angle, double xy_exit_angle, double xz_exit_angle,
	double lambda, unsigned int* path, int& number_of_intersections, double& effective_chord_length, double& a_i_dot_x_k_partially, double& a_i_dot_a_i_partially,
	double* sin_table, double* cos_table, double* scattering_table, double* poly_1_2, double* poly_2_3, double* poly_3_4, double* poly_2_6, double* poly_3_12
) 
{
	double sigma_t1, sigma_t1_theta1, sigma_theta1, determinant_Sigma_1, Sigma_1I[3];
	double sigma_t2, sigma_t2_theta2, sigma_theta2, determinant_Sigma_2, Sigma_2I[3]; 
	double first_term_common_24_1, first_term[4], determinant_first_term;
	double second_term_common_3, second_term[2];
	double t_1, v_1, x_1, y_1;
	int voxel_x = 0, voxel_y = 0, voxel_z = 0;

	unsigned int trig_table_index =  static_cast<unsigned int>((xy_entry_angle - TRIG_TABLE_MIN ) / TRIG_TABLE_STEP + 0.5);
	double sin_term = sin_table[trig_table_index];
	double cos_term = cos_table[trig_table_index];
	 
	double u_in_object = cos_term * x_in_object + sin_term * y_in_object;
	double u_out_object = cos_term * x_out_object + sin_term * y_out_object;

	if( u_in_object > u_out_object )
	{
		xy_entry_angle += PI;
		xy_exit_angle += PI;
		trig_table_index =  static_cast<unsigned int>((xy_entry_angle - TRIG_TABLE_MIN ) / TRIG_TABLE_STEP + 0.5);
		sin_term = sin_table[trig_table_index];
		cos_term = cos_table[trig_table_index];
		u_in_object = cos_term * x_in_object + sin_term * y_in_object;
		u_out_object = cos_term * x_out_object + sin_term * y_out_object;
	}
	double t_in_object = cos_term * y_in_object  - sin_term * x_in_object;
	double t_out_object = cos_term * y_out_object - sin_term * x_out_object;
	
	double T_2[2] = {t_out_object, xy_exit_angle - xy_entry_angle};
	double V_2[2] = {z_out_object, xz_exit_angle - xz_entry_angle};
	double u_1 = MLP_U_STEP;
	double u_2 = abs(u_out_object - u_in_object);
	double depth_2_go = u_2 - u_1;
	double u_shifted = u_in_object;	
	//unsigned int step_number = 1;

	// Scattering Coefficient tables indices
	unsigned int sigma_table_index_step = static_cast<unsigned int>( MLP_U_STEP / COEFF_TABLE_STEP + 0.5 );
	unsigned int sigma_1_coefficient_index = sigma_table_index_step;
	unsigned int sigma_2_coefficient_index = static_cast<unsigned int>( depth_2_go / COEFF_TABLE_STEP + 0.5 );
	
	// Scattering polynomial indices
	unsigned int poly_table_index_step = static_cast<unsigned int>( MLP_U_STEP / POLY_TABLE_STEP + 0.5 );
	unsigned int u_1_poly_index = poly_table_index_step;
	unsigned int u_2_poly_index = static_cast<unsigned int>( u_2 / POLY_TABLE_STEP + 0.5 );

	//precalculated u_2 dependent terms (since u_2 does not change inside while loop)
	double u_2_poly_3_12 = poly_3_12[u_2_poly_index];
	double u_2_poly_2_6 = poly_2_6[u_2_poly_index];
	double u_2_poly_1_2 = poly_1_2[u_2_poly_index];
	double u_1_poly_1_2, u_1_poly_2_3;

	int voxel = first_MLP_voxel_number;
	number_of_intersections = 0;
	path[number_of_intersections] = voxel;
	number_of_intersections++;

	effective_chord_length = EffectiveChordLength_GPU( ( xy_entry_angle + xy_exit_angle ) / 2.0, ( xz_entry_angle + xz_exit_angle) / 2.0 );
	//double effective_chord_length = EffectiveChordLength_GPU( ( xy_entry_angle + xy_exit_angle ) / 2.0, ( xz_entry_angle + xz_exit_angle) / 2.0 );
	//double a_j_times_a_j = effective_chord_length * effective_chord_length;
	a_i_dot_x_k_partially = x[voxel];
	a_i_dot_a_i_partially = powf(effective_chord_length, 2.0);
	//double a_i_dot_x_k = x[voxel];
	//double a_i_dot_a_i = powf(effective_chord_length, 2.0);
	//double a_i_dot_x_k = x[voxel] * effective_chord_length;
	//double a_i_dot_a_i = a_j_times_a_j;

	//while( u_1 < u_2 - configurations.MLP_U_STEP)
	while( depth_2_go > MLP_U_STEP )
	{
		u_1_poly_1_2 = poly_1_2[u_1_poly_index];
		u_1_poly_2_3 = poly_2_3[u_1_poly_index];

		sigma_t1 = poly_3_12[u_1_poly_index];										// poly_3_12(u_1)
		sigma_t1_theta1 =  poly_2_6[u_1_poly_index];								// poly_2_6(u_1) 
		sigma_theta1 = u_1_poly_1_2;												// poly_1_2(u_1)

		sigma_t2 =  u_2_poly_3_12 - powf(u_2, 2.0) * u_1_poly_1_2 + 2 * u_2 * u_1_poly_2_3 - poly_3_4[u_1_poly_index];	// poly_3_12(u_2) - u_2^2 * poly_1_2(u_1) +2u_2*(u_1) - poly_3_4(u_1)
		sigma_t2_theta2 =  u_2_poly_2_6 - u_2 * u_1_poly_1_2 + u_1_poly_2_3;											// poly_2_6(u_2) - u_2*poly_1_2(u_1) + poly_2_3(u_1)
		sigma_theta2 =  u_2_poly_1_2 - u_1_poly_1_2;																	// poly_1_2(u_2) - poly_1_2(u_1)	

		determinant_Sigma_1 = scattering_table[sigma_1_coefficient_index] * ( sigma_t1 * sigma_theta1 - powf( sigma_t1_theta1, 2 ) );//ad-bc
		Sigma_1I[0] = sigma_theta1 / determinant_Sigma_1;
		Sigma_1I[1] = sigma_t1_theta1 / determinant_Sigma_1;	// negative sign is propagated to subsequent calculations instead of here 
		Sigma_1I[2] = sigma_t1 / determinant_Sigma_1;			
			
		determinant_Sigma_2 = scattering_table[sigma_2_coefficient_index] * ( sigma_t2 * sigma_theta2 - powf( sigma_t2_theta2, 2 ) );//ad-bc
		Sigma_2I[0] = sigma_theta2 / determinant_Sigma_2;
		Sigma_2I[1] = sigma_t2_theta2 / determinant_Sigma_2;	// negative sign is propagated to subsequent calculations instead of here 
		Sigma_2I[2] = sigma_t2 / determinant_Sigma_2;
		/**********************************************************************************************************************************************************/
		first_term_common_24_1 = Sigma_2I[0] * depth_2_go - Sigma_2I[1];
		first_term[0] = Sigma_1I[0] + Sigma_2I[0];
		first_term[1] = first_term_common_24_1 - Sigma_1I[1];
		first_term[2] = depth_2_go * Sigma_2I[0] - Sigma_1I[1] - Sigma_2I[1];
		first_term[3] = Sigma_1I[2] + Sigma_2I[2] + depth_2_go * ( first_term_common_24_1 - Sigma_2I[1]);	
		determinant_first_term = first_term[0] * first_term[3] - first_term[1] * first_term[2];
		
		// Calculate MLP t coordinate
		second_term_common_3 = Sigma_2I[0] * t_out_object - Sigma_2I[1] * T_2[1];	
		second_term[0] = Sigma_1I[0] * t_in_object + second_term_common_3;
		second_term[1] = depth_2_go * second_term_common_3 + Sigma_2I[2] * T_2[1] - Sigma_2I[1] * t_out_object - Sigma_1I[1] * t_in_object;	
		t_1 = ( first_term[3] * second_term[0] - first_term[1] * second_term[1] ) / determinant_first_term ;
		//double theta_1 = first_term[2] * second_term[0] + first_term[3] * second_term[1];
		/**********************************************************************************************************************************************************/
		// Calculate MLP v coordinate
		second_term_common_3 = Sigma_2I[0] * z_out_object - Sigma_2I[1] * V_2[1];
		second_term[0] = Sigma_1I[0] * z_in_object + second_term_common_3;
		second_term[1] = depth_2_go * second_term_common_3 + Sigma_2I[2] * V_2[1] - Sigma_2I[1] * z_out_object - Sigma_1I[1] * z_in_object;
		v_1 = ( first_term[3] * second_term[0] - first_term[1] * second_term[1] ) / determinant_first_term ;
		//double phi_1 = first_term[2] * second_term[0] + first_term[3] * second_term[1];
		/**********************************************************************************************************************************************************/
		// Rotate Coordinate From utv to xyz Coordinate System and Determine Which Voxel this Point on the MLP Path is in
		u_shifted += MLP_U_STEP;
		//u_shifted = u_in_object + u_1;
		x_1 = cos_term * u_shifted - sin_term * t_1;
		y_1 = sin_term * u_shifted + cos_term * t_1;

		voxel_x = calculate_voxel_GPU( X_ZERO_COORDINATE, x_1, VOXEL_WIDTH );
		voxel_y = calculate_voxel_GPU( Y_ZERO_COORDINATE, y_1, VOXEL_HEIGHT );
		voxel_z = calculate_voxel_GPU( Z_ZERO_COORDINATE, v_1, VOXEL_THICKNESS);			
		voxel = voxel_x + voxel_y * COLUMNS + voxel_z * COLUMNS * ROWS;

		//if( voxel != path[num_intersections] )
		//{
		//	path[++num_intersections] = voxel;	
		//	a_i_dot_x_k += x[voxel] * effective_chord_length;
		//	//a_i_dot_x_k += x[voxel];
		//	a_i_dot_a_i += a_j_times_a_j;
		//}
		if( voxel != path[number_of_intersections-1] )
		{
			path[number_of_intersections] = voxel;	
			//a_i_dot_x_k += x[voxel] * effective_chord_length;
			a_i_dot_x_k_partially += x[voxel];
			//a_i_dot_a_i += a_j_times_a_j;
			number_of_intersections++;
		}
		u_1 += MLP_U_STEP;
		depth_2_go -= MLP_U_STEP;
		//step_number++;
		//u_1 = step_number * MLP_U_STEP;
		//depth_2_go = u_2 - u_1;
		sigma_1_coefficient_index += sigma_table_index_step;
		sigma_2_coefficient_index -= sigma_table_index_step;
		u_1_poly_index += poly_table_index_step;
	}
	//++num_intersections;
	//a_i_dot_x_k *= effective_chord_length;
	//a_i_dot_a_i *= num_intersections;
	//update_value_history = effective_chord_length * (( b_i - a_i_dot_x_k ) /  a_i_dot_a_i) * lambda;
}
__global__ void block_update_GPU
(
	float* x, float* x_entry, float* y_entry, float* z_entry, float* xy_entry_angle, float* xz_entry_angle, 
	float* x_exit, float* y_exit, float* z_exit, float* xy_exit_angle, float* xz_exit_angle, float* WEPL, 
	unsigned int* first_MLP_voxel, float* x_update, unsigned int* S, int start_proton_id, int num_histories, double lambda,
	double* sin_table, double* cos_table, double* scattering_table, double* poly_1_2, double* poly_2_3, double* poly_3_4, double* poly_2_6, double* poly_3_12
) 
{	
	int voxel=0, number_of_intersections;
	int proton_id =  start_proton_id + threadIdx.x * HISTORIES_PER_THREAD + blockIdx.x * HISTORIES_PER_BLOCK * HISTORIES_PER_THREAD;
	double a_i_dot_a_i, a_i_dot_x_k, effective_chord_length = 0.0;
		
	if( proton_id < num_histories ) 
	{
	  	unsigned int a_i[MAX_INTERSECTIONS];
		
		for( int history = 0; history < HISTORIES_PER_THREAD; history++ ) 
		{	
			if( proton_id < num_histories ) 
			{		  
				number_of_intersections = 0;
				a_i_dot_a_i = 0.0;
				a_i_dot_x_k = 0.0;	
				//b_i = WEPL[proton_id];		
				#if (MLP_ALGORITHM == TABULATED)
					find_MLP_path_GPU_tabulated
					(
						x, WEPL[proton_id], first_MLP_voxel[proton_id] ,x_entry[proton_id], y_entry[proton_id], z_entry[proton_id], x_exit[proton_id], y_exit[proton_id], z_exit[proton_id], xy_entry_angle[proton_id],
						xz_entry_angle[proton_id], xy_exit_angle[proton_id], xz_exit_angle[proton_id], lambda, a_i, number_of_intersections, effective_chord_length, a_i_dot_x_k, a_i_dot_a_i,
						sin_table, cos_table, scattering_table, poly_1_2, poly_2_3, poly_3_4, poly_2_6, poly_3_12
					);
				#else
					find_MLP_path_GPU(x, WEPL[proton_id], first_MLP_voxel[proton_id] ,x_entry[proton_id], y_entry[proton_id], z_entry[proton_id], x_exit[proton_id], y_exit[proton_id], z_exit[proton_id], xy_entry_angle[proton_id],
						      xz_entry_angle[proton_id], xy_exit_angle[proton_id], xz_exit_angle[proton_id], lambda, a_i, number_of_intersections, effective_chord_length, a_i_dot_x_k, a_i_dot_a_i );
				#endif	      
				a_i_dot_a_i *= number_of_intersections;
				a_i_dot_x_k *= effective_chord_length;					      
				
				if( number_of_intersections >= MIN_MLP_LENGTH )
				{
					// Copy a_i to global
					for (int j = 0 ; j < number_of_intersections; ++j) 
					{
						voxel = a_i[j];
						atomicAdd( &( S[voxel]), 1 );
						#if S_CURVE_ON	
							atomicAdd( &(x_update[voxel]), s_curve_scale_GPU(j, number_of_intersections) * effective_chord_length * ( (WEPL[proton_id] - a_i_dot_x_k) /  a_i_dot_a_i ) * lambda ); 						
						#else
							atomicAdd( &(x_update[voxel]), effective_chord_length * ( (WEPL[proton_id] - a_i_dot_x_k) /  a_i_dot_a_i ) * lambda ); 					
						#endif		
					}	
				}
			}
			proton_id++;
		}	
		free(a_i);    	 
	}	
}
__global__ void calculate_x_update_GPU
(
	float* x, float* x_entry, float* y_entry, float* z_entry, float* xy_entry_angle, float* xz_entry_angle, 
	float* x_exit, float* y_exit, float* z_exit, float* xy_exit_angle, float* xz_exit_angle, float* WEPL, 
	unsigned int* first_MLP_voxel, float* x_update, unsigned int* S, int start_proton_id, int num_histories, double lambda,
	double* sin_table, double* cos_table, double* scattering_table, double* poly_1_2, double* poly_2_3, double* poly_3_4, double* poly_2_6, double* poly_3_12
) 
{	
	int number_of_intersections;
	int proton_id =  start_proton_id + threadIdx.x * HISTORIES_PER_THREAD + blockIdx.x * HISTORIES_PER_BLOCK * HISTORIES_PER_THREAD;
	double a_i_dot_a_i, a_i_dot_x_k, effective_chord_length = 0.0;
		
	if( proton_id < num_histories ) 
	{
	  	unsigned int a_i[MAX_INTERSECTIONS];
		
		for( int history = 0; history < HISTORIES_PER_THREAD; history++ ) 
		{	
			if( proton_id < num_histories ) 
			{		  
				number_of_intersections = 0;
				a_i_dot_a_i = 0.0;
				a_i_dot_x_k = 0.0;	
				//b_i = WEPL[proton_id];		
				#if (MLP_ALGORITHM == TABULATED)
					find_MLP_path_GPU_tabulated
					(
						x, WEPL[proton_id], first_MLP_voxel[proton_id] ,x_entry[proton_id], y_entry[proton_id], z_entry[proton_id], x_exit[proton_id], y_exit[proton_id], z_exit[proton_id], xy_entry_angle[proton_id],
						xz_entry_angle[proton_id], xy_exit_angle[proton_id], xz_exit_angle[proton_id], lambda, a_i, number_of_intersections, effective_chord_length, a_i_dot_x_k, a_i_dot_a_i,
						sin_table, cos_table, scattering_table, poly_1_2, poly_2_3, poly_3_4, poly_2_6, poly_3_12
					);
				#else
					find_MLP_path_GPU(x, WEPL[proton_id], first_MLP_voxel[proton_id] ,x_entry[proton_id], y_entry[proton_id], z_entry[proton_id], x_exit[proton_id], y_exit[proton_id], z_exit[proton_id], xy_entry_angle[proton_id],
						      xz_entry_angle[proton_id], xy_exit_angle[proton_id], xz_exit_angle[proton_id], lambda, a_i, number_of_intersections, effective_chord_length, a_i_dot_x_k, a_i_dot_a_i );
				#endif	      
				a_i_dot_a_i *= number_of_intersections;
				a_i_dot_x_k *= effective_chord_length;					      
				
				if( number_of_intersections >= MIN_MLP_LENGTH )
				{
					//voxel = a_i[j];
					DROP_block_update_GPU(proton_id, number_of_intersections, a_i, S, x_update, WEPL, a_i_dot_x_k, a_i_dot_a_i, lambda, effective_chord_length );
					//// Copy a_i to global
					//for (int j = 0 ; j < number_of_intersections; ++j) 
					//{
					//	voxel = a_i[j];
					//	atomicAdd( &( S[voxel]), 1 );
					//	#if S_CURVE_ON	
					//		atomicAdd( &(x_update[voxel]), s_curve_scale_GPU(j, number_of_intersections) * effective_chord_length * ( (WEPL[proton_id] - a_i_dot_x_k) /  a_i_dot_a_i ) * lambda ); 						
					//	#else
					//		atomicAdd( &(x_update[voxel]), effective_chord_length * ( (WEPL[proton_id] - a_i_dot_x_k) /  a_i_dot_a_i ) * lambda ); 					
					//	#endif		
					//}	
				}
			}
			proton_id++;
		}	
		free(a_i);    	 
	}	
}
__global__ void init_image_GPU(float* x_update, unsigned int* S) 
{
	int column_start = VOXELS_PER_THREAD * blockIdx.x;
	int row = blockIdx.y;
	int slice = threadIdx.x;
	int voxel = column_start  + row * COLUMNS + slice * ROWS * COLUMNS;
	
	for( int shift = 0; shift < VOXELS_PER_THREAD; shift++ ) 
	{	  
		if( voxel < NUM_VOXELS ) 
		{   
			x_update[voxel] = 0.0;
			S[voxel] = 0;			
		}
		voxel++;	 
	 }  
}
__global__ void image_update_GPU (float* x, float* x_update, unsigned int* S) 
{
	int column_start = VOXELS_PER_THREAD * blockIdx.x;
	int row = blockIdx.y;
	int slice = threadIdx.x;
	int voxel = column_start  + row * COLUMNS + slice * ROWS * COLUMNS;
	
	for( int shift = 0; shift < VOXELS_PER_THREAD; shift++ ) 
	{	  
		if( (voxel < NUM_VOXELS) && (S[voxel] > 0) ) 
		{		  
			x[voxel] += x_update[voxel] / S[voxel];
			x_update[voxel] = 0.0;
			S[voxel] = 0;
			#if IDENTIFY_X_N_AIR 
				if(x[voxel] < X_N_AIR_THRESHOLD)
					x[voxel] = 0.0;
			#endif
		}
		voxel++;
	}
}
void x_host_2_GPU() 
{
	  cudaMemcpy( x_d, x_h, SIZE_IMAGE_FLOAT, cudaMemcpyHostToDevice);
}
void x_GPU_2_host() 
{
	  cudaMemcpy( x_h, x_d, SIZE_IMAGE_FLOAT, cudaMemcpyDeviceToHost);
}
/***********************************************************************************************************************************************************************************************************************/
/********************************************************* Feasibility Seeking and Iterative Projection Methods ********************************************************************************************************/
/***********************************************************************************************************************************************************************************************************************/
void DROP_setup_update_arrays()
{
	// Hull is not needed during DROP so free its host/GPU arrays
	//free(hull_h);
	cudaFree(hull_d); 
	
	// Allocate GPU memory for x, hull, x_update, and S
	cudaMalloc( (void**) &x_d, 			NUM_VOXELS *sizeof(float));
	cudaMalloc( (void**) &x_update_d, 	NUM_VOXELS *sizeof(float));
	cudaMalloc( (void**) &S_d, 			NUM_VOXELS *sizeof(unsigned int));
	
	cudaMemcpy( x_d, x_h, NUM_VOXELS * sizeof(float), cudaMemcpyHostToDevice );

	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) 
		printf("DROP_setup_update_arrays Error: %s\n", cudaGetErrorString(cudaStatus));
}
void DROP_free_update_arrays()
{
	// Allocate GPU memory for x, hull, x_update, and S
	cudaFree(x_d); 
	cudaFree(x_update_d); 
	cudaFree(S_d); 

	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) 
		printf("Error: %s\n", cudaGetErrorString(cudaStatus));
}
void DROP_allocations(const int num_histories)
{
	unsigned int size_floats		= sizeof(float) * num_histories;
	unsigned int size_ints			= sizeof(int) * num_histories;
	
	cudaMalloc( (void**) &x_entry_d,			size_floats );
	cudaMalloc( (void**) &y_entry_d,			size_floats );
	cudaMalloc( (void**) &z_entry_d,			size_floats );
	cudaMalloc( (void**) &x_exit_d,				size_floats );
	cudaMalloc( (void**) &y_exit_d,				size_floats );
	cudaMalloc( (void**) &z_exit_d,				size_floats );	
	cudaMalloc( (void**) &xy_entry_angle_d,		size_floats );
	cudaMalloc( (void**) &xz_entry_angle_d,		size_floats );
	cudaMalloc( (void**) &xy_exit_angle_d,		size_floats );
	cudaMalloc( (void**) &xz_exit_angle_d,		size_floats );	
	cudaMalloc( (void**) &WEPL_d,				size_floats );
	cudaMalloc( (void**) &first_MLP_voxel_d, 	size_ints );

	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) 
		printf("DROP_allocations Error: %s\n", cudaGetErrorString(cudaStatus));
}
void DROP_host_2_device(const int start_position,const int num_histories)
{
	unsigned int size_floats		= sizeof(float) * num_histories;
	unsigned int size_ints			= sizeof(int) * num_histories;
	     
	cudaMemcpy( x_entry_d,			&x_entry_vector[start_position],			size_floats,	cudaMemcpyHostToDevice );
	cudaMemcpy( y_entry_d,			&y_entry_vector[start_position],			size_floats,	cudaMemcpyHostToDevice );
	cudaMemcpy( z_entry_d,			&z_entry_vector[start_position],			size_floats,	cudaMemcpyHostToDevice );
	cudaMemcpy( x_exit_d,			&x_exit_vector[start_position],				size_floats,	cudaMemcpyHostToDevice );
	cudaMemcpy( y_exit_d,			&y_exit_vector[start_position],				size_floats,	cudaMemcpyHostToDevice );
	cudaMemcpy( z_exit_d,			&z_exit_vector[start_position],				size_floats,	cudaMemcpyHostToDevice );
	cudaMemcpy( xy_entry_angle_d,	&xy_entry_angle_vector[start_position],		size_floats,	cudaMemcpyHostToDevice );
	cudaMemcpy( xz_entry_angle_d,	&xz_entry_angle_vector[start_position],		size_floats,	cudaMemcpyHostToDevice );
	cudaMemcpy( xy_exit_angle_d,	&xy_exit_angle_vector[start_position],		size_floats,	cudaMemcpyHostToDevice );
	cudaMemcpy( xz_exit_angle_d,	&xz_exit_angle_vector[start_position],		size_floats,	cudaMemcpyHostToDevice );
	cudaMemcpy( WEPL_d,				&WEPL_vector[start_position],				size_floats,	cudaMemcpyHostToDevice );
	cudaMemcpy( first_MLP_voxel_d,	&first_MLP_voxel_vector[start_position],	size_ints,		cudaMemcpyHostToDevice );

	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) 
		printf("DROP_host_2_device Error: %s\n", cudaGetErrorString(cudaStatus));
}
void DROP_deallocations()
{	
	cudaFree(x_entry_d);
	cudaFree(y_entry_d);
	cudaFree(z_entry_d);
	cudaFree(x_exit_d);
	cudaFree(y_exit_d);
	cudaFree(z_exit_d);
	cudaFree(xy_entry_angle_d);
	cudaFree(xz_entry_angle_d);
	cudaFree(xy_exit_angle_d);
	cudaFree(xz_exit_angle_d);
	cudaFree(WEPL_d);
	cudaFree(first_MLP_voxel_d);

	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) 
		printf("DROP_deallocations Error: %s\n", cudaGetErrorString(cudaStatus));
}
__device__ void DROP_block_update_GPU(int proton_id, int number_of_intersections, unsigned int* a_i, unsigned int* S, float* x_update, float* WEPL, double a_i_dot_x_k, double a_i_dot_a_i, double lambda, double effective_chord_length )
{
	uint voxel;
	// Copy a_i to global
	for (int j = 0 ; j < number_of_intersections; ++j) 
	{
		voxel = a_i[j];
		atomicAdd( &( S[voxel]), 1 );
		#if S_CURVE_ON	
			atomicAdd( &(x_update[voxel]), s_curve_scale_GPU(j, number_of_intersections) * effective_chord_length * ( (WEPL[proton_id] - a_i_dot_x_k) /  a_i_dot_a_i ) * lambda ); 						
		#else
			atomicAdd( &(x_update[voxel]), effective_chord_length * ( (WEPL[proton_id] - a_i_dot_x_k) /  a_i_dot_a_i ) * lambda ); 					
		#endif		
	}	
}
void DROP_full_tx_iteration(const int num_histories, const int iteration)	
{
	// RECON_TX_MODE = FULL_TX, MLP_ALGORITHM = TABULATED
	cudaError_t cudaStatus;
	char iteration_string[256];
	int remaining_histories = num_histories, start_position = 0, histories_2_process, num_blocks;
	int column_blocks = static_cast<int>(COLUMNS/VOXELS_PER_THREAD);
	dim3 dimBlock( SLICES );
	dim3 dimGrid( column_blocks, ROWS );

	sprintf(iteration_string, "for DROP iteration %d", iteration);		
	timer( START, begin_DROP_iteration, iteration_string);	
	while( remaining_histories > 0 )
	{
		// Proceed using DROP_BLOCK_SIZE histories or all remaining histories if this is less than DROP_BLOCK_SIZE
		if( remaining_histories > DROP_BLOCK_SIZE )
			histories_2_process = DROP_BLOCK_SIZE;
		else
			histories_2_process = remaining_histories;	
		// Set GPU grid/block configuration and perform DROP update calculations		
		num_blocks = static_cast<unsigned int>( (histories_2_process - 1 + HISTORIES_PER_BLOCK*HISTORIES_PER_THREAD) / (HISTORIES_PER_BLOCK*HISTORIES_PER_THREAD) );  
		calculate_x_update_GPU<<< num_blocks, HISTORIES_PER_BLOCK >>>
		( 
			x_d, x_entry_d, y_entry_d, z_entry_d, xy_entry_angle_d, xz_entry_angle_d, x_exit_d, y_exit_d, z_exit_d,  xy_exit_angle_d, 
			xz_exit_angle_d, WEPL_d, first_MLP_voxel_d, x_update_d, S_d, start_position, num_histories, LAMBDA,
			sin_table_d, cos_table_d, scattering_table_d, poly_1_2_d, poly_2_3_d, poly_3_4_d, poly_2_6_d, poly_3_12_d
		);	
		/*block_update_GPU<<< num_blocks, HISTORIES_PER_BLOCK >>>
		( 
			x_d, x_entry_d, y_entry_d, z_entry_d, xy_entry_angle_d, xz_entry_angle_d, x_exit_d, y_exit_d, z_exit_d,  xy_exit_angle_d, 
			xz_exit_angle_d, WEPL_d, first_MLP_voxel_d, x_update_d, S_d, start_position, num_histories, LAMBDA,
			sin_table_d, cos_table_d, scattering_table_d, poly_1_2_d, poly_2_3_d, poly_3_4_d, poly_2_6_d, poly_3_12_d
		);	*/
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) 
			printf("DROP_block_update_GPU Error: %s\n", cudaGetErrorString(cudaStatus));
		
		// Apply DROP update to image
		image_update_GPU<<< dimGrid, dimBlock >>>( x_d, x_update_d, S_d );
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) 
			printf("image_update_GPU Error: %s\n", cudaGetErrorString(cudaStatus));
		remaining_histories -= DROP_BLOCK_SIZE;		
		start_position		+= DROP_BLOCK_SIZE;		
	}// end: while( remaining_histories > 0)		
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching %s Kernel!\n", cudaStatus, "here");

	execution_time_DROP_iteration = timer( STOP, begin_DROP_iteration, iteration_string);	
	execution_times_DROP_iterations.push_back(execution_time_DROP_iteration);		
}
void DROP_partial_tx_iteration( const int num_histories, const int iteration) 
{ 
	// RECON_TX_MODE = PARTIAL_TX, MLP_ALGORITHM = TABULATED
	cudaError_t cudaStatus;
	char iteration_string[256];
	int remaining_histories  = num_histories, start_position = 0, histories_2_process, num_blocks;
	int column_blocks = static_cast<int>(COLUMNS/VOXELS_PER_THREAD);
	dim3 dimBlock( SLICES );
	dim3 dimGrid( column_blocks, ROWS );

	sprintf(iteration_string, "for DROP iteration %d", iteration);		
	timer( START, begin_DROP_iteration, iteration_string);	
	while( remaining_histories > 0 )
	{
		if( remaining_histories > DROP_BLOCK_SIZE )
			histories_2_process = DROP_BLOCK_SIZE;
		else
			histories_2_process = remaining_histories;	
			
		DROP_allocations(histories_2_process);
		DROP_host_2_device( start_position, histories_2_process);
	
		num_blocks = static_cast<int>( (histories_2_process - 1 + HISTORIES_PER_BLOCK*HISTORIES_PER_THREAD) / (HISTORIES_PER_BLOCK*HISTORIES_PER_THREAD) );  
		calculate_x_update_GPU<<< num_blocks, HISTORIES_PER_BLOCK >>>
		( 
			x_d, x_entry_d, y_entry_d, z_entry_d, xy_entry_angle_d, xz_entry_angle_d, x_exit_d, y_exit_d, z_exit_d,  xy_exit_angle_d, 
			xz_exit_angle_d, WEPL_d, first_MLP_voxel_d, x_update_d, S_d, start_position, num_histories, LAMBDA,
			sin_table_d, cos_table_d, scattering_table_d, poly_1_2_d, poly_2_3_d, poly_3_4_d, poly_2_6_d, poly_3_12_d
		);		
	/*	block_update_GPU<<< num_blocks, HISTORIES_PER_BLOCK >>>
		( 
			x_d, x_entry_d, y_entry_d, z_entry_d, xy_entry_angle_d, xz_entry_angle_d, x_exit_d, y_exit_d, z_exit_d,  xy_exit_angle_d, 
			xz_exit_angle_d, WEPL_d, first_MLP_voxel_d, x_update_d, S_d, start_position, num_histories, LAMBDA,
			sin_table_d, cos_table_d, scattering_table_d, poly_1_2_d, poly_2_3_d, poly_3_4_d, poly_2_6_d, poly_3_12_d
		);		*/
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) 
			printf("DROP_block_update_GPU Error: %s\n", cudaGetErrorString(cudaStatus));

		DROP_deallocations();

		image_update_GPU<<< dimGrid, dimBlock >>>( x_d, x_update_d, S_d );

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) 
			printf("image_update_GPU Error: %s\n", cudaGetErrorString(cudaStatus));

		remaining_histories -= DROP_BLOCK_SIZE;
		start_position		+= DROP_BLOCK_SIZE;
	}
	execution_time_DROP_iteration = timer( STOP, begin_DROP_iteration, iteration_string);	
	execution_times_DROP_iterations.push_back(execution_time_DROP_iteration);		
}
void DROP_partial_tx_preallocated_iteration( const int num_histories, const int iteration) 
{
	// RECON_TX_MODE = PARTIAL_TX_PREALLOCATED, MLP_ALGORITHM = TABULATED
	cudaError_t cudaStatus;
	char iteration_string[256];
	int remaining_histories = num_histories, start_position = 0, histories_2_process, num_blocks;
	int column_blocks = static_cast<int>(COLUMNS/VOXELS_PER_THREAD);
	dim3 dimBlock( SLICES );
	dim3 dimGrid( column_blocks, ROWS );

	sprintf(iteration_string, "for DROP iteration %d", iteration);		
	timer( START, begin_DROP_iteration, iteration_string);	
	while( remaining_histories > 0 )
	{
		if( remaining_histories > DROP_BLOCK_SIZE )
			histories_2_process = DROP_BLOCK_SIZE;
		else
			histories_2_process = remaining_histories;	
			
		DROP_host_2_device( start_position, histories_2_process);
	
		num_blocks = static_cast<int>( (histories_2_process - 1 + HISTORIES_PER_BLOCK*HISTORIES_PER_THREAD) / (HISTORIES_PER_BLOCK*HISTORIES_PER_THREAD) );  
		calculate_x_update_GPU<<< num_blocks, HISTORIES_PER_BLOCK >>>
		( 
			x_d, x_entry_d, y_entry_d, z_entry_d, xy_entry_angle_d, xz_entry_angle_d, x_exit_d, y_exit_d, z_exit_d,  xy_exit_angle_d, 
			xz_exit_angle_d, WEPL_d, first_MLP_voxel_d, x_update_d, S_d, start_position, num_histories, LAMBDA,
			sin_table_d, cos_table_d, scattering_table_d, poly_1_2_d, poly_2_3_d, poly_3_4_d, poly_2_6_d, poly_3_12_d
		);	
		/*block_update_GPU<<< num_blocks, HISTORIES_PER_BLOCK >>>
		( 
			x_d, x_entry_d, y_entry_d, z_entry_d, xy_entry_angle_d, xz_entry_angle_d, x_exit_d, y_exit_d, z_exit_d,  xy_exit_angle_d, 
			xz_exit_angle_d, WEPL_d, first_MLP_voxel_d, x_update_d, S_d, start_position, num_histories, LAMBDA,
			sin_table_d, cos_table_d, scattering_table_d, poly_1_2_d, poly_2_3_d, poly_3_4_d, poly_2_6_d, poly_3_12_d
		);	*/
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) 
			printf("DROP_block_update_GPU Error: %s\n", cudaGetErrorString(cudaStatus));

		image_update_GPU<<< dimGrid, dimBlock >>>( x_d, x_update_d, S_d );

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) 
			printf("image_update_GPU Error: %s\n", cudaGetErrorString(cudaStatus));

		remaining_histories -= DROP_BLOCK_SIZE;
		start_position		+= DROP_BLOCK_SIZE;
	}
	execution_time_DROP_iteration = timer( STOP, begin_DROP_iteration, iteration_string);	
	execution_times_DROP_iterations.push_back(execution_time_DROP_iteration);		
}
void DROP_full_tx_iteration(const int num_histories, const int iteration, double relaxation_parameter)	
{
	// RECON_TX_MODE = FULL_TX, MLP_ALGORITHM = TABULATED
	cudaError_t cudaStatus;
	char iteration_string[256];
	int remaining_histories = num_histories, start_position = 0, histories_2_process, num_blocks;
	int column_blocks = static_cast<int>(COLUMNS/VOXELS_PER_THREAD);
	dim3 dimBlock( SLICES );
	dim3 dimGrid( column_blocks, ROWS );

	sprintf(iteration_string, "for DROP iteration %d", iteration);		
	timer( START, begin_DROP_iteration, iteration_string);	
	while( remaining_histories > 0 )
	{
		// Proceed using DROP_BLOCK_SIZE histories or all remaining histories if this is less than DROP_BLOCK_SIZE
		if( remaining_histories > DROP_BLOCK_SIZE )
			histories_2_process = DROP_BLOCK_SIZE;
		else
			histories_2_process = remaining_histories;	
		// Set GPU grid/block configuration and perform DROP update calculations		
		num_blocks = static_cast<unsigned int>( (histories_2_process - 1 + HISTORIES_PER_BLOCK*HISTORIES_PER_THREAD) / (HISTORIES_PER_BLOCK*HISTORIES_PER_THREAD) );  
		calculate_x_update_GPU<<< num_blocks, HISTORIES_PER_BLOCK >>>
		( 
			x_d, x_entry_d, y_entry_d, z_entry_d, xy_entry_angle_d, xz_entry_angle_d, x_exit_d, y_exit_d, z_exit_d,  xy_exit_angle_d, 
			xz_exit_angle_d, WEPL_d, first_MLP_voxel_d, x_update_d, S_d, start_position, num_histories, relaxation_parameter,
			sin_table_d, cos_table_d, scattering_table_d, poly_1_2_d, poly_2_3_d, poly_3_4_d, poly_2_6_d, poly_3_12_d
		);	
		/*block_update_GPU<<< num_blocks, HISTORIES_PER_BLOCK >>>
		( 
			x_d, x_entry_d, y_entry_d, z_entry_d, xy_entry_angle_d, xz_entry_angle_d, x_exit_d, y_exit_d, z_exit_d,  xy_exit_angle_d, 
			xz_exit_angle_d, WEPL_d, first_MLP_voxel_d, x_update_d, S_d, start_position, num_histories, LAMBDA,
			sin_table_d, cos_table_d, scattering_table_d, poly_1_2_d, poly_2_3_d, poly_3_4_d, poly_2_6_d, poly_3_12_d
		);	*/
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) 
			printf("DROP_block_update_GPU Error: %s\n", cudaGetErrorString(cudaStatus));
		
		// Apply DROP update to image
		image_update_GPU<<< dimGrid, dimBlock >>>( x_d, x_update_d, S_d );
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) 
			printf("image_update_GPU Error: %s\n", cudaGetErrorString(cudaStatus));
		remaining_histories -= DROP_BLOCK_SIZE;		
		start_position		+= DROP_BLOCK_SIZE;		
	}// end: while( remaining_histories > 0)		
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching %s Kernel!\n", cudaStatus, "here");

	execution_time_DROP_iteration = timer( STOP, begin_DROP_iteration, iteration_string);	
	execution_times_DROP_iterations.push_back(execution_time_DROP_iteration);		
}
void DROP_partial_tx_iteration( const int num_histories, const int iteration, double relaxation_parameter) 
{ 
	// RECON_TX_MODE = PARTIAL_TX, MLP_ALGORITHM = TABULATED
	cudaError_t cudaStatus;
	char iteration_string[256];
	int remaining_histories  = num_histories, start_position = 0, histories_2_process, num_blocks;
	int column_blocks = static_cast<int>(COLUMNS/VOXELS_PER_THREAD);
	dim3 dimBlock( SLICES );
	dim3 dimGrid( column_blocks, ROWS );

	sprintf(iteration_string, "for DROP iteration %d", iteration);		
	timer( START, begin_DROP_iteration, iteration_string);	
	while( remaining_histories > 0 )
	{
		if( remaining_histories > DROP_BLOCK_SIZE )
			histories_2_process = DROP_BLOCK_SIZE;
		else
			histories_2_process = remaining_histories;	
			
		DROP_allocations(histories_2_process);
		DROP_host_2_device( start_position, histories_2_process);
	
		num_blocks = static_cast<int>( (histories_2_process - 1 + HISTORIES_PER_BLOCK*HISTORIES_PER_THREAD) / (HISTORIES_PER_BLOCK*HISTORIES_PER_THREAD) );  
		calculate_x_update_GPU<<< num_blocks, HISTORIES_PER_BLOCK >>>
		( 
			x_d, x_entry_d, y_entry_d, z_entry_d, xy_entry_angle_d, xz_entry_angle_d, x_exit_d, y_exit_d, z_exit_d,  xy_exit_angle_d, 
			xz_exit_angle_d, WEPL_d, first_MLP_voxel_d, x_update_d, S_d, start_position, num_histories, relaxation_parameter,
			sin_table_d, cos_table_d, scattering_table_d, poly_1_2_d, poly_2_3_d, poly_3_4_d, poly_2_6_d, poly_3_12_d
		);		
	/*	block_update_GPU<<< num_blocks, HISTORIES_PER_BLOCK >>>
		( 
			x_d, x_entry_d, y_entry_d, z_entry_d, xy_entry_angle_d, xz_entry_angle_d, x_exit_d, y_exit_d, z_exit_d,  xy_exit_angle_d, 
			xz_exit_angle_d, WEPL_d, first_MLP_voxel_d, x_update_d, S_d, start_position, num_histories, LAMBDA,
			sin_table_d, cos_table_d, scattering_table_d, poly_1_2_d, poly_2_3_d, poly_3_4_d, poly_2_6_d, poly_3_12_d
		);		*/
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) 
			printf("DROP_block_update_GPU Error: %s\n", cudaGetErrorString(cudaStatus));

		DROP_deallocations();

		image_update_GPU<<< dimGrid, dimBlock >>>( x_d, x_update_d, S_d );

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) 
			printf("image_update_GPU Error: %s\n", cudaGetErrorString(cudaStatus));

		remaining_histories -= DROP_BLOCK_SIZE;
		start_position		+= DROP_BLOCK_SIZE;
	}
	execution_time_DROP_iteration = timer( STOP, begin_DROP_iteration, iteration_string);	
	execution_times_DROP_iterations.push_back(execution_time_DROP_iteration);		
}
void DROP_partial_tx_preallocated_iteration( const int num_histories, const int iteration, double relaxation_parameter) 
{
	// RECON_TX_MODE = PARTIAL_TX_PREALLOCATED, MLP_ALGORITHM = TABULATED
	cudaError_t cudaStatus;
	char iteration_string[256];
	int remaining_histories = num_histories, start_position = 0, histories_2_process, num_blocks;
	int column_blocks = static_cast<int>(COLUMNS/VOXELS_PER_THREAD);
	dim3 dimBlock( SLICES );
	dim3 dimGrid( column_blocks, ROWS );

	sprintf(iteration_string, "for DROP iteration %d", iteration);		
	timer( START, begin_DROP_iteration, iteration_string);	
	while( remaining_histories > 0 )
	{
		if( remaining_histories > DROP_BLOCK_SIZE )
			histories_2_process = DROP_BLOCK_SIZE;
		else
			histories_2_process = remaining_histories;	
			
		DROP_host_2_device( start_position, histories_2_process);
	
		num_blocks = static_cast<int>( (histories_2_process - 1 + HISTORIES_PER_BLOCK*HISTORIES_PER_THREAD) / (HISTORIES_PER_BLOCK*HISTORIES_PER_THREAD) );  
		calculate_x_update_GPU<<< num_blocks, HISTORIES_PER_BLOCK >>>
		( 
			x_d, x_entry_d, y_entry_d, z_entry_d, xy_entry_angle_d, xz_entry_angle_d, x_exit_d, y_exit_d, z_exit_d,  xy_exit_angle_d, 
			xz_exit_angle_d, WEPL_d, first_MLP_voxel_d, x_update_d, S_d, start_position, num_histories, relaxation_parameter,
			sin_table_d, cos_table_d, scattering_table_d, poly_1_2_d, poly_2_3_d, poly_3_4_d, poly_2_6_d, poly_3_12_d
		);	
		/*block_update_GPU<<< num_blocks, HISTORIES_PER_BLOCK >>>
		( 
			x_d, x_entry_d, y_entry_d, z_entry_d, xy_entry_angle_d, xz_entry_angle_d, x_exit_d, y_exit_d, z_exit_d,  xy_exit_angle_d, 
			xz_exit_angle_d, WEPL_d, first_MLP_voxel_d, x_update_d, S_d, start_position, num_histories, LAMBDA,
			sin_table_d, cos_table_d, scattering_table_d, poly_1_2_d, poly_2_3_d, poly_3_4_d, poly_2_6_d, poly_3_12_d
		);	*/
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) 
			printf("DROP_block_update_GPU Error: %s\n", cudaGetErrorString(cudaStatus));

		image_update_GPU<<< dimGrid, dimBlock >>>( x_d, x_update_d, S_d );

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) 
			printf("image_update_GPU Error: %s\n", cudaGetErrorString(cudaStatus));

		remaining_histories -= DROP_BLOCK_SIZE;
		start_position		+= DROP_BLOCK_SIZE;
	}
	execution_time_DROP_iteration = timer( STOP, begin_DROP_iteration, iteration_string);	
	execution_times_DROP_iterations.push_back(execution_time_DROP_iteration);		
}
void DROP_GPU(const unsigned int num_histories)	
{
	// RECON_TX_MODE = FULL_TX, MLP_ALGORITHM = TABULATED
	char iterate_filename[256];
	//char fileNamePNG[512];
	unsigned int start_position = 0;
	unsigned int column_blocks = static_cast<unsigned int>( COLUMNS / VOXELS_PER_THREAD );
	dim3 dimBlock( SLICES );
	dim3 dimGrid( column_blocks, ROWS );	
	// Host and GPU array allocations and host->GPU transfers/initializations for DROP and TVS
	timer( START, begin_DROP, "for all iterations of DROP");	
	setup_MLP_lookup_tables();
	#if (DROP_TX_MODE==FULL_TX)
		DROP_allocations(num_histories);
		DROP_host_2_device( start_position, num_histories);
	#elif (DROP_TX_MODE==PARTIAL_TX_PREALLOCATED)
		DROP_allocations(DROP_BLOCK_SIZE);
	#elif (DROP_TX_MODE==PARTIAL_TX)
		//DROP_partial_tx_iteration(num_histories, iteration);
	#endif 
	#if TVS_ON	
		allocate_perturbation_arrays(false);
		generate_TVS_eta_sequence();
		x_TVS_h = (float*)calloc(NUM_VOXELS, sizeof(float));	
	#endif 

	sprintf(print_statement, "Performing reconstruction with TVS repeated %d times before each DROP iteration and writing output data/images to:", TVS_REPETITIONS);
	print_colored_text(print_statement, CYAN_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );
	print_colored_text(OUTPUT_FOLDER_UNIQUE, LIGHT_PURPLE_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT );		
	for(unsigned int iteration = 1; iteration <= ITERATIONS ; ++iteration) 
	{	    
		sprintf(iterate_filename, "%s%d", "x_", iteration );
		sprintf(print_statement, "Performing iteration %u of image reconstruction...", iteration);
		print_section_header(print_statement, MINOR_SECTION_SEPARATOR, LIGHT_BLUE_TEXT, YELLOW_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT );
		#if TVS_ON && TVS_FIRST
			#if TVS_OLD
				TVS_REPETITIONS = 1;
			#endif
			NTVS_iteration(iteration);	
			x_host_2_GPU();									// Transfer perturbed image back to GPU for update	
		#endif
		// Transfer data for ALL reconstruction_histories before beginning image reconstruction, using the MLP lookup tables each time
		#if (DROP_TX_MODE==FULL_TX)
			DROP_full_tx_iteration(num_histories, iteration);
			print_colored_text("Transferring iterate to host", CYAN_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );
			x_GPU_2_host();
			// Transfer data to GPU as needed and allocate/free the corresponding GPU arrays each kernel launch, using the MLP lookup tables each time
		#elif (DROP_TX_MODE==PARTIAL_TX_PREALLOCATED)
			DROP_partial_tx_preallocated_iteration(num_histories, iteration);
		// Transfer data to GPU as needed but allocate and resuse the GPU arrays each kernel launch, using the MLP lookup tables each time
		#elif (DROP_TX_MODE==PARTIAL_TX)
			DROP_partial_tx_iteration(num_histories, iteration);
		#endif 
		#if TVS_ON && !TVS_FIRST
			#if TVS_OLD
				TVS_REPETITIONS = 1;
			#endif
			NTVS_iteration(iteration);		
		#endif		
		// Transfer the updated image to the host and write it to disk
		if( WRITE_X_KI ) 
		{			
			print_colored_text("Writing iterate to disk", CYAN_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );
			array_2_disk(iterate_filename, OUTPUT_DIRECTORY, OUTPUT_FOLDER_UNIQUE, x_h, COLUMNS, ROWS, SLICES, NUM_VOXELS, true ); 
			// Print the image to a binary file
			write_PNG(iterate_filename, x_h);
			print_colored_text("Finished disk write", CYAN_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );
		}
	}// end: for( unsigned int iteration = 1; iteration < iterations; iteration++)	
	#if (DROP_TX_MODE==FULL_TX)
		DROP_deallocations();
	#elif (DROP_TX_MODE==PARTIAL_TX_PREALLOCATED)
		DROP_deallocations();
	#elif (DROP_TX_MODE==PARTIAL_TX)
		//DROP_partial_tx_iteration(num_histories, iteration);
	#endif 
	#if TVS_ON	
		deallocate_perturbation_arrays(false);
	#endif 	
	#if (MLP_ALGORITHM == TABULATED)
		free_MLP_lookup_tables();	
	#endif 
	execution_time_DROP = timer( STOP, begin_DROP, "for all iterations of DROP");
}
void write_PNG(const char* filename, float* image)
{
	char fileNamePNG[512];
	char command[512];
	char path[512];	
	//print_colored_text("Writing binary image file to", CYAN_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );
	//print_colored_text(path, LIGHT_PURPLE_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT );			
	sprintf(print_statement, "Writing %s.png to disk...", filename);
	print_colored_text(print_statement, CYAN_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );
	
	sprintf(path, "%s%s//%s.dat", OUTPUT_DIRECTORY, OUTPUT_FOLDER_UNIQUE, filename);
	FILE *imageFile = fopen(path, "wb");
	//fwrite(image, sizeof(float), NUM_VOXELS, imageFile);
	float pixel_value;
	for(int k=0;k<SLICES;k++)
	{
		for(int m=0;m<ROWS;m++)		
		{
			for(int n=0;n<COLUMNS;n++)
			{
				pixel_value = image[(k*ROWS*COLUMNS)+(m*COLUMNS)+n]/2;
				fwrite(&pixel_value, sizeof(pixel_value), 1, imageFile);
			}
		}
	}
	fclose(imageFile);
	
	// Convert the binary file to png, using imagemagick
	sprintf(fileNamePNG, "%s%s//%s.png", OUTPUT_DIRECTORY, OUTPUT_FOLDER_UNIQUE, filename);
	//print_colored_text("Writing PNG image file to", CYAN_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );
	//print_colored_text(fileNamePNG, LIGHT_PURPLE_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT );		
	//sprintf(path, "%s%s//%s.png", OUTPUT_DIRECTORY, OUTPUT_FOLDER_UNIQUE, filename);
	sprintf(command, "convert -define quantum:format=floating-point -depth 32 -size %dx%d gray:%s %s", COLUMNS, ROWS*SLICES, path, fileNamePNG);
	system(command);	
}
void DROP_GPU(const unsigned int num_histories, const int iterations, double relaxation_parameter)	
{
	// RECON_TX_MODE = FULL_TX, MLP_ALGORITHM = TABULATED
	char iterate_filename[256];
	unsigned int start_position = 0;
	unsigned int column_blocks = static_cast<unsigned int>( COLUMNS / VOXELS_PER_THREAD );
	dim3 dimBlock( SLICES );
	dim3 dimGrid( column_blocks, ROWS );	
	// Host and GPU array allocations and host->GPU transfers/initializations for DROP and TVS
	timer( START, begin_DROP, "for all iterations of DROP");	
	setup_MLP_lookup_tables();
	#if (DROP_TX_MODE==FULL_TX)
		DROP_allocations(num_histories);
		DROP_host_2_device( start_position, num_histories);
	#elif (DROP_TX_MODE==PARTIAL_TX_PREALLOCATED)
		DROP_allocations(DROP_BLOCK_SIZE);
	#elif (DROP_TX_MODE==PARTIAL_TX)
		//DROP_partial_tx_iteration(num_histories, iteration);
	#endif 
	#if TVS_ON	
		allocate_perturbation_arrays(false);
		generate_TVS_eta_sequence();
		x_TVS_h = (float*)calloc(NUM_VOXELS, sizeof(float));	
	#endif 

	sprintf(print_statement, "Performing reconstruction with TVS repeated %d times before each DROP iteration and writing output data/images to:", TVS_REPETITIONS);
	print_colored_text(print_statement, CYAN_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );
	print_colored_text(OUTPUT_FOLDER_UNIQUE, LIGHT_PURPLE_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT );		
	for(unsigned int iteration = 1; iteration <= iterations ; ++iteration) 
	{	    
		sprintf(iterate_filename, "%s%d", "x_", iteration );
		sprintf(print_statement, "Performing iteration %u of image reconstruction...", iteration);
		print_section_header(print_statement, MINOR_SECTION_SEPARATOR, LIGHT_BLUE_TEXT, YELLOW_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT );
		#if TVS_ON && TVS_FIRST
			#if TVS_OLD
				TVS_REPETITIONS = 1;
			#endif
			NTVS_iteration(iteration);		
		#endif
		// Transfer data for ALL reconstruction_histories before beginning image reconstruction, using the MLP lookup tables each time
		#if (DROP_TX_MODE==FULL_TX)
			DROP_full_tx_iteration(num_histories, iteration, relaxation_parameter);
		// Transfer data to GPU as needed and allocate/free the corresponding GPU arrays each kernel launch, using the MLP lookup tables each time
		#elif (DROP_TX_MODE==PARTIAL_TX_PREALLOCATED)
			DROP_partial_tx_preallocated_iteration(num_histories, iteration, relaxation_parameter);
		// Transfer data to GPU as needed but allocate and resuse the GPU arrays each kernel launch, using the MLP lookup tables each time
		#elif (DROP_TX_MODE==PARTIAL_TX)
			DROP_partial_tx_iteration(num_histories, iteration, relaxation_parameter);
		#endif 
		#if TVS_ON && !TVS_FIRST
			#if TVS_OLD
				TVS_REPETITIONS = 1;
			#endif
			NTVS_iteration(iteration);		
		#endif		
		// Transfer the updated image to the host and write it to disk
		if( WRITE_X_KI ) 
		{
			print_colored_text("Transferring iterate to host", CYAN_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );
			x_GPU_2_host();
			print_colored_text("Writing iterate to disk", CYAN_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );
			array_2_disk(iterate_filename, OUTPUT_DIRECTORY, OUTPUT_FOLDER_UNIQUE, x_h, COLUMNS, ROWS, SLICES, NUM_VOXELS, true ); 
			print_colored_text("Finished disk write", CYAN_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );
	
		}
	}// end: for( unsigned int iteration = 1; iteration < iterations; iteration++)	
	#if (DROP_TX_MODE==FULL_TX)
		DROP_deallocations();
	#elif (DROP_TX_MODE==PARTIAL_TX_PREALLOCATED)
		DROP_deallocations();
	#elif (DROP_TX_MODE==PARTIAL_TX)
		//DROP_partial_tx_iteration(num_histories, iteration);
	#endif 
	#if TVS_ON	
		deallocate_perturbation_arrays(false);
	#endif 	
	#if (MLP_ALGORITHM == TABULATED)
		free_MLP_lookup_tables();	
	#endif 
	DROP_free_update_arrays();
	execution_time_DROP = timer( STOP, begin_DROP, "for all iterations of DROP");
}
void DROP_GPU_PCD(const unsigned int num_histories, const int iterations, double relaxation_parameter)	
{
	// RECON_TX_MODE = FULL_TX, MLP_ALGORITHM = TABULATED
	char iterate_filename[256];
	unsigned int start_position = 0;
	unsigned int column_blocks = static_cast<unsigned int>( COLUMNS / VOXELS_PER_THREAD );
	dim3 dimBlock( SLICES );
	dim3 dimGrid( column_blocks, ROWS );	
	// Host and GPU array allocations and host->GPU transfers/initializations for DROP and TVS
	timer( START, begin_DROP, "for all iterations of DROP");	
	setup_MLP_lookup_tables();
#if (DROP_BRANCHING)
	if (DROP_TX_MODE==FULL_TX)
	{
		DROP_allocations(num_histories);
		DROP_host_2_device( start_position, num_histories);
	}
	else if (DROP_TX_MODE==PARTIAL_TX_PREALLOCATED)
		DROP_allocations(DROP_BLOCK_SIZE);
	else if (DROP_TX_MODE==PARTIAL_TX)
		//DROP_partial_tx_iteration(num_histories, iteration);
#elif PCD_DROP
	#if PCD_DROP_FULL_TX
		DROP_allocations(num_histories);
		DROP_host_2_device( start_position, num_histories);
	#elif PCD_DROP_PARTIAL_TX
		DROP_allocations(DROP_BLOCK_SIZE);
	#elif PCD_DROP_PARTIAL_TX_PREALLOCATED
		//DROP_partial_tx_iteration(num_histories, iteration);
	#endif 	
#endif
	#if TVS_ON	
		allocate_perturbation_arrays(false);
		generate_TVS_eta_sequence();
		x_TVS_h = (float*)calloc(NUM_VOXELS, sizeof(float));	
	#endif 

	sprintf(print_statement, "Performing reconstruction with TVS repeated %d times before each DROP iteration and writing output data/images to:", TVS_REPETITIONS);
	print_colored_text(print_statement, CYAN_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );
	print_colored_text(OUTPUT_FOLDER_UNIQUE, LIGHT_PURPLE_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT );		
	for(unsigned int iteration = 1; iteration <= iterations ; ++iteration) 
	{	    
		sprintf(iterate_filename, "%s%d", "x_", iteration );
		sprintf(print_statement, "Performing iteration %u of image reconstruction...", iteration);
		print_section_header(print_statement, MINOR_SECTION_SEPARATOR, LIGHT_BLUE_TEXT, YELLOW_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT );
		#if TVS_ON && TVS_FIRST
			#if TVS_OLD
				TVS_REPETITIONS = 1;
			#endif
			NTVS_iteration(iteration);		
		#endif
		// Transfer data for ALL reconstruction_histories before beginning image reconstruction, using the MLP lookup tables each time
		#if (DROP_TX_MODE==FULL_TX)
			DROP_full_tx_iteration(num_histories, iteration, relaxation_parameter);
		// Transfer data to GPU as needed and allocate/free the corresponding GPU arrays each kernel launch, using the MLP lookup tables each time
		#elif (DROP_TX_MODE==PARTIAL_TX_PREALLOCATED)
			DROP_partial_tx_preallocated_iteration(num_histories, iteration, relaxation_parameter);
		// Transfer data to GPU as needed but allocate and resuse the GPU arrays each kernel launch, using the MLP lookup tables each time
		#elif (DROP_TX_MODE==PARTIAL_TX)
			DROP_partial_tx_iteration(num_histories, iteration, relaxation_parameter);
		#endif 
		#if TVS_ON && !TVS_FIRST
			#if TVS_OLD
				TVS_REPETITIONS = 1;
			#endif
			NTVS_iteration(iteration);		
		#endif		
		// Transfer the updated image to the host and write it to disk
		if( WRITE_X_KI ) 
		{
			x_GPU_2_host();
			array_2_disk(iterate_filename, OUTPUT_DIRECTORY, OUTPUT_FOLDER_UNIQUE, x_h, COLUMNS, ROWS, SLICES, NUM_VOXELS, true ); 
		}
	}// end: for( unsigned int iteration = 1; iteration < iterations; iteration++)	
	#if (DROP_TX_MODE==FULL_TX)
		DROP_deallocations();
	#elif (DROP_TX_MODE==PARTIAL_TX_PREALLOCATED)
		DROP_deallocations();
	#elif (DROP_TX_MODE==PARTIAL_TX)
		//DROP_partial_tx_iteration(num_histories, iteration);
	#endif 
	#if TVS_ON	
		deallocate_perturbation_arrays(false);
	#endif 	
	#if (MLP_ALGORITHM == TABULATED)
		free_MLP_lookup_tables();	
	#endif 
	DROP_free_update_arrays();
	execution_time_DROP = timer( STOP, begin_DROP, "for all iterations of DROP");
}
/***********************************************************************************************************************************************************************************************************************/
/*************************************************************** Total variation superiorization ***********************************************************************************************************************/
/***********************************************************************************************************************************************************************************************************************/
void generate_TVS_eta_sequence()
{
	TVS_ETA_SEQUENCE_LENGTH = log(TVS_MIN_ETA)/log(A);
	TVS_eta_sequence_h = (float*)calloc( TVS_ETA_SEQUENCE_LENGTH - L_0 + 1, sizeof(float));
	for( int exponent = L_0, element = 0; exponent < TVS_ETA_SEQUENCE_LENGTH; exponent++, element++ )
		TVS_eta_sequence_h[element] = powf(A, exponent);	
	#if TVS_PARALLEL
		cudaMalloc( (void**) &TVS_eta_sequence_d,	TVS_ETA_SEQUENCE_LENGTH * sizeof(float) );
		cudaMemcpy(TVS_eta_sequence_d, TVS_eta_sequence_h, TVS_ETA_SEQUENCE_LENGTH * sizeof(float), cudaMemcpyHostToDevice ); 
	#endif
	//if( TVS_PARALLEL )
	//{
	//	cudaMalloc( (void**) &TVS_eta_sequence_d,	TVS_ETA_SEQUENCE_LENGTH * sizeof(float) );
	//	cudaMemcpy(TVS_eta_sequence_d, TVS_eta_sequence_h, TVS_ETA_SEQUENCE_LENGTH * sizeof(float), cudaMemcpyHostToDevice ); 
	//}
}
template<typename T> float calculate_total_variation( T* image, bool print_TV )
{
	int voxel;
	float total_variation = 0.0;
	// Calculate TV for unperturbed image x
	// Scott had slice = [1,SLICES-1), row = [0, ROWS -1), and column = [0, COLUMNS -1)
	// Not sure why just need to avoid last row and column due to indices [voxel + COLUMNS] and [voxel + 1], respectively 
	for( int slice = 0; slice < SLICES; slice++ )
	{
		for( int row = 0; row < ROWS - 1; row++ )
		{
			for( int column = 0; column < COLUMNS - 1; column++ )
			{
				voxel = column + row * COLUMNS + slice * ROWS * COLUMNS;
				total_variation += sqrt( powf( image[voxel + COLUMNS] - image[voxel], 2 ) + powf( image[voxel + 1] - image[voxel], 2 ) );
				//total_variation += sqrt( ( image[voxel + COLUMNS] - image[voxel] ) * ( image[voxel + COLUMNS] - image[voxel] ) + ( image[voxel + 1] - image[voxel] ) * ( image[voxel + 1] - image[voxel] ) );
			}
		}
	}
	if(print_TV)
	{	
		sprintf(print_statement, "------> Total variation = %6.6lf", total_variation);
		print_colored_text(print_statement, GREEN_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT);
	}
	return total_variation;
}
template<typename T> __device__ float calculate_total_variation_GPU( T* image )
{
	int voxel;
	float total_variation = 0.0;
	// Calculate TV for unperturbed image x
	// Scott had slice = [1,SLICES-1), row = [0, ROWS -1), and column = [0, COLUMNS -1)
	// Not sure why just need to avoid last row and column due to indices [voxel + COLUMNS] and [voxel + 1], respectively 
	for( int slice = 0; slice < SLICES; slice++ )
	{
		for( int row = 0; row < ROWS - 1; row++ )
		{
			for( int column = 0; column < COLUMNS - 1; column++ )
			{
				voxel = column + row * COLUMNS + slice * ROWS * COLUMNS;
				total_variation += sqrt( powf( image[voxel + COLUMNS] - image[voxel], 2 ) + powf( image[voxel + 1] - image[voxel], 2 ) );
				//total_variation += sqrt( ( image[voxel + COLUMNS] - image[voxel] ) * ( image[voxel + COLUMNS] - image[voxel] ) + ( image[voxel + 1] - image[voxel] ) * ( image[voxel + 1] - image[voxel] ) );
			}
		}
	}
	return total_variation;
}
void allocate_perturbation_arrays( bool parallel)
{
	G_x_h		= (float*) calloc( NUM_VOXELS, sizeof(float) );
	G_y_h		= (float*) calloc( NUM_VOXELS, sizeof(float) );
	G_norm_h	= (float*) calloc( NUM_VOXELS, sizeof(float) );
	G_h			= (float*) calloc( NUM_VOXELS, sizeof(float) );
	v_h			= (float*) calloc( NUM_VOXELS, sizeof(float) );
	y_h			= (float*) calloc( NUM_VOXELS, sizeof(float) );
	TV_y_h		= (float*) calloc( 1,		   sizeof(float) );

	if( parallel )
	{
		cudaMalloc( (void**) &G_x_d,	SIZE_IMAGE_FLOAT );
		cudaMalloc( (void**) &G_y_d,	SIZE_IMAGE_FLOAT );
		cudaMalloc( (void**) &G_norm_d,	SIZE_IMAGE_FLOAT );
		cudaMalloc( (void**) &G_d,		SIZE_IMAGE_FLOAT );
		cudaMalloc( (void**) &v_d,		SIZE_IMAGE_FLOAT );	
		cudaMalloc( (void**) &y_d,		SIZE_IMAGE_FLOAT );	
		cudaMalloc( (void**) &TV_y_d,	sizeof(float)	 );	

		cudaMemcpy(G_x_d,		G_x_h,		SIZE_IMAGE_FLOAT, cudaMemcpyHostToDevice ); 
		cudaMemcpy(G_y_d,		G_y_h,		SIZE_IMAGE_FLOAT, cudaMemcpyHostToDevice ); 
		cudaMemcpy(G_norm_d,	G_norm_h,   SIZE_IMAGE_FLOAT, cudaMemcpyHostToDevice ); 
		cudaMemcpy(G_d,			G_h,		SIZE_IMAGE_FLOAT, cudaMemcpyHostToDevice ); 
		cudaMemcpy(v_d,			v_h,		SIZE_IMAGE_FLOAT, cudaMemcpyHostToDevice ); 
		cudaMemcpy(y_d,			y_h,		SIZE_IMAGE_FLOAT, cudaMemcpyHostToDevice ); 
		cudaMemcpy(TV_y_d,		TV_y_h,		sizeof(float)	, cudaMemcpyHostToDevice ); 
	}
}
void deallocate_perturbation_arrays( bool parallel)
{
	free(G_x_h);
	free(G_y_h);
	free(G_norm_h);
	free(G_h);
	free(v_h);
	free(y_h);
	free(TV_y_h);

	if( parallel )
	{
		cudaFree( G_x_d);
		cudaFree(G_y_d);
		cudaFree(G_norm_d);
		cudaFree(G_d);
		cudaFree(v_d);	
		cudaFree(y_d);	
		cudaFree(TV_y_d);	
	}
}
template<typename T> void generate_perturbation_array( T* image)
{
	int row, column, slice, voxel;
	float norm_G = 0.0;

	// 1. Calculate the difference at each pixel with respect to rows and columns and get the normalization factor for this pixel
	for( slice = 0; slice < SLICES; slice++ )
	{
		for( row = 0; row < ROWS - 1; row++ )
		{
			for( column = 0; column < COLUMNS - 1; column++ )
			{
				voxel			= column + row * COLUMNS + slice * ROWS * COLUMNS;
				G_x_h[voxel]	= image[voxel + 1] - image[voxel];
				G_y_h[voxel]	= image[voxel + COLUMNS] - image[voxel];
				G_norm_h[voxel] = sqrt( pow( G_x_h[voxel], 2 ) + pow( G_y_h[voxel], 2 ) );
			}
		}
	}

	// 2. Add the appropriate difference values to each pixel subgradient
	for( slice = 0; slice < SLICES; slice++ )
	{
		for( row = 0; row < ROWS - 1; row++ )
		{
			for( column = 0; column < COLUMNS - 1; column++ )
			{
				voxel = column + row * COLUMNS + slice * ROWS * COLUMNS;
				if( G_norm_h[voxel] > 0.0 )
				{
					G_h[voxel]			 -= ( G_x_h[voxel] + G_y_h[voxel] ) / G_norm_h[voxel];		// Negative signs on x/y terms applied using -=
					G_h[voxel + COLUMNS] += G_y_h[voxel] / G_norm_h[voxel];
					G_h[voxel + 1]		 += G_x_h[voxel] / G_norm_h[voxel];
				}
			}
		}
	}			

	// 3. Get the norm of the subgradient vector 
	for( voxel = 0; voxel < NUM_VOXELS; voxel++ )
		norm_G += pow( G_h[voxel], 2 );
		//G_norm += G_h[voxel] * G_h[voxel];
	norm_G = sqrt(norm_G);
	
	// 4. Normalize the subgradient of the TV to produce v
	// If norm_G = 0, all elements of G_h are zero => all elements of v_h = 0.  
	if( norm_G != 0 )
	{
		for( voxel = 0; voxel < NUM_VOXELS; voxel++ )
			v_h[voxel] = G_h[voxel] / norm_G;			// Standard implementation where steepest descent is applied directly by inserting negative sign here
			//v_h[voxel] = -G_h[voxel] / norm_G;		// Negative sign applied as subtraction in application of perturbation, eliminating unnecessary op
	}
	else
	{
		for( voxel = 0; voxel < NUM_VOXELS; voxel++ )
			v_h[voxel] = 0.0;
	}
	for( voxel = 0; voxel < NUM_VOXELS; voxel++ )
	{
		G_x_h[voxel] = 0.0;
		G_y_h[voxel] = 0.0;
		G_norm_h[voxel] = 0.0;
		G_h[voxel] = 0.0;
	}
}
template<typename T> __global__ void generate_perturbation_array_GPU( float* G_x, float* G_y, float* G_norm, float* G, float* v, T* image )
{
	int column = blockIdx.x;
	int row = blockIdx.y;
	int slice = threadIdx.x;
	
	/*int column = blockIdx.x;
	int slice = blockIdx.y;
	int row = threadIdx.x;*/
	int voxel = column  + row * COLUMNS + slice * ROWS * COLUMNS;
	float norm_G = 0.0;
	//if( voxel < NUM_VOXELS && slice > 1 && slice < SLICES -1)
	if( voxel < NUM_VOXELS )
	{
		// 1. Calculate the difference at each pixel with respect to rows and columns and get the normalization factor for this pixel
		if( (column < COLUMNS - 1) &&  (row < ROWS - 1) && column > 0 && row > 0)
		{
			G_x[voxel]		= image[voxel + 1] - image[voxel];
			G_y[voxel]		= image[voxel + COLUMNS] - image[voxel];
			G_norm[voxel]	= sqrt( powf( G_x[voxel], 2 ) + powf( G_y[voxel], 2 ) );
			//G_norm[voxel] = sqrt( ( G_x[voxel] * G_x[voxel] ) + ( G_y[voxel] * G_y[voxel] ) );
		}

		// 2. Add the appropriate difference values to each pixel subgradient
		if( G_norm[voxel] > 0.0 )
			G[voxel] -= ( G_x[voxel] + G_y[voxel] ) / G_norm[voxel];		// Negative signs on x/y terms applied using -=		
		if( (row > 0) && (G_norm[voxel - COLUMNS] > 0.0) )		
			G[voxel] += G_y[voxel - COLUMNS] / G_norm[voxel - COLUMNS];
		if( (column > 0) && (G_norm[voxel - 1] > 0.0) )		
			G[voxel] += G_x[voxel - 1] / G_norm[voxel - 1];
		
		// 3. Get the norm of the subgradient vector 
		for( voxel = 0; voxel < NUM_VOXELS; voxel++ )
			norm_G += powf( G[voxel], 2 );
			//G_norm += G[voxel] * G[voxel];
		norm_G = sqrt(norm_G);

		// 4. Normalize the subgradient of the TV to produce v
		// If norm_G = 0, all elements of G are zero => all elements of v = 0.  
		if( norm_G != 0 )
			v[voxel] = G[voxel] / norm_G;		// Negative sign applied as subtraction in application of perturbation, eliminating unnecessary op
			//v[voxel] = -G[voxel] / norm_G;	// Standard implementation where steepest descent is applied directly by inserting negative sign here
		else
			v[voxel] = 0.0;
		G[voxel]		= 0.0;
		G_x[voxel]		= 0.0;
		G_y[voxel]		= 0.0;
		G_norm[voxel]	= 0.0;
	}
}
template<typename T>__device__ void generate_perturbation_array_GPU( float* G_x, float* G_y, float* G_norm, float* G, float* v, T* image, int column, int row, int voxel )
{
	float norm_G = 0.0;
	if( voxel < NUM_VOXELS )
	{
		// 1. Calculate the difference at each pixel with respect to rows and columns and get the normalization factor for this pixel
		if( (column < COLUMNS - 1) &&  (row < ROWS - 1) )
		{
			G_x[voxel]		= image[voxel + 1] - image[voxel];
			G_y[voxel]		= image[voxel + COLUMNS] - image[voxel];
			G_norm[voxel]	= sqrt( powf( G_x[voxel], 2 ) + powf( G_y[voxel], 2 ) );
			//G_norm[voxel] = sqrt( ( G_x[voxel] * G_x[voxel] ) + ( G_y[voxel] * G_y[voxel] ) );
		}

		// 2. Add the appropriate difference values to each pixel subgradient
		if( G_norm[voxel] > 0.0 )
			G[voxel] -= ( G_x[voxel] + G_y[voxel] ) / G_norm[voxel];		// Negative signs on x/y terms applied using -=		
		if( (row > 0) && (G_norm[voxel - COLUMNS] > 0.0) )		
			G[voxel] += G_y[voxel - COLUMNS] / G_norm[voxel - COLUMNS];
		if( (column > 0) && (G_norm[voxel - 1] > 0.0) )		
			G[voxel] += G_x[voxel - 1] / G_norm[voxel - 1];
		
		// 3. Get the norm of the subgradient vector 
		for( voxel = 0; voxel < NUM_VOXELS; voxel++ )
			norm_G += powf( G[voxel], 2 );
			//G_norm += G[voxel] * G[voxel];
		norm_G = sqrt(norm_G);

		// 4. Normalize the subgradient of the TV to produce v
		// If norm_G = 0, all elements of G are zero => all elements of v = 0.  
		if( norm_G != 0.0 )
			v[voxel] = G[voxel] / norm_G;		// Negative sign applied as subtraction in application of perturbation, eliminating unnecessary op
		else
			v[voxel] = 0.0;
		G[voxel]		= 0.0;
		G_x[voxel]		= 0.0;
		G_y[voxel]		= 0.0;
		G_norm[voxel]	= 0.0;
	}
}
template<typename T, typename P> void apply_TVS_perturbation( T* image_in, T* image_out, P* perturbation, float perturbation_magnitude, bool* hull )
{
	for( int voxel = 0; voxel < NUM_VOXELS; voxel++ )	// Add product of perturbation factor BETA_K_N and perturbation image to current image 
	{
		if(hull[voxel])
			image_out[voxel] = image_in[voxel] - BETA_K_N * v_h[voxel];			// Negative sign of steepest descent applied here as subtraction to eliminate operation calculating v
	}
}
template<typename T> void iteratively_perturb_image_in_place( T* image, bool* hull, UINT k )
{
	print_colored_text("Iteratively perturbing image in place with TV check...", CYAN_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );	
	float TV_y, TV_y_previous, TV_x;				// Initialize total variation (TV) value variables for unperturbed and current/previous perturbed images
	#if NTVS_ON
		L = randi(L_0 + k - 1, L);								// Randomly choose integer in [k,L] using default engine, resulting in a larger perturbation factor
	#endif
	
	// Perform TVS N_k=TVS_REPETITIONS times using perturbation factor BETA_K_N = A^L, incrementing L after each iteration and each time TV is not improved
	for( int n = 0; n < TVS_REPETITIONS; n++ )
	{
		TV_x = calculate_total_variation(image, DONT_PRINT_TV);		// Calculate total variation of unperturbed image
		TV_y_previous = 0.0;										// Reset TV value variable for previous perturbation of image at beginning of each iteration
		generate_perturbation_array(image);							// Generate non-ascending perturbation array v_h 
		BETA_K_N = TVS_eta_sequence_h[L - L_0];						// Set perturbation factor BETA_K_N to (L-L_0)-th element of precalculated TVS_ETA_SEQUENCE 
		//BETA_K_N = powf(A, L);									// Calculate perturbation factor BETA_K_N = A^L
		//std::copy(image, image + NUM_VOXELS,  x_before_TVS_h);
		for( int voxel = 0; voxel < NUM_VOXELS; voxel++ )			// Add product of perturbation factor BETA_K_N and perturbation image to current image 
		{
			if(hull[voxel])
				image[voxel] -= BETA_K_N * v_h[voxel];					// Negative sign of steepest descent applied here as subtraction to eliminate operation calculating v
		}
		// Iteratively increment L, update BETA_K_N, and reduce the perturbation applied above until the perturbed image's TV improves or its change < TV_THRESHOLD
		// Perturbations can be updated by exploiting: 2^-n - 2^-(n-1) = 2^-(n-1) => perturbation updates can be performed in place using result of previous pertubation:
		//		beta/2 ->	x - beta/2*v = x - (1 - 1/2)*beta*v = (x - beta*v) + beta/2*v = previous result + beta/2*v
		//		beta/4 ->	x - beta/4*v = x - (1 - 1/2 - 1/4)*beta*v = ((x - beta*v) + beta/2*v) + beta/4*v = (x - beta/2*v) + beta/4*v = previous result + beta/4*v
		//		beta/2^n ->	x - beta/2^n*v = x - (1 - 1/2 - 1/4 - ... - 1/2^n)*beta*v = (((x - beta*v) + beta/2*v) + beta/4*v) + ... + beta/2^n*v = (x - beta/2^(n-1)*v) + beta/2^n*v = previous result + beta/4*v
		// For floating point number a and perturbation factor beta/a^n 
		//		beta/a ->	x - beta/a*v	= x - [1 - (1/a - 1) * a] * beta * v = (x - 
		//		beta/a^2 ->	x - beta/a^2*v	= x - [a - (1/a - 1) * a^2] * beta * v = (
		//		beta/a^3 ->	x - beta/a^3*v	= x - [a^2 - (1/a - 1) * a^3] * beta * v = (
		while(true)
		{
			L++;
			TV_y = calculate_total_variation(image, DONT_PRINT_TV);										// Calculate total variation of perturbed image
			if( ( TV_y <= TV_x ) || ( fabs( TV_y - TV_y_previous ) / TV_y < TV_THRESHOLD ) || L > 300 )	// ( fabs( TV_y_previous - TV_y ) / TV_y * 100 < 0.01 )
				break;																					// TVS improved or changed < TV_THRESHOLD so exit loop
			else
			{ 
				BETA_K_N = TVS_eta_sequence_h[L - L_0];													// Update perturbation factor BETA_K_N
				//BETA_K_N *= PERTURB_DOWN_FACTOR;												// Factor to reduce previous perturbation to new desired BETA
				TV_y_previous = TV_y;																	// Save the new TV for next iteration
				for( int voxel = 0; voxel < NUM_VOXELS; voxel++ )										// Exploitation of relation between successive BETA_K_N to generate
				{
					if(hull[voxel])
						image[voxel] = x_before_TVS_h[voxel] - BETA_K_N * v_h[voxel];										// reduced perturbation in place by adding new BETA_K_N*v here
				}
			}
		}
	}
}
template<typename T> void iteratively_perturb_image( T* image, bool* hull, UINT k )
{
	print_colored_text("Iteratively perturbing image with TV check...", CYAN_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );	
	float TV_y, TV_y_previous, TV_x;				// Initialize total variation (TV) value variables for unperturbed and current/previous perturbed images
	#if NTVS_ON
		L = randi(L_0 + k - 1, L);								// Randomly choose integer in [k,L] using default engine, resulting in a larger perturbation factor
	#endif
	
	// Perform TVS N_k=TVS_REPETITIONS times using perturbation factor BETA_K_N = A^L, incrementing L after each iteration and each time TV is not improved
	for( int n = 0; n < TVS_REPETITIONS; n++ )
	{
		generate_perturbation_array(image);							// Generate non-ascending perturbation array v_h 
		BETA_K_N = TVS_eta_sequence_h[L - L_0];						// Set perturbation factor BETA_K_N to (L-L_0)-th element of precalculated TVS_ETA_SEQUENCE 
		//BETA_K_N = powf(A, L);									// Calculate perturbation factor BETA_K_N = A^L
		TV_x = calculate_total_variation(image, DONT_PRINT_TV);		// Calculate total variation of unperturbed image
		TV_y_previous = 0.0;										// Reset TV value variable for previous perturbation of image at beginning of each iteration
		std::copy(image, image + NUM_VOXELS, x_TVS_h );
		//for( int voxel = 0; voxel < NUM_VOXELS; voxel++ )			// Add product of perturbation factor BETA_K_N and perturbation image to current image 
		//{
		//	if(hull[voxel])
		//		image[voxel] = x_TVS_h[voxel] - BETA_K_N * v_h[voxel];					// Negative sign of steepest descent applied here as subtraction to eliminate operation calculating v
		//}
		apply_TVS_perturbation( x_TVS_h, image, v_h, BETA_K_N, hull );
		while(true)
		{
			L++;
			TV_y = calculate_total_variation(image, DONT_PRINT_TV);										// Calculate total variation of perturbed image
			if( ( TV_y <= TV_x ) || ( fabs( TV_y - TV_y_previous ) / TV_y < TV_THRESHOLD ) || L > 300 )	// ( fabs( TV_y_previous - TV_y ) / TV_y * 100 < 0.01 )
				break;																					// TVS improved or changed < TV_THRESHOLD so exit loop
			else
			{ 
				BETA_K_N = TVS_eta_sequence_h[L - L_0];													// Update perturbation factor BETA_K_N
				//BETA_K_N *= PERTURB_DOWN_FACTOR;												// Factor to reduce previous perturbation to new desired BETA
				TV_y_previous = TV_y;																	// Save the new TV for next iteration
				apply_TVS_perturbation( x_TVS_h, image, v_h, BETA_K_N, hull );
				//for( int voxel = 0; voxel < NUM_VOXELS; voxel++ )										// Exploitation of relation between successive BETA_K_N to generate
				//{
				//	if(hull[voxel])
				//		image[voxel] = x_TVS_h[voxel] - BETA_K_N * v_h[voxel];							// reduced perturbation in place by adding new BETA_K_N*v here			
				//}
			}																
		}
	}
}
template<typename T> void iteratively_perturb_image_in_place_GPU( T* image, bool* hull, UINT k )
{
	print_colored_text("Iteratively perturbing image in place with TV check...", CYAN_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );	
	float TV_y, TV_y_previous, TV_x;				// Initialize total variation (TV) value variables for unperturbed and current/previous perturbed images
	//L = randi(TVS_RAND_ENGINE, k, L);				// Randomly choose integer in [k,L] using specified engine, resulting in a larger perturbation factor
	//L = randi(k-1, L);								// Randomly choose integer in [k,L] using default engine, resulting in a larger perturbation factor
	#if NTVS_ON
		L = randi(L_0 + k - 1, L);								// Randomly choose integer in [k,L] using default engine, resulting in a larger perturbation factor
	#endif
	
	// Perform TVS N_k=TVS_REPETITIONS times using perturbation factor BETA_K_N = A^L, incrementing L after each iteration and each time TV is not improved
	for( int n = 0; n < TVS_REPETITIONS; n++ )
	{
		TV_x = calculate_total_variation(image, DONT_PRINT_TV);		// Calculate total variation of unperturbed image
		TV_y_previous = 0.0;										// Reset TV value variable for previous perturbation of image at beginning of each iteration
		generate_perturbation_array(image);							// Generate non-ascending perturbation array v_h 
		BETA_K_N = TVS_eta_sequence_h[L - L_0];						// Set perturbation factor BETA_K_N to (L-L_0)-th element of precalculated TVS_ETA_SEQUENCE 
		//BETA_K_N = powf(A, L);									// Calculate perturbation factor BETA_K_N = A^L
		for( int voxel = 0; voxel < NUM_VOXELS; voxel++ )			// Add product of perturbation factor BETA_K_N and perturbation image to current image 
			image[voxel] -= BETA_K_N * v_h[voxel];					// Negative sign of steepest descent applied here as subtraction to eliminate operation calculating v
		
		// Iteratively increment L, update BETA_K_N, and reduce the perturbation applied above until the perturbed image's TV improves or its change < TV_THRESHOLD
		// Perturbations can be updated by exploiting: 2^-n - 2^-(n-1) = 2^-(n-1) => perturbation updates can be performed in place using the previous result:
		//		beta/2 ->	x - beta/2*v = x - (1 - 1/2)*v = (x - beta*v) + beta/2*v = previous result + beta/2*v
		//		beta/4 ->	x - beta/4*v = x - (1 - 1/2 - 1/4)*v = ((x - beta*v) + beta/2*v) + beta/4*v = (x - beta/2*v) + beta/4*v = previous result + beta/4*v
		//		beta/2^n ->	x - beta/2^n*v = x - (1 - 1/2 - 1/4 - ... - 1/2^n)*v = (((x - beta*v) + beta/2*v) + beta/4*v) + ... + beta/2^n*v = (x - beta/2^(n-1)*v) + beta/2^n*v = previous result + beta/4*v
		while(true)
		{
			L++;
			TV_y = calculate_total_variation(image, DONT_PRINT_TV);								// Calculate total variation of perturbed image
			if( ( TV_y <= TV_x ) || ( fabs( TV_y - TV_y_previous ) / TV_y < TV_THRESHOLD ) )	// ( fabs( TV_y_previous - TV_y ) / TV_y * 100 < 0.01 )
				break;																			// TVS improved or changed < TV_THRESHOLD so exit loop
			else
			{ 
				BETA_K_N = TVS_eta_sequence_h[L - L_0];											// Update perturbation factor BETA_K_N
				TV_y_previous = TV_y;															// Save the new TV for next iteration
				for( int voxel = 0; voxel < NUM_VOXELS; voxel++ )								// Exploitation of relation between successive BETA_K_N to generate
					image[voxel] += BETA_K_N * v_h[voxel];										// reduced perturbation in place by adding new BETA_K_N*v here
			}																
		}
	}
}
template<typename T> void iteratively_perturb_image_unconditional( T* image, bool* hull, UINT k )
{
	print_colored_text("Iteratively perturbing image without TV check...", CYAN_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );	
	#if NTVS_ON
		L = randi(L_0 + k - 1, L);								// Randomly choose integer in [k,L] using default engine, resulting in a larger perturbation factor
	#endif
		
	// Perform TVS N_k=TVS_REPETITIONS times using perturbation factor BETA_K_N = A^L, incrementing L after each iteration and each time TV is not improved
	for( int n = 0; n < TVS_REPETITIONS; n++, L++ )
	{
		generate_perturbation_array(image);					// Generate non-ascending perturbation array v_h 
		BETA_K_N = TVS_eta_sequence_h[L - L_0];				// Set perturbation factor BETA_K_N to (L-L_0)-th element of precalculated TVS_ETA_SEQUENCE 
		//BETA_K_N = powf(A, L);							// Calculate perturbation factor BETA_K_N = A^L
		apply_TVS_perturbation( image, image, v_h, BETA_K_N, hull );
		//for( int voxel = 0; voxel < NUM_VOXELS; voxel++ )	// Add product of perturbation factor BETA_K_N and perturbation image to current image 
		//{
		//	if(hull[voxel])
		//		image[voxel] -= BETA_K_N * v_h[voxel];			// Negative sign of steepest descent applied here as subtraction to eliminate operation calculating v
		//}
	}
}		
void NTVS_iteration(const int iteration)
{
	//x_GPU_2_host();									//effecti Transfer the updated image to the host and write it to disk
	timer( START, begin_TVS_iteration, "for current NTVS iteration");
	print_colored_text("Before NTVS:", YELLOW_TEXT, BLACK_BACKGROUND, UNDERLINE_TEXT );	
	TV_x_values.push_back(calculate_total_variation(x_h, PRINT_TV));
	#if TVS_CONDITIONED	
		std::copy(x_h, x_h + NUM_VOXELS, x_TVS_h );
		iteratively_pe
			rturb_image( x_h, hull_h, iteration);
		//iteratively_perturb_image_in_place( x_h, hull_h, iteration);
	#else
		iteratively_perturb_image_unconditional( x_h, hull_h, iteration);
	#endif
	print_colored_text("After NTVS:", YELLOW_TEXT, BLACK_BACKGROUND, UNDERLINE_TEXT );	
	TV_x_values.push_back(calculate_total_variation(x_h, PRINT_TV));
	execution_time_TVS_iteration = timer( STOP, begin_TVS_iteration, "for current NTVS iteration");	
	execution_time_TVS += execution_time_TVS_iteration;
	execution_times_TVS_iterations.push_back(execution_time_TVS_iteration);
	//x_host_2_GPU();									// Transfer perturbed image back to GPU for update	
}
/***********************************************************************************************************************************************************************************************************************/
/********************************************************************* S-Curve Edge Attenuation Functions **************************************************************************************************************/
/***********************************************************************************************************************************************************************************************************************/
__device__ float sigmoid_GPU(UINT voxel)
{
	// L / ( 1 + exp(-k(x-x0)))
	return ( 1 / ( 1 + exp(-SIGMOID_STEEPNESS * (voxel - SIGMOID_MID_SHIFT)) ));		
}
__device__ float erf_GPU(UINT voxel)
{
	return erf(ROOT_PI_OVER_TWO * voxel);		
}
__device__ float atan_GPU(UINT voxel)
{
	return TWO_OVER_PI * atan(PI_OVER_TWO * voxel );		
}
__device__ float tanh_GPU(UINT voxel)
{
	return tanh(static_cast<float>(voxel) );		
}
__device__ float linear_over_root_GPU(UINT voxel)
{
	return voxel / sqrt( 1 + powf(voxel, 2.0) );		
}
__device__ float s_curve_scale_GPU(UINT MLP_step, UINT MLP_length)
{
	#if (DUAL_SIDED_S_CURVE)
		UINT index = min( MLP_step, MLP_length - 1 - MLP_step);
	#else
		UINT index = MLP_step;
	#endif	

	#if (S_CURVE == SIGMOID)
		return sigmoid_GPU(index);
	#elif (S_CURVE == ERF)
		return erf_GPU(index);
	#elif (S_CURVE == TANH)
		return tanh_GPU(index);
	#elif (S_CURVE == ATAN)
		return atan_GPU(index);
	#elif (S_CURVE == LIN_OVER_ROOT)
		return linear_over_root_GPU(index);
	#endif	
}
/***********************************************************************************************************************************************************************************************************************/
/********************************************************************************************* Image Reconstruction (GPU) **********************************************************************************************/
/***********************************************************************************************************************************************************************************************************************/
void image_reconstruction() 
{  
	/********************************************************************************************************************************************************/
	/*															Perform image reconstruction																*/
	/********************************************************************************************************************************************************/		
	
	/***********************************************************************************************************************************************************************************************************************/
	/****************************************************************** Find MLP endpoints and remove data for protons that did not enter and/or exit the hull *************************************************************/
	/***********************************************************************************************************************************************************************************************************************/
	print_section_header( "Performing image reconstruction", MAJOR_SECTION_SEPARATOR, GREEN_TEXT, YELLOW_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT );
	timer( START, begin_reconstruction, "for complete image reconstruction");		
	reconstruction_cuts();
	/***********************************************************************************************************************************************************************************************************************/
	/********************************************************************************************* Initialize Reconstructed Image ******************************************************************************************/
	/***********************************************************************************************************************************************************************************************************************/
	timer( START, begin_init_image, "for initializing reconstructed image x");
	DROP_setup_update_arrays();		// allocate GPU memory for x, x_update, and S and transfer initial iterate x_0 to x_d
	int column_blocks = static_cast<int>(COLUMNS/VOXELS_PER_THREAD);
	dim3 dimBlock( SLICES );
	dim3 dimGrid( column_blocks, ROWS );
	init_image_GPU<<< dimGrid, dimBlock >>>(x_update_d, S_d);	
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) 
		printf("Error: %s\n", cudaGetErrorString(cudaStatus));	
	execution_time_init_image = timer( STOP, begin_init_image, "for initializing reconstructed image x");	
	/***********************************************************************************************************************************************************************************************************************/
	/************************************************************************************************ Image Reconstruction *************************************************************************************************/
	/***********************************************************************************************************************************************************************************************************************/
	print_section_header( "Performing MLP and image reconstrution...", MINOR_SECTION_SEPARATOR, LIGHT_CYAN_TEXT, YELLOW_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT );
	if( PROJECTION_ALGORITHM == DROP )
		DROP_GPU(reconstruction_histories);
	//DROP_free_update_arrays();	
	print_section_exit( "Finished image reconstruction", SECTION_EXIT_CSTRING, RED_TEXT, RED_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );		
	print_section_header( "Timing information", MINOR_SECTION_SEPARATOR, LIGHT_CYAN_TEXT, YELLOW_TEXT, GRAY_BACKGROUND, DONT_UNDERLINE_TEXT );
	execution_time_reconstruction = timer( STOP, begin_reconstruction, "for complete image reconstruction");		
}
 /***********************************************************************************************************************************************************************************************************************/
/********************************************************************************************* Image Reconstruction (GPU) **********************************************************************************************/
/***********************************************************************************************************************************************************************************************************************/
template< typename T, typename LHS, typename RHS> T discrete_dot_product( LHS*& left, RHS*& right, unsigned int*& elements, unsigned int num_elements )
{
	T sum = 0;
	for( unsigned int i = 0; i < num_elements; i++)
		sum += ( left[i] * right[elements[i]] );
	return sum;
}
template< typename T, typename RHS> T scalar_dot_product( double scalar, RHS*& vector, unsigned int*& elements, unsigned int num_elements )
{
	T sum = 0;
	for( unsigned int i = 0; i < num_elements; i++)
		sum += vector[elements[i]];
	return scalar * sum;
}
double scalar_dot_product2( double scalar, float*& vector, unsigned int*& elements, unsigned int num_elements )
{
	double sum = 0;
	for( unsigned int i = 0; i < num_elements; i++)
		sum += vector[elements[i]];
	return scalar * sum;
}
/***********************************************************************************************************************************************************************************************************************/
/********************************************************************************** Routines for Writing Data Arrays/Vectors to Disk ***********************************************************************************/
/***********************************************************************************************************************************************************************************************************************/
void binary_2_ASCII()
{
	count_histories();
	char filename[256];
	FILE* output_file;
	int start_file_num = 0, end_file_num = 0, histories_2_process = 0;
	while( start_file_num != NUM_FILES )
	{
		while( end_file_num < NUM_FILES )
		{
			if( histories_2_process + histories_per_file[end_file_num] < MAX_GPU_HISTORIES )
				histories_2_process += histories_per_file[end_file_num];
			else
				break;
			end_file_num++;
		}
		read_data_chunk( histories_2_process, start_file_num, end_file_num );
		sprintf( filename, "%s%s/%s%s%d%s", OUTPUT_DIRECTORY, OUTPUT_FOLDER_UNIQUE, PROJECTION_DATA_BASENAME, "_", gantry_angle_h[0], ".txt" );
		output_file = fopen (filename, "w");

		for( unsigned int i = 0; i < histories_2_process; i++ )
		{
			fprintf(output_file, "%3f %3f %3f %3f %3f %3f %3f %3f %3f\n", t_in_1_h[i], t_in_2_h[i], t_out_1_h[i], t_out_2_h[i], v_in_1_h[i], v_in_2_h[i], v_out_1_h[i], v_out_2_h[i], WEPL_h[i]);
		}
		fclose (output_file);
		initial_processing_memory_clean();
		start_file_num = end_file_num;
		histories_2_process = 0;
	} 
}
template<typename T> void array_2_disk( const char* filename_base, const char* directory, const char* folder, T* data, const int x_max, const int y_max, const int z_max, const int elements, const bool single_file )
{
	char filename[256];
	std::ofstream output_file;
	int index;
	int num_files = z_max;
	int z_start = 0;
	int z_end = 1;
	if( single_file )
	{
		num_files = 1;
		z_end = z_max;
	}
	for( int file = 0; file < num_files; file++)
	{
		if( num_files == z_max )
			sprintf( filename, "%s%s/%s_%d.txt", directory, folder, filename_base, file );
		else
			sprintf( filename, "%s%s/%s.txt", directory, folder, filename_base );			
		output_file.open(filename);		
		for(int z = z_start; z < z_end; z++)
		{			
			for(int y = 0; y < y_max; y++)
			{
				for(int x = 0; x < x_max; x++)
				{
					index = x + ( y * x_max ) + ( z * x_max * y_max );
					if( index >= elements )
						break;
					output_file << data[index] << " ";
				}	
				if( index >= elements )
					break;
				output_file << std::endl;
			}
			if( index >= elements )
				break;
		}
		z_start += 1;
		z_end += 1;
		output_file.close();
		OUTPUT_FILE_LIST.push_back(std::string(filename));
		IMAGE_LIST.push_back(std::string(filename));		
	}
}
template<typename T> void vector_2_disk( const char* filename_base, const char* directory, const char* folder, std::vector<T> data, const int x_max, const int y_max, const int z_max, const bool single_file )
{
	char filename[256];
	std::ofstream output_file;
	int elements = data.size();
	int index;
	int num_files = z_max;
	int z_start = 0;
	int z_end = 1;
	if( single_file )
	{
		num_files = 1;
		z_end = z_max;
	}
	for( int file = 0; file < num_files; file++)
	{
		if( num_files == z_max )
			sprintf( filename, "%s%s/%s_%d.txt", directory, folder, filename_base, file );
		else
			sprintf( filename, "%s%s/%s.txt", directory, folder, filename_base );			
		output_file.open(filename);		
		for(int z = z_start; z < z_end; z++)
		{			
			for(int y = 0; y < y_max; y++)
			{
				for(int x = 0; x < x_max; x++)
				{
					index = x + ( y * x_max ) + ( z * x_max * y_max );
					if( index >= elements )
						break;
					output_file << data[index] << " ";
				}	
				if( index >= elements )
					break;
				output_file << std::endl;
			}
			if( index >= elements )
				break;
		}
		z_start += 1;
		z_end += 1;
		output_file.close();
	}
}
template<typename T> void t_bins_2_disk( FILE* output_file, const std::vector<int>& bin_numbers, const std::vector<T>& data, const BIN_ANALYSIS_TYPE type, const BIN_ORGANIZATION bin_order, int bin )
{
	const char* data_format = FLOAT_FORMAT;
	if( typeid(T) == typeid(int) )
		data_format = INT_FORMAT;
	if( typeid(T) == typeid(bool))
		data_format = BOOL_FORMAT;
	std::vector<T> bin_histories;
	unsigned int num_bin_members;
	for( int t_bin = 0; t_bin < T_BINS; t_bin++, bin++ )
	{
		if( bin_order == BY_HISTORY )
		{
			for( unsigned int i = 0; i < data.size(); i++ )
				if( bin_numbers[i] == bin )
					bin_histories.push_back(data[i]);
		}
		else
			bin_histories.push_back(data[bin]);
		num_bin_members = bin_histories.size();
		switch( type )
		{
			case COUNTS:	
				fprintf (output_file, "%d ", num_bin_members);																			
				break;
			case MEANS:		
				fprintf (output_file, "%f ", std::accumulate(bin_histories.begin(), bin_histories.end(), 0.0) / max(num_bin_members, 1 ) );
				break;
			case MEMBERS:	
				for( unsigned int i = 0; i < num_bin_members; i++ )
				{
					//fprintf (output_file, "%f ", bin_histories[i]); 
					fprintf (output_file, data_format, bin_histories[i]); 
					fputs(" ", output_file);
				}					 
				if( t_bin != T_BINS - 1 )
					fputs("\n", output_file);
		}
		bin_histories.resize(0);
		//bin_histories.shrink_to_fit();
	}
}
template<typename T> void bins_2_disk( const char* filename_base, const std::vector<int>& bin_numbers, const std::vector<T>& data, const BIN_ANALYSIS_TYPE type, const BIN_ANALYSIS_FOR which_bins, const BIN_ORGANIZATION bin_order, ... )
{
	//bins_2_disk( "WEPL_dist_pre_test2", empty_parameter, mean_WEPL_h, NUM_BINS, MEANS, ALL_BINS, BY_BIN );
	//bins_2_disk( "WEPL_dist_pre_test2", empty_parameter, sinogram_h, NUM_BINS, MEANS, ALL_BINS, BY_BIN );
	std::vector<int> angles;
	std::vector<int> angular_bins;
	std::vector<int> v_bins;
	if( which_bins == ALL_BINS )
	{
		angular_bins.resize( ANGULAR_BINS);
		v_bins.resize( V_BINS);
		//std::iota( angular_bins.begin(), angular_bins.end(), 0 );
		//std::iota( v_bins.begin(), v_bins.end(), 0 );
	}
	else
	{
		va_list specific_bins;
		va_start( specific_bins, bin_order );
		int num_angles = va_arg(specific_bins, int );
		int* angle_array = va_arg(specific_bins, int* );	
		angles.resize(num_angles);
		std::copy(angle_array, angle_array + num_angles, angles.begin() );

		int num_v_bins = va_arg(specific_bins, int );
		int* v_bins_array = va_arg(specific_bins, int* );	
		v_bins.resize(num_v_bins);
		std::copy(v_bins_array, v_bins_array + num_v_bins, v_bins.begin() );

		va_end(specific_bins);
		angular_bins.resize(angles.size());
		std::transform(angles.begin(), angles.end(), angular_bins.begin(), std::bind2nd(std::divides<int>(), ANGULAR_BIN_SIZE ) );
	}
	
	int num_angles = (int) angular_bins.size();
	int num_v_bins = (int) v_bins.size();
	char filename[256];
	int start_bin, angle;
	FILE* output_file;

	for( int angular_bin = 0; angular_bin < num_angles; angular_bin++)
	{
		angle = angular_bins[angular_bin] * ANGULAR_BIN_SIZE;
		sprintf( filename, "%s%s/%s_%03d%s", OUTPUT_DIRECTORY, OUTPUT_FOLDER_UNIQUE, filename_base, angle, ".txt" );
		output_file = fopen (filename, "w");
		for( int v_bin = 0; v_bin < num_v_bins; v_bin++)
		{			
			start_bin = angular_bins[angular_bin] * T_BINS + v_bins[v_bin] * ANGULAR_BINS * T_BINS;
			t_bins_2_disk( output_file, bin_numbers, data, type, bin_order, start_bin );
			if( v_bin != num_v_bins - 1 )
				fputs("\n", output_file);
		}	
		fclose (output_file);
	}
}
template<typename T> void t_bins_2_disk( FILE* output_file, int*& bin_numbers, T*& data, const unsigned int data_elements, const BIN_ANALYSIS_TYPE type, const BIN_ORGANIZATION bin_order, int bin )
{
	const char* data_format = FLOAT_FORMAT;
	if( typeid(T) == typeid(int) )
		data_format = INT_FORMAT;
	if( typeid(T) == typeid(bool))
		data_format = BOOL_FORMAT;

	std::vector<T> bin_histories;
	//int data_elements = sizeof(data)/sizeof(float);
	unsigned int num_bin_members;
	for( int t_bin = 0; t_bin < T_BINS; t_bin++, bin++ )
	{
		if( bin_order == BY_HISTORY )
		{
			for( unsigned int i = 0; i < data_elements; i++ )
				if( bin_numbers[i] == bin )
					bin_histories.push_back(data[i]);
		}
		else
			bin_histories.push_back(data[bin]);
		num_bin_members = (unsigned int) bin_histories.size();
		switch( type )
		{
			case COUNTS:	
				fprintf (output_file, "%d ", num_bin_members);																			
				break;
			case MEANS:		
				fprintf (output_file, "%f ", std::accumulate(bin_histories.begin(), bin_histories.end(), 0.0) / max(num_bin_members, 1 ) );
				break;
			case MEMBERS:	
				for( unsigned int i = 0; i < num_bin_members; i++ )
				{
					//fprintf (output_file, "%f ", bin_histories[i]); 
					fprintf (output_file, data_format, bin_histories[i]); 
					fputs(" ", output_file);
				}
				if( t_bin != T_BINS - 1 )
					fputs("\n", output_file);
		}
		bin_histories.resize(0);
		//bin_histories.shrink_to_fit();
	}
}
template<typename T>  void bins_2_disk( const char* filename_base, int*& bin_numbers, T*& data, const int data_elements, const BIN_ANALYSIS_TYPE type, const BIN_ANALYSIS_FOR which_bins, const BIN_ORGANIZATION bin_order, ... )
{
	std::vector<int> angles;
	std::vector<int> angular_bins;
	std::vector<int> v_bins;
	if( which_bins == ALL_BINS )
	{
		angular_bins.resize( ANGULAR_BINS);
		v_bins.resize( V_BINS);
		//std::iota( angular_bins.begin(), angular_bins.end(), 0 );
		//std::iota( v_bins.begin(), v_bins.end(), 0 );
	}
	else
	{
		va_list specific_bins;
		va_start( specific_bins, bin_order );
		int num_angles = va_arg(specific_bins, int );
		int* angle_array = va_arg(specific_bins, int* );	
		angles.resize(num_angles);
		std::copy(angle_array, angle_array + num_angles, angles.begin() );

		int num_v_bins = va_arg(specific_bins, int );
		int* v_bins_array = va_arg(specific_bins, int* );	
		v_bins.resize(num_v_bins);
		std::copy(v_bins_array, v_bins_array + num_v_bins, v_bins.begin() );

		va_end(specific_bins);
		angular_bins.resize(angles.size());
		std::transform(angles.begin(), angles.end(), angular_bins.begin(), std::bind2nd(std::divides<int>(), GANTRY_ANGLE_INTERVAL ) );
	}
	//int data_elements = sizeof(data)/sizeof(float);
	//std::cout << std::endl << data_elements << std::endl << std::endl;
	int num_angles = (int) angular_bins.size();
	int num_v_bins = (int) v_bins.size();
	char filename[256];
	int start_bin, angle;
	FILE* output_file;

	for( int angular_bin = 0; angular_bin < num_angles; angular_bin++)
	{
		angle = angular_bins[angular_bin] * (int) GANTRY_ANGLE_INTERVAL;
		sprintf( filename, "%s%s/%s_%03d%s", OUTPUT_DIRECTORY, OUTPUT_FOLDER_UNIQUE, filename_base, angle, ".txt" );
		output_file = fopen (filename, "w");
		for( int v_bin = 0; v_bin < num_v_bins; v_bin++)
		{			
			start_bin = angular_bins[angular_bin] * T_BINS + v_bins[v_bin] * ANGULAR_BINS * T_BINS;
			t_bins_2_disk( output_file, bin_numbers, data, data_elements, type, bin_order, start_bin );
			if( v_bin != num_v_bins - 1 )
				fputs("\n", output_file);
		}	
		fclose (output_file);
	}
}
/***********************************************************************************************************************************************************************************************************************/
/********************************************************************************* Image Position/Voxel Calculation Functions (Host) ***********************************************************************************/
/***********************************************************************************************************************************************************************************************************************/
int calculate_voxel( double zero_coordinate, double current_position, double voxel_size )
{
	return static_cast<int>(abs( current_position - zero_coordinate ) / voxel_size);
}
int positions_2_voxels(const double x, const double y, const double z, int& voxel_x, int& voxel_y, int& voxel_z )
{
	voxel_x = int( ( x - X_ZERO_COORDINATE ) / VOXEL_WIDTH );				
	voxel_y = int( ( Y_ZERO_COORDINATE - y ) / VOXEL_HEIGHT );
	voxel_z = int( ( Z_ZERO_COORDINATE - z ) / VOXEL_THICKNESS );
	return voxel_x + voxel_y * COLUMNS + voxel_z * COLUMNS * ROWS;
}
int position_2_voxel( double x, double y, double z )
{
	int voxel_x = int( ( x - X_ZERO_COORDINATE ) / VOXEL_WIDTH );
	int voxel_y = int( ( Y_ZERO_COORDINATE - y ) / VOXEL_HEIGHT );
	int voxel_z = int( ( Z_ZERO_COORDINATE - z ) / VOXEL_THICKNESS );
	return voxel_x + voxel_y * COLUMNS + voxel_z * COLUMNS * ROWS;
}
void voxel_2_3D_voxels( int voxel, int& voxel_x, int& voxel_y, int& voxel_z )
{
	voxel_x = 0;
    voxel_y = 0;
    voxel_z = 0;
    
    while( voxel - COLUMNS * ROWS > 0 )
	{
		voxel -= COLUMNS * ROWS;
		voxel_z++;
	}
	// => bin = t_bin + angular_bin * T_BINS > 0
	while( voxel - COLUMNS > 0 )
	{
		voxel -= COLUMNS;
		voxel_y++;
	}
	// => bin = t_bin > 0
	voxel_x = voxel;
}
double voxel_2_position( int voxel_i, double voxel_i_size, int num_voxels_i, int coordinate_progression )
{
	// voxel_i = 50, num_voxels_i = 200, middle_voxel = 100, ( 50 - 100 ) * 1 = -50
	double zero_voxel = ( num_voxels_i - 1) / 2.0;
	return coordinate_progression * ( voxel_i - zero_voxel ) * voxel_i_size;
}
void voxel_2_positions( int voxel, double& x, double& y, double& z )
{
	int voxel_x, voxel_y, voxel_z;
	voxel_2_3D_voxels( voxel, voxel_x, voxel_y, voxel_z );
	x = voxel_2_position( voxel_x, VOXEL_WIDTH, COLUMNS, 1 );
	y = voxel_2_position( voxel_y, VOXEL_HEIGHT, ROWS, -1 );
	z = voxel_2_position( voxel_z, VOXEL_THICKNESS, SLICES, -1 );
}
double voxel_2_radius_squared( int voxel )
{
	int voxel_x, voxel_y, voxel_z;
	voxel_2_3D_voxels( voxel, voxel_x, voxel_y, voxel_z );
	double x = voxel_2_position( voxel_x, VOXEL_WIDTH, COLUMNS, 1 );
	double y = voxel_2_position( voxel_y, VOXEL_HEIGHT, ROWS, -1 );
	return pow( x, 2.0 ) + pow( y, 2.0 );
}
/***********************************************************************************************************************************************************************************************************************/
/********************************************************************************* Image Position/Voxel Calculation Functions (Device) *********************************************************************************/
/***********************************************************************************************************************************************************************************************************************/
__device__ int calculate_voxel_GPU( double zero_coordinate, double current_position, double voxel_size )
{
	return abs( current_position - zero_coordinate ) / voxel_size;
}
__device__ int positions_2_voxels_GPU(const double x, const double y, const double z, int& voxel_x, int& voxel_y, int& voxel_z )
{
	voxel_x = int( ( x - X_ZERO_COORDINATE ) / VOXEL_WIDTH );				
	voxel_y = int( ( Y_ZERO_COORDINATE - y ) / VOXEL_HEIGHT );
	voxel_z = int( ( Z_ZERO_COORDINATE - z ) / VOXEL_THICKNESS );
	return voxel_x + voxel_y * COLUMNS + voxel_z * COLUMNS * ROWS;
}
__device__ int position_2_voxel_GPU( double x, double y, double z )
{
	int voxel_x = int( ( x - X_ZERO_COORDINATE ) / VOXEL_WIDTH );
	int voxel_y = int( ( Y_ZERO_COORDINATE - y ) / VOXEL_HEIGHT );
	int voxel_z = int( ( Z_ZERO_COORDINATE - z ) / VOXEL_THICKNESS );
	return voxel_x + voxel_y * COLUMNS + voxel_z * COLUMNS * ROWS;
}
__device__ void voxel_2_3D_voxels_GPU( int voxel, int& voxel_x, int& voxel_y, int& voxel_z )
{
	voxel_x = 0;
    voxel_y = 0;
    voxel_z = 0;
    
    while( voxel - COLUMNS * ROWS > 0 )
	{
		voxel -= COLUMNS * ROWS;
		voxel_z++;
	}
	// => bin = t_bin + angular_bin * T_BINS > 0
	while( voxel - COLUMNS > 0 )
	{
		voxel -= COLUMNS;
		voxel_y++;
	}
	// => bin = t_bin > 0
	voxel_x = voxel;
}
__device__ double voxel_2_position_GPU( int voxel_i, double voxel_i_size, int num_voxels_i, int coordinate_progression )
{
	// voxel_i = 50, num_voxels_i = 200, middle_voxel = 100, ( 50 - 100 ) * 1 = -50
	double zero_voxel = ( num_voxels_i - 1) / 2.0;
	return coordinate_progression * ( voxel_i - zero_voxel ) * voxel_i_size;
}
__device__ void voxel_2_positions_GPU( int voxel, double& x, double& y, double& z )
{
	int voxel_x, voxel_y, voxel_z;
	voxel_2_3D_voxels_GPU( voxel, voxel_x, voxel_y, voxel_z );
	x = voxel_2_position_GPU( voxel_x, VOXEL_WIDTH, COLUMNS, 1 );
	y = voxel_2_position_GPU( voxel_y, VOXEL_HEIGHT, ROWS, -1 );
	z = voxel_2_position_GPU( voxel_z, VOXEL_THICKNESS, SLICES, -1 );
}
__device__ double voxel_2_radius_squared_GPU( int voxel )
{
	int voxel_x, voxel_y, voxel_z;
	voxel_2_3D_voxels_GPU( voxel, voxel_x, voxel_y, voxel_z );
	double x = voxel_2_position_GPU( voxel_x, VOXEL_WIDTH, COLUMNS, 1 );
	double y = voxel_2_position_GPU( voxel_y, VOXEL_HEIGHT, ROWS, -1 );
	return pow( x, 2.0 ) + pow( y, 2.0 );
}
/***********************************************************************************************************************************************************************************************************************/
/********************************************************************************************** Voxel Walk Functions (Host) ********************************************************************************************/
/***********************************************************************************************************************************************************************************************************************/
double distance_remaining( double zero_coordinate, double current_position, int increasing_direction, int step_direction, double voxel_size, int current_voxel )
{
	/* Determine distance from current position to the next voxel edge.  path_projection is used to determine next intersected voxel, but it is possible for two edges to have the same distance in 
	// a particular direction if the path passes through a corner of a voxel.  In this case, we need to advance voxels in two directions simultaneously and to avoid if/else branches
	// to handle every possibility, we simply advance one of the voxel numbers and pass the assumed current_voxel to this function.  Under normal circumstances, this function simply return the 
	// distance to the next edge in a particual direction.  If the path passed through a corner, then this function will return 0 so we will know the voxel needs to be advanced in this direction too.
	*/
	int next_voxel = current_voxel + increasing_direction * step_direction;//  vz = 0, i = -1, s = 1 	
	double next_edge = edge_coordinate( zero_coordinate, next_voxel, voxel_size, increasing_direction, step_direction );
	return abs( next_edge - current_position );
}
double edge_coordinate( double zero_coordinate, int voxel_entered, double voxel_size, int increasing_direction, int step_direction )
{
	// Determine if on left or right edge, since entering a voxel can happen from either side depending on path direction, then calculate the x/y/z coordinate corresponding to the x/y/z edge, respectively
	int on_edge = ( step_direction == increasing_direction ) ? voxel_entered : voxel_entered + 1;
	return zero_coordinate + ( increasing_direction * on_edge * voxel_size );
}
double path_projection( double m, double current_coordinate, double zero_coordinate, int current_voxel, double voxel_size, int increasing_direction, int step_direction )
{
	// Based on the dimensions of a voxel and the current (x,y,z) position, we can determine how far it is to the next edge in the x, y, and z directions.  Since the points where a path crosses 
	// one of these edges each have a corresponding (x,y,z) coordinate, we can determine which edge will be crossed next by comparing the coordinates of the next x/y/z edge in one of the three 
	// directions and determining which is closest to the current position.  For example, the x/y/z edge whose x coordinate is closest to the current x coordinate is the next edge 
	int next_voxel = current_voxel + increasing_direction * step_direction;
	double next_edge = edge_coordinate( zero_coordinate, next_voxel, voxel_size, increasing_direction, step_direction );
	// y = m(x-x0) + y0 => distance = m * (x - x0)
	return m * ( next_edge - current_coordinate );
}
double corresponding_coordinate( double m, double x, double x0, double y0 )
{
	// Using the coordinate returned by edge_coordinate, call this function to determine one of the other coordinates using 
	// y = m(x-x0)+y0 equation determine coordinates in other directions by subsequent calls to this function
	return m * ( x - x0 ) + y0;
}
void take_2D_step
( 
	const int x_move_direction, const int y_move_direction, const int z_move_direction,
	const double dy_dx, const double dz_dx, const double dz_dy, 
	const double dx_dy, const double dx_dz, const double dy_dz, 
	const double x_start, const double y_start, const double z_start, 
	double& x, double& y, double& z, 
	int& voxel_x, int& voxel_y, int& voxel_z, int& voxel,
	double& x_to_go, double& y_to_go, double& z_to_go	
)
{
	// Change in x for Move to Voxel Edge in y
	double y_extension = fabs( dx_dy ) * y_to_go;
	//If Next Voxel Edge is in x or xy Diagonal
	if( x_to_go <= y_extension )
	{
		//printf(" x_to_go <= y_extension \n");
		voxel_x += x_move_direction;					
		x = edge_coordinate( X_ZERO_COORDINATE, voxel_x, VOXEL_WIDTH, X_INCREASING_DIRECTION, x_move_direction );
		y = corresponding_coordinate( dy_dx, x, x_start, y_start );
		x_to_go = VOXEL_WIDTH;
		y_to_go = distance_remaining( Y_ZERO_COORDINATE, y, Z_INCREASING_DIRECTION, y_move_direction, VOXEL_HEIGHT, voxel_y );
	}
	// Else Next Voxel Edge is in y
	else
	{
		//printf(" y_extension < x_extension \n");				
		voxel_y -= y_move_direction;
		y = edge_coordinate( Y_ZERO_COORDINATE, voxel_y, VOXEL_HEIGHT, Y_INCREASING_DIRECTION, y_move_direction );
		x = corresponding_coordinate( dx_dy, y, y_start, x_start );
		x_to_go = distance_remaining( X_ZERO_COORDINATE, x, X_INCREASING_DIRECTION, x_move_direction, VOXEL_WIDTH, voxel_x );
		y_to_go = VOXEL_HEIGHT;
	}
	if( x_to_go == 0 )
	{
		x_to_go = VOXEL_WIDTH;
		voxel_x += x_move_direction;
	}
	if( y_to_go == 0 )
	{
		y_to_go = VOXEL_HEIGHT;
		voxel_y -= y_move_direction;
	}
	voxel_z = max(voxel_z, 0 );
	voxel = voxel_x + voxel_y * COLUMNS + voxel_z * COLUMNS * ROWS;
}
void take_3D_step
( 
	const int x_move_direction, const int y_move_direction, const int z_move_direction,
	const double dy_dx, const double dz_dx, const double dz_dy, 
	const double dx_dy, const double dx_dz, const double dy_dz, 
	const double x_start, const double y_start, const double z_start, 
	double& x, double& y, double& z, 
	int& voxel_x, int& voxel_y, int& voxel_z, int& voxel,
	double& x_to_go, double& y_to_go, double& z_to_go	
)
{
		// Change in z for Move to Voxel Edge in x and y
	double x_extension = fabs( dz_dx ) * x_to_go;
	double y_extension = fabs( dz_dy ) * y_to_go;
	if( (z_to_go <= x_extension  ) && (z_to_go <= y_extension) )
	{
		//printf("z_to_go <= x_extension && z_to_go <= y_extension\n");				
		voxel_z -= z_move_direction;					
		z = edge_coordinate( Z_ZERO_COORDINATE, voxel_z, VOXEL_THICKNESS, Z_INCREASING_DIRECTION, z_move_direction );					
		x = corresponding_coordinate( dx_dz, z, z_start, x_start );
		y = corresponding_coordinate( dy_dz, z, z_start, y_start );
		x_to_go = distance_remaining( X_ZERO_COORDINATE, x, X_INCREASING_DIRECTION, x_move_direction, VOXEL_WIDTH, voxel_x );
		y_to_go = distance_remaining( Y_ZERO_COORDINATE, y, Y_INCREASING_DIRECTION, y_move_direction, VOXEL_HEIGHT, voxel_y );	
		z_to_go = VOXEL_THICKNESS;
	}
	//If Next Voxel Edge is in x or xy Diagonal
	else if( x_extension <= y_extension )
	{
		//printf(" x_extension <= y_extension \n");					
		voxel_x += x_move_direction;
		x = edge_coordinate( X_ZERO_COORDINATE, voxel_x, VOXEL_WIDTH, X_INCREASING_DIRECTION, x_move_direction );
		y = corresponding_coordinate( dy_dx, x, x_start, y_start );
		z = corresponding_coordinate( dz_dx, x, x_start, z_start );
		x_to_go = VOXEL_WIDTH;
		y_to_go = distance_remaining( Y_ZERO_COORDINATE, y, Y_INCREASING_DIRECTION, y_move_direction, VOXEL_HEIGHT, voxel_y );
		z_to_go = distance_remaining( Z_ZERO_COORDINATE, z, Z_INCREASING_DIRECTION, z_move_direction, VOXEL_THICKNESS, voxel_z );
	}
	// Else Next Voxel Edge is in y
	else
	{
		//printf(" y_extension < x_extension \n");
		voxel_y -= y_move_direction;					
		y = edge_coordinate( Y_ZERO_COORDINATE, voxel_y, VOXEL_HEIGHT, Y_INCREASING_DIRECTION, y_move_direction );
		x = corresponding_coordinate( dx_dy, y, y_start, x_start );
		z = corresponding_coordinate( dz_dy, y, y_start, z_start );
		x_to_go = distance_remaining( X_ZERO_COORDINATE, x, X_INCREASING_DIRECTION, x_move_direction, VOXEL_WIDTH, voxel_x );
		y_to_go = VOXEL_HEIGHT;					
		z_to_go = distance_remaining( Z_ZERO_COORDINATE, z, Z_INCREASING_DIRECTION, z_move_direction, VOXEL_THICKNESS, voxel_z );
	}
	if( x_to_go == 0 )
	{
		x_to_go = VOXEL_WIDTH;
		voxel_x += x_move_direction;
	}
	if( y_to_go == 0 )
	{
		y_to_go = VOXEL_HEIGHT;
		voxel_y -= y_move_direction;
	}
	if( z_to_go == 0 )
	{
		z_to_go = VOXEL_THICKNESS;
		voxel_z -= z_move_direction;
	}
	voxel_z = max(voxel_z, 0 );
	voxel = voxel_x + voxel_y * COLUMNS + voxel_z * COLUMNS * ROWS;
	//end_walk = ( voxel == voxel_out ) || ( voxel_x >= COLUMNS ) || ( voxel_y >= ROWS ) || ( voxel_z >= SLICES );
}
/***********************************************************************************************************************************************************************************************************************/
/********************************************************************************************* Voxel Walk Functions (Device) *******************************************************************************************/
/***********************************************************************************************************************************************************************************************************************/
__device__ double distance_remaining_GPU( double zero_coordinate, double current_position, int increasing_direction, int step_direction, double voxel_size, int current_voxel )
{
	/* Determine distance from current position to the next voxel edge.  Based on the dimensions of a voxel and the current (x,y,z) position, we can determine how far it is to
	// the next edge in the x, y, and z directions.  Since the points where a path crosses one of these edges each have a corresponding (x,y,z) coordinate, we can determine
	// which edge will be crossed next by comparing the coordinates of the next x/y/z edge in one of the three directions and determining which is closest the current position.  
	// For example, the edge whose x coordinate is closest to the x coordinate will be encountered next.  However, it is possible for two edges to have the same distance in 
	// a particular direction if the path passes through a corner of a voxel.  In this case we need to advance voxels in two direction simultaneously and to avoid if/else branches
	// to handle every possibility, we simply advance one of the voxel numbers and pass the assumed current_voxel to this function.  If the path passed through a corner, then this
	// function will return 0 for remaining distance and we can advance the voxel number upon review of its return value.
	*/
	int next_voxel = current_voxel + increasing_direction * step_direction;
	double next_edge = edge_coordinate_GPU( zero_coordinate, next_voxel, voxel_size, increasing_direction, step_direction );
	return abs( next_edge - current_position );
}
__device__ double edge_coordinate_GPU( double zero_coordinate, int voxel_entered, double voxel_size, int increasing_direction, int step_direction )
{
	int on_edge = ( step_direction == increasing_direction ) ? voxel_entered : voxel_entered + 1;
	return zero_coordinate + ( increasing_direction * on_edge * voxel_size );
}
__device__ double path_projection_GPU( double m, double x0, double zero_coordinate, int current_voxel, double voxel_size, int increasing_direction, int step_direction )
{

	int next_voxel = current_voxel + increasing_direction * step_direction;
	double x_next_edge = edge_coordinate_GPU( zero_coordinate, next_voxel, voxel_size, increasing_direction, step_direction );
	// y = mx + b: x(2) = [Dx(2)/Dx(1)]*[x(1) - x(1,0)] + x(2,0) => x = (Dx/Dy)*(y - y0) + x0
	return m * ( x_next_edge - x0 );
}
__device__ double corresponding_coordinate_GPU( double m, double x, double x0, double y0 )
{
	// Using the coordinate returned by edge_coordinate, call this function to determine one of the other coordinates using 
	// y = m(x-x0)+y0 equation determine coordinates in other directions by subsequent calls to this function
	return m * ( x - x0 ) + y0;
}
__device__ void take_2D_step_GPU
( 
	const int x_move_direction, const int y_move_direction, const int z_move_direction,
	const double dy_dx, const double dz_dx, const double dz_dy, 
	const double dx_dy, const double dx_dz, const double dy_dz, 
	const double x_start, const double y_start, const double z_start, 
	double& x, double& y, double& z, 
	int& voxel_x, int& voxel_y, int& voxel_z, int& voxel,
	double& x_to_go, double& y_to_go, double& z_to_go	
)
{
	// Change in x for Move to Voxel Edge in y
	double y_extension = fabs( dx_dy ) * y_to_go;
	//If Next Voxel Edge is in x or xy Diagonal
	if( x_to_go <= y_extension )
	{
		//printf(" x_to_go <= y_extension \n");
		voxel_x += x_move_direction;					
		x = edge_coordinate_GPU( X_ZERO_COORDINATE, voxel_x, VOXEL_WIDTH, X_INCREASING_DIRECTION, x_move_direction );
		y = corresponding_coordinate_GPU( dy_dx, x, x_start, y_start );
		x_to_go = VOXEL_WIDTH;
		y_to_go = distance_remaining_GPU( Y_ZERO_COORDINATE, y, Y_INCREASING_DIRECTION, y_move_direction, VOXEL_HEIGHT, voxel_y );
	}
	// Else Next Voxel Edge is in y
	else
	{
		//printf(" y_extension < x_extension \n");				
		voxel_y -= y_move_direction;
		y = edge_coordinate_GPU( Y_ZERO_COORDINATE, voxel_y, VOXEL_HEIGHT, Y_INCREASING_DIRECTION, y_move_direction );
		x = corresponding_coordinate_GPU( dx_dy, y, y_start, x_start );
		x_to_go = distance_remaining_GPU( X_ZERO_COORDINATE, x, X_INCREASING_DIRECTION, x_move_direction, VOXEL_WIDTH, voxel_x );
		y_to_go = VOXEL_HEIGHT;
	}
	if( x_to_go == 0 )
	{
		x_to_go = VOXEL_WIDTH;
		voxel_x += x_move_direction;
	}
	if( y_to_go == 0 )
	{
		y_to_go = VOXEL_HEIGHT;
		voxel_y -= y_move_direction;
	}
	voxel_z = max(voxel_z, 0 );
	voxel = voxel_x + voxel_y * COLUMNS + voxel_z * COLUMNS * ROWS;
}
__device__ void take_3D_step_GPU
( 
	const int x_move_direction, const int y_move_direction, const int z_move_direction,
	const double dy_dx, const double dz_dx, const double dz_dy, 
	const double dx_dy, const double dx_dz, const double dy_dz, 
	const double x_start, const double y_start, const double z_start, 
	double& x, double& y, double& z, 
	int& voxel_x, int& voxel_y, int& voxel_z, int& voxel,
	double& x_to_go, double& y_to_go, double& z_to_go	
)
{
		// Change in z for Move to Voxel Edge in x and y
	double x_extension = fabs( dz_dx ) * x_to_go;
	double y_extension = fabs( dz_dy ) * y_to_go;
	if( (z_to_go <= x_extension  ) && (z_to_go <= y_extension) )
	{
		//printf("z_to_go <= x_extension && z_to_go <= y_extension\n");				
		voxel_z -= z_move_direction;					
		z = edge_coordinate_GPU( Z_ZERO_COORDINATE, voxel_z, VOXEL_THICKNESS, Z_INCREASING_DIRECTION, z_move_direction );					
		x = corresponding_coordinate_GPU( dx_dz, z, z_start, x_start );
		y = corresponding_coordinate_GPU( dy_dz, z, z_start, y_start );
		x_to_go = distance_remaining_GPU( X_ZERO_COORDINATE, x, X_INCREASING_DIRECTION, x_move_direction, VOXEL_WIDTH, voxel_x );
		y_to_go = distance_remaining_GPU( Y_ZERO_COORDINATE, y, Y_INCREASING_DIRECTION, y_move_direction, VOXEL_HEIGHT, voxel_y );	
		z_to_go = VOXEL_THICKNESS;
	}
	//If Next Voxel Edge is in x or xy Diagonal
	else if( x_extension <= y_extension )
	{
		//printf(" x_extension <= y_extension \n");					
		voxel_x += x_move_direction;
		x = edge_coordinate_GPU( X_ZERO_COORDINATE, voxel_x, VOXEL_WIDTH, X_INCREASING_DIRECTION, x_move_direction );
		y = corresponding_coordinate_GPU( dy_dx, x, x_start, y_start );
		z = corresponding_coordinate_GPU( dz_dx, x, x_start, z_start );
		x_to_go = VOXEL_WIDTH;
		y_to_go = distance_remaining_GPU( Y_ZERO_COORDINATE, y, Y_INCREASING_DIRECTION, y_move_direction, VOXEL_HEIGHT, voxel_y );
		z_to_go = distance_remaining_GPU( Z_ZERO_COORDINATE, z, Z_INCREASING_DIRECTION, z_move_direction, VOXEL_THICKNESS, voxel_z );
	}
	// Else Next Voxel Edge is in y
	else
	{
		//printf(" y_extension < x_extension \n");
		voxel_y -= y_move_direction;					
		y = edge_coordinate_GPU( Y_ZERO_COORDINATE, voxel_y, VOXEL_HEIGHT, Y_INCREASING_DIRECTION, y_move_direction );
		x = corresponding_coordinate_GPU( dx_dy, y, y_start, x_start );
		z = corresponding_coordinate_GPU( dz_dy, y, y_start, z_start );
		x_to_go = distance_remaining_GPU( X_ZERO_COORDINATE, x, X_INCREASING_DIRECTION, x_move_direction, VOXEL_WIDTH, voxel_x );
		y_to_go = VOXEL_HEIGHT;					
		z_to_go = distance_remaining_GPU( Z_ZERO_COORDINATE, z, Z_INCREASING_DIRECTION, z_move_direction, VOXEL_THICKNESS, voxel_z );
	}
	if( x_to_go == 0 )
	{
		x_to_go = VOXEL_WIDTH;
		voxel_x += x_move_direction;
	}
	if( y_to_go == 0 )
	{
		y_to_go = VOXEL_HEIGHT;
		voxel_y -= y_move_direction;
	}
	if( z_to_go == 0 )
	{
		z_to_go = VOXEL_THICKNESS;
		voxel_z -= z_move_direction;
	}
	voxel_z = max(voxel_z, 0 );
	voxel = voxel_x + voxel_y * COLUMNS + voxel_z * COLUMNS * ROWS;
	//end_walk = ( voxel == voxel_out ) || ( voxel_x >= COLUMNS ) || ( voxel_y >= ROWS ) || ( voxel_z >= SLICES );
}
/***********************************************************************************************************************************************************************************************************************/
/************************************************************************************************ Host Helper Functions ************************************************************************************************/
/***********************************************************************************************************************************************************************************************************************/
void create_random_number_generator_engines()
{
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	//std::default_random_engine generator (seed);
	std::default_random_engine default_rand_generator(seed);
	std::minstd_rand minstd_rand_generator(seed);
	std::minstd_rand0 minstd_rand0_generator(seed);
	std::mt19937 mt19937_rand_generator(seed);
	std::mt19937_64 mt19937_64_rand_generator(seed);
	std::ranlux24_base ranlux24_base_rand_generator(seed);
	std::ranlux48_base ranlux48_base_rand_generator(seed);
	std::ranlux24 ranlux24_rand_generator(seed);
	std::ranlux48 ranlux48_rand_generator(seed);
	std::knuth_b knuth_b_rand_generator(seed);
}
int randi(int min_value, int max_value)
{
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	//unsigned RANDI_SEED = std::chrono::system_clock::now().time_since_epoch().count();
	//std::default_random_engine generator(RANDI_SEED);
	std::default_random_engine generator(seed);
	std::uniform_int_distribution<int> distribution(min_value, max_value);
	return distribution(generator);
}
int randi(RAND_GENERATORS engine, int min_value, int max_value)
{
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::uniform_int_distribution<int> distribution(min_value, max_value);
	//std::default_random_engine generator(seed);
	//return distribution(generator);
	std::minstd_rand minstd_rand_generator(seed);					
	std::minstd_rand0 minstd_rand0_generator(seed);					
	std::mt19937 mt19937_rand_generator(seed);						
	std::mt19937_64 mt19937_64_rand_generator(seed);				
	std::ranlux24 ranlux24_rand_generator(seed);					
	std::ranlux48 ranlux48_rand_generator(seed);					
	std::knuth_b knuth_b_rand_generator(seed);						
	std::default_random_engine default_rand_generator(seed);	
	//int output;
	switch( engine)
	{
		case MINSTD_RAND:	return  distribution(minstd_rand_generator);
		case MINSTD_RAND0:	return distribution(minstd_rand0_generator);
		case MT19937:		return distribution(mt19937_rand_generator);
		case MT19937_64:	return distribution(mt19937_64_rand_generator);
		case RANLUX24:		return distribution(ranlux24_rand_generator);
		case RANLUX48:		return distribution(ranlux48_rand_generator);
		case KNUTH_B:		return distribution(knuth_b_rand_generator);
		case DEFAULT_RAND:	return distribution(default_rand_generator); 
		default:			return distribution(default_rand_generator);
	}
	//return distribution(default_rand_generator);
	/*switch( engine)
	{
		case MINSTD_RAND:	printf("MINSTD_RAND\n");	return  distribution(minstd_rand_generator);
		case MINSTD_RAND0:	printf("MINSTD_RAND0\n");	return distribution(minstd_rand0_generator);
		case MT19937:		printf("MT19937\n");		return distribution(mt19937_rand_generator);
		case MT19937_64:	printf("MT19937_64\n");		return distribution(mt19937_64_rand_generator);
		case RANLUX24:		printf("RANLUX24\n");		return distribution(ranlux24_rand_generator);
		case RANLUX48:		printf("RANLUX48\n");		return distribution(ranlux48_rand_generator);
		case KNUTH_B:		printf("KNUTH_B\n");		return distribution(knuth_b_rand_generator);
		case DEFAULT_RAND:	printf("DEFAULT_RAND\n");	return distribution(default_rand_generator); 
	}*/
	
	/*switch( engine)
	{
		case MINSTD_RAND:	std::minstd_rand minstd_rand_generator(seed);					return  distribution(minstd_rand_generator);
		case MINSTD_RAND0:	std::minstd_rand0 minstd_rand0_generator(seed);					return distribution(minstd_rand0_generator);
		case MT19937:		std::mt19937 mt19937_rand_generator(seed);						return distribution(mt19937_rand_generator);
		case MT19937_64:	std::mt19937_64 mt19937_64_rand_generator(seed);				return distribution(mt19937_64_rand_generator);
		case RANLUX24:		std::ranlux24 ranlux24_rand_generator(seed);					return distribution(ranlux24_rand_generator);
		case RANLUX48:		std::ranlux48 ranlux48_rand_generator(seed);					return distribution(ranlux48_rand_generator);
		case KNUTH_B:		std::knuth_b knuth_b_rand_generator(seed);						return distribution(knuth_b_rand_generator);
		case DEFAULT_RAND:		std::default_random_engine default_rand_generator(seed);	return distribution(default_rand_generator); 
	}
	*///return output;
	//
	//switch( engine)
	//{
	//	case MINSTD_RAND:	std::minstd_rand minstd_rand_generator(seed);				output=  distribution(minstd_rand_generator);break;
	//	case MINSTD_RAND0:	std::minstd_rand0 minstd_rand0_generator(seed);				output=  distribution(minstd_rand0_generator);break;
	//	case MT19937:		std::mt19937 mt19937_rand_generator(seed);					output=  distribution(mt19937_rand_generator);break;
	//	case MT19937_64:	std::mt19937_64 mt19937_64_rand_generator(seed);				output=  distribution(mt19937_64_rand_generator);break;
	//	case RANLUX24:		std::ranlux24 ranlux24_rand_generator(seed);					output=  distribution(ranlux24_rand_generator);break;
	//	case RANLUX48:		std::ranlux48 ranlux48_rand_generator(seed);					output=  distribution(ranlux48_rand_generator);break;
	//	case KNUTH_B:		std::knuth_b knuth_b_rand_generator(seed);					output=  distribution(knuth_b_rand_generator);break;
	//	case DEFAULT_RAND:		std::default_random_engine default_rand_generator(seed);		output= distribution(default_rand_generator); break;
	//}
	//return output;
	////enum RAND_GENERATORS		{ DEFAULT_RAND, MINSTD_RAND, MINSTD_RAND0, MT19937,				// Defines the available random number generator engines 
	//							MT19937_64, RANLUX24, RANLUX48, KNUTH_B	};					// ...
}
double randn (double mu, double sigma)
{
  double U1, U2, W, mult;
  static double X1, X2;
  static int call = 0;
 
  if (call == 1)
    {
      call = !call;
      return (mu + sigma * (double) X2);
    }
 
  do
    {
      U1 = -1 + ((double) rand () / RAND_MAX) * 2;
      U2 = -1 + ((double) rand () / RAND_MAX) * 2;
      W = pow (U1, 2) + pow (U2, 2);
    }
  while (W >= 1 || W == 0);
 
  mult = sqrt ((-2 * log (W)) / W);
  X1 = U1 * mult;
  X2 = U2 * mult;
 
  call = !call;
 
  return (mu + sigma * (double) X1);
}
void rand_check(uint repetitions)
{
	int rand_integer_result = 0;
	//uint repetitions = 100000;
	uint rand_min = 1;
	uint rand_max = 10;
	uint rand_range = rand_max - rand_min + 1;
	std::vector<uint> occurrences(rand_range, 0);
	std::vector<uint> rand_values(rand_range, 0);
	std::iota(rand_values.begin(), rand_values.end(), rand_min);
	uint rand_index = 0;
	uint num_rand_engines = RAND_GENERATORS::END_RAND_GENERATORS - 1;
	for(int j = 0; j < num_rand_engines; j++)
	{
		printf("RAND_GENERATORS(j) = %d:\n", RAND_GENERATORS(j));
		for(int i = 0; i < repetitions; i++)
		{
			//rand_integer_result = randi(rand_min, rand_max);
			rand_integer_result = randi(RAND_GENERATORS(j), rand_min, rand_max);
			rand_index = rand_integer_result - rand_min;
			occurrences[rand_index]++;
			if(repetitions <= 100)
			{
				//printf("------> rand_integer_result = %d\n", rand_integer_result);
				//pause_execution();
			}

		}
		for(int i = 0; i < rand_range; i++)
		{
			printf("i = %d: %d occurrences\n", rand_values[i], occurrences[i] );
		}			
	}
	for(int i = 0; i < num_rand_engines; i++)
	{
		printf("RAND_GENERATORS[i] = %d\n", static_cast<RAND_GENERATORS>(i) );
	}
}
template<typename T, typename T2> T max_n( int num_args, T2 arg_1, ...)
{
	T2 largest = arg_1;
	T2 value;
	va_list values;
	va_start( values, arg_1 );
	for( int i = 1; i < num_args; i++ )
	{
		value = va_arg( values, T2 );
		largest = ( largest > value ) ? largest : value;
	}
	va_end(values);
	return (T) largest; 
}
template<typename T, typename T2> T min_n( int num_args, T2 arg_1, ...)
{
	T2 smallest = arg_1;
	T2 value;
	va_list values;
	va_start( values, arg_1 );
	for( int i = 1; i < num_args; i++ )
	{
		value = va_arg( values, T2 );
		smallest = ( smallest < value ) ? smallest : value;
	}
	va_end(values);
	return (T) smallest; 
}
template<typename T> T* sequential_numbers( int start_number, int length )
{
	T* sequential_array = (T*)calloc(length,sizeof(T));
	//std::iota( sequential_array, sequential_array + length, start_number );
	return sequential_array;
}
void bin_2_indexes( int& bin_num, int& t_bin, int& v_bin, int& angular_bin )
{
	// => bin = t_bin + angular_bin * T_BINS + v_bin * ANGULAR_BINS * T_BINS > 0
	while( bin_num - ANGULAR_BINS * T_BINS > 0 )
	{
		bin_num -= ANGULAR_BINS * T_BINS;
		v_bin++;
	}
	// => bin = t_bin + angular_bin * T_BINS > 0
	while( bin_num - T_BINS > 0 )
	{
		bin_num -= T_BINS;
		angular_bin++;
	}
	// => bin = t_bin > 0
	t_bin = bin_num;
}
std::string terminal_response(char* system_command) 
{
	#if defined(_WIN32) || defined(_WIN64)
		FILE* pipe = _popen(system_command, "r");
    #else
		FILE* pipe = popen(system_command, "r");
    #endif
    
    if (!pipe) return "ERROR";
    char buffer[256];
    std::string result;
    while(!feof(pipe)) {
    	if(fgets(buffer, 256, pipe) != NULL)
    		result += buffer;
    }
	#if defined(_WIN32) || defined(_WIN64)
		 _pclose(pipe);
    #else
		 pclose(pipe);
    #endif
   
    return result;
}
std::string terminal_response(const char* system_command) 
{
	#if defined(_WIN32) || defined(_WIN64)
		FILE* pipe = _popen(system_command, "r");
    #else
		FILE* pipe = popen(system_command, "r");
    #endif
    
    if (!pipe) return "ERROR";
    char buffer[256];
    std::string result;
    while(!feof(pipe)) {
    	if(fgets(buffer, 256, pipe) != NULL)
    		result += buffer;
    }
	#if defined(_WIN32) || defined(_WIN64)
		 _pclose(pipe);
    #else
		 pclose(pipe);
    #endif
   
    return result;
}
char((&terminal_response( char* system_command, char(&result)[256]))[256])
{
	#if defined(_WIN32) || defined(_WIN64)
		FILE* pipe = _popen(system_command, "r");
    #else
		FILE* pipe = popen(system_command, "r");
    #endif
    
    if (!pipe) 
		{
			strcpy(result, "ERROR");
			return result;
	}
	strcpy(result, "");
    char buffer[128];
    while(!feof(pipe)) {
    	if(fgets(buffer, 128, pipe) != NULL)
    		sprintf( result, "%s%s", result, buffer );
    }
	#if defined(_WIN32) || defined(_WIN64)
		 _pclose(pipe);
    #else
		 pclose(pipe);
    #endif
	return result;
}
char((&terminal_response( const char* system_command, char(&result)[256]))[256])
{
	#if defined(_WIN32) || defined(_WIN64)
		FILE* pipe = _popen(system_command, "r");
    #else
		FILE* pipe = popen(system_command, "r");
    #endif
    
    if (!pipe) 
		{
			strcpy(result, "ERROR");
			return result;
	}
	strcpy(result, "");
    char buffer[128];
    while(!feof(pipe)) {
    	if(fgets(buffer, 128, pipe) != NULL)
    		sprintf( result, "%s%s", result, buffer );
    }
	#if defined(_WIN32) || defined(_WIN64)
		 _pclose(pipe);
    #else
		 pclose(pipe);
    #endif
	return result;
}
/***********************************************************************************************************************************************************************************************************************/
/************************************************************************************* Console Window Print Statement Functions  ***************************************************************************************/
/***********************************************************************************************************************************************************************************************************************/
void change_text_color(const char* text_color_code, const char* background_color_code, const char* underlining_coding, bool reset)
{
	char color_command[256];
	if( !reset )
		sprintf(color_command, "%s \"\033[%s;%s%sm\"", BASH_ECHO_CMD, text_color_code, background_color_code, underlining_coding );
	else
		sprintf(color_command, "%s \"\033[m\"", BASH_ECHO_CMD);
	system(color_command);
}
std::string change_text_color_cmd(const char* text_color_code, const char* background_color_code, const char* underlining_coding, bool reset)
{
	char color_command[256];
	if( !reset )
		sprintf(color_command, "%s \"\033[%s;%s%sm\"", BASH_ECHO_CMD, text_color_code, background_color_code, underlining_coding );
	else
		sprintf(color_command, "%s \"\033[m\"", BASH_ECHO_CMD);
	return std::string(color_command);
}
std::string color_encoding_statement(const char* text_color_code, const char* background_color_code, const char* underlining_coding )
{
	char color_encoding[512];
	sprintf(color_encoding, "%s%s;%s%sm", OPEN_COLOR_ESCAPE_SEQ, text_color_code, background_color_code, underlining_coding );
	std::string statement_str(color_encoding);
	return statement_str;
}
std::string colored_text(const char* statement, const char* text_color_code, const char* background_color_code, const char* underlining_coding )
{
	char color_command[512];
	std::string color_encoding = color_encoding_statement(text_color_code, background_color_code, underlining_coding );
	sprintf(color_command, "%s%s%s", color_encoding.c_str(), statement, CLOSE_COLOR_ESCAPE_SEQ );
	std::string statement_str(color_command);
	return statement_str;
}
std::string colored_text(std::string statement, const char* text_color_code, const char* background_color_code, const char* underlining_coding )
{
	char color_command[256];
	std::string color_encoding = color_encoding_statement(text_color_code, background_color_code, underlining_coding );
	sprintf(color_command, "%s%s%s", color_encoding.c_str(), statement.c_str(), CLOSE_COLOR_ESCAPE_SEQ );
	std::string statement_str(color_command);
	return statement_str;
}
std::string echo_statement(const char* statement, const char* text_color_code, const char* background_color_code, const char* underlining_coding )
{
	char color_command[512];
	std::string color_encoding = color_encoding_statement(text_color_code, background_color_code, underlining_coding );
	sprintf(color_command, "%s \"%s%s%s\"", BASH_ECHO_CMD, color_encoding.c_str(), statement, CLOSE_COLOR_ESCAPE_SEQ);
	std::string echo_command(color_command);
	return echo_command;
}
std::string echo_statement(std::string statement, const char* text_color_code, const char* background_color_code, const char* underlining_coding )
{
	char color_command[512];
	std::string color_encoding = color_encoding_statement(text_color_code, background_color_code, underlining_coding );
	sprintf(color_command, "%s \"%s%s%s\"", BASH_ECHO_CMD, color_encoding.c_str(), statement.c_str(), CLOSE_COLOR_ESCAPE_SEQ);
	std::string echo_command(color_command);
	return echo_command;
}
void print_colored_text(const char* statement, const char* text_color_code, const char* background_color_code, const char* underlining_coding )
{
	std::string echo_command = echo_statement(statement, text_color_code, background_color_code, underlining_coding );
	system(echo_command.c_str());
}
void print_colored_text(std::string statement, const char* text_color_code, const char* background_color_code, const char* underlining_coding )
{
	std::string echo_command = echo_statement(statement, text_color_code, background_color_code, underlining_coding );
	system(echo_command.c_str());
}
void print_section_separator(const char separation_char, const char* text_color_code, const char* background_color_code, const char* underlining_coding )
{
	std::string section_separator_str(CONSOLE_WINDOW_WIDTH, separation_char);
	std::string statement_colored = colored_text(section_separator_str, text_color_code, background_color_code, underlining_coding );
	print_colored_text(statement_colored, text_color_code, background_color_code, underlining_coding );
}
void print_section_header( const char* statement, const char separation_char, const char* separator_text_color_code, const char* text_color_code, const char* background_color_code, const char* underlining_coding )
{
	char header_output[256];
	std::string header_str(statement);
	size_t length = strlen(statement), index = 0, max_line_length = 70, num_dashes, leading_dashes, trailing_dashes, line_length;
	print_section_separator(separation_char, separator_text_color_code, background_color_code, underlining_coding );
	if(separation_char == MAJOR_SECTION_SEPARATOR)
		print_section_separator(separation_char, separator_text_color_code, background_color_code, underlining_coding );	
	while( index < length )
	{
		//i = 0;
		line_length = min(static_cast<int>(length - index), static_cast<int>(max_line_length));
		if( line_length < length - index )
			while( statement[index + line_length] != ' ' )
				line_length--;
		num_dashes = CONSOLE_WINDOW_WIDTH - line_length - 2;
		leading_dashes = num_dashes / 2;		
		trailing_dashes = num_dashes - leading_dashes;
		
		std::string leading_dashes_str(leading_dashes, separation_char);
		std::string trailing_dashes_str(trailing_dashes, separation_char);
		leading_dashes_str.append(" ");
		trailing_dashes_str.insert(0, " ");
		std::string leading_dashes_str_colored = colored_text(leading_dashes_str, separator_text_color_code, background_color_code, underlining_coding );
		std::string trailing_dashes_str_colored = colored_text(trailing_dashes_str, separator_text_color_code, background_color_code, underlining_coding );
		std::string header_substr = header_str.substr (index, line_length);
		std::string header_substr_colored = colored_text(header_substr, text_color_code, background_color_code, underlining_coding );
		sprintf(header_output, "%s \"%s%s%s\"", BASH_ECHO_CMD, leading_dashes_str_colored.c_str(), header_substr_colored.c_str(), trailing_dashes_str_colored.c_str() );
		system(header_output);
		index += line_length;
	}
	if(separation_char == MAJOR_SECTION_SEPARATOR)
		print_section_separator(separation_char, separator_text_color_code, background_color_code, underlining_coding );	
	print_section_separator(separation_char, separator_text_color_code, background_color_code, underlining_coding );
}
void print_section_exit( const char* statement, const char* leading_statement_chars, const char* separator_text_color_code, const char* text_color_code, const char* background_color_code, const char* underlining_coding )
{
	char section_exit_output[256];
	size_t length = strlen(statement);
	size_t num_leading_chars = strlen(leading_statement_chars);
	size_t index = 0, line_length, max_line_length = CONSOLE_WINDOW_WIDTH - 10;
	std::string section_exit_str(statement);	
	std::string leading_spaces(num_leading_chars, ' ');
	std::string leading_chars_str(leading_statement_chars);	
	leading_chars_str.append(" ");		
	std::string leading_chars_str_colored = colored_text(leading_chars_str, separator_text_color_code, background_color_code, underlining_coding );
	while( index < length )
	{
		line_length = min(static_cast<int>(length - index), static_cast<int>(max_line_length));
		if( line_length < length - index )
			while( statement[index + line_length] != ' ' )
				line_length--;
		std::string section_exit_substr = section_exit_str.substr (index, line_length);
		std::string section_exit_substr_colored = colored_text(section_exit_substr, text_color_code, background_color_code, underlining_coding );
		sprintf(section_exit_output, "%s \"%s%s\"", BASH_ECHO_CMD, leading_chars_str_colored.c_str(), section_exit_substr_colored.c_str());		
		leading_chars_str_colored = colored_text(leading_spaces, separator_text_color_code, background_color_code, underlining_coding );	
		system(section_exit_output);
		index += line_length;
	}
	puts("");
}
template<typename T> char print_format_identification( T variable)
{
	std::string typeid_str(typeid(variable).name());
	char type_id = typeid_str.at(typeid_str.length() - 1);
	if(type_id == 'c' && typeid_str.length() > 1 )
		return 's';
	else
		return type_id;
}
template<typename T> void print_labeled_value(const char* statement, T value, const char* statement_color_code, const char* value_color_code, const char* background_color_code, const char* underlining_coding )
{
	char value_string[512];
	char type_identifier = print_format_identification(value);
	if(type_identifier == BOOL_ID_CHAR || type_identifier == INT_ID_CHAR)
		sprintf(value_string, "%d", value);
	if(type_identifier == CHAR_ID_CHAR)
		sprintf(value_string, "%c", value);
	if(type_identifier == FLOAT_ID_CHAR)
		sprintf(value_string, "%6.6lf", value);
	if(type_identifier == STRING_ID_CHAR)
		sprintf(value_string, "%s", value);
	std::string value_string_colored = colored_text(value_string, value_color_code, background_color_code, underlining_coding );	
	sprintf(print_statement, "%s %s", statement, value_string_colored.c_str() );
	print_colored_text(print_statement, statement_color_code, background_color_code, underlining_coding );			
}
void print_multiline_bash_results(const char* command, const char* text_color_code, const char* background_color_code, const char* underlining_coding )
{
	std::string echo_command = echo_statement("${i}", text_color_code, background_color_code, underlining_coding );
	sprintf(system_command, "OIFS=$IFS; IFS=$'\\n'; for i in $(%s); do %s; done; IFS=$OIFS", command, echo_command.c_str());
	system(system_command);
}
/***********************************************************************************************************************************************************************************************************************/
/*********************************************************************************************** Device Helper Functions ***********************************************************************************************/
/***********************************************************************************************************************************************************************************************************************/
cudaError_t CUDA_error_check( char* error_statement )
{
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) 
		printf("%s %s\n", error_statement, cudaGetErrorString(cudaStatus));						  

	return cudaStatus;
}
cudaError_t CUDA_error_check_and_sync_device( char* error_statement, char* sync_statement )
{
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) 
		printf("%s %s\n", error_statement, cudaGetErrorString(cudaStatus));						  
	
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching %s Kernel!\n", cudaStatus, sync_statement);

	return cudaStatus;
}
/***********************************************************************************************************************************************************************************************************************/
/************************************************************************************ Testing Functions and Functions in Development ***********************************************************************************/
/***********************************************************************************************************************************************************************************************************************/
void NTVS_timing_analysis()
{
	//import_image( O*& import_into, char* filename );
	UINT Nk_max = 10;
	float* FBP_image_copy = (float*)calloc(NUM_VOXELS, sizeof(float));
	float* perturbed_FBP_image_copy = (float*)calloc(NUM_VOXELS, sizeof(float));
	std::copy(FBP_image_h, FBP_image_h + NUM_VOXELS, FBP_image_copy);
	std::copy(FBP_image_h, FBP_image_h + NUM_VOXELS, perturbed_FBP_image_copy);
	std::vector<double> execution_times_NTVS_performance_tests;
	std::vector<float> final_TVs_NTVS_performance_tests;
	UINT initial_TVS_repetitions = TVS_REPETITIONS;
	clock_t begin_NTVS=0;
	double execution_time_NTVS = 0;
	UINT repeat_NTVS = 1000;
	allocate_perturbation_arrays(false);
	generate_TVS_eta_sequence();
	float final_TV_value=0.0;
	final_TV_value = calculate_total_variation(perturbed_FBP_image_copy, DONT_PRINT_TV);	// Calculate total variation of unperturbed image
	final_TVs_NTVS_performance_tests.push_back(final_TV_value);
	//******************************
	sprintf(print_statement, "Performing NTVS timing test with TV check and Nk = [1,10] repeated %d times", repeat_NTVS );
	print_colored_text(print_statement, RED_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );
	for( int Nk = 1; Nk <= Nk_max; Nk++ )
	{
		TVS_REPETITIONS = Nk;
		begin_NTVS=0;
		sprintf(print_statement, "for NTVS timing tests with TV check and Nk = %d repeated %d times", Nk, repeat_NTVS );
		timer( START, begin_NTVS, print_statement);	
		for( int i = 0; i < repeat_NTVS; i++ )
		{
			std::copy(FBP_image_h, FBP_image_h + NUM_VOXELS, perturbed_FBP_image_copy);
			iteratively_perturb_image( perturbed_FBP_image_copy, hull_h, 0);
		}
		execution_time_NTVS = timer( STOP, begin_NTVS, print_statement);
		execution_times_NTVS_performance_tests.push_back(execution_time_NTVS);
		final_TV_value = calculate_total_variation(perturbed_FBP_image_copy, DONT_PRINT_TV);	// Calculate total variation of unperturbed image
		final_TVs_NTVS_performance_tests.push_back(final_TV_value);
		
	}
	//******************************
	sprintf(print_statement, "Performing NTVS timing test without TV check and Nk = [1,10] repeated %d times", repeat_NTVS );
	print_colored_text(print_statement, RED_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );
	for( int Nk = 1; Nk <= Nk_max; Nk++ )
	{
		TVS_REPETITIONS = Nk;
		begin_NTVS=0;
		sprintf(print_statement, "for NTVS timing tests without TV check and Nk = %d repeated %d times", Nk, repeat_NTVS );
		timer( START, begin_NTVS, print_statement);	
		for( int i = 0; i < repeat_NTVS; i++ )
		{
			std::copy(FBP_image_h, FBP_image_h + NUM_VOXELS, perturbed_FBP_image_copy);
			iteratively_perturb_image_unconditional( perturbed_FBP_image_copy, hull_h, 0);
		}
		execution_time_NTVS = timer( STOP, begin_NTVS, print_statement);
		execution_times_NTVS_performance_tests.push_back(execution_time_NTVS);
		final_TV_value = calculate_total_variation(perturbed_FBP_image_copy, DONT_PRINT_TV);	// Calculate total variation of unperturbed image
		final_TVs_NTVS_performance_tests.push_back(final_TV_value);
	}
	TVS_REPETITIONS = initial_TVS_repetitions;
	//******************************
	char filename[256];
	//sprintf(cp_command, "%s %s%s//* %s%s", BASH_COPY_DIR, OUTPUT_DIRECTORY, OUTPUT_FOLDER_UNIQUE, CURRENT_RECON_DIR, OUTPUT_FOLDER_UNIQUE);		
	
	//sprintf(filename, "%s%s//NTVS_time_performance_comparison.txt", OUTPUT_DIRECTORY, OUTPUT_FOLDER_UNIQUE);		
	//sprintf(print_statement, "writing NTVS performance comparison results to NTVS_time_performance_comparison.txt");	
	//std::ofstream output_file;
	//output_file.open(filename);	
	//for(int i = 0; i < execution_times_NTVS_performance_tests.size(); i++)
	//	output_file << execution_times_NTVS_performance_tests[i] << " ";
	
	sprintf(print_statement, "writing NTVS performance comparison results to NTVS_time_performance_comparison.txt" );
	print_colored_text(print_statement, RED_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );
	sprintf(filename, "%s//%s//NTVS_time_performance_comparison.txt", CURRENT_RECON_DIR, OUTPUT_FOLDER_UNIQUE);	
	FILE* output_file2 = fopen( filename, "w" );
	for( int i = 0; i < execution_times_NTVS_performance_tests.size(); i++ )
		fprintf(output_file2, "%6.6lf\n",	execution_times_NTVS_performance_tests[i] );	// 1
	fclose(output_file2);	
	sprintf(print_statement, "finished writing NTVS performance comparison results to disk" );
	print_colored_text(print_statement, RED_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );

	//******************************
	sprintf(print_statement, "writing NTVS performance comparison final TV values to NTVS_time_performance_comparison_TVs.txt" );
	print_colored_text(print_statement, RED_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );
	sprintf(filename, "%s//%s//NTVS_time_performance_comparison_final_TVs.txt",  CURRENT_RECON_DIR, OUTPUT_FOLDER_UNIQUE);	
	FILE* output_file3 = fopen( filename, "w" );
	for( int i = 0; i < final_TVs_NTVS_performance_tests.size(); i++ )
		fprintf(output_file3, "%6.6lf\n",	final_TVs_NTVS_performance_tests[i] );	// 1
	fclose(output_file3);	
	sprintf(print_statement, "finished writing NTVS performance comparison results to disk" );
	print_colored_text(print_statement, RED_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );
	//******************************
	//char TV_x_values_path[256];
	//sprintf(TV_x_values_path, "%s%s//%s.txt", OUTPUT_DIRECTORY, OUTPUT_FOLDER_UNIQUE, TV_MEASUREMENTS_FILENAME);	
	//sprintf(print_statement, "Writing %d total variation (TV) measurements to:\n", TV_x_values.size());
	//print_colored_text(print_statement, YELLOW_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );
	//print_colored_text(TV_x_v		alues_path, LIGHT_PURPLE_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );
	//
	//FILE* TV_x_values_file = fopen( TV_x_values_path, "w" );
	//for( int i = 0; i < TV_x_values.size(); i++ )
	//	fprintf(TV_x_values_file, "%6.6lf\n",	TV_x_values[i] );	// 1
	//fclose(TV_x_values_file);	
	deallocate_perturbation_arrays(false);	
}
void test_func()
{
	sprintf( print_statement, "Performing testing of functions currently in development");
	print_section_header( print_statement, MAJOR_SECTION_SEPARATOR, RED_TEXT, RED_TEXT, WHITE_BACKGROUND, DONT_UNDERLINE_TEXT );
	/********************************************************************************************************************************************************/
	/* Perform testing of functions currently in development																						*/
	/********************************************************************************************************************************************************/		
	//char dir[] = "C://Users//Blake//Documents//Education//Research//pCT//pCT_data//reconstruction_data//CTP404_Sensitom_4M//";
	//char folder[] = "Simulated//";
	//char image_fname[] = "FBP_median_filtered.txt";	  
	//char path[256];

	//sprintf(path, "%s%s%s", dir, folder, image_fname);		
	//
	//float* image = (float*)calloc(NUM_VOXELS, sizeof(float));
	//float* image_med = (float*)calloc(NUM_VOXELS, sizeof(float));
	////import_image( image, path );
	//import_text_image( image, path );
	//array_2_disk( FBP_AFTER_FILENAME, dir, folder, image, COLUMNS, ROWS, SLICES, NUM_VOXELS, true );
	//median_filter_2D( image, image_med, FBP_MED_FILTER_RADIUS );
	//sprintf(print_statement, "Median filtering of FBP complete");		
	//print_colored_text( print_statement, RED_TEXT, BLACK_BACKGROUND, DONT_UNDERLINE_TEXT );	
	//
	////std::copy(FBP_median_filtered_h, FBP_median_filtered_h + NUM_VOXELS, FBP_image_h);
	////float* temp = &image[0];
	//image = image_med;
	////free(temp);
	////T* temp = &input_image;
	////input_image = median_filtered_image;
	////FBP_image_h = FBP_image_filtered_h;
	////T* temp = &input_image;
	////input_image = median_filtered_image;
	//
	////import_image( O*& import_into, path );
	//array_2_disk( FBP_IMAGE_FILTER_FILENAME, dir, folder, image, COLUMNS, ROWS, SLICES, NUM_VOXELS, true );
	//		
		
	//char filename[256];
	//char* name = "FBP_med7";
	//sprintf( filename, "%s%s/%s%s", OUTPUT_DIRECTORY, OUTPUT_FOLDER_UNIQUE, name, ".bin" );
	//float* image = (float*)calloc( NUM_VOXELS, sizeof(float));
	//import_image( image, filename );
	//array_2_disk( name, OUTPUT_DIRECTORY, OUTPUT_FOLDER_UNIQUE, image, COLUMNS, ROWS, SLICES, NUM_VOXELS, true );
	//read_config_file();
	//double voxels[4] = {1,2,3,4};
	//std::copy( hull_h, hull_h + NUM_VOXELS, x_h );
	//std::function<double(int, int)> fn1 = my_divide;                    // function
	//int x = 2;
	//int y = 3;
	//cout << func_pass_test(x,y, fn1) << endl;
	//std::function<double(double, double)> fn2 = my_divide2;                    // function
	//double x2 = 2;
	//double y2 = 3;
	//cout << func_pass_test(x2,y2, fn2) << endl;
	//std::function<int(int)> fn2 = &half;                   // function pointer
	//std::function<int(int)> fn3 = third_t();               // function object
	//std::function<int(int)> fn4 = [](int x){return x/4;};  // lambda expression
	//std::function<int(int)> fn5 = std::negate<int>();      // standard function object
	//create_MLP_test_image();
	//array_2_disk( "MLP_image_init", OUTPUT_DIRECTORY, OUTPUT_FOLDER_UNIQUE, MLP_test_image_h, MLP_IMAGE_COLUMNS, MLP_IMAGE_ROWS, MLP_IMAGE_SLICES, MLP_IMAGE_VOXELS, true );
	//MLP_test();
	//array_2_disk( "MLP_image", OUTPUT_DIRECTORY, OUTPUT_FOLDER_UNIQUE, MLP_test_image_h, MLP_IMAGE_COLUMNS, MLP_IMAGE_ROWS, MLP_IMAGE_SLICES, MLP_IMAGE_VOXELS, true );
	//double* x = (double*) calloc(4, sizeof(double) );
	//double* y = (double*) calloc(4, sizeof(double) );
	//double* z = (double*) calloc(4, sizeof(double) );

	//double* x_d, *y_d, *z_d;
	////sinogram_filtered_h = (float*) calloc( NUM_BINS, sizeof(float) );
	//cudaMalloc((void**) &x_d, 4*sizeof(double));
	//cudaMalloc((void**) &y_d, 4*sizeof(double));
	//cudaMalloc((void**) &z_d, 4*sizeof(double));

	//cudaMemcpy( x_d, x, 4*sizeof(double), cudaMemcpyHostToDevice);
	//cudaMemcpy( y_d, y, 4*sizeof(double), cudaMemcpyHostToDevice);
	//cudaMemcpy( z_d, z, 4*sizeof(double), cudaMemcpyHostToDevice);

	//dim3 dimBlock( 1 );
	//dim3 dimGrid( 1 );   	
	//test_func_device<<< dimGrid, dimBlock >>>( x_d, y_d, z_d );

	//cudaMemcpy( x, x_d, 4*sizeof(double), cudaMemcpyDeviceToHost);
	//cudaMemcpy( y, y_d, 4*sizeof(double), cudaMemcpyDeviceToHost);
	//cudaMemcpy( z, z_d, 4*sizeof(double), cudaMemcpyDeviceToHost);

	//for( unsigned int i = 0; i < 4; i++)
	//{
	//	printf("%3f\n", x[i] );
	//	printf("%3f\n", y[i] );
	//	printf("%3f\n", z[i] );
	//	//cout << x[i] << endl; // -8.0
	//	//cout << y[i] << endl;
	//	//cout << z[i] << endl;
	//}
}
