//#ifndef PCT_RECONSTRUCTION_CU
//#define PCT_RECONSTRUCTION_CU
#pragma once
/***********************************************************************************************************************************************************************************************************************/
/********************************************************************************** Proton CT Preprocessing and Image Reconstruction Code ******************************************************************************/
/***********************************************************************************************************************************************************************************************************************/
#include "pCT_Reconstruction.h"

// Includes, CUDA project
//#include <cutil_inline.h>

// Includes, kernels
//#include "pCT_Reconstruction_GPU.cu"
/***********************************************************************************************************************************************************************************************************************/
/********************************************************************************************** Host functions declarations ********************************************************************************************/
/***********************************************************************************************************************************************************************************************************************/

// Execution Control Functions
bool is_bad_angle( const int );	// Just for use with Micah's simultated data
void timer( bool );
void pause_execution();
void exit_program_if( bool );

// Memory transfers and allocations/deallocations
void initial_processing_memory_clean();
void resize_vectors( const int );
void shrink_vectors( const int );
void allocations( const int );
void reallocations( const int );
void post_cut_memory_clean(); 

// Image Initialization/Construction Functions
template<typename T> void initialize_host_image( T*& );
template<typename T> void add_ellipse( T*&, int, double, double, double, double, T );
template<typename T> void add_circle( T*&, int, double, double, double, T );

// Preprocessing setup and initializations 
void write_run_settings();
void assign_SSD_positions();
void initializations();
void count_histories();	
void count_histories_old();
void count_histories_v0();
void count_histories_v1();
void reserve_vector_capacity(); 

// Preprocessing functions
void read_energy_responses( const int, const int, const int );
void read_data_chunk( const int, const int, const int );
void read_data_chunk_old( const int, const int, const int );
void read_data_chunk_v0( const int, const int, const int );
void read_data_chunk_v1( const int, const int, const int );
void apply_tu_shifts( unsigned int );
void convert_mm_2_cm( unsigned int );
void recon_volume_intersections( const int );
void binning( const int );
void calculate_means();
void initialize_stddev();
void sum_squared_deviations( const int, const int );
void calculate_standard_deviations();
void statistical_cuts( const int, const int );
void initialize_sinogram();
void construct_sinogram();
void FBP();
void FBP_image_2_hull();
void filter();
void backprojection();

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
void hull_selection();
template<typename T, typename T2> void averaging_filter( T*&, T2*& );

// MLP: IN DEVELOPMENT
void create_MLP_test_image();	
void MLP_test();
void MLP_test2();
void MLP();
void MLP2();
//void MLP3();
//void MLP( std::vector<float>, std::vector<float>, std::vector<float>, std::vector<float>, std::vector<float>, std::vector<float>, std::vector<float>, std::vector<float>, std::vector<float>, std::vector<float>, std::vector<float>, bool*, float*);
template<typename O> bool find_MLP_endpoints( O*&, double, double, double, double, double, double&, double&, double&, int&, int&, int&, bool);
int find_MLP_path( int*&, double*&, double, double, double, double, double, double, double, double, double, double, int, int, int );
double mean_chord_length( double, double, double, double, double, double );

// Image Reconstruction
void define_initial_iterate();
void create_hull_image_hybrid();
template< typename T, typename L, typename R> T discrete_dot_product( L*&, R*&, int*, unsigned int );
template< typename A, typename X> double update_vector_multiplier( double, A*&, X*&, int*, unsigned int );
template< typename A, typename X> void update_iterate( double, A*&, X*&, int*, unsigned int );
// uses mean chord length for each element of ai instead of individual chord lengths
template< typename T, typename RHS> T scalar_dot_product( double, RHS*&, int*, unsigned int );
double scalar_dot_product2( double, float*&, int*, unsigned int );
template< typename X> double update_vector_multiplier2( double, double, X*&, int*, unsigned int );
double update_vector_multiplier22( double, double, float*&, int*, unsigned int );
template< typename X> void update_iterate2( double, double, X*&, int*, unsigned int );
void update_iterate22( double, double, float*&, int*, unsigned int );
template<typename X, typename U> void calculate_update( double, double, X*&, U*&, int*, unsigned int );
template<typename X, typename U> void update_iterate3( X*&, U*& );

// Write arrays/vectors to file(s)
void binary_2_ASCII();
template<typename T> void array_2_disk( char*, const char*, const char*, T*, const int, const int, const int, const int, const bool );
template<typename T> void vector_2_disk( char*, const char*, const char*, std::vector<T>, const int, const int, const int, const bool );
template<typename T> void t_bins_2_disk( FILE*, const std::vector<int>&, const std::vector<T>&, const BIN_ANALYSIS_TYPE, const int );
template<typename T> void bins_2_disk( const char*, const std::vector<int>&, const std::vector<T>&, const BIN_ANALYSIS_TYPE, const BIN_ANALYSIS_FOR, const BIN_ORGANIZATION, ... );
template<typename T> void t_bins_2_disk( FILE*, int*&, T*&, const int, const BIN_ANALYSIS_TYPE, const BIN_ORGANIZATION, int );
template<typename T> void bins_2_disk( const char*, int*&, T*&, const int, const BIN_ANALYSIS_TYPE, const BIN_ANALYSIS_FOR, const BIN_ORGANIZATION, ... );
FILE* create_MLP_path_file( char* );
//template<typename T> void path_data_2_disk(char*, FILE*, int, T(&)[MAX_INTERSECTIONS], bool );
template<typename T> void path_data_2_disk(char*, FILE*, int, int*, T*&, bool );

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

// Host helper functions		
template< typename T, typename T2> T max_n( int, T2, ...);
template< typename T, typename T2> T min_n( int, T2, ...);
template<typename T> T* sequential_numbers( int, int );
void bin_2_indexes( int, int&, int&, int& );

// New routine test functions
void test_func();
void test_func2( std::vector<int>&, std::vector<double>&);

/***********************************************************************************************************************************************************************************************************************/
/****************************************************************************************** Device (GPU) function declarations *****************************************************************************************/
/***********************************************************************************************************************************************************************************************************************/

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
__global__ void SM_edge_detection_GPU( int*, int* );
__global__ void SM_edge_detection_GPU_2( int*, int* );
__global__ void carve_differences( int*, int* );
template<typename H, typename D> __global__ void averaging_filter_GPU( H*, D*, bool );
template<typename D> __global__ void apply_averaging_filter_GPU( D*, D* );

// MLP: IN DEVELOPMENT
template<typename O> __device__ bool find_MLP_endpoints_GPU( O*&, double, double, double, double, double, double&, double&, double&, int&, int&, int&, bool);
__device__ int find_MLP_path_GPU( int*&, double*&, double, double, double, double, double, double, double, double, double, double, int, int, int );
__device__ void MLP_GPU();

// Image Reconstruction
__global__ void create_hull_image_hybrid_GPU( bool*&, float*& );
//template< typename X> __device__ double update_vector_multiplier2( double, double, X*&, int*, int );
__device__ double scalar_dot_product_GPU_2( double, float*&, int*, int );
__device__ double update_vector_multiplier_GPU_22( double, double, float*&, int*, int );
//template< typename X> __device__ void update_iterate2( double, double, X*&, int*, int );
__device__ void update_iterate_GPU_22( double, double, float*&, int*, int );

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

// Device helper functions

// New routine test functions
__global__ void test_func_GPU( int* );
__global__ void test_func_device( double*, double*, double* );

/***********************************************************************************************************************************************************************************************************************/
/***************************************************************************************************** Program Main ****************************************************************************************************/
/***********************************************************************************************************************************************************************************************************************/
int main(int argc, char** argv)
{
	if( RUN_ON )
	{
		/********************************************************************************************************************************************************/
		/* Start the execution timing clock																														*/
		/********************************************************************************************************************************************************/
		timer( START );
		/********************************************************************************************************************************************************/
		/* Initialize hull detection images and transfer them to the GPU (performed if SC_ON, MSC_ON, or SM_ON is true)											*/
		/********************************************************************************************************************************************************/
		hull_initializations();
		MSC_counts_h = (int*) calloc( NUM_VOXELS, sizeof(int));
		cudaMemcpy( MSC_counts_h,	MSC_counts_d,	NUM_VOXELS * sizeof(int), cudaMemcpyDeviceToHost );	
		array_2_disk( "x_MSC_init", OUTPUT_DIRECTORY, OUTPUT_FOLDER, MSC_counts_h, COLUMNS, ROWS, SLICES, NUM_VOXELS, true );
		/********************************************************************************************************************************************************/
		/* Read the u-coordinates of the detector planes from the config file, allocate and	initialize statistical data arrays, and count the number of 		*/
		/* histories per file, projection, gantry angle, scan, and total.																						*/
		/********************************************************************************************************************************************************/		
		if( DATA_FORMAT == OLD_FORMAT )
			assign_SSD_positions();		// Read the detector plane u-coordinates from config file
		initializations();				// allocate and initialize host and GPU memory for statistical
		count_histories();				// count the number of histories per file, per scan, total, etc.
		reserve_vector_capacity();		// Reserve enough memory so vectors don't grow into another reserved memory space, wasting time since they must be moved
		/********************************************************************************************************************************************************/
		/* Reading the 16 energy detector responses for each of the 5 stages and generate single energy response for each history								*/
		/********************************************************************************************************************************************************/
		int start_file_num = 0, end_file_num = 0, histories_to_process = 0;
		//while( start_file_num != NUM_FILES )
		//{
		//	while( end_file_num < NUM_FILES )
		//	{
		//		if( histories_to_process + histories_per_file[end_file_num] < MAX_GPU_HISTORIES )
		//			histories_to_process += histories_per_file[end_file_num];
		//		else
		//			break;
		//		end_file_num++;
		//	}
		//	//read_energy_responses( histories_to_process, start_file_num, end_file_num );
		//	start_file_num = end_file_num;
		//	histories_to_process = 0;
		//}
		/********************************************************************************************************************************************************/
		/* Iteratively Read and Process Data One Chunk at a Time. There are at Most	MAX_GPU_HISTORIES Per Chunk (i.e. Iteration). On Each Iteration:			*/
		/*	(1) Read data from file																																*/
		/*	(2) Determine which histories traverse the reconstruction volume and store this	information in a boolean array										*/
		/*	(3) Determine which bin each history belongs to																										*/
		/*	(4) Use the boolean array to determine which histories to keep and then push the intermediate data from these histories onto the permanent 			*/
		/*		storage std::vectors																															*/
		/*	(5) Free up temporary host/GPU array memory allocated during iteration																				*/
		/********************************************************************************************************************************************************/
		puts("Iteratively reading data from hard disk");
		puts("Removing proton histories that don't pass through the reconstruction volume");
		puts("Binning the data from those that did...");
		start_file_num = 0, end_file_num = 0, histories_to_process = 0;
		while( start_file_num != NUM_FILES )
		{
			while( end_file_num < NUM_FILES )
			{
				if( histories_to_process + histories_per_file[end_file_num] < MAX_GPU_HISTORIES )
					histories_to_process += histories_per_file[end_file_num];
				else
					break;
				end_file_num++;
			}
			read_data_chunk( histories_to_process, start_file_num, end_file_num );
			recon_volume_intersections( histories_to_process );
			binning( histories_to_process );
			hull_detection( histories_to_process );
			initial_processing_memory_clean();
			start_file_num = end_file_num;
			histories_to_process = 0;
		}
		if( COUNT_0_WEPLS )
			std::cout << "Histories with WEPL = 0 : " << zero_WEPL << std::endl;
		puts("Data reading complete.");
		printf("%d out of %d (%4.2f%%) histories traversed the reconstruction volume\n", recon_vol_histories, total_histories, (double) recon_vol_histories / total_histories * 100  );
		exit_program_if( EXIT_AFTER_BINNING );
		/********************************************************************************************************************************************************/
		/* Reduce vector capacities to their size, the number of histories remaining afterhistories that didn't intersect reconstruction volume were ignored	*/																				
		/********************************************************************************************************************************************************/
		shrink_vectors( recon_vol_histories );
		/********************************************************************************************************************************************************/
		/* Perform thresholding on MSC and SM hulls and write all hull images to file																			*/																					
		/********************************************************************************************************************************************************/
		hull_detection_finish();
		exit_program_if( EXIT_AFTER_HULLS );
		/********************************************************************************************************************************************************/
		/* Calculate the mean WEPL, relative ut-angle, and relative uv-angle for each bin and count the number of histories in each bin							*/											
		/********************************************************************************************************************************************************/
		calculate_means();
		initialize_stddev();
		/********************************************************************************************************************************************************/
		/* Calculate the standard deviation in WEPL, relative ut-angle, and relative uv-angle for each bin.  Iterate through the valid history std::vectors one	*/
		/* chunk at a time, with at most MAX_GPU_HISTORIES per chunk, and calculate the difference between the mean WEPL and WEPL, mean relative ut-angle and	*/ 
		/* relative ut-angle, and mean relative uv-angle and relative uv-angle for each history. The standard deviation is then found by calculating the sum	*/
		/* of these differences for each bin and dividing it by the number of histories in the bin 																*/
		/********************************************************************************************************************************************************/
		puts("Calculating the cumulative sum of the squared deviation in WEPL and relative ut/uv angles over all histories for each bin...");
		int remaining_histories = recon_vol_histories;
		int start_position = 0;
		while( remaining_histories > 0 )
		{
			if( remaining_histories > MAX_GPU_HISTORIES )
				histories_to_process = MAX_GPU_HISTORIES;
			else
				histories_to_process = remaining_histories;
			sum_squared_deviations( start_position, histories_to_process );
			remaining_histories -= MAX_GPU_HISTORIES;
			start_position		+= MAX_GPU_HISTORIES;
		} 
		calculate_standard_deviations();
		/********************************************************************************************************************************************************/
		/* Allocate host memory for the sinogram, initialize it to zeros, allocate memory for it on the GPU, then transfer the initialized sinogram to the GPU	*/
		/********************************************************************************************************************************************************/
		initialize_sinogram();
		/********************************************************************************************************************************************************/
		/* Iterate through the valid history vectors one chunk at a time, with at most MAX_GPU_HISTORIES per chunk, and perform statistical cuts				*/
		/********************************************************************************************************************************************************/
		puts("Performing statistical cuts...");
		remaining_histories = recon_vol_histories, start_position = 0;
		while( remaining_histories > 0 )
		{
			if( remaining_histories > MAX_GPU_HISTORIES )
				histories_to_process = MAX_GPU_HISTORIES;
			else
				histories_to_process = remaining_histories;
			statistical_cuts( start_position, histories_to_process );
			remaining_histories -= MAX_GPU_HISTORIES;
			start_position		+= MAX_GPU_HISTORIES;
		}
		puts("Statistical cuts complete.");
		printf("%d out of %d (%4.2f%%) histories also passed statistical cuts\n", post_cut_histories, total_histories, (double) post_cut_histories / total_histories * 100  );
		/********************************************************************************************************************************************************/
		/* Free host memory for bin number array, free GPU memory for the statistics arrays, and shrink svectors to the number of histories that passed cuts	*/
		/********************************************************************************************************************************************************/		
		post_cut_memory_clean();
		resize_vectors( post_cut_histories );
		shrink_vectors( post_cut_histories );
		exit_program_if( EXIT_AFTER_CUTS );
		/********************************************************************************************************************************************************/
		/* Recalculate the mean WEPL for each bin using	the histories remaining after cuts and use these to produce the sinogram								*/
		/********************************************************************************************************************************************************/
		construct_sinogram();
		exit_program_if( EXIT_AFTER_SINOGRAM );
		/********************************************************************************************************************************************************/
		/* Perform filtered backprojection and write FBP hull to disk																							*/
		/********************************************************************************************************************************************************/
		if( FBP_ON )
			FBP();
		exit_program_if( EXIT_AFTER_FBP );
		hull_selection();
		define_initial_iterate();
		MLP();
		if( WRITE_X )
			array_2_disk("x", OUTPUT_DIRECTORY, OUTPUT_FOLDER, x_h, COLUMNS, ROWS, SLICES, NUM_VOXELS, true );
		array_2_disk("x_hull_after", OUTPUT_DIRECTORY, OUTPUT_FOLDER, x_hull_h, COLUMNS, ROWS, SLICES, NUM_VOXELS, true );
	}
	else
	{
		//binary_2_ASCII();
		test_func();
	}
	/************************************************************************************************************************************************************/
	/* Program has finished execution. Require the user to hit enter to terminate the program and close the terminal/console window								*/ 															
	/************************************************************************************************************************************************************/
	puts("Preprocessing complete.  Press enter to close the console window...");
	exit_program_if(true);
}
/***********************************************************************************************************************************************************************************************************************/
/**************************************************************************************** t/v conversions and energy calibrations **************************************************************************************/
/***********************************************************************************************************************************************************************************************************************/
void read_energy_responses( const int num_histories, const int start_file_num, const int end_file_num )
{
	
	char data_filename[128];
	char magic_number[5];
	int version_id;
	int file_histories;
	float projection_angle, beam_energy;
	int generation_date, preprocess_date;
	int phantom_name_size, data_source_size, prepared_by_size;
	char *phantom_name, *data_source, *prepared_by;
	int data_size;
	//int gantry_position, gantry_angle, scan_histories;
	int gantry_position, gantry_angle, scan_number, scan_histories;
	//int array_index = 0;
	FILE* input_file;

	puts("Reading energy detector responses and performing energy response calibration...");
	//printf("Reading File for Gantry Angle %d from Scan Number %d...\n", gantry_angle, scan_number );
	sprintf(data_filename, "%s%s/%s_%03d%s", INPUT_DIRECTORY, INPUT_FOLDER, INPUT_BASE_NAME, gantry_angle, FILE_EXTENSION );
}
/***********************************************************************************************************************************************************************************************************************/
/********************************************************************************************** Execution Control Functions ********************************************************************************************/
/***********************************************************************************************************************************************************************************************************************/
bool is_bad_angle( const int angle )
{
	static const int bad_angles[] = {0, 80, 84, 88, 92, 96, 100, 180, 260, 264, 268, 272, 276};
	return std::binary_search( bad_angles, bad_angles + sizeof(bad_angles) / sizeof(int), angle );
}
void timer( bool start)
{
	if( start )
		start_time = clock();
	else
	{
		end_time = clock();
		execution_clock_cycles = (end_time - start_time);
		execution_time = double( execution_clock_cycles) / CLOCKS_PER_SEC;
		printf( "Total execution time : %3f [seconds]\n", execution_time );	
	}
}
void pause_execution()
{
	char user_response[20];
	puts("Execution paused.  Hit enter to continue execution.\n");
	fgets(user_response, sizeof(user_response), stdin);
}
void exit_program_if( bool early_exit)
{
	if( early_exit )
	{
		char user_response[20];
		timer( STOP );
		puts("Hit enter to stop...");
		fgets(user_response, sizeof(user_response), stdin);
		exit(1);
	}
}
/***********************************************************************************************************************************************************************************************************************/
/************************************************************************************** Memory Transfers, Maintenance, and Cleaning ************************************************************************************/
/***********************************************************************************************************************************************************************************************************************/
void initializations()
{
	puts("Allocating statistical analysis arrays on host/GPU...");

	bin_counts_h		  = (int*)	 calloc( NUM_BINS, sizeof(int)	 );
	mean_WEPL_h			  = (float*) calloc( NUM_BINS, sizeof(float) );
	mean_rel_ut_angle_h	  = (float*) calloc( NUM_BINS, sizeof(float) );
	mean_rel_uv_angle_h	  = (float*) calloc( NUM_BINS, sizeof(float) );
	
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
void resize_vectors( const int new_size )
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
void shrink_vectors( const int new_capacity )
{
	bin_num_vector.shrink_to_fit();
	//gantry_angle_vector.shrink_to_fit();
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
void initialize_stddev()
{	
	stddev_rel_ut_angle_h = (float*) calloc( NUM_BINS, sizeof(float) );	
	stddev_rel_uv_angle_h = (float*) calloc( NUM_BINS, sizeof(float) );	
	stddev_WEPL_h		  = (float*) calloc( NUM_BINS, sizeof(float) );

	cudaMalloc((void**) &stddev_rel_ut_angle_d,	SIZE_BINS_FLOAT );
	cudaMalloc((void**) &stddev_rel_uv_angle_d,	SIZE_BINS_FLOAT );
	cudaMalloc((void**) &stddev_WEPL_d,			SIZE_BINS_FLOAT );

	cudaMemcpy( stddev_rel_ut_angle_d,	stddev_rel_ut_angle_h,	SIZE_BINS_FLOAT,	cudaMemcpyHostToDevice );
	cudaMemcpy( stddev_rel_uv_angle_d,	stddev_rel_uv_angle_h,	SIZE_BINS_FLOAT,	cudaMemcpyHostToDevice );
	cudaMemcpy( stddev_WEPL_d,			stddev_WEPL_h,			SIZE_BINS_FLOAT,	cudaMemcpyHostToDevice );
}
void allocations( const int num_histories)
{
	bin_num				= (int*)   calloc( num_histories,	sizeof(int)   );		
	gantry_angle		= (int*)   calloc( num_histories,	sizeof(int)   );
	WEPL				= (float*) calloc( num_histories,	sizeof(float) );		
	x_entry				= (float*) calloc( num_histories,	sizeof(float) );		
	y_entry				= (float*) calloc( num_histories,	sizeof(float) );		
	z_entry				= (float*) calloc( num_histories,	sizeof(float) );		
	x_exit				= (float*) calloc( num_histories,	sizeof(float) );		
	y_exit				= (float*) calloc( num_histories,	sizeof(float) );			
	z_exit				= (float*) calloc( num_histories,	sizeof(float) );			
	xy_entry_angle		= (float*) calloc( num_histories,	sizeof(float) );	
	xz_entry_angle		= (float*) calloc( num_histories,	sizeof(float) );	
	xy_exit_angle		= (float*) calloc( num_histories,	sizeof(float) );	
	xz_exit_angle		= (float*) calloc( num_histories,	sizeof(float) );	
}
void reallocations( const int new_size)
{
	bin_num				= (int*)   realloc( bin_num,			new_size * sizeof(int)   );		
	gantry_angle		= (int*)   realloc( gantry_angle,		new_size * sizeof(int)   );
	WEPL				= (float*) realloc( WEPL,				new_size * sizeof(float) );		
	x_entry				= (float*) realloc( x_entry,			new_size * sizeof(float) );		
	y_entry				= (float*) realloc( y_entry,			new_size * sizeof(float) );		
	z_entry				= (float*) realloc( z_entry,			new_size * sizeof(float) );		
	x_exit				= (float*) realloc( x_exit,				new_size * sizeof(float) );		
	y_exit				= (float*) realloc( y_exit,				new_size * sizeof(float) );			
	z_exit				= (float*) realloc( z_exit,				new_size * sizeof(float) );			
	xy_entry_angle		= (float*) realloc( xy_entry_angle,		new_size * sizeof(float) );	
	xz_entry_angle		= (float*) realloc( xz_entry_angle,		new_size * sizeof(float) );	
	xy_exit_angle		= (float*) realloc( xy_exit_angle,		new_size * sizeof(float) );	
	xz_exit_angle		= (float*) realloc( xz_exit_angle,		new_size * sizeof(float) );	
}
void post_cut_memory_clean()
{
	puts("Freeing unnecessary memory, resizing vectors, and shrinking vectors to fit just the remaining histories...");

	free(failed_cuts_h );
	free(stddev_rel_ut_angle_h);
	free(stddev_rel_uv_angle_h);
	free(stddev_WEPL_h);

	cudaFree( failed_cuts_d );
	cudaFree( bin_num_d );
	cudaFree( WEPL_d );
	cudaFree( xy_entry_angle_d );
	cudaFree( xz_entry_angle_d );
	cudaFree( xy_exit_angle_d );
	cudaFree( xz_exit_angle_d );

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
void write_run_settings()
{
	char user_response[20];
	char run_settings_filename[512];
	puts("Reading tracker plane positions...");

	sprintf(run_settings_filename, "%s%s\\run_settings.cfg", INPUT_DIRECTORY, INPUT_FOLDER);
	if( DEBUG_TEXT_ON )
		printf("Opening run settings file %s...\n", run_settings_filename);
	std::ofstream run_settings_file(run_settings_filename);		
	if( !run_settings_file.is_open() ) {
		printf("ERROR: run settings file file not found at %s!\n", run_settings_filename);	
		exit_program_if(true);
	}
	else
	{
		fputs("Found File", stdout);
		fflush(stdout);
		printf("user_response = \"%s\"\n", user_response);
	}
	if( DEBUG_TEXT_ON )
		puts("Loading run settings...");
	run_settings_file << "MAX_GPU_HISTORIES = " << MAX_GPU_HISTORIES << std::endl;
	run_settings_file << "GANTRY_ANGLE_INTERVAL = " << GANTRY_ANGLE_INTERVAL << std::endl;
	run_settings_file << "SSD_T_SIZE = " << SSD_T_SIZE << std::endl;
	run_settings_file << "SSD_V_SIZE = " << SSD_V_SIZE << std::endl;
	run_settings_file << "T_BIN_SIZE = " << T_BIN_SIZE << std::endl;
	run_settings_file << "V_BIN_SIZE = " << V_BIN_SIZE << std::endl;
	run_settings_file << "ANGULAR_BIN_SIZE = " << ANGULAR_BIN_SIZE << std::endl;
	run_settings_file << "GANTRY_ANGLE_INTERVAL = " << GANTRY_ANGLE_INTERVAL << std::endl;
	run_settings_file << "RECON_CYL_RADIUS = " << RECON_CYL_RADIUS << std::endl;
	run_settings_file << "RECON_CYL_HEIGHT = " << RECON_CYL_HEIGHT << std::endl;
	run_settings_file << "COLUMNS = " << COLUMNS << std::endl;
	run_settings_file << "ROWS = " << ROWS << std::endl;
	run_settings_file << "SLICE_THICKNESS" << SLICE_THICKNESS << std::endl;
	//run_settings_file << "RECON_CYL_RADIUS = " << RECON_CYL_RADIUS << std::endl;
	//run_settings_file << "RECON_CYL_HEIGHT = " << RECON_CYL_HEIGHT << std::endl;
	//run_settings_file << "COLUMNS = " << COLUMNS << std::endl;
	//run_settings_file << "ROWS = " << ROWS << std::endl;
	//run_settings_file << "SLICE_THICKNESS" << SLICE_THICKNESS << std::endl;
	run_settings_file.close();
}
void assign_SSD_positions()	//HERE THE COORDINATES OF THE DETECTORS PLANES ARE LOADED, THE CONFIG FILE IS CREATED BY FORD (RWS)
{
	char user_response[20];
	char configFilename[512];
	puts("Reading tracker plane positions...");

	sprintf(configFilename, "%s%s\\scan.cfg", INPUT_DIRECTORY, INPUT_FOLDER);
	if( DEBUG_TEXT_ON )
		printf("Opening config file %s...\n", configFilename);
	std::ifstream configFile(configFilename);		
	if( !configFile.is_open() ) {
		printf("ERROR: config file not found at %s!\n", configFilename);	
		exit_program_if(true);
	}
	else
	{
		fputs("Found File", stdout);
		fflush(stdout);
		printf("user_response = \"%s\"\n", user_response);
	}
	if( DEBUG_TEXT_ON )
		puts("Reading Tracking Plane Positions...");
	for( unsigned int i = 0; i < 8; i++ ) {
		configFile >> SSD_u_Positions[i];
		if( DEBUG_TEXT_ON )
			printf("SSD_u_Positions[%d] = %3f", i, SSD_u_Positions[i]);
	}
	
	configFile.close();

}
void count_histories()
{
	for( int scan_number = 0; scan_number < NUM_SCANS; scan_number++ )
		histories_per_scan[scan_number] = 0;

	histories_per_file =				 (int*) calloc( NUM_SCANS * GANTRY_ANGLES, sizeof(int) );
	histories_per_gantry_angle =		 (int*) calloc( GANTRY_ANGLES, sizeof(int) );
	recon_vol_histories_per_projection = (int*) calloc( GANTRY_ANGLES, sizeof(int) );

	if( DEBUG_TEXT_ON )
		puts("Counting proton histories...\n");
	switch( DATA_FORMAT )
	{
		case OLD_FORMAT : count_histories_old();	break;
		case VERSION_0  : count_histories_v0();		break;
		case VERSION_1  : count_histories_v1();		break;
	}
	if( DEBUG_TEXT_ON )
	{
		for( int file_number = 0, gantry_position_number = 0; file_number < (NUM_SCANS * GANTRY_ANGLES); file_number++, gantry_position_number++ )
		{
			if( file_number % NUM_SCANS == 0 )
				printf("There are a Total of %d Histories From Gantry Angle %d\n", histories_per_gantry_angle[gantry_position_number], int(gantry_position_number* GANTRY_ANGLE_INTERVAL) );			
			printf("* %d Histories are From Scan Number %d\n", histories_per_file[file_number], (file_number % NUM_SCANS) + 1 );
			
		}
		for( int scan_number = 0; scan_number < NUM_SCANS; scan_number++ )
			printf("There are a Total of %d Histories in Scan Number %d \n", histories_per_scan[scan_number], scan_number + 1);
		printf("There are a Total of %d Histories\n", total_histories);
	}
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
			
			sprintf( data_filename, "%s%s/%s_trans%d_%03d%s", INPUT_DIRECTORY, INPUT_FOLDER, INPUT_BASE_NAME, scan_number, gantry_angle, FILE_EXTENSION );
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
	unsigned int magic_number, num_histories, file_number = 0, gantry_position_number = 0;
	for( unsigned int gantry_angle = 0; gantry_angle < 360; gantry_angle += int(GANTRY_ANGLE_INTERVAL), gantry_position_number++ )
	{
		for( unsigned int scan_number = 1; scan_number <= NUM_SCANS; scan_number++, file_number++ )
		{
			sprintf(data_filename, "%s%s/%s_%03d%s", INPUT_DIRECTORY, INPUT_FOLDER, INPUT_BASE_NAME, gantry_angle, FILE_EXTENSION  );
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
			
			fread(&magic_number, 4, 1, data_file );
			if( magic_number != MAGIC_NUMBER_CHECK ) 
			{
				puts("Error: unknown file type (should be PCTD)!\n");
				exit_program_if(true);
			}

			fread(&VERSION_ID, sizeof(int), 1, data_file );			
			if( VERSION_ID == 0 )
			{
				fread(&num_histories, sizeof(int), 1, data_file );
				if( DEBUG_TEXT_ON )
					printf("There are %d Histories for Gantry Angle %d From Scan Number %d\n", num_histories, gantry_angle, scan_number);
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
				//pause_execution();
			}
			else if( VERSION_ID == 1 )
			{
				fread(&num_histories, sizeof(int), 1, data_file );
				if( DEBUG_TEXT_ON )
					printf("There are %d Histories for Gantry Angle %d From Scan Number %d\n", num_histories, gantry_angle, scan_number);
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
				//pause_execution();
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
	//char user_response[20];
	char data_filename[256];
	int num_histories, file_number = 0, gantry_position_number = 0;
	for( int gantry_angle = 0; gantry_angle < 360; gantry_angle += int(GANTRY_ANGLE_INTERVAL), gantry_position_number++ )
	{
		for( int scan_number = 1; scan_number <= NUM_SCANS; scan_number++, file_number++ )
		{
			sprintf(data_filename, "%s%s/%s_%03d%s", INPUT_DIRECTORY, INPUT_FOLDER, INPUT_BASE_NAME, gantry_angle, FILE_EXTENSION  );
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
			
				if( DEBUG_TEXT_ON )
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
			sprintf(data_filename, "%s%s/%s_%03d%s", INPUT_DIRECTORY, INPUT_FOLDER, INPUT_BASE_NAME, gantry_angle, FILE_EXTENSION  );
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
	image = (T*)calloc( IMAGE_VOXELS, sizeof(T));
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
/***********************************************************************************************************************************************************************************************************************/
/************************************************************************************** Data importation, initial cuts, and binning ************************************************************************************/
/***********************************************************************************************************************************************************************************************************************/
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
void apply_tu_shifts( unsigned int num_histories)
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
		sprintf(data_filename, "%s_%03d%s", "ut_entry_angle", gantry_angle, ".txt" );
		array_2_disk( data_filename, OUTPUT_DIRECTORY, OUTPUT_FOLDER, ut_entry_angle, COLUMNS, ROWS, SLICES, num_histories, true );
		sprintf(data_filename, "%s_%03d%s", "uv_entry_angle", gantry_angle, ".txt" );
		array_2_disk( "ut_entry_angle", OUTPUT_DIRECTORY, OUTPUT_FOLDER, uv_entry_angle, COLUMNS, ROWS, SLICES, num_histories, true );
		sprintf(data_filename, "%s_%03d%s", "ut_exit_angle", gantry_angle, ".txt" );
		array_2_disk( "ut_entry_angle", OUTPUT_DIRECTORY, OUTPUT_FOLDER, ut_exit_angle, COLUMNS, ROWS, SLICES, num_histories, true );
		sprintf(data_filename, "%s_%03d%s", "uv_exit_angle", gantry_angle, ".txt" );
		array_2_disk( "ut_entry_angle", OUTPUT_DIRECTORY, OUTPUT_FOLDER, uv_exit_angle, COLUMNS, ROWS, SLICES, num_histories, true );
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
		sprintf( data_filename, "%s%s/%s_trans%d_%03d%s", INPUT_DIRECTORY, INPUT_FOLDER, INPUT_BASE_NAME, scan_number, gantry_angle, FILE_EXTENSION );
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
	char data_filename[128];
	unsigned int gantry_position, gantry_angle, scan_number, file_histories, array_index = 0, histories_read = 0;

	printf("%d histories to be read from %d files\n", num_histories, end_file_num - start_file_num + 1 );
	for( unsigned int file_num = start_file_num; file_num <= end_file_num; file_num++ )
	{	
		gantry_position = file_num / NUM_SCANS;
		gantry_angle = int(gantry_position * GANTRY_ANGLE_INTERVAL);
		scan_number = file_num % NUM_SCANS + 1;
		file_histories = histories_per_file[file_num];
		
		sprintf(data_filename, "%s%s/%s_%03d%s", INPUT_DIRECTORY, INPUT_FOLDER, INPUT_BASE_NAME, gantry_angle, FILE_EXTENSION );
		FILE* data_file = fopen(data_filename, "rb");
		if( data_file == NULL )
		{
			fputs( "Error Opening Data File:  Check that the directories are properly named.", stderr ); 
			exit_program_if(true);
		}
		if( VERSION_ID == 0 )
		{
			printf("\t");
			printf("Reading %d histories for gantry angle %d from scan number %d...\n", file_histories, gantry_angle, scan_number );			
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
			printf("\t");
			printf("Reading %d histories for gantry angle %d from scan number %d...\n", file_histories, gantry_angle, scan_number );			
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
	if( T_SHIFT != 0.0	||  U_SHIFT != 0.0 )
		apply_tu_shifts( num_histories );
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
		sprintf(data_filename, "%s%s/%s_%03d%s", INPUT_DIRECTORY, INPUT_FOLDER, INPUT_BASE_NAME, gantry_angle, FILE_EXTENSION );	
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
					if( WEPL_h[array_index] < 0 )
						printf("WEPL[%d] = %3f\n", i, WEPL_h[array_index] );
					u_in_1_h[array_index]		*= MM_TO_CM;
					u_in_2_h[array_index]		*= MM_TO_CM;
					u_out_1_h[array_index]	*= MM_TO_CM;
					u_out_2_h[array_index]	*= MM_TO_CM;
				}
				gantry_angle_h[array_index] = int(projection_angle);
			}
			data_file.close();
			histories_read += file_histories;
		}
	}
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
		sprintf(data_filename, "%s%s/%s_%03d%s", INPUT_DIRECTORY, INPUT_FOLDER, INPUT_BASE_NAME, gantry_angle, FILE_EXTENSION );	
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
	//printf("There are %d histories in this projection\n", num_histories );
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
	double a = 1 + pow(m, 2);											// x^2 coefficient 
	double b = 2 * m * b_in;											// x coefficient
	double c = pow(b_in, 2) - pow(RECON_CYL_RADIUS, 2 );				// 1 coefficient
	double entry_discriminant = pow(b, 2) - (4 * a * c);				// Quadratic formula discriminant		
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
		double squared_distance_1	= pow( u_intercept_1 - u, 2 ) + pow( t_intercept_1 - t, 2 );
		double squared_distance_2	= pow( u_intercept_2 + u, 2 ) + pow( t_intercept_2 + t, 2 );
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
	//cudaMemcpy( bin_num_d,	bin_num_h,	size_ints,		cudaMemcpyHostToDevice );

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
		array_2_disk( data_filename, OUTPUT_DIRECTORY, OUTPUT_FOLDER, bin_num_h, COLUMNS, ROWS, SLICES, num_histories, true );
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
			//bin_num[recon_vol_histories]			= bin_num[i];			
			//gantry_angle[recon_vol_histories]		= gantry_angle[i];	
			//WEPL[recon_vol_histories]				= WEPL[i]; 		
			//x_entry[recon_vol_histories]			= x_entry[i];		
			//y_entry[recon_vol_histories]			= y_entry[i];		
			//z_entry[recon_vol_histories]			= z_entry[i];		
			//x_exit[recon_vol_histories]				= x_exit[i];			
			//y_exit[recon_vol_histories]				= y_exit[i];			
			//z_exit[recon_vol_histories]				= z_exit[i];			
			//xy_entry_angle[recon_vol_histories]		= xy_entry_angle[i];	
			//xz_entry_angle[recon_vol_histories]		= xz_entry_angle[i];	
			//xy_exit_angle[recon_vol_histories]		= xy_exit_angle[i]; 	
			//xz_exit_angle[recon_vol_histories]		= xz_exit_angle[i];	
			offset++;
			recon_vol_histories++;
		}
	}
	printf( "=======>%d out of %d (%4.2f%%) histories passed intersection cuts\n\n", offset, num_histories, (double) offset / num_histories * 100 );
	
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
				atomicAdd( &mean_rel_ut_angle[bin_num[i]], rel_uv_angle );
				//atomicAdd( &mean_rel_ut_angle[bin_num[i]], relative_ut_angle[i] );
				//atomicAdd( &mean_rel_uv_angle[bin_num[i]], relative_uv_angle[i] );
			}
			//else
				//bin_num[i] = -1;
		}
	}
}
/***********************************************************************************************************************************************************************************************************************/
/******************************************************************************************** Statistical analysis and cuts ********************************************************************************************/
/***********************************************************************************************************************************************************************************************************************/
void calculate_means()
{
	puts("Calculating the Mean for Each Bin Before Cuts...");

	dim3 dimBlock( T_BINS );
	dim3 dimGrid( V_BINS, ANGULAR_BINS );   
	calculate_means_GPU<<< dimGrid, dimBlock >>>
	( 
		bin_counts_d, mean_WEPL_d, mean_rel_ut_angle_d, mean_rel_uv_angle_d
	);

	if( WRITE_WEPL_DISTS )
	{
		cudaMemcpy( mean_WEPL_h,	mean_WEPL_d,	SIZE_BINS_FLOAT, cudaMemcpyDeviceToHost );
		int* empty_parameter;
		bins_2_disk( "WEPL_dist_pre_test2", empty_parameter, mean_WEPL_h, NUM_BINS, MEANS, ALL_BINS, BY_BIN );
	}
	//cudaMemcpy( bin_counts_h,	bin_counts_d,	SIZE_BINS_INT, cudaMemcpyDeviceToHost );
	//cudaMemcpy( mean_WEPL_h,	mean_WEPL_d,	SIZE_BINS_FLOAT, cudaMemcpyDeviceToHost );
	//cudaMemcpy( mean_rel_ut_angle_h,	mean_rel_ut_angle_d,	SIZE_BINS_FLOAT, cudaMemcpyDeviceToHost );
	//cudaMemcpy( mean_rel_uv_angle_h,	mean_rel_uv_angle_d,	SIZE_BINS_FLOAT, cudaMemcpyDeviceToHost );

	//array_2_disk("bin_counts_h_pre", OUTPUT_DIRECTORY, OUTPUT_FOLDER, bin_counts_h, T_BINS, ANGULAR_BINS, V_BINS, NUM_BINS, true );
	//array_2_disk("mean_WEPL_h", OUTPUT_DIRECTORY, OUTPUT_FOLDER, mean_WEPL_h, T_BINS, ANGULAR_BINS, V_BINS, NUM_BINS, true );
	//array_2_disk("mean_rel_ut_angle_h", OUTPUT_DIRECTORY, OUTPUT_FOLDER, mean_rel_ut_angle_h, T_BINS, ANGULAR_BINS, V_BINS, NUM_BINS, true );
	//array_2_disk("mean_rel_uv_angle_h", OUTPUT_DIRECTORY, OUTPUT_FOLDER, mean_rel_uv_angle_h, T_BINS, ANGULAR_BINS, V_BINS, NUM_BINS, true );
	
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
	cudaMalloc((void**) &xy_exit_angle_d,		size_floats);
	cudaMalloc((void**) &xz_exit_angle_d,		size_floats);

	cudaMemcpy( bin_num_d,				&bin_num_vector[start_position],			size_ints, cudaMemcpyHostToDevice);
	cudaMemcpy( WEPL_d,					&WEPL_vector[start_position],				size_floats, cudaMemcpyHostToDevice);
	cudaMemcpy( xy_entry_angle_d,		&xy_entry_angle_vector[start_position],		size_floats, cudaMemcpyHostToDevice);
	cudaMemcpy( xz_entry_angle_d,		&xz_entry_angle_vector[start_position],		size_floats, cudaMemcpyHostToDevice);
	cudaMemcpy( xy_exit_angle_d,		&xy_exit_angle_vector[start_position],		size_floats, cudaMemcpyHostToDevice);
	cudaMemcpy( xz_exit_angle_d,		&xz_exit_angle_vector[start_position],		size_floats, cudaMemcpyHostToDevice);

	//cudaMemcpy( bin_num_d,				&bin_num[start_position],			size_ints, cudaMemcpyHostToDevice);
	//cudaMemcpy( WEPL_d,					&WEPL[start_position],				size_floats, cudaMemcpyHostToDevice);
	//cudaMemcpy( xy_entry_angle_d,		&xy_entry_angle[start_position],		size_floats, cudaMemcpyHostToDevice);
	//cudaMemcpy( xz_entry_angle_d,		&xz_entry_angle[start_position],		size_floats, cudaMemcpyHostToDevice);
	//cudaMemcpy( xy_exit_angle_d,		&xy_exit_angle[start_position],		size_floats, cudaMemcpyHostToDevice);
	//cudaMemcpy( xz_exit_angle_d,		&xz_exit_angle[start_position],		size_floats, cudaMemcpyHostToDevice);

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

		atomicAdd( &stddev_WEPL[bin_num[i]], pow( WEPL_difference, 2 ) );
		atomicAdd( &stddev_rel_ut_angle[bin_num[i]], pow( rel_ut_angle_difference, 2 ) );
		atomicAdd( &stddev_rel_uv_angle[bin_num[i]], pow( rel_uv_angle_difference, 2 ) );
	}
}
void calculate_standard_deviations()
{
	puts("Calculating standard deviations for each bin...");
	dim3 dimBlock( T_BINS );
	dim3 dimGrid( V_BINS, ANGULAR_BINS );   
	calculate_standard_deviations_GPU<<< dimGrid, dimBlock >>>
	( 
		bin_counts_d, stddev_WEPL_d, stddev_rel_ut_angle_d, stddev_rel_uv_angle_d
	);
	//cudaFree( bin_counts_d );
}
__global__ void calculate_standard_deviations_GPU( int* bin_counts, float* stddev_WEPL, float* stddev_rel_ut_angle, float* stddev_rel_uv_angle )
{
	int v = blockIdx.x, angle = blockIdx.y, t = threadIdx.x;
	int bin = t + angle * T_BINS + v * T_BINS * ANGULAR_BINS;
	if( bin_counts[bin] > 0 )
	{
		// SAMPLE_STD_DEV = true/false = 1/0 => std_dev = SUM{i = 1 -> N} [ ( mu - x_i)^2 / ( N - 1/0 ) ]
		stddev_WEPL[bin] = sqrtf( stddev_WEPL[bin] / ( bin_counts[bin] - SAMPLE_STD_DEV ) );		
		stddev_rel_ut_angle[bin] = sqrtf( stddev_rel_ut_angle[bin] / ( bin_counts[bin] - SAMPLE_STD_DEV ) );
		stddev_rel_uv_angle[bin] = sqrtf( stddev_rel_uv_angle[bin] / ( bin_counts[bin] - SAMPLE_STD_DEV ) );
	}
	syncthreads();
	bin_counts[bin] = 0;
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

	//cudaMemcpy( bin_num_d,				&bin_num[start_position],			size_ints, cudaMemcpyHostToDevice);
	//cudaMemcpy( WEPL_d,					&WEPL[start_position],				size_floats, cudaMemcpyHostToDevice);
	//cudaMemcpy( xy_entry_angle_d,		&xy_entry_angle[start_position],		size_floats, cudaMemcpyHostToDevice);
	//cudaMemcpy( xz_entry_angle_d,		&xz_entry_angle[start_position],		size_floats, cudaMemcpyHostToDevice);
	//cudaMemcpy( xy_exit_angle_d,		&xy_exit_angle[start_position],		size_floats, cudaMemcpyHostToDevice);
	//cudaMemcpy( xz_exit_angle_d,		&xz_exit_angle[start_position],		size_floats, cudaMemcpyHostToDevice);

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
			//bin_num[post_cut_histories] = bin_num[start_position + i];
			////gantry_angle[post_cut_histories] = gantry_angle[start_position + i];
			//WEPL[post_cut_histories] = WEPL[start_position + i];
			//x_entry[post_cut_histories] = x_entry[start_position + i];
			//y_entry[post_cut_histories] = y_entry[start_position + i];
			//z_entry[post_cut_histories] = z_entry[start_position + i];
			//x_exit[post_cut_histories] = x_exit[start_position + i];
			//y_exit[post_cut_histories] = y_exit[start_position + i];
			//z_exit[post_cut_histories] = z_exit[start_position + i];
			//xy_entry_angle[post_cut_histories] = xy_entry_angle[start_position + i];
			//xz_entry_angle[post_cut_histories] = xz_entry_angle[start_position + i];
			//xy_exit_angle[post_cut_histories] = xy_exit_angle[start_position + i];
			//xz_exit_angle[post_cut_histories] = xz_exit_angle[start_position + i];
			post_cut_histories++;
		}
	}
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
		bool passed_WEPL_cut = ( abs( mean_WEPL[bin_num[i]] - WEPL[i] ) <= ( SIGMAS_TO_KEEP * stddev_WEPL[bin_num[i]] ) );
		failed_cuts[i] = !passed_ut_cut || !passed_uv_cut || !passed_WEPL_cut;

		if( !failed_cuts[i] )
		{
			atomicAdd( &bin_counts[bin_num[i]], 1 );
			atomicAdd( &sinogram[bin_num[i]], WEPL[i] );			
		}
	}
}
/***********************************************************************************************************************************************************************************************************************/
/********************************************************************************************************* FBP *********************************************************************************************************/
/***********************************************************************************************************************************************************************************************************************/
void initialize_sinogram()
{
	puts("Allocating host/GPU memory and initializing sinogram...");
	sinogram_h = (float*) calloc( NUM_BINS, sizeof(float) );
	cudaMalloc((void**) &sinogram_d, SIZE_BINS_FLOAT );
	cudaMemcpy( sinogram_d,	sinogram_h,	SIZE_BINS_FLOAT, cudaMemcpyHostToDevice );	
}
void construct_sinogram()
{
	puts("Recalculating the mean WEPL for each bin and constructing the sinogram...");
	dim3 dimBlock( T_BINS );
	dim3 dimGrid( V_BINS, ANGULAR_BINS );   
	construct_sinogram_GPU<<< dimGrid, dimBlock >>>( bin_counts_d, sinogram_d );

	if( WRITE_WEPL_DISTS )
	{
		cudaMemcpy( sinogram_h,	sinogram_d,	SIZE_BINS_FLOAT, cudaMemcpyDeviceToHost );
		int* empty_parameter;
		bins_2_disk( "WEPL_dist_post_test2", empty_parameter, sinogram_h, NUM_BINS, MEANS, ALL_BINS, BY_BIN );
	}
	//cudaMemcpy(sinogram_h,  sinogram_d, SIZE_BINS_FLOAT, cudaMemcpyDeviceToHost);
	//array_2_disk("sinogram", OUTPUT_DIRECTORY, OUTPUT_FOLDER, sinogram_h, T_BINS, ANGULAR_BINS, V_BINS, NUM_BINS, true );

	//bin_counts_h		  = (int*)	 calloc( NUM_BINS, sizeof(int) );
	//cudaMemcpy(bin_counts_h, bin_counts_d, SIZE_BINS_INT, cudaMemcpyDeviceToHost) ;
	//array_2_disk( "bin_counts_post", OUTPUT_DIRECTORY, OUTPUT_FOLDER, bin_counts_h, T_BINS, ANGULAR_BINS, V_BINS, NUM_BINS, true );
	cudaFree(bin_counts_d);
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

	puts("Performing backprojection...");

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

	if( WRITE_FBP_IMAGE )
	{
		cudaMemcpy( FBP_image_h, FBP_image_d, SIZE_IMAGE_FLOAT, cudaMemcpyDeviceToHost );
		array_2_disk( "FBP_image_h", OUTPUT_DIRECTORY, OUTPUT_FOLDER, FBP_image_h, COLUMNS, ROWS, SLICES, NUM_VOXELS, true );	
	}

	// Generate FBP hull by thresholding FBP image
	FBP_image_2_hull();

	// Discard FBP image unless it is to be used as the initial iterate x_0 in iterative image reconstruction
	if( X_K0 != FBP_IMAGE && X_K0 != HYBRID )
		free(FBP_image_h);
}
void filter()
{
	puts("Filtering the sinogram...");	

	sinogram_filtered_h = (float*) calloc( NUM_BINS, sizeof(float) );
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
		switch( FBP_FILTER )
		{
			case NONE: 
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
				filtered = pow( pow(T_BIN_SIZE * PI, 2.0) * ( 1.0 - pow(2 * t_bin_sep, 2.0) ), -1.0 );
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

	double delta = GANTRY_ANGLE_INTERVAL * ANGLE_TO_RADIANS;
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
		double delta = GANTRY_ANGLE_INTERVAL * ANGLE_TO_RADIANS;
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
				scale_factor = pow( SOURCE_RADIUS / ( SOURCE_RADIUS + u ), 2 );
		  
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
	puts("Performing thresholding on FBP image to generate FBP hull...");

	FBP_hull_h = (bool*) calloc( COLUMNS * ROWS * SLICES, sizeof(bool) );
	initialize_hull( FBP_hull_h, FBP_hull_d );
	dim3 dimBlock( SLICES );
	dim3 dimGrid( COLUMNS, ROWS );   
	FBP_image_2_hull_GPU<<< dimGrid, dimBlock >>>( FBP_image_d, FBP_hull_d );	
	cudaMemcpy( FBP_hull_h, FBP_hull_d, SIZE_IMAGE_BOOL, cudaMemcpyDeviceToHost );
	
	if( WRITE_FBP_HULL )
		array_2_disk( "x_FBP", OUTPUT_DIRECTORY, OUTPUT_FOLDER, FBP_hull_h, COLUMNS, ROWS, SLICES, NUM_VOXELS, true );

	if( MLP_HULL != FBP_HULL)	
		free(FBP_hull_h);
	cudaFree(FBP_hull_d);
	cudaFree(FBP_image_d);
}
__global__ void FBP_image_2_hull_GPU( float* FBP_image, bool* FBP_hull )
{
	int row = blockIdx.y, column = blockIdx.x, slice = threadIdx.x;
	int voxel = slice * COLUMNS * ROWS + row * COLUMNS + column; 
	double x = -RECON_CYL_RADIUS + ( column + 0.5 )* VOXEL_WIDTH;
	double y = RECON_CYL_RADIUS - (row + 0.5) * VOXEL_HEIGHT;
	double d_squared = pow(x, 2) + pow(y, 2);
	if(FBP_image[voxel] > FBP_THRESHOLD && (d_squared < pow(RECON_CYL_RADIUS, 2) ) ) 
		FBP_hull[voxel] = true; 
	else
		FBP_hull[voxel] = false; 
}
/***********************************************************************************************************************************************************************************************************************/
/*************************************************************************************************** Hull-Detection ****************************************************************************************************/
/***********************************************************************************************************************************************************************************************************************/
void hull_detection( const int histories_to_process)
{
	if( SC_ON  ) 
		SC( histories_to_process );		
	if( MSC_ON )
		MSC( histories_to_process );
	if( SM_ON )
		SM( histories_to_process );   
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
	if( SC_ON )
		initialize_hull( SC_hull_h, SC_hull_d );
	if( MSC_ON )
		initialize_hull( MSC_counts_h, MSC_counts_d );
	if( SM_ON )
		initialize_hull( SM_counts_h, SM_counts_d );
}
template<typename T> void initialize_hull( T*& hull_h, T*& hull_d )
{
	/* Allocate memory and initialize hull on the GPU.  Use the image and reconstruction cylinder parameters to determine the location of the perimeter of  */
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
	if( pow(x, 2) + pow(y, 2) < pow(RECON_CYL_RADIUS, 2) )
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
	if( (i < num_histories) && !missed_recon_volume[i] && (WEPL[i] <= SC_THRESHOLD) )
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
		bool end_walk, debug_run = false;
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
	if( (i < num_histories) && !missed_recon_volume[i] && (WEPL[i] <= MSC_THRESHOLD) )
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
		bool end_walk, debug_run = false;
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
	}// end: if( (i < num_histories) && !missed_recon_volume[i] && (WEPL[i] <= MSC_THRESHOLD) )
}
void MSC_edge_detection()
{
	puts("Performing edge-detection on MSC_counts...");

	dim3 dimBlock( SLICES );
	dim3 dimGrid( COLUMNS, ROWS );   
	MSC_edge_detection_GPU<<< dimGrid, dimBlock >>>( MSC_counts_d );

	puts("MSC hull-detection and edge-detection complete.");	
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
	syncthreads();
	if( max_difference > MSC_DIFF_THRESH )
		MSC_counts[voxel] = 0;
	else
		MSC_counts[voxel] = 1;
	if( pow(x, 2) + pow(y, 2) >= pow(RECON_CYL_RADIUS - max(VOXEL_WIDTH, VOXEL_HEIGHT)/2, 2 ) )
		MSC_counts[voxel] = 0;

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
		bool end_walk, debug_run = false;
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
	}// end: if( (i < num_histories) && !missed_recon_volume[i] && (WEPL[i] <= MSC_THRESHOLD) )
}
void SM_edge_detection()
{
	puts("Performing edge-detection on SM_counts...");	

	/*if( WRITE_SM_COUNTS )
	{
		cudaMemcpy(SM_counts_h,  SM_counts_d,	 SIZE_IMAGE_INT,   cudaMemcpyDeviceToHost);
		array_2_disk("SM_counts", OUTPUT_DIRECTORY, OUTPUT_FOLDER, SM_counts_h, COLUMNS, ROWS, SLICES, NUM_VOXELS, false );
	}*/

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
	
	puts("SM hull-detection and edge-detection complete.");
	//cudaMemcpy(SM_counts_h,  SM_counts_d,	 SIZE_IMAGE_INT,   cudaMemcpyDeviceToHost);
	//cudaFree( SM_counts_d );
	cudaFree( SM_differences_d );
	cudaFree( SM_thresholds_d );

	free(SM_differences_h);
	free(SM_thresholds_h);
	
	/*if( WRITE_SM_HULL )
		array_2_disk("x_SM", OUTPUT_DIRECTORY, OUTPUT_FOLDER, SM_counts_h, COLUMNS, ROWS, SLICES, NUM_VOXELS, true );
	if( MLP_HULL != SM_HULL)
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
		if( pow(x, 2) + pow(y, 2) >= pow(RECON_CYL_RADIUS - max(VOXEL_WIDTH, VOXEL_HEIGHT)/2, 2 ) )
			SM_counts[voxel] = 0;
	}
}
void SM_edge_detection_2()
{
	puts("Performing edge-detection on SM_counts...");

	// Copy the space modeled image from the GPU to the CPU and write it to file.
	cudaMemcpy(SM_counts_h,  SM_counts_d,	 SIZE_IMAGE_INT,   cudaMemcpyDeviceToHost);
	array_2_disk("SM_counts", OUTPUT_DIRECTORY, OUTPUT_FOLDER, SM_counts_h, COLUMNS, ROWS, SLICES, NUM_VOXELS, false );

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

	puts("SM hull-detection complete.  Writing results to disk...");

	cudaMemcpy(SM_counts_h,  SM_counts_d,	 SIZE_IMAGE_INT,   cudaMemcpyDeviceToHost);
	cudaFree( SM_counts_d );
	cudaFree( SM_differences_d );
	cudaFree( SM_thresholds_d );

	free(SM_differences_h);
	free(SM_thresholds_h);
	
	if( WRITE_SM_HULL )
		array_2_disk("x_SM", OUTPUT_DIRECTORY, OUTPUT_FOLDER, SM_counts_h, COLUMNS, ROWS, SLICES, NUM_VOXELS, true );
	if( MLP_HULL != SM_HULL)
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
		if( pow(x, 2) + pow(y, 2) >= pow(RECON_CYL_RADIUS - max(VOXEL_WIDTH, VOXEL_HEIGHT)/2, 2 ) )
			SM_counts[voxel] = 0;
	}
}
void hull_detection_finish()
{
	if( SC_ON )
	{
		SC_hull_h = (bool*) calloc( NUM_VOXELS, sizeof(bool) );
		cudaMemcpy(SC_hull_h,  SC_hull_d, SIZE_IMAGE_BOOL, cudaMemcpyDeviceToHost);
		if( WRITE_SC_HULL )
		{
			puts("Writing SC hull to disk...");
			array_2_disk("x_SC", OUTPUT_DIRECTORY, OUTPUT_FOLDER, SC_hull_h, COLUMNS, ROWS, SLICES, NUM_VOXELS, true );
		}
		if( MLP_HULL != SC_HULL )
			free( SC_hull_h );
		cudaFree(SC_hull_d);
	}
	if( MSC_ON )
	{
		MSC_counts_h = (int*) calloc( NUM_VOXELS, sizeof(int) );
		if( WRITE_MSC_COUNTS )
		{		
			puts("Writing MSC counts to disk...");		
			cudaMemcpy(MSC_counts_h,  MSC_counts_d, SIZE_IMAGE_INT, cudaMemcpyDeviceToHost);
			array_2_disk("MSC_counts_h", OUTPUT_DIRECTORY, OUTPUT_FOLDER, MSC_counts_h, COLUMNS, ROWS, SLICES, NUM_VOXELS, true );	
		}
		if( WRITE_MSC_HULL || (MLP_HULL == MSC_HULL) )
		{
			MSC_edge_detection();
			cudaMemcpy(MSC_counts_h,  MSC_counts_d, SIZE_IMAGE_INT, cudaMemcpyDeviceToHost);
			if( WRITE_MSC_HULL )
			{
				puts("Writing MSC hull to disk...");		
				array_2_disk("x_MSC", OUTPUT_DIRECTORY, OUTPUT_FOLDER, MSC_counts_h, COLUMNS, ROWS, SLICES, NUM_VOXELS, true );	
			}
			cudaFree(MSC_counts_d);
		}
		if( MLP_HULL != MSC_HULL )
			free( MSC_counts_h );		
	}
	if( SM_ON )
	{
		SM_counts_h = (int*) calloc( NUM_VOXELS, sizeof(int) );
		if( WRITE_SM_COUNTS )
		{		
			puts("Writing SM counts to disk...");
			cudaMemcpy(SM_counts_h,  SM_counts_d, SIZE_IMAGE_INT, cudaMemcpyDeviceToHost);
			array_2_disk("SM_counts_h", OUTPUT_DIRECTORY, OUTPUT_FOLDER, SM_counts_h, COLUMNS, ROWS, SLICES, NUM_VOXELS, true );	
		}
		if( WRITE_SM_HULL || (MLP_HULL == SM_HULL) )
		{
			SM_edge_detection();
			cudaMemcpy(SM_counts_h,  SM_counts_d, SIZE_IMAGE_INT, cudaMemcpyDeviceToHost);
			if( WRITE_SM_HULL )
			{
				puts("Writing SM hull to disk...");		
				array_2_disk("x_SM", OUTPUT_DIRECTORY, OUTPUT_FOLDER, SM_counts_h, COLUMNS, ROWS, SLICES, NUM_VOXELS, true );	
			}
			cudaFree(SM_counts_d);
		}
		if( MLP_HULL != SM_HULL )
			free( SM_counts_h );
	}
}
/***********************************************************************************************************************************************************************************************************************/
template<typename H, typename D> void averaging_filter( H*& image_h, D*& image_d )
{
	bool is_hull = ( typeid(bool) == typeid(D) );
	D* new_value_d;
	int new_value_size = NUM_VOXELS * sizeof(D);
	cudaMalloc(&new_value_d, new_value_size );

	dim3 dimBlock( SLICES );
	dim3 dimGrid( COLUMNS, ROWS );   
	averaging_filter_GPU<<< dimGrid, dimBlock >>>( image_d, new_value_d, is_hull );
	//apply_averaging_filter_GPU<<< dimGrid, dimBlock >>>( image_d, new_value_d );
	//cudaFree(new_value_d);
	cudaFree(image_d);
	image_d = new_value_d;
}
template<typename D> __global__ void averaging_filter_GPU( D* image, D* new_value, bool is_hull )
{
	int voxel_x = blockIdx.x;
	int voxel_y = blockIdx.y;	
	int voxel_z = threadIdx.x;
	int voxel = voxel_x + voxel_y * COLUMNS + voxel_z * COLUMNS * ROWS;
	int left_edge = max( voxel_x - AVG_FILTER_RADIUS, 0 );
	int right_edge = min( voxel_x + AVG_FILTER_RADIUS, COLUMNS - 1);
	int top_edge = max( voxel_y - AVG_FILTER_RADIUS, 0 );
	int bottom_edge = min( voxel_y + AVG_FILTER_RADIUS, ROWS - 1);	
	int neighborhood_voxels = ( right_edge - left_edge + 1 ) * ( bottom_edge - top_edge + 1 );
	double sum_threshold = neighborhood_voxels * AVG_FILTER_THRESHOLD;
	double sum = 0;
	// Determine neighborhood sum for voxels whose neighborhood is completely enclosed in image
	// Strip of size floor(AVG_FILTER_SIZE/2) around image perimeter must be ignored
	for( int column = left_edge; column <= right_edge; column++ )
		for( int row = top_edge; row <= bottom_edge; row++ )
			sum += image[column + (row * COLUMNS) + (voxel_z * COLUMNS * ROWS)];
	if( is_hull)
		new_value[voxel] = ( sum > sum_threshold );
	else
		new_value[voxel] = sum / neighborhood_voxels;
}
template<typename T, typename T2> __global__ void apply_averaging_filter_GPU( T* image, T2* new_value )
{
	int voxel_x = blockIdx.x;
	int voxel_y = blockIdx.y;	
	int voxel_z = threadIdx.x;
	int voxel = voxel_x + voxel_y * COLUMNS + voxel_z * COLUMNS * ROWS;
	image[voxel] = new_value[voxel];
}
void hull_selection()
{
	puts("Performing hull selection...");

	x_hull_h = (bool*) calloc( NUM_VOXELS, sizeof(bool) );
	switch( MLP_HULL )
	{
		case SC_HULL  : x_hull_h = SC_hull_h;																							break;
		case MSC_HULL : std::transform( MSC_counts_h, MSC_counts_h + NUM_VOXELS, MSC_counts_h, x_hull_h, std::logical_or<int> () );		break;
		case SM_HULL  : std::transform( SM_counts_h,  SM_counts_h + NUM_VOXELS,  SM_counts_h,  x_hull_h, std::logical_or<int> () );		break;
		case FBP_HULL : x_hull_h = FBP_hull_h;								
	}
	if( WRITE_X_HULL )
	{
		puts("Writing selected hull to disk...");
		array_2_disk("x_hull", OUTPUT_DIRECTORY, OUTPUT_FOLDER, x_hull_h, COLUMNS, ROWS, SLICES, NUM_VOXELS, true );
	}

	// Allocate memory for and transfer hull to the GPU
	cudaMalloc((void**) &x_hull_d, SIZE_IMAGE_BOOL );
	cudaMemcpy( x_hull_d, x_hull_h, SIZE_IMAGE_BOOL, cudaMemcpyHostToDevice );


	if( HULL_FILTER_ON )
	{
		puts("Filtering hull...");
		averaging_filter( x_hull_h, x_hull_d );
		puts("Hull Filtering complete");
		if( WRITE_FILTERED_HULL )
		{
			puts("Writing filtered hull to disk...");
			cudaMemcpy(x_hull_h, x_hull_d, SIZE_IMAGE_BOOL, cudaMemcpyDeviceToHost) ;
			array_2_disk( "x_hull_filtered", OUTPUT_DIRECTORY, OUTPUT_FOLDER, x_hull_h, COLUMNS, ROWS, SLICES, NUM_VOXELS, true );
		}
	}
	puts("Hull selection complete."); 
}
/***********************************************************************************************************************************************************************************************************************/
/****************************************************************************************************** MLP (host) *****************************************************************************************************/
/***********************************************************************************************************************************************************************************************************************/
void create_MLP_test_image()
{	
	//Create space carve object, init to zeros
	MLP_test_image_h = (int*)calloc( MLP_IMAGE_VOXELS, sizeof(int));

	for( int slice = 0; slice < MLP_IMAGE_SLICES; slice++ )
	{
		//add_circle( MLP_test_image_h, slice, 0.0, 0.0, MLP_IMAGE_RECON_CYL_RADIUS, 1 );
		add_ellipse( MLP_test_image_h, slice, 0.0, 0.0, MLP_PHANTOM_A, MLP_PHANTOM_B, 1 );
	}
}
template<typename O> bool find_MLP_endpoints
( 
	O*& image, double x_start, double y_start, double z_start, double xy_angle, double xz_angle, 
	double& x_object, double& y_object, double& z_object, int& voxel_x, int& voxel_y, int& voxel_z, bool entering
)
{	
		//char user_response[20];

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
		//int voxel_x, voxel_y, voxel_z;
		//int voxel_x_out, voxel_y_out, voxel_z_out; 
		int voxel; 
		bool hit_hull = false, end_walk, outside_image;
		// true false
		//bool debug_run = false;
		//bool MLP_image_output = false;
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
		{
			//if( debug_run )
				//puts("z switched");
			z_move_direction *= -1;
		}
		/*if( debug_run )
		{
			cout << "x_move_direction = " << x_move_direction << endl;
			cout << "y_move_direction = " << y_move_direction << endl;
			cout << "z_move_direction = " << z_move_direction << endl;
		}*/
		


		voxel_x = calculate_voxel( X_ZERO_COORDINATE, x, VOXEL_WIDTH );
		voxel_y = calculate_voxel( Y_ZERO_COORDINATE, y, VOXEL_HEIGHT );
		voxel_z = calculate_voxel( Z_ZERO_COORDINATE, z, VOXEL_THICKNESS );

		x_to_go = distance_remaining( X_ZERO_COORDINATE, x, X_INCREASING_DIRECTION, x_move_direction, VOXEL_WIDTH, voxel_x );
		y_to_go = distance_remaining( Y_ZERO_COORDINATE, y, Y_INCREASING_DIRECTION, y_move_direction, VOXEL_HEIGHT, voxel_y );	
		z_to_go = distance_remaining( Z_ZERO_COORDINATE, z, Z_INCREASING_DIRECTION, z_move_direction, VOXEL_THICKNESS, voxel_z );

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

		double dx_dy = pow( tan(xy_angle), -1.0 );
		double dx_dz = pow( tan(xz_angle), -1.0 );
		double dy_dz = tan(xy_angle)/tan(xz_angle);

		//if( debug_run )
		//{
		//	cout << "delta_yx = " << delta_yx << "delta_zx = " << delta_zx << "delta_zy = " << delta_zy << endl;
		//	cout << "dy_dx = " << dy_dx << "dz_dx = " << dz_dx << "dz_dy = " << dz_dy << endl;
		//	cout << "dx_dy = " << dx_dy << "dx_dz = " << dx_dz << "dy_dz = " << dy_dz << endl;
		//}

		/********************************************************************************************/
		/************************* Initialize and Check Exit Conditions *****************************/
		/********************************************************************************************/
		outside_image = (voxel_x >= COLUMNS ) || (voxel_y >= ROWS ) || (voxel_z >= SLICES ) || (voxel_x < 0  ) || (voxel_y < 0 ) || (voxel_z < 0 );		
		if( !outside_image )
		{
			hit_hull = (image[voxel] == 1);		
			//image[voxel] = 4;
		}
		end_walk = outside_image || hit_hull;
		//int j = 0;
		//int j_low_limit = 0;
		//int j_high_limit = 250;
		/*if(debug_run && j <= j_high_limit && j >= j_low_limit )
		{
			printf(" x = %3f y = %3f z = %3f\n",  x, y, z );
			printf(" x_to_go = %3f y_to_go = %3f z_to_go = %3f\n",  x_to_go, y_to_go, z_to_go );
			printf("voxel_x = %d voxel_y = %d voxel_z = %d voxel = %d\n", voxel_x, voxel_y, voxel_z, voxel);
		}*/
		//if( debug_run )
			//fgets(user_response, sizeof(user_response), stdin);
		/********************************************************************************************/
		/*********************************** Voxel Walk Routine *************************************/
		/********************************************************************************************/
		if( z_move_direction != 0 )
		{
			//if(debug_run && j <= j_high_limit && j >= j_low_limit )
				//printf("z_end != z_start\n");
			while( !end_walk )
			{
				// Change in z for Move to Voxel Edge in x and y
				x_extension = delta_zx * x_to_go;
				y_extension = delta_zy * y_to_go;
				//if(debug_run && j <= j_high_limit && j >= j_low_limit )
				//{
				//	printf(" x_extension = %3f y_extension = %3f\n", x_extension, y_extension );
				//	//printf(" x_to_go = %3f y_to_go = %3f z_to_go = %3f\n",  x_to_go, y_to_go, z_to_go );
				//	//printf("voxel_x = %d voxel_y = %d voxel_z = %d voxel = %d\n", voxel_x, voxel_y, voxel_z, voxel);
				//}
				if( (z_to_go <= x_extension  ) && (z_to_go <= y_extension) )
				{
					//printf("z_to_go <= x_extension && z_to_go <= y_extension\n");					
					voxel_z -= z_move_direction;
					
					z = edge_coordinate( Z_ZERO_COORDINATE, voxel_z, VOXEL_THICKNESS, Z_INCREASING_DIRECTION, z_move_direction );					
					x = corresponding_coordinate( dx_dz, z, z_start, x_start );
					y = corresponding_coordinate( dy_dz, z, z_start, y_start );

					/*if(debug_run && j <= j_high_limit && j >= j_low_limit )
					{
						printf(" x = %3f y = %3f z = %3f\n",  x, y, z );
						printf(" x_to_go = %3f y_to_go = %3f z_to_go = %3f\n",  x_to_go, y_to_go, z_to_go );
						printf("voxel_x = %d voxel_y = %d voxel_z = %d voxel = %d\n", voxel_x, voxel_y, voxel_z, voxel);
					}*/

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
				//if(debug_run && j <= j_high_limit && j >= j_low_limit )
				//{
				//	printf(" x = %3f y = %3f z = %3f\n",  x, y, z );
				//	printf(" x_to_go = %3f y_to_go = %3f z_to_go = %3f\n",  x_to_go, y_to_go, z_to_go );
				//	printf("voxel_x = %d voxel_y = %d voxel_z = %d voxel = %d\n", voxel_x, voxel_y, voxel_z, voxel);
				//}
				outside_image = (voxel_x >= COLUMNS ) || (voxel_y >= ROWS ) || (voxel_z >= SLICES ) || (voxel_x < 0  ) || (voxel_y < 0 ) || (voxel_z < 0 );		
				if( !outside_image )
				{
					hit_hull = (image[voxel] == 1);	
					//if( MLP_image_output )
					//{
						//image[voxel] = 4;
					//}
				}
				end_walk = outside_image || hit_hull;
				//j++;
				//if( debug_run )
					//fgets(user_response, sizeof(user_response), stdin);		
			}// end !end_walk 
		}
		else
		{
			//if(debug_run && j <= j_high_limit && j >= j_low_limit )
				//printf("z_end == z_start\n");
			while( !end_walk )
			{
				// Change in x for Move to Voxel Edge in y
				y_extension = y_to_go / delta_yx;
				//If Next Voxel Edge is in x or xy Diagonal
				if( x_to_go <= y_extension )
				{
					//printf(" x_to_go <= y_extension \n");
					voxel_x += x_move_direction;
					
					x = edge_coordinate( X_ZERO_COORDINATE, voxel_x, VOXEL_WIDTH, X_INCREASING_DIRECTION, x_move_direction );
					y = corresponding_coordinate( dy_dx, x, x_start, y_start );

					x_to_go = VOXEL_WIDTH;
					y_to_go = distance_remaining( Y_ZERO_COORDINATE, y, Y_INCREASING_DIRECTION, y_move_direction, VOXEL_HEIGHT, voxel_y );
				}
				// Else Next Voxel Edge is in y
				else
				{
					//printf(" y_extension < x_extension \n");				
					voxel_y -= y_move_direction;

					y = edge_coordinate( Y_ZERO_COORDINATE, voxel_y, VOXEL_HEIGHT, Z_INCREASING_DIRECTION, y_move_direction );
					x = corresponding_coordinate( dx_dy, y, y_start, x_start );

					x_to_go = distance_remaining( X_ZERO_COORDINATE, x, X_INCREASING_DIRECTION, x_move_direction, VOXEL_WIDTH, voxel_x );
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
				/*if(debug_run && j <= j_high_limit && j >= j_low_limit )
				{
					printf(" x = %3f y = %3f z = %3f\n",  x, y, z );
					printf(" x_to_go = %3f y_to_go = %3f z_to_go = %3f\n",  x_to_go, y_to_go, z_to_go );
					printf("voxel_x_in = %d voxel_y_in = %d voxel_z_in = %d\n", voxel_x, voxel_y, voxel_z);
				}*/
				outside_image = (voxel_x >= COLUMNS ) || (voxel_y >= ROWS ) || (voxel_z >= SLICES ) || (voxel_x < 0  ) || (voxel_y < 0 ) || (voxel_z < 0 );		
				if( !outside_image )
				{
					hit_hull = (image[voxel] == 1);		
					//if( MLP_image_output )
					//{
						//image[voxel] = 4;
					//}
				}
				end_walk = outside_image || hit_hull;
				//j++;
				//if( debug_run )
					//fgets(user_response, sizeof(user_response), stdin);		
			}// end: while( !end_walk )
			//printf("i = %d", i );
		}//end: else: z_start != z_end => z_start == z_end
		if( hit_hull )
		{
			x_object = x;
			y_object = y;
			z_object = z;
		}
		return hit_hull;
}
int find_MLP_path
( 
	int*& path, double*& chord_lengths, 
	double x_in_object, double y_in_object, double z_in_object, double x_out_object, double y_out_object, double z_out_object, 
	double xy_entry_angle, double xz_entry_angle, double xy_exit_angle, double xz_exit_angle,
	int voxel_x, int voxel_y, int voxel_z
)
{
	bool debug_output = false, MLP_image_output = false, constant_chord_lengths = true;
	// MLP calculations variables
	int num_intersections = 0;
	double u_0 = 0, u_1 = MLP_U_STEP,  u_2 = 0;
	double T_0[2] = {0, 0};
	double T_2[2] = {0, 0};
	double V_0[2] = {0, 0};
	double V_2[2] = {0, 0};
	double R_0[4] = { 1.0, 0.0, 0.0 , 1.0}; //a,b,c,d
	double R_1[4] = { 1.0, 0.0, 0.0 , 1.0}; //a,b,c,d
	double R_1T[4] = { 1.0, 0.0, 0.0 , 1.0};  //a,c,b,d

	double sigma_2_pre_1, sigma_2_pre_2, sigma_2_pre_3;
	double sigma_1_coefficient, sigma_t1, sigma_t1_theta1, sigma_theta1, determinant_Sigma_1, Sigma_1I[4];
	double common_sigma_2_term_1, common_sigma_2_term_2, common_sigma_2_term_3;
	double sigma_2_coefficient, sigma_t2, sigma_t2_theta2, sigma_theta2, determinant_Sigma_2, Sigma_2I[4]; 
	double first_term_common_13_1, first_term_common_13_2, first_term_common_24_1, first_term_common_24_2, first_term[4], determinant_first_term;
	double second_term_common_1, second_term_common_2, second_term_common_3, second_term_common_4, second_term[2];
	double t_1, v_1, theta_1, phi_1, x_1, y_1, z_1;
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	//double effective_chord_length = mean_chord_length( u_in_object, t_in_object, v_in_object, u_out_object, t_out_object, v_out_object );
	double effective_chord_length = VOXEL_WIDTH;

	int voxel = voxel_x + voxel_y * COLUMNS + voxel_z * COLUMNS * ROWS;
	path[num_intersections] = voxel;
	if(!constant_chord_lengths)
		chord_lengths[num_intersections] = VOXEL_WIDTH;
	num_intersections++;
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
		
	u_0 = 0;
	u_1 = MLP_U_STEP;
	u_2 = abs(u_out_object - u_in_object);		
	//fgets(user_response, sizeof(user_response), stdin);

	//output_file.open(filename);						
				      
	//precalculated u_2 dependent terms (since u_2 does not change inside while loop)
	//u_2 terms
	sigma_2_pre_1 =  pow(u_2, 3.0) * ( A_0_OVER_3 + u_2 * ( A_1_OVER_12 + u_2 * ( A_2_OVER_30 + u_2 * (A_3_OVER_60 + u_2 * ( A_4_OVER_105 + u_2 * A_5_OVER_168 )))));;	//u_2^3 : 1/3, 1/12, 1/30, 1/60, 1/105, 1/168
	sigma_2_pre_2 =  pow(u_2, 2.0) * ( A_0_OVER_2 + u_2 * (A_1_OVER_6 + u_2 * (A_2_OVER_12 + u_2 * ( A_3_OVER_20 + u_2 * (A_4_OVER_30 + u_2 * A_5_OVER_42)))));	//u_2^2 : 1/2, 1/6, 1/12, 1/20, 1/30, 1/42
	sigma_2_pre_3 =  u_2 * ( A_0 +  u_2 * (A_1_OVER_2 +  u_2 * ( A_2_OVER_3 +  u_2 * ( A_3_OVER_4 +  u_2 * ( A_4_OVER_5 + u_2 * A_5_OVER_6 )))));			//u_2 : 1/1, 1/2, 1/3, 1/4, 1/5, 1/6

	while( u_1 < u_2 - MLP_U_STEP)
	//while( u_1 < u_2 - 0.001)
	{
		R_0[1] = u_1 - u_0;
		R_1[1] = u_2 - u_1;
		R_1T[2] = u_2 - u_1;

		sigma_1_coefficient = pow( E_0 * ( 1 + 0.038 * log( (u_1 - u_0)/X_0) ), 2.0 ) / X_0;
		sigma_t1 =  sigma_1_coefficient * ( pow(u_1, 3.0) * ( A_0_OVER_3 + u_1 * (A_1_OVER_12 + u_1 * (A_2_OVER_30 + u_1 * (A_3_OVER_60 + u_1 * (A_4_OVER_105 + u_1 * A_5_OVER_168 ) )))) );	//u_1^3 : 1/3, 1/12, 1/30, 1/60, 1/105, 1/168
		sigma_t1_theta1 =  sigma_1_coefficient * ( pow(u_1, 2.0) * ( A_0_OVER_2 + u_1 * (A_1_OVER_6 + u_1 * (A_2_OVER_12 + u_1 * (A_3_OVER_20 + u_1 * (A_4_OVER_30 + u_1 * A_5_OVER_42))))) );	//u_1^2 : 1/2, 1/6, 1/12, 1/20, 1/30, 1/42															
		sigma_theta1 = sigma_1_coefficient * ( u_1 * ( A_0 + u_1 * (A_1_OVER_2 + u_1 * (A_2_OVER_3 + u_1 * (A_3_OVER_4 + u_1 * (A_4_OVER_5 + u_1 * A_5_OVER_6))))) );			//u_1 : 1/1, 1/2, 1/3, 1/4, 1/5, 1/6																	
		determinant_Sigma_1 = sigma_t1 * sigma_theta1 - pow( sigma_t1_theta1, 2 );//ad-bc
			
		Sigma_1I[0] = sigma_theta1 / determinant_Sigma_1;
		Sigma_1I[1] = -sigma_t1_theta1 / determinant_Sigma_1;
		Sigma_1I[2] = -sigma_t1_theta1 / determinant_Sigma_1;
		Sigma_1I[3] = sigma_t1 / determinant_Sigma_1;

		//sigma 2 terms
		sigma_2_coefficient = pow( E_0 * ( 1 + 0.038 * log( (u_2 - u_1)/X_0) ), 2.0 ) / X_0;
		common_sigma_2_term_1 = u_1 * ( A_0 + u_1 * (A_1_OVER_2 + u_1 * (A_2_OVER_3 + u_1 * (A_3_OVER_4 + u_1 * (A_4_OVER_5 + u_1 * A_5_OVER_6 )))));
		common_sigma_2_term_2 = pow(u_1, 2.0) * ( A_0_OVER_2 + u_1 * (A_1_OVER_3 + u_1 * (A_2_OVER_4 + u_1 * (A_3_OVER_5 + u_1 * (A_4_OVER_6 + u_1 * A_5_OVER_7 )))));
		common_sigma_2_term_3 = pow(u_1, 3.0) * ( A_0_OVER_3 + u_1 * (A_1_OVER_4 + u_1 * (A_2_OVER_5 + u_1 * (A_3_OVER_6 + u_1 * (A_4_OVER_7 + u_1 * A_5_OVER_8 )))));
		sigma_t2 =  sigma_2_coefficient * ( sigma_2_pre_1 - pow(u_2, 2.0) * common_sigma_2_term_1 + 2 * u_2 * common_sigma_2_term_2 - common_sigma_2_term_3 );
		sigma_t2_theta2 =  sigma_2_coefficient * ( sigma_2_pre_2 - u_2 * common_sigma_2_term_1 + common_sigma_2_term_2 );
		sigma_theta2 =  sigma_2_coefficient * ( sigma_2_pre_3 - common_sigma_2_term_1 );				
		determinant_Sigma_2 = sigma_t2 * sigma_theta2 - pow( sigma_t2_theta2, 2 );//ad-bc

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
		first_term[0] = first_term[3] / determinant_first_term;
		first_term[1] = -first_term[1] / determinant_first_term;
		first_term[2] = -first_term[2] / determinant_first_term;
		first_term[3] = first_term[0] / determinant_first_term;

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

		voxel_x = calculate_voxel( X_ZERO_COORDINATE, x_1, VOXEL_WIDTH );
		voxel_y = calculate_voxel( Y_ZERO_COORDINATE, y_1, VOXEL_HEIGHT );
		voxel_z = calculate_voxel( Z_ZERO_COORDINATE, z_1, VOXEL_THICKNESS);
				
		voxel = voxel_x + voxel_y * COLUMNS + voxel_z * COLUMNS * ROWS;
		//cout << "voxel_x = " << voxel_x << "voxel_y = " << voxel_y << "voxel_z = " << voxel_z << "voxel = " << voxel <<endl;
		//fgets(user_response, sizeof(user_response), stdin);

		if( voxel != path[num_intersections - 1] )
		{
			path[num_intersections] = voxel;
			//MLP_test_image_h[voxel] = 0;
			if(!constant_chord_lengths)
				chord_lengths[num_intersections] = effective_chord_length;						
			num_intersections++;
		}
		u_1 += MLP_U_STEP;
	}
	return num_intersections;
}
void MLP_test()
{
	//char user_response[20];
	//double x_entry = -3.0;
	//double y_entry = -sqrt( pow(MLP_IMAGE_RECON_CYL_RADIUS, 2) - pow(x_entry,2) );
	//double z_entry = 0.0;
	//double x_exit = 2.5;
	//double y_exit = sqrt( pow(MLP_IMAGE_RECON_CYL_RADIUS, 2) - pow(x_exit,2) );
	//double z_exit = 0.0;
	//double xy_entry_angle = 25 * PI/180, xz_entry_angle = 0.0;
	//double xy_exit_angle = 45* PI/180, xz_exit_angle = 0.0;
	double x_entry = 2.5;
	double y_entry = sqrt( pow(MLP_IMAGE_RECON_CYL_RADIUS, 2) - pow(x_entry,2) );
	double z_entry = 0.0;
	double x_exit = -3.0;
	double y_exit = -sqrt( pow(MLP_IMAGE_RECON_CYL_RADIUS, 2) - pow(x_exit,2) );
	double z_exit = 1.0;
	double xy_entry_angle = (45) * PI/180+PI, xz_entry_angle = 0.0;
	double xy_exit_angle = (25)* PI/180+PI, xz_exit_angle = 0.0;
	//double xy_entry_angle = (45+180) * PI/180, xz_entry_angle = 0.0;
	//double xy_exit_angle = (25+180)* PI/180, xz_exit_angle = 0.0;
	/********************************************************************************************/
	/**************************** Status Tracking Information ***********************************/
	/********************************************************************************************/
	double x_in_object, y_in_object, z_in_object;
	double x_out_object, y_out_object, z_out_object;
	bool entered_object = false, exited_object = false;
	int voxel_x, voxel_y, voxel_z;
	int voxel_x_int, voxel_y_int, voxel_z_int;

	entered_object = find_MLP_endpoints( MLP_test_image_h, x_entry, y_entry, z_entry, xy_entry_angle, xz_entry_angle, x_in_object, y_in_object, z_in_object, voxel_x, voxel_y, voxel_z, true);	
	exited_object = find_MLP_endpoints( MLP_test_image_h, x_exit, y_exit, z_exit, xy_exit_angle, xz_exit_angle, x_out_object, y_out_object, z_out_object, voxel_x_int, voxel_y_int, voxel_z_int, false);

	printf("entered object = %d\n", entered_object );
	printf("exited object = %d\n", exited_object );
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	char data_filename[256];
	sprintf(data_filename, "%s%s/%s.txt", OUTPUT_DIRECTORY, OUTPUT_FOLDER, MLP_PATH_FILENAME );
	FILE* pFile = create_MLP_path_file( data_filename );

	int* path = (int*)calloc( MAX_INTERSECTIONS, sizeof(int));
	double* chord_lengths = (double*)calloc( MAX_INTERSECTIONS, sizeof(double));	
	int num_intersections = 0;

	//fgets(user_response, sizeof(user_response), stdin);

	//char filename[256];
	//std::ofstream output_file;
	//sprintf( filename, "%s%s/%s.txt", OUTPUT_DIRECTORY, OUTPUT_FOLDER, "path_test");
	//output_file.open(filename);						

	int j = 0;
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	if( entered_object && exited_object )
	{
		//char data_filename[256];	

		num_intersections = find_MLP_path( path, chord_lengths, x_in_object, y_in_object, z_in_object, x_out_object, y_out_object, z_out_object, xy_entry_angle, xz_entry_angle, xy_exit_angle, xz_exit_angle, voxel_x, voxel_y, voxel_z);
		cout << "num_intersections = " << num_intersections << endl;
		//output_file << endl;
		//output_file.close();
		path_data_2_disk(data_filename, pFile, num_intersections, path, path, true);
	}
}
void MLP_test2()
{
	//char user_response[20];
	//double x_entry = -3.0;
	//double y_entry = -sqrt( pow(MLP_IMAGE_RECON_CYL_RADIUS, 2) - pow(x_entry,2) );
	//double z_entry = 0.0;
	//double x_exit = 2.5;
	//double y_exit = sqrt( pow(MLP_IMAGE_RECON_CYL_RADIUS, 2) - pow(x_exit,2) );
	//double z_exit = 0.0;
	//double xy_entry_angle = 25 * PI/180, xz_entry_angle = 0.0;
	//double xy_exit_angle = 45* PI/180, xz_exit_angle = 0.0;
	double x_entry = 2.5;
	double y_entry = sqrt( pow(MLP_IMAGE_RECON_CYL_RADIUS, 2) - pow(x_entry,2) );
	double z_entry = 0.0;
	double x_exit = -3.0;
	double y_exit = -sqrt( pow(MLP_IMAGE_RECON_CYL_RADIUS, 2) - pow(x_exit,2) );
	double z_exit = 1.0;
	double xy_entry_angle = (45) * PI/180+PI, xz_entry_angle = 0.0;
	double xy_exit_angle = (25)* PI/180+PI, xz_exit_angle = 0.0;
	//double xy_entry_angle = (45+180) * PI/180, xz_entry_angle = 0.0;
	//double xy_exit_angle = (25+180)* PI/180, xz_exit_angle = 0.0;
	

	
	//pFile = fopen (data_filename,"w+");
	//path_data_2_disk(data_filename, pFile, num_elements, intersections, voxel_numbers, false);

	/********************************************************************************************/
	/**************************** Status Tracking Information ***********************************/
	/********************************************************************************************/
	//int x_move_direction, y_move_direction, z_move_direction;
	//double x, y, z;
	//double x_inside, y_inside, z_inside;
	//double x_to_go, y_to_go, z_to_go;
	double x_in_object, y_in_object, z_in_object;
	double u_in_object, t_in_object, v_in_object;
	double x_out_object, y_out_object, z_out_object;
	double u_out_object, t_out_object, v_out_object;
	bool entered_object = false, exited_object = false;
	int voxel_x, voxel_y, voxel_z, voxel;
	int voxel_x_int, voxel_y_int, voxel_z_int;

	entered_object = find_MLP_endpoints( MLP_test_image_h, x_entry, y_entry, z_entry, xy_entry_angle, xz_entry_angle, x_in_object, y_in_object, z_in_object, voxel_x, voxel_y, voxel_z, true);	
	exited_object = find_MLP_endpoints( MLP_test_image_h, x_exit, y_exit, z_exit, xy_exit_angle, xz_exit_angle, x_out_object, y_out_object, z_out_object, voxel_x_int, voxel_y_int, voxel_z_int, false);

	printf("entered object = %d\n", entered_object );
	printf("exited object = %d\n", exited_object );
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	char data_filename[256];
	//sprintf(data_filename, "%s%s/%s.txt", OUTPUT_DIRECTORY, OUTPUT_FOLDER, MLP_PATH_FILENAME );
	//FILE* pFile = create_MLP_path_file( data_filename );

	int* path = (int*)calloc( MAX_INTERSECTIONS, sizeof(int));
	double* chord_lengths = (double*)calloc( MAX_INTERSECTIONS, sizeof(double));	
	int path_index = 0;

	double T_0[2] = {0, 0}, T_2[2] = {0, 0}, V_0[2] = {0, 0}, V_2[2] = {0, 0};
	double u_0 = 0, u_1 = MLP_U_STEP,  u_2 = 0;
	//fgets(user_response, sizeof(user_response), stdin);

	char filename[256];
	std::ofstream output_file;
	sprintf( filename, "%s%s/%s.txt", OUTPUT_DIRECTORY, OUTPUT_FOLDER, "path_test");
	//output_file.open(filename);						

	

	int j = 0;
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	double R_0[4] = { 1.0, 0.0, 0.0 , 1.0}; //a,b,c,d
	//double R_0T[4] = { 1.0, 0.0, 0.0 , 1.0}; //a,c,b,d
	double R_1[4] = { 1.0, 0.0, 0.0 , 1.0}; //a,b,c,d
	double R_1T[4] = { 1.0, 0.0, 0.0 , 1.0};  //a,c,b,d

	//double sigma_1_pre_1, sigma_1_pre_2, sigma_1_pre_3;
	double sigma_2_pre_1, sigma_2_pre_2, sigma_2_pre_3;

	double sigma_1_coefficient, sigma_t1, sigma_t1_theta1, sigma_theta1, determinant_Sigma_1, Sigma_1I[4];
	double common_sigma_2_term_1, common_sigma_2_term_2, common_sigma_2_term_3;
	double sigma_2_coefficient, sigma_t2, sigma_t2_theta2, sigma_theta2, determinant_Sigma_2, Sigma_2I[4]; 
	double first_term_common_13_1, first_term_common_13_2, first_term_common_24_1, first_term_common_24_2, first_term[4], determinant_first_term;
	double second_term_common_1, second_term_common_2, second_term_common_3, second_term_common_4, second_term[2];
	double t_1, v_1;
	//double theta_1, phi_1;
	double x_1, y_1, z_1;


	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


	if( entered_object && exited_object )
	{
		//char data_filename[256];
		sprintf(data_filename, "%s%s/%s.txt", OUTPUT_DIRECTORY, OUTPUT_FOLDER, MLP_PATH_FILENAME );
		FILE* pFile = create_MLP_path_file( data_filename );

		voxel = int(voxel_x + voxel_y * MLP_IMAGE_COLUMNS + voxel_z * MLP_IMAGE_COLUMNS * MLP_IMAGE_ROWS);
		//int path[MAX_INTERSECTIONS];
		//double chord_lengths[MAX_INTERSECTIONS];
		//int* path = (int*)calloc( MAX_INTERSECTIONS, sizeof(int));
		//double* chord_lengths = (double*)calloc( MAX_INTERSECTIONS, sizeof(double));	
		path_index = 0;
		MLP_test_image_h[voxel] = 0;
		path[path_index] = voxel;
		chord_lengths[path_index] = 1.0;
		path_index++;

		u_in_object = ( cos( xy_entry_angle ) * x_in_object ) + ( sin( xy_entry_angle ) * y_in_object );
		u_out_object = ( cos( xy_entry_angle ) * x_out_object ) + ( sin( xy_entry_angle ) * y_out_object );
		t_in_object = ( cos( xy_entry_angle ) * y_in_object ) - ( sin( xy_entry_angle ) * x_in_object );
		t_out_object = ( cos( xy_entry_angle ) * y_out_object ) - ( sin( xy_entry_angle ) * x_out_object );
		v_in_object = z_in_object;
		v_out_object = z_out_object;

		if( u_in_object > u_out_object )
		{
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
		
		u_0 = 0;
		u_1 = MLP_U_STEP;
		u_2 = abs(u_out_object - u_in_object);		
		//fgets(user_response, sizeof(user_response), stdin);

		//output_file.open(filename);						

		//precalculated u_0/u_2 terms
		//u_0 terms
		//double sigma_1_pre_1 =  A_0 * u_0 + A_1_OVER_2 * pow(u_0, 2.0) + A_2_OVER_3 * pow(u_0, 3.0) + A_3_OVER_4 * pow(u_0, 4.0) + A_4_OVER_5 * pow(u_0, 5.0) + A_5_OVER_6 * pow(u_0, 6.0);						//1, 1/2, 1/3, 1/4, 1/5, 1/6
		//double sigma_1_pre_2 =  A_0_OVER_2 * pow(u_0, 2.0) + A_1_OVER_3 * pow(u_0, 3.0) + A_2_OVER_4 * pow(u_0, 4.0) + A_3_OVER_5 * pow(u_0, 5.0) + A_4_OVER_6 * pow(u_0, 6.0) + A_5_OVER_7 * pow(u_0, 7.0);	//1/2, 1/3, 1/4, 1/5, 1/6, 1/7
		//double sigma_1_pre_3 =  A_0_OVER_3 * pow(u_0, 3.0) + A_1_OVER_4 * pow(u_0, 4.0) + A_2_OVER_5 * pow(u_0, 5.0) + A_3_OVER_6 * pow(u_0, 6.0) + A_4_OVER_7 * pow(u_0, 7.0) + A_5_OVER_8 * pow(u_0, 8.0);	//1/3, 1/4, 1/5, 1/6, 1/7, 1/8
		//u_2 terms
		sigma_2_pre_1 =  pow(u_2, 3.0) * ( A_0_OVER_3 + u_2 * ( A_1_OVER_12 + u_2 * ( A_2_OVER_30 + u_2 * (A_3_OVER_60 + u_2 * ( A_4_OVER_105 + u_2 * A_5_OVER_168 )))));;	//u_2^3 : 1/3, 1/12, 1/30, 1/60, 1/105, 1/168
		sigma_2_pre_2 =  pow(u_2, 2.0) * ( A_0_OVER_2 + u_2 * (A_1_OVER_6 + u_2 * (A_2_OVER_12 + u_2 * ( A_3_OVER_20 + u_2 * (A_4_OVER_30 + u_2 * A_5_OVER_42)))));	//u_2^2 : 1/2, 1/6, 1/12, 1/20, 1/30, 1/42
		sigma_2_pre_3 =  u_2 * ( A_0 +  u_2 * (A_1_OVER_2 +  u_2 * ( A_2_OVER_3 +  u_2 * ( A_3_OVER_4 +  u_2 * ( A_4_OVER_5 + u_2 * A_5_OVER_6 )))));			//u_2 : 1/1, 1/2, 1/3, 1/4, 1/5, 1/6

		j = 0;
		//while( u_1 <= u_2 )
		while( u_1 <= u_2 - MLP_U_STEP )
		{
			j++;
			R_0[1] = u_1 - u_0;
			//R_0T[2] = u_1 - u_0;
			R_1[1] = u_2 - u_1;
			R_1T[2] = u_2 - u_1;

			//double sigma_1_coefficient = 1.0;
			sigma_1_coefficient = pow( E_0 * ( 1 + 0.038 * log( (u_1 - u_0)/X_0) ), 2.0 ) / X_0;
			sigma_t1 =  sigma_1_coefficient * ( pow(u_1, 3.0) * ( A_0_OVER_3 + u_1 * (A_1_OVER_12 + u_1 * (A_2_OVER_30 + u_1 * (A_3_OVER_60 + u_1 * (A_4_OVER_105 + u_1 * A_5_OVER_168 ) )))) );	//u_1^3 : 1/3, 1/12, 1/30, 1/60, 1/105, 1/168
			sigma_t1_theta1 =  sigma_1_coefficient * ( pow(u_1, 2.0) * ( A_0_OVER_2 + u_1 * (A_1_OVER_6 + u_1 * (A_2_OVER_12 + u_1 * (A_3_OVER_20 + u_1 * (A_4_OVER_30 + u_1 * A_5_OVER_42))))) );	//u_1^2 : 1/2, 1/6, 1/12, 1/20, 1/30, 1/42															
			sigma_theta1 = sigma_1_coefficient * ( u_1 * ( A_0 + u_1 * (A_1_OVER_2 + u_1 * (A_2_OVER_3 + u_1 * (A_3_OVER_4 + u_1 * (A_4_OVER_5 + u_1 * A_5_OVER_6))))) );			//u_1 : 1/1, 1/2, 1/3, 1/4, 1/5, 1/6																	
			determinant_Sigma_1 = sigma_t1 * sigma_theta1 - pow( sigma_t1_theta1, 2 );//ad-bc
			
			if( j == 1)
			{
				cout << "sigma_t1 = " << sigma_t1 << "sigma_t1 = " << sigma_t1/sigma_1_coefficient << endl;
				cout << "sigma_t1_theta1 = " << sigma_t1_theta1 << "sigma_t1_theta1 = " << sigma_t1_theta1/sigma_1_coefficient<< endl;
				cout << "sigma_theta1 = " << sigma_theta1 << "sigma_theta1 = " << sigma_theta1/sigma_1_coefficient<< endl;
				cout << "determinant_Sigma_1 = " << determinant_Sigma_1 << "determinant_Sigma_1 = " << determinant_Sigma_1/sigma_1_coefficient<< endl;
			}
			Sigma_1I[0] = sigma_theta1 / determinant_Sigma_1;
			Sigma_1I[1] = -sigma_t1_theta1 / determinant_Sigma_1;
			Sigma_1I[2] = -sigma_t1_theta1 / determinant_Sigma_1;
			Sigma_1I[3] = sigma_t1 / determinant_Sigma_1;
			//sigma 2 terms
			//double sigma_2_coefficient = 1.0;
			sigma_2_coefficient = pow( E_0 * ( 1 + 0.038 * log( (u_2 - u_1)/X_0) ), 2.0 ) / X_0;
			common_sigma_2_term_1 = u_1 * ( A_0 + u_1 * (A_1_OVER_2 + u_1 * (A_2_OVER_3 + u_1 * (A_3_OVER_4 + u_1 * (A_4_OVER_5 + u_1 * A_5_OVER_6 )))));
			common_sigma_2_term_2 = pow(u_1, 2.0) * ( A_0_OVER_2 + u_1 * (A_1_OVER_3 + u_1 * (A_2_OVER_4 + u_1 * (A_3_OVER_5 + u_1 * (A_4_OVER_6 + u_1 * A_5_OVER_7 )))));
			common_sigma_2_term_3 = pow(u_1, 3.0) * ( A_0_OVER_3 + u_1 * (A_1_OVER_4 + u_1 * (A_2_OVER_5 + u_1 * (A_3_OVER_6 + u_1 * (A_4_OVER_7 + u_1 * A_5_OVER_8 )))));
			sigma_t2 =  sigma_2_coefficient * ( sigma_2_pre_1 - pow(u_2, 2.0) * common_sigma_2_term_1 + 2 * u_2 * common_sigma_2_term_2 - common_sigma_2_term_3 );
			sigma_t2_theta2 =  sigma_2_coefficient * ( sigma_2_pre_2 - u_2 * common_sigma_2_term_1 + common_sigma_2_term_2 );
			sigma_theta2 =  sigma_2_coefficient * ( sigma_2_pre_3 - common_sigma_2_term_1 );				
			determinant_Sigma_2 = sigma_t2 * sigma_theta2 - pow( sigma_t2_theta2, 2 );//ad-bc
	
			if( j == 1)
			{
				cout << "sigma_t2 = " << sigma_t2 << "sigma_t2 = " << sigma_t2/sigma_2_coefficient << endl;
				cout << "sigma_t2_theta2 = " << sigma_t2_theta2 << "sigma_t2_theta2 = " << sigma_t2_theta2/sigma_2_coefficient<< endl;
				cout << "sigma_theta2 = " << sigma_theta2 << "sigma_theta2 = " << sigma_theta2/sigma_2_coefficient<< endl;
				cout << "determinant_Sigma_2 = " << determinant_Sigma_2 << "determinant_Sigma_2 = " << determinant_Sigma_2/sigma_2_coefficient<< endl;
			}
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
			first_term[0] = first_term[3] / determinant_first_term;
			first_term[1] = -first_term[1] / determinant_first_term;
			first_term[2] = -first_term[2] / determinant_first_term;
			first_term[3] = first_term[0] / determinant_first_term;

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
			z_1 = v_in_object + v_1;

			voxel_x = calculate_voxel( X_ZERO_COORDINATE, x_1, VOXEL_WIDTH );
			voxel_y = calculate_voxel( Y_ZERO_COORDINATE, y_1, VOXEL_HEIGHT );
			voxel_z = calculate_voxel( Z_ZERO_COORDINATE, z_1, VOXEL_THICKNESS);
			voxel = voxel_x + voxel_y * MLP_IMAGE_COLUMNS + voxel_z * MLP_IMAGE_COLUMNS * MLP_IMAGE_ROWS;

			if( voxel != path[path_index - 1] )
			{
				path[path_index] = voxel;
				chord_lengths[path_index] = 1.0;
				MLP_test_image_h[voxel] = 0;
				output_file << path[path_index] << " ";
				path_index++;
			}
			u_1 += MLP_U_STEP;
		}
		output_file << endl;
		output_file.close();
		path_data_2_disk(data_filename, pFile, path_index, path, path, true);
	}
}
void MLP()
{
	//char user_response[20];
	//char MLP_test_filename[128];
	char data_filename[256];
	//char updated_image_filename[128];
	//char filename[256];
	char iterate_filename[256];
	double x_entry, y_entry, z_entry, x_exit, y_exit, z_exit;
	double xy_entry_angle, xz_entry_angle, xy_exit_angle, xz_exit_angle;
	double x_in_object, y_in_object, z_in_object, x_out_object, y_out_object, z_out_object;
	double effective_chord_length; 	
	int voxel_x, voxel_y, voxel_z, voxel_x_int, voxel_y_int, voxel_z_int;
	unsigned int num_intersections;
	bool entered_object = false, exited_object = false, debug_output = false, MLP_image_output = false, constant_chord_lengths = true;

	int* path = (int*)calloc( MAX_INTERSECTIONS, sizeof(int));
	double* chord_lengths = (double*)calloc( MAX_INTERSECTIONS, sizeof(double));
	//if(!constant_chord_lengths)
		//chord_lengths = (double*)calloc( MAX_INTERSECTIONS, sizeof(double));	
	double* x_update_h = (double*)calloc( NUM_VOXELS, sizeof(double));

	sprintf(data_filename, "%s%s/%s.txt", OUTPUT_DIRECTORY, OUTPUT_FOLDER, MLP_PATH_FILENAME );
	FILE* pFile = create_MLP_path_file( data_filename );
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	unsigned int block_history = 1;
	//int start_history = 3*x_entry_vector.size()/4;
	unsigned int start_history = 0;
	//int start_history = 10;
	//int end_history = start_history + 12;
	unsigned int end_history = x_entry_vector.size();
	//unsigned int iterations = 20;
	//int end_history = x_entry_vector.size();
	//array_2_disk( "MLP_image_init", OUTPUT_DIRECTORY, OUTPUT_FOLDER, x_hull_h, COLUMNS, ROWS, SLICES, NUM_VOXELS, true );
	//cout << "#histories = " << x_entry_vector.size() << " " << post_cut_histories << endl; 
	//cout << "start i = " << start_history << endl;
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	for( unsigned int iteration = 1; iteration <= ITERATIONS; iteration++ )
	{
		for( unsigned int i = start_history; i < end_history; i++ )
		{		
			x_entry = x_entry_vector[i], y_entry = y_entry_vector[i], z_entry = z_entry_vector[i];
			x_exit = x_exit_vector[i], y_exit = y_exit_vector[i], z_exit = z_exit_vector[i];
			xy_entry_angle = xy_entry_angle_vector[i], xz_entry_angle = xz_entry_angle_vector[i];
			xy_exit_angle = xy_exit_angle_vector[i], xz_exit_angle = xz_exit_angle_vector[i];

			/********************************************************************************************/
			/**************************** Status Tracking Information ***********************************/
			/********************************************************************************************/
			entered_object = false;
			exited_object = false;

			entered_object = find_MLP_endpoints( x_hull_h, x_entry, y_entry, z_entry, xy_entry_angle, xz_entry_angle, x_in_object, y_in_object, z_in_object, voxel_x, voxel_y, voxel_z, true);	
			exited_object = find_MLP_endpoints( x_hull_h, x_exit, y_exit, z_exit, xy_exit_angle, xz_exit_angle, x_out_object, y_out_object, z_out_object, voxel_x_int, voxel_y_int, voxel_z_int, false);

			num_intersections = 0;

			if( entered_object && exited_object )
			{
				//effective_chord_length = mean_chord_length( u_in_object, t_in_object, v_in_object, u_out_object, t_out_object, v_out_object );
				effective_chord_length = VOXEL_WIDTH;

				num_intersections = find_MLP_path( path, chord_lengths, x_in_object, y_in_object, z_in_object, x_out_object, y_out_object, z_out_object, xy_entry_angle, xz_entry_angle, xy_exit_angle, xz_exit_angle, voxel_x, voxel_y, voxel_z);
				//if(constant_chord_lengths)
				update_iterate22( WEPL_vector[i], effective_chord_length, x_h, path, num_intersections );
				//update_iterate2( WEPL_vector[i], effective_chord_length, x_h, path, num_intersections );
				//else
					//update_iterate( WEPL_vector[i], chord_lengths, x_h, path, num_intersections );
				//if( ( i + 1 ) % BLOCK_SIZE == 0 )
				//path_data_2_disk(char* data_filename, FILE* pFile, int voxel_intersections, int* voxel_numbers, T*& data, bool write_sparse)
				//path_data_2_disk(data_filename, pFile, num_intersections, path, path, true);
				//calculate_update( WEPL_vector[i], effective_chord_length, x_update_h, path, num_intersections );
				//if( block_history == BLOCK_SIZE )
				//{					
				//	sprintf(updated_image_filename, "%s_%d_%d", "update_image", i, block_history );
				//	array_2_disk(updated_image_filename, OUTPUT_DIRECTORY, OUTPUT_FOLDER, x_h, COLUMNS, ROWS, SLICES, NUM_VOXELS, true );
				//	update_iterate3( x_h, x_update_h );
				//	block_history = 0;
				//}
				//block_history++;
			}	
		}
		sprintf(iterate_filename, "%s%d", "x_", iteration );		
		if( WRITE_X_KI )
			array_2_disk(iterate_filename, OUTPUT_DIRECTORY, OUTPUT_FOLDER, x_h, COLUMNS, ROWS, SLICES, NUM_VOXELS, true );
	}
}
void MLP2()
{
	//char user_response[20];
	//char MLP_test_filename[128];
	char data_filename[256];
	//char updated_image_filename[128];
	//char filename[256];
	char iterate_filename[256];

	double x_entry, y_entry, z_entry, x_exit, y_exit, z_exit;
	double xy_entry_angle, xz_entry_angle, xy_exit_angle, xz_exit_angle;
	double x_in_object, y_in_object, z_in_object, x_out_object, y_out_object, z_out_object;
	double effective_chord_length; 

	
	int voxel_x, voxel_y, voxel_z, voxel_x_int, voxel_y_int, voxel_z_int;
	int num_intersections;

	bool entered_object = false, exited_object = false, debug_output = false, MLP_image_output = false, constant_chord_lengths = true;

	int* path = (int*)calloc( MAX_INTERSECTIONS, sizeof(int));
	double* chord_lengths = (double*)calloc( MAX_INTERSECTIONS, sizeof(double));
	//if(!constant_chord_lengths)
		//chord_lengths = (double*)calloc( MAX_INTERSECTIONS, sizeof(double));	
	double* x_update_h = (double*)calloc( NUM_VOXELS, sizeof(double));

	sprintf(data_filename, "%s%s/%s.txt", OUTPUT_DIRECTORY, OUTPUT_FOLDER, MLP_PATH_FILENAME );
	FILE* pFile = create_MLP_path_file( data_filename );
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	int block_history = 1;
	//int start_history = 3*x_entry_vector.size()/4;
	int start_history = 0;
	//int start_history = 10;
	//int end_history = start_history + 12;
	int end_history = x_entry_vector.size();
	int iterations = 20;
	//int end_history = x_entry_vector.size();
	//array_2_disk( "MLP_image_init", OUTPUT_DIRECTORY, OUTPUT_FOLDER, x_hull_h, COLUMNS, ROWS, SLICES, NUM_VOXELS, true );
	//cout << "#histories = " << x_entry_vector.size() << " " << post_cut_histories << endl; 
	//cout << "start i = " << start_history << endl;
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	for( int iteration = 1; iteration <= iterations; iteration++ )
	{
		for( int i = start_history; i < end_history; i++ )
		{		
			x_entry = x_entry_vector[i];
			y_entry = y_entry_vector[i];
			z_entry = z_entry_vector[i];
			x_exit = x_exit_vector[i];
			y_exit = y_exit_vector[i];
			z_exit = z_exit_vector[i];
			xy_entry_angle = xy_entry_angle_vector[i];
			xz_entry_angle = xz_entry_angle_vector[i];
			xy_exit_angle = xy_exit_angle_vector[i];;
			xz_exit_angle = xz_exit_angle_vector[i];

			/********************************************************************************************/
			/**************************** Status Tracking Information ***********************************/
			/********************************************************************************************/
			entered_object = false;
			exited_object = false;

			entered_object = find_MLP_endpoints( x_hull_h, x_entry, y_entry, z_entry, xy_entry_angle, xz_entry_angle, x_in_object, y_in_object, z_in_object, voxel_x, voxel_y, voxel_z, true);	
			exited_object = find_MLP_endpoints( x_hull_h, x_exit, y_exit, z_exit, xy_exit_angle, xz_exit_angle, x_out_object, y_out_object, z_out_object, voxel_x_int, voxel_y_int, voxel_z_int, false);

			num_intersections = 0;

			if( entered_object && exited_object )
			{
				//effective_chord_length = mean_chord_length( u_in_object, t_in_object, v_in_object, u_out_object, t_out_object, v_out_object );
				effective_chord_length = VOXEL_WIDTH;

				num_intersections = find_MLP_path( path, chord_lengths, x_in_object, y_in_object, z_in_object, x_out_object, y_out_object, z_out_object, xy_entry_angle, xz_entry_angle, xy_exit_angle, xz_exit_angle, voxel_x, voxel_y, voxel_z);
				//if(constant_chord_lengths)
				update_iterate2( WEPL_vector[i], effective_chord_length, x_h, path, num_intersections );
				//else
					//update_iterate( WEPL_vector[i], chord_lengths, x_h, path, num_intersections );
				//if( ( i + 1 ) % BLOCK_SIZE == 0 )
				//path_data_2_disk(char* data_filename, FILE* pFile, int voxel_intersections, int* voxel_numbers, T*& data, bool write_sparse)
				//path_data_2_disk(data_filename, pFile, num_intersections, path, path, true);
				//calculate_update( WEPL_vector[i], effective_chord_length, x_update_h, path, num_intersections );
				//if( block_history == BLOCK_SIZE )
				//{					
				//	sprintf(updated_image_filename, "%s_%d_%d", "update_image", i, block_history );
				//	array_2_disk(updated_image_filename, OUTPUT_DIRECTORY, OUTPUT_FOLDER, x_h, COLUMNS, ROWS, SLICES, NUM_VOXELS, true );
				//	update_iterate3( x_h, x_update_h );
				//	block_history = 0;
				//}
				//block_history++;
			}	
		}
		sprintf(iterate_filename, "%s%d", "x_", iteration );		
		if( WRITE_X_KI )
			array_2_disk(iterate_filename, OUTPUT_DIRECTORY, OUTPUT_FOLDER, x_h, COLUMNS, ROWS, SLICES, NUM_VOXELS, true );
	}
}
double mean_chord_length( double x_entry, double y_entry, double z_entry, double x_exit, double y_exit, double z_exit )
{

	double xy_angle = atan2( y_exit - y_entry, x_exit - x_entry);
	double xz_angle = atan2( z_exit - z_entry, x_exit - x_entry);

	double int_part;
	double reduced_angle = modf( xy_angle/(PI/2), &int_part );
	double effective_angle_ut = fabs(reduced_angle);
	double effective_angle_uv = fabs(xz_angle );
	//
	double average_pixel_size = ( VOXEL_WIDTH + VOXEL_HEIGHT) / 2;
	double s = MLP_U_STEP;
	double l = average_pixel_size;

	double sin_ut_angle = sin(effective_angle_ut);
	double sin_2_ut_angle = sin(2 * effective_angle_ut);
	double cos_ut_angle = cos(effective_angle_ut);

	double sin_uv_angle = sin(effective_angle_uv);
	double sin_2_uv_angle = sin(2 * effective_angle_uv);
	double cos_uv_angle = cos(effective_angle_uv);

	double sum_ut_angles = sin(effective_angle_ut) + cos(effective_angle_ut);
	double sum_uv_angles = sin(effective_angle_uv) + cos(effective_angle_uv);
	
	////		(L/3) { [(s/L)^2 S{2O} - 6] / [(s/L)S{2O} - 2(C{O} + S{O}) ] } + { [(s/L)^2 S{2O}] / [ 2(C{O} + S{O})] } = (L/3)*[( (s/L)^3 * S{2O}^2 - 12 (C{O} + S{O}) ) / ( 2(s/L)*S{2O}*(C{O} + S{O}) - 4(C{O} + S{O})^2 ]
	////		

	//double chord_length_t = ( l / 6.0 * sum_ut_angles) * ( pow(s/l, 3.0) * pow( sin(2 * effective_angle_ut), 2.0 ) - 12 * sum_ut_angles ) / ( (s/l) * sin(2 * effective_angle_ut) - 2 * sum_ut_angles );
	//
	//// Multiply this by the effective chord in the v-u plane
	//double mean_pixel_width = average_pixel_size / sum_ut_angles;
	//double height_fraction = SLICE_THICKNESS / mean_pixel_width;
	//s = MLP_U_STEP;
	//l = mean_pixel_width;
	//double chord_length_v = ( l / (6.0 * height_fraction * sum_uv_angles) ) * ( pow(s/l, 3.0) * pow( sin(2 * effective_angle_uv), 2.0 ) - 12 * height_fraction * sum_uv_angles ) / ( (s/l) * sin(2 * effective_angle_uv) - 2 * height_fraction * sum_uv_angles );
	//return sqrt(chord_length_t * chord_length_t + chord_length_v*chord_length_v);

	double eff_angle_t,eff_angle_v;
	
	//double xy_angle = atan2( y_exit - y_entry, x_exit - x_entry);
	//double xz_angle = atan2( z_exit - z_entry, x_exit - x_entry);

	//eff_angle_t = modf( xy_angle/(PI/2), &int_part );
	//eff_angle_t = fabs( eff_angle_t);
	eff_angle_t = xy_angle - ( int( xy_angle/(PI/2) ) ) * (PI/2);
	//cout << "eff angle t = " << eff_angle_t << endl;
	eff_angle_v=fabs(xz_angle);
	
	// Get the effective chord in the t-u plane
	double step_fraction=MLP_U_STEP/VOXEL_WIDTH;
	double chord_length_2D=(1/3.0)*((step_fraction*step_fraction*sin(2*eff_angle_t)-6)/(step_fraction*sin(2*eff_angle_t)-2*(cos(eff_angle_t)+sin(eff_angle_t))) + step_fraction*step_fraction*sin(2*eff_angle_t)/(2*(cos(eff_angle_t)+sin(eff_angle_t))));
	
	// Multiply this by the effective chord in the v-u plane
	double mean_pixel_width=VOXEL_WIDTH/(cos(eff_angle_t)+sin(eff_angle_t));
	double height_fraction=SLICE_THICKNESS/mean_pixel_width;
	step_fraction=MLP_U_STEP/mean_pixel_width;
	double chord_length_3D=(1/3.0)*((step_fraction*step_fraction*sin(2*eff_angle_v)-6*height_fraction)/(step_fraction*sin(2*eff_angle_v)-2*(height_fraction*cos(eff_angle_v)+sin(eff_angle_v))) + step_fraction*step_fraction*sin(2*eff_angle_v)/(2*(height_fraction*cos(eff_angle_v)+sin(eff_angle_v))));
	
	//cout << "2D = " << chord_length_2D << " 3D = " << chord_length_3D << endl;
	return VOXEL_WIDTH*chord_length_2D*chord_length_3D;
}
/***********************************************************************************************************************************************************************************************************************/
/******************************************************************************************************* MLP (GPU) *****************************************************************************************************/
/***********************************************************************************************************************************************************************************************************************/
//template<typename O> __device__ bool find_MLP_endpoints_GPU
//( 
//	O*& image, double x_start, double y_start, double z_start, double xy_angle, double xz_angle, 
//	double& x_object, double& y_object, double& z_object, int& voxel_x, int& voxel_y, int& voxel_z, bool entering
//)
//{	
//		//char user_response[20];
//
//		/********************************************************************************************/
//		/********************************* Voxel Walk Parameters ************************************/
//		/********************************************************************************************/
//		int x_move_direction, y_move_direction, z_move_direction;
//		double delta_yx, delta_zx, delta_zy;
//		/********************************************************************************************/
//		/**************************** Status Tracking Information ***********************************/
//		/********************************************************************************************/
//		double x = x_start, y = y_start, z = z_start;
//		double x_to_go, y_to_go, z_to_go;		
//		double x_extension, y_extension;	
//		//int voxel_x, voxel_y, voxel_z;
//		//int voxel_x_out, voxel_y_out, voxel_z_out; 
//		int voxel; 
//		bool hit_hull = false, end_walk, outside_image;
//		// true false
//		bool debug_run = false;
//		bool MLP_image_output = false;
//		/********************************************************************************************/
//		/******************** Initial Conditions and Movement Characteristics ***********************/
//		/********************************************************************************************/	
//		if( !entering )
//		{
//			xy_angle += PI;
//		}
//		x_move_direction = ( cos(xy_angle) >= 0 ) - ( cos(xy_angle) <= 0 );
//		y_move_direction = ( sin(xy_angle) >= 0 ) - ( sin(xy_angle) <= 0 );
//		z_move_direction = ( sin(xz_angle) >= 0 ) - ( sin(xz_angle) <= 0 );
//		if( x_move_direction < 0 )
//		{
//			//if( debug_run )
//				//puts("z switched");
//			z_move_direction *= -1;
//		}
//		/*if( debug_run )
//		{
//			cout << "x_move_direction = " << x_move_direction << endl;
//			cout << "y_move_direction = " << y_move_direction << endl;
//			cout << "z_move_direction = " << z_move_direction << endl;
//		}*/
//		
//
//
//		voxel_x = calculate_voxel( X_ZERO_COORDINATE, x, VOXEL_WIDTH );
//		voxel_y = calculate_voxel( Y_ZERO_COORDINATE, y, VOXEL_HEIGHT );
//		voxel_z = calculate_voxel( Z_ZERO_COORDINATE, z, VOXEL_THICKNESS );
//
//		x_to_go = distance_remaining( X_ZERO_COORDINATE, x, X_INCREASING_DIRECTION, x_move_direction, VOXEL_WIDTH, voxel_x );
//		y_to_go = distance_remaining( Y_ZERO_COORDINATE, y, Y_INCREASING_DIRECTION, y_move_direction, VOXEL_HEIGHT, voxel_y );	
//		z_to_go = distance_remaining( Z_ZERO_COORDINATE, z, Z_INCREASING_DIRECTION, z_move_direction, VOXEL_THICKNESS, voxel_z );
//
//		voxel = voxel_x + voxel_y * COLUMNS + voxel_z * COLUMNS * ROWS;
//		/********************************************************************************************/
//		/***************************** Path and Walk Information ************************************/
//		/********************************************************************************************/
//		// Lengths/Distances as x is Incremented One Voxel tan( xy_hit_hull_angle )
//		delta_yx = fabs(tan(xy_angle));
//		delta_zx = fabs(tan(xz_angle));
//		delta_zy = fabs( tan(xz_angle)/tan(xy_angle));
//
//		double dy_dx = tan(xy_angle);
//		double dz_dx = tan(xz_angle);
//		double dz_dy = tan(xz_angle)/tan(xy_angle);
//
//		double dx_dy = pow( tan(xy_angle), -1.0 );
//		double dx_dz = pow( tan(xz_angle), -1.0 );
//		double dy_dz = tan(xy_angle)/tan(xz_angle);
//
//		//if( debug_run )
//		//{
//		//	cout << "delta_yx = " << delta_yx << "delta_zx = " << delta_zx << "delta_zy = " << delta_zy << endl;
//		//	cout << "dy_dx = " << dy_dx << "dz_dx = " << dz_dx << "dz_dy = " << dz_dy << endl;
//		//	cout << "dx_dy = " << dx_dy << "dx_dz = " << dx_dz << "dy_dz = " << dy_dz << endl;
//		//}
//		/********************************************************************************************/
//		/************************* Initialize and Check Exit Conditions *****************************/
//		/********************************************************************************************/
//		outside_image = (voxel_x >= COLUMNS ) || (voxel_y >= ROWS ) || (voxel_z >= SLICES ) || (voxel_x < 0  ) || (voxel_y < 0 ) || (voxel_z < 0 );		
//		if( !outside_image )
//		{
//			hit_hull = (image[voxel] == 1);		
//			//image[voxel] = 4;
//		}
//		end_walk = outside_image || hit_hull;
//		//int j = 0;
//		//int j_low_limit = 0;
//		//int j_high_limit = 250;
//		/*if(debug_run && j <= j_high_limit && j >= j_low_limit )
//		{
//			printf(" x = %3f y = %3f z = %3f\n",  x, y, z );
//			printf(" x_to_go = %3f y_to_go = %3f z_to_go = %3f\n",  x_to_go, y_to_go, z_to_go );
//			printf("voxel_x = %d voxel_y = %d voxel_z = %d voxel = %d\n", voxel_x, voxel_y, voxel_z, voxel);
//		}*/
//		//if( debug_run )
//			//fgets(user_response, sizeof(user_response), stdin);
//		/********************************************************************************************/
//		/*********************************** Voxel Walk Routine *************************************/
//		/********************************************************************************************/
//		if( z_move_direction != 0 )
//		{
//			//if(debug_run && j <= j_high_limit && j >= j_low_limit )
//				//printf("z_end != z_start\n");
//			while( !end_walk )
//			{
//				// Change in z for Move to Voxel Edge in x and y
//				x_extension = delta_zx * x_to_go;
//				y_extension = delta_zy * y_to_go;
//				//if(debug_run && j <= j_high_limit && j >= j_low_limit )
//				//{
//				//	printf(" x_extension = %3f y_extension = %3f\n", x_extension, y_extension );
//				//	//printf(" x_to_go = %3f y_to_go = %3f z_to_go = %3f\n",  x_to_go, y_to_go, z_to_go );
//				//	//printf("voxel_x = %d voxel_y = %d voxel_z = %d voxel = %d\n", voxel_x, voxel_y, voxel_z, voxel);
//				//}
//				if( (z_to_go <= x_extension  ) && (z_to_go <= y_extension) )
//				{
//					//printf("z_to_go <= x_extension && z_to_go <= y_extension\n");					
//					voxel_z -= z_move_direction;
//					
//					z = edge_coordinate( Z_ZERO_COORDINATE, voxel_z, VOXEL_THICKNESS, Z_INCREASING_DIRECTION, z_move_direction );					
//					x = corresponding_coordinate( dx_dz, z, z_start, x_start );
//					y = corresponding_coordinate( dy_dz, z, z_start, y_start );
//
//					/*if(debug_run && j <= j_high_limit && j >= j_low_limit )
//					{
//						printf(" x = %3f y = %3f z = %3f\n",  x, y, z );
//						printf(" x_to_go = %3f y_to_go = %3f z_to_go = %3f\n",  x_to_go, y_to_go, z_to_go );
//						printf("voxel_x = %d voxel_y = %d voxel_z = %d voxel = %d\n", voxel_x, voxel_y, voxel_z, voxel);
//					}*/
//
//					x_to_go = distance_remaining( X_ZERO_COORDINATE, x, X_INCREASING_DIRECTION, x_move_direction, VOXEL_WIDTH, voxel_x );
//					y_to_go = distance_remaining( Y_ZERO_COORDINATE, y, Y_INCREASING_DIRECTION, y_move_direction, VOXEL_HEIGHT, voxel_y );	
//					z_to_go = VOXEL_THICKNESS;
//				}
//				//If Next Voxel Edge is in x or xy Diagonal
//				else if( x_extension <= y_extension )
//				{
//					//printf(" x_extension <= y_extension \n");			
//					voxel_x += x_move_direction;
//
//					x = edge_coordinate( X_ZERO_COORDINATE, voxel_x, VOXEL_WIDTH, X_INCREASING_DIRECTION, x_move_direction );
//					y = corresponding_coordinate( dy_dx, x, x_start, y_start );
//					z = corresponding_coordinate( dz_dx, x, x_start, z_start );
//
//					x_to_go = VOXEL_WIDTH;
//					y_to_go = distance_remaining( Y_ZERO_COORDINATE, y, Y_INCREASING_DIRECTION, y_move_direction, VOXEL_HEIGHT, voxel_y );
//					z_to_go = distance_remaining( Z_ZERO_COORDINATE, z, Z_INCREASING_DIRECTION, z_move_direction, VOXEL_THICKNESS, voxel_z );
//				}
//				// Else Next Voxel Edge is in y
//				else
//				{
//					//printf(" y_extension < x_extension \n");
//					voxel_y -= y_move_direction;
//					
//					y = edge_coordinate( Y_ZERO_COORDINATE, voxel_y, VOXEL_HEIGHT, Y_INCREASING_DIRECTION, y_move_direction );
//					x = corresponding_coordinate( dx_dy, y, y_start, x_start );
//					z = corresponding_coordinate( dz_dy, y, y_start, z_start );
//
//					x_to_go = distance_remaining( X_ZERO_COORDINATE, x, X_INCREASING_DIRECTION, x_move_direction, VOXEL_WIDTH, voxel_x );
//					y_to_go = VOXEL_HEIGHT;					
//					z_to_go = distance_remaining( Z_ZERO_COORDINATE, z, Z_INCREASING_DIRECTION, z_move_direction, VOXEL_THICKNESS, voxel_z );
//				}
//				// <= VOXEL_ALLOWANCE
//				if( x_to_go == 0 )
//				{
//					x_to_go = VOXEL_WIDTH;
//					voxel_x += x_move_direction;
//				}
//				if( y_to_go == 0 )
//				{
//					y_to_go = VOXEL_HEIGHT;
//					voxel_y -= y_move_direction;
//				}
//				if( z_to_go == 0 )
//				{
//					z_to_go = VOXEL_THICKNESS;
//					voxel_z -= z_move_direction;
//				}
//				
//				voxel_z = max(voxel_z, 0 );
//				voxel = voxel_x + voxel_y * COLUMNS + voxel_z * COLUMNS * ROWS;
//				//if(debug_run && j <= j_high_limit && j >= j_low_limit )
//				//{
//				//	printf(" x = %3f y = %3f z = %3f\n",  x, y, z );
//				//	printf(" x_to_go = %3f y_to_go = %3f z_to_go = %3f\n",  x_to_go, y_to_go, z_to_go );
//				//	printf("voxel_x = %d voxel_y = %d voxel_z = %d voxel = %d\n", voxel_x, voxel_y, voxel_z, voxel);
//				//}
//				outside_image = (voxel_x >= COLUMNS ) || (voxel_y >= ROWS ) || (voxel_z >= SLICES ) || (voxel_x < 0  ) || (voxel_y < 0 ) || (voxel_z < 0 );		
//				if( !outside_image )
//				{
//					hit_hull = (image[voxel] == 1);	
//					//if( MLP_image_output )
//					//{
//					//	image[voxel] = 4;
//					//}
//				}
//				end_walk = outside_image || hit_hull;
//				//j++;
//				//if( debug_run )
//					//fgets(user_response, sizeof(user_response), stdin);		
//			}// end !end_walk 
//		}
//		else
//		{
//			//if(debug_run && j <= j_high_limit && j >= j_low_limit )
//				//printf("z_end == z_start\n");
//			while( !end_walk )
//			{
//				// Change in x for Move to Voxel Edge in y
//				y_extension = y_to_go / delta_yx;
//				//If Next Voxel Edge is in x or xy Diagonal
//				if( x_to_go <= y_extension )
//				{
//					//printf(" x_to_go <= y_extension \n");
//					voxel_x += x_move_direction;
//					
//					x = edge_coordinate( X_ZERO_COORDINATE, voxel_x, VOXEL_WIDTH, X_INCREASING_DIRECTION, x_move_direction );
//					y = corresponding_coordinate( dy_dx, x, x_start, y_start );
//
//					x_to_go = VOXEL_WIDTH;
//					y_to_go = distance_remaining( Y_ZERO_COORDINATE, y, Y_INCREASING_DIRECTION, y_move_direction, VOXEL_HEIGHT, voxel_y );
//				}
//				// Else Next Voxel Edge is in y
//				else
//				{
//					//printf(" y_extension < x_extension \n");				
//					voxel_y -= y_move_direction;
//
//					y = edge_coordinate( Y_ZERO_COORDINATE, voxel_y, VOXEL_HEIGHT, Y_INCREASING_DIRECTION, y_move_direction );
//					x = corresponding_coordinate( dx_dy, y, y_start, x_start );
//
//					x_to_go = distance_remaining( X_ZERO_COORDINATE, x, X_INCREASING_DIRECTION, x_move_direction, VOXEL_WIDTH, voxel_x );
//					y_to_go = VOXEL_HEIGHT;
//				}
//				// <= VOXEL_ALLOWANCE
//				if( x_to_go == 0 )
//				{
//					x_to_go = VOXEL_WIDTH;
//					voxel_x += x_move_direction;
//				}
//				if( y_to_go == 0 )
//				{
//					y_to_go = VOXEL_HEIGHT;
//					voxel_y -= y_move_direction;
//				}
//				voxel = voxel_x + voxel_y * COLUMNS + voxel_z * COLUMNS * ROWS;		
//				/*if(debug_run && j <= j_high_limit && j >= j_low_limit )
//				{
//					printf(" x = %3f y = %3f z = %3f\n",  x, y, z );
//					printf(" x_to_go = %3f y_to_go = %3f z_to_go = %3f\n",  x_to_go, y_to_go, z_to_go );
//					printf("voxel_x_in = %d voxel_y_in = %d voxel_z_in = %d\n", voxel_x, voxel_y, voxel_z);
//				}*/
//				outside_image = (voxel_x >= COLUMNS ) || (voxel_y >= ROWS ) || (voxel_z >= SLICES ) || (voxel_x < 0  ) || (voxel_y < 0 ) || (voxel_z < 0 );		
//				if( !outside_image )
//				{
//					hit_hull = (image[voxel] == 1);		
//					//if( MLP_image_output )
//					//{
//					//	image[voxel] = 4;
//					//}
//				}
//				end_walk = outside_image || hit_hull;
//				//j++;
//				//if( debug_run )
//					//fgets(user_response, sizeof(user_response), stdin);		
//			}// end: while( !end_walk )
//			//printf("i = %d", i );
//		}//end: else: z_start != z_end => z_start == z_end
//		if( hit_hull )
//		{
//			x_object = x;
//			y_object = y;
//			z_object = z;
//		}
//		return hit_hull;
//}
//__device__ int find_MLP_path_GPU
//( 
//	int*& path, double*& chord_lengths, 
//	double x_in_object, double y_in_object, double z_in_object, double x_out_object, double y_out_object, double z_out_object, 
//	double xy_entry_angle, double xz_entry_angle, double xy_exit_angle, double xz_exit_angle,
//	int voxel_x, int voxel_y, int voxel_z
//)
//{
//	double u_in_object, t_in_object, v_in_object, u_out_object, t_out_object, v_out_object;
//	double effective_chord_length; 
//	int num_intersections = 0;
//
//	bool debug_output = false, MLP_image_output = false, constant_chord_lengths = true;
//	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//	double u_0 = 0, u_1 = MLP_U_STEP,  u_2 = 0;
//	double T_0[2] = {0, 0}, T_2[2] = {0, 0}, V_0[2] = {0, 0}, V_2[2] = {0, 0};
//	double R_0[4] = { 1.0, 0.0, 0.0 , 1.0}; //a,b,c,d
//	//double R_0T[4] = { 1.0, 0.0, 0.0 , 1.0}; //a,c,b,d
//	double R_1[4] = { 1.0, 0.0, 0.0 , 1.0}; //a,b,c,d
//	double R_1T[4] = { 1.0, 0.0, 0.0 , 1.0};  //a,c,b,d
//
//	//double sigma_1_pre_1, sigma_1_pre_2, sigma_1_pre_3;
//	double sigma_2_pre_1, sigma_2_pre_2, sigma_2_pre_3;
//	double sigma_1_coefficient, sigma_t1, sigma_t1_theta1, sigma_theta1, determinant_Sigma_1, Sigma_1I[4];
//	double common_sigma_2_term_1, common_sigma_2_term_2, common_sigma_2_term_3;
//	double sigma_2_coefficient, sigma_t2, sigma_t2_theta2, sigma_theta2, determinant_Sigma_2, Sigma_2I[4]; 
//	double first_term_common_13_1, first_term_common_13_2, first_term_common_24_1, first_term_common_24_2, first_term[4], determinant_first_term;
//	double second_term_common_1, second_term_common_2, second_term_common_3, second_term_common_4, second_term[2];
//	double t_1, v_1, theta_1, phi_1, x_1, y_1, z_1;
//	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//	//effective_chord_length = mean_chord_length( u_in_object, t_in_object, v_in_object, u_out_object, t_out_object, v_out_object );
//	effective_chord_length = VOXEL_WIDTH;
//
//	int voxel = voxel_x + voxel_y * COLUMNS + voxel_z * COLUMNS * ROWS;
//	path[num_intersections] = voxel;
//	if(!constant_chord_lengths)
//		chord_lengths[num_intersections] = VOXEL_WIDTH;
//	num_intersections++;
//	//MLP_test_image_h[voxel] = 0;
//
//	u_in_object = ( cos( xy_entry_angle ) * x_in_object ) + ( sin( xy_entry_angle ) * y_in_object );
//	u_out_object = ( cos( xy_entry_angle ) * x_out_object ) + ( sin( xy_entry_angle ) * y_out_object );
//	t_in_object = ( cos( xy_entry_angle ) * y_in_object ) - ( sin( xy_entry_angle ) * x_in_object );
//	t_out_object = ( cos( xy_entry_angle ) * y_out_object ) - ( sin( xy_entry_angle ) * x_out_object );
//	v_in_object = z_in_object;
//	v_out_object = z_out_object;
//
//	if( u_in_object > u_out_object )
//	{
//		//if( debug_output )
//			//cout << "Switched directions" << endl;
//		xy_entry_angle += PI;
//		xy_exit_angle += PI;
//		u_in_object = ( cos( xy_entry_angle ) * x_in_object ) + ( sin( xy_entry_angle ) * y_in_object );
//		u_out_object = ( cos( xy_entry_angle ) * x_out_object ) + ( sin( xy_entry_angle ) * y_out_object );
//		t_in_object = ( cos( xy_entry_angle ) * y_in_object ) - ( sin( xy_entry_angle ) * x_in_object );
//		t_out_object = ( cos( xy_entry_angle ) * y_out_object ) - ( sin( xy_entry_angle ) * x_out_object );
//		v_in_object = z_in_object;
//		v_out_object = z_out_object;
//	}
//	T_0[0] = t_in_object;
//	T_2[0] = t_out_object;
//	T_2[1] = xy_exit_angle - xy_entry_angle;
//	V_0[0] = v_in_object;
//	V_2[0] = v_out_object;
//	V_2[1] = xz_exit_angle - xz_entry_angle;
//		
//	u_0 = 0;
//	u_1 = MLP_U_STEP;
//	u_2 = abs(u_out_object - u_in_object);		
//	//fgets(user_response, sizeof(user_response), stdin);
//
//	//output_file.open(filename);						
//				      
//	//precalculated u_0/u_2 terms
//	//u_0 terms
//	//double sigma_1_pre_1 =  A_0 * u_0 + A_1_OVER_2 * pow(u_0, 2.0) + A_2_OVER_3 * pow(u_0, 3.0) + A_3_OVER_4 * pow(u_0, 4.0) + A_4_OVER_5 * pow(u_0, 5.0) + A_5_OVER_6 * pow(u_0, 6.0);						//1, 1/2, 1/3, 1/4, 1/5, 1/6
//	//double sigma_1_pre_2 =  As_0_OVER_2 * pow(u_0, 2.0) + A_1_OVER_3 * pow(u_0, 3.0) + A_2_OVER_4 * pow(u_0, 4.0) + A_3_OVER_5 * pow(u_0, 5.0) + A_4_OVER_6 * pow(u_0, 6.0) + A_5_OVER_7 * pow(u_0, 7.0);	//1/2, 1/3, 1/4, 1/5, 1/6, 1/7
//	//double sigma_1_pre_3 =  A_0_OVER_3 * pow(u_0, 3.0) + A_1_OVER_4 * pow(u_0, 4.0) + A_2_OVER_5 * pow(u_0, 5.0) + A_3_OVER_6 * pow(u_0, 6.0) + A_4_OVER_7 * pow(u_0, 7.0) + A_5_OVER_8 * pow(u_0, 8.0);	//1/3, 1/4, 1/5, 1/6, 1/7, 1/8
//	//u_2 terms
//	sigma_2_pre_1 =  pow(u_2, 3.0) * ( A_0_OVER_3 + u_2 * ( A_1_OVER_12 + u_2 * ( A_2_OVER_30 + u_2 * (A_3_OVER_60 + u_2 * ( A_4_OVER_105 + u_2 * A_5_OVER_168 )))));;	//u_2^3 : 1/3, 1/12, 1/30, 1/60, 1/105, 1/168
//	sigma_2_pre_2 =  pow(u_2, 2.0) * ( A_0_OVER_2 + u_2 * (A_1_OVER_6 + u_2 * (A_2_OVER_12 + u_2 * ( A_3_OVER_20 + u_2 * (A_4_OVER_30 + u_2 * A_5_OVER_42)))));	//u_2^2 : 1/2, 1/6, 1/12, 1/20, 1/30, 1/42
//	sigma_2_pre_3 =  u_2 * ( A_0 +  u_2 * (A_1_OVER_2 +  u_2 * ( A_2_OVER_3 +  u_2 * ( A_3_OVER_4 +  u_2 * ( A_4_OVER_5 + u_2 * A_5_OVER_6 )))));			//u_2 : 1/1, 1/2, 1/3, 1/4, 1/5, 1/6
//
//	while( u_1 < u_2 - MLP_U_STEP)
//	//while( u_1 < u_2 - 0.001)
//	{
//		R_0[1] = u_1 - u_0;
//		//R_0T[2] = u_1 - u_0;
//		R_1[1] = u_2 - u_1;
//		R_1T[2] = u_2 - u_1;
//
//		sigma_1_coefficient = pow( E_0 * ( 1 + 0.038 * log( (u_1 - u_0)/X_0) ), 2.0 ) / X_0;
//		sigma_t1 =  sigma_1_coefficient * ( pow(u_1, 3.0) * ( A_0_OVER_3 + u_1 * (A_1_OVER_12 + u_1 * (A_2_OVER_30 + u_1 * (A_3_OVER_60 + u_1 * (A_4_OVER_105 + u_1 * A_5_OVER_168 ) )))) );	//u_1^3 : 1/3, 1/12, 1/30, 1/60, 1/105, 1/168
//		sigma_t1_theta1 =  sigma_1_coefficient * ( pow(u_1, 2.0) * ( A_0_OVER_2 + u_1 * (A_1_OVER_6 + u_1 * (A_2_OVER_12 + u_1 * (A_3_OVER_20 + u_1 * (A_4_OVER_30 + u_1 * A_5_OVER_42))))) );	//u_1^2 : 1/2, 1/6, 1/12, 1/20, 1/30, 1/42															
//		sigma_theta1 = sigma_1_coefficient * ( u_1 * ( A_0 + u_1 * (A_1_OVER_2 + u_1 * (A_2_OVER_3 + u_1 * (A_3_OVER_4 + u_1 * (A_4_OVER_5 + u_1 * A_5_OVER_6))))) );			//u_1 : 1/1, 1/2, 1/3, 1/4, 1/5, 1/6																	
//		determinant_Sigma_1 = sigma_t1 * sigma_theta1 - pow( sigma_t1_theta1, 2 );//ad-bc
//			
//		Sigma_1I[0] = sigma_theta1 / determinant_Sigma_1;
//		Sigma_1I[1] = -sigma_t1_theta1 / determinant_Sigma_1;
//		Sigma_1I[2] = -sigma_t1_theta1 / determinant_Sigma_1;
//		Sigma_1I[3] = sigma_t1 / determinant_Sigma_1;
//
//		//sigma 2 terms
//		sigma_2_coefficient = pow( E_0 * ( 1 + 0.038 * log( (u_2 - u_1)/X_0) ), 2.0 ) / X_0;
//		common_sigma_2_term_1 = u_1 * ( A_0 + u_1 * (A_1_OVER_2 + u_1 * (A_2_OVER_3 + u_1 * (A_3_OVER_4 + u_1 * (A_4_OVER_5 + u_1 * A_5_OVER_6 )))));
//		common_sigma_2_term_2 = pow(u_1, 2.0) * ( A_0_OVER_2 + u_1 * (A_1_OVER_3 + u_1 * (A_2_OVER_4 + u_1 * (A_3_OVER_5 + u_1 * (A_4_OVER_6 + u_1 * A_5_OVER_7 )))));
//		common_sigma_2_term_3 = pow(u_1, 3.0) * ( A_0_OVER_3 + u_1 * (A_1_OVER_4 + u_1 * (A_2_OVER_5 + u_1 * (A_3_OVER_6 + u_1 * (A_4_OVER_7 + u_1 * A_5_OVER_8 )))));
//		sigma_t2 =  sigma_2_coefficient * ( sigma_2_pre_1 - pow(u_2, 2.0) * common_sigma_2_term_1 + 2 * u_2 * common_sigma_2_term_2 - common_sigma_2_term_3 );
//		sigma_t2_theta2 =  sigma_2_coefficient * ( sigma_2_pre_2 - u_2 * common_sigma_2_term_1 + common_sigma_2_term_2 );
//		sigma_theta2 =  sigma_2_coefficient * ( sigma_2_pre_3 - common_sigma_2_term_1 );				
//		determinant_Sigma_2 = sigma_t2 * sigma_theta2 - pow( sigma_t2_theta2, 2 );//ad-bc
//
//		Sigma_2I[0] = sigma_theta2 / determinant_Sigma_2;
//		Sigma_2I[1] = -sigma_t2_theta2 / determinant_Sigma_2;
//		Sigma_2I[2] = -sigma_t2_theta2 / determinant_Sigma_2;
//		Sigma_2I[3] = sigma_t2 / determinant_Sigma_2;
//
//		// first_term_common_ij_k: i,j = rows common to, k = 1st/2nd of last 2 terms of 3 term summation in first_term calculation below
//		first_term_common_13_1 = Sigma_2I[0] * R_1[0] + Sigma_2I[1] * R_1[2];
//		first_term_common_13_2 = Sigma_2I[2] * R_1[0] + Sigma_2I[3] * R_1[2];
//		first_term_common_24_1 = Sigma_2I[0] * R_1[1] + Sigma_2I[1] * R_1[3];
//		first_term_common_24_2 = Sigma_2I[2] * R_1[1] + Sigma_2I[3] * R_1[3];
//
//		first_term[0] = Sigma_1I[0] + R_1T[0] * first_term_common_13_1 + R_1T[1] * first_term_common_13_2;
//		first_term[1] = Sigma_1I[1] + R_1T[0] * first_term_common_24_1 + R_1T[1] * first_term_common_24_2;
//		first_term[2] = Sigma_1I[2] + R_1T[2] * first_term_common_13_1 + R_1T[3] * first_term_common_13_2;
//		first_term[3] = Sigma_1I[3] + R_1T[2] * first_term_common_24_1 + R_1T[3] * first_term_common_24_2;
//
//
//		determinant_first_term = first_term[0] * first_term[3] - first_term[1] * first_term[2];
//		first_term[0] = first_term[3] / determinant_first_term;
//		first_term[1] = -first_term[1] / determinant_first_term;
//		first_term[2] = -first_term[2] / determinant_first_term;
//		first_term[3] = first_term[0] / determinant_first_term;
//
//		// second_term_common_i: i = # of term of 4 term summation it is common to in second_term calculation below
//		second_term_common_1 = R_0[0] * T_0[0] + R_0[1] * T_0[1];
//		second_term_common_2 = R_0[2] * T_0[0] + R_0[3] * T_0[1];
//		second_term_common_3 = Sigma_2I[0] * T_2[0] + Sigma_2I[1] * T_2[1];
//		second_term_common_4 = Sigma_2I[2] * T_2[0] + Sigma_2I[3] * T_2[1];
//
//		second_term[0] = Sigma_1I[0] * second_term_common_1 
//						+ Sigma_1I[1] * second_term_common_2 
//						+ R_1T[0] * second_term_common_3 
//						+ R_1T[1] * second_term_common_4;
//		second_term[1] = Sigma_1I[2] * second_term_common_1 
//						+ Sigma_1I[3] * second_term_common_2 
//						+ R_1T[2] * second_term_common_3 
//						+ R_1T[3] * second_term_common_4;
//
//		t_1 = first_term[0] * second_term[0] + first_term[1] * second_term[1];
//		//cout << "t_1 = " << t_1 << endl;
//		//double theta_1 = first_term[2] * second_term[0] + first_term[3] * second_term[1];
//
//		// Do v MLP Now
//		second_term_common_1 = R_0[0] * V_0[0] + R_0[1] * V_0[1];
//		second_term_common_2 = R_0[2] * V_0[0] + R_0[3] * V_0[1];
//		second_term_common_3 = Sigma_2I[0] * V_2[0] + Sigma_2I[1] * V_2[1];
//		second_term_common_4 = Sigma_2I[2] * V_2[0] + Sigma_2I[3] * V_2[1];
//
//		second_term[0]	= Sigma_1I[0] * second_term_common_1
//						+ Sigma_1I[1] * second_term_common_2
//						+ R_1T[0] * second_term_common_3
//						+ R_1T[1] * second_term_common_4;
//		second_term[1]	= Sigma_1I[2] * second_term_common_1
//						+ Sigma_1I[3] * second_term_common_2
//						+ R_1T[2] * second_term_common_3
//						+ R_1T[3] * second_term_common_4;
//		v_1 = first_term[0] * second_term[0] + first_term[1] * second_term[1];
//		//double phi_1 = first_term[2] * second_term[0] + first_term[3] * second_term[1];
//
//		// Rotate Coordinate From utv to xyz Coordinate System and Determine Which Voxel this Point on the MLP Path is in
//		x_1 = ( cos( xy_entry_angle ) * (u_in_object + u_1) ) - ( sin( xy_entry_angle ) * t_1 );
//		y_1 = ( sin( xy_entry_angle ) * (u_in_object + u_1) ) + ( cos( xy_entry_angle ) * t_1 );
//		z_1 = v_1;
//
//		voxel_x = calculate_voxel_GPU( X_ZERO_COORDINATE, x_1, VOXEL_WIDTH );
//		voxel_y = calculate_voxel_GPU( Y_ZERO_COORDINATE, y_1, VOXEL_HEIGHT );
//		voxel_z = calculate_voxel_GPU( Z_ZERO_COORDINATE, z_1, VOXEL_THICKNESS);
//				
//		voxel = voxel_x + voxel_y * COLUMNS + voxel_z * COLUMNS * ROWS;
//		//cout << "voxel_x = " << voxel_x << "voxel_y = " << voxel_y << "voxel_z = " << voxel_z << "voxel = " << voxel <<endl;
//		//fgets(user_response, sizeof(user_response), stdin);
//
//		if( voxel != path[num_intersections - 1] )
//		{
//			path[num_intersections] = voxel;
//			//MLP_test_image_h[voxel] = 0;
//			if(!constant_chord_lengths)
//				chord_lengths[num_intersections] = effective_chord_length;						
//			num_intersections++;
//		}
//		u_1 += MLP_U_STEP;
//	}
//	return num_intersections;
//}
//__device__ void MLP_GPU()
//{
//	//char user_response[20];
//	//char MLP_test_filename[128];
//	char data_filename[256];
//	//char updated_image_filename[128];
//	//char filename[256];
//	char iterate_filename[256];
//	double x_entry, y_entry, z_entry, x_exit, y_exit, z_exit;
//	double xy_entry_angle, xz_entry_angle, xy_exit_angle, xz_exit_angle;
//	double x_in_object, y_in_object, z_in_object, x_out_object, y_out_object, z_out_object;
//	double effective_chord_length; 	
//	int voxel_x, voxel_y, voxel_z, voxel_x_int, voxel_y_int, voxel_z_int;
//	int num_intersections;
//	bool entered_object = false, exited_object = false, debug_output = false, MLP_image_output = false, constant_chord_lengths = true;
//
//	int* path = (int*)calloc( MAX_INTERSECTIONS, sizeof(int));
//	double* chord_lengths = (double*)calloc( MAX_INTERSECTIONS, sizeof(double));
//	//if(!constant_chord_lengths)
//		//chord_lengths = (double*)calloc( MAX_INTERSECTIONS, sizeof(double));	
//	double* x_update_h = (double*)calloc( NUM_VOXELS, sizeof(double));
//
//	sprintf(data_filename, "%s%s/%s.txt", OUTPUT_DIRECTORY, OUTPUT_FOLDER, MLP_PATH_FILENAME );
//	FILE* pFile = create_MLP_path_file( data_filename );
//	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//	int block_history = 1;
//	//int start_history = 3*x_entry_vector.size()/4;
//	int start_history = 0;
//	//int start_history = 10;
//	//int end_history = start_history + 12;
//	int end_history = x_entry_vector.size();
//	int iterations = 20;
//	//int end_history = x_entry_vector.size();
//	//array_2_disk( "MLP_image_init", OUTPUT_DIRECTORY, OUTPUT_FOLDER, x_hull_h, COLUMNS, ROWS, SLICES, NUM_VOXELS, true );
//	//cout << "#histories = " << x_entry_vector.size() << " " << post_cut_histories << endl; 
//	//cout << "start i = " << start_history << endl;
//	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//	for( int iteration = 1; iteration <= iterations; iteration++ )
//	{
//		for( int i = start_history; i < end_history; i++ )
//		{		
//			x_entry = x_entry_vector[i], y_entry = y_entry_vector[i], z_entry = z_entry_vector[i];
//			x_exit = x_exit_vector[i], y_exit = y_exit_vector[i], z_exit = z_exit_vector[i];
//			xy_entry_angle = xy_entry_angle_vector[i], xz_entry_angle = xz_entry_angle_vector[i];
//			xy_exit_angle = xy_exit_angle_vector[i], xz_exit_angle = xz_exit_angle_vector[i];
//
//			/********************************************************************************************/
//			/**************************** Status Tracking Information ***********************************/
//			/********************************************************************************************/
//			entered_object = false;
//			exited_object = false;
//
//			entered_object = find_MLP_endpoints_GPU( x_hull_h, x_entry, y_entry, z_entry, xy_entry_angle, xz_entry_angle, x_in_object, y_in_object, z_in_object, voxel_x, voxel_y, voxel_z, true);	
//			exited_object = find_MLP_endpoints_GPU( x_hull_h, x_exit, y_exit, z_exit, xy_exit_angle, xz_exit_angle, x_out_object, y_out_object, z_out_object, voxel_x_int, voxel_y_int, voxel_z_int, false);
//
//			num_intersections = 0;
//
//			if( entered_object && exited_object )
//			{
//				//effective_chord_length = mean_chord_length( u_in_object, t_in_object, v_in_object, u_out_object, t_out_object, v_out_object );
//				effective_chord_length = VOXEL_WIDTH;
//
//				num_intersections = find_MLP_path_GPU( path, chord_lengths, x_in_object, y_in_object, z_in_object, x_out_object, y_out_object, z_out_object, xy_entry_angle, xz_entry_angle, xy_exit_angle, xz_exit_angle, voxel_x, voxel_y, voxel_z);
//				//if(constant_chord_lengths)
//				update_iterate22_GPU( WEPL_vector[i], effective_chord_length, x_h, path, num_intersections );
//				//update_iterate2( WEPL_vector[i], effective_chord_length, x_h, path, num_intersections );
//				//else
//					//update_iterate( WEPL_vector[i], chord_lengths, x_h, path, num_intersections );
//				//if( ( i + 1 ) % BLOCK_SIZE == 0 )
//				//path_data_2_disk(char* data_filename, FILE* pFile, int voxel_intersections, int* voxel_numbers, T*& data, bool write_sparse)
//				//path_data_2_disk(data_filename, pFile, num_intersections, path, path, true);
//				//calculate_update( WEPL_vector[i], effective_chord_length, x_update_h, path, num_intersections );
//				//if( block_history == BLOCK_SIZE )
//				//{					
//				//	sprintf(updated_image_filename, "%s_%d_%d", "update_image", i, block_history );
//				//	array_2_disk(updated_image_filename, OUTPUT_DIRECTORY, OUTPUT_FOLDER, x_h, COLUMNS, ROWS, SLICES, NUM_VOXELS, true );
//				//	update_iterate3( x_h, x_update_h );
//				//	block_history = 0;
//				//}
//				//block_history++;
//			}	
//		}
//		//sprintf(iterate_filename, "%s%d", "x_", iteration );		
//		//if( WRITE_X_KI )
//			//array_2_disk(iterate_filename, OUTPUT_DIRECTORY, OUTPUT_FOLDER, x_h, COLUMNS, ROWS, SLICES, NUM_VOXELS, true );
//	}
//}
/***********************************************************************************************************************************************************************************************************************/
/********************************************************************************************* Image Reconstruction (host) *********************************************************************************************/
/***********************************************************************************************************************************************************************************************************************/
void define_initial_iterate()
{
	x_h = (float*) calloc( NUM_VOXELS, sizeof(float) );

	switch( X_K0 )
	{
		case X_HULL		: std::copy( x_hull_h, x_hull_h + NUM_VOXELS, x_h );													break;
		case FBP_IMAGE	: x_h = FBP_image_h;																					break;
		case HYBRID		: std::transform(FBP_image_h, FBP_image_h + NUM_VOXELS, x_hull_h, x_h, std::multiplies<float>() );		break;
		case ZEROS		: break;
	}
	//cout << " x_h[708818] = " << x_h[708818] << endl;
	//cout << " x_h[708818] = " << x_h[710377 ] << endl;
	if( WRITE_X_K0 )
		array_2_disk("x_k0", OUTPUT_DIRECTORY, OUTPUT_FOLDER, x_h, COLUMNS, ROWS, SLICES, NUM_VOXELS, true );
	//cout << " x_h[708818] = " << x_h[708818] << endl;
	//cout << " x_h[708818] = " << x_h[710377 ] << endl;
}
void create_hull_image_hybrid()
{
	/*int* SM_differences_h = (int*) calloc( NUM_VOXELS, sizeof(int) );
	int* SM_differences_d;
	cudaMalloc((void**) &SM_differences_d, SIZE_IMAGE_INT );
	cudaMemcpy( SM_differences_d, SM_differences_h, SIZE_IMAGE_INT, cudaMemcpyHostToDevice );*/

	

	dim3 dimBlock( SLICES );
	dim3 dimGrid( COLUMNS, ROWS );   

	create_hull_image_hybrid_GPU<<< dimGrid, dimBlock >>>( x_hull_d, FBP_image_d );
	cudaMemcpy( x_h, FBP_image_d, SIZE_IMAGE_FLOAT, cudaMemcpyDeviceToHost );

	if( WRITE_X_K0 )
		array_2_disk("x_k0", OUTPUT_DIRECTORY, OUTPUT_FOLDER, x_h, COLUMNS, ROWS, SLICES, NUM_VOXELS, true );
}
__global__ void create_hull_image_hybrid_GPU( bool*& x_hull, float*& FBP_image)
{
	int row = blockIdx.y, column = blockIdx.x, slice = threadIdx.x;
	int voxel = column + row * COLUMNS + slice * COLUMNS * ROWS;
	FBP_image[voxel] *= x_hull[voxel];
}
template< typename T, typename LHS, typename RHS> T discrete_dot_product( LHS*& left, RHS*& right, int* elements, unsigned int num_elements )
{
	T sum = 0;
	for( unsigned int i = 0; i < num_elements; i++)
	{
		//cout << "iteration " << i << " " << "num_elements = " << num_elements << " " << elements[i] << " " << left[i] << " " << right[elements[i]] << endl;
		sum += ( left[i] * right[elements[i]] );
	}
	return sum;
}
template< typename A, typename X> double update_vector_multiplier( double bi, A*& a_i, X*& x_k, int* voxels_intersected, unsigned int num_intersections )
{
	// [ ( bi - <ai, x(k)> ) / <ai, ai> ] ai = [ ( bi - <ai, x(k)> ) / ||ai||^2 ] ai 
	/*
	double inner_product_ai_xk = 0.0;
	double norm_ai_squared = 0.0;
	for( unsigned int i = 0; i < num_intersections; i++)
	{
		inner_product_ai_xk	+= a_i[i] * x_k[voxels_intersected[i]];
		norm_ai_squared		+= a_i[i] * a_i[i];
	}
	*/
	double inner_product_ai_xk = discrete_dot_product<double>( a_i, x_k, voxels_intersected, num_intersections );
	//cout << "inner_product_ai_xk = " << inner_product_ai_xk << endl;
	//int* voxel_numbers = sequential_numbers<int>( 0, num_intersections);
	//double norm_ai_squared = discrete_dot_product<double>( a_i, a_i, voxel_numbers, num_intersections );
	double norm_ai_squared = std::inner_product(a_i, a_i + num_intersections, a_i, 0.0 );
	//double ai_sqd = 0;
	//for( unsigned int i = 0; i < num_intersections; i++)
	//{
	//	ai_sqd += a_i[i] * a_i[i];
	//}
	//cout << "ai_sqd = " << ai_sqd << endl;
	//cout << "bi = " << bi << endl;
	//double norm_ai_squared = std::accumulate ( a_i, a_i + , 0.0 );
	//cout << "norm ai squared = " << norm_ai_squared << endl;
	//double norm_ai_squared = discrete_dot_product<double>( a_i, a_i, num_intersections );
	return ( bi - inner_product_ai_xk ) /  norm_ai_squared;
}
template< typename A, typename X> void update_iterate( double bi, A*& a_i, X*& x_k, int* voxels_intersected, unsigned int num_intersections )
{
	// x(K+1) = x(k) + [ ( bi - <ai, x(k)> ) / <ai, ai> ] ai =  x(k) + [ ( bi - <ai, x(k)> ) / ||ai||^2 ] ai 
	//double inner_product_ai_xk = discrete_dot_product<double>( a_i, x_k, voxels_intersected, num_intersections );
	////double norm_ai_squared = std::accumulate ( a_i, a_i, 0.0 );
	//int* voxel_numbers = sequential_numbers<int>( 0, num_intersections);
	//double norm_ai_squared = discrete_dot_product<double>( a_i, a_i, voxel_numbers, num_intersections );
	//double ai_multiplier = ( bi - inner_product_ai_xk ) /  norm_ai_squared;
	//cout << num_intersections << endl;
	double ai_multiplier = update_vector_multiplier( bi, a_i, x_k, voxels_intersected, num_intersections );
	//cout << "ai_multiplier = " << ai_multiplier << endl;
	for( int intersection = 0; intersection < num_intersections; intersection++)
		x_k[voxels_intersected[intersection]] += (LAMBDA * sqrt(bi) )* ai_multiplier * a_i[intersection];
}
/***********************************************************************************************************************************************************************************************************************/
template< typename T, typename RHS> T scalar_dot_product( double scalar, RHS*& right, int* elements, unsigned int num_elements )
{
	T sum = 0;
	for( unsigned int i = 0; i < num_elements; i++)
	{
		//cout << "iteration " << i << " " << "num_elements = " << num_elements << " " << elements[i] <<" " << right[elements[i]] << endl;
		sum += ( scalar * right[elements[i]] );
	}
	return sum;
}
template< typename X> double update_vector_multiplier2( double bi, double mean_chord_length, X*& x_k, int* voxels_intersected, unsigned int num_intersections )
{
	// [ ( bi - <ai, x(k)> ) / <ai, ai> ] ai = [ ( bi - <ai, x(k)> ) / ||ai||^2 ] ai 
	double inner_product_ai_xk = scalar_dot_product<double>( mean_chord_length, x_k, voxels_intersected, num_intersections );
	//double inner_product_ai_xk = scalar_dot_product<double>( mean_chord_length, x_k, voxels_intersected, num_intersections );
	//cout << "inner_product_ai_xk = " << inner_product_ai_xk << endl;
	//int* voxel_numbers = sequential_numbers<int>( 0, num_intersections);
	//double norm_ai_squared = std::inner_product(a_i, a_i + num_intersections, a_i, 0.0 );
	double norm_ai_squared = pow(mean_chord_length, 2.0 ) * num_intersections;
	return ( bi - inner_product_ai_xk ) /  norm_ai_squared;
}
template<typename X> void update_iterate2( double bi, double mean_chord_length, X*& x_k, int* voxels_intersected, unsigned int num_intersections )
{
	// x(K+1) = x(k) + [ ( bi - <ai, x(k)> ) / <ai, ai> ] ai =  x(k) + [ ( bi - <ai, x(k)> ) / ||ai||^2 ] ai 
	double ai_multiplier = update_vector_multiplier2( bi, mean_chord_length, x_k, voxels_intersected, num_intersections );
	//cout << "ai_multiplier = " << ai_multiplier << endl;
	//int middle_intersection = num_intersections / 2;
	int voxel;
	double radius_squared;
	double scale_factor = LAMBDA * ai_multiplier * mean_chord_length;
	//double scaled_lambda;
	for( unsigned int intersection = 0; intersection < num_intersections; intersection++)
	{
		voxel = voxels_intersected[intersection];
		radius_squared = voxel_2_radius_squared( voxel );
		//	1 - a*r(i)^2 DECAY_FACTOR
		//exp(-a*r)  EXPONENTIAL_DECAY
		//exp(-a*r^2)  EXPONENTIAL_SQD_DECAY
		//scaled_lambda = LAMBDA * ( 1 - DECAY_FACTOR * radius_squared );
		// LAMBDA * ( 1 - DECAY_FACTOR * radius_squared );
		// LAMBDA * exp( -EXPONENTIAL_DECAY * sqrt( radius_squared ) );
		// LAMBDA * exp( -EXPONENTIAL_SQD_DECAY * radius_squared );
		//x_k[voxel] +=  scale_factor * ( 1 - DECAY_FACTOR * radius_squared );
		x_k[voxel] +=  scale_factor * exp( -EXPONENTIAL_SQD_DECAY * radius_squared );
		//x_k[voxels_intersected[intersection]] += (LAMBDA / sqrt( abs(middle_intersection - intersection) + 1.0) ) * ai_multiplier * mean_chord_length;
		//x_k[voxels_intersected[intersection]] += (LAMBDA * max(1.0, sqrt(bi) ) ) * ai_multiplier * mean_chord_length;
	}
}
/***********************************************************************************************************************************************************************************************************************/
double scalar_dot_product2( double scalar, float*& right, int* elements, unsigned int num_elements )
{
	double sum = 0;
	for( unsigned int i = 0; i < num_elements; i++)
	{
		//cout << "iteration " << i << " " << "num_elements = " << num_elements << " " << elements[i] <<" " << right[elements[i]] << endl;
		sum += ( scalar * right[elements[i]] );
	}
	return sum;
}
double update_vector_multiplier22( double bi, double mean_chord_length, float*& x_k, int* voxels_intersected, unsigned int num_intersections )
{
	// [ ( bi - <ai, x(k)> ) / <ai, ai> ] ai = [ ( bi - <ai, x(k)> ) / ||ai||^2 ] ai 
	double inner_product_ai_xk = scalar_dot_product2( mean_chord_length, x_k, voxels_intersected, num_intersections );
	//double inner_product_ai_xk = scalar_dot_product<double>( mean_chord_length, x_k, voxels_intersected, num_intersections );
	//cout << "inner_product_ai_xk = " << inner_product_ai_xk << endl;
	//int* voxel_numbers = sequential_numbers<int>( 0, num_intersections);
	//double norm_ai_squared = std::inner_product(a_i, a_i + num_intersections, a_i, 0.0 );
	double norm_ai_squared = pow(mean_chord_length, 2.0 ) * num_intersections;
	return ( bi - inner_product_ai_xk ) /  norm_ai_squared;
}
void update_iterate22( double bi, double mean_chord_length, float*& x_k, int* voxels_intersected, unsigned int num_intersections )
{
	// x(K+1) = x(k) + [ ( bi - <ai, x(k)> ) / <ai, ai> ] ai =  x(k) + [ ( bi - <ai, x(k)> ) / ||ai||^2 ] ai 
	double ai_multiplier = update_vector_multiplier22( bi, mean_chord_length, x_k, voxels_intersected, num_intersections );
	//cout << "ai_multiplier = " << ai_multiplier << endl;
	//int middle_intersection = num_intersections / 2;
	int voxel;
	double radius_squared, update;
	double scale_factor = LAMBDA * ai_multiplier * mean_chord_length;
	//double scaled_lambda;
	for( unsigned int intersection = 0; intersection < num_intersections; intersection++)
	{
		voxel = voxels_intersected[intersection];
		radius_squared = voxel_2_radius_squared( voxel );
		//	1 - a*r(i)^2 DECAY_FACTOR
		//exp(-a*r)  EXPONENTIAL_DECAY
		//exp(-a*r^2)  EXPONENTIAL_SQD_DECAY
		//scaled_lambda = LAMBDA * ( 1 - DECAY_FACTOR * radius_squared );
		// LAMBDA * ( 1 - DECAY_FACTOR * radius_squared );
		// LAMBDA * exp( -EXPONENTIAL_DECAY * sqrt( radius_squared ) );
		// LAMBDA * exp( -EXPONENTIAL_SQD_DECAY * radius_squared );
		//x_k[voxel] +=  scale_factor * ( 1 - DECAY_FACTOR * radius_squared );
		//if( radius_squared > AFFECT_RADIUS_SQD )
			//update = scale_factor * exp( -EXPONENTIAL_SQD_DECAY * radius_squared );
		//else
			//update = scale_factor;
		x_k[voxel] += scale_factor;
		//x_k[voxels_intersected[intersection]] += (LAMBDA / sqrt( abs(middle_intersection - intersection) + 1.0) ) * ai_multiplier * mean_chord_length;
		//x_k[voxels_intersected[intersection]] += (LAMBDA * max(1.0, sqrt(bi) ) ) * ai_multiplier * mean_chord_length;
	}
}
/***********************************************************************************************************************************************************************************************************************/
template<typename X, typename U> void calculate_update( double bi, double mean_chord_length, X*& x_k, U*& image_update,int* voxels_intersected, unsigned int num_intersections )
{
	// x(K+1) = x(k) + [ ( bi - <ai, x(k)> ) / <ai, ai> ] ai =  x(k) + [ ( bi - <ai, x(k)> ) / ||ai||^2 ] ai 
	double ai_multiplier = update_vector_multiplier2( bi, mean_chord_length, x_k, voxels_intersected, num_intersections );
	cout << "ai_multiplier = " << ai_multiplier << endl;
	for( unsigned int intersection = 0; intersection < num_intersections; intersection++)
		image_update[voxels_intersected[intersection]] += LAMBDA * ai_multiplier * mean_chord_length;
		//image_update[voxels_intersected[intersection]] += (LAMBDA * max(1.0, sqrt(bi) ) ) * ai_multiplier * mean_chord_length;
}
template<typename X, typename U> void update_iterate3( X*& x_k, U*& image_update )
{
	for( int voxel = 0; voxel < NUM_VOXELS; voxel++)
	{
		x_k[voxel] += image_update[voxel] / BLOCK_SIZE;
		image_update[voxel] = 0.0;
	}
}
/***********************************************************************************************************************************************************************************************************************/
/********************************************************************************************* Image Reconstruction (GPU) **********************************************************************************************/
/***********************************************************************************************************************************************************************************************************************/
//__device__ double scalar_dot_product_GPU_2( double scalar, float*& right, int* elements, int num_elements )
//{
//	double sum = 0;
//	for( unsigned int i = 0; i < num_elements; i++)
//	{
//		//cout << "iteration " << i << " " << "num_elements = " << num_elements << " " << elements[i] <<" " << right[elements[i]] << endl;
//		sum += ( scalar * right[elements[i]] );
//	}
//	return sum;
//}
//__device__ double update_vector_multiplier_GPU_22( double bi, double mean_chord_length, float*& x_k, int* voxels_intersected, int num_intersections )
//{
//	// [ ( bi - <ai, x(k)> ) / <ai, ai> ] ai = [ ( bi - <ai, x(k)> ) / ||ai||^2 ] ai 
//	double inner_product_ai_xk = scalar_dot_product_GPU_2( mean_chord_length, x_k, voxels_intersected, num_intersections );
//	//double inner_product_ai_xk = scalar_dot_product<double>( mean_chord_length, x_k, voxels_intersected, num_intersections );
//	//cout << "inner_product_ai_xk = " << inner_product_ai_xk << endl;
//	//int* voxel_numbers = sequential_numbers<int>( 0, num_intersections);
//	//double norm_ai_squared = std::inner_product(a_i, a_i + num_intersections, a_i, 0.0 );
//	double norm_ai_squared = pow(mean_chord_length, 2.0 ) * num_intersections;
//	return ( bi - inner_product_ai_xk ) /  norm_ai_squared;
//}
//__device__ void update_iterate22_GPU( double bi, double mean_chord_length, float*& x_k, int* voxels_intersected, int num_intersections )
//{
//	// x(K+1) = x(k) + [ ( bi - <ai, x(k)> ) / <ai, ai> ] ai =  x(k) + [ ( bi - <ai, x(k)> ) / ||ai||^2 ] ai 
//	double ai_multiplier = update_vector_multiplier_GPU_22( bi, mean_chord_length, x_k, voxels_intersected, num_intersections );
//	//cout << "ai_multiplier = " << ai_multiplier << endl;
//	//int middle_intersection = num_intersections / 2;
//	int voxel;
//	double radius_squared;
//	double scale_factor = LAMBDA * ai_multiplier * mean_chord_length;
//	//double scaled_lambda;
//	for( int intersection = 0; intersection < num_intersections; intersection++)
//	{
//		voxel = voxels_intersected[intersection];
//		radius_squared = voxel_2_radius_squared_GPU( voxel );
//		//	1 - a*r(i)^2 DECAY_FACTOR
//		//exp(-a*r)  EXPONENTIAL_DECAY
//		//exp(-a*r^2)  EXPONENTIAL_SQD_DECAY
//		//scaled_lambda = LAMBDA * ( 1 - DECAY_FACTOR * radius_squared );
//		// LAMBDA * ( 1 - DECAY_FACTOR * radius_squared );
//		// LAMBDA * exp( -EXPONENTIAL_DECAY * sqrt( radius_squared ) );
//		// LAMBDA * exp( -EXPONENTIAL_SQD_DECAY * radius_squared );
//		//x_k[voxel] +=  LAMBDA * ( 1 - DECAY_FACTOR * radius_squared ) * ai_multiplier * mean_chord_length;
//		x_k[voxel] +=  scale_factor * exp( -EXPONENTIAL_SQD_DECAY * radius_squared );
//		//x_k[voxels_intersected[intersection]] += (LAMBDA / sqrt( abs(middle_intersection - intersection) + 1.0) ) * ai_multiplier * mean_chord_length;
//		//x_k[voxels_intersected[intersection]] += (LAMBDA * max(1.0, sqrt(bi) ) ) * ai_multiplier * mean_chord_length;
//	}
//}
/***********************************************************************************************************************************************************************************************************************/
/********************************************************************************** Routines for Writing Data Arrays/Vectors to Disk ***********************************************************************************/
/***********************************************************************************************************************************************************************************************************************/
void binary_2_ASCII()
{
	count_histories();
	char filename[256];
	FILE* output_file;
	int start_file_num = 0, end_file_num = 0, histories_to_process = 0;
	while( start_file_num != NUM_FILES )
	{
		while( end_file_num < NUM_FILES )
		{
			if( histories_to_process + histories_per_file[end_file_num] < MAX_GPU_HISTORIES )
				histories_to_process += histories_per_file[end_file_num];
			else
				break;
			end_file_num++;
		}
		read_data_chunk( histories_to_process, start_file_num, end_file_num );
		sprintf( filename, "%s%s/%s%s%d%s", OUTPUT_DIRECTORY, OUTPUT_FOLDER, INPUT_BASE_NAME, "_", gantry_angle_h[0], ".txt" );
		output_file = fopen (filename, "w");

		for( unsigned int i = 0; i < histories_to_process; i++ )
		{
			fprintf(output_file, "%3f %3f %3f %3f %3f %3f %3f %3f %3f\n", t_in_1_h[i], t_in_2_h[i], t_out_1_h[i], t_out_2_h[i], v_in_1_h[i], v_in_2_h[i], v_out_1_h[i], v_out_2_h[i], WEPL_h[i]);
		}
		fclose (output_file);
		initial_processing_memory_clean();
		start_file_num = end_file_num;
		histories_to_process = 0;
	} 
}
template<typename T> void array_2_disk( char* filename_base, const char* directory, const char* folder, T* data, const int x_max, const int y_max, const int z_max, const int elements, const bool single_file )
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
	}
}
template<typename T> void vector_2_disk( char* filename_base, const char* directory, const char* folder, std::vector<T> data, const int x_max, const int y_max, const int z_max, const bool single_file )
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
	char* data_format = FLOAT_FORMAT;
	if( typeid(T) == typeid(int) )
		data_format = INT_FORMAT;
	if( typeid(T) == typeid(bool))
		data_format = BOOL_FORMAT;
	std::vector<T> bin_histories;
	int num_bin_members;
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
		bin_histories.shrink_to_fit();
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
		std::iota( angular_bins.begin(), angular_bins.end(), 0 );
		std::iota( v_bins.begin(), v_bins.end(), 0 );
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
	
	int num_angles = (int) angular_bins.size();
	int num_v_bins = (int) v_bins.size();
	/*for( unsigned int i = 0; i < 3; i++ )
		printf("%d\n", angles[i] );
	for( unsigned int i = 0; i < 3; i++ )
		printf("%d\n", angular_bins[i] );
	for( unsigned int i = 0; i < 3; i++ )
		printf("%d\n", v_bins[i] );*/
	char filename[256];
	int start_bin, angle;
	FILE* output_file;

	for( int angular_bin = 0; angular_bin < num_angles; angular_bin++)
	{
		angle = angular_bins[angular_bin] * GANTRY_ANGLE_INTERVAL;
		//printf("angle = %d\n", angular_bins[angular_bin]);
		sprintf( filename, "%s%s/%s_%03d%s", OUTPUT_DIRECTORY, OUTPUT_FOLDER, filename_base, angle, ".txt" );
		output_file = fopen (filename, "w");
		for( int v_bin = 0; v_bin < num_v_bins; v_bin++)
		{			
			//printf("v bin = %d\n", v_bins[v_bin]);
			start_bin = angular_bins[angular_bin] * T_BINS + v_bins[v_bin] * ANGULAR_BINS * T_BINS;
			t_bins_2_disk( output_file, bin_numbers, data, type, bin_order, start_bin );
			if( v_bin != num_v_bins - 1 )
				fputs("\n", output_file);
		}	
		fclose (output_file);
	}
}
template<typename T> void t_bins_2_disk( FILE* output_file, int*& bin_numbers, T*& data, const int data_elements, const BIN_ANALYSIS_TYPE type, const BIN_ORGANIZATION bin_order, int bin )
{
	char* data_format = FLOAT_FORMAT;
	if( typeid(T) == typeid(int) )
		data_format = INT_FORMAT;
	if( typeid(T) == typeid(bool))
		data_format = BOOL_FORMAT;

	std::vector<T> bin_histories;
	//int data_elements = sizeof(data)/sizeof(float);
	int num_bin_members;
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
		num_bin_members = (int) bin_histories.size();
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
		bin_histories.shrink_to_fit();
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
		std::iota( angular_bins.begin(), angular_bins.end(), 0 );
		std::iota( v_bins.begin(), v_bins.end(), 0 );
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
	/*for( unsigned int i = 0; i < 3; i++ )
		printf("%d\n", angles[i] );
	for( unsigned int i = 0; i < 3; i++ )
		printf("%d\n", angular_bins[i] );
	for( unsigned int i = 0; i < 3; i++ )
		printf("%d\n", v_bins[i] );*/
	char filename[256];
	int start_bin, angle;
	FILE* output_file;

	for( int angular_bin = 0; angular_bin < num_angles; angular_bin++)
	{
		angle = angular_bins[angular_bin] * (int) GANTRY_ANGLE_INTERVAL;
		//printf("angle = %d\n", angular_bins[angular_bin]);
		sprintf( filename, "%s%s/%s_%03d%s", OUTPUT_DIRECTORY, OUTPUT_FOLDER, filename_base, angle, ".txt" );
		output_file = fopen (filename, "w");
		for( int v_bin = 0; v_bin < num_v_bins; v_bin++)
		{			
			//printf("v bin = %d\n", v_bins[v_bin]);
			start_bin = angular_bins[angular_bin] * T_BINS + v_bins[v_bin] * ANGULAR_BINS * T_BINS;
			t_bins_2_disk( output_file, bin_numbers, data, data_elements, type, bin_order, start_bin );
			if( v_bin != num_v_bins - 1 )
				fputs("\n", output_file);
		}	
		fclose (output_file);
	}
}
FILE* create_MLP_path_file( char* data_filename )
{
	FILE * pFile;
	//char data_filename[256];
	//sprintf(data_filename, "%s%s/%s.txt", OUTPUT_DIRECTORY, OUTPUT_FOLDER, MLP_PATH_FILENAME );
	pFile = fopen (data_filename,"w+");
	return pFile;
}
template<typename T> void path_data_2_disk(char* data_filename, FILE* pFile, int voxel_intersections, int* voxel_numbers, T*& data, bool write_sparse)
{
	// Writes either voxel intersection numbers or chord lengths in either dense or sparse format
	T data_value;	
	char* data_format = FLOAT_FORMAT;
	if( typeid(T) == typeid(int) )
		data_format = INT_FORMAT;
	if( typeid(T) == typeid(bool))
		data_format = BOOL_FORMAT;
	freopen (data_filename,"a+", pFile);
	//pFile = freopen (data_filename,"a+", pFile);
	if( write_sparse )
	{
		for( int intersection_num = 0; intersection_num < voxel_intersections; intersection_num++ )
		{
			fprintf (pFile, data_format, data[intersection_num]);	
			fputs(" ", pFile);
		}
	}
	else
	{
		bool intersected = false;
		
		for( int voxel = 0; voxel < NUM_VOXELS; voxel++)
		{
			for( unsigned int i = 0; i < voxel_intersections; i++ )
			{
				if( voxel_numbers[i] == voxel )
				{
					data_value = data[i];
					intersected = true;
				}
			}
			if( typeid(T) == typeid(int) || typeid(T) == typeid(bool) )
				fprintf (pFile, data_format, intersected);
			else
				fprintf (pFile, data_format, data_value);
			if( voxel != NUM_VOXELS - 1 )
				fputc(' ', pFile);
			intersected = false;
			data_value = 0;
		}
	}
	fputc ('\n',pFile);
	fclose (pFile);
}
/***********************************************************************************************************************************************************************************************************************/
/********************************************************************************* Image Position/Voxel Calculation Functions (Host) ***********************************************************************************/
/***********************************************************************************************************************************************************************************************************************/
int calculate_voxel( double zero_coordinate, double current_position, double voxel_size )
{
	return abs( current_position - zero_coordinate ) / voxel_size;
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
	std::iota( sequential_array, sequential_array + length, start_number );
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
/***********************************************************************************************************************************************************************************************************************/
/*********************************************************************************************** Device Helper Functions ***********************************************************************************************/
/***********************************************************************************************************************************************************************************************************************/


/***********************************************************************************************************************************************************************************************************************/
/************************************************************************************ Testing Functions and Functions in Development ***********************************************************************************/
/***********************************************************************************************************************************************************************************************************************/
double radial_lambda( double radius_squared )
{
	//	1 - a*r(i)^2 DECAY_FACTOR
	//exp(-a*r)  EXPONENTIAL_DECAY
	//exp(-a*r^2)  EXPONENTIAL_SQD_DECAY

	return LAMBDA * ( 1 - DECAY_FACTOR * radius_squared );
	return LAMBDA * exp( -EXPONENTIAL_DECAY * sqrt( radius_squared ) );
	return LAMBDA * exp( -EXPONENTIAL_SQD_DECAY * radius_squared );

}
double my_divide( int x, int y ) { return x*y; }
double my_divide2( double x, double y ) { return x*y; }
template<typename T> double func_pass_test( T x, T y, std::function<double(T, T)> func )
{
	return func(x,y);
}
void test_va_arg( const std::vector<int>& data, const BIN_ORGANIZATION bin_order, ... )
{
	//bins_2_disk( "WEPL_dist_pre_test2", empty_parameter, mean_WEPL_h, NUM_BINS, MEANS, ALL_BINS, BY_BIN );
	//bins_2_disk( "WEPL_dist_pre_test2", empty_parameter, sinogram_h, NUM_BINS, MEANS, ALL_BINS, BY_BIN );
	std::vector<int> angles;
	std::vector<int> angular_bins;
	std::vector<int> v_bins;

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
	
	int num_angles2 = (int) angular_bins.size();
	int num_v_bins2 = (int) v_bins.size();

	for( unsigned int i = 0; i < num_angles2; i++ )
		cout << angular_bins[i] << endl;
	for( unsigned int i = 0; i < num_v_bins2; i++ )
		cout << v_bins[i] << endl;
}
void test_func()
{


	int voxel_x = 3;
	int voxel_y = 4;
	int voxel_z = 2;


	//int xtu[voxel_x];
	
	int voxel = voxel_x + voxel_y * COLUMNS + voxel_z * ROWS * COLUMNS;

	voxel_x = 0;
	voxel_y = 0;
	voxel_z = 0;

	voxel_2_3D_voxels( voxel, voxel_x, voxel_y, voxel_z );
	cout << "voxel_x = " << voxel_x << endl;
	cout << "voxel_y = " << voxel_y << endl;
	cout << "voxel_z = " << voxel_z << endl;

	double x = voxel_2_position( voxel_x, VOXEL_WIDTH, COLUMNS, 1 );
	double y = voxel_2_position( voxel_y, VOXEL_HEIGHT, ROWS, -1 );
	double z = voxel_2_position( voxel_z, VOXEL_THICKNESS, SLICES, -1 );

	printf("x = %3f\n", x );
	printf("y = %3f\n", y );
	printf("z = %3f\n", z );

	x = y = z = 0;

	printf("x = %3f\n", x );
	printf("y = %3f\n", y );
	printf("z = %3f\n", z );

	voxel_2_positions( voxel, x, y, z );

	printf("x = %3f\n", x );
	printf("y = %3f\n", y );
	printf("z = %3f\n", z );

	bool t = false;
	bool t2 = false;

	double radius_squared = pow( x, 2.0 ) + pow( y, 2.0 );
	printf("radius_squared = %3f\n",radius_squared );
	//timer( true);
	radius_squared = 0;
	double ai_multiplier = 0.01;
	double mean_chord_length = 0.08;
	unsigned int iterations = 100000000;
	cout << iterations << endl;
	double zz = ai_multiplier * mean_chord_length * LAMBDA;
	double factor = exp( -EXPONENTIAL_SQD_DECAY ) ;
	float* xx = (float*)calloc( NUM_VOXELS, sizeof(float));
	int* voxels_hit = (int*)calloc( 200, sizeof(int));
	voxels_hit[100] = 600000;
	for( unsigned int i = 0; i < iterations; i++ )
	{
		//voxel = voxels_hit[100];
		//voxel = rand() % NUM_VOXELS;
		//radius_squared = voxel_2_radius_squared( voxel );
		//xx[voxel] += pow(EXPONENTIAL_TERM, radius_squared) * zz;
		//xx[voxel] += LAMBDA * exp( -EXPONENTIAL_SQD_DECAY * radius_squared ) * ai_multiplier * mean_chord_length;
		//xx[voxel] += exp( -EXPONENTIAL_SQD_DECAY * sqrt(radius_squared) ) * zz;
		//xx[voxel] += exp( -EXPONENTIAL_SQD_DECAY * radius_squared ) * zz;
		zz = mean_chord_length*mean_chord_length;
		//zz = pow(mean_chord_length, 2.0 );
	}
	//timer( false);
	printf("radius_squared after = %3f\n",radius_squared );
	int val = t ? 1 : (t2 ? 2:3 );
	cout << val << endl;
	// old working *********************************************

	//double sigma_t1_u_0_term = A_0_OVER_3*pow(u_0, 3.0) + A_1_OVER_12*pow(u_0, 4.0) + A_2_OVER_30*pow(u_0, 5.0) + A_3_OVER_60*pow(u_0, 6.0) + A_4_OVER_105*pow(u_0, 7.0) + A_5_OVER_168*pow(u_0, 8.0);								
	//double sigma_t1_theta1_u_0_term	= pow(u_0, 2.0 )*( A_0_OVER_2 + A_1_OVER_6*u_0 + A_2_OVER_12*pow(u_0, 2.0) + A_3_OVER_20*pow(u_0, 3.0) + A_4_OVER_30*pow(u_0, 4.0) + A_5_OVER_42*pow(u_0, 5.0) );
	//double sigma_theta1_u_0_term = A_0 * u_0 + A_1_OVER_2 * pow(u_0, 2.0) + A_2_OVER_3 * pow(u_0, 3.0) + A_3_OVER_4 * pow(u_0, 4.0) + A_4_OVER_5 * pow(u_0, 5.0) + A_5_OVER_6 * pow(u_0, 6.0);

	//double sigma_t2_u_2_term = A_0_OVER_3*pow(u_2, 3.0) + A_1_OVER_12*pow(u_2, 4.0) + A_2_OVER_30*pow(u_2, 5.0) + A_3_OVER_60*pow(u_2, 6.0) + A_4_OVER_105*pow(u_2, 7.0) + A_5_OVER_168*pow(u_2, 8.0);								
	//double sigma_t2_theta2_u_2_term	= pow(u_2, 2.0 )*( A_0_OVER_2 + A_1_OVER_6*u_2 + A_2_OVER_12*pow(u_2, 2.0) + A_3_OVER_20*pow(u_2, 3.0) + A_4_OVER_30*pow(u_2, 4.0) + A_5_OVER_42*pow(u_2, 5.0) );
	//double sigma_theta2_u_2_term = A_0 * u_2 + A_1_OVER_2 * pow(u_2, 2.0) + A_2_OVER_3 * pow(u_2, 3.0) + A_3_OVER_4 * pow(u_2, 4.0) + A_4_OVER_5 * pow(u_2, 5.0) + A_5_OVER_6 * pow(u_2, 6.0);

	//double sigma_1_coefficient = pow( E_0 * ( 1 + 0.038 * log( (u_1 - u_0)/X_0) ), 2.0 ) / X_0;		
	//double sigma_t1 = A_0_OVER_3*pow(u_1, 3.0) + A_1_OVER_12*pow(u_1, 4.0) + A_2_OVER_30*pow(u_1, 5.0) + A_3_OVER_60*pow(u_1, 6.0) + A_4_OVER_105*pow(u_1, 7.0) + A_5_OVER_168*pow(u_1, 8.0) - sigma_t1_u_0_term;
	//double sigma_t1_theta1 = pow(u_1, 2.0 )*( A_0_OVER_2 + A_1_OVER_6*u_1 + A_2_OVER_12*pow(u_1, 2.0) + A_3_OVER_20*pow(u_1, 3.0) + A_4_OVER_30*pow(u_1, 4.0) + A_5_OVER_42*pow(u_1, 5.0) ) - sigma_t1_theta1_u_0_term;
	//double sigma_theta1 = A_0*u_1 + A_1_OVER_2*pow(u_1, 2.0) + A_2_OVER_3*pow(u_1, 3.0) + A_3_OVER_4*pow(u_1, 4.0) + A_4_OVER_5*pow(u_1, 5.0) + A_5_OVER_6*pow(u_1, 6.0) - sigma_theta1_u_0_term;
	//double determinant_Sigma_1 = sigma_t1 * sigma_theta1 - pow( sigma_t1_theta1, 2 );//ad-bc
	//
	//double sigma_2_coefficient = pow( E_0 * ( 1 + 0.038 * log( (u_2 - u_1)/X_0 ) ), 2.0 ) / X_0;
	//double sigma_t2 = sigma_t2_u_2_term - A_0_OVER_3*pow(u_1, 3.0) - A_1_OVER_4*pow(u_1, 4.0) - A_2_OVER_5*pow(u_1, 5.0) - A_3_OVER_6*pow(u_1, 6.0) - A_4_OVER_7*pow(u_1, 7.0) - A_5_OVER_8*pow(u_1, 8.0) 
	//						+ 2*u_2*( A_0_OVER_2*pow(u_1, 2.0) + A_1_OVER_3*pow(u_1, 3.0) + A_2_OVER_4*pow(u_1, 4.0) + A_3_OVER_5*pow(u_1, 5.0) + A_4_OVER_6*pow(u_1, 6.0) + A_5_OVER_7*pow(u_1, 7.0) ) 
	//						- pow(u_2, 2.0) * ( A_0*u_1 + A_1_OVER_2*pow(u_1, 2.0) + A_2_OVER_3*pow(u_1, 3.0) + A_3_OVER_4*pow(u_1, 4.0) + A_4_OVER_5*pow(u_1, 5.0) + A_5_OVER_6*pow(u_1, 6.0) );
	//double sigma_t2_theta2 = sigma_t2_theta2_u_2_term - u_2*u_1*( A_0 +A_1_OVER_2*u_1 + A_2_OVER_3*pow(u_1, 2.0) + A_3_OVER_4*pow(u_1, 3.0) + A_4_OVER_5*pow(u_1, 4.0) + A_5_OVER_6*pow(u_1, 5.0) ) 
	//							+ pow(u_1, 2.0 )*( A_0_OVER_2 + A_1_OVER_3*u_1 + A_2_OVER_4*pow(u_1, 2.0) + A_3_OVER_5*pow(u_1, 3.0) + A_4_OVER_6*pow(u_1, 4.0) + A_5_OVER_7*pow(u_1, 5.0) );
	//double sigma_theta2 = sigma_theta2_u_2_term - ( A_0 * u_1 + A_1_OVER_2 * pow(u_1, 2.0) + A_2_OVER_3 * pow(u_1, 3.0) + A_3_OVER_4 * pow(u_1, 4.0) + A_4_OVER_5 * pow(u_1, 5.0) + A_5_OVER_6 * pow(u_1, 6.0) );				
	//double determinant_Sigma_2 = sigma_t2 * sigma_theta2 - pow( sigma_t2_theta2, 2 );//ad-bc
	// end old working ****************************************************
	// begin newer working ****************************************************
	//sigma_t1 =  sigma_1_coefficient * ( (A_0_OVER_3 * u_1_power_3 + A_1_OVER_12 * u_1_power_4 + A_2_OVER_30 * u_1_power_5 + A_3_OVER_60 * u_1_power_6 + A_4_OVER_105 * u_1_power_7 + A_5_OVER_168 * u_1_power_8 ) );	//u_1^3 : 1/3, 1/12, 1/30, 1/60, 1/105, 1/168
	//sigma_t1_theta1 =  sigma_1_coefficient * ( A_0_OVER_2 * u_1_power_2 + A_1_OVER_6 * u_1_power_3 + A_2_OVER_12 * u_1_power_4 + A_3_OVER_20 * u_1_power_5 + A_4_OVER_30 * u_1_power_6 + A_5_OVER_42 * u_1_power_7 );	//u_1^2 : 1/2, 1/6, 1/12, 1/20, 1/30, 1/42															
	//sigma_theta1 = sigma_1_coefficient * ( A_0 * u_1 + A_1_OVER_2 * u_1_power_2+ A_2_OVER_3 * u_1_power_3 + A_3_OVER_4 * u_1_power_4 + A_4_OVER_5 * u_1_power_5 + A_5_OVER_6 * u_1_power_6 );			//u_1 : 1/1, 1/2, 1/3, 1/4, 1/5, 1/6														
	//determinant_Sigma_1 = sigma_t1 * sigma_theta1 - pow( sigma_t1_theta1, 2 );//ad-bc
	/*sigma_t2 =  sigma_2_coefficient * ( sigma_2_pre_1
					- pow(u_2, 2.0) * ( A_0 * u_1 + A_1_OVER_2 * u_1_power_2 + A_2_OVER_3 * u_1_power_3 + A_3_OVER_4 * u_1_power_4 + A_4_OVER_5 * u_1_power_5 + A_5_OVER_6 * u_1_power_6 )	
					+ 2 * u_2 * ( A_0_OVER_2 * u_1_power_2 + A_1_OVER_3 * u_1_power_3 + A_2_OVER_4 * u_1_power_4 + A_3_OVER_5 * u_1_power_5 + A_4_OVER_6 * u_1_power_6 + A_5_OVER_7 * u_1_power_7 )
					- ( A_0_OVER_3 * u_1_power_3 + A_1_OVER_4 * u_1_power_4 + A_2_OVER_5 * u_1_power_5 + A_3_OVER_6 * u_1_power_6 + A_4_OVER_7 * u_1_power_7 + A_5_OVER_8 * u_1_power_8 ) );
	sigma_t2_theta2 =  sigma_2_coefficient * ( sigma_2_pre_2
							- u_2 * ( A_0 * u_1 + A_1_OVER_2 * u_1_power_2 + A_2_OVER_3 * u_1_power_3 + A_3_OVER_4 * u_1_power_4 + A_4_OVER_5 * u_1_power_5 + A_5_OVER_6 * u_1_power_6 )
							+ ( A_0_OVER_2 * u_1_power_2 + A_1_OVER_3 * u_1_power_3 + A_2_OVER_4 * u_1_power_4 + A_3_OVER_5 * u_1_power_5 + A_4_OVER_6 * u_1_power_6 + A_5_OVER_7 * u_1_power_7 ) );
	sigma_theta2 =  sigma_2_coefficient * ( sigma_2_pre_3 - ( A_0 * u_1 + A_1_OVER_2 * u_1_power_2 + A_2_OVER_3 * u_1_power_3 + A_3_OVER_4 * u_1_power_4 + A_4_OVER_5 * u_1_power_5 + A_5_OVER_6 * u_1_power_6 ) );*/
	// end newer working ****************************************************
	//double voxels[4] = {1,2,3,4};
	//std::copy( x_hull_h, x_hull_h + NUM_VOXELS, x_h );
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
	//array_2_disk( "MLP_image_init", OUTPUT_DIRECTORY, OUTPUT_FOLDER, MLP_test_image_h, MLP_IMAGE_COLUMNS, MLP_IMAGE_ROWS, MLP_IMAGE_SLICES, MLP_IMAGE_VOXELS, true );
	//MLP_test();
	//array_2_disk( "MLP_image", OUTPUT_DIRECTORY, OUTPUT_FOLDER, MLP_test_image_h, MLP_IMAGE_COLUMNS, MLP_IMAGE_ROWS, MLP_IMAGE_SLICES, MLP_IMAGE_VOXELS, true );
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
	//}a
}
void test_func2( std::vector<int>& bin_numbers, std::vector<double>& data )
{
	int angular_bin = 8;
	int v_bin = 14;
	int bin_num = angular_bin * T_BINS + v_bin * ANGULAR_BINS * T_BINS;
	bin_numbers.push_back(bin_num);
	bin_numbers.push_back(bin_num);
	bin_numbers.push_back(bin_num);
	bin_numbers.push_back(bin_num+1);
	bin_numbers.push_back(bin_num+1);
	bin_numbers.push_back(bin_num+3);
	data.push_back(1.1);
	data.push_back(1.2);
	data.push_back(1.3);
	data.push_back(0.1);
	data.push_back(0.1);
	data.push_back(5.4);

	v_bin = 15;
	bin_num = angular_bin * T_BINS + v_bin * ANGULAR_BINS * T_BINS;
	bin_numbers.push_back(bin_num);
	bin_numbers.push_back(bin_num);
	bin_numbers.push_back(bin_num);
	bin_numbers.push_back(bin_num+1);
	bin_numbers.push_back(bin_num+1);
	bin_numbers.push_back(bin_num+3);
	data.push_back(1.1);
	data.push_back(1.2);
	data.push_back(1.3);
	data.push_back(0.1);
	data.push_back(0.1);
	data.push_back(5.4);

	angular_bin = 30;
	v_bin = 14;
	bin_num = angular_bin * T_BINS + v_bin * ANGULAR_BINS * T_BINS;
	bin_numbers.push_back(bin_num);
	bin_numbers.push_back(bin_num);
	bin_numbers.push_back(bin_num);
	bin_numbers.push_back(bin_num+1);
	bin_numbers.push_back(bin_num+1);
	bin_numbers.push_back(bin_num+3);
	data.push_back(1.1);
	data.push_back(1.2);
	data.push_back(1.3);
	data.push_back(0.1);
	data.push_back(0.1);
	data.push_back(5.4);

	v_bin = 16;
	bin_num = angular_bin * T_BINS + v_bin * ANGULAR_BINS * T_BINS;
	bin_numbers.push_back(bin_num);
	bin_numbers.push_back(bin_num);
	bin_numbers.push_back(bin_num);
	bin_numbers.push_back(bin_num+1);
	bin_numbers.push_back(bin_num+1);
	bin_numbers.push_back(bin_num+3);
	data.push_back(1.1);
	data.push_back(1.2);
	data.push_back(1.3);
	data.push_back(0.1);
	data.push_back(0.1);
	data.push_back(5.4);
	//cout << smallest << endl;
	//cout << min_n<double, int>(9, 1, 2, 3, 4, 5, 6, 7, 8, 100 ) << endl;
	//cout << true << endl;
	//FILE * pFile;
	//char data_filename[MAX_INTERSECTIONS];
	//sprintf(data_filename, "%s%s/%s", OUTPUT_DIRECTORY, OUTPUT_FOLDER, "myfile.txt" );
	//pFile = fopen (data_filename,"w+");
	//int ai[1000];
	//cout << pow(ROWS, 2.0) + pow(COLUMNS,2.0) + pow(SLICES,2.0) << " " <<  sqrt(pow(ROWS, 2.0) + pow(COLUMNS,2.0) + pow(SLICES,2.0)) << " " << max_path_elements << endl;
	////pFile = freopen (data_filename,"a+", pFile);
	//for( unsigned int i = 0; i < 10; i++ )
	//{
	//	//int ai[i];
	//	for( int j = 0; j < 10 - i; j++ )
	//	{
	//		ai[j] = j; 
	//		//cout << ai[i] << endl;
	//	}
	//	write_path(data_filename, pFile, 10-i, ai, false);
	//}
	
	//int myints[] = {16,2,77,29};
	//std::vector<int> fifth (myints, myints + sizeof(myints) / sizeof(int) );

	//int x_elements = 5;
	//int y_elements = 10;
	////int x[] = {10, 20,30};
	////int angle_array[];

	//int* x = (int*) calloc( x_elements, sizeof(int));
	//int* y = (int*) calloc( y_elements, sizeof(int));
	//for( unsigned int i = 0; i < x_elements; i++ )
	//{
	//	x[i] = 10*i;
	//}
	//for( unsigned int i = 0; i < y_elements; i++ )
	//{
	//	y[i] = i;
	//}
	////cout << sizeof(&(*x)) << endl;

	//test_va_arg( fifth, BY_BIN, x_elements, x, y_elements, y );
	//else
	//{
	//	//int myints[] = {16,2,77,29};
	//	//std::vector<int> fifth (myints, myints + sizeof(myints) / sizeof(int) );
	//	va_list specific_bins;
	//	va_start( specific_bins, bin_order );
	//	int* angle_array = va_arg(specific_bins, int* );		
	//	int* v_bins_array = va_arg(specific_bins, int* );
	//	std::vector<int> temp ( angle_array,  angle_array + sizeof(angle_array) / sizeof(int) );
	//	angles = temp;
	//	std::vector<int> temp2 ( v_bins_array,  v_bins_array + sizeof(v_bins_array) / sizeof(int) );
	//	v_bins = temp2;
	//	//angles = va_arg(specific_bins, int* );
	//	//v_bins = va_arg(specific_bins, int* );
	//	va_end(specific_bins);
	//	angular_bins.resize(angles.size());
	//	std::transform(angles.begin(), angles.end(), angular_bins.begin(), std::bind2nd(std::divides<int>(), GANTRY_ANGLE_INTERVAL ) );
	//}
	//char* data_format = INT_FORMAT;
	////int x[] = {10, 20,30};
	////int y[] = {1, 2,3};
	//int* x = (int*) calloc( 3, sizeof(int));
	//int* y = (int*) calloc( 3, sizeof(int));
	//for( unsigned int i = 0; i < 3; i++)
	//{
	//	x[i] = 10*i;
	//	y[i] = i;
	//}
	//for( unsigned int i = 0; i < 3; i++)
	//{
	//	cout << x[i] << " " << y[i] << endl;
	//}

	////int type_var;
	//int* intersections = (int*) calloc( 3, sizeof(int));
	//std::iota( intersections, intersections + 3, 0 );
	//double z = discrete_dot_product<double>(x, y, intersections, 3);
	//printf("%d %d %d\n%f %f %f\n", x[1], y[1], z, x[1], y[1], z);
	//create_MLP_test_image();
	//array_2_disk( "MLP_image_init", OUTPUT_DIRECTORY, OUTPUT_FOLDER, MLP_test_image_h, MLP_IMAGE_COLUMNS, MLP_IMAGE_ROWS, MLP_IMAGE_SLICES, MLP_IMAGE_VOXELS, true );
	//MLP_test();
	//array_2_disk( "MLP_image", OUTPUT_DIRECTORY, OUTPUT_FOLDER, MLP_test_image_h, MLP_IMAGE_COLUMNS, MLP_IMAGE_ROWS, MLP_IMAGE_SLICES, MLP_IMAGE_VOXELS, true );
	//int* x = (int*)calloc(10, sizeof(int));
	//int* y = (int*)calloc(10, sizeof(int));
	//std::vector<int*> paths;
	//std::vector<int> num_paths;
	//paths.push_back(x);
	//paths.push_back(y);
	//num_paths.push_back(10);
	//num_paths.push_back(10);

	//std::vector<int> x_vec(10);
	//std::vector<int> y_vec(10);
	//std::vector<std::vector<int>> z_vec;

	//
	//for( int j = 0; j < 10; j++ )
	//{
	//	x[j] = j;
	//	y[j] = 2*j;
	//	x_vec[j] = j;
	//	y_vec[j] = 2*j;
	//}
	//for( unsigned int i = 0; i < paths.size(); i++ )
	//{
	//	for( int j = 0; j < num_paths[i]; j++ )
	//		cout << (paths[i])[j] << endl;
	//}

	//z_vec.push_back(x_vec);
	//z_vec.push_back(y_vec);

	//for( unsigned int i = 0; i < z_vec.size(); i++ )
	//{
	//	for( int j = 0; j < (z_vec[i]).size(); j++)
	//		cout << (z_vec[i])[j] << endl;

	//}

	//std::vector<std::vector<int>> t_vec(5);
	//std::vector<int> temp_vec;
	////temp_vec = new std::vector<int>();
	////std::vector<int> temp_vec = new std::vector<int>(5);
	//for( unsigned int i = 0; i < t_vec.size(); i++ )
	//{
	//	//temp_vec = new std::vector<int>();
	//	//std::vector<int> temp_vec(i);
	//	for( int j = 0; j < i; j++ )
	//	{
	//		temp_vec.push_back(i*j);
	//		//temp_vec[j] = i*j;
	//	}
	//	t_vec[i] = temp_vec;
	//	temp_vec.clear();
	//	//delete temp_vec;
	//}
	//for( unsigned int i = 0; i < t_vec.size(); i++ )
	//{
	//	for( int j = 0; j < t_vec[i].size(); j++ )
	//	{
	//		cout << (t_vec[i])[j] << endl;
	//	}
	//}

	//for( int i = 0, float df = 0.0; i < 10; i++)
	//	cout << "Hello" << endl;
	////int x[] = {2,3,4,6,7};
	////test_func_3();
	//int x[] = {-1, 0, 1};
	//bool y[] = {0,0,0}; 
	//std::transform( x, x + 3, x, y, std::logical_or<int> () );
	//for(unsigned int i = 0; i < 3; i++ )
	//	std::cout << y[i] << std::endl;
	//std::initializer_list<int> mylist;
	//std::cout << sizeof(bool) << sizeof(int) << std::endl;
	//mylist = { 10, 20, 30 };
	////std::array<int,10> y = {1,2,3,4};
	////auto ptr = y.begin();

	//int y[20];
	//int index = 0;
	//for( unsigned int i = 0; i < 20; i++ )
	//	y[index++] = i;
	//for( unsigned int i = 0; i < 20; i++ )
	//	std::cout << y[i] << std::endl;

	//int* il = { 10, 20, 30 };
	//auto p1 = il.begin();
	//auto fn_five = std::bind (my_divide,10,2);               // returns 10/2
  //std::cout << fn_five() << '\n';  

	//std::vector<int> bin_numbers;
	//std::vector<float> WEPLs;
	//test_func2( bin_numbers, WEPLs );
	//int angular_bin = 8;
	//int v_bin = 14;
	//int bin_num = angular_bin * T_BINS + v_bin * ANGULAR_BINS * T_BINS;

	//std::cout << typeid(bin_numbers.size()).name() << std::endl;
	//std::cout << typeid(1).name() << std::endl;
	//printf("%03d %03d\n", bin_numbers.size(), WEPLs.size() );


	///*for( unsigned int i = 0; i < WEPLs.size(); i++ )
	//{
	//	printf("%d %3f\n", bin_numbers[i], WEPLs[i] );
	//}*/
	//char filename[256];
	//FILE* output_file;
	//int angles[] = {32,120};
	//int v_bins[] = {14,15,16};
	//float* sino = (float*) std::calloc( 10, sizeof(float));
	//auto it = std::begin(angles);
	//std::cout << sizeof(&*sino)/sizeof(float) << std::endl << std::endl;
	//std::vector<int> angles_vec(angles, angles + sizeof(angles) / sizeof(int) );
	//std::vector<int> v_bins_vec(v_bins, v_bins + sizeof(v_bins) / sizeof(int) );
	//std::vector<int> angular_bins = angles_vec;
	//std::transform(angles_vec.begin(), angles_vec.end(), angular_bins.begin(), std::bind2nd(std::divides<int>(), GANTRY_ANGLE_INTERVAL ) );
	//int num_angles = sizeof(angles)/sizeof(int);
	//int num_v_bins = sizeof(v_bins)/sizeof(int);
	//std::cout << sizeof(v_bins) << " " << sizeof(angles) << std::endl;
	//std::cout << num_angles << " " << num_v_bins << std::endl;
	//std::cout << angles_vec.size() << " " << angular_bins.size() << std::endl;
	//bins_2_disk( "bin data", bin_numbers, WEPLs, COUNTS, ALL_BINS, BY_HISTORY );
	//bins_2_disk( "bin data", bin_numbers, WEPLs, COUNTS, ALL_BINS, BY_HISTORY, angles_vec, v_bins_vec );
	//bins_2_disk( "bin_counts", bin_numbers, WEPLs, COUNTS, SPECIFIC_BINS, BY_HISTORY, angles_vec, v_bins_vec );
	//bins_2_disk( "bin_means", bin_numbers, WEPLs, MEANS, SPECIFIC_BINS, BY_HISTORY, angles_vec, v_bins_vec );
	//bins_2_disk( "bin_members", bin_numbers, WEPLs, MEMBERS, SPECIFIC_BINS, BY_HISTORY, angles_vec, v_bins_vec );
	//std::transform(angles_vec.begin(), angles_vec.end(), angular_bins.begin(), std::bind2nd(std::divides<int>(), GANTRY_ANGLE_INTERVAL ) );
	//for( unsigned int i = 0; i < angular_bins.size(); i++ )
	//	std::cout << angular_bins[i] << std::endl;
	////std::transform(angles_vec.begin(), angles_vec.end(), angular_bins.begin(), std::bind( std::divides<int>(), 4 ) );

	//
	//auto f1 = std::bind(my_divide, _1, 10);
	////auto triple = std::mem_fn (my_divide, _1);
	//std::transform(angles_vec.begin(), angles_vec.end(), angular_bins.begin(),  f1 );
	//for( unsigned int i = 0; i < angular_bins.size(); i++ )
	//	std::cout << angular_bins[i] << std::endl;
	//int angles[] = {32,120,212};
}
__global__ void test_func_device( double* x, double* y, double* z )
{
	//x = 2;
	//y = 3;
	//z = 4;
}
__global__ void test_func_GPU( int* a)
{
	//int i = threadIdx.x;
	//std::string str;
	double delta_yx = 1.0/1.0;
	double x_to_go = 0.024;
	double y_to_go = 0.015;
	double y_to_go2 = y_to_go;
	double y_move = delta_yx * x_to_go;
	if( -1 )
		printf("-1");
	if( 1 )
		printf("1");
	if( 0 )
		printf("0");
	y_to_go -= !sin(delta_yx)*y_move;

	y_to_go2 -= !sin(delta_yx)*delta_yx * x_to_go;

	printf(" delta_yx = %8f y_move = %8f y_to_go = %8f y_to_go2 = %8f\n", delta_yx, y_move, y_to_go, y_to_go2 );
	double y = 1.36;
	////int voxel_x_out = int( ( x_exit[i] + RECON_CYL_RADIUS ) / VOXEL_WIDTH );
	//int voxel_y_out = int( ( RECON_CYL_RADIUS - y ) / VOXEL_HEIGHT );
	////int voxel_z_out = int( ( RECON_CYL_HEIGHT/2 - z_exit[i] ) /VOXEL_THICKNESS );
	//double voxel_y_float;
	//double y_inside2 = ((( RECON_CYL_RADIUS - y ) / VOXEL_HEIGHT) - voxel_y_out) * VOXEL_HEIGHT;
	//double y_inside = modf( ( RECON_CYL_RADIUS - y) /VOXEL_HEIGHT, &voxel_y_float)*VOXEL_HEIGHT;
	//printf(" voxel_y_float = %8f voxel_y_out = %d\n", voxel_y_float, voxel_y_out );
	//printf(" y_inside = %8f y_inside2 = %8f\n", y_inside, y_inside2 );
	//printf("Hello %d", i);
	float x = 1.0;
	y = 1.0;
	float z = abs(2.0) / abs( x - y );
	float z2 = abs(-2.0) / abs( x - y );
	float z3 = z*x;
	bool less = z < z2;
	bool less2 = x < z;
	bool less3 = x < z2;
	if( less )
		a[0] = 1;
	if( less2 )
		a[1] = 1;
	if( less3 )
		a[2] = 1;

	printf("%3f %3f %3f %d %d %d\n", z, z2, z3, less, less2, less3);
	//int voxel_x = blockIdx.x;
	//int voxel_y = blockIdx.y;	
	//int voxel_z = threadIdx.x;
	//int voxel = voxel_x + voxel_y * COLUMNS + voxel_z * COLUMNS * ROWS;
	//int x = 0, y = 0, z = 0;
	//test_func_device( x, y, z );
	//image[voxel] = x * y * z;
}
//#endif // #ifndef _TVS_DROP_FBP_KERNEL_H_