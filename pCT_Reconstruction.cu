/****************************************************************************************************************************************************************/
/*************************************************** Proton CT Preprocessing and Image Reconstruction Code  *****************************************************/
/****************************************************************************************************************************************************************/
#include "pCT_Reconstruction.h"
/****************************************************************************************************************************************************************/
/***************************************************************** Host functions declarations ******************************************************************/
/****************************************************************************************************************************************************************/

// Preprocessing setup and initializations 
void write_run_settings();
void assign_SSD_positions();
void initializations();
void count_histories();	
void count_histories_old();
void count_histories_v0();
void count_histories_v1();
void reserve_vector_capacity();

// Preprocessing routines
void read_data_chunk( const int, const int, const int );
void read_data_chunk_old( const int, const int, const int );
void read_data_chunk_v0( const int, const int, const int );
void read_data_chunk_v1( const int, const int, const int );
void recon_volume_intersections( const int );
void recon_volume_intersections2( const int );
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
void MLP();
void MLP_entry_exit( int&, int&, int& );
float mean_chord_length( float, float );

// Image Reconstruction
void create_hull_image_hybrid();
void define_initial_iterate();

// Write arrays/vectors to file(s)
template<typename T> void array_2_disk( char*, const char*, const char*, T*, const int, const int, const int, const int, const bool );
template<typename T> void vector_2_disk( char*, const char*, const char*, std::vector<T>, const int, const int, const int, const bool );
void write_t_bin_data( FILE*, const std::vector<int>&, const std::vector<float>&, const BIN_ANALYSIS_TYPE, const int );
void write_bin_data( const char*, const std::vector<int>&, const std::vector<float>&, const BIN_ANALYSIS_TYPE, const BIN_ANALYSIS_FOR, const BIN_ORGANIZATION, ... );
void write_t_bin_data( FILE*, int*&, float*&, const int, const BIN_ANALYSIS_TYPE, const BIN_ORGANIZATION, int );
void write_bin_data( const char*, int*&, float*&, const int, const BIN_ANALYSIS_TYPE, const BIN_ANALYSIS_FOR, const BIN_ORGANIZATION, ... );
void bin_2_txt();

// Memory transfers and allocations/deallocations
void post_cut_memory_clean(); 
void resize_vectors( const int );
void shrink_vectors( const int );
void initial_processing_memory_clean();
void allocations( const int );
void reallocations( const int );

// Helper Functions
void exit_program_if( bool );
void start_execution_timing();
void stop_execution_timing();
void pause_execution();
bool is_bad_angle( const int );								// Just for use with Micah's simultated data
int calculate_x_voxel( const float, const int, const float );
int calculate_y_voxel( const float, const int, const float );
int calculate_slice( const float, const int, const float );
int positions_2_voxels(const float, const float, const float, int&, int&, int& );
void bin_2_indexes( int, int&, int&, int& );

//// New routine test functions
void test_func();
void test_func2( std::vector<int>&, std::vector<float>&);

/****************************************************************************************************************************************************************/
/*************************************************************** Device (GPU) function declarations *************************************************************/
/****************************************************************************************************************************************************************/

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

// Hull-Detection 
template<typename T> __global__ void initialize_hull_GPU( T* );
__device__ int position_2_voxel_GPU( float &x, float &y, float &z );
__device__ void voxel_walk_GPU( bool*&, float, float, float, float, float, float );
__device__ double x_remaining_GPU( double, int, int& );
__device__ double y_remaining_GPU( double, int, int& );
__device__ double z_remaining_GPU( double, int, int& );
__global__ void FBP_image_2_hull_GPU( float*, bool* );
__global__ void SC_GPU( const int, bool*, int*, bool*, float*, float*, float*, float*, float*, float*, float* );
__global__ void MSC_GPU( const int, int*, int*, bool*, float*, float*, float*, float*, float*, float*, float* );
__global__ void SM_GPU( const int, int*, int*, bool*, float*, float*, float*, float*, float*, float*, float* );
__global__ void MSC_edge_detection_GPU( int* );
__global__ void SM_edge_detection_GPU( int*, int* );
__global__ void SM_edge_detection_GPU_2( int*, int* );
__global__ void carve_differences( int*, int* );
template<typename H, typename D> __global__ void averaging_filter_GPU( H*, D*, bool );
template<typename D> __global__ void apply_averaging_filter_GPU( D*, D* );

// Image Reconstruction
__global__ void create_hull_image_hybrid_GPU( bool*&, float*& );

// New routine test functions
__global__ void test_func_GPU( int* );
__device__ void test_func_device( int&, int&, int&);

/****************************************************************************************************************************************************************/
/************************************************************************ Program Main **************************************************************************/
/****************************************************************************************************************************************************************/
int main(int argc, char** argv)
{
	if( RUN_ON )
	{
		/********************************************************************************************************************************************************/
		/* Start the execution timing clock																														*/
		/********************************************************************************************************************************************************/
		start_execution_timing();
		/********************************************************************************************************************************************************/
		/* Initialize hull detection images and transfer them to the GPU (performed if SC_ON, MSC_ON, or SM_ON is true)											*/
		/********************************************************************************************************************************************************/
		hull_initializations();
		/********************************************************************************************************************************************************/
		/* Read the u-coordinates of the detector planes from the config file, allocate and	initialize statistical data arrays, and count the number of 		*/
		/* histories per file, projection, gantry angle, scan, and total.																						*/
		/********************************************************************************************************************************************************/		
		if( DATA_FORMAT == OLD_FORMAT )
			assign_SSD_positions();		// Read the detector plane u-coordinates from config file
		initializations();				// allocate and initialize host and GPU memory for statistical
		count_histories();				// count the number of histories per file, per scan, total, etc.
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
		printf("%d out of %d (%4.2f%%) histories passed through the reconstruction volume.\n", recon_vol_histories, total_histories, (double) recon_vol_histories / total_histories * 100  );
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
		printf("%d out of %d (%4.2f%%) histories passed cuts\n", post_cut_histories, total_histories, (double) post_cut_histories / total_histories * 100  );
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
		//hull_selection();
		//define_initial_iterate();
	}
	else
	{
		//bin_2_txt();
		test_func();
	}
	/************************************************************************************************************************************************************/
	/* Program has finished execution. Require the user to hit enter to terminate the program and close the terminal/console window								*/ 															
	/************************************************************************************************************************************************************/
	puts("Preprocessing complete.  Press enter to close the console window...");
	exit_program_if(true);
}
/****************************************************************************************************************************************************************/
/************************************************************ Preprocessing Setup and Initializations ***********************************************************/
/****************************************************************************************************************************************************************/
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
	for( int i = 0; i < 8; i++ ) {
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
				printf("ERROR: Data format is not Version (%d)!\n", version_id);
				exit_program_if(true);
			}			
		}
	}
}
/****************************************************************************************************************************************************************/
/********************************************************** Data Importation, Initial Cuts, and Binning *********************************************************/
/****************************************************************************************************************************************************************/
void read_data_chunk( const int num_histories, const int start_file_num, const int end_file_num )
{
	// The GPU cannot process all the histories at once, so they are broken up into chunks that can fit on the GPU.  As we iterate 
	// through the data one chunk at a time, we determine which histories enter the reconstruction volume and if they belong to a 
	// valid bin (i.e. t, v, and angular bin number is greater than zero and less than max).  If both are true, we push the bin
	// number, WEPL, and relative entry/exit ut/uv angles to the back of their corresponding std::vector.
	
	unsigned int size_floats = sizeof(float) * num_histories;
	unsigned int size_ints = sizeof(int) * num_histories;

	t_in_1_h         = (float*) malloc(size_floats);
	t_in_2_h         = (float*) malloc(size_floats);
	t_out_1_h        = (float*) malloc(size_floats);
	t_out_2_h        = (float*) malloc(size_floats);
	u_in_1_h         = (float*) malloc(size_floats);
	u_in_2_h         = (float*) malloc(size_floats);
	u_out_1_h        = (float*) malloc(size_floats);
	u_out_2_h        = (float*) malloc(size_floats);
	v_in_1_h         = (float*) malloc(size_floats);
	v_in_2_h         = (float*) malloc(size_floats);
	v_out_1_h        = (float*) malloc(size_floats);
	v_out_2_h        = (float*) malloc(size_floats);		
	WEPL_h           = (float*) malloc(size_floats);
	gantry_angle_h   = (int*)   malloc(size_ints);

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
		case VERSION_1  : read_data_chunk_v1(  num_histories, start_file_num, end_file_num - 1 );	break;
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
	for( int file_num = start_file_num; file_num <= end_file_num; file_num++ )
	{
		gantry_position = file_num / NUM_SCANS;
		gantry_angle = int(gantry_position * GANTRY_ANGLE_INTERVAL);
		scan_number = file_num % NUM_SCANS + 1;
		scan_histories = histories_per_file[file_num];

		printf("Reading File for Gantry Angle %d from Scan Number %d...\n", gantry_angle, scan_number );
		sprintf(data_filename, "%s%s/%s_%03d%s", INPUT_DIRECTORY, INPUT_FOLDER, INPUT_BASE_NAME, gantry_angle, FILE_EXTENSION );	
		std::ifstream data_file(data_filename, std::ios::binary);
		if( data_file == NULL )
		{
			fputs( "File not found:  Check that the directories and files are properly named.", stderr ); 
			exit_program_if(true);
		}

		data_file.read(magic_number, 4);
		magic_number[4] = '\0';
		if( strcmp(magic_number, "PCTD") ) {
			puts("Error: unknown file type (should be PCTD)!\n");
			exit_program_if(true);
		}
		
		data_file.read((char*)&version_id, sizeof(int));
		if( version_id == 0 )
		{
			data_file.read((char*)&file_histories, sizeof(int));
	
			puts("Reading headers from file...\n");
	
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
	
			data_size = num_histories * sizeof(float);
	
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
			
			for( int i = 0; i < num_histories; i++ ) 
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
					t_in_1_h[i]		+= T_SHIFT;
					t_in_2_h[i]		+= T_SHIFT;
					t_out_1_h[i]	+= T_SHIFT;
					t_out_2_h[i]	+= T_SHIFT;
					WEPL_h[i]		*= MM_TO_CM;
					//if( WEPL_h[i] < 0 )
						//printf("WEPL[%d] = %3f\n", i, WEPL_h[i] );
					u_in_1_h[i]		*= MM_TO_CM;
					u_in_2_h[i]		*= MM_TO_CM;
					u_out_1_h[i]	*= MM_TO_CM;
					u_out_2_h[i]	*= MM_TO_CM;
					u_in_1_h[i]		+= U_SHIFT;
					u_in_2_h[i]		+= U_SHIFT;
					u_out_1_h[i]	+= U_SHIFT;
					u_out_2_h[i]	+= U_SHIFT;
					if( COUNT_0_WEPLS && WEPL_h[i] == 0 )
					{
						zero_WEPL++;
						zero_WEPL_files++;
					}
				}
				if( WRITE_SSD_ANGLES )
				{
					ut_entry_angle[i] = atan2( t_in_2_h[i] - t_in_1_h[i], u_in_2_h[i] - u_in_1_h[i] );	
					uv_entry_angle[i] = atan2( v_in_2_h[i] - v_in_1_h[i], u_in_2_h[i] - u_in_1_h[i] );	
					ut_exit_angle[i] = atan2( t_out_2_h[i] - t_out_1_h[i], u_out_2_h[i] - u_out_1_h[i] );	
					uv_exit_angle[i] = atan2( v_out_2_h[i] - v_out_1_h[i], u_out_2_h[i] - u_out_1_h[i] );	
				}
				gantry_angle_h[i] = int(projection_angle);				
			}
			data_file.close();
			if( WRITE_SSD_ANGLES )
			{
				sprintf(data_filename, "%s_%03d%s", "ut_entry_angle", gantry_angle, ".txt" );
				array_2_disk( data_filename, OUTPUT_DIRECTORY, OUTPUT_FOLDER, ut_entry_angle, COLUMNS, ROWS, SLICES, file_histories, true );
				sprintf(data_filename, "%s_%03d%s", "uv_entry_angle", gantry_angle, ".txt" );
				array_2_disk( "ut_entry_angle", OUTPUT_DIRECTORY, OUTPUT_FOLDER, uv_entry_angle, COLUMNS, ROWS, SLICES, file_histories, true );
				sprintf(data_filename, "%s_%03d%s", "ut_exit_angle", gantry_angle, ".txt" );
				array_2_disk( "ut_entry_angle", OUTPUT_DIRECTORY, OUTPUT_FOLDER, ut_exit_angle, COLUMNS, ROWS, SLICES, file_histories, true );
				sprintf(data_filename, "%s_%03d%s", "uv_exit_angle", gantry_angle, ".txt" );
				array_2_disk( "ut_entry_angle", OUTPUT_DIRECTORY, OUTPUT_FOLDER, uv_exit_angle, COLUMNS, ROWS, SLICES, file_histories, true );
			}
		}
	}
	if( COUNT_0_WEPLS )
	{
		std::cout << "Histories in " << gantry_angle_h[0] << "with WEPL = 0 :" << zero_WEPL_files << std::endl;
		zero_WEPL_files = 0;
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
		if( version_id == 0 )
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
			for( int i = 0; i < num_histories; i++ ) 
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
	free(v_in_1_h);
	free(v_in_2_h);
	free(u_in_1_h);
	free(u_in_2_h);
	free(t_out_1_h);
	free(t_out_2_h);
	free(v_out_1_h);
	free(v_out_2_h);
	free(u_out_1_h);
	free(u_out_2_h);

	cudaFree(t_in_1_d);
	cudaFree(t_in_2_d);
	cudaFree(v_in_1_d);
	cudaFree(v_in_2_d);
	cudaFree(u_in_1_d);
	cudaFree(u_in_2_d);
	cudaFree(t_out_1_d);
	cudaFree(t_out_2_d);
	cudaFree(v_out_1_d);
	cudaFree(v_out_2_d);
	cudaFree(u_out_1_d);
	cudaFree(u_out_2_d);
	cudaFree(gantry_angle_d);
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

	cudaMalloc((void**) &WEPL_d,				size_floats);
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
		array_2_disk( data_filename, OUTPUT_DIRECTORY, OUTPUT_FOLDER, bin_num_h, COLUMNS, ROWS, SLICES, num_histories, true );
	}

	int offset = 0;
	for( int i = 0; i < num_histories; i++ )
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
	printf( "%d out of %d histories passed intersection cuts this iteration\n", offset, num_histories );

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

	//cudaFree( bin_num_d );
	cudaFree( xy_entry_angle_d );
	cudaFree( xz_entry_angle_d );
	cudaFree( xy_exit_angle_d );
	cudaFree( xz_exit_angle_d );
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
				else
					bin_num[i] = -1;
			}
	}
}
/****************************************************************************************************************************************************************/
/***************************************************************** Statistical Analysis and Cuts ****************************************************************/
/****************************************************************************************************************************************************************/
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
		write_bin_data( "WEPL_dist_pre_test2", empty_parameter, mean_WEPL_h, NUM_BINS, MEANS, ALL_BINS, BY_BIN );
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
	////cudaMemcpy( xy_exit_angle_d,		&xy_exit_angle[start_position],		size_floats, cudaMemcpyHostToDevice);
	////cudaMemcpy( xz_exit_angle_d,		&xz_exit_angle[start_position],		size_floats, cudaMemcpyHostToDevice);

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
	double WEPL_difference, rel_ut_angle_difference, rel_uv_angle_difference;
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
		WEPL_difference = WEPL[i] - mean_WEPL[bin_num[i]];
		rel_ut_angle_difference = rel_ut_angle - mean_rel_ut_angle[bin_num[i]];
		rel_uv_angle_difference = rel_uv_angle - mean_rel_uv_angle[bin_num[i]];

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
	////cudaMemcpy( xy_exit_angle_d,		&xy_exit_angle[start_position],		size_floats, cudaMemcpyHostToDevice);
	////cudaMemcpy( xz_exit_angle_d,		&xz_exit_angle[start_position],		size_floats, cudaMemcpyHostToDevice);

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

	for( int i = 0; i < num_histories; i++ )
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
			////xy_exit_angle[post_cut_histories] = xy_exit_angle[start_position + i];
			////xz_exit_angle[post_cut_histories] = xz_exit_angle[start_position + i];
			post_cut_histories++;
		}
	}
	//free( failed_cuts_h );
	//cudaFree( failed_cuts_d);
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
/****************************************************************************************************************************************************************/
/************************************************************************** FBP *********************************************************************************/
/****************************************************************************************************************************************************************/
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
		write_bin_data( "WEPL_dist_post_test2", empty_parameter, sinogram_h, NUM_BINS, MEANS, ALL_BINS, BY_BIN );
	}
	//cudaMemcpy(sinogram_h,  sinogram_d, SIZE_BINS_FLOAT, cudaMemcpyDeviceToHost);
	//array_2_disk("sinogram", OUTPUT_DIRECTORY, OUTPUT_FOLDER, sinogram_h, T_BINS, ANGULAR_BINS, V_BINS, NUM_BINS, true );

	//bin_counts_h		  = (int*)	 calloc( NUM_BINS, sizeof(int) );
	//cudaMemcpy(bin_counts_h, bin_counts_d, SIZE_BINS_INT, cudaMemcpyDeviceToHost) ;
	//array_2_disk( "bin_counts_post", OUTPUT_DIRECTORY, OUTPUT_FOLDER, bin_counts_h, T_BINS, ANGULAR_BINS, V_BINS, NUM_BINS, true );
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

	bool FBP_on_GPU = true;
	FBP_image_h = (float*) calloc( VOXELS, sizeof(float) );
	if( FBP_image_h == NULL ) 
	{
		printf("ERROR: Memory not allocated for FBP_image_h!\n");
		exit_program_if(true);
	}
	if( FBP_on_GPU )
	{
		free(sinogram_filtered_h);
		cudaMalloc((void**) &FBP_image_d, SIZE_IMAGE_FLOAT );
		cudaMemcpy( FBP_image_d, FBP_image_h, SIZE_IMAGE_FLOAT, cudaMemcpyHostToDevice );

		dim3 dimBlock( SLICES );
		dim3 dimGrid( COLUMNS, ROWS );   
		backprojection_GPU<<< dimGrid, dimBlock >>>( sinogram_filtered_d, FBP_image_d );
		cudaFree(sinogram_filtered_d);
	}
	else
	{
		cudaMemcpy(sinogram_filtered_h, sinogram_filtered_d, SIZE_BINS_FLOAT, cudaMemcpyDeviceToHost);
		cudaFree(sinogram_filtered_d);
		backprojection();	
		free(sinogram_filtered_h);
	}

	if( WRITE_FBP_IMAGE )
	{
		cudaMemcpy( FBP_image_h, FBP_image_d, SIZE_IMAGE_FLOAT, cudaMemcpyDeviceToHost );
		array_2_disk( "FBP_image_h", OUTPUT_DIRECTORY, OUTPUT_FOLDER, FBP_image_h, COLUMNS, ROWS, SLICES, VOXELS, true );
	}

	// Generate FBP hull by thresholding FBP image
	FBP_image_2_hull();
	//initialize_hull( FBP_hull_h, FBP_hull_d );
	//std::transform(x, x + 10, FBP_hull_h, std::bind2nd(std::less_equal<float>(), 0.6 ) );

	// Discard FBP image unless it is to be used as the initial iterate x_0 in iterative image reconstruction
	if( X_K0 != FBP_IMAGE && X_K0 != HYBRID )
	{
		free(FBP_image_h);
		cudaFree(FBP_image_d);
	}
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
	//for( int i = 0; i < NUM_BINS; i++ )
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
	if ( voxel < VOXELS )
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

	bool FBP_hull_on_GPU = false;
	
	if( FBP_hull_on_GPU )
	{
		initialize_hull( FBP_hull_h, FBP_hull_d );
		dim3 dimBlock( SLICES );
		dim3 dimGrid( COLUMNS, ROWS );   
		FBP_image_2_hull_GPU<<< dimGrid, dimBlock >>>( FBP_image_d, FBP_hull_d );	
		cudaMemcpy( FBP_hull_h, FBP_hull_d, SIZE_IMAGE_BOOL, cudaMemcpyDeviceToHost );
	}
	else
	{
		FBP_hull_h = (bool*) calloc( COLUMNS * ROWS * SLICES, sizeof(bool) );
		std::transform(FBP_image_h, FBP_image_h + VOXELS, FBP_hull_h, std::bind2nd(std::greater<float>(), 0.6 ) );
		/*for( int iteration = 0; iteration < 1; iteration++ )
		{
			for( int slice = 0; slice < SLICES; slice++ )
			{
				for( int row = 0; row < ROWS; row++ )
				{
					for( int column = 0; column < COLUMNS; column++ )
					{
						double x = -RECON_CYL_RADIUS + ( column + 0.5 )* VOXEL_WIDTH;
						double y = RECON_CYL_RADIUS - (row + 0.5) * VOXEL_HEIGHT;
						if( FBP_image_h[( slice * COLUMNS * ROWS ) + ( row * COLUMNS ) + column] > FBP_THRESHOLD && ( pow(x, 2) + pow(y, 2) < pow(RECON_CYL_RADIUS, 2) ) ) 
							FBP_hull_h[( slice * COLUMNS * ROWS ) + ( row * COLUMNS ) + column] = true; 
						else
							FBP_hull_h[( slice * COLUMNS * ROWS ) + ( row * COLUMNS ) + column] = false; 
					}
				}
			}
		}*/
	}
	if( WRITE_FBP_HULL )
		array_2_disk( "x_FBP", OUTPUT_DIRECTORY, OUTPUT_FOLDER, FBP_hull_h, COLUMNS, ROWS, SLICES, VOXELS, true );

	if( MLP_HULL != FBP_HULL)
	{
		//cudaFree();		
		free(FBP_hull_h);
		cudaFree(FBP_hull_d);
	}
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
/****************************************************************************************************************************************************************/
/****************************************************************** Image Initialization ************************************************************************/
/****************************************************************************************************************************************************************/
template<typename T> void initialize_hull( T*& hull_h, T*& hull_d )
{
	/* Allocate memory and initialize hull on the GPU.  Use the image and reconstruction cylinder parameters to determine the location of the perimeter of  */
	/* the reconstruction cylinder, which is centered on the origin (center) of the image.  Assign voxels inside the perimeter of the reconstruction volume */
	/* the value 1 and those outside 0.																														*/

	int image_size = VOXELS * sizeof(T);
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
/****************************************************************************************************************************************************************/
/********************************************************************* Hull Detection ***************************************************************************/
/****************************************************************************************************************************************************************/
void hull_initializations()
{		
	if( SC_ON )
		initialize_hull( SC_hull_h, SC_hull_d );
	if( MSC_ON )
		initialize_hull( MSC_counts_h, MSC_counts_d );
	if( SM_ON )
		initialize_hull( SM_counts_h, SM_counts_d );
}
void hull_detection( const int histories_to_process)
{
	if( SC_ON  ) 
		SC( histories_to_process );		
	if( MSC_ON )
		MSC( histories_to_process );
	if( SM_ON )
		SM( histories_to_process );   
}
__device__ void voxel_walk_GPU( bool*& image, float x_entry, float y_entry, float z_entry, float x_exit, float y_exit, float z_exit )
{
	/********************************************************************************************/
	/********************************* Voxel Walk Parameters ************************************/
	/********************************************************************************************/
	int x_move_direction, y_move_direction, z_move_direction;
	float delta_yx, delta_zx, delta_zy;
	//float x_move = 0, y_move = 0, z_move = 0;
	/********************************************************************************************/
	/**************************** Status Tracking Information ***********************************/
	/********************************************************************************************/
	float x = x_entry, y = y_entry, z = z_entry;
	float x_to_go, y_to_go, z_to_go;		
	float x_extension, y_extension;	
	int voxel_x, voxel_y, voxel_z;
	int voxel, voxel_x_out, voxel_y_out, voxel_z_out, voxel_out; 
	bool end_walk;
	/********************************************************************************************/
	/******************** Initial Conditions and Movement Characteristics ***********************/
	/********************************************************************************************/
	x_move_direction = ( x_entry <= x_exit ) - ( x_entry > x_exit );
	y_move_direction = ( y_entry <= y_exit ) - ( y_entry > y_exit );
	z_move_direction = ( z_entry <= z_exit ) - ( z_entry > z_exit );
	
	x_to_go = x_remaining_GPU( x, x_move_direction, voxel_x );
	y_to_go = y_remaining_GPU( y, -y_move_direction, voxel_y );
	z_to_go = z_remaining_GPU( z, -z_move_direction, voxel_z );
	voxel = voxel_x + voxel_y * COLUMNS + voxel_z * COLUMNS * ROWS;
	/********************************************************************************************/
	/***************************** Path and Walk Information ************************************/
	/********************************************************************************************/
	// Lengths/Distances as x is Incremented One Voxel
	delta_yx = abs( (y_exit - y_entry)/(x_exit - x_entry) );
	delta_zx = abs( (z_exit - z_entry)/(x_exit - x_entry) );
	delta_zy = abs( (z_exit - z_entry)/(y_exit - y_entry) );
	/********************************************************************************************/
	/************************* Initialize and Check Exit Conditions *****************************/
	/********************************************************************************************/
	voxel_x_out = int( ( x_exit + RECON_CYL_RADIUS ) / VOXEL_WIDTH );
	voxel_y_out = int( ( RECON_CYL_RADIUS - y_exit ) / VOXEL_HEIGHT );
	voxel_z_out = int( ( RECON_CYL_HEIGHT/2 - z_exit ) /VOXEL_THICKNESS );
	voxel_out = int(voxel_x_out + voxel_y_out * COLUMNS + voxel_z_out * COLUMNS * ROWS);
		
	end_walk = ( voxel == voxel_out ) || ( voxel_x >= COLUMNS ) || ( voxel_y >= ROWS ) || ( voxel_z >= SLICES );
	if( !end_walk )
		image[voxel] = 0;
	/********************************************************************************************/
	/*********************************** Voxel Walk Routine *************************************/
	/********************************************************************************************/
	if( z_entry != z_exit )
	{
		while( !end_walk )
		{
			// Change in z for Move to Voxel Edge in x and y
			x_extension = delta_zx * x_to_go;
			y_extension = delta_zy * y_to_go;
			if( z_to_go <= x_extension && z_to_go <= y_extension )
			{
				//printf("z_to_go <= x_extension && z_to_go <= y_extension\n");
				//x_move = z_to_go / delta_zx;
				//y_move = z_to_go / delta_zy;
				//z_move = z_to_go;
				//x_to_go -= x_move;
				//y_to_go -= y_move;
				x_to_go -= z_to_go / delta_zx;
				y_to_go -= z_to_go / delta_zy;
				z_to_go = VOXEL_THICKNESS;
				voxel_z -= z_move_direction;
				if( x_to_go <= 0 )
				{
					voxel_x += x_move_direction;
					x_to_go = VOXEL_WIDTH;
				}
				if(	y_to_go <= 0 )
				{
					voxel_y -= y_move_direction;
					y_to_go = VOXEL_HEIGHT;
				}
			}
			//If Next Voxel Edge is in x or xy Diagonal
			else if( x_extension <= y_extension )
			{
				//printf(" x_extension <= y_extension \n");
				//x_move = x_to_go;
				//y_move = delta_yx * x_to_go;
				//z_move = delta_zx * x_to_go;
				x_to_go = VOXEL_WIDTH;
				y_to_go -= delta_yx * x_to_go;
				z_to_go -= delta_zx * x_to_go;
				//x_to_go = VOXEL_WIDTH;
				//y_to_go -= y_move;
				//z_to_go -= z_move;
				voxel_x += x_move_direction;
				if( y_to_go <= 0 )
				{
					y_to_go = VOXEL_HEIGHT;
					voxel_y -= y_move_direction;
				}
			}
			// Else Next Voxel Edge is in y
			else
			{
				//printf(" y_extension < x_extension \n");
				//x_move = y_to_go / delta_yx;
				//y_move = y_to_go;
				//z_move = delta_zy * y_to_go;
				x_to_go -= y_to_go / delta_yx;
				y_to_go = VOXEL_HEIGHT;
				z_to_go -= delta_zy * y_to_go;
				/*x_to_go -= x_move;
				y_to_go = VOXEL_HEIGHT;
				z_to_go -= z_move;*/
				voxel_y -= y_move_direction;
			}
			//x += x_move_direction * x_move;
			//y += y_move_direction * y_move;
			//z += z_move_direction * z_move;				
			voxel = voxel_x + voxel_y * COLUMNS + voxel_z * COLUMNS * ROWS;
			end_walk = ( voxel == voxel_out ) || ( voxel_x >= COLUMNS ) || ( voxel_y >= ROWS ) || ( voxel_z >= SLICES );
			if( !end_walk )
				image[voxel] = 0;
		}
	}
	else
	{
		//printf("hello");
		//int i = 0;
		//printf("z_exit[i] == z_entry[i]\n");
		while( !end_walk )
		{
			// Change in x for Move to Voxel Edge in y
			y_extension = y_to_go / delta_yx;
			//If Next Voxel Edge is in x or xy Diagonal
			if( x_to_go <= y_extension )
			{
				//printf(" x_to_go <= y_extension \n");
				//x_move = x_to_go;
				//y_move = delta_yx * x_to_go;	
				x_to_go = VOXEL_WIDTH;
				y_to_go -= delta_yx * x_to_go;
				//y_to_go -= y_move;
				voxel_x += x_move_direction;
				if( y_to_go <= 0 )
				{
					y_to_go = VOXEL_HEIGHT;
					voxel_y -= y_move_direction;
				}
			}
			// Else Next Voxel Edge is in y
			else
			{
				//printf(" y_extension < x_extension \n");
				//x_move = y_to_go / delta_yx;
				//y_move = y_to_go;
				//x_to_go -= x_move;
				x_to_go -= y_to_go / delta_yx;
				y_to_go = VOXEL_HEIGHT;
				voxel_y -= y_move_direction;
			}
			//x += x_move_direction * x_move;
			//y += y_move_direction * y_move;
			voxel = voxel_x + voxel_y * COLUMNS + voxel_z * COLUMNS * ROWS;			
			end_walk = ( voxel == voxel_out ) || ( voxel_x >= COLUMNS ) || ( voxel_y >= ROWS ) || ( voxel_z >= SLICES );
			if( !end_walk )
				image[voxel] = 0;
			//i++;
		}// end: while( !end_walk )
		//printf("i = %d", i );
	}//end: else: z_entry[i] != z_exit[i] => z_entry[i] == z_exit[i]
}
__device__ double x_remaining_GPU( double x, int x_move_direction, int& voxel_x )
{
	double voxel_x_float;
	double x_inside = modf( (x + RECON_CYL_RADIUS) / VOXEL_WIDTH, &voxel_x_float ) * VOXEL_WIDTH;	
	voxel_x = voxel_x_float;
	return ( x_move_direction > 0 ) * (VOXEL_WIDTH - x_inside) + ( x_move_direction <= 0 ) * x_inside;	
}
__device__ double y_remaining_GPU( double y, int y_move_direction, int& voxel_y )
{
	double voxel_y_float;
	double y_inside = modf( (RECON_CYL_RADIUS - y) / VOXEL_HEIGHT, &voxel_y_float ) * VOXEL_HEIGHT;
	voxel_y = voxel_y_float;
	return ( y_move_direction > 0 ) * (VOXEL_HEIGHT - y_inside) + ( y_move_direction <= 0 ) * y_inside;	
}
__device__ double z_remaining_GPU( double z, int z_move_direction, int& voxel_z )
{
	float voxel_z_float;
	float z_inside = modf( (RECON_CYL_HEIGHT/2 - z) / VOXEL_THICKNESS, &voxel_z_float ) * VOXEL_THICKNESS;
	voxel_z = voxel_z_float;
	return ( z_move_direction > 0 ) * (VOXEL_THICKNESS - z_inside) + ( z_move_direction <= 0 ) * z_inside;
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
/****************************************************************************************************************************************************************/
void SC( const int num_histories )
{
	dim3 dimBlock(THREADS_PER_BLOCK);
	dim3 dimGrid( (int)( num_histories / THREADS_PER_BLOCK ) + 1 );
	SC_GPU<<<dimGrid, dimBlock>>>
	(
		num_histories, SC_hull_d, bin_num_d, missed_recon_volume_d, WEPL_d,
		x_entry_d, y_entry_d, z_entry_d, x_exit_d, y_exit_d, z_exit_d
	);
}
__global__ void SC_GPU
( 
	const int num_histories, bool* SC_hull, int* bin_num, bool* missed_recon_volume, float* WEPL,
	float* x_entry, float* y_entry, float* z_entry, float* x_exit, float* y_exit, float* z_exit
)
{
	int i = threadIdx.x + blockIdx.x * THREADS_PER_BLOCK;
	if( (i < num_histories) && !missed_recon_volume[i] && (WEPL[i] <= SC_THRESHOLD) )
	{
		/********************************************************************************************/
		/********************************* Voxel Walk Parameters ************************************/
		/********************************************************************************************/
		int x_move_direction, y_move_direction, z_move_direction;
		double delta_yx, delta_zx, delta_zy;
		double x_move = 0, y_move = 0, z_move = 0;
		/********************************************************************************************/
		/**************************** Status Tracking Information ***********************************/
		/********************************************************************************************/
		//float x = x_entry[i], y = y_entry[i], z = z_entry[i];
		double x_to_go, y_to_go, z_to_go;		
		double x_extension, y_extension;	
		int voxel_x, voxel_y, voxel_z;
		//int voxel_x_out, voxel_y_out, voxel_z_out; 
		int voxel, voxel_out; 
		bool end_walk;
		/********************************************************************************************/
		/******************** Initial Conditions and Movement Characteristics ***********************/
		/********************************************************************************************/
		x_move_direction = ( x_entry[i] <= x_exit[i] ) - ( x_entry[i] >= x_exit[i] );
		y_move_direction = ( y_entry[i] <= y_exit[i] ) - ( y_entry[i] >= y_exit[i] );
		z_move_direction = ( z_entry[i] <= z_exit[i] ) - ( z_entry[i] >= z_exit[i] );		
		x_to_go = x_remaining_GPU( x_entry[i], x_move_direction, voxel_x );
		y_to_go = y_remaining_GPU( y_entry[i], -y_move_direction, voxel_y );
		z_to_go = z_remaining_GPU( z_entry[i], -z_move_direction, voxel_z );
		voxel = voxel_x + voxel_y * COLUMNS + voxel_z * COLUMNS * ROWS;
		voxel_out = position_2_voxel_GPU( x_exit[i], y_exit[i], z_exit[i] ); 
		/********************************************************************************************/
		/***************************** Path and Walk Information ************************************/
		/********************************************************************************************/
		// Lengths/Distances as x is Incremented One Voxel
		delta_yx = abs( y_exit[i] - y_entry[i] ) / abs( x_exit[i] - x_entry[i] );
		delta_zx = abs( z_exit[i] - z_entry[i] ) / abs( x_exit[i] - x_entry[i] );
		delta_zy = abs( z_exit[i] - z_entry[i] ) / abs( y_exit[i] - y_entry[i] );
		/********************************************************************************************/
		/************************* Initialize and Check Exit Conditions *****************************/
		/********************************************************************************************/		
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
				// Change in z for Move to Voxel Edge in x and y
				x_extension = delta_zx * x_to_go;
				y_extension = delta_zy * y_to_go;
				if( (z_to_go <= x_extension  ) && (z_to_go <= y_extension) )
				{
					//printf("z_to_go <= x_extension && z_to_go <= y_extension\n");
					x_move = z_to_go / delta_zx;
					y_move = z_to_go / delta_zy;
					z_move = z_to_go;
					x_to_go -= x_move * abs(x_move_direction);
					y_to_go -= y_move * abs(y_move_direction);
					//x_to_go -= z_to_go / delta_zx * abs(x_move_direction);
					//y_to_go -= z_to_go / delta_zy * abs(y_move_direction);
					z_to_go = VOXEL_THICKNESS;
					voxel_z -= z_move_direction;
					if( x_to_go == 0 )
					{
						voxel_x += x_move_direction;
						x_to_go = VOXEL_WIDTH;
					}
					if(	y_to_go == 0 )
					{
						voxel_y -= y_move_direction;
						y_to_go = VOXEL_HEIGHT;
					}
				}
				//If Next Voxel Edge is in x or xy Diagonal
				else if( x_extension <= y_extension )
				{
					//printf(" x_extension <= y_extension \n");
					x_move = x_to_go;
					y_move = delta_yx * x_to_go;
					z_move = delta_zx * x_to_go;
					x_to_go = VOXEL_WIDTH;
					y_to_go -= y_move * abs(y_move_direction);
					z_to_go -= z_move * abs(z_move_direction);				
					//y_to_go -= delta_yx * x_to_go * abs(y_move_direction);
					//z_to_go -= delta_zx * x_to_go * abs(z_move_direction);
					//x_to_go = VOXEL_WIDTH;
					voxel_x += x_move_direction;
					if( y_to_go == 0 )
					{
						y_to_go = VOXEL_HEIGHT;
						voxel_y -= y_move_direction;
					}
				}
				// Else Next Voxel Edge is in y
				else
				{
					//printf(" y_extension < x_extension \n");
					x_move = y_to_go / delta_yx;
					y_move = y_to_go;
					z_move = delta_zy * y_to_go;
					x_to_go -= x_move * abs(x_move_direction);
					y_to_go = VOXEL_HEIGHT;
					z_to_go -= z_move * abs(z_move_direction);
					//x_to_go -= y_to_go / delta_yx * abs(x_move_direction);
					//z_to_go -= delta_zy * y_to_go * abs(z_move_direction);
					//y_to_go = VOXEL_HEIGHT;
					voxel_y -= y_move_direction;
				}
				//x += x_move_direction * x_move;
				//y += y_move_direction * y_move;
				//z += z_move_direction * z_move;				
				voxel = voxel_x + voxel_y * COLUMNS + voxel_z * COLUMNS * ROWS;
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
				// Change in x for Move to Voxel Edge in y
				y_extension = y_to_go / delta_yx;
				//If Next Voxel Edge is in x or xy Diagonal
				if( x_to_go <= y_extension )
				{
					//printf(" x_to_go <= y_extension \n");
					x_move = x_to_go;
					y_move = delta_yx * x_to_go;	
					x_to_go = VOXEL_WIDTH;
					//y_to_go -= (delta_yx * x_to_go) * abs(y_move_direction);
					y_to_go -= y_move * abs(y_move_direction);
					voxel_x += x_move_direction;
					if( y_to_go == 0 )
					{
						y_to_go = VOXEL_HEIGHT;
						voxel_y -= y_move_direction;
					}
				}
				// Else Next Voxel Edge is in y
				else
				{
					//printf(" y_extension < x_extension \n");
					x_move = y_to_go / delta_yx;
					y_move = y_to_go;
					x_to_go -= x_move * abs(x_move_direction);
					//x_to_go -= y_to_go / delta_yx * abs(x_move_direction);
					y_to_go = VOXEL_HEIGHT;
					voxel_y -= y_move_direction;
				}
				//x += x_move_direction * x_move;
				//y += y_move_direction * y_move;
				voxel = voxel_x + voxel_y * COLUMNS + voxel_z * COLUMNS * ROWS;		
				end_walk = ( voxel == voxel_out ) || ( voxel_x >= COLUMNS ) || ( voxel_y >= ROWS ) || ( voxel_z >= SLICES );
				if( !end_walk )
					SC_hull[voxel] = 0;
			}// end: while( !end_walk )
			//printf("i = %d", i );
		}//end: else: z_entry[i] != z_exit[i] => z_entry[i] == z_exit[i]
	}// end: if( (i < num_histories) && !missed_recon_volume[i] && (WEPL[i] <= SC_THRESHOLD) )
}
/****************************************************************************************************************************************************************/
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
		/********************************* Voxel Walk Parameters ************************************/
		/********************************************************************************************/
		int x_move_direction, y_move_direction, z_move_direction;
		double delta_yx, delta_zx, delta_zy;
		double x_move = 0, y_move = 0, z_move = 0;
		/********************************************************************************************/
		/**************************** Status Tracking Information ***********************************/
		/********************************************************************************************/
		//double x = x_entry[i], y = y_entry[i], z = z_entry[i];
		double x_to_go, y_to_go, z_to_go;		
		double x_extension, y_extension;	
		int voxel_x, voxel_y, voxel_z;
		//int voxel_x_out, voxel_y_out, voxel_z_out; 
		int voxel, voxel_out; 
		bool end_walk;
		/********************************************************************************************/
		/******************** Initial Conditions and Movement Characteristics ***********************/
		/********************************************************************************************/
		x_move_direction = ( x_entry[i] <= x_exit[i] ) - ( x_entry[i] >= x_exit[i] );
		y_move_direction = ( y_entry[i] <= y_exit[i] ) - ( y_entry[i] >= y_exit[i] );
		z_move_direction = ( z_entry[i] <= z_exit[i] ) - ( z_entry[i] >= z_exit[i] );		
		x_to_go = x_remaining_GPU( x_entry[i], x_move_direction, voxel_x );
		y_to_go = y_remaining_GPU( y_entry[i], -y_move_direction, voxel_y );
		z_to_go = z_remaining_GPU( z_entry[i], -z_move_direction, voxel_z );
		voxel = voxel_x + voxel_y * COLUMNS + voxel_z * COLUMNS * ROWS;
		voxel_out = position_2_voxel_GPU( x_exit[i], y_exit[i], z_exit[i] ); 
		/********************************************************************************************/
		/***************************** Path and Walk Information ************************************/
		/********************************************************************************************/
		// Lengths/Distances as x is Incremented One Voxel
		delta_yx = abs( y_exit[i] - y_entry[i] ) / abs( x_exit[i] - x_entry[i] );
		delta_zx = abs( z_exit[i] - z_entry[i] ) / abs( x_exit[i] - x_entry[i] );
		delta_zy = abs( z_exit[i] - z_entry[i] ) / abs( y_exit[i] - y_entry[i] );
		/********************************************************************************************/
		/************************* Initialize and Check Exit Conditions *****************************/
		/********************************************************************************************/		
		end_walk = ( voxel == voxel_out ) || ( voxel_x >= COLUMNS ) || ( voxel_y >= ROWS ) || ( voxel_z >= SLICES );
		if( !end_walk )
			atomicAdd( &MSC_counts[voxel], 1 );
		/********************************************************************************************/
		/*********************************** Voxel Walk Routine *************************************/
		/********************************************************************************************/
		if( z_move_direction != 0 )
		{
			//printf("z_exit[i] != z_entry[i]\n");
			while( !end_walk )
			{
				// Change in z for Move to Voxel Edge in x and y
				x_extension = delta_zx * x_to_go;
				y_extension = delta_zy * y_to_go;
				if( (z_to_go <= x_extension  ) && (z_to_go <= y_extension) )
				{
					//printf("z_to_go <= x_extension && z_to_go <= y_extension\n");
					x_move = z_to_go / delta_zx;
					y_move = z_to_go / delta_zy;
					z_move = z_to_go;
					x_to_go -= x_move * abs(x_move_direction);
					y_to_go -= y_move * abs(y_move_direction);
					//x_to_go -= z_to_go / delta_zx * abs(x_move_direction);
					//y_to_go -= z_to_go / delta_zy * abs(y_move_direction);
					z_to_go = VOXEL_THICKNESS;
					voxel_z -= z_move_direction;
					if( x_to_go == 0 )
					{
						voxel_x += x_move_direction;
						x_to_go = VOXEL_WIDTH;
					}
					if(	y_to_go == 0 )
					{
						voxel_y -= y_move_direction;
						y_to_go = VOXEL_HEIGHT;
					}
				}
				//If Next Voxel Edge is in x or xy Diagonal
				else if( x_extension <= y_extension )
				{
					//printf(" x_extension <= y_extension \n");
					x_move = x_to_go;
					y_move = delta_yx * x_to_go;
					z_move = delta_zx * x_to_go;
					x_to_go = VOXEL_WIDTH;
					y_to_go -= y_move * abs(y_move_direction);
					z_to_go -= z_move * abs(z_move_direction);				
					//y_to_go -= delta_yx * x_to_go * abs(y_move_direction);
					//z_to_go -= delta_zx * x_to_go * abs(z_move_direction);
					//x_to_go = VOXEL_WIDTH;
					voxel_x += x_move_direction;
					if( y_to_go == 0 )
					{
						y_to_go = VOXEL_HEIGHT;
						voxel_y -= y_move_direction;
					}
				}
				// Else Next Voxel Edge is in y
				else
				{
					//printf(" y_extension < x_extension \n");
					x_move = y_to_go / delta_yx;
					y_move = y_to_go;
					z_move = delta_zy * y_to_go;
					x_to_go -= x_move * abs(x_move_direction);
					y_to_go = VOXEL_HEIGHT;
					z_to_go -= z_move * abs(z_move_direction);
					//x_to_go -= y_to_go / delta_yx * abs(x_move_direction);
					//z_to_go -= delta_zy * y_to_go * abs(z_move_direction);
					//y_to_go = VOXEL_HEIGHT;
					voxel_y -= y_move_direction;
				}
				//x += x_move_direction * x_move;
				//y += y_move_direction * y_move;
				//z += z_move_direction * z_move;				
				voxel = voxel_x + voxel_y * COLUMNS + voxel_z * COLUMNS * ROWS;
				end_walk = ( voxel == voxel_out ) || ( voxel_x >= COLUMNS ) || ( voxel_y >= ROWS ) || ( voxel_z >= SLICES );
				if( !end_walk )
					atomicAdd( &MSC_counts[voxel], 1 );
			}// end !end_walk 
		}
		else
		{
			//printf("z_exit[i] == z_entry[i]\n");
			while( !end_walk )
			{
				// Change in x for Move to Voxel Edge in y
				y_extension = y_to_go / delta_yx;
				//If Next Voxel Edge is in x or xy Diagonal
				if( x_to_go <= y_extension )
				{
					//printf(" x_to_go <= y_extension \n");
					x_move = x_to_go;
					y_move = delta_yx * x_to_go;	
					x_to_go = VOXEL_WIDTH;
					//y_to_go -= (delta_yx * x_to_go) * abs(y_move_direction);
					y_to_go -= y_move * abs(y_move_direction);
					voxel_x += x_move_direction;
					if( y_to_go == 0 )
					{
						y_to_go = VOXEL_HEIGHT;
						voxel_y -= y_move_direction;
					}
				}
				// Else Next Voxel Edge is in y
				else
				{
					//printf(" y_extension < x_extension \n");
					x_move = y_to_go / delta_yx;
					y_move = y_to_go;
					x_to_go -= x_move * abs(x_move_direction);
					//x_to_go -= y_to_go / delta_yx * abs(x_move_direction);
					y_to_go = VOXEL_HEIGHT;
					voxel_y -= y_move_direction;
				}
				//x += x_move_direction * x_move;
				//y += y_move_direction * y_move;
				voxel = voxel_x + voxel_y * COLUMNS + voxel_z * COLUMNS * ROWS;		
				end_walk = ( voxel == voxel_out ) || ( voxel_x >= COLUMNS ) || ( voxel_y >= ROWS ) || ( voxel_z >= SLICES );
				if( !end_walk )
					atomicAdd( &MSC_counts[voxel], 1 );
			}// end: while( !end_walk )
			//printf("i = %d", i );
		}//end: else: z_entry[i] != z_exit[i] => z_entry[i] == z_exit[i]
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
/****************************************************************************************************************************************************************/
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
		/********************************* Voxel Walk Parameters ************************************/
		/********************************************************************************************/
		int x_move_direction, y_move_direction, z_move_direction;
		double delta_yx, delta_zx, delta_zy;
		double x_move = 0, y_move = 0, z_move = 0;
		/********************************************************************************************/
		/**************************** Status Tracking Information ***********************************/
		/********************************************************************************************/
		//double x = x_entry[i], y = y_entry[i], z = z_entry[i];
		double x_to_go, y_to_go, z_to_go;		
		double x_extension, y_extension;	
		int voxel_x, voxel_y, voxel_z;
		//int voxel_x_out, voxel_y_out, voxel_z_out; 
		int voxel, voxel_out; 
		bool end_walk;
		/********************************************************************************************/
		/******************** Initial Conditions and Movement Characteristics ***********************/
		/********************************************************************************************/
		x_move_direction = ( x_entry[i] <= x_exit[i] ) - ( x_entry[i] >= x_exit[i] );
		y_move_direction = ( y_entry[i] <= y_exit[i] ) - ( y_entry[i] >= y_exit[i] );
		z_move_direction = ( z_entry[i] <= z_exit[i] ) - ( z_entry[i] >= z_exit[i] );		
		x_to_go = x_remaining_GPU( x_entry[i], x_move_direction, voxel_x );
		y_to_go = y_remaining_GPU( y_entry[i], -y_move_direction, voxel_y );
		z_to_go = z_remaining_GPU( z_entry[i], -z_move_direction, voxel_z );
		voxel = voxel_x + voxel_y * COLUMNS + voxel_z * COLUMNS * ROWS;
		voxel_out = position_2_voxel_GPU( x_exit[i], y_exit[i], z_exit[i] ); 
		/********************************************************************************************/
		/***************************** Path and Walk Information ************************************/
		/********************************************************************************************/
		// Lengths/Distances as x is Incremented One Voxel
		delta_yx = abs( y_exit[i] - y_entry[i] ) / abs( x_exit[i] - x_entry[i] );
		delta_zx = abs( z_exit[i] - z_entry[i] ) / abs( x_exit[i] - x_entry[i] );
		delta_zy = abs( z_exit[i] - z_entry[i] ) / abs( y_exit[i] - y_entry[i] );
		/********************************************************************************************/
		/************************* Initialize and Check Exit Conditions *****************************/
		/********************************************************************************************/		
		end_walk = ( voxel == voxel_out ) || ( voxel_x >= COLUMNS ) || ( voxel_y >= ROWS ) || ( voxel_z >= SLICES );
		if( !end_walk )
			atomicAdd( &SM_counts[voxel], 1 );
		/********************************************************************************************/
		/*********************************** Voxel Walk Routine *************************************/
		/********************************************************************************************/
		if( z_move_direction != 0 )
		{
			//printf("z_exit[i] != z_entry[i]\n");
			while( !end_walk )
			{
				// Change in z for Move to Voxel Edge in x and y
				x_extension = delta_zx * x_to_go;
				y_extension = delta_zy * y_to_go;
				if( (z_to_go <= x_extension  ) && (z_to_go <= y_extension) )
				{
					//printf("z_to_go <= x_extension && z_to_go <= y_extension\n");
					x_move = z_to_go / delta_zx;
					y_move = z_to_go / delta_zy;
					z_move = z_to_go;
					x_to_go -= x_move * abs(x_move_direction);
					y_to_go -= y_move * abs(y_move_direction);
					//x_to_go -= z_to_go / delta_zx * abs(x_move_direction);
					//y_to_go -= z_to_go / delta_zy * abs(y_move_direction);
					z_to_go = VOXEL_THICKNESS;
					voxel_z -= z_move_direction;
					if( x_to_go == 0 )
					{
						voxel_x += x_move_direction;
						x_to_go = VOXEL_WIDTH;
					}
					if(	y_to_go == 0 )
					{
						voxel_y -= y_move_direction;
						y_to_go = VOXEL_HEIGHT;
					}
				}
				//If Next Voxel Edge is in x or xy Diagonal
				else if( x_extension <= y_extension )
				{
					//printf(" x_extension <= y_extension \n");
					x_move = x_to_go;
					y_move = delta_yx * x_to_go;
					z_move = delta_zx * x_to_go;
					x_to_go = VOXEL_WIDTH;
					y_to_go -= y_move * abs(y_move_direction);
					z_to_go -= z_move * abs(z_move_direction);				
					//y_to_go -= delta_yx * x_to_go * abs(y_move_direction);
					//z_to_go -= delta_zx * x_to_go * abs(z_move_direction);
					//x_to_go = VOXEL_WIDTH;
					voxel_x += x_move_direction;
					if( y_to_go == 0 )
					{
						y_to_go = VOXEL_HEIGHT;
						voxel_y -= y_move_direction;
					}
				}
				// Else Next Voxel Edge is in y
				else
				{
					//printf(" y_extension < x_extension \n");
					x_move = y_to_go / delta_yx;
					y_move = y_to_go;
					z_move = delta_zy * y_to_go;
					x_to_go -= x_move * abs(x_move_direction);
					y_to_go = VOXEL_HEIGHT;
					z_to_go -= z_move * abs(z_move_direction);
					//x_to_go -= y_to_go / delta_yx * abs(x_move_direction);
					//z_to_go -= delta_zy * y_to_go * abs(z_move_direction);
					//y_to_go = VOXEL_HEIGHT;
					voxel_y -= y_move_direction;
				}
				//x += x_move_direction * x_move;
				//y += y_move_direction * y_move;
				//z += z_move_direction * z_move;				
				voxel = voxel_x + voxel_y * COLUMNS + voxel_z * COLUMNS * ROWS;
				end_walk = ( voxel == voxel_out ) || ( voxel_x >= COLUMNS ) || ( voxel_y >= ROWS ) || ( voxel_z >= SLICES );
				if( !end_walk )
					atomicAdd( &SM_counts[voxel], 1 );
			}// end !end_walk 
		}
		else
		{
			//printf("z_exit[i] == z_entry[i]\n");
			while( !end_walk )
			{
				// Change in x for Move to Voxel Edge in y
				y_extension = y_to_go / delta_yx;
				//If Next Voxel Edge is in x or xy Diagonal
				if( x_to_go <= y_extension )
				{
					//printf(" x_to_go <= y_extension \n");
					x_move = x_to_go;
					y_move = delta_yx * x_to_go;	
					x_to_go = VOXEL_WIDTH;
					//y_to_go -= (delta_yx * x_to_go) * abs(y_move_direction);
					y_to_go -= y_move * abs(y_move_direction);
					voxel_x += x_move_direction;
					if( y_to_go == 0 )
					{
						y_to_go = VOXEL_HEIGHT;
						voxel_y -= y_move_direction;
					}
				}
				// Else Next Voxel Edge is in y
				else
				{
					//printf(" y_extension < x_extension \n");
					x_move = y_to_go / delta_yx;
					y_move = y_to_go;
					x_to_go -= x_move * abs(x_move_direction);
					//x_to_go -= y_to_go / delta_yx * abs(x_move_direction);
					y_to_go = VOXEL_HEIGHT;
					voxel_y -= y_move_direction;
				}
				//x += x_move_direction * x_move;
				//y += y_move_direction * y_move;
				voxel = voxel_x + voxel_y * COLUMNS + voxel_z * COLUMNS * ROWS;		
				end_walk = ( voxel == voxel_out ) || ( voxel_x >= COLUMNS ) || ( voxel_y >= ROWS ) || ( voxel_z >= SLICES );
				if( !end_walk )
					atomicAdd( &SM_counts[voxel], 1 );
			}// end: while( !end_walk )
			//printf("i = %d", i );
		}//end: else: z_entry[i] != z_exit[i] => z_entry[i] == z_exit[i]
	}// end: if( (i < num_histories) && !missed_recon_volume[i] && (WEPL[i] <= MSC_THRESHOLD) )
}
void SM_edge_detection()
{
	puts("Performing edge-detection on SM_counts...");	

	/*if( WRITE_SM_COUNTS )
	{
		cudaMemcpy(SM_counts_h,  SM_counts_d,	 SIZE_IMAGE_INT,   cudaMemcpyDeviceToHost);
		array_2_disk("SM_counts", OUTPUT_DIRECTORY, OUTPUT_FOLDER, SM_counts_h, COLUMNS, ROWS, SLICES, VOXELS, false );
	}*/

	int* SM_differences_h = (int*) calloc( VOXELS, sizeof(int) );
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
		array_2_disk("x_SM", OUTPUT_DIRECTORY, OUTPUT_FOLDER, SM_counts_h, COLUMNS, ROWS, SLICES, VOXELS, true );
	if( MLP_HULL != SM_HULL)
		free(SM_counts_h);	*/
}
__global__ void SM_edge_detection_GPU( int* SM_counts, int* SM_threshold )
{
	int row = blockIdx.y, column = blockIdx.x, slice = threadIdx.x;
	float x = ( column - COLUMNS/2 + 0.5 ) * VOXEL_WIDTH;
	float y = ( ROWS/2 - row - 0.5 ) * VOXEL_HEIGHT;
	int voxel = column + row * COLUMNS + slice * COLUMNS * ROWS;
	if( voxel < VOXELS )
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
	array_2_disk("SM_counts", OUTPUT_DIRECTORY, OUTPUT_FOLDER, SM_counts_h, COLUMNS, ROWS, SLICES, VOXELS, false );

	int* SM_differences_h = (int*) calloc( VOXELS, sizeof(int) );
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
		array_2_disk("x_SM", OUTPUT_DIRECTORY, OUTPUT_FOLDER, SM_counts_h, COLUMNS, ROWS, SLICES, VOXELS, true );
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
	if( voxel < VOXELS )
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
		SC_hull_h = (bool*) calloc( VOXELS, sizeof(bool) );
		cudaMemcpy(SC_hull_h,  SC_hull_d, SIZE_IMAGE_BOOL, cudaMemcpyDeviceToHost);
		if( WRITE_SC_HULL )
		{
			puts("Writing SC hull to disk...");
			array_2_disk("x_SC", OUTPUT_DIRECTORY, OUTPUT_FOLDER, SC_hull_h, COLUMNS, ROWS, SLICES, VOXELS, true );
		}
		if( MLP_HULL != SC_HULL )
		{
			free( SC_hull_h );
			cudaFree(SC_hull_d);
		}
	}
	if( MSC_ON )
	{
		MSC_counts_h = (int*) calloc( VOXELS, sizeof(int) );
		if( WRITE_MSC_COUNTS )
		{		
			puts("Writing MSC counts to disk...");		
			cudaMemcpy(MSC_counts_h,  MSC_counts_d, SIZE_IMAGE_INT, cudaMemcpyDeviceToHost);
			array_2_disk("MSC_counts_h", OUTPUT_DIRECTORY, OUTPUT_FOLDER, MSC_counts_h, COLUMNS, ROWS, SLICES, VOXELS, true );	
		}
		if( WRITE_MSC_HULL || (MLP_HULL == MSC_HULL) )
		{
			MSC_edge_detection();
			cudaMemcpy(MSC_counts_h,  MSC_counts_d, SIZE_IMAGE_INT, cudaMemcpyDeviceToHost);
			if( WRITE_MSC_HULL )
			{
				puts("Writing MSC hull to disk...");		
				array_2_disk("x_MSC", OUTPUT_DIRECTORY, OUTPUT_FOLDER, MSC_counts_h, COLUMNS, ROWS, SLICES, VOXELS, true );	
			}
		}
		if( MLP_HULL != MSC_HULL )
			free( MSC_counts_h );
		cudaFree(MSC_counts_d);
	}
	if( SM_ON )
	{
		SM_counts_h = (int*) calloc( VOXELS, sizeof(int) );
		if( WRITE_SM_COUNTS )
		{		
			puts("Writing SM counts to disk...");
			cudaMemcpy(SM_counts_h,  SM_counts_d, SIZE_IMAGE_INT, cudaMemcpyDeviceToHost);
			array_2_disk("SM_counts_h", OUTPUT_DIRECTORY, OUTPUT_FOLDER, SM_counts_h, COLUMNS, ROWS, SLICES, VOXELS, true );	
		}
		if( WRITE_SM_HULL || (MLP_HULL == SM_HULL) )
		{
			SM_edge_detection();
			cudaMemcpy(SM_counts_h,  SM_counts_d, SIZE_IMAGE_INT, cudaMemcpyDeviceToHost);
			if( WRITE_SM_HULL )
			{
				puts("Writing SM hull to disk...");		
				array_2_disk("x_SM", OUTPUT_DIRECTORY, OUTPUT_FOLDER, SM_counts_h, COLUMNS, ROWS, SLICES, VOXELS, true );	
			}
		}
		if( MLP_HULL != SM_HULL )
			free( SM_counts_h );
		cudaFree(SM_counts_d);
	}
}
/****************************************************************************************************************************************************************/
template<typename H, typename D> void averaging_filter( H*& image_h, D*& image_d )
{
	bool is_hull = ( typeid(bool) == typeid(D) );
	D* new_value_d;
	int new_value_size = VOXELS * sizeof(D);
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

	x_hull_h = (bool*) calloc( VOXELS, sizeof(bool) );
	switch( MLP_HULL )
	{
		case SC_HULL  : x_hull_h = SC_hull_h;																						break;
		case MSC_HULL : std::transform( MSC_counts_h, MSC_counts_h + VOXELS, MSC_counts_h, x_hull_h, std::logical_or<int> () );		break;
		case SM_HULL  : std::transform( SM_counts_h,  SM_counts_h + VOXELS,  SM_counts_h,  x_hull_h, std::logical_or<int> () );		break;
		case FBP_HULL : x_hull_h = FBP_hull_h;								
	}
	if( WRITE_X_HULL )
	{
		puts("Writing selected hull to disk...");
		array_2_disk("x_hull", OUTPUT_DIRECTORY, OUTPUT_FOLDER, x_hull_h, COLUMNS, ROWS, SLICES, VOXELS, true );
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
			array_2_disk( "x_hull_filtered", OUTPUT_DIRECTORY, OUTPUT_FOLDER, x_hull_h, COLUMNS, ROWS, SLICES, VOXELS, true );
		}
	}
	puts("Hull selection complete.");
}
/****************************************************************************************************************************************************************/
/*********************************************************************** MLP ********************************************************************************/
/****************************************************************************************************************************************************************/
void create_MLP_test_image()
{
	double x, y;
	//Create space carve object, init to zeros
	MLP_test_image_h = (int*)calloc( MLP_IMAGE_VOXELS, sizeof(int));

	for( int slice = 0; slice < MLP_IMAGE_SLICES; slice++ )
	{
		for( int row = 0; row < MLP_IMAGE_ROWS; row++ )
		{
			for( int column = 0; column < MLP_IMAGE_COLUMNS; column++ )
			{
				x = ( column - MLP_IMAGE_COLUMNS/2 + 0.5) * MLP_IMAGE_VOXEL_WIDTH;
				y = ( MLP_IMAGE_ROWS/2 - row - 0.5 ) * MLP_IMAGE_VOXEL_HEIGHT;
				if( pow( x, 2 ) + pow( y, 2 ) <= pow( double(MLP_IMAGE_RECON_CYL_RADIUS), 2) )
					MLP_test_image_h[slice * MLP_IMAGE_COLUMNS * MLP_IMAGE_ROWS + row * MLP_IMAGE_COLUMNS + column] = 1;
				if( pow( x / MLP_PHANTOM_A, 2 ) + pow( y / MLP_PHANTOM_B, 2 ) <= 1 )
					MLP_test_image_h[slice * MLP_IMAGE_COLUMNS * MLP_IMAGE_ROWS + row * MLP_IMAGE_COLUMNS + column] = 8;
			}
		}
	}
}
void MLP_entry_exit( int*& image, bool entry, float x_start, float y_start, float z_start, float xy_angle, float xz_angle, float x_object, float y_object, float z_object )
{
//	/********************************************************************************************/
//	/********************************* Voxel Walk Parameters ************************************/
//	/********************************************************************************************/
//	int x_move_direction, y_move_direction, z_move_direction;
//	int x_voxel_step, y_voxel_step, z_voxel_step;
//	float delta_x, delta_y, delta_z;
//	float x_move, y_move, z_move;
//	/********************************************************************************************/
//	/**************************** Status Tracking Information ***********************************/
//	/********************************************************************************************/
//	float x, y, z;
//	float x_inside, y_inside, z_inside;
//	float x_to_go, y_to_go, z_to_go;		
//	float x_extension, y_extension;	
//	float voxel_x, voxel_y, voxel_z;
//	float voxel_x_out, voxel_y_out, voxel_z_out, voxel_out; 
//	int voxel;
//	bool outside_image, end_walk;
//	/********************************************************************************************/
//	/************************** Initial and Boundary Conditions *********************************/
//	/********************************************************************************************/
//	// Initial Distance Into Voxel
//	x_inside = modf( ( x_start + RECON_CYL_RADIUS ) /VOXEL_WIDTH, &voxel_x)*VOXEL_WIDTH;	
//	y_inside = modf( ( RECON_CYL_RADIUS - y_entry ) /VOXEL_HEIGHT, &voxel_y)*VOXEL_HEIGHT;
//	z_inside = modf( ( RECON_CYL_HEIGHT/2 - z_entry ) /VOXEL_THICKNESS, &voxel_z)*VOXEL_THICKNESS;
//
//	voxel = int(voxel_x + voxel_y * COLUMNS + voxel_z * COLUMNS * ROWS);
//	voxel_x_out = int( ( x_exit + RECON_CYL_RADIUS ) /VOXEL_WIDTH );
//	voxel_y_out = int( ( RECON_CYL_RADIUS - y_exit ) /VOXEL_HEIGHT );
//	voxel_z_out = int( ( RECON_CYL_HEIGHT/2 - z_exit ) /VOXEL_THICKNESS );
//	voxel_out = int(voxel_x_out + voxel_y_out * COLUMNS + voxel_z_out * COLUMNS * ROWS);
//	/********************************************************************************************/
//	/***************************** Path and Walk Information ************************************/
//	/********************************************************************************************/
//	// Lengths/Distances as x is Incremented One Voxel
//	delta_x = VOXEL_WIDTH;
//	delta_y = abs( (y_exit - y_entry)/(x_exit - x_start) * VOXEL_WIDTH );
//	delta_z = abs( (z_exit - z_entry)/(x_exit - x_start) * VOXEL_WIDTH );
//	// Overwrite NaN if Divisors on delta_i Calculations Above 
//	if( x_start == x_exit )
//	{
//		delta_x = abs( (x_exit - x_entry)/(y_exit - y_entry) * VOXEL_HEIGHT );
//		delta_y = VOXEL_HEIGHT;
//		delta_z = abs( (z_exit - z_entry)/(y_exit - y_entry) * VOXEL_HEIGHT );
//		if( y_entry == y_exit )
//		{
//			delta_x = abs( (x_exit - x_entry)/(z_exit - z_entry) * VOXEL_THICKNESS );
//			delta_y = abs( (y_exit - y_entry)/(z_exit - z_entry) * VOXEL_THICKNESS );;
//			delta_z = VOXEL_THICKNESS;
//		}
//	}
//	x_move = 0, y_move = 0, z_move = 0;
//	/*x_move_direction = ( x_entry <= x_exit ) - ( x_entry > x_exit );
//	y_move_direction = ( y_entry <= y_exit ) - ( y_entry > y_exit );
//	z_move_direction = ( z_entry <= z_exit ) - ( z_entry > z_exit );*/
//	x_move_direction = ( cos(xy_angle) >= 0 ) - ( cos(xy_angle) < 0 );
//	y_move_direction = ( sin(xy_angle) >= 0 ) - ( sin(xy_angle) < 0 );
//	z_move_direction = ( sin(xz_angle) >= 0 ) - ( sin(xz_angle) < 0 );
//	x_voxel_step = x_move_direction;
//	y_voxel_step = -y_move_direction;
//	z_voxel_step = -z_move_direction;
//	/********************************************************************************************/
//	/**************************** Status Tracking Information ***********************************/
//	/********************************************************************************************/
//	x = x_entry, y = y_entry, z = z_entry;
//	x_to_go = ( x_voxel_step > 0 ) * (VOXEL_WIDTH - x_inside) + ( x_voxel_step <= 0 ) * x_inside;
//	y_to_go = ( y_voxel_step > 0 ) * (VOXEL_HEIGHT - y_inside) + ( y_voxel_step <= 0 ) * y_inside;
//	z_to_go = ( z_voxel_step > 0 ) * (VOXEL_THICKNESS - z_inside) + ( z_voxel_step <= 0 ) * z_inside;
//			
//	outside_image = ( voxel_x >= COLUMNS ) || ( voxel_y >= ROWS ) || ( voxel_z >= SLICES );
//	if( !outside_image )
//		image[voxel] = 0;
//	end_walk = ( voxel == voxel_out ) || outside_image;
//	//fgets(user_response, sizeof(user_response), stdin);
//	/********************************************************************************************/
//	/*********************************** Voxel Walk Routine *************************************/
//	/********************************************************************************************/
//	if( z_entry != z_exit )
//	{
//		while( !end_walk )
//		{
//			// Change in z for Move to Voxel Edge in x and y
//			x_extension = delta_z/delta_x * x_to_go;
//			y_extension = delta_z/delta_y * y_to_go;
//			if( z_to_go <= x_extension && z_to_go <= y_extension )
//			{
//				//printf("z_to_go <= x_extension && z_to_go <= y_extension\n");
//				x_move = delta_x / delta_z * z_to_go;
//				y_move = delta_y / delta_z * z_to_go;
//				z_move = z_to_go;
//				x_to_go -= x_move;
//				y_to_go -= y_move;
//				z_to_go = VOXEL_THICKNESS;
//				voxel_z += z_voxel_step;
//				if( x_to_go == 0 )
//				{
//					voxel_x += x_voxel_step;
//					x_to_go = VOXEL_WIDTH;
//				}
//				if(	y_to_go == 0 )
//				{
//					voxel_y += y_voxel_step;
//					y_to_go = VOXEL_HEIGHT;
//				}
//			}
//			//If Next Voxel Edge is in x or xy Diagonal
//			else if( x_extension <= y_extension )
//			{
//				//printf(" x_extension <= y_extension \n");
//				x_move = x_to_go;
//				y_move = delta_y / delta_x * x_to_go;
//				z_move = delta_z / delta_x * x_to_go;
//				x_to_go = VOXEL_WIDTH;
//				y_to_go -= y_move;
//				z_to_go -= z_move;
//				voxel_x += x_voxel_step;
//				if( y_to_go == 0 )
//				{
//					y_to_go = VOXEL_HEIGHT;
//					voxel_y += y_voxel_step;
//				}
//			}
//			// Else Next Voxel Edge is in y
//			else
//			{
//				//printf(" y_extension < x_extension \n");
//				x_move = delta_x / delta_y * y_to_go;
//				y_move = y_to_go;
//				z_move = delta_z / delta_y * y_to_go;
//				x_to_go -= x_move;
//				y_to_go = VOXEL_HEIGHT;
//				z_to_go -= z_move;
//				voxel_y += y_voxel_step;
//			}
//			x += x_move_direction * x_move;
//			y += y_move_direction * y_move;
//			z += z_move_direction * z_move;				
//			//fgets(user_response, sizeof(user_response), stdin);
//			voxel = int(voxel_x + voxel_y * COLUMNS + voxel_z * COLUMNS * ROWS);
//			outside_image = ( voxel_x >= COLUMNS ) || ( voxel_y >= ROWS ) || ( voxel_z >= SLICES );
//			if( !outside_image )
//				image[voxel] = 0;
//			end_walk = ( voxel == voxel_out ) || outside_image;
//		}
//	}
//	else
//	{
//		//printf("z_exit == z_entry\n");
//		while( !end_walk )
//		{
//			// Change in x for Move to Voxel Edge in y
//			y_extension = delta_x/delta_y * y_to_go;
//			//If Next Voxel Edge is in x or xy Diagonal
//			if( x_to_go <= y_extension )
//			{
//				//printf(" x_to_go <= y_extension \n");
//				x_move = x_to_go;
//				y_move = delta_y / delta_x * x_to_go;				
//				x_to_go = VOXEL_WIDTH;
//				y_to_go -= y_move;
//				voxel_x += x_voxel_step;
//				if( y_to_go == 0 )
//				{
//					y_to_go = VOXEL_HEIGHT;
//					voxel_y += y_voxel_step;
//				}
//			}
//			// Else Next Voxel Edge is in y
//			else
//			{
//				//printf(" y_extension < x_extension \n");
//				x_move = delta_x / delta_y * y_to_go;
//				y_move = y_to_go;
//				x_to_go -= x_move;
//				y_to_go = VOXEL_HEIGHT;
//				voxel_y += y_voxel_step;
//			}
//			x += x_move_direction * x_move;
//			y += y_move_direction * y_move;
//			voxel = int(voxel_x + voxel_y * COLUMNS + voxel_z * COLUMNS * ROWS);
//			outside_image = ( voxel_x >= COLUMNS ) || ( voxel_y >= ROWS ) || ( voxel_z >= SLICES );
//			if( !outside_image )
//				image[voxel] = 0;
//			end_walk = ( voxel == voxel_out ) || outside_image;
//			//fgets(user_response, sizeof(user_response), stdin);
//		}// end: while( !end_walk )
//	}//end: else: z_entry_h != z_exit_h => z_entry_h == z_exit_h
}
void MLP_test()
{
	char user_response[20];
	double x_entry = -3.0;
	double y_entry = -sqrt( pow(MLP_IMAGE_RECON_CYL_RADIUS, 2) - pow(x_entry,2) );
	double z_entry = 0.0;
	double x_exit = 2.5;
	double y_exit = sqrt( pow(MLP_IMAGE_RECON_CYL_RADIUS, 2) - pow(x_exit,2) );
	double z_exit = 0.0;
	double xy_entry_angle = 25 * PI/180, xz_entry_angle = 0.0;
	double xy_exit_angle = 45* PI/180, xz_exit_angle = 0.0;
	double x_in_object, y_in_object, z_in_object;
	double u_in_object, t_in_object, v_in_object;
	double x_out_object, y_out_object, z_out_object;
	double u_out_object, t_out_object, v_out_object;

	/********************************************************************************************/
	/**************************** Status Tracking Information ***********************************/
	/********************************************************************************************/
	double voxel_x, voxel_y, voxel_z;
	int voxel;
	int x_move_direction, y_move_direction, z_move_direction;
	int x_voxel_step, y_voxel_step, z_voxel_step;
	double x, y, z;
	double x_inside, y_inside, z_inside;
	double x_to_go, y_to_go, z_to_go;
	double delta_x, delta_y, delta_z;
	double x_extension, y_extension;
	double x_move, y_move, z_move;
	bool end_walk, outside_image;
	bool entered_object = false, exited_object = false;

/********************************************************************************************************/
/******************** Determine if and Where the Proton Enters the Actual Object ************************/
/********************************************************************************************************/

	/********************************************************************************************/
	/************************** Initial and Boundary Conditions *********************************/
	/********************************************************************************************/

	// Initial Distance Into Voxel
	x_inside = modf( ( x_entry + MLP_IMAGE_WIDTH/2 ) / MLP_IMAGE_VOXEL_WIDTH, &voxel_x ) * MLP_IMAGE_VOXEL_WIDTH;	
	y_inside = modf( ( MLP_IMAGE_HEIGHT/2 - y_entry ) / MLP_IMAGE_VOXEL_HEIGHT, &voxel_y ) * MLP_IMAGE_VOXEL_HEIGHT;
	z_inside = modf( ( MLP_IMAGE_THICKNESS/2 - z_entry ) / MLP_IMAGE_VOXEL_THICKNESS, &voxel_z ) * MLP_IMAGE_VOXEL_THICKNESS;
	//printf("voxel_x = %3f \nvoxel_y = %3f \nvoxel_z = %3f\n", voxel_x, voxel_y, voxel_z);
	//printf("x_inside = %3f y_inside = %3f z_inside = %3f\n", x_inside, y_inside, z_inside);
	
	voxel = int(voxel_x + voxel_y * MLP_IMAGE_COLUMNS + voxel_z * MLP_IMAGE_COLUMNS * MLP_IMAGE_ROWS);
	//printf("voxel = %d \n", voxel );
	/********************************************************************************************/
	/***************************** Path and Walk Information ************************************/
	/********************************************************************************************/

	// Lengths/Distances as x is Incremented One Voxel
	delta_x = MLP_IMAGE_VOXEL_WIDTH;
	delta_y = tan( xy_entry_angle ) * MLP_IMAGE_VOXEL_WIDTH;
	delta_z = tan( xz_entry_angle ) * MLP_IMAGE_VOXEL_WIDTH;
	if( x_entry == x_exit )
	{
		delta_x = 0;
		delta_y = MLP_IMAGE_VOXEL_HEIGHT;
		delta_z = tan(xz_entry_angle) / tan(xy_entry_angle) * MLP_IMAGE_VOXEL_HEIGHT;
		if( y_entry == y_exit )
		{
			delta_x = 0;
			delta_y = 0;
			delta_z = MLP_IMAGE_VOXEL_THICKNESS;
		}
	}
	//printf("delta_x = %3f delta_y = %3f delta_z = %3f\n", delta_x, delta_y, delta_z );
	x_move = 0, y_move = 0, z_move = 0;
	/*x_move_direction = ( x_entry <= x_exit ) - ( x_entry > x_exit );
	y_move_direction = ( y_entry <= y_exit ) - ( y_entry > y_exit );
	z_move_direction = ( z_entry <= z_exit ) - ( z_entry > z_exit );*/
	x_move_direction = ( cos(xy_entry_angle) >= 0 ) - ( cos(xy_entry_angle) < 0 );
	y_move_direction = ( sin(xy_entry_angle) >= 0 ) - ( sin(xy_entry_angle) < 0 );
	z_move_direction = ( sin(xy_entry_angle) >= 0 ) - ( sin(xy_entry_angle) < 0 );
	x_voxel_step = x_move_direction;
	y_voxel_step = -y_move_direction;
	z_voxel_step = -z_move_direction;
	/********************************************************************************************/
	/**************************** Status Tracking Information ***********************************/
	/********************************************************************************************/
	x = x_entry, y = y_entry, z = z_entry;
	x_to_go = ( x_voxel_step > 0 ) * ( MLP_IMAGE_VOXEL_WIDTH - x_inside ) + ( x_voxel_step <= 0 ) * x_inside;
	y_to_go = ( y_voxel_step > 0 ) * ( MLP_IMAGE_VOXEL_HEIGHT - y_inside ) + ( y_voxel_step <= 0 ) * y_inside;
	z_to_go = ( z_voxel_step > 0 ) * ( MLP_IMAGE_VOXEL_THICKNESS - z_inside ) + ( z_voxel_step <= 0 ) * z_inside;
	
	//printf("initial values:\n\tx_to_go = %3f\n\ty_to_go = %3f\n\tz_to_go = %3f\n", x_to_go, y_to_go, z_to_go);
	
	outside_image = (voxel_x >= MLP_IMAGE_COLUMNS ) || (voxel_y >= MLP_IMAGE_ROWS ) || (voxel_z >= MLP_IMAGE_SLICES );
	if( !outside_image )
	{
		entered_object = MLP_test_image_h[voxel] == 8;
		MLP_test_image_h[voxel] = 4;
	}
	end_walk = entered_object || outside_image;
	///********************************************************************************************/
	///*********************************** Voxel Walk Routine *************************************/
	///********************************************************************************************/
	if( z_entry != z_exit )
	{
		while( !end_walk )
		{
			// Change in z for Move to Voxel Edge in x and y
			x_extension = delta_z/delta_x * x_to_go;
			y_extension = delta_z/delta_y * y_to_go;
			if( z_to_go <= x_extension && z_to_go <= y_extension )
			{
				//printf("z_to_go <= x_extension && z_to_go <= y_extension\n");
				x_move = delta_x / delta_z * z_to_go;
				y_move = delta_y / delta_z * z_to_go;
				z_move = z_to_go;
				x_to_go -= x_move;
				y_to_go -= y_move;
				z_to_go = MLP_IMAGE_VOXEL_THICKNESS;
				voxel_z += z_voxel_step;
				if( x_to_go == 0 )
				{
					voxel_x += x_voxel_step;
					x_to_go = MLP_IMAGE_VOXEL_WIDTH;
				}
				if(	y_to_go == 0 )
				{
					voxel_y += y_voxel_step;
					y_to_go = MLP_IMAGE_VOXEL_HEIGHT;
				}
			}
			//If Next Voxel Edge is in x or xy Diagonal
			else if( x_extension <= y_extension )
			{
				//printf(" x_extension <= y_extension \n");
				x_move = x_to_go;
				y_move = delta_y / delta_x * x_to_go;
				z_move = delta_z / delta_x * x_to_go;
				x_to_go = MLP_IMAGE_VOXEL_WIDTH;
				y_to_go -= y_move;
				z_to_go -= z_move;
				voxel_x += x_voxel_step;
				if( y_to_go == 0 )
				{
					y_to_go = MLP_IMAGE_VOXEL_HEIGHT;
					voxel_y += y_voxel_step;
				}
			}
			// Else Next Voxel Edge is in y
			else
			{
				//printf(" y_extension < x_extension \n");
				x_move = delta_x / delta_y * y_to_go;
				y_move = y_to_go;
				z_move = delta_z / delta_y * y_to_go;
				x_to_go -= x_move;
				y_to_go = MLP_IMAGE_VOXEL_HEIGHT;
				z_to_go -= z_move;
				voxel_y += y_voxel_step;
			}
			voxel = int(voxel_x + voxel_y * MLP_IMAGE_COLUMNS + voxel_z * MLP_IMAGE_COLUMNS * MLP_IMAGE_ROWS);
			outside_image = (voxel_x >= MLP_IMAGE_COLUMNS ) || (voxel_y >= MLP_IMAGE_ROWS ) || (voxel_z >= MLP_IMAGE_SLICES );
			if( !outside_image )
			{
				entered_object = MLP_test_image_h[voxel] == 8;
				MLP_test_image_h[voxel] = 4;
			}
			x += x_move_direction * x_move;
			y += y_move_direction * y_move;
			z += z_move_direction * z_move;				
			end_walk = entered_object || outside_image;
		}
	}
	else
	{
		//printf("z_exit == z_entry\n");
		while( !end_walk )
		{
			//printf("beginning of loop\n\n");
			//printf("x = %3f y = %3f z = %3f\n", x, y, z );
			//printf("x_to_go = %3f y_to_go = %3f\n", x_to_go, y_to_go);
			//printf("voxel_x = %3f voxel_y = %3f voxel_z = %3f\n", voxel_x, voxel_y, voxel_z);
			// Change in x for Move to Voxel Edge in y
			y_extension = delta_x/delta_y * y_to_go;
			//printf("y_extension = %3f\n", y_extension);
			//If Next Voxel Edge is in x or xy Diagonal
			if( x_to_go <= y_extension )
			{
				//printf(" x_to_go <= y_extension \n");
				x_move = x_to_go;
				y_move = delta_y / delta_x * x_to_go;				
				x_to_go = MLP_IMAGE_VOXEL_WIDTH;
				y_to_go -= y_move;
				voxel_x += x_voxel_step;
				if( y_to_go == 0 )
				{
					y_to_go = MLP_IMAGE_VOXEL_HEIGHT;
					voxel_y += y_voxel_step;
				}
			}
			// Else Next Voxel Edge is in y
			else
			{
				//printf(" y_extension < x_extension \n");
				x_move = delta_x / delta_y * y_to_go;
				y_move = y_to_go;
				x_to_go -= x_move;
				y_to_go = MLP_IMAGE_VOXEL_HEIGHT;
				voxel_y += y_voxel_step;
			}
			voxel = int(voxel_x + voxel_y * MLP_IMAGE_COLUMNS + voxel_z * MLP_IMAGE_COLUMNS * MLP_IMAGE_ROWS);
			//printf("end of loop\n\n");
			//printf("x_move = %3f y_move = %3f\n", x_move, y_move );
			//printf("x = %3f y = %3f z = %3f\n", x, y, z );
			//printf("x_to_go = %3f y_to_go = %3f\n", x_to_go, y_to_go);
			//printf("voxel_x = %3f voxel_y = %3f voxel_z = %3f\n\n", voxel_x, voxel_y, voxel_z);		
			outside_image = (voxel_x >= MLP_IMAGE_COLUMNS ) || (voxel_y >= MLP_IMAGE_ROWS ) || (voxel_z >= MLP_IMAGE_SLICES );
			if( !outside_image )
			{
				entered_object = MLP_test_image_h[voxel] == 8;
				MLP_test_image_h[voxel] = 4;
			}
			//printf("MLP_IMAGE_WIDTH/2 = %3f\n MLP_IMAGE_HEIGHT/2 = %3f", MLP_IMAGE_WIDTH/2 , MLP_IMAGE_HEIGHT/2 );
			x += x_move_direction * x_move;
			y += y_move_direction * y_move;
			end_walk = entered_object || outside_image;
			//fgets(user_response, sizeof(user_response), stdin);
		}// end: while( !end_walk )
	}//end: else: z_entry != z_exit => z_entry == z_exit
	if( entered_object )
	{
		x_in_object = x;
		y_in_object = y;
		z_in_object = z;
	}
/********************************************************************************************************/
/******************** Determine if and Where the Proton Exited the Actual Object ************************/
/********************************************************************************************************/

	/********************************************************************************************/
	/************************** Initial and Boundary Conditions *********************************/
	/********************************************************************************************/

	// Initial Distance Into Voxel
	x_inside = modf( ( x_exit + MLP_IMAGE_WIDTH/2 ) / MLP_IMAGE_VOXEL_WIDTH, &voxel_x ) * MLP_IMAGE_VOXEL_WIDTH;	
	y_inside = modf( ( MLP_IMAGE_HEIGHT/2 - y_exit ) / MLP_IMAGE_VOXEL_HEIGHT, &voxel_y ) * MLP_IMAGE_VOXEL_HEIGHT;
	z_inside = modf( ( MLP_IMAGE_THICKNESS/2 - z_exit ) / MLP_IMAGE_VOXEL_THICKNESS, &voxel_z ) * MLP_IMAGE_VOXEL_THICKNESS;
	//printf("voxel_x = %3f \nvoxel_y = %3f \nvoxel_z = %3f\n", voxel_x, voxel_y, voxel_z);
	//printf("x_inside = %3f y_inside = %3f z_inside = %3f\n", x_inside, y_inside, z_inside);
	
	voxel = int(voxel_x + voxel_y * MLP_IMAGE_COLUMNS + voxel_z * MLP_IMAGE_COLUMNS * MLP_IMAGE_ROWS);
	//printf("voxel = %d \n", voxel );
	/********************************************************************************************/
	/***************************** Path and Walk Information ************************************/
	/********************************************************************************************/

	// Lengths/Distances as x is Incremented One Voxel
	delta_x = MLP_IMAGE_VOXEL_WIDTH;
	delta_y = tan( xy_exit_angle ) * MLP_IMAGE_VOXEL_WIDTH;
	delta_z = tan( xz_exit_angle ) * MLP_IMAGE_VOXEL_WIDTH;
	if( x_entry == x_exit )
	{
		delta_x = 0;
		delta_y = MLP_IMAGE_VOXEL_HEIGHT;
		delta_z = tan(xz_exit_angle) / tan(xy_exit_angle) * MLP_IMAGE_VOXEL_HEIGHT;
		if( y_entry == y_exit )
		{
			delta_x = 0;
			delta_y = 0;
			delta_z = MLP_IMAGE_VOXEL_THICKNESS;
		}
	}
	//printf("delta_x = %3f delta_y = %3f delta_z = %3f\n", delta_x, delta_y, delta_z );
	x_move = 0, y_move = 0, z_move = 0;
	//x_move_direction = ( x_exit <= x_entry ) - ( x_exit > x_entry );
	//y_move_direction = ( y_exit <= y_entry ) - ( y_exit > y_entry );
	//z_move_direction = ( z_exit <= z_entry ) - ( z_exit > z_entry );
	x_move_direction = ( cos(xy_exit_angle) < 0 ) - ( cos(xy_exit_angle) >= 0 );
	y_move_direction = ( sin(xy_exit_angle) < 0 ) - ( sin(xy_exit_angle) >= 0 );
	z_move_direction = ( sin(xy_exit_angle) < 0 ) - ( sin(xy_exit_angle) >= 0 );
	x_voxel_step = x_move_direction;
	y_voxel_step = -y_move_direction;
	z_voxel_step = -z_move_direction;
	/********************************************************************************************/
	/**************************** Status Tracking Information ***********************************/
	/********************************************************************************************/
	x = x_exit, y = y_exit, z = z_exit;
	x_to_go = ( x_voxel_step > 0 ) * ( MLP_IMAGE_VOXEL_WIDTH - x_inside ) + ( x_voxel_step <= 0 ) * x_inside;
	y_to_go = ( y_voxel_step > 0 ) * ( MLP_IMAGE_VOXEL_HEIGHT - y_inside ) + ( y_voxel_step <= 0 ) * y_inside;
	z_to_go = ( z_voxel_step > 0 ) * ( MLP_IMAGE_VOXEL_THICKNESS - z_inside ) + ( z_voxel_step <= 0 ) * z_inside;
	
	//printf("initial values:\n\tx_to_go = %3f\n\ty_to_go = %3f\n\tz_to_go = %3f\n", x_to_go, y_to_go, z_to_go);
	
	outside_image = (voxel_x >= MLP_IMAGE_COLUMNS ) || (voxel_y >= MLP_IMAGE_ROWS ) || (voxel_z >= MLP_IMAGE_SLICES );
	if( !outside_image )
	{
		exited_object = MLP_test_image_h[voxel] == 8;
		MLP_test_image_h[voxel] = 4;
	}
	end_walk = exited_object || outside_image;
	///********************************************************************************************/
	///*********************************** Voxel Walk Routine *************************************/
	///********************************************************************************************/
	if( z_entry != z_exit )
	{
		//printf("z_entry != z_exit\n");
		while( !end_walk )
		{
			// Change in z for Move to Voxel Edge in x and y
			x_extension = delta_z/delta_x * x_to_go;
			y_extension = delta_z/delta_y * y_to_go;
			if( z_to_go <= x_extension && z_to_go <= y_extension )
			{
				//printf("z_to_go <= x_extension && z_to_go <= y_extension\n");
				x_move = delta_x / delta_z * z_to_go;
				y_move = delta_y / delta_z * z_to_go;
				z_move = z_to_go;
				x_to_go -= x_move;
				y_to_go -= y_move;
				z_to_go = MLP_IMAGE_VOXEL_THICKNESS;
				voxel_z += z_voxel_step;
				if( x_to_go == 0 )
				{
					voxel_x += x_voxel_step;
					x_to_go = MLP_IMAGE_VOXEL_WIDTH;
				}
				if(	y_to_go == 0 )
				{
					voxel_y += y_voxel_step;
					y_to_go = MLP_IMAGE_VOXEL_HEIGHT;
				}
			}
			//If Next Voxel Edge is in x or xy Diagonal
			else if( x_extension <= y_extension )
			{
				//printf(" x_extension <= y_extension \n");
				x_move = x_to_go;
				y_move = delta_y / delta_x * x_to_go;
				z_move = delta_z / delta_x * x_to_go;
				x_to_go = MLP_IMAGE_VOXEL_WIDTH;
				y_to_go -= y_move;
				z_to_go -= z_move;
				voxel_x += x_voxel_step;
				if( y_to_go == 0 )
				{
					y_to_go = MLP_IMAGE_VOXEL_HEIGHT;
					voxel_y += y_voxel_step;
				}
			}
			// Else Next Voxel Edge is in y
			else
			{
				//printf(" y_extension < x_extension \n");
				x_move = delta_x / delta_y * y_to_go;
				y_move = y_to_go;
				z_move = delta_z / delta_y * y_to_go;
				x_to_go -= x_move;
				y_to_go = MLP_IMAGE_VOXEL_HEIGHT;
				z_to_go -= z_move;
				voxel_y += y_voxel_step;
			}
			voxel = int( voxel_x + voxel_y * MLP_IMAGE_COLUMNS + voxel_z * MLP_IMAGE_COLUMNS * MLP_IMAGE_ROWS );
			outside_image = ( voxel_x >= MLP_IMAGE_COLUMNS ) || ( voxel_y >= MLP_IMAGE_ROWS ) || ( voxel_z >= MLP_IMAGE_SLICES );
			if( !outside_image )
			{
				exited_object = MLP_test_image_h[voxel] == 8;
				MLP_test_image_h[voxel] = 4;
			}
			x += x_move_direction * x_move;
			y += y_move_direction * y_move;
			z += z_move_direction * z_move;				
			end_walk = exited_object || outside_image;
		}
	}
	else
	{
		//printf("z_entry == z_exit\n");
		while( !end_walk )
		{
			//printf("beginning of loop\n\n");
			//printf("x = %3f y = %3f z = %3f\n", x, y, z );
			//printf("x_to_go = %3f y_to_go = %3f\n", x_to_go, y_to_go);
			//printf("voxel_x = %3f voxel_y = %3f voxel_z = %3f\n", voxel_x, voxel_y, voxel_z);
			// Change in x for Move to Voxel Edge in y
			y_extension = delta_x/delta_y * y_to_go;
			//printf("y_extension = %3f\n", y_extension);
			//If Next Voxel Edge is in x or xy Diagonal
			if( x_to_go <= y_extension )
			{
				//printf(" x_to_go <= y_extension \n");
				x_move = x_to_go;
				y_move = delta_y / delta_x * x_to_go;				
				x_to_go = MLP_IMAGE_VOXEL_WIDTH;
				y_to_go -= y_move;
				voxel_x += x_voxel_step;
				if( y_to_go == 0 )
				{
					y_to_go = MLP_IMAGE_VOXEL_HEIGHT;
					voxel_y += y_voxel_step;
				}
			}
			// Else Next Voxel Edge is in y
			else
			{
				//printf(" y_extension < x_extension \n");
				x_move = delta_x / delta_y * y_to_go;
				y_move = y_to_go;
				x_to_go -= x_move;
				y_to_go = MLP_IMAGE_VOXEL_HEIGHT;
				voxel_y += y_voxel_step;
			}
			voxel = int(voxel_x + voxel_y * MLP_IMAGE_COLUMNS + voxel_z * MLP_IMAGE_COLUMNS * MLP_IMAGE_ROWS);
			/*printf("end of loop\n\n");
			printf("x_move = %3f y_move = %3f\n", x_move, y_move );
			printf("x = %3f y = %3f z = %3f\n", x, y, z );
			printf("x_to_go = %3f y_to_go = %3f\n", x_to_go, y_to_go);
			printf("voxel_x = %3f voxel_y = %3f voxel_z = %3f\n\n", voxel_x, voxel_y, voxel_z);*/		
			outside_image = (voxel_x >= MLP_IMAGE_COLUMNS ) || (voxel_y >= MLP_IMAGE_ROWS ) || (voxel_z >= MLP_IMAGE_SLICES );
			if( !outside_image )
			{
				exited_object = MLP_test_image_h[voxel] == 8;
				MLP_test_image_h[voxel] = 4;
			}
			//printf("MLP_IMAGE_WIDTH/2 = %3f\n MLP_IMAGE_HEIGHT/2 = %3f",MLP_IMAGE_WIDTH/2 , MLP_IMAGE_HEIGHT/2 );
			x += x_move_direction * x_move;
			y += y_move_direction * y_move;
			end_walk = exited_object || outside_image;
			//fgets(user_response, sizeof(user_response), stdin);
		}// end: while( !end_walk )
	}//end: else: z_exit != z_exit => z_exit == z_exit
	if( exited_object )
	{
		x_out_object = x;
		y_out_object = y;
		z_out_object = z;
	}

	x_inside = modf( ( x_in_object + MLP_IMAGE_WIDTH/2 ) / MLP_IMAGE_VOXEL_WIDTH, &voxel_x ) * MLP_IMAGE_VOXEL_WIDTH;	
	y_inside = modf( ( MLP_IMAGE_HEIGHT/2 - y_in_object ) / MLP_IMAGE_VOXEL_HEIGHT, &voxel_y ) * MLP_IMAGE_VOXEL_HEIGHT;
	z_inside = modf( ( MLP_IMAGE_THICKNESS/2 - z_in_object ) / MLP_IMAGE_VOXEL_THICKNESS, &voxel_z ) * MLP_IMAGE_VOXEL_THICKNESS;

	//x_inside = x_remaining_GPU( x_in_object, 1, voxel_x );
	//y_inside = y_remaining_GPU( y_in_object, 1, voxel_y );
	//z_inside = z_remaining_GPU( z_in_object, 1, voxel_z );
	//printf("voxel_x = %3f \nvoxel_y = %3f \nvoxel_z = %3f\n", voxel_x, voxel_y, voxel_z);
	//printf("x_inside = %3f y_inside = %3f z_inside = %3f\n", x_inside, y_inside, z_inside);	
	voxel = int(voxel_x + voxel_y * MLP_IMAGE_COLUMNS + voxel_z * MLP_IMAGE_COLUMNS * MLP_IMAGE_ROWS);

	int path[1000];
	int path_index = 0;
	double chord_lengths[1000];
	MLP_test_image_h[voxel] = 0;
	path[path_index++] = voxel;

	u_in_object = ( cos( xy_entry_angle ) * x_in_object ) + ( sin( xy_entry_angle ) * y_in_object );
	u_out_object = ( cos( xy_entry_angle ) * x_out_object ) + ( sin( xy_entry_angle ) * y_out_object );
	t_in_object = ( cos( xy_entry_angle ) * y_in_object ) - ( sin( xy_entry_angle ) * x_in_object );
	t_out_object = ( cos( xy_entry_angle ) * y_out_object ) - ( sin( xy_entry_angle ) * x_out_object );
	v_in_object = z_in_object;
	v_out_object = z_out_object;
	
	double T_0[2] = { t_in_object, 0 };
	double T_2[2] = { t_out_object, xy_exit_angle - xy_entry_angle };
	double V_0[2] = { v_in_object, xz_entry_angle };
	double V_2[2] = { v_out_object, xz_exit_angle };
	double u_2 = abs(u_out_object - u_in_object);
	double u_0 = 0, u_1 = MLP_u_step;
	double t_1_previous, v_1_previous;
	double x_1_previous = x, y_1_previous = y, z_1_previous = z;
	int voxel_x_previous = voxel_x;
	int voxel_y_previous = voxel_y;
	int voxel_z_previous = voxel_z;
	int voxel_previous = voxel;
	int voxels_passed;
	double chord_segment;
	double chord_fraction;
	double x_to_edge, y_to_edge, z_to_edge;
	//fgets(user_response, sizeof(user_response), stdin);
	while( u_1 <= u_2 - MLP_u_step )
	{
		double R_0[4] = { 1.0, u_1 - u_0, 0.0 , 1.0}; //a,b,c,d
		double R_0T[4] = { 1.0, 0.0, u_1 - u_0 , 1.0}; //a,c,b,d
		double R_1[4] = { 1.0, u_2 - u_1, 0.0 , 1.0}; //a,b,c,d
		double R_1T[4] = { 1.0, 0.0, u_2 - u_1 , 1.0};  //a,c,b,d
	
		double sigma_1_coefficient = pow( E_0 * ( 1 + 0.038 * log( (u_1 - u_0)/X_0) ), 2.0 ) / X_0;
		double sigma_t1 = (A_0/3)*pow(u_1, 3.0) + (A_1/12)*pow(u_1, 4.0) + (A_2/30)*pow(u_1, 5.0) + (A_3/60)*pow(u_1, 6.0) + (A_4/105)*pow(u_1, 7.0) + (A_5/168)*pow(u_1, 8.0);
		double sigma_t1_theta1 = pow(u_1, 2.0 )*( (A_0/2) + (A_1/6)*u_1 + (A_2/12)*pow(u_1, 2.0) + (A_3/20)*pow(u_1, 3.0) + (A_4/30)*pow(u_1, 4.0) + (A_5/42)*pow(u_1, 5.0) );
		double sigma_theta1 = A_0*u_1 + (A_1/2)*pow(u_1, 2.0) + (A_2/3)*pow(u_1, 3.0) + (A_3/4)*pow(u_1, 4.0) + (A_4/5)*pow(u_1, 5.0) + (A_5/6)*pow(u_1, 6.0);	
		double determinant_Sigma_1 = sigma_t1 * sigma_theta1 - pow( sigma_t1_theta1, 2 );//ad-bc
		double Sigma_1I[4] = // Sigma_1 Inverse = [1/det(Sigma_1)]*{ d, -b, -c, a }
		{
			sigma_theta1 / determinant_Sigma_1, 
			-sigma_t1_theta1 / determinant_Sigma_1, 
			-sigma_t1_theta1 / determinant_Sigma_1, 
			sigma_t1 / determinant_Sigma_1 
		};
		double sigma_2_coefficient = pow( E_0 * ( 1 + 0.038 * log( (u_2 - u_1)/X_0 ) ), 2.0 ) / X_0;	
		double sigma_t2  = (A_0/3)*pow(u_2, 3.0) + (A_1/12)*pow(u_2, 4.0) + (A_2/30)*pow(u_2, 5.0) + (A_3/60)*pow(u_2, 6.0) + (A_4/105)*pow(u_2, 7.0) + (A_5/168)*pow(u_2, 8.0) 
						 - (A_0/3)*pow(u_1, 3.0) - (A_1/4)*pow(u_1, 4.0) - (A_2/5)*pow(u_1, 5.0) - (A_3/6)*pow(u_1, 6.0) - (A_4/7)*pow(u_1, 7.0) - (A_5/8)*pow(u_1, 8.0) 
						 + 2*u_2*( (A_0/2)*pow(u_1, 2.0) + (A_1/3)*pow(u_1, 3.0) + (A_2/4)*pow(u_1, 4.0) + (A_3/5)*pow(u_1, 5.0) + (A_4/6)*pow(u_1, 6.0) + (A_5/7)*pow(u_1, 7.0) ) 
						 - pow(u_2, 2.0) * ( A_0*u_1 + (A_1/2)*pow(u_1, 2.0) + (A_2/3)*pow(u_1, 3.0) + (A_3/4)*pow(u_1, 4.0) + (A_4/5)*pow(u_1, 5.0) + (A_5/6)*pow(u_1, 6.0) );
		double sigma_t2_theta2	= pow(u_2, 2.0 )*( (A_0/2) + (A_1/6)*u_2 + (A_2/12)*pow(u_2, 2.0) + (A_3/20)*pow(u_2, 3.0) + (A_4/30)*pow(u_2, 4.0) + (A_5/42)*pow(u_2, 5.0) ) 
								- u_2*u_1*( A_0 + (A_1/2)*u_1 + (A_2/3)*pow(u_1, 2.0) + (A_3/4)*pow(u_1, 3.0) + (A_4/5)*pow(u_1, 4.0) + (A_5/6)*pow(u_1, 5.0) ) 
								+ pow(u_1, 2.0 )*( (A_0/2) + (A_1/3)*u_1 + (A_2/4)*pow(u_1, 2.0) + (A_3/5)*pow(u_1, 3.0) + (A_4/6)*pow(u_1, 4.0) + (A_5/7)*pow(u_1, 5.0) );
		double sigma_theta2 = A_0 * ( u_2 - u_1 ) + ( A_1 / 2 ) * ( pow(u_2, 2.0) - pow(u_1, 2.0) ) + ( A_2 / 3 ) * ( pow(u_2, 3.0) - pow(u_1, 3.0) ) 
							+ ( A_3 / 4 ) * ( pow(u_2, 4.0) - pow(u_1, 4.0) ) + ( A_4 / 5 ) * ( pow(u_2, 5.0) - pow(u_1, 5.0) ) + ( A_5 /6 )*( pow(u_2, 6.0) - pow(u_1, 6.0) );	
		double determinant_Sigma_2 = sigma_t2 * sigma_theta2 - pow( sigma_t2_theta2, 2 );//ad-bc
		double Sigma_2I[4] = // Sigma_2 Inverse = [1/det(Sigma_2)]*{ d, -b, -c, a }
		{
			sigma_theta2 / determinant_Sigma_2, 
			-sigma_t2_theta2 / determinant_Sigma_2, 
			-sigma_t2_theta2 / determinant_Sigma_2, 
			sigma_t2 / determinant_Sigma_2 
		}; 
		double first_term[4] = 
		{
			Sigma_1I[0] + R_1T[0] * ( Sigma_2I[0] * R_1[0] + Sigma_2I[1] * R_1[2] ) + R_1T[1] * ( Sigma_2I[2] * R_1[0] + Sigma_2I[3] * R_1[2] ),
			Sigma_1I[1] + R_1T[0] * ( Sigma_2I[0] * R_1[1] + Sigma_2I[1] * R_1[3] ) + R_1T[1] * ( Sigma_2I[2] * R_1[1] + Sigma_2I[3] * R_1[3] ),
			Sigma_1I[2] + R_1T[2] * ( Sigma_2I[0] * R_1[0] + Sigma_2I[1] * R_1[2] ) + R_1T[3] * ( Sigma_2I[2] * R_1[0] + Sigma_2I[3] * R_1[2] ),
			Sigma_1I[3] + R_1T[2] * ( Sigma_2I[0] * R_1[1] + Sigma_2I[1] * R_1[3] ) + R_1T[3] * ( Sigma_2I[2] * R_1[1] + Sigma_2I[3] * R_1[3] )
		};
		double determinant_first_term = first_term[0] * first_term[3] - first_term[1] * first_term[2];
		first_term[0] = first_term[3] / determinant_first_term;
		first_term[1] = -first_term[1] / determinant_first_term;
		first_term[2] = -first_term[2] / determinant_first_term;
		first_term[3] = first_term[0] / determinant_first_term;
		double second_term[2] = 
		{
			Sigma_1I[0] * ( R_0[0] * T_0[0] + R_0[1] * T_0[1] ) 
			+ Sigma_1I[1] * ( R_0[2] * T_0[0] + R_0[3] * T_0[1] ) 
			+ R_1T[0] * ( Sigma_2I[0] * T_2[0] + Sigma_2I[1] * T_2[1] ) 
			+ R_1T[1] * ( Sigma_2I[2] * T_2[0] + Sigma_2I[3] * T_2[1] )
			, 
			Sigma_1I[2] * ( R_0[0] * T_0[0] + R_0[1] * T_0[1] ) 
			+ Sigma_1I[3] * ( R_0[2] * T_0[0] + R_0[3] * T_0[1] ) 
			+ R_1T[2] * ( Sigma_2I[0] * T_2[0] + Sigma_2I[1] * T_2[1] ) 
			+ R_1T[3] * ( Sigma_2I[2] * T_2[0] + Sigma_2I[3] * T_2[1] )
		};
		double t_1 = first_term[0] * second_term[0] + first_term[1] * second_term[1];
		double theta_1 = first_term[2] * second_term[0] + first_term[3] * second_term[1];

		// Do v MLP Now
		second_term[0]	= Sigma_1I[0] * ( R_0[0] * V_0[0] + R_0[1] * V_0[1] ) 
						+ Sigma_1I[1] * ( R_0[2] * V_0[0] + R_0[3] * V_0[1] ) 
						+ R_1T[0] * ( Sigma_2I[0] * V_2[0] + Sigma_2I[1] * V_2[1] ) 
						+ R_1T[1] * ( Sigma_2I[2] * V_2[0] + Sigma_2I[3] * V_2[1] );
		second_term[1]	= Sigma_1I[2] * ( R_0[0] * V_0[0] + R_0[1] * V_0[1] ) 
						+ Sigma_1I[3] * ( R_0[2] * V_0[0] + R_0[3] * V_0[1] ) 
						+ R_1T[2] * ( Sigma_2I[0] * V_2[0] + Sigma_2I[1] * V_2[1] ) 
						+ R_1T[3] * ( Sigma_2I[2] * V_2[0] + Sigma_2I[3] * V_2[1] );
		double v_1 = first_term[0] * second_term[0] + first_term[1] * second_term[1];
		double phi_1 = first_term[2] * second_term[0] + first_term[3] * second_term[1];

		// Rotate Coordinate From utv to xyz Coordinate System and Determine Which Voxel this Point on the MLP Path is in
		double x_1 = ( cos( xy_entry_angle ) * (u_in_object + u_1) ) - ( sin( xy_entry_angle ) * t_1 );
		double y_1 = ( sin( xy_entry_angle ) * (u_in_object + u_1) ) + ( cos( xy_entry_angle ) * t_1 );
		double z_1 = v_in_object + v_1;
		x_inside = modf( ( x_1 + MLP_IMAGE_WIDTH/2 ) / MLP_IMAGE_VOXEL_WIDTH, &voxel_x ) * MLP_IMAGE_VOXEL_WIDTH;	
		y_inside = modf( ( MLP_IMAGE_HEIGHT/2 - y_1 ) / MLP_IMAGE_VOXEL_HEIGHT, &voxel_y ) * MLP_IMAGE_VOXEL_HEIGHT;
		z_inside = modf( ( MLP_IMAGE_THICKNESS/2 - z_1 ) / MLP_IMAGE_VOXEL_THICKNESS, &voxel_z ) * MLP_IMAGE_VOXEL_THICKNESS;

		x_voxel_step = (voxel_x >= voxel_x_previous ) - (voxel_x <= voxel_x_previous );
		y_voxel_step = (voxel_y >= voxel_y_previous ) - (voxel_y <= voxel_y_previous );
		z_voxel_step = (voxel_z >= voxel_z_previous ) - (voxel_z <= voxel_z_previous );

		x_to_edge = (x_voxel_step < 0) * x_inside + (x_voxel_step > 0) * (VOXEL_WIDTH - x_inside);
		y_to_edge = (y_voxel_step < 0) * y_inside + (y_voxel_step > 0) * (VOXEL_HEIGHT - y_inside);
		z_to_edge = (z_voxel_step < 0) * z_inside + (z_voxel_step > 0) * (VOXEL_THICKNESS - z_inside);

		voxel = int(voxel_x + voxel_y * MLP_IMAGE_COLUMNS + voxel_z * MLP_IMAGE_COLUMNS * MLP_IMAGE_ROWS);
		if( voxel != path[path_index - 1] )
			path[path_index++] = voxel;
		for( int i = 0; i < path_index; i++ )
			printf( "path[i] = %d\n", path[i] );
		printf( "path_index = %d\n\n", path_index );
		fgets(user_response, sizeof(user_response), stdin);
		MLP_test_image_h[voxel] = 0;

		voxels_passed = (voxel_x - voxel_x_previous) + (voxel_y - voxel_y_previous) + (voxel_z - voxel_z_previous);
		chord_segment = sqrt( pow( x_1_previous - x_1, 2 ) + pow( y_1_previous - y_1, 2 ) + pow( z_1_previous - z_1, 2 ) );
		if( voxels_passed == 0 )
		{
			chord_lengths[path_index - 1] += chord_segment;
		}
		else if( voxels_passed == 1 )
		{
			if( x_voxel_step != 0 )
			{
				chord_fraction = x_to_edge / (x_1_previous - x_1);
			}
			else if( y_voxel_step != 0 )
			{
				chord_fraction = y_to_edge / (y_1_previous - y_1);
			}
			else
			{
				chord_fraction = z_to_edge / (z_1_previous - z_1);
			}
			chord_lengths[path_index - 1] += chord_fraction * chord_segment;
			chord_lengths[path_index] += chord_segment - chord_lengths[path_index - 1];
		}
		else if( voxels_passed == 2 )
		{

		}
		else if( voxels_passed == 3 )
		{

		}
		u_1 += MLP_u_step;
		t_1_previous = t_1;
		v_1_previous = v_1;
		x_1_previous = x_1;
		y_1_previous = y_1;
		z_1_previous = z_1;
		voxel_x_previous = voxel_x;
		voxel_y_previous = voxel_y;
		voxel_z_previous = voxel_z;
		voxel_previous = voxel;
	}
}
float mean_chord_length( float, float )
{
	float eff_angle_t,eff_angle_v;
	
	//eff_angle_t=abs_angle_t-((int)(abs_angle_t/(PI/2)))*(PI/2);
	//
	//eff_angle_v=abs(abs_angle_v);
	//
	//// Get the effective chord in the t-u plane
	//float step_fraction=step_length/pixel_size;
	//float chord_length_2D=(1/3.0)*((step_fraction*step_fraction*sin(2*eff_angle_t)-6)/(step_fraction*sin(2*eff_angle_t)-2*(cos(eff_angle_t)+sin(eff_angle_t))) + step_fraction*step_fraction*sin(2*eff_angle_t)/(2*(cos(eff_angle_t)+sin(eff_angle_t))));
	//
	//// Multiply this by the effective chord in the v-u plane
	//float mean_pixel_width=pixel_size/(cos(eff_angle_t)+sin(eff_angle_t));
	//float height_fraction=slice_thickness/mean_pixel_width;
	//step_fraction=step_length/mean_pixel_width;
	//float chord_length_3D=(1/3.0)*((step_fraction*step_fraction*sin(2*eff_angle_v)-6*height_fraction)/(step_fraction*sin(2*eff_angle_v)-2*(height_fraction*cos(eff_angle_v)+sin(eff_angle_v))) + step_fraction*step_fraction*sin(2*eff_angle_v)/(2*(height_fraction*cos(eff_angle_v)+sin(eff_angle_v))));
	//
	return 0;
	//return pixel_size*chord_length_2D*chord_length_3D;
}
/****************************************************************************************************************************************************************/
/******************************************************************* Image Reconstruction ***********************************************************************/
/****************************************************************************************************************************************************************/
void define_initial_iterate()
{
	float* x_h = (float*) calloc( VOXELS, sizeof(float) );

	switch( X_K0 )
	{
		case X_HULL		: std::copy( x_hull_h, x_hull_h + VOXELS, x_h );	break;
		case FBP_IMAGE	: x_h = FBP_image_h;								break;
		case HYBRID		: std::transform(FBP_image_h, FBP_image_h + VOXELS, x_hull_h, x_h, std::multiplies<float>() );
	}

	if( WRITE_X_K0 )
		array_2_disk("x_k0", OUTPUT_DIRECTORY, OUTPUT_FOLDER, x_h, COLUMNS, ROWS, SLICES, VOXELS, true );
}
void create_hull_image_hybrid()
{
	/*int* SM_differences_h = (int*) calloc( VOXELS, sizeof(int) );
	int* SM_differences_d;
	cudaMalloc((void**) &SM_differences_d, SIZE_IMAGE_INT );
	cudaMemcpy( SM_differences_d, SM_differences_h, SIZE_IMAGE_INT, cudaMemcpyHostToDevice );*/

	

	dim3 dimBlock( SLICES );
	dim3 dimGrid( COLUMNS, ROWS );   

	create_hull_image_hybrid_GPU<<< dimGrid, dimBlock >>>( x_hull_d, FBP_image_d );
	cudaMemcpy( x_h, FBP_image_d, SIZE_IMAGE_FLOAT, cudaMemcpyDeviceToHost );

	if( WRITE_X_K0 )
		array_2_disk("x_k0", OUTPUT_DIRECTORY, OUTPUT_FOLDER, x_h, COLUMNS, ROWS, SLICES, VOXELS, true );
}
__global__ void create_hull_image_hybrid_GPU( bool*& x_hull, float*& FBP_image)
{
	int row = blockIdx.y, column = blockIdx.x, slice = threadIdx.x;
	int voxel = column + row * COLUMNS + slice * COLUMNS * ROWS;
	FBP_image[voxel] *= x_hull[voxel];
}
/****************************************************************************************************************************************************************/
/********************************************************* Memory Transfers, Maintenance, and Cleaning **********************************************************/
/****************************************************************************************************************************************************************/
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
/****************************************************************************************************************************************************************/
/******************************************************* Routines for Writing Data Arrays/Vectors to Disk *******************************************************/
/****************************************************************************************************************************************************************/
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
void write_t_bin_data( FILE* output_file, const std::vector<int>& bin_numbers, const std::vector<float>& data, const BIN_ANALYSIS_TYPE type, const BIN_ORGANIZATION bin_order, int bin )
{
	std::vector<float> bin_histories;
	int num_histories;
	for( int t_bin = 0; t_bin < T_BINS; t_bin++, bin++ )
	{
		if( bin_order == BY_HISTORY )
		{
			for( int i = 0; i < data.size(); i++ )
				if( bin_numbers[i] == bin )
					bin_histories.push_back(data[i]);
		}
		else
			bin_histories.push_back(data[bin]);
		num_histories = bin_histories.size();
		switch( type )
		{
			case COUNTS:	
				fprintf (output_file, "%d ", num_histories);																			
				break;
			case MEANS:		
				fprintf (output_file, "%f ", std::accumulate(bin_histories.begin(), bin_histories.end(), 0.0) / max(num_histories, 1 ) );
				break;
			case MEMBERS:	
				for( int i = 0; i < num_histories; i++ ) 
					fprintf (output_file, "%f ", bin_histories[i]); 
				if( t_bin != T_BINS - 1 )
					fputs("\n", output_file);
		}
		bin_histories.resize(0);
		bin_histories.shrink_to_fit();
	}
}
void write_bin_data( const char* filename_base, const std::vector<int>& bin_numbers, const std::vector<float>& data, const BIN_ANALYSIS_TYPE type, const BIN_ANALYSIS_FOR which_bins, const BIN_ORGANIZATION bin_order, ... )
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
		angles = va_arg(specific_bins, std::vector<int> );
		v_bins = va_arg(specific_bins, std::vector<int> );
		va_end(specific_bins);
		angular_bins.resize(angles.size());
		std::transform(angles.begin(), angles.end(), angular_bins.begin(), std::bind2nd(std::divides<int>(), GANTRY_ANGLE_INTERVAL ) );
	}
	
	int num_angles = angular_bins.size();
	int num_v_bins = v_bins.size();
	/*for( int i = 0; i < 3; i++ )
		printf("%d\n", angles[i] );
	for( int i = 0; i < 3; i++ )
		printf("%d\n", angular_bins[i] );
	for( int i = 0; i < 3; i++ )
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
			write_t_bin_data( output_file, bin_numbers, data, type, bin_order, start_bin );
			if( v_bin != num_v_bins - 1 )
				fputs("\n", output_file);
		}	
		fclose (output_file);
	}
}
void write_t_bin_data( FILE* output_file, int*& bin_numbers, float*& data, const int data_elements, const BIN_ANALYSIS_TYPE type, const BIN_ORGANIZATION bin_order, int bin )
{
	std::vector<float> bin_histories;
	//int data_elements = sizeof(data)/sizeof(float);
	int num_bin_members;
	for( int t_bin = 0; t_bin < T_BINS; t_bin++, bin++ )
	{
		if( bin_order == BY_HISTORY )
		{
			for( int i = 0; i < data_elements; i++ )
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
				for( int i = 0; i < num_bin_members; i++ ) 
					fprintf (output_file, "%f ", bin_histories[i]); 
				if( t_bin != T_BINS - 1 )
					fputs("\n", output_file);
		}
		bin_histories.resize(0);
		bin_histories.shrink_to_fit();
	}
}
void write_bin_data( const char* filename_base, int*& bin_numbers, float*& data, const int data_elements, const BIN_ANALYSIS_TYPE type, const BIN_ANALYSIS_FOR which_bins, const BIN_ORGANIZATION bin_order, ... )
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
		angles = va_arg(specific_bins, std::vector<int> );
		v_bins = va_arg(specific_bins, std::vector<int> );
		va_end(specific_bins);
		angular_bins.resize(angles.size());
		std::transform(angles.begin(), angles.end(), angular_bins.begin(), std::bind2nd(std::divides<int>(), GANTRY_ANGLE_INTERVAL ) );
	}
	//int data_elements = sizeof(data)/sizeof(float);
	//std::cout << std::endl << data_elements << std::endl << std::endl;
	int num_angles = angular_bins.size();
	int num_v_bins = v_bins.size();
	/*for( int i = 0; i < 3; i++ )
		printf("%d\n", angles[i] );
	for( int i = 0; i < 3; i++ )
		printf("%d\n", angular_bins[i] );
	for( int i = 0; i < 3; i++ )
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
			write_t_bin_data( output_file, bin_numbers, data, data_elements, type, bin_order, start_bin );
			if( v_bin != num_v_bins - 1 )
				fputs("\n", output_file);
		}	
		fclose (output_file);
	}
}
void bin_2_txt()
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

		for( int i = 0; i < histories_to_process; i++ )
		{
			fprintf(output_file, "%3f %3f %3f %3f %3f %3f %3f %3f %3f\n", t_in_1_h[i], t_in_2_h[i], t_out_1_h[i], t_out_2_h[i], v_in_1_h[i], v_in_2_h[i], v_out_1_h[i], v_out_2_h[i], WEPL_h[i]);
		}
		fclose (output_file);
		initial_processing_memory_clean();
		start_file_num = end_file_num;
		histories_to_process = 0;
	} 
}
/****************************************************************************************************************************************************************/
/*********************************************************************** Helper Functions ***********************************************************************/
/****************************************************************************************************************************************************************/
bool is_bad_angle( const int angle )
{
	static const int bad_angles[] = {0, 80, 84, 88, 92, 96, 100, 180, 260, 264, 268, 272, 276};
	return std::binary_search( bad_angles, bad_angles + sizeof(bad_angles) / sizeof(int), angle );
}
int calculate_x_voxel(const float x, const int x_voxels, const float voxel_width )
{
	// x @ x_voxel(0): x(0) = -COLUMNS * VOXEL_WIDTH / 2.0 => voxel_x = (  x - x(0) ) / VOXEL_WIDTH;
	return int( ( COLUMNS / 2.0 ) + ( x / VOXEL_WIDTH ) );
}
int calculate_y_voxel(const float y, const int y_voxels, const float voxel_height )
{
	// y @ y_voxel(0): y(0) = ROWS * VOXEL_HEIGHT / 2.0 => voxel_y = ( y(0) - y ) / VOXEL_HEIGHT;
	return int( ( ROWS / 2.0) - ( y / VOXEL_HEIGHT ) );
}
int calculate_slice(const float z, const int z_voxels, const float voxel_thickness )
{
	// z @ z_voxel(0): z(0) = SLICES * VOXEL_THICKNESS / 2.0 => voxel_z = ( z(0) - z ) / VOXEL_THICKNESS
	return  int( ( SLICES / 2.0 ) - ( z / VOXEL_THICKNESS ) );
}
int positions_2_voxels(const float x, const float y, const float z, int& voxel_x, int& voxel_y, int& voxel_z )
{
	voxel_x = ( x + RECON_CYL_RADIUS ) / VOXEL_WIDTH;				
	voxel_y = ( RECON_CYL_RADIUS - y ) / VOXEL_HEIGHT;
	voxel_z = ( RECON_CYL_HEIGHT/2 - z ) /VOXEL_THICKNESS;
	return voxel_x + voxel_y * COLUMNS + voxel_z * COLUMNS * ROWS;
}
__device__ int position_2_voxel_GPU( float &x, float &y, float &z )
{
	int voxel_x = ( x + RECON_CYL_RADIUS ) / VOXEL_WIDTH;
	int voxel_y = ( RECON_CYL_RADIUS - y ) / VOXEL_HEIGHT;
	int voxel_z = ( RECON_CYL_HEIGHT/2 - z ) /VOXEL_THICKNESS;
	return voxel_x + voxel_y * COLUMNS + voxel_z * COLUMNS * ROWS;
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
void exit_program_if( bool early_exit)
{
	if( early_exit )
	{
		char user_response[20];
		stop_execution_timing();
		puts("Hit enter to stop...");
		fgets(user_response, sizeof(user_response), stdin);
		exit(1);
	}
}
void start_execution_timing()
{
	start_time = clock();
}
void stop_execution_timing()
{
	end_time = clock();
	execution_time = (end_time - start_time) / CLOCKS_PER_SEC;
	printf( "Total execution time : %3f\n", double(execution_time) );		
}
void pause_execution()
{
	char user_response[20];
	fgets(user_response, sizeof(user_response), stdin);
}
/****************************************************************************************************************************************************************/
/********************************************************************** Testing Functions ***********************************************************************/
/****************************************************************************************************************************************************************/
int my_divide( int x, int y) {return x*y;}
void test_func()
{
	//auto fn_five = std::bind (my_divide,10,2);               // returns 10/2
  //std::cout << fn_five() << '\n';  

	std::vector<int> bin_numbers;
	std::vector<float> WEPLs;
	test_func2( bin_numbers, WEPLs );
	int angular_bin = 8;
	int v_bin = 14;
	int bin_num = angular_bin * T_BINS + v_bin * ANGULAR_BINS * T_BINS;

	std::cout << typeid(bin_numbers.size()).name() << std::endl;
	std::cout << typeid(1).name() << std::endl;
	printf("%03d %03d\n", bin_numbers.size(), WEPLs.size() );


	/*for( int i = 0; i < WEPLs.size(); i++ )
	{
		printf("%d %3f\n", bin_numbers[i], WEPLs[i] );
	}*/
	char filename[256];
	FILE* output_file;
	int angles[] = {32,120};
	int v_bins[] = {14,15,16};
	float* sino = (float*) std::calloc( 10, sizeof(float));
	auto it = std::begin(angles);
	std::cout << sizeof(&*sino)/sizeof(float) << std::endl << std::endl;
	std::vector<int> angles_vec(angles, angles + sizeof(angles) / sizeof(int) );
	std::vector<int> v_bins_vec(v_bins, v_bins + sizeof(v_bins) / sizeof(int) );
	std::vector<int> angular_bins = angles_vec;
	//std::transform(angles_vec.begin(), angles_vec.end(), angular_bins.begin(), std::bind2nd(std::divides<int>(), GANTRY_ANGLE_INTERVAL ) );
	//int num_angles = sizeof(angles)/sizeof(int);
	//int num_v_bins = sizeof(v_bins)/sizeof(int);
	//std::cout << sizeof(v_bins) << " " << sizeof(angles) << std::endl;
	//std::cout << num_angles << " " << num_v_bins << std::endl;
	//std::cout << angles_vec.size() << " " << angular_bins.size() << std::endl;
	//write_bin_data( "bin data", bin_numbers, WEPLs, COUNTS, ALL_BINS, BY_HISTORY );
	//write_bin_data( "bin data", bin_numbers, WEPLs, COUNTS, ALL_BINS, BY_HISTORY, angles_vec, v_bins_vec );
	//write_bin_data( "bin_counts", bin_numbers, WEPLs, COUNTS, SPECIFIC_BINS, BY_HISTORY, angles_vec, v_bins_vec );
	//write_bin_data( "bin_means", bin_numbers, WEPLs, MEANS, SPECIFIC_BINS, BY_HISTORY, angles_vec, v_bins_vec );
	//write_bin_data( "bin_members", bin_numbers, WEPLs, MEMBERS, SPECIFIC_BINS, BY_HISTORY, angles_vec, v_bins_vec );
	//std::transform(angles_vec.begin(), angles_vec.end(), angular_bins.begin(), std::bind2nd(std::divides<int>(), GANTRY_ANGLE_INTERVAL ) );
	//for( int i = 0; i < angular_bins.size(); i++ )
	//	std::cout << angular_bins[i] << std::endl;
	////std::transform(angles_vec.begin(), angles_vec.end(), angular_bins.begin(), std::bind( std::divides<int>(), 4 ) );

	//
	//auto f1 = std::bind(my_divide, _1, 10);
	////auto triple = std::mem_fn (my_divide, _1);
	//std::transform(angles_vec.begin(), angles_vec.end(), angular_bins.begin(),  f1 );
	//for( int i = 0; i < angular_bins.size(); i++ )
	//	std::cout << angular_bins[i] << std::endl;
	//int angles[] = {32,120,212};
	/*BIN_ANALYSIS_TYPE type = MEMBERS;
	switch( type )
	{
	case MEMBERS: printf("members\n");
	case !MEMBERS: printf("not members\n");
	}
	type = COUNTS;
	switch( type )
	{
	case MEMBERS: printf("members\n");
	case !MEMBERS: printf("not members\n");
	}
	type = MEANS;
	switch( type )
	{
	case MEMBERS: printf("members\n");
	case !MEMBERS: printf("not members\n");
	}*/

}
void test_func2( std::vector<int>& bin_numbers, std::vector<float>& data )
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
}
__device__ void test_func_device( int& x, int& y, int& z )
{
	x = 2;
	y = 3;
	z = 4;
}
__global__ void test_func_GPU( int* a)
{
	//int i = threadIdx.x;
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
