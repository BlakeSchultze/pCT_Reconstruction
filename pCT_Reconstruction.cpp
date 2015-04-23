//#pragma once
//
//#include "pCT_Reconstruction.h"
////#include "C:\Users\Blake\Documents\GitHub\pCT_Reconstruction\Configurations.h"
////#include "C:\Users\Blake\Documents\GitHub\pCT_Reconstruction\Globals.h"
////#include "C:\Users\Blake\Documents\GitHub\pCT_Reconstruction\Constants.h"
//
//// Execution Control Functions
//bool is_bad_angle( const int );	// Just for use with Micah's simultated data
//void timer( bool, clock_t, clock_t);
//void pause_execution();
//void exit_program_if( bool );
//
//// Memory transfers and allocations/deallocations
//void initial_processing_memory_clean();
//void resize_vectors( unsigned int );
//void shrink_vectors( unsigned int );
//void allocations( const unsigned int );
//void reallocations( const unsigned int );
//void post_cut_memory_clean(); 
//
//// Image Initialization/Construction Functions
//template<typename T> void initialize_host_image( T*& );
//template<typename T> void initialize_hull( T*&, T*& );
//void hull_initializations();
//template<typename T> void add_ellipse( T*&, int, double, double, double, double, T );
//template<typename T> void add_circle( T*&, int, double, double, double, T );
//
//// Preprocessing setup and initializations 
//void apply_execution_arguments();
//void assign_SSD_positions();
//void initializations();
//void count_histories();	
//void count_histories_v0();
//void reserve_vector_capacity(); 
//
//// Preprocessing functions
//void read_energy_responses( const int, const int, const int );
//void read_data_chunk( const uint, const uint, const uint );
//void read_data_chunk_v0( const uint, const uint, const uint );
//void read_data_chunk_v02( const uint, const uint, const uint );
//void apply_tuv_shifts( unsigned int );
//void convert_mm_2_cm( unsigned int );
//
//template<typename T> void array_2_disk( char*, char*, DISK_WRITE_MODE, T*, const int, const int, const int, const int, const bool );
//template<typename T> void vector_2_disk( char*, char*, DISK_WRITE_MODE, std::vector<T>, const int, const int, const int, const bool );
//template<typename T> void t_bins_2_disk( FILE*, const std::vector<int>&, const std::vector<T>&, const BIN_ANALYSIS_TYPE, const int );
//template<typename T> void bins_2_disk( char*, char*, DISK_WRITE_MODE, const std::vector<int>&, const std::vector<T>&, const BIN_ANALYSIS_TYPE, const BIN_ANALYSIS_FOR, const BIN_ORGANIZATION, ... );
//template<typename T> void t_bins_2_disk( FILE*, int*&, T*&, const unsigned int, const BIN_ANALYSIS_TYPE, const BIN_ORGANIZATION, int );
//template<typename T> void bins_2_disk( char*, char*, DISK_WRITE_MODE, int*&, T*&, const int, const BIN_ANALYSIS_TYPE, const BIN_ANALYSIS_FOR, const BIN_ORGANIZATION, ... );
//void combine_data_sets();
//
//void read_energy_responses( const int num_histories, const int start_file_num, const int end_file_num )
//{
//	
//	//char data_filename[128];
//	//char magic_number[5];
//	//int version_id;
//	//int file_histories;
//	//float projection_angle, beam_energy;
//	//int generation_date, preprocess_date;
//	//int phantom_name_size, data_source_size, prepared_by_size;
//	//char *phantom_name, *data_source, *prepared_by;
//	//int data_size;
//	////int gantry_position, gantry_angle, scan_histories;
//	//int gantry_position, gantry_angle, scan_number, scan_histories;
//	////int array_index = 0;
//	//FILE* input_file;
//
//	//puts("Reading energy detector responses and performing energy response calibration...");
//	////printf("Reading File for Gantry Angle %d from Scan Number %d...\n", gantry_angle, scan_number );
//	//sprintf(data_filename, "%s%s/%s_%03d%s", PROJECTION_DATA_DIR, INPUT_FOLDER, INPUT_DATA_BASENAME, gantry_angle, FILE_EXTENSION );
//}
///***********************************************************************************************************************************************************************************************************************/
///********************************************************** Read and set execution arguments, preprocessing/reconstruction configurations, settings, and parameters ****************************************************/
///***********************************************************************************************************************************************************************************************************************/
//void apply_execution_arguments()
//{
//	//num_run_arguments = num_arguments;
//	//run_arguments = arguments; 
//	//if(num_arguments == 4)
//	//{
//	//  
//	//  METHOD = atoi(run_arguments[1]);
//	//  ETA = atof(run_arguments[2]);
//	//  PSI_SIGN = atoi(run_arguments[3]);	  
//	//}
//	//printf("num_arguments = %d\n", num_arguments);
//	//printf("num_run_arguments = %d\n", num_run_arguments);
//	//printf("chars = %s\n", run_arguments[2]);
//	//printf("atof = %3f\n", atof(run_arguments[2]));
//	/*if( num_arguments > 1 )
//		PREPROCESSING_DIR = arguments[1];
//	if( num_run_arguments > 2 )
//	{
//		parameter_container.LAMBDA = atof(run_arguments[2]); 
//		LAMBDA = atof(run_arguments[2]);
//		CONSTANT_LAMBDA_SCALE = VOXEL_WIDTH * LAMBDA;
//	}
//	if( num_run_arguments > 3 )
//	{
//		num_voxel_scales =  num_run_arguments - 3;
//		voxel_scales = (double*)calloc( num_voxel_scales, sizeof(double) ); 
//		for( unsigned int i = 3; i < num_run_arguments; i++ )
//			voxel_scales[i-3] = atof(run_arguments[i]);
//	}*/	
//	//			  1				   2		   3	 4	  5    6   ...  N + 3  
//	// ./pCT_Reconstruction [.cfg address] [LAMBDA] [C1] [C2] [C3] ... [CN]
//	//switch( true )
//	//{
//	//	case (num_arguments >= 4): 
//	//		num_voxel_scales =  num_run_arguments - 3;
//	//		voxel_scales = (double*)calloc( num_voxel_scales, sizeof(double) ); 
//	//		for( unsigned int i = 3; i < num_run_arguments; i++ )
//	//			voxel_scales[i-3] = atof(run_arguments[i]);
//	//	case (num_arguments >= 3): 
//	//		parameter_container.LAMBDA = atof(run_arguments[2]); 
//	//		LAMBDA = atof(run_arguments[2]);
//	//	case (num_arguments >= 2): 
//	//		PREPROCESSING_DIR = arguments[1];
//	//	case default: break;
//	//}
//	//printf("LAMBDA = %3f\n", LAMBDA);
//	//
//	//cout << "voxels to be scaled = " << num_voxel_scales << endl;
//	//for( unsigned int i = 0; i < num_voxel_scales; i++ )
//	//	printf("voxel_scale[%d] = %3f\n", i, voxel_scales[i] );
//}
///***********************************************************************************************************************************************************************************************************************/
///************************************************************************************** Memory Transfers, Maintenance, and Cleaning ************************************************************************************/
///***********************************************************************************************************************************************************************************************************************/
//void initializations()
//{
//	puts("Allocating statistical analysis arrays on host/GPU...");
//
//	histories_per_scan		= (int*)	calloc( NUM_SCANS,	sizeof(int)	);
//	bin_counts_h			= (int*)	calloc( NUM_BINS,	sizeof(int)	);
//	mean_WEPL_h				= (float*)	calloc( NUM_BINS,	sizeof(float) );
//	mean_rel_ut_angle_h		= (float*)	calloc( NUM_BINS,	sizeof(float) );
//	mean_rel_uv_angle_h		= (float*)	calloc( NUM_BINS,	sizeof(float) );
//	
//	if( ( bin_counts_h == NULL ) || (mean_WEPL_h == NULL) || (mean_rel_ut_angle_h == NULL) || (mean_rel_uv_angle_h == NULL) )
//	{
//		puts("std dev allocation error\n");
//		exit(1);
//	}
//
//	cudaMalloc((void**) &bin_counts_d,			SIZE_BINS_INT );
//	cudaMalloc((void**) &mean_WEPL_d,			SIZE_BINS_FLOAT );
//	cudaMalloc((void**) &mean_rel_ut_angle_d,	SIZE_BINS_FLOAT );
//	cudaMalloc((void**) &mean_rel_uv_angle_d,	SIZE_BINS_FLOAT );
//
//	cudaMemcpy( bin_counts_d,			bin_counts_h,			SIZE_BINS_INT,		cudaMemcpyHostToDevice );
//	cudaMemcpy( mean_WEPL_d,			mean_WEPL_h,			SIZE_BINS_FLOAT,	cudaMemcpyHostToDevice );
//	cudaMemcpy( mean_rel_ut_angle_d,	mean_rel_ut_angle_h,	SIZE_BINS_FLOAT,	cudaMemcpyHostToDevice );
//	cudaMemcpy( mean_rel_uv_angle_d,	mean_rel_uv_angle_h,	SIZE_BINS_FLOAT,	cudaMemcpyHostToDevice );
//}
//void reserve_vector_capacity()
//{
//	// Reserve enough memory for vectors to hold all histories.  If a vector grows to the point where the next memory address is already allocated to another
//	// object, the system must first move the vector to a new location in memory which can hold the existing vector and new element.  The eventual size of these
//	// vectors is quite large and the possibility of this happening is high for one or more vectors and it can happen multiple times as the vector grows.  Moving 
//	// a vector and its contents is a time consuming process, especially as it becomes large, so we reserve enough memory to guarantee this does not happen.
//	bin_num_vector.reserve( total_histories );
//	gantry_angle_vector.reserve( total_histories );
//	WEPL_vector.reserve( total_histories );
//	x_entry_vector.reserve( total_histories );
//	y_entry_vector.reserve( total_histories );
//	z_entry_vector.reserve( total_histories );
//	x_exit_vector.reserve( total_histories );
//	y_exit_vector.reserve( total_histories );
//	z_exit_vector.reserve( total_histories );
//	xy_entry_angle_vector.reserve( total_histories );
//	xz_entry_angle_vector.reserve( total_histories );
//	xy_exit_angle_vector.reserve( total_histories );
//	xz_exit_angle_vector.reserve( total_histories );
//}
//void initial_processing_memory_clean()
//{
//	//clear_input_memory
//	//free( missed_recon_volume_h );
//	free( gantry_angle_h );
//	cudaFree( x_entry_d );
//	cudaFree( y_entry_d );
//	cudaFree( z_entry_d );
//	cudaFree( x_exit_d );
//	cudaFree( y_exit_d );
//	cudaFree( z_exit_d );
//	cudaFree( missed_recon_volume_d );
//	cudaFree( bin_num_d );
//	cudaFree( WEPL_d);
//}
//void resize_vectors( unsigned int new_size )
//{
//	bin_num_vector.resize( new_size );
//	gantry_angle_vector.resize( new_size );
//	WEPL_vector.resize( new_size );
//	x_entry_vector.resize( new_size );	
//	y_entry_vector.resize( new_size );	
//	z_entry_vector.resize( new_size );
//	x_exit_vector.resize( new_size );
//	y_exit_vector.resize( new_size );
//	z_exit_vector.resize( new_size );
//	xy_entry_angle_vector.resize( new_size );	
//	xz_entry_angle_vector.resize( new_size );	
//	xy_exit_angle_vector.resize( new_size );
//	xz_exit_angle_vector.resize( new_size );
//}
//void shrink_vectors( unsigned int new_capacity )
//{
//	bin_num_vector.shrink_to_fit();
//	gantry_angle_vector.shrink_to_fit();
//	WEPL_vector.shrink_to_fit();
//	x_entry_vector.shrink_to_fit();	
//	y_entry_vector.shrink_to_fit();	
//	z_entry_vector.shrink_to_fit();	
//	x_exit_vector.shrink_to_fit();	
//	y_exit_vector.shrink_to_fit();	
//	z_exit_vector.shrink_to_fit();	
//	xy_entry_angle_vector.shrink_to_fit();	
//	xz_entry_angle_vector.shrink_to_fit();	
//	xy_exit_angle_vector.shrink_to_fit();	
//	xz_exit_angle_vector.shrink_to_fit();	
//}
//void initialize_stddev()
//{	
//	stddev_rel_ut_angle_h = (float*) calloc( NUM_BINS, sizeof(float) );	
//	stddev_rel_uv_angle_h = (float*) calloc( NUM_BINS, sizeof(float) );	
//	stddev_WEPL_h		  = (float*) calloc( NUM_BINS, sizeof(float) );
//	if( ( stddev_rel_ut_angle_h == NULL ) || (stddev_rel_uv_angle_h == NULL) || (stddev_WEPL_h == NULL) )
//	{
//		puts("std dev allocation error\n");
//		exit(1);
//	}
//	cudaMalloc((void**) &stddev_rel_ut_angle_d,	SIZE_BINS_FLOAT );
//	cudaMalloc((void**) &stddev_rel_uv_angle_d,	SIZE_BINS_FLOAT );
//	cudaMalloc((void**) &stddev_WEPL_d,			SIZE_BINS_FLOAT );
//
//	cudaMemcpy( stddev_rel_ut_angle_d,	stddev_rel_ut_angle_h,	SIZE_BINS_FLOAT,	cudaMemcpyHostToDevice );
//	cudaMemcpy( stddev_rel_uv_angle_d,	stddev_rel_uv_angle_h,	SIZE_BINS_FLOAT,	cudaMemcpyHostToDevice );
//	cudaMemcpy( stddev_WEPL_d,			stddev_WEPL_h,			SIZE_BINS_FLOAT,	cudaMemcpyHostToDevice );
//}
//void post_cut_memory_clean()
//{
//	puts("Freeing unnecessary memory, resizing vectors, and shrinking vectors to fit just the remaining histories...");
//
//	//free(failed_cuts_h );
//	free(stddev_rel_ut_angle_h);
//	free(stddev_rel_uv_angle_h);
//	free(stddev_WEPL_h);
//
//	//cudaFree( failed_cuts_d );
//	//cudaFree( bin_num_d );
//	//cudaFree( WEPL_d );
//	//cudaFree( xy_entry_angle_d );
//	//cudaFree( xz_entry_angle_d );
//	//cudaFree( xy_exit_angle_d );
//	//cudaFree( xz_exit_angle_d );
//
//	cudaFree( mean_rel_ut_angle_d );
//	cudaFree( mean_rel_uv_angle_d );
//	cudaFree( mean_WEPL_d );
//	cudaFree( stddev_rel_ut_angle_d );
//	cudaFree( stddev_rel_uv_angle_d );
//	cudaFree( stddev_WEPL_d );
//}
///***********************************************************************************************************************************************************************************************************************/
///**************************************************************************************** Preprocessing setup and initializations **************************************************************************************/
///***********************************************************************************************************************************************************************************************************************/
//void assign_SSD_positions()	//HERE THE COORDINATES OF THE DETECTORS PLANES ARE LOADED, THE CONFIG FILE IS CREATED BY FORD (RWS)
//{
//	char user_response[20];
//	char configFilename[512];
//	puts("Reading tracker plane positions...");
//
//	sprintf(configFilename, "%s\\scan.cfg", PREPROCESSING_DIR);
//	if( DEBUG_TEXT_ON )
//		printf("Opening config file %s...\n", configFilename);
//	std::ifstream configFile(configFilename);		
//	if( !configFile.is_open() ) {
//		printf("ERROR: config file not found at %s!\n", configFilename);	
//		exit_program_if(true);
//	}
//	else
//	{
//		fputs("Found File", stdout);
//		fflush(stdout);
//		printf("user_response = \"%s\"\n", user_response);
//	}
//	if( DEBUG_TEXT_ON )
//		puts("Reading Tracking Plane Positions...");
//	for( unsigned int i = 0; i < 8; i++ ) {
//		configFile >> SSD_u_Positions[i];
//		if( DEBUG_TEXT_ON )
//			printf("SSD_u_Positions[%d] = %3f", i, SSD_u_Positions[i]);
//	}
//	
//	configFile.close();
//
//}
//void count_histories()
//{
//	for( uint scan_number = 0; scan_number < NUM_SCANS; scan_number++ )
//		histories_per_scan[scan_number] = 0;
//
//	histories_per_file =				 (int*) calloc( NUM_SCANS * GANTRY_ANGLES, sizeof(int) );
//	histories_per_gantry_angle =		 (int*) calloc( GANTRY_ANGLES, sizeof(int) );
//	recon_vol_histories_per_projection = (int*) calloc( GANTRY_ANGLES, sizeof(int) );
//
//	if( DEBUG_TEXT_ON )
//		puts("Counting proton histories...\n");
//	switch( DATA_FORMAT )
//	{
//		case VERSION_0  : count_histories_v0();		break;
//	}
//	if( DEBUG_TEXT_ON )
//	{
//		for( uint file_number = 0, gantry_position_number = 0; file_number < (NUM_SCANS * GANTRY_ANGLES); file_number++, gantry_position_number++ )
//		{
//			if( file_number % NUM_SCANS == 0 )
//				printf("There are a Total of %d Histories From Gantry Angle %d\n", histories_per_gantry_angle[gantry_position_number], int(gantry_position_number* GANTRY_ANGLE_INTERVAL) );			
//			printf("* %d Histories are From Scan Number %d\n", histories_per_file[file_number], (file_number % NUM_SCANS) + 1 );
//			
//		}
//		for( uint scan_number = 0; scan_number < NUM_SCANS; scan_number++ )
//			printf("There are a Total of %d Histories in Scan Number %d \n", histories_per_scan[scan_number], scan_number + 1);
//		printf("There are a Total of %d Histories\n", total_histories);
//	}
//}
//void count_histories_v0()
//{
//	char data_filename[256];
//	float projection_angle;
//	unsigned int magic_number, num_histories, file_number = 0, gantry_position_number = 0;
//	for( unsigned int gantry_angle = 0; gantry_angle < 360; gantry_angle += int(GANTRY_ANGLE_INTERVAL), gantry_position_number++ )
//	{
//		for( unsigned int scan_number = 1; scan_number <= NUM_SCANS; scan_number++, file_number++ )
//		{
//			sprintf(data_filename, "%s/%s_%03d%s", PREPROCESSING_DIR, INPUT_DATA_BASENAME, gantry_angle, FILE_EXTENSION  );
//			//sprintf(data_filename, "%s/%s_%03d%s", PREPROCESSING_DIR, INPUT_DATA_BASENAME, gantry_position_number, FILE_EXTENSION  );
//			/*
//			Contains the following headers:
//				Magic number identifier: "PCTD" (4-byte string)
//				Format version identifier (integer)
//				Number of events in file (integer)
//				Projection angle (float | degrees)
//				Beam energy (float | MeV)
//				Acquisition/generation date (integer | Unix time)
//				Pre-process date (integer | Unix time)
//				Phantom name or description (variable length string)
//				Data source (variable length string)
//				Prepared by (variable length string)
//				* Note on variable length strings: each variable length string should be preceded with an integer containing the number of characters in the string.			
//			*/
//			FILE* data_file = fopen(data_filename, "rb");
//			if( data_file == NULL )
//			{
//				fputs( "Error Opening Data File:  Check that the directories are properly named.", stderr ); 
//				exit_program_if(true);
//			}
//			
//			fread(&magic_number, 4, 1, data_file );
//			if( magic_number != MAGIC_NUMBER_CHECK ) 
//			{
//				puts("Error: unknown file type (should be PCTD)!\n");
//				exit_program_if(true);
//			}
//
//			fread(&VERSION_ID, sizeof(int), 1, data_file );		
//			if( VERSION_ID == 0 )
//			{
//				DATA_FORMAT	= VERSION_0;
//				fread(&num_histories, sizeof(int), 1, data_file );
//				if( DEBUG_TEXT_ON )
//					printf("There are %d Histories for Gantry Angle %d From Scan Number %d\n", num_histories, gantry_angle, scan_number);
//				histories_per_file[file_number] = num_histories;
//				histories_per_gantry_angle[gantry_position_number] += num_histories;
//				histories_per_scan[scan_number-1] += num_histories;
//				total_histories += num_histories;
//			
//				fread(&projection_angle, sizeof(float), 1, data_file );
//				projection_angles.push_back(projection_angle);
//
//				fseek( data_file, 2 * sizeof(int) + sizeof(float), SEEK_CUR );
//				fread(&PHANTOM_NAME_SIZE, sizeof(int), 1, data_file );
//
//				fseek( data_file, PHANTOM_NAME_SIZE, SEEK_CUR );
//				fread(&DATA_SOURCE_SIZE, sizeof(int), 1, data_file );
//
//				fseek( data_file, DATA_SOURCE_SIZE, SEEK_CUR );
//				fread(&PREPARED_BY_SIZE, sizeof(int), 1, data_file );
//
//				fseek( data_file, PREPARED_BY_SIZE, SEEK_CUR );
//				fclose(data_file);
//				SKIP_2_DATA_SIZE = 4 + 7 * sizeof(int) + 2 * sizeof(float) + PHANTOM_NAME_SIZE + DATA_SOURCE_SIZE + PREPARED_BY_SIZE;
//				//pause_execution();
//			}
//			else if( VERSION_ID == 1 )
//			{
//				DATA_FORMAT = VERSION_1;
//				fread(&num_histories, sizeof(int), 1, data_file );
//				if( DEBUG_TEXT_ON )
//					printf("There are %d Histories for Gantry Angle %d From Scan Number %d\n", num_histories, gantry_angle, scan_number);
//				histories_per_file[file_number] = num_histories;
//				histories_per_gantry_angle[gantry_position_number] += num_histories;
//				histories_per_scan[scan_number-1] += num_histories;
//				total_histories += num_histories;
//			
//				fread(&projection_angle, sizeof(float), 1, data_file );
//				projection_angles.push_back(projection_angle);
//
//				fseek( data_file, 2 * sizeof(int) + sizeof(float), SEEK_CUR );
//				fread(&PHANTOM_NAME_SIZE, sizeof(int), 1, data_file );
//
//				fseek( data_file, PHANTOM_NAME_SIZE, SEEK_CUR );
//				fread(&DATA_SOURCE_SIZE, sizeof(int), 1, data_file );
//
//				fseek( data_file, DATA_SOURCE_SIZE, SEEK_CUR );
//				fread(&PREPARED_BY_SIZE, sizeof(int), 1, data_file );
//
//				fseek( data_file, PREPARED_BY_SIZE, SEEK_CUR );
//				fclose(data_file);
//				SKIP_2_DATA_SIZE = 4 + 7 * sizeof(int) + 2 * sizeof(float) + PHANTOM_NAME_SIZE + DATA_SOURCE_SIZE + PREPARED_BY_SIZE;
//				//pause_execution();
//			}
//			else 
//			{
//				DATA_FORMAT = OLD_FORMAT;
//				printf("ERROR: Data format is not Version (%d)!\n", VERSION_ID);
//				exit_program_if(true);
//			}						
//		}
//	}
//}
///***********************************************************************************************************************************************************************************************************************/
///******************************************************************************************* Image initialization/Construction *****************************************************************************************/
///***********************************************************************************************************************************************************************************************************************/
//template<typename T> void initialize_host_image( T*& image )
//{
//	image = (T*)calloc( IMAGE_VOXELS, sizeof(T));
//}
//template<typename T> void add_ellipse( T*& image, int slice, double x_center, double y_center, double semi_major_axis, double semi_minor_axis, T value )
//{
//	double x, y;
//	for( int row = 0; row < ROWS; row++ )
//	{
//		for( int column = 0; column < COLUMNS; column++ )
//		{
//			x = ( column - COLUMNS/2 + 0.5) * VOXEL_WIDTH;
//			y = ( ROWS/2 - row - 0.5 ) * VOXEL_HEIGHT;
//			if( pow( ( x - x_center) / semi_major_axis, 2 ) + pow( ( y - y_center )  / semi_minor_axis, 2 ) <= 1 )
//				image[slice * COLUMNS * ROWS + row * COLUMNS + column] = value;
//		}
//	}
//}
//template<typename T> void add_circle( T*& image, int slice, double x_center, double y_center, double radius, T value )
//{
//	double x, y;
//	for( int row = 0; row < ROWS; row++ )
//	{
//		for( int column = 0; column < COLUMNS; column++ )
//		{
//			x = ( column - COLUMNS/2 + 0.5) * VOXEL_WIDTH;
//			//x_center = ( center_column - COLUMNS/2 + 0.5) * VOXEL_WIDTH;
//			y = ( ROWS/2 - row - 0.5 ) * VOXEL_HEIGHT;
//			//y_center = ( center_row - COLUMNS/2 + 0.5) * VOXEL_WIDTH;
//			if( pow( (x - x_center), 2 ) + pow( (y - y_center), 2 ) <= pow( radius, 2) )
//				image[slice * COLUMNS * ROWS + row * COLUMNS + column] = value;
//		}
//	}
//}	
//template<typename O> void import_image( O*& import_into, char* directory, char* filename_base, DISK_WRITE_MODE format )
//{
//	char filename[256];
//	FILE* input_file;	
//	switch( format )
//	{
//		case TEXT	:	sprintf( filename, "%s/%s.txt", directory, filename_base  );	
//						input_file = fopen(filename, "r" );								
//						break;
//		case BINARY	:	sprintf( filename, "%s/%s.bin", directory, filename_base );
//						input_file = fopen(filename, "rb" );
//	}
//	//FILE* input_file = fopen(filename, "rb" );
//	O* temp = (O*)calloc(NUM_VOXELS, sizeof(O) );
//	fread(temp, sizeof(O), NUM_VOXELS, input_file );
//	free(import_into);
//	import_into = temp;
//}
///***********************************************************************************************************************************************************************************************************************/
///************************************************************************************** Data importation, initial cuts, and binning ************************************************************************************/
///***********************************************************************************************************************************************************************************************************************/
//
//void convert_mm_2_cm( unsigned int num_histories )
//{
//	for( unsigned int i = 0; i < num_histories; i++ ) 
//	{
//		// Convert the input data from mm to cm
//		v_in_1_h[i]	 *= MM_TO_CM;
//		v_in_2_h[i]	 *= MM_TO_CM;
//		v_out_1_h[i] *= MM_TO_CM;
//		v_out_2_h[i] *= MM_TO_CM;
//		t_in_1_h[i]	 *= MM_TO_CM;
//		t_in_2_h[i]	 *= MM_TO_CM;
//		t_out_1_h[i] *= MM_TO_CM;
//		t_out_2_h[i] *= MM_TO_CM;
//		u_in_1_h[i]	 *= MM_TO_CM;
//		u_in_2_h[i]	 *= MM_TO_CM;
//		u_out_1_h[i] *= MM_TO_CM;
//		u_out_2_h[i] *= MM_TO_CM;
//		WEPL_h[i]	 *= (float)MM_TO_CM;
//	}
//
//}
//void apply_tuv_shifts( unsigned int num_histories)
//{
//	for( unsigned int i = 0; i < num_histories; i++ ) 
//	{
//		// Correct for any shifts in u/t coordinates
//		t_in_1_h[i]	 += T_SHIFT;
//		t_in_2_h[i]	 += T_SHIFT;
//		t_out_1_h[i] += T_SHIFT;
//		t_out_2_h[i] += T_SHIFT;
//		u_in_1_h[i]	 += U_SHIFT;
//		u_in_2_h[i]	 += U_SHIFT;
//		u_out_1_h[i] += U_SHIFT;
//		u_out_2_h[i] += U_SHIFT;
//		v_in_1_h[i]	 += V_SHIFT;
//		v_in_2_h[i]	 += V_SHIFT;
//		v_out_1_h[i] += V_SHIFT;
//		v_out_2_h[i] += V_SHIFT;
//		if( WRITE_SSD_ANGLES )
//		{
//			ut_entry_angle[i] = atan2( t_in_2_h[i] - t_in_1_h[i], u_in_2_h[i] - u_in_1_h[i] );	
//			uv_entry_angle[i] = atan2( v_in_2_h[i] - v_in_1_h[i], u_in_2_h[i] - u_in_1_h[i] );	
//			ut_exit_angle[i] = atan2( t_out_2_h[i] - t_out_1_h[i], u_out_2_h[i] - u_out_1_h[i] );	
//			uv_exit_angle[i] = atan2( v_out_2_h[i] - v_out_1_h[i], u_out_2_h[i] - u_out_1_h[i] );	
//		}
//	}
//	if( WRITE_SSD_ANGLES )
//	{
//		char data_filename[256];
//		sprintf(data_filename, "%s_%03d", "ut_entry_angle", gantry_angle_h );
//		array_2_disk( data_filename, PREPROCESSING_DIR, TEXT, ut_entry_angle, COLUMNS, ROWS, SLICES, num_histories, true );
//		sprintf(data_filename, "%s_%03d", "uv_entry_angle", gantry_angle_h );
//		array_2_disk( data_filename, PREPROCESSING_DIR, TEXT, uv_entry_angle, COLUMNS, ROWS, SLICES, num_histories, true );
//		sprintf(data_filename, "%s_%03d", "ut_exit_angle", gantry_angle_h );
//		array_2_disk( data_filename, PREPROCESSING_DIR, TEXT, ut_exit_angle, COLUMNS, ROWS, SLICES, num_histories, true );
//		sprintf(data_filename, "%s_%03d", "uv_exit_angle", gantry_angle_h );
//		array_2_disk( data_filename, PREPROCESSING_DIR, TEXT, uv_exit_angle, COLUMNS, ROWS, SLICES, num_histories, true );
//	}
//}
//void read_data_chunk( const uint num_histories, const uint start_file_num, const uint end_file_num )
//{
//	// The GPU cannot process all the histories at once, so they are broken up into chunks that can fit on the GPU.  As we iterate 
//	// through the data one chunk at a time, we determine which histories enter the reconstruction volume and if they belong to a 
//	// valid bin (i.e. t, v, and angular bin number is greater than zero and less than max).  If both are true, we push the bin
//	// number, WEPL, and relative entry/exit ut/uv angles to the back of their corresponding std::vector.
//	
//	unsigned int size_floats = sizeof(float) * num_histories;
//	unsigned int size_ints = sizeof(int) * num_histories;
//
//	t_in_1_h		= (float*) malloc(size_floats);
//	t_in_2_h		= (float*) malloc(size_floats);
//	t_out_1_h		= (float*) malloc(size_floats);
//	t_out_2_h		= (float*) malloc(size_floats);
//	u_in_1_h		= (float*) malloc(size_floats);
//	u_in_2_h		= (float*) malloc(size_floats);
//	u_out_1_h		= (float*) malloc(size_floats);
//	u_out_2_h		= (float*) malloc(size_floats);
//	v_in_1_h		= (float*) malloc(size_floats);
//	v_in_2_h		= (float*) malloc(size_floats);
//	v_out_1_h		= (float*) malloc(size_floats);
//	v_out_2_h		= (float*) malloc(size_floats);		
//	WEPL_h			= (float*) malloc(size_floats);
//	gantry_angle_h	= (int*)   malloc(size_ints);
//
//	if( WRITE_SSD_ANGLES )
//	{
//		ut_entry_angle	= (float*) malloc(size_floats);
//		uv_entry_angle	= (float*) malloc(size_floats);
//		ut_exit_angle	= (float*) malloc(size_floats);
//		uv_exit_angle	= (float*) malloc(size_floats);
//	}
//	switch( DATA_FORMAT )
//	{	
//		case VERSION_0  : read_data_chunk_v02(  num_histories, start_file_num, end_file_num - 1 ); break;
//	}
//}
//void read_data_chunk_v0( const uint num_histories, const uint start_file_num, const uint end_file_num )
//{	
//	/*
//	Event data:
//	Data is be stored with all of one type in a consecutive row, meaning the first entries will be N t0 values, where N is the number of events in the file. Next will be N t1 values, etc. This more closely matches the data structure in memory.
//	Detector coordinates in mm relative to a phantom center, given in the detector coordinate system:
//		t0 (float * N)
//		t1 (float * N)
//		t2 (float * N)
//		t3 (float * N)
//		v0 (float * N)
//		v1 (float * N)
//		v2 (float * N)
//		v3 (float * N)
//		u0 (float * N)
//		u1 (float * N)
//		u2 (float * N)
//		u3 (float * N)
//		WEPL in mm (float * N)
//	*/
//	char data_filename[128];
//	unsigned int gantry_position, gantry_angle, scan_number, file_histories, array_index = 0, histories_read = 0;
//	FILE* data_file;
//	printf("%d histories to be read from %d files\n", num_histories, end_file_num - start_file_num + 1 );
//	for( unsigned int file_num = start_file_num; file_num <= end_file_num; file_num++ )
//	{	
//		gantry_position = file_num / NUM_SCANS;
//		gantry_angle = int(gantry_position * GANTRY_ANGLE_INTERVAL);
//		scan_number = file_num % NUM_SCANS + 1;
//		file_histories = histories_per_file[file_num];
//		
//		//sprintf(data_filename, "%s/%s_%03d%s", PROJECTION_DATA_DIR, INPUT_DATA_BASENAME, gantry_angle, FILE_EXTENSION );
//		sprintf(data_filename, "%s/%s_%03d%s", PROJECTION_DATA_DIR, INPUT_DATA_BASENAME, gantry_position, FILE_EXTENSION );
//		if( strcmp(FILE_EXTENSION, ".bin") == 0 )
//			data_file = fopen(data_filename, "rb");
//		else if( strcmp(FILE_EXTENSION, ".txt") == 0 )
//			data_file = fopen(data_filename, "r");
//		if( data_file == NULL )
//		{
//			fputs( "Error Opening Data File:  Check that the directories are properly named.", stderr ); 
//			exit_program_if(true);
//		}
//		if( VERSION_ID == 0 )
//		{
//			printf("\t");
//			printf("Reading %d histories for gantry angle %d from scan number %d...\n", file_histories, gantry_angle, scan_number );			
//			fseek( data_file, SKIP_2_DATA_SIZE, SEEK_SET );
//
//			fread( &t_in_1_h[histories_read],  sizeof(float), file_histories, data_file );
//			fread( &t_in_2_h[histories_read],  sizeof(float), file_histories, data_file );
//			fread( &t_out_1_h[histories_read], sizeof(float), file_histories, data_file );
//			fread( &t_out_2_h[histories_read], sizeof(float), file_histories, data_file );
//			fread( &v_in_1_h[histories_read],  sizeof(float), file_histories, data_file );
//			fread( &v_in_2_h[histories_read],  sizeof(float), file_histories, data_file );
//			fread( &v_out_1_h[histories_read], sizeof(float), file_histories, data_file );
//			fread( &v_out_2_h[histories_read], sizeof(float), file_histories, data_file );
//			fread( &u_in_1_h[histories_read],  sizeof(float), file_histories, data_file );
//			fread( &u_in_2_h[histories_read],  sizeof(float), file_histories, data_file );
//			fread( &u_out_1_h[histories_read], sizeof(float), file_histories, data_file );
//			fread( &u_out_2_h[histories_read], sizeof(float), file_histories, data_file );
//			fread( &WEPL_h[histories_read],    sizeof(float), file_histories, data_file );
//			fclose(data_file);
//
//			histories_read += file_histories;
//			for( unsigned int i = 0; i < file_histories; i++, array_index++ ) 
//				gantry_angle_h[array_index] = int(projection_angles[file_num]);							
//		}
//		else if( VERSION_ID == 1 )
//		{
//			printf("\t");
//			printf("Reading %d histories for gantry angle %d from scan number %d...\n", file_histories, gantry_angle, scan_number );			
//			fseek( data_file, SKIP_2_DATA_SIZE, SEEK_SET );
//
//			fread( &t_in_1_h[histories_read],  sizeof(float), file_histories, data_file );
//			fread( &t_in_2_h[histories_read],  sizeof(float), file_histories, data_file );
//			fread( &t_out_1_h[histories_read], sizeof(float), file_histories, data_file );
//			fread( &t_out_2_h[histories_read], sizeof(float), file_histories, data_file );
//			fread( &v_in_1_h[histories_read],  sizeof(float), file_histories, data_file );
//			fread( &v_in_2_h[histories_read],  sizeof(float), file_histories, data_file );
//			fread( &v_out_1_h[histories_read], sizeof(float), file_histories, data_file );
//			fread( &v_out_2_h[histories_read], sizeof(float), file_histories, data_file );
//			fread( &u_in_1_h[histories_read],  sizeof(float), file_histories, data_file );
//			fread( &u_in_2_h[histories_read],  sizeof(float), file_histories, data_file );
//			fread( &u_out_1_h[histories_read], sizeof(float), file_histories, data_file );
//			fread( &u_out_2_h[histories_read], sizeof(float), file_histories, data_file );
//			fread( &WEPL_h[histories_read],    sizeof(float), file_histories, data_file );
//			fclose(data_file);
//
//			histories_read += file_histories;
//			for( unsigned int i = 0; i < file_histories; i++, array_index++ ) 
//				gantry_angle_h[array_index] = int(projection_angles[file_num]);							
//		}
//	}
//	convert_mm_2_cm( num_histories );
//	if( T_SHIFT != 0.0	||  U_SHIFT != 0.0 ||  V_SHIFT != 0.0)
//		apply_tuv_shifts( num_histories );
//}
//void read_data_chunk_v02( const uint num_histories, const uint start_file_num, const uint end_file_num )
//{
//	/*
//	Contains the following headers:
//		Magic number identifier: "PCTD" (4-byte string)
//		Format version identifier (integer)
//		Number of events in file (integer)
//		Projection angle (float | degrees)
//		Beam energy (float | MeV)
//		Acquisition/generation date (integer | Unix time)
//		Pre-process date (integer | Unix time)
//		Phantom name or description (variable length string)
//		Data source (variable length string)
//		Prepared by (variable length string)
//	* Note on variable length strings: each variable length string should be preceded with an integer containing the number of characters in the string.
//	
//	Event data:
//	Data is be stored with all of one type in a consecutive row, meaning the first entries will be N t0 values, where N is the number of events in the file. Next will be N t1 values, etc. This more closely matches the data structure in memory.
//	Detector coordinates in mm relative to a phantom center, given in the detector coordinate system:
//		t0 (float * N)
//		t1 (float * N)
//		t2 (float * N)
//		t3 (float * N)
//		v0 (float * N)
//		v1 (float * N)
//		v2 (float * N)
//		v3 (float * N)
//		u0 (float * N)
//		u1 (float * N)
//		u2 (float * N)
//		u3 (float * N)
//		WEPL in mm (float * N)
//	*/
//	//char user_response[20];
//	char data_filename[128];
//	std::ifstream data_file;
//	int array_index = 0, histories_read = 0;
//	for( uint file_num = start_file_num; file_num <= end_file_num; file_num++ )
//	{
//		int gantry_position = file_num / NUM_SCANS;
//		int gantry_angle = int(gantry_position * GANTRY_ANGLE_INTERVAL);
//		int scan_number = file_num % NUM_SCANS + 1;
//
//		printf("Reading File for Gantry Angle %d from Scan Number %d...\n", gantry_angle, scan_number );
//		//sprintf(data_filename, "%s/%s_%03d%s", PROJECTION_DATA_DIR, INPUT_DATA_BASENAME, gantry_angle, FILE_EXTENSION );
//		sprintf(data_filename, "%s/%s_%03d%s", PROJECTION_DATA_DIR, INPUT_DATA_BASENAME, gantry_position, FILE_EXTENSION );
//		if( strcmp(FILE_EXTENSION, ".bin") == 0 )
//			data_file.open(data_filename, std::ios::binary);
//		else if( strcmp(FILE_EXTENSION, ".txt") == 0 )
//			data_file.open(data_filename);
//
//		//sprintf(data_filename, "%s%s/%s_%03d%s", PROJECTION_DATA_DIR, INPUT_FOLDER, INPUT_DATA_BASENAME, gantry_angle, FILE_EXTENSION );	
//		//std::ifstream data_file(data_filename, std::ios::binary);
//		if( data_file == NULL )
//		{
//			fputs( "File not found:  Check that the directories and files are properly named.", stderr ); 
//			exit_program_if(true);
//		}
//		char magic_number[5];
//		data_file.read(magic_number, 4);
//		magic_number[4] = '\0';
//		if( strcmp(magic_number, "PCTD") ) {
//			puts("Error: unknown file type (should be PCTD)!\n");
//			exit_program_if(true);
//		}
//		int version_id;
//		data_file.read((char*)&version_id, sizeof(int));
//		if( version_id == 0 )
//		{
//			int file_histories;
//			data_file.read((char*)&file_histories, sizeof(int));
//	
//			puts("Reading headers from file...\n");
//	
//			float projection_angle, beam_energy;
//			int generation_date, preprocess_date;
//			int phantom_name_size, data_source_size, prepared_by_size;
//			char *phantom_name, *data_source, *prepared_by;
//	
//			data_file.read((char*)&projection_angle, sizeof(float));
//			data_file.read((char*)&beam_energy, sizeof(float));
//			data_file.read((char*)&generation_date, sizeof(int));
//			data_file.read((char*)&preprocess_date, sizeof(int));
//			data_file.read((char*)&phantom_name_size, sizeof(int));
//			phantom_name = (char*)malloc(phantom_name_size);
//			data_file.read(phantom_name, phantom_name_size);
//			data_file.read((char*)&data_source_size, sizeof(int));
//			data_source = (char*)malloc(data_source_size);
//			data_file.read(data_source, data_source_size);
//			data_file.read((char*)&prepared_by_size, sizeof(int));
//			prepared_by = (char*)malloc(prepared_by_size);
//			data_file.read(prepared_by, prepared_by_size);
//	
//			printf("Loading %d histories from file\n", file_histories);
//	
//			int data_size = file_histories * sizeof(float);
//	
//			data_file.read((char*)&t_in_1_h[histories_read], data_size);
//			data_file.read((char*)&t_in_2_h[histories_read], data_size);
//			data_file.read((char*)&t_out_1_h[histories_read], data_size);
//			data_file.read((char*)&t_out_2_h[histories_read], data_size);
//			data_file.read((char*)&v_in_1_h[histories_read], data_size);
//			data_file.read((char*)&v_in_2_h[histories_read], data_size);
//			data_file.read((char*)&v_out_1_h[histories_read], data_size);
//			data_file.read((char*)&v_out_2_h[histories_read], data_size);
//			data_file.read((char*)&u_in_1_h[histories_read], data_size);
//			data_file.read((char*)&u_in_2_h[histories_read], data_size);
//			data_file.read((char*)&u_out_1_h[histories_read], data_size);
//			data_file.read((char*)&u_out_2_h[histories_read], data_size);
//			data_file.read((char*)&WEPL_h[histories_read], data_size);
//	
//			double max_v = 0;
//			double min_v = 0;
//			double max_WEPL = 0;
//			double min_WEPL = 0;
//			convert_mm_2_cm( num_histories );
//			for( unsigned int i = 0; i < file_histories; i++, array_index++ ) 
//			{				
//				if( (v_in_1_h[array_index]) > max_v )
//					max_v = v_in_1_h[array_index];
//				if( (v_in_2_h[array_index]) > max_v )
//					max_v = v_in_2_h[array_index];
//				if( (v_out_1_h[array_index]) > max_v )
//					max_v = v_out_1_h[array_index];
//				if( (v_out_2_h[array_index]) > max_v )
//					max_v = v_out_2_h[array_index];
//					
//				if( (v_in_1_h[array_index]) < min_v )
//					min_v = v_in_1_h[array_index];
//				if( (v_in_2_h[array_index]) < min_v )
//					min_v = v_in_2_h[array_index];
//				if( (v_out_1_h[array_index]) < min_v )
//					min_v = v_out_1_h[array_index];
//				if( (v_out_2_h[array_index]) < min_v )
//					min_v = v_out_2_h[array_index];
//
//				if( (WEPL_h[array_index]) > max_WEPL )
//					max_WEPL = WEPL_h[array_index];
//				if( (WEPL_h[array_index]) < min_WEPL )
//					min_WEPL = WEPL_h[array_index];
//				gantry_angle_h[array_index] = (int(projection_angle) + 270)%360;
//			}
//			printf("max_WEPL = %3f\n", max_WEPL );
//			printf("min_WEPL = %3f\n", min_WEPL );
//			data_file.close();
//			histories_read += file_histories;
//		}
//	}
//}
//
///***********************************************************************************************************************************************************************************************************************/
///********************************************************************************** Routines for Writing Data Arrays/Vectors to Disk ***********************************************************************************/
///***********************************************************************************************************************************************************************************************************************/
//void binary_2_ASCII()
//{
//	count_histories();
//	char filename[256];
//	FILE* output_file;
//	uint histories_to_process = 0;
//	for( uint gantry_position = 0; gantry_position < NUM_FILES; gantry_position++ )
//	{
//		histories_to_process = histories_per_file[gantry_position];
//		read_data_chunk( histories_to_process, gantry_position, gantry_position + 1 );
//		sprintf( filename, "%s/%s%s%03d%s", PREPROCESSING_DIR, INPUT_DATA_BASENAME, "_", gantry_position, ".txt" );
//		output_file = fopen (filename, "w");
//
//		for( unsigned int i = 0; i < histories_to_process; i++ )
//		{
//			fprintf(output_file, "%3f %3f %3f %3f %3f %3f %3f %3f %3f\n", t_in_1_h[i], t_in_2_h[i], t_out_1_h[i], t_out_2_h[i], v_in_1_h[i], v_in_2_h[i], v_out_1_h[i], v_out_2_h[i], WEPL_h[i]);
//		}
//		fclose (output_file);
//		initial_processing_memory_clean();
//		histories_to_process = 0;
//	} 
//}
//template<typename T> void array_2_disk( char* filename_base, char* filepath, DISK_WRITE_MODE format, T* data, const int x_max, const int y_max, const int z_max, const int num_elements, const bool single_file )
//{
//	char filename[256];
//	std::ofstream output_file;
//	int index;
//	int z_start = 0;
//	int z_end = single_file ? z_max : 1;
//	int num_files = single_file ? 1 : z_max;
//	//if( single_file )
//	//{
//	//	num_files = 1;
//	//	z_end = z_max;
//	//}
//	for( int file = 0; file < num_files; file++)
//	{
//		//if( num_files == z_max )
//		if( !single_file )
//			sprintf( filename, "%s/%s_%d", filepath, filename_base, file );
//		else
//			sprintf( filename, "%s/%s", filepath, filename_base );			
//		switch( format )
//		{
//			case TEXT	:	sprintf( filename, "%s.txt", filename );	
//							output_file.open(filename);					break;
//			case BINARY	:	sprintf( filename, "%s.bin", filepath );
//							output_file.open(filename, std::ofstream::binary);
//		}
//		//output_file.open(filename);		
//		for(int z = z_start; z < z_end; z++)
//		{			
//			for(int y = 0; y < y_max; y++)
//			{
//				for(int x = 0; x < x_max; x++)
//				{
//					index = x + ( y * x_max ) + ( z * x_max * y_max );
//					if( index >= num_elements )
//						break;
//					output_file << data[index] << " ";
//				}	
//				if( index >= num_elements )
//					break;
//				output_file << std::endl;
//			}
//			if( index >= num_elements )
//				break;
//		}
//		z_start += 1;
//		z_end += 1;
//		output_file.close();
//	}
//}
//template<typename T> void vector_2_disk( char* filename_base, char* filepath, DISK_WRITE_MODE format, std::vector<T> data, const int x_max, const int y_max, const int z_max, const bool single_file )
//{
//	char filename[256];
//	std::ofstream output_file;
//	int elements = data.size();
//	int index;
//	int z_start = 0;
//	int z_end = single_file ? z_max : 1;
//	int num_files = single_file ? 1 : z_max;
//	//if( single_file )
//	//{
//	//	num_files = 1;
//	//	z_end = z_max;
//	//}
//	for( int file = 0; file < num_files; file++)
//	{
//		//if( num_files == z_max )
//		if( !single_file )
//			sprintf( filename, "%s/%s_%d", filepath, filename_base, file );
//		else
//			sprintf( filename, "%s/%s", filepath, filename_base );			
//		switch( format )
//		{
//			case TEXT	:	sprintf( filename, "%s.txt", filename );	
//							output_file.open(filename);					break;
//			case BINARY	:	sprintf( filename, "%s.bin", filepath );
//							output_file.open(filename, std::ofstream::binary);
//		}
//		//output_file.open(filename);		
//		for(int z = z_start; z < z_end; z++)
//		{			
//			for(int y = 0; y < y_max; y++)
//			{
//				for(int x = 0; x < x_max; x++)
//				{
//					index = x + ( y * x_max ) + ( z * x_max * y_max );
//					if( index >= elements )
//						break;
//					output_file << data[index] << " ";
//				}	
//				if( index >= elements )
//					break;
//				output_file << std::endl;
//			}
//			if( index >= elements )
//				break;
//		}
//		z_start += 1;
//		z_end += 1;
//		output_file.close();
//	}
//}
//template<typename T> void t_bins_2_disk( FILE* output_file, const std::vector<int>& bin_numbers, const std::vector<T>& data, const BIN_ANALYSIS_TYPE type, const BIN_ORGANIZATION bin_order, int bin )
//{
//	char* data_format = FLOAT_FORMAT;
//	if( typeid(T) == typeid(int) )
//		data_format = INT_FORMAT;
//	if( typeid(T) == typeid(bool))
//		data_format = BOOL_FORMAT;
//	std::vector<T> bin_histories;
//	unsigned int num_bin_members;
//	for( int t_bin = 0; t_bin < T_BINS; t_bin++, bin++ )
//	{
//		if( bin_order == BY_HISTORY )
//		{
//			for( unsigned int i = 0; i < data.size(); i++ )
//				if( bin_numbers[i] == bin )
//					bin_histories.push_back(data[i]);
//		}
//		else
//			bin_histories.push_back(data[bin]);
//		num_bin_members = bin_histories.size();
//		switch( type )
//		{
//			case COUNTS:	
//				fprintf (output_file, "%d ", num_bin_members);																			
//				break;
//			case MEANS:		
//				fprintf (output_file, "%f ", std::accumulate(bin_histories.begin(), bin_histories.end(), 0.0) / max(num_bin_members, 1 ) );
//				break;
//			case MEMBERS:	
//				for( unsigned int i = 0; i < num_bin_members; i++ )
//				{
//					//fprintf (output_file, "%f ", bin_histories[i]); 
//					fprintf (output_file, data_format, bin_histories[i]); 
//					fputs(" ", output_file);
//				}					 
//				if( t_bin != T_BINS - 1 )
//					fputs("\n", output_file);
//		}
//		bin_histories.resize(0);
//		bin_histories.shrink_to_fit();
//	}
//}
//template<typename T> void bins_2_disk( char* filename_base, char* filepath, DISK_WRITE_MODE format, const std::vector<int>& bin_numbers, const std::vector<T>& data, const BIN_ANALYSIS_TYPE type, const BIN_ANALYSIS_FOR which_bins, const BIN_ORGANIZATION bin_order, ... )
//{
//	//bins_2_disk( "WEPL_dist_pre_test2", empty_parameter, mean_WEPL_h, NUM_BINS, MEANS, ALL_BINS, BY_BIN );
//	//bins_2_disk( "WEPL_dist_pre_test2", empty_parameter, sinogram_h, NUM_BINS, MEANS, ALL_BINS, BY_BIN );
//	std::vector<int> angles;
//	std::vector<int> angular_bins;
//	std::vector<int> v_bins;
//	if( which_bins == ALL_BINS )
//	{
//		angular_bins.resize( ANGULAR_BINS);
//		v_bins.resize( V_BINS);
//		std::iota( angular_bins.begin(), angular_bins.end(), 0 );
//		std::iota( v_bins.begin(), v_bins.end(), 0 );
//	}
//	else
//	{
//		va_list specific_bins;
//		va_start( specific_bins, bin_order );
//		int num_angles = va_arg(specific_bins, int );
//		int* angle_array = va_arg(specific_bins, int* );	
//		angles.resize(num_angles);
//		std::copy(angle_array, angle_array + num_angles, angles.begin() );
//
//		int num_v_bins = va_arg(specific_bins, int );
//		int* v_bins_array = va_arg(specific_bins, int* );	
//		v_bins.resize(num_v_bins);
//		std::copy(v_bins_array, v_bins_array + num_v_bins, v_bins.begin() );
//
//		va_end(specific_bins);
//		angular_bins.resize(angles.size());
//		std::transform(angles.begin(), angles.end(), angular_bins.begin(), std::bind2nd(std::divides<int>(), GANTRY_ANGLE_INTERVAL ) );
//	}
//	
//	int num_angles = (int) angular_bins.size();
//	int num_v_bins = (int) v_bins.size();
//	/*for( unsigned int i = 0; i < 3; i++ )
//		printf("%d\n", angles[i] );
//	for( unsigned int i = 0; i < 3; i++ )
//		printf("%d\n", angular_bins[i] );
//	for( unsigned int i = 0; i < 3; i++ )
//		printf("%d\n", v_bins[i] );*/
//	char filename[256];
//	int start_bin, angle;
//	FILE* output_file;
//
//	for( int angular_bin = 0; angular_bin < num_angles; angular_bin++)
//	{
//		angle = angular_bins[angular_bin] * GANTRY_ANGLE_INTERVAL;
//		//printf("angle = %d\n", angular_bins[angular_bin]);
//		//sprintf( filename, "%s%s/%s_%03d%s", filepath, filename_base, angle, ".txt" );
//		//output_file = fopen (filename, "w");	
//		sprintf( filename, "%s/%s_%03d", filepath, filename_base, angular_bin );	
//		switch( format )
//		{
//			case TEXT	:	sprintf( filename, "%s.txt", filename );	
//							output_file = fopen (filename, "w");				break;
//			case BINARY	:	sprintf( filename, "%s.bin", filepath );
//							output_file = fopen (filename, "wb");
//		}
//		for( int v_bin = 0; v_bin < num_v_bins; v_bin++)
//		{			
//			//printf("v bin = %d\n", v_bins[v_bin]);
//			start_bin = angular_bins[angular_bin] * T_BINS + v_bins[v_bin] * ANGULAR_BINS * T_BINS;
//			t_bins_2_disk( output_file, bin_numbers, data, type, bin_order, start_bin );
//			if( v_bin != num_v_bins - 1 )
//				fputs("\n", output_file);
//		}	
//		fclose (output_file);
//	}
//}
//template<typename T> void t_bins_2_disk( FILE* output_file, int*& bin_numbers, T*& data, const unsigned int num_elements, const BIN_ANALYSIS_TYPE type, const BIN_ORGANIZATION bin_order, int bin )
//{
//	char* data_format = FLOAT_FORMAT;
//	if( typeid(T) == typeid(int) )
//		data_format = INT_FORMAT;
//	if( typeid(T) == typeid(bool))
//		data_format = BOOL_FORMAT;
//
//	std::vector<T> bin_histories;
//	//int data_elements = sizeof(data)/sizeof(float);
//	unsigned int num_bin_members;
//	for( int t_bin = 0; t_bin < T_BINS; t_bin++, bin++ )
//	{
//		if( bin_order == BY_HISTORY )
//		{
//			for( unsigned int i = 0; i < num_elements; i++ )
//				if( bin_numbers[i] == bin )
//					bin_histories.push_back(data[i]);
//		}
//		else
//			bin_histories.push_back(data[bin]);
//		num_bin_members = (unsigned int) bin_histories.size();
//		switch( type )
//		{
//			case COUNTS:	
//				fprintf (output_file, "%d ", num_bin_members);																			
//				break;
//			case MEANS:		
//				fprintf (output_file, "%f ", std::accumulate(bin_histories.begin(), bin_histories.end(), 0.0) / max(num_bin_members, 1 ) );
//				break;
//			case MEMBERS:	
//				for( unsigned int i = 0; i < num_bin_members; i++ )
//				{
//					//fprintf (output_file, "%f ", bin_histories[i]); 
//					fprintf (output_file, data_format, bin_histories[i]); 
//					fputs(" ", output_file);
//				}
//				if( t_bin != T_BINS - 1 )
//					fputs("\n", output_file);
//		}
//		bin_histories.resize(0);
//		bin_histories.shrink_to_fit();
//	}
//}
//template<typename T>  void bins_2_disk( char* filename_base, char* filepath, DISK_WRITE_MODE format, int*& bin_numbers, T*& data, const int num_elements, const BIN_ANALYSIS_TYPE type, const BIN_ANALYSIS_FOR which_bins, const BIN_ORGANIZATION bin_order, ... )
//{
//	std::vector<int> angles;
//	std::vector<int> angular_bins;
//	std::vector<int> v_bins;
//	if( which_bins == ALL_BINS )
//	{
//		angular_bins.resize( ANGULAR_BINS);
//		v_bins.resize( V_BINS);
//		std::iota( angular_bins.begin(), angular_bins.end(), 0 );
//		std::iota( v_bins.begin(), v_bins.end(), 0 );
//	}
//	else
//	{
//		va_list specific_bins;
//		va_start( specific_bins, bin_order );
//		int num_angles = va_arg(specific_bins, int );
//		int* angle_array = va_arg(specific_bins, int* );	
//		angles.resize(num_angles);
//		std::copy(angle_array, angle_array + num_angles, angles.begin() );
//
//		int num_v_bins = va_arg(specific_bins, int );
//		int* v_bins_array = va_arg(specific_bins, int* );	
//		v_bins.resize(num_v_bins);
//		std::copy(v_bins_array, v_bins_array + num_v_bins, v_bins.begin() );
//
//		va_end(specific_bins);
//		angular_bins.resize(angles.size());
//		std::transform(angles.begin(), angles.end(), angular_bins.begin(), std::bind2nd(std::divides<int>(), GANTRY_ANGLE_INTERVAL ) );
//	}
//	//int data_elements = sizeof(data)/sizeof(float);
//	//std::cout << std::endl << data_elements << std::endl << std::endl;
//	int num_angles = (int) angular_bins.size();
//	int num_v_bins = (int) v_bins.size();
//	/*for( unsigned int i = 0; i < 3; i++ )
//		printf("%d\n", angles[i] );
//	for( unsigned int i = 0; i < 3; i++ )
//		printf("%d\n", angular_bins[i] );
//	for( unsigned int i = 0; i < 3; i++ )
//		printf("%d\n", v_bins[i] );*/
//	char filename[256];
//	int start_bin, angle;
//	FILE* output_file;
//
//	for( int angular_bin = 0; angular_bin < num_angles; angular_bin++)
//	{
//		angle = angular_bins[angular_bin] * (int) GANTRY_ANGLE_INTERVAL;
//		//printf("angle = %d\n", angular_bins[angular_bin]);
//		//sprintf( filename, "%s%s/%s_%03d%s", filepath, filename_base, angle, ".txt" );
//		//output_file = fopen (filename, "w");	
//		sprintf( filename, "%s/%s_%03d", filepath, filename_base, angular_bin );	
//		switch( format )
//		{
//			case TEXT	:	sprintf( filename, "%s.txt", filename );	
//							output_file = fopen (filename, "w");				break;
//			case BINARY	:	sprintf( filename, "%s.bin", filepath );
//							output_file = fopen (filename, "wb");
//		}
//		for( int v_bin = 0; v_bin < num_v_bins; v_bin++)
//		{			
//			//printf("v bin = %d\n", v_bins[v_bin]);
//			start_bin = angular_bins[angular_bin] * T_BINS + v_bins[v_bin] * ANGULAR_BINS * T_BINS;
//			t_bins_2_disk( output_file, bin_numbers, data, num_elements, type, bin_order, start_bin );
//			if( v_bin != num_v_bins - 1 )
//				fputs("\n", output_file);
//		}	
//		fclose (output_file);
//	}
//}
//void combine_data_sets()
//{
//	char input_filename1[256];
//	char input_filename2[256];
//	char output_filename[256];
//	const char INPUT_FOLDER1[]	   = "input_CTP404";
//	const char INPUT_FOLDER2[]	   = "CTP404_4M";
//	const char MERGED_FOLDER[]	   = "my_merged";
//
//	char magic_number1[4], magic_number2[4];
//	int version_id1, version_id2;
//	int file_histories1, file_histories2, total_histories;
//
//	float projection_angle1, beam_energy1;
//	int generation_date1, preprocess_date1;
//	int phantom_name_size1, data_source_size1, prepared_by_size1;
//	char *phantom_name1, *data_source1, *prepared_by1;
//	
//	float projection_angle2, beam_energy2;
//	int generation_date2, preprocess_date2;
//	int phantom_name_size2, data_source_size2, prepared_by_size2;
//	char *phantom_name2, *data_source2, *prepared_by2;
//
//	float* t_in_1_h1, * t_in_1_h2, *t_in_2_h1, * t_in_2_h2; 
//	float* t_out_1_h1, * t_out_1_h2, * t_out_2_h1, * t_out_2_h2;
//	float* v_in_1_h1, * v_in_1_h2, * v_in_2_h1, * v_in_2_h2;
//	float* v_out_1_h1, * v_out_1_h2, * v_out_2_h1, * v_out_2_h2;
//	float* u_in_1_h1, * u_in_1_h2, * u_in_2_h1, * u_in_2_h2;
//	float* u_out_1_h1, * u_out_1_h2, * u_out_2_h1, * u_out_2_h2;
//	float* WEPL_h1, * WEPL_h2;
//
//	for( unsigned int gantry_angle = 0; gantry_angle < 360; gantry_angle += int(GANTRY_ANGLE_INTERVAL) )
//	{	
//		cout << gantry_angle << endl;
//		sprintf(input_filename1, "%s%s/%s_%03d%s", PROJECTION_DATA_DIR, INPUT_FOLDER1, INPUT_DATA_BASENAME, gantry_angle, FILE_EXTENSION );
//		sprintf(input_filename2, "%s%s/%s_%03d%s", PROJECTION_DATA_DIR, INPUT_FOLDER2, INPUT_DATA_BASENAME, gantry_angle, FILE_EXTENSION );
//		sprintf(output_filename, "%s%s/%s_%03d%s", PROJECTION_DATA_DIR, MERGED_FOLDER, INPUT_DATA_BASENAME, gantry_angle, FILE_EXTENSION );
//
//		printf("%s\n", input_filename1 );
//		printf("%s\n", input_filename2 );
//		printf("%s\n", output_filename );
//
//		FILE* input_file1 = fopen(input_filename1, "rb");
//		FILE* input_file2 = fopen(input_filename2, "rb");
//		FILE* output_file = fopen(output_filename, "wb");
//
//		if( (input_file1 == NULL) ||  (input_file2 == NULL)  || (output_file == NULL)  )
//		{
//			fputs( "Error Opening Data File:  Check that the directories are properly named.", stderr ); 
//			exit_program_if(true);
//		}
//
//		fread(&magic_number1, sizeof(char), 4, input_file1 );
//		fread(&magic_number2, sizeof(char), 4, input_file2 );
//		fwrite( &magic_number1, sizeof(char), 4, output_file );
//		//if( magic_number != MAGIC_NUMBER_CHECK ) 
//		//{
//		//	puts("Error: unknown file type (should be PCTD)!\n");
//		//	exit_program_if(true);
//		//}
//
//		fread(&version_id1, sizeof(int), 1, input_file1 );
//		fread(&version_id2, sizeof(int), 1, input_file2 );
//		fwrite( &version_id1, sizeof(int), 1, output_file );
//
//		fread(&file_histories1, sizeof(int), 1, input_file1 );
//		fread(&file_histories2, sizeof(int), 1, input_file2 );
//		total_histories = file_histories1 + file_histories2;
//		fwrite( &total_histories, sizeof(int), 1, output_file );
//
//		puts("Reading headers from files...\n");
//	
//		fread(&projection_angle1, sizeof(float), 1, input_file1 );
//		fread(&projection_angle2, sizeof(float), 1, input_file2 );
//		fwrite( &projection_angle1, sizeof(float), 1, output_file );
//			
//		fread(&beam_energy1, sizeof(float), 1, input_file1 );
//		fread(&beam_energy2, sizeof(float), 1, input_file2 );
//		fwrite( &beam_energy1, sizeof(float), 1, output_file );
//
//		fread(&generation_date1, sizeof(int), 1, input_file1 );
//		fread(&generation_date2, sizeof(int), 1, input_file2 );
//		fwrite( &generation_date1, sizeof(int), 1, output_file );
//
//		fread(&preprocess_date1, sizeof(int), 1, input_file1 );
//		fread(&preprocess_date2, sizeof(int), 1, input_file2 );
//		fwrite( &preprocess_date1, sizeof(int), 1, output_file );
//
//		fread(&phantom_name_size1, sizeof(int), 1, input_file1 );
//		fread(&phantom_name_size2, sizeof(int), 1, input_file2 );
//		fwrite( &phantom_name_size1, sizeof(int), 1, output_file );
//
//		phantom_name1 = (char*)malloc(phantom_name_size1);
//		phantom_name2 = (char*)malloc(phantom_name_size2);
//
//		fread(phantom_name1, phantom_name_size1, 1, input_file1 );
//		fread(phantom_name2, phantom_name_size2, 1, input_file2 );
//		fwrite( phantom_name1, phantom_name_size1, 1, output_file );
//
//		fread(&data_source_size1, sizeof(int), 1, input_file1 );
//		fread(&data_source_size2, sizeof(int), 1, input_file2 );
//		fwrite( &data_source_size1, sizeof(int), 1, output_file );
//
//		data_source1 = (char*)malloc(data_source_size1);
//		data_source2 = (char*)malloc(data_source_size2);
//
//		fread(data_source1, data_source_size1, 1, input_file1 );
//		fread(data_source2, data_source_size2, 1, input_file2 );
//		fwrite( &data_source1, data_source_size1, 1, output_file );
//
//		fread(&prepared_by_size1, sizeof(int), 1, input_file1 );
//		fread(&prepared_by_size2, sizeof(int), 1, input_file2 );
//		fwrite( &prepared_by_size1, sizeof(int), 1, output_file );
//
//		prepared_by1 = (char*)malloc(prepared_by_size1);
//		prepared_by2 = (char*)malloc(prepared_by_size2);
//
//		fread(prepared_by1, prepared_by_size1, 1, input_file1 );
//		fread(prepared_by2, prepared_by_size2, 1, input_file2 );
//		fwrite( &prepared_by1, prepared_by_size1, 1, output_file );
//
//		puts("Reading data from files...\n");
//
//		t_in_1_h1 = (float*)calloc( file_histories1, sizeof(float ) );
//		t_in_1_h2 = (float*)calloc( file_histories2, sizeof(float ) );
//		t_in_2_h1 = (float*)calloc( file_histories1, sizeof(float ) );
//		t_in_2_h2 = (float*)calloc( file_histories2, sizeof(float ) );
//		t_out_1_h1 = (float*)calloc( file_histories1, sizeof(float ) );
//		t_out_1_h2 = (float*)calloc( file_histories2, sizeof(float ) );
//		t_out_2_h1 = (float*)calloc( file_histories1, sizeof(float ) );
//		t_out_2_h2 = (float*)calloc( file_histories2, sizeof(float ) );
//		v_in_1_h1 = (float*)calloc( file_histories1, sizeof(float ) );
//		v_in_1_h2 = (float*)calloc( file_histories2, sizeof(float ) );		
//		v_in_2_h1 = (float*)calloc( file_histories1, sizeof(float ) );
//		v_in_2_h2 = (float*)calloc( file_histories2, sizeof(float ) );
//		v_out_1_h1 = (float*)calloc( file_histories1, sizeof(float ) );
//		v_out_1_h2 = (float*)calloc( file_histories2, sizeof(float ) );
//		v_out_2_h1 = (float*)calloc( file_histories1, sizeof(float ) );
//		v_out_2_h2 = (float*)calloc( file_histories2, sizeof(float ) );
//		u_in_1_h1 = (float*)calloc( file_histories1, sizeof(float ) );
//		u_in_1_h2 = (float*)calloc( file_histories2, sizeof(float ) );
//		u_in_2_h1 = (float*)calloc( file_histories1, sizeof(float ) );
//		u_in_2_h2 = (float*)calloc( file_histories2, sizeof(float ) );
//		u_out_1_h1 = (float*)calloc( file_histories1, sizeof(float ) );
//		u_out_1_h2 = (float*)calloc( file_histories2, sizeof(float ) );
//		u_out_2_h1 = (float*)calloc( file_histories1, sizeof(float ) );
//		u_out_2_h2 = (float*)calloc( file_histories2, sizeof(float ) );
//		WEPL_h1 = (float*)calloc( file_histories1, sizeof(float ) );
//		WEPL_h2 = (float*)calloc( file_histories2, sizeof(float ) );
//
//		fread( t_in_1_h1,  sizeof(float), file_histories1, input_file1 );
//		fread( t_in_2_h1,  sizeof(float), file_histories1, input_file1 );
//		fread( t_out_1_h1,  sizeof(float), file_histories1, input_file1 );
//		fread( t_out_2_h1,  sizeof(float), file_histories1, input_file1 );
//		fread( v_in_1_h1,  sizeof(float), file_histories1, input_file1 );
//		fread( v_in_2_h1,  sizeof(float), file_histories1, input_file1 );
//		fread( v_out_1_h1,  sizeof(float), file_histories1, input_file1 );
//		fread( v_out_2_h1,  sizeof(float), file_histories1, input_file1 );
//		fread( u_in_1_h1,  sizeof(float), file_histories1, input_file1 );
//		fread( u_in_2_h1,  sizeof(float), file_histories1, input_file1 );
//		fread( u_out_1_h1,  sizeof(float), file_histories1, input_file1 );
//		fread( u_out_2_h1,  sizeof(float), file_histories1, input_file1 );
//		fread( WEPL_h1,  sizeof(float), file_histories1, input_file1 );
//
//		fread( t_in_1_h2,  sizeof(float), file_histories2, input_file2 );
//		fread( t_in_2_h2,  sizeof(float), file_histories2, input_file2 );
//		fread( t_out_1_h2,  sizeof(float), file_histories2, input_file2 );
//		fread( t_out_2_h2,  sizeof(float), file_histories2, input_file2 );
//		fread( v_in_1_h2,  sizeof(float), file_histories2, input_file2 );
//		fread( v_in_2_h2,  sizeof(float), file_histories2, input_file2 );
//		fread( v_out_1_h2,  sizeof(float), file_histories2, input_file2 );
//		fread( v_out_2_h2,  sizeof(float), file_histories2, input_file2 );
//		fread( u_in_1_h2,  sizeof(float), file_histories2, input_file2 );
//		fread( u_in_2_h2,  sizeof(float), file_histories2, input_file2 );
//		fread( u_out_1_h2,  sizeof(float), file_histories2, input_file2 );
//		fread( u_out_2_h2,  sizeof(float), file_histories2, input_file2 );
//		fread( WEPL_h2,  sizeof(float), file_histories2, input_file2 );
//
//		fwrite( t_in_1_h1, sizeof(float), file_histories1, output_file );
//		fwrite( t_in_1_h2, sizeof(float), file_histories2, output_file );		
//		fwrite( t_in_2_h1, sizeof(float), file_histories1, output_file );
//		fwrite( t_in_2_h2, sizeof(float), file_histories2, output_file );		
//		fwrite( t_out_1_h1, sizeof(float), file_histories1, output_file );
//		fwrite( t_out_1_h2, sizeof(float), file_histories2, output_file );		
//		fwrite( t_out_2_h1, sizeof(float), file_histories1, output_file );
//		fwrite( t_out_2_h2, sizeof(float), file_histories2, output_file );	
//
//		fwrite( v_in_1_h1, sizeof(float), file_histories1, output_file );
//		fwrite( v_in_1_h2, sizeof(float), file_histories2, output_file );		
//		fwrite( v_in_2_h1, sizeof(float), file_histories1, output_file );
//		fwrite( v_in_2_h2, sizeof(float), file_histories2, output_file );		
//		fwrite( v_out_1_h1, sizeof(float), file_histories1, output_file );
//		fwrite( v_out_1_h2, sizeof(float), file_histories2, output_file );		
//		fwrite( v_out_2_h1, sizeof(float), file_histories1, output_file );
//		fwrite( v_out_2_h2, sizeof(float), file_histories2, output_file );	
//
//		fwrite( u_in_1_h1, sizeof(float), file_histories1, output_file );
//		fwrite( u_in_1_h2, sizeof(float), file_histories2, output_file );		
//		fwrite( u_in_2_h1, sizeof(float), file_histories1, output_file );
//		fwrite( u_in_2_h2, sizeof(float), file_histories2, output_file );	
//		fwrite( u_out_1_h1, sizeof(float), file_histories1, output_file );
//		fwrite( u_out_1_h2, sizeof(float), file_histories2, output_file );	
//		fwrite( u_out_2_h1, sizeof(float), file_histories1, output_file );
//		fwrite( u_out_2_h2, sizeof(float), file_histories2, output_file );	
//
//		fwrite( WEPL_h1, sizeof(float), file_histories1, output_file );
//		fwrite( WEPL_h2, sizeof(float), file_histories2, output_file );
//		
//		free( t_in_1_h1 );
//		free( t_in_1_h2 );
//		free( t_in_2_h1 );
//		free( t_in_2_h2 );
//		free( t_out_1_h1 );
//		free( t_out_1_h2 );
//		free( t_out_2_h1 );
//		free( t_out_2_h2 );
//
//		free( v_in_1_h1 );
//		free( v_in_1_h2 );
//		free( v_in_2_h1 );
//		free( v_in_2_h2 );
//		free( v_out_1_h1 );
//		free( v_out_1_h2 );
//		free( v_out_2_h1 );
//		free( v_out_2_h2 );
//
//		free( u_in_1_h1 );
//		free( u_in_1_h2 );
//		free( u_in_2_h1 );
//		free( u_in_2_h2 );
//		free( u_out_1_h1 );
//		free( u_out_1_h2 );
//		free( u_out_2_h1 );
//		free( u_out_2_h2 );
//
//		free( WEPL_h1 );
//		free( WEPL_h2 );
//
//		fclose(input_file1);						
//		fclose(input_file2);	
//		fclose(output_file);	
//
//		puts("Finished");
//		pause_execution();
//	}
//
//}
///***********************************************************************************************************************************************************************************************************************/
///************************************************************************************************ Host Helper Functions ************************************************************************************************/
///***********************************************************************************************************************************************************************************************************************/
//template<typename T, typename T2> T max_n( int num_args, T2 arg_1, ...)
//{
//	T2 largest = arg_1;
//	T2 value;
//	va_list values;
//	va_start( values, arg_1 );
//	for( int i = 1; i < num_args; i++ )
//	{
//		value = va_arg( values, T2 );
//		largest = ( largest > value ) ? largest : value;
//	}
//	va_end(values);
//	return (T) largest; 
//}
//template<typename T, typename T2> T min_n( int num_args, T2 arg_1, ...)
//{
//	T2 smallest = arg_1;
//	T2 value;
//	va_list values;
//	va_start( values, arg_1 );
//	for( int i = 1; i < num_args; i++ )
//	{
//		value = va_arg( values, T2 );
//		smallest = ( smallest < value ) ? smallest : value;
//	}
//	va_end(values);
//	return (T) smallest; 
//}
//void timer( bool start, clock_t start_time, clock_t end_time)
//{
//	if( start )
//		start_time = clock();
//	else
//	{
//		end_time = clock();
//		clock_t execution_clock_cycles = (end_time - start_time) - pause_cycles;
//		double execution_time = double( execution_clock_cycles) / CLOCKS_PER_SEC;
//		printf( "Total execution time : %3f [seconds]\n", execution_time );	
//	}
//}
//void pause_execution()
//{
//	clock_t pause_start, pause_end;
//	pause_start = clock();
//	//char user_response[20];
//	puts("Execution paused.  Hit enter to continue execution.\n");
//	 //Clean the stream and ask for input
//	//std::cin.ignore ( std::numeric_limits<std::streamsize>::max(), '\n' );
//	std::cin.get();
//
//	pause_end = clock();
//	pause_cycles += pause_end - pause_start;
//}
//void exit_program_if( bool early_exit)
//{
//	if( early_exit )
//	{
//		char user_response[20];
//		timer( STOP, program_start, program_end );
//		puts("Hit enter to stop...");
//		fgets(user_response, sizeof(user_response), stdin);
//		exit(1);
//	}
//}
//template<typename T> T* sequential_numbers( int start_number, int length )
//{
//	T* sequential_array = (T*)calloc(length,sizeof(T));
//	std::iota( sequential_array, sequential_array + length, start_number );
//	return sequential_array;
//}
//void bin_2_indexes( int& bin_num, int& t_bin, int& v_bin, int& angular_bin )
//{
//	// => bin = t_bin + angular_bin * T_BINS + v_bin * ANGULAR_BINS * T_BINS > 0
//	while( bin_num - ANGULAR_BINS * T_BINS > 0 )
//	{
//		bin_num -= ANGULAR_BINS * T_BINS;
//		v_bin++;
//	}
//	// => bin = t_bin + angular_bin * T_BINS > 0
//	while( bin_num - T_BINS > 0 )
//	{
//		bin_num -= T_BINS;
//		angular_bin++;
//	}
//	// => bin = t_bin > 0
//	t_bin = bin_num;
//}