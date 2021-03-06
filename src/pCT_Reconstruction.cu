//#ifndef PCT_RECONSTRUCTION_CU
//#define PCT_RECONSTRUCTION_CU
#pragma once
/***********************************************************************************************************************************************************************************************************************/
/********************************************************************************** Proton CT Preprocessing and Image Reconstruction Code ******************************************************************************/
/***********************************************************************************************************************************************************************************************************************/
#include "pCT_Reconstruction.h"
//#include "C:\Users\Blake\Documents\Visual Studio 2010\Projects\robust_pct\robust_pct\RMS_Image_Analysis.cu"
//#include "C:\Users\Blake\Documents\Visual Studio 2010\Projects\robust_pct\robust_pct\RMS_Image_Analysis.cpp"
#include "C:\Users\Blake\Documents\GitHub\pCT_Reconstruction\Configurations.h"
//#include "C:\Users\Blake\Documents\GitHub\pCT_Reconstruction\Globals.h"
//#include "C:\Users\Blake\Documents\GitHub\pCT_Reconstruction\Constants.h"

// Includes, CUDA project
//#include <cutil_inline.h>

// Includes, kernels
//#include "pCT_Reconstruction_GPU.cu" 

/***********************************************************************************************************************************************************************************************************************/
/***************************************************************************************************** Program Main ****************************************************************************************************/
/***********************************************************************************************************************************************************************************************************************/
int main( unsigned int num_arguments, char** arguments )
{	
	//configure_execution( num_arguments, arguments );
	if( RUN_ON )
	{
		if( parameters.IMPORT_PREPROCESSING )
		{
			import_hull();
			import_x_0();
			puts("Reading hull entry/exit coordinates from disk...");
			reconstruction_histories = import_histories();
		}
		//pause_execution();
		else
		{
			/********************************************************************************************************************************************************/
			/* Start the execution timing clock																														*/
			/********************************************************************************************************************************************************/
			timer( START, program_start, program_end );
			/********************************************************************************************************************************************************/
			/* Initialize hull detection images and transfer them to the GPU (performed if parameters.SC_ON, parameters.MSC_ON, or parameters.SM_ON is true)											*/
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
			reserve_vector_capacity();		// Reserve enough memory so vectors don't grow into another reserved memory space, wasting time since they must be moved
			/********************************************************************************************************************************************************/
			/* Iteratively Read and Process Data One Chunk at a Time. There are at Most	parameters.MAX_GPU_HISTORIES Per Chunk (i.e. Iteration). On Each Iteration:			*/
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
			uint start_file_num = 0, end_file_num = 0, histories_to_process = 0;
			while( start_file_num != parameters.NUM_FILES )
			{
				while( end_file_num < parameters.NUM_FILES )
				{
					if( histories_to_process + histories_per_file[end_file_num] < parameters.MAX_GPU_HISTORIES )
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
			puts("Data reading complete.");
			printf("%d out of %d (%4.2f%%) histories traversed the reconstruction volume\n", recon_vol_histories, total_histories, static_cast<double>( recon_vol_histories / total_histories * 100)  );
			exit_program_if( parameters.EXIT_AFTER_BINNING );
			/********************************************************************************************************************************************************/
			/* Reduce vector capacities to their size, the number of histories remaining after histories that didn't intersect reconstruction volume were ignored	*/																				
			/********************************************************************************************************************************************************/		
			shrink_vectors( recon_vol_histories );
			/********************************************************************************************************************************************************/
			/* Perform thresholding on MSC and SM hulls and write all hull images to file																			*/																					
			/********************************************************************************************************************************************************/
			hull_detection_finish();
			exit_program_if( parameters.EXIT_AFTER_HULLS );
			/********************************************************************************************************************************************************/
			/* Calculate the mean WEPL, relative ut-angle, and relative uv-angle for each bin and count the number of histories in each bin							*/											
			/********************************************************************************************************************************************************/
			calculate_means();
			initialize_stddev();
			/********************************************************************************************************************************************************/
			/* Calculate the standard deviation in WEPL, relative ut-angle, and relative uv-angle for each bin.  Iterate through the valid history std::vectors one	*/
			/* chunk at a time, with at most parameters.MAX_GPU_HISTORIES per chunk, and calculate the difference between the mean WEPL and WEPL, mean relative ut-angle and	*/ 
			/* relative ut-angle, and mean relative uv-angle and relative uv-angle for each history. The standard deviation is then found by calculating the sum	*/
			/* of these differences for each bin and dividing it by the number of histories in the bin 																*/
			/********************************************************************************************************************************************************/
			puts("Calculating the cumulative sum of the squared deviation in WEPL and relative ut/uv angles over all histories for each bin...");
			int remaining_histories = recon_vol_histories;
			int start_position = 0;
			while( remaining_histories > 0 )
			{
				if( remaining_histories > parameters.MAX_CUTS_HISTORIES )
					histories_to_process = parameters.MAX_CUTS_HISTORIES;
				else
					histories_to_process = remaining_histories;
				sum_squared_deviations( start_position, histories_to_process );
				remaining_histories -= parameters.MAX_CUTS_HISTORIES;
				start_position		+= parameters.MAX_CUTS_HISTORIES;
			} 
			calculate_standard_deviations();
			/********************************************************************************************************************************************************/
			/* Allocate host memory for the sinogram, initialize it to zeros, allocate memory for it on the GPU, then transfer the initialized sinogram to the GPU	*/
			/********************************************************************************************************************************************************/
			initialize_sinogram();
			/********************************************************************************************************************************************************/
			/* Iterate through the valid history vectors one chunk at a time, with at most parameters.MAX_GPU_HISTORIES per chunk, and perform statistical cuts				*/
			/********************************************************************************************************************************************************/
			puts("Performing statistical cuts...");
			remaining_histories = recon_vol_histories, start_position = 0;
			while( remaining_histories > 0 )
			{
				if( remaining_histories > parameters.MAX_CUTS_HISTORIES )
					histories_to_process = parameters.MAX_CUTS_HISTORIES;
				else
					histories_to_process = remaining_histories;
				statistical_cuts( start_position, histories_to_process );
				remaining_histories -= parameters.MAX_CUTS_HISTORIES;
				start_position		+= parameters.MAX_CUTS_HISTORIES;
			}
			puts("Statistical cuts complete.");
			printf("%d out of %d (%4.2f%%) histories also passed statistical cuts\n", post_cut_histories, total_histories, static_cast<double>( post_cut_histories / total_histories * 100 ) );
			/********************************************************************************************************************************************************/
			/* Free host memory for bin number array, free GPU memory for the statistics arrays, and shrink svectors to the number of histories that passed cuts	*/
			/********************************************************************************************************************************************************/		
			post_cut_memory_clean();
			resize_vectors( post_cut_histories );
			shrink_vectors( post_cut_histories );
			exit_program_if( parameters.EXIT_AFTER_CUTS );
			/********************************************************************************************************************************************************/
			/* Recalculate the mean WEPL for each bin using	the histories remaining after cuts and use these to produce the sinogram								*/
			/********************************************************************************************************************************************************/
			construct_sinogram();
			exit_program_if( parameters.EXIT_AFTER_SINOGRAM );
			/********************************************************************************************************************************************************/
			/* Perform filtered backprojection and write FBP hull to disk																							*/
			/********************************************************************************************************************************************************/
			if( parameters.FBP_ON )
				FBP();
			exit_program_if( parameters.EXIT_AFTER_FBP );
			define_hull();
			define_x_0();
			print_section_exit("Preprocessing complete", "====>" );
		}
		if( parameters.PERFORM_RECONSTRUCTION )  
		{
			if( parameters.MLP_IN_LOOP )
			{

			}
			else
			{
				acquire_preprocessed_data();
				//image_reconstruction_import_preprocessing_blocks();
				image_reconstruction_import_preprocessing();
			}
			print_section_exit("Reconstruction complete", "====>" );
		}			
	}
	else
	{
		test_func();
		//double RMS_error = perform_RMS_analysis( arguments[1], arguments[2], arguments[3], false );
		//cout << "RMS error = " << RMS_error << endl;
	}
	/************************************************************************************************************************************************************/
	/* Program has finished execution. Require the user to hit enter to terminate the program and close the terminal/console window								*/ 															
	/************************************************************************************************************************************************************/
	//puts("-------------------------------------------------------------------------------");
	//puts("----------------------- Program has finished executing ------------------------");
	//puts("-------------------------------------------------------------------------------\n");
	print_section_header("Program has finished executing", '-' );
	if( !parameters.STDOUT_2_DISK || !parameters.USER_INPUT_REQUESTS_OFF )
		exit_program_if(true);
	//apply_permissions();
}
/***********************************************************************************************************************************************************************************************************************/
/**************************************************************************************** t/v conversions and energy calibrations **************************************************************************************/
/***********************************************************************************************************************************************************************************************************************/
void read_energy_responses( const int num_histories, const int start_file_num, const int end_file_num )
{
	
	//char data_filename[128];
	//char magic_number[5];
	//int version_id;
	//int file_histories;
	//float projection_angle, beam_energy;
	//int generation_date, preprocess_date;
	//int phantom_name_size, data_source_size, prepared_by_size;
	//char *phantom_name, *data_source, *prepared_by;
	//int data_size;
	////int gantry_position, gantry_angle, scan_histories;
	//int gantry_position, gantry_angle, scan_number, scan_histories;
	////int array_index = 0;
	//FILE* input_file;

	//puts("Reading energy detector responses and performing energy response calibration...");
	////printf("Reading File for Gantry Angle %d from Scan Number %d...\n", gantry_angle, scan_number );
	//sprintf(data_filename, "%s%s/%s_%03d%s", PROJECTION_DATA_DIR, INPUT_FOLDER, PROJECTION_DATA_BASENAME, gantry_angle, PROJECTION_DATA_FILE_EXTENSION );
}
/***********************************************************************************************************************************************************************************************************************/
/********************************************************** Read and set execution arguments, preprocessing/reconstruction configurations, settings, and parameters ****************************************************/
/***********************************************************************************************************************************************************************************************************************/
void apply_execution_arguments(unsigned int num_arguments, char** arguments)
{
	NUM_RUN_ARGUMENTS = num_arguments;
	RUN_ARGUMENTS = arguments; 
	uint config_path_index, i = 1;
	if( NUM_RUN_ARGUMENTS % 2 == 0 )
	{
		CONFIG_PATH_PASSED = true;
		NUM_PARAMETERS_2_CHANGE = (NUM_RUN_ARGUMENTS / 2) - 1;	
	}
	else
	{
		CONFIG_PATH_PASSED = false;
		NUM_PARAMETERS_2_CHANGE = NUM_RUN_ARGUMENTS / 2;
	}
	for( ; i <= NUM_PARAMETERS_2_CHANGE; i++ );
	for( unsigned int j = 1; j < NUM_RUN_ARGUMENTS; j++ )
		cout << RUN_ARGUMENTS[j] << endl;
	
	// n =				  1			   1		   2			2		   3		   3	 ...	 n			 n       
	// i =	 0			  1			   2		   3			4		   5		   6	 ...   2n - 1  		 2n		  2n + 1
	// [program name][parameter 1][new val 1][parameter 2][new val 2][parameter 3][new val 3]...[parameter n][new val n][cfg path]
	//"C:\Users\Blake\Documents\Visual Studio 2010\Projects\robust_pct\robust_pct\settings.cfg"
	//"C:\Users\Blake\Documents\pCT_Data\object_name\Experimental\MMDDYYYY\run_number\Output\MMDDYYYY\Reconstruction\MMDDYYYY\settings.cfg"
	//"C:\Users\Blake\Documents\pCT_Data\object_name\Experimental\MMDDYYYY\run_number\Output\MMDDYYYY"
	if( CONFIG_PATH_PASSED )
	{
		config_path_index = 2 * i - 1;
		PROJECTION_DATA_DIR = (char*) calloc( strlen(RUN_ARGUMENTS[config_path_index]) + 1, sizeof(char));
		std::copy( RUN_ARGUMENTS[config_path_index], &RUN_ARGUMENTS[config_path_index][strlen(RUN_ARGUMENTS[config_path_index])], PROJECTION_DATA_DIR );
		//puts(PROJECTION_DATA_DIR);
		print_section_header("Config file location passed as command line argument and set to : ",'*');
		print_section_separator('-');
		printf("%s\n", PROJECTION_DATA_DIR );
		print_section_separator('-');
	}
}
void configure_execution( unsigned int num_arguments, char** arguments )
{
	//print_copyright_notice();
	set_execution_date();
	apply_execution_arguments( num_arguments, arguments );
	CONFIG_OBJECT config_object = config_file_2_object();			
	read_config_file();
	set_dependent_parameters();
	set_IO_file_extensions();
	set_IO_directories();
	set_IO_filenames();
	set_IO_filepaths();
	set_images_2_use();
	existing_data_check();
	parameters_2_GPU();
	if( parameters.STDOUT_2_DISK )
	{
		freopen(STDOUT_FILENAME,"w+",stdout);
		freopen(STDIN_FILENAME,"w+",stdin);
		freopen(STDERR_FILENAME,"w+",stderr);
	}
	print_section_exit("Finished reading configurations and setting program options and parameters", "====>" );
		
}
void apply_permissions()
{
	system("cd /local/organized_data");
	system("chmod -R 777 *");
	system("cd /local/pct_code");
	system("chmod -R 777 *");
	system("cd /local/pct_code/Blake");
}
/***********************************************************************************************************************************************************************************************************************/
/************************************************************************************** Memory Transfers, Maintenance, and Cleaning ************************************************************************************/
/***********************************************************************************************************************************************************************************************************************/
void initializations()
{
	puts("Allocating statistical analysis arrays on host/GPU...");

	histories_per_scan					= (int*) calloc( parameters.NUM_SCANS,	 sizeof(int)	);
	histories_per_file					= (int*) calloc( parameters.NUM_FILES,	 sizeof(int) );
	histories_per_gantry_angle			= (int*) calloc( parameters.GANTRY_ANGLES, sizeof(int) );
	recon_vol_histories_per_projection	= (int*) calloc( parameters.GANTRY_ANGLES, sizeof(int) );

	bin_counts_h			= (int*)	calloc( parameters.NUM_BINS,	sizeof(int)	);
	mean_WEPL_h				= (float*)	calloc( parameters.NUM_BINS,	sizeof(float) );
	mean_rel_ut_angle_h		= (float*)	calloc( parameters.NUM_BINS,	sizeof(float) );
	mean_rel_uv_angle_h		= (float*)	calloc( parameters.NUM_BINS,	sizeof(float) );
	
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
	bin_number_vector.reserve( total_histories );
	gantry_angle_vector.reserve( total_histories );
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
	free( WEPL_h );
	free( gantry_angle_h );
	cudaFree( x_entry_d );
	cudaFree( y_entry_d );
	cudaFree( z_entry_d );
	cudaFree( x_exit_d );
	cudaFree( y_exit_d );
	cudaFree( z_exit_d );
	cudaFree( missed_recon_volume_d );
	cudaFree( bin_number_d );
	cudaFree( WEPL_d);
}
void resize_vectors( unsigned int new_size )
{
	bin_number_vector.resize( new_size );
	gantry_angle_vector.resize( new_size );
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
void shrink_vectors( unsigned int new_capacity )
{
	bin_number_vector.shrink_to_fit();
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
void initialize_stddev()
{	
	stddev_rel_ut_angle_h = (float*) calloc( parameters.NUM_BINS, sizeof(float) );	
	stddev_rel_uv_angle_h = (float*) calloc( parameters.NUM_BINS, sizeof(float) );	
	stddev_WEPL_h		  = (float*) calloc( parameters.NUM_BINS, sizeof(float) );
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
	puts("Freeing unnecessary memory, resizing vectors, and shrinking vectors to fit just the remaining histories...");

	//free(failed_cuts_h );
	free(stddev_rel_ut_angle_h);
	free(stddev_rel_uv_angle_h);
	free(stddev_WEPL_h);

	//cudaFree( failed_cuts_d );
	//cudaFree( bin_number_d );
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
void assign_SSD_positions()	//HERE THE COORDINATES OF THE DETECTORS PLANES ARE LOADED, THE CONFIG FILE IS CREATED BY FORD (RWS)
{
	char user_response[20];
	char configFilename[512];
	puts("Reading tracker plane positions...");

	sprintf(configFilename, "%s\\scan.cfg", PREPROCESSING_DIR);
	if( parameters.DEBUG_TEXT_ON )
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
	if( parameters.DEBUG_TEXT_ON )
		puts("Reading Tracking Plane Positions...");
	for( unsigned int i = 0; i < 8; i++ ) {
		configFile >> SSD_u_Positions[i];
		if( parameters.DEBUG_TEXT_ON )
			printf("SSD_u_Positions[%d] = %3f", i, SSD_u_Positions[i]);
	}
	
	configFile.close();

}
void count_histories()
{
	for( uint scan_number = 0; scan_number < parameters.NUM_SCANS; scan_number++ )
		histories_per_scan[scan_number] = 0;

	histories_per_file =				 (int*) calloc( parameters.NUM_SCANS * parameters.GANTRY_ANGLES, sizeof(int) );
	histories_per_gantry_angle =		 (int*) calloc( parameters.GANTRY_ANGLES, sizeof(int) );
	recon_vol_histories_per_projection = (int*) calloc( parameters.GANTRY_ANGLES, sizeof(int) );

	if( parameters.DEBUG_TEXT_ON )
		puts("Counting proton histories...\n");
	switch( DATA_FORMAT )
	{
		case VERSION_0  : count_histories_v0();		break;
	}
	if( parameters.DEBUG_TEXT_ON )
	{
		for( uint file_number = 0, gantry_position_number = 0; file_number < (parameters.NUM_SCANS * parameters.GANTRY_ANGLES); file_number++, gantry_position_number++ )
		{
			if( file_number % parameters.NUM_SCANS == 0 )
				printf("There are a Total of %d Histories From Gantry Angle %d\n", histories_per_gantry_angle[gantry_position_number], static_cast<int>(gantry_position_number * parameters.GANTRY_ANGLE_INTERVAL) );			
			printf("* %d Histories are From Scan Number %d\n", histories_per_file[file_number], (file_number % parameters.NUM_SCANS) + 1 );
			
		}
		for( uint scan_number = 0; scan_number < parameters.NUM_SCANS; scan_number++ )
			printf("There are a Total of %d Histories in Scan Number %d \n", histories_per_scan[scan_number], scan_number + 1);
		printf("There are a Total of %d Histories\n", total_histories);
	}
}
void count_histories_v0()
{
	char data_filename[256];
	float projection_angle;
	 
	//unsigned int magic_number, num_histories, file_number = 0, gantry_position_number = 0;
	unsigned int num_histories, file_number = 0, gantry_position_number = 0;
	char magic_number[4];
	for( unsigned int gantry_angle = 0; gantry_angle < 360; gantry_angle += static_cast<int>(parameters.GANTRY_ANGLE_INTERVAL), gantry_position_number++ )
	{
		for( unsigned int scan_number = 1; scan_number <= parameters.NUM_SCANS; scan_number++, file_number++ )
		{
			sprintf(data_filename, "%s/%s_%03d%s", PROJECTION_DATA_DIR, PROJECTION_DATA_BASENAME, gantry_angle, PROJECTION_DATA_FILE_EXTENSION  );
			//sprintf(data_filename, "%s/%s_%03d%s", PROJECTION_DATA_DIR, PROJECTION_DATA_BASENAME, gantry_position_number, PROJECTION_DATA_FILE_EXTENSION  );
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
				log_write_test();
				exit_program_if(true);
			}
			
			fread(&magic_number, 4, 1, data_file );
			//if( magic_number != MAGIC_NUMBER_CHECK ) 
			//if( strcmp(magic_number, MAGIC_NUMBER_CHARS ) != 0 ) 
			if( std::string(magic_number) != MAGIC_NUMBER_STRING ) 
			{
				puts("Error: unknown file type (should be PCTD)!\n");
				exit_program_if(true);
			}

			fread(&VERSION_ID, sizeof(int), 1, data_file );		
			
			if( VERSION_ID == 0 )
			{
				DATA_FORMAT	= VERSION_0;
				fread(&num_histories, sizeof(int), 1, data_file );
				if( parameters.DEBUG_TEXT_ON )
					printf("There are %d Histories for Gantry Angle %d From Scan Number %d\n", num_histories, gantry_angle, scan_number);
				histories_per_file[file_number] = num_histories;
				histories_per_gantry_angle[gantry_position_number] += num_histories;
				histories_per_scan[scan_number-1] += num_histories;
				total_histories += num_histories;
			
				fread(&PROJECTION_ANGLE, sizeof(float), 1, data_file );
				projection_angles.push_back(PROJECTION_ANGLE);
				fread(&BEAM_ENERGY_IN, sizeof(float), 1, data_file );
				fread(&GENERATION_DATE, sizeof(int), 1, data_file );
				fread(&CALIBRATION_DATE, sizeof(int), 1, data_file );

				fread(&PHANTOM_NAME_SIZE, sizeof(int), 1, data_file );
				PHANTOM_NAME = (char*)malloc(PHANTOM_NAME_SIZE );
				fread(PHANTOM_NAME,  PHANTOM_NAME_SIZE, 1, data_file );
				//char* ACQUIRED_BY, * CALIBRATED_BY, * PREPROCESSED_BY, * RECONSTRUCTED_BY, * COMMENTS;
				fread(&DATA_SOURCE_SIZE, sizeof(int), 1, data_file );
				DATA_SOURCE = (char*)malloc(DATA_SOURCE_SIZE );
				fread(DATA_SOURCE, DATA_SOURCE_SIZE, 1, data_file );

				fread(&CALIBRATED_BY_SIZE, sizeof(int), 1, data_file );
				CALIBRATED_BY = (char*)malloc(ACQUIRED_BY_SIZE );
				fread(CALIBRATED_BY, CALIBRATED_BY_SIZE, 1, data_file );

				fclose(data_file);
				SKIP_2ATA_SIZE = sizeof(magic_number) + sizeof(VERSION_ID) + sizeof(num_histories) + sizeof(PROJECTION_ANGLE) + sizeof(BEAM_ENERGY_IN) + sizeof(GENERATION_DATE) + sizeof(CALIBRATION_DATE) + sizeof(PHANTOM_NAME_SIZE) + PHANTOM_NAME_SIZE + sizeof(DATA_SOURCE_SIZE) + DATA_SOURCE_SIZE + sizeof(CALIBRATED_BY_SIZE) + CALIBRATED_BY_SIZE;
				//printf("%d\n", SKIP_2ATA_SIZE);
				//SKIP_2ATA_SIZE = 4 + 7 * sizeof(int) + 2 * sizeof(float) + PHANTOM_NAME_SIZE + DATA_SOURCE_SIZE + CALIBRATED_BY_SIZE;
				//printf("%d\n", SKIP_2ATA_SIZE);
			}
			else if( VERSION_ID == 1 )
			{
				DATA_FORMAT = VERSION_1;
				fread(&num_histories, sizeof(int), 1, data_file );
				if( parameters.DEBUG_TEXT_ON )
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
				fread(&CALIBRATED_BY_SIZE, sizeof(int), 1, data_file );

				fseek( data_file, CALIBRATED_BY_SIZE, SEEK_CUR );
				fclose(data_file);
				SKIP_2ATA_SIZE = 4 + 7 * sizeof(int) + 2 * sizeof(float) + PHANTOM_NAME_SIZE + DATA_SOURCE_SIZE + CALIBRATED_BY_SIZE;
				//pause_execution();
			}
			else 
			{
				DATA_FORMAT = OLD_FORMAT;
				printf("ERROR: Data format is not Version (%d)!\n", VERSION_ID);
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
	image = (T*)calloc( parameters.NUM_VOXELS, sizeof(T));
}
template<typename T> void add_ellipse( T*& image, int slice, double x_center, double y_center, double semi_major_axis, double semi_minor_axis, T value )
{
	double x, y;
	for( int row = 0; row < parameters.ROWS; row++ )
	{
		for( int column = 0; column < parameters.COLUMNS; column++ )
		{
			x = ( column - parameters.COLUMNS/2 + 0.5) * parameters.VOXEL_WIDTH;
			y = ( parameters.ROWS/2 - row - 0.5 ) * parameters.VOXEL_HEIGHT;
			if( pow( ( x - x_center) / semi_major_axis, 2 ) + pow( ( y - y_center )  / semi_minor_axis, 2 ) <= 1 )
				image[slice * parameters.COLUMNS * parameters.ROWS + row * parameters.COLUMNS + column] = value;
		}
	}
}
template<typename T> void add_circle( T*& image, int slice, double x_center, double y_center, double radius, T value )
{
	double x, y;
	for( int row = 0; row < parameters.ROWS; row++ )
	{
		for( int column = 0; column < parameters.COLUMNS; column++ )
		{
			x = ( column - parameters.COLUMNS/2 + 0.5) * parameters.VOXEL_WIDTH;
			//x_center = ( center_column - parameters.COLUMNS/2 + 0.5) * parameters.VOXEL_WIDTH;
			y = ( parameters.ROWS/2 - row - 0.5 ) * parameters.VOXEL_HEIGHT;
			//y_center = ( center_row - parameters.COLUMNS/2 + 0.5) * parameters.VOXEL_WIDTH;
			if( pow( (x - x_center), 2 ) + pow( (y - y_center), 2 ) <= pow( radius, 2) )
				image[slice * parameters.COLUMNS * parameters.ROWS + row * parameters.COLUMNS + column] = value;
		}
	}
}	
template<typename O> unsigned int import_image( O*& import_into, char* directory, char* filename_base, DISK_WRITE_MODE format )
{
	char filename[256];
	FILE* input_file;	
	std::ifstream file_stream;
	int voxels = 0;
	long int end_file;
	O value;
	clock_t start, stop, clock_cycles;
	start = clock();
	switch( format )
	{
		case TEXT	:	puts("Reading text image from disk to array");
						sprintf( filename, "%s/%s.txt", directory, filename_base  );	
						file_stream.open(filename);
						while( file_stream >>  value )
							voxels++;
						file_stream.close();
						file_stream.open(filename);
						import_into = (O*)calloc(voxels, sizeof(O) );
						voxels = 0;
						//while( file_stream >>  import_into[voxels++] );	
						while( file_stream >>  value )
							import_into[voxels++] = value;
						file_stream.close();
						//cout << voxels << endl;
						stop = clock();
						clock_cycles = stop - start;
						cout << "text image to array time = " << double( clock_cycles) / CLOCKS_PER_SEC << endl;
						break;
		case BINARY	:	puts("Reading binary image from disk to array");
						sprintf( filename, "%s/%s.bin", directory, filename_base );
						input_file = fopen(filename, "rb" );
						fseek (input_file, 0, SEEK_END);   // non-portable
						end_file = ftell (input_file);
						voxels = end_file / sizeof(O);
						fread(import_into, sizeof(O), voxels, input_file );
						fclose(input_file);
						stop = clock();
						clock_cycles = stop - start;
						cout << "binary image to array time = " << double( clock_cycles) / CLOCKS_PER_SEC << endl;
	}
	
	return voxels;
}
template<typename O> unsigned int import_image( std::vector<O>& import_into, char* directory, char* filename_base, DISK_WRITE_MODE format )
{
	char filename[256];
	FILE* input_file;	
	std::ifstream file_stream;
	int voxels = 0;
	long int end_file;
	O value;
	clock_t start, stop, clock_cycles;
	start = clock();
	switch( format )
	{
		case TEXT	:	puts("Reading text image from disk to vector");
						sprintf( filename, "%s/%s.txt", directory, filename_base  );	
						file_stream.open(filename);
						while( file_stream >>  value )
							import_into.push_back( value);
						file_stream.close();
						stop = clock();
						clock_cycles = stop - start;
						cout << "text image to vector time = " << double( clock_cycles) / CLOCKS_PER_SEC << endl;
						break;
		case BINARY	:	puts("Reading binary image from disk to vector");
						sprintf( filename, "%s/%s.bin", directory, filename_base );
						input_file = fopen(filename, "rb" );
						fseek (input_file, 0, SEEK_END);   // non-portable
						end_file = ftell (input_file);
						voxels = end_file / sizeof(O);
						import_into.resize(voxels);
						fread(&import_into[0], sizeof(O), voxels, input_file );
						fclose(input_file);
						stop = clock();
						clock_cycles = stop - start;
						cout << "binary image to vector time = " << double( clock_cycles) / CLOCKS_PER_SEC << endl;
	}
	return import_into.size();
}
template<typename T> void binary_2_txt_images( char* path, char* filename, T*& image )
{
	print_section_header( "Importing binary image and writing a copy to disk in text format...", '*' );	
	image = (T*) calloc( parameters.NUM_VOXELS, sizeof(T) );
	import_image( image, path, filename, BINARY );
	array_2_disk(filename, path, TEXT, image, parameters.COLUMNS, parameters.ROWS, parameters.SLICES, parameters.NUM_VOXELS, true );
	puts("Binary image has been imported and a text image has been generated and written to disk with the same filename.");
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
	}

}
void apply_tuv_shifts( unsigned int num_histories)
{
	for( unsigned int i = 0; i < num_histories; i++ ) 
	{
		// Correct for any shifts in u/t coordinates
		t_in_1_h[i]	 += parameters.T_SHIFT;
		t_in_2_h[i]	 += parameters.T_SHIFT;
		t_out_1_h[i] += parameters.T_SHIFT;
		t_out_2_h[i] += parameters.T_SHIFT;
		u_in_1_h[i]	 += parameters.U_SHIFT;
		u_in_2_h[i]	 += parameters.U_SHIFT;
		u_out_1_h[i] += parameters.U_SHIFT;
		u_out_2_h[i] += parameters.U_SHIFT;
		v_in_1_h[i]	 += parameters.V_SHIFT;
		v_in_2_h[i]	 += parameters.V_SHIFT;
		v_out_1_h[i] += parameters.V_SHIFT;
		v_out_2_h[i] += parameters.V_SHIFT;
		if( parameters.WRITE_SSD_ANGLES )
		{
			ut_entry_angle[i] = atan2( t_in_2_h[i] - t_in_1_h[i], u_in_2_h[i] - u_in_1_h[i] );	
			uv_entry_angle[i] = atan2( v_in_2_h[i] - v_in_1_h[i], u_in_2_h[i] - u_in_1_h[i] );	
			ut_exit_angle[i] = atan2( t_out_2_h[i] - t_out_1_h[i], u_out_2_h[i] - u_out_1_h[i] );	
			uv_exit_angle[i] = atan2( v_out_2_h[i] - v_out_1_h[i], u_out_2_h[i] - u_out_1_h[i] );	
		}
	}
	if( parameters.WRITE_SSD_ANGLES )
	{
		char data_filename[256];
		sprintf(data_filename, "%s_%03d", "ut_entry_angle", gantry_angle_h );
		array_2_disk( data_filename, PREPROCESSING_DIR, TEXT, ut_entry_angle, parameters.COLUMNS, parameters.ROWS, parameters.SLICES, num_histories, true );
		sprintf(data_filename, "%s_%03d", "uv_entry_angle", gantry_angle_h );
		array_2_disk( data_filename, PREPROCESSING_DIR, TEXT, uv_entry_angle, parameters.COLUMNS, parameters.ROWS, parameters.SLICES, num_histories, true );
		sprintf(data_filename, "%s_%03d", "ut_exit_angle", gantry_angle_h );
		array_2_disk( data_filename, PREPROCESSING_DIR, TEXT, ut_exit_angle, parameters.COLUMNS, parameters.ROWS, parameters.SLICES, num_histories, true );
		sprintf(data_filename, "%s_%03d", "uv_exit_angle", gantry_angle_h );
		array_2_disk( data_filename, PREPROCESSING_DIR, TEXT, uv_exit_angle, parameters.COLUMNS, parameters.ROWS, parameters.SLICES, num_histories, true );
	}
}
void read_data_chunk( const uint num_histories, const uint start_file_num, const uint end_file_num )
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

	if( parameters.WRITE_SSD_ANGLES )
	{
		ut_entry_angle	= (float*) malloc(size_floats);
		uv_entry_angle	= (float*) malloc(size_floats);
		ut_exit_angle	= (float*) malloc(size_floats);
		uv_exit_angle	= (float*) malloc(size_floats);
	}
	switch( DATA_FORMAT )
	{	
		case VERSION_0  : read_data_chunk_v02(  num_histories, start_file_num, end_file_num - 1 ); break;
	}
}
void read_data_chunk_v0( const uint num_histories, const uint start_file_num, const uint end_file_num )
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
	FILE* data_file;
	printf("%d histories to be read from %d files\n", num_histories, end_file_num - start_file_num + 1 );
	for( unsigned int file_num = start_file_num; file_num <= end_file_num; file_num++ )
	{	
		gantry_position = file_num / parameters.NUM_SCANS;
		gantry_angle = static_cast<int>(gantry_position * parameters.GANTRY_ANGLE_INTERVAL);
		scan_number = file_num % parameters.NUM_SCANS + 1;
		file_histories = histories_per_file[file_num];
		
		//sprintf(data_filename, "%s/%s_%03d%s", PROJECTION_DATA_DIR, PROJECTION_DATA_BASENAME, gantry_angle, PROJECTION_DATA_FILE_EXTENSION );
		sprintf(data_filename, "%s/%s_%03d%s", PROJECTION_DATA_DIR, PROJECTION_DATA_BASENAME, gantry_position, PROJECTION_DATA_FILE_EXTENSION );
		if( strcmp(PROJECTION_DATA_FILE_EXTENSION, ".bin") == 0 )
			data_file = fopen(data_filename, "rb");
		else if( strcmp(PROJECTION_DATA_FILE_EXTENSION, ".txt") == 0 )
			data_file = fopen(data_filename, "r");
		if( data_file == NULL )
		{
			fputs( "Error Opening Data File:  Check that the directories are properly named.", stderr ); 
			exit_program_if(true);
		}
		if( VERSION_ID == 0 )
		{
			printf("\t");
			printf("Reading %d histories for gantry angle %d from scan number %d...\n", file_histories, gantry_angle, scan_number );			
			fseek( data_file, SKIP_2ATA_SIZE, SEEK_SET );

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
				gantry_angle_h[array_index] = static_cast<int>(projection_angles[file_num]);							
		}
		else if( VERSION_ID == 1 )
		{
			printf("\t");
			printf("Reading %d histories for gantry angle %d from scan number %d...\n", file_histories, gantry_angle, scan_number );			
			fseek( data_file, SKIP_2ATA_SIZE, SEEK_SET );

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
				gantry_angle_h[array_index] = static_cast<int>(projection_angles[file_num]);							
		}
	}
	convert_mm_2_cm( num_histories );
	if( parameters.T_SHIFT != 0.0	||  parameters.U_SHIFT != 0.0 ||  parameters.V_SHIFT != 0.0)
		apply_tuv_shifts( num_histories );
}
void read_data_chunk_v02( const uint num_histories, const uint start_file_num, const uint end_file_num )
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
	std::ifstream data_file;
	int array_index = 0, histories_read = 0;
	for( uint file_num = start_file_num; file_num <= end_file_num; file_num++ )
	{
		int gantry_position = file_num / parameters.NUM_SCANS;
		int gantry_angle = static_cast<int>(gantry_position * parameters.GANTRY_ANGLE_INTERVAL);
		int scan_number = file_num % parameters.NUM_SCANS + 1;

		printf("Reading File for Gantry Angle %d from Scan Number %d...\n", gantry_angle, scan_number );
		//sprintf(data_filename, "%s/%s_%03d%s", PROJECTION_DATA_DIR, PROJECTION_DATA_BASENAME, gantry_angle, PROJECTION_DATA_FILE_EXTENSION );
		sprintf(data_filename, "%s/%s_%03d%s", PROJECTION_DATA_DIR, PROJECTION_DATA_BASENAME, gantry_position, PROJECTION_DATA_FILE_EXTENSION );
		if( strcmp(PROJECTION_DATA_FILE_EXTENSION, ".bin") == 0 )
			data_file.open(data_filename, std::ios::binary);
		else if( strcmp(PROJECTION_DATA_FILE_EXTENSION, ".txt") == 0 )
			data_file.open(data_filename);

		//sprintf(data_filename, "%s%s/%s_%03d%s", PROJECTION_DATA_DIR, INPUT_FOLDER, PROJECTION_DATA_BASENAME, gantry_angle, PROJECTION_DATA_FILE_EXTENSION );	
		//std::ifstream data_file(data_filename, std::ios::binary);
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
	
			double max_v = 0;
			double min_v = 0;
			double max_WEPL = 0;
			double min_WEPL = 0;
			convert_mm_2_cm( num_histories );
			for( int i = 0; i < file_histories; i++, array_index++ ) 
			{				
				if( (v_in_1_h[array_index]) > max_v )
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
					min_WEPL = WEPL_h[array_index];
				gantry_angle_h[array_index] = (static_cast<int>(projection_angle) + 270)%360;
			}
			printf("max_WEPL = %3f\n", max_WEPL );
			printf("min_WEPL = %3f\n", min_WEPL );
			data_file.close();
			histories_read += file_histories;
		}
	}
}
void recon_volume_intersections( const uint num_histories )
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
	dim3 dimGrid(static_cast<int>(num_histories/THREADS_PER_BLOCK)+1);
	recon_volume_intersections_GPU<<<dimGrid, dimBlock>>>
	(
		parameters_d, num_histories, gantry_angle_d, missed_recon_volume_d,
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
	configurations* parameters, uint num_histories, int* gantry_angle, bool* missed_recon_volume, float* t_in_1, float* t_in_2, float* t_out_1, float* t_out_2, float* u_in_1, float* u_in_2, 
	float* u_out_1, float* u_out_2, float* v_in_1, float* v_in_2, float* v_out_1, float* v_out_2, float* x_entry, float* y_entry, float* z_entry, float* x_exit, 
	float* y_exit, float* z_exit, float* xy_entry_angle, float* xz_entry_angle, float* xy_exit_angle, float* xz_exit_angle
)
{
	/************************************************************************************************************************************************************/
	/*		Determine if the proton path passes through the reconstruction volume (i.e. intersects the reconstruction cylinder twice) and if it does, determine	*/ 
	/*	the x, y, and z positions in the global/object coordinate system where the proton enters and exits the reconstruction volume.  The origin of the object */
	/*	coordinate system is defined to be at the center of the reconstruction cylinder so that its volume is bounded by:										*/
	/*																																							*/
	/*													-parameters->RECON_CYL_RADIUS	<= x <= parameters->RECON_CYL_RADIUS															*/
	/*													-parameters->RECON_CYL_RADIUS	<= y <= parameters->RECON_CYL_RADIUS															*/
	/*													-parameters->RECON_CYL_HEIGHT/2 <= z <= parameters->RECON_CYL_HEIGHT/2															*/																									
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
		bool entered = calculate_intercepts( parameters, u_in_2[i], t_in_2[i], ut_entry_angle, u_entry, t_entry );
		
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
		bool exited = calculate_intercepts( parameters, u_out_1[i], t_out_1[i], ut_exit_angle, u_exit, t_exit );

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
		/* reconstruction volume or may have only passed through part way.  If |z_entry|> parameters->RECON_CYL_HEIGHT/2, then data is erroneous since the source			*/
		/* is around z=0 and we do not want to use this history.  If |z_entry| < parameters->RECON_CYL_HEIGHT/2 and |z_exit| > parameters->RECON_CYL_HEIGHT/2 then we want to use the	*/ 
		/* history but the x_exit and y_exit positions need to be calculated again based on how far through the cylinder the proton passed before exiting		*/
		/********************************************************************************************************************************************************/
		if( entered && exited )
		{
			if( ( abs(z_entry[i]) < parameters->RECON_CYL_HEIGHT * 0.5 ) && ( abs(z_exit[i]) > parameters->RECON_CYL_HEIGHT * 0.5 ) )
			{
				double recon_cyl_fraction = abs( ( ( (z_exit[i] >= 0) - (z_exit[i] < 0) ) * parameters->RECON_CYL_HEIGHT * 0.5 - z_entry[i] ) / ( z_exit[i] - z_entry[i] ) );
				x_exit[i] = x_entry[i] + recon_cyl_fraction * ( x_exit[i] - x_entry[i] );
				y_exit[i] = y_entry[i] + recon_cyl_fraction * ( y_exit[i] - y_entry[i] );
				z_exit[i] = ( (z_exit[i] >= 0) - (z_exit[i] < 0) ) * parameters->RECON_CYL_HEIGHT * 0.5;
			}
			else if( abs(z_entry[i]) > parameters->RECON_CYL_HEIGHT * 0.5 )
			{
				entered = false;
				exited = false;
			}
			if( ( abs(z_entry[i]) > parameters->RECON_CYL_HEIGHT * 0.5 ) && ( abs(z_exit[i]) < parameters->RECON_CYL_HEIGHT * 0.5 ) )
			{
				double recon_cyl_fraction = abs( ( ( (z_exit[i] >= 0) - (z_exit[i] < 0) ) * parameters->RECON_CYL_HEIGHT * 0.5 - z_exit[i] ) / ( z_exit[i] - z_entry[i] ) );
				x_entry[i] = x_exit[i] + recon_cyl_fraction * ( x_exit[i] - x_entry[i] );
				y_entry[i] = y_exit[i] + recon_cyl_fraction * ( y_exit[i] - y_entry[i] );
				z_entry[i] = ( (z_entry[i] >= 0) - (z_entry[i] < 0) ) * parameters->RECON_CYL_HEIGHT * 0.5;
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
__device__ bool calculate_intercepts( configurations* parameters, double u, double t, double ut_angle, double& u_intercept, double& t_intercept )
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
	double c = pow(b_in, 2) - pow(parameters->RECON_CYL_RADIUS, 2 );				// 1 coefficient
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
void binning( const uint num_histories )
{
	unsigned int size_floats	= sizeof(float) * num_histories;
	unsigned int size_ints		= sizeof(int) * num_histories;
	unsigned int size_bool		= sizeof(bool) * num_histories;

	missed_recon_volume_h		= (bool*)  calloc( num_histories, sizeof(bool)	);	
	bin_number_h				= (int*)   calloc( num_histories, sizeof(int)   );
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
	cudaMalloc((void**) &bin_number_d,	size_ints );

	cudaMemcpy( WEPL_d,		WEPL_h,		size_floats,	cudaMemcpyHostToDevice) ;
	cudaMemcpy( bin_number_d,	bin_number_h,	size_ints,		cudaMemcpyHostToDevice );

	dim3 dimBlock( THREADS_PER_BLOCK );
	dim3 dimGrid( static_cast<int>( num_histories/THREADS_PER_BLOCK ) + 1 );
	binning_GPU<<<dimGrid, dimBlock>>>
	( 
		parameters_d, num_histories, bin_counts_d, bin_number_d, missed_recon_volume_d,
		x_entry_d, y_entry_d, z_entry_d, x_exit_d, y_exit_d, z_exit_d, 
		mean_WEPL_d, mean_rel_ut_angle_d, mean_rel_uv_angle_d, WEPL_d, 
		xy_entry_angle_d, xz_entry_angle_d, xy_exit_angle_d, xz_exit_angle_d
	);
	cudaMemcpy( missed_recon_volume_h,		missed_recon_volume_d,		size_bool,		cudaMemcpyDeviceToHost );
	cudaMemcpy( bin_number_h,					bin_number_d,					size_ints,		cudaMemcpyDeviceToHost );
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
	if( parameters.WRITE_BIN_WEPLS )
	{
		sprintf(data_filename, "%s_%03d", "bin_numbers", gantry_angle_h[0] );
		array_2_disk( data_filename, PREPROCESSING_DIR, TEXT, bin_number_h, parameters.COLUMNS, parameters.ROWS, parameters.SLICES, num_histories, true );
	}

	// Push data from valid histories  (i.e. missed_recon_volume = FALSE) onto the end of each vector
	int offset = 0;
	for( unsigned int i = 0; i < num_histories; i++ )
	{
		if( !missed_recon_volume_h[i] && ( bin_number_h[i] >= 0 ) )
		{
			bin_number_vector.push_back( bin_number_h[i] );
			gantry_angle_vector.push_back( gantry_angle_h[i] );
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
	printf( "=======>%d out of %d (%4.2f%%) histories passed intersection cuts\n\n", offset, num_histories, static_cast<int>( offset / num_histories * 100 ));
	
	free( missed_recon_volume_h ); 
	free( bin_number_h );
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
		bin_number_d;
	*/
}
__global__ void binning_GPU
( 
	configurations* parameters, uint num_histories, int* bin_counts, int* bin_num, bool* missed_recon_volume, 
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
		path_angle = atan2( ( y_exit[i] - y_entry[i] ), ( x_exit[i] - x_entry[i] ) );
		if( path_angle < 0 )
			path_angle += 2*PI;
		angle_bin = static_cast<int> ( ( path_angle * RADIANS_TO_ANGLE / parameters->ANGULAR_BIN_SIZE ) + 0.5) % parameters->ANGULAR_BINS;	
		angle = angle_bin * parameters->ANGULAR_BIN_SIZE * ANGLE_TO_RADIANS;

		// Calculate t/v of midpoint and find t/v bin closest to this value
		t = y_midpath * cos(angle) - x_midpath * sin(angle);
		t_bin = static_cast<int>( (t / parameters->T_BIN_SIZE ) + parameters->T_BINS / 2 );			
			
		v = z_midpath;
		v_bin = static_cast<int>( (v / parameters->V_BIN_SIZE ) + parameters->V_BINS / 2 );
		
		// For histories with valid angular/t/v bin #, calculate bin #, add to its count and WEPL/relative angle sums
		if( (t_bin >= 0) && (v_bin >= 0) && (t_bin < parameters->T_BINS) && (v_bin < parameters->V_BINS) )
		{
			bin_num[i] = t_bin + angle_bin * parameters->T_BINS + v_bin * parameters->T_BINS * parameters->ANGULAR_BINS;
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
/***********************************************************************************************************************************************************************************************************************/
/******************************************************************************************** Statistical analysis and cuts ********************************************************************************************/
/***********************************************************************************************************************************************************************************************************************/
void calculate_means()
{
	puts("Calculating the Mean for Each Bin Before Cuts...");
	//cudaMemcpy( mean_WEPL_h,	mean_WEPL_d,	SIZE_BINS_FLOAT, cudaMemcpyDeviceToHost );
	//	int* empty_parameter;
	//	bins_2_disk( "WEPL_dist_pre_test2", empty_parameter, mean_WEPL_h, NUM_BINS, MEANS, ALL_BINS, BY_BIN );

	dim3 dimBlock( parameters.T_BINS );
	dim3 dimGrid( parameters.V_BINS, parameters.ANGULAR_BINS );   
	calculate_means_GPU<<< dimGrid, dimBlock >>>
	( 
		parameters_d, bin_counts_d, mean_WEPL_d, mean_rel_ut_angle_d, mean_rel_uv_angle_d
	);

	if( parameters.WRITE_WEPL_DISTS )
	{
		cudaMemcpy( mean_WEPL_h,	mean_WEPL_d,	SIZE_BINS_FLOAT, cudaMemcpyDeviceToHost );
		int* empty_parameter;
		bins_2_disk( "WEPL_dist_pre_test2", PREPROCESSING_DIR, TEXT, empty_parameter, mean_WEPL_h, parameters.NUM_BINS, MEANS, ALL_BINS, BY_BIN );
	}
	bin_counts_h		  = (int*)	 calloc( parameters.NUM_BINS, sizeof(int) );
	cudaMemcpy(bin_counts_h, bin_counts_d, SIZE_BINS_INT, cudaMemcpyDeviceToHost) ;
	cudaMemcpy( mean_WEPL_h,	mean_WEPL_d,	SIZE_BINS_FLOAT, cudaMemcpyDeviceToHost );
	cudaMemcpy( mean_rel_ut_angle_h,	mean_rel_ut_angle_d,	SIZE_BINS_FLOAT, cudaMemcpyDeviceToHost );
	cudaMemcpy( mean_rel_uv_angle_h,	mean_rel_uv_angle_d,	SIZE_BINS_FLOAT, cudaMemcpyDeviceToHost );

	array_2_disk("bin_counts_h_pre", PREPROCESSING_DIR, TEXT, bin_counts_h, parameters.T_BINS, parameters.ANGULAR_BINS, parameters.V_BINS, parameters.NUM_BINS, true );
	array_2_disk("mean_WEPL_h", PREPROCESSING_DIR, TEXT, mean_WEPL_h, parameters.T_BINS, parameters.ANGULAR_BINS, parameters.V_BINS, parameters.NUM_BINS, true );
	array_2_disk("mean_rel_ut_angle_h", PREPROCESSING_DIR, TEXT, mean_rel_ut_angle_h, parameters.T_BINS, parameters.ANGULAR_BINS, parameters.V_BINS, parameters.NUM_BINS, true );
	array_2_disk("mean_rel_uv_angle_h", PREPROCESSING_DIR, TEXT, mean_rel_uv_angle_h, parameters.T_BINS, parameters.ANGULAR_BINS, parameters.V_BINS, parameters.NUM_BINS, true );
	
	free(bin_counts_h);
	free(mean_WEPL_h);
	free(mean_rel_ut_angle_h);
	free(mean_rel_uv_angle_h);
}
__global__ void calculate_means_GPU( configurations* parameters, int* bin_counts, float* mean_WEPL, float* mean_rel_ut_angle, float* mean_rel_uv_angle )
{
	int v = blockIdx.x, angle = blockIdx.y, t = threadIdx.x;
	int bin = t + angle * parameters->T_BINS + v * parameters->T_BINS * parameters->ANGULAR_BINS;
	if( bin_counts[bin] > 0 )
	{
		mean_WEPL[bin] /= bin_counts[bin];		
		mean_rel_ut_angle[bin] /= bin_counts[bin];
		mean_rel_uv_angle[bin] /= bin_counts[bin];
	}
}
void sum_squared_deviations( const uint start_position, const uint num_histories )
{
	unsigned int size_floats = sizeof(float) * num_histories;
	unsigned int size_ints = sizeof(int) * num_histories;

	cudaMalloc((void**) &bin_number_d,				size_ints);
	cudaMalloc((void**) &WEPL_d,				size_floats);
	cudaMalloc((void**) &xy_entry_angle_d,		size_floats);
	cudaMalloc((void**) &xz_entry_angle_d,		size_floats);
	cudaMalloc((void**) &xy_exit_angle_d,		size_floats);
	cudaMalloc((void**) &xz_exit_angle_d,		size_floats);

	cudaMemcpy( bin_number_d,				&bin_number_vector[start_position],			size_ints, cudaMemcpyHostToDevice);
	cudaMemcpy( WEPL_d,					&WEPL_vector[start_position],				size_floats, cudaMemcpyHostToDevice);
	cudaMemcpy( xy_entry_angle_d,		&xy_entry_angle_vector[start_position],		size_floats, cudaMemcpyHostToDevice);
	cudaMemcpy( xz_entry_angle_d,		&xz_entry_angle_vector[start_position],		size_floats, cudaMemcpyHostToDevice);
	cudaMemcpy( xy_exit_angle_d,		&xy_exit_angle_vector[start_position],		size_floats, cudaMemcpyHostToDevice);
	cudaMemcpy( xz_exit_angle_d,		&xz_exit_angle_vector[start_position],		size_floats, cudaMemcpyHostToDevice);


	//cudaMemcpy( bin_number_d,				&bin_num[start_position],			size_ints, cudaMemcpyHostToDevice);
	//cudaMemcpy( WEPL_d,					&WEPL[start_position],				size_floats, cudaMemcpyHostToDevice);
	//cudaMemcpy( xy_entry_angle_d,		&xy_entry_angle[start_position],		size_floats, cudaMemcpyHostToDevice);
	//cudaMemcpy( xz_entry_angle_d,		&xz_entry_angle[start_position],		size_floats, cudaMemcpyHostToDevice);
	//cudaMemcpy( xy_exit_angle_d,		&xy_exit_angle[start_position],		size_floats, cudaMemcpyHostToDevice);
	//cudaMemcpy( xz_exit_angle_d,		&xz_exit_angle[start_position],		size_floats, cudaMemcpyHostToDevice);

	dim3 dimBlock(THREADS_PER_BLOCK);
	dim3 dimGrid(static_cast<int>(num_histories/THREADS_PER_BLOCK)+1);
	sum_squared_deviations_GPU<<<dimGrid, dimBlock>>>
	( 
		parameters_d, num_histories, bin_number_d, mean_WEPL_d, mean_rel_ut_angle_d, mean_rel_uv_angle_d, 
		WEPL_d, xy_entry_angle_d, xz_entry_angle_d,  xy_exit_angle_d, xz_exit_angle_d,
		stddev_WEPL_d, stddev_rel_ut_angle_d, stddev_rel_uv_angle_d
	);
	cudaFree( bin_number_d );
	cudaFree( WEPL_d );
	cudaFree( xy_entry_angle_d );
	cudaFree( xz_entry_angle_d );
	cudaFree( xy_exit_angle_d );
	cudaFree( xz_exit_angle_d );
}
__global__ void sum_squared_deviations_GPU
( 
	configurations* parameters, uint num_histories, int* bin_num, float* mean_WEPL, float* mean_rel_ut_angle, float* mean_rel_uv_angle,  
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
	dim3 dimBlock( parameters.T_BINS );
	dim3 dimGrid( parameters.V_BINS, parameters.ANGULAR_BINS );   
	calculate_standard_deviations_GPU<<< dimGrid, dimBlock >>>
	( 
		parameters_d, bin_counts_d, stddev_WEPL_d, stddev_rel_ut_angle_d, stddev_rel_uv_angle_d
	);
	cudaMemcpy( stddev_rel_ut_angle_h,	stddev_rel_ut_angle_d,	SIZE_BINS_FLOAT,	cudaMemcpyDeviceToHost );
	cudaMemcpy( stddev_rel_uv_angle_h,	stddev_rel_uv_angle_d,	SIZE_BINS_FLOAT,	cudaMemcpyDeviceToHost );
	cudaMemcpy( stddev_WEPL_h,			stddev_WEPL_d,			SIZE_BINS_FLOAT,	cudaMemcpyDeviceToHost );

	array_2_disk("stddev_rel_ut_angle_h", PREPROCESSING_DIR, TEXT, stddev_rel_ut_angle_h, parameters.T_BINS, parameters.ANGULAR_BINS, parameters.V_BINS, parameters.NUM_BINS, true );
	array_2_disk("stddev_rel_uv_angle_h", PREPROCESSING_DIR, TEXT, stddev_rel_uv_angle_h, parameters.T_BINS, parameters.ANGULAR_BINS, parameters.V_BINS, parameters.NUM_BINS, true );
	array_2_disk("stddev_WEPL_h", PREPROCESSING_DIR, TEXT, stddev_WEPL_h, parameters.T_BINS, parameters.ANGULAR_BINS, parameters.V_BINS, parameters.NUM_BINS, true );
	//cudaFree( bin_counts_d );
}
__global__ void calculate_standard_deviations_GPU( configurations* parameters, int* bin_counts, float* stddev_WEPL, float* stddev_rel_ut_angle, float* stddev_rel_uv_angle )
{
	int v = blockIdx.x, angle = blockIdx.y, t = threadIdx.x;
	int bin = t + angle * parameters->T_BINS + v * parameters->T_BINS * parameters->ANGULAR_BINS;
	if( bin_counts[bin] > 1 )
	{
		// SAMPLE_STDEV = true/false = 1/0 => std_dev = SUM{i = 1 -> N} [ ( mu - x_i)^2 / ( N - 1/0 ) ]
		stddev_WEPL[bin] = sqrtf( stddev_WEPL[bin] / ( bin_counts[bin] - SAMPLE_STDEV ) );		
		stddev_rel_ut_angle[bin] = sqrtf( stddev_rel_ut_angle[bin] / ( bin_counts[bin] - SAMPLE_STDEV ) );
		stddev_rel_uv_angle[bin] = sqrtf( stddev_rel_uv_angle[bin] / ( bin_counts[bin] - SAMPLE_STDEV ) );
	}
	syncthreads();
	bin_counts[bin] = 0;
}
void statistical_cuts( const uint start_position, const uint num_histories )
{
	unsigned int size_floats = sizeof(float) * num_histories;
	unsigned int size_ints = sizeof(int) * num_histories;
	unsigned int size_bools = sizeof(bool) * num_histories;

	failed_cuts_h = (bool*) calloc ( num_histories, sizeof(bool) );
	
	cudaMalloc( (void**) &bin_number_d,			size_ints );
	cudaMalloc( (void**) &WEPL_d,				size_floats );
	cudaMalloc( (void**) &xy_entry_angle_d,		size_floats );
	cudaMalloc( (void**) &xz_entry_angle_d,		size_floats );
	cudaMalloc( (void**) &xy_exit_angle_d,		size_floats );
	cudaMalloc( (void**) &xz_exit_angle_d,		size_floats );
	cudaMalloc( (void**) &failed_cuts_d,		size_bools );

	cudaMemcpy( bin_number_d,				&bin_number_vector[start_position],			size_ints,		cudaMemcpyHostToDevice );
	cudaMemcpy( WEPL_d,					&WEPL_vector[start_position],				size_floats,	cudaMemcpyHostToDevice );
	cudaMemcpy( xy_entry_angle_d,		&xy_entry_angle_vector[start_position],		size_floats,	cudaMemcpyHostToDevice );
	cudaMemcpy( xz_entry_angle_d,		&xz_entry_angle_vector[start_position],		size_floats,	cudaMemcpyHostToDevice );
	cudaMemcpy( xy_exit_angle_d,		&xy_exit_angle_vector[start_position],		size_floats,	cudaMemcpyHostToDevice );
	cudaMemcpy( xz_exit_angle_d,		&xz_exit_angle_vector[start_position],		size_floats,	cudaMemcpyHostToDevice );
	cudaMemcpy( failed_cuts_d,			failed_cuts_h,								size_bools,		cudaMemcpyHostToDevice );

	//cudaMemcpy( bin_number_d,				&bin_num[start_position],			size_ints, cudaMemcpyHostToDevice);
	//cudaMemcpy( WEPL_d,					&WEPL[start_position],				size_floats, cudaMemcpyHostToDevice);
	//cudaMemcpy( xy_entry_angle_d,		&xy_entry_angle[start_position],		size_floats, cudaMemcpyHostToDevice);
	//cudaMemcpy( xz_entry_angle_d,		&xz_entry_angle[start_position],		size_floats, cudaMemcpyHostToDevice);
	//cudaMemcpy( xy_exit_angle_d,		&xy_exit_angle[start_position],		size_floats, cudaMemcpyHostToDevice);
	//cudaMemcpy( xz_exit_angle_d,		&xz_exit_angle[start_position],		size_floats, cudaMemcpyHostToDevice);

	dim3 dimBlock(THREADS_PER_BLOCK);
	dim3 dimGrid( static_cast<int>( num_histories / THREADS_PER_BLOCK ) + 1 );  
	statistical_cuts_GPU<<< dimGrid, dimBlock >>>
	( 
		parameters_d, num_histories, bin_counts_d, bin_number_d, sinogram_d, WEPL_d, 
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
			bin_number_vector[post_cut_histories] = bin_number_vector[start_position + i];
			gantry_angle_vector[post_cut_histories] = gantry_angle_vector[start_position + i];
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
	
	cudaFree(bin_number_d);
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
		bin_number_d;
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
	configurations* parameters, uint num_histories, int* bin_counts, int* bin_num, float* sinogram, float* WEPL, 
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
		bool passed_ut_cut = ( abs( rel_ut_angle - mean_rel_ut_angle[bin_num[i]] ) < ( parameters->SIGMAS_2_KEEP * stddev_rel_ut_angle[bin_num[i]] ) );
		bool passed_uv_cut = ( abs( rel_uv_angle - mean_rel_uv_angle[bin_num[i]] ) < ( parameters->SIGMAS_2_KEEP * stddev_rel_uv_angle[bin_num[i]] ) );
		//bool passed_uv_cut = true;
		bool passed_WEPL_cut = ( abs( mean_WEPL[bin_num[i]] - WEPL[i] ) <= ( parameters->SIGMAS_2_KEEP * stddev_WEPL[bin_num[i]] ) );
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
	sinogram_h = (float*) calloc( parameters.NUM_BINS, sizeof(float) );
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
	puts("Recalculating the mean WEPL for each bin and constructing the sinogram...");
	bin_counts_h		  = (int*)	 calloc( parameters.NUM_BINS, sizeof(int) );
	cudaMemcpy(bin_counts_h, bin_counts_d, SIZE_BINS_INT, cudaMemcpyDeviceToHost) ;
	array_2_disk( "bin_counts_pre", PREPROCESSING_DIR, TEXT, bin_counts_h, parameters.T_BINS, parameters.ANGULAR_BINS, parameters.V_BINS, parameters.NUM_BINS, true );

	cudaMemcpy(sinogram_h,  sinogram_d, SIZE_BINS_FLOAT, cudaMemcpyDeviceToHost);
	array_2_disk("sinogram_pre", PREPROCESSING_DIR, TEXT, sinogram_h, parameters.T_BINS, parameters.ANGULAR_BINS, parameters.V_BINS, parameters.NUM_BINS, true );

	dim3 dimBlock( parameters.T_BINS );
	dim3 dimGrid( parameters.V_BINS, parameters.ANGULAR_BINS );   
	construct_sinogram_GPU<<< dimGrid, dimBlock >>>( parameters_d, bin_counts_d, sinogram_d );

	if( parameters.WRITE_WEPL_DISTS )
	{
		cudaMemcpy( sinogram_h,	sinogram_d,	SIZE_BINS_FLOAT, cudaMemcpyDeviceToHost );
		int* empty_parameter;
		bins_2_disk( "WEPL_dist_post_test2", PREPROCESSING_DIR, TEXT, empty_parameter, sinogram_h, parameters.NUM_BINS, MEANS, ALL_BINS, BY_BIN );
	}
	cudaMemcpy(sinogram_h,  sinogram_d, SIZE_BINS_FLOAT, cudaMemcpyDeviceToHost);
	array_2_disk("sinogram", PREPROCESSING_DIR, TEXT, sinogram_h, parameters.T_BINS, parameters.ANGULAR_BINS, parameters.V_BINS, parameters.NUM_BINS, true );

	cudaMemcpy(bin_counts_h, bin_counts_d, SIZE_BINS_INT, cudaMemcpyDeviceToHost) ;
	array_2_disk( "bin_counts_post", PREPROCESSING_DIR, TEXT, bin_counts_h, parameters.T_BINS, parameters.ANGULAR_BINS, parameters.V_BINS, parameters.NUM_BINS, true );
	cudaFree(bin_counts_d);
}
__global__ void construct_sinogram_GPU( configurations* parameters, int* bin_counts, float* sinogram )
{
	int v = blockIdx.x, angle = blockIdx.y, t = threadIdx.x;
	int bin = t + angle * parameters->T_BINS + v * parameters->T_BINS * parameters->ANGULAR_BINS;
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

	FBP_h = (float*) calloc( parameters.NUM_VOXELS, sizeof(float) );
	if( FBP_h == NULL ) 
	{
		printf("ERROR: Memory not allocated for FBP_h!\n");
		exit_program_if(true);
	}

	free(sinogram_filtered_h);
	cudaMalloc((void**) &FBP_d, SIZE_IMAGE_FLOAT );
	cudaMemcpy( FBP_d, FBP_h, SIZE_IMAGE_FLOAT, cudaMemcpyHostToDevice );

	dim3 dimBlock( parameters.SLICES );
	dim3 dimGrid( parameters.COLUMNS, parameters.ROWS );   
	backprojection_GPU<<< dimGrid, dimBlock >>>( parameters_d, sinogram_filtered_d, FBP_d );
	cudaFree(sinogram_filtered_d);

	cudaMemcpy( FBP_h, FBP_d, SIZE_IMAGE_FLOAT, cudaMemcpyDeviceToHost );
	array_2_disk( "FBP_h", PREPROCESSING_DIR, TEXT, FBP_h, parameters.COLUMNS, parameters.ROWS, parameters.SLICES, parameters.NUM_VOXELS, true );	

	//if( parameters.IMPORT_FILTERED_FBP)
	//{
	//	//char filename[256];
	//	//char* name = "FBP_med7";		
	//	//sprintf( filename, "%s%s/%s%s", OUTPUTIRECTORY, OUTPUT_FOLDER, name, ".bin" );
	//	//import_image( image, filename );
	//	float* image = (float*)calloc( parameters.NUM_VOXELS, sizeof(float));
	//	import_image( image, PREPROCESSING_DIR, FBP_BASENAME, TEXT );
	//	FBP_h = image;
	//	array_2_disk( "FBP_after", PREPROCESSING_DIR, TEXT, image, parameters.COLUMNS, parameters.ROWS, parameters.SLICES, parameters.NUM_VOXELS, true );
	//}
	if( parameters.AVG_FILTER_FBP )
	{
		puts("Applying average filter to FBP image...");
		//cout << FBP_d << endl;
		//float* FBP_filtered_d;
		FBP_filtered_h = FBP_h;
		cudaMalloc((void**) &FBP_filtered_d, SIZE_IMAGE_FLOAT );
		cudaMemcpy( FBP_filtered_d, FBP_filtered_h, SIZE_IMAGE_FLOAT, cudaMemcpyHostToDevice );

		//averaging_filter( FBP_h, FBP_filtered_d, FBP_AVG_FILTER_RADIUS, false, FBP_FILTER_THRESHOLD );
		//averaging_filter( FBP_filtered_h, FBP_filtered_d, parameters.FBP_AVG_FILTER_RADIUS, false, parameters.FBP_AVG_THRESHOLD );
		puts("FBP Filtering complete");
		if( parameters.WRITE_AVG_FBP )
		{
			puts("Writing filtered hull to disk...");
			//cudaMemcpy(FBP_h, FBP_filtered_d, SIZE_IMAGE_FLOAT, cudaMemcpyDeviceToHost);
			//array_2_disk( "FBP_filtered", OUTPUTIRECTORY, OUTPUT_FOLDER, FBP_h, parameters.COLUMNS, parameters.ROWS, parameters.SLICES, parameters.NUM_VOXELS, true );
			cudaMemcpy(FBP_filtered_h, FBP_filtered_d, SIZE_IMAGE_FLOAT, cudaMemcpyDeviceToHost) ;
			//cout << FBP_d << endl;
			array_2_disk( "FBP_filtered", PREPROCESSING_DIR, TEXT, FBP_filtered_h, parameters.COLUMNS, parameters.ROWS, parameters.SLICES, parameters.NUM_VOXELS, true );
			//FBP_h = FBP_filtered_h;
		}
		cudaFree(FBP_filtered_d);
	}
	else if( parameters.MEDIAN_FILTER_FBP )
	{
		puts("Applying median filter to FBP image...");
		//cout << FBP_d << endl;
		//float* FBP_filtered_d;
		//FBP_median_filtered_h = FBP_h;
		//cudaMalloc((void**) &FBP_median_filtered_d, SIZE_IMAGE_FLOAT );
		//cudaMemcpy( FBP_median_filtered_d, FBP_median_filtered_h, SIZE_IMAGE_FLOAT, cudaMemcpyHostToDevice );
		FBP_median_filtered_2D_h = (float*)calloc(parameters.NUM_VOXELS, sizeof(float));
		FBP_median_filtered_3D_h = (float*)calloc(parameters.NUM_VOXELS, sizeof(float));
		//averaging_filter( FBP_h, FBP_filtered_d, FBP_FILTER_RADIUS, false, FBP_FILTER_THRESHOLD );
		//median_filter_2D( FBP_h, FBP_median_filtered_2D_h, parameters.FBP_MED_FILTER_RADIUS );
		//median_filter_2D( FBP_h, FBP_median_filtered_3D_h, parameters.FBP_MED_FILTER_RADIUS );
		median_filter_2D( FBP_h, parameters.FBP_MED_FILTER_RADIUS );
		puts("FBP median filtering complete");
		if( parameters.WRITE_MEDIAN_FBP )
		{
			puts("Writing filtered hull to disk...");
			//cudaMemcpy(FBP_h, FBP_filtered_d, SIZE_IMAGE_FLOAT, cudaMemcpyDeviceToHost);
			//array_2_disk( "FBP_filtered", OUTPUTIRECTORY, OUTPUT_FOLDER, FBP_h, parameters.COLUMNS, parameters.ROWS, parameters.SLICES, parameters.NUM_VOXELS, true );
			//cudaMemcpy(FBP_median_filtered_h, FBP_median_filtered_d, SIZE_IMAGE_FLOAT, cudaMemcpyDeviceToHost) ;
			//cout << FBP_d << endl;
			array_2_disk( FBP_MEDIAN_2D_FILENAME, PREPROCESSING_DIR, TEXT, FBP_median_filtered_2D_h, parameters.COLUMNS, parameters.ROWS, parameters.SLICES, parameters.NUM_VOXELS, true );
			array_2_disk( FBP_MEDIAN_3D_FILENAME, PREPROCESSING_DIR, TEXT, FBP_median_filtered_3D_h, parameters.COLUMNS, parameters.ROWS, parameters.SLICES, parameters.NUM_VOXELS, true );
			//FBP_h = FBP_filtered_h;
		}
		cudaFree(FBP_filtered_d);
	}
	
	// Generate FBP hull by thresholding FBP image
	if( X_HULL == FBP_HULL )
		FBP_2_hull();

	// Discard FBP image unless it is to be used as the initial iterate x_0 in iterative image reconstruction
	if( parameters.X_0_TYPE != X_FBP && parameters.X_0_TYPE != HYBRID && X_HULL != FBP_HULL	)
	{
		free(FBP_h);
		cudaFree(FBP_d);
	}
}
void filter()
{
	puts("Filtering the sinogram...");	

	sinogram_filtered_h = (float*) calloc( parameters.NUM_BINS, sizeof(float) );
	if( sinogram_filtered_h == NULL )
	{
		puts("ERROR: Memory allocation for sinogram_filtered_h failed.");
		exit(1);
	}
	cudaMalloc((void**) &sinogram_filtered_d, SIZE_BINS_FLOAT);
	cudaMemcpy( sinogram_filtered_d, sinogram_filtered_h, SIZE_BINS_FLOAT, cudaMemcpyHostToDevice);

	dim3 dimBlock( parameters.T_BINS );
	dim3 dimGrid( parameters.V_BINS, parameters.ANGULAR_BINS );   	
	filter_GPU<<< dimGrid, dimBlock >>>( parameters_d, sinogram_d, sinogram_filtered_d );
}
__global__ void filter_GPU( configurations* parameters, float* sinogram, float* sinogram_filtered )
{		
	int v_bin = blockIdx.x, angle_bin = blockIdx.y, t_bin = threadIdx.x;
	int t_bin_ref, t_bin_sep, strip_index; 
	double filtered, t, scale_factor;
	double v = ( v_bin - parameters->V_BINS/2 ) * parameters->V_BIN_SIZE + parameters->V_BIN_SIZE/2.0;
	
	// Loop over strips for this strip
	for( t_bin_ref = 0; t_bin_ref < parameters->T_BINS; t_bin_ref++ )
	{
		t = ( t_bin_ref - parameters->T_BINS/2 ) * parameters->T_BIN_SIZE + parameters->T_BIN_SIZE/2.0;
		t_bin_sep = t_bin - t_bin_ref;
		// scale_factor = r . path = cos(theta_{r,path})
		scale_factor = SOURCE_RADIUS / sqrt( SOURCE_RADIUS * SOURCE_RADIUS + t * t + v * v );
		switch( parameters->FBP_FILTER_TYPE )
		{
			case NONE: 
				break;
			case RAM_LAK:
				if( t_bin_sep == 0 )
					filtered = 1.0 / ( 4.0 * pow( parameters->RAM_LAK_TAU, 2.0 ) );
				else if( t_bin_sep % 2 == 0 )
					filtered = 0;
				else
					filtered = -1.0 / ( pow( parameters->RAM_LAK_TAU * PI * t_bin_sep, 2.0 ) );	
				break;
			case SHEPP_LOGAN:
				filtered = pow( pow(parameters->T_BIN_SIZE * PI, 2.0) * ( 1.0 - pow(2 * t_bin_sep, 2.0) ), -1.0 );
		}
		strip_index = ( v_bin * parameters->ANGULAR_BINS * parameters->T_BINS ) + ( angle_bin * parameters->T_BINS );
		sinogram_filtered[strip_index + t_bin] += parameters->T_BIN_SIZE * sinogram[strip_index + t_bin_ref] * filtered * scale_factor;
	}
}
void backprojection()
{
	//// Check that we don't have any corruptions up until now
	//for( unsigned int i = 0; i < NUM_BINS; i++ )
	//	if( sinogram_filtered_h[i] != sinogram_filtered_h[i] )
	//		printf("We have a nan in bin #%d\n", i);

	double delta = parameters.GANTRY_ANGLE_INTERVAL * ANGLE_TO_RADIANS;
	int voxel;
	double x, y, z;
	double u, t, v;
	double detector_number_t, detector_number_v;
	double eta, epsilon;
	double scale_factor;
	int t_bin, v_bin, bin, bin_below;
	// Loop over the voxels
	for( uint slice = 0; slice < parameters.SLICES; slice++ )
	{
		for( uint column = 0; column < parameters.COLUMNS; column++ )
		{

			for( uint row = 0; row < parameters.ROWS; row++ )
			{
				voxel = column +  ( row * parameters.COLUMNS ) + ( slice * parameters.COLUMNS * parameters.ROWS);
				x = -parameters.RECON_CYL_RADIUS + ( column + 0.5 )* parameters.VOXEL_WIDTH;
				y = parameters.RECON_CYL_RADIUS - (row + 0.5) * parameters.VOXEL_HEIGHT;
				z = -parameters.RECON_CYL_HEIGHT / 2.0 + (slice + 0.5) * parameters.SLICE_THICKNESS;
				// If the voxel is outside the cylinder defining the reconstruction volume, set RSP to air
				if( ( x * x + y * y ) > ( parameters.RECON_CYL_RADIUS * parameters.RECON_CYL_RADIUS ) )
					FBP_h[voxel] = static_cast<float>(RSP_AIR);							
				else
				{	  
					// Sum over projection angles
					for( uint angle_bin = 0; angle_bin < parameters.ANGULAR_BINS; angle_bin++ )
					{
						// Rotate the pixel position to the beam-detector coordinate system
						u = x * cos( angle_bin * delta ) + y * sin( angle_bin * delta );
						t = -x * sin( angle_bin * delta ) + y * cos( angle_bin * delta );
						v = z;

						// Project to find the detector number
						detector_number_t = ( t - u *( t / ( SOURCE_RADIUS + u ) ) ) / parameters.T_BIN_SIZE + parameters.T_BINS/2.0;
						t_bin = static_cast<int>( detector_number_t);
						if( t_bin > detector_number_t )
							t_bin -= 1;
						eta = detector_number_t - t_bin;

						// Now project v to get detector number in v axis
						detector_number_v = ( v - u * ( v / ( SOURCE_RADIUS + u ) ) ) / parameters.V_BIN_SIZE + parameters.V_BINS/2.0;
						v_bin = static_cast<int>( detector_number_v);
						if( v_bin > detector_number_v )
							v_bin -= 1;
						epsilon = detector_number_v - v_bin;

						// Calculate the fan beam scaling factor
						scale_factor = pow( SOURCE_RADIUS / ( SOURCE_RADIUS + u ), 2 );
		  
						// Compute the back-projection
						bin = t_bin + angle_bin * parameters.T_BINS + v_bin * parameters.ANGULAR_BINS * parameters.T_BINS;
						bin_below = bin + ( parameters.ANGULAR_BINS * parameters.T_BINS );

						// If in last v_vin, there is no bin below so only use adjacent bins
						if( v_bin == parameters.V_BINS - 1 || ( bin < 0 ) )
							FBP_h[voxel] += static_cast<float>(scale_factor * ( ( ( 1 - eta ) * sinogram_filtered_h[bin] ) + ( eta * sinogram_filtered_h[bin + 1] ) ) );
					/*	if( t_bin < parameters.T_BINS - 1 )
								FBP_h[voxel] += scale_factor * ( ( ( 1 - eta ) * sinogram_filtered_h[bin] ) + ( eta * sinogram_filtered_h[bin + 1] ) );
							if( v_bin < parameters.V_BINS - 1 )
								FBP_h[voxel] += scale_factor * ( ( ( 1 - epsilon ) * sinogram_filtered_h[bin] ) + ( epsilon * sinogram_filtered_h[bin_below] ) );
							if( t_bin == parameters.T_BINS - 1 && v_bin == parameters.V_BINS - 1 )
								FBP_h[voxel] += scale_factor * sinogram_filtered_h[bin];*/
						else 
						{
							// Technically this is to be multiplied by delta as well, but since delta is constant, it is more accurate numerically to multiply result by delta instead
							FBP_h[voxel] += static_cast<float>(scale_factor * ( ( 1 - eta ) * ( 1 - epsilon ) * sinogram_filtered_h[bin] 
							+ eta * ( 1 - epsilon ) * sinogram_filtered_h[bin + 1]
							+ ( 1 - eta ) * epsilon * sinogram_filtered_h[bin_below]
							+ eta * epsilon * sinogram_filtered_h[bin_below + 1] ));
						} 
					}
					FBP_h[voxel] *= static_cast<float>(delta);
				}
			}
		}
	}
}
__global__ void backprojection_GPU( configurations* parameters, float* sinogram_filtered, float* FBP )
{
	int row = blockIdx.y, column = blockIdx.x, slice = threadIdx.x;
	int voxel = slice * parameters->COLUMNS * parameters->ROWS + row * parameters->COLUMNS + column;	
	if ( voxel < parameters->NUM_VOXELS )
	{
		double delta = parameters->GANTRY_ANGLE_INTERVAL * ANGLE_TO_RADIANS;
		double u, t, v;
		double detector_number_t, detector_number_v;
		double eta, epsilon;
		double scale_factor;
		int t_bin, v_bin, bin;
		double x = -parameters->RECON_CYL_RADIUS + ( column + 0.5 )* parameters->VOXEL_WIDTH;
		double y = parameters->RECON_CYL_RADIUS - (row + 0.5) * parameters->VOXEL_HEIGHT;
		double z = -parameters->RECON_CYL_HEIGHT / 2.0 + (slice + 0.5) * parameters->SLICE_THICKNESS;

		//// If the voxel is outside a cylinder contained in the reconstruction volume, set to air
		if( ( x * x + y * y ) > ( parameters->RECON_CYL_RADIUS * parameters->RECON_CYL_RADIUS ) )
			FBP[( slice * parameters->COLUMNS * parameters->ROWS) + ( row * parameters->COLUMNS ) + column] = RSP_AIR;							
		else
		{	  
			// Sum over projection angles
			for( int angle_bin = 0; angle_bin < parameters->ANGULAR_BINS; angle_bin++ )
			{
				// Rotate the pixel position to the beam-detector coordinate system
				u = x * cos( angle_bin * delta ) + y * sin( angle_bin * delta );
				t = -x * sin( angle_bin * delta ) + y * cos( angle_bin * delta );
				v = z;

				// Project to find the detector number
				detector_number_t = ( t - u *( t / ( SOURCE_RADIUS + u ) ) ) / parameters->T_BIN_SIZE + parameters->T_BINS/2.0;
				t_bin = static_cast<int>( detector_number_t);
				//if( t_bin > detector_number_t )
				//	t_bin -= 1;
				eta = detector_number_t - t_bin;

				// Now project v to get detector number in v axis
				detector_number_v = ( v - u * ( v / ( SOURCE_RADIUS + u ) ) ) / parameters->V_BIN_SIZE + parameters->V_BINS/2.0;
				v_bin = static_cast<int>( detector_number_v);
				//if( v_bin > detector_number_v )
				//	v_bin -= 1;
				epsilon = detector_number_v - v_bin;

				// Calculate the fan beam scaling factor
				scale_factor = pow( SOURCE_RADIUS / ( SOURCE_RADIUS + u ), 2 );
		  
				//bin_num[i] = t_bin + angle_bin * parameters->T_BINS + v_bin * parameters->T_BINS * parameters->ANGULAR_BINS;
				// Compute the back-projection
				bin = t_bin + angle_bin * parameters->T_BINS + v_bin * parameters->ANGULAR_BINS * parameters->T_BINS;
				// not sure why this won't compile without calculating the index ahead of time instead inside []s
				//int index = parameters->ANGULAR_BINS * parameters->T_BINS;

				//if( ( ( bin + parameters->ANGULAR_BINS * parameters->T_BINS + 1 ) >= NUM_BINS ) || ( bin < 0 ) );
				if( v_bin == parameters->V_BINS - 1 || ( bin < 0 ) )
					FBP[voxel] += scale_factor * ( ( ( 1 - eta ) * sinogram_filtered[bin] ) + ( eta * sinogram_filtered[bin + 1] ) ) ;
					//printf("The bin selected for this voxel does not exist!\n Slice: %d\n Column: %d\n Row: %d\n", slice, column, row);
				else 
				{
					// not sure why this won't compile without calculating the index ahead of time instead inside []s
					/*FBP[voxel] += delta * ( ( 1 - eta ) * ( 1 - epsilon ) * sinogram_filtered[bin] 
					+ eta * ( 1 - epsilon ) * sinogram_filtered[bin + 1]
					+ ( 1 - eta ) * epsilon * sinogram_filtered[bin + parameters->ANGULAR_BINS * parameters->T_BINS]
					+ eta * epsilon * sinogram_filtered[bin + parameters->ANGULAR_BINS * parameters->T_BINS + 1] ) * scale_factor;*/

					// Multilpying by the gantry angle interval for each gantry angle is equivalent to multiplying the final answer by 2*PI and is better numerically
					// so multiplying by delta each time should be replaced by FBP_h[voxel] *= 2 * PI after all contributions have been made, which is commented out below
					FBP[voxel] += scale_factor * ( ( 1 - eta ) * ( 1 - epsilon ) * sinogram_filtered[bin] 
					+ eta * ( 1 - epsilon ) * sinogram_filtered[bin + 1]
					+ ( 1 - eta ) * epsilon * sinogram_filtered[bin + ( parameters->ANGULAR_BINS * parameters->T_BINS)]
					+ eta * epsilon * sinogram_filtered[bin + ( parameters->ANGULAR_BINS * parameters->T_BINS) + 1] );
				}				
			}
			FBP[voxel] *= delta; 
		}
	}
}
void FBP_2_hull()
{
	puts("Performing thresholding on FBP image to generate FBP hull...");

	FBP_hull_h = (bool*) calloc( parameters.COLUMNS * parameters.ROWS * parameters.SLICES, sizeof(bool) );
	initialize_hull( FBP_hull_h, FBP_hull_d );
	dim3 dimBlock( parameters.SLICES );
	dim3 dimGrid( parameters.COLUMNS, parameters.ROWS );   
	FBP_2_hull_GPU<<< dimGrid, dimBlock >>>( parameters_d, FBP_d, FBP_hull_d );	
	cudaMemcpy( FBP_hull_h, FBP_hull_d, SIZE_IMAGE_BOOL, cudaMemcpyDeviceToHost );
	
	if( parameters.WRITE_FBP_HULL )
		array_2_disk( "hull_FBP", PREPROCESSING_DIR, TEXT, FBP_hull_h, parameters.COLUMNS, parameters.ROWS, parameters.SLICES, parameters.NUM_VOXELS, true );

	if( parameters.HULL_TYPE != FBP_HULL)	
		free(FBP_hull_h);
	cudaFree(FBP_hull_d);
	cudaFree(FBP_d);
}
__global__ void FBP_2_hull_GPU( configurations* parameters, float* FBP, bool* FBP_hull )
{
	int row = blockIdx.y, column = blockIdx.x, slice = threadIdx.x;
	int voxel = slice * parameters->COLUMNS * parameters->ROWS + row * parameters->COLUMNS + column; 
	double x = -parameters->RECON_CYL_RADIUS + ( column + 0.5 )* parameters->VOXEL_WIDTH;
	double y = parameters->RECON_CYL_RADIUS - (row + 0.5) * parameters->VOXEL_HEIGHT;
	double d_squared = pow(x, 2) + pow(y, 2);
	if( ( FBP[voxel] > FBP_THRESHOLD ) && ( d_squared < pow(parameters->RECON_CYL_RADIUS, 2) ) ) 
		FBP_hull[voxel] = true; 
	else
		FBP_hull[voxel] = false; 
}
/***********************************************************************************************************************************************************************************************************************/
/*************************************************************************************************** Hull-Detection ****************************************************************************************************/
/***********************************************************************************************************************************************************************************************************************/
void hull_initializations()
{		
	if( parameters.SC_ON )
		initialize_hull( SC_hull_h, SC_hull_d );
	if( parameters.MSC_ON )
		initialize_hull( MSC_counts_h, MSC_counts_d );
	if( parameters.SM_ON )
		initialize_hull( SM_counts_h, SM_counts_d );
}
template<typename T> void initialize_hull( T*& hull_h, T*& hull_d )
{
	/* Allocate memory and initialize hull on the GPU.  Use the image and reconstruction cylinder parameters to determine the location of the perimeter of  */
	/* the reconstruction cylinder, which is centered on the origin (center) of the image.  Assign voxels inside the perimeter of the reconstruction volume */
	/* the value 1 and those outside 0.																														*/

	int image_size = parameters.NUM_VOXELS * sizeof(T);
	cudaMalloc((void**) &hull_d, image_size );

	dim3 dimBlock( parameters.SLICES );
	dim3 dimGrid( parameters.COLUMNS, parameters.ROWS );   
	initialize_hull_GPU<<< dimGrid, dimBlock >>>( parameters_d, hull_d );	
}
template<typename T> __global__ void initialize_hull_GPU( configurations* parameters, T* hull )
{
	int row = blockIdx.y, column = blockIdx.x, slice = threadIdx.x;
	int voxel = column + ( row * parameters->COLUMNS ) + ( slice * parameters->COLUMNS * parameters->ROWS );
	double x = ( column - parameters->COLUMNS/2 + 0.5) * parameters->VOXEL_WIDTH;
	double y = ( parameters->ROWS/2 - row - 0.5) * parameters->VOXEL_HEIGHT;
	if( pow(x, 2) + pow(y, 2) < pow(parameters->RECON_CYL_RADIUS, 2) )
		hull[voxel] = 1;
	else
		hull[voxel] = 0;
}
void hull_detection( const uint histories_to_process)
{
	if( parameters.SC_ON  ) 
		SC( histories_to_process );		
	if( parameters.MSC_ON )
		MSC( histories_to_process );
	if( parameters.SM_ON )
		SM( histories_to_process );   
}
__global__ void carve_differences( configurations* parameters, int* carve_differences, int* image )
{
	int row = blockIdx.y, column = blockIdx.x, slice = threadIdx.x;
	int voxel = column + row * parameters->COLUMNS + slice * parameters->COLUMNS * parameters->ROWS;
	if( (row != 0) && (row != parameters->ROWS - 1) && (column != 0) && (column != parameters->COLUMNS - 1) )
	{
		int difference, max_difference = 0;
		for( int current_row = row - 1; current_row <= row + 1; current_row++ )
		{
			for( int current_column = column - 1; current_column <= column + 1; current_column++ )
			{
				difference = image[voxel] - image[current_column + current_row * parameters->COLUMNS + slice * parameters->COLUMNS * parameters->ROWS];
				if( difference > max_difference )
					max_difference = difference;
			}
		}
		carve_differences[voxel] = max_difference;
	}
}
void SC( const uint num_histories )
{
	dim3 dimBlock(THREADS_PER_BLOCK);
	dim3 dimGrid( static_cast<int>( num_histories / THREADS_PER_BLOCK ) + 1 );
	SC_GPU<<<dimGrid, dimBlock>>>
	(
		parameters_d, num_histories, SC_hull_d, bin_number_d, missed_recon_volume_d, WEPL_d,
		x_entry_d, y_entry_d, z_entry_d, x_exit_d, y_exit_d, z_exit_d
	);
	//pause_execution();
}
__global__ void SC_GPU
( 
	configurations* parameters, const uint num_histories, bool* SC_hull, int* bin_num, bool* missed_recon_volume, float* WEPL,
	float* x_entry, float* y_entry, float* z_entry, float* x_exit, float* y_exit, float* z_exit
)
{// 15 doubles, 11 integers, 2 booleans
	int i = threadIdx.x + blockIdx.x * THREADS_PER_BLOCK;
	if
	( 
			( i < num_histories							) 
		&&	( !missed_recon_volume[i]					) 
		&&	( WEPL[i] <= parameters->SC_THRESHOLD		) 
		&&	( WEPL[i] >= parameters->IGNORE_WEPL_BELOW	) 
	)
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

		voxel_x = calculate_voxel_GPU( parameters->X_ZERO_COORDINATE, x_entry[i], parameters->VOXEL_WIDTH );
		voxel_y = calculate_voxel_GPU( parameters->Y_ZERO_COORDINATE, y_entry[i], parameters->VOXEL_HEIGHT );
		voxel_z = calculate_voxel_GPU( parameters->Z_ZERO_COORDINATE, z_entry[i], parameters->VOXEL_THICKNESS );		
		voxel = voxel_x + voxel_y * parameters->COLUMNS + voxel_z * parameters->COLUMNS * parameters->ROWS;

		x_to_go = distance_remaining_GPU( parameters->X_ZERO_COORDINATE,	x, X_INCREASING_DIRECTION, x_move_direction, parameters->VOXEL_WIDTH,	 voxel_x );
		y_to_go = distance_remaining_GPU( parameters->Y_ZERO_COORDINATE,	y, Y_INCREASING_DIRECTION,  y_move_direction, parameters->VOXEL_HEIGHT,	 voxel_y );
		z_to_go = distance_remaining_GPU( parameters->Z_ZERO_COORDINATE,	z, Z_INCREASING_DIRECTION,  z_move_direction, parameters->VOXEL_THICKNESS, voxel_z );				
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
		voxel_x_out = calculate_voxel_GPU( parameters->X_ZERO_COORDINATE, x_exit[i], parameters->VOXEL_WIDTH );
		voxel_y_out = calculate_voxel_GPU( parameters->Y_ZERO_COORDINATE, y_exit[i], parameters->VOXEL_HEIGHT );
		voxel_z_out = calculate_voxel_GPU( parameters->Z_ZERO_COORDINATE, z_exit[i], parameters->VOXEL_THICKNESS );
		voxel_out = voxel_x_out + voxel_y_out * parameters->COLUMNS + voxel_z_out * parameters->COLUMNS * parameters->ROWS;

		end_walk = ( voxel == voxel_out ) || ( voxel_x >= parameters->COLUMNS ) || ( voxel_y >= parameters->ROWS ) || ( voxel_z >= parameters->SLICES );
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
					parameters, x_move_direction, y_move_direction, z_move_direction,
					dy_dx, dz_dx, dz_dy, 
					dx_dy, dx_dz, dy_dz, 
					x_entry[i], y_entry[i], z_entry[i], 
					x, y, z, 
					voxel_x, voxel_y, voxel_z, voxel,
					x_to_go, y_to_go, z_to_go
				);
				//voxel = voxel_x + voxel_y * parameters->COLUMNS + voxel_z * parameters->COLUMNS * parameters->ROWS;
				end_walk = ( voxel == voxel_out ) || ( voxel_x >= parameters->COLUMNS ) || ( voxel_y >= parameters->ROWS ) || ( voxel_z >= parameters->SLICES );
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
					parameters, x_move_direction, y_move_direction, z_move_direction,
					dy_dx, dz_dx, dz_dy, 
					dx_dy, dx_dz, dy_dz, 
					x_entry[i], y_entry[i], z_entry[i], 
					x, y, z, 
					voxel_x, voxel_y, voxel_z, voxel,
					x_to_go, y_to_go, z_to_go
				);
				//voxel = voxel_x + voxel_y * parameters->COLUMNS + voxel_z * parameters->COLUMNS * parameters->ROWS;
				end_walk = ( voxel == voxel_out ) || ( voxel_x >= parameters->COLUMNS ) || ( voxel_y >= parameters->ROWS ) || ( voxel_z >= parameters->SLICES );
				if( !end_walk )
					SC_hull[voxel] = 0;
			}// end: while( !end_walk )
		}//end: else: z_start != z_end => z_start == z_end
	}// end: if( (i < num_histories) && !missed_recon_volume[i] && (WEPL[i] <= parameters->SC_THRESHOLD) )
}
/***********************************************************************************************************************************************************************************************************************/
void MSC( const uint num_histories )
{
	dim3 dimBlock(THREADS_PER_BLOCK);
	//dim3 dimGrid(static_cast<int>((num_histories/STRIDE)/THREADS_PER_BLOCK)+1);
	dim3 dimGrid( static_cast<int>( num_histories / THREADS_PER_BLOCK ) + 1 );
	MSC_GPU<<<dimGrid, dimBlock>>>
	(
		parameters_d, num_histories, MSC_counts_d, bin_number_d, missed_recon_volume_d, WEPL_d,
		x_entry_d, y_entry_d, z_entry_d, x_exit_d, y_exit_d, z_exit_d
	);
}
__global__ void MSC_GPU
( 
	configurations* parameters, const uint num_histories, int* MSC_counts, int* bin_num, bool* missed_recon_volume, float* WEPL,
	float* x_entry, float* y_entry, float* z_entry, float* x_exit, float* y_exit, float* z_exit
)
{
	//int thread_num = threadIdx.x + blockIdx.x * THREADS_PER_BLOCK;
	//unsigned int start_element = thread_num * STRIDE;
	//unsigned int i = start_element;
	int i = threadIdx.x + blockIdx.x * THREADS_PER_BLOCK;
	//for( int iteration = 0; iteration < STRIDE; iteration++, i++ )
	//{		
	if
	( 
			( i < num_histories							) 
		&&	( !missed_recon_volume[i]					) 
		&&	( WEPL[i] <= parameters->MSC_THRESHOLD		) 
		&&	( WEPL[i] >= parameters->IGNORE_WEPL_BELOW	) 
	)
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

		voxel_x = calculate_voxel_GPU( parameters->X_ZERO_COORDINATE, x_entry[i], parameters->VOXEL_WIDTH );
		voxel_y = calculate_voxel_GPU( parameters->Y_ZERO_COORDINATE, y_entry[i], parameters->VOXEL_HEIGHT );
		voxel_z = calculate_voxel_GPU( parameters->Z_ZERO_COORDINATE, z_entry[i], parameters->VOXEL_THICKNESS );		
		voxel = voxel_x + voxel_y * parameters->COLUMNS + voxel_z * parameters->COLUMNS * parameters->ROWS;

		x_to_go = distance_remaining_GPU( parameters->X_ZERO_COORDINATE,	x, X_INCREASING_DIRECTION, x_move_direction, parameters->VOXEL_WIDTH,	 voxel_x );
		y_to_go = distance_remaining_GPU( parameters->Y_ZERO_COORDINATE,	y, Y_INCREASING_DIRECTION,  y_move_direction, parameters->VOXEL_HEIGHT,	 voxel_y );
		z_to_go = distance_remaining_GPU( parameters->Z_ZERO_COORDINATE,	z, Z_INCREASING_DIRECTION,  z_move_direction, parameters->VOXEL_THICKNESS, voxel_z );				
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
		voxel_x_out = calculate_voxel_GPU( parameters->X_ZERO_COORDINATE, x_exit[i], parameters->VOXEL_WIDTH );
		voxel_y_out = calculate_voxel_GPU( parameters->Y_ZERO_COORDINATE, y_exit[i], parameters->VOXEL_HEIGHT );
		voxel_z_out = calculate_voxel_GPU( parameters->Z_ZERO_COORDINATE, z_exit[i], parameters->VOXEL_THICKNESS );
		voxel_out = voxel_x_out + voxel_y_out * parameters->COLUMNS + voxel_z_out * parameters->COLUMNS * parameters->ROWS;

		end_walk = ( voxel == voxel_out ) || ( voxel_x >= parameters->COLUMNS ) || ( voxel_y >= parameters->ROWS ) || ( voxel_z >= parameters->SLICES );
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
					parameters, x_move_direction, y_move_direction, z_move_direction,
					dy_dx, dz_dx, dz_dy, dx_dy, dx_dz, dy_dz, 
					x_entry[i], y_entry[i], z_entry[i], 
					x, y, z, 
					voxel_x, voxel_y, voxel_z, voxel,
					x_to_go, y_to_go, z_to_go
				);
				end_walk = ( voxel == voxel_out ) || ( voxel_x >= parameters->COLUMNS ) || ( voxel_y >= parameters->ROWS ) || ( voxel_z >= parameters->SLICES );
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
					parameters, x_move_direction, y_move_direction, z_move_direction,
					dy_dx, dz_dx, dz_dy, dx_dy, dx_dz, dy_dz, 
					x_entry[i], y_entry[i], z_entry[i], 
					x, y, z, 
					voxel_x, voxel_y, voxel_z, voxel,
					x_to_go, y_to_go, z_to_go
				);				
				end_walk = ( voxel == voxel_out ) || ( voxel_x >= parameters->COLUMNS ) || ( voxel_y >= parameters->ROWS ) || ( voxel_z >= parameters->SLICES );
				if( !end_walk )
					atomicAdd(&MSC_counts[voxel], 1);
			}// end: while( !end_walk )
		}//end: else: z_start != z_end => z_start == z_end
	}// end: if( (i < num_histories) && !missed_recon_volume[i] && (WEPL[i] <= parameters->MSC_THRESHOLD) && ( WEPL[i] >= parameters->IGNORE_WEPL_BELOW	) )
	//} // end for( int iteration = 0; iteration < STRIDE; iteration++, i++ )
}
void MSC_edge_detection()
{
	puts("Performing edge-detection on MSC_counts...");

	dim3 dimBlock( parameters.SLICES );
	dim3 dimGrid( parameters.COLUMNS, parameters.ROWS );   
	MSC_edge_detection_GPU<<< dimGrid, dimBlock >>>( parameters_d, MSC_counts_d );

	puts("MSC hull-detection and edge-detection complete.");	
}
__global__ void MSC_edge_detection_GPU( configurations* parameters, int* MSC_counts )
{
	int row = blockIdx.y, column = blockIdx.x, slice = threadIdx.x;
	int voxel = column + row * parameters->COLUMNS + slice * parameters->COLUMNS * parameters->ROWS;
	float x = ( column - parameters->COLUMNS/2 + 0.5 ) * parameters->VOXEL_WIDTH;
	float y = ( parameters->ROWS/2 - row - 0.5 ) * parameters->VOXEL_HEIGHT;
	int difference, max_difference = 0;
	if( (row != 0) && (row != parameters->ROWS - 1) && (column != 0) && (column != parameters->COLUMNS - 1) )
	{		
		for( int current_row = row - 1; current_row <= row + 1; current_row++ )
		{
			for( int current_column = column - 1; current_column <= column + 1; current_column++ )
			{
				difference = MSC_counts[voxel] - MSC_counts[current_column + current_row * parameters->COLUMNS + slice * parameters->COLUMNS * parameters->ROWS];
				if( difference > max_difference )
					max_difference = difference;
			}
		}
	}
	syncthreads();
	if( max_difference > parameters->MSC_DIFF_THRESH )
		MSC_counts[voxel] = 0;
	else
		MSC_counts[voxel] = 1;
	if( pow(x, 2) + pow(y, 2) >= pow(parameters->RECON_CYL_RADIUS - max(parameters->VOXEL_WIDTH, parameters->VOXEL_HEIGHT)/2, 2 ) )
		MSC_counts[voxel] = 0;

}
/***********************************************************************************************************************************************************************************************************************/
void SM( const uint num_histories)
{
	dim3 dimBlock(THREADS_PER_BLOCK);
	dim3 dimGrid( static_cast<int>( num_histories / THREADS_PER_BLOCK ) + 1 );
	SM_GPU <<< dimGrid, dimBlock >>>
	(
		parameters_d, num_histories, SM_counts_d, bin_number_d, missed_recon_volume_d, WEPL_d,
		x_entry_d, y_entry_d, z_entry_d, x_exit_d, y_exit_d, z_exit_d
	);
}
__global__ void SM_GPU
( 
	configurations* parameters, const uint num_histories, int* SM_counts, int* bin_num, bool* missed_recon_volume, float* WEPL,
	float* x_entry, float* y_entry, float* z_entry, float* x_exit, float* y_exit, float* z_exit
)
{
	int i = threadIdx.x + blockIdx.x * THREADS_PER_BLOCK;
	//if( (i < num_histories) && !missed_recon_volume[i] && (WEPL[i] >= parameters->SM_LOWER_THRESHOLD ) && (WEPL[i] <= parameters->SM_UPPER_THRESHOLD ) )
	if( (i < num_histories) && !missed_recon_volume[i] && (WEPL[i] >= parameters->SM_LOWER_THRESHOLD) )
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

		voxel_x = calculate_voxel_GPU( parameters->X_ZERO_COORDINATE, x_entry[i], parameters->VOXEL_WIDTH );
		voxel_y = calculate_voxel_GPU( parameters->Y_ZERO_COORDINATE, y_entry[i], parameters->VOXEL_HEIGHT );
		voxel_z = calculate_voxel_GPU( parameters->Z_ZERO_COORDINATE, z_entry[i], parameters->VOXEL_THICKNESS );		
		voxel = voxel_x + voxel_y * parameters->COLUMNS + voxel_z * parameters->COLUMNS * parameters->ROWS;

		x_to_go = distance_remaining_GPU( parameters->X_ZERO_COORDINATE,	x, X_INCREASING_DIRECTION, x_move_direction, parameters->VOXEL_WIDTH,	 voxel_x );
		y_to_go = distance_remaining_GPU( parameters->Y_ZERO_COORDINATE,	y, Y_INCREASING_DIRECTION,  y_move_direction, parameters->VOXEL_HEIGHT,	 voxel_y );
		z_to_go = distance_remaining_GPU( parameters->Z_ZERO_COORDINATE,	z, Z_INCREASING_DIRECTION,  z_move_direction, parameters->VOXEL_THICKNESS, voxel_z );				
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
		voxel_x_out = calculate_voxel_GPU( parameters->X_ZERO_COORDINATE, x_exit[i], parameters->VOXEL_WIDTH );
		voxel_y_out = calculate_voxel_GPU( parameters->Y_ZERO_COORDINATE, y_exit[i], parameters->VOXEL_HEIGHT );
		voxel_z_out = calculate_voxel_GPU( parameters->Z_ZERO_COORDINATE, z_exit[i], parameters->VOXEL_THICKNESS );
		voxel_out = voxel_x_out + voxel_y_out * parameters->COLUMNS + voxel_z_out * parameters->COLUMNS * parameters->ROWS;

		end_walk = ( voxel == voxel_out ) || ( voxel_x >= parameters->COLUMNS ) || ( voxel_y >= parameters->ROWS ) || ( voxel_z >= parameters->SLICES );
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
					parameters, x_move_direction, y_move_direction, z_move_direction,
					dy_dx, dz_dx, dz_dy, dx_dy, dx_dz, dy_dz, 
					x_entry[i], y_entry[i], z_entry[i], 
					x, y, z, 
					voxel_x, voxel_y, voxel_z, voxel,
					x_to_go, y_to_go, z_to_go
				);
				end_walk = ( voxel == voxel_out ) || ( voxel_x >= parameters->COLUMNS ) || ( voxel_y >= parameters->ROWS ) || ( voxel_z >= parameters->SLICES );
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
					parameters, x_move_direction, y_move_direction, z_move_direction,
					dy_dx, dz_dx, dz_dy, dx_dy, dx_dz, dy_dz, 
					x_entry[i], y_entry[i], z_entry[i], 
					x, y, z, 
					voxel_x, voxel_y, voxel_z, voxel,
					x_to_go, y_to_go, z_to_go
				);				
				end_walk = ( voxel == voxel_out ) || ( voxel_x >= parameters->COLUMNS ) || ( voxel_y >= parameters->ROWS ) || ( voxel_z >= parameters->SLICES );
				if( !end_walk )
					atomicAdd(&SM_counts[voxel], 1);
			}// end: while( !end_walk )
		}//end: else: z_start != z_end => z_start == z_end
	}// end: if( (i < num_histories) && !missed_recon_volume[i] && (WEPL[i] <= parameters->MSC_THRESHOLD) )
}
void SM_edge_detection()
{
	puts("Performing edge-detection on SM_counts...");	

	/*if( parameters.WRITE_SM_COUNTS )
	{
		cudaMemcpy(SM_counts_h,  SM_counts_d,	 SIZE_IMAGE_INT,   cudaMemcpyDeviceToHost);
		array_2_disk("SM_counts", OUTPUTIRECTORY, OUTPUT_FOLDER, SM_counts_h, parameters.COLUMNS, parameters.ROWS, parameters.SLICES, parameters.NUM_VOXELS, false );
	}*/

	int* SM_differences_h = (int*) calloc( parameters.NUM_VOXELS, sizeof(int) );
	int* SM_differences_d;	
	cudaMalloc((void**) &SM_differences_d, SIZE_IMAGE_INT );
	cudaMemcpy( SM_differences_d, SM_differences_h, SIZE_IMAGE_INT, cudaMemcpyHostToDevice );

	dim3 dimBlock( parameters.SLICES );
	dim3 dimGrid( parameters.COLUMNS, parameters.ROWS );   
	carve_differences<<< dimGrid, dimBlock >>>( parameters_d, SM_differences_d, SM_counts_d );
	
	cudaMemcpy( SM_differences_h, SM_differences_d, SIZE_IMAGE_INT, cudaMemcpyDeviceToHost );

	int* SM_thresholds_h = (int*) calloc( parameters.SLICES, sizeof(int) );
	int voxel;	
	int max_difference = 0;
	for( uint slice = 0; slice < parameters.SLICES; slice++ )
	{
		for( uint pixel = 0; pixel < parameters.COLUMNS * parameters.ROWS; pixel++ )
		{
			voxel = pixel + slice * parameters.COLUMNS * parameters.ROWS;
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
	unsigned int threshold_size = parameters.SLICES * sizeof(int);
	cudaMalloc((void**) &SM_thresholds_d, threshold_size );
	cudaMemcpy( SM_thresholds_d, SM_thresholds_h, threshold_size, cudaMemcpyHostToDevice );

	SM_edge_detection_GPU<<< dimGrid, dimBlock >>>( parameters_d, SM_counts_d, SM_thresholds_d);
	
	puts("SM hull-detection and edge-detection complete.");
	cudaFree( SM_differences_d );
	cudaFree( SM_thresholds_d );

	free(SM_differences_h);
	free(SM_thresholds_h);
}
__global__ void SM_edge_detection_GPU( configurations* parameters, int* SM_counts, int* SM_threshold )
{
	int row = blockIdx.y, column = blockIdx.x, slice = threadIdx.x;
	float x = ( column - parameters->COLUMNS/2 + 0.5 ) * parameters->VOXEL_WIDTH;
	float y = ( parameters->ROWS/2 - row - 0.5 ) * parameters->VOXEL_HEIGHT;
	int voxel = column + row * parameters->COLUMNS + slice * parameters->COLUMNS * parameters->ROWS;
	if( voxel < parameters->NUM_VOXELS )
	{
		if( SM_counts[voxel] > parameters->SM_SCALE_THRESHOLD * SM_threshold[slice] )
			SM_counts[voxel] = 1;
		else
			SM_counts[voxel] = 0;
		if( pow(x, 2) + pow(y, 2) >= pow(parameters->RECON_CYL_RADIUS - max(parameters->VOXEL_WIDTH, parameters->VOXEL_HEIGHT)/2, 2 ) )
			SM_counts[voxel] = 0;
	}
}
void SM_edge_detection_2()
{
	puts("Performing edge-detection on SM_counts...");

	// Copy the space modeled image from the GPU to the CPU and write it to file.
	cudaMemcpy(SM_counts_h,  SM_counts_d,	 SIZE_IMAGE_INT,   cudaMemcpyDeviceToHost);
	array_2_disk("SM_counts", PREPROCESSING_DIR, TEXT, SM_counts_h, parameters.COLUMNS, parameters.ROWS, parameters.SLICES, parameters.NUM_VOXELS, false );

	int* SM_differences_h = (int*) calloc( parameters.NUM_VOXELS, sizeof(int) );
	int* SM_differences_d;
	cudaMalloc((void**) &SM_differences_d, SIZE_IMAGE_INT );
	cudaMemcpy( SM_differences_d, SM_differences_h, SIZE_IMAGE_INT, cudaMemcpyHostToDevice );

	dim3 dimBlock( parameters.SLICES );
	dim3 dimGrid( parameters.COLUMNS, parameters.ROWS );   

	carve_differences<<< dimGrid, dimBlock >>>( parameters_d, SM_differences_d, SM_counts_d );
	cudaMemcpy( SM_differences_h, SM_differences_d, SIZE_IMAGE_INT, cudaMemcpyDeviceToHost );

	int* SM_thresholds_h = (int*) calloc( parameters.SLICES, sizeof(int) );
	int voxel;	
	int max_difference = 0;
	for( uint slice = 0; slice < parameters.SLICES; slice++ )
	{
		for( uint pixel = 0; pixel < parameters.COLUMNS * parameters.ROWS; pixel++ )
		{
			voxel = pixel + slice * parameters.COLUMNS * parameters.ROWS;
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
	unsigned int threshold_size = parameters.SLICES * sizeof(int);
	cudaMalloc((void**) &SM_thresholds_d, threshold_size );
	cudaMemcpy( SM_thresholds_d, SM_thresholds_h, threshold_size, cudaMemcpyHostToDevice );

	SM_edge_detection_GPU<<< dimGrid, dimBlock >>>( parameters_d, SM_counts_d, SM_thresholds_d);

	puts("SM hull-detection complete.  Writing results to disk...");
	
	cudaFree( SM_differences_d );
	cudaFree( SM_thresholds_d );

	free(SM_differences_h);
	free(SM_thresholds_h);
}
__global__ void SM_edge_detection_GPU_2( configurations* parameters, int* SM_counts, int* SM_differences )
{
	int row = blockIdx.y, column = blockIdx.x, slice = threadIdx.x;
	int voxel = column + row * parameters->COLUMNS + slice * parameters->COLUMNS * parameters->ROWS;
	int difference, max_difference = 0;
	if( (row != 0) && (row != parameters->ROWS - 1) && (column != 0) && (column != parameters->COLUMNS - 1) )
	{
		for( int current_row = row - 1; current_row <= row + 1; current_row++ )
		{
			for( int current_column = column - 1; current_column <= column + 1; current_column++ )
			{
				difference = SM_counts[voxel] - SM_counts[current_column + current_row * parameters->COLUMNS + slice * parameters->COLUMNS * parameters->ROWS];
				if( difference > max_difference )
					max_difference = difference;
			}
		}
		SM_differences[voxel] = max_difference;
	}
	syncthreads();
	int slice_threshold;
	max_difference = 0;
	for( int pixel = 0; pixel < parameters->COLUMNS * parameters->ROWS; pixel++ )
	{
		voxel = pixel + slice * parameters->COLUMNS * parameters->ROWS;
		if( SM_differences[voxel] > max_difference )
		{
			max_difference = SM_differences[voxel];
			slice_threshold = SM_counts[voxel];
		}
	}
	syncthreads();
	float x = ( column - parameters->COLUMNS/2 + 0.5 ) * parameters->VOXEL_WIDTH;
	float y = ( parameters->ROWS/2 - row - 0.5 ) * parameters->VOXEL_HEIGHT;
	if( voxel < parameters->NUM_VOXELS )
	{
		if( SM_counts[voxel] > parameters->SM_SCALE_THRESHOLD * slice_threshold )
			SM_counts[voxel] = 1;
		else
			SM_counts[voxel] = 0;
		if( pow(x, 2) + pow(y, 2) >= pow(parameters->RECON_CYL_RADIUS - max(parameters->VOXEL_WIDTH, parameters->VOXEL_HEIGHT)/2, 2 ) )
			SM_counts[voxel] = 0;
	}
}
void hull_detection_finish()
{
	if( parameters.SC_ON )
	{
		SC_hull_h = (bool*) calloc( parameters.NUM_VOXELS, sizeof(bool) );
		cudaMemcpy(SC_hull_h,  SC_hull_d, SIZE_IMAGE_BOOL, cudaMemcpyDeviceToHost);
		if( parameters.HULL_TYPE != SC_HULL )
		{
			free( SC_hull_h );
			cudaFree(SC_hull_d);
		}	
	}
	if( parameters.MSC_ON )
	{
		MSC_counts_h = (int*) calloc( parameters.NUM_VOXELS, sizeof(int) );
		if( parameters.WRITE_MSC_COUNTS )
		{		
			puts("Writing MSC counts to disk...");		
			cudaMemcpy(MSC_counts_h,  MSC_counts_d, SIZE_IMAGE_INT, cudaMemcpyDeviceToHost);
			array_2_disk("MSC_counts_h", PREPROCESSING_DIR, TEXT, MSC_counts_h, parameters.COLUMNS, parameters.ROWS, parameters.SLICES, parameters.NUM_VOXELS, true );	
		}
		if( (parameters.HULL_TYPE == MSC_HULL) )
		{
			MSC_edge_detection();
			cudaMemcpy(MSC_counts_h,  MSC_counts_d, SIZE_IMAGE_INT, cudaMemcpyDeviceToHost);
			puts("Writing MSC hull to disk...");		
			array_2_disk("hull_MSC", PREPROCESSING_DIR, TEXT, MSC_counts_h, parameters.COLUMNS, parameters.ROWS, parameters.SLICES, parameters.NUM_VOXELS, true );	
		}
		if( parameters.HULL_TYPE != MSC_HULL )
		{
			free( MSC_counts_h );		
			cudaFree( MSC_counts_d );
		}
	}
	if( parameters.SM_ON )
	{
		SM_counts_h = (int*) calloc( parameters.NUM_VOXELS, sizeof(int) );
		if( parameters.WRITE_SM_COUNTS )
		{		
			puts("Writing SM counts to disk...");
			cudaMemcpy(SM_counts_h,  SM_counts_d, SIZE_IMAGE_INT, cudaMemcpyDeviceToHost);
			array_2_disk("SM_counts_h", PREPROCESSING_DIR, TEXT, SM_counts_h, parameters.COLUMNS, parameters.ROWS, parameters.SLICES, parameters.NUM_VOXELS, true );	
		}
		if( parameters.HULL_TYPE == SM_HULL ) 
		{
			SM_edge_detection();
			cudaMemcpy(SM_counts_h,  SM_counts_d, SIZE_IMAGE_INT, cudaMemcpyDeviceToHost);
			puts("Writing SM hull to disk...");		
			array_2_disk("hull_SM", PREPROCESSING_DIR, TEXT, SM_counts_h, parameters.COLUMNS, parameters.ROWS, parameters.SLICES, parameters.NUM_VOXELS, true );	
		}
		if( parameters.HULL_TYPE != SM_HULL )
		{
			free( SM_counts_h );
			cudaFree( SM_counts_d );
		}
	}
}
void define_hull()
{
	puts("Performing hull selection...");

	hull_h = (bool*) calloc( parameters.NUM_VOXELS, sizeof(bool) );
	switch( parameters.HULL_TYPE )
	{
		case IMPORT_HULL	: import_image( hull_h, PREPROCESSING_DIR, HULL_BASENAME, TEXT );													break;
		case SC_HULL		: hull_h = SC_hull_h;																							break;
		case MSC_HULL		: std::transform( MSC_counts_h, MSC_counts_h + parameters.NUM_VOXELS, MSC_counts_h, hull_h, std::logical_or<int>() );		break;
		case SM_HULL		: std::transform( SM_counts_h,  SM_counts_h + parameters.NUM_VOXELS,  SM_counts_h,  hull_h, std::logical_or<int> () );		break;
		case FBP_HULL		: hull_h = FBP_hull_h;								
	}
	puts("Writing selected hull to disk and transferring it to GPU...");
	array_2_disk(HULL_BASENAME, PREPROCESSING_DIR, TEXT, hull_h, parameters.COLUMNS, parameters.ROWS, parameters.SLICES, parameters.NUM_VOXELS, true );

	// Allocate memory for and transfer hull to the GPU
	cudaMalloc((void**) &hull_d, SIZE_IMAGE_BOOL );
	cudaMemcpy( hull_d, hull_h, SIZE_IMAGE_BOOL, cudaMemcpyHostToDevice );

	if( parameters.AVG_FILTER_HULL && parameters.MEDIAN_FILTER_HULL)
	{
		puts("Performing average and median filtering of hull...");
		averaging_filter( hull_h, hull_d, parameters.HULL_AVG_FILTER_RADIUS, true, parameters.HULL_RSP_THRESHOLD );
		median_filter_2D( hull_h, parameters.HULL_MED_FILTER_RADIUS );
		cudaMemcpy( hull_d, hull_h, SIZE_IMAGE_BOOL, cudaMemcpyHostToDevice );
		puts("Average and median filtering of hull complete.  Writing filtered hull to disk...");
	}
	else if( parameters.AVG_FILTER_HULL )
	{
		puts("Performing average filtering of hull...");
		averaging_filter( hull_h, hull_d, parameters.HULL_AVG_FILTER_RADIUS, true, parameters.HULL_RSP_THRESHOLD );
		puts("Average filtering of hull complete.  Writing filtered hull to disk...");
	}
	else if( parameters.MEDIAN_FILTER_HULL )
	{
		puts("Performing median filtering of hull...");
		median_filter_2D( hull_h, parameters.HULL_MED_FILTER_RADIUS );
		cudaMemcpy( hull_d, hull_h, SIZE_IMAGE_BOOL, cudaMemcpyHostToDevice );
		puts("Median filtering of hull complete.  Writing filtered hull to disk...");
	}	
	if( parameters.AVG_FILTER_HULL ||  parameters.MEDIAN_FILTER_HULL )
		array_2_disk(HULL_2_USE_FILENAME, PREPROCESSING_DIR, TEXT, hull_h, parameters.COLUMNS, parameters.ROWS, parameters.SLICES, parameters.NUM_VOXELS, true );	
	if( parameters.EXPORT_PREPROCESSING )
		export_hull();
	puts("Hull selection complete."); 
}
void define_x_0()
{
	puts("Defining initital iterate...");

	x_h = (float*) calloc( parameters.NUM_VOXELS, sizeof(float) );

	switch( parameters.X_0_TYPE )
	{	
		case IMPORT_X_0	:	import_image( x_h, X_0_PATH, X_0_BASENAME, TEXT );														break;
		case X_HULL		:	std::copy( hull_h, hull_h + parameters.NUM_VOXELS, x_h );												break;												
		case X_FBP		:	x_h = FBP_h; 																							break;
		case HYBRID		:	std::transform(FBP_h, FBP_h + parameters.NUM_VOXELS, hull_h, x_h, std::multiplies<float>() );			break;	
		case ZEROS		:	break;
		default			:	puts("ERROR: Invalid initial iterate selected");
							exit(1);
	}
	puts("Writing hull to disk...");
	array_2_disk(X_0_FILENAME, PREPROCESSING_DIR, TEXT, x_h, parameters.COLUMNS, parameters.ROWS, parameters.SLICES, parameters.NUM_VOXELS, true );
	cudaMalloc((void**) &x_d, SIZE_IMAGE_FLOAT );
	cudaMemcpy( x_d, x_h, SIZE_IMAGE_FLOAT, cudaMemcpyHostToDevice );
		
	if( parameters.AVG_FILTER_X_0 && parameters.MEDIAN_FILTER_X_0)
	{
		puts("Performing average and median filtering of initial iterate...");
		averaging_filter( x_h, x_d, parameters.X_0_AVG_FILTER_RADIUS, false, 0.0 );
		median_filter_2D( x_h, parameters.X_0_MED_FILTER_RADIUS );
		cudaMemcpy( x_d, x_h, SIZE_IMAGE_FLOAT, cudaMemcpyHostToDevice );
		puts("Average and median filtering of hull complete.  Writing filtered hull to disk...");
	}
	else if( parameters.AVG_FILTER_X_0 )
	{
		puts("Performing average and median filtering of initial iterate...");
		averaging_filter( x_h, x_d, parameters.X_0_AVG_FILTER_RADIUS, false, 0.0 );
		puts("Average filtering of hull complete.  Writing filtered hull to disk...");
	}
	else if( parameters.MEDIAN_FILTER_X_0 )
	{
		puts("Performing average and median filtering of initial iterate...");
		median_filter_2D( x_h, parameters.X_0_MED_FILTER_RADIUS );
		cudaMalloc((void**) &x_d, SIZE_IMAGE_FLOAT );
		cudaMemcpy( x_d, x_h, SIZE_IMAGE_FLOAT, cudaMemcpyHostToDevice );
		puts("Median filtering of hull complete.  Writing filtered hull to disk...");
	}	

	if( parameters.AVG_FILTER_X_0 ||  parameters.MEDIAN_FILTER_X_0 )
		array_2_disk(X_0_2_USE_FILENAME, PREPROCESSING_DIR, TEXT, x_h, parameters.COLUMNS, parameters.ROWS, parameters.SLICES, parameters.NUM_VOXELS, true );	
	if( parameters.EXPORT_PREPROCESSING )
		export_x_0();
	puts("Deinition of initial iterate selection complete."); 
}
/***********************************************************************************************************************************************************************************************************************/
/*********************************************************************************************** Image filtering functions *********************************************************************************************/
/***********************************************************************************************************************************************************************************************************************/
template<typename H, typename D> void averaging_filter( H*& image_h, D*& image_d, int radius, bool perform_threshold, double threshold_value )
{
	//bool is_hull = ( typeid(bool) == typeid(D) );
	D* new_value_d;
	int new_value_size = parameters.NUM_VOXELS * sizeof(D);
	cudaMalloc(&new_value_d, new_value_size );

	dim3 dimBlock( parameters.SLICES );
	dim3 dimGrid( parameters.COLUMNS, parameters.ROWS );   
	averaging_filter_GPU<<< dimGrid, dimBlock >>>( parameters_d, image_d, new_value_d, radius, perform_threshold, threshold_value );
	//apply_averaging_filter_GPU<<< dimGrid, dimBlock >>>( image_d, new_value_d );
	//cudaFree(new_value_d);
	cudaMemcpy( image_h, new_value_d, SIZE_IMAGE_FLOAT, cudaMemcpyDeviceToHost );
	cudaFree(image_d);
	image_d = new_value_d;
}
template<typename D> __global__ void averaging_filter_GPU( configurations* parameters, D* image, D* new_value, int radius, bool perform_threshold, double threshold_value )
{
	int voxel_x = blockIdx.x;
	int voxel_y = blockIdx.y;	
	int voxel_z = threadIdx.x;
	int voxel = voxel_x + voxel_y * parameters->COLUMNS + voxel_z * parameters->COLUMNS * parameters->ROWS;
	unsigned int left_edge = max( voxel_x - radius, 0 );
	unsigned int right_edge = min( voxel_x + radius, parameters->COLUMNS - 1);
	unsigned int top_edge = max( voxel_y - radius, 0 );
	unsigned int bottom_edge = min( voxel_y + radius, parameters->ROWS - 1);	
	int neighborhood_voxels = ( right_edge - left_edge + 1 ) * ( bottom_edge - top_edge + 1 );
	double sum_threshold = neighborhood_voxels * threshold_value;
	double sum = 0.0;
	// Determine neighborhood sum for voxels whose neighborhood is completely enclosed in image
	// Strip of size floor(AVG_FILTER_SIZE/2) around image perimeter must be ignored
	for( int column = left_edge; column <= right_edge; column++ )
		for( int row = top_edge; row <= bottom_edge; row++ )
			sum += image[column + (row * parameters->COLUMNS) + (voxel_z * parameters->COLUMNS * parameters->ROWS)];
	if( perform_threshold)
		new_value[voxel] = ( sum > sum_threshold );
	else
		new_value[voxel] = sum / neighborhood_voxels;
}
template<typename T, typename T2> __global__ void apply_averaging_filter_GPU( configurations* parameters, T* image, T2* new_value )
{
	int voxel_x = blockIdx.x;
	int voxel_y = blockIdx.y;	
	int voxel_z = threadIdx.x;
	int voxel = voxel_x + voxel_y * parameters->COLUMNS + voxel_z * parameters->COLUMNS * parameters->ROWS;
	image[voxel] = new_value[voxel];
}
template<typename T> void average_filter_2D( T*& input_image, unsigned int radius )
{
	T* average_filtered_image = (T*) calloc( parameters.NUM_VOXELS, sizeof(T) );

	unsigned int neighborhood_voxels = (2*radius + 1 ) * (2*radius + 1 );
	unsigned int target_voxel, voxel;
	double sum = 0;
	for( unsigned int target_slice = 0; target_slice < parameters.SLICES; target_slice++ )
	{
		for( unsigned int target_column = radius; target_column < parameters.COLUMNS - radius; target_column++ )
		{
			for( unsigned int target_row = radius; target_row < parameters.ROWS - radius; target_row++ )
			{
				sum = 0;
				target_voxel = target_column + target_row * parameters.COLUMNS + target_slice * parameters.COLUMNS * parameters.ROWS;
				for( unsigned int column = target_column - radius; column <= target_column + radius; column++ )
				{
					for( unsigned int row = target_row - radius; row <=  target_row + radius; row++ )
					{
						voxel = column + row * parameters.COLUMNS + target_slice * parameters.COLUMNS * parameters.ROWS;
						sum += input_image[voxel];
						//neighborhood[i] = image_h[voxel2];
						//i++;
					}
				}
				average_filtered_image[target_voxel] = sum / neighborhood_voxels;
			}
		}
	}
	//free(input_image);
	//input_image = median_filtered_image;
	std::copy(average_filtered_image, average_filtered_image + parameters.NUM_VOXELS, input_image);
	free(average_filtered_image);
}
template<typename T> void median_filter_2D( T*& input_image, unsigned int radius )
{
	T* median_filtered_image = (T*) calloc( parameters.NUM_VOXELS, sizeof(T) );

	unsigned int neighborhood_voxels = (2*radius + 1 ) * (2*radius + 1 );
	unsigned int middle = neighborhood_voxels/2;
	unsigned int target_voxel, voxel;
	//T* neighborhood = (T*)calloc( neighborhood_voxels, sizeof(T));
	std::vector<T> neighborhood;
	for( unsigned int target_slice = 0; target_slice < parameters.SLICES; target_slice++ )
	{
		for( unsigned int target_column = radius; target_column < parameters.COLUMNS - radius; target_column++ )
		{
			for( unsigned int target_row = radius; target_row < parameters.ROWS - radius; target_row++ )
			{
				target_voxel = target_column + target_row * parameters.COLUMNS + target_slice * parameters.COLUMNS * parameters.ROWS;
				for( unsigned int column = target_column - radius; column <= target_column + radius; column++ )
				{
					for( unsigned int row = target_row - radius; row <=  target_row + radius; row++ )
					{
						voxel = column + row * parameters.COLUMNS + target_slice * parameters.COLUMNS * parameters.ROWS;
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
	//input_image = (T*) calloc( parameters.NUM_VOXELS, sizeof(T) );
	//std::copy(median_filtered_image, median_filtered_image + parameters.NUM_VOXELS, input_image);
	

}
template<typename T> void median_filter_2D( T*& input_image, T*& median_filtered_image, unsigned int radius )
{
	//T* median_filtered_image = (T*) calloc( parameters.NUM_VOXELS, sizeof(T) );

	unsigned int neighborhood_voxels = (2*radius + 1 ) * (2*radius + 1 );
	unsigned int middle = neighborhood_voxels/2;
	unsigned int target_voxel, voxel;
	//T* neighborhood = (T*)calloc( neighborhood_voxels, sizeof(T));
	std::vector<T> neighborhood;
	for( unsigned int target_slice = 0; target_slice < parameters.SLICES; target_slice++ )
	{
		for( unsigned int target_column = radius; target_column < parameters.COLUMNS - radius; target_column++ )
		{
			for( unsigned int target_row = radius; target_row < parameters.ROWS - radius; target_row++ )
			{
				target_voxel = target_column + target_row * parameters.COLUMNS + target_slice * parameters.COLUMNS * parameters.ROWS;
				for( unsigned int column = target_column - radius; column <= target_column + radius; column++ )
				{
					for( unsigned int row = target_row - radius; row <=  target_row + radius; row++ )
					{
						voxel = column + row * parameters.COLUMNS + target_slice * parameters.COLUMNS * parameters.ROWS;
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
	//free(input_image);
	//input_image = (T*) calloc( parameters.NUM_VOXELS, sizeof(T) );
	//std::copy(median_filtered_image, median_filtered_image + parameters.NUM_VOXELS, input_image);
	//input_image = median_filtered_image;

}
template<typename D> __global__ void median_filter_GPU( configurations* parameters, D* image, D* new_value, int radius, bool perform_threshold, double threshold_value )
{
//	int voxel_x = blockIdx.x;
//	int voxel_y = blockIdx.y;	
//	int voxel_z = threadIdx.x;
//	int voxel = voxel_x + voxel_y * parameters->COLUMNS + voxel_z * parameters->COLUMNS * parameters->ROWS;
//	unsigned int left_edge = max( voxel_x - radius, 0 );
//	unsigned int right_edge = min( voxel_x + radius, parameters->COLUMNS - 1);
//	unsigned int top_edge = max( voxel_y - radius, 0 );
//	unsigned int bottom_edge = min( voxel_y + radius, parameters->ROWS - 1);	
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
//			current_value =  image[column + (row * parameters->COLUMNS) + (voxel_z * parameters->COLUMNS * parameters->ROWS)];
//			for( int column2 = left_edge; column2 <= right_edge; column2++ )
//			{
//				for( int row2 = top_edge; row2 <= bottom_edge; row2++ )
//				{
//					if(  image[column2 + (row2 * parameters->COLUMNS) + (voxel_z * parameters->COLUMNS * parameters->ROWS)] < current_value)
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
template<typename T> void median_filter_3D( T*& input_image, T*& median_filtered_image, unsigned int radius )
{
	//T* median_filtered_image = (T*) calloc( parameters.NUM_VOXELS, sizeof(T) );

	unsigned int neighborhood_voxels = (2*radius + 1 ) * (2*radius + 1 )* (2*radius + 1 );
	unsigned int middle = neighborhood_voxels/2;
	unsigned int target_voxel, voxel;
	//T* neighborhood = (T*)calloc( neighborhood_voxels, sizeof(T));
	std::vector<T> neighborhood;
	for( unsigned int target_slice = 0; target_slice < parameters.SLICES; target_slice++ )
	{
		for( unsigned int target_column = radius; target_column < parameters.COLUMNS - radius; target_column++ )
		{
			for( unsigned int target_row = radius; target_row < parameters.ROWS - radius; target_row++ )
			{
				target_voxel = target_column + target_row * parameters.COLUMNS + target_slice * parameters.COLUMNS * parameters.ROWS;
				if( (target_slice >= radius ) && (target_slice < parameters.SLICES - radius) )
				{					
					neighborhood_voxels = (2*radius + 1 ) * (2*radius + 1 ) * (2*radius + 1 );
					middle = neighborhood_voxels/2;
					for( unsigned int slice = target_slice - radius; slice <= target_slice + radius; slice++ )
					{
						for( unsigned int column = target_column - radius; column <= target_column + radius; column++ )
						{
							for( unsigned int row = target_row - radius; row <=  target_row + radius; row++ )
							{
								voxel = column + row * parameters.COLUMNS + slice * parameters.COLUMNS * parameters.ROWS;
								neighborhood.push_back(input_image[voxel]);
								//neighborhood[i] = image_h[voxel2];
								//i++;
							}
						}
					}
				}
				else
				{
					neighborhood_voxels = (2*radius + 1 ) * (2*radius + 1 );
					middle = neighborhood_voxels/2;
					for( unsigned int column = target_column - radius; column <= target_column + radius; column++ )
					{
						for( unsigned int row = target_row - radius; row <=  target_row + radius; row++ )
						{
							voxel = column + row * parameters.COLUMNS + target_slice * parameters.COLUMNS * parameters.ROWS;
							neighborhood.push_back(input_image[voxel]);
							//neighborhood[i] = image_h[voxel2];
							//i++;
						}
					}
				}
				std::sort( neighborhood.begin(), neighborhood.end());
				median_filtered_image[target_voxel] = neighborhood[middle];
				neighborhood.clear();
			}
		}
	}

}
template<typename T> void test_median_filter_radii(T*& image, char* output_basename )
{
	float* median_filtered_2D_h = (float*) calloc( parameters.NUM_VOXELS, sizeof(float) );	
	float* median_filtered_3D_h = (float*) calloc( parameters.NUM_VOXELS, sizeof(float) );	
	char basename2D_w_radius[256];
	char basename3D_w_radius[256];
	uint widths[] = {3,5,7};
	uint radius;
	std::vector<uint> filter_widths(widths, widths + sizeof(widths) / sizeof(uint) );

	for( int i = 0; i < filter_widths.size(); i++ )
	{
		radius = filter_widths[i];
		printf("Generating all possible median filtered images for filter width %d...\n", radius);
		median_filter_2D( image, median_filtered_2D_h, radius);
		printf("2D median filter of radius %d applied to image.\n", radius);
		median_filter_3D( image, median_filtered_3D_h, radius );
		printf("3D median filter of radius %d applied to image.\n", radius);
		sprintf(basename2D_w_radius, "%s_2D_%d", output_basename,  radius );
		sprintf(basename3D_w_radius, "%s_3D_%d", output_basename,  radius );
		array_2_disk(basename2D_w_radius, PREPROCESSING_DIR, TEXT, median_filtered_2D_h, parameters.COLUMNS, parameters.ROWS, parameters.SLICES, parameters.NUM_VOXELS, true );
		array_2_disk(basename2D_w_radius, PREPROCESSING_DIR, BINARY, median_filtered_2D_h, parameters.COLUMNS, parameters.ROWS, parameters.SLICES, parameters.NUM_VOXELS, true );
		array_2_disk(basename3D_w_radius, PREPROCESSING_DIR, TEXT, median_filtered_3D_h, parameters.COLUMNS, parameters.ROWS, parameters.SLICES, parameters.NUM_VOXELS, true );
		array_2_disk(basename3D_w_radius, PREPROCESSING_DIR, BINARY, median_filtered_3D_h, parameters.COLUMNS, parameters.ROWS, parameters.SLICES, parameters.NUM_VOXELS, true );
		printf("2D/3D median filter of radius %d applied to image and written to disk in text and binary formats\n", radius);
	}
	//char check_bin_convert[256];
	//sprintf(check_bin_convert, "FBP_median_3D_7" );
	//binary_2_txt_images( PREPROCESSING_DIR, check_bin_convert, median_filtered_3D_h );
	free(median_filtered_2D_h);
	free(median_filtered_3D_h);
	//binary_2_txt_images( PREPROCESSING_DIR, basename2D_w_radius, median_filtered_2D_h );
}
/***********************************************************************************************************************************************************************************************************************/
/****************************************************************************************** Root Mean Square (RMS) error analysis **************************************************************************************/
/***********************************************************************************************************************************************************************************************************************/
template<typename I1, typename I2> double calculate_RMS_error( I1*& image_1, I2*& image_2, unsigned int num_elements )
{
	double RMS_error = 0.0;
	for( int i = 0; i < num_elements; i++ )
	{
		RMS_error += pow( image_1[i] - image_2[i], 2 );
	}
	return sqrt( RMS_error / num_elements );
}
template<typename I1, typename I2> double calculate_RMS_error( std::vector<I1> image_1, std::vector<I2> image_2, unsigned int num_elements )
{
	double RMS_error = 0.0;
	for( int i = 0; i < num_elements; i++ )
	{
		RMS_error += pow( image_1[i] - image_2[i], 2 );
	}
	return sqrt( RMS_error / num_elements );
}
double perform_RMS_analysis( char* directory, char* filename_1, char* filename_2, bool arrays )
{
	unsigned int num_voxels_1, num_voxels_2;
	double RMS_error;
	if( arrays )
	{
		float* image_1, * image_2;
		num_voxels_1 = import_image( image_1, directory, filename_1, TEXT );
		num_voxels_2 = import_image( image_2, directory, filename_2, TEXT );
		RMS_error = calculate_RMS_error( image_1, image_2, num_voxels_1 );
	}
	else
	{
		std::vector<float> image_1_vec;
		std::vector<float> image_2_vec;
		num_voxels_1 = import_image( image_1_vec, directory, filename_1, TEXT );
		num_voxels_2 = import_image( image_2_vec, directory, filename_2, TEXT );
		RMS_error = calculate_RMS_error( image_1_vec, image_2_vec, num_voxels_1 );
	}
	return RMS_error;
}
/***********************************************************************************************************************************************************************************************************************/
/**************************************************************************************************** History Ordering *************************************************************************************************/
/***********************************************************************************************************************************************************************************************************************/
void generate_history_sequence(ULL N, ULL offset_prime )
{
	puts("Generating cyclic and exhaustive ordering of histories...");
	history_sequence = (ULL*)calloc( reconstruction_histories, sizeof(ULL));
    history_sequence[0] = 0;
    for( ULL i = 1; i < N; i++ )
        history_sequence[i] = ( history_sequence[i-1] + offset_prime ) % N;
	puts("History order generation complete.");
}
void verify_history_sequence(ULL N, ULL offset_prime )
{
	for( ULL i = 1; i < N; i++ )
    {
        if(history_sequence[i] == 1)
        {
            printf("repeats at i = %llu\n", i);
            printf("sequence[i] = %llu\n", history_sequence[i]);
            break;
        }
        if(history_sequence[i] > N)
        {
            printf("exceeds at i = %llu\n", i);
            printf("sequence[i] = %llu\n", history_sequence[i]);
            break;
        }
    }
}
void print_history_sequence(ULL print_start, ULL print_end )
{
    for( ULL i = print_start; i < print_end; i++ )
		printf("history_sequence[i] = %llu\n", history_sequence[i]);
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
void generate_scattering_coefficient_table()
{
	double scattering_coefficient;
	scattering_table_file = fopen( COEFFICIENT_FILENAME, "wb" );
	int i = 0;
	double depth = 0.0;
	scattering_table_h = (double*)calloc( DEPTH_TABLE_ELEMENTS + 1, sizeof(double));
	for( int step_num = 0; step_num <= DEPTH_TABLE_ELEMENTS; step_num++ )
	{
		depth = step_num * DEPTH_TABLE_STEP;
		scattering_coefficient = pow( E_0 * ( 1 + 0.038 * log(depth / X0) ), 2.0 ) / X0;
		scattering_table_h[i] = scattering_coefficient;
		//fwrite( &scattering_coefficient, sizeof(float), 1, scattering_table_file );
		i++;
	}
	//for( float depth = 0.0; depth <= DEPTH_TABLE_RANGE; depth+= DEPTH_TABLE_STEP )
	//{
	//	scattering_coefficient = pow( E_0 * ( 1 + 0.038 * log(depth / X0) ), 2.0 ) / X0;
	//	scattering_table_h[i] = scattering_coefficient;
	//	//fwrite( &scattering_coefficient, sizeof(float), 1, scattering_table_file );
	//	i++;
	//}
	fwrite(scattering_table_h, sizeof(double), DEPTH_TABLE_ELEMENTS + 1, scattering_table_file );
	fclose(scattering_table_file);
	//for( int step_num = 0; step_num <= DEPTH_TABLE_ELEMENTS; step_num++ )
	//	cout << scattering_table_h[step_num] << endl;
	cout << "elements = " << i << endl;
	cout << "DEPTH_TABLE_ELEMENTS = " << DEPTH_TABLE_ELEMENTS << endl;
	//cout << scattering_table_h[i-1] << endl;
	//cout << (pow( E_0 * ( 1 + 0.038 * log(DEPTH_TABLE_RANGE / X0) ), 2.0 ) / X0) << endl;
	
}
void import_scattering_coefficient_table()
{
	scattering_table_h = (double*)calloc( DEPTH_TABLE_ELEMENTS + 1, sizeof(double));
	scattering_table_file = fopen( COEFFICIENT_FILENAME, "rb" );
	fread(scattering_table_h, sizeof(double), DEPTH_TABLE_ELEMENTS + 1, scattering_table_file );
	fclose(scattering_table_file);
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
	cout << "i = " << i << endl;														
	cout << "POLY_TABLE_ELEMENTS = " << POLY_TABLE_ELEMENTS << endl;			
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
void tables_2_GPU()
{
	unsigned int size_trig_tables			= ( TRIG_TABLE_ELEMENTS	 + 1 ) * sizeof(double);
	unsigned int size_coefficient_tables	= ( DEPTH_TABLE_ELEMENTS + 1 ) * sizeof(double);
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
/***********************************************************************************************************************************************************************************************************************/
/***************************************************************************************************** Chord Length ****************************************************************************************************/
/***********************************************************************************************************************************************************************************************************************/
double mean_chord_length2( double x_entry, double y_entry, double z_entry, double x_exit, double y_exit, double z_exit, double xy_dim, double z_dim )
{
    double xy_angle = atan2( y_exit - y_entry, x_exit - x_entry);
    double xz_angle = atan2( z_exit - z_entry, x_exit - x_entry);

    double max_value_xy = xy_dim;
    double min_value_xy = xy_dim/sqrt(2.0);
    double range_xy = max_value_xy - min_value_xy;
    double A_xy = range_xy/2;
    double base_level_xy = (max_value_xy + min_value_xy)/2;
    double xy_dist_sqd = pow(base_level_xy + A_xy * cos(4*xy_angle), 2.0);

    //double max_value_xz = z_dim;
    //double min_value_xz = z_dim/sqrt(2.0);
    //double range_xz = max_value_xz - min_value_xz;
    //double A_xz = range_xz/2;
    //double base_level_xz = (max_value_xz + min_value_xz)/2;
    //double xz_dist_sqd = pow(base_level_xz + A_xz * cos(4*xz_angle), 2.0);
    double xz_dist_sqd = pow(sin(xz_angle), 2.0);

    return sqrt( xy_dist_sqd + xz_dist_sqd);
}
double mean_chord_length( double x_entry, double y_entry, double z_entry, double x_exit, double y_exit, double z_exit )
{

	double xy_angle = atan2( y_exit - y_entry, x_exit - x_entry);
	double xz_angle = atan2( z_exit - z_entry, x_exit - x_entry);

	//double int_part;
	//double reduced_angle = modf( xy_angle/(PI/2), &int_part );
	double reduced_angle = xy_angle - ( static_cast<int>( xy_angle/(PI/2) ) ) * (PI/2);
	double effective_angle_ut = fabs(reduced_angle);
	double effective_angle_uv = fabs(xz_angle );
	//
	double average_pixel_size = ( parameters.VOXEL_WIDTH + parameters.VOXEL_HEIGHT) / 2;
	double s = parameters.MLP_U_STEP;
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

	double chord_length_t = ( l / 6.0 * sum_ut_angles) * ( pow(s/l, 3.0) * pow( sin(2 * effective_angle_ut), 2.0 ) - 12 * sum_ut_angles ) / ( (s/l) * sin(2 * effective_angle_ut) - 2 * sum_ut_angles );
	
	// Multiply this by the effective chord in the v-u plane
	double mean_pixel_width = average_pixel_size / sum_ut_angles;
	double height_fraction = parameters.SLICE_THICKNESS / mean_pixel_width;
	s = parameters.MLP_U_STEP;
	l = mean_pixel_width;
	double chord_length_v = ( l / (6.0 * height_fraction * sum_uv_angles) ) * ( pow(s/l, 3.0) * pow( sin(2 * effective_angle_uv), 2.0 ) - 12 * height_fraction * sum_uv_angles ) / ( (s/l) * sin(2 * effective_angle_uv) - 2 * height_fraction * sum_uv_angles );
	return sqrt(chord_length_t * chord_length_t + chord_length_v*chord_length_v);

	//double eff_angle_t,eff_angle_v;
	//
	////double xy_angle = atan2( y_exit - y_entry, x_exit - x_entry);
	////double xz_angle = atan2( z_exit - z_entry, x_exit - x_entry);

	////eff_angle_t = modf( xy_angle/(PI/2), &int_part );
	////eff_angle_t = fabs( eff_angle_t);
	////eff_angle_t = xy_angle - ( static_cast<int> xy_angle/(PI/2) ) ) * (PI/2);
	//eff_angle_t = effective_angle_ut;
	//
	////cout << "eff angle t = " << eff_angle_t << endl;
	//eff_angle_v=fabs(xz_angle);
	//
	//// Get the effective chord in the t-u plane
	//double step_fraction=parameters.MLP_U_STEP/parameters.VOXEL_WIDTH;
	//double chord_length_2D=(1/3.0)*((step_fraction*step_fraction*sin(2*eff_angle_t)-6)/(step_fraction*sin(2*eff_angle_t)-2*(cos(eff_angle_t)+sin(eff_angle_t))) + step_fraction*step_fraction*sin(2*eff_angle_t)/(2*(cos(eff_angle_t)+sin(eff_angle_t))));
	//
	//// Multiply this by the effective chord in the v-u plane
	//double mean_pixel_width=parameters.VOXEL_WIDTH/(cos(eff_angle_t)+sin(eff_angle_t));
	//double height_fraction=parameters.SLICE_THICKNESS/mean_pixel_width;
	//step_fraction=parameters.MLP_U_STEP/mean_pixel_width;
	//double chord_length_3D=(1/3.0)*((step_fraction*step_fraction*sin(2*eff_angle_v)-6*height_fraction)/(step_fraction*sin(2*eff_angle_v)-2*(height_fraction*cos(eff_angle_v)+sin(eff_angle_v))) + step_fraction*step_fraction*sin(2*eff_angle_v)/(2*(height_fraction*cos(eff_angle_v)+sin(eff_angle_v))));
	//
	////cout << "2D = " << chord_length_2D << " 3D = " << chord_length_3D << endl;
	//return parameters.VOXEL_WIDTH*chord_length_2D*chord_length_3D;
}
double EffectiveChordLength(double abs_angle_t, double abs_angle_v)
{
	
	double eff_angle_t,eff_angle_v;
	
	eff_angle_t=abs_angle_t-(static_cast<int>(abs_angle_t/(PI/2)))*(PI/2);
	
	eff_angle_v=fabs(abs_angle_v);
	
	// Get the effective chord in the t-u plane
	double step_fraction=parameters.MLP_U_STEP/parameters.VOXEL_WIDTH;
	double chord_length_2D=(1/3.0)*((step_fraction*step_fraction*sinf(2*eff_angle_t)-6)/(step_fraction*sinf(2*eff_angle_t)-2*(cosf(eff_angle_t)+sinf(eff_angle_t))) + step_fraction*step_fraction*sinf(2*eff_angle_t)/(2*(cosf(eff_angle_t)+sinf(eff_angle_t))));
	
	// Multiply this by the effective chord in the v-u plane
	double mean_pixel_width=parameters.VOXEL_WIDTH/(cosf(eff_angle_t)+sinf(eff_angle_t));
	double height_fraction=parameters.VOXEL_THICKNESS/mean_pixel_width;
	step_fraction=parameters.MLP_U_STEP/mean_pixel_width;
	double chord_length_3D=(1/3.0)*((step_fraction*step_fraction*sinf(2*eff_angle_v)-6*height_fraction)/(step_fraction*sinf(2*eff_angle_v)-2*(height_fraction*cosf(eff_angle_v)+sinf(eff_angle_v))) + step_fraction*step_fraction*sinf(2*eff_angle_v)/(2*(height_fraction*cosf(eff_angle_v)+sinf(eff_angle_v))));
	
	return parameters.VOXEL_WIDTH*chord_length_2D*chord_length_3D;
	 
}
float EffectiveChordLength(float abs_angle_t, float abs_angle_v)
{
	
	float eff_angle_t,eff_angle_v;
	
	eff_angle_t=abs_angle_t-(static_cast<int>(abs_angle_t/(PI/2)))*(PI/2);
	
	eff_angle_v=fabs(abs_angle_v);
	
	// Get the effective chord in the t-u plane
	float step_fraction=parameters.MLP_U_STEP/parameters.VOXEL_WIDTH;
	float chord_length_2D=(1/3.0)*((step_fraction*step_fraction*sinf(2*eff_angle_t)-6)/(step_fraction*sinf(2*eff_angle_t)-2*(cosf(eff_angle_t)+sinf(eff_angle_t))) + step_fraction*step_fraction*sinf(2*eff_angle_t)/(2*(cosf(eff_angle_t)+sinf(eff_angle_t))));
	
	// Multiply this by the effective chord in the v-u plane
	float mean_pixel_width=parameters.VOXEL_WIDTH/(cosf(eff_angle_t)+sinf(eff_angle_t));
	float height_fraction=parameters.VOXEL_THICKNESS/mean_pixel_width;
	step_fraction=parameters.MLP_U_STEP/mean_pixel_width;
	float chord_length_3D=(1/3.0)*((step_fraction*step_fraction*sinf(2*eff_angle_v)-6*height_fraction)/(step_fraction*sinf(2*eff_angle_v)-2*(height_fraction*cosf(eff_angle_v)+sinf(eff_angle_v))) + step_fraction*step_fraction*sinf(2*eff_angle_v)/(2*(height_fraction*cosf(eff_angle_v)+sinf(eff_angle_v))));
	
	return parameters.VOXEL_WIDTH*chord_length_2D*chord_length_3D;
	 
}
/***********************************************************************************************************************************************************************************************************************/
/****************************************************************************************************** MLP (host) *****************************************************************************************************/
/***********************************************************************************************************************************************************************************************************************/
template<typename O> bool find_MLP_endpoints
( 
	O*& image, double x_start, double y_start, double z_start, double xy_angle, double xz_angle, 
	double& x_hit, double& y_hit, double& z_hit, int& voxel_x, int& voxel_y, int& voxel_z, bool entering
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
			xy_angle += PI;
		
		x_move_direction = ( cos(xy_angle) >= 0 ) - ( cos(xy_angle) <= 0 );
		y_move_direction = ( sin(xy_angle) >= 0 ) - ( sin(xy_angle) <= 0 );
		z_move_direction = ( sin(xz_angle) >= 0 ) - ( sin(xz_angle) <= 0 );
		
		if( x_move_direction < 0 )
			z_move_direction *= -1;

		voxel_x = calculate_voxel( parameters.X_ZERO_COORDINATE, x, parameters.VOXEL_WIDTH );
		voxel_y = calculate_voxel( parameters.Y_ZERO_COORDINATE, y, parameters.VOXEL_HEIGHT );
		voxel_z = calculate_voxel( parameters.Z_ZERO_COORDINATE, z, parameters.VOXEL_THICKNESS );

		x_to_go = distance_remaining( parameters.X_ZERO_COORDINATE, x, X_INCREASING_DIRECTION, x_move_direction, parameters.VOXEL_WIDTH, voxel_x );
		y_to_go = distance_remaining( parameters.Y_ZERO_COORDINATE, y, Y_INCREASING_DIRECTION, y_move_direction, parameters.VOXEL_HEIGHT, voxel_y );	
		z_to_go = distance_remaining( parameters.Z_ZERO_COORDINATE, z, Z_INCREASING_DIRECTION, z_move_direction, parameters.VOXEL_THICKNESS, voxel_z );

		voxel = voxel_x + voxel_y * parameters.COLUMNS + voxel_z * parameters.COLUMNS * parameters.ROWS;
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
		/********************************************************************************************/
		/************************* Initialize and Check Exit Conditions *****************************/
		/********************************************************************************************/
		outside_image = static_cast<bool>((voxel_x >= parameters.COLUMNS ) || (voxel_y >= parameters.ROWS ) || (voxel_z >= parameters.SLICES ) || (voxel_x < 0  ) || (voxel_y < 0 ) || (voxel_z < 0 ) );		
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
					//printf("z_to_go <= x_extension && z_to_go <= y_extension\n");					
					voxel_z -= z_move_direction;
					
					z = edge_coordinate( parameters.Z_ZERO_COORDINATE, voxel_z, parameters.VOXEL_THICKNESS, Z_INCREASING_DIRECTION, z_move_direction );					
					x = corresponding_coordinate( dx_dz, z, z_start, x_start );
					y = corresponding_coordinate( dy_dz, z, z_start, y_start );

					x_to_go = distance_remaining( parameters.X_ZERO_COORDINATE, x, X_INCREASING_DIRECTION, x_move_direction, parameters.VOXEL_WIDTH, voxel_x );
					y_to_go = distance_remaining( parameters.Y_ZERO_COORDINATE, y, Y_INCREASING_DIRECTION, y_move_direction, parameters.VOXEL_HEIGHT, voxel_y );	
					z_to_go = parameters.VOXEL_THICKNESS;
				}
				//If Next Voxel Edge is in x or xy Diagonal
				else if( x_extension <= y_extension )
				{
					//printf(" x_extension <= y_extension \n");			
					voxel_x += x_move_direction;

					x = edge_coordinate( parameters.X_ZERO_COORDINATE, voxel_x, parameters.VOXEL_WIDTH, X_INCREASING_DIRECTION, x_move_direction );
					y = corresponding_coordinate( dy_dx, x, x_start, y_start );
					z = corresponding_coordinate( dz_dx, x, x_start, z_start );

					x_to_go = parameters.VOXEL_WIDTH;
					y_to_go = distance_remaining( parameters.Y_ZERO_COORDINATE, y, Y_INCREASING_DIRECTION, y_move_direction, parameters.VOXEL_HEIGHT, voxel_y );
					z_to_go = distance_remaining( parameters.Z_ZERO_COORDINATE, z, Z_INCREASING_DIRECTION, z_move_direction, parameters.VOXEL_THICKNESS, voxel_z );
				}
				// Else Next Voxel Edge is in y
				else
				{
					//printf(" y_extension < x_extension \n");
					voxel_y -= y_move_direction;
					
					y = edge_coordinate( parameters.Y_ZERO_COORDINATE, voxel_y, parameters.VOXEL_HEIGHT, Y_INCREASING_DIRECTION, y_move_direction );
					x = corresponding_coordinate( dx_dy, y, y_start, x_start );
					z = corresponding_coordinate( dz_dy, y, y_start, z_start );

					x_to_go = distance_remaining( parameters.X_ZERO_COORDINATE, x, X_INCREASING_DIRECTION, x_move_direction, parameters.VOXEL_WIDTH, voxel_x );
					y_to_go = parameters.VOXEL_HEIGHT;					
					z_to_go = distance_remaining( parameters.Z_ZERO_COORDINATE, z, Z_INCREASING_DIRECTION, z_move_direction, parameters.VOXEL_THICKNESS, voxel_z );
				}
				if( x_to_go == 0 )
				{
					x_to_go = parameters.VOXEL_WIDTH;
					voxel_x += x_move_direction;
				}
				if( y_to_go == 0 )
				{
					y_to_go = parameters.VOXEL_HEIGHT;
					voxel_y -= y_move_direction;
				}
				if( z_to_go == 0 )
				{
					z_to_go = parameters.VOXEL_THICKNESS;
					voxel_z -= z_move_direction;
				}			
				voxel_z = max(voxel_z, 0 );
				voxel = voxel_x + voxel_y * parameters.COLUMNS + voxel_z * parameters.COLUMNS * parameters.ROWS;
				outside_image = (voxel_x >= parameters.COLUMNS ) || (voxel_y >= parameters.ROWS ) || (voxel_z >= parameters.SLICES ) || (voxel_x < 0  ) || (voxel_y < 0 ) || (voxel_z < 0 );		
				if( !outside_image )
					hit_hull = (image[voxel] == 1);	
				end_walk = outside_image || hit_hull;	
			}// end !end_walk 
		}
		else
		{
			while( !end_walk )
			{
				// Change in x for Move to Voxel Edge in y
				y_extension = y_to_go / delta_yx;
				//If Next Voxel Edge is in x or xy Diagonal
				if( x_to_go <= y_extension )
				{
					//printf(" x_to_go <= y_extension \n");
					voxel_x += x_move_direction;
					
					x = edge_coordinate( parameters.X_ZERO_COORDINATE, voxel_x, parameters.VOXEL_WIDTH, X_INCREASING_DIRECTION, x_move_direction );
					y = corresponding_coordinate( dy_dx, x, x_start, y_start );

					x_to_go = parameters.VOXEL_WIDTH;
					y_to_go = distance_remaining( parameters.Y_ZERO_COORDINATE, y, Y_INCREASING_DIRECTION, y_move_direction, parameters.VOXEL_HEIGHT, voxel_y );
				}
				// Else Next Voxel Edge is in y
				else
				{
					//printf(" y_extension < x_extension \n");				
					voxel_y -= y_move_direction;

					y = edge_coordinate( parameters.Y_ZERO_COORDINATE, voxel_y, parameters.VOXEL_HEIGHT, Z_INCREASING_DIRECTION, y_move_direction );
					x = corresponding_coordinate( dx_dy, y, y_start, x_start );

					x_to_go = distance_remaining( parameters.X_ZERO_COORDINATE, x, X_INCREASING_DIRECTION, x_move_direction, parameters.VOXEL_WIDTH, voxel_x );
					y_to_go = parameters.VOXEL_HEIGHT;
				}
				if( x_to_go == 0 )
				{
					x_to_go = parameters.VOXEL_WIDTH;
					voxel_x += x_move_direction;
				}
				if( y_to_go == 0 )
				{
					y_to_go = parameters.VOXEL_HEIGHT;
					voxel_y -= y_move_direction;
				}
				voxel = voxel_x + voxel_y * parameters.COLUMNS + voxel_z * parameters.COLUMNS * parameters.ROWS;		
				outside_image = (voxel_x >= parameters.COLUMNS ) || (voxel_y >= parameters.ROWS ) || (voxel_z >= parameters.SLICES ) || (voxel_x < 0  ) || (voxel_y < 0 ) || (voxel_z < 0 );		
				if( !outside_image )
					hit_hull = (image[voxel] == 1);		
				end_walk = outside_image || hit_hull;	
			}// end: while( !end_walk )
		}//end: else: z_start != z_end => z_start == z_end
		if( hit_hull )
		{
			x_hit = x;
			y_hit = y;
			z_hit = z;
		}
		return hit_hull;
}
void collect_MLP_endpoints()
{
	puts("Calculating hull entry/exit coordinates and writing results to disk...");
	/*************************************************************************************************************************************************************************/
	/***************************************************************** Variable Declarations and Instantiations **************************************************************/
	/*************************************************************************************************************************************************************************/
	double x_in_object, y_in_object, z_in_object, x_out_object, y_out_object, z_out_object;	
	int voxel_x, voxel_y, voxel_z, voxel_x_int, voxel_y_int, voxel_z_int;
	bool entered_object = false, exited_object = false;

	cout << "vector histories = " << (unsigned int)x_entry_vector.size() << endl;
	cout << "post_cut_histories = " << post_cut_histories << endl;
	/*************************************************************************************************************************************************************************/
	/******************************************************************** Perform MLP endpoint calculations ******************************************************************/
	/*************************************************************************************************************************************************************************/
	for( unsigned int i = 0; i < post_cut_histories; i++ )
	{		
		/********************************************************************************************************************************************************************/
		/***************************************** Determine if proton entered and exited object and if so, where these occurred ********************************************/
		/********************************************************************************************************************************************************************/
		entered_object = find_MLP_endpoints( hull_h, x_entry_vector[i], y_entry_vector[i], z_entry_vector[i], xy_entry_angle_vector[i], xz_entry_angle_vector[i], x_in_object, y_in_object, z_in_object, voxel_x, voxel_y, voxel_z, true);	
		exited_object = find_MLP_endpoints( hull_h, x_exit_vector[i], y_exit_vector[i], z_exit_vector[i], xy_exit_angle_vector[i], xz_exit_angle_vector[i], x_out_object, y_out_object, z_out_object, voxel_x_int, voxel_y_int, voxel_z_int, false);
		/********************************************************************************************************************************************************************/
		/***************************************************** Shift data down if proton entered and exited object **********************************************************/
		/********************************************************************************************************************************************************************/		
		if( entered_object && exited_object )
		{			
			voxel_x_vector.push_back(voxel_x);
			voxel_y_vector.push_back(voxel_y);
			voxel_z_vector.push_back(voxel_z);
			x_entry_vector[reconstruction_histories] = x_in_object;
			y_entry_vector[reconstruction_histories] = y_in_object;
			z_entry_vector[reconstruction_histories] = z_in_object;
			x_exit_vector[reconstruction_histories] = x_out_object;
			y_exit_vector[reconstruction_histories] = y_out_object;
			z_exit_vector[reconstruction_histories] = z_out_object;
			xy_entry_angle_vector[reconstruction_histories] = xy_entry_angle_vector[i];
			xz_entry_angle_vector[reconstruction_histories] = xz_entry_angle_vector[i];
			xy_exit_angle_vector[reconstruction_histories] = xy_exit_angle_vector[i];
			xz_exit_angle_vector[reconstruction_histories] = xz_exit_angle_vector[i];
			WEPL_vector[reconstruction_histories] = WEPL_vector[i];
			bin_number_vector[reconstruction_histories] = bin_number_vector[i];
			gantry_angle_vector[reconstruction_histories] = gantry_angle_vector[i];
			reconstruction_histories++;
		}
	}
	resize_vectors( reconstruction_histories );
	shrink_vectors( reconstruction_histories );

	puts("Acquistion of hull entry/exit coordinates complete.");
	printf("%d of %d protons passed through hull\n", reconstruction_histories, post_cut_histories);
}
unsigned int calculate_MLP
( 
	unsigned int*& path, float*& chord_lengths, 
	double x_in_object, double y_in_object, double z_in_object, double x_out_object, double y_out_object, double z_out_object, 
	double xy_entry_angle, double xz_entry_angle, double xy_exit_angle, double xz_exit_angle,
	int voxel_x, int voxel_y, int voxel_z
)
{
	//bool debug_output = false, MLP_image_output = false;
	//bool constant_chord_lengths = true;
	// MLP calculations variables
	int num_intersections = 0;
	double u_0 = 0.0, u_1 = parameters.MLP_U_STEP,  u_2 = 0.0;
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
	//double theta_1, phi_1;
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	//double effective_chord_length = mean_chord_length( u_in_object, t_in_object, v_in_object, u_out_object, t_out_object, v_out_object );
	//double effective_chord_length = parameters.VOXEL_WIDTH;

	int voxel = voxel_x + voxel_y * parameters.COLUMNS + voxel_z * parameters.COLUMNS * parameters.ROWS;
	path[num_intersections] = voxel;
	//if(!constant_chord_lengths)
		//chord_lengths[num_intersections] = parameters.VOXEL_WIDTH;
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
		
	u_0 = 0.0;
	u_1 = parameters.MLP_U_STEP;
	u_2 = abs(u_out_object - u_in_object);		
	//fgets(user_response, sizeof(user_response), stdin);

	//output_file.open(filename);						
				      
	//precalculated u_2 dependent terms (since u_2 does not change inside while loop)
	//u_2 terms
	sigma_2_pre_1 =  pow(u_2, 3.0) * ( A_0_OVER_3 + u_2 * ( A_1_OVER_12 + u_2 * ( A_2_OVER_30 + u_2 * (A_3_OVER_60 + u_2 * ( A_4_OVER_105 + u_2 * A_5_OVER_168 )))));;	//u_2^3 : 1/3, 1/12, 1/30, 1/60, 1/105, 1/168
	sigma_2_pre_2 =  pow(u_2, 2.0) * ( A_0_OVER_2 + u_2 * (A_1_OVER_6 + u_2 * (A_2_OVER_12 + u_2 * ( A_3_OVER_20 + u_2 * (A_4_OVER_30 + u_2 * A_5_OVER_42)))));	//u_2^2 : 1/2, 1/6, 1/12, 1/20, 1/30, 1/42
	sigma_2_pre_3 =  u_2 * ( A_0 +  u_2 * (A_1_OVER_2 +  u_2 * ( A_2_OVER_3 +  u_2 * ( A_3_OVER_4 +  u_2 * ( A_4_OVER_5 + u_2 * A_5_OVER_6 )))));			//u_2 : 1/1, 1/2, 1/3, 1/4, 1/5, 1/6

	while( u_1 < u_2 - parameters.MLP_U_STEP)
	//while( u_1 < u_2 - 0.001)
	{
		R_0[1] = u_1;
		R_1[1] = u_2 - u_1;
		R_1T[2] = u_2 - u_1;

		sigma_1_coefficient = pow( E_0 * ( 1 + 0.038 * log( (u_1 - u_0)/X0) ), 2.0 ) / X0;
		sigma_t1 =  sigma_1_coefficient * ( pow(u_1, 3.0) * ( A_0_OVER_3 + u_1 * (A_1_OVER_12 + u_1 * (A_2_OVER_30 + u_1 * (A_3_OVER_60 + u_1 * (A_4_OVER_105 + u_1 * A_5_OVER_168 ) )))) );	//u_1^3 : 1/3, 1/12, 1/30, 1/60, 1/105, 1/168
		sigma_t1_theta1 =  sigma_1_coefficient * ( pow(u_1, 2.0) * ( A_0_OVER_2 + u_1 * (A_1_OVER_6 + u_1 * (A_2_OVER_12 + u_1 * (A_3_OVER_20 + u_1 * (A_4_OVER_30 + u_1 * A_5_OVER_42))))) );	//u_1^2 : 1/2, 1/6, 1/12, 1/20, 1/30, 1/42															
		sigma_theta1 = sigma_1_coefficient * ( u_1 * ( A_0 + u_1 * (A_1_OVER_2 + u_1 * (A_2_OVER_3 + u_1 * (A_3_OVER_4 + u_1 * (A_4_OVER_5 + u_1 * A_5_OVER_6))))) );			//u_1 : 1/1, 1/2, 1/3, 1/4, 1/5, 1/6																	
		determinant_Sigma_1 = sigma_t1 * sigma_theta1 - pow( sigma_t1_theta1, 2 );//ad-bc
			
		Sigma_1I[0] = sigma_theta1 / determinant_Sigma_1;
		Sigma_1I[1] = -sigma_t1_theta1 / determinant_Sigma_1;
		Sigma_1I[2] = -sigma_t1_theta1 / determinant_Sigma_1;
		Sigma_1I[3] = sigma_t1 / determinant_Sigma_1;

		//sigma 2 terms
		sigma_2_coefficient = pow( E_0 * ( 1 + 0.038 * log( (u_2 - u_1)/X0) ), 2.0 ) / X0;
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

		voxel_x = calculate_voxel( parameters.X_ZERO_COORDINATE, x_1, parameters.VOXEL_WIDTH );
		voxel_y = calculate_voxel( parameters.Y_ZERO_COORDINATE, y_1, parameters.VOXEL_HEIGHT );
		voxel_z = calculate_voxel( parameters.Z_ZERO_COORDINATE, z_1, parameters.VOXEL_THICKNESS);
				
		voxel = voxel_x + voxel_y * parameters.COLUMNS + voxel_z * parameters.COLUMNS * parameters.ROWS;
		//cout << "voxel_x = " << voxel_x << "voxel_y = " << voxel_y << "voxel_z = " << voxel_z << "voxel = " << voxel <<endl;
		//fgets(user_response, sizeof(user_response), stdin);

		if( voxel != path[num_intersections - 1] )
		{
			path[num_intersections] = voxel;
			//MLP_test_image_h[voxel] = 0;
			//if(!constant_chord_lengths)
				//chord_lengths[num_intersections] = effective_chord_length;						
			num_intersections++;
		}
		u_1 += parameters.MLP_U_STEP;
	}
	return num_intersections;
}
unsigned int calculate_MLP2
( 
	unsigned int*& path, float*& chord_lengths, 
	double x_in_object, double y_in_object, double z_in_object, double x_out_object, double y_out_object, double z_out_object, 
	double xy_entry_angle, double xz_entry_angle, double xy_exit_angle, double xz_exit_angle,
	int voxel_x, int voxel_y, int voxel_z
)
{
	
	int num_intersections = 0;
	double sigma_2_pre_1, sigma_2_pre_2, sigma_2_pre_3;
	double sigma_1_coefficient, sigma_t1, sigma_t1_theta1, sigma_theta1, determinant_Sigma_1, Sigma_1I[3];
	double common_sigma_2_term_1, common_sigma_2_term_2, common_sigma_2_term_3;
	double sigma_2_coefficient, sigma_t2, sigma_t2_theta2, sigma_theta2, determinant_Sigma_2, Sigma_2I[3]; 
	double first_term_common_24_1, first_term_common_24_2, first_term[4], determinant_first_term;
	double second_term_common_3, second_term_common_4, second_term[2];
	double t_1, v_1, x_1, y_1, z_1;
	//double first_term_inversion_temp;
	int voxel = voxel_x + voxel_y * parameters.COLUMNS + voxel_z * parameters.COLUMNS * parameters.ROWS;
	path[num_intersections] = voxel;
	num_intersections++;

	//double cos_term;	// lookup
	//double sin_term;	// lookup

	double u_in_object = ( cos( xy_entry_angle ) * x_in_object ) + ( sin( xy_entry_angle ) * y_in_object );
	double u_out_object = ( cos( xy_entry_angle ) * x_out_object ) + ( sin( xy_entry_angle ) * y_out_object );
	double t_in_object = ( cos( xy_entry_angle ) * y_in_object ) - ( sin( xy_entry_angle ) * x_in_object );
	double t_out_object = ( cos( xy_entry_angle ) * y_out_object ) - ( sin( xy_entry_angle ) * x_out_object );
	//double v_in_object = z_in_object;
	//double v_out_object = z_out_object;

	/*double u_in_object = cos_term * x_in_object + sin_term * y_in_object;
	double u_out_object = cos_term * x_out_object + sin_term * y_out_object;*/

	if( u_in_object > u_out_object )
	{
		//if( debug_output )
			//cout << "Switched directions" << endl;
		xy_entry_angle += PI;
		xy_exit_angle += PI;
		//cos_term;	// lookup
		//sin_term;	// lookup
		u_in_object = ( cos( xy_entry_angle ) * x_in_object ) + ( sin( xy_entry_angle ) * y_in_object );
		u_out_object = ( cos( xy_entry_angle ) * x_out_object ) + ( sin( xy_entry_angle ) * y_out_object );
		t_in_object = ( cos( xy_entry_angle ) * y_in_object ) - ( sin( xy_entry_angle ) * x_in_object );
		t_out_object = ( cos( xy_entry_angle ) * y_out_object ) - ( sin( xy_entry_angle ) * x_out_object );
		//double u_in_object = cos_term * x_in_object + sin_term * y_in_object;
		//double u_out_object = cos_term * x_out_object + sin_term * y_out_object;
	}

	//double t_in_object = cos_term * y_in_object  - sin_term * x_in_object;
	//double t_out_object = cos_term * y_out_object - sin_term * x_out_object;
	/*T_0[0] = t_in_object;
	T_2[0] = t_out_object;
	T_2[1] = xy_exit_angle - xy_entry_angle;
	V_0[0] = v_in_object;
	V_2[0] = v_out_object;
	V_2[1] = xz_exit_angle - xz_entry_angle;*/
	
	//double T_0[2] = {t_in_object, 0.0};
	double T_2[2] = {t_out_object, xy_exit_angle - xy_entry_angle};
	//double V_0[2] = {z_in_object, 0.0};
	double V_2[2] = {z_out_object, xz_exit_angle - xz_entry_angle};
	//double R_0[4] = { 1.0, 0.0, 0.0 , 1.0}; //a,b,c,d
	//double R_1[4] = { 1.0, 0.0, 0.0 , 1.0}; //a,b,c,d
	double R_1T[4] = { 1.0, 0.0, 0.0 , 1.0};  //a,c,b,d
	double u_1 = parameters.MLP_U_STEP,  u_2 = abs(u_out_object - u_in_object);
	//u_0 = 0.0;
	//u_1 = parameters.MLP_U_STEP;
	//u_2 = abs(u_out_object - u_in_object);		
	//fgets(user_response, sizeof(user_response), stdin);

	//output_file.open(filename);						
				      
	//precalculated u_2 dependent terms (since u_2 does not change inside while loop)
	//u_2 terms
	sigma_2_pre_1 =  pow(u_2, 3.0) * ( A_0_OVER_3 + u_2 * ( A_1_OVER_12 + u_2 * ( A_2_OVER_30 + u_2 * (A_3_OVER_60 + u_2 * ( A_4_OVER_105 + u_2 * A_5_OVER_168 )))));;	//u_2^3 : 1/3, 1/12, 1/30, 1/60, 1/105, 1/168
	sigma_2_pre_2 =  pow(u_2, 2.0) * ( A_0_OVER_2 + u_2 * (A_1_OVER_6 + u_2 * (A_2_OVER_12 + u_2 * ( A_3_OVER_20 + u_2 * (A_4_OVER_30 + u_2 * A_5_OVER_42)))));	//u_2^2 : 1/2, 1/6, 1/12, 1/20, 1/30, 1/42
	sigma_2_pre_3 =  u_2 * ( A_0 +  u_2 * (A_1_OVER_2 +  u_2 * ( A_2_OVER_3 +  u_2 * ( A_3_OVER_4 +  u_2 * ( A_4_OVER_5 + u_2 * A_5_OVER_6 )))));			//u_2 : 1/1, 1/2, 1/3, 1/4, 1/5, 1/6

	while( u_1 < u_2 - parameters.MLP_U_STEP)
	//while( u_1 < u_2 - 0.001)
	{
		//R_0[1] = u_1;
		//R_1[1] = u_2 - u_1;
		R_1T[2] = u_2 - u_1;

		// 26 + 15 + 13 + 12 + 12 + 3 = 81
		sigma_1_coefficient = pow( E_0 * ( 1 + 0.038 * log( u_1 / X0) ), 2.0 ) / X0;
		sigma_t1 = pow(u_1, 3.0) * ( A_0_OVER_3 + u_1 * (A_1_OVER_12 + u_1 * (A_2_OVER_30 + u_1 * (A_3_OVER_60 + u_1 * (A_4_OVER_105 + u_1 * A_5_OVER_168 ) ))) );	//u_1^3 : 1/3, 1/12, 1/30, 1/60, 1/105, 1/168
		sigma_t1_theta1 =  pow(u_1, 2.0) * ( A_0_OVER_2 + u_1 * (A_1_OVER_6 + u_1 * (A_2_OVER_12 + u_1 * (A_3_OVER_20 + u_1 * (A_4_OVER_30 + u_1 * A_5_OVER_42)))) );	//u_1^2 : 1/2, 1/6, 1/12, 1/20, 1/30, 1/42															
		sigma_theta1 = u_1 * ( A_0 + u_1 * (A_1_OVER_2 + u_1 * (A_2_OVER_3 + u_1 * (A_3_OVER_4 + u_1 * (A_4_OVER_5 + u_1 * A_5_OVER_6)))) );			//u_1 : 1/1, 1/2, 1/3, 1/4, 1/5, 1/6																	
		
		// 8
		determinant_Sigma_1 = sigma_1_coefficient * ( sigma_t1 * sigma_theta1 - pow( sigma_t1_theta1, 2 ) );//ad-bc
		Sigma_1I[0] = sigma_theta1 / determinant_Sigma_1;
		Sigma_1I[1] = -sigma_t1_theta1 / determinant_Sigma_1;
		//Sigma_1I[2] = -sigma_t1_theta1 / determinant_Sigma_1;
		Sigma_1I[2] = sigma_t1 / determinant_Sigma_1;

		// 26 + 11 + 12 + 12 + 8 + 4 + 2 + 3  = 78
		//sigma 2 terms
		sigma_2_coefficient = pow( E_0 * ( 1 + 0.038 * log( (u_2 - u_1)/X0) ), 2.0 ) / X0;
		common_sigma_2_term_1 = u_1 * ( A_0 + u_1 * (A_1_OVER_2 + u_1 * (A_2_OVER_3 + u_1 * (A_3_OVER_4 + u_1 * (A_4_OVER_5 + u_1 * A_5_OVER_6 )))));
		common_sigma_2_term_2 = pow(u_1, 2.0) * ( A_0_OVER_2 + u_1 * (A_1_OVER_3 + u_1 * (A_2_OVER_4 + u_1 * (A_3_OVER_5 + u_1 * (A_4_OVER_6 + u_1 * A_5_OVER_7 )))));
		common_sigma_2_term_3 = pow(u_1, 3.0) * ( A_0_OVER_3 + u_1 * (A_1_OVER_4 + u_1 * (A_2_OVER_5 + u_1 * (A_3_OVER_6 + u_1 * (A_4_OVER_7 + u_1 * A_5_OVER_8 )))));
		sigma_t2 =  sigma_2_pre_1 - pow(u_2, 2.0) * common_sigma_2_term_1 + 2 * u_2 * common_sigma_2_term_2 - common_sigma_2_term_3;
		sigma_t2_theta2 =  sigma_2_pre_2 - u_2 * common_sigma_2_term_1 + common_sigma_2_term_2 ;
		sigma_theta2 =  sigma_2_pre_3 - common_sigma_2_term_1;				
			
		// 8
		determinant_Sigma_2 = sigma_2_coefficient * ( sigma_t2 * sigma_theta2 - pow( sigma_t2_theta2, 2 ) );//ad-bc
		Sigma_2I[0] = sigma_theta2 / determinant_Sigma_2;
		Sigma_2I[1] = -sigma_t2_theta2 / determinant_Sigma_2;
		//Sigma_2I[2] = -sigma_t2_theta2 / determinant_Sigma_2;
		Sigma_2I[2] = sigma_t2 / determinant_Sigma_2;

		/**********************************************************************************************************************************************************/
		// 12
		// first_term_common_ij_k: i,j = rows common to, k = 1st/2nd of last 2 terms of 3 term summation in first_term calculation below
		/*first_term_common_13_1 = Sigma_2I[0] * R_1[0] + Sigma_2I[1] * R_1[2];	// Sigma_2I[0] * 1 + Sigma_2I[1] * 0 = Sigma_2I[0]
		first_term_common_13_2 = Sigma_2I[2] * R_1[0] + Sigma_2I[3] * R_1[2];	// Sigma_2I[2] * 1 + Sigma_2I[3] * 0 = Sigma_2I[2]
		first_term_common_24_1 = Sigma_2I[0] * R_1[1] + Sigma_2I[1] * R_1[3];	// Sigma_2I[0] * (u_2-u_1) + Sigma_2I[1] * 1 = Sigma_2I[0] * (u_2-u_1) + Sigma_2I[1]
		first_term_common_24_2 = Sigma_2I[2] * R_1[1] + Sigma_2I[3] * R_1[3];*/	// Sigma_2I[2] * (u_2-u_1) + Sigma_2I[3] * 1 = Sigma_2I[2] * (u_2-u_1) + Sigma_2I[3]
		//first_term_common_13_1 = Sigma_2I[0];
		//first_term_common_13_2 = Sigma_2I[2];

		// 2 + 2 = 4			R_1[1] = R_1T[2] = u_2 - u_1
		first_term_common_24_1 = Sigma_2I[0] * R_1T[2] + Sigma_2I[1];
		first_term_common_24_2 = Sigma_2I[1] * R_1T[2] + Sigma_2I[2];

		// 16
		/*first_term[0] = Sigma_1I[0] + R_1T[0] * first_term_common_13_1 + R_1T[1] * first_term_common_13_2;	//Sigma_1I[0] + 1 * Sigma_2I[0] + 0 * Sigma_2I[2] = Sigma_1I[0] + Sigma_2I[0]
		first_term[1] = Sigma_1I[1] + R_1T[0] * first_term_common_24_1 + R_1T[1] * first_term_common_24_2;		//Sigma_1I[1] + 1 * first_term_common_24_1 + 0 * first_term_common_24_2 = Sigma_1I[1] + first_term_common_24_1
		first_term[2] = Sigma_1I[2] + R_1T[2] * first_term_common_13_1 + R_1T[3] * first_term_common_13_2;		// Sigma_1I[2] + (u_2-u_1) * Sigma_2I[0] + 1 * Sigma_2I[2] = Sigma_1I[2] + (u_2-u_1) * Sigma_2I[0] + Sigma_2I[2]
		first_term[3] = Sigma_1I[3] + R_1T[2] * first_term_common_24_1 + R_1T[3] * first_term_common_24_2;*/	// Sigma_1I[3] + (u_2-u_1) * first_term_common_24_1 + 1 * first_term_common_24_2 = Sigma_1I[3] + (u_2-u_1) * first_term_common_24_1 + first_term_common_24_2
		
		// 1 + 1 + 3 + 3 = 8		R_1T[2] = (u_2-u_1)
		first_term[0] = Sigma_1I[0] + Sigma_2I[0];
		first_term[1] = Sigma_1I[1] + first_term_common_24_1;
		first_term[2] = Sigma_1I[1] + R_1T[2] * Sigma_2I[0] + Sigma_2I[1];
		first_term[3] = Sigma_1I[2] + R_1T[2] * first_term_common_24_1 + first_term_common_24_2;

		// 3 + 5 = 8
		determinant_first_term = first_term[0] * first_term[3] - first_term[1] * first_term[2];
		//first_term_inversion_temp = first_term[0];
		//first_term[0] = first_term[3] / determinant_first_term;
		//first_term[1] = -first_term[1] / determinant_first_term;
		//first_term[2] = -first_term[2] / determinant_first_term;
		//first_term[3] = first_term_inversion_temp / determinant_first_term;

		// 3
		//determinant_first_term = first_term[0] * first_term[3] - first_term[1] * first_term[2];

		// 12
		// second_term_common_i: i = # of term of 4 term summation it is common to in second_term calculation below
		/*second_term_common_1 = R_0[0] * T_0[0] + R_0[1] * T_0[1];				// 1 * t_in_object + u_1 * 0 = t_in_object
		second_term_common_2 = R_0[2] * T_0[0] + R_0[3] * T_0[1];				// 0 * t_in_object + 1 * 0 = 0
		second_term_common_3 = Sigma_2I[0] * T_2[0] + Sigma_2I[1] * T_2[1];		// Sigma_2I[0] * t_out_object + Sigma_2I[1] * (xy_exit_angle - xy_entry_angle)
		second_term_common_4 = Sigma_2I[2] * T_2[0] + Sigma_2I[3] * T_2[1];*/	// Sigma_2I[2] * t_out_object + Sigma_2I[3] * (xy_exit_angle - xy_entry_angle)
		
		// 3 + 3 = 6		T_2[1] = (xy_exit_angle - xy_entry_angle)
		second_term_common_3 = Sigma_2I[0] * t_out_object + Sigma_2I[1] * T_2[1];	
		second_term_common_4 = Sigma_2I[1] * t_out_object + Sigma_2I[2] * T_2[1];

		// 14
		/*second_term[0] = Sigma_1I[0] * second_term_common_1				// Sigma_1I[0] * second_term_common_1 = Sigma_1I[0] * t_in_object
						+ Sigma_1I[1] * second_term_common_2				// + Sigma_1I[1] * second_term_common_2 = 0
						+ R_1T[0] * second_term_common_3					// + 1 * second_term_common_3 = second_term_common_3
						+ R_1T[1] * second_term_common_4;*/					// + 0 * second_term_common_4 = 0
		
		// 2
		second_term[0] = Sigma_1I[0] * t_in_object + second_term_common_3;
		/*second_term[1] = Sigma_1I[2] * second_term_common_1				// Sigma_1I[2] * second_term_common_1 = Sigma_1I[2] * t_in_object
						+ Sigma_1I[3] * second_term_common_2				// + Sigma_1I[3] * second_term_common_2 = 0
						+ R_1T[2] * second_term_common_3					// + R_1T[2] * second_term_common_3 = (u_2-u_1) * second_term_common_3
						+ R_1T[3] * second_term_common_4;*/					// + R_1T[3] * second_term_common_4; = second_term_common_4
		
		// 4					R_1T[2] = (u_2 - u_1)
		second_term[1] = Sigma_1I[1] * t_in_object +  R_1T[2] * second_term_common_3 + second_term_common_4;
		
		// 3
		// t_1 = first_term[0] * second_term[0] + first_term[1] * second_term[1];
		
		// 4		first_term[0] -> first_term[3] & first_term[1] -> -first_term[1] 
		t_1 = ( first_term[3] * second_term[0] - first_term[1] * second_term[1] ) / determinant_first_term ;
		//cout << "t_1 = " << t_1 << endl;
		//double theta_1 = first_term[2] * second_term[0] + first_term[3] * second_term[1];

		// t: 65 -> 31
		/**********************************************************************************************************************************************************/
		// 12
		// Do v MLP Now
		/*second_term_common_1 = R_0[0] * V_0[0] + R_0[1] * V_0[1];				// v_in_object + u_1 * 0 = v_in_object
		second_term_common_2 = R_0[2] * V_0[0] + R_0[3] * V_0[1];				// 0
		second_term_common_3 = Sigma_2I[0] * V_2[0] + Sigma_2I[1] * V_2[1];		// Sigma_2I[0] * v_out_object + Sigma_2I[1] * (xz_exit_angle - xz_entry_angle)
		second_term_common_4 = Sigma_2I[2] * V_2[0] + Sigma_2I[3] * V_2[1];*/	// Sigma_2I[2] * v_out_object + Sigma_2I[3] * (xz_exit_angle - xz_entry_angle)
		
		// 			V_2[1] = (xz_exit_angle - xz_entry_angle)
		//second_term_common_1 = v_in_object;						
		//second_term_common_2 = 0;	

		// 3 + 3 = 6
		second_term_common_3 =  Sigma_2I[0] * z_out_object + Sigma_2I[1] * V_2[1];
		second_term_common_4 = Sigma_2I[1] * z_out_object + Sigma_2I[2] * V_2[1];

		// 14
		/*second_term[0]	= Sigma_1I[0] * second_term_common_1			// Sigma_1I[0] * v_in_object	
						+ Sigma_1I[1] * second_term_common_2				// + Sigma_1I[1] * 0 = 0
						+ R_1T[0] * second_term_common_3					// + 1 * second_term_common_3 = second_term_common_3
						+ R_1T[1] * second_term_common_4;*/					// + 0 * second_term_common_4 = 0
		//  2 
		second_term[0] = Sigma_1I[0] * z_in_object + second_term_common_3;
		
		/*second_term[1]	= Sigma_1I[2] * second_term_common_1			// Sigma_1I[2] * v_in_object
						+ Sigma_1I[3] * second_term_common_2				// + Sigma_1I[3] * 0 = 0
						+ R_1T[2] * second_term_common_3					// + R_1T[2] * second_term_common_3
						+ R_1T[3] * second_term_common_4;*/					// + 1 * second_term_common_4 = second_term_common_4
		
		// 4
		second_term[1] = Sigma_1I[1] * z_in_object + R_1T[2] * second_term_common_3 + second_term_common_4;
		
		
		// 3
		//v_1 = first_term[0] * second_term[0] + first_term[1] * second_term[1];

		// 4
		v_1 = ( first_term[3] * second_term[0] - first_term[1] * second_term[1] ) / determinant_first_term ;

		//double phi_1 = first_term[2] * second_term[0] + first_term[3] * second_term[1];

		// v: 29 -> 16

		// t + v: 94 -> 47
		/**********************************************************************************************************************************************************/
		// 47 -> 13
		// Rotate Coordinate From utv to xyz Coordinate System and Determine Which Voxel this Point on the MLP Path is in
		// 1 + 1 + 1 + 3 + 3 = 9 -> 7 w/ precalculated sin/cos
		// cos_term = lookup
		// sin_term = lookup
		// u_shifted = u_in_object + u_1
		/*x_1 = cos_term * u_shifted - sin_term * t_1 ;
		y_1 = sin_term * u_shifted + cos_term * t_1;*/

		// 24 + 24 + 1 = 49 
		x_1 = ( cos( xy_entry_angle ) * (u_in_object + u_1) ) - ( sin( xy_entry_angle ) * t_1 );
		y_1 = ( sin( xy_entry_angle ) * (u_in_object + u_1) ) + ( cos( xy_entry_angle ) * t_1 );
		z_1 = v_1;

		// 12 + 5 = 17
		/*voxel_x = calculate_voxel( parameters.X_ZERO_COORDINATE, x_1, parameters.VOXEL_WIDTH );
		voxel_y = calculate_voxel( parameters.Y_ZERO_COORDINATE, y_1, parameters.VOXEL_HEIGHT );
		voxel_z = calculate_voxel( parameters.Z_ZERO_COORDINATE, v_1, parameters.VOXEL_THICKNESS);*/
		// 12
		voxel_x = calculate_voxel( parameters.X_ZERO_COORDINATE, x_1, parameters.VOXEL_WIDTH );
		voxel_y = calculate_voxel( parameters.Y_ZERO_COORDINATE, y_1, parameters.VOXEL_HEIGHT );
		voxel_z = calculate_voxel( parameters.Z_ZERO_COORDINATE, z_1, parameters.VOXEL_THICKNESS);
		voxel = voxel_x + voxel_y * parameters.COLUMNS + voxel_z * parameters.COLUMNS * parameters.ROWS;
		//cout << "voxel_x = " << voxel_x << "voxel_y = " << voxel_y << "voxel_z = " << voxel_z << "voxel = " << voxel <<endl;
		//fgets(user_response, sizeof(user_response), stdin);

		// 5
		if( voxel != path[num_intersections - 1] )
		{
			path[num_intersections] = voxel;
			//MLP_test_image_h[voxel] = 0;
			//if(!constant_chord_lengths)
				//chord_lengths[num_intersections] = effective_chord_length;						
			num_intersections++;
		}
		u_1 += parameters.MLP_U_STEP;

		// 71 -> 31

		// 175 + 94 + 71 = 339 ->  47 + 31
	}
	return num_intersections;
}
unsigned int calculate_MLP3
( 
	unsigned int*& path, float*& chord_lengths, 
	double x_in_object, double y_in_object, double z_in_object, double x_out_object, double y_out_object, double z_out_object, 
	double xy_entry_angle, double xz_entry_angle, double xy_exit_angle, double xz_exit_angle,
	int voxel_x, int voxel_y, int voxel_z
)
{
	double sigma_2_pre_1, sigma_2_pre_2, sigma_2_pre_3;
	double sigma_1_coefficient, sigma_t1, sigma_t1_theta1, sigma_theta1, determinant_Sigma_1, Sigma_1I[3];
	double common_sigma_2_term_1, common_sigma_2_term_2, common_sigma_2_term_3;
	double sigma_2_coefficient, sigma_t2, sigma_t2_theta2, sigma_theta2, determinant_Sigma_2, Sigma_2I[3]; 
	double first_term_common_24_1, first_term_common_24_2, first_term[4], determinant_first_term;
	double second_term_common_3, second_term_common_4, second_term[2];
	double t_1, v_1, x_1, y_1;
	int voxel = voxel_x + voxel_y * parameters.COLUMNS + voxel_z * parameters.COLUMNS * parameters.ROWS;
	
	int num_intersections = 0;
	path[num_intersections] = voxel;
	//num_intersections++;

	int trig_table_index =  (xy_entry_angle - TRIG_TABLE_MIN ) / TRIG_TABLE_STEP + 0.5;
	double sin_term = sin_table_h[trig_table_index];
	double cos_term = cos_table_h[trig_table_index];
	
	double u_in_object = cos_term * x_in_object + sin_term * y_in_object;
	double u_out_object = cos_term * x_out_object + sin_term * y_out_object;

	if( u_in_object > u_out_object )
	{
		xy_entry_angle += PI;
		xy_exit_angle += PI;
		trig_table_index =  (xy_entry_angle - TRIG_TABLE_MIN ) / TRIG_TABLE_STEP + 0.5;
		sin_term = sin_table_h[trig_table_index];
		cos_term = cos_table_h[trig_table_index];
		u_in_object = cos_term * x_in_object + sin_term * y_in_object;
		u_out_object = cos_term * x_out_object + sin_term * y_out_object;
	}

	double t_in_object = cos_term * y_in_object  - sin_term * x_in_object;
	double t_out_object = cos_term * y_out_object - sin_term * x_out_object;
	
	double T_2[2] = {t_out_object, xy_exit_angle - xy_entry_angle};
	double V_2[2] = {z_out_object, xz_exit_angle - xz_entry_angle};
	double R_1T[4] = { 1.0, 0.0, 0.0 , 1.0};  //a,c,b,d
	double u_1 = parameters.MLP_U_STEP,  u_2 = abs(u_out_object - u_in_object);
	double u_shifted = u_in_object;							      
	//precalculated u_2 dependent terms (since u_2 does not change inside while loop)
	//u_2 terms
	sigma_2_pre_1 =  pow(u_2, 3.0) * ( A_0_OVER_3 + u_2 * ( A_1_OVER_12 + u_2 * ( A_2_OVER_30 + u_2 * (A_3_OVER_60 + u_2 * ( A_4_OVER_105 + u_2 * A_5_OVER_168 )))));;	//u_2^3 : 1/3, 1/12, 1/30, 1/60, 1/105, 1/168
	sigma_2_pre_2 =  pow(u_2, 2.0) * ( A_0_OVER_2 + u_2 * (A_1_OVER_6 + u_2 * (A_2_OVER_12 + u_2 * ( A_3_OVER_20 + u_2 * (A_4_OVER_30 + u_2 * A_5_OVER_42)))));	//u_2^2 : 1/2, 1/6, 1/12, 1/20, 1/30, 1/42
	sigma_2_pre_3 =  u_2 * ( A_0 +  u_2 * (A_1_OVER_2 +  u_2 * ( A_2_OVER_3 +  u_2 * ( A_3_OVER_4 +  u_2 * ( A_4_OVER_5 + u_2 * A_5_OVER_6 )))));			//u_2 : 1/1, 1/2, 1/3, 1/4, 1/5, 1/6

	while( u_1 < u_2 - parameters.MLP_U_STEP)
	//while( u_1 < u_2 - 0.001)
	{
		R_1T[2] = u_2 - u_1;

		// 26 + 15 + 13 + 12 + 12 + 3 = 81
		sigma_1_coefficient = pow( E_0 * ( 1 + 0.038 * log( u_1 / X0 ) ), 2.0 ) / X0;
		sigma_t1 = pow(u_1, 3.0) * ( A_0_OVER_3 + u_1 * (A_1_OVER_12 + u_1 * (A_2_OVER_30 + u_1 * (A_3_OVER_60 + u_1 * (A_4_OVER_105 + u_1 * A_5_OVER_168 ) ))) );	//u_1^3 : 1/3, 1/12, 1/30, 1/60, 1/105, 1/168
		sigma_t1_theta1 =  pow(u_1, 2.0) * ( A_0_OVER_2 + u_1 * (A_1_OVER_6 + u_1 * (A_2_OVER_12 + u_1 * (A_3_OVER_20 + u_1 * (A_4_OVER_30 + u_1 * A_5_OVER_42)))) );	//u_1^2 : 1/2, 1/6, 1/12, 1/20, 1/30, 1/42															
		sigma_theta1 = u_1 * ( A_0 + u_1 * (A_1_OVER_2 + u_1 * (A_2_OVER_3 + u_1 * (A_3_OVER_4 + u_1 * (A_4_OVER_5 + u_1 * A_5_OVER_6)))) );			//u_1 : 1/1, 1/2, 1/3, 1/4, 1/5, 1/6																	
		
		// 4
		determinant_Sigma_1 = sigma_1_coefficient * ( sigma_t1 * sigma_theta1 - pow( sigma_t1_theta1, 2 ) );//ad-bc
		Sigma_1I[0] = sigma_theta1 / determinant_Sigma_1;
		Sigma_1I[1] = -sigma_t1_theta1 / determinant_Sigma_1;
		//Sigma_1I[2] = -sigma_t1_theta1 / determinant_Sigma_1;
		Sigma_1I[2] = sigma_t1 / determinant_Sigma_1;

		// 26 + 11 + 12 + 12 + 8 + 4 + 2 + 3  = 78
		//sigma 2 terms
		sigma_2_coefficient = pow( E_0 * ( 1 + 0.038 * log( (u_2 - u_1)/X0) ), 2.0 ) / X0;
		common_sigma_2_term_1 = u_1 * ( A_0 + u_1 * (A_1_OVER_2 + u_1 * (A_2_OVER_3 + u_1 * (A_3_OVER_4 + u_1 * (A_4_OVER_5 + u_1 * A_5_OVER_6 )))));
		common_sigma_2_term_2 = pow(u_1, 2.0) * ( A_0_OVER_2 + u_1 * (A_1_OVER_3 + u_1 * (A_2_OVER_4 + u_1 * (A_3_OVER_5 + u_1 * (A_4_OVER_6 + u_1 * A_5_OVER_7 )))));
		common_sigma_2_term_3 = pow(u_1, 3.0) * ( A_0_OVER_3 + u_1 * (A_1_OVER_4 + u_1 * (A_2_OVER_5 + u_1 * (A_3_OVER_6 + u_1 * (A_4_OVER_7 + u_1 * A_5_OVER_8 )))));
		sigma_t2 =  sigma_2_pre_1 - pow(u_2, 2.0) * common_sigma_2_term_1 + 2 * u_2 * common_sigma_2_term_2 - common_sigma_2_term_3;
		sigma_t2_theta2 =  sigma_2_pre_2 - u_2 * common_sigma_2_term_1 + common_sigma_2_term_2 ;
		sigma_theta2 =  sigma_2_pre_3 - common_sigma_2_term_1;				
			
		// 4
		determinant_Sigma_2 = sigma_2_coefficient * ( sigma_t2 * sigma_theta2 - pow( sigma_t2_theta2, 2 ) );//ad-bc
		Sigma_2I[0] = sigma_theta2 / determinant_Sigma_2;
		Sigma_2I[1] = -sigma_t2_theta2 / determinant_Sigma_2;
		//Sigma_2I[2] = -sigma_t2_theta2 / determinant_Sigma_2;
		Sigma_2I[2] = sigma_t2 / determinant_Sigma_2;

		/**********************************************************************************************************************************************************/
		// 2 + 2 = 4			R_1[1] = R_1T[2] = u_2 - u_1
		first_term_common_24_1 = Sigma_2I[0] * R_1T[2] + Sigma_2I[1];
		first_term_common_24_2 = Sigma_2I[1] * R_1T[2] + Sigma_2I[2];

		// 1 + 1 + 3 + 3 = 8		R_1T[2] = (u_2-u_1)
		first_term[0] = Sigma_1I[0] + Sigma_2I[0];
		first_term[1] = Sigma_1I[1] + first_term_common_24_1;
		first_term[2] = Sigma_1I[1] + R_1T[2] * Sigma_2I[0] + Sigma_2I[1];
		first_term[3] = Sigma_1I[2] + R_1T[2] * first_term_common_24_1 + first_term_common_24_2;

		// 3
		determinant_first_term = first_term[0] * first_term[3] - first_term[1] * first_term[2];

		second_term_common_3 = Sigma_2I[0] * t_out_object + Sigma_2I[1] * T_2[1];	
		second_term_common_4 = Sigma_2I[1] * t_out_object + Sigma_2I[2] * T_2[1];
		
		// 2
		second_term[0] = Sigma_1I[0] * t_in_object + second_term_common_3;
		
		// 4					R_1T[2] = (u_2 - u_1)
		second_term[1] = Sigma_1I[1] * t_in_object +  R_1T[2] * second_term_common_3 + second_term_common_4;
		
		// 4		first_term[0] -> first_term[3] & first_term[1] -> -first_term[1] 
		t_1 = ( first_term[3] * second_term[0] - first_term[1] * second_term[1] ) / determinant_first_term ;
		//double theta_1 = first_term[2] * second_term[0] + first_term[3] * second_term[1];

		// t: 65 -> 31
		/**********************************************************************************************************************************************************/
		// Do v MLP Now
		
		// 3 + 3 = 6
		second_term_common_3 =  Sigma_2I[0] * z_out_object + Sigma_2I[1] * V_2[1];
		second_term_common_4 = Sigma_2I[1] * z_out_object + Sigma_2I[2] * V_2[1];

		//  2 + 4
		second_term[0] = Sigma_1I[0] * z_in_object + second_term_common_3;
		second_term[1] = Sigma_1I[1] * z_in_object + R_1T[2] * second_term_common_3 + second_term_common_4;

		// 4
		v_1 = ( first_term[3] * second_term[0] - first_term[1] * second_term[1] ) / determinant_first_term ;
		//double phi_1 = first_term[2] * second_term[0] + first_term[3] * second_term[1];

		// v: 29 -> 16

		// t + v: 94 -> 47
		/**********************************************************************************************************************************************************/
		// 47 -> 13
		// Rotate Coordinate From utv to xyz Coordinate System and Determine Which Voxel this Point on the MLP Path is in
		// 1 + 1 + 1 + 3 + 3 = 9 -> 7 w/ precalculated sin/cos
		
		u_shifted = u_in_object + u_1;
		x_1 = cos_term * u_shifted - sin_term * t_1 ;
		y_1 = sin_term * u_shifted + cos_term * t_1;

		// 12 + 5 = 17
		voxel_x = calculate_voxel( parameters.X_ZERO_COORDINATE, x_1, parameters.VOXEL_WIDTH );
		voxel_y = calculate_voxel( parameters.Y_ZERO_COORDINATE, y_1, parameters.VOXEL_HEIGHT );
		voxel_z = calculate_voxel( parameters.Z_ZERO_COORDINATE, v_1, parameters.VOXEL_THICKNESS);
		voxel = voxel_x + voxel_y * parameters.COLUMNS + voxel_z * parameters.COLUMNS * parameters.ROWS;

		// 4
		//if( voxel != path[num_intersections - 1] )
		//	path[num_intersections++] = voxel;	
		if( voxel != path[num_intersections] )
			path[++num_intersections] = voxel;	
		u_1 += parameters.MLP_U_STEP;

		// 70 -> 30

		// 167 + 94 + 70 = 331 ->  47 + 30
	}
	++num_intersections;
	return num_intersections;
}
unsigned int calculate_MLP4
( 
	unsigned int*& path, float*& chord_lengths, 
	double x_in_object, double y_in_object, double z_in_object, double x_out_object, double y_out_object, double z_out_object, 
	double xy_entry_angle, double xz_entry_angle, double xy_exit_angle, double xz_exit_angle,
	int voxel_x, int voxel_y, int voxel_z
)
{
	double sigma_2_pre_1, sigma_2_pre_2, sigma_2_pre_3;
	double sigma_1_coefficient, sigma_t1, sigma_t1_theta1, sigma_theta1, determinant_Sigma_1, Sigma_1I[3];
	double common_sigma_2_term_1, common_sigma_2_term_2, common_sigma_2_term_3;
	double sigma_2_coefficient, sigma_t2, sigma_t2_theta2, sigma_theta2, determinant_Sigma_2, Sigma_2I[3]; 
	double first_term_common_24_1, first_term_common_24_2, first_term[4], determinant_first_term;
	double second_term_common_3, second_term_common_4, second_term[2];
	double t_1, v_1, x_1, y_1;
	int voxel = voxel_x + voxel_y * parameters.COLUMNS + voxel_z * parameters.COLUMNS * parameters.ROWS;
	
	int num_intersections = 0;
	path[num_intersections] = voxel;
	//num_intersections++;

	int trig_table_index =  (xy_entry_angle - TRIG_TABLE_MIN ) / TRIG_TABLE_STEP + 0.5;
	double sin_term = sin_table_h[trig_table_index];
	double cos_term = cos_table_h[trig_table_index];
	
	double u_in_object = cos_term * x_in_object + sin_term * y_in_object;
	double u_out_object = cos_term * x_out_object + sin_term * y_out_object;

	if( u_in_object > u_out_object )
	{
		xy_entry_angle += PI;
		xy_exit_angle += PI;
		trig_table_index =  (xy_entry_angle - TRIG_TABLE_MIN ) / TRIG_TABLE_STEP + 0.5;
		sin_term = sin_table_h[trig_table_index];
		cos_term = cos_table_h[trig_table_index];
		u_in_object = cos_term * x_in_object + sin_term * y_in_object;
		u_out_object = cos_term * x_out_object + sin_term * y_out_object;
	}

	double t_in_object = cos_term * y_in_object  - sin_term * x_in_object;
	double t_out_object = cos_term * y_out_object - sin_term * x_out_object;
	
	double T_2[2] = {t_out_object, xy_exit_angle - xy_entry_angle};
	double V_2[2] = {z_out_object, xz_exit_angle - xz_entry_angle};
	double u_1 = parameters.MLP_U_STEP;
	double u_2 = abs(u_out_object - u_in_object);
	double depth_2_go = u_2 - u_1;
	double u_shifted = u_in_object;	
						// [#] Difference between the index of consecutive elements of the poly table for u_1 dependent terms
	
	// Scattering Coefficient tables indices
	unsigned int sigma_table_index_step = static_cast<unsigned int>( parameters.MLP_U_STEP / DEPTH_TABLE_STEP + 0.5 );
	//unsigned int depth_table_last_index = static_cast<unsigned int>( depth_2_go / DEPTH_TABLE_STEP + 0.5 );
	//unsigned int depth_2_go_index = depth_table_last_index - depth_table_index_step;
	unsigned int sigma_1_coefficient_index = sigma_table_index_step;
	unsigned int sigma_2_coefficient_index = static_cast<unsigned int>( depth_2_go / DEPTH_TABLE_STEP + 0.5 );
	
	sigma_2_pre_1 =  pow(u_2, 3.0) * ( A_0_OVER_3 + u_2 * ( A_1_OVER_12 + u_2 * ( A_2_OVER_30 + u_2 * (A_3_OVER_60 + u_2 * ( A_4_OVER_105 + u_2 * A_5_OVER_168 )))));;	//u_2^3 : 1/3, 1/12, 1/30, 1/60, 1/105, 1/168
	sigma_2_pre_2 =  pow(u_2, 2.0) * ( A_0_OVER_2 + u_2 * (A_1_OVER_6 + u_2 * (A_2_OVER_12 + u_2 * ( A_3_OVER_20 + u_2 * (A_4_OVER_30 + u_2 * A_5_OVER_42)))));	//u_2^2 : 1/2, 1/6, 1/12, 1/20, 1/30, 1/42
	sigma_2_pre_3 =  u_2 * ( A_0 +  u_2 * (A_1_OVER_2 +  u_2 * ( A_2_OVER_3 +  u_2 * ( A_3_OVER_4 +  u_2 * ( A_4_OVER_5 + u_2 * A_5_OVER_6 )))));			//u_2 : 1/1, 1/2, 1/3, 1/4, 1/5, 1/6
	
	while( depth_2_go > parameters.MLP_U_STEP )
	{
		//depth_2_go = u_2 - u_1;
		//depth_2_go_index -= depth_table_index_step;
		//depth_index += depth_table_index_step;
		//depth_2_go -= parameters.MLP_U_STEP;
		// 26 + 15 + 13 + 12 + 12 + 3 = 81
		sigma_1_coefficient = scattering_table_h[sigma_1_coefficient_index];
		sigma_t1 = pow(u_1, 3.0) * ( A_0_OVER_3 + u_1 * (A_1_OVER_12 + u_1 * (A_2_OVER_30 + u_1 * (A_3_OVER_60 + u_1 * (A_4_OVER_105 + u_1 * A_5_OVER_168 ) ))) );	//u_1^3 : 1/3, 1/12, 1/30, 1/60, 1/105, 1/168
		sigma_t1_theta1 =  pow(u_1, 2.0) * ( A_0_OVER_2 + u_1 * (A_1_OVER_6 + u_1 * (A_2_OVER_12 + u_1 * (A_3_OVER_20 + u_1 * (A_4_OVER_30 + u_1 * A_5_OVER_42)))) );	//u_1^2 : 1/2, 1/6, 1/12, 1/20, 1/30, 1/42															
		sigma_theta1 = u_1 * ( A_0 + u_1 * (A_1_OVER_2 + u_1 * (A_2_OVER_3 + u_1 * (A_3_OVER_4 + u_1 * (A_4_OVER_5 + u_1 * A_5_OVER_6)))) );			//u_1 : 1/1, 1/2, 1/3, 1/4, 1/5, 1/6																	
		
		// 4
		determinant_Sigma_1 = sigma_1_coefficient * ( sigma_t1 * sigma_theta1 - pow( sigma_t1_theta1, 2 ) );//ad-bc
		Sigma_1I[0] = sigma_theta1 / determinant_Sigma_1;
		Sigma_1I[1] = -sigma_t1_theta1 / determinant_Sigma_1;
		Sigma_1I[2] = sigma_t1 / determinant_Sigma_1;

		// 26 + 11 + 12 + 12 + 8 + 4 + 2 + 3  = 78
		sigma_2_coefficient = scattering_table_h[sigma_2_coefficient_index];
		common_sigma_2_term_1 = u_1 * ( A_0 + u_1 * (A_1_OVER_2 + u_1 * (A_2_OVER_3 + u_1 * (A_3_OVER_4 + u_1 * (A_4_OVER_5 + u_1 * A_5_OVER_6 )))));
		common_sigma_2_term_2 = pow(u_1, 2.0) * ( A_0_OVER_2 + u_1 * (A_1_OVER_3 + u_1 * (A_2_OVER_4 + u_1 * (A_3_OVER_5 + u_1 * (A_4_OVER_6 + u_1 * A_5_OVER_7 )))));
		common_sigma_2_term_3 = pow(u_1, 3.0) * ( A_0_OVER_3 + u_1 * (A_1_OVER_4 + u_1 * (A_2_OVER_5 + u_1 * (A_3_OVER_6 + u_1 * (A_4_OVER_7 + u_1 * A_5_OVER_8 )))));
		sigma_t2 =  sigma_2_pre_1 - pow(u_2, 2.0) * common_sigma_2_term_1 + 2 * u_2 * common_sigma_2_term_2 - common_sigma_2_term_3;
		sigma_t2_theta2 =  sigma_2_pre_2 - u_2 * common_sigma_2_term_1 + common_sigma_2_term_2 ;
		sigma_theta2 =  sigma_2_pre_3 - common_sigma_2_term_1;				
			
		// 4
		determinant_Sigma_2 = sigma_2_coefficient * ( sigma_t2 * sigma_theta2 - pow( sigma_t2_theta2, 2 ) );//ad-bc
		Sigma_2I[0] = sigma_theta2 / determinant_Sigma_2;
		Sigma_2I[1] = -sigma_t2_theta2 / determinant_Sigma_2;
		Sigma_2I[2] = sigma_t2 / determinant_Sigma_2;

		/**********************************************************************************************************************************************************/
		// 2 + 2 = 4			R_1[1] = R_1T[2] = u_2 - u_1
		first_term_common_24_1 = Sigma_2I[0] * depth_2_go + Sigma_2I[1];
		first_term_common_24_2 = Sigma_2I[1] * depth_2_go + Sigma_2I[2];

		// 1 + 1 + 3 + 3 = 8		R_1T[2] = (u_2-u_1)
		first_term[0] = Sigma_1I[0] + Sigma_2I[0];
		first_term[1] = Sigma_1I[1] + first_term_common_24_1;
		first_term[2] = Sigma_1I[1] + depth_2_go * Sigma_2I[0] + Sigma_2I[1];
		first_term[3] = Sigma_1I[2] + depth_2_go * first_term_common_24_1 + first_term_common_24_2;

		// 3
		determinant_first_term = first_term[0] * first_term[3] - first_term[1] * first_term[2];

		// 3 + 3 = 6		T_2[1] = (xy_exit_angle - xy_entry_angle)
		second_term_common_3 = Sigma_2I[0] * t_out_object + Sigma_2I[1] * T_2[1];	
		second_term_common_4 = Sigma_2I[1] * t_out_object + Sigma_2I[2] * T_2[1];

		// 2 + 4			R_1T[2] = (u_2 - u_1)
		second_term[0] = Sigma_1I[0] * t_in_object + second_term_common_3;				
		second_term[1] = Sigma_1I[1] * t_in_object +  depth_2_go * second_term_common_3 + second_term_common_4;
		
		// 4		first_term[0] -> first_term[3] & first_term[1] -> -first_term[1] 
		t_1 = ( first_term[3] * second_term[0] - first_term[1] * second_term[1] ) / determinant_first_term ;
		//double theta_1 = first_term[2] * second_term[0] + first_term[3] * second_term[1];

		// t: 65 -> 31
		/**********************************************************************************************************************************************************/
		// Do v MLP Now
		// 3 + 3 = 6
		second_term_common_3 =  Sigma_2I[0] * z_out_object + Sigma_2I[1] * V_2[1];
		second_term_common_4 = Sigma_2I[1] * z_out_object + Sigma_2I[2] * V_2[1];

		//  2 + 4
		second_term[0] = Sigma_1I[0] * z_in_object + second_term_common_3;
		second_term[1] = Sigma_1I[1] * z_in_object + depth_2_go * second_term_common_3 + second_term_common_4;

		// 4
		v_1 = ( first_term[3] * second_term[0] - first_term[1] * second_term[1] ) / determinant_first_term ;
		//double phi_1 = first_term[2] * second_term[0] + first_term[3] * second_term[1];
		/**********************************************************************************************************************************************************/
		// 47 -> 13
		// Rotate Coordinate From utv to xyz Coordinate System and Determine Which Voxel this Point on the MLP Path is in
		// 1 + 1 + 1 + 3 + 3 = 9 -> 7 w/ precalculated sin/cos
		
		u_shifted += parameters.MLP_U_STEP;
		x_1 = cos_term * u_shifted - sin_term * t_1;
		y_1 = sin_term * u_shifted + cos_term * t_1;

		// 12 + 5 = 17
		voxel_x = calculate_voxel( parameters.X_ZERO_COORDINATE, x_1, parameters.VOXEL_WIDTH );
		voxel_y = calculate_voxel( parameters.Y_ZERO_COORDINATE, y_1, parameters.VOXEL_HEIGHT );
		voxel_z = calculate_voxel( parameters.Z_ZERO_COORDINATE, v_1, parameters.VOXEL_THICKNESS);
		voxel = voxel_x + voxel_y * parameters.COLUMNS + voxel_z * parameters.COLUMNS * parameters.ROWS;

		// 4
		if( voxel != path[num_intersections] )
			path[++num_intersections] = voxel;	

		u_1 += parameters.MLP_U_STEP;
		depth_2_go -= parameters.MLP_U_STEP;
		sigma_1_coefficient_index += sigma_table_index_step;
		sigma_2_coefficient_index -= sigma_table_index_step;
		// 70 -> 30

		// 167 + 94 + 70 = 331 ->  47 + 30
	}
	++num_intersections;
	return num_intersections;
}
unsigned int calculate_MLP5
( 
	unsigned int*& path, float*& chord_lengths, 
	double x_in_object, double y_in_object, double z_in_object, double x_out_object, double y_out_object, double z_out_object, 
	double xy_entry_angle, double xz_entry_angle, double xy_exit_angle, double xz_exit_angle,
	int voxel_x, int voxel_y, int voxel_z
)
{
	double sigma_t1, sigma_t1_theta1, sigma_theta1, determinant_Sigma_1, Sigma_1I[3];
	double sigma_t2, sigma_t2_theta2, sigma_theta2, determinant_Sigma_2, Sigma_2I[3]; 
	double first_term_common_24_1, first_term[4], determinant_first_term;
	double second_term_common_3, second_term[2];
	double t_1, v_1, x_1, y_1;
	
	int voxel = voxel_x + voxel_y * parameters.COLUMNS + voxel_z * parameters.COLUMNS * parameters.ROWS;	
	int num_intersections = 0;
	path[num_intersections] = voxel;
	//num_intersections++;

	unsigned int trig_table_index =  static_cast<unsigned int>((xy_entry_angle - TRIG_TABLE_MIN ) / TRIG_TABLE_STEP + 0.5);
	double sin_term = sin_table_h[trig_table_index];
	double cos_term = cos_table_h[trig_table_index];
	 
	double u_in_object = cos_term * x_in_object + sin_term * y_in_object;
	double u_out_object = cos_term * x_out_object + sin_term * y_out_object;

	if( u_in_object > u_out_object )
	{
		xy_entry_angle += PI;
		xy_exit_angle += PI;
		trig_table_index =  static_cast<unsigned int>((xy_entry_angle - TRIG_TABLE_MIN ) / TRIG_TABLE_STEP + 0.5);
		sin_term = sin_table_h[trig_table_index];
		cos_term = cos_table_h[trig_table_index];
		u_in_object = cos_term * x_in_object + sin_term * y_in_object;
		u_out_object = cos_term * x_out_object + sin_term * y_out_object;
	}

	double t_in_object = cos_term * y_in_object  - sin_term * x_in_object;
	double t_out_object = cos_term * y_out_object - sin_term * x_out_object;
	
	double T_2[2] = {t_out_object, xy_exit_angle - xy_entry_angle};
	double V_2[2] = {z_out_object, xz_exit_angle - xz_entry_angle};
	double u_1 = parameters.MLP_U_STEP;
	double u_2 = abs(u_out_object - u_in_object);
	double depth_2_go = u_2 - u_1;
	double u_shifted = u_in_object;	
	unsigned int step_number = 1;

	// Scattering Coefficient tables indices
	unsigned int sigma_table_index_step = static_cast<unsigned int>( parameters.MLP_U_STEP / DEPTH_TABLE_STEP + 0.5 );
	unsigned int sigma_1_coefficient_index = sigma_table_index_step;
	unsigned int sigma_2_coefficient_index = static_cast<unsigned int>( depth_2_go / DEPTH_TABLE_STEP + 0.5 );
	
	// Scattering polynomial indices
	unsigned int poly_table_index_step = static_cast<unsigned int>( parameters.MLP_U_STEP / POLY_TABLE_STEP + 0.5 );
	unsigned int u_1_poly_index = poly_table_index_step;
	unsigned int u_2_poly_index = static_cast<unsigned int>( u_2 / POLY_TABLE_STEP + 0.5 );

	//precalculated u_2 dependent terms (since u_2 does not change inside while loop)
	double u_2_poly_3_12 = poly_3_12_h[u_2_poly_index];
	double u_2_poly_2_6 = poly_2_6_h[u_2_poly_index];
	double u_2_poly_1_2 = poly_1_2_h[u_2_poly_index];
	double u_1_poly_1_2, u_1_poly_2_3;

	//while( u_1 < u_2 - parameters.MLP_U_STEP)
	while( depth_2_go > parameters.MLP_U_STEP )
	{
		u_1_poly_1_2 = poly_1_2_h[u_1_poly_index];
		u_1_poly_2_3 = poly_2_3_h[u_1_poly_index];

		// Sigma_1: 3
		sigma_t1 = poly_3_12_h[u_1_poly_index];										// poly_3_12(u_1)
		sigma_t1_theta1 =  poly_2_6_h[u_1_poly_index];								// poly_2_6(u_1) 
		sigma_theta1 = u_1_poly_1_2;												// poly_1_2(u_1)

		// Sigma_2: 7 + 3 + 1 = 11									//
		sigma_t2 =  u_2_poly_3_12 - pow(u_2, 2.0) * u_1_poly_1_2 + 2 * u_2 * u_1_poly_2_3 - poly_3_4_h[u_1_poly_index];	// poly_3_12(u_2) - u_2^2 * poly_1_2(u_1) +2u_2*(u_1) - poly_3_4(u_1)
		sigma_t2_theta2 =  u_2_poly_2_6 - u_2 * u_1_poly_1_2 + u_1_poly_2_3;											// poly_2_6(u_2) - u_2*poly_1_2(u_1) + poly_2_3(u_1)
		sigma_theta2 =  u_2_poly_1_2 - u_1_poly_1_2;																	// poly_1_2(u_2) - poly_1_2(u_1)	

		// 4 + 1 + 1 + 1 = 7
		determinant_Sigma_1 = scattering_table_h[sigma_1_coefficient_index] * ( sigma_t1 * sigma_theta1 - pow( sigma_t1_theta1, 2 ) );//ad-bc
		Sigma_1I[0] = sigma_theta1 / determinant_Sigma_1;
		Sigma_1I[1] = sigma_t1_theta1 / determinant_Sigma_1;	// negative sign is propagated to subsequent calculations instead of here 
		Sigma_1I[2] = sigma_t1 / determinant_Sigma_1;			
			
		// 4 + 1 + 1 + 1 = 7
		determinant_Sigma_2 = scattering_table_h[sigma_2_coefficient_index] * ( sigma_t2 * sigma_theta2 - pow( sigma_t2_theta2, 2 ) );//ad-bc
		Sigma_2I[0] = sigma_theta2 / determinant_Sigma_2;
		Sigma_2I[1] = sigma_t2_theta2 / determinant_Sigma_2;	// negative sign is propagated to subsequent calculations instead of here 
		Sigma_2I[2] = sigma_t2 / determinant_Sigma_2;

		// Inverse Scattering Matrices: Total Ops = 2 + 3 + 11 + 7 + 7 = 30
		/**********************************************************************************************************************************************************/
		// 2 + 1 + 1 + 3 + 3 + 3 = 13
		first_term_common_24_1 = Sigma_2I[0] * depth_2_go - Sigma_2I[1];
		//first_term_common_24_2 = Sigma_2I[2] - Sigma_2I[1] * depth_2_go;
		first_term[0] = Sigma_1I[0] + Sigma_2I[0];
		first_term[1] = first_term_common_24_1 - Sigma_1I[1];
		first_term[2] = depth_2_go * Sigma_2I[0] - Sigma_1I[1] - Sigma_2I[1];
		first_term[3] = Sigma_1I[2] + Sigma_2I[2] + depth_2_go * ( first_term_common_24_1 - Sigma_2I[1]);	
		//first_term[3] = Sigma_1I[2] + Sigma_2I[2] + depth_2_go * first_term_common_24_1  - depth_2_go * Sigma_2I[1];
		determinant_first_term = first_term[0] * first_term[3] - first_term[1] * first_term[2];
		
		// 3 + 2 + 7 + 4 = 16
		second_term_common_3 = Sigma_2I[0] * t_out_object - Sigma_2I[1] * T_2[1];	
		second_term[0] = Sigma_1I[0] * t_in_object + second_term_common_3;
		second_term[1] = depth_2_go * second_term_common_3 + Sigma_2I[2] * T_2[1] - Sigma_2I[1] * t_out_object - Sigma_1I[1] * t_in_object;	
		t_1 = ( first_term[3] * second_term[0] - first_term[1] * second_term[1] ) / determinant_first_term ;
		//double theta_1 = first_term[2] * second_term[0] + first_term[3] * second_term[1];
		// t: Total Ops = 17 + 16 = 33
		/**********************************************************************************************************************************************************/
		// Do v MLP Now
		// 3 + 2 + 7 + 4 = 16
		second_term_common_3 = Sigma_2I[0] * z_out_object - Sigma_2I[1] * V_2[1];
		second_term[0] = Sigma_1I[0] * z_in_object + second_term_common_3;
		second_term[1] = depth_2_go * second_term_common_3 + Sigma_2I[2] * V_2[1] - Sigma_2I[1] * z_out_object - Sigma_1I[1] * z_in_object;
		v_1 = ( first_term[3] * second_term[0] - first_term[1] * second_term[1] ) / determinant_first_term ;
		//double phi_1 = first_term[2] * second_term[0] + first_term[3] * second_term[1];
		// v: Total Ops = 16
		/**********************************************************************************************************************************************************/
		// Rotate Coordinate From utv to xyz Coordinate System and Determine Which Voxel this Point on the MLP Path is in
		
		// 1 + 3 + 3 = 7
		//u_shifted += parameters.MLP_U_STEP;
		u_shifted = u_in_object + u_1;
		x_1 = cos_term * u_shifted - sin_term * t_1;
		y_1 = sin_term * u_shifted + cos_term * t_1;

		// 4 + 4 + 4 + 5 = 17
		voxel_x = calculate_voxel( parameters.X_ZERO_COORDINATE, x_1, parameters.VOXEL_WIDTH );
		voxel_y = calculate_voxel( parameters.Y_ZERO_COORDINATE, y_1, parameters.VOXEL_HEIGHT );
		voxel_z = calculate_voxel( parameters.Z_ZERO_COORDINATE, v_1, parameters.VOXEL_THICKNESS);
		voxel = voxel_x + voxel_y * parameters.COLUMNS + voxel_z * parameters.COLUMNS * parameters.ROWS;

		// 3
		if( voxel != path[num_intersections] )
			path[++num_intersections] = voxel;	

		// 1 + 1 + 1 + 1 + 1 = 5
		//u_1 += parameters.MLP_U_STEP;
		//depth_2_go -= parameters.MLP_U_STEP;
		step_number++;
		u_1 = step_number * parameters.MLP_U_STEP;
		depth_2_go = u_2 - u_1;
		sigma_1_coefficient_index += sigma_table_index_step;
		sigma_2_coefficient_index -= sigma_table_index_step;
		u_1_poly_index += poly_table_index_step;

		// Total: 7 + 17 + 3 + 5 = 32
		// Complete = 30 + 17 + 16 + 13 + 32 = 108
	}
	++num_intersections;
	return num_intersections;
}
__device__ double EffectiveChordLength_GPU(configurations* parameters, double abs_angle_t, double abs_angle_v)
{	
	double eff_angle_t,eff_angle_v;	
	eff_angle_t=abs_angle_t-((int)(abs_angle_t/(PI/2)))*(PI/2);
	eff_angle_v=fabs(abs_angle_v);
	
	// Get the effective chord in the t-u plane
	double step_fraction=parameters->MLP_U_STEP / parameters->VOXEL_WIDTH;
	double chord_length_2D=(1/3.0)*((step_fraction*step_fraction*sinf(2*eff_angle_t)-6)/(step_fraction*sinf(2*eff_angle_t)-2*(cosf(eff_angle_t)+sinf(eff_angle_t))) + step_fraction*step_fraction*sinf(2*eff_angle_t)/(2*(cosf(eff_angle_t)+sinf(eff_angle_t))));
	
	// Multiply this by the effective chord in the v-u plane
	double mean_pixel_width=parameters->VOXEL_WIDTH/(cosf(eff_angle_t)+sinf(eff_angle_t));
	double height_fraction=parameters->VOXEL_THICKNESS/mean_pixel_width;
	step_fraction=parameters->MLP_U_STEP/mean_pixel_width;
	double chord_length_3D=(1/3.0)*((step_fraction*step_fraction*sinf(2*eff_angle_v)-6*height_fraction)/(step_fraction*sinf(2*eff_angle_v)-2*(height_fraction*cosf(eff_angle_v)+sinf(eff_angle_v))) + step_fraction*step_fraction*sinf(2*eff_angle_v)/(2*(height_fraction*cosf(eff_angle_v)+sinf(eff_angle_v))));
	
	return parameters->VOXEL_WIDTH*chord_length_2D*chord_length_3D;
	 
}
template<typename O> __device__ bool find_MLP_endpoints_GPU
(
	configurations* parameters, O* image, double x_start, double y_start, double z_start, double xy_angle, double xz_angle, 
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
		


		voxel_x = calculate_voxel_GPU( parameters->X_ZERO_COORDINATE, x, parameters->VOXEL_WIDTH );
		voxel_y = calculate_voxel_GPU( parameters->Y_ZERO_COORDINATE, y, parameters->VOXEL_HEIGHT );
		voxel_z = calculate_voxel_GPU( parameters->Z_ZERO_COORDINATE, z, parameters->VOXEL_THICKNESS );

		x_to_go = distance_remaining_GPU( parameters->X_ZERO_COORDINATE, x, X_INCREASING_DIRECTION, x_move_direction, parameters->VOXEL_WIDTH, voxel_x );
		y_to_go = distance_remaining_GPU( parameters->Y_ZERO_COORDINATE, y, Y_INCREASING_DIRECTION, y_move_direction, parameters->VOXEL_HEIGHT, voxel_y );	
		z_to_go = distance_remaining_GPU( parameters->Z_ZERO_COORDINATE, z, Z_INCREASING_DIRECTION, z_move_direction, parameters->VOXEL_THICKNESS, voxel_z );

		voxel = voxel_x + voxel_y * parameters->COLUMNS + voxel_z * parameters->COLUMNS * parameters->ROWS;
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
		outside_image = (voxel_x >= parameters->COLUMNS ) || (voxel_y >= parameters->ROWS ) || (voxel_z >= parameters->SLICES ) || (voxel_x < 0  ) || (voxel_y < 0 ) || (voxel_z < 0 );		
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
					
					z = edge_coordinate_GPU( parameters->Z_ZERO_COORDINATE, voxel_z, parameters->VOXEL_THICKNESS, Z_INCREASING_DIRECTION, z_move_direction );					
					x = corresponding_coordinate_GPU( dx_dz, z, z_start, x_start );
					y = corresponding_coordinate_GPU( dy_dz, z, z_start, y_start );

					/*if(debug_run && j <= j_high_limit && j >= j_low_limit )
					{
						printf(" x = %3f y = %3f z = %3f\n",  x, y, z );
						printf(" x_to_go = %3f y_to_go = %3f z_to_go = %3f\n",  x_to_go, y_to_go, z_to_go );
						printf("voxel_x = %d voxel_y = %d voxel_z = %d voxel = %d\n", voxel_x, voxel_y, voxel_z, voxel);
					}*/

					x_to_go = distance_remaining_GPU( parameters->X_ZERO_COORDINATE, x, X_INCREASING_DIRECTION, x_move_direction, parameters->VOXEL_WIDTH, voxel_x );
					y_to_go = distance_remaining_GPU( parameters->Y_ZERO_COORDINATE, y, Y_INCREASING_DIRECTION, y_move_direction, parameters->VOXEL_HEIGHT, voxel_y );	
					z_to_go = parameters->VOXEL_THICKNESS;
				}
				//If Next Voxel Edge is in x or xy Diagonal
				else if( x_extension <= y_extension )
				{
					//printf(" x_extension <= y_extension \n");			
					voxel_x += x_move_direction;

					x = edge_coordinate_GPU( parameters->X_ZERO_COORDINATE, voxel_x, parameters->VOXEL_WIDTH, X_INCREASING_DIRECTION, x_move_direction );
					y = corresponding_coordinate_GPU( dy_dx, x, x_start, y_start );
					z = corresponding_coordinate_GPU( dz_dx, x, x_start, z_start );

					x_to_go = parameters->VOXEL_WIDTH;
					y_to_go = distance_remaining_GPU( parameters->Y_ZERO_COORDINATE, y, Y_INCREASING_DIRECTION, y_move_direction, parameters->VOXEL_HEIGHT, voxel_y );
					z_to_go = distance_remaining_GPU( parameters->Z_ZERO_COORDINATE, z, Z_INCREASING_DIRECTION, z_move_direction, parameters->VOXEL_THICKNESS, voxel_z );
				}
				// Else Next Voxel Edge is in y
				else
				{
					//printf(" y_extension < x_extension \n");
					voxel_y -= y_move_direction;
					
					y = edge_coordinate_GPU( parameters->Y_ZERO_COORDINATE, voxel_y, parameters->VOXEL_HEIGHT, Y_INCREASING_DIRECTION, y_move_direction );
					x = corresponding_coordinate_GPU( dx_dy, y, y_start, x_start );
					z = corresponding_coordinate_GPU( dz_dy, y, y_start, z_start );

					x_to_go = distance_remaining_GPU( parameters->X_ZERO_COORDINATE, x, X_INCREASING_DIRECTION, x_move_direction, parameters->VOXEL_WIDTH, voxel_x );
					y_to_go = parameters->VOXEL_HEIGHT;					
					z_to_go = distance_remaining_GPU( parameters->Z_ZERO_COORDINATE, z, Z_INCREASING_DIRECTION, z_move_direction, parameters->VOXEL_THICKNESS, voxel_z );
				}
				// <= VOXEL_ALLOWANCE
				if( x_to_go == 0 )
				{
					x_to_go = parameters->VOXEL_WIDTH;
					voxel_x += x_move_direction;
				}
				if( y_to_go == 0 )
				{
					y_to_go = parameters->VOXEL_HEIGHT;
					voxel_y -= y_move_direction;
				}
				if( z_to_go == 0 )
				{
					z_to_go = parameters->VOXEL_THICKNESS;
					voxel_z -= z_move_direction;
				}
				
				voxel_z = max(voxel_z, 0 );
				voxel = voxel_x + voxel_y * parameters->COLUMNS + voxel_z * parameters->COLUMNS * parameters->ROWS;
				//if(debug_run && j <= j_high_limit && j >= j_low_limit )
				//{
				//	printf(" x = %3f y = %3f z = %3f\n",  x, y, z );
				//	printf(" x_to_go = %3f y_to_go = %3f z_to_go = %3f\n",  x_to_go, y_to_go, z_to_go );
				//	printf("voxel_x = %d voxel_y = %d voxel_z = %d voxel = %d\n", voxel_x, voxel_y, voxel_z, voxel);
				//}
				outside_image = (voxel_x >= parameters->COLUMNS ) || (voxel_y >= parameters->ROWS ) || (voxel_z >= parameters->SLICES ) || (voxel_x < 0  ) || (voxel_y < 0 ) || (voxel_z < 0 );		
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
					
					x = edge_coordinate_GPU( parameters->X_ZERO_COORDINATE, voxel_x, parameters->VOXEL_WIDTH, X_INCREASING_DIRECTION, x_move_direction );
					y = corresponding_coordinate_GPU( dy_dx, x, x_start, y_start );

					x_to_go = parameters->VOXEL_WIDTH;
					y_to_go = distance_remaining_GPU( parameters->Y_ZERO_COORDINATE, y, Y_INCREASING_DIRECTION, y_move_direction, parameters->VOXEL_HEIGHT, voxel_y );
				}
				// Else Next Voxel Edge is in y
				else
				{
					//printf(" y_extension < x_extension \n");				
					voxel_y -= y_move_direction;

					y = edge_coordinate_GPU( parameters->Y_ZERO_COORDINATE, voxel_y, parameters->VOXEL_HEIGHT, Z_INCREASING_DIRECTION, y_move_direction );
					x = corresponding_coordinate_GPU( dx_dy, y, y_start, x_start );

					x_to_go = distance_remaining_GPU( parameters->X_ZERO_COORDINATE, x, X_INCREASING_DIRECTION, x_move_direction, parameters->VOXEL_WIDTH, voxel_x );
					y_to_go = parameters->VOXEL_HEIGHT;
				}
				// <= VOXEL_ALLOWANCE
				if( x_to_go == 0 )
				{
					x_to_go = parameters->VOXEL_WIDTH;
					voxel_x += x_move_direction;
				}
				if( y_to_go == 0 )
				{
					y_to_go = parameters->VOXEL_HEIGHT;
					voxel_y -= y_move_direction;
				}
				voxel = voxel_x + voxel_y * parameters->COLUMNS + voxel_z * parameters->COLUMNS * parameters->ROWS;		
				/*if(debug_run && j <= j_high_limit && j >= j_low_limit )
				{
					printf(" x = %3f y = %3f z = %3f\n",  x, y, z );
					printf(" x_to_go = %3f y_to_go = %3f z_to_go = %3f\n",  x_to_go, y_to_go, z_to_go );
					printf("voxel_x_in = %d voxel_y_in = %d voxel_z_in = %d\n", voxel_x, voxel_y, voxel_z);
				}*/
				outside_image = (voxel_x >= parameters->COLUMNS ) || (voxel_y >= parameters->ROWS ) || (voxel_z >= parameters->SLICES ) || (voxel_x < 0  ) || (voxel_y < 0 ) || (voxel_z < 0 );		
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

__device__ void find_MLP_path_GPU 
(
	configurations* parameters, float* x, double b_i, unsigned int first_MLP_voxel_number,
	double x_in_object, double y_in_object, double z_in_object, double x_out_object, double y_out_object, double z_out_object, 
	double xy_entry_angle, double xz_entry_angle, double xy_exit_angle, double xz_exit_angle,
	float lambda, unsigned int* path, float& update_value_history, int& num_intersections_historty
) 
{
	num_intersections_historty = 0;
	double u_0 = 0.0, u_1 = parameters->MLP_U_STEP,  u_2 = 0.0;
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
	
	double a_i_dot_x_k = 0.0;
	double a_i_dot_a_i = 0.0;
	
	//double theta_1, phi_1;
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	double effective_chord_length = EffectiveChordLength_GPU( parameters, ( xy_entry_angle + xy_exit_angle ) / 2.0, ( xz_entry_angle + xz_exit_angle) / 2.0 );
	double a_j_times_a_j = effective_chord_length * effective_chord_length;
	
	//double effective_chord_length = mean_chord_length( u_in_object, t_in_object, v_in_object, u_out_object, t_out_object, v_out_object );
	//double effective_chord_length = VOXEL_WIDTH;

	int voxel_x = 0, voxel_y = 0, voxel_z = 0;
	
	int voxel = first_MLP_voxel_number;
	
	path[num_intersections_historty] = voxel;
	//path[num_intersections_historty] = voxel;
	a_i_dot_x_k += x[voxel] * effective_chord_length;
	a_i_dot_a_i += a_j_times_a_j;
	//atomicAdd(S[voxel], 1);
	//if(!constant_chord_lengths)
		//chord_lengths[num_intersections] = VOXEL_WIDTH;
	num_intersections_historty++;
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
	u_1 = parameters->MLP_U_STEP;
	u_2 = abs(u_out_object - u_in_object);		
	//fgets(user_response, sizeof(user_response), stdin);

	//output_file.open(filename);						
				      
	//precalculated u_2 dependent terms (since u_2 does not change inside while loop)
	//u_2 terms
	sigma_2_pre_1 =  u_2*u_2*u_2 * ( A_0_OVER_3 + u_2 * ( A_1_OVER_12 + u_2 * ( A_2_OVER_30 + u_2 * (A_3_OVER_60 + u_2 * ( A_4_OVER_105 + u_2 * A_5_OVER_168 )))));;	//u_2^3 : 1/3, 1/12, 1/30, 1/60, 1/105, 1/168
	sigma_2_pre_2 =  u_2*u_2 * ( A_0_OVER_2 + u_2 * (A_1_OVER_6 + u_2 * (A_2_OVER_12 + u_2 * ( A_3_OVER_20 + u_2 * (A_4_OVER_30 + u_2 * A_5_OVER_42)))));	//u_2^2 : 1/2, 1/6, 1/12, 1/20, 1/30, 1/42
	sigma_2_pre_3 =  u_2 * ( A_0 +  u_2 * (A_1_OVER_2 +  u_2 * ( A_2_OVER_3 +  u_2 * ( A_3_OVER_4 +  u_2 * ( A_4_OVER_5 + u_2 * A_5_OVER_6 )))));			//u_2 : 1/1, 1/2, 1/3, 1/4, 1/5, 1/6

	while( u_1 < u_2 - parameters->MLP_U_STEP)
	//while( u_1 < u_2 - 0.001)
	{
		R_0[1] = u_1;
		R_1[1] = u_2 - u_1;
		R_1T[2] = u_2 - u_1;

		sigma_1_coefficient = pow( E_0 * ( 1 + 0.038 * log( (u_1 - u_0)/X0) ), 2.0 ) / X0;
		sigma_t1 =  sigma_1_coefficient * ( pow(u_1, 3.0) * ( A_0_OVER_3 + u_1 * (A_1_OVER_12 + u_1 * (A_2_OVER_30 + u_1 * (A_3_OVER_60 + u_1 * (A_4_OVER_105 + u_1 * A_5_OVER_168 ) )))) );	//u_1^3 : 1/3, 1/12, 1/30, 1/60, 1/105, 1/168
		sigma_t1_theta1 =  sigma_1_coefficient * ( pow(u_1, 2.0) * ( A_0_OVER_2 + u_1 * (A_1_OVER_6 + u_1 * (A_2_OVER_12 + u_1 * (A_3_OVER_20 + u_1 * (A_4_OVER_30 + u_1 * A_5_OVER_42))))) );	//u_1^2 : 1/2, 1/6, 1/12, 1/20, 1/30, 1/42															
		sigma_theta1 = sigma_1_coefficient * ( u_1 * ( A_0 + u_1 * (A_1_OVER_2 + u_1 * (A_2_OVER_3 + u_1 * (A_3_OVER_4 + u_1 * (A_4_OVER_5 + u_1 * A_5_OVER_6))))) );			//u_1 : 1/1, 1/2, 1/3, 1/4, 1/5, 1/6																	
		determinant_Sigma_1 = sigma_t1 * sigma_theta1 - pow( sigma_t1_theta1, 2 );//ad-bc
			
		Sigma_1I[0] = sigma_theta1 / determinant_Sigma_1;
		Sigma_1I[1] = -sigma_t1_theta1 / determinant_Sigma_1;
		Sigma_1I[2] = -sigma_t1_theta1 / determinant_Sigma_1;
		Sigma_1I[3] = sigma_t1 / determinant_Sigma_1;

		//sigma 2 terms
		sigma_2_coefficient = pow( E_0 * ( 1 + 0.038 * log( (u_2 - u_1)/X0) ), 2.0 ) / X0;
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

		voxel_x = calculate_voxel_GPU( parameters->X_ZERO_COORDINATE, x_1, parameters->VOXEL_WIDTH );
		voxel_y = calculate_voxel_GPU( parameters->Y_ZERO_COORDINATE, y_1, parameters->VOXEL_HEIGHT );
		voxel_z = calculate_voxel_GPU( parameters->Z_ZERO_COORDINATE, z_1, parameters->VOXEL_THICKNESS);
				
		voxel = voxel_x + voxel_y * parameters->COLUMNS + voxel_z * parameters->COLUMNS * parameters->ROWS;
		//cout << "voxel_x = " << voxel_x << "voxel_y = " << voxel_y << "voxel_z = " << voxel_z << "voxel = " << voxel <<endl;
		//fgets(user_response, sizeof(user_response), stdin);

		if( voxel != path[num_intersections_historty - 1] )
		{
			//path[num_intersections_historty] = voxel;
			path[num_intersections_historty] = voxel;
			a_i_dot_x_k += x[voxel] * effective_chord_length;
			a_i_dot_a_i += a_j_times_a_j;
			//atomicAdd(S[voxel], 1);
			//MLP_test_image_h[voxel] = 0;
			//if(!constant_chord_lengths)
				//chord_lengths[num_intersections] = effective_chord_length;						
			num_intersections_historty++;
		}
		u_1 += parameters->MLP_U_STEP;
	}
	
	
	
	update_value_history = effective_chord_length * (( b_i - a_i_dot_x_k ) /  a_i_dot_a_i) * lambda;
	
	//return update_value;
	
	
  
}
__device__ void find_MLP_path_GPU_tabulated
(
	configurations* parameters, float* x, double b_i, unsigned int first_MLP_voxel_number,
	double x_in_object, double y_in_object, double z_in_object, double x_out_object, double y_out_object, double z_out_object, 
	double xy_entry_angle, double xz_entry_angle, double xy_exit_angle, double xz_exit_angle,
	float lambda, unsigned int* path, float& update_value_history, int& num_intersections,
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
	double u_1 = parameters->MLP_U_STEP;
	double u_2 = abs(u_out_object - u_in_object);
	double depth_2_go = u_2 - u_1;
	double u_shifted = u_in_object;	
	unsigned int step_number = 1;

	// Scattering Coefficient tables indices
	unsigned int sigma_table_index_step = static_cast<unsigned int>( parameters->MLP_U_STEP / DEPTH_TABLE_STEP + 0.5 );
	unsigned int sigma_1_coefficient_index = sigma_table_index_step;
	unsigned int sigma_2_coefficient_index = static_cast<unsigned int>( depth_2_go / DEPTH_TABLE_STEP + 0.5 );
	
	// Scattering polynomial indices
	unsigned int poly_table_index_step = static_cast<unsigned int>( parameters->MLP_U_STEP / POLY_TABLE_STEP + 0.5 );
	unsigned int u_1_poly_index = poly_table_index_step;
	unsigned int u_2_poly_index = static_cast<unsigned int>( u_2 / POLY_TABLE_STEP + 0.5 );

	//precalculated u_2 dependent terms (since u_2 does not change inside while loop)
	double u_2_poly_3_12 = poly_3_12[u_2_poly_index];
	double u_2_poly_2_6 = poly_2_6[u_2_poly_index];
	double u_2_poly_1_2 = poly_1_2[u_2_poly_index];
	double u_1_poly_1_2, u_1_poly_2_3;

	int voxel = first_MLP_voxel_number;
	num_intersections = 0;
	path[num_intersections] = voxel;
	//num_intersections++;

	double effective_chord_length = EffectiveChordLength_GPU( parameters, ( xy_entry_angle + xy_exit_angle ) / 2.0, ( xz_entry_angle + xz_exit_angle) / 2.0 );
	//double a_j_times_a_j = effective_chord_length * effective_chord_length;
	double a_i_dot_x_k = x[voxel];
	double a_i_dot_a_i = effective_chord_length * effective_chord_length;
	//double a_i_dot_x_k = x[voxel] * effective_chord_length;
	//double a_i_dot_a_i = a_j_times_a_j;

	//while( u_1 < u_2 - parameters.MLP_U_STEP)
	while( depth_2_go > parameters->MLP_U_STEP )
	{
		u_1_poly_1_2 = poly_1_2[u_1_poly_index];
		u_1_poly_2_3 = poly_2_3[u_1_poly_index];

		sigma_t1 = poly_3_12[u_1_poly_index];										// poly_3_12(u_1)
		sigma_t1_theta1 =  poly_2_6[u_1_poly_index];								// poly_2_6(u_1) 
		sigma_theta1 = u_1_poly_1_2;												// poly_1_2(u_1)

		sigma_t2 =  u_2_poly_3_12 - pow(u_2, 2.0) * u_1_poly_1_2 + 2 * u_2 * u_1_poly_2_3 - poly_3_4[u_1_poly_index];	// poly_3_12(u_2) - u_2^2 * poly_1_2(u_1) +2u_2*(u_1) - poly_3_4(u_1)
		sigma_t2_theta2 =  u_2_poly_2_6 - u_2 * u_1_poly_1_2 + u_1_poly_2_3;											// poly_2_6(u_2) - u_2*poly_1_2(u_1) + poly_2_3(u_1)
		sigma_theta2 =  u_2_poly_1_2 - u_1_poly_1_2;																	// poly_1_2(u_2) - poly_1_2(u_1)	

		determinant_Sigma_1 = scattering_table[sigma_1_coefficient_index] * ( sigma_t1 * sigma_theta1 - pow( sigma_t1_theta1, 2 ) );//ad-bc
		Sigma_1I[0] = sigma_theta1 / determinant_Sigma_1;
		Sigma_1I[1] = sigma_t1_theta1 / determinant_Sigma_1;	// negative sign is propagated to subsequent calculations instead of here 
		Sigma_1I[2] = sigma_t1 / determinant_Sigma_1;			
			
		determinant_Sigma_2 = scattering_table[sigma_2_coefficient_index] * ( sigma_t2 * sigma_theta2 - pow( sigma_t2_theta2, 2 ) );//ad-bc
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
		//u_shifted += parameters.MLP_U_STEP;
		u_shifted = u_in_object + u_1;
		x_1 = cos_term * u_shifted - sin_term * t_1;
		y_1 = sin_term * u_shifted + cos_term * t_1;

		voxel_x = calculate_voxel_GPU( parameters->X_ZERO_COORDINATE, x_1, parameters->VOXEL_WIDTH );
		voxel_y = calculate_voxel_GPU( parameters->Y_ZERO_COORDINATE, y_1, parameters->VOXEL_HEIGHT );
		voxel_z = calculate_voxel_GPU( parameters->Z_ZERO_COORDINATE, v_1, parameters->VOXEL_THICKNESS);			
		voxel = voxel_x + voxel_y * parameters->COLUMNS + voxel_z * parameters->COLUMNS * parameters->ROWS;

		if( voxel != path[num_intersections] )
		{
			path[++num_intersections] = voxel;	
			//a_i_dot_x_k += x[voxel] * effective_chord_length;
			a_i_dot_x_k += x[voxel];
			//a_i_dot_a_i += a_j_times_a_j;
		}
		//u_1 += parameters.MLP_U_STEP;
		//depth_2_go -= parameters.MLP_U_STEP;
		step_number++;
		u_1 = step_number * parameters->MLP_U_STEP;
		depth_2_go = u_2 - u_1;
		sigma_1_coefficient_index += sigma_table_index_step;
		sigma_2_coefficient_index -= sigma_table_index_step;
		u_1_poly_index += poly_table_index_step;
	}
	++num_intersections;
	a_i_dot_x_k *= effective_chord_length;
	a_i_dot_a_i *= num_intersections;
	update_value_history = effective_chord_length * (( b_i - a_i_dot_x_k ) /  a_i_dot_a_i) * lambda;
}
__global__ void collect_MLP_endpoints_GPU
(
	configurations* parameters, bool* intersected_hull, unsigned int* first_MLP_voxel, bool* x_hull, 
	float* x_entry, float* y_entry, float* z_entry, float* xy_entry_angle, float* xz_entry_angle, 
	float* x_exit, float* y_exit, float* z_exit, float* xy_exit_angle, float* xz_exit_angle, int start_proton_id, int num_histories
)
{
  
	bool entered_object = false, exited_object = false;
	int voxel_x = 0, voxel_y = 0, voxel_z = 0, voxel_x_int = 0, voxel_y_int = 0, voxel_z_int = 0;
	double x_in_object = 0.0, y_in_object = 0.0, z_in_object = 0.0, x_out_object = 0.0, y_out_object = 0.0, z_out_object = 0.0;
		
	int proton_id = blockIdx.x * blockDim.x + threadIdx.x + start_proton_id;
		
	if( proton_id < num_histories ) 
	{
		intersected_hull[proton_id] = false;
		first_MLP_voxel[proton_id] = parameters->NUM_VOXELS;
		entered_object = find_MLP_endpoints_GPU( parameters, x_hull, x_entry[proton_id], y_entry[proton_id], z_entry[proton_id], xy_entry_angle[proton_id], xz_entry_angle[proton_id], x_in_object, y_in_object, z_in_object, voxel_x, voxel_y, voxel_z, true);	
		exited_object = find_MLP_endpoints_GPU( parameters, x_hull, x_exit[proton_id], y_exit[proton_id], z_exit[proton_id], xy_exit_angle[proton_id], xz_exit_angle[proton_id], x_out_object, y_out_object, z_out_object, voxel_x_int, voxel_y_int, voxel_z_int, false);
				
		if( entered_object && exited_object ) 
		{	  
			intersected_hull[proton_id] = true;
			first_MLP_voxel[proton_id] = voxel_x + parameters->COLUMNS * voxel_y + parameters->ROWS * parameters->COLUMNS * voxel_z;
			x_entry[proton_id] = x_in_object;
			y_entry[proton_id] = y_in_object;
			z_entry[proton_id] = z_in_object;
			x_exit[proton_id] = x_out_object;
			y_exit[proton_id] = y_out_object;
			z_exit[proton_id] = z_out_object;
		}	  
	}
}
__global__ void block_update_GPU
(
	configurations* parameters, float* x, float* x_entry, float* y_entry, float* z_entry, float* xy_entry_angle, float* xz_entry_angle, 
	float* x_exit, float* y_exit, float* z_exit, float* xy_exit_angle, float* xz_exit_angle, float* WEPL, unsigned int* first_MLP_voxel, 
	float* x_update, unsigned int* intersection_counts, int start_proton_id, int post_cut_protons, float lambda
) 
{
	// Shared memory
	//__shared__ double* x_block_update;
	//__shared__ unsigned int* voxel_intersections;
	
	//x_block_update = (double*)malloc( NUM_VOXELS * sizeof(double));
	//voxel_intersections = (unsigned int*)malloc( MAX_INTERSECTIONS * sizeof(unsigned int));

	if( start_proton_id < post_cut_protons ) 
	{	  
		unsigned int* a_i;
		a_i = (unsigned int*)malloc( MAX_INTERSECTIONS * sizeof(unsigned int));
		
		// Initialize arrays to zero
		for(int j = 0 ; j < MAX_INTERSECTIONS; ++j) 
		      a_i[j] = 0;		
		
		for( int history = 0; history < HISTORIES_PER_THREAD; history++ ) 
		{
		
			//int proton_id = blockIdx.x * blockDim.x + threadIdx.x + start_proton_id;  
			//int proton_id =  history + threadIdx.x * HISTORIES_PER_THREAD + start_proton_id; 		
			int proton_id =  start_proton_id + history + threadIdx.x * HISTORIES_PER_THREAD + blockIdx.x * HISTORIES_PER_BLOCK;
			
			if( proton_id < post_cut_protons ) 
			{  
				float update_value_history = 0.0;
				int num_intersections_historty = 0;
	
				//collect_MLP_endpoints_GPU(x, x_hull, x_entry, y_entry, z_entry, xy_entry_angle, xz_entry_angle, x_exit, y_exit, z_exit,  xy_exit_angle, 
					//	xz_exit_angle, WEPL, lambda, proton_id, post_cut_protons, a_i, update_value_history, num_intersections_historty); // Each thread runs a proton	
				find_MLP_path_GPU(parameters, x, WEPL[proton_id], first_MLP_voxel[proton_id] ,x_entry[proton_id], y_entry[proton_id], z_entry[proton_id], x_exit[proton_id], y_exit[proton_id], z_exit[proton_id], xy_entry_angle[proton_id],
						      xz_entry_angle[proton_id], xy_exit_angle[proton_id], xz_exit_angle[proton_id], lambda, a_i, update_value_history, num_intersections_historty);

				// Copy a_i to global	
				for (int j = 0 ; j < num_intersections_historty; ++j) 
				{	  
					atomicAdd( &( intersection_counts[a_i[j]]), 1 );
					atomicAdd( &( x_update[a_i[j]]) , update_value_history ); 
				}	
			}	
		}
		free(a_i);  	
	}
}
__global__ void block_update_GPU_tabulated
(
	configurations* parameters, float* x, float* x_entry, float* y_entry, float* z_entry, float* xy_entry_angle, float* xz_entry_angle, 
	float* x_exit, float* y_exit, float* z_exit, float* xy_exit_angle, float* xz_exit_angle, float* WEPL, unsigned int* first_MLP_voxel, 
	float* x_update, unsigned int* intersection_counts, int start_proton_id, int post_cut_protons, float lambda,
	double* sin_table, double* cos_table, double* scattering_table, double* poly_1_2, double* poly_2_3, double* poly_3_4, double* poly_2_6, double* poly_3_12
) 
{
	// Shared memory
	//__shared__ double* x_block_update;
	//__shared__ unsigned int* voxel_intersections;
	
	//x_block_update = (double*)malloc( NUM_VOXELS * sizeof(double));
	//voxel_intersections = (unsigned int*)malloc( MAX_INTERSECTIONS * sizeof(unsigned int));

	if( start_proton_id < post_cut_protons ) 
	{	  
		unsigned int* a_i;
		a_i = (unsigned int*)malloc( MAX_INTERSECTIONS * sizeof(unsigned int));
		
		// Initialize arrays to zero
		for(int j = 0 ; j < MAX_INTERSECTIONS; ++j) 
		      a_i[j] = 0;		
		
		for( int history = 0; history < HISTORIES_PER_THREAD; history++ ) 
		{
		
			//int proton_id = blockIdx.x * blockDim.x + threadIdx.x + start_proton_id;  
			//int proton_id =  history + threadIdx.x * HISTORIES_PER_THREAD + start_proton_id; 		
			int proton_id =  start_proton_id + history + threadIdx.x * HISTORIES_PER_THREAD + blockIdx.x * HISTORIES_PER_BLOCK;
			
			if( proton_id < post_cut_protons ) 
			{  
				float update_value_history = 0.0;
				int num_intersections_historty = 0;
	
				//collect_MLP_endpoints_GPU(x, x_hull, x_entry, y_entry, z_entry, xy_entry_angle, xz_entry_angle, x_exit, y_exit, z_exit,  xy_exit_angle, 
					//	xz_exit_angle, WEPL, lambda, proton_id, post_cut_protons, a_i, update_value_history, num_intersections_historty); // Each thread runs a proton	
				find_MLP_path_GPU_tabulated
				(
					parameters, x, WEPL[proton_id], first_MLP_voxel[proton_id] ,x_entry[proton_id], y_entry[proton_id], z_entry[proton_id], x_exit[proton_id], y_exit[proton_id], z_exit[proton_id], xy_entry_angle[proton_id],
					xz_entry_angle[proton_id], xy_exit_angle[proton_id], xz_exit_angle[proton_id], lambda, a_i, update_value_history, num_intersections_historty,
					sin_table, cos_table, scattering_table, poly_1_2, poly_2_3, poly_3_4, poly_2_6, poly_3_12
				);

				// Copy a_i to global	
				for (int j = 0 ; j < num_intersections_historty; ++j) 
				{	  
					atomicAdd( &( intersection_counts[a_i[j]]), 1 );
					atomicAdd( &( x_update[a_i[j]]) , update_value_history ); 
				}	
			}	
		}
		free(a_i);  	
	}
}
// x_d, x_update, and voxel_intersections are global
__global__ void image_update_GPU (configurations* parameters, float* x, float* x_update, unsigned int* voxel_intersections) {
  
        int row = blockIdx.y, column = blockIdx.x, slice = threadIdx.x;
	int voxel = column + row * parameters->COLUMNS + slice * parameters->COLUMNS * parameters->ROWS;
	/*int voxel = 0;
	int column_start = VOXELS_PER_THREAD * blockIdx.x;
	int row = blockIdx.y;
	int slice = threadIdx.x;
	int voxel_start = column_start  + row * COLUMNS + slice * ROWS * COLUMNS;
	*/
	for( int shift = 0; shift < VOXELS_PER_THREAD; shift++ ) {
	  
		//voxel = (column_start  + shift) +  row * COLUMNS + slice * ROWS * COLUMNS;
		
		if(voxel < parameters->NUM_VOXELS && voxel_intersections[voxel] > 0 ) {
		  
			atomicAdd(&(x[voxel]) , x_update[voxel] / voxel_intersections[voxel]);
			x_update[voxel] = 0.0;
			voxel_intersections[voxel]=0;
		}

	}


	
	//if( voxel < NUM_VOXELS && num_intersections[voxel] > 0 )
		//x_k[voxel] += x_update[voxel] / num_intersections[voxel];

	//int voxel = threadIdx.x;
       /* if(voxel < NUM_VOXELS && voxel_intersections[voxel] > 0 ) {
	  atomicAdd(&(x[voxel]) , x_update[voxel] / voxel_intersections[voxel]);
	  x_update[voxel] = 0.0;
	  voxel_intersections[voxel]=0;
	}*/
}
void gpu_memory_allocation( const int num_histories ) {
  
	  cudaFree(x_d);
	  cudaFree(x_update_d);
	  cudaFree(intersection_counts_d);
	  cudaFree(hull_d);
	  
	  unsigned int size_floats		= sizeof(float) * num_histories;
	  unsigned int size_ints		= sizeof(int) * num_histories;
	  unsigned int size_bool		= sizeof(bool) * num_histories;
  
	  
	  
	  
  
	  cudaMalloc( (void**) &x_d, 				parameters.NUM_VOXELS *sizeof(float));
	  cudaMalloc( (void**) &x_update_d, 			parameters.NUM_VOXELS *sizeof(float));
	  cudaMalloc( (void**) &intersection_counts_d, 		parameters.NUM_VOXELS *sizeof(unsigned int));
	  cudaMalloc( (void**) &hull_d, 			parameters.NUM_VOXELS *sizeof(bool));
	  
	  cudaMalloc( (void**) &x_entry_d,			size_floats );
	  cudaMalloc( (void**) &y_entry_d,			size_floats );
	  cudaMalloc( (void**) &z_entry_d,			size_floats );
	  cudaMalloc( (void**) &x_exit_d,			size_floats );
	  cudaMalloc( (void**) &y_exit_d,			size_floats );
	  cudaMalloc( (void**) &z_exit_d,			size_floats );
	  
	  
	  cudaMalloc( (void**) &intersected_hull_d, 		size_bool );
	  cudaMalloc( (void**) &first_MLP_voxel_d, 		size_ints );
	  
	  
	  //cudaMalloc( (void**) &WEPL_d,				size_floats );
	  cudaMalloc( (void**) &xy_entry_angle_d,		size_floats );
	  cudaMalloc( (void**) &xz_entry_angle_d,		size_floats );
	  cudaMalloc( (void**) &xy_exit_angle_d,		size_floats );
	  cudaMalloc( (void**) &xz_exit_angle_d,		size_floats );
	  

}	
void transfer_host_to_device( const int start_position, const int num_histories ) {
  
	  
	
	  unsigned int size_floats		= sizeof(float) * num_histories;
	  unsigned int size_ints		= sizeof(int) * num_histories;
	  unsigned int size_bool		= sizeof(bool) * num_histories;
	  //int start_position = 0;
	  
	  intersected_hull_h = (bool*)calloc( num_histories, sizeof(bool) );
	  cudaMemcpy( intersected_hull_d, intersected_hull_h, num_histories *sizeof(bool),cudaMemcpyHostToDevice );
	  
	  
	  cudaMemcpy( x_d, x_h, parameters.NUM_VOXELS *sizeof(float),cudaMemcpyHostToDevice ); // copy initial iterate from host to device
	  cudaMemcpy( hull_d, hull_h, parameters.NUM_VOXELS *sizeof(bool),cudaMemcpyHostToDevice );
	  
	  
	  
	  cudaMemcpy( x_entry_d,				&x_entry_vector[start_position],		size_floats,	cudaMemcpyHostToDevice );
	  cudaMemcpy( y_entry_d,				&y_entry_vector[start_position],		size_floats,	cudaMemcpyHostToDevice );
	  cudaMemcpy( z_entry_d,				&z_entry_vector[start_position],		size_floats,	cudaMemcpyHostToDevice );
	  cudaMemcpy( x_exit_d,					&x_exit_vector[start_position],			size_floats,	cudaMemcpyHostToDevice );
	  cudaMemcpy( y_exit_d,					&y_exit_vector[start_position],			size_floats,	cudaMemcpyHostToDevice );
	  cudaMemcpy( z_exit_d,					&z_exit_vector[start_position],			size_floats,	cudaMemcpyHostToDevice );
	  //cudaMemcpy( WEPL_d,					&WEPL_vector[start_position],			size_floats,	cudaMemcpyHostToDevice );
	  cudaMemcpy( xy_entry_angle_d,				&xy_entry_angle_vector[start_position],		size_floats,	cudaMemcpyHostToDevice );
	  cudaMemcpy( xz_entry_angle_d,				&xz_entry_angle_vector[start_position],		size_floats,	cudaMemcpyHostToDevice );
	  cudaMemcpy( xy_exit_angle_d,				&xy_exit_angle_vector[start_position],		size_floats,	cudaMemcpyHostToDevice );
	  cudaMemcpy( xz_exit_angle_d,				&xz_exit_angle_vector[start_position],		size_floats,	cudaMemcpyHostToDevice );
	  
	  
}
void transfer_image_device_to_host() {

	  cudaMemcpy( x_h, x_d, parameters.NUM_VOXELS *sizeof(float), cudaMemcpyDeviceToHost);
	  
  
}
void transfer_intermediate_results_device_to_host( const int start_position, const int num_histories ) {
  
	unsigned int size_floats		= sizeof(float) * num_histories;
	unsigned int size_ints			= sizeof(int) * num_histories;
	unsigned int size_bool			= sizeof(bool) * num_histories;
  
	
	first_MLP_voxel_vector.resize( num_histories );
	  
	cudaMemcpy(intersected_hull_h, intersected_hull_d, size_bool, cudaMemcpyDeviceToHost);
	cudaMemcpy(&first_MLP_voxel_vector[start_position], first_MLP_voxel_d, size_ints, cudaMemcpyDeviceToHost);
	cudaMemcpy(&x_entry_vector[start_position], x_entry_d, size_floats, cudaMemcpyDeviceToHost);
	cudaMemcpy(&y_entry_vector[start_position], y_entry_d, size_floats, cudaMemcpyDeviceToHost);
	cudaMemcpy(&z_entry_vector[start_position], z_entry_d, size_floats, cudaMemcpyDeviceToHost);
	cudaMemcpy(&x_exit_vector[start_position], x_exit_d, size_floats, cudaMemcpyDeviceToHost);
	cudaMemcpy(&y_exit_vector[start_position], y_exit_d, size_floats, cudaMemcpyDeviceToHost);
	cudaMemcpy(&z_exit_vector[start_position], z_exit_d, size_floats, cudaMemcpyDeviceToHost);
	
	cudaFree(intersected_hull_d);
	cudaFree(first_MLP_voxel_d);
	//cudaFree(WEPL_d);
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


}
void transfer_intermediate_results_host_to_device( const int num_histories ) {
  
	unsigned int size_floats		= sizeof(float) * num_histories;
	unsigned int size_ints			= sizeof(int) * num_histories;
	unsigned int size_bool			= sizeof(bool) * num_histories;
	int start_position = 0;
	
	
	/*cudaFree(intersected_hull_d);
	cudaFree(first_MLP_voxel_d);
	//cudaFree(WEPL_d);
	cudaFree(x_entry_d);
	cudaFree(y_entry_d);
	cudaFree(z_entry_d);
	cudaFree(x_exit_d);
	cudaFree(y_exit_d);
	cudaFree(z_exit_d);
	cudaFree(xy_entry_angle_d);
	cudaFree(xz_entry_angle_d);
	cudaFree(xy_exit_angle_d);
	cudaFree(xz_exit_angle_d);*/
  
	//cudaFree(WEPL_d);
	
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
	
	cudaMalloc( (void**) &WEPL_d,				size_floats );
	cudaMalloc( (void**) &first_MLP_voxel_d, 		size_ints );
	
	cudaMemcpy( x_entry_d,					&x_entry_vector[start_position],		size_floats,	cudaMemcpyHostToDevice );
	cudaMemcpy( y_entry_d,					&y_entry_vector[start_position],		size_floats,	cudaMemcpyHostToDevice );
	cudaMemcpy( z_entry_d,					&z_entry_vector[start_position],		size_floats,	cudaMemcpyHostToDevice );
	cudaMemcpy( x_exit_d,					&x_exit_vector[start_position],			size_floats,	cudaMemcpyHostToDevice );
	cudaMemcpy( y_exit_d,					&y_exit_vector[start_position],			size_floats,	cudaMemcpyHostToDevice );
	cudaMemcpy( z_exit_d,					&z_exit_vector[start_position],			size_floats,	cudaMemcpyHostToDevice );
	
	cudaMemcpy( xy_entry_angle_d,				&xy_entry_angle_vector[start_position],		size_floats,	cudaMemcpyHostToDevice );
	cudaMemcpy( xz_entry_angle_d,				&xz_entry_angle_vector[start_position],		size_floats,	cudaMemcpyHostToDevice );
	cudaMemcpy( xy_exit_angle_d,				&xy_exit_angle_vector[start_position],		size_floats,	cudaMemcpyHostToDevice );
	cudaMemcpy( xz_exit_angle_d,				&xz_exit_angle_vector[start_position],		size_floats,	cudaMemcpyHostToDevice );
	
	cudaMemcpy( WEPL_d,					&WEPL_vector[start_position],			size_floats,	cudaMemcpyHostToDevice );
	cudaMemcpy( first_MLP_voxel_d,				&first_MLP_voxel_vector[start_position],	size_ints,	cudaMemcpyHostToDevice );
	
	  
}
__global__ void init_image_GPU(configurations* parameters, float* x_update, unsigned int* intersection_counts) {

	 int row = blockIdx.x, column = blockIdx.y, slice = threadIdx.x;
	 int voxel = column + row * parameters->COLUMNS + slice * parameters->COLUMNS * parameters->ROWS;
	 
	/*int voxel = 0;
	int column_start = VOXELS_PER_THREAD * blockIdx.x;
	int row = blockIdx.y;
	int slice = threadIdx.x;
	int voxel_start = column_start  + row * COLUMNS + slice * ROWS * COLUMNS;
	 */
	for( int shift = 0; shift < VOXELS_PER_THREAD; shift++ ) {
	  
		//voxel = (column_start  + shift) +  row * COLUMNS + slice * ROWS * COLUMNS;
		
		if( voxel < parameters->NUM_VOXELS ) {
	   
			x_update[voxel] = 0.0;
			intersection_counts[voxel] = 0;
			
		}
	 
	 }
  
}
void reconstruction_cuts (const int start_position, const int num_histories) 
{
	unsigned int size_floats		= sizeof(float) * num_histories;
	unsigned int size_ints			= sizeof(int) * num_histories;
	unsigned int size_bool			= sizeof(bool) * num_histories;
	
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
	cudaMalloc( (void**) &intersected_hull_d, 		size_bool );
	cudaMalloc( (void**) &first_MLP_voxel_d, 		size_ints );
 
	intersected_hull_h = (bool*)calloc( num_histories, sizeof(bool) );
	cudaMemcpy( intersected_hull_d, 	intersected_hull_h, 					size_bool,		cudaMemcpyHostToDevice );  
	cudaMemcpy( x_entry_d,				&x_entry_vector[start_position],		size_floats,	cudaMemcpyHostToDevice );
	cudaMemcpy( y_entry_d,				&y_entry_vector[start_position],		size_floats,	cudaMemcpyHostToDevice );
	cudaMemcpy( z_entry_d,				&z_entry_vector[start_position],		size_floats,	cudaMemcpyHostToDevice );
	cudaMemcpy( x_exit_d,				&x_exit_vector[start_position],			size_floats,	cudaMemcpyHostToDevice );
	cudaMemcpy( y_exit_d,				&y_exit_vector[start_position],			size_floats,	cudaMemcpyHostToDevice );
	cudaMemcpy( z_exit_d,				&z_exit_vector[start_position],			size_floats,	cudaMemcpyHostToDevice );
	cudaMemcpy( xy_entry_angle_d,		&xy_entry_angle_vector[start_position],	size_floats,	cudaMemcpyHostToDevice );
	cudaMemcpy( xz_entry_angle_d,		&xz_entry_angle_vector[start_position],	size_floats,	cudaMemcpyHostToDevice );
	cudaMemcpy( xy_exit_angle_d,		&xy_exit_angle_vector[start_position],	size_floats,	cudaMemcpyHostToDevice );
	cudaMemcpy( xz_exit_angle_d,		&xz_exit_angle_vector[start_position],	size_floats,	cudaMemcpyHostToDevice );
		
	puts("Collecting MLP endpoints...");
	
	dim3 dimBlock(THREADS_PER_BLOCK);
	dim3 dimGrid( static_cast<int>( (num_histories - 1 + THREADS_PER_BLOCK ) / THREADS_PER_BLOCK  ) );  
	collect_MLP_endpoints_GPU<<< dimGrid, dimBlock >>>
	( 
		parameters_d, intersected_hull_d, first_MLP_voxel_d, hull_d, 
		x_entry_d, y_entry_d, z_entry_d, xy_entry_angle_d, xz_entry_angle_d, 
		x_exit_d, y_exit_d, z_exit_d, xy_exit_angle_d, xz_exit_angle_d, 0, 	num_histories
	);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) 
		printf("Error: %s\n", cudaGetErrorString(err));	  
	puts("Memory device to host...");  

	//Transfer from device to host
	cudaMemcpy(intersected_hull_h,						intersected_hull_d, size_bool,		cudaMemcpyDeviceToHost);
	cudaMemcpy(&first_MLP_voxel_vector[start_position], first_MLP_voxel_d,	size_ints,		cudaMemcpyDeviceToHost);
	cudaMemcpy(&x_entry_vector[start_position],			x_entry_d,			size_floats,	cudaMemcpyDeviceToHost);
	cudaMemcpy(&y_entry_vector[start_position],			y_entry_d,			size_floats,	cudaMemcpyDeviceToHost);
	cudaMemcpy(&z_entry_vector[start_position],			z_entry_d,			size_floats,	cudaMemcpyDeviceToHost);
	cudaMemcpy(&x_exit_vector[start_position],			x_exit_d,			size_floats,	cudaMemcpyDeviceToHost);
	cudaMemcpy(&y_exit_vector[start_position],			y_exit_d,			size_floats,	cudaMemcpyDeviceToHost);
	cudaMemcpy(&z_exit_vector[start_position],			z_exit_d,			size_floats,	cudaMemcpyDeviceToHost);
	
	//first_MLP_voxel_vector.resize( num_histories );
	
	puts("Free GPU memory...");
	//Free GPU memory
	cudaFree(intersected_hull_d);
	cudaFree(first_MLP_voxel_d);
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
		
	puts("Vectors manupilating...");
	
	for( int i = 0; i < num_histories; i++ ) 
	{	    
		if( intersected_hull_h[i] ) 
		{
			first_MLP_voxel_vector.at(reconstruction_histories) = first_MLP_voxel_vector.at( i + start_position );
			//printf("%d\n",first_MLP_voxel_vector[i] );
			bin_number_vector[reconstruction_histories] = bin_number_vector[ i + start_position ];
			gantry_angle_vector[reconstruction_histories] = gantry_angle_vector[ i + start_position ];
			WEPL_vector[reconstruction_histories] = WEPL_vector[ i + start_position ];
			x_entry_vector[reconstruction_histories] = x_entry_vector[ i + start_position ];
			y_entry_vector[reconstruction_histories] = y_entry_vector[ i + start_position ];
			z_entry_vector[reconstruction_histories] = z_entry_vector[ i + start_position ];
			x_exit_vector[reconstruction_histories] = x_exit_vector[ i + start_position ];
			y_exit_vector[reconstruction_histories] = y_exit_vector[ i + start_position ];
			z_exit_vector[reconstruction_histories] = z_exit_vector[ i + start_position ];
			xy_entry_angle_vector[reconstruction_histories] = xy_entry_angle_vector[ i + start_position ];
			xz_entry_angle_vector[reconstruction_histories] = xz_entry_angle_vector[ i + start_position ];
			xy_exit_angle_vector[reconstruction_histories] = xy_exit_angle_vector[ i + start_position ];
			xz_exit_angle_vector[reconstruction_histories] = xz_exit_angle_vector[ i + start_position ];
			reconstruction_histories++;
		}
	}
	free(intersected_hull_h);
}
void reconstruction_cuts2 (const int start_position, const int num_histories) 
{
	unsigned int size_floats		= sizeof(float) * num_histories;
	unsigned int size_ints			= sizeof(int) * num_histories;
	unsigned int size_bool			= sizeof(bool) * num_histories;
	
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
	//cudaMalloc( (void**) &intersected_hull_d, 		size_bool );
	cudaMalloc( (void**) &first_MLP_voxel_d, 		size_ints );
 
	//intersected_hull_h = (bool*)calloc( num_histories, sizeof(bool) );
	//cudaMemcpy( intersected_hull_d, 	intersected_hull_h, 					size_bool,		cudaMemcpyHostToDevice );  
	first_MLP_voxel_h = (unsigned int*)calloc( num_histories, sizeof(unsigned int) );
	
	cudaMemcpy( first_MLP_voxel_d, 		first_MLP_voxel_h, 						size_ints,		cudaMemcpyHostToDevice );  
	cudaMemcpy( x_entry_d,				&x_entry_vector[start_position],		size_floats,	cudaMemcpyHostToDevice );
	cudaMemcpy( y_entry_d,				&y_entry_vector[start_position],		size_floats,	cudaMemcpyHostToDevice );
	cudaMemcpy( z_entry_d,				&z_entry_vector[start_position],		size_floats,	cudaMemcpyHostToDevice );
	cudaMemcpy( x_exit_d,				&x_exit_vector[start_position],			size_floats,	cudaMemcpyHostToDevice );
	cudaMemcpy( y_exit_d,				&y_exit_vector[start_position],			size_floats,	cudaMemcpyHostToDevice );
	cudaMemcpy( z_exit_d,				&z_exit_vector[start_position],			size_floats,	cudaMemcpyHostToDevice );
	cudaMemcpy( xy_entry_angle_d,		&xy_entry_angle_vector[start_position],	size_floats,	cudaMemcpyHostToDevice );
	cudaMemcpy( xz_entry_angle_d,		&xz_entry_angle_vector[start_position],	size_floats,	cudaMemcpyHostToDevice );
	cudaMemcpy( xy_exit_angle_d,		&xy_exit_angle_vector[start_position],	size_floats,	cudaMemcpyHostToDevice );
	cudaMemcpy( xz_exit_angle_d,		&xz_exit_angle_vector[start_position],	size_floats,	cudaMemcpyHostToDevice );
		
	puts("Collecting MLP endpoints...");

	dim3 dimBlock(THREADS_PER_BLOCK);
	dim3 dimGrid( static_cast<int>( (num_histories - 1 + THREADS_PER_BLOCK ) / THREADS_PER_BLOCK  ) );  
	collect_MLP_endpoints_GPU<<< dimGrid, dimBlock >>>
	( 
		parameters_d, intersected_hull_d, first_MLP_voxel_d, hull_d, 
		x_entry_d, y_entry_d, z_entry_d, xy_entry_angle_d, xz_entry_angle_d, 
		x_exit_d, y_exit_d, z_exit_d, xy_exit_angle_d, xz_exit_angle_d, 0, 	num_histories
	);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) 
		printf("Error: %s\n", cudaGetErrorString(err));	  
	puts("Memory device to host...");  

	//Transfer from device to host
	//cudaMemcpy(intersected_hull_h,						intersected_hull_d, size_bool,		cudaMemcpyDeviceToHost);
	//cudaMemcpy(&first_MLP_voxel_vector[start_position], first_MLP_voxel_d,	size_ints,		cudaMemcpyDeviceToHost);
	cudaMemcpy(first_MLP_voxel_h,						first_MLP_voxel_d,	size_ints,		cudaMemcpyDeviceToHost);
	cudaMemcpy(&x_entry_vector[start_position],			x_entry_d,			size_floats,	cudaMemcpyDeviceToHost);
	cudaMemcpy(&y_entry_vector[start_position],			y_entry_d,			size_floats,	cudaMemcpyDeviceToHost);
	cudaMemcpy(&z_entry_vector[start_position],			z_entry_d,			size_floats,	cudaMemcpyDeviceToHost);
	cudaMemcpy(&x_exit_vector[start_position],			x_exit_d,			size_floats,	cudaMemcpyDeviceToHost);
	cudaMemcpy(&y_exit_vector[start_position],			y_exit_d,			size_floats,	cudaMemcpyDeviceToHost);
	cudaMemcpy(&z_exit_vector[start_position],			z_exit_d,			size_floats,	cudaMemcpyDeviceToHost);
	
	//first_MLP_voxel_vector.resize( num_histories );
	
	puts("Free GPU memory...");
	//Free GPU memory
	//cudaFree(intersected_hull_d);
	cudaFree(first_MLP_voxel_d);
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
		
	puts("Vectors manupilating...");
	
	for( int i = 0; i < num_histories; i++ ) 
	{	    
		//if( intersected_hull_h[i] ) 
		if( first_MLP_voxel_h[i] != parameters.NUM_VOXELS ) 
		{
			//first_MLP_voxel_vector.at(reconstruction_histories) = first_MLP_voxel_vector.at( i + start_position );
			//printf("%d\n",first_MLP_voxel_vector[i] );
			first_MLP_voxel_vector[reconstruction_histories] = first_MLP_voxel_h[i];
			bin_number_vector[reconstruction_histories] = bin_number_vector[ i + start_position ];
			gantry_angle_vector[reconstruction_histories] = gantry_angle_vector[ i + start_position ];
			WEPL_vector[reconstruction_histories] = WEPL_vector[ i + start_position ];
			x_entry_vector[reconstruction_histories] = x_entry_vector[ i + start_position ];
			y_entry_vector[reconstruction_histories] = y_entry_vector[ i + start_position ];
			z_entry_vector[reconstruction_histories] = z_entry_vector[ i + start_position ];
			x_exit_vector[reconstruction_histories] = x_exit_vector[ i + start_position ];
			y_exit_vector[reconstruction_histories] = y_exit_vector[ i + start_position ];
			z_exit_vector[reconstruction_histories] = z_exit_vector[ i + start_position ];
			xy_entry_angle_vector[reconstruction_histories] = xy_entry_angle_vector[ i + start_position ];
			xz_entry_angle_vector[reconstruction_histories] = xz_entry_angle_vector[ i + start_position ];
			xy_exit_angle_vector[reconstruction_histories] = xy_exit_angle_vector[ i + start_position ];
			xz_exit_angle_vector[reconstruction_histories] = xz_exit_angle_vector[ i + start_position ];
			reconstruction_histories++;
		}
	}
	free(first_MLP_voxel_h);
	//free(intersected_hull_h);
}
void image_reconstruction_GPU() 
{	  
	char iterate_filename[256];
	clock_t begin1, end1, begin2, end2;
	float time_spent1, time_spent2;
	int i = 0; 
	reconstruction_histories = 0;
	  
	//sprintf(iterate_filename, "%s%d", "x_", 0 );
	sprintf(iterate_filename, "%s_%d", X_2_USE_PATH_BASE, 0, X_FILE_EXTENSION );
	if( WRITE_X_KI ) 
	{
		//transfer_device_to_host();		
		array_2_disk(iterate_filename, PREPROCESSING_DIR, X_WRITE_MODE, x_h, parameters.COLUMNS, parameters.ROWS, parameters.SLICES, parameters.NUM_VOXELS, true ); 
	}
	printf("%d\n",post_cut_histories);  
  
	//post_cut_histories = 50000000;	
	//num_histories = 60000000;
	  	 
	//int num_blocks = static_cast<int>( (post_cut_histories - 1 + THREADS_PER_BLOCK )/ THREADS_PER_BLOCK ) ;
	//int num_blocks = static_cast<int>((BLOCK_SIZE + HISTORIES_PER_BLOCK - 1)/HISTORIES_PER_BLOCK);
	//int num_blocks = static_cast<int>((BLOCK_SIZE - 1 + THREADS_PER_BLOCK) / THREADS_PER_BLOCK);
	  
	first_MLP_voxel_vector.resize( post_cut_histories );  
	cudaFree(x_d);
	//cudaFree(x_update_d);
	//cudaFree(intersection_counts_d);
	cudaFree(hull_d);
	  
	cudaMalloc( (void**) &x_d, 					parameters.NUM_VOXELS *sizeof(float));
	cudaMalloc( (void**) &x_update_d, 			parameters.NUM_VOXELS *sizeof(float));
	cudaMalloc( (void**) &intersection_counts_d, 	parameters.NUM_VOXELS *sizeof(unsigned int));
	cudaMalloc( (void**) &hull_d, 				parameters.NUM_VOXELS *sizeof(bool));
	  
	cudaMemcpy( hull_d, hull_h, parameters.NUM_VOXELS *sizeof(bool),cudaMemcpyHostToDevice );
	cudaMemcpy( x_d, x_h, parameters.NUM_VOXELS *sizeof(float),cudaMemcpyHostToDevice );
	
	puts("Starting collecting MLP endpoints...");
	int remaining_histories = post_cut_histories, start_position = 0, histories_to_process = 0;
	  
	printf("start remaining histories: %d\n", remaining_histories);
	  
	while( remaining_histories > 0 )
	{
		printf("*******\n");
	 
		if( remaining_histories > parameters.MAX_GPU_HISTORIES )
			histories_to_process = parameters.MAX_GPU_HISTORIES;
		else
			histories_to_process = remaining_histories;			
		reconstruction_cuts( start_position, histories_to_process );		
		remaining_histories -= parameters.MAX_GPU_HISTORIES;
		start_position		+= parameters.MAX_GPU_HISTORIES;
	}
		
	puts("MLP endpoints finished cuts complete.");
	printf("RECON hist %d\n", reconstruction_histories);

	first_MLP_voxel_vector.resize( reconstruction_histories );
	resize_vectors(reconstruction_histories);
	
	transfer_intermediate_results_host_to_device( reconstruction_histories );
	
	cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess) 
			printf("Error: %s\n", cudaGetErrorString(err));
	  
	i=0;
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	dim3 dimBlock( parameters.SLICES );
	dim3 dimGrid( parameters.COLUMNS, parameters.ROWS );
	//dim3 dimGrid( COLUMNS/VOXELS_PER_THREAD, ROWS );
	
	init_image_GPU<<< dimGrid, dimBlock >>>(parameters_d, x_update_d, intersection_counts_d);
	err = cudaGetLastError();
	if (err != cudaSuccess) 
		printf("Error: %s\n", cudaGetErrorString(err));
	  
	 //int remaining_histories = BLOCK_SIZE % THREADS_PER_BLOCK;
	  
	 //history_sequence = (ULL*)calloc( reconstruction_histories, sizeof(ULL));
	 //generate_history_sequence(post_cut_histories, PRIME_OFFSET, history_sequence );
	int num_blocks;
	//int num_blocks = static_cast<int>((DROP_BLOCK_SIZE + HISTORIES_PER_BLOCK - 1)/HISTORIES_PER_BLOCK);
	
	//begin = clock();
	for(int iteration = 1; iteration <= parameters.ITERATIONS ; ++iteration) {
	    
		float total_time1=0.0, total_time2 = 0.0;
		i=0;
		printf("Performing iteration %u of image reconstruction\n", iteration);
		begin1 = clock();
		//transfer_intermediate_results_host_to_device( reconstruction_histories );
		
		remaining_histories = reconstruction_histories;
		//dim3 dimBlock( SLICES );
		//dim3 dimGrid( COLUMNS/VOXELS_PER_THREAD, ROWS );
		while( remaining_histories > 0 )
		{
			if( remaining_histories > parameters.BLOCK_SIZE )
				histories_to_process = parameters.BLOCK_SIZE;
			else
				histories_to_process = remaining_histories;	
			
			num_blocks = static_cast<int>( (histories_to_process - 1 + HISTORIES_PER_BLOCK) / HISTORIES_PER_BLOCK);  			
			block_update_GPU<<< num_blocks, HISTORIES_PER_BLOCK >>>(parameters_d, x_d, x_entry_d, y_entry_d, z_entry_d, xy_entry_angle_d, xz_entry_angle_d, x_exit_d, y_exit_d, z_exit_d,  xy_exit_angle_d, 
					 xz_exit_angle_d, WEPL_d, first_MLP_voxel_d, x_update_d, intersection_counts_d, i, reconstruction_histories, parameters.LAMBDA);	
			
			err = cudaGetLastError();
			if (err != cudaSuccess) 
				printf("Error: %s\n", cudaGetErrorString(err));

			cudaEventRecord(start, 0);
			image_update_GPU<<< dimGrid, dimBlock >>>(parameters_d, x_d, x_update_d, intersection_counts_d);
			cudaEventRecord(stop, 0);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&time, start, stop);
			
			//time_spent2 = (float)(end2 - begin2) / CLOCKS_PER_SEC;
			total_time2+=time;
			remaining_histories -= parameters.BLOCK_SIZE;
		}
		//while ( i < reconstruction_histories ) 
		//{	

		//	//begin1 = clock();
		//  
		//	block_update_GPU<<< num_blocks, HISTORIES_PER_BLOCK >>>(parameters_d, x_d, x_entry_d, y_entry_d, z_entry_d, xy_entry_angle_d, xz_entry_angle_d, x_exit_d, y_exit_d, z_exit_d,  xy_exit_angle_d, 
		//			 xz_exit_angle_d, WEPL_d, first_MLP_voxel_d, x_update_d, intersection_counts_d, i, reconstruction_histories, parameters.LAMBDA);					  
		//	//end1 = clock();
		//	//time_spent1 = (float)(end1 - begin1) / CLOCKS_PER_SEC;
		//	//total_time1+=time_spent1;
		//	//printf( "time spent on block_update_GPU: %lf\n", time_spent );
		//	
		//	err = cudaGetLastError();
		//	if (err != cudaSuccess) 
		//		printf("Error: %s\n", cudaGetErrorString(err));

		//	cudaEventRecord(start, 0);
		//	image_update_GPU<<< dimGrid, dimBlock >>>(parameters_d, x_d, x_update_d, intersection_counts_d);
		//	cudaEventRecord(stop, 0);
		//	cudaEventSynchronize(stop);
		//	cudaEventElapsedTime(&time, start, stop);
		//	
		//	//time_spent2 = (float)(end2 - begin2) / CLOCKS_PER_SEC;
		//	total_time2+=time;
		//	//printf( "time spent on image_update_GPU: %lf\n", time_spent );
		//  
		//	i+=num_blocks*HISTORIES_PER_BLOCK*HISTORIES_PER_THREAD;
		//}
		end1 = clock();
		time_spent1 = (float)(end1 - begin1) / CLOCKS_PER_SEC;
	  
		//printf( "time spent on block_update: %lf ,and image_update: %lf milliseconds\n", 0.0f, total_time2 );
		printf( "time spent on iteration %d : %lf seconds\n", iteration, time_spent1 );
		
		//sprintf(iterate_filename, "%s%d", "x_", iteration );
		
		if( WRITE_X_KI ) 
		{
			transfer_image_device_to_host();
			//array_2_disk(iterate_filename, OUTPUT_DIRECTORY, OUTPUT_FOLDER, x_h, parameters.COLUMNS, parameters.ROWS, parameters.SLICES, parameters.NUM_VOXELS, true ); 
			sprintf(iterate_filename, "%s_%d", X_2_USE_PATH_BASE, iteration, X_FILE_EXTENSION );		
			array_2_disk(iterate_filename, RECONSTRUCTION_DIR, X_WRITE_MODE, x_h, parameters.COLUMNS, parameters.ROWS, parameters.SLICES, parameters.NUM_VOXELS, true );
		}  
	}
}
void image_reconstruction_GPU_tabulated() 
{	  
	char iterate_filename[256];
	clock_t begin1, end1, begin2, end2;
	float time_spent1, time_spent2;
	reconstruction_histories = 0;
	//------------------------------------------------------------------------------------------------------------------------//
	//------------------------------------------ Export preprocessing images  ------------------------------------------------//
	//------------------------------------------------------------------------------------------------------------------------//	  
	export_hull();
	export_x_0();
	//------------------------------------------------------------------------------------------------------------------------//
	//------------------------- Clear global GPU memory and allocate for reconstruction image data  --------------------------//
	//------------------------------------------------------------------------------------------------------------------------//	  
	cudaFree(x_d);
	cudaFree(hull_d);
	  
	cudaMalloc( (void**) &x_d, 					parameters.NUM_VOXELS *sizeof(float));
	cudaMalloc( (void**) &x_update_d, 			parameters.NUM_VOXELS *sizeof(float));
	cudaMalloc( (void**) &intersection_counts_d, 	parameters.NUM_VOXELS *sizeof(unsigned int));
	cudaMalloc( (void**) &hull_d, 				parameters.NUM_VOXELS *sizeof(bool));
	  
	cudaMemcpy( hull_d, hull_h, parameters.NUM_VOXELS *sizeof(bool),cudaMemcpyHostToDevice );
	cudaMemcpy( x_d, x_h, parameters.NUM_VOXELS *sizeof(float),cudaMemcpyHostToDevice );
	//------------------------------------------------------------------------------------------------------------------------//
	//----------------------------- Collect MLP endpoints and remove histories missing the hull  -----------------------------//
	//------------------------------------------------------------------------------------------------------------------------//	
	printf("Starting collecting MLP endpoints for %d histories...\n", post_cut_histories);
	int remaining_histories = post_cut_histories, start_position = 0, histories_to_process = 0;
	first_MLP_voxel_vector.resize( post_cut_histories );  

	while( remaining_histories > 0 )
	{
		if( remaining_histories > parameters.MAX_GPU_HISTORIES )
			histories_to_process = parameters.MAX_GPU_HISTORIES;
		else
			histories_to_process = remaining_histories;			
		reconstruction_cuts( start_position, histories_to_process );		
		remaining_histories -= parameters.MAX_GPU_HISTORIES;
		start_position		+= parameters.MAX_GPU_HISTORIES;
	}
		
	printf("MLP endpoints calculated and reconstruction cuts complete.  Total reconstruction histories = %d\n", reconstruction_histories );

	first_MLP_voxel_vector.resize( reconstruction_histories );
	resize_vectors(reconstruction_histories);
	//------------------------------------------------------------------------------------------------------------------------//
	//-------------------------------------- Generate and transfer MLP lookup tables to GPU ----------------------------------//
	//------------------------------------------------------------------------------------------------------------------------//
	generate_trig_tables();
	//import_trig_tables();
	generate_scattering_coefficient_table();
	//import_scattering_coefficient_table();
	generate_polynomial_tables();
	//import_polynomial_tables();
	tables_2_GPU();
	//------------------------------------------------------------------------------------------------------------------------//
	//------------------------------------------------ Begin image reconstruction --------------------------------------------//
	//------------------------------------------------------------------------------------------------------------------------//
	transfer_intermediate_results_host_to_device( reconstruction_histories );
	
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) 
		printf("Error: %s\n", cudaGetErrorString(err));
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	dim3 dimBlock( parameters.SLICES );
	dim3 dimGrid( parameters.COLUMNS, parameters.ROWS );
	//dim3 dimGrid( COLUMNS/VOXELS_PER_THREAD, ROWS );
	
	init_image_GPU<<< dimGrid, dimBlock >>>(parameters_d, x_update_d, intersection_counts_d);
	err = cudaGetLastError();
	if (err != cudaSuccess) 
		printf("Error: %s\n", cudaGetErrorString(err));
  
	 //history_sequence = (ULL*)calloc( reconstruction_histories, sizeof(ULL));
	 //generate_history_sequence(post_cut_histories, PRIME_OFFSET, history_sequence );
	
	int num_blocks, i;
	//begin = clock();
	for(int iteration = 1; iteration <= parameters.ITERATIONS ; ++iteration) 
	{
	    i = 0;
		float total_time1 = 0.0, total_time2 = 0.0;
		printf("Performing iteration %u of image reconstruction\n", iteration);
		begin1 = clock();
		//transfer_intermediate_results_host_to_device( reconstruction_histories );
		
		remaining_histories = reconstruction_histories;
		//dim3 dimBlock( SLICES );
		//dim3 dimGrid( COLUMNS/VOXELS_PER_THREAD, ROWS );
		while( remaining_histories > 0 )
		{
			if( remaining_histories > parameters.BLOCK_SIZE )
				histories_to_process = parameters.BLOCK_SIZE;
			else
				histories_to_process = remaining_histories;	
			
			num_blocks = static_cast<int>( (histories_to_process - 1 + HISTORIES_PER_BLOCK) / HISTORIES_PER_BLOCK);  			
			//begin1 = clock();
			block_update_GPU_tabulated<<< num_blocks, HISTORIES_PER_BLOCK >>>
			(
				parameters_d, x_d, x_entry_d, y_entry_d, z_entry_d, xy_entry_angle_d, xz_entry_angle_d, x_exit_d, y_exit_d, z_exit_d,  xy_exit_angle_d, 
				xz_exit_angle_d, WEPL_d, first_MLP_voxel_d, x_update_d, intersection_counts_d, i, reconstruction_histories, parameters.LAMBDA,
				sin_table_d, cos_table_d, scattering_table_d, poly_1_2_d, poly_2_3_d, poly_3_4_d, poly_2_6_d, poly_3_12_d
			);	
			//end1 = clock();
			//time_spent1 = (float)(end1 - begin1) / CLOCKS_PER_SEC;
			//total_time1+=time_spent1;
			//printf( "time spent on block_update_GPU: %lf\n", time_spent );
			
			err = cudaGetLastError();
			if (err != cudaSuccess) 
				printf("Error: %s\n", cudaGetErrorString(err));

			cudaEventRecord(start, 0);
			image_update_GPU<<< dimGrid, dimBlock >>>(parameters_d, x_d, x_update_d, intersection_counts_d);
			cudaEventRecord(stop, 0);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&time, start, stop);
			
			//time_spent2 = (float)(end2 - begin2) / CLOCKS_PER_SEC;
			total_time2 += time;
			//printf( "time spent on image_update_GPU: %lf\n", time_spent );
			remaining_histories -= parameters.BLOCK_SIZE;
			i += parameters.BLOCK_SIZE;
		}
		end1 = clock();
		time_spent1 = (float)(end1 - begin1) / CLOCKS_PER_SEC;
	  
		//printf( "time spent on block_update: %lf ,and image_update: %lf milliseconds\n", 0.0f, total_time2 );
		printf( "time spent on iteration %d : %lf seconds\n", iteration, time_spent1 );

		if( WRITE_X_KI ) 
		{
			transfer_image_device_to_host();
			//array_2_disk(iterate_filename, OUTPUT_DIRECTORY, OUTPUT_FOLDER, x_h, parameters.COLUMNS, parameters.ROWS, parameters.SLICES, parameters.NUM_VOXELS, true ); 
			sprintf(iterate_filename, "%s_%d", X_2_USE_PATH_BASE, iteration, X_FILE_EXTENSION );		
			array_2_disk(iterate_filename, RECONSTRUCTION_DIR, X_WRITE_MODE, x_h, parameters.COLUMNS, parameters.ROWS, parameters.SLICES, parameters.NUM_VOXELS, true );
		}  
	}
}
/***********************************************************************************************************************************************************************************************************************/
/************************************************************************************************ Preprocessing Data I/O ***********************************************************************************************/
/***********************************************************************************************************************************************************************************************************************/
void export_hull()
{
	puts("Writing image reconstruction hull to disk...");
	hull_file = fopen(HULL_2_USE_PATH, "wb");
	//fwrite( parameters.ROWS, sizeof(uint), 1, hull_file );
	//fwrite( parameters.COLUMNS, sizeof(uint), 1, hull_file );
	//fwrite( parameters.SLICES, sizeof(uint), 1, hull_file );
	fwrite( hull_h, sizeof(bool), parameters.NUM_VOXELS, hull_file );
	fclose(hull_file);
	puts("Finished writing image reconstruction hull to disk.");
}
void export_x_0()
{
	puts("Writing initial iterate to disk...");
	x_0_file = fopen(X_0_2_USE_PATH, "wb");
	//fwrite( parameters.ROWS, sizeof(uint), 1, x_0_file );
	//fwrite( parameters.COLUMNS, sizeof(uint), 1, x_0_file );
	//fwrite( parameters.SLICES, sizeof(uint), 1, x_0_file );
	fwrite( x_h, sizeof(float), parameters.NUM_VOXELS, x_0_file );
	fclose(x_0_file);
	puts("Finished writing initial iterate to disk.");
}
void export_WEPL()
{
	puts("Writing WEPL data to disk...");
	WEPL_file = fopen(WEPL_PATH, "wb");
	fwrite( &reconstruction_histories, sizeof(unsigned int), 1, WEPL_file );
	fwrite( &WEPL_vector[0], sizeof(float), WEPL_vector.size(), WEPL_file );
	fclose(WEPL_file);
	//vector_2_disk(WEPL_BASENAME, PREPROCESSING_DIR, WEPL_WRITE_MODE, WEPL_vector, WEPL_vector.size(), 1, 1, true );
	print_section_exit("Finished writing WEPL data to disk.", "====>");
}
void export_histories()
{
	puts("Writing MLP endpoints to disk...");
	histories_file = fopen(HISTORIES_PATH, "wb");
	fwrite( &reconstruction_histories, sizeof(unsigned int), 1, histories_file );
	fwrite( &voxel_x_vector[0], sizeof(int), voxel_x_vector.size(), histories_file );
	fwrite( &voxel_y_vector[0], sizeof(int), voxel_y_vector.size(), histories_file);
	fwrite( &voxel_z_vector[0], sizeof(int), voxel_z_vector.size(), histories_file );	
	fwrite( &x_entry_vector[0], sizeof(float), x_entry_vector.size(), histories_file);
	fwrite( &y_entry_vector[0], sizeof(float), y_entry_vector.size(), histories_file);
	fwrite( &z_entry_vector[0], sizeof(float), z_entry_vector.size(), histories_file);
	fwrite( &x_exit_vector[0], sizeof(float), x_exit_vector.size(), histories_file );
	fwrite( &y_exit_vector[0], sizeof(float), y_exit_vector.size(), histories_file );
	fwrite( &z_exit_vector[0], sizeof(float), z_exit_vector.size(), histories_file );
	fwrite( &xy_entry_angle_vector[0], sizeof(float), xy_entry_angle_vector.size(), histories_file );
	fwrite( &xz_entry_angle_vector[0], sizeof(float), xz_entry_angle_vector.size(), histories_file );
	fwrite( &xy_exit_angle_vector[0], sizeof(float), xy_exit_angle_vector.size(), histories_file );
	fwrite( &xz_exit_angle_vector[0], sizeof(float), xz_exit_angle_vector.size(), histories_file );
	fwrite( &bin_number_vector[0], sizeof(int), bin_number_vector.size(), histories_file );
	fwrite( &gantry_angle_vector[0], sizeof(int), gantry_angle_vector.size(), histories_file );
	fclose(histories_file);
	print_section_exit("Finished writing histories to disk.", "====>");
}
void export_voxels_per_path()
{
	puts("Writing # of intersections per MLP path to disk...");
	voxels_per_path_file = fopen(VOXELS_PER_PATH_PATH, "wb");
	fwrite( &reconstruction_histories, sizeof(uint), 1, voxels_per_path_file );
	fwrite( &voxels_per_path_vector[0], sizeof(uint), voxels_per_path_vector.size(), voxels_per_path_file );
	fclose(voxels_per_path_file);
	print_section_exit("Finished writing # of intersections per MLP path to disk.", "====>");
}
void export_avg_chord_lengths()
{
	puts("Writing # of intersections per MLP path to disk...");
	avg_chord_lengths_file = fopen(AVG_CHORDS_PATH, "wb");
	fwrite( &reconstruction_histories, sizeof(uint), 1, avg_chord_lengths_file );
	fwrite( &avg_chord_length_vector[0], sizeof(float), avg_chord_length_vector.size(), avg_chord_lengths_file );
	fclose(avg_chord_lengths_file);
	print_section_exit("Finished writing # of intersections per MLP path to disk.", "====>");
}
void accumulate_and_export_preprocessed_data()
{
	unsigned int start_history = 0, end_history = reconstruction_histories;
	unsigned int voxels_in_MLP;
	unsigned int* MLP = (unsigned int*)calloc( MAX_INTERSECTIONS, sizeof(unsigned int));
	float* chord_lengths = (float*)calloc( 1, sizeof(float));

	float avg_xy_angle, avg_xz_angle, effective_chord_length;
	ULL i;	
	puts("Precalculating MLP paths and writing them to disk...");
	if( !file_exists(MLP_PATH) || parameters.PREPROCESS_OVERWRITE_OK )
	{
		MLP_file  = fopen( MLP_PATH,  "wb" );
		WEPL_file = fopen( WEPL_PATH, "wb" );
		fprintf( MLP_file,	"%u\n", reconstruction_histories );	
		fprintf( WEPL_file, "%u\n", reconstruction_histories );
		for( unsigned int n = start_history; n < end_history; n++ )
		{		
			i = history_sequence[n];			
			voxels_in_MLP = calculate_MLP( MLP, chord_lengths, x_entry_vector[i], y_entry_vector[i], z_entry_vector[i], x_exit_vector[i], y_exit_vector[i], z_exit_vector[i], xy_entry_angle_vector[i], xz_entry_angle_vector[i], xy_exit_angle_vector[i], xz_exit_angle_vector[i], voxel_x_vector[i], voxel_y_vector[i], voxel_z_vector[i]);
			
			avg_xy_angle = atanf( (y_exit_vector[i] - y_entry_vector[i] ) / ( x_exit_vector[i] - x_entry_vector[i] ) );
			avg_xz_angle = atanf( ( z_exit_vector[i] - z_entry_vector[i] ) / ( x_exit_vector[i] - x_entry_vector[i] ) );
			effective_chord_length = EffectiveChordLength( avg_xy_angle, avg_xz_angle );
			
			voxels_per_path_vector.push_back( voxels_in_MLP );
			avg_chord_length_vector.push_back( effective_chord_length );
			
			fwrite( MLP,				sizeof(uint),	voxels_in_MLP,	MLP_file  );
			fwrite( &WEPL_vector[i],	sizeof(float),	1,				WEPL_file );
		}
		fclose(MLP_file);
		fclose(WEPL_file);	

		export_histories();
		export_voxels_per_path();
		export_avg_chord_lengths();
		export_hull();
		export_x_0();	
	}
	else
		puts("Preprocessed data files already exist and overwriting this data is not permitted according to the value of PREPROCESS_OVERWRITE_OK in the config file.");
	
	import_WEPL();
	puts("MLP path calculations complete.  These paths, their effective chord lengths, voxels per path, and WEPL measurements have been written to disk in reconstruction history order.");
}
void import_hull()
{
	puts("Importing image reconstruction hull from disk...");
	hull_file = fopen(HULL_2_USE_PATH, "rb");
	//fread( parameters.ROWS, sizeof(uint), 1, hull_file );
	//fread( parameters.COLUMNS, sizeof(uint), 1, hull_file );
	//fread( parameters.SLICES, sizeof(uint), 1, hull_file );
	//array_2_disk(HULL_2_USE_FILENAME, PREPROCESSING_DIR, HULL_WRITE_MODE, hull_h, parameters.COLUMNS, parameters.ROWS, parameters.SLICES, parameters.NUM_VOXELS, true );
	hull_h = (bool*)calloc( parameters.NUM_VOXELS, sizeof(bool) );
	fread( hull_h, sizeof(bool), parameters.NUM_VOXELS, hull_file );
	fclose(hull_file);
	puts("Finished importing image reconstruction hull from disk.");
}
void import_x_0()
{
		puts("Reading image reconstruction hull from disk...");
		x_0_file = fopen(X_0_2_USE_PATH, "rb");
		x_h = (float*)calloc( parameters.NUM_VOXELS, sizeof(float) );
		fread( x_h, sizeof(float), parameters.NUM_VOXELS, x_0_file );
		fclose(x_0_file);
		puts("Finished reading image reconstruction hull from disk.");
}
uint import_WEPL()
{
	//unsigned int histories;
	puts("Reading WEPL data from disk...");
	WEPL_file = fopen(WEPL_PATH, "rb");
	fread( &WEPL_histories, sizeof(unsigned int), 1, WEPL_file );
	WEPL_vector.resize(WEPL_histories);
	fread( &WEPL_vector[0], sizeof(float), WEPL_histories, WEPL_file );
	fclose(WEPL_file);
	//vector_2_disk(WEPL_BASENAME, PREPROCESSING_DIR, WEPL_WRITE_MODE, WEPL_vector, WEPL_vector.size(), 1, 1, true );
	print_section_exit("Finished reading WEPL data from disk.", "====>");
	return WEPL_histories;
}
uint import_histories()
{
	//unsigned int histories;
	puts("Reading MLP endpoints from disk...");
	histories_file = fopen(HISTORIES_PATH, "rb");
	fread( &hull_intersection_histories, sizeof(unsigned int), 1, histories_file );
	
	voxel_x_vector.resize(hull_intersection_histories);
	voxel_y_vector.resize(hull_intersection_histories);
	voxel_z_vector.resize(hull_intersection_histories);
	resize_vectors( hull_intersection_histories );
	shrink_vectors( hull_intersection_histories );

	fread( &voxel_x_vector[0],			sizeof(uint),	hull_intersection_histories, histories_file );
	fread( &voxel_y_vector[0],			sizeof(uint),	hull_intersection_histories, histories_file);
	fread( &voxel_z_vector[0],			sizeof(uint),	hull_intersection_histories, histories_file );	
	fread( &x_entry_vector[0],			sizeof(float),	hull_intersection_histories, histories_file);
	fread( &y_entry_vector[0],			sizeof(float),	hull_intersection_histories, histories_file);
	fread( &z_entry_vector[0],			sizeof(float),	hull_intersection_histories, histories_file);
	fread( &x_exit_vector[0],			sizeof(float),	hull_intersection_histories, histories_file );
	fread( &y_exit_vector[0],			sizeof(float),	hull_intersection_histories, histories_file );
	fread( &z_exit_vector[0],			sizeof(float),	hull_intersection_histories, histories_file );
	fread( &xy_entry_angle_vector[0],	sizeof(float),	hull_intersection_histories, histories_file );
	fread( &xz_entry_angle_vector[0],	sizeof(float),	hull_intersection_histories, histories_file );
	fread( &xy_exit_angle_vector[0],	sizeof(float),	hull_intersection_histories, histories_file );
	fread( &xz_exit_angle_vector[0],	sizeof(float),	hull_intersection_histories, histories_file );
	fread( &bin_number_vector[0],		sizeof(uint),	hull_intersection_histories, histories_file );
	fread( &gantry_angle_vector[0],		sizeof(uint),	hull_intersection_histories, histories_file );
	fclose(histories_file);
	print_section_exit("Finished reading histories from disk.", "====>");
	return hull_intersection_histories;
}
uint import_voxels_per_path()
{
	//unsigned int histories;
	puts("Reading # of intersections per MLP path to disk...");
	voxels_per_path_file = fopen(VOXELS_PER_PATH_PATH, "rb");
	fread( &voxels_per_path_histories, sizeof(unsigned int), 1, voxels_per_path_file );
	voxels_per_path_vector.resize(voxels_per_path_histories);
	fread( &voxels_per_path_vector[0], sizeof(uint), voxels_per_path_histories, voxels_per_path_file );
	fclose(voxels_per_path_file);
	print_section_exit("Finished reading # of intersections per MLP path from disk.", "====>");
	return voxels_per_path_histories;
}
uint import_avg_chord_lengths()
{
	//unsigned int histories;
	puts("Reading # of intersections per MLP path to disk...");
	avg_chord_lengths_file = fopen(AVG_CHORDS_PATH, "rb");
	fread( &avg_chords_histories, sizeof(unsigned int), 1, avg_chord_lengths_file );
	avg_chord_length_vector.resize(avg_chords_histories);
	fread( &avg_chord_length_vector[0], sizeof(float), avg_chords_histories, avg_chord_lengths_file );
	fclose(avg_chord_lengths_file);
	print_section_exit("Finished reading # of intersections per MLP path from disk.", "====>");
	return avg_chords_histories;
}
uint import_MLP()
{
	MLP_file = fopen(MLP_PATH, "rb");
	//uint histories;
	fread( &MLP_histories, sizeof(unsigned int), 1, MLP_file );
	begin_MLP_data = ftell(MLP_file);
	fseek( MLP_file, 0, SEEK_END );
	long int end_MLP_data = ftell(MLP_file);
	long int total_voxels = end_MLP_data - begin_MLP_data;
	fseek( MLP_file, begin_MLP_data, SEEK_SET );
	MLP_vector.resize(total_voxels);
	fread( &MLP_vector[0], sizeof(uint), total_voxels, MLP_file );
	fclose(MLP_file);
	return MLP_histories;
}
void import_images()
{
	// Import hull and initial iterate x_0
	import_hull();
	import_x_0();
}
bool init_import_preprocessed_data()
{
	if( !existing_data_check() )
	{
		puts("ERROR: One or more preprocessed data file(s) are missing and reconstruction cannot proceed.  Verify these files exist and are properly named and generate this data if necessary");
		return false;
	}

	// Open each preprocessed data file for subsequent reading
	MLP_file				= fopen(MLP_PATH,				"rb"	);
	voxels_per_path_file	= fopen(VOXELS_PER_PATH_PATH,	"rb"	);
	WEPL_file				= fopen(WEPL_PATH,				"rb"	);
	avg_chord_lengths_file	= fopen(AVG_CHORDS_PATH,		"rb"	);
	//histories_file			= fopen(HISTORIES_PATH,			"rb"	);
	
	// Read the # of histories whose data was written to each file so these can be verified to be the same, which they must be
	fread( &MLP_histories,					sizeof(uint),	1, MLP_file					);
	fread( &voxels_per_path_histories,		sizeof(uint),	1, voxels_per_path_file		);
	fread( &WEPL_histories,					sizeof(uint),	1, WEPL_file				);
	fread( &avg_chords_histories,			sizeof(uint),	1, avg_chord_lengths_file	);
	//fread( &hull_intersection_histories,	sizeof(unsigned int),	1, histories_file			);
	
	// Record the locations where the data begins in each file, after the history # position
	begin_MLP_data					= ftell(MLP_file);
	begin_avg_chord_lengths_data	= ftell(avg_chord_lengths_file);  
	begin_WEPL_data					= ftell(WEPL_file);  
	begin_voxels_per_path_data		= ftell(voxels_per_path_file);  
	//begin_histories_data			= ftell(histories_file);  
	
	if( ( MLP_histories != WEPL_histories ) ||	( MLP_histories != voxels_per_path_histories ) || ( MLP_histories != avg_chords_histories ) )
		return false;

	// As long as each data file contains data for the same # of histories, the global variable reconstruction_histories can be set by any of the individual history # values
	reconstruction_histories = MLP_histories;

	if( !parameters.IMPORT_DATA_ITERATIVELY )
	{
		voxels_per_path_vector.resize(reconstruction_histories);
		WEPL_vector.resize(reconstruction_histories);
		avg_chord_length_vector.resize(reconstruction_histories);
	
		fread( &voxels_per_path_vector[0],	sizeof(uint),	reconstruction_histories, voxels_per_path_file );
		fread( &WEPL_vector[0],				sizeof(float),	reconstruction_histories, WEPL_file );
		fread( &avg_chord_length_vector[0], sizeof(float),	reconstruction_histories, avg_chord_lengths_file );
	
		fclose(voxels_per_path_file);
		fclose(WEPL_file);
		fclose(avg_chord_lengths_file);

		cudaMalloc( (void**) &voxels_per_path_d,	reconstruction_histories * sizeof(uint)		);
		cudaMalloc( (void**) &WEPLs_d,				reconstruction_histories * sizeof(float)	);
		cudaMalloc( (void**) &avg_chord_lengths_d,	reconstruction_histories * sizeof(float)	);
		
		cudaMemcpy( voxels_per_path_d,		&voxels_per_path_vector[0],		reconstruction_histories * sizeof(uint),	cudaMemcpyHostToDevice );
		cudaMemcpy( WEPLs_d,				&WEPL_vector[0],				reconstruction_histories * sizeof(float),	cudaMemcpyHostToDevice );
		cudaMemcpy( avg_chord_lengths_d,	&avg_chord_length_vector[0],	reconstruction_histories * sizeof(float),	cudaMemcpyHostToDevice );	
	}
	if( !parameters.IMPORT_MLP_ITERATIVELY )
	{
		fseek( MLP_file, 0, SEEK_END );
		long int end_MLP_data = ftell(MLP_file);
		long int total_voxels = end_MLP_data - begin_MLP_data;
		fseek( MLP_file, begin_MLP_data, SEEK_SET );
		MLP_vector.resize(total_voxels);
		fread( &MLP_vector[0], sizeof(uint), total_voxels, MLP_file );
		fclose(MLP_file);

		cudaMalloc( (void**) &MLPs_d,total_voxels * sizeof(uint) );
		cudaMemcpy( MLPs_d,	&MLP_vector[0],	total_voxels * sizeof(uint),	cudaMemcpyHostToDevice );
	}
	// GPU_blocks_per_DROP_block
	
	//uint num_DROP_iterations;
	//int DROP_block_size = 0;
	uint block_histories = parameters.BLOCK_SIZE;
	//num_DROP_iterations	= static_cast<uint>( ceil( static_cast<double>(reconstruction_histories / parameters.BLOCK_SIZE) ) );
	//GPU_blocks_per_DROP_block = static_cast<uint>( ceil( static_cast<double>( parameters.BLOCK_SIZE / THREADS_PER_BLOCK) ) );
	for( int block_start_history = 0; block_start_history < reconstruction_histories; )
	{
		if( block_start_history + parameters.BLOCK_SIZE >= reconstruction_histories )					
				block_histories = reconstruction_histories - block_start_history;	
		//MLP_block_sizes[num_DROP_blocks] = 0;
		MLP_block_sizes.push_back(0);
		for( int block_history = 0; block_history < block_histories; block_history++ )
		{
			MLP_block_sizes[num_DROP_blocks] += voxels_per_path_vector[block_start_history + block_history];
			//DROP_block_size += voxels_per_path_vector[i + j];
		}
		//MLP_block_sizes.push_back(DROP_block_size);
		num_DROP_blocks++;
		block_start_history += parameters.BLOCK_SIZE;
	}
	return true;
}
// DROP iterative import WEPL, voxels_per_path, avg_chord_lengths data, iterative or full import MLP
void allocate_precalculated_block_arrays(uint block_histories )
{
	block_WEPLs_h				= (float*)	calloc( block_histories,		sizeof(float)	);
	block_avg_chord_lengths_h	= (float*)	calloc( block_histories,		sizeof(float)	);	
	block_voxels_per_path_h		= (uint*)	calloc( block_histories,		sizeof(uint)	);
	
	cudaMalloc( (void**) &block_WEPLs_d,				block_histories			* sizeof(float)	);
	cudaMalloc( (void**) &block_avg_chord_lengths_d,	block_histories			* sizeof(float)	);
	cudaMalloc( (void**) &block_voxels_per_path_d,		block_histories			* sizeof(uint)	);
}

uint vector_2_block_array( uint current_history, uint block_histories )
{
	// Copy block array data from appropriate portion of corresponding global host vectors
	//std::copy( &WEPL_vector[current_history],				&WEPL_vector[current_history] + block_histories,				block_WEPLs_h );
	//std::copy( &voxels_per_path_vector[current_history],	&voxels_per_path_vector[current_history] + block_histories,		block_voxels_per_path_h );
	//std::copy( &avg_chord_length_vector[current_history],	&avg_chord_length_vector[current_history] + block_histories,	block_avg_chord_lengths_h );
	// Copy DROP block data from host vectors to arrays on the GPU
	cudaMemcpy( block_WEPLs_d,				&WEPL_vector[current_history],				block_histories * sizeof(float),	cudaMemcpyHostToDevice);
	cudaMemcpy( block_voxels_per_path_d,	&voxels_per_path_vector[current_history],	block_histories * sizeof(uint),		cudaMemcpyHostToDevice);
	cudaMemcpy( block_avg_chord_lengths_d,	&avg_chord_length_vector[current_history],	block_histories * sizeof(float),	cudaMemcpyHostToDevice);

	uint total_voxels = 0;
	for( uint i = 0; i < block_histories; i++ )
		total_voxels += voxels_per_path_vector[current_history + i];
	
	return total_voxels;
}
uint iterative_read_2_block_array( uint current_history, uint block_histories )
{
	// Import block array data directly from disk
	fread( block_WEPLs_h,				sizeof(float), block_histories, WEPL_file				);
	fread( block_voxels_per_path_h,		sizeof(float), block_histories, voxels_per_path_file	);
	fread( block_avg_chord_lengths_h,	sizeof(float), block_histories, avg_chord_lengths_file	);
		
	// Copy DROP block data from host arrays to arrays on the GPU
	cudaMemcpy( block_WEPLs_d,				block_WEPLs_h,				block_histories * sizeof(float),	cudaMemcpyHostToDevice);
	cudaMemcpy( block_voxels_per_path_d,	block_voxels_per_path_h,	block_histories * sizeof(uint),		cudaMemcpyHostToDevice);
	cudaMemcpy( block_avg_chord_lengths_d,	block_avg_chord_lengths_h,	block_histories * sizeof(float),	cudaMemcpyHostToDevice);
		
	uint total_voxels = 0;
	for( uint i = 0; i < block_histories; i++ )
		total_voxels += block_voxels_per_path_h[i];
	
	return total_voxels;
}
void MLP_vector_2_block_array( uint& current_MLP_index, uint total_voxels )
{
	// Copy block array data from appropriate portion of corresponding global host vectors
	cudaMalloc( (void**)&block_paths_d, total_voxels * sizeof(uint) );									// Allocate GPU memory to store MLP from each history in DROP block
	//std::copy( &MLP_vector[current_MLP_voxel], &MLP_vector[current_MLP_voxel] + total_voxels,	block_paths_h );
	cudaMemcpy( block_paths_d, &MLP_vector[current_MLP_index], total_voxels * sizeof(uint), cudaMemcpyHostToDevice );		// Copy MLP paths for each history in the DROP block from host to GPU
	current_MLP_index += total_voxels;	
}
void iterative_MLP_2_block_array( uint total_voxels )
{
	cudaMalloc( (void**)&block_paths_d, total_voxels * sizeof(uint) );									// Allocate GPU memory to store MLP from each history in DROP block
	block_paths_h = (uint*) calloc( total_voxels, sizeof(uint));										// Allocate array to store MLP from each history in DROP block
	fread( block_paths_h, sizeof(uint), total_voxels, MLP_file );										// Import MLP paths for each of the histories in the DROP block
	cudaMemcpy( block_paths_d, block_paths_h, total_voxels * sizeof(uint), cudaMemcpyHostToDevice );		// Copy MLP paths for each history in the DROP block from host to GPU
}
/***********************************************************************************************************************************************************************************************************************/
/********************************************************************************************** Image Reconstruction (host) ********************************************************************************************/
/***********************************************************************************************************************************************************************************************************************/
void acquire_preprocessed_data()
{
	//EXPORT_PREPROCESSING
	//IMPORT_PREPROCESSING
	//PREPROCESS_OVERWRITE_OK
	//RECON_OVERWRITE_OK
	//MLP_IN_LOOP
	//IMPORT_DATA_ITERATIVELY

	/*************************************************************************************************************************************************************************/
	/******************** Collect preprocessed data for histories that enter hull and write it to disk in the order it will be used for reconstruction ***********************/
	/*************************************************************************************************************************************************************************/	
	if( parameters.IMPORT_PREPROCESSING )
	{
		/*************************************************************************************************************************************************************************/
		/********************* Open preprocessed data files, verify the # of histories is consistent, and leave read pointers for each at beginning of data **********************/
		/*************************************************************************************************************************************************************************/	
		bool preprocessed_data_is_valid = init_import_preprocessed_data();
		if( preprocessed_data_is_valid )
		{
			import_images();
			init_import_preprocessed_data();
			if( parameters.IMPORT_DATA_ITERATIVELY )
				allocate_precalculated_block_arrays( parameters.BLOCK_SIZE );

		}
		else
		{
			puts("ERROR: The # of histories in the preprocessed data files do not match.  Please verify the correct files are being accessed and regenerate this data if necessary");
			exit_program_if( true );
		}
		generate_history_sequence(reconstruction_histories, PRIME_OFFSET );
	}
	else
	{
		collect_MLP_endpoints();
		generate_history_sequence(reconstruction_histories, PRIME_OFFSET );
		accumulate_and_export_preprocessed_data();
	}
	
}
void image_reconstruction_import_preprocessing_blocks()
{
	/*************************************************************************************************************************************************************************/
	/************************************************************************ Perform image reconstruction *******************************************************************/
	/*************************************************************************************************************************************************************************/	
	puts("Starting image reconstruction...");
	char iterate_filename[256];	
	uint start_history = 0, block_MLP_voxels, block_histories = parameters.BLOCK_SIZE;
	allocate_precalculated_block_arrays( block_histories );
	for( unsigned int iteration = 1; iteration <= parameters.ITERATIONS; iteration++ )
	{
		printf("Performing iteration %u of image reconstruction\n", iteration);
		/*********************************************************************************************************************************************************************/
		/**************** Fill data arrays for each block, calculate updates and # of times each voxel is intersected by a path in block, and update image x *****************/
		/*********************************************************************************************************************************************************************/
		for( unsigned int current_history = start_history; current_history < reconstruction_histories;  )
		{		
			// If there are less than BLOCK_SIZE histories remaining to be processed, the last DROP block will contain only the remaining unused histories
			//i = history_sequence[n];			
			if( current_history + block_histories >= reconstruction_histories )					
				block_histories = reconstruction_histories - current_history;					
			
			// Read DROP block data from disk or copy from global host vectors, then copy this data from host arrays to arrays on GPU
			//fill_precalculated_block_arrays( current_history, block_histories );		// Fill preprocessed data host arrays for each history in the DROP block and copy these to GPU
			if( parameters.IMPORT_DATA_ITERATIVELY )
				block_MLP_voxels = iterative_read_2_block_array( current_history, block_histories );
			else
				block_MLP_voxels = vector_2_block_array( current_history, block_histories );
			
			// 
			if( parameters.IMPORT_MLP_ITERATIVELY )
				iterative_MLP_2_block_array( block_MLP_voxels );
			else
				MLP_vector_2_block_array( current_MLP_voxel, block_MLP_voxels );
			
			// If histories in block < THREADS_PER_BLOCK, single kernel can be used for calculations/update.  Otherwise, synchronization is imposed by using separate kernels 
			if( block_histories <= THREADS_PER_BLOCK )
				DROP_single_block_precalculated_MLP_blocks( block_histories );
			else
				DROP_precalculated_MLP_blocks( block_histories );
			
			current_history += parameters.BLOCK_SIZE;
			//current_MLP_voxel 
			free(block_paths_h);
			cudaFree(block_paths_d);
		}	
		// Return the read pointers of each file to the beginning of the data (after # of histories value)
		if( parameters.IMPORT_MLP_ITERATIVELY )
			fseek( MLP_file, begin_MLP_data, SEEK_SET );
		if( parameters.IMPORT_DATA_ITERATIVELY )
		{
			fseek( WEPL_file, begin_WEPL_data, SEEK_SET );
			//fseek( histories_file, begin_histories_data, SEEK_SET );
			fseek( voxels_per_path_file, begin_voxels_per_path_data, SEEK_SET );
			fseek( avg_chord_lengths_file, begin_avg_chord_lengths_data, SEEK_SET );
		}
		cudaMemcpy( x_h, x_d, SIZE_IMAGE_FLOAT, cudaMemcpyDeviceToHost);
		sprintf(iterate_filename, "%s_%d", X_2_USE_PATH_BASE, iteration, X_FILE_EXTENSION );		
		array_2_disk(iterate_filename, RECONSTRUCTION_DIR, X_WRITE_MODE, x_h, parameters.COLUMNS, parameters.ROWS, parameters.SLICES, parameters.NUM_VOXELS, true );
	}
	fclose(MLP_file);
	puts("Image reconstruction complete.  Writing reconstructed image to disk...");
	array_2_disk( X_2_USE_PATH_BASE, RECONSTRUCTION_DIR, TEXT, x_h, parameters.COLUMNS, parameters.ROWS, parameters.SLICES, parameters.NUM_VOXELS, true );
	puts("Reconstructed image written to disk.");
}
void image_reconstruction_import_preprocessing()
{
	init_import_preprocessed_data();
	import_images();
	/*************************************************************************************************************************************************************************/
	/************************************************************************ Perform image reconstruction *******************************************************************/
	/*************************************************************************************************************************************************************************/	
	puts("Starting image reconstruction...");
	char iterate_filename[256];	
	uint start_history = 0, block_MLP_voxels, block_histories = parameters.BLOCK_SIZE;
	//allocate_precalculated_block_arrays( block_histories );
	for( unsigned int iteration = 1; iteration <= parameters.ITERATIONS; iteration++ )
	{
		printf("Performing iteration %u of image reconstruction\n", iteration);
		/*********************************************************************************************************************************************************************/
		/**************** Fill data arrays for each block, calculate updates and # of times each voxel is intersected by a path in block, and update image x *****************/
		/*********************************************************************************************************************************************************************/
		for( unsigned int current_history = start_history; current_history < reconstruction_histories;  )
		{		
			// If there are less than BLOCK_SIZE histories remaining to be processed, the last DROP block will contain only the remaining unused histories
			//i = history_sequence[n];			
			if( current_history + block_histories >= reconstruction_histories )					
				block_histories = reconstruction_histories - current_history;					
			
			// Read DROP block data from disk or copy from global host vectors, then copy this data from host arrays to arrays on GPU
			//fill_precalculated_block_arrays( current_history, block_histories );		// Fill preprocessed data host arrays for each history in the DROP block and copy these to GPU
			// 
			if( parameters.IMPORT_MLP_ITERATIVELY )
				iterative_MLP_2_block_array( block_MLP_voxels );
			else
				MLP_vector_2_block_array( current_MLP_voxel, block_MLP_voxels );

			// If histories in block < THREADS_PER_BLOCK, single kernel can be used for calculations/update.  Otherwise, synchronization is imposed by using separate kernels 
			if( block_histories <= THREADS_PER_BLOCK )
				DROP_single_block_precalculated_MLP( block_histories, current_history );
			else
				DROP_precalculated_MLP( block_histories, current_history );
			
			current_history += parameters.BLOCK_SIZE;
			free(block_paths_h);
			cudaFree(block_paths_d);
		}	
		// Return the read pointers of each file to the beginning of the data (after # of histories value)
		if( parameters.IMPORT_MLP_ITERATIVELY )
			fseek( MLP_file, begin_MLP_data, SEEK_SET );
		cudaMemcpy( x_h, x_d, SIZE_IMAGE_FLOAT, cudaMemcpyDeviceToHost);
		sprintf(iterate_filename, "%s_%d", X_2_USE_PATH_BASE, iteration, X_FILE_EXTENSION );		
		array_2_disk(iterate_filename, RECONSTRUCTION_DIR, X_WRITE_MODE, x_h, parameters.COLUMNS, parameters.ROWS, parameters.SLICES, parameters.NUM_VOXELS, true );
	}
	fclose(MLP_file);
	puts("Image reconstruction complete.  Writing reconstructed image to disk...");
	array_2_disk( X_2_USE_PATH_BASE, RECONSTRUCTION_DIR, TEXT, x_h, parameters.COLUMNS, parameters.ROWS, parameters.SLICES, parameters.NUM_VOXELS, true );
	puts("Reconstructed image written to disk.");
}
void image_reconstruction_import_preprocessing_test()
{
	init_import_preprocessed_data();
	import_images();
	/*************************************************************************************************************************************************************************/
	/************************************************************************ Perform image reconstruction *******************************************************************/
	/*************************************************************************************************************************************************************************/	
	puts("Starting image reconstruction...");
	char iterate_filename[256];	
	uint start_history = 0, block_MLP_voxels, block_histories = parameters.BLOCK_SIZE;
	//allocate_precalculated_block_arrays( block_histories );
	for( unsigned int iteration = 1; iteration <= parameters.ITERATIONS; iteration++ )
	{
		printf("Performing iteration %u of image reconstruction\n", iteration);
		/*********************************************************************************************************************************************************************/
		/**************** Fill data arrays for each block, calculate updates and # of times each voxel is intersected by a path in block, and update image x *****************/
		/*********************************************************************************************************************************************************************/
		for( unsigned int current_history = start_history; current_history < reconstruction_histories;  )
		{		
			// If there are less than BLOCK_SIZE histories remaining to be processed, the last DROP block will contain only the remaining unused histories
			//i = history_sequence[n];			
			if( current_history + block_histories >= reconstruction_histories )					
				block_histories = reconstruction_histories - current_history;					
			
			// Read DROP block data from disk or copy from global host vectors, then copy this data from host arrays to arrays on GPU
			//fill_precalculated_block_arrays( current_history, block_histories );		// Fill preprocessed data host arrays for each history in the DROP block and copy these to GPU
			// 
			if( parameters.IMPORT_MLP_ITERATIVELY )
				iterative_MLP_2_block_array( block_MLP_voxels );
			else
				MLP_vector_2_block_array( current_MLP_voxel, block_MLP_voxels );

			// If histories in block < THREADS_PER_BLOCK, single kernel can be used for calculations/update.  Otherwise, synchronization is imposed by using separate kernels 
			//if( block_histories <= THREADS_PER_BLOCK )
				DROP_single_block_precalculated_MLP( block_histories, current_history );
			//else
				//DROP_precalculated_MLP( block_histories, current_history );
			
			current_history += parameters.BLOCK_SIZE;
			free(block_paths_h);
			cudaFree(block_paths_d);
		}	
		// Return the read pointers of each file to the beginning of the data (after # of histories value)
		if( parameters.IMPORT_MLP_ITERATIVELY )
			fseek( MLP_file, begin_MLP_data, SEEK_SET );
		cudaMemcpy( x_h, x_d, SIZE_IMAGE_FLOAT, cudaMemcpyDeviceToHost);
		sprintf(iterate_filename, "%s_%d", X_2_USE_PATH_BASE, iteration, X_FILE_EXTENSION );		
		array_2_disk(iterate_filename, RECONSTRUCTION_DIR, X_WRITE_MODE, x_h, parameters.COLUMNS, parameters.ROWS, parameters.SLICES, parameters.NUM_VOXELS, true );
	}
	fclose(MLP_file);
	puts("Image reconstruction complete.  Writing reconstructed image to disk...");
	array_2_disk( X_2_USE_PATH_BASE, RECONSTRUCTION_DIR, TEXT, x_h, parameters.COLUMNS, parameters.ROWS, parameters.SLICES, parameters.NUM_VOXELS, true );
	puts("Reconstructed image written to disk.");
}
/********************************************************************************************************* ART *********************************************************************************************************/
__global__ void DROP_calculate_update_GPU_blocks(configurations* parameters, uint block_histories, uint* paths, uint* voxels_per_path, float* avg_chord_lengths, float* WEPLs, uint* S, float* x_update, float* x )
{
	// Determine the history # i corresponding to current thread, which is in the range [0, BLOCK_SIZE -1] and provides index # for data arrays (except paths)
	int thread = threadIdx.x;
	int block = blockIdx.x;
	int i = threadIdx.x + THREADS_PER_BLOCK * block;
	
	if( i < block_histories )
	{
		// Extract corresponding # of voxels in MLP, average chord length, and WEPL measurement for history i
		const int voxels_in_MLP = voxels_per_path[i];	//							=> # of intersected voxels in MLP for history i
		float avg_chord_length = avg_chord_lengths[i];	// ai := avg_chord_length	=> chord length for each intersected voxel j in MLP for history i
		float WEPL = WEPLs[i];							// bi := WEPL				=> WEPL measurement corresponding to history i
		//__shared__ int* MLP;
		//MLP = (int*)malloc( voxels_in_MLP * sizeof(int) );

		// Determine index of "paths" array where MLP for history i begins by determining how many total # of voxels for all histories preceeding it ( i.e. histories [0, i - 1] )
		int MLP_start_index = 0;
		for( int history = 0; history < i; history++ )
			MLP_start_index += voxels_per_path[history];

		// x(K+1) = x(k) + LAMBDA * [ ( bi - <ai, x(k)> ) / <ai, ai> ] ai =  x(k) + parameters.LAMBDA * [ ( WEPL - <ai, x(k)> ) / ||ai||^2 ] ai 
		float ai_dot_xk = 0;											// <ai, x(k)> := ai_dot_xk = <avg_chord_length, x> = SUM(j) { avg_chord_length * x[j] }
		float ai_dot_ai = pow(avg_chord_length, 2) * voxels_in_MLP;		// <ai, ai> := ai_dot_ai = <avg_chord_length, avg_chord_length> = SUM(j) { avg_chord_length * x[j] } = voxels_in_MLP * avg_chord_length ^ 2
		int j;															// j := paths[first_MLP_voxel + i] | i <= voxels_in_MLP (i.e., paths array contains indices of each intersected voxel in MLP
	
		// <ai, x(k)> := ai_dot_xk = SUM(j) { avg_chord_length * x[j] }
		for( int MLP_offset_index = 0; MLP_offset_index < voxels_in_MLP; MLP_offset_index++ )
		{	
			j = paths[MLP_start_index + MLP_offset_index];
			//MLP[MLP_offset_index] = j;
			ai_dot_xk += avg_chord_length * x[j];
			atomicAdd( &S[j], 1 );						// S[j]++
		}

		//  update_value := LAMBDA * [ ( bi - <ai, x(k)> ) / ||ai||^2 ]
		float update_value = parameters->LAMBDA * ( WEPL - ai_dot_xk) / ai_dot_ai;
	
		// x(k+1) = x(k) + update_value * ai
		for( int MLP_offset_index = 0; MLP_offset_index < voxels_in_MLP; MLP_offset_index++ )
		{
			j = paths[MLP_start_index + MLP_offset_index];
			atomicAdd( &x_update[j], update_value * avg_chord_length );
			//j = MLP[MLP_offset_index];
			//atomicAdd( &x_update[j], update_value * avg_chord_length );
			//atomicAdd( &x_update[MLP[MLP_offset_index]], update_value * avg_chord_length );
		}
	}
}
__global__ void DROP_single_block_GPU_blocks(configurations* parameters, uint* paths, uint* voxels_per_path, float* avg_chord_lengths, float* WEPLs, uint* S, float* x_update, float* x )
{
	// Determine the history # i corresponding to current thread, which is in the range [0, BLOCK_SIZE -1] and provides index # for data arrays (except paths)
	int i = threadIdx.x;
	
	// Extract corresponding # of voxels in MLP, average chord length, and WEPL measurement for history i
	const int voxels_in_MLP = voxels_per_path[i];	//							=> # of intersected voxels in MLP for history i
	float avg_chord_length = avg_chord_lengths[i];	// ai := avg_chord_length	=> chord length for each intersected voxel j in MLP for history i
	float WEPL = WEPLs[i];							// bi := WEPL				=> WEPL measurement corresponding to history i
	//__shared__ int* MLP, * S_shared;
	//__shared__ float* x_update_shared, * x_shared;
	//__shared__ int* MLP;
	//MLP = (int*)malloc( voxels_in_MLP * sizeof(int) );
	//__shared__ int* S_shared;
	//S_shared = (int*)malloc( parameters->NUM_VOXELS * sizeof(int) );
	//__shared__ float* x_update_shared;
	//x_update_shared = (float*)malloc( parameters->NUM_VOXELS * sizeof(float) );
	//__shared__ float* x_shared;
	//x_shared = (float*)malloc( parameters->NUM_VOXELS * sizeof(float) );

	// Determine index of "paths" array where MLP for history i begins by determining how many total # of voxels for all histories preceeding it ( i.e. histories [0, i - 1] )
	int MLP_start_index = 0;
	for( int history = 0; history < i; history++ )
		MLP_start_index += voxels_per_path[history];

	// x(K+1) = x(k) + LAMBDA * [ ( bi - <ai, x(k)> ) / <ai, ai> ] ai =  x(k) + parameters.LAMBDA * [ ( WEPL - <ai, x(k)> ) / ||ai||^2 ] ai 
	float ai_dot_xk = 0;											// <ai, x(k)> := ai_dot_xk = <avg_chord_length, x> = SUM(j) { avg_chord_length * x[j] }
	float ai_dot_ai = pow(avg_chord_length, 2) * voxels_in_MLP;		// <ai, ai> := ai_dot_ai = <avg_chord_length, avg_chord_length> = SUM(j) { avg_chord_length * x[j] } = voxels_in_MLP * avg_chord_length ^ 2
	int j;															// j := paths[first_MLP_voxel + i] | i <= voxels_in_MLP (i.e., paths array contains indices of each intersected voxel in MLP
	
	// <ai, x(k)> := ai_dot_xk = SUM(j) { avg_chord_length * x[j] }
	for( int MLP_offset_index = 0; MLP_offset_index < voxels_in_MLP; MLP_offset_index++ )
	{	
		j = paths[MLP_start_index + MLP_offset_index];
		//MLP[MLP_offset_index] = j;
		ai_dot_xk += avg_chord_length * x[j];
		atomicAdd( &S[j], 1 );								// S[j]++
		//ai_dot_xk += avg_chord_length * x_shared[j];
		//atomicAdd( &S_shared[j], 1 );						// S[j]++
	}

	//  update_value := LAMBDA * [ ( bi - <ai, x(k)> ) / ||ai||^2 ]
	float update_value = parameters->LAMBDA * ( WEPL - ai_dot_xk) / ai_dot_ai;
	
	// x(k+1) = x(k) + update_value * ai
	for( int MLP_offset_index = 0; MLP_offset_index < voxels_in_MLP; MLP_offset_index++ )
	{
		j = paths[MLP_start_index + MLP_offset_index];
		//j = MLP[MLP_start_index + MLP_offset_index];
		atomicAdd( &x_update[j], update_value * avg_chord_length );
		//atomicAdd( &x_update[j], update_value * avg_chord_length );
		//atomicAdd( &x_update_shared[j], update_value * avg_chord_length );
	}
	syncthreads();
	for( int MLP_offset_index = 0; MLP_offset_index < voxels_in_MLP; MLP_offset_index++ )
	{
		j = paths[MLP_start_index + MLP_offset_index];
		//j = MLP[MLP_start_index + MLP_offset_index];
		atomicAdd( &x[j], x_update[j] / S[j] );		
		//atomicAdd( &x[j], x_update[j] / S[j] ) );
		//atomicAdd( &x[j], x_update_shared[j] / S_shared[j] ) );
	}
}
void DROP_single_block_precalculated_MLP_blocks( uint block_histories )
{
	// Accumulate the update values from each thread into x_update[j] and add 1 to S[j] for all voxels j intersected by its MLP and once all threads in the
	// single GPU block finish these calculations (imposed by syncthreads() function), have each thread update the voxels of the image x its MLP intersected
	dim3 dimBlock( block_histories );
	dim3 dimGrid( 1 );   	
	DROP_single_block_GPU_blocks<<< dimGrid, dimBlock >>>(parameters_d, block_paths_d, block_voxels_per_path_d, block_avg_chord_lengths_d, block_WEPLs_d, S_d, x_update_d, x_d );
}
void DROP_precalculated_MLP_blocks( uint block_histories )
{
	// Accumulate the update values from each thread into x_update[j] and add 1 to S[j] for all voxels j intersected by its MLP
	uint num_blocks = static_cast<uint>( ceil( static_cast<double>(block_histories / THREADS_PER_BLOCK) ) );
	dim3 dimBlock( THREADS_PER_BLOCK );
	dim3 dimGrid( num_blocks );   	
	DROP_calculate_update_GPU_blocks<<< dimGrid, dimBlock >>>(parameters_d, block_histories, block_paths_d, block_voxels_per_path_d, block_avg_chord_lengths_d, block_WEPLs_d, S_d, x_update_d, x_d );

	// Update each voxel j of the image x in parallel using the values of x_update[j] and S[j]
	
	// Apply update of image x using 1 dimensional grid configuration
	//dimBlock = parameters.COLUMNS * parameters.ROWS * parameters.SLICES;
	//DROP_apply_1D_update_GPU<<< dimGrid, dimBlock >>>(parameters_d, S_d, x_update_d, x_d );
	
	// Apply update of image x using 3 dimensional grid configuration
	dimBlock = parameters.SLICES;  
	dimGrid = dim3(parameters.COLUMNS, parameters.ROWS );
	DROP_apply_3D_update_GPU<<< dimGrid, dimBlock >>>(parameters_d, S_d, x_update_d, x_d );
}
/********************************************************************************************************* ART *********************************************************************************************************/
__global__ void DROP_calculate_update_GPU(configurations* parameters, uint current_history, uint total_histories, uint block_histories, uint* paths, uint* voxels_per_path, float* avg_chord_lengths, float* WEPLs, uint* S, float* x_update, float* x )
{
	// Determine the history # i corresponding to current thread, which is in the range [0, BLOCK_SIZE -1] and provides index # for data arrays (except paths)
	int thread = threadIdx.x;
	int block = blockIdx.x;
	int i = current_history + thread + THREADS_PER_BLOCK * block;
	int block_history = thread + THREADS_PER_BLOCK * block;

	if( (block_history < block_histories) && (i < total_histories) )
	{
		// Extract corresponding # of voxels in MLP, average chord length, and WEPL measurement for history i
		const int voxels_in_MLP = voxels_per_path[i];	//							=> # of intersected voxels in MLP for history i
		float avg_chord_length = avg_chord_lengths[i];	// ai := avg_chord_length	=> chord length for each intersected voxel j in MLP for history i
		float WEPL = WEPLs[i];							// bi := WEPL				=> WEPL measurement corresponding to history i
		//__shared__ int* MLP;
		//MLP = (int*)malloc( voxels_in_MLP * sizeof(int) );

		// Determine index of "paths" array where MLP for history i begins by determining how many total # of voxels for all histories preceeding it ( i.e. histories [0, i - 1] )
		int MLP_start_index = 0;
		for( int history = current_history; history < i; history++ )
			MLP_start_index += voxels_per_path[history];

		// x(K+1) = x(k) + LAMBDA * [ ( bi - <ai, x(k)> ) / <ai, ai> ] ai =  x(k) + parameters.LAMBDA * [ ( WEPL - <ai, x(k)> ) / ||ai||^2 ] ai 
		float ai_dot_xk = 0;											// <ai, x(k)> := ai_dot_xk = <avg_chord_length, x> = SUM(j) { avg_chord_length * x[j] }
		float ai_dot_ai = pow(avg_chord_length, 2) * voxels_in_MLP;		// <ai, ai> := ai_dot_ai = <avg_chord_length, avg_chord_length> = SUM(j) { avg_chord_length * x[j] } = voxels_in_MLP * avg_chord_length ^ 2
		int j;															// j := paths[first_MLP_voxel + i] | i <= voxels_in_MLP (i.e., paths array contains indices of each intersected voxel in MLP
	
		// <ai, x(k)> := ai_dot_xk = SUM(j) { avg_chord_length * x[j] }
		for( int MLP_offset_index = 0; MLP_offset_index < voxels_in_MLP; MLP_offset_index++ )
		{	
			j = paths[MLP_start_index + MLP_offset_index];
			//MLP[MLP_offset_index] = j;
			ai_dot_xk += avg_chord_length * x[j];
			atomicAdd( &S[j], 1 );						// S[j]++
		}

		//  update_value := LAMBDA * [ ( bi - <ai, x(k)> ) / ||ai||^2 ]
		float update_value = parameters->LAMBDA * ( WEPL - ai_dot_xk) / ai_dot_ai;
	
		// x(k+1) = x(k) + update_value * ai
		for( int MLP_offset_index = 0; MLP_offset_index < voxels_in_MLP; MLP_offset_index++ )
		{
			j = paths[MLP_start_index + MLP_offset_index];
			atomicAdd( &x_update[j], update_value * avg_chord_length );
			//j = MLP[MLP_offset_index];
			//atomicAdd( &x_update[j], update_value * avg_chord_length );
			//atomicAdd( &x_update[MLP[MLP_offset_index]], update_value * avg_chord_length );
		}
	}
}
__global__ void DROP_single_block_GPU(configurations* parameters, uint current_history, uint* paths, uint* voxels_per_path, float* avg_chord_lengths, float* WEPLs, uint* S, float* x_update, float* x )
{
	// Determine the history # i corresponding to current thread, which is in the range [0, BLOCK_SIZE -1] and provides index # for data arrays (except paths)
	int i = current_history + threadIdx.x;
	
	// Extract corresponding # of voxels in MLP, average chord length, and WEPL measurement for history i
	const int voxels_in_MLP = voxels_per_path[i];	//							=> # of intersected voxels in MLP for history i
	float avg_chord_length = avg_chord_lengths[i];	// ai := avg_chord_length	=> chord length for each intersected voxel j in MLP for history i
	float WEPL = WEPLs[i];							// bi := WEPL				=> WEPL measurement corresponding to history i
	//__shared__ int* MLP, * S_shared;
	//__shared__ float* x_update_shared, * x_shared;
	//__shared__ int* MLP;
	//MLP = (int*)malloc( voxels_in_MLP * sizeof(int) );
	//__shared__ int* S_shared;
	//S_shared = (int*)malloc( parameters->NUM_VOXELS * sizeof(int) );
	//__shared__ float* x_update_shared;
	//x_update_shared = (float*)malloc( parameters->NUM_VOXELS * sizeof(float) );
	//__shared__ float* x_shared;
	//x_shared = (float*)malloc( parameters->NUM_VOXELS * sizeof(float) );

	// Determine index of "paths" array where MLP for history i begins by determining how many total # of voxels for all histories preceeding it ( i.e. histories [0, i - 1] )
	int MLP_start_index = 0;
	for( int history = current_history; history < i; history++ )
		MLP_start_index += voxels_per_path[history];

	// x(K+1) = x(k) + LAMBDA * [ ( bi - <ai, x(k)> ) / <ai, ai> ] ai =  x(k) + parameters.LAMBDA * [ ( WEPL - <ai, x(k)> ) / ||ai||^2 ] ai 
	float ai_dot_xk = 0;											// <ai, x(k)> := ai_dot_xk = <avg_chord_length, x> = SUM(j) { avg_chord_length * x[j] }
	float ai_dot_ai = pow(avg_chord_length, 2) * voxels_in_MLP;		// <ai, ai> := ai_dot_ai = <avg_chord_length, avg_chord_length> = SUM(j) { avg_chord_length * x[j] } = voxels_in_MLP * avg_chord_length ^ 2
	int j;															// j := paths[first_MLP_voxel + i] | i <= voxels_in_MLP (i.e., paths array contains indices of each intersected voxel in MLP
	
	// <ai, x(k)> := ai_dot_xk = SUM(j) { avg_chord_length * x[j] }
	for( int MLP_offset_index = 0; MLP_offset_index < voxels_in_MLP; MLP_offset_index++ )
	{	
		j = paths[MLP_start_index + MLP_offset_index];
		//MLP[MLP_offset_index] = j;
		ai_dot_xk += avg_chord_length * x[j];
		atomicAdd( &S[j], 1 );								// S[j]++
		//ai_dot_xk += avg_chord_length * x_shared[j];
		//atomicAdd( &S_shared[j], 1 );						// S[j]++
	}

	//  update_value := LAMBDA * [ ( bi - <ai, x(k)> ) / ||ai||^2 ]
	float update_value = parameters->LAMBDA * ( WEPL - ai_dot_xk) / ai_dot_ai;
	
	// x(k+1) = x(k) + update_value * ai
	for( int MLP_offset_index = 0; MLP_offset_index < voxels_in_MLP; MLP_offset_index++ )
	{
		j = paths[MLP_start_index + MLP_offset_index];
		//j = MLP[MLP_start_index + MLP_offset_index];
		atomicAdd( &x_update[j], update_value * avg_chord_length );
		//atomicAdd( &x_update[j], update_value * avg_chord_length );
		//atomicAdd( &x_update_shared[j], update_value * avg_chord_length );
	}
	syncthreads();
	for( int MLP_offset_index = 0; MLP_offset_index < voxels_in_MLP; MLP_offset_index++ )
	{
		j = paths[MLP_start_index + MLP_offset_index];
		//j = MLP[MLP_start_index + MLP_offset_index];
		atomicAdd( &x[j], x_update[j] / S[j] );		
		//atomicAdd( &x[j], x_update[j] / S[j] ) );
		//atomicAdd( &x[j], x_update_shared[j] / S_shared[j] ) );
	}
}
void DROP_single_block_precalculated_MLP( uint block_histories, uint current_history )
{
	// Accumulate the update values from each thread into x_update[j] and add 1 to S[j] for all voxels j intersected by its MLP and once all threads in the
	// single GPU block finish these calculations (imposed by syncthreads() function), have each thread update the voxels of the image x its MLP intersected
	dim3 dimBlock( block_histories );
	dim3 dimGrid( 1 );   	
	DROP_single_block_GPU<<< dimGrid, dimBlock >>>(parameters_d, current_history, block_paths_d, voxels_per_path_d, avg_chord_lengths_d, WEPLs_d, S_d, x_update_d, x_d );
}
void DROP_precalculated_MLP( uint block_histories, uint current_history )
{
	// Accumulate the update values from each thread into x_update[j] and add 1 to S[j] for all voxels j intersected by its MLP
	uint num_blocks = static_cast<uint>( ceil( static_cast<double>(block_histories / THREADS_PER_BLOCK) ) );
	dim3 dimBlock( THREADS_PER_BLOCK );
	dim3 dimGrid( num_blocks );   	
	DROP_calculate_update_GPU<<< dimGrid, dimBlock >>>(parameters_d, current_history, reconstruction_histories, block_histories, block_paths_d, voxels_per_path_d, avg_chord_lengths_d, WEPLs_d, S_d, x_update_d, x_d );

	// Update each voxel j of the image x in parallel using the values of x_update[j] and S[j]
	
	// Apply update of image x using 1 dimensional grid configuration
	//dimBlock = parameters.COLUMNS * parameters.ROWS * parameters.SLICES;
	//DROP_apply_1D_update_GPU<<< dimGrid, dimBlock >>>(parameters_d, S_d, x_update_d, x_d );
	
	// Apply update of image x using 3 dimensional grid configuration
	dimBlock = parameters.SLICES;  
	dimGrid = dim3(parameters.COLUMNS, parameters.ROWS );
	DROP_apply_3D_update_GPU<<< dimGrid, dimBlock >>>(parameters_d, S_d, x_update_d, x_d );
}
/********************************************************************************************************* ART *********************************************************************************************************/
__global__ void DROP_apply_1D_update_GPU(configurations* parameters, uint*& S, float*& x_update, float*& x )
{
	//int column = threadIdx.x;
	//int row = threadIdx.y;
	//int slice = blockIdx.x;
	//int voxel = column + row * parameters.COLUMNS + slice * parameters.COLUMNS * parameters.ROWS;
	int voxel = threadIdx.x;
	if( ( voxel < parameters->NUM_VOXELS ) && ( S[voxel] > 0 ) )
	{
		x[voxel] += x_update[voxel] / S[voxel];
		x_update[voxel] = 0.0;
		S[voxel] = 0;
	}
}
__global__ void DROP_apply_3D_update_GPU( configurations* parameters, uint*& S, float*& x_update, float*& x  )
{
	int column = blockIdx.x, row = blockIdx.y, slice = threadIdx.x;
	int voxel = column + row * parameters->COLUMNS + slice * parameters->COLUMNS * parameters->ROWS;
	if( voxel < parameters->NUM_VOXELS && S[voxel] > 0 )
	{
		x[voxel] += x_update[voxel] / S[voxel];
		x_update[voxel] = 0.0;
		S[voxel] = 0;
	}
}
/********************************************************************************************************* ART *********************************************************************************************************/
template< typename T, typename LHS, typename RHS> T discrete_dot_product( LHS*& left, RHS*& right, unsigned int*& elements, unsigned int num_elements )
{
	T sum = 0;
	for( unsigned int i = 0; i < num_elements; i++)
		sum += ( left[i] * right[elements[i]] );
	return sum;
}
template< typename A, typename X> double update_vector_multiplier( double bi, A*& a_i, X*& x_k, unsigned int*& voxels_intersected, unsigned int num_intersections )
{
	// [ ( bi - <ai, x(k)> ) / <ai, ai> ] ai = [ ( bi - <ai, x(k)> ) / ||ai||^2 ] ai 
	double inner_product_ai_xk = discrete_dot_product<double>( a_i, x_k, voxels_intersected, num_intersections );
	double norm_ai_squared = std::inner_product(a_i, a_i + num_intersections, a_i, 0.0 );
	return ( bi - inner_product_ai_xk ) /  norm_ai_squared;
}
template< typename A, typename X> void update_iterate( double bi, A*& a_i, X*& x_k, unsigned int*& voxels_intersected, unsigned int num_intersections )
{
	// x(K+1) = x(k) + [ ( bi - <ai, x(k)> ) / <ai, ai> ] ai =  x(k) + [ ( bi - <ai, x(k)> ) / ||ai||^2 ] ai 
	double ai_multiplier = update_vector_multiplier( bi, a_i, x_k, voxels_intersected, num_intersections );
	for( int intersection = 0; intersection < num_intersections; intersection++)
		x_k[voxels_intersected[intersection]] += (parameters.LAMBDA * sqrt(bi) )* ai_multiplier * a_i[intersection];
}
/***********************************************************************************************************************************************************************************************************************/
template< typename T, typename RHS> T scalar_dot_product( double scalar, RHS*& vector, unsigned int*& elements, unsigned int num_elements )
{
	T sum = 0;
	for( unsigned int i = 0; i < num_elements; i++)
		sum += vector[elements[i]];
	return scalar * sum;
}
template< typename X> double update_vector_multiplier2( double bi, double mean_chord_length, X*& x_k, unsigned int*& voxels_intersected, unsigned int num_intersections )
{
	// [ ( bi - <ai, x(k)> ) / <ai, ai> ] ai = [ ( bi - <ai, x(k)> ) / ||ai||^2 ] ai 
	double inner_product_ai_xk = scalar_dot_product<double>( mean_chord_length, x_k, voxels_intersected, num_intersections );
	double norm_ai_squared = pow(mean_chord_length, 2.0 ) * num_intersections;
	return ( bi - inner_product_ai_xk ) /  norm_ai_squared;
}
template<typename X> void update_iterate2( double bi, double mean_chord_length, X*& x_k, unsigned int*& voxels_intersected, unsigned int num_intersections )
{
	// x(K+1) = x(k) + [ ( bi - <ai, x(k)> ) / <ai, ai> ] ai =  x(k) + [ ( bi - <ai, x(k)> ) / ||ai||^2 ] ai 
	double ai_multiplier = update_vector_multiplier2( bi, mean_chord_length, x_k, voxels_intersected, num_intersections );
	//cout << "ai_multiplier = " << ai_multiplier << endl;
	//int middle_intersection = num_intersections / 2;
	unsigned int voxel;
	double radius_squared;
	double scale_factor = parameters.LAMBDA * ai_multiplier * mean_chord_length;
	//double scaled_lambda;
	for( unsigned int intersection = 0; intersection < num_intersections; intersection++)
	{
		voxel = voxels_intersected[intersection];
		radius_squared = voxel_2_radius_squared( voxel );
		//	1 - a*r(i)^2 DECAY_FACTOR
		//exp(-a*r)  EXPONENTIALECAY
		//exp(-a*r^2)  EXPONENTIAL_SQDECAY
		//scaled_lambda = parameters.LAMBDA * ( 1 - DECAY_FACTOR * radius_squared );
		// parameters.LAMBDA * ( 1 - DECAY_FACTOR * radius_squared );
		// parameters.LAMBDA * exp( -EXPONENTIALECAY * sqrt( radius_squared ) );
		// parameters.LAMBDA * exp( -EXPONENTIAL_SQDECAY * radius_squared );
		//x_k[voxel] +=  scale_factor * ( 1 - DECAY_FACTOR * radius_squared );
		x_k[voxel] +=  scale_factor * exp( -EXPONENTIAL_SQDECAY * radius_squared );
		//x_k[voxels_intersected[intersection]] += (parameters.LAMBDA / sqrt( abs(middle_intersection - intersection) + 1.0) ) * ai_multiplier * mean_chord_length;
		//x_k[voxels_intersected[intersection]] += (parameters.LAMBDA * max(1.0, sqrt(bi) ) ) * ai_multiplier * mean_chord_length;
	}
}
/***********************************************************************************************************************************************************************************************************************/
double scalar_dot_product2( double scalar, float*& vector, unsigned int*& elements, unsigned int num_elements )
{
	double sum = 0;
	for( unsigned int i = 0; i < num_elements; i++)
		sum += vector[elements[i]];
	return scalar * sum;
}
void update_iterate22( double bi, double mean_chord_length, float*& x_k, unsigned int*& voxels_intersected, unsigned int num_intersections )
{
	// x(K+1) = x(k) + [ ( bi - <ai, x(k)> ) / <ai, ai> ] ai =  x(k) + [ ( bi - <ai, x(k)> ) / ||ai||^2 ] ai 
	double a_i_dot_x_k = 0;
	for( unsigned int i = 0; i < num_intersections; i++)
		a_i_dot_x_k += x_k[voxels_intersected[i]];
	double scale_factor =  parameters.LAMBDA * ( bi - mean_chord_length * a_i_dot_x_k ) /  ( num_intersections * pow( mean_chord_length, 2.0)  ) * mean_chord_length;	
	//for( unsigned int intersection = 0; intersection < num_voxel_scales; intersection++ )
		//x_k[voxels_intersected[intersection]] += voxel_scales[intersection] * scale_factor;
	for( unsigned int intersection = 0; intersection < num_intersections; intersection++)
	//for( unsigned int intersection = num_voxel_scales; intersection < num_intersections; intersection++)
		x_k[voxels_intersected[intersection]] += scale_factor;
}
/**************************************************************************************************** DROP ******************************************************************************************************/
void DROP_blocks_robust1( unsigned int*& a_i, float*& x_k, double bi, unsigned int num_intersections, double mean_chord_length, double*& x_update, unsigned int*& voxel_intersections )
{
	// x(K+1) = x(k) + parameters.LAMBDA * [ ( bi - <ai, x(k)> ) / <ai, ai> ] ai =  x(k) + parameters.LAMBDA * [ ( bi - <ai, x(k)> ) / ( ||ai||^2 +/- Psi(i) )] ai 
	//  (1) <ai, x(k)>
	double a_i_dot_x_k = scalar_dot_product2( mean_chord_length, x_k, a_i, num_intersections );
	// (2) ( bi - <ai, x(k)> ) / <ai, ai>
	double a_i_dot_a_i =  pow(mean_chord_length, 2.0) * num_intersections;
	double residual =  bi - a_i_dot_x_k;
	for( unsigned int intersection = 0; intersection < num_intersections; intersection++)
	{
	        double psi_i = ((1.0 - x_k[a_i[intersection]]) * parameters.ETA) * parameters.PSI_SIGN;
		
		double update_value = parameters.LAMBDA * (residual / (a_i_dot_a_i + psi_i)); 
		x_update[a_i[intersection]] += update_value;
		voxel_intersections[a_i[intersection]] += 1;
	}
}
void DROP_blocks_robust2( unsigned int*& a_i, float*& x_k, double bi, unsigned int num_intersections, double mean_chord_length, double*& x_update, unsigned int*& voxel_intersections, double*& norm_Ai )
{
	// x(K+1) = x(k) + parameters.LAMBDA * [ ( bi - <ai, x(k)> ) / <ai, ai> ] ai =  x(k) + parameters.LAMBDA * [ ( bi - <ai, x(k)> ) / ||ai||^2 ] ai 
	//  (1) <ai, x(k)>
	double a_i_dot_x_k = scalar_dot_product2( mean_chord_length, x_k, a_i, num_intersections );
	// (2) ( bi - <ai, x(k)> ) / <ai, ai>
	double scaled_residual = ( bi - a_i_dot_x_k ) /  (  pow(mean_chord_length, 2.0) * num_intersections );
	// (3) parameters.LAMBDA * [ ( bi - <ai, x(k)> ) / ||ai||^2 ]
	double update_value = mean_chord_length * parameters.LAMBDA * scaled_residual;	
	for( unsigned int intersection = 0; intersection < num_intersections; intersection++)
	{
		x_update[a_i[intersection]] += update_value;
		voxel_intersections[a_i[intersection]] += 1;
		norm_Ai[a_i[intersection]] += pow( mean_chord_length, 2.0 );
	}
}
void DROP_update_robust2( float*& x_k, double*& x_update, unsigned int*& voxel_intersections, double*& norm_Ai )
{
	double psi_i;
	for( unsigned int voxel = 0; voxel < parameters.NUM_VOXELS; voxel++ )
	{
		if( voxel_intersections[voxel] > 0 )
		{
			psi_i = parameters.PSI_SIGN*(1-x_k[voxel])*parameters.ETA;
			//x_k[voxel] += x_update[voxel] / voxel_intersections[voxel];
			x_k[voxel] += ( norm_Ai[voxel]/ (norm_Ai[voxel] + psi_i))  * x_update[voxel] / voxel_intersections[voxel];		
			x_update[voxel] = 0;
			voxel_intersections[voxel] = 0;
		}
	}
}
void DROP_blocks3
( 
	unsigned int*& path_voxels, float*& x_k, double bi, unsigned int path_intersections, double mean_chord_length, 
	double*& x_update, unsigned int*& block_voxels, unsigned int& block_intersections
)
{
	// x(K+1) = x(k) + parameters.LAMBDA * [ ( bi - <ai, x(k)> ) / <ai, ai> ] ai =  x(k) + parameters.LAMBDA * [ ( bi - <ai, x(k)> ) / ||ai||^2 ] ai 
	//  (1) <ai, x(k)>
	double a_i_dot_x_k = scalar_dot_product2( mean_chord_length, x_k, path_voxels, path_intersections );
	// (2) ( bi - <ai, x(k)> ) / <ai, ai>
	double scaled_residual = ( bi - a_i_dot_x_k ) /  (  pow(mean_chord_length, 2.0) * path_intersections );
	// (3) parameters.LAMBDA * [ ( bi - <ai, x(k)> ) / ||ai||^2 ]
	double update_value = mean_chord_length * parameters.LAMBDA * scaled_residual;	
	//for( unsigned int intersection = 0; intersection < num_voxel_scales; intersection++ )
	//{
	//	x_update[a_i[intersection]] += voxel_scales[intersection] * update_value;
	//	voxel_intersections[a_i[intersection]] += 1;
	//}
	//bool new_block_voxel = true;
	for( unsigned int path_intersection = 0; path_intersection < path_intersections; path_intersection++)
	{
		block_voxels[block_intersections] = path_voxels[path_intersection];
		x_update[block_voxels[block_intersections]] += update_value;
		//block_counts[block_intersections] +=1;
		block_intersections++;
	}
}
void DROP_update3( float*& x_k, double*& x_update, unsigned int*& block_voxels, unsigned int& block_intersections )
{
	int count;
	for( unsigned int block_intersection = 0; block_intersection < block_intersections; block_intersection++ )
	{
		count = static_cast<int>(std::count(block_voxels, &block_voxels[block_intersections], block_voxels[block_intersection] ));
		x_k[block_voxels[block_intersection]] += x_update[block_voxels[block_intersection]] / count;
		//x_update[block_intersection] = 0;
	}
	memset(x_update, 0.0, parameters.NUM_VOXELS );
	block_intersections = 0;
}
void DROP_blocks2
( 
	unsigned int*& path_voxels, float*& x_k, double bi, unsigned int path_intersections, double mean_chord_length, 
	double*& x_update, unsigned int*& block_voxels, unsigned int& block_intersections, unsigned int*& block_counts 
)
{
	// x(K+1) = x(k) + parameters.LAMBDA * [ ( bi - <ai, x(k)> ) / <ai, ai> ] ai =  x(k) + parameters.LAMBDA * [ ( bi - <ai, x(k)> ) / ||ai||^2 ] ai 
	//  (1) <ai, x(k)>
	double a_i_dot_x_k = scalar_dot_product2( mean_chord_length, x_k, path_voxels, path_intersections );
	// (2) ( bi - <ai, x(k)> ) / <ai, ai>
	double scaled_residual = ( bi - a_i_dot_x_k ) /  (  pow(mean_chord_length, 2.0) * path_intersections );
	// (3) parameters.LAMBDA * [ ( bi - <ai, x(k)> ) / ||ai||^2 ]
	double update_value = mean_chord_length * parameters.LAMBDA * scaled_residual;	
	//for( unsigned int intersection = 0; intersection < num_voxel_scales; intersection++ )
	//{
	//	x_update[a_i[intersection]] += voxel_scales[intersection] * update_value;
	//	voxel_intersections[a_i[intersection]] += 1;
	//}
	//bool new_block_voxel = true;
	for( unsigned int path_intersection = 0; path_intersection < path_intersections; path_intersection++)
	{
		//for( unsigned int block_intersection = 0; block_intersection < block_intersections; block_intersection++ )
		//{
			//if( path_voxels[path_intersection] == block_voxels[block_intersection] )
			unsigned int* ptr = std::find( block_voxels, block_voxels + block_intersections, path_voxels[path_intersection] );
			
			if( ptr == block_voxels + block_intersections )
			{
				block_voxels[block_intersections] = path_voxels[path_intersection];
				x_update[block_intersections] += update_value;
				block_counts[block_intersections] +=1;
				block_intersections++;
			}
			else
			{
				x_update[*ptr] += update_value;
				block_counts[*ptr] +=1;
			}
			//else
			//{
			//	x_update[block_intersection] += update_value;
			//	block_counts[block_intersection] +=1;
			//}
		//}
	}
}
void DROP_update2( float*& x_k, double*& x_update, unsigned int*& block_voxels, unsigned int& block_intersections, unsigned int*& block_counts )
{
	for( unsigned int block_intersection = 0; block_intersection < block_intersections; block_intersection++ )
	{
			x_k[block_voxels[block_intersection]] += x_update[block_intersection] / block_counts[block_intersection];
			x_update[block_intersection] = 0;
			block_counts[block_intersection] = 0;
	}
	block_intersections = 0;
}
void DROP_blocks( unsigned int*& a_i, float*& x_k, double bi, unsigned int num_intersections, double mean_chord_length, double*& x_update, unsigned int*& voxel_intersections )
{
	// x(K+1) = x(k) + parameters.LAMBDA * [ ( bi - <ai, x(k)> ) / <ai, ai> ] ai =  x(k) + parameters.LAMBDA * [ ( bi - <ai, x(k)> ) / ||ai||^2 ] ai 
	//  (1) <ai, x(k)>
	double a_i_dot_x_k = scalar_dot_product2( mean_chord_length, x_k, a_i, num_intersections );
	// (2) ( bi - <ai, x(k)> ) / <ai, ai>
	double scaled_residual = ( bi - a_i_dot_x_k ) /  (  pow(mean_chord_length, 2.0) * num_intersections );
	// (3) parameters.LAMBDA * [ ( bi - <ai, x(k)> ) / ||ai||^2 ]
	double update_value = mean_chord_length * parameters.LAMBDA * scaled_residual;	
	//for( unsigned int intersection = 0; intersection < num_voxel_scales; intersection++ )
	//{
	//	x_update[a_i[intersection]] += voxel_scales[intersection] * update_value;
	//	voxel_intersections[a_i[intersection]] += 1;
	//}
	for( unsigned int intersection = 0; intersection < num_intersections; intersection++)
	{
		x_update[a_i[intersection]] += update_value;
		voxel_intersections[a_i[intersection]] += 1;
	}
}
void DROP_update( float*& x_k, double*& x_update, unsigned int*& voxel_intersections )
{
	for( unsigned int voxel = 0; voxel < parameters.NUM_VOXELS; voxel++ )
	{
		if( voxel_intersections[voxel] > 0 )
		{
			x_k[voxel] += x_update[voxel] / voxel_intersections[voxel];
			x_update[voxel] = 0;
			voxel_intersections[voxel] = 0;
		}
	}
}
void update_x(float*& x_k, double*& x_update, unsigned int*& voxel_intersections )
{
	for( unsigned int voxel = 0; voxel < parameters.NUM_VOXELS; voxel++ )
	{
		if( voxel_intersections[voxel] > 0 )
		{
			x_k[voxel] += x_update[voxel] / voxel_intersections[voxel];
			x_update[voxel] = 0;
			voxel_intersections[voxel] = 0;
		}
	}
	//cudaMemcpy( block_voxels_d, block_voxels_h, SIZE_IMAGE_UINT, cudaMemcpyHostToDevice );
	//cudaMemcpy( x_update_d, x_update_h, SIZE_IMAGE_DOUBLE, cudaMemcpyHostToDevice );
	//
	//dim3 dimBlock( parameters.SLICES );
	//dim3 dimGrid( parameters.COLUMNS, parameters.ROWS );   

	//update_x_GPU<<< dimGrid, dimBlock >>>( x_d, x_update_d, block_voxels_d );
	//cudaMemcpy( x_h, x_d, SIZE_IMAGE_FLOAT, cudaMemcpyDeviceToHost );
}
/***********************************************************************************************************************************************************************************************************************/
/********************************************************************************* Image Position/Voxel Calculation Functions (Host) ***********************************************************************************/
/***********************************************************************************************************************************************************************************************************************/
int calculate_voxel( double zero_coordinate, double current_position, double voxel_size )
{
	return static_cast<int>( abs( current_position - zero_coordinate ) / voxel_size );
}
int positions_2_voxels(const double x, const double y, const double z, int& voxel_x, int& voxel_y, int& voxel_z )
{
	voxel_x = static_cast<int>( ( x - parameters.X_ZERO_COORDINATE ) / parameters.VOXEL_WIDTH );				
	voxel_y = static_cast<int>( ( parameters.Y_ZERO_COORDINATE - y ) / parameters.VOXEL_HEIGHT );
	voxel_z = static_cast<int>( ( parameters.Z_ZERO_COORDINATE - z ) / parameters.VOXEL_THICKNESS );
	return voxel_x + voxel_y * parameters.COLUMNS + voxel_z * parameters.COLUMNS * parameters.ROWS;
}
int position_2_voxel( double x, double y, double z )
{
	int voxel_x = static_cast<int>( ( x - parameters.X_ZERO_COORDINATE ) / parameters.VOXEL_WIDTH );
	int voxel_y = static_cast<int>( ( parameters.Y_ZERO_COORDINATE - y ) / parameters.VOXEL_HEIGHT );
	int voxel_z = static_cast<int>( ( parameters.Z_ZERO_COORDINATE - z ) / parameters.VOXEL_THICKNESS );
	return voxel_x + voxel_y * parameters.COLUMNS + voxel_z * parameters.COLUMNS * parameters.ROWS;
}
void voxel_2_3D_voxels( int voxel, int& voxel_x, int& voxel_y, int& voxel_z )
{
	voxel_x = 0;
    voxel_y = 0;
    voxel_z = 0;
    
    while( voxel - parameters.COLUMNS * parameters.ROWS > 0 )
	{
		voxel -= parameters.COLUMNS * parameters.ROWS;
		voxel_z++;
	}
	// => bin = t_bin + angular_bin * parameters.T_BINS > 0
	while( voxel - parameters.COLUMNS > 0 )
	{
		voxel -= parameters.COLUMNS;
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
	x = voxel_2_position( voxel_x, parameters.VOXEL_WIDTH, parameters.COLUMNS, 1 );
	y = voxel_2_position( voxel_y, parameters.VOXEL_HEIGHT, parameters.ROWS, -1 );
	z = voxel_2_position( voxel_z, parameters.VOXEL_THICKNESS, parameters.SLICES, -1 );
}
double voxel_2_radius_squared( int voxel )
{
	int voxel_x, voxel_y, voxel_z;
	voxel_2_3D_voxels( voxel, voxel_x, voxel_y, voxel_z );
	double x = voxel_2_position( voxel_x, parameters.VOXEL_WIDTH, parameters.COLUMNS, 1 );
	double y = voxel_2_position( voxel_y, parameters.VOXEL_HEIGHT, parameters.ROWS, -1 );
	return pow( x, 2.0 ) + pow( y, 2.0 );
}
/***********************************************************************************************************************************************************************************************************************/
/********************************************************************************* Image Position/Voxel Calculation Functions (Device) *********************************************************************************/
/***********************************************************************************************************************************************************************************************************************/
__device__ int calculate_voxel_GPU( double zero_coordinate, double current_position, double voxel_size )
{
	return abs( current_position - zero_coordinate ) / voxel_size;
}
__device__ int positions_2_voxels_GPU(configurations* parameters, const double x, const double y, const double z, int& voxel_x, int& voxel_y, int& voxel_z )
{
	voxel_x = static_cast<int>( ( x - parameters->X_ZERO_COORDINATE) / parameters->VOXEL_WIDTH );				
	voxel_y = static_cast<int>( ( parameters->Y_ZERO_COORDINATE - y ) / parameters->VOXEL_HEIGHT );
	voxel_z = static_cast<int>( ( parameters->Z_ZERO_COORDINATE - z ) / parameters->VOXEL_THICKNESS );
	return voxel_x + voxel_y * parameters->COLUMNS + voxel_z * parameters->COLUMNS * parameters->ROWS;
}
__device__ int position_2_voxel_GPU( configurations* parameters, double x, double y, double z )
{
	int voxel_x = static_cast<int>( ( x - parameters->X_ZERO_COORDINATE ) / parameters->VOXEL_WIDTH );
	int voxel_y = static_cast<int>( ( parameters->Y_ZERO_COORDINATE - y ) / parameters->VOXEL_HEIGHT );
	int voxel_z = static_cast<int>( ( parameters->Z_ZERO_COORDINATE - z ) / parameters->VOXEL_THICKNESS );
	return voxel_x + voxel_y * parameters->COLUMNS + voxel_z * parameters->COLUMNS * parameters->ROWS;
}
__device__ void voxel_2_3D_voxels_GPU( configurations* parameters, int voxel, int& voxel_x, int& voxel_y, int& voxel_z )
{
	voxel_x = 0;
    voxel_y = 0;
    voxel_z = 0;
    
    while( voxel - parameters->COLUMNS * parameters->ROWS > 0 )
	{
		voxel -= parameters->COLUMNS * parameters->ROWS;
		voxel_z++;
	}
	// => bin = t_bin + angular_bin * parameters->T_BINS > 0
	while( voxel - parameters->COLUMNS > 0 )
	{
		voxel -= parameters->COLUMNS;
		voxel_y++;
	}
	// => bin = t_bin > 0
	voxel_x = voxel;
}
__device__ double voxel_2_position_GPU( configurations* parameters, int voxel_i, double voxel_i_size, int num_voxels_i, int coordinate_progression )

{
	// voxel_i = 50, num_voxels_i = 200, middle_voxel = 100, ( 50 - 100 ) * 1 = -50
	double zero_voxel = ( num_voxels_i - 1) / 2.0;
	return coordinate_progression * ( voxel_i - zero_voxel ) * voxel_i_size;
}
__device__ void voxel_2_positions_GPU( configurations* parameters, int voxel, double& x, double& y, double& z )
{
	int voxel_x, voxel_y, voxel_z;
	voxel_2_3D_voxels_GPU( parameters, voxel, voxel_x, voxel_y, voxel_z );
	x = voxel_2_position_GPU( parameters, voxel_x, parameters->VOXEL_WIDTH, parameters->COLUMNS, 1 );
	y = voxel_2_position_GPU( parameters, voxel_y, parameters->VOXEL_HEIGHT, parameters->ROWS, -1 );
	z = voxel_2_position_GPU( parameters, voxel_z, parameters->VOXEL_THICKNESS, parameters->SLICES, -1 );
}
__device__ double voxel_2_radius_squared_GPU( configurations* parameters, int voxel )
{
	int voxel_x, voxel_y, voxel_z;
	voxel_2_3D_voxels_GPU( parameters, voxel, voxel_x, voxel_y, voxel_z );
	double x = voxel_2_position_GPU( parameters, voxel_x, parameters->VOXEL_WIDTH, parameters->COLUMNS, 1 );
	double y = voxel_2_position_GPU( parameters, voxel_y, parameters->VOXEL_HEIGHT, parameters->ROWS, -1 );
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
	// Determine if on left/top or right/bottom edge, since entering a voxel can happen from either side depending on path direction, then calculate the x/y/z coordinate corresponding to the x/y/z edge, respectively
	// If stepping in direction of increasing x/y/z voxel #, entered on left/top edge, otherwise it entered on right/bottom edge.  Left/bottom edge is voxel_entered * voxel_size from 0 coordinate of first x/y/z voxel
	int on_edge = ( step_direction == increasing_direction ) ? voxel_entered : voxel_entered + 1;
	return (increasing_direction > 0 ) ? zero_coordinate + ( on_edge * voxel_size ) : zero_coordinate - ( on_edge * voxel_size );
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
		x = edge_coordinate( parameters.X_ZERO_COORDINATE, voxel_x, parameters.VOXEL_WIDTH, X_INCREASING_DIRECTION, x_move_direction );
		y = corresponding_coordinate( dy_dx, x, x_start, y_start );
		x_to_go = parameters.VOXEL_WIDTH;
		y_to_go = distance_remaining( parameters.Y_ZERO_COORDINATE, y, Z_INCREASING_DIRECTION, y_move_direction, parameters.VOXEL_HEIGHT, voxel_y );
	}
	// Else Next Voxel Edge is in y
	else
	{
		//printf(" y_extension < x_extension \n");				
		voxel_y -= y_move_direction;
		y = edge_coordinate( parameters.Y_ZERO_COORDINATE, voxel_y, parameters.VOXEL_HEIGHT, Y_INCREASING_DIRECTION, y_move_direction );
		x = corresponding_coordinate( dx_dy, y, y_start, x_start );
		x_to_go = distance_remaining( parameters.X_ZERO_COORDINATE, x, X_INCREASING_DIRECTION, x_move_direction, parameters.VOXEL_WIDTH, voxel_x );
		y_to_go = parameters.VOXEL_HEIGHT;
	}
	if( x_to_go == 0 )
	{
		x_to_go = parameters.VOXEL_WIDTH;
		voxel_x += x_move_direction;
	}
	if( y_to_go == 0 )
	{
		y_to_go = parameters.VOXEL_HEIGHT;
		voxel_y -= y_move_direction;
	}
	voxel_z = max(voxel_z, 0 );
	voxel = voxel_x + voxel_y * parameters.COLUMNS + voxel_z * parameters.COLUMNS * parameters.ROWS;
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
		z = edge_coordinate( parameters.Z_ZERO_COORDINATE, voxel_z, parameters.VOXEL_THICKNESS, Z_INCREASING_DIRECTION, z_move_direction );					
		x = corresponding_coordinate( dx_dz, z, z_start, x_start );
		y = corresponding_coordinate( dy_dz, z, z_start, y_start );
		x_to_go = distance_remaining( parameters.X_ZERO_COORDINATE, x, X_INCREASING_DIRECTION, x_move_direction, parameters.VOXEL_WIDTH, voxel_x );
		y_to_go = distance_remaining( parameters.Y_ZERO_COORDINATE, y, Y_INCREASING_DIRECTION, y_move_direction, parameters.VOXEL_HEIGHT, voxel_y );	
		z_to_go = parameters.VOXEL_THICKNESS;
	}
	//If Next Voxel Edge is in x or xy Diagonal
	else if( x_extension <= y_extension )
	{
		//printf(" x_extension <= y_extension \n");					
		voxel_x += x_move_direction;
		x = edge_coordinate( parameters.X_ZERO_COORDINATE, voxel_x, parameters.VOXEL_WIDTH, X_INCREASING_DIRECTION, x_move_direction );
		y = corresponding_coordinate( dy_dx, x, x_start, y_start );
		z = corresponding_coordinate( dz_dx, x, x_start, z_start );
		x_to_go = parameters.VOXEL_WIDTH;
		y_to_go = distance_remaining( parameters.Y_ZERO_COORDINATE, y, Y_INCREASING_DIRECTION, y_move_direction, parameters.VOXEL_HEIGHT, voxel_y );
		z_to_go = distance_remaining( parameters.Z_ZERO_COORDINATE, z, Z_INCREASING_DIRECTION, z_move_direction, parameters.VOXEL_THICKNESS, voxel_z );
	}
	// Else Next Voxel Edge is in y
	else
	{
		//printf(" y_extension < x_extension \n");
		voxel_y -= y_move_direction;					
		y = edge_coordinate( parameters.Y_ZERO_COORDINATE, voxel_y, parameters.VOXEL_HEIGHT, Y_INCREASING_DIRECTION, y_move_direction );
		x = corresponding_coordinate( dx_dy, y, y_start, x_start );
		z = corresponding_coordinate( dz_dy, y, y_start, z_start );
		x_to_go = distance_remaining( parameters.X_ZERO_COORDINATE, x, X_INCREASING_DIRECTION, x_move_direction, parameters.VOXEL_WIDTH, voxel_x );
		y_to_go = parameters.VOXEL_HEIGHT;					
		z_to_go = distance_remaining( parameters.Z_ZERO_COORDINATE, z, Z_INCREASING_DIRECTION, z_move_direction, parameters.VOXEL_THICKNESS, voxel_z );
	}
	if( x_to_go == 0 )
	{
		x_to_go = parameters.VOXEL_WIDTH;
		voxel_x += x_move_direction;
	}
	if( y_to_go == 0 )
	{
		y_to_go = parameters.VOXEL_HEIGHT;
		voxel_y -= y_move_direction;
	}
	if( z_to_go == 0 )
	{
		z_to_go = parameters.VOXEL_THICKNESS;
		voxel_z -= z_move_direction;
	}
	voxel_z = max(voxel_z, 0 );
	voxel = voxel_x + voxel_y * parameters.COLUMNS + voxel_z * parameters.COLUMNS * parameters.ROWS;
	//end_walk = ( voxel == voxel_out ) || ( voxel_x >= parameters.COLUMNS ) || ( voxel_y >= parameters.ROWS ) || ( voxel_z >= parameters.SLICES );
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
	configurations* parameters, const int x_move_direction, const int y_move_direction, const int z_move_direction,
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
		x = edge_coordinate_GPU( parameters->X_ZERO_COORDINATE, voxel_x, parameters->VOXEL_WIDTH, X_INCREASING_DIRECTION, x_move_direction );
		y = corresponding_coordinate_GPU( dy_dx, x, x_start, y_start );
		x_to_go = parameters->VOXEL_WIDTH;
		y_to_go = distance_remaining_GPU( parameters->Y_ZERO_COORDINATE, y, Y_INCREASING_DIRECTION, y_move_direction, parameters->VOXEL_HEIGHT, voxel_y );
	}
	// Else Next Voxel Edge is in y
	else
	{
		//printf(" y_extension < x_extension \n");				
		voxel_y -= y_move_direction;
		y = edge_coordinate_GPU( parameters->Y_ZERO_COORDINATE, voxel_y, parameters->VOXEL_HEIGHT, Y_INCREASING_DIRECTION, y_move_direction );
		x = corresponding_coordinate_GPU( dx_dy, y, y_start, x_start );
		x_to_go = distance_remaining_GPU( parameters->X_ZERO_COORDINATE, x, X_INCREASING_DIRECTION, x_move_direction, parameters->VOXEL_WIDTH, voxel_x );
		y_to_go = parameters->VOXEL_HEIGHT;
	}
	if( x_to_go == 0 )
	{
		x_to_go = parameters->VOXEL_WIDTH;
		voxel_x += x_move_direction;
	}
	if( y_to_go == 0 )
	{
		y_to_go = parameters->VOXEL_HEIGHT;
		voxel_y -= y_move_direction;
	}
	voxel_z = max(voxel_z, 0 );
	voxel = voxel_x + voxel_y * parameters->COLUMNS + voxel_z * parameters->COLUMNS * parameters->ROWS;
}
__device__ void take_3D_step_GPU
( 
	configurations* parameters, const int x_move_direction, const int y_move_direction, const int z_move_direction,
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
		z = edge_coordinate_GPU( parameters->Z_ZERO_COORDINATE, voxel_z, parameters->VOXEL_THICKNESS, Z_INCREASING_DIRECTION, z_move_direction );					
		x = corresponding_coordinate_GPU( dx_dz, z, z_start, x_start );
		y = corresponding_coordinate_GPU( dy_dz, z, z_start, y_start );
		x_to_go = distance_remaining_GPU( parameters->X_ZERO_COORDINATE, x, X_INCREASING_DIRECTION, x_move_direction, parameters->VOXEL_WIDTH, voxel_x );
		y_to_go = distance_remaining_GPU( parameters->Y_ZERO_COORDINATE, y, Y_INCREASING_DIRECTION, y_move_direction, parameters->VOXEL_HEIGHT, voxel_y );	
		z_to_go = parameters->VOXEL_THICKNESS;
	}
	//If Next Voxel Edge is in x or xy Diagonal
	else if( x_extension <= y_extension )
	{
		//printf(" x_extension <= y_extension \n");					
		voxel_x += x_move_direction;
		x = edge_coordinate_GPU( parameters->X_ZERO_COORDINATE, voxel_x, parameters->VOXEL_WIDTH, X_INCREASING_DIRECTION, x_move_direction );
		y = corresponding_coordinate_GPU( dy_dx, x, x_start, y_start );
		z = corresponding_coordinate_GPU( dz_dx, x, x_start, z_start );
		x_to_go = parameters->VOXEL_WIDTH;
		y_to_go = distance_remaining_GPU( parameters->Y_ZERO_COORDINATE, y, Y_INCREASING_DIRECTION, y_move_direction, parameters->VOXEL_HEIGHT, voxel_y );
		z_to_go = distance_remaining_GPU( parameters->Z_ZERO_COORDINATE, z, Z_INCREASING_DIRECTION, z_move_direction, parameters->VOXEL_THICKNESS, voxel_z );
	}
	// Else Next Voxel Edge is in y
	else
	{
		//printf(" y_extension < x_extension \n");
		voxel_y -= y_move_direction;					
		y = edge_coordinate_GPU( parameters->Y_ZERO_COORDINATE, voxel_y, parameters->VOXEL_HEIGHT, Y_INCREASING_DIRECTION, y_move_direction );
		x = corresponding_coordinate_GPU( dx_dy, y, y_start, x_start );
		z = corresponding_coordinate_GPU( dz_dy, y, y_start, z_start );
		x_to_go = distance_remaining_GPU( parameters->X_ZERO_COORDINATE, x, X_INCREASING_DIRECTION, x_move_direction, parameters->VOXEL_WIDTH, voxel_x );
		y_to_go = parameters->VOXEL_HEIGHT;					
		z_to_go = distance_remaining_GPU( parameters->Z_ZERO_COORDINATE, z, Z_INCREASING_DIRECTION, z_move_direction, parameters->VOXEL_THICKNESS, voxel_z );
	}
	if( x_to_go == 0 )
	{
		x_to_go = parameters->VOXEL_WIDTH;
		voxel_x += x_move_direction;
	}
	if( y_to_go == 0 )
	{
		y_to_go = parameters->VOXEL_HEIGHT;
		voxel_y -= y_move_direction;
	}
	if( z_to_go == 0 )
	{
		z_to_go = parameters->VOXEL_THICKNESS;
		voxel_z -= z_move_direction;
	}
	voxel_z = max(voxel_z, 0 );
	voxel = voxel_x + voxel_y * parameters->COLUMNS + voxel_z * parameters->COLUMNS * parameters->ROWS;
	//end_walk = ( voxel == voxel_out ) || ( voxel_x >= parameters->COLUMNS ) || ( voxel_y >= parameters->ROWS ) || ( voxel_z >= parameters->SLICES );
}
/***********************************************************************************************************************************************************************************************************************/
/********************************************************************************** Routines for Writing Data Arrays/Vectors to Disk ***********************************************************************************/
/***********************************************************************************************************************************************************************************************************************/
void binary_2_ASCII()
{
	count_histories();
	char filename[256];
	FILE* output_file;
	uint histories_to_process = 0;
	for( uint gantry_position = 0; gantry_position < parameters.NUM_FILES; gantry_position++ )
	{
		histories_to_process = histories_per_file[gantry_position];
		read_data_chunk( histories_to_process, gantry_position, gantry_position + 1 );
		sprintf( filename, "%s/%s%s%03d%s", PREPROCESSING_DIR, PROJECTION_DATA_BASENAME, "_", gantry_position, ".txt" );
		output_file = fopen (filename, "w");

		for( unsigned int i = 0; i < histories_to_process; i++ )
		{
			fprintf(output_file, "%3f %3f %3f %3f %3f %3f %3f %3f %3f\n", t_in_1_h[i], t_in_2_h[i], t_out_1_h[i], t_out_2_h[i], v_in_1_h[i], v_in_2_h[i], v_out_1_h[i], v_out_2_h[i], WEPL_h[i]);
		}
		fclose (output_file);
		initial_processing_memory_clean();
		histories_to_process = 0;
	} 
}
template<typename T> void array_2_disk( char* filename_base, char* filepath, DISK_WRITE_MODE format, T* data, const int x_max, const int y_max, const int z_max, const int num_elements, const bool single_file )
{
	char filename[256];
	std::ofstream output_file;
	int index;
	int z_start = 0;
	int z_end = single_file ? z_max : 1;
	int num_files = single_file ? 1 : z_max;
	//if( single_file )
	//{
	//	num_files = 1;
	//	z_end = z_max;
	//}
	for( int file = 0; file < num_files; file++)
	{
		//if( num_files == z_max )
		if( !single_file )
			sprintf( filename, "%s/%s_%d", filepath, filename_base, file );
		else
			sprintf( filename, "%s/%s", filepath, filename_base );			
		switch( format )
		{
			case TEXT	:	sprintf( filename, "%s.txt", filename );	
							puts("writing text format");
							output_file.open(filename);					break;
			case BINARY	:	sprintf( filename, "%s.bin", filepath );
							puts("writing binary format");
							output_file.open(filename, std::ofstream::binary);
		}
		//output_file.open(filename);		
		for(int z = z_start; z < z_end; z++)
		{			
			for(int y = 0; y < y_max; y++)
			{
				for(int x = 0; x < x_max; x++)
				{
					index = x + ( y * x_max ) + ( z * x_max * y_max );
					if( index >= num_elements )
						break;
					output_file << data[index] << " ";
				}	
				if( index >= num_elements )
					break;
				output_file << std::endl;
			}
			if( index >= num_elements )
				break;
		}
		z_start += 1;
		z_end += 1;
		output_file.close();
	}
}
template<typename T> void vector_2_disk( char* filename_base, char* filepath, DISK_WRITE_MODE format, std::vector<T> data, const int x_max, const int y_max, const int z_max, const bool single_file )
{
	char filename[256];
	std::ofstream output_file;
	int elements = data.size();
	int index;
	int z_start = 0;
	int z_end = single_file ? z_max : 1;
	int num_files = single_file ? 1 : z_max;
	//if( single_file )
	//{
	//	num_files = 1;
	//	z_end = z_max;
	//}
	for( int file = 0; file < num_files; file++)
	{
		//if( num_files == z_max )
		if( !single_file )
			sprintf( filename, "%s/%s_%d", filepath, filename_base, file );
		else
			sprintf( filename, "%s/%s", filepath, filename_base );			
switch( format )
		{
			case TEXT	:	sprintf( filename, "%s.txt", filename );	
							puts("writing text format");
							output_file.open(filename);					break;
			case BINARY	:	sprintf( filename, "%s.bin", filepath );
							puts("writing binary format");
							output_file.open(filename, std::ofstream::binary);
		}
		//output_file.open(filename);		
		for(int z = z_start; z < z_end; z++)
		{			
			for(int y = 0; y < y_max; y++)
			{
				for(int x = 0; x < x_max; x++)
				{
					index = x + ( y * x_max ) + ( z * x_max * y_max );
					if( index >= num_elements )
						break;
					output_file << data[index] << " ";
				}	
				if( index >= num_elements )
					break;
				output_file << std::endl;
			}
			if( index >= num_elements )
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
	unsigned int num_bin_members;
	for( int t_bin = 0; t_bin < parameters.T_BINS; t_bin++, bin++ )
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
				if( t_bin != parameters.T_BINS - 1 )
					fputs("\n", output_file);
		}
		bin_histories.resize(0);
		bin_histories.shrink_to_fit();
	}
}
template<typename T> void bins_2_disk( char* filename_base, char* filepath, DISK_WRITE_MODE format, const std::vector<int>& bin_numbers, const std::vector<T>& data, const BIN_ANALYSIS_TYPE type, const BIN_ANALYSIS_FOR which_bins, const BIN_ORGANIZATION bin_order, ... )
{
	//bins_2_disk( "WEPL_dist_pre_test2", empty_parameter, mean_WEPL_h, parameters.NUM_BINS, MEANS, ALL_BINS, BY_BIN );
	//bins_2_disk( "WEPL_dist_pre_test2", empty_parameter, sinogram_h, parameters.NUM_BINS, MEANS, ALL_BINS, BY_BIN );
	std::vector<int> angles;
	std::vector<int> angular_bins;
	std::vector<int> v_bins;
	if( which_bins == ALL_BINS )
	{
		angular_bins.resize( parameters.ANGULAR_BINS);
		v_bins.resize( parameters.V_BINS);
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
		std::transform(angles.begin(), angles.end(), angular_bins.begin(), std::bind2nd(std::divides<int>(), parameters.GANTRY_ANGLE_INTERVAL ) );
	}
	
	int num_angles = static_cast<int>( angular_bins.size() );
	int num_v_bins = static_cast<int>( v_bins.size() );
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
		angle = angular_bins[angular_bin] * parameters.GANTRY_ANGLE_INTERVAL;
		//printf("angle = %d\n", angular_bins[angular_bin]);
		//sprintf( filename, "%s%s/%s_%03d%s", filepath, filename_base, angle, ".txt" );
		//output_file = fopen (filename, "w");	
		sprintf( filename, "%s/%s_%03d", filepath, filename_base, angular_bin );	
		switch( format )
		{
			case TEXT	:	sprintf( filename, "%s.txt", filename );	
							output_file = fopen (filename, "w");				break;
			case BINARY	:	sprintf( filename, "%s.bin", filepath );
							output_file = fopen (filename, "wb");
		}
		for( int v_bin = 0; v_bin < num_v_bins; v_bin++)
		{			
			//printf("v bin = %d\n", v_bins[v_bin]);
			start_bin = angular_bins[angular_bin] * parameters.T_BINS + v_bins[v_bin] * parameters.ANGULAR_BINS * parameters.T_BINS;
			t_bins_2_disk( output_file, bin_numbers, data, type, bin_order, start_bin );
			if( v_bin != num_v_bins - 1 )
				fputs("\n", output_file);
		}	
		fclose (output_file);
	}
}
template<typename T> void t_bins_2_disk( FILE* output_file, int*& bin_numbers, T*& data, const unsigned int num_elements, const BIN_ANALYSIS_TYPE type, const BIN_ORGANIZATION bin_order, int bin )
{
	char* data_format = FLOAT_FORMAT;
	if( typeid(T) == typeid(int) )
		data_format = INT_FORMAT;
	if( typeid(T) == typeid(bool))
		data_format = BOOL_FORMAT;

	std::vector<T> bin_histories;
	//int data_elements = sizeof(data)/sizeof(float);
	unsigned int num_bin_members;
	for( int t_bin = 0; t_bin < parameters.T_BINS; t_bin++, bin++ )
	{
		if( bin_order == BY_HISTORY )
		{
			for( unsigned int i = 0; i < num_elements; i++ )
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
				if( t_bin != parameters.T_BINS - 1 )
					fputs("\n", output_file);
		}
		bin_histories.resize(0);
		bin_histories.shrink_to_fit();
	}
}
template<typename T>  void bins_2_disk( char* filename_base, char* filepath, DISK_WRITE_MODE format, int*& bin_numbers, T*& data, const int num_elements, const BIN_ANALYSIS_TYPE type, const BIN_ANALYSIS_FOR which_bins, const BIN_ORGANIZATION bin_order, ... )
{
	std::vector<int> angles;
	std::vector<int> angular_bins;
	std::vector<int> v_bins;
	if( which_bins == ALL_BINS )
	{
		angular_bins.resize( parameters.ANGULAR_BINS);
		v_bins.resize( parameters.V_BINS);
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
		std::transform(angles.begin(), angles.end(), angular_bins.begin(), std::bind2nd(std::divides<int>(), parameters.GANTRY_ANGLE_INTERVAL ) );
	}
	int num_angles = static_cast<int>( angular_bins.size() );
	int num_v_bins = static_cast<int>( v_bins.size() );
	char filename[256];
	int start_bin, angle;
	FILE* output_file;

	for( int angular_bin = 0; angular_bin < num_angles; angular_bin++)
	{
		angle = static_cast<int>(angular_bins[angular_bin] * parameters.GANTRY_ANGLE_INTERVAL);
		sprintf( filename, "%s/%s_%03d", filepath, filename_base, angular_bin );	
		switch( format )
		{
			case TEXT	:	sprintf( filename, "%s.txt", filename );	
							output_file = fopen (filename, "w");				break;
			case BINARY	:	sprintf( filename, "%s.bin", filepath );
							output_file = fopen (filename, "wb");
		}
		for( int v_bin = 0; v_bin < num_v_bins; v_bin++)
		{			
			start_bin = angular_bins[angular_bin] * parameters.T_BINS + v_bins[v_bin] * parameters.ANGULAR_BINS * parameters.T_BINS;
			t_bins_2_disk( output_file, bin_numbers, data, num_elements, type, bin_order, start_bin );
			if( v_bin != num_v_bins - 1 )
				fputs("\n", output_file);
		}	
		fclose (output_file);
	}
}
void combine_data_sets()
{
	char input_filename1[256];
	char input_filename2[256];
	char output_filename[256];
	const char INPUT_FOLDER1[]	   = "input_CTP404";
	const char INPUT_FOLDER2[]	   = "CTP404_4M";
	const char MERGED_FOLDER[]	   = "my_merged";

	char magic_number1[4], magic_number2[4];
	int version_id1, version_id2;
	int file_histories1, file_histories2, total_histories;

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

	for( unsigned int gantry_angle = 0; gantry_angle < 360; gantry_angle += static_cast<int>(parameters.GANTRY_ANGLE_INTERVAL) )
	{	
		cout << gantry_angle << endl;
		sprintf(input_filename1, "%s%s/%s_%03d%s", PROJECTION_DATA_DIR, INPUT_FOLDER1, PROJECTION_DATA_BASENAME, gantry_angle, PROJECTION_DATA_FILE_EXTENSION );
		sprintf(input_filename2, "%s%s/%s_%03d%s", PROJECTION_DATA_DIR, INPUT_FOLDER2, PROJECTION_DATA_BASENAME, gantry_angle, PROJECTION_DATA_FILE_EXTENSION );
		sprintf(output_filename, "%s%s/%s_%03d%s", PROJECTION_DATA_DIR, MERGED_FOLDER, PROJECTION_DATA_BASENAME, gantry_angle, PROJECTION_DATA_FILE_EXTENSION );

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
		total_histories = file_histories1 + file_histories2;
		fwrite( &total_histories, sizeof(int), 1, output_file );

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

		print_section_exit("Finished combining data sets.", "====>");
		pause_execution();
	}

}
void truncate_data_set(uint reduced_num_histories)
{
	char input_filename1[256];
	char output_filename[256];
	const char INPUT_FOLDER1[]	   = "input_CTP404";
	const char MERGED_FOLDER[]	   = "my_merged";

	char magic_number1[4];
	int version_id1;
	int file_histories1;
	//int histories_per_angle = reduced_num_histories/90;

	float projection_angle1, beam_energy1;
	int generation_date1, preprocess_date1;
	int phantom_name_size1, data_source_size1, prepared_by_size1;
	char *phantom_name1, *data_source1, *prepared_by1;

	float* t_in_1_h1, *t_in_2_h1; 
	float* t_out_1_h1, * t_out_2_h1;
	float* v_in_1_h1, * v_in_2_h1;
	float* v_out_1_h1, * v_out_2_h1;
	float* u_in_1_h1, * u_in_2_h1;
	float* u_out_1_h1, * u_out_2_h1;
	float* WEPL_h1;

	for( unsigned int gantry_angle = 0; gantry_angle < 360; gantry_angle += static_cast<int>(parameters.GANTRY_ANGLE_INTERVAL) )
	{	
		cout << gantry_angle << endl;
		sprintf(input_filename1, "%s%s/%s_%03d%s", PROJECTION_DATA_DIR, INPUT_FOLDER1, PROJECTION_DATA_BASENAME, gantry_angle, PROJECTION_DATA_FILE_EXTENSION );
		sprintf(output_filename, "%s%s/%s_%03d%s", PROJECTION_DATA_DIR, MERGED_FOLDER, PROJECTION_DATA_BASENAME, gantry_angle, PROJECTION_DATA_FILE_EXTENSION );

		printf("%s\n", input_filename1 );
		printf("%s\n", output_filename );

		FILE* input_file1 = fopen(input_filename1, "rb");
		FILE* output_file = fopen(output_filename, "wb");

		if( (input_file1 == NULL) || (output_file == NULL)  )
		{
			fputs( "Error Opening Data File:  Check that the directories are properly named.", stderr ); 
			exit_program_if(true);
		}

		fread(&magic_number1, sizeof(char), 4, input_file1 );
		fwrite( &magic_number1, sizeof(char), 4, output_file );
		//if( magic_number != MAGIC_NUMBER_CHECK ) 
		//{
		//	puts("Error: unknown file type (should be PCTD)!\n");
		//	exit_program_if(true);
		//}

		fread(&version_id1, sizeof(int), 1, input_file1 );
		fwrite( &version_id1, sizeof(int), 1, output_file );

		fread(&file_histories1, sizeof(int), 1, input_file1 );
		fwrite( &total_histories, sizeof(int), 1, output_file );

		puts("Reading headers from files...\n");
	
		fread(&projection_angle1, sizeof(float), 1, input_file1 );
		fwrite( &projection_angle1, sizeof(float), 1, output_file );
			
		fread(&beam_energy1, sizeof(float), 1, input_file1 );
		fwrite( &beam_energy1, sizeof(float), 1, output_file );

		fread(&generation_date1, sizeof(int), 1, input_file1 );
		fwrite( &generation_date1, sizeof(int), 1, output_file );

		fread(&preprocess_date1, sizeof(int), 1, input_file1 );
		fwrite( &preprocess_date1, sizeof(int), 1, output_file );

		fread(&phantom_name_size1, sizeof(int), 1, input_file1 );
		fwrite( &phantom_name_size1, sizeof(int), 1, output_file );

		phantom_name1 = (char*)malloc(phantom_name_size1);

		fread(phantom_name1, phantom_name_size1, 1, input_file1 );
		fwrite( phantom_name1, phantom_name_size1, 1, output_file );

		fread(&data_source_size1, sizeof(int), 1, input_file1 );
		fwrite( &data_source_size1, sizeof(int), 1, output_file );

		data_source1 = (char*)malloc(data_source_size1);

		fread(data_source1, data_source_size1, 1, input_file1 );
		fwrite( &data_source1, data_source_size1, 1, output_file );

		fread(&prepared_by_size1, sizeof(int), 1, input_file1 );
		fwrite( &prepared_by_size1, sizeof(int), 1, output_file );

		prepared_by1 = (char*)malloc(prepared_by_size1);

		fread(prepared_by1, prepared_by_size1, 1, input_file1 );
		fwrite( &prepared_by1, prepared_by_size1, 1, output_file );

		puts("Reading data from files...\n");

		t_in_1_h1 = (float*)calloc( file_histories1, sizeof(float ) );
		t_in_2_h1 = (float*)calloc( file_histories1, sizeof(float ) );
		t_out_1_h1 = (float*)calloc( file_histories1, sizeof(float ) );
		t_out_2_h1 = (float*)calloc( file_histories1, sizeof(float ) );
		v_in_1_h1 = (float*)calloc( file_histories1, sizeof(float ) );
		v_in_2_h1 = (float*)calloc( file_histories1, sizeof(float ) );
		v_out_1_h1 = (float*)calloc( file_histories1, sizeof(float ) );
		v_out_2_h1 = (float*)calloc( file_histories1, sizeof(float ) );
		u_in_1_h1 = (float*)calloc( file_histories1, sizeof(float ) );
		u_in_2_h1 = (float*)calloc( file_histories1, sizeof(float ) );
		u_out_1_h1 = (float*)calloc( file_histories1, sizeof(float ) );
		u_out_2_h1 = (float*)calloc( file_histories1, sizeof(float ) );
		WEPL_h1 = (float*)calloc( file_histories1, sizeof(float ) );
		
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

		fwrite( t_in_1_h1, sizeof(float), file_histories1, output_file );
		fwrite( t_in_2_h1, sizeof(float), file_histories1, output_file );
		fwrite( t_out_1_h1, sizeof(float), file_histories1, output_file );
		fwrite( t_out_2_h1, sizeof(float), file_histories1, output_file );
		
		fwrite( v_in_1_h1, sizeof(float), file_histories1, output_file );
		fwrite( v_in_2_h1, sizeof(float), file_histories1, output_file );
		fwrite( v_out_1_h1, sizeof(float), file_histories1, output_file );
		fwrite( v_out_2_h1, sizeof(float), file_histories1, output_file );
		
		fwrite( u_in_1_h1, sizeof(float), file_histories1, output_file );
		fwrite( u_in_2_h1, sizeof(float), file_histories1, output_file );
		fwrite( u_out_1_h1, sizeof(float), file_histories1, output_file );
		fwrite( u_out_2_h1, sizeof(float), file_histories1, output_file );
		
		fwrite( WEPL_h1, sizeof(float), file_histories1, output_file );
		
		free( t_in_1_h1 );
		free( t_in_2_h1 );
		free( t_out_1_h1 );
		free( t_out_2_h1 );
		
		free( v_in_1_h1 );
		free( v_in_2_h1 );
		free( v_out_1_h1 );
		free( v_out_2_h1 );
		
		free( u_in_1_h1 );
		free( u_in_2_h1 );
		free( u_out_1_h1 );
		free( u_out_2_h1 );
		
		free( WEPL_h1 );
		
		fclose(input_file1);						
		fclose(output_file);	

		print_section_exit("Finished combining data sets.", "====>");
		pause_execution();
	}

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
void timer( bool start, clock_t start_time, clock_t end_time)
{
	if( start )
		start_time = clock();
	else
	{
		end_time = clock();
		clock_t execution_clock_cycles = (end_time - start_time) - pause_cycles;
		double execution_time = double( execution_clock_cycles) / CLOCKS_PER_SEC;
		puts("-------------------------------------------------------------------------------");
		printf( "----------------- Total execution time : %3f [seconds] -------------------\n", execution_time );	
		puts("-------------------------------------------------------------------------------");
	}
}
void pause_execution()
{
	clock_t pause_start, pause_end;
	pause_start = clock();
	//char user_response[20];
	puts("Execution paused.  Hit enter to continue execution.\n");
	 //Clean the stream and ask for input
	std::cin.ignore ( std::numeric_limits<std::streamsize>::max(), '\n' );
	std::cin.get();

	pause_end = clock();
	pause_cycles += pause_end - pause_start;
}
void exit_program_if( bool early_exit)
{
	if( early_exit )
	{
		char user_response[20];		
		timer( STOP, program_start, program_end );
		//puts("*******************************************************************************");
		//puts("************** Press enter to exit and close the console window ***************");
		//puts("*******************************************************************************");
		print_section_header( "Press 'ENTER' to exit program and close console window", '*' );
		fgets(user_response, sizeof(user_response), stdin);
		exit(1);
	}
}
template<typename T> T* sequential_numbers( int start_number, int length )
{
	T* sequential_array = (T*)calloc(length,sizeof(T));
	std::iota( sequential_array, sequential_array + length, start_number );
	return sequential_array;
}
void bin_2_indexes( int& bin_num, int& t_bin, int& v_bin, int& angular_bin )
{
	// => bin = t_bin + angular_bin * parameters.T_BINS + v_bin * parameters.ANGULAR_BINS * parameters.T_BINS > 0
	while( bin_num - parameters.ANGULAR_BINS * parameters.T_BINS > 0 )
	{
		bin_num -= parameters.ANGULAR_BINS * parameters.T_BINS;
		v_bin++;
	}
	// => bin = t_bin + angular_bin * parameters.T_BINS > 0
	while( bin_num - parameters.T_BINS > 0 )
	{
		bin_num -= parameters.T_BINS;
		angular_bin++;
	}
	// => bin = t_bin > 0
	t_bin = bin_num;
}
/***********************************************************************************************************************************************************************************************************************/
/************************************************************************************************* IO Helper Functions *************************************************************************************************/
/***********************************************************************************************************************************************************************************************************************/
template<typename T> T cin_until_valid( T* valid_input, int num_valid, char* cin_message )
{
	T input;
	//cout << sizeof(*valid_input)  << " " << sizeof(valid_input)<< endl;
	//&& (std::find(valid_input, valid_input + sizeof(valid_input)/sizeof(*valid_input), input ) > &valid_input[sizeof(valid_input)/sizeof(*valid_input)-1])			
	while(puts(cin_message) || ((std::cin >> input) && (std::find(valid_input, valid_input + num_valid, input ) > &valid_input[num_valid-1]) ) )			
	{
		std::cout << "Invalid selection";
		std::cin.clear();
		std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
	}
	std::cin.clear();
	std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
	return input;
}
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
template<typename T> char((&minimize_trailing_zeros( T value, char(&buf)[64]) )[64])
{
	sprintf(buf, "%-.*G", 16, value);
	if( std::strchr(buf, '.' ) == NULL )
		strcat(buf, ".0");
	return buf;
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
bool directory_exists(char* dir_name )
{
	char mkdir_command[256]= "cd ";
	strcat( mkdir_command, dir_name);
	return !system( mkdir_command );
}
unsigned int create_unique_dir( char* dir_name )
{
	unsigned int i = 0;
	char mkdir_command[256];//= "mkdir ";
	char error_response[256];
	char* statement_beginning = "A subirectory or file ";
	char* statement_ending = " already exists";
	sprintf(mkdir_command, "mkdir \"%s\"", dir_name);
	//freopen("out.txt","a+",stdin);
	while( system(mkdir_command) )
	{
		//std::string text = buffer.str();
		//std::cout << "-> " << text << "<- " << endl;
		//printf( "-> %s <-\n", text );
		if( parameters.STDOUT_2_DISK )
		{
			sprintf(error_response, "%s %s_%d %s\n", statement_beginning, dir_name, i, statement_ending );
			puts(error_response);
		}
		if( (strlen(mkdir_command) + strlen(statement_beginning) + strlen(statement_ending) - 6) % CONSOLE_WINDOW_WIDTH != 0 )
			puts("");	
		sprintf(mkdir_command, "mkdir \"%s_%d\"", dir_name, ++i);
	}
	//fclose("out.txt");
	if( i != 0 )
		sprintf(dir_name, "%s_%d", dir_name, i);
	return i;
}
void get_dir_filenames(std::vector<std::string> &out, const std::string &directory)
{
	#if defined(_WIN32) || defined(_WIN64)
		HANDLE dir;
		WIN32_FIND_DATA file_data;
		if ((dir = FindFirstFile((directory + "/*").c_str(), &file_data)) == INVALID_HANDLE_VALUE)
			return; /* No files found */

		do {
			const std::string file_name = file_data.cFileName;
			const std::string full_file_name = directory + "/" + file_name;
			const bool is_directory = (file_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0;

			if (file_name[0] == '.')
				continue;

			if (is_directory)
				continue;

			out.push_back(full_file_name);
		} while (FindNextFile(dir, &file_data));

		FindClose(dir);
	#else
		DIR *dir;
		class dirent *ent;
		class stat st;

		dir = opendir(directory);
		while ((ent = readdir(dir)) != NULL) {
			const string file_name = ent->d_name;
			const string full_file_name = directory + "/" + file_name;

			if (file_name[0] == '.')
				continue;

			if (stat(full_file_name.c_str(), &st) == -1)
				continue;


			const bool is_directory = (st.st_mode & S_IFDIR) != 0;

			if (is_directory)
				continue;

			out.push_back(full_file_name);
		}
		closedir(dir);
	#endif
} // get_dir_filenames
void get_dir_filenames_matching(const std::string &directory, const std::string &beginning_with, std::vector<std::string> &path, std::vector<std::string> &file)
{
	#if defined(_WIN32) || defined(_WIN64)
		HANDLE dir;
		WIN32_FIND_DATA file_data;
		if ((dir = FindFirstFile((directory + "/" + beginning_with + "*").c_str(), &file_data)) == INVALID_HANDLE_VALUE)
			return; /* No files found */

		do {
			const std::string file_name = file_data.cFileName;
			const std::string full_file_name = directory + "/" + file_name;
			const bool is_directory = (file_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0;

			if (file_name[0] == '.')
				continue;

			if (is_directory)
				continue;

			path.push_back(full_file_name);
			file.push_back(file_name);
		} while (FindNextFile(dir, &file_data));

		FindClose(dir);
	#else
		DIR *dir;
		class dirent *ent;
		class stat st;

		dir = opendir(directory);
		while ((ent = readdir(dir)) != NULL) {
			const string file_name = ent->d_name;
			const string full_file_name = directory + "/" + file_name;

			if (file_name[0] == '.')
				continue;

			if (stat(full_file_name.c_str(), &st) == -1)
				continue;

			const bool is_directory = (st.st_mode & S_IFDIR) != 0;

			if (is_directory)
				continue;

			path.push_back(full_file_name);
			file.push_back(file_name);
		}
		closedir(dir);
	#endif
} // get_dir_filenames_matching
bool file_exists3 (const char* file_location) 
{
    #if defined(_WIN32) || defined(_WIN64)
		return file_location && ( PathFileExists (file_location) != 0 );
    #else
		struct stat sb;
		return file_location && (stat (file_location, &sb) == 0 ;
    #endif
} 
void rename_subdirectory_files(char* string_2_find, char* string_2_replace, char* string_replacement )
{
	char chnage_name_command[256];
	sprintf( chnage_name_command, "Get-ChildItem -Filter \"*%s*\" -Recurse | Rename-Item -NewName {$_.name -replace '%s','%s' }", string_2_find, string_2_replace, string_replacement);
	system("Powershell");
}
//void rotate_image()
//{
//	import_image( O*& import_into, char* directory, char* filename_base, DISK_WRITE_MODE format )
//}
void fgets_validated(char *line, int buf_size, FILE* input_file)
{
    bool done = false;
    while(!done)
    {
        if( fgets(line, buf_size, input_file ) == NULL )										// Read a line from the file
            return;
		while( *line == ' ' || *line == '\t' )
			line++;
        if( std::find_if(line, &line[strlen(line)], blank_line ) == ( &line[strlen(line)] ) )	// Skip lines with only "\n", "\t", and/or " "
			continue;
        else if( strncmp( line, "//", 2 ) == 0 )												// Skip any comment lines
            continue;
        else																					// Got a valid data line so return with this line
			done = true;    
    }
}
struct generic_IO_container read_key_value_pair( FILE* input_file )
{
	//puts("pair");
	char key[128], equal_sign[128], value[256], comments[512], buf[64];	
	char* open_quote_pos;
	const uint buf_size = 1024;
	char line[buf_size];
	size_t string_length;
	generic_IO_container input_value;
	// Remove leading spaces/tabs and return first line which does not begin with comment command "//" and is not blank
	fgets_validated(line, buf_size, input_file);	
	//puts("after fgets");
	// Having now read a non comment/blank line and removed leading spaces/tabs, parse it into {key}{=}{value}//{comments} format
	sscanf (line, " %s %s %s //%s", &key, &equal_sign, &value, &comments);
	input_value.key = (char*) calloc( strlen(key) + 1, sizeof(char));
	std::copy( key, &key[strlen(key)], input_value.key  );
	printf("key = %s ", input_value.key);

	if( strcmp( value, "true" ) == 0 || strcmp( value, "false" ) == 0 )
	{
		input_value.input_type_ID = BOOLEAN;
		input_value.boolean_input = ( strcmp( value, "true" ) == 0 );
		input_value.integer_input = static_cast<int>(input_value.boolean_input);
		input_value.double_input = static_cast<double>(input_value.integer_input);
		input_value.string_input = (char*) calloc( strlen(value) + 1, sizeof(char) );
		std::copy( value, &value[strlen(value)], input_value.string_input );
		printf( "is a boolean with value = %s\n", input_value.string_input );
		return input_value;
	}
	else
	{
		open_quote_pos = std::strchr(value, '"' );
		if( open_quote_pos == NULL)
		{
			if( std::strchr(value, '.' ) == NULL )
			{
				input_value.input_type_ID = INTEGER;
				sscanf( value, "%d", &input_value.integer_input );
				input_value.double_input = static_cast<double>(input_value.integer_input);
				input_value.boolean_input = static_cast<bool>(input_value.integer_input);
				input_value.string_input = (char*) calloc( 10, sizeof(char) );
				//itoa(input_value.integer_input, input_value.string_input, 10);
				sprintf(input_value.string_input,"%d",input_value.integer_input);
				printf( "is an integer with value = %d\n",input_value.integer_input );
				//printf( "is an integer with value = %s\n",input_value.string_input );	
			}
			else
			{
				input_value.input_type_ID = DOUBLE;
				sscanf( value, "%lf", &input_value.double_input );
				input_value.integer_input = static_cast<int>(input_value.double_input);
				input_value.boolean_input = static_cast<bool>(input_value.integer_input);
				input_value.string_input = (char*) calloc( 10, sizeof(char) );
				//itoa(input_value.double_input, input_value.string_input, 10);
				sprintf(input_value.string_input,"%lf",input_value.double_input);
				printf("is floating point with value %s\n", minimize_trailing_zeros( input_value.double_input,buf) );
				//printf("is floating point with value %s\n", input_value.string_input);
			}
		}
		else
		{
			input_value.input_type_ID = STRING;
			string_length =  std::strcspn ( open_quote_pos + 1, "\"" ) + 1;
			input_value.string_input = (char*) calloc( string_length + 1, sizeof(char) );
			std::copy( open_quote_pos + 1, &open_quote_pos[string_length], input_value.string_input );
			printf( "is a string with value \"%s\"\n",input_value.string_input );
		}
	}
	return input_value;
}
/***********************************************************************************************************************************************************************************************************************/
/************************************************************************************* Console Window Print Statement Functions  ***************************************************************************************/
/***********************************************************************************************************************************************************************************************************************/
void print_section_separator(char separation_char )
{
	char section_separator[CONSOLE_WINDOW_WIDTH];
	for( int n = 0; n < CONSOLE_WINDOW_WIDTH; n++ )
		section_separator[n] = separation_char;
	section_separator[CONSOLE_WINDOW_WIDTH - 1] = '\0';
	puts(section_separator);
}
void construct_header_line( char* statement, char separation_char, char* header )
{
	size_t length = strlen(statement), max_line_length = 70;
	size_t num_dashes = CONSOLE_WINDOW_WIDTH - length - 2;
	size_t leading_dashes = num_dashes / 2;
	size_t trailing_dashes = num_dashes - leading_dashes;
	size_t i = 0, line_length, index = 0;

	line_length = min(length - index, max_line_length);
	if( line_length < length - index )
		while( statement[index + line_length] != ' ' )
			line_length--;
	num_dashes = CONSOLE_WINDOW_WIDTH - line_length - 2;
	leading_dashes = num_dashes / 2;		
	trailing_dashes = num_dashes - leading_dashes;

	for( ; i < leading_dashes; i++ )
		header[i] = separation_char;
	if( statement[index] != ' ' )
		header[i++] = ' ';
	else
		header[i++] = separation_char;
	for( int j = 0; j < line_length; j++)
		header[i++] = statement[index++];
	header[i++] = ' ';
	for( int j = 0; j < trailing_dashes; j++)
		header[i++] = separation_char;
	header[CONSOLE_WINDOW_WIDTH - 1] = '\0';
}
void print_section_header( char* statement, char separation_char )
{
	print_section_separator(separation_char);
	char header[CONSOLE_WINDOW_WIDTH];
	size_t length = strlen(statement), max_line_length = 70;
	size_t num_dashes = CONSOLE_WINDOW_WIDTH - length - 2;
	size_t leading_dashes = num_dashes / 2;
	size_t trailing_dashes = num_dashes - leading_dashes;
	size_t i, line_length, index = 0;

	while( index < length )
	{
		i = 0;
		line_length = min(length - index, max_line_length);
		if( line_length < length - index )
			while( statement[index + line_length] != ' ' )
				line_length--;
		num_dashes = CONSOLE_WINDOW_WIDTH - line_length - 2;
		leading_dashes = num_dashes / 2;		
		trailing_dashes = num_dashes - leading_dashes;

		for( ; i < leading_dashes; i++ )
			header[i] = separation_char;
		if( statement[index] != ' ' )
			header[i++] = ' ';
		else
			header[i++] = separation_char;
		for( int j = 0; j < line_length; j++)
			header[i++] = statement[index++];
		header[i++] = ' ';
		for( int j = 0; j < trailing_dashes; j++)
			header[i++] = separation_char;
		header[CONSOLE_WINDOW_WIDTH - 1] = '\0';
		puts(header);		
	}
	print_section_separator(separation_char);
	puts("");
}
void print_section_exit( char* statement, char* leading_statement_chars )
{
	puts("");
	//print_section_separator('_');
	char header[CONSOLE_WINDOW_WIDTH];
	size_t length = strlen(statement), max_line_length = 70;
	//int additional_chars = CONSOLE_WINDOW_WIDTH - length - 2;
	size_t leading_chars = strlen(leading_statement_chars);
	size_t i = 0, index = 0, line_length;

	while( index < length )
	{
		//i = 0;
		line_length = min(length - index, max_line_length);
		if( line_length < length - index )
			while( statement[index + line_length] != ' ' )
				line_length--;
		//additional_chars = CONSOLE_WINDOW_WIDTH - line_length - 2;
		leading_chars = strlen(leading_statement_chars);

		if( index == 0 )
		{
			for( ; i < leading_chars; i++ )
				header[i] = leading_statement_chars[i];
		}
		else
		{
			for( ; i < leading_chars; i++ )
				header[i] = ' '; 
		}
		
		header[i++] = ' ';
		if( statement[index] == ' ' )
			index++;
		for( int j = 0; j < line_length; j++ )
			header[i++] = statement[index++];
		for( ; i < CONSOLE_WINDOW_WIDTH; i++ )
			header[i] = ' ';
		header[CONSOLE_WINDOW_WIDTH - 1] = '\0';
		puts(header);		
		i = 0;
	}
	//print_section_separator(separation_char);
	puts("");
}
void print_copyright_notice()
{
	//char notice[1024] = "";	
	const size_t string_length = strlen(COPYRIGHT_NOTICE) + strlen(HEADER_STATEMENT) + strlen(BLAKE_CONTACT) + strlen(REINHARD_CONTACT) + strlen(KEITH_CONTACT) + strlen(PANIZ_CONTACT);
	//char program_header[string_length + 6];
	char* program_header = (char*)calloc(string_length + 6, sizeof(char) );
	sprintf(program_header, "%s\n%s\n%s\n%s\n%s\n%s\n", COPYRIGHT_NOTICE, HEADER_STATEMENT, BLAKE_CONTACT, REINHARD_CONTACT, KEITH_CONTACT, PANIZ_CONTACT );
	puts(program_header);
}
void std_IO_2_disk()
{
	//system("C:/Users/Blake/Documents/Visual\" Studio 2010\"/Projects/robust_pct/x64/Debug/robust_pct.exe>log.out 2>&1");
	//system("CMD.exe 1>&outpt.txt 2>&1");
	//system("3>&1 4>&2");
	//system("trap 'exec 2>&4 1>&3' 0 1 2 3");
	//system("1>log.out 2>&1");
	//system("2>&1");
	//freopen(STDOUT_FILENAME,"w",stdout);
	//freopen(STDERR_FILENAME,"w",stderr);
	//system("2>& 1");
	//freopen(STDOUT_FILENAME,"w",stdout);
	//system("2>& 1");
	//system("1>log.out 2>&1");
	//if( parameters.STDOUT_2ISK )
	//{
	//	//system("rexec 3>&1 4>&2");
	//	//system("rtrap 'exec 2>&4 1>&3' 0 1 2 3");
	//	//system("rexec 1>log.out 2>&1");
	//	//system("robust_pct 2>& test.txt");
	//	//std::stringstream buffer;
	//	//std::streambuf * old = std::cout.rdbuf(buffer.rdbuf());
	//	//std::streambuf * old = std::cin.rdbuf(buffer.rdbuf());
	//	//std::streambuf * old2 = std::cerr.rdbuf(buffer.rdbuf());
	//	//freopen(STDOUT_FILENAME,"a+",stdin);
	//	//freopen(STDOUT_FILENAME,"w",stderr);
	//	freopen(STDOUT_FILENAME,"w+",stdout);
	//	//system("2>&1");
	//	//system("mkdir 2>& out.txt");
	//	//int file = open("myfile.txt", O_APPEND | O_WRONLY);
	//	//if(file < 0)    return 1;
 //
	//	//Now we redirect standard output to the file using dup2
	//	//if(dup2(file,1) < 0)    return 1;
	//	//system("script out.txt");
	//	//system("script C:/Users/Blake/Documents/Visual Studio 2010/Projects/robust_pct/robust_pct/out.txt");
	//}
}
/***********************************************************************************************************************************************************************************************************************/
/****************************************************************************** Log Maintaining Functions and Functions in Development *********************************************************************************/
/***********************************************************************************************************************************************************************************************************************/
LOG_OBJECT read_log()
{
	char entry_item[LOG_READ_BUFFER];
	LOG_LINE log_entry;
	LOG_OBJECT log;

	std::fstream log_file(LOG_FILENAME, std::fstream::in | std::fstream::out);
	
	if( !log_file.is_open() )
		log_file.open(LOG_FILENAME);
	else
	{
		log_file << std::skipws;
		while( !log_file.eof() )
		{		
			log_entry.clear();
			for( int i = 0; i < NUM_LOG_FIELDS; i++ )
			{
				log_file.getline(entry_item, LOG_READ_BUFFER, ',');
				log_entry.push_back(std::string(entry_item));
				puts(entry_item);
			}
			log.push_back(log_entry);
			log_file.get();			
		}
	}
	log_file.close();
	return log;
}
std::vector<int> scan_log_4_matches( LOG_OBJECT log )
{
	uint entry_number = LOG_FIRST_LINE;
	std::vector<int> match_info;
	bool object_match, scan_match, run_date_match, run_num_match, proj_date_match, preprocess_date_match, recon_date_match;
	bool projection_data_entry, preprocessing_data_entry, recon_data_entry;
	
	bool complete_match = true;
	//bool projection_match = true;
	bool empty_preprocessing, empty_reconstruction;
	//bool partial_match;
	bool incomplete;
	//char* empty_preprocess_date = (char*)calloc(DATE_ENTRIES_SIZE, sizeof(char) );
	//char* empty_recon_date = (char*)calloc(DATE_ENTRIES_SIZE, sizeof(char) );
	for( ; entry_number < log.size(); entry_number++ )
	{
		printf("Reading Log Entry %d : \n", entry_number  + 1 );

		object_match = ( (log[entry_number][0]).compare(std::string(OBJECT)) == 0);
		scan_match = ( (log[entry_number][1]).compare(std::string(SCAN_TYPE)) == 0);
		run_date_match = ( (log[entry_number][2]).compare(std::string(RUN_DATE)) == 0);
		run_num_match = ( (log[entry_number][3]).compare(std::string(RUN_NUMBER)) == 0);
		proj_date_match = ( (log[entry_number][4]).compare(std::string(PROJECTION_DATA_DATE)) == 0);
		
		empty_preprocessing = ( (log[entry_number][5]).compare(std::string("")) == 0);
		preprocess_date_match = ( (log[entry_number][5]).compare(std::string(PREPROCESS_DATE)) == 0);

		empty_reconstruction = ( (log[entry_number][6]).compare(std::string("")) == 0);
		recon_date_match = ( (log[entry_number][6]).compare(std::string(RECONSTRUCTION_DATE)) == 0);
		
		projection_data_entry = object_match && scan_match && run_date_match && run_num_match && proj_date_match;
		preprocessing_data_entry = projection_data_entry && preprocess_date_match;
		recon_data_entry = preprocessing_data_entry && recon_date_match;
		
		printf("object_match = %d\n", object_match );
		printf("scan_match = %d\n", scan_match );
		printf("run_date_match = %d\n", run_date_match );
		printf("run_num_match = %d\n", run_num_match );
		printf("proj_date_match = %d\n", proj_date_match );

		printf("empty_preprocessing = %d\n", empty_preprocessing );		
		printf("preprocess_date_match = %d\n", preprocess_date_match );

		printf("empty_reconstruction = %d\n", empty_reconstruction );
		printf("recon_date_match = %d\n", recon_date_match );

		printf("projection_data_entry = %d\n", projection_data_entry );
		printf("preprocessing_data_entry = %d\n", preprocessing_data_entry );
		printf("recon_data_entry = %d\n", recon_data_entry );
		
		puts("");
		incomplete = projection_data_entry & ( empty_preprocessing || empty_reconstruction );
		complete_match = projection_data_entry && preprocess_date_match && recon_date_match;
			
		if( incomplete || complete_match )
		{
			printf("Match found at %d\n", entry_number + 1 );
			match_info.push_back(complete_match);
			match_info.push_back(entry_number);
			match_info.push_back(empty_preprocessing);
			match_info.push_back(empty_reconstruction);
		}
	}
	//match_info.push_back(0);
	//match_info.push_back(entry_number);
	//match_info.push_back(0 );
	//match_info.push_back(0 );
	return match_info;
}
std::string format_log_entry(const char* entry_array, uint length  )
{
	std::string entry_string(entry_array);
	entry_string.resize(length, ' ');
	if( strlen(entry_array) > length )
	{
		puts("ERROR: length of string to be written is larger than container"); 
		exit_program_if(true);
	}
	return entry_string;
}
LOG_LINE construct_log_field_names( )
{
	LOG_LINE log_field_names;
	for( int i = 0; i < NUM_LOG_FIELDS; i++ )
		log_field_names.push_back( format_log_entry( LOG_ITEMS[i], LOG_FIELD_WIDTHS[i]  )  );
	return log_field_names;
}
LOG_LINE construct_log_entry()
{
	LOG_LINE current_run_entry;

	PREPROCESSED_BY = (char*) calloc(AUTHOR_ENTRIES_SIZE, sizeof(char) );
	RECONSTRUCTED_BY = (char*) calloc(AUTHOR_ENTRIES_SIZE, sizeof(char) );
	COMMENTS = (char*) calloc(COMMENT_ENTRIES_SIZE, sizeof(char) );
	CONFIG_LINK = (char*) calloc(CONFIG_LINK_SIZE, sizeof(char) );
	sprintf(PREPROCESSED_BY, "%s", "Blake Schultze" );
	sprintf(RECONSTRUCTED_BY, "%s", "Blake Schultze" );
	sprintf(CONFIG_LINK, "file:///%s/%s", PREPROCESSING_DIR, CONFIG_FILENAME );
	sprintf(COMMENTS, "%s", "Insert Comments Here" );

	current_run_entry.push_back( format_log_entry( OBJECT, OBJECT_ENTRIES_SIZE  )  );
	current_run_entry.push_back( format_log_entry( SCAN_TYPE, TYPE_ENTRIES_SIZE  )  );
	current_run_entry.push_back( format_log_entry( RUN_DATE, DATE_ENTRIES_SIZE  )  );
	current_run_entry.push_back( format_log_entry( RUN_NUMBER, RUN_NUM_ENTRIES_SIZE  )  );
	current_run_entry.push_back( format_log_entry( ACQUIRED_BY, AUTHOR_ENTRIES_SIZE  )  );
	current_run_entry.push_back( format_log_entry( PROJECTION_DATA_DATE, DATE_ENTRIES_SIZE  )  );
	current_run_entry.push_back( format_log_entry( CALIBRATED_BY, AUTHOR_ENTRIES_SIZE  )  );
	current_run_entry.push_back( format_log_entry( PREPROCESS_DATE, DATE_ENTRIES_SIZE  )  );
	current_run_entry.push_back( format_log_entry( PREPROCESSED_BY, AUTHOR_ENTRIES_SIZE  )  );

	if( parameters.PERFORM_RECONSTRUCTION )
		current_run_entry.push_back( format_log_entry( RECONSTRUCTION_DATE, DATE_ENTRIES_SIZE  )  );
	else
		current_run_entry.push_back( format_log_entry( "", DATE_ENTRIES_SIZE  )  );
	current_run_entry.push_back( format_log_entry( RECONSTRUCTED_BY, AUTHOR_ENTRIES_SIZE  )  );

	current_run_entry.push_back( format_log_entry( CONFIG_LINK, CONFIG_LINK_SIZE  )  );
	current_run_entry.push_back( format_log_entry( COMMENTS, COMMENT_ENTRIES_SIZE  )  );
	return current_run_entry;
}
void new_log_entry( LOG_OBJECT log_object )
{
	
	LOG_LINE new_entry = construct_log_entry();
	/*LOG_LINE new_entry;
	new_entry.push_back( "aab" );
	new_entry.push_back( "aab" );
	new_entry.push_back( "aab" );
	new_entry.push_back( "aab" );
	new_entry.push_back( "aab" );
	new_entry.push_back( "aab"  );
	new_entry.push_back( "aab" );*/

	int start_row = LOG_FIRST_LINE, end_row = static_cast<int>(log_object.size()) - 1, log_item = 0;
	while( (log_item < NUM_LOG_FIELDS) && (start_row <= end_row) )
	{
		while( ((new_entry[log_item]).compare(std::string(log_object[start_row][log_item])) > 0) && (start_row < end_row) )
			start_row++;
		while( ((new_entry[log_item]).compare(std::string(log_object[end_row][log_item])) < 0) && (start_row < end_row) ) 
			end_row--;			
		log_item++;
	}
	printf("Added as log entry = %d\n", start_row );
	log_object.insert(log_object.begin() + start_row, new_entry );
	write_log( log_object);

	//printf("%s\n", new_entry[j]);
	//printf("%s\n", std::string(log_object[start_row][j]));
	//printf("%s\n", std::string(log_object[end_row][j]));
}
void add_log_entry( LOG_ENTRIES entry )
{	
	LOG_OBJECT log_o = read_log();
	//print_log(log_o);
	std::string new_entry;
	std::vector<std::string> log_entry;
	std::vector<int> log_entry_info = scan_log_4_matches( log_o );
	uint index;
	size_t num_matches = log_entry_info.size() / 4;
	int complete_match_index = -1, incomplete_match_index = -1;	
	bool entry_present = false, incomplete_entry = false;
	bool empty_preprocessing, empty_reconstruction;
	
	cout << "Complete/incomplete matches = " << num_matches << endl;
	for( int i = 0; i < num_matches; i++ )
	{
		index = 4 * i;
		if( log_entry_info[index] )
		{		
			entry_present = true;
			complete_match_index = log_entry_info[index + 1];
			printf("Complete match found at entry %d\n", complete_match_index + 1 );
		}
		else 
		{
			incomplete_entry = true;
			incomplete_match_index = log_entry_info[index + 1];
			empty_preprocessing = log_entry_info[index + 2];
			empty_reconstruction = log_entry_info[index + 3];
			printf("Incomplete match found at entry %d\n", incomplete_match_index + 1 );
		}
	}
	
	if( incomplete_entry )
	{
		log_entry = log_o[incomplete_match_index];
		new_entry = format_log_entry(PREPROCESS_DATE, DATE_ENTRIES_SIZE  );
		if( empty_preprocessing )
			log_entry[PREPROCESS_DATE_L] = new_entry;
		new_entry = format_log_entry(RECONSTRUCTION_DATE, DATE_ENTRIES_SIZE  );
		if( empty_reconstruction )
			log_entry[RECONSTRUCTION_DATE_L] = new_entry;
		log_o[incomplete_match_index] = log_entry;
		write_log( log_o);
	}
	else if( !entry_present )
		new_log_entry( log_o );
	else
		puts("Entry already existed so either output data was overwritten or configurations were improperly specified.");
	
}
void print_log( LOG_OBJECT log )
{
	for( int i = 0; i < log.size(); i++ )
	{
		printf("Log Entry %d : \n", i );
		for( int j = 0; j < NUM_LOG_FIELDS; j++ )
			printf("%s\n", log[i][j] );
		puts("");
	}
}
void write_log( LOG_OBJECT log_object)
{
	std::ofstream log_file(LOG_FILENAME);
	
	if( !log_file.is_open() )
		log_file.open(LOG_FILENAME);
	else
	{
		for( int i = 0; i < log_object.size(); i++ )
		{
			log_file << std::noskipws;		
			for( int j = 0; j < NUM_LOG_FIELDS; j++ )
			{
				log_file <<  log_object[i][j] << ",";
			}
			log_file << endl;
		}
	}
	log_file.close();
}
void log_write_test()
{
	std::ofstream log_file(LOG_FILENAME);
	LOG_LINE log_field_names = construct_log_field_names();
	for( int i = 0; i < NUM_LOG_FIELDS; i++ )
		log_file << log_field_names[i] <<",";
	log_file << endl;

	ACQUIRED_BY = (char*)calloc( AUTHOR_ENTRIES_SIZE, sizeof(char) );
	PREPROCESSED_BY = (char*)calloc( AUTHOR_ENTRIES_SIZE, sizeof(char) );
	RECONSTRUCTED_BY = (char*)calloc( AUTHOR_ENTRIES_SIZE, sizeof(char) );
	COMMENTS = (char*)calloc( AUTHOR_ENTRIES_SIZE, sizeof(char) );
	CONFIG_LINK = (char*) calloc(CONFIG_LINK_SIZE, sizeof(char) );

	if( parameters.DATA_TYPE == EXPERIMENTAL )
		sprintf(ACQUIRED_BY, "%s", "UCSC" );
	else if( parameters.DATA_TYPE == SIMULATED_G )
		sprintf(ACQUIRED_BY, "%s", "Valentina Giacometti" );
	else if( parameters.DATA_TYPE == SIMULATED_T )
		sprintf(ACQUIRED_BY, "%s", "Tai Dou" );
	else
		sprintf(ACQUIRED_BY, "%s", "" );
	sprintf(PREPROCESSED_BY, "%s", "Blake Schultze" );
	sprintf(RECONSTRUCTED_BY, "%s", "Blake Schultze" );
	sprintf(CONFIG_LINK, "file:///%s/%s ", PREPROCESSING_DIR, CONFIG_FILENAME );
	sprintf(COMMENTS, "%s", "Insert Comments Here" );

	log_file << OBJECT <<"," << SCAN_TYPE << "," << RUN_DATE << "," << RUN_NUMBER << "," << ACQUIRED_BY << "," << PROJECTION_DATA_DATE << ","  << CALIBRATED_BY << ","  << PREPROCESS_DATE << ","  << PREPROCESSED_BY << "," << RECONSTRUCTION_DATE << "," << RECONSTRUCTED_BY << "," << CONFIG_LINK << "," << COMMENTS << ","<< endl;
	log_file << OBJECT <<"," << SCAN_TYPE << "," << RUN_DATE << "," << RUN_NUMBER << "," << ACQUIRED_BY << "," << PROJECTION_DATA_DATE << ","  << CALIBRATED_BY << ","  << PREPROCESS_DATE << ","  << PREPROCESSED_BY << "," << RECONSTRUCTION_DATE << "," << RECONSTRUCTED_BY << "," << CONFIG_LINK << "," << COMMENTS << ","<< endl;
	log_file << OBJECT <<"," << SCAN_TYPE << "," << RUN_DATE << "," << RUN_NUMBER << "," << ACQUIRED_BY << "," << PROJECTION_DATA_DATE << ","  << CALIBRATED_BY << ","  << PREPROCESS_DATE << ","  << PREPROCESSED_BY << "," << RECONSTRUCTION_DATE << "," << RECONSTRUCTED_BY << "," << CONFIG_LINK << "," << COMMENTS << ","<< endl;
	log_file << OBJECT <<"," << SCAN_TYPE << "," << RUN_DATE << "," << RUN_NUMBER << "," << ACQUIRED_BY << "," << PROJECTION_DATA_DATE << ","  << CALIBRATED_BY << ","  << PREPROCESS_DATE << ","  << PREPROCESSED_BY << "," << RECONSTRUCTION_DATE << "," << RECONSTRUCTED_BY << "," << CONFIG_LINK << "," << COMMENTS << ","<< endl;
	log_file << OBJECT <<"," << SCAN_TYPE << "," << RUN_DATE << "," << RUN_NUMBER << "," << ACQUIRED_BY << "," << PROJECTION_DATA_DATE << ","  << CALIBRATED_BY << ","  << PREPROCESS_DATE << ","  << PREPROCESSED_BY << "," << RECONSTRUCTION_DATE << "," << RECONSTRUCTED_BY << "," << CONFIG_LINK << "," << COMMENTS << ","<< endl;
	log_file << OBJECT <<"," << SCAN_TYPE << "," << RUN_DATE << "," << RUN_NUMBER << "," << ACQUIRED_BY << "," << PROJECTION_DATA_DATE << ","  << CALIBRATED_BY << ","  << PREPROCESS_DATE << ","  << PREPROCESSED_BY << "," << RECONSTRUCTION_DATE << "," << RECONSTRUCTED_BY << "," << CONFIG_LINK << "," << COMMENTS << ","<< endl;
	log_file << OBJECT <<"," << SCAN_TYPE << "," << RUN_DATE << "," << RUN_NUMBER << "," << ACQUIRED_BY << "," << PROJECTION_DATA_DATE << ","  << CALIBRATED_BY << ","  << PREPROCESS_DATE << ","  << PREPROCESSED_BY << "," << RECONSTRUCTION_DATE << "," << RECONSTRUCTED_BY << "," << CONFIG_LINK << "," << COMMENTS << ","<< endl;
	log_file << OBJECT <<"," << SCAN_TYPE << "," << RUN_DATE << "," << RUN_NUMBER << "," << ACQUIRED_BY << "," << PROJECTION_DATA_DATE << ","  << CALIBRATED_BY << ","  << PREPROCESS_DATE << ","  << PREPROCESSED_BY << "," << RECONSTRUCTION_DATE << "," << RECONSTRUCTED_BY << "," << CONFIG_LINK << "," << COMMENTS << ",";
	log_file.close();
}
void log_write_test2()
{
	std::ofstream log_file(LOG_FILENAME);
	log_file << OBJECT <<","  << SCAN_TYPE << ","  << RUN_DATE << ","  << RUN_NUMBER << ","  << PROJECTION_DATA_DATE << ","  << PREPROCESS_DATE << ","  << RECONSTRUCTION_DATE << ","  << endl;
	log_file << OBJECT <<","  << SCAN_TYPE << ","  << RUN_DATE << ","  << RUN_NUMBER << ","  << PROJECTION_DATA_DATE << ","  << PREPROCESS_DATE << ","  << RECONSTRUCTION_DATE << ","  << endl;
	log_file << OBJECT <<","  << SCAN_TYPE << ","  << RUN_DATE << ","  << RUN_NUMBER << ","  << PROJECTION_DATA_DATE << ","  << PREPROCESS_DATE << ","  << RECONSTRUCTION_DATE << ","  << endl;
	log_file << OBJECT <<","  << SCAN_TYPE << ","  << RUN_DATE << ","  << RUN_NUMBER << ","  << PROJECTION_DATA_DATE << ","  << PREPROCESS_DATE << ","  << RECONSTRUCTION_DATE << ","  << endl;
	log_file << OBJECT <<","  << SCAN_TYPE << ","  << RUN_DATE << ","  << RUN_NUMBER << ","  << PROJECTION_DATA_DATE << ","  << PREPROCESS_DATE << ","  << RECONSTRUCTION_DATE << ","  << endl;
	log_file << OBJECT <<","  << SCAN_TYPE << ","  << RUN_DATE << ","  << RUN_NUMBER << "a,"  << PROJECTION_DATA_DATE << "a,"  << PREPROCESS_DATE << "a,"  << RECONSTRUCTION_DATE << ","  << endl;
	log_file << OBJECT <<","  << SCAN_TYPE << "a,"  << RUN_DATE << "a,"  << RUN_NUMBER << "a,"  << PROJECTION_DATA_DATE << ","  << PREPROCESS_DATE << ","  << RECONSTRUCTION_DATE << ","  << endl;
	log_file << OBJECT <<","  << SCAN_TYPE << "a,"  << RUN_DATE << ","  << RUN_NUMBER << ","  << PROJECTION_DATA_DATE << ","  << PREPROCESS_DATE << ","  << RECONSTRUCTION_DATE << ","  << endl;
	log_file << OBJECT <<"a,"  << SCAN_TYPE << "b,"  << RUN_DATE << "a,"  << RUN_NUMBER << ","  << PROJECTION_DATA_DATE << ","  << PREPROCESS_DATE << ","  << RECONSTRUCTION_DATE << ","  << endl;
	log_file << OBJECT <<"a,"  << SCAN_TYPE << "c,"  << RUN_DATE << "a,"  << RUN_NUMBER << ","  << PROJECTION_DATA_DATE << ","  << PREPROCESS_DATE << ","  << RECONSTRUCTION_DATE << ","  << endl;
	log_file << OBJECT <<"b,"  << SCAN_TYPE << ","  << RUN_DATE << ","  << RUN_NUMBER << ","  << PROJECTION_DATA_DATE << ","  << PREPROCESS_DATE << ","  << RECONSTRUCTION_DATE << ","  << endl;
	log_file << OBJECT <<"b,"  << SCAN_TYPE << ","  << RUN_DATE << ","  << RUN_NUMBER << ","  << PROJECTION_DATA_DATE << ","  << PREPROCESS_DATE << ","  << RECONSTRUCTION_DATE << ","  << endl;
	log_file << OBJECT <<"b,"  << SCAN_TYPE << ","  << RUN_DATE << ","  << RUN_NUMBER << ","  << PROJECTION_DATA_DATE << ","  << PREPROCESS_DATE << ","  << RECONSTRUCTION_DATE << ","  << endl;

	log_file.close();
}
void log_write_test3()
{
	std::ofstream log_file(LOG_FILENAME);
	log_file << "aaa" <<","  << "aaa" << ","  << "aaa" << ","  << "aaa" << ","  << "aaa" << ","  << "aaa" << ","  << "aaa" << ","  << endl;
	log_file << "aaa" <<","  << "aaa" << ","  << "aaa" << ","  << "aaa" << ","  << "aaa" << ","  << "aaa" << ","  << "aaa" << ","  << endl;
	log_file << "aaa" <<","  << "aaa" << ","  << "aaa" << ","  << "aaa" << ","  << "aaa" << ","  << "aaa" << ","  << "aaa" << ","  << endl;
	log_file << "aaa" <<","  << "aaa" << ","  << "aaa" << ","  << "aaa" << ","  << "aaa" << ","  << "aaa" << ","  << "aaa" << ","  << endl;
	log_file << "aaa" <<","  << "aaa" << ","  << "aaa" << ","  << "aaa" << ","  << "aaa" << ","  << "aaa" << ","  << "aaa" << ","  << endl;
	log_file << "aab" <<","  << "aab" << ","  << "aaa" << ","  << "aaa" << ","  << "aaa" << ","  << "aaa" << ","  << "aaa" << ","  << endl;
	log_file << "aab" <<","  << "aab" << ","  << "aab" << ","  << "aaa" << ","  << "aaa" << ","  << "aaa" << ","  << "aaa" << ","  << endl;
	log_file << "aab" <<","  << "aab" << ","  << "aab" << ","  << "aab" << ","  << "aaa" << ","  << "aaa" << ","  << "aaa" << ","  << endl;
	log_file << "aab" <<","  << "aab" << ","  << "aab" << ","  << "aab" << ","  << "aab" << ","  << "aaa" << ","  << "aaa" << ","  << endl;
	/**/
	log_file << "aab" <<","  << "aab" << ","  << "aab" << ","  << "aab" << ","  << "aab" << ","  << "aab" << ","  << "aac" << ","  << endl;
	log_file << "aab" <<","  << "aab" << ","  << "aab" << ","  << "aab" << ","  << "aab" << ","  << "aab" << ","  << "aac" << ","  << endl;
	log_file << "aab" <<","  << "aab" << ","  << "aab" << ","  << "aab" << ","  << "aac" << ","  << "aaa" << ","  << "aab" << ","  << endl;
	log_file << "aab" <<","  << "aab" << ","  << "aab" << ","  << "aab" << ","  << "aac" << ","  << "aab" << ","  << "aab" << ","  << endl;
	log_file << "aab" <<","  << "aab" << ","  << "aab" << ","  << "aac" << ","  << "aab" << ","  << "aab" << ","  << "aab" << ","  << endl;
	log_file << "aac" <<","  << "aac" << ","  << "aaa" << ","  << "aaa" << ","  << "aaa" << ","  << "aaa" << ","  << "aaa" << ","  << endl;
	log_file << "aac" <<","  << "aac" << ","  << "aaa" << ","  << "aaa" << ","  << "aaa" << ","  << "aaa" << ","  << "aaa" << "," ;

	log_file.close();
}
/***********************************************************************************************************************************************************************************************************************/
/*********************************************************************************************** Device Helper Functions ***********************************************************************************************/
/***********************************************************************************************************************************************************************************************************************/

/***********************************************************************************************************************************************************************************************************************/
/********************************************************************************* Testing Host Functions and Functions in Development *********************************************************************************/
/***********************************************************************************************************************************************************************************************************************/
unsigned int read_MLP_endpoints2()
{
	char endpoints_filename[256];
	//sprintf(endpoints_filename, "%s%s/%s", OUTPUTIRECTORY, OUTPUT_FOLDER, MLP_ENDPOINTS_FILENAME );
	sprintf(endpoints_filename, "D:\\Visual Studio 2010\\Projects\\pCT_Reconstruction\\Output\\CTP404_merged\\MLP_endpoints_r=1.bin" );
	FILE* read_MLP_endpoints = fopen(endpoints_filename, "rb");
	unsigned int histories;
	fread( &histories, sizeof(unsigned int), 1, read_MLP_endpoints );
	printf("%d\n",histories );
	////D:\\Visual Studio 2010\\Projects\\pCT_Reconstruction\\Output\\CTP404_merged\\MLP_endpoints_r=1.bin
	//resize_vectors( histories );
	//shrink_vectors( histories );
	//voxel_x_vector.resize(histories);
	//voxel_y_vector.resize(histories);
	//voxel_z_vector.resize(histories);

	////fread( &reconstruction_histories, sizeof(unsigned int), 1, read_MLP_endpoints );
	//fread( &voxel_x_vector[0], sizeof(int), voxel_x_vector.size(), read_MLP_endpoints );
	//fread( &voxel_y_vector[0], sizeof(int), voxel_y_vector.size(), read_MLP_endpoints);
	//fread( &voxel_z_vector[0], sizeof(int), voxel_z_vector.size(), read_MLP_endpoints );
	//fread( &bin_num_vector[0], sizeof(int), bin_num_vector.size(), read_MLP_endpoints );
	//fread( &WEPL_vector[0], sizeof(float), WEPL_vector.size(), read_MLP_endpoints );
	//fread( &x_entry_vector[0], sizeof(float), x_entry_vector.size(), read_MLP_endpoints);
	//fread( &y_entry_vector[0], sizeof(float), y_entry_vector.size(), read_MLP_endpoints);
	//fread( &z_entry_vector[0], sizeof(float), z_entry_vector.size(), read_MLP_endpoints);
	//fread( &x_exit_vector[0], sizeof(float), x_exit_vector.size(), read_MLP_endpoints );
	//fread( &y_exit_vector[0], sizeof(float), y_exit_vector.size(), read_MLP_endpoints );
	//fread( &z_exit_vector[0], sizeof(float), z_exit_vector.size(), read_MLP_endpoints );
	//fread( &xy_entry_angle_vector[0], sizeof(float), xy_entry_angle_vector.size(), read_MLP_endpoints );
	//fread( &xz_entry_angle_vector[0], sizeof(float), xz_entry_angle_vector.size(), read_MLP_endpoints );
	//fread( &xy_exit_angle_vector[0], sizeof(float), xy_exit_angle_vector.size(), read_MLP_endpoints );
	//fread( &xz_exit_angle_vector[0], sizeof(float), xz_exit_angle_vector.size(), read_MLP_endpoints );
	fclose(read_MLP_endpoints);
	return histories;
}
__global__ void test_func_DROP_GPU( configurations* parameters, int num_histories, int* voxels_per_path, int* paths_d, float* result)
{
	int n = threadIdx.x;
	const int voxels_in_MLP = voxels_per_path[n];
	int start_path = 0;
	for( int i = 0; i < n; i++ )
		start_path += voxels_per_path[i];
		//float x[N];
	//__shared__ int a_shared[MAX_INTERSECTIONS];
	//int* a2 = (int*)calloc(10, sizeof(int) );
	__shared__ float* thread_path;
	thread_path = (float*)malloc(voxels_in_MLP * sizeof(float) );
	//std::copy(&paths_d[start_path], &paths_d[start_path + voxels_in_path], thread_path );
	//int* a_d;
	//cudaMalloc((void**)&a_d, 10*sizeof(int));
	//float *x = new float[N];
	for( int i = 0; i < voxels_in_MLP; i++ )
		thread_path[i] = 1;

	for( int i = 0; i < voxels_in_MLP; i++ )
		//result[n] += thread_path[i];
		result[n] = parameters->ANGULAR_BIN_SIZE;
	//result = thread_path;
	//atomicAdd(&result, thread_path );
	//std::copy(&paths_d[start_path], &paths_d[start_path + voxels_in_path], thread_path );
	//int* a_d;
	//cudaMalloc((void**)&a_d, 10*sizeof(int));
	//float *x = new float[N];
//	float sum = 0;
	/*for( int i = 0; i < voxels_in_path; i++ )
		result[n] += 1.0;*/
	//result[n] = start_path;
	/*for( int i = start_path; i < start_path + voxels_in_path; i++ )
		result[n] += paths_d[i];*/
	//result[n] = sum;
}
void test_func_DROP_host()
{
	const int num_histories = 10;
	int* voxels_per_path = (int*)calloc(num_histories, sizeof(int) );
	int total_voxels = 0;
	
	for( int i = 0; i < num_histories; i++ )
	{
		voxels_per_path[i] = 5*(i+1);
		total_voxels += voxels_per_path[i];
	}
	int* paths = (int*)calloc(total_voxels, sizeof(int) );
	for( int i = 0; i < total_voxels; i++ )
	{
		paths[i] = 1;
	}

	float* result = (float*)calloc(num_histories, sizeof(float) );
	float* result_d;
	int* voxels_per_path_d, *paths_d;
	//const int N = 50;
	cudaMalloc((void**)&result_d, num_histories*sizeof(float));
	cudaMalloc((void**)&voxels_per_path_d, num_histories*sizeof(int));
	cudaMalloc((void**)&paths_d, total_voxels*sizeof(int));


	cudaMemcpy( result_d, result, num_histories*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy( voxels_per_path_d, voxels_per_path, num_histories*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy( paths_d, paths, total_voxels*sizeof(int), cudaMemcpyHostToDevice);

	dim3 dimBlock( num_histories );
	dim3 dimGrid( 1 );   	
	test_func_DROP_GPU<<< dimGrid, dimBlock >>>(parameters_d, num_histories, voxels_per_path_d, paths_d, result_d);

	cudaMemcpy( result, result_d, num_histories*sizeof(float), cudaMemcpyDeviceToHost);
	//puts("hello");
	for( unsigned int i = 0; i < num_histories; i++)
	{
		printf("%d = %3f\n", i, result[i] );
		//cout << x[i] << endl; // -8.0
		//cout << y[i] << endl;
		//cout << z[i] << endl;
	}
}
int edge_coordinate_2_voxel( float coordinate, float zero_coordinate, float voxel_size, int increasing_direction, int moving_direction )
{
	int voxel = round(increasing_direction * ((coordinate - zero_coordinate ) / voxel_size) );
	return ( moving_direction == increasing_direction ) ? voxel : voxel - 1;
}
int edge_coordinate_2_voxel( double coordinate, double zero_coordinate, double voxel_size, int increasing_direction, int moving_direction )
{
	int voxel = round(increasing_direction * ((coordinate - zero_coordinate ) / voxel_size) );
	return ( moving_direction == increasing_direction ) ? voxel : voxel - 1;
}
void hull_entry_2_voxels( double x, double y, double z, double angle_xy, double angle_xz, int& voxel_x, int& voxel_y, int& voxel_z )
{
	int x_move = ( cos(angle_xy) >= 0 ) ? 1 : -1;
	int y_move = ( sin(angle_xy) >= 0 ) ? 1 : -1;
	int z_move = ( sin(angle_xz) >= 0 ) ? 1 : -1;
	cout << "x_move = " << x_move << endl;
	cout << "y_move = " << y_move << endl;
	cout << "z_move = " << z_move << endl;
	double vx = (x - parameters.X_ZERO_COORDINATE ) / parameters.VOXEL_WIDTH;
	double vy = (parameters.Y_ZERO_COORDINATE - y) / parameters.VOXEL_HEIGHT;
	double vz = (parameters.Z_ZERO_COORDINATE - z) / parameters.VOXEL_THICKNESS;
	cout << "vx = " << vx << endl;
	cout << "vy = " << vy << endl;
	cout << "vz = " << vz << endl;
	voxel_x = calculate_voxel( parameters.X_ZERO_COORDINATE, x, parameters.VOXEL_WIDTH );
	voxel_y = calculate_voxel( parameters.Y_ZERO_COORDINATE, y, parameters.VOXEL_HEIGHT );
	voxel_z = calculate_voxel( parameters.Z_ZERO_COORDINATE, z, parameters.VOXEL_THICKNESS );
	cout << "voxel_x = " << voxel_x << endl;
	cout << "voxel_y = " << voxel_y << endl;
	cout << "voxel_z = " << voxel_z << endl;
	double x_diff = abs(vx - round(vx) );
	double y_diff = abs(vy - round(vy) );
	double z_diff = abs(vz - round(vz) );
	cout << "x_diff = " << x_diff << endl;
	cout << "y_diff = " << y_diff << endl;
	cout << "z_diff = " << z_diff << endl;
	if( x_diff <= y_diff )
	{
		if( x_diff <= z_diff )
		{
			voxel_x = edge_coordinate_2_voxel( x, parameters.X_ZERO_COORDINATE, parameters.VOXEL_WIDTH, X_INCREASING_DIRECTION, x_move );			
			puts("x");
		}
		else
		{
			voxel_z = edge_coordinate_2_voxel( z, parameters.Z_ZERO_COORDINATE, parameters.VOXEL_THICKNESS, Z_INCREASING_DIRECTION, z_move );		
			puts("z");
		}
	}
	else
	{
		if( y_diff <= z_diff )
		{
			voxel_y = edge_coordinate_2_voxel( y, parameters.Y_ZERO_COORDINATE, parameters.VOXEL_HEIGHT, Y_INCREASING_DIRECTION, y_move );	
			puts("y");
		}
		else
		{
			voxel_z = edge_coordinate_2_voxel( z, parameters.Z_ZERO_COORDINATE, parameters.VOXEL_THICKNESS, Z_INCREASING_DIRECTION, z_move );		
			puts("z");
		}
	}
}
double lookup_sin( double angle )
{
	int lookup_index =  (angle - TRIG_TABLE_MIN ) / TRIG_TABLE_STEP + 0.5;
	return sin_table_h[lookup_index];
}
double lookup_cos( double angle )
{
	int lookup_index =  (angle - TRIG_TABLE_MIN ) / TRIG_TABLE_STEP + 0.5;
	return cos_table_h[lookup_index];
}
void test_trig_table()
{
	generate_trig_tables();
	import_trig_tables();
	//printf("%3f\n", sin_table_h[0] );
	//printf("%3f\n", cos_table_h[0] );
	int element = 0;
	float val = 0;
	printf("step = %3f\n", TRIG_TABLE_STEP );
	printf("min = %3f\n", TRIG_TABLE_MIN );
	printf("max = %3f\n", TRIG_TABLE_MAX );
	printf("range = %3f\n", TRIG_TABLE_RANGE );
	printf("TRIG_TABLE_ELEMENTS = %d\n", TRIG_TABLE_ELEMENTS );
	for( int i = 0; i <= TRIG_TABLE_ELEMENTS; i++ )
	{
		//element = static_cast<int>((i - TRIG_TABLE_MIN ) / TRIG_TABLE_STEP);
		val =  TRIG_TABLE_MIN + i * TRIG_TABLE_STEP;
		//printf("i = %3f\n", i );
		printf("val = %3f\n", val );
		printf("i = %d\n", i );
		printf("sin true val = %3f\n", sin(val) );
		printf("sin table val = %3f\n", sin_table_h[i] );
		printf("sin table lookup = %3f\n", lookup_sin(val) );
		printf("cos true val = %3f\n", cos(val) );
		printf("cos table val = %3f\n", cos_table_h[i] );
		printf("cos table lookup = %3f\n", lookup_cos(val) );
	}
	cout << cos(TRIG_TABLE_MAX-TRIG_TABLE_STEP) << endl;
	cout << sin(TRIG_TABLE_MAX-TRIG_TABLE_STEP) << endl;
	cout << cos(TRIG_TABLE_MAX) << endl;
	cout << sin(TRIG_TABLE_MAX) << endl;

	val = PI + TRIG_TABLE_STEP/3;
	element = (val - TRIG_TABLE_MIN ) / TRIG_TABLE_STEP + 0.5;
	cout << cos(val) << endl;
	cout << cos_table_h[element] << endl;
	cout << cos_table_h[element+1] << endl;
	cout << sin(val) << endl;
	cout << sin_table_h[element] << endl;
	cout << sin_table_h[element+1] << endl;

	double error_round = 0.0, error_truncate = 0.0, error_pt5 = 0.0, error_pt5plus = 0.0, error_truncplus1 = 0.0, error_roundplus1 = 0.0;
	float true_sin, true_cos;
	double error_round1 = 0.0, error_truncate1 = 0.0, error_pt51 = 0.0, error_pt5plus1 = 0.0, error_truncplus11 = 0.0, error_roundplus11 = 0.0;
	double max_error_round1 = 0.0, max_error_truncate1 = 0.0, max_error_pt51 = 0.0, max_error_pt5plus1 = 0.0, max_error_truncplus11 = 0.0, max_error_roundplus11 = 0.0;
	
	for( float angle = TRIG_TABLE_MIN; angle < TRIG_TABLE_MAX - 1.5*TRIG_TABLE_STEP; angle+=(TRIG_TABLE_STEP/10) )
	{
		//element = static_cast<int>((i - TRIG_TABLE_MIN ) / TRIG_TABLE_STEP);
		//val =  TRIG_TABLE_MIN + i * TRIG_TABLE_STEP;
		element = (angle - TRIG_TABLE_MIN ) / TRIG_TABLE_STEP + 0.5; 
		true_sin = sin(angle);
		true_cos = cos(angle);
		error_pt51 = abs(sin_table_h[element] - true_sin) + abs(cos_table_h[element] - true_cos);
		error_pt5 += error_pt51;
		//error_pt5 += abs(sin_table_h[element] - true_sin) + abs(cos_table_h[element] - true_cos);
		if( error_pt51 > max_error_pt51 )
			max_error_pt51 = error_pt51;
		error_pt5plus1 = abs(sin_table_h[element+1] - true_sin) + abs(cos_table_h[element+1] - true_cos);
		error_pt5plus += error_pt5plus1;
		//error_pt5plus += abs(sin_table_h[element+1] - true_sin) + abs(cos_table_h[element+1] - true_cos);
		if( error_pt5plus1 > max_error_pt5plus1 )
			max_error_pt5plus1 = error_pt5plus1;
		element = (angle - TRIG_TABLE_MIN ) / TRIG_TABLE_STEP; 
		error_truncate1 = abs(sin_table_h[element] - true_sin) + abs(cos_table_h[element] - true_cos);
		error_truncate += error_truncate1;
		//error_truncate += abs(sin_table_h[element] - true_sin) + abs(cos_table_h[element] - true_cos);
		if( error_truncate1 > max_error_truncate1 )
			max_error_truncate1 = error_truncate1;
		error_truncplus11 = abs(sin_table_h[element+1] - true_sin) + abs(cos_table_h[element+1] - true_cos);
		error_truncplus1 += error_truncplus11;
		//error_truncplus1 += abs(sin_table_h[element+1] - true_sin) + abs(cos_table_h[element+1] - true_cos);
		if( error_truncplus11 > max_error_truncplus11 )
			max_error_truncplus11 = error_truncplus11;
		element = round((angle - TRIG_TABLE_MIN ) / TRIG_TABLE_STEP); 
		error_round1 = abs(sin_table_h[element] - true_sin) + abs(cos_table_h[element] - true_cos);
		error_round += error_round1;
		//error_round += abs(sin_table_h[element] - true_sin) + abs(cos_table_h[element] - true_cos);
		if( error_round1 > max_error_round1 )
			max_error_round1 = error_round1;
		error_roundplus11 = abs(sin_table_h[element+1] - true_sin) + abs(cos_table_h[element+1] - true_cos);
		error_roundplus1 += error_roundplus11;
		//error_roundplus1 += abs(sin_table_h[element+1] - true_sin) + abs(cos_table_h[element+1] - true_cos);
		if( error_roundplus11 > max_error_roundplus11 )
			max_error_roundplus11 = error_roundplus11;
	}
	printf("TRIG_TABLE_ELEMENTS = %d\n", TRIG_TABLE_ELEMENTS );
	
	cout << max_error_pt51 << endl;
	cout << max_error_pt5plus1 << endl;
	cout << max_error_truncate1 << endl;
	cout << max_error_truncplus11 << endl;
	cout << max_error_round1 << endl;
	cout << max_error_roundplus11 << endl;
		
	cout << error_pt5 << endl;
	cout << error_pt5plus << endl;
	cout << error_truncate << endl;
	cout << error_truncplus1 << endl;
	cout << error_round << endl;
	cout << error_roundplus1 << endl;
}
void MLP5_test()
{
	double scattering_coefficient;
	double u_1 = parameters.MLP_U_STEP;
	double u_2 = 20.0;
	double depth_2_go = u_2 - u_1;
	double u_shifted = 5.0;	

	double poly_1_2, poly_2_3, poly_3_4, poly_2_6, poly_3_12;
	double u_1_poly_1_2, u_1_poly_2_3, u_1_poly_3_4, u_1_poly_2_6, u_1_poly_3_12;
	double u_2_poly_1_2, u_2_poly_2_3, u_2_poly_3_4, u_2_poly_2_6, u_2_poly_3_12;

	// Scattering Coefficient tables indices
	unsigned int sigma_table_index_step = static_cast<unsigned int>( parameters.MLP_U_STEP / DEPTH_TABLE_STEP + 0.5 );
	unsigned int sigma_1_coefficient_index = sigma_table_index_step;
	unsigned int sigma_2_coefficient_index = static_cast<unsigned int>( depth_2_go / DEPTH_TABLE_STEP + 0.5 );
	
	// Scattering polynomial indices
	unsigned int poly_table_index_step = static_cast<unsigned int>( parameters.MLP_U_STEP / POLY_TABLE_STEP + 0.5 );
	unsigned int u_1_poly_index = poly_table_index_step;
	unsigned int u_2_poly_index = static_cast<unsigned int>( u_2 / POLY_TABLE_STEP + 0.5 );

	//precalculated u_2 dependent terms (since u_2 does not change inside while loop)
	//double u_2_poly_3_12 = poly_3_12_h[u_2_poly_index];
	//double u_2_poly_2_6 = poly_2_6_h[u_2_poly_index];
	//double u_2_poly_1_2 = poly_1_2_h[u_2_poly_index];
	//double u_1_poly_1_2, u_1_poly_2_3;

	//while( u_1 < u_2 - parameters.MLP_U_STEP)
	while( depth_2_go > parameters.MLP_U_STEP )
	{
		cout << "u_1 = " << u_1 << endl;
		cout << "depth_2_go = " << depth_2_go << endl;
		cout << "sigma_1_coefficient_index = " << sigma_1_coefficient_index << endl;
		cout << "sigma_2_coefficient_index = " << sigma_2_coefficient_index << endl;
		cout << "u_1_poly_index = " << u_1_poly_index << endl;
		cout << "sigma_table_index_step = " << sigma_table_index_step << endl;
		cout << "poly_table_index_step = " << poly_table_index_step << endl;
			
		poly_1_2  = u_1 * ( A_0		   + u_1 * (A_1_OVER_2  + u_1 * (A_2_OVER_3  + u_1 * (A_3_OVER_4  + u_1 * (A_4_OVER_5   + u_1 * A_5_OVER_6   )))) );	// 1, 2, 3, 4, 5, 6
		poly_2_3  = pow(u_1, 2) * ( A_0_OVER_2 + u_1 * (A_1_OVER_3  + u_1 * (A_2_OVER_4  + u_1 * (A_3_OVER_5  + u_1 * (A_4_OVER_6   + u_1 * A_5_OVER_7   )))) );	// 2, 3, 4, 5, 6, 7
		poly_3_4 = pow(u_1, 3) * ( A_0_OVER_3 + u_1 * (A_1_OVER_4  + u_1 * (A_2_OVER_5  + u_1 * (A_3_OVER_6  + u_1 * (A_4_OVER_7   + u_1 * A_5_OVER_8   )))) );	// 3, 4, 5, 6, 7, 8
		poly_2_6 = pow(u_1, 2) * ( A_0_OVER_2 + u_1 * (A_1_OVER_6  + u_1 * (A_2_OVER_12 + u_1 * (A_3_OVER_20 + u_1 * (A_4_OVER_30  + u_1 * A_5_OVER_42  )))) );	// 2, 6, 12, 20, 30, 42
		poly_3_12 = pow(u_1, 3) * ( A_0_OVER_3 + u_1 * (A_1_OVER_12 + u_1 * (A_2_OVER_30 + u_1 * (A_3_OVER_60 + u_1 * (A_4_OVER_105 + u_1 * A_5_OVER_168 )))) );	// 3, 12, 30, 60, 105, 168		
		
		u_1_poly_1_2 = poly_1_2_h[u_1_poly_index];
		u_1_poly_2_3 = poly_2_3_h[u_1_poly_index];
		u_1_poly_3_4 = poly_3_4_h[u_1_poly_index];
		u_1_poly_2_6 = poly_2_6_h[u_1_poly_index];
		u_1_poly_3_12 = poly_3_12_h[u_1_poly_index];

		puts("u_1s");
		scattering_coefficient = pow( E_0 * ( 1 + 0.038 * log(u_1 / X0) ), 2.0 ) / X0;
		cout << scattering_coefficient << " vs. " << scattering_table_h[sigma_1_coefficient_index] << endl;
		cout << poly_1_2 << " vs. " << u_1_poly_1_2 << endl;
		cout << poly_2_3 << " vs. " << u_1_poly_2_3 << endl;
		cout << poly_3_4 << " vs. " << u_1_poly_3_4 << endl;
		cout << poly_2_6 << " vs. " << u_1_poly_2_6 << endl;
		cout << poly_3_12 << " vs. " << u_1_poly_3_12 << endl;
		
		poly_1_2  = u_2 * ( A_0		   + u_2 * (A_1_OVER_2  + u_2 * (A_2_OVER_3  + u_2 * (A_3_OVER_4  + u_2 * (A_4_OVER_5   + u_2 * A_5_OVER_6   )))) );	// 1, 2, 3, 4, 5, 6
		poly_2_3  = pow(u_2, 2) * ( A_0_OVER_2 + u_2 * (A_1_OVER_3  + u_2 * (A_2_OVER_4  + u_2 * (A_3_OVER_5  + u_2 * (A_4_OVER_6   + u_2 * A_5_OVER_7   )))) );	// 2, 3, 4, 5, 6, 7
		poly_3_4 = pow(u_2, 3) * ( A_0_OVER_3 + u_2 * (A_1_OVER_4  + u_2 * (A_2_OVER_5  + u_2 * (A_3_OVER_6  + u_2 * (A_4_OVER_7   + u_2 * A_5_OVER_8   )))) );	// 3, 4, 5, 6, 7, 8
		poly_2_6 = pow(u_2, 2) * ( A_0_OVER_2 + u_2 * (A_1_OVER_6  + u_2 * (A_2_OVER_12 + u_2 * (A_3_OVER_20 + u_2 * (A_4_OVER_30  + u_2 * A_5_OVER_42  )))) );	// 2, 6, 12, 20, 30, 42
		poly_3_12 = pow(u_2, 3) * ( A_0_OVER_3 + u_2 * (A_1_OVER_12 + u_2 * (A_2_OVER_30 + u_2 * (A_3_OVER_60 + u_2 * (A_4_OVER_105 + u_2 * A_5_OVER_168 )))) );	// 3, 12, 30, 60, 105, 168		
		
		u_2_poly_1_2 = poly_1_2_h[u_2_poly_index];
		u_2_poly_2_3 = poly_2_3_h[u_2_poly_index];
		u_2_poly_3_4 = poly_3_4_h[u_2_poly_index];
		u_2_poly_2_6 = poly_2_6_h[u_2_poly_index];
		u_2_poly_3_12 = poly_3_12_h[u_2_poly_index];

		puts("u_2s");
		
		scattering_coefficient = pow( E_0 * ( 1 + 0.038 * log((u_2-u_1) / X0) ), 2.0 ) / X0;
		cout << scattering_coefficient << " vs. " << scattering_table_h[sigma_2_coefficient_index] << endl;
		cout << poly_1_2 << " vs. " << u_2_poly_1_2 << endl;
		cout << poly_2_3 << " vs. " << u_2_poly_2_3 << endl;
		cout << poly_3_4 << " vs. " << u_2_poly_3_4 << endl;
		cout << poly_2_6 << " vs. " << u_2_poly_2_6 << endl;
		cout << poly_3_12 << " vs. " << u_2_poly_3_12 << endl;
		// Sigma_1: 3
		//sigma_t1 = poly_3_12_h[u_1_poly_index];										// poly_3_12(u_1)
		//sigma_t1_theta1 =  poly_2_6_h[u_1_poly_index];								// poly_2_6(u_1) 
		//sigma_theta1 = u_1_poly_1_2;												// poly_1_2(u_1)

		//// Sigma_2: 7 + 3 + 1 = 11									//
		//sigma_t2 =  u_2_poly_3_12 - pow(u_2, 2.0) * u_1_poly_1_2 + 2 * u_2 * u_1_poly_2_3 - poly_3_4_h[u_1_poly_index];	// poly_3_12(u_2) - u_2^2 * poly_1_2(u_1) +2u_2*(u_1) - poly_3_4(u_1)
		//sigma_t2_theta2 =  u_2_poly_2_6 - u_2 * u_1_poly_1_2 + u_1_poly_2_3;											// poly_2_6(u_2) - u_2*poly_1_2(u_1) + poly_2_3(u_1)
		//sigma_theta2 =  u_2_poly_1_2 - u_1_poly_1_2;																	// poly_1_2(u_2) - poly_1_2(u_1)	

		//// 4 + 1 + 1 + 1 = 7
		//determinant_Sigma_1 = scattering_table_h[sigma_1_coefficient_index] * ( sigma_t1 * sigma_theta1 - pow( sigma_t1_theta1, 2 ) );//ad-bc
		//Sigma_1I[0] = sigma_theta1 / determinant_Sigma_1;
		//Sigma_1I[1] = sigma_t1_theta1 / determinant_Sigma_1;	// negative sign is propagated to subsequent calculations instead of here 
		//Sigma_1I[2] = sigma_t1 / determinant_Sigma_1;			
		//	
		//// 4 + 1 + 1 + 1 = 7
		//determinant_Sigma_2 = scattering_table_h[sigma_2_coefficient_index] * ( sigma_t2 * sigma_theta2 - pow( sigma_t2_theta2, 2 ) );//ad-bc
		//Sigma_2I[0] = sigma_theta2 / determinant_Sigma_2;
		//Sigma_2I[1] = sigma_t2_theta2 / determinant_Sigma_2;	// negative sign is propagated to subsequent calculations instead of here 
		//Sigma_2I[2] = sigma_t2 / determinant_Sigma_2;

		u_1 += parameters.MLP_U_STEP;
		depth_2_go -= parameters.MLP_U_STEP;
		sigma_1_coefficient_index += sigma_table_index_step;
		sigma_2_coefficient_index -= sigma_table_index_step;
		u_1_poly_index += poly_table_index_step;

		
		pause_execution();
	}
}

void test_func()
{	
	generate_trig_tables();
	//free(sin_table_h);
	//free(cos_table_h);
	//import_trig_tables();
	//test_trig_table();
	//uint i = 0;
	generate_scattering_coefficient_table();
	//cout << scattering_table_h[0] << endl;
	//cout << scattering_table_h[1] << endl;
	//cout << scattering_table_h[++i] << endl;
	//cout << i << endl;
	//cout << scattering_table_h[i++] << endl;
	//cout << i << endl;
	//cout << scattering_table_h[2] << endl;
	//cout << scattering_table_h[DEPTH_TABLE_ELEMENTS-1] << endl;
	//free(scattering_table_h);
	//import_scattering_coefficient_table();
	generate_polynomial_tables();
	//MLP5_test();
	//import_polynomial_tables();
	//for( int i = 0; i <= POLY_TABLE_ELEMENTS; i++ )
	//{
	//	cout << " i1 = " << (i*POLY_TABLE_STEP) << " = " << poly_1_2_h[i] << endl;
	//}
	//for( int i = 0; i <= POLY_TABLE_ELEMENTS; i++ )
	//{
	//	cout << " i2 = " << (i*POLY_TABLE_STEP) << " = " << poly_2_3_h[i] << endl;
	//}
	//for( int i = 0; i <= POLY_TABLE_ELEMENTS; i++ )
	//{
	//	cout << " i3 = " << (i*POLY_TABLE_STEP) << " = " << poly_3_4_h[i] << endl;
	//}
	//for( int i = 0; i <= POLY_TABLE_ELEMENTS; i++ )
	//{
	//	cout << " i4 = " << (i*POLY_TABLE_STEP) << " = " << poly_2_6_h[i] << endl;
	//}
	//for( int i = 0; i <= POLY_TABLE_ELEMENTS; i++ )
	//{
	//	cout << " i5 = " << (i*POLY_TABLE_STEP) << " = " << poly_3_12_h[i] << endl;
	//}
	//cout << poly_1_2_h[0] << endl;
	//cout << poly_1_2_h[1] << endl;
	//cout << poly_1_2_h[POLY_TABLE_ELEMENTS-1] << endl;
	//cout << (POLY_TABLE_ELEMENTS * POLY_TABLE_STEP)<< endl;
	/*free(poly_1_2_h);
	free(poly_2_3_h);
	free(poly_3_4_h);
	free(poly_2_6_h);
	free(poly_3_12_h);
	import_polynomial_tables();
	cout << poly_1_2_h[0] << endl;
	cout << poly_1_2_h[1] << endl;
	cout << poly_1_2_h[POLY_TABLE_ELEMENTS] << endl;

	cout << poly_2_3_h[0] << endl;
	cout << poly_2_3_h[1] << endl;
	cout << poly_2_3_h[POLY_TABLE_ELEMENTS] << endl;

	cout << poly_3_4_h[0] << endl;
	cout << poly_3_4_h[1] << endl;
	cout << poly_3_4_h[POLY_TABLE_ELEMENTS] << endl;

	cout << poly_2_6_h[0] << endl;
	cout << poly_2_6_h[1] << endl;
	cout << poly_2_6_h[POLY_TABLE_ELEMENTS] << endl;

	cout << poly_3_12_h[0] << endl;
	cout << poly_3_12_h[1] << endl;
	cout << poly_3_12_h[POLY_TABLE_ELEMENTS] << endl;*/

	//int orig[] = {1,2,3,4,5,6,7,8};
	//std::vector<int> orig_vec( orig, orig + 8);
	//std::vector<int> new_vec(8,1);
	//orig_vec = new_vec;
	//new_vec.~vector();
	////new_vec[0] = 10;
	//for( int i = 0; i < 8; i++)
	//	cout << orig_vec[i] << endl;
	/*reconstruction_histories = 10;
	for( int i = 0; i < reconstruction_histories; i++ )
		WEPL_vector.push_back( 2.35 );
	export_WEPL();
	for( int i = 0; i < reconstruction_histories; i++ )
		WEPL_vector[i] = 1.35;
	for( int i = 0; i < WEPL_vector.size(); i++ )
		cout << WEPL_vector[i] << endl;
	uint histories = import_WEPL();
	cout << histories << endl << endl;
	for( int i = 0; i < WEPL_vector.size(); i++ )
		cout << WEPL_vector[i] << endl;*/

	//image_reconstruction_import_preprocessing();
	//cout << directory_exists( PREPROCESSING_DIR ) << endl;
	//int moving_x = 1, moving_y = 1, moving_z = 1;
	/*int voxel_x, voxel_y, voxel_z;
	double x = parameters.X_ZERO_COORDINATE + 2 * parameters.VOXEL_WIDTH;
	double y = parameters.Y_ZERO_COORDINATE - 4.5 * parameters.VOXEL_HEIGHT;
	double z = parameters.Z_ZERO_COORDINATE - 8.9 * parameters.VOXEL_THICKNESS;
	double angle_xy = 45 * ANGLE_TO_RADIANS;
	double angle_xz = 45 * ANGLE_TO_RADIANS;
	hull_entry_2_voxels( x, y, z, angle_xy, angle_xz, voxel_x, voxel_y, voxel_z );
	cout << "voxel_x = " << voxel_x << endl;
	cout << "voxel_y = " << voxel_y << endl;
	cout << "voxel_z = " << voxel_z << endl;*/
	/*
	int moving_direction = 1;
	int voxel = 7;
	double x = edge_coordinate( parameters.X_ZERO_COORDINATE, voxel, parameters.VOXEL_WIDTH, X_INCREASING_DIRECTION, moving_direction );
	cout << x << endl;
	int voxel_x = edge_coordinate_2_voxel( x, parameters.X_ZERO_COORDINATE, parameters.VOXEL_WIDTH, X_INCREASING_DIRECTION, moving_direction );
	cout << vx << endl;
	
	moving_direction = 1;
	voxel = 1;
	x = edge_coordinate( parameters.X_ZERO_COORDINATE, voxel, parameters.VOXEL_WIDTH, X_INCREASING_DIRECTION, moving_direction );
	cout << x << endl;
	vx = edge_coordinate_2_voxel( x,parameters.X_ZERO_COORDINATE, parameters.VOXEL_WIDTH, X_INCREASING_DIRECTION, moving_direction );
	cout << vx << endl;
	
	moving_direction = -1;
	voxel = 4;
	x = edge_coordinate( parameters.X_ZERO_COORDINATE, voxel, parameters.VOXEL_WIDTH, X_INCREASING_DIRECTION, moving_direction );
	cout << x << endl;
	vx = edge_coordinate_2_voxel( x,parameters.X_ZERO_COORDINATE, parameters.VOXEL_WIDTH, X_INCREASING_DIRECTION, moving_direction );
	cout << vx << endl;
	
	moving_direction = -1;
	voxel = 4;
	x = edge_coordinate( parameters.X_ZERO_COORDINATE, voxel, parameters.VOXEL_WIDTH, X_INCREASING_DIRECTION, moving_direction );
	cout << x << endl;
	vx = edge_coordinate_2_voxel( x,parameters.X_ZERO_COORDINATE, parameters.VOXEL_WIDTH, X_INCREASING_DIRECTION, moving_direction );
	cout << vx << endl;

	moving_direction = 1;
	voxel = 7;
	double y = edge_coordinate( parameters.Y_ZERO_COORDINATE, voxel, parameters.VOXEL_HEIGHT, Y_INCREASING_DIRECTION, moving_direction );
	cout << y << endl;
	int vy = edge_coordinate_2_voxel( y,parameters.Y_ZERO_COORDINATE, parameters.VOXEL_HEIGHT, Y_INCREASING_DIRECTION, moving_direction );
	cout << vy << endl;
	
	moving_direction = 1;
	voxel = 9;
	y = edge_coordinate( parameters.Y_ZERO_COORDINATE, voxel, parameters.VOXEL_HEIGHT, Y_INCREASING_DIRECTION, moving_direction );
	cout << y << endl;
	vy = edge_coordinate_2_voxel( y,parameters.Y_ZERO_COORDINATE, parameters.VOXEL_HEIGHT, Y_INCREASING_DIRECTION, moving_direction );
	cout << vy << endl;
	
	moving_direction = -1;
	voxel = 7;
	y = edge_coordinate( parameters.Y_ZERO_COORDINATE, voxel, parameters.VOXEL_HEIGHT, Y_INCREASING_DIRECTION, moving_direction );
	cout << y << endl;
	vy = edge_coordinate_2_voxel( y,parameters.Y_ZERO_COORDINATE, parameters.VOXEL_HEIGHT, Y_INCREASING_DIRECTION, moving_direction );
	cout << vy << endl;
	
	moving_direction = -1;
	voxel = 4;
	y = edge_coordinate( parameters.Y_ZERO_COORDINATE, voxel, parameters.VOXEL_HEIGHT, Y_INCREASING_DIRECTION, moving_direction );
	cout << y << endl;
	vy = edge_coordinate_2_voxel( y,parameters.Y_ZERO_COORDINATE, parameters.VOXEL_HEIGHT, Y_INCREASING_DIRECTION, moving_direction );
	cout << vy << endl;*/
	//int vx = edge_coordinate_2_voxel( -parameters.RECON_CYL_RADIUS,parameters.X_ZERO_COORDINATE, parameters.VOXEL_WIDTH, X_INCREASING_DIRECTION, 1 );
	//cout << vx << endl;
	//vx = edge_coordinate_2_voxel( -parameters.RECON_CYL_RADIUS,parameters.X_ZERO_COORDINATE, parameters.VOXEL_WIDTH, X_INCREASING_DIRECTION, -1 );
	//cout << vx << endl;
	//vx = edge_coordinate_2_voxel( -parameters.RECON_CYL_RADIUS + parameters.VOXEL_WIDTH,parameters.X_ZERO_COORDINATE, parameters.VOXEL_WIDTH, X_INCREASING_DIRECTION, 1 );
	//cout << vx << endl;
	//vx = edge_coordinate_2_voxel( -parameters.RECON_CYL_RADIUS + parameters.VOXEL_WIDTH,parameters.X_ZERO_COORDINATE, parameters.VOXEL_WIDTH, X_INCREASING_DIRECTION, -1 );
	//cout << vx << endl << endl;

	////cout << parameters.RECON_CYL_RADIUS << endl;
	////cout << parameters.VOXEL_HEIGHT << endl;
	////cout << parameters.Y_ZERO_COORDINATE << endl;
	//int vy = edge_coordinate_2_voxel( parameters.RECON_CYL_RADIUS,parameters.Y_ZERO_COORDINATE, parameters.VOXEL_HEIGHT, Y_INCREASING_DIRECTION, 1 );
	//cout << vy << endl;
	//vy = edge_coordinate_2_voxel( parameters.RECON_CYL_RADIUS,parameters.Y_ZERO_COORDINATE, parameters.VOXEL_HEIGHT, Y_INCREASING_DIRECTION, -1 );
	//cout << vy << endl;
	//vy = edge_coordinate_2_voxel( parameters.RECON_CYL_RADIUS - parameters.VOXEL_HEIGHT,parameters.Y_ZERO_COORDINATE, parameters.VOXEL_HEIGHT, Y_INCREASING_DIRECTION, 1 );
	//cout << vy << endl;
	//vy = edge_coordinate_2_voxel( parameters.RECON_CYL_RADIUS - parameters.VOXEL_HEIGHT,parameters.Y_ZERO_COORDINATE, parameters.VOXEL_HEIGHT, Y_INCREASING_DIRECTION, -1 );
	//cout << vy << endl;

	

	//test_transfer();
	//test_func_DROP_host();
	/*char* path = "C:/Users/Blake/Documents/Visual Studio 2010/Projects/robust_pct/robust_pct/MLP_read_test.bin";
	FILE* MLP_out_file = fopen(path, "wb");
	int num_entries_out = 50;
	uint* MLP_test_out = (uint*) calloc( num_entries_out, sizeof(uint) );
	for( int i = 0; i < num_entries_out; i++ )
		MLP_test_out[i] = i;
	fwrite( &num_entries_out, sizeof( uint), 1, MLP_out_file );
	fwrite( MLP_test_out, sizeof( uint), num_entries_out, MLP_out_file );
	fclose(MLP_out_file);

	
	int num_entries_in;
	FILE* MLP_in_file = fopen(path, "rb");
	fread( &num_entries_in, sizeof(int), 1, MLP_in_file );
	long int begin_paths = ftell(MLP_in_file);
	uint* MLP_test_in = (uint*) calloc( num_entries_in, sizeof(uint) );

	fread( MLP_test_in, sizeof(unsigned int), num_entries_in, MLP_in_file );
	for( int i = 0; i < num_entries_in; i++ )
		printf("%d\n", MLP_test_in[i] );

	fseek( MLP_in_file, begin_paths, SEEK_SET );
	uint* MLP_test_in2 = (uint*) calloc( num_entries_in, sizeof(uint) );

	fread( MLP_test_in2, sizeof(unsigned int), num_entries_in, MLP_in_file );
	for( int i = 0; i < num_entries_in; i++ )
		printf("%d\n", MLP_test_in2[i] );*/
	//bool data_sizes_match = ( ( MLP_histories == WEPL_histories ) && (WEPL_histories == voxels_per_path_histories ) && ( voxels_per_path_histories == hull_intersection_histories ) );
	
	
	//long int begin_paths = ftell(MLP_in_file);
	//fseek( MLP_in_file, begin_paths, SEEK_SET );

	//apply_execution_arguments( num_arguments, arguments );
	//set_execution_date();
	//CONFIG_OBJECT config_object = config_file_2_object();			
	//read_config_file();
	//set_dependent_parameters();
	//set_IO_file_extensions();
	//set_IO_directories();
	//set_IO_filenames();
	//set_IO_filepaths();
	//set_images_2_use();
	//existing_data_check();
	//parameters_2_GPU();
	////print_copyright_notice();
	////view_config_file();
	//existing_data_check();

	//std::vector<float> test_vec(15, 2.0);
	//for( int i = 0; i < static_cast<int>(test_vec.size()); i++ )
	//	cout << test_vec[i] << endl;
	////vector_2_disk("test_vec", "D:\\pCT_Data\\Output\\CTP404\\CTP404_merged\\", TEXT, test_vec, test_vec.size(), 1, 1, true);
	//char* filename = "D:\\pCT_Data\\Output\\CTP404\\CTP404_merged\\test_vec.bin";
	//FILE* outfile = fopen(filename, "wb");
	//fwrite( &test_vec[0], sizeof(float), test_vec.size(), outfile );
	//fclose(outfile);

	//FILE* infile = fopen(filename, "rb");
	//std::vector<float> invec;
	//invec.reserve(test_vec.size());
	//invec.resize(test_vec.size());
	//fread( &invec[0], sizeof(float), test_vec.size(), infile );
	//for( int i = 0; i < test_vec.size(); i++ )
	//	cout << invec[i] << endl;
	//fclose(infile);
	/*cout << file_exists3(MLP_PATH) << endl;
	cout << file_exists3(WEPL_PATH) << endl;
	cout << file_exists3(HISTORIES_PATH) << endl;*/

	/*cout << file_exists3(MLP_PATH) << endl;
	cout << file_exists3(WEPL_PATH) << endl;
	cout << file_exists3(HISTORIES_PATH) << endl;*/

	//bool MLPs =  file_exists3(MLP_PATH);
	//bool WEPLs =  file_exists3(WEPL_PATH);
	//bool histories_p =  file_exists3(HISTORIES_PATH);
	//cout << std::string(MLP_PATH) << endl;
	//cout << MLPs << endl;
	//cout << WEPLs << endl;
	//cout << histories_p << endl;
		//view_config_file();
	//initializations();
	//count_histories_v0();
	//log_write_test();
	//log_write_test2();
	//log_write_test3();
	//LOG_OBJECT log_o = read_log();
	//new_log_entry( log_o );
	//print_log( log_o );
	//write_log(log_o);
	
	//char* FBP_path = "C:/Users/Blake/Documents/Visual Studio 2010/Projects/robust_pct/robust_pct/Images";
	////char* filename = "FBP_image_h";
	//char* FBP_median_basename = "FBP_median";
	//char* FBP_average_basename = "FBP_average";
	//char* filename = "FBP_med_new7";
	////float* FBP = (float*) calloc( parameters.NUM_VOXELS, sizeof(float) );
	//printf("%d\n", parameters.ROWS );
	//printf("%d\n", parameters.COLUMNS );
	//printf("%d\n", parameters.SLICES );
	//printf("%d\n", parameters.NUM_VOXELS );
	//printf("%3f\n", parameters.SSD_V_SIZE );
	//printf("%3f\n", parameters.RECON_CYL_HEIGHT );
	//printf("%d\n", parameters.NUM_FILES);							// *[#] 1 file per gantry angle per translation
	//printf("%d\n", parameters.GANTRY_ANGLES);					// *[#] Total number of projection angles
	//printf("%d\n", parameters.T_BINS);					// *[#] Number of bins (i.e. quantization levels) for t (lateral) direction 
	//printf("%d\n", parameters.V_BINS);					// *[#] Number of bins (i.e. quantization levels) for v (vertical) direction
	//printf("%d\n", parameters.ANGULAR_BINS);					// *[#] Number of bins (i.e. quantization levels) for path angle 
	//printf("%d\n", parameters.NUM_BINS);						// *[#] Total number of bins corresponding to possible 3-tuples [ANGULAR_BIN, T_BIN, V_BIN],
	//printf("%d\n", parameters.SLICES);				// *[#] Number of voxels in the z direction (i.e., number of slices) of image
	//printf("%d\n", parameters.NUM_VOXELS);							// *[#] Total number of voxels (i.e. 3-tuples [column, row, slice]) in image	
	//printf("%3f\n", parameters.RECON_CYLIAMETER);							// *[cm] Diameter of reconstruction cylinder
	//printf("%3f\n", parameters.RECON_CYL_HEIGHT);									// *[cm] Height of reconstruction cylinder
	//printf("%3f\n", parameters.IMAGE_WIDTH);									// *[cm] Distance between left and right edges of each slice in image
	//printf("%3f\n", parameters.IMAGE_HEIGHT);									// *[cm] Distance between top and bottom edges of each slice in image
	//printf("%3f\n", parameters.IMAGE_THICKNESS);									// *[cm] Distance between bottom of bottom slice and top of the top slice of image
	//printf("%3f\n", parameters.VOXEL_WIDTH);						// *[cm] distance between left and right edges of each voxel in image
	//printf("%3f\n", parameters.VOXEL_HEIGHT);							// *[cm] distance between top and bottom edges of each voxel in image
	//printf("%3f\n", parameters.X_ZERO_COORDINATE);								// *[cm] x-coordinate corresponding to left edge of 1st voxel (i.e. column) in image space
	//printf("%3f\n", parameters.Y_ZERO_COORDINATE);								// *[cm] y-coordinate corresponding to top edge of 1st voxel (i.e. row) in image space
	//printf("%3f\n", parameters.Z_ZERO_COORDINATE);								// *[cm] z-coordinate corresponding to top edge of 1st voxel (i.e. slice) in image space
	//printf("%3f\n", parameters.RAM_LAK_TAU);
	////RECON_CYL_HEIGHT(SSD_V_SIZE - 1.0),									// *[cm] Height of reconstruction cylinder

	////import_image( FBP, FBP_path, filename, TEXT );
	////import_image( FBP, FBP_path, filename, BINARY );
	////printf("%d\n", parameters.NUM_VOXELS );
	//puts("Image Imported");

	//average_filter_2D( FBP, 2 );
	//array_2_disk( FBP_average_basename, FBP_path, TEXT, FBP, parameters.COLUMNS, parameters.ROWS, parameters.SLICES, parameters.NUM_VOXELS, true );
	//binary_2_txt_images( FBP_path, filename, FBP );
	//binary_2_txt_images( PREPROCESSING_DIR, filename, FBP );
	//test_median_filter_radii(FBP, FBP_median_basename );
	//median_filter_FBP_2D(FBP, parameters.FBP_MED_FILTER_RADIUS);
	//median_filter_FBP_3D(FBP, parameters.FBP_MED_FILTER_RADIUS);
	//puts("FBP image filtered");

	//char* FBP_path = "C:/Users/Blake/Documents/Visual Studio 2010/Projects/robust_pct/robust_pct";
	//char* filename = "FBP_image_h";
	//////char* filename = "FBP_med_new7";
	//////char* filename = "FBP_med_composed7";
	//float* FBP = (float*) calloc( parameters.NUM_VOXELS, sizeof(float) );
	//float* filtered_2D = (float*) calloc( parameters.NUM_VOXELS, sizeof(float) );
	//float* filtered_3D = (float*) calloc( parameters.NUM_VOXELS, sizeof(float) );
	////sprintf(FBP_MEDIAN_2D_PATH,"%s/%s_3D%d%s", PREPROCESSING_DIR, FBP_MEDIAN_BASENAME, parameters.FBP_MED_FILTER_RADIUS, FBP_MEDIANS_FILE_EXTENSION);
	////sprintf(FBP_MEDIAN_3D_PATH,"%s/%s_2D%s", PREPROCESSING_DIR, FBP_MEDIAN_BASENAME, parameters.FBP_MED_FILTER_RADIUS, FBP_MEDIANS_FILE_EXTENSION);
	//
	////
	////
	//////import_image( FBP, FBP_path, filename, BINARY );
	//import_image( FBP, FBP_path, filename, TEXT );
	//puts("FBP image imported");
	//////filtered = FBP;
	//////array_2_disk("FBP_imported", FBP_path, TEXT, filtered, parameters.COLUMNS, parameters.ROWS, parameters.SLICES, parameters.NUM_VOXELS, true );
	//////array_2_disk("FBP_med_composed7", FBP_path, TEXT, FBP, parameters.COLUMNS, parameters.ROWS, parameters.SLICES, parameters.NUM_VOXELS, true );

	////puts("FBP image imported and rewritten");
	//uint radius = 2;
	////median_filter_3D( FBP, filtered, radius );
	////median_filter_3D( FBP, filtered, radius );
	//median_filter_FBP_2D(FBP, FBP_median_filtered_2D_h, parameters.FBP_MED_FILTER_RADIUS);
	//median_filter_FBP_3D(FBP, FBP_median_filtered_3D_h, parameters.FBP_MED_FILTER_RADIUS);
	////FBP_MEDIAN_2D_PATH
	////char* filename_filt2D, * filename_filt3D, * filt_base = "FBP_median";
	////sprintf(filename_filt2D, "%s_2D%d", filt_base, radius );
	////sprintf(filename_filt3D, "%s_3D%d", filt_base, radius );
	//puts("FBP image filtered");

	//array_2_disk(filename_filt2D, FBP_path, TEXT, filtered_2D, parameters.COLUMNS, parameters.ROWS, parameters.SLICES, parameters.NUM_VOXELS, true );
	//array_2_disk(filename_filt3D, FBP_path, TEXT, filtered_3D, parameters.COLUMNS, parameters.ROWS, parameters.SLICES, parameters.NUM_VOXELS, true );
	//puts("filtered FBP image written to disk");
	

	//std::function<double(int, int)> fn1 = my_divide;                    // function
	//cout << func_pass_test(x,y, fn1) << endl;
	//std::function<double(double, double)> fn2 = my_divide2;                    // function
	//double x2 = 2;
	//double y2 = 3;
	//cout << func_pass_test(x2,y2, fn2) << endl;
	//std::function<int(int)> fn2 = &half;                   // function pointer
	//std::function<int(int)> fn3 = third_t();               // function object
	//std::function<int(int)> fn4 = [](int x){return x/4;};  // lambda expression
	//std::function<int(int)> fn5 = std::negate<int>();      // standard function object

	//test_transfer();
	//add_log_entry();
}
void test_func2( std::vector<int>& bin_numbers, std::vector<double>& data )
{
	int angular_bin = 8;
	int v_bin = 14;
	int bin_num = angular_bin * parameters.T_BINS + v_bin * parameters.ANGULAR_BINS * parameters.T_BINS;
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
	bin_num = angular_bin * parameters.T_BINS + v_bin * parameters.ANGULAR_BINS * parameters.T_BINS;
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
	bin_num = angular_bin * parameters.T_BINS + v_bin * parameters.ANGULAR_BINS * parameters.T_BINS;
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
	bin_num = angular_bin * parameters.T_BINS + v_bin * parameters.ANGULAR_BINS * parameters.T_BINS;
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
void test_transfer()
{
	unsigned int N_x = 3;
	unsigned int N_y = 4;
	unsigned int N_z = 5;

	float* x = (float*) calloc(N_x, sizeof(float) );
	float* y = (float*) calloc(N_y, sizeof(float) );
	float* z = (float*) calloc(N_z, sizeof(float) );

	float* x_d, *y_d, *z_d;

	cudaMalloc((void**) &x_d, N_x*sizeof(float));
	cudaMalloc((void**) &y_d, N_y*sizeof(float));
	cudaMalloc((void**) &z_d, N_z*sizeof(float));

	cudaMemcpy( x_d, x, N_x*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy( y_d, y, N_y*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy( z_d, z, N_z*sizeof(float), cudaMemcpyHostToDevice);

	dim3 dimBlock =  N_z;
	dim3 dimGrid = dim3(N_x, N_y );
	test_transfer_GPU<<< dimGrid, dimBlock >>>(x_d, y_d, z_d );

	cudaMemcpy( x, x_d, N_x*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy( y, y_d, N_y*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy( z, z_d, N_z*sizeof(float), cudaMemcpyDeviceToHost);

	for( unsigned int i = 0; i < 1; i++)
	{
		printf("%3f\n", x[i] );
		printf("%3f\n", y[i] );
		printf("%3f\n", z[i] );
		//cout << x[i] << endl; // -8.0
		//cout << y[i] << endl;
		//cout << z[i] << endl;
	}
}
__global__ void test_transfer_GPU(float* x, float* y, float* z)
{
	int column = blockIdx.x, row = blockIdx.y, slice = threadIdx.x;
	 atomicAdd( &x[column], static_cast<float>(1) );
	 atomicAdd( &y[row], static_cast<float>(1) );
	 atomicAdd( &z[slice], static_cast<float>(1) );
	//x = 2;
	//y = 3;
	//z = 4;
}
/***********************************************************************************************************************************************************************************************************************/
/********************************************************************************** Testing GPU Functions and Functions in Development *********************************************************************************/
/***********************************************************************************************************************************************************************************************************************/


__global__ void test_func_device( configurations* parameters, double* x, double* y, double* z )
{
	//x = 2;
	//y = 3;
	//z = 4;
		for( unsigned int i = 0; i < 4; i++)
	{
		//printf("%3f\n", x[i] );
		//printf("%3f\n", y[i] );
		//printf("%3f\n", z[i] );
		printf("%d\n", parameters->COLUMNS );
		//cout << x[i] << endl; // -8.0
		//cout << y[i] << endl;
		//cout << z[i] << endl;
	}
}
__global__ void test_func_GPU( configurations* parameters, int* a)
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
	////int voxel_x_out = int( ( x_exit[i] + parameters->RECON_CYL_RADIUS ) / parameters->VOXEL_WIDTH );
	//int voxel_y_out = int( ( parameters->RECON_CYL_RADIUS - y ) / parameters->VOXEL_HEIGHT );
	////int voxel_z_out = int( ( parameters->RECON_CYL_HEIGHT/2 - z_exit[i] ) /VOXEL_THICKNESS );
	//double voxel_y_float;
	//double y_inside2 = ((( parameters->RECON_CYL_RADIUS - y ) / parameters->VOXEL_HEIGHT) - voxel_y_out) * parameters->VOXEL_HEIGHT;
	//double y_inside = modf( ( parameters->RECON_CYL_RADIUS - y) /parameters->VOXEL_HEIGHT, &voxel_y_float)*parameters->VOXEL_HEIGHT;
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
	//int voxel = voxel_x + voxel_y * parameters->COLUMNS + voxel_z * parameters->COLUMNS * parameters->ROWS;
	//int x = 0, y = 0, z = 0;
	//test_func_device( x, y, z );
	//image[voxel] = x * y * z;
}