//********************************************************************************************************************************************************//
//*********************************************** Proton CT Preprocessing and Image Reconstruction Code  *************************************************//
//********************************************************************************************************************************************************//
#include "pCT_Reconstruction.h"

//********************************************************************************************************************************************************//
//********************************************************************** Host Code ***********************************************************************//
//********************************************************************************************************************************************************//

// Preprocessing setup and initializations 
void assign_SSD_positions();
void initializations();
void count_histories_old();
void count_histories_v0();
void count_histories_v1();
void reserve_vector_capacity();

// Preprocessing routines
void iterative_data_read_old( const int, const int, const int );
void iterative_data_read_v0( const int, const int, const int );
void iterative_data_read_v1( const int, const int, const int );
void recon_volume_intersections( const int );
void bin_valid_histories( const int );
void calculate_means();
void sum_differences( const int, const int );
void calculate_std_devs();
void statistical_cuts( const int, const int );
void initialize_sinogram();
void construct_sinogram();
void filter();
void backprojection();

// Hull-Detection 
void initialize_SC_hull( bool*&, bool*& );
void initialize_MSC_hull( int*&, int*& );
void initialize_SM_hull( int*&, int*& );
void initialize_float_image( float*&, float*& );
void SC( int );
void MSC( int );
void MSC_threshold();
void SM( int );
void SM_threshold();
void SM_threshold_2();
void averaging_filter( bool*&, bool*&, const int);

// MLP
void create_MLP_test_image();	// In development
void MLP_test();				// In development

// Write arrays/vectors to file(s)
template<typename T> void write_array_to_disk( char*, const char*, const char*, T*, const int, const int, const int, const int, const bool );
template<typename T> void write_vector_to_disk( char*, const char*, const char*, vector<T>, const int, const int, const int, const bool );

// Memory transfers and allocations/deallocations
void post_cut_memory_clean(); 
void resize_vectors( const int );
void shrink_vectors( const int );
void initial_processing_memory_clean();

// Helper Functions
bool bad_data_angle( const int );
int calculate_x_voxel(const float, const int, const float);
int calculate_y_voxel(const float, const int, const float);
int calculate_slice(const float, const int, const float);

// New routine test functions
void test_func();

//********************************************************************************************************************************************************//
//****************************************************************** Device (GPU) Code *******************************************************************//
//********************************************************************************************************************************************************//

// Preprocessing routines
__global__ void recon_volume_intersections_kernel( int, int*, bool*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*);
__global__ void bin_valid_histories_kernel( int, int*, int*, bool*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float* );
__global__ void calculate_means_kernel( int*, float*, float*, float* );
__global__ void sum_differences_kernel( int, int*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*  );
__global__ void calculate_std_devs_kernel( int*, float*, float*, float* );
__global__ void statistical_cuts_kernel( int, int*, int*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, bool*, float*, float* );
__global__ void construct_sinogram_kernel( int*, float* );
__global__ void filter_kernel( float*, float* );

// Hull-Detection 
__device__ void voxel_walk( bool*&, float, float, float, float, float, float );
__global__ void SC_kernel( int, bool*, int*, bool*, float*, float*, float*, float*, float*, float*, float* );
__global__ void MSC_kernel( int, int*, int*, bool*, float*, float*, float*, float*, float*, float*, float* );
__global__ void SM_kernel( int, int*, int*, bool*, float*, float*, float*, float*, float*, float*, float* );
__global__ void MSC_threshold_kernel( int* );
__global__ void SM_threshold_kernel( int*, int* );
__global__ void SM_threshold_kernel_2( int*, int* );
__global__ void carve_differences( int*, int* );
__global__ void averaging_filter_kernel( bool*, const int, const float );

// New routine test functions
__global__ void test_func_kernel( int*, int);

/************************************************************************************************************************************************************/
/******************************************************************** Program Main **************************************************************************/
/************************************************************************************************************************************************************/
int main(int argc, char** argv)
{
	char user_response[20];
	/*
	puts("Hit enter to stop...");
	fgets(user_response, sizeof(user_response), stdin);
	exit(1);
	*/
	/********************************************************************************************/
	/* Start the Execution Timing Clock															*/
	/********************************************************************************************/
	clock_t start,end;
	start = clock();
	/********************************************************************************************/
	/* Initialize Hull Detection Images and Transfer Them to the GPU							*/
	/********************************************************************************************/
	if( SC_ON )
		initialize_SC_hull( SC_image_h, SC_image_d );
	if( MSC_ON )
		initialize_MSC_hull( MSC_image_h, MSC_image_d );
	if( SM_ON )
		initialize_SM_hull( SM_image_h, SM_image_d );
	write_array_to_disk("x_sc_int", output_directory, output_folder, MSC_image_h, COLUMNS, ROWS, SLICES, VOXELS, true );
	write_array_to_disk("x_sc_int", output_directory, output_folder, MSC_image_h, COLUMNS, ROWS, SLICES, VOXELS, false );
	exit(1);
	/********************************************************************************************/
	/* Read the u-Coordinates of the Detector Planes from the Config File, Allocate and			*/
	/* Initialize Statistical Data Arrays, and Count the Number of Histories Per File,			*/
	/* Projection, Gantry Angle, Scan, and Total.  Request Input from User to Continue.			*/
	/********************************************************************************************/
	puts("Reading tracker plane positions and initializing storage arrays...");
	if( CONFIG_FILE)
		assign_SSD_positions(); // Read the detector plane u-coordinates from config file
	initializations();			// allocate and initialize host and GPU memory for binning
	if( VERSION_OLD )
		count_histories_old();	// count the number of histories per file, per scan, total, etc.
	else if( VERSION_0 )
		count_histories_v0();		// count the number of histories per file, per scan, total, etc.
	else
		count_histories_v1();
	/********************************************************************************************/
	/* Iteratively Read and Process Data One Chunk at a Time. There are at Most					*/
	/* MAX_GPU_HISTORIES Per Chunk (i.e. Iteration). On Each Iteration:							*/
	/*	(1) Read Data from File																	*/
	/*	(2) Determine Which Histories Traverse the Reconstruction Volume and Store this			*/
	/*		Information in a Boolean Array														*/
	/*	(3) Determine Which Bin Each History Belongs to											*/
	/*	(4) Use the Boolean Array to Determine Which Histories to Keep and then Push			*/
	/*		the Intermediate Data from these Histories onto the Permanent Storage Vectors		*/
	/*	(5) Free Up Temporary Host/GPU Array Memory Allocated During Iteration					*/
	/********************************************************************************************/
	puts("Iteratively Reading Data from Hard Disk");
	puts("Removing Proton Histories that Don't Pass Through the Reconstruction Volume");
	puts("Binning the Data from Those that Did...");
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
		if( VERSION_OLD )
			iterative_data_read_old( histories_to_process, start_file_num, end_file_num - 1 );
		else if( VERSION_0 )
			iterative_data_read_v0( histories_to_process, start_file_num, end_file_num - 1 );
		else
			iterative_data_read_v1( histories_to_process, start_file_num, end_file_num - 1 );
		recon_volume_intersections( histories_to_process );
		bin_valid_histories( histories_to_process );
		if(  SC_ON && (!bad_data_angle( gantry_angle_h[0] ) || !RESTRICTED_ANGLES ) ) 
			SC( histories_to_process );		
		if( MSC_ON )
			MSC( histories_to_process );
		if( SM_ON )
			SM( histories_to_process );                                                                                                                                                                             
		initial_processing_memory_clean();
		start_file_num = end_file_num;
		histories_to_process = 0;
	}	
	/********************************************************************************************/
	/* Shrink vectors so capacity reduced to size, which is number of histories remaining after */
	/* histories that didn't intersect reconstruction volume were ignored						*/																					
	/********************************************************************************************/
	shrink_vectors( recon_vol_histories );
	/********************************************************************************************/
	/* Perform Thresholding on MSC and SM Hulls and Write All Hull Images to File				*/																					
	/********************************************************************************************/
	puts("\nPerforming Hull Thresholding and Writing Hull Images to Disk...");
	if( SC_ON )
	{
		cudaMemcpy(SC_image_h,  SC_image_d, MEM_SIZE_IMAGE_BOOL, cudaMemcpyDeviceToHost);
		write_array_to_disk("x_sc", output_directory, output_folder, SC_image_h, COLUMNS, ROWS, SLICES, VOXELS, true );
	}
	if( MSC_ON )
		MSC_threshold();
	if( SM_ON )
		SM_threshold();
	exit(1);
	/********************************************************************************************/
	/* Calculate the Mean WEPL, Relative ut-Angle, and Relative uv-Angle for Each Bin and Count */
	/* the Number of Histories in Each Bin														*/													
	///********************************************************************************************/
	puts("Calculating the Mean for Each Bin Before Cuts...");
	calculate_means();
	/********************************************************************************************/
	/* Calculate the Standard Deviation in WEPL, Relative ut-Angle, and Relative uv-Angle for	*/
	/* Each Bin.  Iterate Through the Valid History Vectors One Chunk at a Time, With at Most	*/
	/* MAX_GPU_HISTORIES Per Chunk, and Calculate the Difference Between the Mean WEPL and WEPL,*/
	/* Mean Relative ut-Angle and Relative ut-Angle, and Mean Relative uv-Angle and	Relative	*/
	/* uv-Angle for Each History. The Standard Deviation is then Found By Calculating the Sum	*/
	/* of these Differences for Each Bin and Dividing it by the Number of Histories in the Bin 	*/
	/********************************************************************************************/
	puts("Summing up the Difference Between Individual Measurements and the Mean for Each Bin...");
	int remaining_histories = recon_vol_histories;
	int start_position = 0;
	while( remaining_histories > 0 )
	{
		if( remaining_histories > MAX_GPU_HISTORIES )
			histories_to_process = MAX_GPU_HISTORIES;
		else
			histories_to_process = remaining_histories;
		sum_differences( start_position, histories_to_process );
		remaining_histories -= MAX_GPU_HISTORIES;
		start_position += MAX_GPU_HISTORIES;
	}
	puts("Calculating Standard Deviations for Each Bin...");
	calculate_std_devs();
	/********************************************************************************************/
	/* Allocate Memory for the Sinogram on the Host, Initialize it to Zeros, Allocate Memory 	*/
	/* for it on the GPU, then Transfer the Initialized Sinogram to the GPU						*/
	/********************************************************************************************/
	initialize_sinogram();
	/********************************************************************************************/
	/* Iterate Through the Valid History Vectors One Chunk at a Time, With at Most				*/
	/* MAX_GPU_HISTORIES Per Chunk, and Perform Statistical Cuts 								*/
	/********************************************************************************************/
	puts("Performing Statistical Cuts...");
	remaining_histories = recon_vol_histories, start_position = 0;
	while( remaining_histories > 0 )
	{
		if( remaining_histories > MAX_GPU_HISTORIES )
			histories_to_process = MAX_GPU_HISTORIES;
		else
			histories_to_process = remaining_histories;
		statistical_cuts( start_position, histories_to_process );
		remaining_histories -= MAX_GPU_HISTORIES;
		start_position += MAX_GPU_HISTORIES;
	}
	printf("%d out of %d histories passed cuts\n", post_cut_histories, total_histories );
	/********************************************************************************************/
	/* Free the host memory for the bin number array and gpu memory for the statistics arrays	*/
	/* and shrink the vectors to fit exactly the number of histories that passed cuts			*/
	/********************************************************************************************/
	puts("Freeing unnecessary memory and shrinking vectors to just fit remaining histories...");
	post_cut_memory_clean();
	resize_vectors( post_cut_histories );
	shrink_vectors( post_cut_histories );	
	/********************************************************************************************/
	/* Recalculate the Mean WEPL for Each Bin Using	the Histories Remaining After Cuts and Use	*/
	/* these to Produce the Sinogram															*/
	///********************************************************************************************/
	puts("Calculating the Elements of the Sinogram...");
	construct_sinogram();
	/********************************************************************************************/
	/* Perform Filtered Backprojection and Write FBP Hull to Disk								*/
	/********************************************************************************************/
	if( FBP_ON )
	{
		filter();
		backprojection();
	}
	/********************************************************************************************/
	/* End Program Execution Timing Clock and Print	the Total Execution Time to Console Window	*/
	/********************************************************************************************/
	//end = clock();
	//printf("Total execution time : %3f\n",(double)(end-start)/1000);
	/********************************************************************************************/
	/* Program Has Finished Execution. Require the User to Hit the Enter Key to Terminate the	*/
	/* Program and Close the Terminal/Console Window											*/ 															
	/********************************************************************************************/
	puts("Preprocessing complete.  Press any key to close the console window...");
	fgets(user_response, sizeof(user_response), stdin);
}
/************************************************************************************************************************************************************/
/******************************************************** Preprocessing Setup and Initializations ***********************************************************/
/************************************************************************************************************************************************************/
void assign_SSD_positions()	//HERE THE COORDINATES OF THE DETECTORS PLANES ARE LOADED, THE CONFIG FILE IS CREATED BY FORD (RWS)
{
	char user_response[20];
	char configFilename[512];
	sprintf(configFilename, "%s%s\\scan.cfg", input_directory, input_folder);
	if( DEBUG_TEXT_ON )
		printf("Opening config file %s...\n", configFilename);
	ifstream configFile(configFilename);		
	if( !configFile.is_open() ) {
		printf("ERROR: config file not found at %s!\n", configFilename);	
		fputs("Didn't Find File", stdout);
		fflush(stdout); 
		printf("text = \"%s\"\n", user_response);
		fgets(user_response, sizeof(user_response), stdin);
		exit(1);
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
void initializations()
{
	for( int scan_number = 0; scan_number < NUM_SCANS; scan_number++ )
		histories_per_scan[scan_number] = 0;

	histories_per_file =				 (int*) calloc( NUM_SCANS * GANTRY_ANGLES, sizeof(int) );
	histories_per_gantry_angle =		 (int*) calloc( GANTRY_ANGLES, sizeof(int) );
	recon_vol_histories_per_projection = (int*) calloc( GANTRY_ANGLES, sizeof(int) );

	bin_counts_h		  = (int*)	 calloc( NUM_BINS, sizeof(int) );
	mean_WEPL_h			  = (float*) calloc( NUM_BINS, sizeof(float) );
	mean_rel_ut_angle_h	  = (float*) calloc( NUM_BINS, sizeof(float) );
	mean_rel_uv_angle_h	  = (float*) calloc( NUM_BINS, sizeof(float) );
	stddev_rel_ut_angle_h = (float*) calloc( NUM_BINS, sizeof(float) );	
	stddev_rel_uv_angle_h = (float*) calloc( NUM_BINS, sizeof(float) );	
	stddev_WEPL_h		  = (float*) calloc( NUM_BINS, sizeof(float) );

	cudaMalloc((void**) &bin_counts_d,			MEM_SIZE_BINS_INTS );
	cudaMalloc((void**) &mean_WEPL_d,			MEM_SIZE_BINS_FLOATS );
	cudaMalloc((void**) &mean_rel_ut_angle_d,	MEM_SIZE_BINS_FLOATS );
	cudaMalloc((void**) &mean_rel_uv_angle_d,	MEM_SIZE_BINS_FLOATS );
	cudaMalloc((void**) &stddev_rel_ut_angle_d,	MEM_SIZE_BINS_FLOATS );
	cudaMalloc((void**) &stddev_rel_uv_angle_d,	MEM_SIZE_BINS_FLOATS );
	cudaMalloc((void**) &stddev_WEPL_d,			MEM_SIZE_BINS_FLOATS );

	cudaMemcpy( bin_counts_d,			bin_counts_h,			MEM_SIZE_BINS_INTS,		cudaMemcpyHostToDevice );
	cudaMemcpy( mean_WEPL_d,			mean_WEPL_h,			MEM_SIZE_BINS_FLOATS,	cudaMemcpyHostToDevice );
	cudaMemcpy( mean_rel_ut_angle_d,	mean_rel_ut_angle_h,	MEM_SIZE_BINS_FLOATS,	cudaMemcpyHostToDevice );
	cudaMemcpy( mean_rel_uv_angle_d,	mean_rel_uv_angle_h,	MEM_SIZE_BINS_FLOATS,	cudaMemcpyHostToDevice );
	cudaMemcpy( stddev_rel_ut_angle_d,	stddev_rel_ut_angle_h,	MEM_SIZE_BINS_FLOATS,	cudaMemcpyHostToDevice );
	cudaMemcpy( stddev_rel_uv_angle_d,	stddev_rel_uv_angle_h,	MEM_SIZE_BINS_FLOATS,	cudaMemcpyHostToDevice );
	cudaMemcpy( stddev_WEPL_d,			stddev_WEPL_h,			MEM_SIZE_BINS_FLOATS,	cudaMemcpyHostToDevice );
}
void count_histories_old()
{
	if( DEBUG_TEXT_ON )
		printf("Counting histories...\n");
	char user_response[20];
	char data_filename[128];
	int file_size, num_histories, file_number = 0, gantry_position_number = 0;
	for( int gantry_angle = 0; gantry_angle < 360; gantry_angle += GANTRY_ANGLE_INTERVAL, gantry_position_number++ )
	{
		for( int scan_number = 1; scan_number <= NUM_SCANS; scan_number++, file_number++ )
		{
			
			sprintf( data_filename, "%s%s/%s_trans%d_%03d%s", input_directory, input_folder, input_base_name, scan_number, gantry_angle, file_extension );
			//printf("Name = %s", data_filename );
			FILE *data_file = fopen(data_filename, "rb");
			if( data_file == NULL )
			{
				fputs( "Error Opening Data File:  Check that the directories are properly named.", stderr ); 
				fgets(user_response, sizeof(user_response), stdin);
				exit(1);
			}
			fseek( data_file, 0, SEEK_END );
			file_size = ftell( data_file );
			if( BINARY_ENCODING )
			{
				if( file_size % BYTES_PER_HISTORY ) 
				{
					printf("ERROR! Problem with bytes_per_history!\n");
					fgets(user_response, sizeof(user_response), stdin);
					exit(2);
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
				printf("There are %d Histories for Gantry Angle %d From Scan Number %d\n",num_histories, gantry_angle, scan_number);
		}
	}
	if( DEBUG_TEXT_ON )
	{
		for( int file_number = 0, int gantry_position_number = 0; file_number < (NUM_SCANS * GANTRY_ANGLES); file_number++, gantry_position_number++ )
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
void count_histories_v0()
{
	if( DEBUG_TEXT_ON )
		puts("Counting histories...\n");

	char user_response[20];
	char data_filename[256];
	int file_size, num_histories, file_number = 0, gantry_position_number = 0;
	for( int gantry_angle = 0; gantry_angle < 360; gantry_angle += GANTRY_ANGLE_INTERVAL, gantry_position_number++ )
	{
		for( int scan_number = 1; scan_number <= NUM_SCANS; scan_number++, file_number++ )
		{
			sprintf(data_filename, "%s%s/%s_%03d%s", input_directory, input_folder, input_base_name, gantry_angle, file_extension  );
			//cout << data_filename << endl;
			ifstream data_file(data_filename, ios::binary);
			if( data_file == NULL )
			{
				fputs( "File not found:  Check that the directories and files are properly named.", stderr ); 
				fgets(user_response, sizeof(user_response), stdin);
				exit(1);
			}
			char magic_number[5];
			data_file.read(magic_number, 4);
			magic_number[4] = '\0';
			if( strcmp(magic_number, "PCTD") ) {
				puts("Error: unknown file type (should be PCTD)!\n");
				fgets(user_response, sizeof(user_response), stdin);
				exit(1);
			}
			int version_id;
			data_file.read((char*)&version_id, sizeof(int));
			if( version_id == 0 )
			{
				int num_histories;
				data_file.read((char*)&num_histories, sizeof(int));						
				data_file.close();
				histories_per_file[file_number] = num_histories;
				histories_per_gantry_angle[gantry_position_number] += num_histories;
				histories_per_scan[scan_number-1] += num_histories;
				total_histories += num_histories;
			
				if( DEBUG_TEXT_ON )
					printf("There are %d Histories for Gantry Angle %d From Scan Number %d\n",num_histories, gantry_angle, scan_number);
			}
			else 
			{
				printf("ERROR: Unsupported format version (%d)!\n", version_id);
				fgets(user_response, sizeof(user_response), stdin);
				exit(1);
			}			
		}
	}
	if( DEBUG_TEXT_ON )
	{
		for( int file_number = 0, int gantry_position_number = 0; file_number < (NUM_SCANS * GANTRY_ANGLES); file_number++, gantry_position_number++ )
		{
			if( file_number % NUM_SCANS == 0 )
				printf("There are a Total of %d Histories From Gantry Angle %d\n", histories_per_gantry_angle[gantry_position_number], int(gantry_position_number* GANTRY_ANGLE_INTERVAL) );			
			printf("* %d Histories are From Scan Number %d\n", histories_per_file[file_number], (file_number % NUM_SCANS) + 1 );
			
		}
		for( int scan_number = 0; scan_number < NUM_SCANS; scan_number++ )
			printf("There are a Total of %d Histories in Scan Number %d \n", histories_per_scan[scan_number], scan_number + 1);
		printf("There are a Total of %d Histories\n", total_histories);
	}
	// The GPU cannot process all the histories at once, so they are broken up into chunks that can fit on the GPU.  As we iterate 
	// through the data one chunk at a time, we determine which histories enter the reconstruction volume and if they belong to a 
	// valid bin (i.e. t, v, and angular bin number is greater than zero and less than max).  If both are true, we append the bin
	// number, WEPL, and relative entry/exit ut/uv angles to the following four arrays.  We do not know ahead of time how many 
	// valid histories there will be, so memory is allocated to accomodate every history and the actual number of valid histories
	// are counted. Although we waste some host memory, we can avoid writing intermediate information to file or keeping the raw 
	// data and recalculating it every time its needed. Once all the data is processed and we know how many valid histories we 
	// have, we simply ignore the illegitimate elements of the four arrays to avoid transferring invalid and unnecessary data to 
	// and from the GPU.
}
void count_histories_v1()
{
	if( DEBUG_TEXT_ON )
		printf("Counting histories...\n");

	char user_response[20];
	char data_filename[128];
	int file_size, num_histories, file_number = 0, gantry_position_number = 0;
	for( int gantry_angle = 0; gantry_angle < 360; gantry_angle += GANTRY_ANGLE_INTERVAL, gantry_position_number++ )
	{
		for( int scan_number = 1; scan_number <= NUM_SCANS; scan_number++, file_number++ )
		{
			
			sprintf(data_filename, "%s%s/%s_%03d%%s", input_directory, input_folder, input_base_name, gantry_angle, file_extension  );
			FILE *data_file = fopen(data_filename, "rb");
			if( data_file == NULL )
			{
				fputs( "Error Opening Data File:  Check that the directories are properly named.", stderr ); 
				fgets(user_response, sizeof(user_response), stdin);
				exit(1);
			}
			fseek( data_file, 0, SEEK_END );
			file_size = ftell( data_file );
			if( BINARY_ENCODING )
			{
				if( file_size % BYTES_PER_HISTORY ) 
				{
					printf("ERROR! Problem with bytes_per_history!\n");
					fgets(user_response, sizeof(user_response), stdin);
					exit(2);
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
				printf("There are %d Histories for Gantry Angle %d From Scan Number %d\n",num_histories, gantry_angle, scan_number);
		}
	}
	if( DEBUG_TEXT_ON )
	{
		for( int file_number = 0, int gantry_position_number = 0; file_number < (NUM_SCANS * GANTRY_ANGLES); file_number++, gantry_position_number++ )
		{
			if( file_number % NUM_SCANS == 0 )
				printf("There are a Total of %d Histories From Gantry Angle %d\n", histories_per_gantry_angle[gantry_position_number], int(gantry_position_number* GANTRY_ANGLE_INTERVAL) );			
			printf("* %d Histories are From Scan Number %d\n", histories_per_file[file_number], (file_number % NUM_SCANS) + 1 );
			
		}
		for( int scan_number = 0; scan_number < NUM_SCANS; scan_number++ )
			printf("There are a Total of %d Histories in Scan Number %d \n", histories_per_scan[scan_number], scan_number + 1);
		printf("There are a Total of %d Histories\n", total_histories);
	}
	// The GPU cannot process all the histories at once, so they are broken up into chunks that can fit on the GPU.  As we iterate 
	// through the data one chunk at a time, we determine which histories enter the reconstruction volume and if they belong to a 
	// valid bin (i.e. t, v, and angular bin number is greater than zero and less than max).  If both are true, we append the bin
	// number, WEPL, and relative entry/exit ut/uv angles to the following four arrays.  We do not know ahead of time how many 
	// valid histories there will be, so memory is allocated to accomodate every history and the actual number of valid histories
	// are counted. Although we waste some host memory, we can avoid writing intermediate information to file or keeping the raw 
	// data and recalculating it every time its needed. Once all the data is processed and we know how many valid histories we 
	// have, we simply ignore the illegitimate elements of the four arrays to avoid transferring invalid and unnecessary data to 
	// and from the GPU.
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
	//xy_exit_angle_vector.reserve( total_histories );
	//xz_exit_angle_vector.reserve( total_histories );
	relative_ut_angle_vector.reserve( total_histories );
	relative_uv_angle_vector.reserve( total_histories );
}
/************************************************************************************************************************************************************/
/********************************************************* Data Importation, Initial Cuts, and Binning ******************************************************/
/************************************************************************************************************************************************************/
void iterative_data_read_old( const int num_histories, const int start_file_num, const int end_file_num )
{
		unsigned int mem_size_hist_floats = sizeof(float) * num_histories;
	unsigned int mem_size_hist_ints = sizeof(int) * num_histories;

	t_in_1_h         = (float*) malloc(mem_size_hist_floats);
	t_in_2_h         = (float*) malloc(mem_size_hist_floats);
	t_out_1_h        = (float*) malloc(mem_size_hist_floats);
	t_out_2_h        = (float*) malloc(mem_size_hist_floats);
	u_in_1_h         = (float*) malloc(mem_size_hist_floats);
	u_in_2_h         = (float*) malloc(mem_size_hist_floats);
	u_out_1_h        = (float*) malloc(mem_size_hist_floats);
	u_out_2_h        = (float*) malloc(mem_size_hist_floats);
	v_in_1_h         = (float*) malloc(mem_size_hist_floats);
	v_in_2_h         = (float*) malloc(mem_size_hist_floats);
	v_out_1_h        = (float*) malloc(mem_size_hist_floats);
	v_out_2_h        = (float*) malloc(mem_size_hist_floats);		
	WEPL_h           = (float*) malloc(mem_size_hist_floats);
	gantry_angle_h   = (int*)   malloc(mem_size_hist_ints);

	int array_index = 0, gantry_position, gantry_angle, scan_number, scan_histories;
	float v_data[4], t_data[4], WEPL_data, gantry_angle_data, dummy_data;
	char tracker_plane[4];
	char data_filename[128];
	FILE* data_file;

	for( int file_num = start_file_num; file_num <= end_file_num; file_num++ )
	{
		gantry_position = file_num / NUM_SCANS;
		gantry_angle = gantry_position * GANTRY_ANGLE_INTERVAL;
		scan_number = file_num % NUM_SCANS + 1;
		scan_histories = histories_per_file[file_num];

		printf("Reading File for Gantry Angle %d from Scan Number %d...\n", gantry_angle, scan_number );
		sprintf( data_filename, "%s%s/%s_trans%d_%03d%s", input_directory, input_folder, input_base_name, scan_number, gantry_angle, file_extension );
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
				v_in_1_h[array_index]	= v_data[0] * 0.1;
				v_in_2_h[array_index]	= v_data[1] * 0.1;
				v_out_1_h[array_index]	= v_data[2] * 0.1;
				v_out_2_h[array_index]	= v_data[3] * 0.1;
				t_in_1_h[array_index]	= t_data[0] * 0.1;
				t_in_2_h[array_index]	= t_data[1] * 0.1;
				t_out_1_h[array_index]	= t_data[2] * 0.1;
				t_out_2_h[array_index]	= t_data[3] * 0.1;
				WEPL_h[array_index]		= WEPL_data * 0.1;
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
				u_in_1_h[array_index]	*= 0.1;
				u_in_2_h[array_index]	*= 0.1;
				u_out_1_h[array_index]	*= 0.1;
				u_out_2_h[array_index]	*= 0.1;
			}
			gantry_angle_h[array_index] = int(gantry_angle_data);
		}
		fclose(data_file);		
	}
}
void iterative_data_read_v0( const int num_histories, const int start_file_num, const int end_file_num )
{
	unsigned int mem_size_hist_floats = sizeof(float) * num_histories;
	unsigned int mem_size_hist_ints = sizeof(int) * num_histories;

	t_in_1_h         = (float*) malloc(mem_size_hist_floats);
	t_in_2_h         = (float*) malloc(mem_size_hist_floats);
	t_out_1_h        = (float*) malloc(mem_size_hist_floats);
	t_out_2_h        = (float*) malloc(mem_size_hist_floats);
	u_in_1_h         = (float*) malloc(mem_size_hist_floats);
	u_in_2_h         = (float*) malloc(mem_size_hist_floats);
	u_out_1_h        = (float*) malloc(mem_size_hist_floats);
	u_out_2_h        = (float*) malloc(mem_size_hist_floats);
	v_in_1_h         = (float*) malloc(mem_size_hist_floats);
	v_in_2_h         = (float*) malloc(mem_size_hist_floats);
	v_out_1_h        = (float*) malloc(mem_size_hist_floats);
	v_out_2_h        = (float*) malloc(mem_size_hist_floats);		
	WEPL_h           = (float*) malloc(mem_size_hist_floats);
	gantry_angle_h   = (int*)   malloc(mem_size_hist_ints);

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
	char user_response[20];
	char data_filename[128];
	int array_index = 0;
	float min_WEPL = 20, max_WEPL = -20;
	for( int file_num = start_file_num; file_num <= end_file_num; file_num++ )
	{
		int gantry_position = file_num / NUM_SCANS;
		int gantry_angle = gantry_position * GANTRY_ANGLE_INTERVAL;
		int scan_number = file_num % NUM_SCANS + 1;
		int scan_histories = histories_per_file[file_num];

		printf("Reading File for Gantry Angle %d from Scan Number %d...\n", gantry_angle, scan_number );
		sprintf(data_filename, "%s%s/%s_%03d%s", input_directory, input_folder, input_base_name, gantry_angle, file_extension );	
		ifstream data_file(data_filename, ios::binary);
		if( data_file == NULL )
		{
			fputs( "File not found:  Check that the directories and files are properly named.", stderr ); 
			fgets(user_response, sizeof(user_response), stdin);
			exit(1);
		}
		char magic_number[5];
		data_file.read(magic_number, 4);
		magic_number[4] = '\0';
		if( strcmp(magic_number, "PCTD") ) {
			puts("Error: unknown file type (should be PCTD)!\n");
			fgets(user_response, sizeof(user_response), stdin);
			exit(1);
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
					v_in_1_h[i]		*= 0.1;
					v_in_2_h[i]		*= 0.1;
					v_out_1_h[i]	*= 0.1;
					v_out_2_h[i]	*= 0.1;
					t_in_1_h[i]		*= 0.1;
					t_in_2_h[i]		*= 0.1;
					t_out_1_h[i]	*= 0.1;
					t_out_2_h[i]	*= 0.1;
					WEPL_h[i]		*= 0.1;
					if( WEPL_h[i] < 0 )
						printf("WEPL[%d] = %3f\n", i, WEPL_h[i] );
					u_in_1_h[i]		*= 0.1;
					u_in_2_h[i]		*= 0.1;
					u_out_1_h[i]	*= 0.1;
					u_out_2_h[i]	*= 0.1;
				}
				gantry_angle_h[i] = int(projection_angle);
			}
			data_file.close();
		}
	}
}
void iterative_data_read_v1( const int num_histories, const int start_file_num, const int end_file_num ){
	unsigned int mem_size_hist_floats = sizeof(float) * num_histories;
	unsigned int mem_size_hist_ints = sizeof(int) * num_histories;

	t_in_1_h         = (float*) malloc(mem_size_hist_floats);
	t_in_2_h         = (float*) malloc(mem_size_hist_floats);
	t_out_1_h        = (float*) malloc(mem_size_hist_floats);
	t_out_2_h        = (float*) malloc(mem_size_hist_floats);
	u_in_1_h         = (float*) malloc(mem_size_hist_floats);
	u_in_2_h         = (float*) malloc(mem_size_hist_floats);
	u_out_1_h        = (float*) malloc(mem_size_hist_floats);
	u_out_2_h        = (float*) malloc(mem_size_hist_floats);
	v_in_1_h         = (float*) malloc(mem_size_hist_floats);
	v_in_2_h         = (float*) malloc(mem_size_hist_floats);
	v_out_1_h        = (float*) malloc(mem_size_hist_floats);
	v_out_2_h        = (float*) malloc(mem_size_hist_floats);		
	WEPL_h           = (float*) malloc(mem_size_hist_floats);
	gantry_angle_h   = (int*)   malloc(mem_size_hist_ints);

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
	char user_response[20];
	char data_filename[128];
	int array_index = 0;
	float min_WEPL = 20, max_WEPL = -20;
	for( int file_num = start_file_num; file_num <= end_file_num; file_num++ )
	{
		int gantry_position = file_num / NUM_SCANS;
		int gantry_angle = gantry_position * GANTRY_ANGLE_INTERVAL;
		int scan_number = file_num % NUM_SCANS + 1;
		int scan_histories = histories_per_file[file_num];

		printf("Reading File for Gantry Angle %d from Scan Number %d...\n", gantry_angle, scan_number );
		sprintf(data_filename, "%s%s/%s_%03d%s", input_directory, input_folder, input_base_name, gantry_angle, file_extension );	
		ifstream data_file(data_filename, ios::binary);
		if( data_file == NULL )
		{
			fputs( "File not found:  Check that the directories and files are properly named.", stderr ); 
			fgets(user_response, sizeof(user_response), stdin);
			exit(1);
		}
		char magic_number[5];
		data_file.read(magic_number, 4);
		magic_number[4] = '\0';
		if( strcmp(magic_number, "PCTD") ) {
			puts("Error: unknown file type (should be PCTD)!\n");
			fgets(user_response, sizeof(user_response), stdin);
			exit(1);
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
					v_in_1_h[i]		*= 0.1;
					v_in_2_h[i]		*= 0.1;
					v_out_1_h[i]	*= 0.1;
					v_out_2_h[i]	*= 0.1;
					t_in_1_h[i]		*= 0.1;
					t_in_2_h[i]		*= 0.1;
					t_out_1_h[i]	*= 0.1;
					t_out_2_h[i]	*= 0.1;
					WEPL_h[i]		*= 0.1;
					if( WEPL_h[i] < 0 )
						printf("WEPL[%d] = %3f\n", i, WEPL_h[i] );
					u_in_1_h[i]		*= 0.1;
					u_in_2_h[i]		*= 0.1;
					u_out_1_h[i]	*= 0.1;
					u_out_2_h[i]	*= 0.1;
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
	unsigned int mem_size_hist_floats = sizeof(float) * num_histories;
	unsigned int mem_size_hist_ints = sizeof(int) * num_histories;
	unsigned int mem_size_hist_bool = sizeof(bool) * num_histories;

	// Allocate GPU memory
	cudaMalloc((void**) &t_in_1_d,					mem_size_hist_floats);
	cudaMalloc((void**) &t_in_2_d,					mem_size_hist_floats);
	cudaMalloc((void**) &t_out_1_d,					mem_size_hist_floats);
	cudaMalloc((void**) &t_out_2_d,					mem_size_hist_floats);
	cudaMalloc((void**) &u_in_1_d,					mem_size_hist_floats);
	cudaMalloc((void**) &u_in_2_d,					mem_size_hist_floats);
	cudaMalloc((void**) &u_out_1_d,					mem_size_hist_floats);
	cudaMalloc((void**) &u_out_2_d,					mem_size_hist_floats);
	cudaMalloc((void**) &v_in_1_d,					mem_size_hist_floats);
	cudaMalloc((void**) &v_in_2_d,					mem_size_hist_floats);
	cudaMalloc((void**) &v_out_1_d,					mem_size_hist_floats);
	cudaMalloc((void**) &v_out_2_d,					mem_size_hist_floats);		
	cudaMalloc((void**) &WEPL_d,					mem_size_hist_floats);
	cudaMalloc((void**) &gantry_angle_d,			mem_size_hist_ints);

	cudaMalloc((void**) &x_entry_d,					mem_size_hist_floats);
	cudaMalloc((void**) &y_entry_d,					mem_size_hist_floats);
	cudaMalloc((void**) &z_entry_d,					mem_size_hist_floats);
	cudaMalloc((void**) &x_exit_d,					mem_size_hist_floats);
	cudaMalloc((void**) &y_exit_d,					mem_size_hist_floats);
	cudaMalloc((void**) &z_exit_d,					mem_size_hist_floats);
	cudaMalloc((void**) &xy_entry_angle_d,			mem_size_hist_floats);	
	cudaMalloc((void**) &xz_entry_angle_d,			mem_size_hist_floats);
	cudaMalloc((void**) &xy_exit_angle_d,			mem_size_hist_floats);
	cudaMalloc((void**) &xz_exit_angle_d,			mem_size_hist_floats);
	cudaMalloc((void**) &relative_ut_angle_d,		mem_size_hist_floats);
	cudaMalloc((void**) &relative_uv_angle_d,		mem_size_hist_floats);
	cudaMalloc((void**) &traversed_recon_volume_d,	mem_size_hist_bool);	

	cudaMemcpy(t_in_1_d,		t_in_1_h,		mem_size_hist_floats, cudaMemcpyHostToDevice) ;
	cudaMemcpy(t_in_2_d,		t_in_2_h,		mem_size_hist_floats, cudaMemcpyHostToDevice) ;
	cudaMemcpy(t_out_1_d,		t_out_1_h,		mem_size_hist_floats, cudaMemcpyHostToDevice) ;
	cudaMemcpy(t_out_2_d,		t_out_2_h,		mem_size_hist_floats, cudaMemcpyHostToDevice) ;
	cudaMemcpy(u_in_1_d,		u_in_1_h,		mem_size_hist_floats, cudaMemcpyHostToDevice) ;
	cudaMemcpy(u_in_2_d,		u_in_2_h,		mem_size_hist_floats, cudaMemcpyHostToDevice) ;
	cudaMemcpy(u_out_1_d,		u_out_1_h,		mem_size_hist_floats, cudaMemcpyHostToDevice) ;
	cudaMemcpy(u_out_2_d,		u_out_2_h,		mem_size_hist_floats, cudaMemcpyHostToDevice) ;
	cudaMemcpy(v_in_1_d,		v_in_1_h,		mem_size_hist_floats, cudaMemcpyHostToDevice) ;
	cudaMemcpy(v_in_2_d,		v_in_2_h,		mem_size_hist_floats, cudaMemcpyHostToDevice) ;
	cudaMemcpy(v_out_1_d,		v_out_1_h,		mem_size_hist_floats, cudaMemcpyHostToDevice) ;
	cudaMemcpy(v_out_2_d,		v_out_2_h,		mem_size_hist_floats, cudaMemcpyHostToDevice) ;
	cudaMemcpy(gantry_angle_d,	gantry_angle_h,	mem_size_hist_ints,   cudaMemcpyHostToDevice) ;
	cudaMemcpy(WEPL_d,			WEPL_h,			mem_size_hist_floats, cudaMemcpyHostToDevice) ;

	dim3 dimBlock(THREADS_PER_BLOCK);
	dim3 dimGrid((int)(num_histories/THREADS_PER_BLOCK)+1);
	recon_volume_intersections_kernel<<<dimGrid, dimBlock>>>
	(
		num_histories, gantry_angle_d, traversed_recon_volume_d, WEPL_d,
		t_in_1_d, t_in_2_d, t_out_1_d, t_out_2_d,
		u_in_1_d, u_in_2_d, u_out_1_d, u_out_2_d,
		v_in_1_d, v_in_2_d, v_out_1_d, v_out_2_d, 	
		x_entry_d, y_entry_d, z_entry_d, x_exit_d, y_exit_d, z_exit_d, 		
		xy_entry_angle_d, xz_entry_angle_d, xy_exit_angle_d, xz_exit_angle_d,
		relative_ut_angle_d, relative_uv_angle_d
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
__global__ void recon_volume_intersections_kernel
(
	int num_histories, int* gantry_angle, bool* traversed_recon_volume, float* WEPL,
	float* t_in_1, float* t_in_2, float* t_out_1, float* t_out_2,
	float* u_in_1, float* u_in_2, float* u_out_1, float* u_out_2,
	float* v_in_1, float* v_in_2, float* v_out_1, float* v_out_2, 	
	float* x_entry, float* y_entry, float* z_entry, float* x_exit, float* y_exit, float* z_exit, 	
	float* xy_entry_angle, float* xz_entry_angle, float* xy_exit_angle, float* xz_exit_angle,
	float* relative_ut_angle, float* relative_uv_angle
)
{
	/*
			Determine if the proton path passes through the reconstruction volume (i.e. intersects the reconstruction 
		cylinder twice) and if it does, determine the x, y, and z positions in the global/object coordinate system where 
		the proton enters and exits the reconstruction volume.  The origin of the object coordinate system is defined to 
		be at the center of the reconstruction cylinder so that its volume is bounded by:

			-RECON_CYL_RADIUS	<= x <= RECON_CYL_RADIUS
			-RECON_CYL_RADIUS	<= y <= RECON_CYL_RADIUS 
			-RECON_CYL_HEIGHT/2 <= z <= RECON_CYL_HEIGHT/2

			First, the coordinates of the points where the proton path intersected the entry/exit detectors must be 
		calculated.  Since the detectors records data in the detector coordinate system, data in the utv coordinate 
		system must be converted into the global/object coordinate system.  The coordinate transformation can be 
		accomplished using a rotation matrix with an angle of rotation determined by the angle between the two 
		coordinate systems, which is the gantry_angle, in this case:

		Rotate ut-coordinate system to xy-coordinate system
				x = cos( gantry_angle ) * u - sin( gantry_angle ) * t
				y = sin( gantry_angle ) * u + cos( gantry_angle ) * t
		Rotate xy-coordinate system to ut-coordinate system
				u = cos( gantry_angle ) * x + sin( gantry_angle ) * y
				t = cos( gantry_angle ) * y - sin( gantry_angle ) * x

			 If a proton passes through the reconstruction volume, then the line defining its path in the
		xy-plane will intersect the circle defining the boundary of the reconstruction cylinder in the xy-plane twice.  
		We can determine if the proton path passes through the reconstruction volume by equating the equations of the 
		proton path and the circle.  This produces a second order polynomial which we must solve:

										  f(x)_proton = f(x)_cylinder
												 mx+b = sqrt(r^2 - x^2)
								  m^2x^2 + 2mbx + b^2 = r^2 - x^2
					(m^2 + 1)x^2 + 2mbx + (b^2 - r^2) = 0
										ax^2 + bx + c = 0
												=>	a = m^2 + 1
													b = 2mb
													c = b^2 - r^2

			We can solve this using the quadratic formula ([-b +/- sqrt(b^2-4ac)]/2a).  If the proton passed through the 
		reconstruction volume, then the determinant will be greater than zero ( b^2-4ac > 0 ) and the quadratic formula 
		will return two unique points of intersection.  The intersection point closest to where the proton entry/exit 
		path intersects the entry/exit
		detector plane is calculated and The proton entry/exit path If the determinant <= 0, then the proton path does not go through the reconstruction 
		volume and we need not determine intersection coordinates.  Two points are returned by the quadratic formula
		for each reconstruction cylinder intersection, the coordinates closest to the point where the entry/exit path
		intersected the detector plane are determined 

			If the exit/entry path travels through the cone bounded by y=|x| && y=-|x| the x_coordinates will be small
		and the difference between the entry and exit x-coordinates will approach zero, causing instabilities in trig
		functions and slope calculations ( x difference in denominator).  To overcome these innaccurate calculations, 
		coordinates for these proton paths will be rotated PI/2 radians(90 degrees) prior to calculations and rotated
		back when they are completed using a rotation matrix transformation again:
		
		 Positive Rotation By 90 Degrees
				x' = cos( 90 ) * x - sin( 90 ) * y = -y
				y' = sin( 90 ) * x + cos( 90 ) * y = x
		 Negative Rotation By 90 Degree
				x' = cos( 90 ) * x + sin( 90 ) * y = y
				y' = cos( 90 ) * y - sin( 90 ) * x = -x
	*/
	float a = 0, b = 0, c = 0;
	float x_intercept_1, x_intercept_2, y_intercept_1, y_intercept_2, squared_distance_1, squared_distance_2;
	float x_temp, y_temp;
	int i = threadIdx.x + blockIdx.x * THREADS_PER_BLOCK;
	float rotation_angle_radians = gantry_angle[i] * ANGLE_TO_RADIANS;
	traversed_recon_volume[i] = false;
	if( i < num_histories )
	{
		/***************************************************************************************************************/
		/**************************************** Check entry information **********************************************/
		/***************************************************************************************************************/
		
		// Determine if the proton path enters the reconstruction volume.  The proton path is defined using the entry angle and
		// position where the proton intersected the entry SSD which is closest to the object.  If this line projected onto the 
		// xy plane intersects the reconstruction cylinder, the line will intersect the circle in the xy plane which describes the
		// boundary of the reconstruction cylinder twice and its entry elevation will be within the height of the cylinder.   

		// Relevant angles in radians: gantry angle, proton path entry angle in ut and xy planes.
		float ut_entry_angle = atan2f( t_in_2[i] - t_in_1[i], u_in_2[i] - u_in_1[i] );	
		xy_entry_angle[i] = ut_entry_angle + rotation_angle_radians;
		if( xy_entry_angle[i] < 0 )
			xy_entry_angle[i] += TWO_PI;

		// Rotate entry detector positions
		float x_in = ( cosf( rotation_angle_radians ) * u_in_2[i] ) - ( sinf( rotation_angle_radians ) * t_in_2[i] );
		float y_in = ( sinf( rotation_angle_radians ) * u_in_2[i] ) + ( cosf( rotation_angle_radians ) * t_in_2[i] );

		// Determine if entry points should be rotated
		bool entry_in_cone = 
		( (xy_entry_angle[i] > PI_OVER_4) && (xy_entry_angle[i] < THREE_PI_OVER_4) ) 
		|| 
		( (xy_entry_angle[i] > FIVE_PI_OVER_4) && (xy_entry_angle[i] < SEVEN_PI_OVER_4) );

		// Rotate x_in & y_in by 90 degrees, if necessary
		if( entry_in_cone )
		{
			x_temp = x_in;
			y_temp = y_in;		
			x_in = -y_temp;
			y_in = x_temp;
			xy_entry_angle[i] += PI_OVER_2;
		}

		float m_in = tanf( xy_entry_angle[i] );	// proton entry path slope
		float b_in = y_in - m_in * x_in;				// proton entry path y-intercept
		
		// Quadratic formula coefficients
		a = 1 + pow(m_in, 2);								// x^2 coefficient 
		b = 2 * m_in * b_in;								// x coefficient
		c = pow(b_in, 2) - pow(RECON_CYL_RADIUS, 2 );		// 1 coefficient
		float entry_discriminant = pow(b, 2) - (4 * a * c);	// Quadratic formula discriminant		
		bool entered = ( entry_discriminant > 0 );			// Proton path intersected twice
		
		// Find both intersection points of the circle; closest one to the entry SSDs is the entry position
		// Notice that x_intercept_2 = ( -b - sqrt(...) ) / ( 2 * a ) has the negative sign pulled out and following calculations modified as necessary
		// e.g. x_intercept_2 = -x_real_2
		//		y_intercept_2 = -y_real_2
		//		squared_distance_2 = sqd_real_2		since (x_intercept_2 + x_in)^2 = (-x_intercept_2 - x_in)^2 = (x_real_2 - x_in)^2 (same for y term)
		// This negation is also considered when assigning x_entry/y_entry using -x_intercept_2/y_intercept_2 *(TRUE/FALSE = 1/0) 
		if( entered )
		{
			x_intercept_1 = ( sqrtf(entry_discriminant) - b ) / ( 2 * a );
			x_intercept_2 = ( sqrtf(entry_discriminant) + b ) / ( 2 * a );
			y_intercept_1 = m_in * x_intercept_1 + b_in;
			y_intercept_2 = m_in * x_intercept_2 - b_in;
			squared_distance_1 = pow(x_intercept_1 - x_in, 2) + pow(y_intercept_1 - y_in, 2);
			squared_distance_2 = pow(x_intercept_2 + x_in, 2) + pow(y_intercept_2 + y_in, 2);
			x_entry[i] = x_intercept_1 * (squared_distance_1 <= squared_distance_2) - x_intercept_2 * (squared_distance_1 > squared_distance_2);
			y_entry[i] = y_intercept_1 * (squared_distance_1 <= squared_distance_2) - y_intercept_2 * (squared_distance_1 > squared_distance_2);
		}
		// Unrotate by 90 degrees, if necessary
		if( entry_in_cone )
		{
			x_temp = x_entry[i];
			y_temp = y_entry[i];
			x_entry[i] = y_temp;
			y_entry[i] = -x_temp;
			xy_entry_angle[i] -= PI_OVER_2;
		}
		/***************************************************************************************************************/
		/****************************************** Check exit information *********************************************/
		/***************************************************************************************************************/
		
		// Repeat the procedure above, this time to determine if the proton path exited the reconstruction volume and if so, the
		// x,y,z position where it exited
		float ut_exit_angle = atan2f( t_out_2[i] - t_out_1[i], u_out_2[i] - u_out_1[i] );
		xy_exit_angle[i] = ut_exit_angle + rotation_angle_radians;
		if( xy_exit_angle[i] < 0 )
			xy_exit_angle[i] += TWO_PI;

		// Rotate exit detector positions
		float x_out = ( cosf(rotation_angle_radians) * u_out_1[i] ) - ( sinf(rotation_angle_radians) * t_out_1[i] );
		float y_out = ( sinf(rotation_angle_radians) * u_out_1[i] ) + ( cosf(rotation_angle_radians) * t_out_1[i] );

		// Determine if exit points should be rotated
		bool exit_in_cone = 
		( (xy_exit_angle[i] > PI_OVER_4) &&	(xy_exit_angle[i] < THREE_PI_OVER_4) ) 
		|| 
		( (xy_exit_angle[i] > FIVE_PI_OVER_4) && (xy_exit_angle[i] < SEVEN_PI_OVER_4) );

		// Rotate x_out & y_out by 90 degrees, if necessary
		if( exit_in_cone )
		{
			x_temp = x_out;
			y_temp = y_out;		
			x_out = -y_temp;
			y_out = x_temp;	
			xy_exit_angle[i] += PI_OVER_2;
		}	

		float m_out = tanf( xy_exit_angle[i] );	// proton entry path slope
		float b_out = y_out - m_out * x_out;			// proton entry path y-intercept
		
		// Quadratic formula coefficients
		a = 1 + pow(m_out, 2);								// x^2 coefficient 
		b = 2 * m_out * b_out;								// x coefficient
		c = pow(b_out, 2) - pow(RECON_CYL_RADIUS, 2);		// 1 coefficient
		float exit_discriminant = pow(b, 2)  - (4 * a * c); // Quadratic formula discriminant
		bool exited = ( exit_discriminant > 0 );			// Proton path intersected twice
		
			
		// Find both intersection points of the circle; closest one to the exit SSDs is the exit position
		if( exited )
		{
			x_intercept_1 = ( sqrtf(exit_discriminant) - b ) / ( 2 * a );
			x_intercept_2 = ( sqrtf(exit_discriminant) + b ) / ( 2 * a );// -x calculated
			y_intercept_1 = m_out * x_intercept_1 + b_out;
			y_intercept_2 = m_out * x_intercept_2 - b_out;// -y calculated
			squared_distance_1 = pow(x_intercept_1 - x_out, 2) + pow(y_intercept_1 - y_out, 2);
			squared_distance_2 = pow(x_intercept_2 + x_out, 2) + pow(y_intercept_2 + y_out, 2);// modified due to -x and -y calcs above
			x_exit[i] = x_intercept_1 * (squared_distance_1 <= squared_distance_2) - x_intercept_2 * (squared_distance_1 > squared_distance_2);
			y_exit[i] = y_intercept_1 * (squared_distance_1 <= squared_distance_2) - y_intercept_2 * (squared_distance_1 > squared_distance_2);
		}
		// Unrotate by 90 degrees, if necessary
		if( exit_in_cone )
		{
			x_temp = x_exit[i];
			y_temp = y_exit[i];
			x_exit[i] = y_temp;
			y_exit[i] = -x_temp;	
			xy_exit_angle[i] -= PI_OVER_2;
		}
		/***************************************************************************************************************/
		/***************************************** Check z(v) direction ************************************************/
		/***************************************************************************************************************/		

		// Relevant angles/slopes in radians for entry and exit in the uv plane
		float uv_entry_slope = ( v_in_2[i] - v_in_1[i] ) / ( u_in_2[i] - u_in_1[i] );
		float uv_exit_slope = ( v_out_2[i] - v_out_1[i] ) / ( u_out_2[i] - u_out_1[i] );
		
		float uv_entry_angle = atan2( v_in_2[i] - v_in_1[i], u_in_2[i] - u_in_1[i] );
		float uv_exit_angle = atan2( v_out_2[i] - v_out_1[i],  u_out_2[i] - u_out_1[i] );

		xz_entry_angle[i] = uv_entry_angle;
		xz_exit_angle[i] = uv_exit_angle;
		if( xz_entry_angle[i] < 0 )
			xz_entry_angle[i] += TWO_PI;
		if( xz_exit_angle[i] < 0 )
			xz_exit_angle[i] += TWO_PI;

		// Calculate the u coordinate for the entry and exit points of the reconstruction volume and then use the uv slope calculated 
		// from the detector entry and exit positions to determine the z position of the proton as it entered and exited the 
		// reconstruction volume 
		/*
			u-coordinate of the entry and exit points of the reconsruction cylinder can be found using an inverse rotation 
				u = cos( gantry_angle ) * x + sin( gantry_angle ) * y
		*/
		float u_entry = ( cosf( rotation_angle_radians ) * x_entry[i] ) + ( sinf( rotation_angle_radians ) * y_entry[i] );
		float u_exit = ( cosf(rotation_angle_radians) * x_exit[i] ) + ( sinf(rotation_angle_radians) * y_exit[i] );
		z_entry[i] = v_in_2[i] + uv_entry_slope * ( u_entry - u_in_2[i] );
		z_exit[i] = v_out_1[i] - uv_exit_slope * ( u_out_1[i] - u_exit );

		// Even if the proton path intersected the circle describing the boundary of the cylinder twice, it may not have actually
		// passed through the reconstruction volume or may have only passed through part way.  If |z_entry|> RECON_CYL_HEIGHT/2 ,
		// then something off happened since the the source is around z=0 and we do not want to use this history.  If the 
		// |z_entry| < RECON_CYL_HEIGHT/2 and |z_exit| > RECON_CYL_HEIGHT/2 then we want to use the history but the x_exit and
		// y_exit positions need to be calculated again based on how far through the cylinder the proton passed before exiting it
		if( entered && exited )
		{
			if( ( fabs(z_entry[i]) <= RECON_CYL_HEIGHT * 0.5 ) && ( fabs(z_exit[i]) > RECON_CYL_HEIGHT * 0.5 ) )
			{
				float recon_cyl_fraction = fabs( ( ( (z_exit[i] >= 0) - (z_exit[i] < 0) ) * RECON_CYL_HEIGHT * 0.5 - z_entry[i] ) / ( z_exit[i] - z_entry[i] ) );
				x_exit[i] = x_entry[i] + recon_cyl_fraction * ( x_exit[i] - x_entry[i] );
				y_exit[i] = y_entry[i] + recon_cyl_fraction * ( y_exit[i] - y_entry[i] );
				z_exit[i] = ( (z_exit[i] >= 0) - (z_exit[i] < 0) ) * RECON_CYL_HEIGHT * 0.5;
			}
			else if( fabs(z_entry[i]) > RECON_CYL_HEIGHT * 0.5 )
			{
				entered = false;
				exited = false;
			}
			// Check the measurement locations. Do not allow more than 5 cm difference in entry and exit in t and v. This gets 
			// rid of spurious events.
			if( ( fabs(t_out_1[i] - t_in_2[i]) > 5 ) || ( fabs(v_out_1[i] - v_in_2[i]) > 5 ) )
			{
				entered = false;
				exited = false;
			}
		}
		relative_ut_angle[i] = ut_exit_angle - ut_entry_angle;
		relative_uv_angle[i] = uv_exit_angle - uv_entry_angle;

		// Proton passed through the reconstruction volume only if it both entered and exited the reconstruction cylinder
		traversed_recon_volume[i] = entered && exited;
	}	
}
void bin_valid_histories( const int num_histories )
{
	unsigned int mem_size_hist_floats	= sizeof(float) * num_histories;
	unsigned int mem_size_hist_ints		= sizeof(int) * num_histories;
	unsigned int mem_size_hist_bool		= sizeof(bool) * num_histories;

	traversed_recon_volume_h	= (bool*)  calloc( num_histories, sizeof(bool)	);
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
	relative_ut_angle_h			= (float*) calloc( num_histories, sizeof(float) );
	relative_uv_angle_h			= (float*) calloc( num_histories, sizeof(float) );

	cudaMalloc((void**) &bin_num_d,	mem_size_hist_ints );
	cudaMemcpy( bin_num_d,	bin_num_h,	mem_size_hist_ints, cudaMemcpyHostToDevice );

	dim3 dimBlock( THREADS_PER_BLOCK );
	dim3 dimGrid( (int)( num_histories/THREADS_PER_BLOCK ) + 1 );
	bin_valid_histories_kernel<<<dimGrid, dimBlock>>>
	( 
		num_histories, bin_counts_d, bin_num_d, traversed_recon_volume_d, 
		x_entry_d, y_entry_d, z_entry_d, x_exit_d, y_exit_d, z_exit_d, 
		mean_WEPL_d, mean_rel_ut_angle_d, mean_rel_uv_angle_d, WEPL_d, 
		xy_entry_angle_d, xz_entry_angle_d, xy_exit_angle_d, xz_exit_angle_d,
		relative_ut_angle_d, relative_uv_angle_d
	);
	cudaMemcpy( traversed_recon_volume_h,	traversed_recon_volume_d,	mem_size_hist_bool,		cudaMemcpyDeviceToHost );
	cudaMemcpy( bin_num_h,					bin_num_d,					mem_size_hist_ints,		cudaMemcpyDeviceToHost );
	cudaMemcpy( x_entry_h,					x_entry_d,					mem_size_hist_floats,	cudaMemcpyDeviceToHost );
	cudaMemcpy( y_entry_h,					y_entry_d,					mem_size_hist_floats,	cudaMemcpyDeviceToHost );
	cudaMemcpy( z_entry_h,					z_entry_d,					mem_size_hist_floats,	cudaMemcpyDeviceToHost );
	cudaMemcpy( x_exit_h,					x_exit_d,					mem_size_hist_floats,	cudaMemcpyDeviceToHost );
	cudaMemcpy( y_exit_h,					y_exit_d,					mem_size_hist_floats,	cudaMemcpyDeviceToHost );
	cudaMemcpy( z_exit_h,					z_exit_d,					mem_size_hist_floats,	cudaMemcpyDeviceToHost );
	cudaMemcpy( xy_entry_angle_h,			xy_entry_angle_d,			mem_size_hist_floats,	cudaMemcpyDeviceToHost );
	cudaMemcpy( xz_entry_angle_h,			xz_entry_angle_d,			mem_size_hist_floats,	cudaMemcpyDeviceToHost );
	cudaMemcpy( xy_exit_angle_h,			xy_exit_angle_d,			mem_size_hist_floats,	cudaMemcpyDeviceToHost );
	cudaMemcpy( xz_exit_angle_h,			xz_exit_angle_d,			mem_size_hist_floats,	cudaMemcpyDeviceToHost );
	cudaMemcpy( relative_ut_angle_h,		relative_ut_angle_d,		mem_size_hist_floats,	cudaMemcpyDeviceToHost );
	cudaMemcpy( relative_uv_angle_h,		relative_uv_angle_d,		mem_size_hist_floats,	cudaMemcpyDeviceToHost );

	int offset = 0;
	for( int i = 0; i < num_histories; i++ )
	{
		if( traversed_recon_volume_h[i] && ( bin_num_h[i] >= 0 ) )
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
			//xy_exit_angle_vector.push_back( xy_exit_angle_h[i] );
			//xz_exit_angle_vector.push_back( xz_exit_angle_h[i] );
			relative_ut_angle_vector.push_back( relative_ut_angle_h[i] );
			relative_uv_angle_vector.push_back( relative_uv_angle_h[i] );
			offset++;
			recon_vol_histories++;
		}
	}
	printf( "%d out of %d histories passed intersection cuts this iteration\n", offset, num_histories );

	free( traversed_recon_volume_h ); 
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
	free( relative_ut_angle_h );
	free( relative_uv_angle_h );

	//cudaFree( bin_num_d );
	cudaFree( xy_entry_angle_d );
	cudaFree( xz_entry_angle_d );
	cudaFree( xy_exit_angle_d );
	cudaFree( xz_exit_angle_d );
	cudaFree( relative_ut_angle_d );
	cudaFree( relative_uv_angle_d );
}
__global__ void bin_valid_histories_kernel
( 
	int num_histories, int* bin_counts, int* bin_num, bool* traversed_recon_volume, 
	float* x_entry, float* y_entry, float* z_entry, float* x_exit, float* y_exit, float* z_exit, 
	float* mean_WEPL, float* mean_rel_ut_angle, float* mean_rel_uv_angle, float* WEPL, 
	float* xy_entry_angle, float* xz_entry_angle, float* xy_exit_angle, float* xz_exit_angle,
	float* relative_ut_angle, float* relative_uv_angle
)
{
	int i = threadIdx.x + blockIdx.x * THREADS_PER_BLOCK;
	if( i < num_histories )
	{
		if( traversed_recon_volume[i] )
		{
			float x_midpath, y_midpath, z_midpath, path_angle;
			int angle_bin, t_bin, v_bin;
			float angle, t, v;

			x_midpath = ( x_entry[i] + x_exit[i] ) / 2;
			y_midpath = ( y_entry[i] + y_exit[i] ) / 2;
			z_midpath = ( z_entry[i] + z_exit[i] ) / 2;

			path_angle = atan2( ( y_exit[i] - y_entry[i] ) , ( x_exit[i] - x_entry[i] ) );
			if( path_angle < 0 )
				path_angle += 2*PI;
			angle_bin = int( ( path_angle * RADIANS_TO_ANGLE / ANGULAR_BIN_SIZE ) + 0.5) % ANGULAR_BINS;	
			angle = angle_bin * ANGULAR_BIN_SIZE * ANGLE_TO_RADIANS;

			t = y_midpath * cosf(angle) - x_midpath * sinf(angle);
			t_bin = int( (t / T_BIN_SIZE ) + T_BINS/2);
			
			v = z_midpath;
			v_bin = int( (v / V_BIN_SIZE ) + V_BINS/2);

			if( (t_bin >= 0) && (v_bin >= 0) && (t_bin < T_BINS) && (v_bin < V_BINS) )
			{
				bin_num[i] = t_bin + angle_bin * T_BINS + v_bin * T_BINS * ANGULAR_BINS;
				atomicAdd( &bin_counts[bin_num[i]], 1 );
				atomicAdd( &mean_WEPL[bin_num[i]], WEPL[i] );
				atomicAdd( &mean_rel_ut_angle[bin_num[i]], relative_ut_angle[i] );
				atomicAdd( &mean_rel_uv_angle[bin_num[i]], relative_uv_angle[i] );
			}
			else
				bin_num[i] = -1;
		}
	}
}

/************************************************************************************************************************************************************/
/*************************************************************** Statistical Analysis and Cuts **************************************************************/
/************************************************************************************************************************************************************/
void calculate_means()
{
	dim3 dimBlock( T_BINS );
	dim3 dimGrid( V_BINS, ANGULAR_BINS );   
	calculate_means_kernel<<< dimGrid, dimBlock >>>
	( 
		bin_counts_d, mean_WEPL_d, mean_rel_ut_angle_d, mean_rel_uv_angle_d
	);

	//cudaMemcpy( bin_counts_h,	bin_counts_d,	MEM_SIZE_BINS_INTS, cudaMemcpyDeviceToHost );
	//cudaMemcpy( mean_WEPL_h,	mean_WEPL_d,	MEM_SIZE_BINS_FLOATS, cudaMemcpyDeviceToHost );
	//cudaMemcpy( mean_rel_ut_angle_h,	mean_rel_ut_angle_d,	MEM_SIZE_BINS_FLOATS, cudaMemcpyDeviceToHost );
	//cudaMemcpy( mean_rel_uv_angle_h,	mean_rel_uv_angle_d,	MEM_SIZE_BINS_FLOATS, cudaMemcpyDeviceToHost );

	//write_array_to_disk("bin_counts_h_pre", output_directory, output_folder, bin_counts_h, T_BINS, ANGULAR_BINS, V_BINS, NUM_BINS, true );
	//write_array_to_disk("mean_WEPL_h", output_directory, output_folder, mean_WEPL_h, T_BINS, ANGULAR_BINS, V_BINS, NUM_BINS, true );
	//write_array_to_disk("mean_rel_ut_angle_h", output_directory, output_folder, mean_rel_ut_angle_h, T_BINS, ANGULAR_BINS, V_BINS, NUM_BINS, true );
	//write_array_to_disk("mean_rel_uv_angle_h", output_directory, output_folder, mean_rel_uv_angle_h, T_BINS, ANGULAR_BINS, V_BINS, NUM_BINS, true );
	
	free(bin_counts_h);
	free(mean_WEPL_h);
	free(mean_rel_ut_angle_h);
	free(mean_rel_uv_angle_h);
}
__global__ void calculate_means_kernel( int* bin_counts, float* mean_WEPL, float* mean_rel_ut_angle, float* mean_rel_uv_angle )
{
	int v = blockIdx.x;
	int angle = blockIdx.y;
	int t = threadIdx.x;
	int bin = t + angle * T_BINS + v * T_BINS * ANGULAR_BINS;
	if( bin_counts[bin] > 0 )
	{
		mean_WEPL[bin] /= bin_counts[bin];		
		mean_rel_ut_angle[bin] /= bin_counts[bin];
		mean_rel_uv_angle[bin] /= bin_counts[bin];
	}
}
void sum_differences( const int start_position, const int num_histories )
{
	unsigned int mem_size_hist_floats = sizeof(float) * num_histories;
	unsigned int mem_size_hist_ints = sizeof(int) * num_histories;

	cudaMalloc((void**) &bin_num_d,				mem_size_hist_ints);
	cudaMalloc((void**) &WEPL_d,				mem_size_hist_floats);
	cudaMalloc((void**) &xy_entry_angle_d,		mem_size_hist_floats);
	cudaMalloc((void**) &xz_entry_angle_d,		mem_size_hist_floats);
	cudaMalloc((void**) &xy_exit_angle_d,		mem_size_hist_floats);
	cudaMalloc((void**) &xz_exit_angle_d,		mem_size_hist_floats);
	//cudaMalloc((void**) &xy_exit_angle_d,		mem_size_hist_floats);
	//cudaMalloc((void**) &xz_exit_angle_d,		mem_size_hist_floats);
	cudaMalloc((void**) &relative_ut_angle_d,	mem_size_hist_floats);
	cudaMalloc((void**) &relative_uv_angle_d,	mem_size_hist_floats);

	cudaMemcpy( bin_num_d,				&bin_num_vector[start_position],			mem_size_hist_ints, cudaMemcpyHostToDevice);
	cudaMemcpy( WEPL_d,					&WEPL_vector[start_position],				mem_size_hist_floats, cudaMemcpyHostToDevice);
	cudaMemcpy( xy_entry_angle_d,		&xy_entry_angle_vector[start_position],		mem_size_hist_floats, cudaMemcpyHostToDevice);
	cudaMemcpy( xz_entry_angle_d,		&xz_entry_angle_vector[start_position],		mem_size_hist_floats, cudaMemcpyHostToDevice);
	//cudaMemcpy( xy_exit_angle_d,		&xy_exit_angle_vector[start_position],		mem_size_hist_floats, cudaMemcpyHostToDevice);
	//cudaMemcpy( xz_exit_angle_d,		&xz_exit_angle_vector[start_position],		mem_size_hist_floats, cudaMemcpyHostToDevice);
	cudaMemcpy( relative_ut_angle_d,	&relative_ut_angle_vector[start_position],	mem_size_hist_floats, cudaMemcpyHostToDevice);
	cudaMemcpy( relative_uv_angle_d,	&relative_uv_angle_vector[start_position],	mem_size_hist_floats, cudaMemcpyHostToDevice);

	dim3 dimBlock(THREADS_PER_BLOCK);
	dim3 dimGrid((int)(num_histories/THREADS_PER_BLOCK)+1);
	sum_differences_kernel<<<dimGrid, dimBlock>>>
	( 
		num_histories, bin_num_d, mean_WEPL_d, mean_rel_ut_angle_d, mean_rel_uv_angle_d, 
		WEPL_d, xy_entry_angle_d, xz_entry_angle_d,  xy_entry_angle_d, xz_entry_angle_d,//xy_exit_angle_d, xz_exit_angle_d,
		stddev_WEPL_d, stddev_rel_ut_angle_d, stddev_rel_uv_angle_d, relative_ut_angle_d, relative_uv_angle_d
	);
	cudaFree( bin_num_d );
	cudaFree( WEPL_d );
	cudaFree( xy_entry_angle_d );
	cudaFree( xz_entry_angle_d );
	//cudaFree( xy_exit_angle_d );
	//cudaFree( xz_exit_angle_d );
	cudaFree( relative_ut_angle_d );
	cudaFree( relative_uv_angle_d );
}
__global__ void sum_differences_kernel
( 
	int num_histories, int* bin_num, float* mean_WEPL, float* mean_rel_ut_angle, float* mean_rel_uv_angle,  
	float* WEPL, float* xy_entry_angle, float* xz_entry_angle, float* xy_exit_angle, float* xz_exit_angle,
	float* stddev_WEPL, float* stddev_rel_ut_angle, float* stddev_rel_uv_angle, float* relative_ut_angle, float* relative_uv_angle 
)
{
	float WEPL_difference, rel_ut_angle_difference, rel_uv_angle_difference;

	int i = threadIdx.x + blockIdx.x * THREADS_PER_BLOCK;
	if( i < num_histories )
	{
	/*	float ut_diff = xy_exit_angle[i] - xy_entry_angle[i];
		if( fabs(ut_diff) > PI )
		{
			printf("Hello\n");
			if( xy_entry_angle[i] > PI )
				xy_entry_angle[i] -= TWO_PI;
			if( xy_exit_angle[i] > PI )
				xy_exit_angle[i] -= TWO_PI;
			ut_diff = xy_exit_angle[i] - xy_entry_angle[i];
		}
		float uv_diff = xz_exit_angle[i] - xz_entry_angle[i];
		if( fabs(uv_diff) > PI )
		{
			if( xz_entry_angle[i] > PI )
				xz_entry_angle[i] -= TWO_PI;
			if( xz_exit_angle[i] > PI )
				xz_exit_angle[i] -= TWO_PI;
			uv_diff = xz_exit_angle[i] - xz_entry_angle[i];
		}*/
		WEPL_difference = WEPL[i] - mean_WEPL[bin_num[i]];
		rel_ut_angle_difference = relative_ut_angle[i] - mean_rel_ut_angle[bin_num[i]];
		rel_uv_angle_difference = relative_uv_angle[i] - mean_rel_uv_angle[bin_num[i]];
		//rel_ut_angle_difference = ut_diff - mean_rel_ut_angle[bin_num[i]];
		//rel_uv_angle_difference = uv_diff - mean_rel_uv_angle[bin_num[i]];

		atomicAdd( &stddev_WEPL[bin_num[i]], WEPL_difference * WEPL_difference);
		atomicAdd( &stddev_rel_ut_angle[bin_num[i]], rel_ut_angle_difference * rel_ut_angle_difference );
		atomicAdd( &stddev_rel_uv_angle[bin_num[i]], rel_uv_angle_difference * rel_uv_angle_difference );
	}
}
void calculate_std_devs()
{
	dim3 dimBlock( T_BINS );
	dim3 dimGrid( V_BINS, ANGULAR_BINS );   
	calculate_std_devs_kernel<<< dimGrid, dimBlock >>>
	( 
		bin_counts_d, stddev_WEPL_d, stddev_rel_ut_angle_d, stddev_rel_uv_angle_d
	);
	//cudaFree( bin_counts_d );
}
__global__ void calculate_std_devs_kernel( int* bin_counts, float* stddev_WEPL, float* stddev_rel_ut_angle, float* stddev_rel_uv_angle )
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
	unsigned int mem_size_hist_floats = sizeof(float) * num_histories;
	unsigned int mem_size_hist_ints = sizeof(int) * num_histories;
	unsigned int mem_size_hist_bools = sizeof(bool) * num_histories;

	passed_cuts_h = (bool*) calloc (num_histories, sizeof(bool) );
	
	cudaMalloc( (void**) &bin_num_d,			mem_size_hist_ints );
	cudaMalloc( (void**) &WEPL_d,				mem_size_hist_floats );
	cudaMalloc( (void**) &xy_entry_angle_d,		mem_size_hist_floats );
	cudaMalloc( (void**) &xz_entry_angle_d,		mem_size_hist_floats );
	//cudaMalloc( (void**) &xy_exit_angle_d,		mem_size_hist_floats );
	//cudaMalloc( (void**) &xz_exit_angle_d,		mem_size_hist_floats );
	cudaMalloc( (void**) &relative_ut_angle_d,	mem_size_hist_floats );
	cudaMalloc( (void**) &relative_uv_angle_d,	mem_size_hist_floats );
	cudaMalloc( (void**) &passed_cuts_d,		mem_size_hist_bools );

	cudaMemcpy( bin_num_d,				&bin_num_vector[start_position],			mem_size_hist_ints,		cudaMemcpyHostToDevice );
	cudaMemcpy( WEPL_d,					&WEPL_vector[start_position],				mem_size_hist_floats,	cudaMemcpyHostToDevice );
	cudaMemcpy( xy_entry_angle_d,		&xy_entry_angle_vector[start_position],		mem_size_hist_floats,	cudaMemcpyHostToDevice );
	cudaMemcpy( xz_entry_angle_d,		&xz_entry_angle_vector[start_position],		mem_size_hist_floats,	cudaMemcpyHostToDevice );
	//cudaMemcpy( xy_exit_angle_d,		&xy_exit_angle_vector[start_position],		mem_size_hist_floats,	cudaMemcpyHostToDevice );
	//cudaMemcpy( xz_exit_angle_d,		&xz_exit_angle_vector[start_position],		mem_size_hist_floats,	cudaMemcpyHostToDevice );
	cudaMemcpy( relative_ut_angle_d,	&relative_ut_angle_vector[start_position],	mem_size_hist_floats,	cudaMemcpyHostToDevice );
	cudaMemcpy( relative_uv_angle_d,	&relative_uv_angle_vector[start_position],	mem_size_hist_floats,	cudaMemcpyHostToDevice );
	cudaMemcpy( passed_cuts_d,			passed_cuts_h,								mem_size_hist_bools,	cudaMemcpyHostToDevice );
	//puts("Before kernel");
	dim3 dimBlock(THREADS_PER_BLOCK);
	dim3 dimGrid( int( num_histories / THREADS_PER_BLOCK ) + 1 );  
	statistical_cuts_kernel<<< dimGrid, dimBlock >>>
	( 
		num_histories, bin_counts_d, bin_num_d, sinogram_d, WEPL_d, 
		xy_entry_angle_d, xz_entry_angle_d, xy_entry_angle_d, xz_entry_angle_d,//xy_exit_angle_d, xz_exit_angle_d, 
		mean_WEPL_d, mean_rel_ut_angle_d, mean_rel_uv_angle_d, 
		stddev_WEPL_d, stddev_rel_ut_angle_d, stddev_rel_uv_angle_d, 
		passed_cuts_d, relative_ut_angle_d, relative_uv_angle_d
	);
	//puts("After kernel");
	cudaMemcpy( passed_cuts_h, passed_cuts_d, mem_size_hist_bools, cudaMemcpyDeviceToHost);

	//printf("start iteration %d\n", iteration );
	for( int i = 0; i < num_histories; i++ )
	{
		if( passed_cuts_h[i] )
		{
			//printf("start i = %d\n", i );
			//printf("index = %d\n", start_position + i );
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
			//xy_exit_angle_vector[post_cut_histories] = xy_exit_angle_vector[start_position + i];
			//xz_exit_angle_vector[post_cut_histories] = xz_exit_angle_vector[start_position + i];
			relative_ut_angle_vector[post_cut_histories] = relative_ut_angle_vector[start_position + i];
			relative_uv_angle_vector[post_cut_histories] = relative_uv_angle_vector[start_position + i];
			post_cut_histories++;
		}
	}
	//printf("end iteration %d\n", iteration );
}
__global__ void statistical_cuts_kernel
( 
	int num_histories, int* bin_counts, int* bin_num, float* sinogram, float* WEPL, 
	float* xy_entry_angle, float* xz_entry_angle, float* xy_exit_angle, float* xz_exit_angle, 
	float* mean_WEPL, float* mean_rel_ut_angle, float* mean_rel_uv_angle,
	float* stddev_WEPL, float* stddev_rel_ut_angle, float* stddev_rel_uv_angle, 
	bool* passed_cuts, float* relative_ut_angle, float* relative_uv_angle	
)
{
	int i = threadIdx.x + blockIdx.x * THREADS_PER_BLOCK;
	if( i < num_histories )
	{
		/*float ut_diff = xy_exit_angle[i] - xy_entry_angle[i];
		if( ut_diff > PI )
		{
			if( xy_entry_angle[i] > PI )
				xy_entry_angle[i] -= TWO_PI;
			if( xy_exit_angle[i] > PI )
				xy_exit_angle[i] -= TWO_PI;
			ut_diff = xy_exit_angle[i] - xy_entry_angle[i];
		}
		float uv_diff = xz_exit_angle[i] - xz_entry_angle[i];
		if( uv_diff > PI )
		{
			if( xz_entry_angle[i] > PI )
				xz_entry_angle[i] -= TWO_PI;
			if( xz_exit_angle[i] > PI )
				xz_exit_angle[i] -= TWO_PI;
			uv_diff = xz_exit_angle[i] - xz_entry_angle[i];
		}*/
		bool passed_ut_cut = ( fabs( relative_ut_angle[i] - mean_rel_ut_angle[bin_num[i]] ) < ( SIGMAS_TO_KEEP * stddev_rel_ut_angle[bin_num[i]] ) );
		bool passed_uv_cut = ( fabs( relative_uv_angle[i] - mean_rel_uv_angle[bin_num[i]] ) < ( SIGMAS_TO_KEEP * stddev_rel_uv_angle[bin_num[i]] ) );
		/*bool passed_ut_cut = ( fabs( ut_diff - mean_rel_ut_angle[bin_num[i]] ) < ( SIGMAS_TO_KEEP * stddev_rel_ut_angle[bin_num[i]] ) );
		bool passed_uv_cut = ( fabs( uv_diff - mean_rel_uv_angle[bin_num[i]] ) < ( SIGMAS_TO_KEEP * stddev_rel_uv_angle[bin_num[i]] ) );*/
		bool passed_WEPL_cut = ( fabs( mean_WEPL[bin_num[i]] - WEPL[i] ) <= ( SIGMAS_TO_KEEP * stddev_WEPL[bin_num[i]] ) );
		passed_cuts[i] = passed_ut_cut && passed_uv_cut && passed_WEPL_cut;

		if( passed_cuts[i] )
		{
			atomicAdd( &sinogram[bin_num[i]], WEPL[i] );
			atomicAdd( &bin_counts[bin_num[i]], 1 );
		}
	}
}
/************************************************************************************************************************************************************/
/*********************************************************************** MLP ********************************************************************************/
/************************************************************************************************************************************************************/
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
void MLP_test()
{
	char user_response[20];
	float x_entry = -3.0;
	float y_entry = -sqrtf( pow(MLP_IMAGE_RECON_CYL_RADIUS, 2) - pow(x_entry,2) );
	float z_entry = 0.0;
	float x_exit = 2.5;
	float y_exit = sqrtf( pow(MLP_IMAGE_RECON_CYL_RADIUS, 2) - pow(x_exit,2) );
	float z_exit = 0.0;
	float xy_entry_angle = 25 * PI/180, xz_entry_angle = 0.0;
	float xy_exit_angle = 45* PI/180, xz_exit_angle = 0.0;
	float x_in_object, y_in_object, z_in_object;
	float u_in_object, t_in_object, v_in_object;
	float x_out_object, y_out_object, z_out_object;
	float u_out_object, t_out_object, v_out_object;

	/********************************************************************************************/
	/**************************** Status Tracking Information ***********************************/
	/********************************************************************************************/
	float voxel_x, voxel_y, voxel_z;
	int voxel;
	int x_move_direction, y_move_direction, z_move_direction;
	int x_voxel_step, y_voxel_step, z_voxel_step;
	float x, y, z;
	float x_inside, y_inside, z_inside;
	float x_to_go, y_to_go, z_to_go;
	float delta_x, delta_y, delta_z;
	float x_extension, y_extension;
	float x_move, y_move, z_move;
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
	delta_y = tanf( xy_entry_angle ) * MLP_IMAGE_VOXEL_WIDTH;
	delta_z = tanf( xz_entry_angle ) * MLP_IMAGE_VOXEL_WIDTH;
	if( x_entry == x_exit )
	{
		delta_x = 0;
		delta_y = MLP_IMAGE_VOXEL_HEIGHT;
		delta_z = tanf(xz_entry_angle) / tanf(xy_entry_angle) * MLP_IMAGE_VOXEL_HEIGHT;
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
	x_move_direction = ( cosf(xy_entry_angle) >= 0 ) - ( cosf(xy_entry_angle) < 0 );
	y_move_direction = ( sinf(xy_entry_angle) >= 0 ) - ( sinf(xy_entry_angle) < 0 );
	z_move_direction = ( sinf(xy_entry_angle) >= 0 ) - ( sinf(xy_entry_angle) < 0 );
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
	delta_y = tanf( xy_exit_angle ) * MLP_IMAGE_VOXEL_WIDTH;
	delta_z = tanf( xz_exit_angle ) * MLP_IMAGE_VOXEL_WIDTH;
	if( x_entry == x_exit )
	{
		delta_x = 0;
		delta_y = MLP_IMAGE_VOXEL_HEIGHT;
		delta_z = tanf(xz_exit_angle) / tanf(xy_exit_angle) * MLP_IMAGE_VOXEL_HEIGHT;
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
	x_move_direction = ( cosf(xy_exit_angle) < 0 ) - ( cosf(xy_exit_angle) >= 0 );
	y_move_direction = ( sinf(xy_exit_angle) < 0 ) - ( sinf(xy_exit_angle) >= 0 );
	z_move_direction = ( sinf(xy_exit_angle) < 0 ) - ( sinf(xy_exit_angle) >= 0 );
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
	//printf("voxel_x = %3f \nvoxel_y = %3f \nvoxel_z = %3f\n", voxel_x, voxel_y, voxel_z);
	//printf("x_inside = %3f y_inside = %3f z_inside = %3f\n", x_inside, y_inside, z_inside);	
	voxel = int(voxel_x + voxel_y * MLP_IMAGE_COLUMNS + voxel_z * MLP_IMAGE_COLUMNS * MLP_IMAGE_ROWS);

	int path[1000];
	int path_index = 0;
	double chord_lengths[1000];
	MLP_test_image_h[voxel] = 0;
	path[path_index++] = voxel;

	u_in_object = ( cosf( xy_entry_angle ) * x_in_object ) + ( sinf( xy_entry_angle ) * y_in_object );
	u_out_object = ( cosf( xy_entry_angle ) * x_out_object ) + ( sinf( xy_entry_angle ) * y_out_object );
	t_in_object = ( cosf( xy_entry_angle ) * y_in_object ) - ( sinf( xy_entry_angle ) * x_in_object );
	t_out_object = ( cosf( xy_entry_angle ) * y_out_object ) - ( sinf( xy_entry_angle ) * x_out_object );
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
		float sigma_t1 = (A_0/3)*pow(u_1, 3.0) + (A_1/12)*pow(u_1, 4.0) + (A_2/30)*pow(u_1, 5.0) + (A_3/60)*pow(u_1, 6.0) + (A_4/105)*pow(u_1, 7.0) + (A_5/168)*pow(u_1, 8.0);
		float sigma_t1_theta1 = pow(u_1, 2.0 )*( (A_0/2) + (A_1/6)*u_1 + (A_2/12)*pow(u_1, 2.0) + (A_3/20)*pow(u_1, 3.0) + (A_4/30)*pow(u_1, 4.0) + (A_5/42)*pow(u_1, 5.0) );
		float sigma_theta1 = A_0*u_1 + (A_1/2)*pow(u_1, 2.0) + (A_2/3)*pow(u_1, 3.0) + (A_3/4)*pow(u_1, 4.0) + (A_4/5)*pow(u_1, 5.0) + (A_5/6)*pow(u_1, 6.0);	
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
		double x_1 = ( cosf( xy_entry_angle ) * (u_in_object + u_1) ) - ( sinf( xy_entry_angle ) * t_1 );
		double y_1 = ( sinf( xy_entry_angle ) * (u_in_object + u_1) ) + ( cosf( xy_entry_angle ) * t_1 );
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
/************************************************************************************************************************************************************/
/************************************************************************ FBP *******************************************************************************/
/************************************************************************************************************************************************************/
void initialize_sinogram()
{
	sinogram_h = (float*) calloc( NUM_BINS, sizeof(float) );
	cudaMalloc((void**) &sinogram_d, MEM_SIZE_BINS_FLOATS );
	cudaMemcpy( sinogram_d,	sinogram_h,	MEM_SIZE_BINS_FLOATS, cudaMemcpyHostToDevice );	
}
void construct_sinogram()
{
	dim3 dimBlock( T_BINS );
	dim3 dimGrid( V_BINS, ANGULAR_BINS );   
	construct_sinogram_kernel<<< dimGrid, dimBlock >>>( bin_counts_d, sinogram_d );

	//cudaMemcpy(sinogram_h,  sinogram_d, MEM_SIZE_BINS_FLOATS, cudaMemcpyDeviceToHost);
	//write_array_to_disk("sinogram", output_directory, output_folder, sinogram_h, T_BINS, ANGULAR_BINS, V_BINS, NUM_BINS, false );

	//bin_counts_h		  = (int*)	 calloc( NUM_BINS, sizeof(int) );
	//cudaMemcpy(bin_counts_h, bin_counts_d, MEM_SIZE_BINS_INTS, cudaMemcpyDeviceToHost) ;
	//write_array_to_disk( "bin_counts_post", output_directory, output_folder, bin_counts_h, T_BINS, ANGULAR_BINS, V_BINS, NUM_BINS, true );
}
__global__ void construct_sinogram_kernel( int* bin_counts, float* sinogram )
{
	int v = blockIdx.x, angle = blockIdx.y, t = threadIdx.x;
	int bin = t + angle * T_BINS + v * T_BINS * ANGULAR_BINS;
	if( bin_counts[bin] > 0 )
		sinogram[bin] /= bin_counts[bin];		
}
void filter()
{
	puts("Doing the filtering...");		
	sinogram_filtered_h = (float*) calloc( NUM_BINS, sizeof(float) );
	
	cudaMalloc((void**) &sinogram_filtered_d, MEM_SIZE_BINS_FLOATS);
	cudaMemcpy( sinogram_filtered_d, sinogram_filtered_h, MEM_SIZE_BINS_FLOATS, cudaMemcpyHostToDevice);

	dim3 dimBlock( T_BINS );
	dim3 dimGrid( V_BINS, ANGULAR_BINS );   	
	filter_kernel<<< dimGrid, dimBlock >>>( sinogram_d, sinogram_filtered_d );

	cudaMemcpy(sinogram_filtered_h, sinogram_filtered_d, MEM_SIZE_BINS_FLOATS, cudaMemcpyDeviceToHost) ;

	free(sinogram_h);

	cudaFree(sinogram_d);
	cudaFree(sinogram_filtered_d);
}
__global__ void filter_kernel( float* sinogram, float* sinogram_filtered )
{	
	int t_bin_ref,angle_bin,t_bin,v_bin,t_bin_sep;
	float filtered,t,v,scale_factor;
	
	v_bin = blockIdx.x;
	angle_bin = blockIdx.y;
	t_bin = threadIdx.x;
	
	v = ( v_bin - V_BINS/2 ) * V_BIN_SIZE + V_BIN_SIZE/2.0;
	
	// Loop over strips for this strip
	for( t_bin_ref = 0; t_bin_ref < T_BINS; t_bin_ref++ )
	{
		t = ( t_bin_ref - T_BINS/2 ) * T_BIN_SIZE + T_BIN_SIZE/2.0;
		t_bin_sep = t_bin - t_bin_ref;
		// scale_factor = r . path = cos(theta_{r,path})
		scale_factor = SOURCE_RADIUS / sqrtf( SOURCE_RADIUS * SOURCE_RADIUS + t * t + v * v );
		
		switch( FILTER_NUM )
		{
			case 0: // Ram-Lak
				if( t_bin_sep == 0 )
					filtered = 1.0 / ( 8.0 * powf( T_BIN_SIZE, 2.0 ) );
				else if( t_bin_sep % 2 == 0 )
					filtered = 0;
				else
					filtered = -1.0 / ( 2.0 * powf( T_BIN_SIZE * PI * t_bin_sep, 2.0 ) );					
			case 1: // Shepp-Logan filter
				filtered = powf( powf(T_BIN_SIZE * PI, 2.0) * ( 1.0 - powf(2 * t_bin_sep, 2.0) ), -1.0 );
		}
		int strip_index = ( v_bin * ANGULAR_BINS * T_BINS ) + ( angle_bin * T_BINS );
		sinogram_filtered[strip_index + t_bin] += T_BIN_SIZE * sinogram[strip_index + t_bin_ref] * filtered * scale_factor;
	}
}
void backprojection()
{
	puts("Doing the backprojection...");
	printf("DEBUG: MEM_SIZE_IMAGE_FLOAT = %u\n", MEM_SIZE_IMAGE_FLOAT); 

	// Allocate host memory
	puts("DEBUG: Allocate host memory");

	char user_response[20];
	X_h = (float*) calloc( VOXELS, sizeof(float) );

	if( X_h == NULL ) 
	{
		printf("ERROR: Memory not allocated for X_h!\n");
		fgets(user_response, sizeof(user_response), stdin);
		exit(1);
	}
	
	// Check that we don't have any corruptions up until now
	for( int i = 0; i < NUM_BINS; i++ )
		if( sinogram_filtered_h[i] != sinogram_filtered_h[i] )
			printf("We have a nan in bin #%d\n", i);

	float delta = GANTRY_ANGLE_INTERVAL * ANGLE_TO_RADIANS;
	// Loop over the voxels
	for( int slice = 0; slice < SLICES; slice++ )
	{
		for( int column = 0; column < COLUMNS; column++ )
		{

			for( int row = 0; row < ROWS; row++ )
			{
				float x = -RECON_CYL_RADIUS + ( column + 0.5 )* VOXEL_WIDTH;
				float y = RECON_CYL_RADIUS - (row + 0.5) * VOXEL_HEIGHT;
				float z = -RECON_CYL_HEIGHT / 2.0 + (slice + 0.5) * SLICE_THICKNESS;
				//// If the voxel is outside a cylinder contained in the reconstruction volume, set to air
				if( ( x * x + y * y ) > ( RECON_CYL_RADIUS * RECON_CYL_RADIUS ) )
					X_h[( slice * COLUMNS * ROWS) + ( row * COLUMNS ) + column] = 0.00113;							
				else
				{	  
					// Sum over projection angles
					for( int angle_bin = 0; angle_bin < ANGULAR_BINS; angle_bin++ )
					{
						// Rotate the pixel position to the beam-detector co-ordinate system
						float u = x * cosf( angle_bin * delta ) + y * sinf( angle_bin * delta );
						float t = -x * sinf( angle_bin * delta ) + y * cosf( angle_bin * delta );
						float v = z;

						// Project to find the detector number
						float detector_number_t = ( t - u *( t / ( SOURCE_RADIUS + u ) ) ) / T_BIN_SIZE + T_BINS/2.0;
						int t_bin = int( detector_number_t);
						if( t_bin > detector_number_t )
							t_bin -= 1;
						float eta = detector_number_t - t_bin;

						// Now project v to get detector number in v axis
						float detector_number_v = ( v - u * ( v / ( SOURCE_RADIUS + u ) ) ) / V_BIN_SIZE + V_BINS/2.0;
						int v_bin = int( detector_number_v);
						if( v_bin > detector_number_v )
							v_bin -= 1;
						float epsilon = detector_number_v - v_bin;

						// Calculate the fan beam scaling factor
						float scale_factor = powf( SOURCE_RADIUS / ( SOURCE_RADIUS + u ), 2 );
		  
						//bin_num[i] = t_bin + angle_bin * T_BINS + v_bin * T_BINS * ANGULAR_BINS;
						// Compute the back-projection
						int bin = t_bin + angle_bin * T_BINS + v_bin * ANGULAR_BINS * T_BINS;
						int voxel = slice * COLUMNS * ROWS + row * COLUMNS + column;
						// not sure why this won't compile without calculating the index ahead of time instead inside []s
						int index = ANGULAR_BINS * T_BINS;

						//if( ( ( bin + ANGULAR_BINS * T_BINS + 1 ) >= NUM_BINS ) || ( bin < 0 ) );
						if( v_bin == V_BINS - 1 || ( bin < 0 ) )
						{
							X_h[voxel] += delta * 2 *( ( 1 - eta ) * ( 1 - epsilon ) * sinogram_filtered_h[bin] 
							+ eta * ( 1 - epsilon ) * sinogram_filtered_h[bin + 1]) * scale_factor;
						}
							//printf("The bin selected for this voxel does not exist!\n Slice: %d\n Column: %d\n Row: %d\n", slice, column, row);
						else 
						{
							// not sure why this won't compile without calculating the index ahead of time instead inside []s
							/*X_h[voxel] += delta * ( ( 1 - eta ) * ( 1 - epsilon ) * sinogram_filtered_h[bin] 
							+ eta * ( 1 - epsilon ) * sinogram_filtered_h[bin + 1]
							+ ( 1 - eta ) * epsilon * sinogram_filtered_h[bin + index]
							+ eta * epsilon * sinogram_filtered_h[bin + index + 1] ) * scale_factor;*/
							X_h[voxel] += delta * ( ( 1 - eta ) * ( 1 - epsilon ) * sinogram_filtered_h[bin] 
							+ eta * ( 1 - epsilon ) * sinogram_filtered_h[bin + 1]
							+ ( 1 - eta ) * epsilon * sinogram_filtered_h[bin + index]
							+ eta * epsilon * sinogram_filtered_h[bin + index + 1] ) * scale_factor;

							// Multilpying by the gantry angle interval for each gantry angle is equivalent to multiplying the final answer by 2*PI and is better numerically
							// so multiplying by delta each time should be replaced by X_h[voxel] *= 2 * PI after all contributions have been made, which is commented out below
							/*X_h[voxel] += scale_factor * ( ( 1 - eta ) * ( 1 - epsilon ) * sinogram_filtered_h[bin] 
							+ eta * ( 1 - epsilon ) * sinogram_filtered_h[bin + 1]
							+ ( 1 - eta ) * epsilon * sinogram_filtered_h[bin + index]
							+ eta * epsilon * sinogram_filtered_h[bin + index + 1] );*/

							if(X_h[voxel]!=X_h[voxel])
								printf("We have a nan in slice %d, column %d, and row %d\n", slice, column, row);
						}
						//X_h[voxel] *= 2 * PI; 
					}
				}
			}
		}
	}
	free(sinogram_filtered_h);
	FBP_object_h = (int*) calloc( COLUMNS * ROWS * SLICES, sizeof(int) );

	for( int slice = 0; slice < SLICES; slice++ )
	{
		for( int row = 0; row < ROWS; row++ )
		{
			for( int column = 0; column < COLUMNS; column++ )
			{
				float x = -RECON_CYL_RADIUS + ( column + 0.5 )* VOXEL_WIDTH;
				float y = RECON_CYL_RADIUS - (row + 0.5) * VOXEL_HEIGHT;
				float d_squared = powf(x, 2) + powf(y, 2);
				if(X_h[( slice * COLUMNS * ROWS ) + ( row * COLUMNS ) + column] > FBP_THRESHOLD && (d_squared < powf(RECON_CYL_RADIUS, 2) ) ) 
					FBP_object_h[( slice * COLUMNS * ROWS ) + ( row * COLUMNS ) + column] = 1; 
				else
					FBP_object_h[( slice * COLUMNS * ROWS ) + ( row * COLUMNS ) + column] = 0; 
			}

		}
	}
	//write_array_to_disk( "FBP_object", output_directory, output_folder, FBP_object_h, COLUMNS, ROWS, SLICES, VOXELS, false );
	write_array_to_disk( "X_h", output_directory, output_folder, X_h, COLUMNS, ROWS, SLICES, VOXELS, false );
	write_array_to_disk( "x_FBP", output_directory, output_folder, FBP_object_h, COLUMNS, ROWS, SLICES, VOXELS, true );
}
/************************************************************************************************************************************************************/
/****************************************************************** Image Initialization  *******************************************************************/
/************************************************************************************************************************************************************/
void initialize_SC_hull( bool*& SC_hull_h, bool*& SC_hull_d )
{
	/* Allocate Memory and Initialize Images for Hull Detection Algorithms.  Use the Image and	*/
	/* Reconstruction Cylinder Parameters to Determine the Location of the Perimeter of the		*/
	/* Reconstruction Cylinder, Which is Centered on the Origin (Center) of the Image.  Assign	*/
	/* Voxels Inside the Perimeter of the Reconstruction Volume the Value 1 and Those Outside 0	*/

	// Allocate memory for the hull image on the host and initialize to zeros
	SC_hull_h = (bool*)calloc( VOXELS, sizeof(bool));

	float x, y;
	// Set the inner cylinder of the hull image to 1s
	for( int slice = 0; slice < SLICES; slice++ )
		for( int row = 0; row < ROWS; row++ )
			for( int column = 0; column < COLUMNS; column++ )
			{
				x = ( column - COLUMNS/2 + 0.5) * VOXEL_WIDTH;
				y = ( ROWS/2 - row - 0.5) * VOXEL_HEIGHT;
				if( ( (x * x) + (y * y) ) < float(RECON_CYL_RADIUS * RECON_CYL_RADIUS) )
					SC_hull_h[slice * COLUMNS * ROWS + row * COLUMNS + column] = true;
			}
	// Allocate memory for the initialized hull image on the GPU and then transfer it to the GPU	
	cudaMalloc((void**) &SC_hull_d,	MEM_SIZE_IMAGE_BOOL);
	cudaMemcpy(SC_hull_d, SC_hull_h, MEM_SIZE_IMAGE_BOOL, cudaMemcpyHostToDevice) ;
}
void initialize_MSC_hull( int*& MSC_hull_h, int*& MSC_hull_d )
{
	/* Allocate Memory and Initialize Images for Hull Detection Algorithms.  Use the Image and	*/
	/* Reconstruction Cylinder Parameters to Determine the Location of the Perimeter of the		*/
	/* Reconstruction Cylinder, Which is Centered on the Origin (Center) of the Image.  Assign	*/
	/* Voxels Inside the Perimeter of the Reconstruction Volume the Value 1 and Those Outside 0	*/
	
	// Allocate memory for the hull image on the host and initialize to zeros
	MSC_hull_h = (int*)calloc( VOXELS, sizeof(int));
	
	float x, y;
	// Set the inner cylinder of the hull image to 1s
	for( int slice = 0; slice < SLICES; slice++ )
		for( int row = 0; row < ROWS; row++ )
			for( int column = 0; column < COLUMNS; column++ )
			{
				x = ( column - COLUMNS/2 + 0.5 ) * VOXEL_WIDTH;
				y = ( ROWS/2 - row - 0.5 ) * VOXEL_HEIGHT;
				if( ( (x * x) + (y * y) ) < float(RECON_CYL_RADIUS * RECON_CYL_RADIUS) )
					MSC_hull_h[slice * COLUMNS * ROWS + row * COLUMNS + column] = 1;
			}
	// Allocate memory for the initialized hull image on the GPU and then transfer it to the GPU
	cudaMalloc((void**) &MSC_hull_d,	MEM_SIZE_IMAGE_INT);
	cudaMemcpy(MSC_hull_d, MSC_hull_h, MEM_SIZE_IMAGE_INT, cudaMemcpyHostToDevice) ;
}
void initialize_SM_hull( int*& SM_hull_h, int*& SM_hull_d )
{
	/* Allocate Memory and Initialize Images for Hull Detection Algorithms.  Use the Image and	*/
	/* Reconstruction Cylinder Parameters to Determine the Location of the Perimeter of the		*/
	/* Reconstruction Cylinder, Which is Centered on the Origin (Center) of the Image.  Assign	*/
	/* Voxels Inside the Perimeter of the Reconstruction Volume the Value 1 and Those Outside 0	*/
	
	// Allocate memory for the hull image on the host and initialize to zeros
	SM_hull_h = (int*)calloc( VOXELS, sizeof(int));
	float x, y;
	// Set the inner cylinder of the hull image to 1s
	for( int slice = 0; slice < SLICES; slice++ )
		for( int row = 0; row < ROWS; row++ )
			for( int column = 0; column < COLUMNS; column++ )
			{
				x = ( column - COLUMNS/2 + 0.5 ) * VOXEL_WIDTH;
				y = ( ROWS/2 - row - 0.5 ) * VOXEL_HEIGHT;
				if( ( (x * x) + (y * y) ) < float(RECON_CYL_RADIUS * RECON_CYL_RADIUS) )
					SM_hull_h[slice * COLUMNS * ROWS + row * COLUMNS + column] = 1;
			}
	// Allocate memory for the initialized hull image on the GPU and then transfer it to the GPU
	cudaMalloc((void**) &SM_hull_d,	MEM_SIZE_IMAGE_INT);
	cudaMemcpy(SM_hull_d, SM_hull_h, MEM_SIZE_IMAGE_INT, cudaMemcpyHostToDevice) ;
}
void initialize_float_image( float*& float_image_h, float*& float_image_d )
{
	//Create space carve object, init to zeros
	float_image_h = (float*)calloc( VOXELS, sizeof(float));

	double x, y;
	// Set inner cylinder to 1s
	for( int slice = 0; slice < SLICES; slice++ )
		for( int row = 0; row < ROWS; row++ )
			for( int column = 0; column < COLUMNS; column++ )
			{
				x = ( column - COLUMNS/2 + 0.5 ) * VOXEL_WIDTH;
				y = ( ROWS/2 - row - 0.5 ) * VOXEL_HEIGHT;
				if( ( (x * x) + (y * y) ) < double(RECON_CYL_RADIUS * RECON_CYL_RADIUS) )
					float_image_h[slice * COLUMNS * ROWS + row * COLUMNS + column] = 1;
			}
	cudaMalloc((void**) &float_image_d,	MEM_SIZE_IMAGE_FLOAT);
	cudaMemcpy(float_image_d, float_image_h, MEM_SIZE_IMAGE_FLOAT, cudaMemcpyHostToDevice) ;
}
/************************************************************************************************************************************************************/
/******************************************************************* Hull Detection *************************************************************************/
/************************************************************************************************************************************************************/
__device__ void voxel_walk( bool*& image, float x_entry, float y_entry, float z_entry, float x_exit, float y_exit, float z_exit )
{
	/********************************************************************************************/
	/********************************* Voxel Walk Parameters ************************************/
	/********************************************************************************************/
	int x_move_direction, y_move_direction, z_move_direction;
	int x_voxel_step, y_voxel_step, z_voxel_step;
	float delta_x, delta_y, delta_z;
	float x_move, y_move, z_move;
	/********************************************************************************************/
	/**************************** Status Tracking Information ***********************************/
	/********************************************************************************************/
	float x, y, z;
	float x_inside, y_inside, z_inside;
	float x_to_go, y_to_go, z_to_go;		
	float x_extension, y_extension;	
	float voxel_x, voxel_y, voxel_z;
	float voxel_x_out, voxel_y_out, voxel_z_out, voxel_out; 
	int voxel;
	bool outside_image, end_walk;
	/********************************************************************************************/
	/************************** Initial and Boundary Conditions *********************************/
	/********************************************************************************************/
	// Initial Distance Into Voxel
	x_inside = modf( ( x_entry + RECON_CYL_RADIUS ) /VOXEL_WIDTH, &voxel_x)*VOXEL_WIDTH;	
	y_inside = modf( ( RECON_CYL_RADIUS - y_entry ) /VOXEL_HEIGHT, &voxel_y)*VOXEL_HEIGHT;
	z_inside = modf( ( RECON_CYL_HEIGHT/2 - z_entry ) /VOXEL_THICKNESS, &voxel_z)*VOXEL_THICKNESS;

	voxel = int(voxel_x + voxel_y * COLUMNS + voxel_z * COLUMNS * ROWS);
	voxel_x_out = int( ( x_exit + RECON_CYL_RADIUS ) /VOXEL_WIDTH );
	voxel_y_out = int( ( RECON_CYL_RADIUS - y_exit ) /VOXEL_HEIGHT );
	voxel_z_out = int( ( RECON_CYL_HEIGHT/2 - z_exit ) /VOXEL_THICKNESS );
	voxel_out = int(voxel_x_out + voxel_y_out * COLUMNS + voxel_z_out * COLUMNS * ROWS);
	/********************************************************************************************/
	/***************************** Path and Walk Information ************************************/
	/********************************************************************************************/
	// Lengths/Distances as x is Incremented One Voxel
	delta_x = VOXEL_WIDTH;
	delta_y = abs( (y_exit - y_entry)/(x_exit - x_entry) * VOXEL_WIDTH );
	delta_z = abs( (z_exit - z_entry)/(x_exit - x_entry) * VOXEL_WIDTH );
	// Overwrite NaN if Divisors on delta_i Calculations Above 
	if( x_entry == x_exit )
	{
		delta_x = abs( (x_exit - x_entry)/(y_exit - y_entry) * VOXEL_HEIGHT );
		delta_y = VOXEL_HEIGHT;
		delta_z = abs( (z_exit - z_entry)/(y_exit - y_entry) * VOXEL_HEIGHT );
		if( y_entry == y_exit )
		{
			delta_x = abs( (x_exit - x_entry)/(z_exit - z_entry) * VOXEL_THICKNESS );
			delta_y = abs( (y_exit - y_entry)/(z_exit - z_entry) * VOXEL_THICKNESS );;
			delta_z = VOXEL_THICKNESS;
		}
	}
	x_move = 0, y_move = 0, z_move = 0;
	x_move_direction = ( x_entry <= x_exit ) - ( x_entry > x_exit );
	y_move_direction = ( y_entry <= y_exit ) - ( y_entry > y_exit );
	z_move_direction = ( z_entry <= z_exit ) - ( z_entry > z_exit );
	x_voxel_step = x_move_direction;
	y_voxel_step = -y_move_direction;
	z_voxel_step = -z_move_direction;
	/********************************************************************************************/
	/**************************** Status Tracking Information ***********************************/
	/********************************************************************************************/
	x = x_entry, y = y_entry, z = z_entry;
	x_to_go = ( x_voxel_step > 0 ) * (VOXEL_WIDTH - x_inside) + ( x_voxel_step <= 0 ) * x_inside;
	y_to_go = ( y_voxel_step > 0 ) * (VOXEL_HEIGHT - y_inside) + ( y_voxel_step <= 0 ) * y_inside;
	z_to_go = ( z_voxel_step > 0 ) * (VOXEL_THICKNESS - z_inside) + ( z_voxel_step <= 0 ) * z_inside;
			
	outside_image = ( voxel_x >= COLUMNS ) || ( voxel_y >= ROWS ) || ( voxel_z >= SLICES );
	if( !outside_image )
		image[voxel] = 0;
	end_walk = ( voxel == voxel_out ) || outside_image;
	//fgets(user_response, sizeof(user_response), stdin);
	/********************************************************************************************/
	/*********************************** Voxel Walk Routine *************************************/
	/********************************************************************************************/
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
				z_to_go = VOXEL_THICKNESS;
				voxel_z += z_voxel_step;
				if( x_to_go == 0 )
				{
					voxel_x += x_voxel_step;
					x_to_go = VOXEL_WIDTH;
				}
				if(	y_to_go == 0 )
				{
					voxel_y += y_voxel_step;
					y_to_go = VOXEL_HEIGHT;
				}
			}
			//If Next Voxel Edge is in x or xy Diagonal
			else if( x_extension <= y_extension )
			{
				//printf(" x_extension <= y_extension \n");
				x_move = x_to_go;
				y_move = delta_y / delta_x * x_to_go;
				z_move = delta_z / delta_x * x_to_go;
				x_to_go = VOXEL_WIDTH;
				y_to_go -= y_move;
				z_to_go -= z_move;
				voxel_x += x_voxel_step;
				if( y_to_go == 0 )
				{
					y_to_go = VOXEL_HEIGHT;
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
				y_to_go = VOXEL_HEIGHT;
				z_to_go -= z_move;
				voxel_y += y_voxel_step;
			}
			x += x_move_direction * x_move;
			y += y_move_direction * y_move;
			z += z_move_direction * z_move;				
			//fgets(user_response, sizeof(user_response), stdin);
			voxel = int(voxel_x + voxel_y * COLUMNS + voxel_z * COLUMNS * ROWS);
			outside_image = ( voxel_x >= COLUMNS ) || ( voxel_y >= ROWS ) || ( voxel_z >= SLICES );
			if( !outside_image )
				image[voxel] = 0;
			end_walk = ( voxel == voxel_out ) || outside_image;
		}
	}
	else
	{
		//printf("z_exit == z_entry\n");
		while( !end_walk )
		{
			// Change in x for Move to Voxel Edge in y
			y_extension = delta_x/delta_y * y_to_go;
			//If Next Voxel Edge is in x or xy Diagonal
			if( x_to_go <= y_extension )
			{
				//printf(" x_to_go <= y_extension \n");
				x_move = x_to_go;
				y_move = delta_y / delta_x * x_to_go;				
				x_to_go = VOXEL_WIDTH;
				y_to_go -= y_move;
				voxel_x += x_voxel_step;
				if( y_to_go == 0 )
				{
					y_to_go = VOXEL_HEIGHT;
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
				y_to_go = VOXEL_HEIGHT;
				voxel_y += y_voxel_step;
			}
			x += x_move_direction * x_move;
			y += y_move_direction * y_move;
			voxel = int(voxel_x + voxel_y * COLUMNS + voxel_z * COLUMNS * ROWS);
			outside_image = ( voxel_x >= COLUMNS ) || ( voxel_y >= ROWS ) || ( voxel_z >= SLICES );
			if( !outside_image )
				image[voxel] = 0;
			end_walk = ( voxel == voxel_out ) || outside_image;
			//fgets(user_response, sizeof(user_response), stdin);
		}// end: while( !end_walk )
	}//end: else: z_entry_h != z_exit_h => z_entry_h == z_exit_h
}
void SC( int num_histories )
{
	dim3 dimBlock(THREADS_PER_BLOCK);
	dim3 dimGrid((int)(num_histories/THREADS_PER_BLOCK)+1);
	SC_kernel<<<dimGrid, dimBlock>>>
	(
		num_histories, SC_image_d, bin_num_d, traversed_recon_volume_d, WEPL_d,
		x_entry_d, y_entry_d, z_entry_d, x_exit_d, y_exit_d, z_exit_d
	);
}
__global__ void SC_kernel
( 
	int num_histories, bool* SC_image, int* bin_num, bool* traversed_recon_volume, float* WEPL,
	float* x_entry, float* y_entry, float* z_entry, float* x_exit, float* y_exit, float* z_exit
)
{
	int i = threadIdx.x + blockIdx.x * THREADS_PER_BLOCK;
	if( (i < num_histories) && traversed_recon_volume[i] && (WEPL[i] <= SC_THRESHOLD) && (bin_num[i] >= 0) )
	{
		voxel_walk( SC_image, x_entry[i], y_entry[i], z_entry[i], x_exit[i], y_exit[i], z_exit[i] );
	}// end: if( (i < num_histories) && traversed_recon_volume[i] && (WEPL[i] <= PURE_SC_THRESH) && (bin_num[i] >= 0) )
}
/************************************************************************************************************************************************************/
void MSC( int num_histories )
{
	dim3 dimBlock(THREADS_PER_BLOCK);
	dim3 dimGrid((int)(num_histories/THREADS_PER_BLOCK)+1);
	MSC_kernel<<<dimGrid, dimBlock>>>
	(
		num_histories, MSC_image_d, bin_num_d, traversed_recon_volume_d, WEPL_d,
		x_entry_d, y_entry_d, z_entry_d, x_exit_d, y_exit_d, z_exit_d
	);
}
__global__ void MSC_kernel
( 
	int num_histories, int* MSC_image, int* bin_num, bool* traversed_recon_volume, float* WEPL,
	float* x_entry, float* y_entry, float* z_entry, float* x_exit, float* y_exit, float* z_exit
)
{
	int i = threadIdx.x + blockIdx.x * THREADS_PER_BLOCK;
	if( (i < num_histories) && traversed_recon_volume[i] && (WEPL[i] < MSC_THRESHOLD) && (bin_num[i] >= 0) )
	{
		//char user_response[20];
		/********************************************************************************************/
		/********************************* Voxel Walk Parameters ************************************/
		/********************************************************************************************/
		int x_move_direction, y_move_direction, z_move_direction;
		int x_voxel_step, y_voxel_step, z_voxel_step;
		float delta_x, delta_y, delta_z;
		float x_move, y_move, z_move;
		/********************************************************************************************/
		/**************************** Status Tracking Information ***********************************/
		/********************************************************************************************/
		float x, y, z;
		float x_inside, y_inside, z_inside;
		float x_to_go, y_to_go, z_to_go;		
		float x_extension, y_extension;	
		float voxel_x, voxel_y, voxel_z;
		float voxel_x_out, voxel_y_out, voxel_z_out, voxel_out; 
		int voxel;
		bool outside_image, end_walk;
		/********************************************************************************************/
		/************************** Initial and Boundary Conditions *********************************/
		/********************************************************************************************/
		// Initial Distance Into Voxel
		x_inside = modf( ( x_entry[i] + RECON_CYL_RADIUS) /VOXEL_WIDTH, &voxel_x)*VOXEL_WIDTH;	
		y_inside = modf( ( RECON_CYL_RADIUS - y_entry[i]) /VOXEL_HEIGHT, &voxel_y)*VOXEL_HEIGHT;
		z_inside = modf( ( RECON_CYL_HEIGHT/2 - z_entry[i]) /VOXEL_THICKNESS, &voxel_z)*VOXEL_THICKNESS;

		voxel = int(voxel_x + voxel_y * COLUMNS + voxel_z * COLUMNS * ROWS);
		voxel_x_out = int( ( x_exit[i] + RECON_CYL_RADIUS ) /VOXEL_WIDTH );
		voxel_y_out = int( ( RECON_CYL_RADIUS - y_exit[i] ) /VOXEL_HEIGHT );
		voxel_z_out = int( ( RECON_CYL_HEIGHT/2 - z_exit[i] ) /VOXEL_THICKNESS );
		voxel_out = int(voxel_x_out + voxel_y_out * COLUMNS + voxel_z_out * COLUMNS * ROWS);
		/********************************************************************************************/
		/***************************** Path and Walk Information ************************************/
		/********************************************************************************************/
		// Lengths/Distances as x is Incremented One Voxel
		delta_x = VOXEL_WIDTH;
		delta_y = abs( (y_exit[i] - y_entry[i])/(x_exit[i] - x_entry[i]) * VOXEL_WIDTH );
		delta_z = abs( (z_exit[i] - z_entry[i])/(x_exit[i] - x_entry[i]) * VOXEL_WIDTH );
		// Overwrite NaN if Divisors on delta_i Calculations Above 
		if( x_entry[i] == x_exit[i] )
		{
			delta_x = abs( (x_exit[i] - x_entry[i])/(y_exit[i] - y_entry[i]) * VOXEL_HEIGHT );
			delta_y = VOXEL_HEIGHT;
			delta_z = abs( (z_exit[i] - z_entry[i])/(y_exit[i] - y_entry[i]) * VOXEL_HEIGHT );
			if( y_entry[i] == y_exit[i] )
			{
				delta_x = abs( (x_exit[i] - x_entry[i])/(z_exit[i] - z_entry[i]) * VOXEL_THICKNESS );
				delta_y = abs( (y_exit[i] - y_entry[i])/(z_exit[i] - z_entry[i]) * VOXEL_THICKNESS );;
				delta_z = VOXEL_THICKNESS;
			}
		}
		x_move = 0, y_move = 0, z_move = 0;
		x_move_direction = ( x_entry[i] <= x_exit[i] ) - ( x_entry[i] > x_exit[i] );
		y_move_direction = ( y_entry[i] <= y_exit[i] ) - ( y_entry[i] > y_exit[i] );
		z_move_direction = ( z_entry[i] <= z_exit[i] ) - ( z_entry[i] > z_exit[i] );
		x_voxel_step = x_move_direction;
		y_voxel_step = -y_move_direction;
		z_voxel_step = -z_move_direction;
		/********************************************************************************************/
		/**************************** Status Tracking Information ***********************************/
		/********************************************************************************************/
		x = x_entry[i], y = y_entry[i], z = z_entry[i];
		x_to_go = ( x_voxel_step > 0 ) * (VOXEL_WIDTH - x_inside) + ( x_voxel_step <= 0 ) * x_inside;
		y_to_go = ( y_voxel_step > 0 ) * (VOXEL_HEIGHT - y_inside) + ( y_voxel_step <= 0 ) * y_inside;
		z_to_go = ( z_voxel_step > 0 ) * (VOXEL_THICKNESS - z_inside) + ( z_voxel_step <= 0 ) * z_inside;
			
		outside_image = ( voxel_x >= COLUMNS ) || ( voxel_y >= ROWS ) || ( voxel_z >= SLICES );
		if( !outside_image )
			atomicAdd( &MSC_image[voxel], 1 );
		end_walk = ( voxel == voxel_out ) || outside_image;
		//fgets(user_response, sizeof(user_response), stdin);
		/********************************************************************************************/
		/*********************************** Voxel Walk Routine *************************************/
		/********************************************************************************************/
		if( z_entry[i] != z_exit[i] )
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
					z_to_go = VOXEL_THICKNESS;
					voxel_z += z_voxel_step;
					if( x_to_go == 0 )
					{
						voxel_x += x_voxel_step;
						x_to_go = VOXEL_WIDTH;
					}
					if(	y_to_go == 0 )
					{
						voxel_y += y_voxel_step;
						y_to_go = VOXEL_HEIGHT;
					}
				}
				//If Next Voxel Edge is in x or xy Diagonal
				else if( x_extension <= y_extension )
				{
					//printf(" x_extension <= y_extension \n");
					x_move = x_to_go;
					y_move = delta_y / delta_x * x_to_go;
					z_move = delta_z / delta_x * x_to_go;
					x_to_go = VOXEL_WIDTH;
					y_to_go -= y_move;
					z_to_go -= z_move;
					voxel_x += x_voxel_step;
					if( y_to_go == 0 )
					{
						y_to_go = VOXEL_HEIGHT;
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
					y_to_go = VOXEL_HEIGHT;
					z_to_go -= z_move;
					voxel_y += y_voxel_step;
				}
				x += x_move_direction * x_move;
				y += y_move_direction * y_move;
				z += z_move_direction * z_move;				
				//fgets(user_response, sizeof(user_response), stdin);
				voxel = int(voxel_x + voxel_y * COLUMNS + voxel_z * COLUMNS * ROWS);
				outside_image = ( voxel_x >= COLUMNS ) || ( voxel_y >= ROWS ) || ( voxel_z >= SLICES );
				if( !outside_image )
					atomicAdd( &MSC_image[voxel], 1 );
				end_walk = ( voxel == voxel_out ) || outside_image;
			}
		}
		else
		{
			//printf("z_exit[i] == z_entry[i]\n");
			while( !end_walk )
			{
				// Change in x for Move to Voxel Edge in y
				y_extension = delta_x/delta_y * y_to_go;
				//If Next Voxel Edge is in x or xy Diagonal
				if( x_to_go <= y_extension )
				{
					//printf(" x_to_go <= y_extension \n");
					x_move = x_to_go;
					y_move = delta_y / delta_x * x_to_go;				
					x_to_go = VOXEL_WIDTH;
					y_to_go -= y_move;
					voxel_x += x_voxel_step;
					if( y_to_go == 0 )
					{
						y_to_go = VOXEL_HEIGHT;
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
					y_to_go = VOXEL_HEIGHT;
					voxel_y += y_voxel_step;
				}
				x += x_move_direction * x_move;
				y += y_move_direction * y_move;
				voxel = int(voxel_x + voxel_y * COLUMNS + voxel_z * COLUMNS * ROWS);
				outside_image = ( voxel_x >= COLUMNS ) || ( voxel_y >= ROWS ) || ( voxel_z >= SLICES );
				if( !outside_image )
					atomicAdd( &MSC_image[voxel], 1 );
				end_walk = ( voxel == voxel_out ) || outside_image;
				//fgets(user_response, sizeof(user_response), stdin);
			}// end: while( !end_walk )
		}//end: else: z_entry[i] != z_exit[i] => z_entry[i] == z_exit[i]
	}// end: if( (i < num_histories) && traversed_recon_volume[i] && (WEPL[i] <= PURE_SC_THRESH) && (bin_num[i] >= 0) )
}
void MSC_threshold()
{
	cudaMemcpy(MSC_image_h,  MSC_image_d,	 MEM_SIZE_IMAGE_INT, cudaMemcpyDeviceToHost);
	write_array_to_disk("MSC_image", output_directory, output_folder, MSC_image_h, COLUMNS, ROWS, SLICES, VOXELS, false );

	dim3 dimBlock( SLICES );
	dim3 dimGrid( COLUMNS, ROWS );   

	MSC_threshold_kernel<<< dimGrid, dimBlock >>>( MSC_image_d );

	cudaMemcpy(MSC_image_h,  MSC_image_d,	 MEM_SIZE_IMAGE_INT, cudaMemcpyDeviceToHost);

	write_array_to_disk("MSC_image_thresholded", output_directory, output_folder, MSC_image_h, COLUMNS, ROWS, SLICES, VOXELS, false );
	write_array_to_disk("x_MSC", output_directory, output_folder, MSC_image_h, COLUMNS, ROWS, SLICES, VOXELS, true );

	cudaFree( MSC_image_d );
	free(MSC_image_h);
}
__global__ void MSC_threshold_kernel( int* MSC_image )
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
				difference = MSC_image[voxel] - MSC_image[current_column + current_row * COLUMNS + slice * COLUMNS * ROWS];
				if( difference > max_difference )
					max_difference = difference;
			}
		}
	}
	syncthreads();
	if( max_difference > MSC_DIFF_THRESH )
		MSC_image[voxel] = 0;
	else if( MSC_image[voxel] == 0 )
		MSC_image[voxel] = 0;
	else
		MSC_image[voxel] = 1;
	if( powf(x, 2) + pow(y, 2) >= powf(RECON_CYL_RADIUS - max(VOXEL_WIDTH, VOXEL_HEIGHT)/2, 2 ) )
		MSC_image[voxel] = 0;

}
/************************************************************************************************************************************************************/
void SM( int num_histories)
{
	dim3 dimBlock(THREADS_PER_BLOCK);
	dim3 dimGrid((int)(num_histories/THREADS_PER_BLOCK)+1);
	SM_kernel<<<dimGrid, dimBlock>>>
	(
		num_histories, SM_image_d, bin_num_d, traversed_recon_volume_d, WEPL_d,
		x_entry_d, y_entry_d, z_entry_d, x_exit_d, y_exit_d, z_exit_d
	);
}
__global__ void SM_kernel
( 
	int num_histories, int* SM_image, int* bin_num, bool* traversed_recon_volume, float* WEPL,
	float* x_entry, float* y_entry, float* z_entry, float* x_exit, float* y_exit, float* z_exit
)
{
	int i = threadIdx.x + blockIdx.x * THREADS_PER_BLOCK;
	//if( (i < num_histories) && traversed_recon_volume[i] && (WEPL[i] >= SM_LOWER_THRESHOLD) && (bin_num[i] >= 0) )
	//{
	//	//char user_response[20];
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
	//	x_inside = modf( ( x_entry[i] + RECON_CYL_RADIUS ) /VOXEL_WIDTH, &voxel_x)*VOXEL_WIDTH;	
	//	y_inside = modf( ( RECON_CYL_RADIUS - y_entry[i] ) /VOXEL_HEIGHT, &voxel_y)*VOXEL_HEIGHT;
	//	z_inside = modf( ( RECON_CYL_HEIGHT/2 - z_entry[i] ) /VOXEL_THICKNESS, &voxel_z)*VOXEL_THICKNESS;

	//	voxel = int(voxel_x + voxel_y * COLUMNS + voxel_z * COLUMNS * ROWS);
	//	voxel_x_out = int( ( x_exit[i] + RECON_CYL_RADIUS ) /VOXEL_WIDTH );
	//	voxel_y_out = int( ( RECON_CYL_RADIUS - y_exit[i] ) /VOXEL_HEIGHT );
	//	voxel_z_out = int( ( RECON_CYL_HEIGHT/2 - z_exit[i] ) /VOXEL_THICKNESS );
	//	voxel_out = int(voxel_x_out + voxel_y_out * COLUMNS + voxel_z_out * COLUMNS * ROWS);
	//	/********************************************************************************************/
	//	/***************************** Path and Walk Information ************************************/
	//	/********************************************************************************************/
	//	// Lengths/Distances as x is Incremented One Voxel
	//	delta_x = VOXEL_WIDTH;
	//	delta_y = abs( (y_exit[i] - y_entry[i])/(x_exit[i] - x_entry[i]) * VOXEL_WIDTH );
	//	delta_z = abs( (z_exit[i] - z_entry[i])/(x_exit[i] - x_entry[i]) * VOXEL_WIDTH );
	//	// Overwrite NaN if Divisors on delta_i Calculations Above 
	//	if( x_entry[i] == x_exit[i] )
	//	{
	//		delta_x = abs( (x_exit[i] - x_entry[i])/(y_exit[i] - y_entry[i]) * VOXEL_HEIGHT );
	//		delta_y = VOXEL_HEIGHT;
	//		delta_z = abs( (z_exit[i] - z_entry[i])/(y_exit[i] - y_entry[i]) * VOXEL_HEIGHT );
	//		if( y_entry[i] == y_exit[i] )
	//		{
	//			delta_x = abs( (x_exit[i] - x_entry[i])/(z_exit[i] - z_entry[i]) * VOXEL_THICKNESS );
	//			delta_y = abs( (y_exit[i] - y_entry[i])/(z_exit[i] - z_entry[i]) * VOXEL_THICKNESS );;
	//			delta_z = VOXEL_THICKNESS;
	//		}
	//	}
	//	x_move = 0, y_move = 0, z_move = 0;
	//	x_move_direction = ( x_entry[i] <= x_exit[i] ) - ( x_entry[i] > x_exit[i] );
	//	y_move_direction = ( y_entry[i] <= y_exit[i] ) - ( y_entry[i] > y_exit[i] );
	//	z_move_direction = ( z_entry[i] <= z_exit[i] ) - ( z_entry[i] > z_exit[i] );
	//	x_voxel_step = x_move_direction;
	//	y_voxel_step = -y_move_direction;
	//	z_voxel_step = -z_move_direction;
	//	/********************************************************************************************/
	//	/**************************** Status Tracking Information ***********************************/
	//	/********************************************************************************************/
	//	x = x_entry[i], y = y_entry[i], z = z_entry[i];
	//	x_to_go = ( x_voxel_step > 0 ) * (VOXEL_WIDTH - x_inside) + ( x_voxel_step <= 0 ) * x_inside;
	//	y_to_go = ( y_voxel_step > 0 ) * (VOXEL_HEIGHT - y_inside) + ( y_voxel_step <= 0 ) * y_inside;
	//	z_to_go = ( z_voxel_step > 0 ) * (VOXEL_THICKNESS - z_inside) + ( z_voxel_step <= 0 ) * z_inside;
	//		
	//	outside_image = ( voxel_x >= COLUMNS ) || ( voxel_y >= ROWS ) || ( voxel_z >= SLICES );
	//	if( !outside_image )
	//		atomicAdd( &SM_image[voxel], 1 );
	//	end_walk = ( voxel == voxel_out ) || outside_image;
	//	//fgets(user_response, sizeof(user_response), stdin);
	//	/********************************************************************************************/
	//	/*********************************** Voxel Walk Routine *************************************/
	//	/********************************************************************************************/
	//	if( z_entry[i] != z_exit[i] )
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
	//				atomicAdd( &SM_image[voxel], 1 );
	//			end_walk = ( voxel == voxel_out ) || outside_image;
	//		}
	//	}
	//	else
	//	{
	//		//printf("z_exit[i] == z_entry[i]\n");
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
	//				atomicAdd( &SM_image[voxel], 1 );
	//			end_walk = ( voxel == voxel_out ) || outside_image;
	//			//fgets(user_response, sizeof(user_response), stdin);
	//		}// end: while( !end_walk )
	//	}//end: else: z_entry[i] != z_exit[i] => z_entry[i] == z_exit[i]
	//}// end: if( (i < num_histories) && traversed_recon_volume[i] && (WEPL[i] >= SPACE_MODEL_LOWER_THRESHOLD) && (WEPL[i] <= SPACE_MODEL_UPPER_THRESHOLD) && (bin_num[i] >= 0) )
}
void SM_threshold()
{
	// Copy the space modeled image from the GPU to the CPU and write it to file.
	cudaMemcpy(SM_image_h,  SM_image_d,	 MEM_SIZE_IMAGE_INT,   cudaMemcpyDeviceToHost);
	write_array_to_disk("SM_image", output_directory, output_folder, SM_image_h, COLUMNS, ROWS, SLICES, VOXELS, false );

	int* SM_differences_h = (int*) calloc( VOXELS, sizeof(int) );
	int* SM_differences_d;
	cudaMalloc((void**) &SM_differences_d, MEM_SIZE_IMAGE_INT );
	cudaMemcpy( SM_differences_d, SM_differences_h, MEM_SIZE_IMAGE_INT, cudaMemcpyHostToDevice );

	dim3 dimBlock( SLICES );
	dim3 dimGrid( COLUMNS, ROWS );   

	carve_differences<<< dimGrid, dimBlock >>>( SM_differences_d, SM_image_d );
	cudaMemcpy( SM_differences_h, SM_differences_d, MEM_SIZE_IMAGE_INT, cudaMemcpyDeviceToHost );

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
				SM_thresholds_h[slice] = SM_image_h[voxel];
			}
		}
		printf( "Slice %d : The maximum space_model difference = %d and the space_model threshold = %d\n", slice, max_difference, SM_thresholds_h[slice] );
		max_difference = 0;
	}

	int* SM_thresholds_d;
	unsigned int threshold_size = SLICES * sizeof(int);
	cudaMalloc((void**) &SM_thresholds_d, threshold_size );
	cudaMemcpy( SM_thresholds_d, SM_thresholds_h, threshold_size, cudaMemcpyHostToDevice );

	SM_threshold_kernel<<< dimGrid, dimBlock >>>( SM_image_d, SM_thresholds_d);
	
	cudaMemcpy(SM_image_h,  SM_image_d,	 MEM_SIZE_IMAGE_INT,   cudaMemcpyDeviceToHost);
	//write_array_to_disk("space_model_thresholded", output_directory, output_folder, SM_image_h, COLUMNS, ROWS, SLICES, VOXELS, false );
	write_array_to_disk("x_SM", output_directory, output_folder, SM_image_h, COLUMNS, ROWS, SLICES, VOXELS, true );

	cudaFree( SM_differences_d );
	cudaFree( SM_thresholds_d );
	cudaFree( SM_image_d );

	free(SM_differences_h);
	free(SM_thresholds_h);
	free(SM_image_h);
}
__global__ void SM_threshold_kernel( int* SM_image, int* SM_threshold )
{
	int row = blockIdx.y, column = blockIdx.x, slice = threadIdx.x;
	float x = ( column - COLUMNS/2 + 0.5 ) * VOXEL_WIDTH;
	float y = ( ROWS/2 - row - 0.5 ) * VOXEL_HEIGHT;
	int voxel = column + row * COLUMNS + slice * COLUMNS * ROWS;
	if( voxel < VOXELS )
	{
		if( SM_image[voxel] > SM_THRESHOLD_MULTIPLIER * SM_threshold[slice] )
			SM_image[voxel] = 1;
		else
			SM_image[voxel] = 0;
		if( powf(x, 2) + pow(y, 2) >= powf(RECON_CYL_RADIUS - max(VOXEL_WIDTH, VOXEL_HEIGHT)/2, 2 ) )
			SM_image[voxel] = 0;
	}
}
void SM_threshold_2()
{
	// Copy the space modeled image from the GPU to the CPU and write it to file.
	cudaMemcpy(SM_image_h,  SM_image_d,	 MEM_SIZE_IMAGE_INT,   cudaMemcpyDeviceToHost);
	write_array_to_disk("SM_image", output_directory, output_folder, SM_image_h, COLUMNS, ROWS, SLICES, VOXELS, false );

	int* SM_differences_h = (int*) calloc( VOXELS, sizeof(int) );
	int* SM_differences_d;
	cudaMalloc((void**) &SM_differences_d, MEM_SIZE_IMAGE_INT );
	cudaMemcpy( SM_differences_d, SM_differences_h, MEM_SIZE_IMAGE_INT, cudaMemcpyHostToDevice );

	dim3 dimBlock( SLICES );
	dim3 dimGrid( COLUMNS, ROWS );   

	carve_differences<<< dimGrid, dimBlock >>>( SM_differences_d, SM_image_d );
	cudaMemcpy( SM_differences_h, SM_differences_d, MEM_SIZE_IMAGE_INT, cudaMemcpyDeviceToHost );

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
				SM_thresholds_h[slice] = SM_image_h[voxel];
			}
		}
		printf( "Slice %d : The maximum space_model difference = %d and the space_model threshold = %d\n", slice, max_difference, SM_thresholds_h[slice] );
		max_difference = 0;
	}

	int* SM_thresholds_d;
	unsigned int threshold_size = SLICES * sizeof(int);
	cudaMalloc((void**) &SM_thresholds_d, threshold_size );
	cudaMemcpy( SM_thresholds_d, SM_thresholds_h, threshold_size, cudaMemcpyHostToDevice );

	SM_threshold_kernel<<< dimGrid, dimBlock >>>( SM_image_d, SM_thresholds_d);
	
	cudaMemcpy(SM_image_h,  SM_image_d,	 MEM_SIZE_IMAGE_INT,   cudaMemcpyDeviceToHost);
	//write_array_to_disk("space_model_thresholded", output_directory, output_folder, SM_image_h, COLUMNS, ROWS, SLICES, VOXELS, false );
	write_array_to_disk("x_SM", output_directory, output_folder, SM_image_h, COLUMNS, ROWS, SLICES, VOXELS, true );

	cudaFree( SM_differences_d );
	cudaFree( SM_thresholds_d );
	cudaFree( SM_image_d );

	free(SM_differences_h);
	free(SM_thresholds_h);
	free(SM_image_h);
}
__global__ void SM_threshold_kernel_2( int* SM_image, int* SM_differences )
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
				difference = SM_image[voxel] - SM_image[current_column + current_row * COLUMNS + slice * COLUMNS * ROWS];
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
			slice_threshold = SM_image[voxel];
		}
	}
	syncthreads();
	float x = ( column - COLUMNS/2 + 0.5 ) * VOXEL_WIDTH;
	float y = ( ROWS/2 - row - 0.5 ) * VOXEL_HEIGHT;
	if( voxel < VOXELS )
	{
		if( SM_image[voxel] > SM_THRESHOLD_MULTIPLIER * slice_threshold )
			SM_image[voxel] = 1;
		else
			SM_image[voxel] = 0;
		if( powf(x, 2) + pow(y, 2) >= powf(RECON_CYL_RADIUS - max(VOXEL_WIDTH, VOXEL_HEIGHT)/2, 2 ) )
			SM_image[voxel] = 0;
	}
}
/************************************************************************************************************************************************************/
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
void averaging_filter( bool*& image_h, bool*& image_d, const int filter_size )
{
	initialize_SC_hull(image_h, image_d);

	float threshold = 0;
	
	dim3 dimBlock( SLICES );
	dim3 dimGrid( COLUMNS, ROWS );   
	averaging_filter_kernel<<< dimGrid, dimBlock >>>( image_d, filter_size, threshold);

	cudaMemcpy(image_h, image_d, MEM_SIZE_IMAGE_INT, cudaMemcpyDeviceToHost) ;
	write_array_to_disk( "test", output_directory, output_folder, image_h, COLUMNS, ROWS, SLICES, VOXELS, true );
}
__global__ void averaging_filter_kernel( bool* image, const int filter_size, const float threshold )
{
	int voxel_x = blockIdx.x;
	int voxel_y = blockIdx.y;	
	int voxel_z = threadIdx.x;
	int voxel = voxel_x + voxel_y * COLUMNS + voxel_z * COLUMNS * ROWS;
	int sum = image[voxel];
	if( (voxel_x > 0) && (voxel_y > 0) && (voxel_x < COLUMNS - 1) && (voxel_y < ROWS - 1) )
	{
		for( int i = voxel_x - filter_size/2; i <= voxel_x + filter_size/2; i++ )
			for( int j = voxel_y - filter_size/2; j <= voxel_y + filter_size/2; j++ )
				sum += image[i + j * COLUMNS + voxel_z * COLUMNS * ROWS];
	}
	//value[voxel] = sum > threshold;
	syncthreads();
	image[voxel] = sum > threshold;
}

/************************************************************************************************************************************************************/
/******************************************************** Memory Transfers, Maintenance, and Cleaning *******************************************************/
/************************************************************************************************************************************************************/
void initial_processing_memory_clean()
{
	free( gantry_angle_h );
	cudaFree( x_entry_d );
	cudaFree( y_entry_d );
	cudaFree( z_entry_d );
	cudaFree( x_exit_d );
	cudaFree( y_exit_d );
	cudaFree( z_exit_d );
	cudaFree( traversed_recon_volume_d );
	cudaFree( bin_num_d );
	cudaFree( WEPL_d);
}
void post_cut_memory_clean()
{
	free(passed_cuts_h );
	free(stddev_rel_ut_angle_h);
	free(stddev_rel_uv_angle_h);
	free(stddev_WEPL_h);

	cudaFree( passed_cuts_d );
	cudaFree( bin_num_d );
	cudaFree( WEPL_d );
	cudaFree( xy_entry_angle_d );
	cudaFree( xz_entry_angle_d );
	//cudaFree( xy_exit_angle_d );
	//cudaFree( xz_exit_angle_d );
	cudaFree( relative_ut_angle_d );
	cudaFree( relative_uv_angle_d );

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
	//xy_exit_angle_vector.resize( new_size );
	//xz_exit_angle_vector.resize( new_size );
	relative_ut_angle_vector.resize( new_size );
	relative_uv_angle_vector.resize( new_size );
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
	//xy_exit_angle_vector.shrink_to_fit();	
	//xz_exit_angle_vector.shrink_to_fit();	
	relative_ut_angle_vector.shrink_to_fit();
	relative_uv_angle_vector.shrink_to_fit();
}
/************************************************************************************************************************************************************/
/****************************************************** Routines for Writing Data Arrays/Vectors to Disk ****************************************************/
/************************************************************************************************************************************************************/
template<typename T> void write_array_to_disk( char* filename_base, const char* directory, const char* folder, T* data, const int x_max, const int y_max, const int z_max, const int elements, const bool single_file )
{
	char filename[256];
	ofstream output_file;
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
				output_file << endl;
			}
			if( index >= elements )
				break;
		}
		z_start += 1;
		z_end += 1;
		output_file.close();
	}
}
template<typename T> void write_vector_to_disk( char* filename_base, const char* directory, const char* folder, vector<T> data, const int x_max, const int y_max, const int z_max, const bool single_file )
{
	char filename[256];
	ofstream output_file;
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
				output_file << endl;
			}
			if( index >= elements )
				break;
		}
		z_start += 1;
		z_end += 1;
		output_file.close();
	}
}
/********************************************************************* Helper Functions *********************************************************************/
/************************************************************************************************************************************************************/
bool bad_data_angle( const int angle )
{
	static const int bad_angles_array[] = {80, 84, 88, 92, 96, 100, 00, 180, 260, 264, 268, 272, 276};
	vector<int> bad_angles(bad_angles_array, bad_angles_array + sizeof(bad_angles_array) / sizeof(bad_angles_array[0]) );
	bool bad_angle = false;
	for( int i = 0; i < bad_angles.size(); i++ )
		if( angle == bad_angles[i] )
			bad_angle = true;
	return bad_angle;
}
int calculate_x_voxel(const float x_position, const int x_voxels, const float voxel_width )
{
	// -10 100 1 [-50 49] -40
	float x_width = x_voxels * voxel_width;//100
	float x_range = x_width/2;//50
	return (  x_position + x_range) / voxel_width;//-10+50/1 = 40
	//[0 99]
}
int calculate_y_voxel(const float y_position, const int y_voxels, const float voxel_height )
{
	// 10 100 1 [-50 49] 40
	float y_width = y_voxels * voxel_height;//100
	float y_range = y_width/2;//50
	return ( y_range - y_position ) / voxel_height;
}
int calculate_slice(const float z_position, const int z_voxels, const float voxel_thickness )
{
	// -10 100 1 [-50 49] -40
	float z_width = z_voxels * voxel_thickness;//100
	float z_range = z_width/2;//50
	return ( z_range - z_position ) / voxel_thickness;
}
/************************************************************************************************************************************************************/
/****************************************************************** Testing Functions ***********************************************************************/
/************************************************************************************************************************************************************/
void test_func()
{
	char user_response[20];
	//fgets(user_response, sizeof(user_response), stdin);
	bool* passed_cuts_h = (bool*)calloc (30, sizeof(bool));
	for( int i = 0; i < 30; i++ )
	{
			bin_num_vector.push_back(i);
			WEPL_vector.push_back(i);
			x_entry_vector.push_back(i);
			y_entry_vector.push_back(i);
			z_entry_vector.push_back(i);
			x_exit_vector.push_back(i);
			y_exit_vector.push_back(i);
			z_exit_vector.push_back(i);
			xy_entry_angle_vector.push_back(i);
			xz_entry_angle_vector.push_back(i);
			xy_exit_angle_vector.push_back(i);
			xz_exit_angle_vector.push_back(i);
			passed_cuts_h[i] = i%2;
	}
	for( int i = 0; i < 30; i++ )
	{
			printf("bin_num_vector[%d] = %d\n", i, bin_num_vector[i]);
			printf("WEPL_vector[%d] = %3f\n", i, WEPL_vector[i]);
			printf("x_entry_vector[%d] = %3f\n", i, x_entry_vector[i]);
			printf("y_entry_vector[%d] = %3f\n", i, y_entry_vector[i]);
			printf("z_entry_vector[%d] = %3f\n", i, z_entry_vector[i]);
			printf("x_exit_vector[%d] = %3f\n", i, x_exit_vector[i]);
			printf("y_exit_vector[%d] = %3f\n", i, y_exit_vector[i]);
			printf("z_exit_vector[%d] = %3f\n", i, z_exit_vector[i]);
			printf("xy_entry_angle_vector[%d] = %3f\n", i, xy_entry_angle_vector[i]);
			printf("xz_entry_angle_vector[%d] = %3f\n", i, xz_entry_angle_vector[i]);
			printf("xy_exit_angle_vector[%d] = %3f\n", i, xy_exit_angle_vector[i]);
			printf("xz_exit_angle_vector[%d] = %3f\n", i, xz_exit_angle_vector[i]);
			printf("passed_cuts_h[%d] = %d\n", i, passed_cuts_h[i]);
			fgets(user_response, sizeof(user_response), stdin);
	}
	int start_position = 0;
	int post_cut_histories = 0;
	for( int iteration = 0; iteration < 6; iteration++ )
	{		
		printf("start iteration %d\n", iteration );
		for( int i = 0; i < 5; i++ )
		{
			if( passed_cuts_h[start_position + i] )
			{
				printf("start i = %d\n", i );
				printf("index = %d\n", start_position + i );
				bin_num_vector[post_cut_histories] = bin_num_vector[start_position + i];
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
				printf("end i = %d\n", i );
			}
		}
		start_position += 5;
		printf("end iteration %d\n", iteration );
	}

	bin_num_vector.resize(post_cut_histories);
	WEPL_vector.resize(post_cut_histories);
	x_entry_vector.resize(post_cut_histories);
	y_entry_vector.resize(post_cut_histories);
	z_entry_vector.resize(post_cut_histories);
	x_exit_vector.resize(post_cut_histories);
	y_exit_vector.resize(post_cut_histories);
	z_exit_vector.resize(post_cut_histories);
	xy_entry_angle_vector.resize(post_cut_histories);
	xz_entry_angle_vector.resize(post_cut_histories);
	xy_exit_angle_vector.resize(post_cut_histories);
	xz_exit_angle_vector.resize(post_cut_histories);
	
	printf("post_cuts\n\n\n");
	printf("post_cut_histories = %d\n\n", post_cut_histories);
	for( int i = 0; i < post_cut_histories; i++ )
	{
			printf("bin_num_vector[%d] = %d\n", i, bin_num_vector[i]);
			printf("WEPL_vector[%d] = %3f\n", i, WEPL_vector[i]);
			printf("x_entry_vector[%d] = %3f\n", i, x_entry_vector[i]);
			printf("y_entry_vector[%d] = %3f\n", i, y_entry_vector[i]);
			printf("z_entry_vector[%d] = %3f\n", i, z_entry_vector[i]);
			printf("x_exit_vector[%d] = %3f\n", i, x_exit_vector[i]);
			printf("y_exit_vector[%d] = %3f\n", i, y_exit_vector[i]);
			printf("z_exit_vector[%d] = %3f\n", i, z_exit_vector[i]);
			printf("xy_entry_angle_vector[%d] = %3f\n", i, xy_entry_angle_vector[i]);
			printf("xz_entry_angle_vector[%d] = %3f\n", i, xz_entry_angle_vector[i]);
			printf("xy_exit_angle_vector[%d] = %3f\n", i, xy_exit_angle_vector[i]);
			printf("xz_exit_angle_vector[%d] = %3f\n", i, xz_exit_angle_vector[i]);
			printf("passed_cuts_h[%d] = %d\n", i, passed_cuts_h[i]);
			fgets(user_response, sizeof(user_response), stdin);
	}
}
__global__ void test_func_kernel( int* test_array, int vec_array_elements )
{
	for(int i = 0; i < vec_array_elements; i++ )
		test_array[i] *= 2;
}