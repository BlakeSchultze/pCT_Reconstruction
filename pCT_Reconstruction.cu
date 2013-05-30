//THIS IS THE MAIN PROGRAM (RWS)

//********************************************************************************************************************************************************//
//************************************************Proton CT Reconstruction Code for Execution on GPU******************************************************//
//*********************************************FBP with FDK Cone Beam Binning Followed by TVS Algorithm***************************************************//
//********************************************************************************************************************************************************//
#include "pCT_Reconstruction.h"

void create_MLP_test_image();
void MLP_test();
void construct_pure_space_carve_object();
void construct_space_carve_object();
void construct_space_model_object();
void write_integer_array_to_files( char*, const char*, int*, int, int, int );
void write_integer_array_to_file( char*, const char*, int*, int, int, int );
void write_float_array_to_files( char*, const char*, float*, int, int, int );
void read_SSD_positions();
void initializations();
void count_histories();
void iterative_data_read( int, int, int );
void recon_volume_intersections( int );
void bin_valid_histories( int );
void pure_space_carve( int );
void space_carve( int );
void space_model( int );
void space_carve_threshold();
void space_model_threshold();
void vector_to_array_transfer();
void calculate_means();
void sum_differences( int, int );
void calculate_std_devs();
void statistical_cuts( int, int );
void construct_sinogram();
void filter();
void backprojection();

__global__ void recon_volume_intersections_kernel( int, int*, bool*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, 
	float*, float*, float*, float*, float*, float*, float*, float*, float*);
__global__ void bin_valid_histories_kernel( int, int*, int*, bool*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float* );
__global__ void pure_space_carve_kernel( int, int*, int*, bool*, float*, float*, float*, float*, float*, float*, float* );
__global__ void space_carve_kernel( int, int*, int*, bool*, float*, float*, float*, float*, float*, float*, float* );
__global__ void space_model_kernel( int, int*, int*, bool*, float*, float*, float*, float*, float*, float*, float* );
__global__ void pure_space_carve_threshold_kernel( int* );
__global__ void space_carve_threshold_kernel( int* );
__global__ void space_model_threshold_kernel( int*, int* );
__global__ void carve_differences( int*, int* );
__global__ void calculate_means_kernel( int*, float*, float*, float* );
__global__ void sum_differences_kernel( int, int*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*  );
__global__ void calculate_std_devs_kernel( int*, float*, float*, float* );
__global__ void statistical_cuts_kernel( int, int*, int*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float* );
__global__ void construct_sinogram_kernel( int*, float* );
__global__ void filter_kernel( float*, float* );

/********************************************************************************************/
/*************************************** Program Main ***************************************/
/********************************************************************************************/
int main(int argc, char** argv)
{
	char text[20];
	/*something = 100;
	printf("something_else = %d\n", something_else );
	something += 26;
	printf("something_else = %d\n", something_else );
	printf("IMAGE_WIDTH =%3f, VOXELS_X = %d\n", IMAGE_WIDTH , VOXELS_X);
	printf("IMAGE_WIDTH / VOXELS_X = %3f\n", IMAGE_WIDTH / VOXELS_X);
	printf("VOXEL_WIDTH =%3f, VOXEL_HEIGHT = %3f\n", VOXEL_WIDTH , VOXEL_HEIGHT);

	cout << IMAGE_WIDTH << endl<< VOXELS_X << endl<< IMAGE_WIDTH / VOXELS_X << endl << IMAGE_WIDTH * VOXELS_X << endl;
	//IMAGE_WIDTH / VOXELS_X
	printf("min(VOXEL_WIDTH, VOXEL_HEIGHT) / 2 = %3f\n", min(VOXEL_WIDTH, VOXEL_HEIGHT) / 2 );
	printf("MLP_u_step = %3f\n", MLP_u_step );*/
	/********************************************************************************************/
	/* Allocate MLP Test Image and Perform MLP Test Procedure									*/
	/********************************************************************************************/
	create_MLP_test_image();
	write_integer_array_to_file("MLP_test_image_in", output_dir, MLP_test_image_h, MLP_IMAGE_VOXELS_X, MLP_IMAGE_VOXELS_Y, MLP_IMAGE_VOXELS_Z );
	MLP_test();
	//write_integer_array_to_file("MLP_test_image_out", output_dir, MLP_test_image_h, MLP_IMAGE_VOXELS_X, MLP_IMAGE_VOXELS_Y, MLP_IMAGE_VOXELS_Z );

	/********************************************************************************************/
	/* Allocate Memory and Initialize Images for Hull Detection Algorithms.  Use the Image and	*/
	/* Reconstruction Cylinder Parameters to Determine the Location of the Perimeter of the		*/
	/* Reconstruction Cylinder, Which is Centered on the Origin (Center) of the Image.  Assign	*/
	/* Voxels Inside the Perimeter of the Reconstruction Volume the Value 1 and Those Outside 0	*/
	/********************************************************************************************/
	//construct_pure_space_carve_object();
	//construct_space_carve_object(); // allocate and initialize hull object to be space carved
	//construct_space_model_object(); // allocate and initialize hull object to be space modeled

	/********************************************************************************************/
	/* Read the u-Coordinates of the Detector Planes from the Config File, Allocate and			*/
	/* Initialize Data Arrays, and Count the Number of Histories Per File, Projection, Gantry	*/
	/* Angle, Translation, and Total.  Request Input from User to Continue.						*/
	/********************************************************************************************/
	//read_SSD_positions(); // Read the detector plane u-coordinates from config file
	//initializations(); // allocate and initialize host and GPU memory for binning
	//count_histories(); // count the number of histories per file, per translation , total, etc.
 //	fgets(text, sizeof text, stdin);

	///********************************************************************************************/
	///* Iteratively Read and Process Data One Chunk at a Time. There are at Most					*/
	///* MAX_GPU_HISTORIES Per Chunk (i.e. Iteration). On Each Iteration:							*/
	///*	(1) Read Data from File																	*/
	///*	(2) Determine Which Histories Traverse the Reconstruction Volume and Store this			*/
	///*		Information in a Boolean Array														*/
	///*	(3) Determine Which Bin Each History Belongs to											*/
	///*	(4) Use the Boolean Array to Determine Which Histories to Keep and then Push			*/
	///*		the Intermediate Data from these Histories onto the Temporary Storage Vectors		*/
	///*	(5) Perform Space Carving, Modified Space Carving, and/or Space Modeling				*/
	///*	(6) Free Up Temporary Host/GPU Array Memory Allocated During Iteration					*/
	///********************************************************************************************/
	//printf("\nDetermine and Write the Bin Number, WEPL Value, Relative UT Angle, and Relative UV Angle of Histories that Intersect " 
	//	"the Reconstruction Volume and Belong to a Valid Bin to Host Arrays and Use These Valid Histories to Space Carve\n\n");
	//int start_file_num = 0, end_file_num = 0, num_histories = 0;
	//while( start_file_num != NUM_FILES )
	//{
	//	while( end_file_num < NUM_FILES )
	//	{
	//		if( num_histories + histories_per_file[end_file_num] < MAX_GPU_HISTORIES )
	//			num_histories += histories_per_file[end_file_num];
	//		else
	//			break;
	//		end_file_num++;
	//	}
	//	iterative_data_read( num_histories, start_file_num, end_file_num - 1 );
	//	printf("num_histories = %d \n", num_histories);
	//	recon_volume_intersections( num_histories );
	//	bin_valid_histories( num_histories );		
	//	//space_carve( num_histories );
	//	//space_model( num_histories );
	//	//if(  //true
	//	//	gantry_angle_h[0] != 60 &&
	//	//	gantry_angle_h[0] != 64 &&
	//	//	gantry_angle_h[0] != 68 &&
	//	//	gantry_angle_h[0] != 72 && 
	//	//	gantry_angle_h[0] != 76 && 
	//	//	gantry_angle_h[0] != 80 && 
	//	//	gantry_angle_h[0] != 84 && 
	//	//	gantry_angle_h[0] != 88 && 
	//	//	gantry_angle_h[0] != 92 &&
	//	//	gantry_angle_h[0] != 96 && 
	//	//	gantry_angle_h[0] != 100 && 
	//	//	gantry_angle_h[0] != 104 && 
	//	//	gantry_angle_h[0] != 00 && 
	//	//	gantry_angle_h[0] != 180 && 
	//	//	gantry_angle_h[0] != 260 && 
	//	//	gantry_angle_h[0] != 264 && 
	//	//	gantry_angle_h[0] != 268 && 
	//	//	gantry_angle_h[0] != 272 && 
	//	//	gantry_angle_h[0] != 276 &&  
	//	//	gantry_angle_h[0] != 280  
	//	//) 
	//	//{
	//		//pure_space_carve( num_histories );
	//		//space_carve_2( num_histories );
	//		//if( gantry_angle_h[0] == 132 )
	//			//space_carve_cpu( num_histories );
	//		//else
	//			//space_carve_cpu( num_histories );
	//		free( gantry_angle_h );
	//		cudaFree( x_entry_d );
	//		cudaFree( y_entry_d );
	//		cudaFree( z_entry_d );
	//		cudaFree( x_exit_d );
	//		cudaFree( y_exit_d );
	//		cudaFree( z_exit_d );
	//		cudaFree( traversed_recon_volume_d );
	//		cudaFree( bin_num_d );
	//		cudaFree( WEPL_d);
	//	//}
	//	start_file_num = end_file_num;
	//	num_histories = 0;
	//}	

	///********************************************************************************************/
	///* Allocate Memory for valid_... Arrays Using the Length of the Vectors then Transfer the   */
	///* Data From the Vectors into the Arrays													*/
	///********************************************************************************************/
	//vector_to_array_transfer();

	///********************************************************************************************/
	///* Apply Thresholds to Space Carve and Space Model Images									*/													
	///********************************************************************************************/
	////space_carve_threshold();
	////space_model_threshold();
	//
	///********************************************************************************************/
	///* Transfer Hull Detection Images from GPU and Write to File								*/													
	///********************************************************************************************/
	////cudaMemcpy(pure_space_carve_object_h,  pure_space_carve_object_d, IMAGE_INT_MEM_SIZE, cudaMemcpyDeviceToHost);
	////write_integer_array_to_file("x_pure_sc", output_dir, pure_space_carve_object_h, VOXELS_X, VOXELS_Y, VOXELS_Z );
	////write_integer_array_to_files("sc", output_dir, pure_space_carve_object_h, VOXELS_X, VOXELS_Y, VOXELS_Z );
	////printf("\nCalculating the Mean for Each Bin Before Cuts\n");

	///********************************************************************************************/
	///* Calculate the Mean WEPL, Relative ut-Angle, and Relative uv-Angle for Each Bin and Count */
	///* the Number of Histories in Each Bin														*/													
	///********************************************************************************************/
	//calculate_means();

	/********************************************************************************************/
	/* Calculate the Standard Deviation in WEPL, Relative ut-Angle, and Relative uv-Angle for	*/
	/* Each Bin.  Iterate Through the Valid History Arrays One Chunk at a Time, With at Most	*/
	/* MAX_GPU_HISTORIES Per Chunk, and Calculate the Difference Between the Mean WEPL and WEPL,*/
	/* Mean Relative ut-Angle and Relative ut-Angle, and Mean Relative uv-Angle and	Relative	*/
	/* uv-Angle for Each History. The Standard Deviation is then Found By Calculating the Sum	*/
	/* of these Differences for Each Bin and Dividing it by the Number of Histories in the Bin 	*/
	/********************************************************************************************/
	//printf("\nSumming up the Difference Between Histories and the Mean for Each Bin\n");
	//int remaining_histories = recon_vol_histories;
	//int start_position = 0;
	//while( remaining_histories > 0 )
	//{
	//	if( remaining_histories > MAX_GPU_HISTORIES )
	//		sum_differences( start_position, MAX_GPU_HISTORIES );
	//	else
	//		sum_differences( start_position, remaining_histories );
	//	remaining_histories -= MAX_GPU_HISTORIES;
	//	start_position += MAX_GPU_HISTORIES;
	//}
	//printf("\nCalculating Standard Deviations for Each Bin\n");
	//calculate_std_devs();

	/********************************************************************************************/
	/* Allocate Memory for the Sinogram on the Host, Initialize it to Zeros, Allocate Memory on	*/
	/* theGPU for the Sinogram, then Transfer the Initialized Sinogram to the GPU				*/
	/********************************************************************************************/
	//sinogram_h = (float*) calloc( NUM_BINS, sizeof(float) );
	//cudaMalloc((void**) &sinogram_d, mem_size_bins_floats );
	//cudaMemcpy( sinogram_d,	sinogram_h,	mem_size_bins_floats, cudaMemcpyHostToDevice );
	
	/********************************************************************************************/
	/* Iterate Through the Valid History Arrays One Chunk at a Time, With at Most				*/
	/* MAX_GPU_HISTORIES Per Chunk, and Perform Statistical Cuts 								*/
	/********************************************************************************************/
	//remaining_histories = recon_vol_histories;
	//start_position = 0;
	//printf("\nMaking Statistical Cuts\n");
	//while( remaining_histories > 0 )
	//{
	//	if( remaining_histories > MAX_GPU_HISTORIES )
	//		statistical_cuts( start_position, MAX_GPU_HISTORIES );
	//	else
	//		statistical_cuts( start_position, remaining_histories );
	//	remaining_histories -= MAX_GPU_HISTORIES;
	//	start_position += MAX_GPU_HISTORIES;
	//}

	/********************************************************************************************/
	/* Free the Host Memory for the Bin Number Array and GPU Memory for the Statistics Arrays	*/
	/********************************************************************************************/
	//free(valid_bin_num);
	//cudaFree( mean_rel_ut_angle_d );
	//cudaFree( mean_rel_uv_angle_d );
	//cudaFree( mean_WEPL_d );
	//cudaFree( stddev_rel_ut_angle_d );
	//cudaFree( stddev_rel_uv_angle_d );
	//cudaFree( stddev_WEPL_d );

	/********************************************************************************************/
	/* Recalculate the Mean WEPL for Each Bin Using	the Histories Remaining After Cuts and Use	*/
	/* these to Produce the Sinogram for FBP													*/
	/********************************************************************************************/
	//printf("\nConstruct the Sinogram\n\n");
	//construct_sinogram();

	/********************************************************************************************/
	/* Perform Filtered Backprojection															*/
	/********************************************************************************************/
	//filter();
	//backprojection();
	
	/********************************************************************************************/
	/* Program Has Finished Execution. Require the User to Hit the Enter Key to Terminate the	*/
	/* Program and Close the Terminal/Console Window											*/ 															
	/********************************************************************************************/
	printf("\nHit enter to close the console window\n");
	fgets(text, sizeof text, stdin);
}
void create_MLP_test_image()
{
	float x, y;
	//Create space carve object, init to zeros
	MLP_test_image_h = (int*)calloc( MLP_IMAGE_VOXELS, sizeof(int));

	//// Set inner cylinder to 1s
	//for( int slice = 0; slice < MLP_IMAGE_VOXELS_Z; slice++ )
	//{
	//	for( int row = 0; row < MLP_IMAGE_VOXELS_Y; row++ )
	//	{
	//		for( int column = 0; column < MLP_IMAGE_VOXELS_X; column++ )
	//		{
	//			x = ( column - MLP_IMAGE_VOXELS_X/2 );
	//			y = ( MLP_IMAGE_VOXELS_Y/2 - row );
	//			if( x < 0 )
	//			{
	//				if( y < 0 )
	//				{
	//					if( pow(x+0.5, 2) + pow(y+0.5, 2) <= pow((float)MLP_IMAGE_RECON_CYL_RADIUS_VOXELS, 2) )
	//						MLP_test_image_h[slice * MLP_IMAGE_VOXELS_X * MLP_IMAGE_VOXELS_Y + row * MLP_IMAGE_VOXELS_X + column] = 1;
	//					if( pow((x+0.5)/MLP_PHANTOM_A_VOXELS, 2) + pow((y+0.5)/MLP_PHANTOM_B_VOXELS, 2) <= 1 )
	//						MLP_test_image_h[slice * MLP_IMAGE_VOXELS_X * MLP_IMAGE_VOXELS_Y + row * MLP_IMAGE_VOXELS_X + column] = 8;
	//				}
	//				else
	//				{
	//					if( pow(x+0.5, 2) + pow(y-0.5, 2) <= pow((float)MLP_IMAGE_RECON_CYL_RADIUS_VOXELS, 2) )
	//						MLP_test_image_h[slice * MLP_IMAGE_VOXELS_X * MLP_IMAGE_VOXELS_Y + row * MLP_IMAGE_VOXELS_X + column] = 1;
	//					if( pow((x+0.5)/MLP_PHANTOM_A_VOXELS, 2) + pow((y-0.5)/MLP_PHANTOM_B_VOXELS, 2) <= 1 )
	//						MLP_test_image_h[slice * MLP_IMAGE_VOXELS_X * MLP_IMAGE_VOXELS_Y + row * MLP_IMAGE_VOXELS_X + column] = 8;
	//				}
	//			}
	//			else
	//			{
	//				if( y < 0 )
	//				{
	//					if( pow(x-0.5, 2) + pow(y+0.5, 2) <= pow((float)MLP_IMAGE_RECON_CYL_RADIUS_VOXELS, 2) )
	//						MLP_test_image_h[slice * MLP_IMAGE_VOXELS_X * MLP_IMAGE_VOXELS_Y + row * MLP_IMAGE_VOXELS_X + column] = 1;
	//					if( pow((x-0.5)/MLP_PHANTOM_A_VOXELS, 2) + pow((y+0.5)/MLP_PHANTOM_B_VOXELS, 2) <= 1 )
	//						MLP_test_image_h[slice * MLP_IMAGE_VOXELS_X * MLP_IMAGE_VOXELS_Y + row * MLP_IMAGE_VOXELS_X + column] = 8;
	//				}
	//				else
	//				{
	//					if( pow(x-0.5, 2) + pow(y-0.5, 2) <= pow((float)MLP_IMAGE_RECON_CYL_RADIUS_VOXELS, 2) )
	//						MLP_test_image_h[slice * MLP_IMAGE_VOXELS_X * MLP_IMAGE_VOXELS_Y + row * MLP_IMAGE_VOXELS_X + column] = 1;
	//					if( pow((x-0.5)/MLP_PHANTOM_A_VOXELS, 2) + pow((y-0.5)/MLP_PHANTOM_B_VOXELS, 2) <= 1 )
	//						MLP_test_image_h[slice * MLP_IMAGE_VOXELS_X * MLP_IMAGE_VOXELS_Y + row * MLP_IMAGE_VOXELS_X + column] = 8;
	//				}
	//			}
	//			//if( pow(x/MLP_phantom_a, 2) + pow(y/MLP_phantom_b, 2) <= 1 )
	//			//if( pow((x+0.5)/MLP_phantom_a, 2) + pow((y+0.5)/MLP_phantom_b, 2) <= 1 )
	//				//MLP_test_image_h[slice * MLP_image_voxels_x * MLP_image_voxels_y + row * MLP_image_voxels_x + column] = 8;
	//		}
	//	}
	//}
	//cudaMalloc((void**) &MLP_test_image_d,	MLP_image_size);
	//cudaMemcpy(MLP_test_image_d, MLP_test_image_h, MLP_image_size, cudaMemcpyHostToDevice) ;


	//// Set inner cylinder to 1s
	//for( int slice = 0; slice < MLP_IMAGE_VOXELS_Z; slice++ )
	//{
	//	for( int row = 0; row < MLP_IMAGE_VOXELS_Y; row++ )
	//	{
	//		for( int column = 0; column < MLP_IMAGE_VOXELS_X; column++ )
	//		{
	//			if( x < 0 )
	//			{
	//				x = ( column - MLP_IMAGE_VOXELS_X/2 + 0.5) * MLP_IMAGE_VOXEL_WIDTH;
	//				if( y < 0 )
	//				{
	//					y = ( MLP_IMAGE_VOXELS_Y/2 - row + 0.5 ) * MLP_IMAGE_VOXEL_HEIGHT;
	//					if( pow( x, 2 ) + pow( y, 2 ) <= pow( float(MLP_IMAGE_RECON_CYL_RADIUS), 2) )
	//						MLP_test_image_h[slice * MLP_IMAGE_VOXELS_X * MLP_IMAGE_VOXELS_Y + row * MLP_IMAGE_VOXELS_X + column] = 1;
	//					if( pow( x / MLP_PHANTOM_A, 2 ) + pow( y / MLP_PHANTOM_B, 2 ) <= 1 )
	//						MLP_test_image_h[slice * MLP_IMAGE_VOXELS_X * MLP_IMAGE_VOXELS_Y + row * MLP_IMAGE_VOXELS_X + column] = 8;
	//				}
	//				else
	//				{
	//					y = ( MLP_IMAGE_VOXELS_Y/2 - row - 0.5 ) * MLP_IMAGE_VOXEL_HEIGHT;
	//					if( pow( x, 2 ) + pow( y, 2 ) <= pow( float(MLP_IMAGE_RECON_CYL_RADIUS), 2) )
	//						MLP_test_image_h[slice * MLP_IMAGE_VOXELS_X * MLP_IMAGE_VOXELS_Y + row * MLP_IMAGE_VOXELS_X + column] = 1;
	//					if( pow( x / MLP_PHANTOM_A, 2 ) + pow( y / MLP_PHANTOM_B, 2 ) <= 1 )
	//						MLP_test_image_h[slice * MLP_IMAGE_VOXELS_X * MLP_IMAGE_VOXELS_Y + row * MLP_IMAGE_VOXELS_X + column] = 8;
	//				}
	//			}
	//			else
	//			{
	//				x = ( column - MLP_IMAGE_VOXELS_X/2 - 0.5) * MLP_IMAGE_VOXEL_WIDTH;
	//				if( y < 0 )
	//				{
	//					y = ( MLP_IMAGE_VOXELS_Y/2 - row + 0.5 ) * MLP_IMAGE_VOXEL_HEIGHT;
	//					if( pow( x, 2 ) + pow( y, 2 ) <= pow( float(MLP_IMAGE_RECON_CYL_RADIUS), 2) )
	//						MLP_test_image_h[slice * MLP_IMAGE_VOXELS_X * MLP_IMAGE_VOXELS_Y + row * MLP_IMAGE_VOXELS_X + column] = 1;
	//					if( pow( x / MLP_PHANTOM_A, 2 ) + pow( y / MLP_PHANTOM_B, 2 ) <= 1 )
	//						MLP_test_image_h[slice * MLP_IMAGE_VOXELS_X * MLP_IMAGE_VOXELS_Y + row * MLP_IMAGE_VOXELS_X + column] = 8;
	//				}
	//				else
	//				{
	//					y = ( MLP_IMAGE_VOXELS_Y/2 - row - 0.5 ) * MLP_IMAGE_VOXEL_HEIGHT;
	//					if( pow( x, 2 ) + pow( y, 2 ) <= pow( float(MLP_IMAGE_RECON_CYL_RADIUS), 2) )
	//						MLP_test_image_h[slice * MLP_IMAGE_VOXELS_X * MLP_IMAGE_VOXELS_Y + row * MLP_IMAGE_VOXELS_X + column] = 1;
	//					if( pow( x / MLP_PHANTOM_A, 2 ) + pow( y / MLP_PHANTOM_B, 2 ) <= 1 )
	//						MLP_test_image_h[slice * MLP_IMAGE_VOXELS_X * MLP_IMAGE_VOXELS_Y + row * MLP_IMAGE_VOXELS_X + column] = 8;
	//				}
	//			}
	//			//if( pow(x/MLP_phantom_a, 2) + pow(y/MLP_phantom_b, 2) <= 1 )
	//			//if( pow((x+0.5)/MLP_phantom_a, 2) + pow((y+0.5)/MLP_phantom_b, 2) <= 1 )
	//				//MLP_test_image_h[slice * MLP_image_voxels_x * MLP_image_voxels_y + row * MLP_image_voxels_x + column] = 8;
	//		}
	//	}
	//}
	// Set inner cylinder to 1s
	for( int slice = 0; slice < MLP_IMAGE_VOXELS_Z; slice++ )
	{
		for( int row = 0; row < MLP_IMAGE_VOXELS_Y; row++ )
		{
			for( int column = 0; column < MLP_IMAGE_VOXELS_X; column++ )
			{
				x = ( column - MLP_IMAGE_VOXELS_X/2 + 0.5) * MLP_IMAGE_VOXEL_WIDTH;
				y = ( MLP_IMAGE_VOXELS_Y/2 - row - 0.5 ) * MLP_IMAGE_VOXEL_HEIGHT;
				if( pow( x, 2 ) + pow( y, 2 ) <= pow( float(MLP_IMAGE_RECON_CYL_RADIUS), 2) )
					MLP_test_image_h[slice * MLP_IMAGE_VOXELS_X * MLP_IMAGE_VOXELS_Y + row * MLP_IMAGE_VOXELS_X + column] = 1;
				if( pow( x / MLP_PHANTOM_A, 2 ) + pow( y / MLP_PHANTOM_B, 2 ) <= 1 )
					MLP_test_image_h[slice * MLP_IMAGE_VOXELS_X * MLP_IMAGE_VOXELS_Y + row * MLP_IMAGE_VOXELS_X + column] = 8;
			}
		}
	}
}
void MLP_test()
{
	char text[20];
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
	
	voxel = int(voxel_x + voxel_y * MLP_IMAGE_VOXELS_X + voxel_z * MLP_IMAGE_VOXELS_X * MLP_IMAGE_VOXELS_Y);
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
	
	outside_image = (voxel_x >= MLP_IMAGE_VOXELS_X ) || (voxel_y >= MLP_IMAGE_VOXELS_Y ) || (voxel_z >= MLP_IMAGE_VOXELS_Z );
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
			voxel = int(voxel_x + voxel_y * MLP_IMAGE_VOXELS_X + voxel_z * MLP_IMAGE_VOXELS_X * MLP_IMAGE_VOXELS_Y);
			outside_image = (voxel_x >= MLP_IMAGE_VOXELS_X ) || (voxel_y >= MLP_IMAGE_VOXELS_Y ) || (voxel_z >= MLP_IMAGE_VOXELS_Z );
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
			voxel = int(voxel_x + voxel_y * MLP_IMAGE_VOXELS_X + voxel_z * MLP_IMAGE_VOXELS_X * MLP_IMAGE_VOXELS_Y);
			//printf("end of loop\n\n");
			//printf("x_move = %3f y_move = %3f\n", x_move, y_move );
			//printf("x = %3f y = %3f z = %3f\n", x, y, z );
			//printf("x_to_go = %3f y_to_go = %3f\n", x_to_go, y_to_go);
			//printf("voxel_x = %3f voxel_y = %3f voxel_z = %3f\n\n", voxel_x, voxel_y, voxel_z);		
			outside_image = (voxel_x >= MLP_IMAGE_VOXELS_X ) || (voxel_y >= MLP_IMAGE_VOXELS_Y ) || (voxel_z >= MLP_IMAGE_VOXELS_Z );
			if( !outside_image )
			{
				entered_object = MLP_test_image_h[voxel] == 8;
				MLP_test_image_h[voxel] = 4;
			}
			//printf("MLP_IMAGE_WIDTH/2 = %3f\n MLP_IMAGE_HEIGHT/2 = %3f", MLP_IMAGE_WIDTH/2 , MLP_IMAGE_HEIGHT/2 );
			x += x_move_direction * x_move;
			y += y_move_direction * y_move;
			end_walk = entered_object || outside_image;
			//fgets(text, sizeof text, stdin);
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
	
	voxel = int(voxel_x + voxel_y * MLP_IMAGE_VOXELS_X + voxel_z * MLP_IMAGE_VOXELS_X * MLP_IMAGE_VOXELS_Y);
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
	
	outside_image = (voxel_x >= MLP_IMAGE_VOXELS_X ) || (voxel_y >= MLP_IMAGE_VOXELS_Y ) || (voxel_z >= MLP_IMAGE_VOXELS_Z );
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
			voxel = int( voxel_x + voxel_y * MLP_IMAGE_VOXELS_X + voxel_z * MLP_IMAGE_VOXELS_X * MLP_IMAGE_VOXELS_Y );
			outside_image = ( voxel_x >= MLP_IMAGE_VOXELS_X ) || ( voxel_y >= MLP_IMAGE_VOXELS_Y ) || ( voxel_z >= MLP_IMAGE_VOXELS_Z );
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
			voxel = int(voxel_x + voxel_y * MLP_IMAGE_VOXELS_X + voxel_z * MLP_IMAGE_VOXELS_X * MLP_IMAGE_VOXELS_Y);
			/*printf("end of loop\n\n");
			printf("x_move = %3f y_move = %3f\n", x_move, y_move );
			printf("x = %3f y = %3f z = %3f\n", x, y, z );
			printf("x_to_go = %3f y_to_go = %3f\n", x_to_go, y_to_go);
			printf("voxel_x = %3f voxel_y = %3f voxel_z = %3f\n\n", voxel_x, voxel_y, voxel_z);*/		
			outside_image = (voxel_x >= MLP_IMAGE_VOXELS_X ) || (voxel_y >= MLP_IMAGE_VOXELS_Y ) || (voxel_z >= MLP_IMAGE_VOXELS_Z );
			if( !outside_image )
			{
				exited_object = MLP_test_image_h[voxel] == 8;
				MLP_test_image_h[voxel] = 4;
			}
			//printf("MLP_IMAGE_WIDTH/2 = %3f\n MLP_IMAGE_HEIGHT/2 = %3f",MLP_IMAGE_WIDTH/2 , MLP_IMAGE_HEIGHT/2 );
			x += x_move_direction * x_move;
			y += y_move_direction * y_move;
			end_walk = exited_object || outside_image;
			//fgets(text, sizeof text, stdin);
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
	voxel = int(voxel_x + voxel_y * MLP_IMAGE_VOXELS_X + voxel_z * MLP_IMAGE_VOXELS_X * MLP_IMAGE_VOXELS_Y);

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
	//fgets(text, sizeof text, stdin);
	while( u_1 <= u_2 - MLP_u_step )
	{
		double R_0[4] = { 1.0, u_1 - u_0, 0.0 , 1.0}; //a,b,c,d
		double R_0T[4] = { 1.0, 0.0, u_1 - u_0 , 1.0}; //a,c,b,d
		double R_1[4] = { 1.0, u_2 - u_1, 0.0 , 1.0}; //a,b,c,d
		double R_1T[4] = { 1.0, 0.0, u_2 - u_1 , 1.0};  //a,c,b,d
	
		double sigma_1_coefficient = pow( E_0 * ( 1 + 0.038 * log( (u_1 - u_0)/X_0) ), 2.0 ) / X_0;
		float sigma_t1 = (a_0/3)*pow(u_1, 3.0) + (a_1/12)*pow(u_1, 4.0) + (a_2/30)*pow(u_1, 5.0) + (a_3/60)*pow(u_1, 6.0) + (a_4/105)*pow(u_1, 7.0) + (a_5/168)*pow(u_1, 8.0);
		float sigma_t1_theta1 = pow(u_1, 2.0 )*( (a_0/2) + (a_1/6)*u_1 + (a_2/12)*pow(u_1, 2.0) + (a_3/20)*pow(u_1, 3.0) + (a_4/30)*pow(u_1, 4.0) + (a_5/42)*pow(u_1, 5.0) );
		float sigma_theta1 = a_0*u_1 + (a_1/2)*pow(u_1, 2.0) + (a_2/3)*pow(u_1, 3.0) + (a_3/4)*pow(u_1, 4.0) + (a_4/5)*pow(u_1, 5.0) + (a_5/6)*pow(u_1, 6.0);	
		double determinant_Sigma_1 = sigma_t1 * sigma_theta1 - pow( sigma_t1_theta1, 2 );//ad-bc
		double Sigma_1I[4] = // Sigma_1 Inverse = [1/det(Sigma_1)]*{ d, -b, -c, a }
		{
			sigma_theta1 / determinant_Sigma_1, 
			-sigma_t1_theta1 / determinant_Sigma_1, 
			-sigma_t1_theta1 / determinant_Sigma_1, 
			sigma_t1 / determinant_Sigma_1 
		};
		double sigma_2_coefficient = pow( E_0 * ( 1 + 0.038 * log( (u_2 - u_1)/X_0 ) ), 2.0 ) / X_0;	
		double sigma_t2  = (a_0/3)*pow(u_2, 3.0) + (a_1/12)*pow(u_2, 4.0) + (a_2/30)*pow(u_2, 5.0) + (a_3/60)*pow(u_2, 6.0) + (a_4/105)*pow(u_2, 7.0) + (a_5/168)*pow(u_2, 8.0) 
						 - (a_0/3)*pow(u_1, 3.0) - (a_1/4)*pow(u_1, 4.0) - (a_2/5)*pow(u_1, 5.0) - (a_3/6)*pow(u_1, 6.0) - (a_4/7)*pow(u_1, 7.0) - (a_5/8)*pow(u_1, 8.0) 
						 + 2*u_2*( (a_0/2)*pow(u_1, 2.0) + (a_1/3)*pow(u_1, 3.0) + (a_2/4)*pow(u_1, 4.0) + (a_3/5)*pow(u_1, 5.0) + (a_4/6)*pow(u_1, 6.0) + (a_5/7)*pow(u_1, 7.0) ) 
						 - pow(u_2, 2.0) * ( a_0*u_1 + (a_1/2)*pow(u_1, 2.0) + (a_2/3)*pow(u_1, 3.0) + (a_3/4)*pow(u_1, 4.0) + (a_4/5)*pow(u_1, 5.0) + (a_5/6)*pow(u_1, 6.0) );
		double sigma_t2_theta2	= pow(u_2, 2.0 )*( (a_0/2) + (a_1/6)*u_2 + (a_2/12)*pow(u_2, 2.0) + (a_3/20)*pow(u_2, 3.0) + (a_4/30)*pow(u_2, 4.0) + (a_5/42)*pow(u_2, 5.0) ) 
								- u_2*u_1*( a_0 + (a_1/2)*u_1 + (a_2/3)*pow(u_1, 2.0) + (a_3/4)*pow(u_1, 3.0) + (a_4/5)*pow(u_1, 4.0) + (a_5/6)*pow(u_1, 5.0) ) 
								+ pow(u_1, 2.0 )*( (a_0/2) + (a_1/3)*u_1 + (a_2/4)*pow(u_1, 2.0) + (a_3/5)*pow(u_1, 3.0) + (a_4/6)*pow(u_1, 4.0) + (a_5/7)*pow(u_1, 5.0) );
		double sigma_theta2 = a_0 * ( u_2 - u_1 ) + ( a_1 / 2 ) * ( pow(u_2, 2.0) - pow(u_1, 2.0) ) + ( a_2 / 3 ) * ( pow(u_2, 3.0) - pow(u_1, 3.0) ) 
							+ ( a_3 / 4 ) * ( pow(u_2, 4.0) - pow(u_1, 4.0) ) + ( a_4 / 5 ) * ( pow(u_2, 5.0) - pow(u_1, 5.0) ) + ( a_5 /6 )*( pow(u_2, 6.0) - pow(u_1, 6.0) );	
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

		voxel = int(voxel_x + voxel_y * MLP_IMAGE_VOXELS_X + voxel_z * MLP_IMAGE_VOXELS_X * MLP_IMAGE_VOXELS_Y);
		if( voxel != path[path_index - 1] )
			path[path_index++] = voxel;
		for( int i = 0; i < path_index; i++ )
			printf( "path[i] = %d\n", path[i] );
		printf( "path_index = %d\n\n", path_index );
		fgets(text, sizeof text, stdin);
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
void construct_space_carve_object()
{
	//Create space carve object, init to zeros
	space_carve_object_h = (int*)calloc( VOXELS, sizeof(int));

	// Set inner cylinder to 1s
	for( int slice = 0; slice < VOXELS_Z; slice++ )
		for( int row = 0; row < VOXELS_Y; row++ )
			for( int column = 0; column < VOXELS_X; column++ )
			{
				float x = ( column - VOXELS_X/2 + 0.5 ) * VOXEL_WIDTH;
				float y = ( VOXELS_Y/2 - row - 0.5 ) * VOXEL_HEIGHT;
				if( ( (x * x) + (y * y) ) < float(RECON_CYL_RADIUS * RECON_CYL_RADIUS) )
					space_carve_object_h[slice * VOXELS_X * VOXELS_Y + row * VOXELS_X + column] = 1;
			}
	cudaMalloc((void**) &space_carve_object_d,	IMAGE_INT_MEM_SIZE);
	cudaMemcpy(space_carve_object_d, space_carve_object_h, IMAGE_INT_MEM_SIZE, cudaMemcpyHostToDevice) ;

}
void construct_pure_space_carve_object()
{
	//Create space carve object, init to zeros
	pure_space_carve_object_h = (int*)calloc( VOXELS, sizeof(int));

	// Set inner cylinder to 1s
	for( int slice = 0; slice < VOXELS_Z; slice++ )
		for( int row = 0; row < VOXELS_Y; row++ )
			for( int column = 0; column < VOXELS_X; column++ )
			{
				float x = ( column - VOXELS_X/2 + 0.5 ) * VOXEL_WIDTH;
				float y = ( VOXELS_Y/2 - row - 0.5 ) * VOXEL_HEIGHT;
				if( ( (x * x) + (y * y) ) < float(RECON_CYL_RADIUS * RECON_CYL_RADIUS) )
					pure_space_carve_object_h[slice * VOXELS_X * VOXELS_Y + row * VOXELS_X + column] = 1;
			}
	cudaMalloc((void**) &pure_space_carve_object_d,	IMAGE_INT_MEM_SIZE);
	cudaMemcpy(pure_space_carve_object_d, pure_space_carve_object_h, IMAGE_INT_MEM_SIZE, cudaMemcpyHostToDevice) ;

}
void construct_space_model_object()
{
	//Create space carve object, init to zeros
	space_model_object_h = (int*)calloc( VOXELS, sizeof(int));

	// Set inner cylinder to 1s
	for( int slice = 0; slice < VOXELS_Z; slice++ )
		for( int row = 0; row < VOXELS_Y; row++ )
			for( int column = 0; column < VOXELS_X; column++ )
			{
				float x = ( column - VOXELS_X/2 + 0.5 ) * VOXEL_WIDTH;
				float y = ( VOXELS_Y/2 - row - 0.5 ) * VOXEL_HEIGHT;
				if( ( (x * x) + (y * y) ) < float(RECON_CYL_RADIUS * RECON_CYL_RADIUS) )
					space_model_object_h[slice * VOXELS_X * VOXELS_Y + row * VOXELS_X + column] = 1;
			}

	unsigned int space_model_object_size = VOXELS * sizeof(int);
	cudaMalloc( (void**) &space_model_object_d,	space_model_object_size );
	cudaMemcpy( space_model_object_d, space_model_object_h, IMAGE_INT_MEM_SIZE, cudaMemcpyHostToDevice ) ;

}
void write_integer_array_to_files( char* output_filename_base, const char* output_directory, int* integer_array, int x_max, int y_max, int z_max )
{
	// Write each slice of the space carved object to a separate file
	for(int z = 0; z < z_max; z++)
	{
		ofstream output_file;
		char output_filename[128];
		sprintf( output_filename, "%s/%s_%d.txt", output_directory, output_filename_base, z );
		output_file.open(output_filename);
			for(int y = 0; y < y_max; y++)
			{
				for(int x = 0; x < x_max; x++)
					output_file << integer_array[(z*x_max*y_max)+(y*x_max)+x] << " ";
				output_file << endl;
			}
		output_file.close();	
	}
}
void write_integer_array_to_file( char* output_filename_base, const char* output_directory, int* integer_array, int x_max, int y_max, int z_max )
{
	// Write each slice of the space carved object to a single file
	ofstream output_file;
	char output_filename[128];
	sprintf( output_filename, "%s/%s.txt", output_directory, output_filename_base );
	output_file.open(output_filename);
	for( int z = 0; z < z_max; z++ )
	{		
			for( int y = 0; y < y_max; y++ )
			{
				for( int x = 0; x < x_max; x++ )
					output_file << integer_array[( z * x_max * y_max ) + ( y * x_max ) + x] << " ";	
				output_file << endl;
			}		
	}//607,999
	output_file.close();
}
void write_float_array_to_files( char* output_filename_base, const char* output_directory, float* float_array, int x_max, int y_max, int z_max )
{
	// Write each slice of the space carved object to a separate file
	for(int z = 0; z < z_max; z++)
	{
		ofstream output_file;
		char output_filename[128];
		sprintf( output_filename, "%s/%s_%d.txt", output_directory, output_filename_base, z );
		output_file.open(output_filename);
			for(int y = 0; y < y_max; y++)
			{
				for(int x = 0; x < x_max; x++)
					output_file << float_array[(z*x_max*y_max)+(y*x_max)+x] << " ";
				output_file << endl;
			}
		output_file.close();	
	}
}
void read_SSD_positions() //HERE THE COORDINATES OF THE DETECTORS PLANES ARE LOADED, THE CONFIG FILE IS CREATED BY FORD (RWS)
{
	//char configFilename[512];
	//sprintf(configFilename, "%s/scan.cfg", input_dir);
	//
	//printf("Opening config file %s...\n", configFilename);
	//ifstream configFile(configFilename);
	//char text[20];
	////fgets(text, sizeof text, stdin);
	//if( !configFile.is_open() ) {
	//	printf("ERROR: config file not found at %s!\n", configFilename);	
	//	fputs("Didn't Find File", stdout);
	//	fflush(stdout); 
	//	//fgets(text, sizeof text, stdin);
	//	printf("text = \"%s\"\n", text);
	//	exit(1);
	//}
	//else
	//{
	//	fputs("Found File", stdout);
	//	fflush(stdout);
	//	//fgets(text, sizeof text, stdin);
	//	printf("text = \"%s\"\n", text);
	//}
	//
	//printf("Reading tracking plane positions...\n");
	//for( int i = 0; i < 8; i++ ) {
	//	configFile >> SSD_u_Positions[i];
	//	printf("SSD_u_Positions[%d] = %3f", i, SSD_u_Positions[i]);
	//}
	//
	//configFile.close();
	//Lucy3
	/*SSD_u_Positions[0] = -206.93;
	SSD_u_Positions[1] = -197.73;
	SSD_u_Positions[2] = -108.93;
	SSD_u_Positions[3] = -99.73;
	SSD_u_Positions[4] = 99.73;
	SSD_u_Positions[5] = 108.93;
	SSD_u_Positions[6] = 197.73;
	SSD_u_Positions[7] = 206.93;*/

	////Lucy3 500,450,-450.-500
	//SSD_u_Positions[0] = -206.93;
	//SSD_u_Positions[1] = -197.73;
	//SSD_u_Positions[2] = -108.93;
	//SSD_u_Positions[3] = -99.73;
	//SSD_u_Positions[4] = 99.73;
	//SSD_u_Positions[5] = 108.93;
	//SSD_u_Positions[6] = 197.73;
	//SSD_u_Positions[7] = 206.93;

	////Simulated_Data 9-21 100,50,-50,-100
	/*SSD_u_Positions[0] = -100.00;
	SSD_u_Positions[1] = -100.00;
	SSD_u_Positions[2] = -50.00;
	SSD_u_Positions[3] = -50.00;
	SSD_u_Positions[4] = 50.00;
	SSD_u_Positions[5] = 50.00;
	SSD_u_Positions[6] = 100.00;
	SSD_u_Positions[7] = 100.00;*/

	//////Simulated_Data 9-21 100,50,-50,-100
	/*SSD_u_Positions[0] = -100.00;
	SSD_u_Positions[1] = -100.00;
	SSD_u_Positions[2] = -50.00;
	SSD_u_Positions[3] = -50.00;
	SSD_u_Positions[4] = 50.00;
	SSD_u_Positions[5] = 50.00;
	SSD_u_Positions[6] = 100.00;
	SSD_u_Positions[7] = 100.00;*/

	//////Simulated_Data 9-21 100,50,-50,-100
	SSD_u_Positions[0] = -200.00;
	SSD_u_Positions[1] = -200.00;
	SSD_u_Positions[2] = -150.00;
	SSD_u_Positions[3] = -150.00;
	SSD_u_Positions[4] = 150.00;
	SSD_u_Positions[5] = 150.00;
	SSD_u_Positions[6] = 200.00;
	SSD_u_Positions[7] = 200.00;

}
void initializations()
{
	for( int translation = 0; translation < TRANSLATIONS; translation++ )
		histories_per_translation[translation] = 0;

	histories_per_file =				 (int*) calloc( TRANSLATIONS * GANTRY_ANGLES, sizeof(int) );
	histories_per_gantry_angle =		 (int*) calloc( GANTRY_ANGLES, sizeof(int) );
	recon_vol_histories_per_projection = (int*) calloc( GANTRY_ANGLES, sizeof(int) );

	bin_counts_h		  = (int*)	 calloc( NUM_BINS, sizeof(int) );
	mean_WEPL_h			  = (float*) calloc( NUM_BINS, sizeof(float) );
	mean_rel_ut_angle_h	  = (float*) calloc( NUM_BINS, sizeof(float) );
	mean_rel_uv_angle_h	  = (float*) calloc( NUM_BINS, sizeof(float) );
	stddev_rel_ut_angle_h = (float*) calloc( NUM_BINS, sizeof(float) );	
	stddev_rel_uv_angle_h = (float*) calloc( NUM_BINS, sizeof(float) );	
	stddev_WEPL_h		  = (float*) calloc( NUM_BINS, sizeof(float) );

	cudaMalloc((void**) &bin_counts_d,			mem_size_bins_ints );
	cudaMalloc((void**) &mean_WEPL_d,			mem_size_bins_floats );
	cudaMalloc((void**) &mean_rel_ut_angle_d,	mem_size_bins_floats );
	cudaMalloc((void**) &mean_rel_uv_angle_d,	mem_size_bins_floats );
	cudaMalloc((void**) &stddev_rel_ut_angle_d,	mem_size_bins_floats );
	cudaMalloc((void**) &stddev_rel_uv_angle_d,	mem_size_bins_floats );
	cudaMalloc((void**) &stddev_WEPL_d,			mem_size_bins_floats );

	cudaMemcpy( bin_counts_d,		   bin_counts_h,		  mem_size_bins_ints,	cudaMemcpyHostToDevice );
	cudaMemcpy( mean_WEPL_d,		   mean_WEPL_h,		      mem_size_bins_floats, cudaMemcpyHostToDevice );
	cudaMemcpy( mean_rel_ut_angle_d,   mean_rel_ut_angle_h,   mem_size_bins_floats, cudaMemcpyHostToDevice );
	cudaMemcpy( mean_rel_uv_angle_d,   mean_rel_uv_angle_h,   mem_size_bins_floats, cudaMemcpyHostToDevice );
	cudaMemcpy( stddev_rel_ut_angle_d, stddev_rel_ut_angle_h, mem_size_bins_floats, cudaMemcpyHostToDevice );
	cudaMemcpy( stddev_rel_uv_angle_d, stddev_rel_uv_angle_h, mem_size_bins_floats, cudaMemcpyHostToDevice );
}
void count_histories()
{
	printf("Counting histories...\n");
	char data_filename[128];
	int file_size, num_histories, file_number = 0, gantry_position_number = 0;
	for( int gantry_angle = 0; gantry_angle < 360; gantry_angle += GANTRY_ANGLE_INTERVAL, gantry_position_number++ )
	{
		for( int translation = 1; translation <= TRANSLATIONS; translation++, file_number++ )
		{
			if( binary_data_files )
			{
				sprintf( data_filename, "%s/%s_trans%d_%03d.dat", input_dir, input_base_name, translation, gantry_angle );
				FILE *data_file = fopen(data_filename, "rb");
				if( data_file == NULL )
				{
					fputs( "Error Opening Data File:  Check that the directories are properly named.", stderr ); 
					exit(1);
				}
				fseek( data_file, 0, SEEK_END );
				file_size = ftell( data_file );
				if( file_size % BYTES_PER_HISTORY ) 
				{
					printf("ERROR! Problem with bytes_per_history!\n");
					exit(2);
				}
				num_histories = file_size / BYTES_PER_HISTORY;				
				fclose(data_file);
			}
			else
			{
				sprintf( data_filename, "%s/%s_trans%d_%03d.txt", input_dir, input_base_name, translation, gantry_angle );
				FILE *data_file = fopen(data_filename, "r");
				if( data_file == NULL )
				{
					fputs( "Error Opening Data File:  Check that the directories are properly named.", stderr ); 
					exit (1);
				}
				fseek( data_file, 0, SEEK_END );
				num_histories = ftell( data_file );
				fclose(data_file);
			}
			histories_per_file[file_number] = num_histories;
			histories_per_gantry_angle[gantry_position_number] += num_histories;
			histories_per_translation[translation-1] += num_histories;
			total_histories += num_histories;
			
			if(debug_text_on)
				printf("There are %d histories in the file for translation %d gantry_angle %d\n",num_histories, translation, gantry_angle);
		}
	}
	if(debug_text_on)
	{
		for( int file_num = 0, int gantry_position_number = 0; file_num < (TRANSLATIONS * GANTRY_ANGLES); file_num++ )
		{
			printf("There are %d histories in the file for translation %d gantry_angle %d\n", histories_per_file[file_num], (file_num%TRANSLATIONS) + 1, (file_num/TRANSLATIONS )*GANTRY_ANGLE_INTERVAL);
			if(file_num%2 == 1)
			{
				printf("There are %d histories in gantry_angle %d\n", histories_per_gantry_angle[gantry_position_number], (file_num/TRANSLATIONS )*GANTRY_ANGLE_INTERVAL);
				gantry_position_number++;
			}				
		}
		printf("There are %d histories in translation 1 \n",histories_per_translation[0]);
		printf("There are %d histories in translation 2 \n",histories_per_translation[1]);
		printf("There are a total of %d histories\n",total_histories);
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
void vector_to_array_transfer()
{
	unsigned int histories_remaining = WEPL_vector.size();

	valid_bin_num = (int*)malloc( bin_num_vector.size() * sizeof(int) );
	copy(bin_num_vector.begin(), bin_num_vector.end(), valid_bin_num);
	vector<int>().swap(bin_num_vector);

	valid_WEPL = (float*)malloc( WEPL_vector.size() * sizeof(float) );
	copy(WEPL_vector.begin(), WEPL_vector.end(), valid_WEPL);
	vector<float>().swap(WEPL_vector);

	valid_x_entry = (float*)malloc( x_entry_vector.size() * sizeof(float) );
	copy(x_entry_vector.begin(), x_entry_vector.end(), valid_x_entry);
	vector<float>().swap(x_entry_vector);

	valid_y_entry = (float*)malloc( y_entry_vector.size() * sizeof(float) );
	copy(y_entry_vector.begin(), y_entry_vector.end(), valid_y_entry);
	vector<float>().swap(y_entry_vector);

	valid_z_entry = (float*)malloc( z_entry_vector.size() * sizeof(float) );
	copy(z_entry_vector.begin(), z_entry_vector.end(), valid_z_entry);
	vector<float>().swap(z_entry_vector);

	valid_x_exit = (float*)malloc( x_exit_vector.size() * sizeof(float) );
	copy(x_exit_vector.begin(), x_exit_vector.end(), valid_x_exit);
	vector<float>().swap(x_exit_vector);

	valid_y_exit = (float*)malloc( y_exit_vector.size() * sizeof(float) );
	copy(y_exit_vector.begin(), y_exit_vector.end(), valid_y_exit);
	vector<float>().swap(y_exit_vector);

	valid_z_exit = (float*)malloc( z_exit_vector.size() * sizeof(float) );
	copy(z_exit_vector.begin(), z_exit_vector.end(), valid_z_exit);
	vector<float>().swap(z_exit_vector);

	valid_xy_entry_angle = (float*)malloc( xy_entry_angle_vector.size() * sizeof(float) );
	copy(xy_entry_angle_vector.begin(), xy_entry_angle_vector.end(), valid_xy_entry_angle);
	vector<float>().swap(xy_entry_angle_vector);

	valid_xz_entry_angle = (float*)malloc( xz_entry_angle_vector.size() * sizeof(float) );
	copy(xz_entry_angle_vector.begin(), xz_entry_angle_vector.end(), valid_xz_entry_angle);
	vector<float>().swap(xz_entry_angle_vector);

	valid_xy_exit_angle = (float*)malloc( xy_exit_angle_vector.size() * sizeof(float) );
	copy(xy_exit_angle_vector.begin(), xy_exit_angle_vector.end(), valid_xy_exit_angle);
	vector<float>().swap(xy_exit_angle_vector);

	valid_xz_exit_angle = (float*)malloc( xz_exit_angle_vector.size() * sizeof(float) );
	copy(xz_exit_angle_vector.begin(), xz_exit_angle_vector.end(), valid_xz_exit_angle);
	vector<float>().swap(xz_exit_angle_vector);

	/*
	valid_bin_num			= (int*)   calloc( recon_vol_histories,	sizeof(int) );
	valid_WEPL				= (float*) calloc( recon_vol_histories,	sizeof(float) );
	valid_x_entry			= (float*) calloc( recon_vol_histories,	sizeof(float) );
	valid_y_entry			= (float*) calloc( recon_vol_histories,	sizeof(float) );
	valid_z_entry			= (float*) calloc( recon_vol_histories,	sizeof(float) );
	valid_x_exit			= (float*) calloc( recon_vol_histories,	sizeof(float) );
	valid_y_exit			= (float*) calloc( recon_vol_histories,	sizeof(float) );
	valid_z_exit			= (float*) calloc( recon_vol_histories,	sizeof(float) );
	valid_xy_entry_angle	= (float*) calloc( recon_vol_histories,	sizeof(float) );
	valid_xz_entry_angle	= (float*) calloc( recon_vol_histories,	sizeof(float) );
	valid_xy_exit_angle		= (float*) calloc( recon_vol_histories,	sizeof(float) );
	valid_xz_exit_angle		= (float*) calloc( recon_vol_histories,	sizeof(float) );
	*/
	
	printf("Capacity of bin_num_vector = %3f\n", bin_num_vector.capacity() );
	printf("Capacity of WEPL_vector = %3f\n", WEPL_vector.capacity() );
	printf("Capacity of x_entry_vector = %3f\n", x_entry_vector.capacity() );
	printf("Capacity of y_entry_vector = %3f\n", y_entry_vector.capacity() );
	printf("Capacity of z_entry_vector = %3f\n", z_entry_vector.capacity() );
	printf("Capacity of x_exit_vector = %3f\n", x_exit_vector.capacity() );
	printf("Capacity of y_exit_vector = %3f\n", y_exit_vector.capacity() );
	printf("Capacity of z_exit_vector = %3f\n", z_exit_vector.capacity() );
	printf("Capacity of xy_entry_angle_vector = %3f\n", xy_entry_angle_vector.capacity() );
	printf("Capacity of xz_entry_angle_vector = %3f\n", xz_entry_angle_vector.capacity() );
	printf("Capacity of xy_exit_angle_vector = %3f\n", xy_exit_angle_vector.capacity() );
	printf("Capacity of xz_exit_angle_vector = %3f\n", xz_exit_angle_vector.capacity() );
//		vector<int>	bin_num_vector;			
//vector<float> WEPL_vector;		
//vector<float> x_entry_vector;		
//vector<float> y_entry_vector;		
//vector<float> z_entry_vector;		
//vector<float> x_exit_vector;			
//vector<float> y_exit_vector;			
//vector<float> z_exit_vector;			
//vector<float> xy_entry_angle_vector;	
//vector<float> xz_entry_angle_vector;	
//vector<float> xy_exit_angle_vector;	
//vector<float> xz_exit_angle_vector;
}
void iterative_data_read( int num_histories, int start_file_num, int end_file_num )
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

	char data_filename[128];
	int array_index = 0;
	//float min_WEPL = 20, max_rel_angle = -100;
	for( int file_num = start_file_num; file_num <= end_file_num; file_num++ )
	{
		int gantry_position = file_num / TRANSLATIONS;
		int gantry_angle = gantry_position * GANTRY_ANGLE_INTERVAL;
		int translation = file_num % TRANSLATIONS + 1;
		int translation_histories = histories_per_file[file_num];

		printf("Reading file for gantry angle %d from translation %d\n", gantry_angle, translation );
		sprintf( data_filename, "%s/%s_trans%d_%03d.dat", input_dir, input_base_name, translation, gantry_angle );
		//sprintf( data_filename2, "%s/%s_trans%d_%03d_out.txt", input_dir, input_base_name, translation, gantry_angle );
		FILE* data_file = fopen( data_filename, "rb" );	
	//	ofstream output_file;
		//output_file.open( data_filename2 );

		for( int history = 0; history < translation_histories; history++, array_index++ ) 
		{
			float v_data[4], t_data[4], WEPL_data, gantry_angle_data, dummy_data;
			char detector_number[4];

			fread(&v_data,				sizeof(float),	4, data_file);
			fread(&t_data,				sizeof(float),	4, data_file);
			fread(&detector_number,		sizeof(char),	4, data_file);
			fread(&WEPL_data,			sizeof(float),	1, data_file);
			fread(&gantry_angle_data,	sizeof(float),	1, data_file);
			fread(&dummy_data,			sizeof(float),	1, data_file); // dummy read because each event has an extra 4 bytes, for some reason

			// Convert the mm displacements to cm
			v_in_1_h[array_index]	= v_data[0] * 0.1;
			v_in_2_h[array_index]	= v_data[1] * 0.1;
			v_out_1_h[array_index]	= v_data[2] * 0.1;
			v_out_2_h[array_index]	= v_data[3] * 0.1;
			t_in_1_h[array_index]	= t_data[0] * 0.1;
			t_in_2_h[array_index]	= t_data[1] * 0.1;
			t_out_1_h[array_index]	= t_data[2] * 0.1;
			t_out_2_h[array_index]	= t_data[3] * 0.1;
			WEPL_h[array_index]		= WEPL_data * 0.1;
			u_in_1_h[array_index]	= SSD_u_Positions[0] * 0.1;
			u_in_2_h[array_index]	= SSD_u_Positions[2] * 0.1;
			u_out_1_h[array_index]	= SSD_u_Positions[4] * 0.1;
			u_out_2_h[array_index]	= SSD_u_Positions[6] * 0.1;
			/*u_in_1_h[array_index]	= SSD_u_Positions[0];
			u_in_2_h[array_index]	= SSD_u_Positions[2];
			u_out_1_h[array_index]	= SSD_u_Positions[4];
			u_out_2_h[array_index]	= SSD_u_Positions[6];*/
			gantry_angle_h[array_index] = int(gantry_angle_data);

			//output_file << v_in_1_h[array_index] <<" "<< v_in_2_h[array_index] <<" "<< v_out_1_h[array_index] <<" "<< v_out_2_h[array_index] <<" "
			//	<< t_in_1_h[array_index] <<" "<< t_in_2_h[array_index] <<" "<< t_out_1_h[array_index] <<" "<< t_out_2_h[array_index] <<" "
				//<< WEPL_h[array_index] <<"\n";

			/*fwrite(&v_data,sizeof(float),4,data_file_out);
			fwrite(&t_data,sizeof(float),4,data_file_out);
			fwrite(&detector_number,sizeof(char),4,data_file_out);
			fwrite(&WEPL_data,sizeof(float),1,data_file_out);
			fwrite(&gantry_angle_data,sizeof(float),1,data_file_out);
			fwrite(&dummy_data,sizeof(float),1,data_file_out); */
	/*		u_in_1_h[array_index]	= SSD_u_Positions[int(detector_number[0])] * 0.1;
			u_in_2_h[array_index]	= SSD_u_Positions[int(detector_number[1])] * 0.1;
			u_out_1_h[array_index]	= SSD_u_Positions[int(detector_number[2])] * 0.1;
			u_out_2_h[array_index]	= SSD_u_Positions[int(detector_number[3])] * 0.1;*/
			//if( file_num == 0 && history < 100 )
			//{
			//	printf("Detector Number 0 = %c\n", detector_number[0] );
			//	printf("Detector Number 1 = %c\n", detector_number[1] );
			//	printf("Detector Number 2 = %c\n", detector_number[2] );
			//	printf("Detector Number 3 = %c\n\n", detector_number[3] );

			///*	printf("Detector Number 0 = %d\n", atoi(detector_number[0]) );
			//	printf("Detector Number 1 = %d\n", atoi(detector_number[1]) );
			//	printf("Detector Number 2 = %d\n", atoi(detector_number[2]) );
			//	printf("Detector Number 3 = %d\n\n", atoi(detector_number[3]) );*/
			//	
			//
			//	printf("v_in_1_h = %3f\n", v_in_1_h[array_index]);
			//	printf("v_in_2_h = %3f\n", v_in_2_h[array_index]);
			//	printf("v_out_1_h = %3f\n", v_out_1_h[array_index]);
			//	printf("v_out_2_h = %3f\n", v_out_2_h[array_index]);
			//	printf("t_in_1_h = %3f\n", t_in_1_h[array_index]);
			//	printf("t_in_2_h = %3f\n", t_in_2_h[array_index]);
			//	printf("t_out_1_h = %3f\n", t_out_1_h[array_index]);
			//	printf("t_out_2_h = %3f\n", t_out_2_h[array_index]);
			//	printf("u_in_1_h = %3f\n", u_in_1_h[array_index]);
			//	printf("u_in_2_h = %3f\n", u_in_2_h[array_index]);
			//	printf("u_out_1_h = %3f\n", u_out_1_h[array_index]);
			//	printf("u_out_2_h = %3f\n", u_out_2_h[array_index]);
			//	printf("gantry_angle_h = %3f\n", gantry_angle_h[array_index]);
			//	printf("WEPL_h = %3f\n\n", WEPL_h[array_index]);
			//}
			//char text[20];
			//fgets(text, sizeof text, stdin);
		}
		fclose(data_file);
		//output_file.close();
		
	}
}
void recon_volume_intersections( int num_histories )
{
	printf("There are %d histories in this projection\n", num_histories );
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
	cudaMalloc((void**) &traversed_recon_volume_d,	mem_size_hist_bool);	
	cudaMalloc((void**) &xy_entry_angle_d,			mem_size_hist_floats);	
	cudaMalloc((void**) &xz_entry_angle_d,			mem_size_hist_floats);
	cudaMalloc((void**) &xy_exit_angle_d,			mem_size_hist_floats);
	cudaMalloc((void**) &xz_exit_angle_d,			mem_size_hist_floats);

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
__global__ void recon_volume_intersections_kernel
(
	int num_histories, int* gantry_angle, bool* traversed_recon_volume, float* WEPL,
	float* t_in_1, float* t_in_2, float* t_out_1, float* t_out_2,
	float* u_in_1, float* u_in_2, float* u_out_1, float* u_out_2,
	float* v_in_1, float* v_in_2, float* v_out_1, float* v_out_2, 	
	float* x_entry, float* y_entry, float* z_entry, float* x_exit, float* y_exit, float* z_exit, 	
	float* xy_entry_angle, float* xz_entry_angle, float* xy_exit_angle, float* xz_exit_angle
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
	float rotation_angle_radians = gantry_angle[i] * PI/180;
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
			xy_entry_angle[i] += 2*PI;

		// Rotate entry detector positions
		float x_in = ( cosf( rotation_angle_radians ) * u_in_2[i] ) - ( sinf( rotation_angle_radians ) * t_in_2[i] );
		float y_in = ( sinf( rotation_angle_radians ) * u_in_2[i] ) + ( cosf( rotation_angle_radians ) * t_in_2[i] );

		// Determine if entry points should be rotated
		bool entry_in_cone = 
		( 
			(xy_entry_angle[i] > 0.25*PI) &&	// 1/4 PI
			(xy_entry_angle[i] < 0.75*PI)		// 3/4 PI
		) 
		|| 
		( 
			(xy_entry_angle[i] > 1.25*PI) &&	// 5/4 PI
			(xy_entry_angle[i] < 1.75*PI)		// 7/4 PI
		);

		// Rotate x_in & y_in by 90 degrees, if necessary
		if( entry_in_cone )
		{
			x_temp = x_in;
			y_temp = y_in;		
			x_in = -y_temp;
			y_in = x_temp;
			xy_entry_angle[i] += PI/2;
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
		}
		/***************************************************************************************************************/
		/****************************************** Check exit information *********************************************/
		/***************************************************************************************************************/
		
		// Repeat the procedure above, this time to determine if the proton path exited the reconstruction volume and if so, the
		// x,y,z position where it exited
		float ut_exit_angle = atan2f( t_out_2[i] - t_out_1[i], u_out_2[i] - u_out_1[i] );
		xy_exit_angle[i] = ut_exit_angle + rotation_angle_radians;
		if( xy_exit_angle[i] < 0 )
			xy_exit_angle[i] += 2*PI;

		// Rotate exit detector positions
		float x_out = ( cosf(rotation_angle_radians) * u_out_1[i] ) - ( sinf(rotation_angle_radians) * t_out_1[i] );
		float y_out = ( sinf(rotation_angle_radians) * u_out_1[i] ) + ( cosf(rotation_angle_radians) * t_out_1[i] );

		// Determine if exit points should be rotated
		bool exit_in_cone = 
		( 
			(xy_exit_angle[i] > 0.25*PI) &&	// 1/4 PI
			(xy_exit_angle[i] < 0.75*PI)		// 3/4 PI
		) 
		|| 
		( 
			(xy_exit_angle[i] > 1.25*PI) &&	// 5/4 PI
			(xy_exit_angle[i] < 1.75*PI)		// 7/4 PI
		);

		// Rotate x_out & y_out by 90 degrees, if necessary
		if( exit_in_cone )
		{
			x_temp = x_out;
			y_temp = y_out;		
			x_out = -y_temp;
			y_out = x_temp;	
			xy_exit_angle[i] += PI/2;
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
		}
		/***************************************************************************************************************/
		/***************************************** Check z(v) direction ************************************************/
		/***************************************************************************************************************/		

		// Relevant angles/slopes in radians for entry and exit in the uv plane
		float uv_entry_slope = ( v_in_2[i] - v_in_1[i] ) / ( u_in_2[i] - u_in_1[i] );
		float uv_exit_slope = ( v_out_2[i] - v_out_1[i] ) / ( u_out_2[i] - u_out_1[i] );
		/*
		float uv_entry_angle = atan2( v_in_2[i] - v_in_1[i], u_in_2[i] - u_in_1[i] );
		float uv_exit_angle = atan2( v_out_2[i] - v_out_1[i],  u_out_2[i] - u_out_1[i] );
		xz_entry_angle[i] = uv_entry_angle + rotation_angle_radians;
		xz_exit_angle[i] = uv_exit_angle + rotation_angle_radians;
		*/
		xz_entry_angle[i] = atan2( v_in_2[i] - v_in_1[i], u_in_2[i] - u_in_1[i] );
		xz_exit_angle[i] = atan2( v_out_2[i] - v_out_1[i],  u_out_2[i] - u_out_1[i] );

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

		// Proton passed through the reconstruction volume only if it both entered and exited the reconstruction cylinder
		traversed_recon_volume[i] = entered && exited;
	}	
}
void bin_valid_histories( int num_histories )
{
	unsigned int mem_size_hist_floats	= sizeof(float) * num_histories;
	unsigned int mem_size_hist_ints		= sizeof(int) * num_histories;
	unsigned int mem_size_hist_bool		= sizeof(bool) * num_histories;

	traversed_recon_volume_h	= (bool*)  calloc( num_histories, sizeof(bool)	);
	bin_num_h					= (int*)   calloc( num_histories, sizeof(int)   );

	x_entry_h	= (float*) calloc( num_histories, sizeof(float) );
	y_entry_h	= (float*) calloc( num_histories, sizeof(float) );
	z_entry_h	= (float*) calloc( num_histories, sizeof(float) );
	x_exit_h	= (float*) calloc( num_histories, sizeof(float) );
	y_exit_h	= (float*) calloc( num_histories, sizeof(float) );
	z_exit_h	= (float*) calloc( num_histories, sizeof(float) );	

	xy_entry_angle_h	= (float*) calloc( num_histories, sizeof(float) );	
	xz_entry_angle_h	= (float*) calloc( num_histories, sizeof(float) );
	xy_exit_angle_h		= (float*) calloc( num_histories, sizeof(float) );
	xz_exit_angle_h		= (float*) calloc( num_histories, sizeof(float) );

	//cudaMalloc((void**) &bin_num_d,	mem_size_hist_ints );
	//cudaMemcpy( bin_num_d,	bin_num_h,	mem_size_hist_ints, cudaMemcpyHostToDevice );

	dim3 dimBlock( THREADS_PER_BLOCK );
	dim3 dimGrid( (int)( num_histories/THREADS_PER_BLOCK ) + 1 );
	bin_valid_histories_kernel<<<dimGrid, dimBlock>>>
	( 
		num_histories, bin_counts_d, bin_num_d, traversed_recon_volume_d, 
		x_entry_d, y_entry_d, z_entry_d, x_exit_d, y_exit_d, z_exit_d, 
		mean_WEPL_d, mean_rel_ut_angle_d, mean_rel_uv_angle_d, WEPL_d, 
		xy_entry_angle_d, xz_entry_angle_d, xy_exit_angle_d, xz_exit_angle_d
	);
	cudaMemcpy( traversed_recon_volume_h,	traversed_recon_volume_d,	mem_size_hist_bool, cudaMemcpyDeviceToHost );
	cudaMemcpy( bin_num_h,					bin_num_d,					mem_size_hist_ints, cudaMemcpyDeviceToHost );
	cudaMemcpy( x_entry_h,					x_entry_d,					mem_size_hist_floats, cudaMemcpyDeviceToHost );
	cudaMemcpy( y_entry_h,					y_entry_d,					mem_size_hist_floats, cudaMemcpyDeviceToHost );
	cudaMemcpy( z_entry_h,					z_entry_d,					mem_size_hist_floats, cudaMemcpyDeviceToHost );
	cudaMemcpy( x_exit_h,					x_exit_d,					mem_size_hist_floats, cudaMemcpyDeviceToHost );
	cudaMemcpy( y_exit_h,					y_exit_d,					mem_size_hist_floats, cudaMemcpyDeviceToHost );
	cudaMemcpy( z_exit_h,					z_exit_d,					mem_size_hist_floats, cudaMemcpyDeviceToHost );

	cudaMemcpy( xy_entry_angle_h,	xy_entry_angle_d,	mem_size_hist_floats, cudaMemcpyDeviceToHost );
	cudaMemcpy( xz_entry_angle_h,	xz_entry_angle_d,	mem_size_hist_floats, cudaMemcpyDeviceToHost );
	cudaMemcpy( xy_exit_angle_h,	xy_exit_angle_d,	mem_size_hist_floats, cudaMemcpyDeviceToHost );
	cudaMemcpy( xz_exit_angle_h,	xz_exit_angle_d,	mem_size_hist_floats, cudaMemcpyDeviceToHost );

	//char text[20];
	unsigned int outside_circle = 0;
	unsigned int offset = 0;
	for( int i = 0; i < num_histories; i++ )
	{
		if( traversed_recon_volume_h[i] && ( bin_num_h[i] >= 0 ) )
		{
			bin_num_vector.push_back( bin_num_h[i] );
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
	printf( "%d histories out of %d histories passed this intersection cuts this iteration\n", offset, num_histories );
	printf( "%d histories outside_circle \n", outside_circle );
	valid_array_position += offset;

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

	//cudaFree( bin_num_d );
	cudaFree( xy_entry_angle_d );
	cudaFree( xz_entry_angle_d );
	cudaFree( xy_exit_angle_d );
	cudaFree( xz_exit_angle_d );
}
__global__ void bin_valid_histories_kernel
( 
	int num_histories, int* bin_counts, int* bin_num, bool* traversed_recon_volume, 
	float* x_entry, float* y_entry, float* z_entry, float* x_exit, float* y_exit, float* z_exit, 
	float* mean_WEPL, float* mean_rel_ut_angle, float* mean_rel_uv_angle, float* WEPL, 
	float* xy_entry_angle, float* xz_entry_angle, float* xy_exit_angle, float* xz_exit_angle
)
{
	int i = threadIdx.x + blockIdx.x * THREADS_PER_BLOCK;
	if( i < num_histories )
	{
		if( traversed_recon_volume[i] )
		{
			float path_angle_radians, closest_source_angle_radians;
			float x_midpath, y_midpath, z_midpath, t, v;
			int closest_source_bin, t_bin_num, v_bin_num;

			x_midpath = ( x_entry[i] + x_exit[i] ) / 2;
			y_midpath = ( y_entry[i] + y_exit[i] ) / 2;
			z_midpath = ( z_entry[i] + z_exit[i] ) / 2;

			path_angle_radians = atan2( ( y_exit[i] - y_entry[i] ) , ( x_exit[i] - x_entry[i] ) );
			if( path_angle_radians < 0 )
				path_angle_radians += 2*PI;
			closest_source_bin = int( ( path_angle_radians * 180.0/PI / ANGULAR_BIN_SIZE ) + 0.5) % ANGULAR_BINS;	
			closest_source_angle_radians = closest_source_bin * ANGULAR_BIN_SIZE * PI/180.0;

			t = y_midpath * cosf(closest_source_angle_radians) - x_midpath * sinf(closest_source_angle_radians);
			t_bin_num = int( (t / T_BIN_SIZE ) + T_BINS/2);
			
			v = z_midpath;
			v_bin_num = int( (v / V_BIN_SIZE ) + V_BINS/2);

			if( (t_bin_num >= 0) && (v_bin_num >= 0) && (t_bin_num < T_BINS) && (v_bin_num < V_BINS) )
			{
				bin_num[i] = t_bin_num + closest_source_bin * T_BINS + v_bin_num * T_BINS * ANGULAR_BINS;
				atomicAdd( &bin_counts[bin_num[i]], 1 );
				atomicAdd( &mean_WEPL[bin_num[i]], WEPL[i] );
				atomicAdd( &mean_rel_ut_angle[bin_num[i]], xy_entry_angle[i] - xy_exit_angle[i] );
				atomicAdd( &mean_rel_uv_angle[bin_num[i]], xz_entry_angle[i] - xz_exit_angle[i] );
			}
			else
				bin_num[i] = -1;
		}
	}
}
void pure_space_carve( int num_histories )
{
	dim3 dimBlock(THREADS_PER_BLOCK);
	dim3 dimGrid((int)(num_histories/THREADS_PER_BLOCK)+1);
	pure_space_carve_kernel<<<dimGrid, dimBlock>>>
	(
		num_histories, pure_space_carve_object_d, bin_num_d, traversed_recon_volume_d, WEPL_d,
		x_entry_d, y_entry_d, z_entry_d, x_exit_d, y_exit_d, z_exit_d
	);
}
__global__ void pure_space_carve_kernel
( 
	int num_histories, int* pure_space_carve_object, int* bin_num, bool* traversed_recon_volume, float* WEPL,
	float* x_entry, float* y_entry, float* z_entry, float* x_exit, float* y_exit, float* z_exit
)
{
	int i = threadIdx.x + blockIdx.x * THREADS_PER_BLOCK;
	if( (i < num_histories) && traversed_recon_volume[i] && (WEPL[i] <= PURE_SPACE_CARVE_THRESHOLD) && (bin_num[i] >= 0) )
	{
		//char text[20];
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
		x_inside = modf( ( x_entry[i] + RECON_CYL_RADIUS ) /VOXEL_WIDTH, &voxel_x)*VOXEL_WIDTH;	
		y_inside = modf( ( RECON_CYL_RADIUS - y_entry[i] ) /VOXEL_HEIGHT, &voxel_y)*VOXEL_HEIGHT;
		z_inside = modf( ( RECON_CYL_HEIGHT/2 - z_entry[i] ) /VOXEL_THICKNESS, &voxel_z)*VOXEL_THICKNESS;

		voxel = int(voxel_x + voxel_y * VOXELS_X + voxel_z * VOXELS_X * VOXELS_Y);
		voxel_x_out = int( ( x_exit[i] + RECON_CYL_RADIUS ) /VOXEL_WIDTH );
		voxel_y_out = int( ( RECON_CYL_RADIUS - y_exit[i] ) /VOXEL_HEIGHT );
		voxel_z_out = int( ( RECON_CYL_HEIGHT/2 - z_exit[i] ) /VOXEL_THICKNESS );
		voxel_out = int(voxel_x_out + voxel_y_out * VOXELS_X + voxel_z_out * VOXELS_X * VOXELS_Y);
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
			
		outside_image = ( voxel_x >= VOXELS_X ) || ( voxel_y >= VOXELS_Y ) || ( voxel_z >= VOXELS_Z );
		if( !outside_image )
			pure_space_carve_object[voxel] = 0;
		end_walk = ( voxel == voxel_out ) || outside_image;
		//fgets(text, sizeof text, stdin);
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
				//fgets(text, sizeof text, stdin);
				voxel = int(voxel_x + voxel_y * VOXELS_X + voxel_z * VOXELS_X * VOXELS_Y);
				outside_image = ( voxel_x >= VOXELS_X ) || ( voxel_y >= VOXELS_Y ) || ( voxel_z >= VOXELS_Z );
				if( !outside_image )
					pure_space_carve_object[voxel] = 0;
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
				voxel = int(voxel_x + voxel_y * VOXELS_X + voxel_z * VOXELS_X * VOXELS_Y);
				outside_image = ( voxel_x >= VOXELS_X ) || ( voxel_y >= VOXELS_Y ) || ( voxel_z >= VOXELS_Z );
				if( !outside_image )
					pure_space_carve_object[voxel] = 0;
				end_walk = ( voxel == voxel_out ) || outside_image;
				//fgets(text, sizeof text, stdin);
			}// end: while( !end_walk )
		}//end: else: z_entry_h[i] != z_exit_h[i] => z_entry_h[i] == z_exit_h[i]
	}// end: if( (i < num_histories) && traversed_recon_volume[i] && (WEPL[i] <= PURE_SC_THRESH) && (bin_num[i] >= 0) )
}
void space_carve( int num_histories )
{
	dim3 dimBlock(THREADS_PER_BLOCK);
	dim3 dimGrid((int)(num_histories/THREADS_PER_BLOCK)+1);
	space_carve_kernel<<<dimGrid, dimBlock>>>
	(
		num_histories, space_carve_object_d, bin_num_d, traversed_recon_volume_d, WEPL_d,
		x_entry_d, y_entry_d, z_entry_d, x_exit_d, y_exit_d, z_exit_d
	);
}
__global__ void space_carve_kernel
( 
	int num_histories, int* space_carve_object, int* bin_num, bool* traversed_recon_volume, float* WEPL,
	float* x_entry, float* y_entry, float* z_entry, float* x_exit, float* y_exit, float* z_exit
)
{
	int i = threadIdx.x + blockIdx.x * THREADS_PER_BLOCK;
	if( (i < num_histories) && traversed_recon_volume[i] && (WEPL[i] <= SPACE_CARVE_THRESHOLD) && (bin_num[i] >= 0) )
	{
		//char text[20];
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
		x_inside = modf( ( x_entry[i] + RECON_CYL_RADIUS ) /VOXEL_WIDTH, &voxel_x)*VOXEL_WIDTH;	
		y_inside = modf( ( RECON_CYL_RADIUS - y_entry[i] ) /VOXEL_HEIGHT, &voxel_y)*VOXEL_HEIGHT;
		z_inside = modf( ( RECON_CYL_HEIGHT/2 - z_entry[i] ) /VOXEL_THICKNESS, &voxel_z)*VOXEL_THICKNESS;

		voxel = int(voxel_x + voxel_y * VOXELS_X + voxel_z * VOXELS_X * VOXELS_Y);
		voxel_x_out = int( ( x_exit[i] + RECON_CYL_RADIUS ) /VOXEL_WIDTH );
		voxel_y_out = int( ( RECON_CYL_RADIUS - y_exit[i] ) /VOXEL_HEIGHT );
		voxel_z_out = int( ( RECON_CYL_HEIGHT/2 - z_exit[i] ) /VOXEL_THICKNESS );
		voxel_out = int(voxel_x_out + voxel_y_out * VOXELS_X + voxel_z_out * VOXELS_X * VOXELS_Y);
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
			
		outside_image = ( voxel_x >= VOXELS_X ) || ( voxel_y >= VOXELS_Y ) || ( voxel_z >= VOXELS_Z );
		if( !outside_image )
			atomicAdd( &space_carve_object[voxel], 1 );
		end_walk = ( voxel == voxel_out ) || outside_image;
		//fgets(text, sizeof text, stdin);
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
				//fgets(text, sizeof text, stdin);
				voxel = int(voxel_x + voxel_y * VOXELS_X + voxel_z * VOXELS_X * VOXELS_Y);
				outside_image = ( voxel_x >= VOXELS_X ) || ( voxel_y >= VOXELS_Y ) || ( voxel_z >= VOXELS_Z );
				if( !outside_image )
					atomicAdd( &space_carve_object[voxel], 1 );
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
				voxel = int(voxel_x + voxel_y * VOXELS_X + voxel_z * VOXELS_X * VOXELS_Y);
				outside_image = ( voxel_x >= VOXELS_X ) || ( voxel_y >= VOXELS_Y ) || ( voxel_z >= VOXELS_Z );
				if( !outside_image )
					atomicAdd( &space_carve_object[voxel], 1 );
				end_walk = ( voxel == voxel_out ) || outside_image;
				//fgets(text, sizeof text, stdin);
			}// end: while( !end_walk )
		}//end: else: z_entry[i] != z_exit[i] => z_entry[i] == z_exit[i]
	}// end: if( (i < num_histories) && traversed_recon_volume[i] && (WEPL[i] <= PURE_SC_THRESH) && (bin_num[i] >= 0) )
}
void space_model( int num_histories)
{
	dim3 dimBlock(THREADS_PER_BLOCK);
	dim3 dimGrid((int)(num_histories/THREADS_PER_BLOCK)+1);
	space_model_kernel<<<dimGrid, dimBlock>>>
	(
		num_histories, space_model_object_d, bin_num_d, traversed_recon_volume_d, WEPL_d,
		x_entry_d, y_entry_d, z_entry_d, x_exit_d, y_exit_d, z_exit_d
	);
}
__global__ void space_model_kernel
( 
	int num_histories, int* space_model_object, int* bin_num, bool* traversed_recon_volume, float* WEPL,
	float* x_entry, float* y_entry, float* z_entry, float* x_exit, float* y_exit, float* z_exit
)
{
	int i = threadIdx.x + blockIdx.x * THREADS_PER_BLOCK;
	if( (i < num_histories) && traversed_recon_volume[i] && (WEPL[i] >= SPACE_MODEL_LOWER_THRESHOLD) && (WEPL[i] <= SPACE_MODEL_UPPER_THRESHOLD) && (bin_num[i] >= 0) )
	{
		//char text[20];
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
		x_inside = modf( ( x_entry[i] + RECON_CYL_RADIUS ) /VOXEL_WIDTH, &voxel_x)*VOXEL_WIDTH;	
		y_inside = modf( ( RECON_CYL_RADIUS - y_entry[i] ) /VOXEL_HEIGHT, &voxel_y)*VOXEL_HEIGHT;
		z_inside = modf( ( RECON_CYL_HEIGHT/2 - z_entry[i] ) /VOXEL_THICKNESS, &voxel_z)*VOXEL_THICKNESS;

		voxel = int(voxel_x + voxel_y * VOXELS_X + voxel_z * VOXELS_X * VOXELS_Y);
		voxel_x_out = int( ( x_exit[i] + RECON_CYL_RADIUS ) /VOXEL_WIDTH );
		voxel_y_out = int( ( RECON_CYL_RADIUS - y_exit[i] ) /VOXEL_HEIGHT );
		voxel_z_out = int( ( RECON_CYL_HEIGHT/2 - z_exit[i] ) /VOXEL_THICKNESS );
		voxel_out = int(voxel_x_out + voxel_y_out * VOXELS_X + voxel_z_out * VOXELS_X * VOXELS_Y);
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
			
		outside_image = ( voxel_x >= VOXELS_X ) || ( voxel_y >= VOXELS_Y ) || ( voxel_z >= VOXELS_Z );
		if( !outside_image )
			atomicAdd( &space_model_object[voxel], 1 );
		end_walk = ( voxel == voxel_out ) || outside_image;
		//fgets(text, sizeof text, stdin);
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
				//fgets(text, sizeof text, stdin);
				voxel = int(voxel_x + voxel_y * VOXELS_X + voxel_z * VOXELS_X * VOXELS_Y);
				outside_image = ( voxel_x >= VOXELS_X ) || ( voxel_y >= VOXELS_Y ) || ( voxel_z >= VOXELS_Z );
				if( !outside_image )
					atomicAdd( &space_model_object[voxel], 1 );
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
				voxel = int(voxel_x + voxel_y * VOXELS_X + voxel_z * VOXELS_X * VOXELS_Y);
				outside_image = ( voxel_x >= VOXELS_X ) || ( voxel_y >= VOXELS_Y ) || ( voxel_z >= VOXELS_Z );
				if( !outside_image )
					atomicAdd( &space_model_object[voxel], 1 );
				end_walk = ( voxel == voxel_out ) || outside_image;
				//fgets(text, sizeof text, stdin);
			}// end: while( !end_walk )
		}//end: else: z_entry[i] != z_exit[i] => z_entry[i] == z_exit[i]
	}// end: if( (i < num_histories) && traversed_recon_volume[i] && (WEPL[i] >= SPACE_MODEL_LOWER_THRESHOLD) && (WEPL[i] <= SPACE_MODEL_UPPER_THRESHOLD) && (bin_num[i] >= 0) )
}
void space_carve_threshold()
{
	cudaMemcpy(space_carve_object_h,  space_carve_object_d,	 IMAGE_INT_MEM_SIZE, cudaMemcpyDeviceToHost);
	write_integer_array_to_files("space_carve", output_dir, space_carve_object_h, VOXELS_X, VOXELS_Y, VOXELS_Z );

	dim3 dimBlock( VOXELS_Z );
	dim3 dimGrid( VOXELS_X, VOXELS_Y );   

	space_carve_threshold_kernel<<< dimGrid, dimBlock >>>( space_carve_object_d );

	cudaMemcpy(space_carve_object_h,  space_carve_object_d,	 IMAGE_INT_MEM_SIZE, cudaMemcpyDeviceToHost);
	write_integer_array_to_files("space_carve_thresholded", output_dir, space_carve_object_h, VOXELS_X, VOXELS_Y, VOXELS_Z );
	//printf("JHello\n");
	write_integer_array_to_file("x_sc", output_dir, space_carve_object_h, VOXELS_X, VOXELS_Y, VOXELS_Z );
	//printf("JHello\n");
	cudaFree( space_carve_object_d );
	//printf("JHello\n");
	free(space_carve_object_h);
	//printf("JHello\n");
}
__global__ void space_carve_threshold_kernel( int* space_carve_object )
{
	int row = blockIdx.y, column = blockIdx.x, slice = threadIdx.x;
	int voxel = column + row * VOXELS_X + slice * VOXELS_X * VOXELS_Y;
	float x = ( column - VOXELS_X/2 + 0.5 ) * VOXEL_WIDTH;
	float y = ( VOXELS_Y/2 - row - 0.5 ) * VOXEL_HEIGHT;
	int difference, max_difference = 0;
	if( (row != 0) && (row != VOXELS_Y - 1) && (column != 0) && (column != VOXELS_X - 1) )
	{		
		for( int current_row = row - 1; current_row <= row + 1; current_row++ )
		{
			for( int current_column = column - 1; current_column <= column + 1; current_column++ )
			{
				difference = space_carve_object[voxel] - space_carve_object[current_column + current_row * VOXELS_X + slice * VOXELS_X * VOXELS_Y];
				if( difference > max_difference )
					max_difference = difference;
			}
		}
	}
	syncthreads();
	if( max_difference > SPACE_CARVE_DIFF_THRESH )
		space_carve_object[voxel] = 0;
	else if( space_carve_object[voxel] == 0 )
		space_carve_object[voxel] = 0;
	else
		space_carve_object[voxel] = 1;
	if( x * x + y * y > RECON_CYL_RADIUS * RECON_CYL_RADIUS )
		space_carve_object[voxel] = 0;

}
void space_model_threshold()
{
	printf("JHello\n");
	// Copy the space modeled image from the GPU to the CPU and write it to file.
	cudaMemcpy(space_model_object_h,  space_model_object_d,	 IMAGE_INT_MEM_SIZE,   cudaMemcpyDeviceToHost);
	write_integer_array_to_files("space_model", output_dir, space_model_object_h, VOXELS_X, VOXELS_Y, VOXELS_Z );
	printf("JHello\n");
	int* model_differenes_h = (int*) calloc( VOXELS, sizeof(int) );
	int* model_differenes_d;
	cudaMalloc((void**) &model_differenes_d, IMAGE_INT_MEM_SIZE );
	cudaMemcpy( model_differenes_d, model_differenes_h, IMAGE_INT_MEM_SIZE, cudaMemcpyHostToDevice );

	dim3 dimBlock( VOXELS_Z );
	dim3 dimGrid( VOXELS_X, VOXELS_Y );   

	carve_differences<<< dimGrid, dimBlock >>>( model_differenes_d, space_model_object_d );
	cudaMemcpy( model_differenes_h,  model_differenes_d, IMAGE_INT_MEM_SIZE, cudaMemcpyDeviceToHost );
	int* space_model_thresholds_h = (int*) calloc( VOXELS_Z, sizeof(int) );
	int voxel;	
	int max_difference = 0;
	for( int slice = 0; slice < VOXELS_Z; slice++ )
	{
		for( int pixel = 0; pixel < VOXELS_X * VOXELS_Y; pixel++ )
		{
			voxel = pixel + slice * VOXELS_X * VOXELS_Y;
			if( model_differenes_h[voxel] > max_difference )
			{
				max_difference = model_differenes_h[voxel];
				space_model_thresholds_h[slice] = space_model_object_h[voxel];
			}
		}
		printf( "Slice %d : The maximum space_model difference = %d and the space_model threshold = %d\n", slice, max_difference, space_model_thresholds_h[slice] );
		max_difference = 0;
	}
	//for( int slice = 0; slice < VOXELS_Z; slice++ )
		//printf( "The maximum space_model difference = %d and the space_model threshold = %d\n", max_difference, threshold[slice] );
	int* space_model_thresholds_d;
	unsigned int threshold_size = VOXELS_Z * sizeof(int);
	cudaMalloc((void**) &space_model_thresholds_d, threshold_size );
	cudaMemcpy( space_model_thresholds_d, space_model_thresholds_h, threshold_size, cudaMemcpyHostToDevice );

	space_model_threshold_kernel<<< dimGrid, dimBlock >>>( space_model_object_d, space_model_thresholds_d);
	
	cudaMemcpy(space_model_object_h,  space_model_object_d,	 IMAGE_INT_MEM_SIZE,   cudaMemcpyDeviceToHost);
	write_integer_array_to_files("space_model_thresholded", output_dir, space_model_object_h, VOXELS_X, VOXELS_Y, VOXELS_Z );
	write_integer_array_to_file("x_sm", output_dir, space_model_object_h, VOXELS_X, VOXELS_Y, VOXELS_Z );

	cudaFree( space_model_object_d );
	free(space_model_object_h);
}
__global__ void space_model_threshold_kernel( int* space_model_object, int* space_model_threshold )
{
	int row = blockIdx.y, column = blockIdx.x, slice = threadIdx.x;
	float x = ( column - VOXELS_X/2 + 0.5 ) * VOXEL_WIDTH;
	float y = ( VOXELS_Y/2 - row - 0.5 ) * VOXEL_HEIGHT;
	int voxel = column + row * VOXELS_X + slice * VOXELS_X * VOXELS_Y;
	if( voxel < VOXELS )
	{
		//if( space_model_object[voxel] > SPACE_MODEL_INTERSECTIONS_THRESHOLD )
		if( space_model_object[voxel] > 1.0 * space_model_threshold[slice] )
			space_model_object[voxel] = 1;
		else
			space_model_object[voxel] = 0;
		if( x * x + y * y > RECON_CYL_RADIUS * RECON_CYL_RADIUS )
			space_model_object[voxel] = 0;
	}
}
__global__ void carve_differences( int* carve_differences, int* space_carve_object )
{
	int row = blockIdx.y, column = blockIdx.x, slice = threadIdx.x;
	int voxel = column + row * VOXELS_X + slice * VOXELS_X * VOXELS_Y;
	if( (row != 0) && (row != VOXELS_Y - 1) && (column != 0) && (column != VOXELS_X - 1) )
	{
		int difference, max_difference = 0;
		for( int current_row = row - 1; current_row <= row + 1; current_row++ )
		{
			for( int current_column = column - 1; current_column <= column + 1; current_column++ )
			{
				difference = space_carve_object[voxel] - space_carve_object[current_column + current_row * VOXELS_X + slice * VOXELS_X * VOXELS_Y];
				if( difference > max_difference )
					max_difference = difference;
			}
		}
		carve_differences[voxel] = max_difference;
	}
}
void calculate_means()
{
	dim3 dimBlock( T_BINS );
	dim3 dimGrid( V_BINS, ANGULAR_BINS );   
	calculate_means_kernel<<< dimGrid, dimBlock >>>
	( 
		bin_counts_d, mean_WEPL_d, mean_rel_ut_angle_d, mean_rel_uv_angle_d
	);

	//cudaMemcpy( bin_counts_h,	bin_counts_d,	mem_size_bins_ints, cudaMemcpyDeviceToHost );
	//cudaMemcpy( mean_WEPL_h,	mean_WEPL_d,	mem_size_bins_floats, cudaMemcpyDeviceToHost );
	//cudaMemcpy( mean_rel_ut_angle_h,	mean_rel_ut_angle_d,	mem_size_bins_floats, cudaMemcpyDeviceToHost );
	//cudaMemcpy( mean_rel_uv_angle_h,	mean_rel_uv_angle_d,	mem_size_bins_floats, cudaMemcpyDeviceToHost );

	//write_integer_array_to_files("bin_counts_h", output_dir, bin_counts_h, T_BINS, ANGULAR_BINS, V_BINS );
	//write_float_array_to_files("mean_WEPL_h", output_dir, mean_WEPL_h, T_BINS, ANGULAR_BINS, V_BINS );
	//write_float_array_to_files("mean_rel_ut_angle_h", output_dir, mean_rel_ut_angle_h, T_BINS, ANGULAR_BINS, V_BINS );
	//write_float_array_to_files("mean_rel_uv_angle_h", output_dir, mean_rel_uv_angle_h, T_BINS, ANGULAR_BINS, V_BINS );
	
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
void sum_differences( int start_position, int num_histories )
{
	unsigned int mem_size_hist_floats = sizeof(float) * num_histories;
	unsigned int mem_size_hist_ints = sizeof(int) * num_histories;

	cudaMalloc((void**) &bin_num_d,			mem_size_hist_ints);
	cudaMalloc((void**) &WEPL_d,			mem_size_hist_floats);
	cudaMalloc((void**) &xy_entry_angle_d,	mem_size_hist_floats);
	cudaMalloc((void**) &xz_entry_angle_d,	mem_size_hist_floats);
	cudaMalloc((void**) &xy_exit_angle_d,	mem_size_hist_floats);
	cudaMalloc((void**) &xz_exit_angle_d,	mem_size_hist_floats);

	cudaMemcpy( bin_num_d,			&valid_bin_num[start_position],			mem_size_hist_ints, cudaMemcpyHostToDevice);
	cudaMemcpy( WEPL_d,				&valid_WEPL[start_position],			mem_size_hist_floats, cudaMemcpyHostToDevice);
	cudaMemcpy( xy_entry_angle_d,	&valid_xy_entry_angle[start_position],	mem_size_hist_floats, cudaMemcpyHostToDevice);
	cudaMemcpy( xz_entry_angle_d,	&valid_xz_entry_angle[start_position],	mem_size_hist_floats, cudaMemcpyHostToDevice);
	cudaMemcpy( xy_exit_angle_d,	&valid_xy_exit_angle[start_position],	mem_size_hist_floats, cudaMemcpyHostToDevice);
	cudaMemcpy( xz_exit_angle_d,	&valid_xz_exit_angle[start_position],	mem_size_hist_floats, cudaMemcpyHostToDevice);

	dim3 dimBlock(THREADS_PER_BLOCK);
	dim3 dimGrid((int)(num_histories/THREADS_PER_BLOCK)+1);
	sum_differences_kernel<<<dimGrid, dimBlock>>>
	( 
		num_histories, bin_num_d, mean_WEPL_d, mean_rel_ut_angle_d, mean_rel_uv_angle_d, 
		WEPL_d, xy_entry_angle_d, xz_entry_angle_d, xy_exit_angle_d, xz_exit_angle_d, 
		stddev_WEPL_d, stddev_rel_ut_angle_d, stddev_rel_uv_angle_d
	);
	cudaFree( bin_num_d );
	cudaFree( WEPL_d );
	cudaFree( xy_entry_angle_d );
	cudaFree( xz_entry_angle_d );
	cudaFree( xy_exit_angle_d );
	cudaFree( xz_exit_angle_d );
}
__global__ void sum_differences_kernel
( 
	int num_histories, int* bin_num, float* mean_WEPL, float* mean_rel_ut_angle, float* mean_rel_uv_angle,  
	float* WEPL, float* xy_entry_angle, float* xz_entry_angle, float* xy_exit_angle, float* xz_exit_angle,
	float* stddev_WEPL, float* stddev_rel_ut_angle, float* stddev_rel_uv_angle 
)
{
	float WEPL_difference, rel_ut_angle_difference, rel_uv_angle_difference;

	int i = threadIdx.x + blockIdx.x * THREADS_PER_BLOCK;

	if( i < num_histories )
	{
		WEPL_difference = WEPL[i] - mean_WEPL[bin_num[i]];
		rel_ut_angle_difference = xy_entry_angle[i] - xy_exit_angle[i] - mean_rel_ut_angle[bin_num[i]];
		rel_uv_angle_difference = xz_entry_angle[i] - xz_exit_angle[i] - mean_rel_uv_angle[bin_num[i]];

		atomicAdd( &stddev_WEPL[bin_num[i]], WEPL_difference * WEPL_difference );
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
}
__global__ void calculate_std_devs_kernel( int* bin_counts, float* stddev_WEPL, float* stddev_rel_ut_angle, float* stddev_rel_uv_angle )
{
	int v = blockIdx.x, angle = blockIdx.y, t = threadIdx.x;
	int bin = t + angle * T_BINS + v * T_BINS * ANGULAR_BINS;
	if( bin_counts[bin] > 0 )
	{
		stddev_WEPL[bin] = sqrtf( stddev_WEPL[bin] / bin_counts[bin] );		
		stddev_rel_ut_angle[bin] = sqrtf( stddev_rel_ut_angle[bin] / bin_counts[bin] );
		stddev_rel_uv_angle[bin] = sqrtf( stddev_rel_uv_angle[bin] / bin_counts[bin] );
	}
	syncthreads();
	bin_counts[bin] = 0;
}
void statistical_cuts( int start_position, int num_histories )
{
	unsigned int mem_size_hist_floats = sizeof(float) * num_histories;
	unsigned int mem_size_hist_ints = sizeof(int) * num_histories;

	cudaMalloc((void**) &bin_num_d,			mem_size_hist_ints);
	cudaMalloc((void**) &WEPL_d,			mem_size_hist_floats);
	cudaMalloc((void**) &xy_entry_angle_d,	mem_size_hist_floats);
	cudaMalloc((void**) &xz_entry_angle_d,	mem_size_hist_floats);
	cudaMalloc((void**) &xy_exit_angle_d,	mem_size_hist_floats);
	cudaMalloc((void**) &xz_exit_angle_d,	mem_size_hist_floats);

	cudaMemcpy( bin_num_d,			&valid_bin_num[start_position],			mem_size_hist_ints,		cudaMemcpyHostToDevice);
	cudaMemcpy( WEPL_d,				&valid_WEPL[start_position],			mem_size_hist_floats,	cudaMemcpyHostToDevice);
	cudaMemcpy( xy_entry_angle_d,	&valid_xy_entry_angle[start_position],	mem_size_hist_floats, cudaMemcpyHostToDevice);
	cudaMemcpy( xz_entry_angle_d,	&valid_xz_entry_angle[start_position],	mem_size_hist_floats, cudaMemcpyHostToDevice);
	cudaMemcpy( xy_exit_angle_d,	&valid_xy_exit_angle[start_position],	mem_size_hist_floats, cudaMemcpyHostToDevice);
	cudaMemcpy( xz_exit_angle_d,	&valid_xz_exit_angle[start_position],	mem_size_hist_floats, cudaMemcpyHostToDevice);

	dim3 dimBlock(THREADS_PER_BLOCK);
	dim3 dimGrid((int)(num_histories/THREADS_PER_BLOCK)+1);  
	statistical_cuts_kernel<<< dimGrid, dimBlock >>>
	( 
		num_histories, bin_counts_d, bin_num_d, sinogram_d, WEPL_d, 
		xy_entry_angle_d, xz_entry_angle_d, xy_exit_angle_d, xz_exit_angle_d, 
		mean_WEPL_d, mean_rel_ut_angle_d, mean_rel_uv_angle_d, 
		stddev_WEPL_d, stddev_rel_ut_angle_d, stddev_rel_uv_angle_d
	);
	cudaFree( bin_num_d );
	cudaFree( WEPL_d );
	cudaFree( xy_entry_angle_d );
	cudaFree( xz_entry_angle_d );
	cudaFree( xy_exit_angle_d );
	cudaFree( xz_exit_angle_d );
}
__global__ void statistical_cuts_kernel
( 
	int num_histories, int* bin_counts, int* bin_num, float* sinogram, float* WEPL, 
	float* xy_entry_angle, float* xz_entry_angle, float* xy_exit_angle, float* xz_exit_angle, 
	float* mean_WEPL, float* mean_rel_ut_angle, float* mean_rel_uv_angle,
	float* stddev_WEPL, float* stddev_rel_ut_angle, float* stddev_rel_uv_angle
	
)
{
	bool passed_ut_cut, passed_uv_cut, passed_WEPL_cut;
	int i = threadIdx.x + blockIdx.x * THREADS_PER_BLOCK;
	if( i < num_histories )
	{
		if( fabs( xy_entry_angle[i] - xy_exit_angle[i] - mean_rel_ut_angle[bin_num[i]] ) < SIGMAS_TO_KEEP * stddev_rel_ut_angle[bin_num[i]] )
			passed_ut_cut = true;
		if( fabs( xz_entry_angle[i] - xz_exit_angle[i] - mean_rel_uv_angle[bin_num[i]] ) < SIGMAS_TO_KEEP * stddev_rel_uv_angle[bin_num[i]] )
			passed_uv_cut = true;
		if( fabs( mean_WEPL[bin_num[i]] - WEPL[i] ) < SIGMAS_TO_KEEP * stddev_WEPL[bin_num[i]] )
			passed_WEPL_cut = true;
		if( passed_ut_cut && passed_uv_cut && passed_WEPL_cut )
		{
			atomicAdd( &sinogram[bin_num[i]], WEPL[i] );
			atomicAdd( &bin_counts[bin_num[i]], 1 );
		}
	}
}
void construct_sinogram()
{
	dim3 dimBlock( T_BINS );
	dim3 dimGrid( V_BINS, ANGULAR_BINS );   
	construct_sinogram_kernel<<< dimGrid, dimBlock >>>( bin_counts_d, sinogram_d );

	cudaFree(bin_counts_d);
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
	cout << "Doing the filtering\n";		
	sinogram_filtered_h = (float*) calloc( NUM_BINS, sizeof(float) );
	
	cudaMalloc((void**) &sinogram_filtered_d, mem_size_bins_floats);
	cudaMemcpy( sinogram_filtered_d, sinogram_filtered_h, mem_size_bins_floats, cudaMemcpyHostToDevice);

	dim3 dimBlock( T_BINS );
	dim3 dimGrid( V_BINS, ANGULAR_BINS );   	
	filter_kernel<<< dimGrid, dimBlock >>>( sinogram_d, sinogram_filtered_d );

	cudaMemcpy(sinogram_filtered_h, sinogram_filtered_d, mem_size_bins_floats, cudaMemcpyDeviceToHost) ;

	free(sinogram_h);

	cudaFree(sinogram_d);
	cudaFree(sinogram_filtered_d);
}
__global__ void filter_kernel( float* sinogram, float* sinogram_filtered )
{	
	int n_prime,m,n,k,difference;
	float filtered,s,v,scale_factor;
	
	// v detector index
	k = blockIdx.x;
	
	// binned projection index
	m = blockIdx.y;
	
	// t detector index
	n = threadIdx.x;
	
	// Calculate the cone beam scaling factor
	v = ( k - V_BINS/2 ) * V_BIN_SIZE + V_BIN_SIZE/2.0;
	
	// Loop over strips for this strip
	for( n_prime = 0; n_prime < T_BINS; n_prime++ )
	{
		// Calculate the fan beam scaling factor
		s = ( n_prime - T_BINS/2 ) * T_BIN_SIZE + T_BIN_SIZE/2.0;
		
		scale_factor = SOURCE_RADIUS / sqrtf( SOURCE_RADIUS * SOURCE_RADIUS + s * s + v * v );
		
		difference = n - n_prime;
		
		// Ram-Lak
		//if(difference==0){ filtered=1.0/(8.0*d_t*d_t); }
		//else if(difference%2==0){ filtered=0; }
		//else{ filtered=-1.0/(2.0*d_t*d_t*PI*PI*difference*difference); }
		
		// Shepp-Logan filter
		filtered = ( 1.0 / ( PI * PI * T_BIN_SIZE * T_BIN_SIZE ) ) * ( 1.0 / ( 1.0 - ( 4.0 * difference * difference ) ) );
		// not sure why this won't compile without calculating the index ahead of time instead inside []s
		int index1 = ( k * ANGULAR_BINS * T_BINS ) + ( m * T_BINS ) + n;
		int index2 = ( k * ANGULAR_BINS * T_BINS ) + ( m * T_BINS ) + n_prime;
		sinogram_filtered[index1] += T_BIN_SIZE * sinogram[index2] * filtered * scale_factor;
	}
}
void backprojection()
{
	cout << "Doing the backprojection\n";
	printf("DEBUG: IMAGE_FLOAT_MEM_SIZE = %u\n", IMAGE_FLOAT_MEM_SIZE); 

	// Allocate host memory
	printf("DEBUG: Allocate host memory\n");

	X_h = (float*) calloc( VOXELS, sizeof(float) );

	if( X_h == NULL ) 
	{
		printf("ERROR: Memory not allocated for X_h!\n");
		exit(1);
	}
	
	// Check that we don't have any corruptions up until now
	for( int i = 0; i < NUM_BINS; i++ )
		if( sinogram_filtered_h[i] != sinogram_filtered_h[i] )
			printf("We have a nan in bin #%d\n", i);

	float delta = GANTRY_ANGLE_INTERVAL * PI/180.0;
	// Loop over the voxels
	for( int slice = 0; slice < VOXELS_Z; slice++ )
	{
		for( int column = 0; column < VOXELS_X; column++ )
		{

			for( int row = 0; row < VOXELS_Y; row++ )
			{

				// Get the spatial co-ordinates of the pixel
				float x = -RECON_CYL_RADIUS + column * VOXEL_WIDTH + VOXEL_WIDTH * 0.5;
				float y = RECON_CYL_RADIUS - row * VOXEL_HEIGHT - VOXEL_HEIGHT * 0.5;
				float z = -RECON_CYL_HEIGHT * 0.5 + slice * SLICE_THICKNESS + SLICE_THICKNESS * 0.5;

				// If the voxel is outside a cylinder contained in the reconstruction volume, set to air
				if( ( x * x + y * y ) > ( RECON_CYL_RADIUS * RECON_CYL_RADIUS ) )
					X_h[( slice * VOXELS_X * VOXELS_Y) + ( row * VOXELS_X ) + column] = 0.00113;
				else
				{
	  
					// Sum over projection angles
					for( int m = 0; m < ANGULAR_BINS; m++ )
					{
						// Rotate the pixel position to the beam-detector co-ordinate system
						float u = x * cosf( m * delta ) + y * sinf( m * delta );
						float t = -x * sinf( m * delta ) + y * cosf( m * delta );
						float v = z;

						// Project to find the detector number
						float detector_number_t = ( t - u *( t / ( SOURCE_RADIUS + u ) ) ) / T_BIN_SIZE + T_BINS * 0.5;
						int k = int( detector_number_t );
						if( k > detector_number_t )
							k -= 1;
						float eta = detector_number_t - k;

						// Now project v to get detector number in v axis
						float detector_number_v = ( v - u * ( v / ( SOURCE_RADIUS + u ) ) ) / V_BIN_SIZE + V_BINS * 0.5;
						int l = int( detector_number_v );
						if( l > detector_number_v )
							l -= 1;
						float epsilon = detector_number_v - l;

						// Calculate the fan beam scaling factor
						float scale_factor = ( SOURCE_RADIUS / ( SOURCE_RADIUS + u ) ) * ( SOURCE_RADIUS / ( SOURCE_RADIUS + u ) );
		  
						// Compute the back-projection
						int bin = l * ANGULAR_BINS * T_BINS + m * T_BINS + k;
						int voxel = slice * VOXELS_X * VOXELS_Y + row * VOXELS_X + column;
						// not sure why this won't compile without calculating the index ahead of time instead inside []s
						int index = ANGULAR_BINS * T_BINS;

						if( ( ( bin + ANGULAR_BINS * T_BINS + 1 ) >= NUM_BINS ) || ( bin < 0 ) )
							printf("The bin selected for this voxel does not exist!\n Slice: %d\n Column: %d\n Row: %d\n", slice, column, row);
						else 
						{
							// not sure why this won't compile without calculating the index ahead of time instead inside []s
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
	FBP_object_h = (int*) calloc( VOXELS_X * VOXELS_Y * VOXELS_Z, sizeof(int) );

	for( int k = 0; k < VOXELS_Z; k++ )
	{
		for( int m = 0; m < VOXELS_Y; m++ )
		{
			for( int n = 0; n < VOXELS_X; n++ )
			{
				if(X_h[( k * VOXELS_X * VOXELS_Y ) + ( m * VOXELS_X ) + n] < FBP_THRESHOLD )
					FBP_object_h[( k * VOXELS_X * VOXELS_Y ) + ( m * VOXELS_X ) + n] = 0; 
				else
					FBP_object_h[( k * VOXELS_X * VOXELS_Y ) + ( m * VOXELS_X ) + n] = 1; 
			}

		}
	}

	write_integer_array_to_files( "FBP_object", output_dir, FBP_object_h, VOXELS_X, VOXELS_Y, VOXELS_Z );
	write_integer_array_to_file( "x_FBP", output_dir, FBP_object_h, VOXELS_X, VOXELS_Y, VOXELS_Z );
}