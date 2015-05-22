//#ifndef PCT_RECONSTRUCTION_H
//#define PCT_RECONSTRUCTION_H
#pragma once
/***************************************************************************************************************************************************************************/
/********************************************************************* Header for pCT reconstruction program ***************************************************************/
/***************************************************************************************************************************************************************************/
//#include <Windows4Root.h>
//#include "w32pragma.h"
//#include <TROOT.h>
//#include <TMath.h>
//#include <C:\Users\Blake\Documents\GitHub\pCT_Reconstruction\Configurations.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>

//#include <boost/lambda/lambda.hpp>
//#include <boost/filesystem/operations.hpp>
//#include <boost/filesystem/path.hpp>

#include <algorithm>    // std::transform
#include <array>
#include <bitset>
#include <cmath>
#include <cstdarg>		// va_list, va_arg, va_start, va_end, va_copy
#include <cstdio>		// printf, sprintf,  
#include <cstdlib>		// rand, srand
#include <ctime>		// clock(), time()
#include <fcntl.h>
#include <fstream>
#include <functional>	// std::multiplies, std::plus, std::function, std::negate
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <map>
#include <new>			 
#include <numeric>		// inner_product, partial_sum, adjacent_difference, accumulate
//#include <omp.h>		// OpenMP
#include <process.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include "sys/types.h"	// stat f
#include "sys/stat.h"	// stat functions
#include <tuple>
#include <typeinfo>		//operator typeid
#include <type_traits>	// is_pod
//#include <unistd.h>
#include <utility>		// for std::move
#include <vector>
#if defined(_WIN32) || defined(_WIN64)
	#include <windows.h>
	#include "Shlwapi.h"	// Requires: Shlwapi.lib, Shlwapi.h, Shlwapi.dll (version 4.71 or later)
#endif
/***************************************************************************************************************************************************************************/
/************************************************************** Global typedef and namespace usage definitions *************************************************************/
/***************************************************************************************************************************************************************************/
//using namespace std::placeholders; 
//using namespace std;
using std::cout;
using std::endl;
typedef unsigned long long ULL;
typedef unsigned int uint;
typedef std::vector<std::vector<std::string> > LOG_OBJECT;
typedef std::vector<std::string> LOG_LINE;
//typedef std::vector<std::vector<char*> > log_object;
typedef std::vector<std::vector<std::string> > CONFIG_OBJECT;
typedef std::vector<std::string> CONFIG_LINE;
/***************************************************************************************************************************************************************************/
/************************************************************************ User defined data types **********************************************************************/
/***************************************************************************************************************************************************************************/
struct generic_IO_container
{	
	unsigned int input_type_ID;
	bool boolean_input;						// type_ID = 1
	int integer_input;						// type_ID = 2
	double double_input;					// type_ID = 3
	char* string_input;						// type_ID = 4
	char* key;
};

enum DATA_FORMATS { OLD_FORMAT, VERSION_0, VERSION_1 };						// Define the data formats that are supported
enum DISK_WRITE_MODE { TEXT, BINARY };										// Experimental or simulated data
enum BIN_ANALYSIS_TYPE { MEANS, COUNTS, MEMBERS };							// Choices for what information about the binned data is desired 
enum BIN_ANALYSIS_FOR { ALL_BINS, SPECIFIC_BINS };							// Choices for which bins the desired data should come from
enum BIN_ORGANIZATION { BY_BIN, BY_HISTORY };								// Binned data is either organized in order by bin or by history w/ bin # specified separately
enum BIN_ANALYSIS_OF { WEPLS, ANGLES, POSITIONS, BIN_NUMS };				// Choices for which type of binned data is desired
enum IMAGE_DEFINED_BY { SIZE_VOXELS, DIMENSIONS_VOXELS, SIZE_DIMENSIONS};	// Image size defined by 2 of voxel dimenensions, image dimensions, and image discretization

enum SCAN_TYPES { EXPERIMENTAL, SIMULATED_G, SIMULATED_T, ST_END };			// Experimental or simulated data
enum HULL_TYPES {IMPORT_HULL, SC_HULL, MSC_HULL, SM_HULL, FBP_HULL, HT_END};// Define valid choices for which hull to use in MLP calculations
enum FILTER_TYPES {RAM_LAK, SHEPP_LOGAN, NONE, FT_END};						// Define the types of filters that are available for use in FBP
enum X_0_TYPES { IMPORT_X_0, X_HULL, X_FBP, HYBRID, ZEROS, X0_END };		// Define valid choices for which hull to use in MLP calculations
enum RECON_ALGORITHMS { ART, DROP, BIP, SAP, ROBUST1, ROBUST2, RA_END };	// Define valid choices for iterative projection algorithm to use
enum LOG_ENTRIES {OBJECT_L, SCAN_TYPE_L, RUN_DATE_L, RUN_NUMBER_L, 
	ACQUIRED_BY_L, PROJECTION_DATA_DATE_L, CALIBRATED_BY_L, 
	PREPROCESS_DATE_L, PREPROCESSED_BY_L, RECONSTRUCTION_DATE_L, 
	RECONSTRUCTED_BY_L, COMMENTS_L};
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*--------------------------------------------------------------------- Preprocessing option parameters -------------------------------------------------------------------*/
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
char* OBJECT, *SCAN_TYPE, *RUN_DATE, *RUN_NUMBER, *PROJECTION_DATA_DATE, *PREPROCESS_DATE, *RECONSTRUCTION_DATE;
char* USER_NAME;
char* PATH_2_PCT_DATA_DIR, *DATA_TYPE_DIR, *PROJECTION_DATA_DIR, *PREPROCESSING_DIR, *RECONSTRUCTION_DIR;
//char* OVERWRITING_PREPROCESS_DATE, OVERWRITING_RECON_DATE;
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*--------------------------------------------------------------- IO file extension character array variables -------------------------------------------------------------*/
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
char PROJECTION_DATA_FILE_EXTENSION[4], RADIOGRAPHS_FILE_EXTENSION[4], WEPL_DISTS_FILE_EXTENSION[4];
char HULL_FILE_EXTENSION[4], FBP_FILE_EXTENSION[4], FBP_MEDIANS_FILE_EXTENSION[4], X_0_FILE_EXTENSION[4], X_FILE_EXTENSION[4];	
char MLP_FILE_EXTENSION[4], VOXELS_PER_PATH_FILE_EXTENSION[4], WEPL_FILE_EXTENSION[4], HISTORIES_FILE_EXTENSION[4];
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*------------------------------------------------------------------ IO filename character array variables ----------------------------------------------------------------*/
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
char* HULL_FILENAME, *FBP_FILENAME, *X_0_FILENAME, *X_K_FILENAME;
char* MLP_FILENAME, *VOXELS_PER_PATH_FILENAME, *WEPL_FILENAME, *HISTORIES_FILENAME;
char* HULL_MEDIAN_2D_FILENAME, *FBP_MEDIAN_2D_FILENAME, *X_0_MEDIAN_2D_FILENAME, *X_K_MEDIAN_2D_FILENAME;
char* HULL_MEDIAN_3D_FILENAME, *FBP_MEDIAN_3D_FILENAME, *X_0_MEDIAN_3D_FILENAME, *X_K_MEDIAN_3D_FILENAME;
char* HULL_AVG_2D_FILENAME, *FBP_AVG_2D_FILENAME, *X_0_AVG_2D_FILENAME, *X_K_AVG_2D_FILENAME;
char* HULL_AVG_3D_FILENAME, *FBP_AVG_3D_FILENAME, *X_0_AVG_3D_FILENAME, *X_K_AVG_3D_FILENAME;
char* HULL_COMBO_2D_FILENAME, *FBP_COMBO_2D_FILENAME, *X_0_COMBO_2D_FILENAME, *X_K_COMBO_2D_FILENAME;
char* HULL_COMBO_3D_FILENAME, *FBP_COMBO_3D_FILENAME, *X_0_COMBO_3D_FILENAME, *X_K_COMBO_3D_FILENAME;
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*------------------------------------------------------------------ Full IO path character array variables ---------------------------------------------------------------*/
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
char* CONFIG_PATH, *CONFIG_OUT_PATH, *LOG_PATH, *STDOUT_PATH, *STDERR_PATH;
char* MLP_PATH, *VOXELS_PER_PATH_PATH, *WEPL_PATH, *HISTORIES_PATH;
char* HULL_PATH, *FBP_PATH, *X_0_PATH, *X_K_PATH;
char* HULL_MEDIAN_2D_PATH, *FBP_MEDIAN_2D_PATH, *X_0_MEDIAN_2D_PATH, *X_K_MEDIAN_2D_PATH;
char* HULL_MEDIAN_3D_PATH, *FBP_MEDIAN_3D_PATH, *X_0_MEDIAN_3D_PATH, *X_K_MEDIAN_3D_PATH;
char* HULL_AVG_2D_PATH, *FBP_AVG_2D_PATH, *X_0_AVG_2D_PATH, *X_K_AVG_2D_PATH;
char* HULL_AVG_3D_PATH, *FBP_AVG_3D_PATH, *X_0_AVG_3D_PATH, *X_K_AVG_3D_PATH;
char* HULL_COMBO_2D_PATH, *FBP_COMBO_2D_PATH, *X_0_COMBO_2D_PATH, *X_K_COMBO_2D_PATH;
char* HULL_COMBO_3D_PATH, *FBP_COMBO_3D_PATH, *X_0_COMBO_3D_PATH, *X_K_COMBO_3D_PATH;
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------- Selected images filename/path character array variables ------------------------------------------------------*/
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
char* X_FILENAME_BASE, *X_MEDIAN_2D_FILENAME_BASE, *X_MEDIAN_3D_FILENAME_BASE, *X_AVG_2D_FILENAME_BASE, *X_AVG_3D_FILENAME_BASE, *X_COMBO_2D_FILENAME_BASE, *X_COMBO_3D_FILENAME_BASE;
char* X_PATH_BASE, *X_MEDIAN_2D_PATH_BASE, *X_MEDIAN_3D_PATH_BASE, *X_AVG_2D_PATH_BASE, *X_AVG_3D_PATH_BASE, *X_COMBO_2D_PATH_BASE, *X_COMBO_3D_PATH_BASE;
char* HULL_2_USE_FILENAME, *FBP_2_USE_FILENAME, *X_0_2_USE_FILENAME, *X_K_2_USE_FILENAME, *X_2_USE_FILENAME_BASE;
char* HULL_2_USE_PATH, *FBP_2_USE_PATH, *X_0_2_USE_PATH, *X_K_2_USE_PATH, *X_2_USE_PATH_BASE;
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*--------------------------------------------------------------------- Preprocessing option parameters -------------------------------------------------------------------*/
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
bool OBJECT_SET = false, RUN_DATE_SET = false, RUN_NUMBER_SET = false, USER_NAME_SET = false;
bool PROJECTION_DATA_DATE_SET = false, PREPROCESS_DATE_SET = false, RECONSTRUCTION_DATE_SET = false;
bool PROJECTION_DATA_DIR_SET = false, PREPROCESSING_DIR_SET = false, RECONSTRUCTION_DIR_SET = false;
bool PATH_2_PCT_DATA_DIR_SET, PROJECTION_DATA_DIR_CONSTRUCTABLE, PREPROCESSING_DIR_CONSTRUCTABLE, RECONSTRUCTION_DIR_CONSTRUCTABLE;
/***************************************************************************************************************************************************************************/
/******************************************************************************* Constants *********************************************************************************/
/***************************************************************************************************************************************************************************/
const bool SAMPLE_STD_DEV		= true;									// Use sample/population standard deviation (T/F) in statistical cuts (i.e. divisor is N/N-1)
unsigned int MAGIC_NUMBER_CHECK = int('DTCP');
#define STRIDE					5
#define SIGMAS_2_KEEP			3										// [#] Number of standard deviations from mean to allow before cutting the history 
#define THREADS_PER_BLOCK		1024									// [#] Number of threads assigned to each block on the GPU
#define BYTES_PER_HISTORY		48										// [bytes] Data size of each history, 44 for actual data and 4 empty bytes, for old data format
#define SOURCE_RADIUS			260.7									// [cm] Distance  to source/scatterer 
#define FBP_THRESHOLD			0.6										// [cm] RSP threshold used to generate FBP_hull from FBP
#define MAX_INTERSECTIONS		1000									// Limit on the # of intersections expected for proton's MLP; = # voxels along image diagonal
/***************************************************************************************************************************************************************************/
/************************************************************************* For Use In Development **************************************************************************/
/***************************************************************************************************************************************************************************/
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*-------------------------------------------------------------------- Execution and early exit options -------------------------------------------------------------------*/
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/***************************************************************************************************************************************************************************/
/************************************************************************ End of Parameter Definitions *********************************************************************/
/***************************************************************************************************************************************************************************/
char user_response[20];
bool print_scan_type( uint n)
{
	switch(n)
	{
		case 0:		puts("set to EXPERIMENTAL");			break;
		case 1:		puts("set to SIMULATED_G");				break;
		case 2:		puts("set to SIMULATED_T");				break;
		default:	puts("ERROR: Invalid DATA_TYPE selection.\n\n=>Correct the configuration file and rerun the program.\n");
					return true;
	}
	return false;
}
bool print_hull_type( uint n)
{
	switch(n)
	{
		case 0:		puts("set to IMPORT_HULL");				break;
		case 1:		puts("set to SC_HULL");					break;
		case 2:		puts("set to MSC_HULL");				break;
		case 3:		puts("set to SM_HULL");					break;
		case 4:		puts("set to FBP_HULL");				break;
		default:	puts("ERROR: Invalid HULL_TYPE selection.\n\n=>Correct the configuration file and rerun the program.\n");
					return true;
	}
	return false;
}
bool print_filter_type( uint n)
{
	switch(n)
	{
		case 0:		puts("set to RAM_LAK");					break;
		case 1:		puts("set to SHEPP_LOGAN");				break;
		case 2:		puts("set to NONE");					break;
		default:	puts("ERROR: Invalid FILTER_TYPE selection.\n\n=>Correct the configuration file and rerun the program.\n");
					return true;
	}
	return false;
}
bool print_x_0_type( uint n)
{
	switch(n)
	{
		case 0:		puts("set to IMPORT_X_0");				break;
		case 1:		puts("set to X_HULL");					break;
		case 2:		puts("set to X_FBP");					break;
		case 3:		puts("set to HYBRID");					break;
		case 4:		puts("set to ZEROS");					break;
		default:	puts("ERROR: Invalid X_0_TYPE selection.\n\n=>Correct the configuration file and rerun the program.\n");
					return true;
	}
	return false;
}
bool print_recon_algorithm( uint n)
{
	switch(n)
	{
		case 0:		puts("set to ART");					break;
		case 1:		puts("set to DROP");				break;
		case 2:		puts("set to BIP");					break;
		case 3:		puts("set to SAP");					break;
		case 4:		puts("set to ROBUST1");				break;
		case 5:		puts("set to ROBUST2");				break;
		default:	puts("ERROR: Invalid RECON_ALGORITHM selection.\n\n=>Correct the configuration file and rerun the program.\n");
					return true;
	}
	return false;
}
//#endif // _PCT_RECONSTRUCTION_H_