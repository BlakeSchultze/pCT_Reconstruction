#pragma once

//#include <C:\Users\Blake\Documents\GitHub\pCT_Reconstruction\pCT_Reconstruction.h>

/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------- Output filenames ----------------------------------------------------------------------------*/
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
char PCT_DATA_DIR_NAME[]		= "pCT_Data";
char EXPERIMENTAL_DIR_NAME[]	= "Experimental";
char SIMULATIONS_DIR_NAME[]		= "Simulated";
char RUN_DIR_BASENAME[]			= "Run";
char RAW_DATA_DIR_NAME[]		= "Input";
char PROJECTION_DATA_DIR_NAME[]	= "Output";
char RECONSTRUCTION_DIR_NAME[]	= "Reconstruction";
char PCT_IMAGES_DIR_NAME[]		= "Images";
char REF_IMAGES_DIR_NAME[]		= "Reference_Images";

char CONFIG_FILENAME[]			= "export_testing.cfg";
char INPUT_DATA_BASENAME[]		= "projection";							// Prefix of the input data set BASENAME
char FILE_EXTENSION[]			= ".bin";								// File extension for the input data
char SC_HULL_BASENAME[]			= "hull_SC";
char MSC_HULL_BASENAME[]		= "hull_MSC";
char SM_HULL_BASENAME[]			= "hull_SM";
char FBP_HULL_BASENAME[]		= "hull_FBP";
char HULL_BASENAME[]			= "hull";
char FBP_BASENAME[]				= "FBP";
char X_0_BASENAME[]				= "x_0";
char MLP_BASENAME[]				= "MLP";
char RECON_HISTORIES_BASENAME[]	= "histories";
char FBP_MEDIAN_BASENAME[]		= "FBPmed";
char FBP_AVERAGE_BASENAME[]		= "FBP_avg";
char X_BASENAME[]				= "x";

char HULL_FILE_EXTENSION[]		= ".txt";
char FBP_FILE_EXTENSION[]		= ".txt";
char X_0_FILE_EXTENSION[]		= ".txt";
char MLP_FILE_EXTENSION[]		= ".bin";
char HISTORIES_FILE_EXTENSION[] = ".bin";
char X_FILE_EXTENSION[]			= ".txt";
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------- MLP Parameters ----------------------------------------------------------------------------*/
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
#define E_0						13.6									// [MeV/c] empirical constant
#define X0						36.08									// [cm] radiation length
#define RSP_AIR					0.00113									// [cm/cm] Approximate RSP of air
// Coefficients of 5th order polynomial fit to the term [1 / ( beta^2(u)*p^2(u) )] present in scattering covariance matrices Sigma 1/2 for:
#define BEAM_ENERGY				200										// 200 MeV protons
#define A_0						7.457  * pow( 10, -6.0  )
#define A_1						4.548  * pow( 10, -7.0  )
#define A_2						-5.777 * pow( 10, -8.0  )
#define A_3						1.301  * pow( 10, -8.0  )
#define A_4						-9.228 * pow( 10, -10.0 )
#define A_5						2.687  * pow( 10, -11.0 )

// Common fractional values of A_i coefficients appearing in polynomial expansions of MLP calculations, precalculating saves time
#define A_0_OVER_2				A_0/2 
#define A_0_OVER_3				A_0/3
#define A_1_OVER_2				A_1/2
#define A_1_OVER_3				A_1/3
#define A_1_OVER_4				A_1/4
#define A_1_OVER_6				A_1/6
#define A_1_OVER_12				A_1/12
#define A_2_OVER_3				A_2/3
#define A_2_OVER_4				A_2/4
#define A_2_OVER_5				A_2/5
#define A_2_OVER_12				A_2/12
#define A_2_OVER_30				A_2/30
#define A_3_OVER_4				A_3/4
#define A_3_OVER_5				A_3/5
#define A_3_OVER_6				A_3/6
#define A_3_OVER_20				A_3/20
#define A_3_OVER_60				A_3/60
#define A_4_OVER_5				A_4/5
#define A_4_OVER_6				A_4/6
#define A_4_OVER_7				A_4/7
#define A_4_OVER_30				A_4/30
#define A_4_OVER_105			A_4/105
#define A_5_OVER_6				A_5/6
#define A_5_OVER_7				A_5/7
#define A_5_OVER_8				A_5/8
#define A_5_OVER_42				A_5/42
#define A_5_OVER_168			A_5/168
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------ Precalculated Constants ------------------------------------------------------------------------*/
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
#define PHI						((1 + sqrt(5.0) ) / 2)					// Positive golden ratio, positive solution of PHI^2-PHI-1 = 0; also PHI = a/b when a/b = (a + b) / a 
#define PHI_NEGATIVE			((1 - sqrt(5.0) ) / 2)					// Negative golden ratio, negative solution of PHI^2-PHI-1 = 0; 
#define PI_OVER_4				( atan( 1.0 ) )							// 1*pi/4 radians =   pi/4 radians = 45 degrees
#define PI_OVER_2				( 2 * atan( 1.0 ) )						// 2*pi/4 radians =   pi/2 radians = 90 degrees
#define THREE_PI_OVER_4			( 3 * atan( 1.0 ) )						// 3*pi/4 radians = 3*pi/4 radians = 135 degrees
#define PI						( 4 * atan( 1.0 ) )						// 4*pi/4 radians =   pi   radians = 180 degrees
#define FIVE_PI_OVER_4			( 5 * atan( 1.0 ) )						// 5*pi/4 radians = 5*pi/4 radians = 225 degrees
#define SIX_PI_OVER_4			( 5 * atan( 1.0 ) )						// 6*pi/4 radians = 3*pi/2 radians = 270 degrees
#define SEVEN_PI_OVER_4			( 7 * atan( 1.0 ) )						// 7*pi/4 radians = 7*pi/4 radians = 315 degrees
#define TWO_PI					( 8 * atan( 1.0 ) )						// 8*pi/4 radians = 2*pi   radians = 360 degrees = 0 degrees
#define ANGLE_TO_RADIANS		( PI/180.0 )							// Convertion from angle to radians
#define RADIANS_TO_ANGLE		( 180.0/PI )							// Convertion from radians to angle
#define ROOT_TWO				sqrtf(2.0)								// 2^(1/2) = Square root of 2 
#define MM_TO_CM				0.1										// 10 [mm] = 1 [cm] => 1 [mm] = 0.1 [cm]
#define START					true									// Used as an alias for true for starting timer
#define STOP					false									// Used as an alias for false for stopping timer
#define RIGHT					1										// Specifies that moving right corresponds with an increase in x position, used in voxel walk 
#define LEFT					-1										// Specifies that moving left corresponds with a decrease in x position, used in voxel walk 
#define UP						1										// Specifies that moving up corresponds with an increase in y/z position, used in voxel walk 
#define DOWN					-1										// Specifies that moving down corresponds with a decrease in y/z position, used in voxel walk 
#define X_INCREASING_DIRECTION	RIGHT									// [#] specifies direction (LEFT/RIGHT) along x-axis in which voxel #s increase
#define Y_INCREASING_DIRECTION	DOWN									// [#] specifies direction (UP/DOWN) along y-axis in which voxel #s increase
#define Z_INCREASING_DIRECTION	DOWN									// [#] specifies direction (UP/DOWN) along z-axis in which voxel #s increase
#define BOOL_FORMAT				"%d"									// Specifies format to use for writing/printing boolean data using {s/sn/f/v/vn}printf statements
#define CHAR_FORMAT				"%c"									// Specifies format to use for writing/printing character data using {s/sn/f/v/vn}printf statements
#define INT_FORMAT				"%d"									// Specifies format to use for writing/printing integer data using {s/sn/f/v/vn}printf statements
#define FLOAT_FORMAT			"%f"									// Specifies format to use for writing/printing floating point data using {s/sn/f/v/vn}printf statements
#define STRING_FORMAT			"%s"									// Specifies format to use for writing/printing strings data using {s/sn/f/v/vn}printf statements
#define BOOLEAN					1
#define INTEGER					2
#define DOUBLE					3
#define STRING					4
