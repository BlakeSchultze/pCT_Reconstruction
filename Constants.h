#pragma once

//#include <C:\Users\Blake\Documents\GitHub\pCT_Reconstruction\pCT_Reconstruction.h>

//#define BLOCK_SIZE				320000//15000000//3840// # of paths to use for each update: ART = 1, 
#define DROP_BLOCK_SIZE			3200//3840// # of paths to use for each update: ART = 1, 
//#define THREADS_PER_BLOCK			320
#define HISTORIES_PER_BLOCK 			320
#define ENDPOINTS_PER_BLOCK 			1
#define HISTORIES_PER_THREAD 			1
#define ENDPOINTS_PER_THREAD 			1
#define VOXELS_PER_THREAD 			1
const bool WRITE_X_KI			= true;
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------- pCT data format directory names ------------------------------------------------------------------*/
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
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------- pCT data format input file names -----------------------------------------------------------------*/
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
char PROJECTION_DATA_BASENAME[]	= "projection";							// Prefix of the files containing the projection data (tracker/WEPL/gantry angle) used as input to preprocessing
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*--------------------------------------------------------------- Configuration and execution logging file names ----------------------------------------------------------*/
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
char CONFIG_FILENAME[]			= "settings.cfg";						// Name of the file used to control the program options/parameters as key=value pairs
char CONFIG_OUT_FILENAME[]		= "settings_out.cfg";					// Name of the file used to control the program options/parameters as key=value pairs
char LOG_FILENAME[]				= "log.csv";							// Name of the file logging the execution information associated with each data set generated
char STDOUT_FILENAME[]			= "stdout.txt";							// Name of the file where the standard output stream stdout is redirected
char STDIN_FILENAME[]			= "stdin.txt";							// Name of the file where the standard input stream stdin is redirected
char STDERR_FILENAME[]			= "stderr.txt";							// Name of the file where the standard error stream stderr is redirected
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------- Tabulated data file names ---------------------------------------------------------------------*/
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
char SIN_TABLE_FILENAME[]	= "sin_table.bin";							// Prefix of the file containing the tabulated values of sine function for angles [0, 2PI]
char COS_TABLE_FILENAME[]	= "cos_table.bin";							// Prefix of the file containing the tabulated values of cosine function for angles [0, 2PI]
char COEFFICIENT_FILENAME[]	= "coefficient.bin";						// Prefix of the file containing the tabulated values of the scattering coefficient for u_2-u_1/u_1 values in increments of 0.001
char POLY_1_2_FILENAME[]	= "poly_1_2.bin";							// Prefix of the file containing the tabulated values of the polynomial with coefficients {1,2,3,4,5,6} in increments of 0.001
char POLY_2_3_FILENAME[]	= "poly_2_3.bin";							// Prefix of the file containing the tabulated values of the polynomial with coefficients {2,3,4,5,6,7} in increments of 0.001
char POLY_3_4_FILENAME[]	= "poly_3_4.bin";							// Prefix of the file containing the tabulated values of the polynomial with coefficients {3,4,5,6,7,8} in increments of 0.001
char POLY_2_6_FILENAME[]	= "poly_2_6.bin";							// Prefix of the file containing the tabulated values of the polynomial with coefficients {2,6,12,20,30,42} in increments of 0.001
char POLY_3_12_FILENAME[]	= "poly_3_12.bin";							// Prefix of the file containing the tabulated values of the polynomial with coefficients {3,12,30,60,105,168} in increments of 0.001
#define TRIG_TABLE_MIN	-2 * PI											// [degrees] Minimum angle contained in the sin/cos lookup table used for MLP
#define TRIG_TABLE_MAX	4 * PI											// [degrees] Maximum angle contained in the sin/cos lookup table used for MLP
#define TRIG_TABLE_RANGE	(TRIG_TABLE_MAX - TRIG_TABLE_MIN)				// [degrees] Range of angles contained in the sin/cos lookup table used for MLP
#define TRIG_TABLE_STEP		(0.001 * ANGLE_TO_RADIANS)						// [degrees] Step of 1/4 degree for elements of sin/cos lookup table used for MLP
#define TRIG_TABLE_ELEMENTS	static_cast<int>(TRIG_TABLE_RANGE / TRIG_TABLE_STEP + 0.5)			// [#] # of elements contained in the sin/cos lookup table used for MLP
#define DEPTH_TABLE_RANGE	40.0										// [cm] Range of depths u contained in the polynomial lookup tables used for MLP
#define DEPTH_TABLE_STEP		0.00005										// [cm] Step of 1/1000 cm for elements of the polynomial lookup tables used for MLP
#define DEPTH_TABLE_ELEMENTS	static_cast<int>(DEPTH_TABLE_RANGE / DEPTH_TABLE_STEP + 0.5 )			// [#] # of elements contained in the polynomial lookup tables used for MLP
//#define DEPTH_TABLE_SHIFT	static_cast<int>(DEPTH_TABLE_RANGE / DEPTH_TABLE_STEP + 0.5 )			// [#] # of elements contained in the polynomial lookup tables used for MLP

#define POLY_TABLE_RANGE	40.0										// [cm] Range of depths u contained in the polynomial lookup tables used for MLP
#define POLY_TABLE_STEP		0.00005										// [cm] Step of 1/1000 cm for elements of the polynomial lookup tables used for MLP
#define POLY_TABLE_ELEMENTS	static_cast<int>(POLY_TABLE_RANGE / POLY_TABLE_STEP + 0.5 )			// [#] # of elements contained in the polynomial lookup tables used for MLP
//#define POLY_TABLE_SHIFT	static_cast<int>(POLY_TABLE_RANGE / POLY_TABLE_STEP + 0.5 )			// [#] # of elements contained in the polynomial lookup tables used for MLP
//#define INDEX_SHIFT_4_U_1	50											// [#] Difference between the index of consecutive elements of the poly table for u_1 dependent terms
//#define INDEX_SHIFT_4_U_1	0.05 / POLY_TABLE_STEP						// [#] Difference between the index of consecutive elements of the poly table for u_1 dependent terms
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*------------------------------------------------------------------- Preprocessing data IO file base names ---------------------------------------------------------------*/
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
char RADIOGRAPHS_RAW_BASENAME[]	= "radiographs_raw";					// Prefix of the file containing the radiograph images from each projection angle prior to performing cuts 
char RADIOGRAPHS_BASENAME[]		= "radiographs";						// Prefix of the file containing the radiograph images from each projection angle after performing cuts (i.e. rearranged sinogram)
char WEPL_DISTS_RAW_BASENAME[]	= "WEPL_distributions_raw";				// Prefix of the file containing the WEPL distribution images from each projection angle prior to performing cuts 
char WEPL_DISTS_BASENAME[]		= "WEPL_distributions";					// Prefix of the file containing the WEPL distribution images each projection angle after performing cuts 
char SC_HULL_BASENAME[]			= "hull_SC";							// Prefix of the file containing the SC hull image
char MSC_HULL_BASENAME[]		= "hull_MSC";							// Prefix of the file containing the MSC hull image 
char SM_HULL_BASENAME[]			= "hull_SM";							// Prefix of the file containing the SM hull image 
char FBP_HULL_BASENAME[]		= "hull_FBP";							// Prefix of the file containing the FBP hull image 
char HULL_BASENAME[]			= "hull";								// Prefix of the file containing the SC, MSC, SM, or FBP hull image as specified by the settings.cfg file 
char FBP_BASENAME[]				= "FBP";								// Prefix of the file containing the FBP image
//char FBP_MEDIAN_2D_BASENAME[]	= "FBP_median_2D";						// Prefix of the file containing the 2D median filtered FBP image
//char FBP_MEDIAN_3D_BASENAME[]	= "FBP_median_3D";						// Prefix of the file containing the 3D median filtered FBP image
//char FBP_AVERAGE_BASENAME[]		= "FBP_avg";							// Prefix of the file containing the average filtered FBP image
char X_0_BASENAME[]				= "x_0";								// Prefix of the file containing the FBP, hull, or FBP/hull hybrid initial iterate image as specified by the settings.cfg file
char X_BASENAME[]				= "x";									// Prefix of the file containing the reconstructed images after each of the N iterations (e.g., x_1, x_2, x_3, ..., x_N)
char MLP_BASENAME[]				= "MLP";								// Prefix of the file containing the MLP path data
char WEPL_BASENAME[]			= "WEPL";								// Prefix of the file containing the WEPL measurements for each MLP path
char HISTORIES_BASENAME[]		= "histories";							// Prefix of the file containing the x/y/z hull entry/exit coordinates/angles, x/y/z hull entry voxels, gantry angle, and bin # for each reconstruction history
char VOXELS_PER_PATH_BASENAME[]	= "voxels_per_path";					// Prefix of the file containing the # of intersected voxels per MLP path
char AVG_CHORDS_BASENAME[]		= "avg_chord_lengths";					// Prefix of the file containing the effective (average) chord length for each MLP path

char MEDIAN_FILTER_2D_POSTFIX[]	= "med_2D_r";							// Postfix added to filename for 2D median filtered images
char MEDIAN_FILTER_3D_POSTFIX[]	= "med_3D_r";							// Postfix added to filename for 3D median filtered images
char AVERAGE_FILTER_2D_POSTFIX[]= "avg_2D_r";							// Postfix added to filename for 2D average filtered images
char AVERAGE_FILTER_3D_POSTFIX[]= "avg_3D_r";							// Postfix added to filename for 3D average filtered images

DISK_WRITE_MODE PROJECTION_DATA_WRITE_MODE	= BINARY;					// Disk write mode for the files containing the projection data (tracker/WEPL/gantry angle) used as input to preprocessing
DISK_WRITE_MODE RADIOGRAPHS_WRITE_MODE		= TEXT;						// Disk write mode for the files containing the radiograph images from each projection angle before/after performing cuts
DISK_WRITE_MODE WEPL_DISTS_WRITE_MODE		= TEXT;						// Disk write mode for the files containing the WEPL distribution images from each projection angle before/after performing cuts
DISK_WRITE_MODE HULL_WRITE_MODE				= TEXT;						// Disk write mode for the file containing the SC, MSC, SM, or FBP hull image as specified by the settings.cfg file 
DISK_WRITE_MODE FBP_WRITE_MODE				= TEXT;						// Disk write mode for the file containing the FBP image
DISK_WRITE_MODE FBP_MEDIANS_WRITE_MODE		= TEXT;						// Disk write mode for the file containing the median filtered FBP images
DISK_WRITE_MODE X_0_WRITE_MODE				= TEXT;						// Disk write mode for the file containing the FBP, hull, or FBP/hull hybrid initial iterate image as specified by the settings.cfg file
DISK_WRITE_MODE MLP_WRITE_MODE				= BINARY;					// Disk write mode for the file containing the MLP path data
DISK_WRITE_MODE HISTORIES_WRITE_MODE		= BINARY;					// Disk write mode for the file containing the x/y/z hull entry/exit coordinates/angles, x/y/z hull entry voxels, gantry angle, and bin # for each reconstruction history
DISK_WRITE_MODE WEPL_WRITE_MODE				= BINARY;					// Disk write mode for the file containing the WEPL measurement for each MLP data
DISK_WRITE_MODE VOXELS_PER_PATH_WRITE_MODE	= BINARY;					// Disk write mode for the file containing the number of intersected voxels in each MLP path
DISK_WRITE_MODE AVG_CHORDS_WRITE_MODE		= BINARY;					// Disk write mode for the file containing the effective (average) chord length for each MLP path
DISK_WRITE_MODE X_WRITE_MODE				= TEXT;						// Disk write mode for the file containing the reconstructed images after each of the N iterations (e.g., x_1, x_2, x_3, ..., x_N)
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
#define CONSOLE_WINDOW_WIDTH	80
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------ Data log parameters ----------------------------------------------------------------------------*/
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
#define NUM_LOG_FIELDS			13
#define OBJECT_ENTRIES_SIZE		64
#define DATE_ENTRIES_SIZE		20
#define TYPE_ENTRIES_SIZE		16	
#define RUN_NUM_ENTRIES_SIZE	28
#define AUTHOR_ENTRIES_SIZE		64
#define CONFIG_LINK_SIZE		256
#define COMMENT_ENTRIES_SIZE	512
#define LOG_FIRST_LINE			1
#define LOG_READ_BUFFER			1024
const char* LOG_ITEMS[NUM_LOG_FIELDS] = {"Phantom Name", "Scan Type", "Run Date", "Run Number", "Acquired By", "Projection Data Date", "Calibrated By", "Preprocessing Date", "Preprocessed By", "Reconstruction Date", "Reconstructed By", "Link to Config File", "Comments" };
const uint LOG_FIELD_WIDTHS[NUM_LOG_FIELDS] = {OBJECT_ENTRIES_SIZE, TYPE_ENTRIES_SIZE, DATE_ENTRIES_SIZE, RUN_NUM_ENTRIES_SIZE, AUTHOR_ENTRIES_SIZE, DATE_ENTRIES_SIZE, AUTHOR_ENTRIES_SIZE, DATE_ENTRIES_SIZE, AUTHOR_ENTRIES_SIZE, DATE_ENTRIES_SIZE, AUTHOR_ENTRIES_SIZE, CONFIG_LINK_SIZE, COMMENT_ENTRIES_SIZE };
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------- Config file parameters ---------------------------------------------------------------------------*/
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
#define NUM_CONFIG_FIELDS		4
#define CONFIG_LINE_WIDTH		180
#define KEY_FIELD_WIDTH			24
#define EQUALS_FIELD_WIDTH		2
#define VALUE_FIELD_WIDTH		34
#define COMMENT_FIELD_WIDTH		120
#define CONFIG_READ_BUFFER		1024
const uint CONFIG_FIELD_WIDTHS[NUM_CONFIG_FIELDS] = {KEY_FIELD_WIDTH, EQUALS_FIELD_WIDTH, VALUE_FIELD_WIDTH, COMMENT_FIELD_WIDTH};
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------- Config file parameters ---------------------------------------------------------------------------*/
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/

#define COPYRIGHT_NOTICE	"Copyright (C) 2015: Blake Schultze"
#define HEADER_STATEMENT	"This program reconstructs Proton Computed Tomography (pCT) images from scan measurements (e.g. proton gantry angle, proton tracker coordinates, and Water Equivalent Path Length (WEPL) data) acquired from experiments or simulations.  The program operates in two distinct phases: (1) preprocessing of data to remove problematic/unnecessary data and generate the object's hull and each valid proton's hull entry/exit coordinates/angles and the image reconstruction (Ax = b) parameters A (MLP path data), x0 (initial iterate/guess), and b (WEPL measurements) and (2) performs image reconstruction using an iterative projection method algorithm (e.g., ART, DROP, SAP, BIP, etc.).  These 2 phases of image reconstruction can be performed separately, allowing one to use the same preprocessed data previously generated to perform multiple image reconstructions as this aspect of pCT is the primary area of interest for numerous research groups and most of these have no tools for generating the requisite preprocessed data.  This program thus serves a dual purpose: (1) as a tool to continue researching the optimal methods for processing measurement data while (2) providing a tool to generate the data required for image reconstruction so groups researching data acquisition can analyze the effects modifications to the scanner system and data calibration have on reconstructed images and provide image reconstruction researchers with a tool to generate the input data needed to perform and analyze their reconstruction algorithms without either of these groups needing to spend the time to understand the details of preprocessing or generate programs to generate it themselves. This also establishes a standardized data format and organization scheme and provides a common framework/collection of tools, allowing various groups to operate independently while still being able to assess the merit/suitability of proposed improvements directly since the accuracy/quality of results can be compared directly when using the same input data.  There are numerous instances where GPGPU programming can be applied to exploit the inherent data/task parallelism to improve computational efficiency and reduce execution time.  The portions of the program where parallel programming has been implemented and/or the complexity of the programming used to exploit it as much as possible both continue to increase, but to make the program operable by a larger portion of the research community (including those with limited sequential/parallel programming), the program's underlying programming code and its compilation/execution are abstracted from its operation.  The code can be accessed by users and can be downloaded and modified on their local machine, but all options/parameters a user may want to modify were identified and their values are specified as key/value pairs in an external text file.  The program repository includes this configuration file (\"settings.cfg\") with the key for all of these options/parameters and their default value along with an explanation of what each means/controls.  The program does not need to be recompiled each time the value of an option/parameter is changed since the program's source code is not modified, thus, users do not need to know how to program/compile code nor do they need to understand every option/parameter, they need only change the value of the keys they wish to modify inside the configuration text file and launch the program's .exe file and it will import these option/parameter values and generate the generate the resulting data/images.  The program must know the name and location of all input data and given that preprocessed data generated by the program can later be used as input to this or another researchers' program, an extremely important aspect of operating the program solely through an external text file and sharing data/results is the naming/format of all input/output data/image files and the structure of the hierarchy of directories where they are stored.  The location of a file/folder in the hierarchy of pCT data directories transparently encodes information like for which object, type of scan (experimental/simulated), date of scan/calibration/preprocessing/reconstruction, etc. is data associated, not only making it easier to find data/images and share data/results, but also reducing the required information that must be specified by the user to operate the program since any information that cannot be inferred by the program automatically must be specified as a parameter in the configuration file.  Thus, the program adheres to and enforces the established data format and organizational scheme and automatically creates the files/folders required to maintain this with the appropriate names.  Although file names are strictly enforced, there is some flexibility in the location of input data by providing a key which can be used to specify the path to where it is located.  However, the program was designed so that its .exe and config file be placed in the same folder as the input tracker/WEPL data and upon execution, the program parses the path to the executable to determine the scan information and automatically read the associated data and create/write to the appropriate locations.  Ideally, all users would adhere to the standardized data format and organizational scheme, making it possible for users to regularly synchronize their local data storage with the main server so they always have up-to-date software and data, but to prevent situations where a user that cannot follow this recommendation is unable to use the program, some paths can be specified explicitly, but the output data and subdirectories created to house it still follows the format.  A significant amount of time was spent developing the file/folder naming scheme and organizational hierarchy that would balance the desire to easily locate data with the indivudal needs of researchers that generate this data and the ease to which they can conform to it, both its programmatic simplicity and with consideration to the format/structure to which they have established for their own use and are accustomed.  In cases where a choice between programming simplicity and ease of use had to be made, user experience was given higher importance, hopefully making it possible for all those interested to use this and other pCT programs.  Please direct all issues/complications with pCT programs and any comments/suggestions to the developer of that program and Reinhard Schulte."
#define BLAKE_CONTACT		"Blake Schultze (Blake_Schultze@baylor.edu)"
#define REINHARD_CONTACT	"Reinhard Schulte (rschulte@llu.edu)"
#define KEITH_CONTACT		"Keith Schubert (Keith_Schubert@baylor.edu)"
#define PANIZ_CONTACT		"Paniz Karbasi (Paniz_Karbasi@baylor.edu)"
#define header_length		strlen(copyright_notice) + strlen(header_statement) + strlen(blake_contact) + strlen(reinhard_contact) + strlen(keith_contact) + strlen(paniz_contact)