######################################################################################################################################
############################### Determine the run date for the data in the current raw data directory ################################
######################################################################################################################################
error_flag="false"															# Initialize error_flag indicating unknown Phantom ID 
data_path=$PWD																# Query path to current raw data directory
run_date="${data_path##*/}"													# Extract run date from last directory in the path
echo "Run Date = ${run_date}"												# Print run date to terminal window
######################################################################################################################################
########################## Determine path to pCT data directory (parent directory of raw_data directory) #############################
######################################################################################################################################
data_directory="raw_data"													# Name of directory where all raw data is stored
pCT_path=${data_path%%${data_directory}*}									# Extract pCT data path from raw data path
echo "Path to pCT data directory = ${pCT_path}"								# Print path to pCT data directory to terminal window
######################################################################################################################################
############################# Find and create/organize links to raw data files for each angle in [0,360) #############################
######################################################################################################################################
if [[ ! -n $1 ]]; then														# If an angle interval was passed as an execution argument
	printf -v angle_interval "%d" $1										# Print execution argument $1 to angle interval variable
else 
	angle_interval=4														# Print execution argument $1 to angle interval variable
fi	
for ((angle=0; angle<360; angle+=$angle_interval)); do						# For each angle in [0,360) in steps of "angle_interval"
	##################################################################################################################################
	######################### Parse file names to determine object name and run # for the current angle xxx ##########################
	##################################################################################################################################
	printf -v angle_str "%03d" $angle										# Convert each angle into string of the form "xxx"
	for file in *_$angle_str.*; do											# For each file found for the current angle xxx
		IFS='_'																# Delimiter for splitting path into its directory names
		set -- $file														# Split file names into '_' separated tokens $1, $2, ...
		if [[ "$file" != "*_${angle_str}.*" ]]; then							# Make sure there are 2+ tokens so "$2" is valid token
			##########################################################################################################################
			################### Combine tokens until reaching xxx.extension to allow run #s like "Run_100_sup_..." ###################
			##########################################################################################################################
			extension=${file##*${angle_str}}								# Extract file extension from file name
			file_ending="${angle_str}${extension}"							# Set string used to stop parsing file names
			objectID="${1}"													# Set object name as 1st token
			run_num="${2}"													# Initialize run_num to 1st token after 1st '_'
			j=3																# Set integer # for next token # to access
			var="$j"														# Convert integer j into string to access tokens by #
			while [[ ${!var} != $file_ending ]]; do							# Parse while next token is not xxx.extension
				run_num="${run_num}_${!var}" 								# Append token to run # directory name
				j=$(($j+1))													# Advance token #
				var="$j"													# Token # integer -> string so !var accesses next token
			done
			##########################################################################################################################
			################ Use objectID to lookup the appropriate object folder name corresponding to this phantom #################
			##########################################################################################################################
			if [[ $objectID == "Emp" ]]; then object_name="Empty"
			elif [[ $objectID == "CalEmp" ]]; then object_name="Calibration"
			elif [[ $objectID == "Calib" ]]; then object_name="Calibration"
			elif [[ $objectID == "Rod" ]]; then object_name="Rod"
			elif [[ $objectID == "Water" ]]; then object_name="Water"	
			elif [[ $objectID == "Sensitom" ]]; then object_name="CTP404_Sensitom"
			elif [[ $objectID == "LowCon" ]]; then object_name="CTP515_Low_Contrast"
			elif [[ $objectID == "Dose16" ]]; then object_name="CTP554_Dose"
			elif [[ $objectID == "CIRSPHP0" ]]; then object_name="CTP404_PedHead_0"
			elif [[ $objectID == "CIRSPHP1" ]]; then object_name="CTP404_PedHead_1"
			elif [[ $objectID == "LMUDECT" ]]; then object_name="CTP404_LMU_DECT"
			elif [[ $objectID == "CIRSEdge" ]]; then object_name="CIRS_Edge"
			else error_flag="true"
			fi
			##########################################################################################################################
			############################# Construct path/file names for the raw data and the links to it #############################
			##########################################################################################################################
			data_file="${objectID}_${run_num}_${file_ending}"
			link_path="${pCT_path}organized_data/${object_name}/Experimental/${run_date}/Run_${run_num}/Input/"
			link_file="raw_${angle_str}.bin"
			##########################################################################################################################
			################## Create appropriate directories/subdirectories and create/organize the raw data links ##################
			##########################################################################################################################
			mkdir -p "${link_path}"											# Create directories/subdirectories for links
			ln -s "${data_path}/${data_file}" "${link_path}${link_file}"	# Create the soft links to the raw data
			##########################################################################################################################
			################### Print scan properties/characteristics and directory/file names of data and links## ###################
			##########################################################################################################################
			echo "/------------------------------------------------------------------------------/"
			echo -e "File path = \n\n${data_path}/${data_file}\n"
			echo "File name = $file"
			echo "File extension = $extension"
			echo "Object ID = ${objectID}"
			echo "Object Name = ${object_name}"
			echo "Run # = ${run_num}"
			echo -e "Link path = \n\n${link_path}${link_file}"
		fi
		if [[ $error_flag == "true" ]]
		then
			echo "ERROR: Unknown Phantom ID encountered"
			break
		fi
	done
done
