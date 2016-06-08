#!/bin/bash
######################################################################################################################################
########################### Determine the run date for the data in the current projection data directory #############################
######################################################################################################################################
execution_path=$PWD
error_flag="false"                                                          # Initialize error_flag indicating unknown Phantom ID 
data_path=$PWD  
angle_interval=4  
scan_type='Experimental'                                                          # Query path to current raw data directory
while getopts 'p:i:t:' opt; do
    case $opt in
        p) data_path=${OPTARG};;
        i) printf -v angle_interval "%d" ${OPTARG};;
        t) scan_type=${OPTARG};;
        *) error "Unexpected option ${flag}";;
    esac
done

run_date="${data_path##*/}"                                                 # Extract run date from last directory in the path
#scan_type_path="${data_path%/$run_date}"                                    # Remove run date from path so scan type folder is on end
#scan_type="${scan_type_path##*/}"                                           # Identify scan type from last folder on scan_type_path
data_directory="preprocessed_data"                                            # Name of directory where all projection data is stored
output_folder="projection_data"                                            # Name of directory where all projection data is stored
pCT_path=${data_path%%${data_directory}*}                                   # Extract pCT data path from projection data path
echo  "Execution path:$execution_path"
echo  "pCT data parent directory:$pCT_path"                                 # Print path to pCT data directory to terminal window
echo  "Raw data path:$data_directory"
echo  "Scan Type: ${scan_type}"                                             # Print scan type: Experimental or Simulated
echo  "Run Date:$run_date"
echo  "Angle interval:$angle_interval" 
cd "${data_path}"
######################################################################################################################################
######################### Find and create/organize links to projection data files for each angle in [0,360) ##########################
######################################################################################################################################
for ((angle=0; angle<360; angle+=$angle_interval)); do                      # For each angle in [0,360) in steps of "angle_interval"
    ##################################################################################################################################
    ########################## Parse file names to determine object ID and run # for the current angle xxx ###########################
    ##################################################################################################################################
    printf -v angle_str "%03d" $angle                                       # Convert each angle into string of the form "xxx"
    for file in *_$angle_str.*; do                                          # For each file found for the current angle xxx
        IFS='_'                                                             # Delimiter for splitting path into its directory names
        set -- $file                                                        # Split file names into '_' separated tokens $1, $2, ...
        if [[ $file != "*_${angle_str}.*" ]]; then                          # Make sure there are 2+ tokens so "$2" is valid token
            ##########################################################################################################################
            ################### Combine tokens until reaching xxx.extension to allow run #s like "Run_100_sup_..." ###################
            ##########################################################################################################################
            extension=${file##*${angle_str}}                                # Extract file extension from file name
            file_ending="${angle_str}${extension}"                          # Set string used to stop parsing file names
            objectID="${1}"                                                 # Set object ID as 1st token
            run_num="${2}"                                                  # Initialize run_num to 1st token after 1st '_'
            j=3                                                             # Set integer # for next token # to access
            var="$j"                                                        # Convert integer j into string to access tokens by #
            while [ ${!var} != "$file_ending" ]; do                         # Parse while next token is not xxx.extension
                run_num="${run_num}_${!var}"                                # Append token to run # directory name
                j=$(($j+1))                                                 # Advance token #
                var="$j"                                                    # Token # integer -> string so !var accesses next token
            done
            data_file="${objectID}_${run_num}_${file_ending}"
            echo "$data_file"
			if [[ $angle == 0 ]]; then preprocessed_date=$(date +"%y-%m-%d" -r "${data_file}"); echo hello; fi                                                # Extract run date from last directory in the path            
            ##########################################################################################################################
            ################ Use objectID to lookup the appropriate object folder name corresponding to this phantom #################
            ##########################################################################################################################
            if [[ $objectID == "Emp" ]]; then object_name="Empty"
            elif [[ $objectID == "CalEmp" ]]; then object_name="Calibration"
            elif [[ $objectID == "Calib" ]]; then object_name="Calibration"
            elif [[ $objectID == "Rod" ]]; then object_name="Rod"
            elif [[ $objectID == "Water" ]]; then object_name="Water"   
            elif [[ $objectID == "Sensitom" ]]; then object_name="CTP404_Sensitom"
            elif [[ $objectID == "LinePair" ]]; then object_name="CTP528_Linepair"
            elif [[ $objectID == "LowCon" ]]; then object_name="CTP515_Low_Contrast"
            elif [[ $objectID == "Dose16" ]]; then object_name="CTP554_Dose"
            elif [[ $objectID == "CIRSPHP0" ]]; then object_name="HN715_PedHead_0"
            elif [[ $objectID == "CIRSPHP1" ]]; then object_name="HN715_PedHead_1"
            elif [[ $objectID == "LMUDECT" ]]; then object_name="LMU_DECT"
            elif [[ $objectID == "CIRSEdge" ]]; then object_name="CIRS_Edge"
            elif [[ $objectID == "Birks" ]]; then object_name="Birks"
            else error_flag="true"
            fi
            ##########################################################################################################################
            ######################### Construct path/file names for the projection data and the links to it ##########################
            ##########################################################################################################################
            data_file="${objectID}_${run_num}_${file_ending}"
            link_path="${pCT_path}organized_data/${object_name}/${scan_type}/${run_date}/${run_num}/Output/${preprocessed_date}/"
            link_file="projection_${angle_str}.bin"
            ##########################################################################################################################
            ################### Print scan properties/characteristics and directory/file names of data and links## ###################
            ##########################################################################################################################
            echo "/------------------------------------------------------------------------------/"
            echo "Projection data filename = $file"
            echo "Preprocessed date = ${preprocessed_date}"
            echo "Object Name = ${object_name}"
            echo "Run # = ${run_num}"
            echo -e "Link destination = ${link_path}${link_file}"
            ##########################################################################################################################
            ############## Create appropriate directories/subdirectories and create/organize the projection data links ###############
            ##########################################################################################################################
            mkdir -p "${link_path}"                                         # Create directories/subdirectories for links
            ln -s "${data_path}/${data_file}" "${link_path}${link_file}"    # Create the soft links to the projection data
        fi
        if [[ $error_flag == "true" ]]
        then
            echo "ERROR: Unknown Phantom ID '$objectID' encountered"
            error_flag=false;
            break
        fi
    done
done
