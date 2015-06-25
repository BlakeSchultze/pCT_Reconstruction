#!/bin/bash
#run_date="$1"
run_date="${PWD##*/}"
full_path=$PWD
printf -v angle_interval "%d" $1
IFS='_'
for ((angle=0; angle<360; angle+=$angle_interval))
do
	printf -v angle_str "%03d" $angle
	extension="$angle_str.dat.root.reco.root.bin"
	for file in *_$angle_str.*
	do
		set -- $file
		echo "$file"
		if [ "$file" != "*_${angle_str}.*" ]
		then
			j=2
			var="$j"
			run_num_dir=""
			while [ ${!var} != $extension ]
			do			
				run_num_dir="${run_num_dir}_${!var}" 
				j=$(($j+1))
				var="$j"
			done
			path_out="${1}/Experimental/${run_date}/Run${run_num_dir}/Input"
			mkdir -p "${path_out}"
			ln -s "$full_path/${1}${run_num_dir}_$angle_str.dat.root.reco.root.bin" "${path_out}/projection_${angle_str}.bin"
		fi
	done
done
