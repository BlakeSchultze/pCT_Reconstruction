#!/bin/bash
run_date="$1"
printf -v angles "%d" $(($2-1))
printf -v angle_interval "%d" $3
IFS='_'
for ((i=0; i<=$angles; i++))
do
	printf -v angle_str "%03d" $(($angle_interval*$i))
	extension="$angle_str.dat.root.reco.root.bin"
	for file in *_$angle_str.*
	do
		set -- $file
		echo "$file"
		if [ "$file" != "*_${angle_str}.*" ]
		then
			path_out="${1}/Experimental/${run_date}/Run_${2}"
			if [ $3 != $extension ]
			then 
				path_out="${path_out}_${3}" 
			fi
			path_out="${path_out}/Input"
			mkdir -p "${path_out}"
			ln -s $angle_str.dat.root.reco.root.bin "${path_out}/projection_${angle_str}.bin"
		fi
	done
done
