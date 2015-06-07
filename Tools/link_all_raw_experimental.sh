#!/bin/bash
run_date="$1"
echo "$run_date"
for i in {0..89}
do
	
	pnum=$((4*($i)))
	pnumstr=$(printf "%03d" $pnum)
	IFS='_'
	extension=$pnumstr.dat.root.reco.root.bin
	for file in *_$pnumstr.*
	do
		set -- $file
		echo "$file"
		if [ "$file" != "*_${pnumstr}.*" ]
		then
			path_out="${1}/Experimental/${run_date}/Run_${2}"
			if [ $3 != $extension ]
			then 
				path_out="${path_out}_${3}" 
			fi
			path_out="${path_out}/Input"
			mkdir -p "${path_out}"
			ln -s $pnumstr.dat.root.reco.root.bin "${path_out}/projection_${pnumstr}.bin"
		fi
	done
done
