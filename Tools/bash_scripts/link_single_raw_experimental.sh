#!/bin/bash

for i in {0..0}
do
	pnum=$((4*($i)))
	pnumstr=$(printf "%03d" $pnum)
	filename=$(ls *_$pnumstr.*)
	IFS='_'
	set -- $filename
	path_out="${1}/Experimental/YYMMDD/Run_${2}"
	extension=$pnumstr.dat.root.reco.root.bin
	if [ $3 != $extension ]
	then 
		path_out="${path_out}_${3}/Input" 
	else
		path_out="${path_out}/Input"
	fi
	mkdir -p "${path_out}"
	ln -s $pnumstr.dat.root.reco.root.bin "${path_out}/projection_${pnumstr}.bin"
done
