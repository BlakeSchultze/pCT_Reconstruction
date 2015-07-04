#!/bin/bash
if [ ! -n $1 ]
then
	printf -v angle_interval "%d" $1
else 
	angle_interval=4
fi	
for ((angle=0; angle<360; angle+=$angle_interval))
do
	pnum=$((4*($i)))
	pnumstr=$(printf "%03d" $pnum)
	cp *$pnumstr.dat.root.reco.root.bin projection_$pnumstr.bin
done
