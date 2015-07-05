#!/bin/bash
# if [ ! -n $1 ]
# then
	# printf -v angle_interval "%d" $1
# else 
	# angle_interval=4
# fi	
# for ((angle=0; angle<360; angle+=$angle_interval))
# do
	# pnum=$((4*($i)))
	# pnumstr=$(printf "%03d" $pnum)
	# ln -s *$pnumstr.dat.root.reco.root.bin projection_$pnumstr.bin
# done
for angle in {0..89};do printf -v angle_str "%03d" ((4*($angle)));cp Edge_0057_$angle_str.dat.root.reco.root.bin projection_$angle_str.bin;done;