#!/bin/bash
#echo "$1"
#angle_interval=4
#printf -v angle_interval "%d" 4
#angle_interval=$(printf "%d" $1)
#echo $angle_interval
#for (( angle=0; angle<360 ; angle+=4 )) 
for angle in {0..89};do printf -v angle_str "%03d" ((4*($angle)));cp Edge_0057_$angle_str.dat.root.reco.root.bin projection_$angle_str.bin;done;