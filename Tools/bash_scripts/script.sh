#!/bin/bash

proj_step=4 

for i in {0..89}
do
pro=$((i*($proj_step)))
fname=$(printf 'Edge_0057_%03d.dat.root.reco.root.bin' $pro)
lnfname=$(printf 'projection_%03d.bin' $pro)

cp $fname $lnfname
done