#!/bin/bash

for i in {0..89}
do
pnum=$((4*($i)))
pnumstr=$(printf "%03d" $pnum)
ln -s $pnumstr.out.root.reco.root.bin projection_$pnumstr.bin
done
