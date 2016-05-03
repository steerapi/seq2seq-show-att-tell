#!/bin/sh
name="seq2seq_att"
suffix="$1_test"
th project.lua -gpuid 2 -classifier ${name} -test 1 -train 0 -outfile out_${name}${suffix}.txt -lossfile loss_${name}${suffix}.png | tee -a log_${name}${suffix}.txt