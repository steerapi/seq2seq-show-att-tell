#!/bin/sh
name="seq2seq_att"
suffix="$1_noinit"
th project.lua -gpuid 2 -classifier ${name} -test 1 -train 0 -outfile out_${name}${suffix}.txt -train_from ${name}${suffix}.t7 | tee -a log_${name}${suffix}.txt