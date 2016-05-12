#!/bin/sh
name="seq2seq_att"
suffix="$1_noinit"
th project.lua -gpuid 1 -init_dec 0 -epochs 20 -classifier ${name} -test 0 -train 1 -lossfile loss_${name}${suffix}.png -savefile ${name}${suffix}.t7 -train_from ${name}${suffix}.t7 | tee -a log_${name}${suffix}.txt