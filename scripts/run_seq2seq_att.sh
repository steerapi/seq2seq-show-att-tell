#!/bin/sh
name="seq2seq_att"
suffix="$1"
th project.lua -classifier ${name} -test 0 -train 1 -lossfile loss_${name}${suffix}.png | tee -a log_${name}${suffix}.txt