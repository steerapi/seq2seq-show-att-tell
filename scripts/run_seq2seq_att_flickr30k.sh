#!/bin/sh
name="seq2seq_att"
suffix="$1_flickr30k"
th project.lua -gpuid 1 -target_size 18463 -trainfile "new_processed_data/proc_train_flickr30k.h5" -validfile "new_processed_data/proc_val_flickr30k.h5" -testfile "new_processed_data/proc_test_flickr30k.h5" -init_dec 0 -epochs 20 -classifier ${name}  -test 0 -train 1 -lossfile loss_${name}${suffix}.png -savefile ${name}${suffix}.t7 -train_from ${name}${suffix}.t7 | tee -a log_${name}${suffix}.txt

