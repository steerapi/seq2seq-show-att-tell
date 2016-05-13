#!/bin/sh
name="seq2seq_att"
suffix="$1_coco"
th project.lua -gpuid 2 -targ_dict "mydata/idx_to_word_coco.txt" -target_size 27466 -trainfile "new_processed_data/proc_train_coco.h5" -validfile "new_processed_data/proc_val_coco.h5" -testfile "new_processed_data/proc_test_coco.h5"  -init_dec 0 -classifier ${name} -test 1 -train 0 -outfile out_${name}${suffix}.txt -train_from ${name}${suffix}.t7 | tee -a log_${name}${suffix}.txt