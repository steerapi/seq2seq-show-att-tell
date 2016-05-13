#!/bin/sh
name="seq2seq_att"
suffix="$1_coco"
th project.lua -gpuid 3 -target_size 27466 -trainfile "new_processed_data/proc_train_coco.h5" -validfile "new_processed_data/proc_val_coco.h5" -testfile "new_processed_data/proc_test_coco.h5" -init_dec 0 -epochs 20 -classifier ${name} -pre_word_vecs_dec './data/correctly_idxed_glove_weights.hdf5' -test 0 -train 1 -lossfile loss_${name}${suffix}.png -savefile ${name}${suffix}.t7 -train_from ${name}${suffix}.t7 | tee -a log_${name}${suffix}.txt