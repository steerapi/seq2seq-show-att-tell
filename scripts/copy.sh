#!/bin/sh
name=$1
mkdir $2
cp $name.t7 $2
cp loss_$name.png $2
cp loss_$name.png.th $2
cp test_$name.txt $2
cp log_$name.txt $2
cp out_$name.txt $2