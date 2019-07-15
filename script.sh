#!/bin/bash
for x in {1..34};
do /home/opt/anaconda3/envs/pi_radar/bin/python /home/awf/guille/input_matrix_pa2.py $x;
done
