#!/bin/bash
for i in 
do
    echo "----------------CORRIENDO CON RATIO = $i ss -----------------------"
    /home/lac/miniconda2/envs/lac/bin/python /home/lac/PIRadar2019/src/algo.py > a$i.log 2>&1
done
