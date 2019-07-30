#!/bin/bash
for i in 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 1.1 1.2 1.3 1.4 1.5 1.6 1.7
do
    echo "----------------CORRIENDO CON RATIO = $i -----------------------"
    /home/lac/miniconda2/envs/lac/bin/python /home/lac/PIRadar2019/src/vgg16_1estacion.py $i > ComparativaSMOTE$i.log 2>&1
done
