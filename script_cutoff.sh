#!/bin/bash
for i in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5
do
    echo "----------------CORRIENDO CON RATIO = $i cs -----------------------"
    /home/lac/miniconda2/envs/lac/bin/python /home/lac/PIRadar2019/src/test_1estacion_confusion.py $i cs > ScoreConCutoff_CS$i.log 2>&1
    echo "----------------CORRIENDO CON RATIO = $i ss -----------------------"
    /home/lac/miniconda2/envs/lac/bin/python /home/lac/PIRadar2019/src/test_1estacion_confusion.py $i ss > ScoreConCutoff_SS$i.log 2>&1
done
