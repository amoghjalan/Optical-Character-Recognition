#!/bin/bash

echo Enter path to the dataset:

read path

python3 run.py $path >> output.txt
