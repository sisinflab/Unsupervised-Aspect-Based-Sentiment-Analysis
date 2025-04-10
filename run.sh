#!/bin/bash


INPUT_PATH="...data/test_category/Rest14_test.csv"

### TRAIN ###
python -u script/train.py

### TEST ###
python -u script/test.py --input_path "$INPUT_PATH"


### TEST CATEGORY###
python -u script/test_category.py