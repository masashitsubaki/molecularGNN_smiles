#!/bin/bash

DATASET=HIV
# DATASET=yourdata

radius=1
# radius=2

python preprocess_data.py $DATASET $radius
