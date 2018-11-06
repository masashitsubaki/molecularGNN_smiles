#!/bin/bash

DATASET=HIV
# DATASET=yourdata

radius=2

python preprocess_data.py $DATASET $radius
