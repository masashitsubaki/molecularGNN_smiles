#!/bin/bash

DATASET=photovoltaic
# DATASET=yourdata

radius=1
# radius=2

python preprocess_data.py $DATASET $radius
