#!/bin/bash

DATASET=HIV
# DATASET=yourdata

# radius=0  # w/o fingerprints (i.e., atoms).
# radius=1
radius=2
# radius=3

python preprocess_data.py $DATASET $radius
