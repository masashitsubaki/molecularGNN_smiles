#!/bin/bash

DATASET=HIV
# DATASET=yourdata

radius=2

dim=25

layer=3

batch=32

lr=1e-3

lr_decay=0.9

decay_interval=50

iteration=1000

setting=$DATASET--radius$radius--dim$dim--layer$layer--batch$batch--lr$lr--lr_decay$lr_decay--decay_interval$decay_interval
python run_training.py $DATASET $radius $dim $layer $batch $lr $lr_decay $decay_interval $iteration $setting
