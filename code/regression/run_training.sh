#!/bin/bash

DATASET=photovoltaic
# DATASET=yourdata

# radius=1
radius=2
# radius=3

update=sum
# update=mean

# output=sum
output=mean

dim=25
hidden_layer=6
output_layer=3
batch=32
lr=1e-3
lr_decay=0.9
decay_interval=10
weight_decay=1e-6
iteration=300

setting=$DATASET--radius$radius--update_$update--output_$output--dim$dim--hidden_layer$hidden_layer--output_layer$output_layer--batch$batch--lr$lr--lr_decay$lr_decay--decay_interval$decay_interval--weight_decay$weight_decay--iteration$iteration
python run_training.py $DATASET $radius $update $output $dim $hidden_layer $output_layer $batch $lr $lr_decay $decay_interval $weight_decay $iteration $setting
