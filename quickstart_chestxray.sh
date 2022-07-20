#!/bin/bash
#Around 5 GB used
# splitfolders --output ChestX --ratio .8 .1 .1 --move \
# -- COVID19_Pneumonia_Normal_Chest_Xray_PA_Dataset

time python dataset_preparation.py \
--dataset_dir COVID19_Pneumonia_Normal_Chest_Xray_PA_Dataset \
--percentage 0.2 \
--split_train_test
#To start from here type this command : 
# tail -n +17 quickstart_chestxray.sh | bash

#Create directories for train et eval models
mkdir MoCo_train_checkpoints && \
mkdir MoCo_eval_checkpoints
python -c "import torch; import torchvision; print('\n Torch version:\t', torch.__version__, '\n Torchvision version:\t', torchvision.__version__)"
# #Launch training process
time python pre_train.py \
--epochs 1 \
--batch_size 16 \
--lr 0.6 \
--results-dir "MoCo_train_checkpoints/" \
--dataset "folder" \
--root_folder "pretext" \
--cos \
--knn-k 4000 \
--bn-splits 1
touch MoCo_train_checkpoints/linear_eval.log
zip -r chest_pretext.zip MoCo_train_checkpoints
./gdrive upload chest_pretext.zip
time python linear_eval.py \
--epochs 1 \
--batch_size 16 \
--lr 0.6 \
--model-dir "MoCo_train_checkpoints/" \
--dataset-ft "folder" \
--results_dir "MoCo_eval_checkpoints/" \
--root_folder "downstream" \
--cos \
--num_classes 3 \
-pt-ssl

# #Zip the result and upload them to drive

zip -r chest_dowstr.zip MoCo_eval_checkpoints
./gdrive upload chest_dowstr.zip
./vast stop instance ${VAST_CONTAINERLABEL:2}