#!/bin/bash
# cd && wget -nc https://raw.githubusercontent.com/vast-ai/vast-python/master/vast.py -O vast; chmod +x vast; \
# cat ~/.ssh/authorized_keys | md5sum | awk '{print $1}' > ssh_key_hv; echo -n $VAST_CONTAINERLABEL | md5sum | awk '{print $1}' > instance_id_hv; head -c -1 -q ssh_key_hv instance_id_hv > ~/.vast_api_key; \
# ./vast start instance ${VAST_CONTAINERLABEL:2} && \
# bash -e SSL_MoCo_New/fine_tune_scripts/fine_tune_chest.sh && \
# ./vast stop instance ${VAST_CONTAINERLABEL:2}
cd SSL_MoCo_New
##Download pretrained model
gdown --fuzzy -O temp.zip https://drive.google.com/file/d/1JTRGAKh9vAcp6ZBmdnJ1riWq03pXgnKV/view?usp=sharing
unzip -n temp.zip
wget -nc https://md-datasets-public-files-prod.s3.eu-west-1.amazonaws.com/898720e7-9fcd-49f0-87ba-08c979e6f35e -O chest.zip && unzip -n chest.zip
splitfolders --output ChestX --ratio .8 .1 .1 --move \
-- COVID19_Pneumonia_Normal_Chest_Xray_PA_Dataset

time python dataset_preparation.py \
--dataset_dir ChestX \
--percentage 0.05 \
--split_train_test
python -c "import torch; import torchvision; print('\n Torch version:\t', torch.__version__, '\n Torchvision version:\t', torchvision.__version__)"
##Download model
mkdir -p MoCo_eval_checkpoints
time python linear_eval.py \
--epochs 500 \
--batch_size 256 \
--lr 0.01 \
--model-dir "MoCo_train_checkpoints/" \
--dataset-ft "folder" \
--results_dir "MoCo_eval_checkpoints/" \
--root_folder "downstream" \
--cos \
--patience 25 \
--num_classes 10 \
-pt-ssl

zip -r chest_finetune.zip MoCo_train_checkpoints MoCo_eval_checkpoints
cd
./gdrive upload SSL_MoCo_New/chest_finetune.zip
