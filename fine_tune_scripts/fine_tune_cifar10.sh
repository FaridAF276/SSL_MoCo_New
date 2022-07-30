#!/bin/bash
# cd && wget -nc https://raw.githubusercontent.com/vast-ai/vast-python/master/vast.py -O vast; chmod +x vast; \
# cat ~/.ssh/authorized_keys | md5sum | awk '{print $1}' > ssh_key_hv; echo -n $VAST_CONTAINERLABEL | md5sum | awk '{print $1}' > instance_id_hv; head -c -1 -q ssh_key_hv instance_id_hv > ~/.vast_api_key; \
# ./vast start instance ${VAST_CONTAINERLABEL:2} && \
# bash -e SSL_MoCo_New/fine_tune_scripts/fine_tune_cifar10.sh && \
# ./vast stop instance ${VAST_CONTAINERLABEL:2}
cd SSL_MoCo_New
##Download pretrained model
gdown --fuzzy -O temp.zip https://drive.google.com/file/d/1OWweZjrJklhQG6NITTB-KlDDPpxGjl3L/view?usp=sharing
unzip -n temp.zip
gdown --fuzzy -O cifar.zip https://drive.google.com/file/d/1ny6vBH54X0qV07EsNhgddHFGYgoROswy/view?usp=sharing && unzip -n cifar.zip
mkdir -p MoCo_eval_checkpoints
time python dataset_preparation.py \
--dataset_dir cifar10 \
--percentage 0.20 \
--split_train_test
python -c "import torch; import torchvision; print('\n Torch version:\t', torch.__version__, '\n Torchvision version:\t', torchvision.__version__)"
##Download model
mkdir -p MoCo_eval_checkpoints
time python linear_eval.py \
--epochs 500 \
--batch_size 512 \
--lr 1e-3 \
--model-dir "MoCo_train_checkpoints/" \
--dataset-ft "folder" \
--patience 25 \
--results_dir "MoCo_eval_checkpoints/" \
--root_folder "downstream" \
--cos \
--num_classes 10 \
-pt-ssl

zip -r cifar10_finetune.zip MoCo_train_checkpoints MoCo_eval_checkpoints
cd
./gdrive upload SSL_MoCo_New/cifar10_finetune.zip
