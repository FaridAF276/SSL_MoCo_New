#!/bin/bash
# cd && wget -nc https://raw.githubusercontent.com/vast-ai/vast-python/master/vast.py -O vast; chmod +x vast; \
# cat ~/.ssh/authorized_keys | md5sum | awk '{print $1}' > ssh_key_hv; echo -n $VAST_CONTAINERLABEL | md5sum | awk '{print $1}' > instance_id_hv; head -c -1 -q ssh_key_hv instance_id_hv > ~/.vast_api_key; \
# ./vast start instance ${VAST_CONTAINERLABEL:2} && \
# bash -e SSL_MoCo_New/fine_tune_scripts/fine_tune_stl10.sh && \
# ./vast stop instance ${VAST_CONTAINERLABEL:2}
cd SSL_MoCo_New
##Download pretrained model
gdown --fuzzy -O temp.zip https://drive.google.com/file/d/1OWweZjrJklhQG6NITTB-KlDDPpxGjl3L/view?usp=sharing
unzip -n temp.zip
gdown --fuzzy -O stl10.zip https://drive.google.com/file/d/1B0GLPjsXgtWLhV5SvBT1Kti1QHKWql2t/view?usp=sharing && unzip -n stl10.zip -d stl10
mkdir -p MoCo_eval_checkpoints
rm -rf stl10/unlabelled/
time python dataset_preparation.py \
--dataset_dir stl10 \
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
--results_dir "MoCo_eval_checkpoints/" \
--root_folder "downstream" \
--patience 25 \
--cos \
--num_classes 10 \
-pt-ssl

zip -r stl10_finetune.zip MoCo_train_checkpoints MoCo_eval_checkpoints
cd
./gdrive upload SSL_MoCo_New/stl10_finetune.zip
