#!/bin/bash

## Best cifar10 : 0.01
##Best stl10 : 0.01
##Best Imagenet : 
##Best ChestXRay : 
set -e 
# apt-get install -y git zip vim unzip fastjar && git clone https://github.com/FaridAF276/SSL_MoCo_New.git && chmod +x SSL_MoCo_New/find_best_lr.sh && ./SSL_MoCo_New/find_best_lr.sh
# cat ~/.ssh/authorized_keys | md5sum | awk '{print $1}' > ssh_key_hv; echo -n $VAST_CONTAINERLABEL | md5sum | awk '{print $1}' > instance_id_hv; head -c -1 -q ssh_key_hv instance_id_hv > ~/.vast_api_key; 
apt-get install -y wget; wget https://raw.githubusercontent.com/vast-ai/vast-python/master/vast.py -O vast; 
chmod +x vast; 
sleep 5
# ./vast start instance ${VAST_CONTAINERLABEL:2}; 
#start from here : # tail -n +9 find_best_lr.sh | bash
cd SSL_MoCo_New
#cd SSL_MoCo_New && chmod +x SSL_MoCo_New/find_best_lr.sh && ./SSL_MoCo_New/find_best_lr.sh
apt update -y
pip install pandas matplotlib tensorboard Pillow split-folders gdown
#Download and connect with gdrive
# wget https://github.com/prasmussen/gdrive/releases/download/2.1.1/gdrive_2.1.1_linux_386.tar.gz
# tar -xvf gdrive_2.1.1_linux_386.tar.gz
# ./gdrive about
#Download ImageNet dataset
# wget https://data.mendeley.com/public-files/datasets/jctsfj2sfn/files/148dd4e7-636b-404b-8a3c-6938158bc2c0/file_downloaded && \
# unzip file_downloaded
# mkdir lr_find 
# python find_lr.py \
# --lr_min=0.5 --lr_max=1 --epochs=5 \
# --batch_size=16 \
# --bn-splits=1 \
# --results-dir=lr_find \
# --dataset=folder \
# --root_folder=ChestX
'''

# splitfolders --output imgnt --ratio .8 .1 .1 --move \
# -- imagenet
gdown --fuzzy https://drive.google.com/file/d/1_dRbJEpMH7436l8aU4xrGHcFIE9i5TX7/view?usp=sharing && unzip tiny-imagenet-200.zip && \
python find_lr.py \
--lr_min=0.1 --lr_max=0.4 --epochs=5 \
--nb_value 5 \
--batch_size=16 \
--bn-splits=1 \
--results-dir=lr_find \
--dataset=folder \
--root_folder=imagenet

'''

python find_lr.py \
--lr_min=0.5 --lr_max=1 --epochs=5 \
--batch_size=16 \
--bn-splits=1 \
--results-dir=lr_find \
--dataset=stl10

# python find_lr.py \
# --lr_min=0.5 --lr_max=1 --epochs=5 \
# --batch_size=16 \
# --bn-splits=1 \
# --results-dir=lr_find \
# --dataset=cifar10
# ./vast stop instance ${VAST_CONTAINERLABEL:2}
