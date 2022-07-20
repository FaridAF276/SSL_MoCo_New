#!/bin/bash

# apt-get install -y git zip vim unzip fastjar && git clone https://github.com/FaridAF276/SSL_MoCo_New.git && chmod +x SSL_MoCo_New/setup_env.sh && ./SSL_MoCo_New/setup_env.sh
chmod +x SSL_MoCo_New/quickstart_chestxray.sh SSL_MoCo_New/quickstart_imagenet.sh SSL_MoCo_New/quickstart_cifar10.sh
apt-get install -y wget; 
wget https://raw.githubusercontent.com/vast-ai/vast-python/master/vast.py -O vast; chmod +x vast;
cat ~/.ssh/authorized_keys | md5sum | awk '{print $1}' > ssh_key_hv; echo -n $VAST_CONTAINERLABEL | md5sum | awk '{print $1}' > instance_id_hv; head -c -1 -q ssh_key_hv instance_id_hv > ~/.vast_api_key;
./vast start instance ${VAST_CONTAINERLABEL:2}
#shell script
# tail -n +17 quickstart_chestxray.sh | bash
apt update -y
pip install pandas matplotlib tensorboard Pillow split-folders
#Download and connect with gdrive
wget https://github.com/prasmussen/gdrive/releases/download/2.1.1/gdrive_2.1.1_linux_386.tar.gz
tar -xvf gdrive_2.1.1_linux_386.tar.gz
./gdrive about