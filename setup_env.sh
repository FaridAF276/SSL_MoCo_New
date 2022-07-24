#!/bin/bash

# apt-get install -y git zip vim unzip fastjar && git clone https://github.com/FaridAF276/SSL_MoCo_New.git && chmod +x SSL_MoCo_New/setup_env.sh && ./SSL_MoCo_New/setup_env.sh
chmod +x SSL_MoCo_New/quickstart_chestxray.sh SSL_MoCo_New/quickstart_imagenet.sh SSL_MoCo_New/quickstart_cifar10.sh SSL_MoCo_New/find_best_lr.sh
apt-get install -y wget;
#shell script
# tail -n +17 quickstart_chestxray.sh | bash
apt update -y
pip install pandas matplotlib tensorboard Pillow split-folders gdown pandas
#Download and connect with gdrive
wget -nc https://github.com/prasmussen/gdrive/releases/download/2.1.1/gdrive_2.1.1_linux_386.tar.gz
tar -xvf gdrive_2.1.1_linux_386.tar.gz
./gdrive about