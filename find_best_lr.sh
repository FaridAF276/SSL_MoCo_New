# apt-get install -y git zip vim unzip fastjar && git clone https://github.com/FaridAF276/SSL_MoCo_New.git && cd SSL_MoCo_New && chmod +x find_best_lr.sh && ./find_best_lr.sh
apt update -y
pip install pandas matplotlib tensorboard Pillow split-folders
#Download and connect with gdrive
# wget https://github.com/prasmussen/gdrive/releases/download/2.1.1/gdrive_2.1.1_linux_386.tar.gz
# tar -xvf gdrive_2.1.1_linux_386.tar.gz
# ./gdrive about
#Download ImageNet dataset
wget https://data.mendeley.com/public-files/datasets/jctsfj2sfn/files/148dd4e7-636b-404b-8a3c-6938158bc2c0/file_downloaded && \
unzip file_downloaded
splitfolders --output ChestX --ratio .8 .1 .1 --move \
-- COVID19_Pneumonia_Normal_Chest_Xray_PA_Dataset
mkdir lr_find 
python find_lr.py \
--lr=0.5 --epochs=10 \
--batch_size=16 \
--bn-splits=1 \
--results-dir=lr_find \
--dataset=folder \
--root_folder=ChestX

gdown --fuzzy https://drive.google.com/file/d/1NeBMqfrgLPJcb6_w9-2QZ7ZgYeSzG__u/view?usp=sharing && unzip tiny_imagenet_200.zip

python find_lr.py \
--lr=0.5 --epochs=10 \
--batch_size=16 \
--bn-splits=1 \
--results-dir=lr_find \
--dataset=folder \
--root_folder=imagenet


python find_lr.py \
--lr=0.5 --epochs=10 \
--batch_size=16 \
--bn-splits=1 \
--results-dir=lr_find \
--dataset=cifar10


python find_lr.py \
--lr=0.5 --epochs=10 \
--batch_size=16 \
--bn-splits=1 \
--results-dir=lr_find \
--dataset=stl10
