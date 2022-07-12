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
--lr=0.5 --epochs=5 \
--batch_size=16 \
--bn-splits=8 \
--results-dir=lr_find \
--dataset=folder \
--root_folder=ChestX
