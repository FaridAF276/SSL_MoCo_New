# apt-get install -y git && git clone https://github.com/FaridAF276/SSL_MoCo_New.git && cd SSL_MoCo_New && chmod +x swav.sh && ./swav.sh
#Setup everything to install apex correctly (configure vast.ai with cuda 10.1 and pytorch 1.4.0)
apt-get install -y unzip zip git wget
apt-get install -y libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6
wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh
bash ~/Anaconda3-2022.05-Linux-x86_64.sh
git clone https://github.com/facebookresearch/swav.git && cd swav
source ~/anaconda3/etc/profile.d/conda.sh # Or path to where your conda is
conda create -y --name=swav python=3.6.6
conda activate swav
conda install -y pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch
conda install -y -c conda-forge cudatoolkit-dev=10.1.243 opencv
conda install -y -c anaconda pandas
which pip
git clone "https://github.com/NVIDIA/apex"
cd apex
git checkout 4a1aa97e31ca87514e17c3cd3bbc03f4204579d0
python setup.py install --cuda_ext
python -c 'import apex; from apex.parallel import LARC' # should run and return nothing
python -c 'import apex; from apex.parallel import SyncBatchNorm; print(SyncBatchNorm.__module__)' # should run and return apex.parallel.optimized_sync_batchnorm 
cd ~/swav/
conda install -y -c conda-forge gdrive gdown


##Download dataset

gdown --fuzzy https://drive.google.com/file/d/1NeBMqfrgLPJcb6_w9-2QZ7ZgYeSzG__u/view?usp=sharing
unzip tiny_imagenet_200.zip

python -m torch.distributed.launch --nproc_per_node=1 main_swav.py \
--data_path imagenet/train \
--epochs 2 \
--base_lr 0.6 \
--final_lr 0.0006 \
--warmup_epochs 0 \
--batch_size 32 \
--size_crops 224 96 \
--nmb_crops 2 6 \
--min_scale_crops 0.14 0.05 \
--max_scale_crops 1. 0.14 \
--use_fp16 true \
--freeze_prototypes_niters 5005 \
--queue_length 3840 \
--epoch_queue_starts 15
