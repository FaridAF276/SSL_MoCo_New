# apt-get install -y git && git clone https://github.com/FaridAF276/SSL_MoCo_New.git && cd SSL_MoCo_New && chmod +x swav.sh && ./swav.sh
apt-get install -y unzip zip git wget
wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh
bash ~/Anaconda3-2022.05-Linux-x86_64.sh
git clone https://github.com/facebookresearch/swav.git
cd swav
conda create --name=swav python=3.6.6
conda install -y pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch
conda install -y -c conda-forge cudatoolkit-dev=10.1.243 opencv
conda install -y -c anaconda pandas
which pip
git clone "https://github.com/NVIDIA/apex"
cd apex
git checkout 4a1aa97e31ca87514e17c3cd3bbc03f4204579d0
conda install -c nvidia/label/cuda-10.1.243 cuda-nvcc
python setup.py install --cuda_ext
python -c 'import apex; from apex.parallel import LARC' # should run and return nothing
python -c 'import apex; from apex.parallel import SyncBatchNorm; print(SyncBatchNorm.__module__)' # should run and return apex.parallel.optimized_sync_batchnorm
source ~/anaconda3/etc/profile.d/conda.sh # Or path to where your conda is
conda activate 