pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html

pip install git+https://github.com/openai/CLIP.git

python -m pip install 'git+https://github.com/facebookresearch/detectron2.git' --user

# install java
sudo apt-get install default-jre -y

pip install "git+https://github.com/salaniz/pycocoevalcap.git"

# build the project
pip install -e . --user

