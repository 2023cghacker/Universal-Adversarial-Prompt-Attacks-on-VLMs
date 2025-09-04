# apt update
# apt install git
# git clone https://github.com/2023cghacker/Universal-Adversarial-Prompt-Attacks-on-VLMs.git

conda create -n myenv python=3.10 -y
conda activate myenv
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers
pip install accelerate
pip install matplotlib
pip install tiktoken
pip install protobuf
pip install sentencepiece


# scp -P port -r local_dirpath root@serverip:server_dirpath