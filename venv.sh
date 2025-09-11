# apt update
# apt install git
# git clone https://github.com/2023cghacker/Universal-Adversarial-Prompt-Attacks-on-VLMs.git

apt install python3.10 python3.10-venv python3-pip
python3.10 -m venv myenv
source myenv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers
pip install accelerate
pip install matplotlib
pip install tiktoken
pip install protobuf
pip install sentencepiece
pip install datasets
pip install qwen-omni-utils[decord] -U
pip install qwen-vl-utils[decord]==0.0.8

# scp -P port -r local_dirpath root@serverip:server_dirpath