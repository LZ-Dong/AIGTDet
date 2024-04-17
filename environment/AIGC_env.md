AIGC_env

cuda=12.2

https://github.com/huggingface/transformers/blob/main/README_zh-hans.md

conda create -n AIGC_env python=3.9

conda activate AIGC_env

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install transformers
